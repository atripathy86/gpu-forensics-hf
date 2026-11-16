"""Read-only Hugging Face Hub access: URL parsing, revision pinning, range reads.

Only metadata and byte ranges of safetensors files are fetched.  Pickle-based
checkpoints are never downloaded or deserialized and no repository code runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from . import __version__

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
USER_AGENT = f"gpu-forensics/{__version__}"
SAFETENSORS_DTYPES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
                      "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


class HubError(Exception):
    """A structured, reportable failure talking to the Hub."""

    def __init__(self, kind: str, message: str, url: str = ""):
        super().__init__(f"{kind}: {message}" + (f" ({url})" if url else ""))
        self.kind, self.message, self.url = kind, message, url


def cache_dir() -> Path:
    root = os.environ.get("GPU_FORENSICS_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "gpu-forensics")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    path = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "token"
    return path.read_text().strip() if path.exists() else None


def http_get(url: str, start: int | None = None, end: int | None = None, retries: int = 3) -> bytes:
    """GET ``url`` (optionally bytes ``start..end`` inclusive) with retries and auth."""
    headers = {"User-Agent": USER_AGENT}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    token = _token()
    if token and url.startswith(ENDPOINT):
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            if start is not None and len(data) != end - start + 1:
                raise HubError("range", f"expected {end - start + 1} bytes, got {len(data)}", url)
            return data
        except urllib.error.HTTPError as exc:
            kind = {401: "unauthorized", 403: "forbidden", 404: "not_found"}.get(exc.code)
            if kind:
                raise HubError(kind, f"HTTP {exc.code}", url) from exc
            if attempt == retries - 1:
                raise HubError("http", f"HTTP {exc.code}", url) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == retries - 1:
                raise HubError("network", str(getattr(exc, "reason", exc)), url) from exc
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


_RESOLVED: dict[str, str] = {}


def direct_url(url: str) -> str:
    """Follow the Hub's redirect once so repeated range reads go straight to storage."""
    if url not in _RESOLVED:
        headers = {"User-Agent": USER_AGENT, "Range": "bytes=0-0"}
        token = _token()
        if token and url.startswith(ENDPOINT):
            headers["Authorization"] = f"Bearer {token}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                response.read()
                _RESOLVED[url] = response.geturl()
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            return url
    return _RESOLVED[url]


def get_range(url: str, start: int, end: int) -> bytes:
    """Range read through the cached direct URL, re-resolving once if it expired."""
    try:
        return http_get(direct_url(url), start, end)
    except HubError as exc:
        if exc.kind not in ("forbidden", "unauthorized", "http") or url not in _RESOLVED:
            raise
        _RESOLVED.pop(url, None)
        return http_get(direct_url(url), start, end)


def get_json(url: str):
    return json.loads(http_get(url))


@dataclass(frozen=True)
class ModelRef:
    repo: str
    revision: str = "main"
    sha: str | None = None

    @property
    def pinned(self) -> str:
        return self.sha or self.revision

    def file_url(self, filename: str) -> str:
        return f"{ENDPOINT}/{self.repo}/resolve/{urllib.parse.quote(self.pinned, safe='')}/{filename}"

    def to_dict(self) -> dict:
        return asdict(self) | {"url": f"{ENDPOINT}/{self.repo}"}


_URL = re.compile(
    r"^(?:https?://(?:www\.)?huggingface\.co/)?"
    r"(?P<repo>[\w.-]+(?:/[\w.-]+)?)"
    r"(?:/(?:tree|blob|resolve)/(?P<rev>[^/]+)(?:/.*)?)?/?$"
)


def parse(value: str, revision: str | None = None) -> ModelRef:
    """Accept ``org/name``, ``name``, ``org/name@rev`` or a huggingface.co URL."""
    value = value.strip()
    rev = None
    if "@" in value and not value.startswith("http"):
        value, rev = value.rsplit("@", 1)
    if value.startswith("http") and not re.match(r"^https?://(www\.)?huggingface\.co/", value):
        raise ValueError(f"not a Hugging Face model URL: {value}")
    match = _URL.match(value)
    if not match or match.group("repo") in {"models", "datasets", "spaces"}:
        raise ValueError(f"expected a Hugging Face model URL or repository ID, got {value!r}")
    rev = revision or rev or match.group("rev") or "main"
    return ModelRef(match.group("repo"), urllib.parse.unquote(rev))


def model_info(ref: ModelRef) -> dict:
    rev = urllib.parse.quote(ref.revision, safe="")
    return get_json(f"{ENDPOINT}/api/models/{ref.repo}/revision/{rev}")


def resolve(ref: ModelRef) -> tuple[ModelRef, dict]:
    """Pin ``ref`` to a commit SHA; returns the pinned ref and the model info."""
    info = model_info(ref)
    return ModelRef(info.get("id", ref.repo), ref.revision, info.get("sha")), info


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple
    filename: str
    start: int  # absolute byte offset of the tensor data in the file
    end: int

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    def to_dict(self) -> dict:
        return {"name": self.name, "dtype": self.dtype, "shape": list(self.shape), "file": self.filename,
                "byte_range": [self.start, self.end]}


def _cached(key: str, fetch):
    path = cache_dir() / "meta" / (hashlib.sha256(key.encode()).hexdigest()[:32] + ".json")
    if path.exists():
        return json.loads(path.read_text())
    value = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return value


def safetensors_header(ref: ModelRef, filename: str) -> dict:
    """Parse a safetensors header with two range requests (cached when pinned)."""
    url = ref.file_url(filename)

    def fetch():
        n = struct.unpack("<Q", get_range(url, 0, 7))[0]
        if n > 100_000_000:
            raise HubError("format", f"implausible safetensors header length {n}", url)
        return {"length": n, "header": json.loads(get_range(url, 8, 7 + n))}

    return _cached(f"{url}#header", fetch) if ref.sha else fetch()


def tensor_inventory(ref: ModelRef, files: list[str], workers: int = 8) -> dict[str, TensorInfo]:
    """Every tensor's dtype, shape and absolute byte range across ``files``."""

    def one(filename):
        meta = safetensors_header(ref, filename)
        base = 8 + meta["length"]
        out = {}
        for name, item in meta["header"].items():
            if name == "__metadata__":
                continue
            a, b = item["data_offsets"]
            out[name] = TensorInfo(name, item["dtype"], tuple(item["shape"]), filename, base + a, base + b)
        return out

    inventory: dict[str, TensorInfo] = {}
    with ThreadPoolExecutor(workers) as pool:
        for part in pool.map(one, files):
            inventory.update(part)
    return inventory


def read_rows(ref: ModelRef, tensor: TensorInfo, rows, workers: int = 16):
    """Raw little-endian element bytes for selected rows of a 2-D tensor, grouped
    into contiguous range requests.  Returns ``{row: bytes}``."""
    width = tensor.shape[1]
    item = SAFETENSORS_DTYPES[tensor.dtype]
    stride = width * item
    rows = sorted({int(r) for r in rows})
    runs, run = [], []
    for r in rows:
        if run and (r != run[-1] + 1 or len(run) >= 64):
            runs.append(run)
            run = []
        run.append(r)
    if run:
        runs.append(run)
    url = ref.file_url(tensor.filename)

    def fetch(run):
        a = tensor.start + run[0] * stride
        data = get_range(url, a, a + len(run) * stride - 1)
        return {r: data[i * stride:(i + 1) * stride] for i, r in enumerate(run)}

    out = {}
    with ThreadPoolExecutor(workers) as pool:
        for part in pool.map(fetch, runs):
            out.update(part)
    return out
