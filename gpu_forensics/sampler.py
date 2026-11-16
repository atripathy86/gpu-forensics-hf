"""Stage B: select embedding rows likely to still hold their initial values.

Rows of tokens that never occur in training receive no gradient; at most a
uniform weight-decay shrink.  They therefore share one per-row standard
deviation (the init std times a common factor), while trained rows drift apart.
Candidates come from the tokenizer (padding/reserved ids) and from a cheap
statistical scan read with HTTP range requests.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from . import hub
from .recipes import bf16_bits_to_float
from .report import TensorSample


def decode(raw: bytes, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        return bf16_bits_to_float(np.frombuffer(raw, "<u2")).astype(np.float64)
    if dtype == "F16":
        return np.frombuffer(raw, "<f2").astype(np.float64)
    if dtype == "F32":
        return np.frombuffer(raw, "<f4").astype(np.float64)
    raise ValueError(f"cannot decode {dtype}")


def fetch_rows(ref: hub.ModelRef, tensor: hub.TensorInfo, rows) -> tuple[np.ndarray, str]:
    """Rows as float64 plus the SHA-256 of their raw little-endian bytes.

    Pinned reads are cached on disk under the commit SHA."""
    rows = [int(r) for r in rows]
    cache = None
    if ref.sha:
        key = hashlib.sha256(json.dumps([ref.repo, ref.sha, tensor.name, rows]).encode()).hexdigest()[:32]
        cache = hub.cache_dir() / "rows" / f"{key}.npy"
        if cache.exists():
            raw = np.load(cache)
            return np.stack([decode(r.tobytes(), tensor.dtype) for r in raw]), hashlib.sha256(raw.tobytes()).hexdigest()
    parts = hub.read_rows(ref, tensor, rows)
    raw = np.stack([np.frombuffer(parts[r], np.uint8) for r in rows])
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, raw)
    return np.stack([decode(r.tobytes(), tensor.dtype) for r in raw]), hashlib.sha256(raw.tobytes()).hexdigest()


def row_stats(x: np.ndarray) -> dict:
    mean = x.mean(1)
    std = x.std(1)
    z = (x - mean[:, None]) / np.where(std > 0, std, 1)[:, None]
    return {"mean": mean, "std": std, "kurtosis": (z**4).mean(1) - 3}


def init_cluster(std: np.ndarray, tol: float = 0.03, prefer=None) -> tuple[float | None, np.ndarray]:
    """Locate the init scale: the most *isolated* narrow cluster of per-row std values.

    Untrained rows share one std (up to sampling error, about ``1/sqrt(2*width)``
    in log terms, which ``tol`` must exceed); trained rows spread smoothly.  Each
    candidate centre is scored by its count within ``tol`` minus the count in the
    denser of the two equal-width flanking bands (``tol..3*tol`` either side), so
    neither a dense broad population nor its edge counts as a cluster.  Returns the cluster centre and a member mask."""
    ok = std > 0
    if ok.sum() < 3:
        return None, np.zeros(std.shape, bool)
    logs = np.log(std[ok])
    best, centre = 0.0, None
    for c in (np.log(prefer),) if prefer else logs:
        d = logs - c
        core = int((np.abs(d) < tol).sum())
        left = int(((d <= -tol) & (d > -3 * tol)).sum())
        right = int(((d >= tol) & (d < 3 * tol)).sum())
        excess = core - max(left, right)  # the denser flank: edges of a broad population are not clusters
        # Without tokenizer evidence, at least half of the core must stand above background.
        if excess > best and (prefer is not None or excess >= 0.5 * core):
            best, centre = excess, c
    if centre is None or best < 3:
        return None, np.zeros(std.shape, bool)
    centre = float(np.median(logs[np.abs(logs - centre) < tol]))
    mask = np.zeros(std.shape, bool)
    mask[ok] = np.abs(logs - centre) < tol
    return float(np.exp(centre)), mask


def clusters(std: np.ndarray, tol: float, k: int = 20) -> list[tuple[float, np.ndarray]]:
    """Isolated std clusters of at least 4 rows, smallest scale first.

    Untrained rows never grow (weight decay only shrinks them), while trained
    rows usually end up larger than the init, so the smallest isolated cluster is
    the most likely init population even when it holds only a handful of rows."""
    found, remaining = [], np.ones(std.shape, bool)
    for _ in range(k):
        masked = np.where(remaining, std, 0.0)
        scale, mask = init_cluster(masked, tol)
        if scale is None:
            break
        remaining &= ~mask
        if mask.sum() >= 4:
            found.append((scale, mask))
    return sorted(found, key=lambda c: c[0])


def select_rows(ref: hub.ModelRef, tensor: hub.TensorInfo, tokenizer: dict | None = None,
                coarse: int = 256, tail: int = 1024, head: int = 512, max_rows: int = 64,
                rng_seed: int = 0) -> dict:
    """Pick likely-untrained rows.  Deterministic for a given configuration.

    Scans tokenizer candidates, evenly spaced rows, and the head and tail of the
    tensor (byte-fallback and reserved tokens cluster there).  Returns the rows of
    the preferred cluster plus ``alternates`` from other isolated clusters."""
    rows_total, width = tensor.shape
    tokenizer = tokenizer or {}
    tok_rows = set(tokenizer.get("reserved_ids", []))
    if tokenizer.get("size") and tokenizer["size"] < rows_total:
        tok_rows.update(range(tokenizer["size"], rows_total))
    tok_rows = {r for r in tok_rows if 0 <= r < rows_total}
    scan = set(np.linspace(0, rows_total - 1, coarse).astype(int).tolist())
    scan.update(range(max(0, rows_total - tail), rows_total, 2))
    scan.update(range(0, min(head, rows_total)))
    tok_list = sorted(tok_rows)
    if len(tok_list) > 4 * max_rows:  # subsample long padding ranges deterministically
        tok_list = sorted(np.random.default_rng(rng_seed).choice(tok_list, 4 * max_rows, replace=False).tolist())
    scan.update(tok_list)
    scanned = sorted(scan)
    values, _ = fetch_rows(ref, tensor, scanned)
    stats = row_stats(values)
    tol = max(0.03, 3.0 / np.sqrt(2.0 * width))
    # normal_/trunc_normal_/uniform_ rows have excess kurtosis in [-1.2, 0]; trained rows are heavy-tailed
    shape_ok = (stats["kurtosis"] > -1.4) & (stats["kurtosis"] < 6 * np.sqrt(24.0 / width))
    tok_mask = np.isin(scanned, tok_list)
    candidates = []
    if tok_mask.sum() >= 3:
        prefer, members = init_cluster(np.where(tok_mask, stats["std"], 0.0), tol)
        if prefer is not None and members[tok_mask].mean() >= 0.5:
            scale, mask = init_cluster(stats["std"], tol, prefer=prefer)
            if scale is not None:
                candidates.append(("tokenizer+std-cluster", scale, mask))
    for scale, mask in clusters(np.where(shape_ok, stats["std"], 0.0), tol):
        if not any(abs(np.log(scale / c[1])) < tol for c in candidates):
            candidates.append(("std-cluster", scale, mask))

    groups = []
    for method, scale, mask in candidates:
        mask = mask & shape_ok & (np.abs(stats["mean"]) < 6 * scale / np.sqrt(width))
        chosen = [r for r, m in zip(scanned, mask) if m]
        backed = [r for r in chosen if r in tok_rows]
        if method.startswith("tokenizer") and len(backed) >= min(8, max_rows):
            chosen = backed
        # Later rows sit past every launch grid; tokenizer-backed rows first.
        chosen.sort(key=lambda r: (r not in tok_rows, -r))
        if len(chosen) >= 2:
            groups.append({"rows": sorted(chosen[:max_rows]), "init_scale": scale, "method": method,
                           "members": int(mask.sum()),
                           "kurtosis_median": float(np.median(stats["kurtosis"][mask]))})
    primary = groups[0] if groups else {"rows": [], "init_scale": None, "method": "none"}
    return primary | {"scanned_rows": len(scanned), "tokenizer_candidates": len(tok_rows), "std_tolerance": tol,
                      "alternates": groups[1:]}


def sample(ref: hub.ModelRef, tensor: hub.TensorInfo, rows, selection: dict | None = None, known=None):
    """Read rows with provenance.  ``known`` models use the hash-checked local copies in ``rows/``."""
    if known is not None:
        import fetch

        values, digest = fetch.rows(known), known["sha256"]
        selection = (selection or {}) | {"source": "rows/ (hash-verified copy of the pinned revision)"}
    else:
        values, digest = fetch_rows(ref, tensor, rows)
    stride = tensor.shape[1] * hub.SAFETENSORS_DTYPES[tensor.dtype]
    provenance = TensorSample(
        tensor=tensor.name, dtype=tensor.dtype, shape=list(tensor.shape), file=tensor.filename,
        rows=[int(r) for r in rows],
        byte_ranges=[[tensor.start + int(r) * stride, tensor.start + (int(r) + 1) * stride - 1] for r in rows],
        sha256=digest,
        rng_position="row * width + column within the tensor-parallel shard (row-major, contiguous)",
        selection=selection or {},
    )
    return values, provenance
