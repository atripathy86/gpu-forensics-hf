"""Shared fixtures: an in-memory Hugging Face Hub serving a synthetic checkpoint."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from gpu_forensics import hub
from gpu_forensics.recipes import Recipe, bf16, matrix

REPO = "test-org/tiny-model"
SHA = "0123456789abcdef0123456789abcdef01234567"
SHAPE = (4096, 256)
TOKENIZER_SIZE = 4000
SEED = 777_777_777
THREADS = 132 * 2048


def safetensors_bytes(tensors: dict[str, tuple[str, np.ndarray]]) -> bytes:
    header, blobs, offset = {}, [], 0
    for name, (dtype, array) in tensors.items():
        raw = array.tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    text = json.dumps(header).encode()
    text += b" " * (-len(text) % 8)
    return struct.pack("<Q", len(text)) + text + b"".join(blobs)


def synthetic_embedding(seed=SEED, threads=THREADS, dist="normal", std=0.02, trained_scale=(0.7, 3.0)):
    """Rows >= TOKENIZER_SIZE keep their init; the rest look trained."""
    rec = Recipe(dist=dist, seed=seed, threads=threads, rows=SHAPE[0], width=SHAPE[1], dtype="bfloat16",
                 params={"std": std, "low": -std, "high": std, "bound_sigmas": 2.0})
    init_rows = list(range(TOKENIZER_SIZE, SHAPE[0]))
    rng = np.random.default_rng(1)
    # trained rows: heavy-tailed (Student t, 4 dof) with a spread of scales overlapping the init std
    w = (rng.standard_t(4, SHAPE) / np.sqrt(2) * rng.uniform(*trained_scale, (SHAPE[0], 1)) * std).astype(np.float32)
    w[init_rows] = matrix(rec, init_rows).astype(np.float32)
    bits = (bf16(w).view(np.uint32) >> 16).astype("<u2")
    return bits, rec


class FakeHub:
    def __init__(self, files: dict[str, bytes], info: dict):
        self.files, self.info, self.requests = files, info, []

    def get(self, url, start=None, end=None, retries=3):
        self.requests.append((url, start, end))
        if "/api/models/" in url:
            return json.dumps(self.info).encode()
        name = url.split(f"/resolve/{SHA}/")[-1] if f"/resolve/{SHA}/" in url else url.split("/resolve/main/")[-1]
        if name not in self.files:
            raise hub.HubError("not_found", "HTTP 404", url)
        data = self.files[name]
        return data if start is None else data[start:end + 1]


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    bits, rec = synthetic_embedding()
    vocab = {f"tok{i}": i for i in range(TOKENIZER_SIZE)}
    files = {
        "config.json": json.dumps({"architectures": ["TinyForCausalLM"], "vocab_size": SHAPE[0],
                                   "hidden_size": SHAPE[1], "initializer_range": 0.02}).encode(),
        "model.safetensors": safetensors_bytes({"model.embed_tokens.weight": ("BF16", bits),
                                                "lm_head.weight": ("BF16", bits)}),
        "tokenizer.json": json.dumps({"model": {"vocab": vocab}, "added_tokens": []}).encode(),
        "README.md": b"# tiny\nTrained on a laptop.\n",
    }
    info = {"id": REPO, "sha": SHA, "siblings": [{"rfilename": n} for n in files], "tags": [], "cardData": {}}
    fake = FakeHub(files, info)
    monkeypatch.setattr(hub, "http_get", fake.get)
    monkeypatch.setattr(hub, "direct_url", lambda url: url)
    monkeypatch.setenv("GPU_FORENSICS_CACHE", str(tmp_path / "cache"))
    fake.recipe = rec
    return fake
