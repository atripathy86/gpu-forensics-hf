"""Bit-exact checks against fixtures produced on real GPUs by tools/synthesize_fixtures.py.

Skipped when ``fixtures/`` holds no recordings."""

import json
from pathlib import Path

import numpy as np
import pytest

from gpu_forensics.layouts import philox_offset_increment
from gpu_forensics.recipes import Recipe, values

FIXTURES = sorted((Path(__file__).resolve().parents[1] / "fixtures").glob("*.json"))


@pytest.mark.skipif(not FIXTURES, reason="no GPU fixtures recorded")
@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_cpu_model_reproduces_gpu(path):
    meta = json.loads(path.read_text())
    observed = np.load(path.with_suffix(".npy"))
    threads = meta["device"]["threads"]
    offset = sum(philox_offset_increment(n, threads) for n in meta["preceding_numels"])
    rec = Recipe(dist=meta["dist"], seed=meta["seed"], threads=threads, rows=meta["shape"][0],
                 width=meta["shape"][1], offset=offset, dtype=meta["dtype"], params=meta["params"])
    rebuilt = np.stack([values(rec, r) for r in meta["rows"]]).astype(np.float32)
    exact = float(np.mean(rebuilt == observed))
    assert exact > 0.999, f"{meta['case']}: {exact:.4f} bit-exact"
