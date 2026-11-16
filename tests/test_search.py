"""Phase 4: exhaustive seed search."""
import json

import numpy as np
import pytest

from gpu_forensics import search
from gpu_forensics.recipes import Recipe, matrix, unit_normal, words

ROWS = [129100, 129150, 129200, 129250]


BIG = (129280, 7168)


@pytest.mark.parametrize("dist,code", [("normal", 0), ("uniform", 1)])
def test_sign_model_matches_reconstruction(dist, code):
    pos = np.random.default_rng(0).integers(0, 10**9, 20000).astype(np.uint64)
    w, lane = words(pos, 270336, 99)
    bits = search.sign_bits(w, lane, code)
    if dist == "normal":
        ref = unit_normal(pos, 270336, 99) > 0
    else:
        ref = w[lane, np.arange(lane.size)] >= 2**31
    assert np.mean(bits == ref) == 1.0


@pytest.mark.parametrize("backend", ["cpu", "numpy"])
@pytest.mark.parametrize("dist", ["normal", "uniform", "trunc_normal"])
def test_planted_seed_recovered(backend, dist):
    seed = 123_456_789
    rec = Recipe(dist=dist, seed=seed, threads=148 * 2048, rows=BIG[0], width=BIG[1], dtype="bfloat16",
                 params={"std": 0.006, "low": -0.01, "high": 0.01, "bound_sigmas": 2.0})
    obs = matrix(rec, ROWS)
    hyps = search.hypotheses_for([132 * 2048, 148 * 2048, 152 * 2048])
    span = 1 << (20 if backend == "cpu" else 12)
    res = search.search(obs, ROWS, BIG, dist, hyps, seed_range=(seed - span, seed + 7), n=64, max_miss=6,
                        backend_name=backend)
    assert res["status"] == "complete"
    found = [(h["seed"], h["hypotheses"][0]["threads"]) for h in res["hits"]]
    assert (seed, 148 * 2048) in found
    assert len(found) == 1


def test_search_is_resumable(tmp_path):
    rec = Recipe(dist="normal", seed=5000, threads=270336, rows=BIG[0], width=BIG[1])
    obs = matrix(rec, ROWS)
    cp = tmp_path / "cp.json"
    hyps = search.hypotheses_for([270336])
    first = search.search(obs, ROWS, BIG, "normal", hyps, seed_range=(0, 8192), chunk=4096, checkpoint=cp)
    state = json.loads(cp.read_text())
    assert state["done"] == [[0, 4096], [4096, 8192]]
    again = search.search(obs, ROWS, BIG, "normal", hyps, seed_range=(0, 8192), chunk=4096, checkpoint=cp)
    assert again["hits"] == first["hits"] and again["elapsed_seconds"] < 1


def test_search_budget_reports_partial_coverage():
    rec = Recipe(dist="normal", seed=1, threads=270336, rows=BIG[0], width=BIG[1])
    res = search.search(matrix(rec, ROWS), ROWS, BIG, "normal", search.hypotheses_for([270336]),
                        seed_range=(0, 1 << 20), chunk=1 << 16, budget_seconds=0)
    assert res["status"] == "budget_exhausted" and res["seeds_covered"] < 1 << 20


def test_tp_hypothesis_requires_single_shard():
    rows = [10, 129000]
    obs = np.zeros((2, BIG[1]))
    obs[:, ::2] = 1
    _, groups = search.plan_positions(rows, [np.arange(4)] * 2, obs, BIG, "normal",
                                      [search.Hypothesis(270336, tp=1), search.Hypothesis(270336, tp=2)])
    assert len(groups) == 1 and groups[0][1][0].tp == 1


def test_expected_false_hits_small():
    assert search.expected_false_hits(64, 6, 2**32, 20) < 1

