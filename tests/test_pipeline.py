"""Phase 5: statistics, classification, end-to-end pipeline and CLI; golden replay of published models."""
import json
from pathlib import Path

import numpy as np
import pytest

import gpuid
from gpu_forensics import recipes, stats
from gpu_forensics.cli import main
from gpu_forensics.engine import fingerprint, parse_rows, recipe_from_known, revalidate
from gpu_forensics.recipes import Recipe, matrix

from .conftest import REPO, SEED, THREADS

ROOT = Path(__file__).resolve().parents[1]


MODELS = json.loads((ROOT / "models.json").read_text())


BIG = (129280, 7168)


@pytest.mark.parametrize("model", MODELS, ids=[m["name"] for m in MODELS])
def test_published_examples_reproduce(model):
    """Golden test: every published example is recovered with its SM count and r > threshold."""
    import fetch

    observed = fetch.rows(model)
    rec = recipe_from_known(model, model["shape"])
    rows = model["rows"][:12]
    obs = observed[:12]
    from gpu_forensics.stats import score_rows, sweep

    scores = sweep(rec, rows[:4], obs[:4], [s * 2048 for s in range(16, 257)])
    best = max(scores, key=scores.get)
    assert best == model["sms"] * 2048
    r = score_rows(rec.with_(threads=best), rows, obs)
    assert np.median(r) > 0.9999
    control = score_rows(rec.with_(threads=best, seed=rec.seed + 1), rows, obs)
    assert abs(np.median(control)) < 0.05


def test_ling_exact_trunc_normal_model():
    """The bf16 uniform_ path (bounds cast, upper bound folded) lifts Ling from r~0.983 to ~1."""
    model = next(m for m in MODELS if m["name"].startswith("Ling"))
    import fetch

    obs = fetch.rows(model)[:20]
    rec = recipe_from_known(model, model["shape"])
    from gpu_forensics.stats import score_rows

    assert np.median(score_rows(rec, model["rows"][:20], obs)) > 0.9999
    assert recipes.corr(obs[0], gpuid.draw(np.arange(model["shape"][1], dtype=np.uint64)
                                            + np.uint64(model["rows"][0] * model["shape"][1]),
                                            152 * 2048, 3952, 0, "trunc_normal")) < 0.99


def _synthetic(seed=42, threads=152 * 2048, rows=16, noise=0.0, dist="normal"):
    rec = Recipe(dist=dist, seed=seed, threads=threads, rows=BIG[0], width=4096, params={"std": 0.006})
    rws = list(range(BIG[0] - rows, BIG[0]))
    obs = matrix(rec, rws)
    if noise:
        obs = obs + np.random.default_rng(3).standard_normal(obs.shape) * noise
    return rec, rws, obs


def _validate(rec, rws, obs, null=30):
    fit, hold = stats.split(rws)
    idx = {r: i for i, r in enumerate(rws)}
    of = np.stack([obs[idx[r]] for r in fit])
    sw = stats.sweep(rec, fit, of, [132 * 2048, 148 * 2048, 152 * 2048, 108 * 2048])
    v = stats.validate(rec, rws, obs, fit, hold, sw, null_seeds=null)
    return stats.classify(v, float(np.median(stats.score_rows(rec, fit, of)))), v


def test_classify_supported():
    rec, rws, obs = _synthetic()
    (cls, _), v = _validate(rec, rws, obs)
    assert cls == "supported"
    assert v.holdout_median_r > 0.9999 and v.runner_up["threads"] != rec.threads


def test_classify_weak_signal():
    """Init values buried in noise (r~0.05 per element) are only detectable in aggregate."""
    rec, rws, obs = _synthetic(rows=64, noise=0.006 * 20)
    (cls, _), v = _validate(rec, rws, obs, null=60)
    assert cls == "weak", (v.holdout_median_r, v.fisher_z, v.null)


def test_classify_noise_inconclusive():
    rec, rws, _ = _synthetic()
    noise = np.random.default_rng(9).standard_normal((len(rws), 4096))
    (cls, _), _ = _validate(rec, rws, noise)
    assert cls == "inconclusive"


def test_fit_only_success_is_downgraded():
    rec, rws, obs = _synthetic()
    fit, hold = stats.split(rws)
    obs = obs.copy()
    for i, r in enumerate(rws):
        if r in hold:
            obs[i] = np.random.default_rng(i).standard_normal(obs.shape[1])
    (cls, reason), _ = _validate(rec, rws, obs)
    assert cls == "inconclusive" and "held-out" in reason


def test_parse_rows():
    assert parse_rows("1,3-5", 10) == [1, 3, 4, 5]
    assert parse_rows("tail:3", 10) == [7, 8, 9]
    assert len(parse_rows("random:4", 10)) == 4
    with pytest.raises(ValueError):
        parse_rows("11", 10)


def test_end_to_end_screen(fake_hub, monkeypatch):
    """Unknown model, seed in metadata: blind row selection + layout sweep + held-out validation."""
    fake_hub.files["README.md"] = f"We used random seed {SEED} for initialization.".encode()
    report = fingerprint(REPO, seed="screen", dist="normal", quiet=True, null_seeds=30)
    assert report.classification == "supported", report.reason
    assert report.best.seed == SEED and report.best.threads == THREADS
    assert all(r >= 4000 for r in report.sample.rows)
    assert report.validation.exact_fraction > 0.99


def test_end_to_end_exhaustive_search(fake_hub):
    report = fingerprint(REPO, seed="search", dist="normal", quiet=True, null_seeds=30,
                         seed_range=(SEED - (1 << 20), SEED + 1000), backend="cpu")
    assert report.classification == "supported", report.reason
    assert report.seed_search["status"] == "found" and report.best.seed == SEED


def test_end_to_end_seed_not_found_is_inconclusive(fake_hub):
    report = fingerprint(REPO, seed="search", dist="normal", quiet=True, seed_range=(0, 1 << 16), backend="cpu")
    assert report.classification == "inconclusive"
    assert report.seed_search["searches"][0]["seeds_covered"] == 1 << 16


def test_cli_fingerprint_and_validate(fake_hub, tmp_path, capsys):
    fake_hub.files["README.md"] = f"seed: {SEED}".encode()
    out = tmp_path / "r.json"
    assert main(["fingerprint", REPO, "--seed", "screen", "--dist", "normal", "--quiet", "--null-seeds", "20",
                 "--output", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["schema_version"] == "1.0" and report["classification"] == "supported"
    capsys.readouterr()
    assert main(["validate", str(out), "--null-seeds", "20", "--format", "text", "--quiet"]) == 0
    assert "supported" in capsys.readouterr().out
    again = revalidate(report, null_seeds=10, quiet=True)
    assert again["revalidated"]["sample_sha256_verified"]


def test_cli_rejects_bad_url(capsys):
    with pytest.raises(SystemExit):
        main(["inspect", "https://example.com/x"])



def test_split_spreads_fit_rows():
    fit, hold = stats.split(list(range(100)))
    assert fit[0] == 0 and fit[-1] == 98 and len(fit) == 4
    assert not set(fit) & set(hold) and len(fit) + len(hold) == 100
    assert stats.split([5]) == ([5], [])
