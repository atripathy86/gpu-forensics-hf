"""Stage E: held-out validation, null distributions and evidence classification."""

from __future__ import annotations

import math

import numpy as np

from .layouts import describe, effective_threads
from .recipes import Recipe, corr, exact_fraction, values
from .report import ValidationResult

SUPPORTED_R = 0.95
NULL_P = 0.01
STRONG_Z = 10.0
WEAK_Z = 5.0


def split(rows: list[int], max_fit: int = 4) -> tuple[list[int], list[int]]:
    """Deterministic split: up to ``max_fit`` even-indexed rows, spread evenly over the
    sample (so fit rows span tensor-parallel shards), fit; everything else is held out."""
    even = rows[0::2]
    picks = sorted(set(np.linspace(0, len(even) - 1, min(max_fit, len(even))).round().astype(int).tolist()))
    fit = [even[i] for i in picks]
    return fit, [r for r in rows if r not in set(fit)]


def score_rows(recipe: Recipe, rows, observed, cols=None) -> np.ndarray:
    """Per-row correlation; ``observed`` must already be restricted to ``cols``."""
    return np.asarray([corr(y, values(recipe, r, cols)) for r, y in zip(rows, observed)])


def sweep(recipe: Recipe, rows, observed, threads_list, cols=None) -> dict[int, float]:
    """Median correlation of the fit rows under every layout hypothesis."""
    return {int(g): float(np.median(score_rows(recipe.with_(threads=int(g)), rows, observed, cols)))
            for g in threads_list}


def distinguishable(recipe: Recipe, rows, a: int, b: int, cols=None) -> bool:
    """Whether layouts ``a`` and ``b`` generate different values at these positions."""
    for r in rows[:2]:
        if not np.array_equal(values(recipe.with_(threads=a), r, cols), values(recipe.with_(threads=b), r, cols)):
            return True
    return False


def aggregate_r(recipe: Recipe, rows, observed, cols=None) -> float:
    """Correlation over all elements of ``rows`` pooled together."""
    rec = np.concatenate([values(recipe, r, cols) for r in rows])
    return corr(np.asarray(observed).ravel(), rec)


def null_distribution(recipe: Recipe, rows, observed, k: int = 100, cols=None) -> np.ndarray:
    """Pooled correlation under wrong seeds ``seed+1 .. seed+k``."""
    return np.asarray([aggregate_r(recipe.with_(seed=recipe.seed + i), rows, observed, cols)
                       for i in range(1, k + 1)])


def fisher_z(r: float, n: int) -> float:
    r = min(max(r, -0.999999999), 0.999999999)
    return math.atanh(r) * math.sqrt(max(n - 3, 1))


def validate(recipe: Recipe, rows, observed, fit_rows, holdout_rows, sweep_scores: dict[int, float],
             null_seeds: int = 100, null_cols: int = 4096, threshold: float = 0.99,
             storage_dtype: str = "BF16") -> ValidationResult:
    """Score the best hypothesis on held-out rows against wrong-seed and wrong-layout alternatives."""
    index = {r: i for i, r in enumerate(rows)}
    hold = holdout_rows or fit_rows
    warnings = []
    if not holdout_rows:
        warnings.append("only one sampled row: held-out score reuses the fitting row")
    obs_h = np.stack([observed[index[r]] for r in hold])
    scores = score_rows(recipe, hold, obs_h)
    flat_obs = obs_h.ravel()
    flat_rec = np.concatenate([values(recipe, r) for r in hold])
    agg = corr(flat_obs, flat_rec)
    width = obs_h.shape[1]
    cols = np.sort(np.random.default_rng(0).choice(width, min(width, null_cols), replace=False))
    null_rows = hold[:16]
    obs_n = np.stack([observed[index[r]][cols] for r in null_rows])
    best_n = aggregate_r(recipe, null_rows, obs_n, cols)
    nulls = null_distribution(recipe, null_rows, obs_n, null_seeds, cols)
    mean, sd = float(nulls.mean()), float(nulls.std()) or 1e-12
    null = {"statistic": "pooled r over null rows x columns", "observed": best_n, "seeds": null_seeds,
            "mean": mean, "sd": sd, "max": float(nulls.max()),
            "z": (best_n - mean) / sd, "p_value": (1 + int((nulls >= best_n).sum())) / (null_seeds + 1),
            "columns": int(len(cols)), "rows": len(null_rows)}

    same, ranked = [], sorted(sweep_scores.items(), key=lambda kv: -kv[1])
    runner = {}
    for g, r in ranked:
        if g == recipe.threads:
            continue
        g_eff = effective_threads(recipe.locate(hold[0])[2], g)
        if g_eff == effective_threads(recipe.locate(hold[0])[2], recipe.threads) or \
                not distinguishable(recipe, fit_rows, g, recipe.threads):
            same.append(g)
            continue
        if not runner:
            runner = {"threads": g, "median_r": r, "devices": describe(g)}
    if same:
        warnings.append(f"{len(same)} other layout(s) generate identical values at the sampled positions")

    exact = None
    if recipe.dist == "normal":
        unit = recipe.with_(dtype="float32", params={"std": 1.0, "mean": 0.0})
        target = {"BF16": "bfloat16", "F16": "float16"}.get(storage_dtype, "float32")
        exact = exact_fraction(flat_obs, np.concatenate([values(unit, r) for r in hold]), target)
    return ValidationResult(
        fit_rows=list(fit_rows), holdout_rows=list(hold),
        holdout_median_r=float(np.median(scores)), holdout_min_r=float(scores.min()),
        rows_matched=f"{int((scores > threshold).sum())}/{len(scores)}",
        exact_fraction=exact, aggregate_r=agg, fisher_z=fisher_z(agg, flat_obs.size),
        null=null, runner_up=runner, indistinguishable_threads=sorted(same), warnings=warnings)


def classify(v: ValidationResult, fit_r: float) -> tuple[str, str]:
    """Evidence rules (documented in README):

    supported    held-out median r >= 0.95, no wrong seed scores as high (p <= max(0.01, 1/(K+1))),
                 null z >= 10, and every distinguishable layout trails by more than 0.5
    weak         aggregate held-out signal significant (Fisher z >= 5, null z >= 5, same p rule)
    inconclusive everything else, including a fit-only success
    """
    runner = v.runner_up.get("median_r")
    p_ok = v.null["p_value"] <= max(NULL_P, 1.0 / (v.null["seeds"] + 1))
    margin_ok = runner is None or (v.holdout_median_r - runner) > 0.5
    if (v.holdout_median_r >= SUPPORTED_R and p_ok and v.null["z"] >= STRONG_Z
            and margin_ok and v.holdout_rows):
        return "supported", (f"held-out median r={v.holdout_median_r:.4f} vs wrong-seed null "
                             f"{v.null['mean']:+.4f}±{v.null['sd']:.4f} (z={v.null['z']:.0f}); "
                             f"runner-up layout r={runner if runner is None else round(runner, 4)}")
    if fit_r >= SUPPORTED_R and v.holdout_median_r < 0.5:
        return "inconclusive", "fits the fitting rows but not the held-out rows"
    if v.fisher_z >= WEAK_Z and v.null["z"] >= WEAK_Z and p_ok:
        return "weak", (f"aggregate held-out r={v.aggregate_r:.4f} (Fisher z={v.fisher_z:.1f}, null z="
                        f"{v.null['z']:.1f}) is significant but below the per-row threshold")
    return "inconclusive", (f"held-out median r={v.holdout_median_r:.4f} is not distinguishable from the "
                            f"wrong-seed null (z={v.null['z']:.1f}, p={v.null['p_value']:.3f})")
