"""End-to-end pipeline: inspect -> sample -> seeds -> layouts -> validation -> report."""

from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np

from . import hub, sampler, search, stats
from .inspector import find_known, inspect
from .layouts import candidate_threads, catalogue_threads, describe, equivalent_configs
from .recipes import Recipe
from .report import LIMITATIONS, CandidateHypothesis, ForensicReport, dumps

INIT_DTYPE = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}
SCREEN_COLS = 512
SWEEP_COLS = 2048


def log(msg: str, quiet: bool = False):
    if not quiet:
        import sys
        print(f"[gpu-forensics] {msg}", file=sys.stderr, flush=True)


def recipe_from_known(model: dict, shape) -> Recipe:
    params = {"std": model.get("std", 1.0)}
    if model["dist"] == "trunc_normal":
        params = {"bound_sigmas": model.get("bound_sigmas", 3.0)}
    return Recipe(dist=model["dist"], seed=int(model["seed"]), threads=int(model.get("sms", 132)) * 2048,
                  rows=shape[0], width=shape[1], offset=int(model.get("cpu_offset", 0)),
                  dtype=model.get("dtype", "bfloat16"), tp=int(model.get("tp", 1)),
                  seed_per_rank=int(model.get("seed_per_rank", 0)), params=params)


def variants(dists, init_dtype: str) -> list[dict]:
    """Distribution/parameter families tried when the init operation is unknown.

    Weights are often initialized in float32 and only cast to bf16/fp16 for
    storage, so both the storage dtype and float32 are tried."""
    out = []
    for dtype in dict.fromkeys([init_dtype, "float32"]):
        for d in dists:
            if d == "normal":
                out.append({"dist": "normal", "dtype": dtype, "params": {"std": 1.0}})
            elif d == "uniform":
                out.append({"dist": "uniform", "dtype": dtype, "params": {"low": -1.0, "high": 1.0}})
            else:
                for k in (2.0, 3.0):
                    out.append({"dist": "trunc_normal", "dtype": dtype, "params": {"bound_sigmas": k}})
                out.append({"dist": "trunc_normal", "dtype": dtype,
                            "params": {"std": 0.02, "a": -2.0, "b": 2.0}})
    return out


def _cols(width: int, n: int, seed: int = 0) -> np.ndarray:
    return np.sort(np.random.default_rng(seed).choice(width, min(width, n), replace=False))


def screen(seeds, variant_list, threads_list, shape, rows, observed, tps=((1, 0),), offsets=(0,)):
    """Score cheap candidates on a few columns; returns ranked (r, recipe)."""
    cols = _cols(shape[1], SCREEN_COLS)
    obs = observed[:, cols]
    results = []
    for seed in seeds:
        for v in variant_list:
            for tp, spr in tps:
                for off in offsets:
                    base = Recipe(dist=v["dist"], seed=int(seed), threads=threads_list[0], rows=shape[0],
                                  width=shape[1], offset=off, dtype=v["dtype"], tp=tp, seed_per_rank=spr,
                                  params=v["params"])
                    for g in threads_list:
                        rec = base.with_(threads=g)
                        r = float(np.median(stats.score_rows(rec, rows, obs, cols)))
                        results.append((r, rec))
    results.sort(key=lambda x: -x[0])
    return results


def seed_candidates(explicit, insp, known) -> list[tuple[int, str]]:
    out = []
    if explicit:
        out += [(s, "explicit") for s in explicit]
    if known:
        out.append((int(known["seed"]), "models.json"))
    for h in (insp.hints or {}).get("seeds", []):
        out.append((int(h["seed"]), h["source"]))
    out += [(s, "convention") for s in search.CONVENTIONAL]
    seen, uniq = set(), []
    for s, src in out:
        if s not in seen:
            seen.add(s)
            uniq.append((s, src))
    return uniq


def search_rows(rows, shape, k: int = 8):
    """Up to ``k`` sampled rows that share one eighth of the tensor (one TP shard for tp<=8)."""
    eighth = -(-shape[0] // 8)
    buckets: dict[int, list[int]] = {}
    for r in rows:
        buckets.setdefault(r // eighth, []).append(r)
    best = max(buckets.values(), key=len)
    return best[-k:]


def fingerprint(value: str, revision: str | None = None, seed: str = "auto", dist: str = "auto",
                tensor_name: str | None = None, rows_spec: str | None = None, offsets=(0,), tps=(1,),
                search_budget: float | None = 3600.0, backend: str = "auto", null_seeds: int = 100,
                max_rows: int = 64, quiet: bool = False, seed_range=(0, search.SEED_SPACE)) -> ForensicReport:
    started = time.time()
    known = find_known(hub.parse(value).repo)
    if known and revision is None and hub.parse(value).revision == "main":
        revision = known["revision"]  # pin known models to the revision their recipe was recorded at
    insp = inspect(value, revision)
    model = insp.model
    report = ForensicReport(model=model, classification="inconclusive", reason="", inspection=asdict(insp),
                            limitations=list(LIMITATIONS))
    report.inspection.pop("tokenizer", None)

    def finish(classification, reason):
        report.classification, report.reason = classification, reason
        report.runtime_seconds = round(time.time() - started, 2)
        return report

    if insp.feasibility.get("verdict") == "infeasible" and not tensor_name:
        return finish("inconclusive", "; ".join(insp.feasibility.get("blockers", [])) or "infeasible")
    ref = hub.ModelRef(model["repo"], model["revision"], model.get("sha"))

    # Stage B: tensor and rows
    if tensor_name or not insp.embedding:
        name = tensor_name or (known or {}).get("tensor")
        if not name:
            return finish("inconclusive", "no embedding tensor identified; pass --tensor")
        index = hub.get_json(ref.file_url("model.safetensors.index.json")) \
            if insp.files.get("safetensors", 0) > 1 else None
        file = index["weight_map"][name] if index else "model.safetensors"
        tensor = hub.tensor_inventory(ref, [file])[name]
    else:
        e = insp.embedding
        tensor = hub.TensorInfo(e["name"], e["dtype"], tuple(e["shape"]), e["file"], *e["byte_range"])
    shape = tensor.shape
    if len(shape) != 2:
        return finish("inconclusive", f"{tensor.name} is not a 2-D tensor")

    selection = {}
    if rows_spec:
        rows = parse_rows(rows_spec, shape[0])
        selection = {"method": "user", "spec": rows_spec}
    elif known and known["tensor"] == tensor.name and seed in ("auto", "known"):
        rows = list(known["rows"])
        selection = {"method": "models.json"}
    else:
        log(f"scanning {tensor.name} {list(shape)} for untrained rows", quiet)
        selection = sampler.select_rows(ref, tensor, inspector_tokenizer(insp), max_rows=max_rows)
        rows = selection.pop("rows")
        if len(rows) < 1:
            return finish("inconclusive", "no rows with an initialization-like distribution were found")
    log(f"reading {len(rows)} rows", quiet)
    local = known and selection.get("method") == "models.json" and known.get("revision") == ref.sha
    observed, sample = sampler.sample(ref, tensor, rows, selection, known=known if local else None)
    report.sample = sample
    fit_rows, holdout_rows = stats.split(rows)
    index = {r: i for i, r in enumerate(rows)}
    obs_fit = np.stack([observed[index[r]] for r in fit_rows])

    # Stage C/D: seeds and layouts
    dists = ["normal", "uniform", "trunc_normal"] if dist == "auto" else [dist]
    init_dtype = INIT_DTYPE.get(tensor.dtype, "float32")
    tp_pairs = sorted({(t, s if t > 1 else 0) for t in tps for s in (0, 1)})
    best: Recipe | None = None
    if known and seed in ("auto", "known") and known["tensor"] == tensor.name:
        best = recipe_from_known(known, shape)
        report.seed_search = {"status": "known", "summary": f"seed {best.seed} from models.json"}
    elif seed != "search":
        explicit = [int(s) for s in seed.split(",")] if seed not in ("auto", "known", "screen") else []
        cands = seed_candidates(explicit, insp, None)
        log(f"screening {len(cands)} seed candidates", quiet)
        ranked = screen([s for s, _ in cands], variants(dists, init_dtype), catalogue_threads(), shape,
                        fit_rows[:4], obs_fit[:4], tps=tp_pairs, offsets=offsets)
        for alt in (selection.get("alternates") or []) if not (ranked and ranked[0][0] > 0.5) else []:
            log(f"retrying with the alternate row cluster at std {alt['init_scale']:.3g}", quiet)
            alt_obs, alt_sample = sampler.sample(ref, tensor, alt["rows"], {k: v for k, v in alt.items()
                                                                            if k != "rows"})
            a_fit, a_hold = stats.split(alt["rows"])
            a_index = {r: i for i, r in enumerate(alt["rows"])}
            a_obs_fit = np.stack([alt_obs[a_index[r]] for r in a_fit])
            alt_ranked = screen([s for s, _ in cands], variants(dists, init_dtype), catalogue_threads(), shape,
                                a_fit[:4], a_obs_fit[:4], tps=tp_pairs, offsets=offsets)
            if alt_ranked and alt_ranked[0][0] > 0.5:
                ranked, rows, observed, report.sample = alt_ranked, list(alt["rows"]), alt_obs, alt_sample
                fit_rows, holdout_rows, index, obs_fit = a_fit, a_hold, a_index, a_obs_fit
                break
        if ranked and ranked[0][0] > 0.5:
            best = ranked[0][1]
            src = dict(cands)[best.seed]
            report.seed_search = {"status": "screened", "summary": f"seed {best.seed} ({src}) r={ranked[0][0]:.4f}",
                                  "candidates": len(cands)}
        elif explicit:
            log("explicit seed did not match at offset 0; scanning offsets", quiet)
            ranked = screen(explicit, variants(dists, init_dtype), catalogue_threads(), shape, fit_rows[:1],
                            obs_fit[:1], tps=tp_pairs, offsets=range(0, 8192, 4))
            if ranked and ranked[0][0] > 0.5:
                best = ranked[0][1]
                report.seed_search = {"status": "offset_scan", "summary": f"offset {best.offset}"}
        if best is None:
            report.seed_search = {"status": "not_found", "summary": f"{len(cands)} candidate seeds screened",
                                  "candidates": len(cands)}
    if best is None and seed in ("auto", "search"):
        best = exhaustive(report, ref, tensor, rows, observed, shape, dists, init_dtype, offsets, tps,
                          search_budget, backend, quiet, seed_range)
    if best is None:
        return finish("inconclusive", "seed not recovered: " + (report.seed_search or {}).get("summary", ""))

    log("sweeping layouts", quiet)
    cols = _cols(shape[1], SWEEP_COLS)
    threads_list = sorted(set(candidate_threads()) | set(catalogue_threads()) | {best.threads})
    sweep = stats.sweep(best, fit_rows, obs_fit[:, cols], threads_list, cols)
    top = max(sweep, key=sweep.get)
    best = best.with_(threads=top)
    if dist == "auto" and not known:
        options = [best.with_(dist=v["dist"], dtype=v["dtype"], params=v["params"])
                   for v in variants(dists, init_dtype)]
        best = max(options, key=lambda r: np.median(stats.score_rows(r, fit_rows, obs_fit[:, cols], cols)))
    fit_r = float(np.median(stats.score_rows(best, fit_rows, obs_fit)))
    report.layout_sweep = [{"threads": g, "median_r": r} for g, r in
                           sorted(sweep.items(), key=lambda kv: -kv[1])[:10]]

    log("validating on held-out rows", quiet)
    thr = (known or {}).get("thr", 0.99)
    v = stats.validate(best, rows, observed, fit_rows, holdout_rows, sweep, null_seeds=null_seeds,
                       threshold=thr, storage_dtype=tensor.dtype)
    report.validation = v
    report.best = CandidateHypothesis(seed=best.seed, threads=best.threads, dist=best.dist, offset=best.offset,
                                      dtype=best.dtype, tp=best.tp, seed_per_rank=best.seed_per_rank,
                                      params=best.params, fit_median_r=fit_r, devices=describe(best.threads),
                                      configs=equivalent_configs(best.threads))
    if known and known.get("revision") and known["revision"] != ref.sha:
        v.warnings.append("known recipe applied to a different revision than it was recorded for")
    return finish(*stats.classify(v, fit_r))


def exhaustive(report, ref, tensor, rows, observed, shape, dists, init_dtype, offsets, tps, budget, backend,
               quiet, seed_range=(0, search.SEED_SPACE)) -> Recipe | None:
    srows = search_rows(rows, shape)
    idx = {r: i for i, r in enumerate(rows)}
    obs = np.stack([observed[idx[r]] for r in srows])
    hyps = search.hypotheses_for(catalogue_threads(), tps, offsets)
    families = sorted({"normal" if d == "normal" else "uniform" for d in dists})
    key = f"{ref.repo}@{ref.sha}:{tensor.name}:{seed_range[0]}-{seed_range[1]}".replace("/", "_")
    results = []
    for fam in families:
        log(f"exhaustive {fam} seed search over 2^32 seeds x {len(hyps)} hypotheses", quiet)
        cp = hub.cache_dir() / "search" / f"{key}-{fam}.json"
        res = search.search(obs, srows, shape, fam, hyps, seed_range=tuple(seed_range), n=64, max_miss=6,
                            budget_seconds=budget,
                            checkpoint=cp, backend_name=backend,
                            progress=lambda a, b, hi, n: log(f"  {b / hi:6.1%} searched, {n} hits", quiet))
        res["family"] = fam
        results.append(res)
        for hit in res["hits"]:
            for h in hit["hypotheses"]:
                for spr in ((0, 1) if h["tp"] > 1 else (0,)):
                    base = hit["seed"] - hit["rank"] * spr
                    for v in variants([fam] if fam == "normal" else ["uniform", "trunc_normal"], init_dtype):
                        rec = Recipe(dist=v["dist"], seed=base, threads=h["threads"], rows=shape[0], width=shape[1],
                                     offset=h["offset"], dtype=v["dtype"], tp=h["tp"], seed_per_rank=spr,
                                     params=v["params"])
                        r = float(np.median(stats.score_rows(rec, srows[:2], obs[:2])))
                        hit.setdefault("confirmations", []).append({"recipe": rec.to_dict(), "median_r": r})
                        if r > 0.5:
                            report.seed_search = {"status": "found", "searches": results,
                                                  "summary": f"seed {base} found by exhaustive search"}
                            return rec
    report.seed_search = {"status": "not_found", "searches": results,
                          "summary": "; ".join(f"{r['family']}: {r['status']}, {r['summary']}" for r in results)}
    return None


def inspector_tokenizer(insp) -> dict:
    return insp.tokenizer or {}


def parse_rows(spec: str, total: int) -> list[int]:
    """``1,2,10-20`` or ``random:N`` (deterministic) or ``tail:N``."""
    if spec.startswith("random:"):
        n = int(spec.split(":")[1])
        return sorted(np.random.default_rng(0).choice(total, min(n, total), replace=False).tolist())
    if spec.startswith("tail:"):
        n = int(spec.split(":")[1])
        return list(range(total - n, total))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    bad = [r for r in out if not 0 <= r < total]
    if bad:
        raise ValueError(f"rows out of range: {bad[:5]}")
    return out


def revalidate(report: dict, null_seeds: int = 200, fresh_rows: int = 0, quiet: bool = False) -> dict:
    """Re-fetch the recorded sample (and optionally fresh rows) and recompute validation."""
    best, sample = report.get("best"), report.get("sample")
    if not best or not sample:
        report["validation_note"] = "nothing to validate: report has no hypothesis"
        return report
    m = report["model"]
    ref = hub.ModelRef(m["repo"], m["revision"], m.get("sha"))
    lo, hi = sample["byte_ranges"][0]
    stride = hi - lo + 1
    start = lo - sample["rows"][0] * stride
    tensor = hub.TensorInfo(sample["tensor"], sample["dtype"], tuple(sample["shape"]), sample["file"], start,
                            start + sample["shape"][0] * stride)
    observed, digest = sampler.fetch_rows(ref, tensor, sample["rows"])
    if digest != sample["sha256"]:
        raise ValueError("sample bytes no longer match the recorded SHA-256")
    rec = Recipe(dist=best["dist"], seed=best["seed"], threads=best["threads"], rows=tensor.shape[0],
                 width=tensor.shape[1], offset=best["offset"], dtype=best["dtype"], tp=best["tp"],
                 seed_per_rank=best["seed_per_rank"], params=best["params"])
    rows = list(sample["rows"])
    v0 = report.get("validation") or {}
    fit = v0.get("fit_rows") or stats.split(rows)[0]
    hold = [r for r in rows if r not in set(fit)]
    if fresh_rows:
        log(f"selecting {fresh_rows} fresh rows", quiet)
        sel = sampler.select_rows(ref, tensor, {}, max_rows=fresh_rows + len(rows), rng_seed=1)
        new = [r for r in sel["rows"] if r not in set(rows)][:fresh_rows]
        if new:
            extra, _ = sampler.fetch_rows(ref, tensor, new)
            observed = np.concatenate([observed, extra])
            rows += new
            hold = new
    sweep = {s["threads"]: s["median_r"] for s in report.get("layout_sweep", [])}
    v = stats.validate(rec, rows, observed, fit, hold, sweep, null_seeds=null_seeds, storage_dtype=tensor.dtype)
    cls, reason = stats.classify(v, best.get("fit_median_r") or 0.0)
    report["validation"] = asdict(v)
    report["classification"], report["reason"] = cls, reason
    report["revalidated"] = {"null_seeds": null_seeds, "fresh_rows": len(hold) if fresh_rows else 0,
                             "sample_sha256_verified": True}
    return __import__("json").loads(dumps(report))
