"""Stage C: seed discovery.

Candidates are tried in increasing cost: explicit seeds, metadata hints and
conventions (scored directly), then an exhaustive, resumable sign-pattern search
over the 32-bit seed space for each layout hypothesis.  Survivors of the search
are confirmed by full reconstruction before being reported.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import hub
from ..layouts import effective_threads
from ..recipes import Recipe

HERE = Path(__file__).resolve().parent
DIST_CODE = {"normal": 0, "uniform": 1, "trunc_normal": 1}
SEED_SPACE = 2**32

# Seeds seen in published models and common conventions; cheap to try first.
CONVENTIONAL = [0, 1, 2, 3, 7, 10, 42, 123, 233, 666, 777, 1000, 1024, 1234, 1337, 2020, 2021, 2022, 2023, 2024,
                2025, 2026, 2048, 3407, 4096, 12345, 31337, 54321, 65536, 114514, 123456, 1234567, 12345678,
                20256586, 953049583, 2760, 3952, 2**31 - 1, 2**32 - 1]


def _build(source: str, compiler: list[str]) -> Path | None:
    src = HERE / source
    digest = hashlib.sha256(src.read_bytes() + " ".join(compiler).encode()).hexdigest()[:16]
    out = hub.cache_dir() / "build" / f"{src.stem}-{digest}.so"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.so")
    try:
        subprocess.run([*compiler, str(src), "-o", str(tmp)], check=True, capture_output=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return None
    tmp.rename(out)
    return out


def _load(path: Path):
    lib = ctypes.CDLL(str(path))
    fn = lib.philox_search
    p = np.ctypeslib.ndpointer
    fn.argtypes = [p(np.uint64), p(np.uint64), p(np.int32), p(np.uint8), ctypes.c_int32, ctypes.c_int32,
                   ctypes.c_int32, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int32, p(np.uint64), p(np.int32),
                   p(np.int32), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64)]
    fn.restype = ctypes.c_int64
    return fn


def backend(name: str = "auto"):
    """Return ``(name, callable)`` for the requested search backend.

    ``auto`` selects the CPU (OpenMP) library and falls back to NumPy.  The CUDA
    backend is only used when requested explicitly."""
    if name == "cuda":
        nvcc = shutil.which("nvcc")
        lib = nvcc and _build("philox_search.cu", [nvcc, "-O3", "-shared", "-Xcompiler", "-fPIC"])
        if not lib:
            raise RuntimeError("CUDA backend unavailable (nvcc build failed)")
        return "cuda", _load(lib)
    if name in ("auto", "cpu"):
        cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
        lib = cc and _build("philox_search.c", [cc, "-O3", "-fopenmp", "-shared", "-fPIC"])
        if lib:
            return "cpu", _load(lib)
        if name == "cpu":
            raise RuntimeError("CPU backend unavailable (no C compiler with OpenMP)")
    return "numpy", _numpy_search


def _philox_keys(c0, c1, c2, c3, k0, k1):
    """Philox-4x32-10 broadcasting over key arrays (NumPy fallback)."""
    lo, s = np.uint64(0xFFFFFFFF), np.uint64(32)
    c0, c1, c2, c3 = (np.asarray(x, np.uint64) & lo for x in (c0, c1, c2, c3))
    k0, k1 = np.asarray(k0, np.uint64) & lo, np.asarray(k1, np.uint64) & lo
    for _ in range(10):
        p0, p1 = np.uint64(0xD2511F53) * c0, np.uint64(0xCD9E8D57) * c2
        c0, c1, c2, c3 = (p1 >> s) ^ c1 ^ k0, p1 & lo, (p0 >> s) ^ c3 ^ k1, p0 & lo
        k0, k1 = (k0 + np.uint64(0x9E3779B9)) & lo, (k1 + np.uint64(0xBB67AE85)) & lo
    return np.stack([c0, c1, c2, c3])


def sign_bits(w: np.ndarray, lane: np.ndarray, dist: int) -> np.ndarray:
    """Sign model shared with the native kernels; ``w`` is ``[4, ...]``."""
    idx = np.indices(lane.shape)
    if dist == 0:
        y = np.where(lane < 2, w[1], w[3])
        odd = (y < 0x40000000) | (y >= 0xC0000000)
        return np.where(lane % 2 == 1, odd, y < 0x80000000).astype(np.uint8)
    return (w[(lane, *idx)] >= 0x80000000).astype(np.uint8)


def _numpy_search(t, call, lane, expect, n, groups, dist, seed_lo, seed_hi, max_miss, out_seed, out_group,
                  out_score, max_out, total):
    hits = 0
    t, call, lane, expect = (a.reshape(groups, n) for a in (t, call, lane, expect))
    for lo in range(seed_lo, seed_hi, 1 << 16):
        seeds = np.arange(lo, min(seed_hi, lo + (1 << 16)), dtype=np.uint64)[:, None]
        for g in range(groups):
            w = _philox_keys(call[g] & 0xFFFFFFFF, call[g] >> np.uint64(32), t[g] & 0xFFFFFFFF,
                             t[g] >> np.uint64(32), seeds, seeds >> np.uint64(32))
            miss = (sign_bits(w, np.broadcast_to(lane[g], w.shape[1:]), dist) != expect[g]).sum(1)
            for s, m in zip(seeds[miss <= max_miss, 0], miss[miss <= max_miss]):
                if hits < max_out:
                    out_seed[hits], out_group[hits], out_score[hits] = s, g, n - m
                hits += 1
    total._obj.value = hits
    return min(hits, max_out)


@dataclass
class Hypothesis:
    """One layout/sharding assumption under which the seed is searched."""

    threads: int
    tp: int = 1
    offset: int = 0
    label: dict = field(default_factory=dict)


def plan_positions(rows, cols_by_row, observed, shape, dist: str, hypotheses: list[Hypothesis]):
    """Per-hypothesis (t, call, lane) arrays for the sampled positions.

    Hypotheses whose arrays coincide (e.g. all positions below G) are merged."""
    rows_total, width = shape
    expect, flat_rows, flat_cols = [], [], []
    centre = float(np.median(observed))
    for i, (r, cols) in enumerate(zip(rows, cols_by_row)):
        flat_rows += [r] * len(cols)
        flat_cols += list(cols)
        expect += list((observed[i][cols] > centre).astype(np.uint8))
    flat_rows, flat_cols = np.asarray(flat_rows, np.uint64), np.asarray(flat_cols, np.uint64)
    groups, keys = [], {}
    for h in hypotheses:
        shard_rows = -(-rows_total // h.tp)
        ranks = {int(r) // shard_rows for r in flat_rows}
        if len(ranks) != 1:
            continue  # a single search assumes one seed: all rows must share a shard
        rank = ranks.pop()
        numel = min(shard_rows, rows_total - rank * shard_rows) * width
        g = np.uint64(effective_threads(numel, h.threads))
        pos = (flat_rows % np.uint64(shard_rows)) * np.uint64(width) + flat_cols
        t = pos % g
        lane = ((pos // g) % np.uint64(4)).astype(np.int32)
        call = np.uint64(h.offset // 4) + pos // (np.uint64(4) * g)
        key = hashlib.sha256(t.tobytes() + lane.tobytes() + call.tobytes()).hexdigest()
        if key in keys:
            groups[keys[key]][1].append(h)
            continue
        keys[key] = len(groups)
        groups.append(((t, call, lane), [h], rank))
    return np.asarray(expect, np.uint8), groups


def pick_positions(observed: np.ndarray, rows, n: int = 48, seed: int = 0):
    """Choose ``n`` positions spread over rows, avoiding values near the centre."""
    rng = np.random.default_rng(seed)
    per = max(1, -(-n // len(rows)))
    centre = float(np.median(observed))
    spread = float(np.std(observed)) or 1.0
    cols_by_row = []
    for row in observed:
        clear = np.flatnonzero(np.abs(row - centre) > 0.25 * spread)
        pool = clear if clear.size >= per else np.arange(row.size)
        cols_by_row.append(np.sort(rng.choice(pool, min(per, pool.size), replace=False)))
    total = sum(len(c) for c in cols_by_row)
    while total > n:  # trim to exactly n
        i = int(np.argmax([len(c) for c in cols_by_row]))
        cols_by_row[i] = cols_by_row[i][:-1]
        total -= 1
    return cols_by_row


def search(observed, rows, shape, dist: str, hypotheses: list[Hypothesis], seed_range=(0, SEED_SPACE),
           n: int = 48, max_miss: int = 4, chunk: int = 1 << 28, budget_seconds: float | None = None,
           checkpoint: Path | None = None, backend_name: str = "auto", progress=None) -> dict:
    """Exhaustive sign-pattern search.  Resumable via ``checkpoint``."""
    name, fn = backend(backend_name)
    cols_by_row = pick_positions(observed, rows, n)
    expect1, groups = plan_positions(rows, cols_by_row, observed, shape, dist, hypotheses)
    if not groups:
        return {"status": "skipped", "summary": "no hypothesis keeps the sampled rows in one shard"}
    n_eff = len(expect1)
    t = np.ascontiguousarray(np.concatenate([g[0][0] for g in groups]), np.uint64)
    call = np.ascontiguousarray(np.concatenate([g[0][1] for g in groups]), np.uint64)
    lane = np.ascontiguousarray(np.concatenate([g[0][2] for g in groups]), np.int32)
    expect = np.ascontiguousarray(np.tile(expect1, len(groups)), np.uint8)

    state = {"done": [], "hits": []}
    fingerprint = hashlib.sha256(t.tobytes() + call.tobytes() + lane.tobytes() + expect.tobytes()
                                 + f"{dist}{max_miss}".encode()).hexdigest()[:16]
    if checkpoint and Path(checkpoint).exists():
        saved = json.loads(Path(checkpoint).read_text())
        if saved.get("fingerprint") == fingerprint:
            state = saved
    lo, hi = seed_range
    start = time.time()
    status = "complete"
    max_out = 4096
    for a in range(lo, hi, chunk):
        b = min(hi, a + chunk)
        if [a, b] in state["done"]:
            continue
        if budget_seconds is not None and time.time() - start > budget_seconds:
            status = "budget_exhausted"
            break
        out_seed = np.zeros(max_out, np.uint64)
        out_group = np.zeros(max_out, np.int32)
        out_score = np.zeros(max_out, np.int32)
        total = ctypes.c_int64(0)
        written = fn(t, call, lane, expect, n_eff, len(groups), DIST_CODE[dist], a, b, max_miss,
                     out_seed, out_group, out_score, max_out, ctypes.byref(total))
        if written < 0:
            raise RuntimeError(f"{name} search backend failed")
        for s, g, sc in zip(out_seed[:written], out_group[:written], out_score[:written]):
            state["hits"].append([int(s), int(g), int(sc)])
        state["done"].append([a, b])
        state["fingerprint"] = fingerprint
        if checkpoint:
            Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
            Path(checkpoint).write_text(json.dumps(state))
        if progress:
            progress(a, b, hi, len(state["hits"]))
    covered = sum(b - a for a, b in state["done"])
    hits = [{"seed": s, "score": sc, "rank": groups[g][2],
             "hypotheses": [{"threads": h.threads, "tp": h.tp, "offset": h.offset} for h in groups[g][1]]}
            for s, g, sc in state["hits"]]
    return {
        "status": status if covered < hi - lo or status != "complete" else "complete",
        "backend": name,
        "positions": n_eff,
        "max_mismatches": max_miss,
        "layout_groups": len(groups),
        "seeds_covered": covered,
        "seed_range": [lo, hi],
        "elapsed_seconds": round(time.time() - start, 2),
        "hits": hits,
        "summary": f"{covered:,} seeds x {len(groups)} layout groups, {len(hits)} sign-pattern hits",
    }


def expected_false_hits(n: int, max_miss: int, seeds: int, groups: int) -> float:
    from math import comb
    p = sum(comb(n, k) for k in range(max_miss + 1)) / 2**n
    return p * seeds * groups


def hypotheses_for(threads_list, tps=(1,), offsets=(0,)) -> list[Hypothesis]:
    return [Hypothesis(g, tp, o) for g in threads_list for tp in tps for o in offsets]


def recipe_from_hit(hit, hyp: dict, dist, shape, dtype, params) -> Recipe:
    return Recipe(dist=dist, seed=hit["seed"], threads=hyp["threads"], rows=shape[0], width=shape[1],
                  offset=hyp["offset"], dtype=dtype, tp=hyp["tp"], params=params)
