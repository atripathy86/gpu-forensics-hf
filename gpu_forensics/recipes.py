"""CPU reconstruction of PyTorch CUDA initialization operations.

Element ``i`` of a CUDA ``normal_``/``uniform_`` call is produced by thread
``t = i % G`` from Philox subsequence ``t``; lane ``(i // G) % 4`` of the call
``offset / 4 + i // (4 G)`` supplies the value.  ``gpuid.philox`` is the
reference Philox-4x32-10 implementation (checked against Random123 test vectors).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np
from scipy.special import erfinv

from gpuid import philox

from .layouts import effective_threads

DISTS = ("normal", "uniform", "trunc_normal")
DTYPES = ("float32", "bfloat16", "float16")
_LO = np.uint64(0xFFFFFFFF)
_INV = np.float32(2.0**-32)
_TWO_PI = np.float32(6.2831855)


def bf16(x) -> np.ndarray:
    """Round float32 values to bfloat16 (round-to-nearest-even), returned as float32."""
    bits = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    bits = (bits + np.uint64(0x7FFF) + ((bits >> np.uint64(16)) & np.uint64(1))) >> np.uint64(16)
    return (bits << np.uint64(16)).astype(np.uint32).view(np.float32)


def cast(x, dtype: str) -> np.ndarray:
    x = np.asarray(x, np.float32)
    if dtype == "bfloat16":
        return bf16(x)
    if dtype == "float16":
        return x.astype(np.float16).astype(np.float32)
    return x


def bf16_bits_to_float(bits) -> np.ndarray:
    return (np.asarray(bits, np.uint32) << 16).view(np.float32)


@dataclass(frozen=True)
class Recipe:
    """One initialization operation over a 2-D tensor ``[rows, width]``."""

    dist: str
    seed: int
    threads: int
    rows: int
    width: int
    offset: int = 0
    dtype: str = "float32"
    tp: int = 1
    seed_per_rank: int = 0
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.dist not in DISTS:
            raise ValueError(f"unsupported distribution {self.dist!r}; expected one of {DISTS}")
        if self.dtype not in DTYPES:
            raise ValueError(f"unsupported dtype {self.dtype!r}; expected one of {DTYPES}")

    def with_(self, **changes) -> Recipe:
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def rows_per_shard(self) -> int:
        return -(-self.rows // self.tp)

    def locate(self, row: int) -> tuple[int, int, int]:
        """(seed, local row, numel of the RNG call) for a global row index."""
        shard_rows = self.rows_per_shard
        rank, local = divmod(int(row), shard_rows)
        numel = min(shard_rows, self.rows - rank * shard_rows) * self.width
        return self.seed + rank * self.seed_per_rank, local, numel


def words(pos, threads: int, seed: int, offset: int = 0):
    """Philox output word feeding each flat position, plus its lane."""
    p, g = np.asarray(pos, np.uint64), np.uint64(threads)
    t = p % g
    lane = ((p // g) % np.uint64(4)).astype(np.int64)
    call = np.uint64(offset // 4) + p // (np.uint64(4) * g)
    seed = int(seed)
    w = np.stack(philox(call & _LO, call >> np.uint64(32), t & _LO, t >> np.uint64(32), seed, seed >> 32))
    return w, lane


def unit_normal(pos, threads, seed, offset=0) -> np.ndarray:
    """``curand_normal4`` output (float32): Box-Muller on word pairs (0,1) and (2,3)."""
    w, lane = words(pos, threads, seed, offset)
    i = np.arange(lane.size)
    x = w[np.where(lane < 2, 0, 2), i].astype(np.float32)
    y = w[np.where(lane < 2, 1, 3), i].astype(np.float32)
    u = x * _INV + _INV / np.float32(2)
    tp = _INV * _TWO_PI
    v = y * tp + tp / np.float32(2)
    s = np.sqrt(np.float32(-2) * np.log(u))
    return np.where(lane % 2 == 0, np.sin(v) * s, np.cos(v) * s).astype(np.float32)


def unit_uniform(pos, threads, seed, offset=0) -> np.ndarray:
    """``curand_uniform4`` output (float32) in (0, 1]."""
    w, lane = words(pos, threads, seed, offset)
    return w[lane, np.arange(lane.size)].astype(np.float32) * _INV + _INV / np.float32(2)


def uniform_values(u, low, high, dtype) -> np.ndarray:
    """PyTorch ``uniform_kernel``: bounds cast to the tensor dtype, upper bound folded to lower."""
    frm = cast(np.float32(low), dtype)
    to = cast(np.float32(high), dtype)
    rng = cast(to - frm, dtype)
    value = cast(u * rng + frm, dtype)
    return np.where(value == to, frm, value).astype(np.float32)


def trunc_normal_bounds(mean=0.0, std=1.0, a=-2.0, b=2.0) -> tuple[float, float]:
    """The ``uniform_`` bounds that ``torch.nn.init.trunc_normal_`` passes."""
    cdf = lambda x: (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0  # noqa: E731
    return 2 * cdf((a - mean) / std) - 1, 2 * cdf((b - mean) / std) - 1


def values(recipe: Recipe, row: int, cols=None) -> np.ndarray:
    """Reconstructed initial values for one row (or a column subset), as float64."""
    seed, local, numel = recipe.locate(row)
    cols = np.arange(recipe.width, dtype=np.uint64) if cols is None else np.asarray(cols, np.uint64)
    pos = np.uint64(local * recipe.width) + cols
    g = effective_threads(numel, recipe.threads)
    p, dt = recipe.params, recipe.dtype
    if recipe.dist == "normal":
        z = unit_normal(pos, g, seed, recipe.offset)
        out = cast(z * np.float32(p.get("std", 1.0)) + np.float32(p.get("mean", 0.0)), dt)
    elif recipe.dist == "uniform":
        u = unit_uniform(pos, g, seed, recipe.offset)
        out = uniform_values(u, p.get("low", 0.0), p.get("high", 1.0), dt)
    else:
        mean, std = p.get("mean", 0.0), p.get("std", 1.0)
        a, b = p.get("a", -2.0), p.get("b", 2.0)
        if "bound_sigmas" in p:
            a, b = mean - p["bound_sigmas"] * std, mean + p["bound_sigmas"] * std
        lo, hi = p.get("u_low"), p.get("u_high")
        if lo is None or hi is None:
            lo, hi = trunc_normal_bounds(mean, std, a, b)
        u = uniform_values(unit_uniform(pos, g, seed, recipe.offset), lo, hi, dt)
        with np.errstate(over="ignore", invalid="ignore"):
            x = cast(erfinv(u.astype(np.float64)).astype(np.float32), dt)
            x = cast(x * np.float32(std * math.sqrt(2.0)), dt)
            x = cast(x + np.float32(mean), dt)
        out = np.clip(x, cast(np.float32(a), dt), cast(np.float32(b), dt))
    return out.astype(np.float64)


def matrix(recipe: Recipe, rows, cols=None) -> np.ndarray:
    return np.stack([values(recipe, r, cols) for r in rows])


def corr(a, b) -> float:
    a = np.asarray(a, np.float64) - np.mean(a)
    b = np.asarray(b, np.float64) - np.mean(b)
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / den) if den else 0.0


def row_corr(observed, reconstructed) -> np.ndarray:
    return np.array([corr(o, r) for o, r in zip(observed, reconstructed)])


def exact_fraction(observed, reconstructed, dtype: str) -> float:
    """Fraction of elements reproduced bit-for-bit after fitting a single scale.

    The scale stands in for the unknown ``std``; it is fitted by least squares,
    snapped to the nearest float32, and applied before casting to ``dtype``.
    """
    o = np.asarray(observed, np.float64).ravel()
    r = np.asarray(reconstructed, np.float64).ravel()
    den = float(r @ r)
    if not den:
        return 0.0
    k = np.float32(float(o @ r) / den)
    best = 0.0
    for scale in {k, np.float32(float(f"{k:.3g}")), np.float32(float(f"{k:.2g}")), np.float32(1.0)}:
        rebuilt = cast(r.astype(np.float32) * scale, dtype)
        best = max(best, float(np.mean(rebuilt == cast(o.astype(np.float32), dtype))))
    return best
