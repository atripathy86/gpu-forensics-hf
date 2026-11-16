"""Phase 3: CPU reconstruction of CUDA init kernels and layout model."""
import numpy as np
import pytest

import gpuid
from gpu_forensics import layouts
from gpu_forensics.recipes import Recipe, bf16, uniform_values, unit_normal, unit_uniform


def test_unit_normal_matches_upstream_draw():
    pos = np.random.default_rng(0).integers(0, 10**9, 5000).astype(np.uint64)
    for g in (221184, 270336, 303104):
        np.testing.assert_array_equal(unit_normal(pos, g, 953049583, 3776), gpuid.draw(pos, g, 953049583, 3776))


def test_first_g_positions_are_layout_independent():
    pos = np.arange(1000, dtype=np.uint64)
    np.testing.assert_array_equal(unit_normal(pos, 32768, 5), unit_normal(pos, 311296, 5))


def test_bf16_round_to_nearest_even():
    x = np.array([1.0, 1.00390625, 1.005859375, -2.5e-8, 3.0e38], np.float32)
    out = bf16(x)
    assert out[0] == 1.0
    assert out[1] == 1.0  # tie rounds to even
    assert out[2] == np.float32(1.0078125)
    assert np.all((out.view(np.uint32) & 0xFFFF) == 0)


def test_uniform_upper_bound_folds_to_lower():
    u = np.array([1.0, 0.5, 2.0**-33], np.float32)
    v = uniform_values(u, -1.0, 1.0, "float32")
    assert v[0] == -1.0 and v[1] == 0.0


def test_uniform_in_range():
    u = unit_uniform(np.arange(10000, dtype=np.uint64), 270336, 1)
    assert u.min() > 0 and u.max() <= 1


def test_recipe_validation():
    with pytest.raises(ValueError):
        Recipe(dist="laplace", seed=0, threads=1, rows=1, width=1)
    with pytest.raises(ValueError):
        Recipe(dist="normal", seed=0, threads=1, rows=1, width=1, dtype="int8")


def test_tensor_parallel_locate():
    rec = Recipe(dist="normal", seed=2760, threads=1, rows=200064, width=8, tp=4, seed_per_rank=1)
    assert rec.locate(0) == (2760, 0, 50016 * 8)
    assert rec.locate(150048) == (2763, 0, 50016 * 8)


def test_layout_helpers():
    assert layouts.effective_threads(10**9, 132 * 2048) == 270336
    assert layouts.effective_threads(1000, 270336) == 1024  # small tensors never reach the cap
    assert layouts.philox_offset_increment(1000, 270336) == 4
    assert layouts.philox_offset_increment(270336 * 4 * 3, 270336) == 12
    assert (132, 2048) in layouts.equivalent_configs(270336)
    assert "GB200" in layouts.describe(311296) and "MI300X" in layouts.describe(311296)
    assert layouts.Device("GB10", 48, 1536).threads == 73728

