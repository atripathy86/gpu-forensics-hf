"""Phase 2: range reads and untrained-row selection."""
import numpy as np

from gpu_forensics import hub, sampler
from gpu_forensics.recipes import values as rv

from .conftest import REPO, SHAPE, TOKENIZER_SIZE, safetensors_bytes, synthetic_embedding


def test_range_reads_and_decode(fake_hub):
    ref, _ = hub.resolve(hub.parse(REPO))
    inv = hub.tensor_inventory(ref, ["model.safetensors"])
    t = inv["model.embed_tokens.weight"]
    assert t.shape == SHAPE and t.dtype == "BF16"
    values, digest = sampler.fetch_rows(ref, t, [4000, 4001, 4095])
    expected = fake_hub.recipe

    np.testing.assert_allclose(values[0], rv(expected, 4000), rtol=0, atol=0)
    assert len(digest) == 64
    # contiguous rows are fetched in one request
    ranged = [r for r in fake_hub.requests if r[1] is not None and r[1] > 1000]
    assert len(ranged) == 2


def test_select_rows_finds_untrained(fake_hub):
    ref, _ = hub.resolve(hub.parse(REPO))
    t = hub.tensor_inventory(ref, ["model.safetensors"])["model.embed_tokens.weight"]
    sel = sampler.select_rows(ref, t, {"size": TOKENIZER_SIZE}, max_rows=32)
    assert len(sel["rows"]) == 32
    assert all(r >= TOKENIZER_SIZE for r in sel["rows"])
    assert abs(sel["init_scale"] - 0.02) < 0.001


def _embedding_tensor(fake_hub, trained_scale):
    bits, _ = synthetic_embedding(trained_scale=trained_scale)
    fake_hub.files["model.safetensors"] = safetensors_bytes({"model.embed_tokens.weight": ("BF16", bits)})
    ref, _ = hub.resolve(hub.parse(REPO))
    return ref, hub.tensor_inventory(ref, ["model.safetensors"])["model.embed_tokens.weight"]


def test_select_rows_without_tokenizer(fake_hub):
    """Statistics alone find the init rows when their std stands apart from trained rows."""
    ref, t = _embedding_tensor(fake_hub, (1.5, 4.0))
    sel = sampler.select_rows(ref, t, {}, max_rows=16, tail=256)
    assert sel["rows"] and all(r >= TOKENIZER_SIZE for r in sel["rows"])


def test_select_rows_refuses_ambiguous_population(fake_hub):
    """Without tokenizer evidence, an init std buried inside the trained population is not claimed."""
    ref, t = _embedding_tensor(fake_hub, (0.7, 3.0))
    sel = sampler.select_rows(ref, t, {}, max_rows=16, tail=256)
    assert not [r for r in sel["rows"] if r < TOKENIZER_SIZE]


def test_init_cluster():
    std = np.concatenate([np.full(10, 0.006) * (1 + np.linspace(-0.01, 0.01, 10)), np.linspace(0.01, 0.1, 50)])
    scale, mask = sampler.init_cluster(std)
    assert abs(scale - 0.006) < 1e-4 and mask[:10].all() and mask[10:].sum() < 3



def test_smallest_isolated_cluster_preferred():
    """A broad trained population and a dense secondary cluster must not beat a small init cluster."""
    rng = np.random.default_rng(0)
    std = np.concatenate([np.full(13, 2.47e-6) * (1 + rng.normal(0, 0.005, 13)),  # weight-decayed init rows
                          0.143 * (1 + rng.normal(0, 0.005, 300)),                 # second tight cluster
                          np.exp(rng.uniform(np.log(0.01), np.log(0.1), 2000))])   # trained rows
    found = sampler.clusters(std, 0.03)
    assert abs(found[0][0] - 2.47e-6) < 1e-7 and found[0][1][:13].all()
    assert any(abs(scale - 0.143) < 0.005 for scale, _ in found)
