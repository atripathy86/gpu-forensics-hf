"""Phase 1: read-only inspection."""
import json

import numpy as np
import pytest

from gpu_forensics import hub
from gpu_forensics.inspector import inspect

from .conftest import REPO, SHA, SHAPE, TOKENIZER_SIZE, safetensors_bytes


def test_inspect_synthetic(fake_hub):
    result = inspect(REPO)
    assert result.model["sha"] == SHA
    assert result.embedding["name"] == "model.embed_tokens.weight"
    assert result.embedding["padding_rows"] == SHAPE[0] - TOKENIZER_SIZE
    assert result.hints["std"] == 0.02
    assert result.feasibility["verdict"] == "requires_seed_search"
    assert result.remote_code_executed is False


def test_inspect_missing_repo(fake_hub):
    fake_hub.info = None

    def fail(url, *a, **k):
        raise hub.HubError("unauthorized", "HTTP 401", url)

    hub.http_get = fail
    result = inspect("nobody/nothing")
    assert result.feasibility["verdict"] == "infeasible"
    assert result.diagnostics[0]["kind"] == "unauthorized"


def test_pickle_only_repo_is_infeasible(fake_hub):
    fake_hub.info["siblings"] = [{"rfilename": "pytorch_model.bin"}, {"rfilename": "config.json"}]
    result = inspect(REPO)
    assert result.feasibility["verdict"] == "infeasible"
    assert "pickle" in result.feasibility["blockers"][0]
    assert not any("pytorch_model.bin" in u for u, *_ in fake_hub.requests)


def test_quantized_embedding_is_infeasible(fake_hub, monkeypatch):


    fake_hub.files["model.safetensors"] = safetensors_bytes(
        {"model.embed_tokens.weight": ("I8", np.zeros(SHAPE, np.int8))})
    fake_hub.files["config.json"] = json.dumps({"vocab_size": SHAPE[0],
                                                "quantization_config": {"quant_method": "bnb"}}).encode()
    result = inspect(REPO)
    assert result.feasibility["verdict"] == "infeasible"


@pytest.mark.network
def test_inspect_real_single_file_model():
    result = inspect("https://huggingface.co/Qwen/Qwen2.5-0.5B")
    assert result.embedding["shape"] == [151936, 896]
    assert result.embedding["padding_rows"] > 0


@pytest.mark.network
def test_inspect_real_single_segment_url():
    result = inspect("https://huggingface.co/bert-base-uncased")
    assert result.model["repo"] in ("bert-base-uncased", "google-bert/bert-base-uncased")
    assert result.embedding is not None

