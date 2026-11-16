"""Phase 0: model reference parsing."""
import pytest

from gpu_forensics import hub


@pytest.mark.parametrize("value,repo,rev", [
    ("bert-base-uncased", "bert-base-uncased", "main"),
    ("https://huggingface.co/bert-base-uncased", "bert-base-uncased", "main"),
    ("https://huggingface.co/Qwen/Qwen2.5-0.5B/", "Qwen/Qwen2.5-0.5B", "main"),
    ("https://huggingface.co/org/model/tree/v1.0", "org/model", "v1.0"),
    ("https://huggingface.co/org/model/blob/abc123/config.json", "org/model", "abc123"),
    ("https://huggingface.co/org/model/resolve/refs%2Fpr%2F1/x.safetensors", "org/model", "refs/pr/1"),
    ("org/model@deadbeef", "org/model", "deadbeef"),
])
def test_parse(value, repo, rev):
    ref = hub.parse(value)
    assert (ref.repo, ref.revision) == (repo, rev)


def test_parse_revision_flag_wins():
    assert hub.parse("org/model@a", "b").revision == "b"


@pytest.mark.parametrize("value", ["https://example.com/org/model", "https://huggingface.co/datasets", "a/b/c/d"])
def test_parse_rejects(value):
    with pytest.raises(ValueError):
        hub.parse(value)

