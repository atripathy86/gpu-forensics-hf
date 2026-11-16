"""Stage A: read-only repository inspection and feasibility assessment."""

from __future__ import annotations

import json
import re
from pathlib import Path

import gpuid

from . import hub
from .layouts import catalogue_threads
from .report import ModelInspection

MODELS_JSON = Path(gpuid.__file__).resolve().parent / "models.json"
EMBEDDING_NAMES = re.compile(
    r"(embed_tokens|word_embeddings|tok_embeddings|wte|embed\.weight$|embedding\.weight$|embeddings?\.weight$)")
FLOAT_DTYPES = {"F32", "BF16", "F16"}
RESERVED_TOKEN = re.compile(r"reserved|unused|placeholder|<\|?extra_?\d+|<pad_?\d*>", re.I)
SEED_TEXT = re.compile(r"\bseed\s*(?:[:=]|of|is|was|to)?\s*`?(\d{1,10})\b", re.I)
STD_TEXT = re.compile(
    r"(?:init(?:ializer)?[_ ](?:std|range)|init[_ ]std)\s*[:=]?\s*`?([0-9]*\.[0-9]+(?:e-?\d+)?)", re.I)
INIT_TEXT = re.compile(r"\b(trunc(?:ated)?[_ ]normal|xavier|kaiming|normal_|uniform_|mitchell|small_init)\b", re.I)
PICKLE_SUFFIX = (".bin", ".pt", ".pth", ".ckpt", ".pkl")


def known_recipes() -> list[dict]:
    return json.loads(MODELS_JSON.read_text()) if MODELS_JSON.exists() else []


def find_known(repo: str) -> dict | None:
    for model in known_recipes():
        if model["repo"].lower() == repo.lower():
            return model
    return None


def _text(ref: hub.ModelRef, filename: str, limit: int = 2_000_000) -> str | None:
    try:
        data = hub.http_get(ref.file_url(filename))
    except hub.HubError:
        return None
    return data[:limit].decode("utf-8", "replace")


def _json(ref: hub.ModelRef, filename: str):
    text = _text(ref, filename, limit=200_000_000)
    try:
        return json.loads(text) if text else None
    except json.JSONDecodeError:
        return None


def tokenizer_summary(ref: hub.ModelRef, names: set[str]) -> dict:
    """Vocabulary size and reserved-token ids, read from tokenizer files only."""
    ids: set[int] = set()
    reserved: list[int] = []
    source = None
    if "tokenizer.json" in names:
        tok = _json(ref, "tokenizer.json")
        if tok:
            source = "tokenizer.json"
            vocab = (tok.get("model") or {}).get("vocab") or {}
            if isinstance(vocab, dict):
                ids.update(int(i) for i in vocab.values())
            elif isinstance(vocab, list):
                ids.update(range(len(vocab)))
            for added in tok.get("added_tokens") or []:
                ids.add(int(added["id"]))
                if RESERVED_TOKEN.search(added.get("content", "")):
                    reserved.append(int(added["id"]))
    if "tokenizer_config.json" in names:
        cfg = _json(ref, "tokenizer_config.json") or {}
        for key, added in (cfg.get("added_tokens_decoder") or {}).items():
            ids.add(int(key))
            if RESERVED_TOKEN.search(added.get("content", "")):
                reserved.append(int(key))
        source = source or "tokenizer_config.json"
    if not ids:
        return {"source": source}
    return {"source": source, "size": max(ids) + 1, "distinct_ids": len(ids),
            "reserved_ids": sorted(set(reserved))}


def find_embedding(inventory: dict, vocab_size: int | None) -> hub.TensorInfo | None:
    matrices = [t for t in inventory.values() if len(t.shape) == 2]
    named = [t for t in matrices if EMBEDDING_NAMES.search(t.name) and "position" not in t.name]
    if vocab_size:
        exact = [t for t in named if t.shape[0] == vocab_size] or [t for t in matrices if t.shape[0] == vocab_size]
        if exact:
            return max(exact, key=lambda t: (EMBEDDING_NAMES.search(t.name) is not None, -len(t.name)))
    return max(named, key=lambda t: t.numel) if named else None


def _hints(config: dict, readme: str | None, card: dict) -> dict:
    hints: dict = {}
    for key in ("initializer_range", "init_std", "initializer_factor", "embedding_init_std", "init_method_std"):
        if isinstance(config.get(key), (int, float)):
            hints.setdefault("std", float(config[key]))
            hints.setdefault("std_source", f"config.json:{key}")
    for key in ("seed", "random_seed", "init_seed"):
        if isinstance(config.get(key), int):
            hints.setdefault("seeds", []).append({"seed": config[key], "source": f"config.json:{key}"})
    if readme:
        for m in SEED_TEXT.finditer(readme):
            hints.setdefault("seeds", []).append({"seed": int(m.group(1)), "source": "README.md"})
        for m in STD_TEXT.finditer(readme):
            hints.setdefault("std_mentions", []).append(float(m.group(1)))
        methods = sorted({m.group(1).lower() for m in INIT_TEXT.finditer(readme)})
        if methods:
            hints["init_methods_mentioned"] = methods
    if card.get("base_model"):
        hints["base_model"] = card["base_model"]
    return hints


def inspect(value: str, revision: str | None = None, deep: bool = False, tokenizer: bool = True) -> ModelInspection:
    ref = hub.parse(value, revision)
    result = ModelInspection(model=ref.to_dict())
    try:
        ref, info = hub.resolve(ref)
    except hub.HubError as exc:
        result.diagnostics.append({"stage": "resolve", "kind": exc.kind, "message": str(exc)})
        result.feasibility = {"verdict": "infeasible", "blockers": [f"repository unavailable, private or gated "
                                                                    f"({exc.kind})"], "reasons": []}
        return result
    result.model = ref.to_dict() | {"gated": info.get("gated", False), "tags": info.get("tags", [])}
    names = {s["rfilename"] for s in info.get("siblings", [])}
    card = info.get("cardData") or {}

    known = find_known(ref.repo)
    if known:
        result.known_recipe = {k: v for k, v in known.items() if k not in ("rows", "sha256")} | {
            "rows": len(known["rows"])}
        if known.get("revision") and known["revision"] != ref.sha:
            result.warnings.append(f"known recipe was recorded at revision {known['revision']}, inspecting {ref.sha}")

    config = _json(ref, "config.json") or {}
    text_cfg = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    vocab = config.get("vocab_size") or text_cfg.get("vocab_size")
    result.architecture = {
        "architectures": config.get("architectures", []),
        "model_type": config.get("model_type"),
        "vocab_size": vocab,
        "hidden_size": config.get("hidden_size") or text_cfg.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers") or text_cfg.get("num_hidden_layers"),
        "tie_word_embeddings": config.get("tie_word_embeddings", text_cfg.get("tie_word_embeddings")),
        "torch_dtype": config.get("torch_dtype") or config.get("dtype"),
        "requires_remote_code": bool(config.get("auto_map")),
    }
    gated_blocked = bool(info.get("gated")) and not config
    if not config:
        result.diagnostics.append({"stage": "config", "kind": "gated" if gated_blocked else "missing",
                                   "message": "config.json not readable"})

    st_files = sorted(n for n in names if n.endswith(".safetensors") and "/" not in n)
    weight_map = {}
    if "model.safetensors.index.json" in names:
        index = _json(ref, "model.safetensors.index.json") or {}
        weight_map = index.get("weight_map", {})
        st_files = sorted(set(weight_map.values()))
    pickles = sorted(n for n in names if n.endswith(PICKLE_SUFFIX))
    result.files = {"safetensors": len(st_files), "pickle_checkpoints": len(pickles), "total": len(names),
                    "adapter": "adapter_config.json" in names}

    readme = _text(ref, "README.md") if "README.md" in names else None
    result.hints = _hints(config, readme, card)
    if tokenizer:
        result.tokenizer = tokenizer_summary(ref, names)

    reasons, blockers = [], []
    if gated_blocked:
        blockers.append("gated repository: accept its license on huggingface.co and set HF_TOKEN")
    if not st_files and not gated_blocked:
        blockers.append("no safetensors weights (pickle checkpoints are never loaded)" if pickles
                        else "no weight files found")
    quant = (config.get("quantization_config") or {}).get("quant_method", "unknown") \
        if config.get("quantization_config") else None
    if result.files["adapter"]:
        blockers.append("adapter-only repository; inspect the base model instead")
    tags = {t.lower() for t in info.get("tags", [])}
    if tags & {"merge", "mergekit"}:
        result.warnings.append("merged model: initialization signal is usually destroyed")
    if card.get("base_model"):
        result.warnings.append(f"derived from {card['base_model']}: any fingerprint belongs to the base model's "
                               "initialization")
    if result.architecture["tie_word_embeddings"]:
        result.warnings.append("tied embeddings: unused-token rows still receive output-layer gradients, "
                               "so the signal is likely weak")

    embedding = None
    if st_files and not gated_blocked:
        try:
            if deep or not weight_map:
                inventory = hub.tensor_inventory(ref, st_files)
            else:
                guess = [n for n in weight_map if EMBEDDING_NAMES.search(n) and "position" not in n]
                shards = sorted({weight_map[n] for n in guess}) or st_files[:1]
                inventory = hub.tensor_inventory(ref, shards)
            if deep:
                dtypes: dict = {}
                for t in inventory.values():
                    dtypes[t.dtype] = dtypes.get(t.dtype, 0) + 1
                result.files["tensor_count"] = len(inventory)
                result.files["tensor_dtypes"] = dtypes
                result.files["parameters"] = sum(t.numel for t in inventory.values())
            elif weight_map:
                result.files["tensor_count"] = len(weight_map)
            else:
                result.files["tensor_count"] = len(inventory)
            embedding = find_embedding(inventory, vocab)
        except hub.HubError as exc:
            result.diagnostics.append({"stage": "safetensors", "kind": exc.kind, "message": str(exc)})
            blockers.append(f"safetensors headers unreadable ({exc.kind})")

    if embedding is not None:
        result.embedding = embedding.to_dict()
        if embedding.dtype not in FLOAT_DTYPES:
            blockers.append(f"embedding stored as {embedding.dtype}; only F32/BF16/F16 can be reconstructed")
        elif quant:
            result.warnings.append(f"checkpoint is {quant}-quantized but the embedding is {embedding.dtype}")
        if embedding.numel <= min(catalogue_threads()):
            blockers.append("embedding is smaller than any GPU launch grid: layouts are indistinguishable")
        tok_size = result.tokenizer.get("size")
        padding = embedding.shape[0] - tok_size if tok_size else None
        result.embedding["padding_rows"] = max(padding, 0) if padding is not None else None
        result.embedding["reserved_rows"] = len(result.tokenizer.get("reserved_ids", []))
        if padding and padding > 0:
            reasons.append(f"{padding} embedding rows beyond the tokenizer are likely untrained")
        if result.embedding["reserved_rows"]:
            reasons.append(f"{result.embedding['reserved_rows']} reserved/unused tokens")
        if not padding or padding <= 0:
            reasons.append("no padding rows; untrained rows must be found statistically")
    elif st_files and not blockers:
        blockers.append("no embedding matrix identified")
    unreadable = embedding is None or embedding.dtype not in FLOAT_DTYPES
    if quant and unreadable and not any("stored as" in b for b in blockers):
        blockers.append(f"quantized checkpoint ({quant})")

    seeds = result.hints.get("seeds") or []
    if known:
        reasons.append(f"known recipe: seed {known['seed']}, {known['dist']}")
    elif seeds:
        reasons.append(f"{len(seeds)} seed candidate(s) from metadata")
    else:
        reasons.append("seed unknown: requires search (fingerprint --seed search)")
    if result.architecture["requires_remote_code"]:
        result.warnings.append("repository defines custom modeling code (auto_map); it was not executed and may "
                               "use a custom initializer")

    verdict = "infeasible" if blockers else ("feasible" if known or seeds else "requires_seed_search")
    result.feasibility = {"verdict": verdict, "blockers": blockers, "reasons": reasons}
    return result
