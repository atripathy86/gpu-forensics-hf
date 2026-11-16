"""Read-only CLI for GPU initialization-layout forensics.

The known-model path deliberately delegates reconstruction to the forked
upstream implementation in ``gpuid.py`` and ``fetch.py``.  The generic path
inspects Hugging Face metadata and reports whether enough information exists
for a defensible analysis; it never executes repository code.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fetch import rows
from gpuid import corr, draw

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models.json"
GPU = {
    108: "A100/A800",
    132: "Hopper (H100/H800/H200-compatible)",
    148: "Blackwell B200/B300-compatible",
    152: "Blackwell GB200/GB300-compatible",
}


def load_models():
    return json.loads(MODELS.read_text())


def resolve_model(value):
    models = load_models()
    for model in models:
        if value == model["name"] or value.rstrip("/") in {
            "https://huggingface.co/" + model["repo"],
            model["repo"],
        }:
            return model
    return None


def hf_get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "gpu-forensics/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def model_ref(value):
    value = value.rstrip("/")
    match = re.match(r"https?://huggingface\.co/([^/]+/[^/]+)(?:/tree/([^/]+))?$", value)
    if match:
        return match.group(1), match.group(2) or "main"
    if not value.startswith("http") and value:
        return value, "main"
    raise ValueError("expected a Hugging Face URL or repository ID")


def inspect(value):
    repo, revision = model_ref(value)
    base = f"https://huggingface.co/{repo}/resolve/{revision}/"
    result = {
        "repository": repo,
        "revision": revision,
        "remote_code_executed": False,
        "feasible": False,
        "diagnostics": [],
    }
    for filename in ("config.json", "model.safetensors.index.json"):
        try:
            result[filename] = json.loads(hf_get(base + filename))
        except Exception as exc:  # network and missing-file errors are report data
            result["diagnostics"].append(f"{filename}: {exc}")
    config = result.get("config.json", {})
    index = result.get("model.safetensors.index.json", {})
    result["architecture"] = config.get("architectures", [])
    result["tensor_count"] = len(index.get("weight_map", {}))
    result["feasible"] = bool(index.get("weight_map"))
    result["diagnostics"].append(
        "A seed and initialization operation are still required for fingerprinting."
    )
    return result


def match(model, observed, sms, shift=0):
    volume, width = model["shape"]
    shard_rows = -(-volume // model.get("tp", 1))
    scores = []
    for row, values in zip(model["rows"], observed):
        position = np.uint64(row % shard_rows * width) + np.arange(width, dtype=np.uint64)
        seed = model["seed"] + shift + row // shard_rows * model.get("seed_per_rank", 0)
        scores.append(corr(values, draw(position, sms * 2048, seed, model.get("cpu_offset", 0), model["dist"])))
    return np.asarray(scores)


def fingerprint(value, output=None):
    model = resolve_model(value)
    if model is None:
        result = inspect(value)
        result.update({"classification": "inconclusive", "reason": "No known seed/layout recipe is available."})
    else:
        observed = rows(model)
        fitting = dict(model, rows=model["rows"][:4])
        sweep = {}
        for sms in range(16, 257):
            sweep[sms] = float(np.median(match(fitting, observed[:4], sms)))
        best, second = sorted(sweep, key=sweep.get, reverse=True)[:2]
        scores = match(model, observed, best)
        control = match(model, observed, best, 1)
        threshold = model.get("thr", 0.99)
        result = {
            "model": model["name"],
            "repository": model["repo"],
            "revision": model["revision"],
            "seed": model["seed"],
            "best_sms": best,
            "threads": best * 2048,
            "gpu_layout": GPU.get(best, "unknown layout"),
            "rows_matched": f"{int((scores > threshold).sum())}/{len(scores)}",
            "median_correlation": float(np.median(scores)),
            "seed_plus_one_median_correlation": float(np.median(control)),
            "next_best_sms": second,
            "next_best_median_correlation": sweep[second],
            "classification": "supported" if float(np.median(scores)) > 0.9 else "weak",
            "validation": "This run uses the upstream fitting/sample procedure; use validate for a held-out check.",
        }
    result["tool"] = "atripathy86/gpu-forensics-hf"
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    text = json.dumps(result, indent=2, sort_keys=True)
    if output:
        Path(output).write_text(text + "\n")
    print(text)
    return result


def validate(path, output=None):
    report = json.loads(Path(path).read_text())
    report["validation"] = {
        "status": "limited",
        "held_out": False,
        "note": "The published samples do not contain a separate held-out partition; upstream replay evidence is preserved, but this report does not claim independent validation.",
        "inconclusive_if": ["unknown seed", "transformed/quantized weights", "no preserved initialization signal"],
    }
    report["classification"] = report.get("classification", "inconclusive")
    text = json.dumps(report, indent=2, sort_keys=True)
    if output:
        Path(output).write_text(text + "\n")
    print(text)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gpu-forensics")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect_parser = sub.add_parser("inspect")
    inspect_parser.add_argument("model")
    fingerprint_parser = sub.add_parser("fingerprint")
    fingerprint_parser.add_argument("model")
    fingerprint_parser.add_argument("--output")
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("report")
    validate_parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            print(json.dumps(inspect(args.model), indent=2, sort_keys=True))
        elif args.command == "fingerprint":
            fingerprint(args.model, args.output)
        else:
            validate(args.report, args.output)
    except (ValueError, urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
