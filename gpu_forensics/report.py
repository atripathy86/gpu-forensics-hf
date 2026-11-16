"""Report data contracts with deterministic JSON serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone

import numpy as np

from . import __version__

SCHEMA_VERSION = "1.0"


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dumps(value) -> str:
    return json.dumps(_plain(value), indent=2, sort_keys=True)


@dataclass
class ModelInspection:
    model: dict
    files: dict = field(default_factory=dict)
    architecture: dict = field(default_factory=dict)
    embedding: dict | None = None
    tokenizer: dict = field(default_factory=dict)
    hints: dict = field(default_factory=dict)
    known_recipe: dict | None = None
    warnings: list = field(default_factory=list)
    diagnostics: list = field(default_factory=list)
    feasibility: dict = field(default_factory=dict)
    remote_code_executed: bool = False


@dataclass
class TensorSample:
    tensor: str
    dtype: str
    shape: list
    file: str
    rows: list
    byte_ranges: list
    sha256: str
    rng_position: str
    selection: dict = field(default_factory=dict)


@dataclass
class CandidateHypothesis:
    seed: int
    threads: int
    dist: str
    offset: int = 0
    dtype: str = "float32"
    tp: int = 1
    seed_per_rank: int = 0
    params: dict = field(default_factory=dict)
    fit_median_r: float | None = None
    devices: str = ""
    configs: list = field(default_factory=list)


@dataclass
class ValidationResult:
    fit_rows: list
    holdout_rows: list
    holdout_median_r: float
    holdout_min_r: float
    rows_matched: str
    exact_fraction: float | None
    aggregate_r: float
    fisher_z: float
    null: dict
    runner_up: dict
    indistinguishable_threads: list
    warnings: list = field(default_factory=list)


@dataclass
class ForensicReport:
    model: dict
    classification: str
    reason: str
    best: CandidateHypothesis | None = None
    validation: ValidationResult | None = None
    sample: TensorSample | None = None
    seed_search: dict | None = None
    layout_sweep: list = field(default_factory=list)
    inspection: dict | None = None
    limitations: list = field(default_factory=list)
    runtime_seconds: float | None = None
    schema_version: str = SCHEMA_VERSION
    tool: str = f"gpu-forensics {__version__}"
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return dumps(self)


LIMITATIONS = [
    "Identifies the thread layout of the GPU that initialized the weights, not the hardware used for training.",
    "Devices with the same SMs x threads-per-SM product produce identical layouts (e.g. GB200/GB300 and a "
    "2-way-partitioned MI300X), so a match is not proof of a vendor or product.",
    "CPU, meta-device or custom-kernel initialization cannot be fingerprinted by this method.",
]


def render(report: dict) -> str:
    """Human-readable summary of a report dict."""
    m = report.get("model", {})
    lines = [f"model:          {m.get('repo')} @ {m.get('sha') or m.get('revision')}",
             f"classification: {report.get('classification')}",
             f"reason:         {report.get('reason')}"]
    best = report.get("best")
    if best:
        lines += [f"seed:           {best['seed']}  dist={best['dist']}  offset={best['offset']}  "
                  f"tp={best['tp']}",
                  f"layout:         G={best['threads']} threads  ({best['devices']})",
                  "configs:        " + ", ".join(f"{s} SMs x {t}" for s, t in best["configs"])]
    v = report.get("validation")
    if v:
        lines += [f"held-out:       median r={v['holdout_median_r']:.6f} over {len(v['holdout_rows'])} rows "
                  f"({v['rows_matched']} matched)",
                  f"null:           seed+k pooled r mean={v['null'].get('mean', 0):+.4f} "
                  f"sd={v['null'].get('sd', 0):.4f}  p={v['null'].get('p_value')}",
                  f"runner-up:      G={v['runner_up'].get('threads')} r={v['runner_up'].get('median_r')}",
                  f"aggregate:      r={v['aggregate_r']:.6f}  Fisher z={v['fisher_z']:.1f}"]
        if v.get("exact_fraction") is not None:
            lines.append(f"bit-exact:      {v['exact_fraction']:.3f} of held-out elements")
    s = report.get("seed_search")
    if s:
        lines.append(f"seed search:    {s.get('status')}  {s.get('summary', '')}")
    for w in (v or {}).get("warnings", []) + (report.get("inspection") or {}).get("warnings", []):
        lines.append(f"warning:        {w}")
    return "\n".join(lines)
