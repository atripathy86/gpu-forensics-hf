# Validation report

Date: 2026-11-16

This report records local validation of the `atripathy86/gpu-forensics-hf`
fork against the published [Unemployment Labs checkpoint-forensics report](https://unemploymentlabs.com/checkpoint-forensics)
and the upstream [macrogrants/checkpoint-forensics](https://github.com/macrogrants/checkpoint-forensics) reference implementation.

## Environment

- Fork: `git@github.com:atripathy86/gpu-forensics-hf.git`
- Upstream: `git@github.com:macrogrants/checkpoint-forensics.git`
- `uv`: 0.12.3
- Environment: `.venv/` created and managed by `uv sync`
- Dependencies: `numpy 2.5.5`, `scipy 1.18.1` from `uv.lock`
- Samples: pinned `rows/` files validated against the SHA-256 hashes in `models.json`
- Remote model code execution: none

## Reproduction with uv

From the repository root, recreate the locked environment and run every command
through `uv`:

```bash
uv sync
uv run python replay.py
uv run gpu-forensics inspect https://huggingface.co/bert-base-uncased
uv run gpu-forensics fingerprint https://huggingface.co/deepseek-ai/DeepSeek-V3-Base \
  --output /tmp/deepseek-report.json
uv run gpu-forensics validate /tmp/deepseek-report.json
```

The pinned model examples can be selected by URL as documented in `README.md`.
`REFETCH=1 uv run python replay.py` re-fetches the pinned rows and verifies
 their hashes.

## Upstream replay

Command:

```text
uv run python replay.py
```

Observed output:

```text
model                     seed   SMs      rows  median r   seed+1     next best  gpu
DeepSeek-V3-Base     953049583   132       8/8  0.999999  -0.0038   183 (0.016)  Hopper
DeepSeek-V4-Flash    953049583   148   461/469  0.999999   0.0016   209 (0.023)  Blackwell B200/B300
Kimi-K3               20256586   152     13/13  0.999996   0.0042    69 (0.018)  Blackwell GB200/GB300
MiniMax-M3                2760   148     17/17  0.999990   0.0017    87 (0.018)  Blackwell B200/B300
Ling-3.1-flash-base       3952   152   384/384  0.983436   0.0004   192 (0.025)  Blackwell GB200/GB300
```

These results match the upstream README and the Unemployment Labs report:
132 SMs for DeepSeek-V3, 148 for DeepSeek-V4-Flash and MiniMax-M3, and 152
for Kimi K3 and Ling-3.1-flash-base. The match counts, correlations, control
seed behavior, and runner-up examples also match the published reference.

## New CLI

Known-model fingerprint command:

```text
uv run gpu-forensics fingerprint \
  https://huggingface.co/deepseek-ai/DeepSeek-V3-Base \
  --output /tmp/deepseek-report.json
```

Relevant output:

```json
{
  "best_sms": 132,
  "gpu_layout": "Hopper (H100/H800/H200-compatible)",
  "median_correlation": 0.9999986288599336,
  "next_best_sms": 183,
  "rows_matched": "8/8",
  "seed": 953049583,
  "seed_plus_one_median_correlation": -0.0037853181504305636,
  "threads": 270336
}
```

The CLI delegates known-model reconstruction to the retained upstream
`gpuid.py`/`fetch.py` implementation, so its result is numerically consistent
with `replay.py`. It adds structured provenance and classification output.

Generic inspection was also exercised with `uv run gpu-forensics inspect
https://huggingface.co/bert-base-uncased` against the public `bert-base-uncased`
repository. It read `config.json`, reported the `BertForMaskedLM`
architecture, reported the missing safetensors index as a diagnostic, and
returned `feasible: false` without executing remote code. This is the intended
safe/inconclusive behavior for models without a usable forensic input path.

Validation of a saved report:

```text
uv run gpu-forensics validate /tmp/deepseek-report.json
```

The result records `held_out: false` because the published upstream samples
are fitting/reference samples rather than a separate held-out partition. The
tool therefore preserves the strong published replay evidence without making
an unsupported claim of independent held-out validation.

## Interpretation

The implementation faithfully reproduces the published numerical fingerprints
for the five supplied models. As stated by the source report, these are
fingerprints of initialization layout and support—but do not alone prove—the
GPU vendor, exact product, or the hardware used for later training.
