# Validation report

Date: 2026-11-16 · Tool: `gpu-forensics 0.2.0` · Report schema `1.0`

This report records how `gpu-forensics` was validated:
- against the five published checkpoint-forensics examples;
- on blind runs that don't use any recorded recipe;
- across a corpus of 19 public Hugging Face models.

All runs were read-only. Only metadata and byte ranges of safetensors files were
fetched, and no repository code was executed.

## Environment

- Host: aarch64, 20 CPU cores, Python 3.12, `uv`-managed `.venv`
- Dependencies: `numpy 2.5.5`, `scipy 1.18.1` (locked in `uv.lock`)
- Seed-search backend: OpenMP CPU library compiled on demand with gcc 13. The local GB10 GPU was not used.
- Tests: `uv run pytest` gives 58 passed and 1 skipped (GPU fixtures not yet recorded); `uv run pytest -m network` gives 2 passed; `ruff` reports no issues.

## 1. Reproduction of the published examples

`uv run python replay.py` still prints the original table unchanged:

```text
model                     seed   SMs      rows  median r   seed+1     next best  gpu
DeepSeek-V3-Base     953049583   132       8/8  0.999999  -0.0038   183 (0.016)  Hopper
DeepSeek-V4-Flash    953049583   148   461/469  0.999999   0.0016   209 (0.023)  Blackwell B200/B300
Kimi-K3               20256586   152     13/13  0.999996   0.0042    69 (0.018)  Blackwell GB200/GB300
MiniMax-M3                2760   148     17/17  0.999990   0.0017    87 (0.018)  Blackwell B200/B300
Ling-3.1-flash-base       3952   152   384/384  0.983436   0.0004   192 (0.025)  Blackwell GB200/GB300
```

`gpu-forensics fingerprint <url>` uses the recorded recipes and validates on held-out rows:

| Model | Layout `G` | Held-out rows matched | Held-out median r | Bit-exact | Class |
|---|---|---|---|---|---|
| DeepSeek-V3-Base | 270,336 (132 SMs) | 4/4 | 1.000000 | 1.000 | supported |
| DeepSeek-V4-Flash | 303,104 (148 SMs) | 457/465 | 0.999999 | 0.952 | supported |
| Kimi-K3 (offset 3776) | 311,296 (152 SMs) | 9/9 | 0.999995 | 0.668 | supported |
| MiniMax-M3 (tp=4, seed+rank) | 303,104 (148 SMs) | 13/13 | 0.999990 | 0.437 | supported |
| Ling-3.1-flash-base | 311,296 (152 SMs) | 380/380 | 0.999994 | n/a | supported |

**Ling.** Ling now reconstructs at r=0.99999, up from 0.983. On a bf16 tensor,
PyTorch's `uniform_kernel` casts its bounds to bf16 and replaces any value that
rounds to the upper bound with the lower bound. The earlier model missed this,
which produced the outliers that held r at 0.983 (a ≈2% tail of sign-flipped
extremes).

## 2. Blind runs

These runs don't use the rows or recipe in `models.json`. Rows are selected by
`sampler.select_rows` from tokenizer evidence and isolated per-row std clusters.

| Model | Picked rows (std) | Overlap with published rows | Result |
|---|---|---|---|
| DeepSeek-V3-Base | 64 (0.0060) | different rows, same init population | supported, 60/60 held-out, bit-exact |
| DeepSeek-V4-Flash | 64 (0.0060) | 64/64 | supported, 60/60 held-out |
| Kimi-K3 | 13 (2.5e-6) | 13/13 | supported with `--seed 20256586` (offset scan found 3776), 9/9 |
| MiniMax-M3 | 16 (3.6e-4) | 16/16 | supported with `--tp 1,4` (recovers tp=4, seed+rank), 12/12 |
| Ling-3.1-flash-base | 56 (1.4e-7) | 46/56 | supported, 52/52 held-out |

**DeepSeek-V4-Flash.** The tokenizer covers every row of the embedding, so the
untrained rows had to be found by statistics alone. A tight cluster of trained
rows at std 0.143 is rejected in favour of the smaller, isolated init cluster at 0.006.

**Exhaustive search.** `gpu-forensics fingerprint deepseek-ai/DeepSeek-V3-Base --seed search --dist normal`
used no seed hints. It searched all 2³² seeds against 18 layout groups on the CPU,
recovered seed 953049583 with 64/64 sign agreements, and classified the result
`supported`. Wall time was 752 s, while other jobs were running on the same machine.

## 3. Corpus

Command: `tools/corpus.py tools/corpus.txt --fingerprint screen`. It ran in
2 min 40 s with metadata cached. This pass screens metadata seeds and conventions
only; no exhaustive search was run.

| model | verdict | embedding | padding rows | init-like rows | result |
|---|---|---|---|---|---|
| deepseek-ai/DeepSeek-V3-Base | feasible | BF16 [129280, 7168] | 465 | 64 (std 0.005997) | supported: H100 SXM / H800 / H200, seed 953049583 |
| deepseek-ai/DeepSeek-V4-Flash | feasible | BF16 [129280, 4096] | 0 | 64 (std 0.005996) | supported: B200 / B300, seed 953049583 |
| moonshotai/Kimi-K3 | feasible | BF16 [163840, 7168] | 0 | 13 (std 2.466e-06) | inconclusive at offset 0 (needs `--seed 20256586` or `--offsets 3776`) |
| MiniMaxAI/MiniMax-M3 | feasible | BF16 [200064, 6144] | 3 | 16 (std 0.0003578) | supported: B200 / B300, seed 2760 (10/12; `--tp 1,4` gives 12/12) |
| inclusionAI/Ling-3.1-flash-base | feasible | BF16 [157184, 4096] | 31 | 56 (std 1.352e-07) | supported: GB200 / GB300 / MI300X (2-way partition), seed 3952 |
| Qwen/Qwen2.5-0.5B | requires_seed_search | BF16 [151936, 896] | 271 | 6 (std 0.01148) | inconclusive (seed not in screening set) |
| Qwen/Qwen2.5-7B | requires_seed_search | BF16 [152064, 3584] | 399 | 14 (std 0.0006063) | inconclusive (seed not in screening set) |
| allenai/OLMo-2-0425-1B | requires_seed_search | F32 [100352, 2048] | 74 | 64 (std 0.01968) | inconclusive (seed not in screening set) |
| allenai/OLMo-2-1124-7B | requires_seed_search | F32 [100352, 4096] | 74 | 64 (std 0.01974) | inconclusive (seed not in screening set) |
| HuggingFaceTB/SmolLM2-135M | requires_seed_search | BF16 [49152, 576] | 0 | 0 | inconclusive (no init-like rows found) |
| EleutherAI/pythia-160m | requires_seed_search | F16 [50304, 768] | 27 | 64 (std 0.03433) | inconclusive (seed not in screening set) |
| openai-community/gpt2 | requires_seed_search | F32 [50257, 768] | 0 | 21 (std 0.1004) | inconclusive (seed not in screening set) |
| microsoft/phi-2 | requires_seed_search | F16 [51200, 2560] | 905 | 7 (std 0.004904) | inconclusive (seed not in screening set) |
| deepseek-ai/DeepSeek-V2-Lite | requires_seed_search | BF16 [102400, 2048] | 2398 | 64 (std 0.005999) | **supported: A100 / A800, seed 953049583** |
| mistralai/Mistral-7B-v0.1 | requires_seed_search | BF16 [32000, 4096] | 0 | 64 (std 8.48e-36) | **supported: H100 SXM / H800 / H200, seed 0** |
| ibm-granite/granite-3.0-2b-base | requires_seed_search | F32 [49152, 2048] | 0 | 7 (std 0.01625) | inconclusive (seed not in screening set) |
| bigscience/bloom-560m | requires_seed_search | F16 [250880, 1024] | 200 | 5 (std 0.006257) | inconclusive (seed not in screening set) |
| TheBloke/Llama-2-7B-GPTQ | requires_seed_search | F16 [32000, 4096] | 0 | 64 (std 5.617e-06) | inconclusive (seed not in screening set) |
| meta-llama/Llama-3.1-8B | infeasible: gated repository, set HF_TOKEN | | | | |

### Results beyond the published set

The corpus produced two `supported` results for models outside the published set.

**DeepSeek-V2-Lite** (`604d5664`)
- seed 953049583, the same seed as V3 and V4;
- `normal_`, layout G=221,184 (A100/A800-compatible);
- 60/60 held-out rows at r=0.999999, 99.98–100% bit-exact;
- wrong-seed null z=145, runner-up layout r=0.036.

**Mistral-7B-v0.1** (`27d67f1b`)
- seed 0;
- fp32 `trunc_normal_` with the default absolute bounds a=−2, b=2, then cast to bf16;
- layout G=270,336 (H100/H800/H200-compatible);
- 60/60 held-out rows at r=0.999997, null z=277;
- the untrained rows are the unused byte-fallback tokens (ids 82+), shrunk to a std of about 8e-36.

The first Mistral pass assumed the init ran in bf16 and reached only r=0.984.
Trying fp32 init variants closed the gap.

These are statements about the **initialization layout**, which is compatible with
the listed devices. They say nothing about where training ran.

### Inconclusive models

For most of the inconclusive models, rows were found but the seed is not in the
metadata or convention set. The next step is `--seed search`, which takes about
10 minutes per distribution family on this CPU. Other causes of inconclusive results:
- **SmolLM2:** no isolated init cluster was found.
- **Qwen:** tied embeddings weaken the signal.
- **Llama-2-GPTQ:** derived from Llama-2.

## 4. Known gaps

- **GPU ground truth not recorded.** `tools/synthesize_fixtures.py` and `tests/test_fixtures.py` are ready, but recording on the local GB10 was deferred by request.
- **CUDA search backend not validated on hardware.**
- **Philox offsets not predicted automatically.** They are found by an explicit-seed offset scan or supplied with `--offsets`; automatic prediction from module init order is not implemented.
