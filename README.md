# checkpoint-forensics

This is the code for my report, [Top Chinese Frontier AI Models Carry the Fingerprint of Export-Controlled NVIDIA Blackwell GPUs](https://unemploymentlabs.com/checkpoint-forensics).

Almost every AI model carries a hidden signature of the GPU that created it. You can use this to check that yourself for five models from DeepSeek, Moonshot, MiniMax and Ant Group and it runs on a laptop in a few seconds.

This repository extends the replay of [checkpoint-forensics](https://github.com/macrogrants/checkpoint-forensics)
with the **[`gpu-forensics` CLI](#gpu-forensics-cli)**, which runs the same
analysis on any Hugging Face model. It finds untrained rows automatically,
recovers unknown seeds, validates on held-out rows and writes JSON reports.

- [Replay of the published models](#replay-of-the-published-models)
- [How it works](#how-it-works)
- [On a real GPU](#on-a-real-gpu)
- [Notes](#notes)
- [gpu-forensics CLI](#gpu-forensics-cli): [commands](#commands), [classification rules](#classification-rules), [results](#results-on-published-models)

## Replay of the published models

    pip install numpy scipy
    python replay.py

With uv: `uv sync && uv run python replay.py`.

| Model | Seed | SMs | Rows matched | Median r | Seed+1 | Next best | GPU |
|---|---|---|---|---|---|---|---|
| DeepSeek-V3-Base | 953049583 | 132 | 8/8 | 0.999999 | -0.0038 | 183 (0.016) | Hopper |
| DeepSeek-V4-Flash | 953049583 | 148 | 461/469 | 0.999999 | 0.0016 | 209 (0.023) | Blackwell B200/B300 |
| Kimi-K3 | 20256586 | 152 | 13/13 | 0.999996 | 0.0042 | 69 (0.018) | Blackwell GB200/GB300 |
| MiniMax-M3 | 2760 | 148 | 17/17 | 0.999990 | 0.0017 | 87 (0.018) | Blackwell B200/B300 |
| Ling-3.1-flash-base | 3952 | 152 | 384/384 | 0.983436 | 0.0004 | 192 (0.025) | Blackwell GB200/GB300 |

For each model it rebuilds the untouched embedding rows from the seed under every SM count from 16 to 256 and prints the one that matches. seed+1 is a control and next best is the runner-up SM count. V4-Flash is 461/469 because 8 of those rows got partly trained.

## How it works

PyTorch's CUDA `normal_` launches 2,048 threads per SM and every thread gets its own Philox stream. Element `i` is written by thread `i mod G` where `G = 2048 × SMs`. A seed gives the same numbers on any GPU but they end up in different positions depending on the SM count.

| GPU | SMs | Threads (G) |
|---|---|---|
| A100 / A800 | 108 | 221,184 |
| H100 / H800 / H200 | 132 | 270,336 |
| B200 / B300 | 148 | 303,104 |
| GB200 / GB300 | 152 | 311,296 |

Embedding rows for tokens that never appear in training never get a gradient so they still hold the init. With the right seed and SM count you get them back at r > 0.9999. Anything else gives about 0.

The seeds were brute forced over all 2^32 values.

## On a real GPU

    python gpu_check.py

Same check with stock PyTorch (`torch.cuda.manual_seed` then `normal_`) so nothing in this repo is generating the numbers. Rows only match on a GPU with the right SM count. This is what we got on the three GPUs we tried.

| GPU | DeepSeek-V3 | DeepSeek-V4-Flash | Kimi K3 | MiniMax-M3 |
|---|---|---|---|---|
| H100 (132 SMs) | 8/8, bit for bit | 0/469 | | |
| B300 (148 SMs) | 0/8 | 461/469 | 0/13 | 17/17 |
| GB200 (152 SMs) | | | 13/13 | |

Seed+1 matched nothing on any of them.

## Notes

- `rows/` is the rows copied from Hugging Face, 7 MB total. `REFETCH=1 python replay.py` downloads them again from the pinned revisions. They get checked against the hashes in `models.json` either way.
- This is the GPU that created the starting weights. It doesn't say where the training happened after that.
- H100, H800 and H200 are all 132 SMs. GB200 and GB300 are both 152 and so is an AMD MI300X in two-way partition mode. A stock MI300X is 304.
- Ling uses `trunc_normal_` and our model of the bf16 path only gets to r 0.98. The [`gpu-forensics` CLI](#gpu-forensics-cli) models PyTorch's bf16 `uniform_` path exactly and reaches r ≈ 0.99999.
- MiniMax's embedding is split into 4 shards and shard n uses seed 2760 + n.

## gpu-forensics CLI

**`gpu-forensics`** is a command-line tool that applies the method above to any
Hugging Face model. Given a model URL, it estimates which GPU thread layout
initialized the weights. It reads only metadata and small byte ranges of
safetensors files, never executes repository code, and classifies its evidence as
`supported`, `weak` or `inconclusive`.

Compared with the replay, which needs a recorded seed and hand-picked rows for each
model, the CLI:

- **Inspects any repository.** It pins the model to a commit and reports whether
  analysis is feasible, flagging blockers such as quantized or pickle-only weights.
- **Finds untrained rows automatically**, using tokenizer evidence and per-row
  statistics.
- **Recovers unknown seeds.** It tries metadata and known seeds, then runs an
  exhaustive, resumable 2³² search (about 10 minutes on a 20-core CPU).
- **Uses a broader layout model.** The launch grid is SMs × threads per SM (1024,
  1536 or 2048). `normal_`, `uniform_` and `trunc_normal_` are reconstructed
  exactly, including bf16 rounding, Philox offsets, tensor-parallel shards and
  fp32 init.
- **Validates on held-out rows** against wrong-seed and wrong-layout
  alternatives, and writes a reproducible JSON report.

```bash
uv sync
uv run gpu-forensics fingerprint https://huggingface.co/deepseek-ai/DeepSeek-V3-Base --format text
```

```text
model:          deepseek-ai/DeepSeek-V3-Base @ afb92e1fa402c2be2a9eb085312bb02e0384d6c7
classification: supported
reason:         held-out median r=1.0000 vs wrong-seed null +0.0014±0.0124 (z=80); runner-up layout r=0.0454
seed:           953049583  dist=normal  offset=0  tp=1
layout:         G=270336 threads  (H100 SXM / H800 / H200)
configs:        264 SMs x 1024, 176 SMs x 1536, 132 SMs x 2048
held-out:       median r=1.000000 over 4 rows (4/4 matched)
bit-exact:      1.000 of held-out elements
```

The result describes the hardware that **created the starting weights**, not the
hardware used for training.

### Why this works

PyTorch's CUDA `normal_`/`uniform_` kernels launch
`G = min(SMs × threadsPerSM / 256, ⌈numel/256⌉) × 256` threads. Thread `i mod G`
writes element `i` from its own Philox stream. A seed therefore yields the same
numbers on every GPU, but **where** each number lands depends on `G`. Embedding
rows for tokens that never occur in training receive no gradient, so they keep
those initial values.

The fingerprint is `G`, not the SM count alone. Datacenter parts run 2,048 threads
per SM, but many workstation and consumer parts run 1,536. So `G` identifies an
equivalence class of devices, for example H100 = 132×2048 = 176×1536.

| Layout `G` | Devices with this layout |
|---|---|
| 221,184 | A100 / A800 |
| 270,336 | H100 SXM / H800 / H200 |
| 303,104 | B200 / B300 |
| 311,296 | GB200 / GB300, MI300X (2-way partition) |
| 73,728 | GB10 (48 × 1536) |

The full catalogue is in `gpu_forensics/layouts.py`, and every report lists all
(SMs, threads/SM) pairs that produce the winning `G`.

### Commands

#### `inspect`: is analysis feasible?

```bash
uv run gpu-forensics inspect https://huggingface.co/Qwen/Qwen2.5-0.5B
uv run gpu-forensics inspect org/model@<revision> --deep   # full tensor/dtype census
```

`inspect` reads only metadata:
- It accepts `org/name`, single-name repos, `org/name@rev`, and `huggingface.co` URLs including `/tree/`, `/blob/` and `/resolve/` forms.
- It pins the revision to a commit SHA.
- It reads `config.json`, the model card, the tokenizer files, and safetensors headers. Headers are read with two range requests per shard.

It then reports:
- the embedding tensor, its dtype, and its shape
- **padding rows** (embedding rows beyond the tokenizer) and reserved tokens
- init hints: `initializer_range`, seeds mentioned in the README, init method names
- blockers, such as pickle-only checkpoints, a quantized embedding, adapter-only repos, or gated repos without `HF_TOKEN`
- warnings, such as a merged or derived model, tied embeddings, custom `auto_map` code, or a partially quantized checkpoint

The verdict is one of `feasible`, `requires_seed_search` or `infeasible`.

#### `fingerprint`: run the pipeline

```bash
uv run gpu-forensics fingerprint <model> [--seed auto|screen|search|known|N[,N]] \
    [--dist auto|normal|uniform|trunc_normal] [--rows 'tail:64'|'random:500'|'1,5,10-20'] \
    [--offsets 0,3776 | 0:8192:4] [--tp 1,2,4,8] [--search-budget SECONDS] \
    [--backend auto|cpu|numpy|cuda] [--output report.json] [--format json|text]
```

1. **Sample.** The tool finds untrained rows without downloading the checkpoint. It range-reads three groups of candidates:
   - tokenizer padding rows and reserved tokens;
   - evenly spaced rows across the tensor;
   - the first 512 rows (unused byte-fallback tokens sit there) and the tail.

   Untrained rows share a single per-row std that stands out from its neighbours. Trained rows spread out, and their kurtosis is heavier. The tool keeps every isolated std cluster that has a near-zero mean and init-like kurtosis, and prefers the smallest one: untrained rows only shrink under weight decay, while trained rows usually grow. The remaining clusters are retried as alternates if screening the first one fails.
2. **Seeds.** Candidates are tried in order of cost:
   1. explicit `--seed`
   2. a `models.json` recipe
   3. seeds from metadata and the README
   4. common conventions
   5. an exhaustive 2³² search

   Unless `--seed search` is given, each candidate is screened against every catalogued layout. For an explicit seed that doesn't match at offset 0, the tool also scans Philox offsets.
3. **Exhaustive search.** This runs a native Philox brute-force over sign patterns. The sign of a normal or uniform draw needs only the raw Philox word, with no transcendental functions, so most seeds are rejected after a few evaluations. Each candidate is checked against 64 positions with up to 6 sign mismatches allowed, so false hits are expected less than once across the whole space. On a 20-core CPU (OpenMP), 2³² seeds × 18 layouts take about 10 minutes. The search is chunked, checkpointed and resumable, and honours `--search-budget`. If the budget runs out, the result is `inconclusive` with the seed range that was covered.
4. **Layouts.** With the seed fixed, the tool sweeps every `G` from 16–320 SMs at 1024, 1536 and 2048 threads/SM, plus the device catalogue. With `--dist auto` it also compares init families:
   - `normal_`
   - `uniform_`
   - `trunc_normal_` (±2σ, ±3σ or absolute bounds)

   Each family is tried in both the storage dtype and float32, because weights are often initialized in fp32 and cast to bf16 for release.
5. **Validation.** The pipeline fits on even-indexed rows and scores the remaining rows. Validation includes:
   - a wrong-seed null distribution (pooled r for `seed+1..seed+K`), with an empirical p-value and z-score;
   - the runner-up layout, with any layouts that are indistinguishable at the sampled positions listed explicitly;
   - a pooled Fisher-z for weak signals;
   - the fraction of held-out elements that match bit for bit.

#### `validate`: re-check a saved report

```bash
uv run gpu-forensics validate report.json --null-seeds 500 --fresh-rows 32 --format text
```

`validate` re-reads the recorded sample and refuses to continue if its SHA-256
differs from the one in the report. It then recomputes the held-out and null
statistics. `--fresh-rows N` also selects rows that were never part of the
original sample and uses them as an independent hold-out set.

#### `search-seed`: run the exhaustive search on its own

```bash
uv run gpu-forensics search-seed <model> --rows tail:8 --dist normal --checkpoint search.json
```

### Classification rules

| Class | Rule |
|---|---|
| `supported` | All of the following hold: held-out median r ≥ 0.95; no wrong seed scores as high (p ≤ max(0.01, 1/(K+1))); null z ≥ 10; every distinguishable layout trails by more than 0.5 |
| `weak` | Pooled held-out signal is significant (Fisher z ≥ 5, null z ≥ 5, same p rule) but below the per-row threshold. This is the trained-weights regime |
| `inconclusive` | Everything else, including a high fit with no held-out support, an unrecovered seed, or an infeasible repository |

The CLI exits with code 0 for `supported` or `weak`, 3 for `inconclusive`, and 2 for Hub errors.

### Report format

Reports are deterministic JSON with `schema_version`. They contain:
- the pinned model reference
- a bounded inspection summary
- sample provenance: tensor, shard file, rows, byte ranges, SHA-256, the RNG-position assumption, and how the rows were selected
- the seed-search record: status, coverage, backend, hits and confirmations
- the top layouts from the sweep
- the best hypothesis: seed, `G`, distribution and parameters, dtype, Philox offset, tensor-parallel degree, seed-per-rank, matching devices, and equivalent configurations
- validation statistics
- classification, reason, limitations and runtime

### Results on published models

All five reproduce as `supported` on held-out rows, using the known recipes:

| Model | Seed | Layout | Held-out rows matched | Held-out median r | Bit-exact |
|---|---|---|---|---|---|
| DeepSeek-V3-Base | 953049583 | H100/H800/H200 (132 SMs) | 4/4 | 1.000000 | 1.000 |
| DeepSeek-V4-Flash | 953049583 | B200/B300 (148 SMs) | 457/465 | 0.999999 | 0.952 |
| Kimi-K3 (offset 3776) | 20256586 | GB200/GB300 (152 SMs) | 9/9 | 0.999995 | 0.668 |
| MiniMax-M3 (tp=4, seed+rank) | 2760 | B200/B300 (148 SMs) | 13/13 | 0.999990 | 0.437 |
| Ling-3.1-flash-base | 3952 | GB200/GB300 (152 SMs) | 380/380 | 0.999994 | n/a |

The bit-exact fraction is lower for Kimi-K3 and MiniMax-M3 because their exact
init std isn't recorded, so the scale is fitted.

Ling's `trunc_normal_` now reconstructs at r ≈ 0.99999 (previously 0.983). The
fix is to model PyTorch's `uniform_kernel` on bf16 tensors: the bounds are cast
to bf16, and a value that rounds to the upper bound is replaced by the lower bound.

**Blind runs.** These use no `models.json` rows or recipe, and the tool finds the init rows of all five models on its own:
- **DeepSeek-V3:** supported, 60/60 held-out rows, bit-exact.
- **DeepSeek-V4-Flash:** supported, 60/60. The tokenizer covers every row, so the rows were found by statistics alone.
- **Ling-3.1-flash-base:** supported, 52/52.
- **MiniMax-M3:** supported with `--tp 1,4`, which recovers tp=4 and seed+rank.
- **Kimi-K3:** supported with `--seed 20256586`. The offset scan finds offset 3776.
- **Exhaustive search:** `--seed search` recovers DeepSeek-V3's seed with no hints, in about 12 minutes on a 20-core CPU.

**Corpus.** A screening pass over 19 public models (`tools/corpus.py`) found two `supported` results outside the published set:
- **DeepSeek-V2-Lite:** seed 953049583, A100/A800-compatible layout, bit-exact on held-out rows.
- **Mistral-7B-v0.1:** seed 0, fp32 `trunc_normal_`, H100/H800/H200-compatible layout, r=0.999997 on 60/60 held-out rows.

Most other models came back `inconclusive` because their seed isn't in the screening set; `--seed search` is the next step for them.

`docs/report.md` records the full validation log.

### Development

```bash
uv sync
uv run pytest            # offline: golden replay of the 5 models, synthetic Hub, search, statistics
uv run pytest -m network # live Hub checks
uv run ruff check .
uv run python replay.py  # five-model replay table
```

The offline tests include:
- Philox and Box-Muller agreement with the reference implementation
- planted-seed recovery with the CPU and NumPy backends
- resumable and budget-limited searches
- tensor-parallel shard handling
- an end-to-end fake Hub serving a synthetic safetensors checkpoint
- classification of supported, weak, noise and fit-only cases

**Ground truth on hardware.** `tools/synthesize_fixtures.py` records stock
PyTorch output for several cases on a CUDA device: `normal_`, `uniform_` and
`trunc_normal_` in fp32, bf16 and fp16, plus Philox offsets advanced by earlier
RNG calls. `tests/test_fixtures.py` then requires the CPU model to reproduce
these recordings bit for bit. `tools/corpus.py tools/corpus.txt` runs
inspection and fingerprinting over a list of models.

The CUDA search backend (`--backend cuda`, `gpu_forensics/search/philox_search.cu`)
is experimental and not yet validated on hardware. The OpenMP CPU backend is the
reference.

### Limitations

- The tool detects the initialization layout, not where training ran.
- Layouts with the same `G` are indistinguishable, so a match does not prove the vendor or the exact product.
- CPU, meta-device and custom-kernel initialization cannot be fingerprinted.
- Quantized, merged and heavily fine-tuned embeddings usually come back `inconclusive`.
- Tied embeddings weaken the signal.
- The exhaustive search assumes a 32-bit seed and the offsets and tensor-parallel degrees you pass in.

## License

MIT. See `LICENSE`.
