# Gap analysis and roadmap

Date: 2026-11-16

> **Status (v0.2.0):** Phases 0–5 are implemented in the `gpu_forensics/` package, and
> Phases 6–7 have tooling (`tools/synthesize_fixtures.py`, `tests/test_fixtures.py`,
> `tools/corpus.py`). See "Implementation status" at the end. The gap analysis below
> describes the tree as it was at `81b1c47`.
Scope: compares the current tree (`81b1c47`) against `docs/init.md` and `docs/PLAN.md`, then lays out what to build next.

## 1. Verdict

**Faithful for the upstream reproduction. Not yet faithful for the arbitrary-model pipeline that `init.md` asks for.**

`init.md` recommends forking `checkpoint-forensics`, reproducing the five published examples, and then building the generic Hugging Face ingestion and seed-discovery pipeline around it. The first two steps are done well. The pipeline (Stages A–E) exists mostly as CLI scaffolding: for any model not in `models.json`, `fingerprint` returns `inconclusive` without sampling a single weight.

| `init.md` requirement | Status | Evidence |
|---|---|---|
| Fork upstream, keep MIT and attribution | Done | `LICENSE`, README header, `gpuid.py`/`fetch.py`/`replay.py` unchanged |
| Reproduce the five published examples | Done | `uv run python replay.py` matches the upstream table exactly (re-run 2026-11-16) |
| CLI with `inspect` / `fingerprint` / `validate` | Partial | All three exist in `gpu_forensics.py`; `--seed` and `--revision` are missing |
| **Stage A**: inspect config, model card, safetensors metadata, init hints, pinned revision | Partial | Reads `config.json` and the shard index only. No model card, no safetensors headers (shapes/dtypes), no seed/init hints, revision never resolved to a commit SHA, no `huggingface_hub`, no auth for gated repos |
| **Stage B**: select promising tensors, unused-token rows, range reads | Missing for new models | `fetch.rows` range-reads only the rows already listed in `models.json`. There's no tensor selection and no unused-row detection |
| **Stage C**: seed discovery (metadata → conventions → accelerated search) | Missing | Seeds come only from `models.json` |
| **Stage D**: SM count × threads/SM × distribution × offset × dtype | Partial | Sweeps SM 16–256 at a fixed 2048 threads/SM. Only `normal` and an approximate `trunc_normal` (Ling r≈0.98). No `uniform_`, no offset search, no threads/SM other than 2048 |
| **Stage E**: correlation, held-out, significance, competing hypotheses, explicit inconclusive | Mostly missing | `validate` computes nothing: it stamps `held_out: false` onto the JSON. `fingerprint` scores on all rows, including the 4 it fit on. Classification is `median r > 0.9`, with no significance test and no runner-up margin |
| Machine-readable report | Partial | JSON is produced, but it has no schema version or sample provenance, and `inspect` dumps the whole index (9 MB for DeepSeek-V3) |

### Defects found while checking (reproduced locally)

1. **`docs/report.md` claims something that doesn't happen.** It says `inspect https://huggingface.co/bert-base-uncased` "read config.json … returned feasible: false". It actually fails with `error: expected a Hugging Face URL or repository ID`, because `model_ref` requires an `org/name` URL.
2. **Pinned-revision URLs miss known models.** `fingerprint https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/tree/afb92e1…` (the exact pinned revision) returns `inconclusive`, because `resolve_model` compares only bare URLs and repo IDs.
3. **Single-file checkpoints are reported infeasible.** `Qwen/Qwen2.5-0.5B` has `model.safetensors` and no index. `inspect` reports `feasible: false, tensor_count: 0`.
4. **Fit and test rows overlap.** `fingerprint` picks the SM count from `rows[:4]` and then scores `rows` (all of them), so the reported score includes the fitting rows. A real held-out split costs nothing here: DeepSeek-V3 has 8 rows, V4-Flash 469, Ling 384.
5. **`validate` says held-out data doesn't exist.** Its note ("published samples do not contain a separate held-out partition") is wrong, for the reason in item 4.
6. **`match()` is duplicated** verbatim between `replay.py` and `gpu_forensics.py`.
7. **No tests, lint, or CI.** `PLAN.md` §Test plan has not been started.
8. **`inspect` output is unbounded.** It embeds the raw `config.json` and the full `weight_map`.

Against `PLAN.md` §Acceptance criteria: 1 is met; 2 and 4 are partly met (4 only for the five known models); 3, 5, 6 (provenance and inconclusive semantics) and 7 are not met.

## 2. Roadmap

Ordered by value per effort. Each phase ends with a verifiable check.

### Phase 0: Correctness and hygiene (small, do first)

- [ ] Fix `model_ref`: accept single-segment repos (`bert-base-uncased`), `/tree/<rev>`, `/blob/<rev>/…`, `/resolve/<rev>/…`, and an `org/name@rev` shorthand. Add a `--revision` flag.
- [ ] Resolve every revision to a commit SHA (`GET /api/models/{repo}/revision/{rev}`) and record it in the report.
- [ ] `resolve_model` should match on repo, and on revision when one is given. If the revision differs from the pinned one, warn and still use the recipe.
- [ ] Support single-file `model.safetensors`, and flag `.bin`/`.pt`-only repos as unsupported (never unpickle them).
- [ ] Move `match()` into one shared module (`recipes.py`) and have `replay.py` import it. Keep `replay.py` output byte-identical to upstream.
- [ ] Do a real held-out split in `fingerprint`: fit on one row subset, report fit and held-out scores separately. Make `validate` recompute on held-out rows instead of stamping text.
- [ ] Have `inspect` emit a bounded summary (architecture, dtype, vocab size, embedding tensor names, shapes, shard count) rather than raw files.
- [ ] Correct `docs/report.md` item 1 once the parser is fixed and the command has been re-run.
- [ ] Add `pytest` and `ruff`. Golden test: the `replay.py` table for all five models, offline from `rows/`. Add a GitHub Actions workflow.

**Check:** the golden test passes; the bert, Qwen-0.5B and pinned-revision DeepSeek commands all behave correctly.

### Phase 1: Stage A, a useful `inspect`

- [ ] Read every shard's safetensors header with two range requests per shard (8-byte length, then the header) to get each tensor's dtype, shape and byte offsets without downloading weights.
- [ ] Parse model-card front matter and README text, plus `generation_config.json`, `training_args`-style files, and any `*.yaml` config, for `seed`, `init_std`, `initializer_range`, and `init_fn`.
- [ ] Detect blockers: `quantization_config` (GPTQ/AWQ/FP8/bnb), FP8/INT dtypes on the embedding, `merge`/`adapter` tags, a `base_model` field (point at the base model instead).
- [ ] Compute a feasibility score:
  - an embedding tensor is present in BF16/FP16/FP32;
  - the embedding has more rows than the tokenizer has tokens, which suggests untrained padding rows (compare against `tokenizer.json` / `tokenizer_config.json` added tokens);
  - the tensor holds well over 2048 × 16 elements. The first G elements are identical under every layout (see the `gpuid.py` header), so only elements past that discriminate.
- [ ] Use `HF_TOKEN` for gated repos, and turn 401, 403, 404 and timeouts into structured diagnostics.
- [ ] Decide between `huggingface_hub` (which `init.md` suggests, for auth, caching and revision resolution) and the current dependency-free `urllib`. Recommendation: `huggingface_hub`, with a thin wrapper so tests can stub it.

**Check:** a 20-model inspection corpus (small and large, gated, quantized, merged, single-file) produces correct feasibility verdicts with no crashes.

### Phase 2: Stage B, sampler and unused-row detection

- [ ] Generalise `fetch.rows` to any tensor, row set and dtype (BF16/FP16/FP32), with an on-disk cache keyed by `(repo, sha, tensor, rows)` and a SHA-256 recorded per sample.
- [ ] Find unused rows without a tokenizer, using statistics alone. Rows that never got a gradient have per-row std ≈ init std (e.g. 0.006 or 0.02) and near-zero mean, while trained rows drift. Range-read a coarse sample of rows (say every 64th), flag the outliers that look like init, then densify around them.
- [ ] Use tokenizer evidence too: padding rows beyond `len(tokenizer)`, `<|reserved_*|>` and `<unusedN>` tokens, and declared-but-unused special tokens.
- [ ] Record per-sample provenance: shard, byte range, global row, dtype, and the assumed RNG position (row × width + TP-shard adjustment).
- [ ] Sample deterministically from a recorded config, so the same report can be re-derived.

**Check:** on the five known models, unused-row detection recovers a superset of the rows in `models.json`, without using those lists.

### Phase 3: Stage D, a wider hypothesis space

Do this before seed search, because the search needs a correct forward model.

- [ ] Parameterise threads/SM. `blocks_per_sm = maxThreadsPerMultiProcessor / 256`, so it is 2048 on datacenter parts but 1536 on many consumer and workstation parts (e.g. sm_86/89/120). The hypothesis is (SMs, threads/SM), i.e. G, not SMs alone. Report every GPU family that maps to the winning G (an equivalence class), including the MI300X partition caveat.
- [ ] Add `uniform_` (used by `kaiming_uniform_`/`xavier_uniform_`, the `nn.Linear` defaults) and exact `normal_` paths for fp32 and bf16/fp16 outputs.
- [ ] Make `trunc_normal_` exact. Ling stops at r≈0.98, which means the bf16 rounding path isn't modelled precisely. Get it to bit-exact on a real GPU (Phase 6).
- [ ] Model Philox offsets. Each CUDA RNG call advances the offset by `((numel − 1) / (256 · grid.x · unroll) + 1) · 4`, and `grid.x` depends on the SM count. So a tensor's offset depends on the sizes of all earlier initialised tensors and on G: it is a second, independent fingerprint, and it predicts offsets such as Kimi's `cpu_offset: 3776` from module init order. Implement an offset predictor from the parameter list plus G, and a fallback that searches small offsets.
- [ ] Support tensor-parallel layouts generically: `tp`, `seed_per_rank` and the row partition as hypothesis parameters, not per-model constants.

**Check:** synthetic tensors generated by stock PyTorch on the local GPU for each (dist, dtype, offset) combination are reproduced bit-exactly by the CPU model.

### Phase 4: Stage C, seed discovery

- [ ] Try candidates in cost order:
  1. `--seed`
  2. metadata hints (Phase 1)
  3. a convention list: 0, 1, 42, 1234, 1337, 2024–2026, 3407, the seeds already seen in `models.json` (DeepSeek reused 953049583 across generations), and Unix timestamps around the release window
  4. exhaustive search
- [ ] Use the layout-independent prefix where possible. Elements with flat index < G are identical for every layout, so if any trusted untrained values sit inside the first 32,768 elements, the seed can be found without guessing the GPU. Otherwise, search seed × {plausible G values} (about 10 families, not 241 SM counts).
- [ ] Build a native search kernel. Per seed candidate, generate about 8 values at the sampled positions, compare signs or a z-scored dot product, reject early, and confirm survivors with the full NumPy model. Philox-4x32-10 is cheap: 2³² × ~10 layouts fits in minutes on one GPU. Implementation options: a CUDA kernel via CuPy or `torch.utils.cpp_extension`, plus a C/OpenMP CPU fallback. The local machine has an NVIDIA GB10.
- [ ] Make the search resumable: checkpoint completed seed ranges to disk, shard across processes, and record the searched space in the report so a failed search is still evidence.
- [ ] Cap cost with `--max-seeds` and a time budget. An exhausted budget yields `inconclusive` plus the space covered.

**Check:** blind recovery of all five published seeds (DeepSeek, Kimi, MiniMax, Ling) using only `repo`, `tensor` and the auto-detected rows, plus seeds for synthetic GB10-generated tensors.

### Phase 5: Stage E, statistics and verdicts

- [ ] Fit on subset A and score on a disjoint subset B (rows, or element ranges within rows).
- [ ] Build a null distribution: scores under seed+k for k = 1…1000 and under every other G, giving an empirical p-value and a z-score of the winner against the null.
- [ ] Add a weak-signal mode for trained weights, like the r≈0.066 that `init.md` cites for DeepSeek-V4-Flash: aggregate across many elements and report a Fisher-z test with z = atanh(r)·√(N−3), not a per-row threshold.
- [ ] Require a margin over the runner-up: the gap between the best and second-best G must exceed the null spread.
- [ ] Count exact matches: once the init std is known, compare bf16 bit patterns directly, since bit-for-bit agreement is far stronger evidence than r.
- [ ] Write the classification rules down and test them:
  - `supported`: held-out exact or r > 0.99 and the null p-value below threshold
  - `weak`: significant aggregate signal only
  - `inconclusive`: everything else, including any fit-only success
- [ ] Version the JSON schema and implement the `PLAN.md` dataclasses (`ModelRef`, `TensorSample`, `CandidateHypothesis`, `ValidationResult`, `ForensicReport`), with deterministic serialisation. Add a human-readable renderer.

**Check:** a test that a high fit score with no held-out support is downgraded; synthetic weak-signal tensors (init + heavy noise) land in `weak`; seed+1 and random data land in `inconclusive`.

### Phase 6: Ground truth on real hardware

- [ ] Run `gpu_check.py` on the local GB10. It should match none of the five models (a negative control), and the result goes in `docs/report.md`.
- [ ] Add `gpu_check.py --synthesize`: initialise tensors with stock PyTorch at known seeds, distributions, dtypes and offsets on the local GPU, and save them as safetensors fixtures for Phases 3–5. Record `multi_processor_count` and `max_threads_per_multi_processor` alongside them.
- [ ] If others can run the same script on A100/H100/B200, collect their fixtures to cover more of the G table.

### Phase 7: External validation corpus

- [ ] Choose open models whose seed, init code and hardware are publicly documented (for example, fully open training releases that publish configs and cluster details), and treat them as blind tests: run the pipeline without the disclosed answer, then compare.
- [ ] Publish the results table, misses and inconclusives included, in `docs/report.md`.

## 3. Proposed package shape (once Phase 0–2 land)

```
gpu_forensics/
  cli.py          # argparse entry point (inspect / fingerprint / validate / search-seed)
  hub.py          # ModelRef, revision resolution, range reads, cache
  inspect.py      # metadata, safetensors headers, feasibility
  sampler.py      # tensor/row selection, unused-row detection
  recipes.py      # init-operation models: normal/uniform/trunc_normal, offsets, TP
  search/         # seed search: python driver + CUDA/C kernels
  stats.py        # held-out, null distributions, Fisher-z, classification
  report.py       # dataclasses, JSON schema, text renderer
gpuid.py fetch.py replay.py gpu_check.py   # upstream files kept as-is (provenance)
tests/
```

## 4. Risks and open questions

- **Seed search may hit the limits of the forward model.** If the init path isn't modelled bit-exactly (as with Ling's `trunc_normal_`), early rejection must rely on correlation rather than exact matches, which needs more values per candidate and slows the search.
- **Many labs initialise on CPU or the meta device and then materialise**, or use custom init kernels. Both are undetectable by design and should be reported as such, not forced.
- **G equivalence classes are not unique to a vendor or product.** The report wording must not claim a specific GPU, only "layout compatible with …".
- **Ethics and wording.** The upstream report makes export-control claims. This tool should report layout evidence and confidence, and leave the attribution conclusions to the reader.

## 5. Implementation status (v0.2.0)

| Phase | Status | Where |
|---|---|---|
| 0 Correctness/hygiene | Done. URL forms, SHA pinning, single-file checkpoints, real held-out split, bounded `inspect`, pytest/ruff/CI | `hub.py`, `engine.py`, `tests/`, `.github/workflows/ci.yml` |
| 1 Inspect | Done. Safetensors headers via range reads, model-card/config/README hints, tokenizer padding and reserved ids, blockers (pickle, quantized embedding, adapters, gated repos), warnings (merged, derived, tied, `auto_map`), verdict | `inspector.py` |
| 2 Sampler | Done. Tokenizer candidates plus a head/tail/coarse scan; isolated std clusters (smallest first, alternates kept) with mean and kurtosis filters; coalesced range reads, SHA-256 provenance, on-disk cache. Blind selection finds the init rows of all five published models | `sampler.py` |
| 3 Hypothesis space | Done. `G` = SMs × threads/SM (1024/1536/2048), device catalogue and equivalence classes, `uniform_`, exact bf16 `trunc_normal_` (Ling r 0.983 → 0.99999), dtype casts, Philox offsets and an offset-increment model, TP and seed-per-rank | `layouts.py`, `recipes.py` |
| 4 Seed discovery | Done. Explicit, `models.json`, metadata and conventions, then exhaustive native sign-pattern search (OpenMP; NumPy fallback; CUDA experimental), resumable, with a budget | `search/` |
| 5 Statistics | Done. Fit/held-out split, pooled wrong-seed null with p and z, runner-up and indistinguishable layouts, Fisher z, bit-exact fraction, documented rules, versioned JSON | `stats.py`, `report.py` |
| 6 Ground truth | Tooling ready, **not yet run**: generation on the local GB10 is deferred by request | `tools/synthesize_fixtures.py`, `tests/test_fixtures.py` |
| 7 Corpus | Done for screening: 19 models; 6 supported, including DeepSeek-V2-Lite (A100-compatible) and Mistral-7B-v0.1 (H100-compatible) outside the published set. See `docs/report.md` | `tools/corpus.py`, `tools/corpus.txt` |

Deviations from the plan:
- **Hub access.** It uses `urllib` with `HF_TOKEN` rather than `huggingface_hub`, which keeps the project to two dependencies (numpy and scipy). Tests stub one function.
- **Upstream scripts.** `replay.py` and the other upstream scripts are kept verbatim for provenance, rather than refactored to import the package.
- **Seed search.** It needed no GPU: the CPU covers 2³² seeds × 18 layouts in about 10 minutes.

Remaining work:
- Run the fixtures on A100, H100, B200 and GB10 hardware.
- Validate the CUDA backend.
- Predict offsets automatically from module init order (currently `--offsets` or an explicit-seed offset scan).
- Run `--seed search` across the inconclusive corpus models.
- Search seeds wider than 32 bits.
