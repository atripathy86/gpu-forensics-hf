# GPU Initialization Forensics for Hugging Face Models

## Summary

This project will be an `atripathy86` fork/adaptation of [macrogrants/checkpoint-forensics](https://github.com/macrogrants/checkpoint-forensics), an MIT-licensed research implementation for estimating which GPU layout initialized an AI model's starting weights.

The upstream project supplies the core Philox reconstruction, GPU-layout comparison, model-specific examples, and initial checkpoint sampling code. This project will preserve that work and add a general Hugging Face workflow: inspect an arbitrary model repository, sample useful tensors without downloading entire checkpoints, discover or accept initialization seeds, compare plausible GPU layouts, validate on held-out weights, and emit an evidence report that can explicitly be inconclusive.

The result identifies initialization hardware evidence, not necessarily the hardware used throughout training.

## Upstream credit and project delta

The upstream repository and its authors must remain credited in the README, source headers where upstream code is retained, documentation, and distribution metadata. Preserve the MIT license and its notice. Cite the Unemployment Labs research that motivates the method.

The work beyond the upstream repository is:

- A reusable Python package and CLI named `gpu-forensics`.
- Hugging Face URL/repository parsing with pinned revision support.
- Read-only model inspection for configuration, model-card metadata, tensor indexes, safetensors metadata, initialization hints, and seed candidates.
- Range-based safetensors sampling that avoids downloading complete checkpoints.
- Generic tensor-selection heuristics favoring embeddings, unused-token rows, and other parameters likely to retain initialization signal.
- Seed-discovery orchestration, with published/configured seeds tried before optional accelerated search.
- Support for additional initialization distributions, tensor layouts, dtypes, conversion behavior, and Philox offsets.
- Held-out validation, competing-hypothesis comparison, statistical evidence scoring, and machine-readable reports.
- Clear limitations and an explicit `inconclusive` outcome instead of forced GPU attribution.

## Implementation stages

### 1. Bootstrap the fork

- Start from the upstream repository rather than reimplementing Philox behavior.
- Import upstream files and examples with their provenance intact.
- Isolate upstream-compatible reconstruction functions behind a small internal API so later CLI and Hugging Face code does not depend on script internals.
- Add project metadata, dependency constraints, formatting/linting configuration, and a test runner.
- Reproduce all upstream published examples before changing behavior.

### 2. Inspect Hugging Face repositories

Implement an inspection layer using `huggingface_hub` that accepts either a model URL or repository ID and supports an explicitly pinned revision.

The inspector should read, without executing remote model code:

- `config.json` and model-card metadata.
- Safetensors indexes and tensor names, shapes, dtypes, and shard locations.
- Published seeds, initialization methods, training configuration, and other forensic hints.
- Repository and revision identifiers needed for reproducibility.

Failures such as missing indexes, unavailable revisions, unsupported formats, authentication errors, and network errors should become structured diagnostics rather than opaque crashes.

### 3. Select and retrieve useful weights

Add a sampler that chooses promising tensors and records their exact provenance.

- Prefer embedding matrices, especially rows associated with unused or minimally trained tokens.
- Consider other parameters likely to preserve initialization signal.
- Avoid beginning with heavily trained attention weights unless no better candidates exist.
- Read selected tensor ranges or rows from safetensors shards instead of downloading complete files.
- Preserve tensor shape, dtype, global position, shard, byte range, and RNG-position assumptions in the sample metadata.
- Make sampling deterministic from a recorded configuration.

The sampler must not claim that an observed tensor position is valid for reconstruction until the original initialization operation, memory order, and offset assumptions are established.

### 4. Reconstruct candidate layouts

Wrap and extend the upstream reconstruction implementation to test combinations of:

- Candidate seed and Philox offset.
- SM count and threads per SM.
- Initialization distribution: at minimum normal, uniform, and truncated normal where the reference behavior is known.
- Tensor shape, flattening/order, dtype, and floating-point conversion behavior.

Retain the hardware-layout principle from PyTorch's `calc_execution_policy()`: large operations can reach a grid limit derived from `multiProcessorCount` and `maxThreadsPerMultiProcessor`, while small operations may not contain enough evidence to distinguish layouts.

Candidate GPU families should be reported as layout hypotheses, for example A100/A800, H100/H800/H200, B200/B300, or GB200/GB300 where supported by the evidence. The report must avoid treating SM count as a unique vendor proof because compatible layouts may be reproduced by other hardware.

### 5. Discover seeds

Implement seed handling in increasing cost order:

1. Use an explicitly supplied seed.
2. Read seeds from model metadata, configuration, published examples, and documented conventions.
3. Try model-specific candidate rules.
4. Offer optional accelerated search for unknown seeds.

The CLI and documentation must state that exhaustive 32-bit search spans roughly 4.29 billion candidates and is not a practical Python-loop operation. Seed search should be bounded, resumable, parallelizable where safe, and recorded in the output. An unsuccessful search must produce an informative inconclusive result.

### 6. Validate and report evidence

For each candidate, compute correlations between observed and reconstructed values, then rank candidates without relying only on the fitting sample.

Validation must include:

- Held-out tensors or weight ranges.
- Competing SM counts and initialization hypotheses.
- Exact-match statistics where applicable.
- Sample size and statistical significance information.
- Repeated or independently selected samples when available.
- Warnings for trained, quantized, merged, fine-tuned, transformed, CPU-initialized, or custom-kernel weights.

Reports should be available as human-readable text and JSON and include model ID, pinned revision, sample provenance, candidate parameters, scores, validation results, evidence strength, limitations, runtime/resource information, and a final classification of supported, weak, or inconclusive.

## Intended CLI

```text
gpu-forensics inspect https://huggingface.co/org/model

gpu-forensics fingerprint \
  https://huggingface.co/org/model \
  --seed auto \
  --revision <commit-or-tag> \
  --output report.json

gpu-forensics validate report.json --output validated-report.json
```

`inspect` should be safe and inexpensive, reporting feasibility and available metadata. `fingerprint` should perform sampling and candidate comparison. `validate` should run held-out and competing-hypothesis checks against a saved analysis. Commands should support explicit cache and resource controls and should never execute arbitrary repository code.

## Public interfaces and data contracts

- `ModelRef`: repository ID, resolved revision, optional URL, and local/cache information.
- `ModelInspection`: architecture, tensor inventory, initialization hints, seed candidates, supported formats, and diagnostics.
- `TensorSample`: tensor identity, shape, dtype, source shard, byte/range provenance, values, and RNG-position assumptions.
- `CandidateHypothesis`: seed, offset, distribution, layout parameters, reconstruction settings, and score.
- `ValidationResult`: fitting scores, held-out scores, significance details, competing candidates, and warnings.
- `ForensicReport`: immutable input provenance, selected samples, ranked hypotheses, final classification, evidence strength, limitations, and tool version.

These structures should serialize deterministically to JSON so reports can be archived, compared, and independently reviewed.

## Test and validation plan

- Reproduce every upstream example and compare outputs against the upstream implementation.
- Unit-test Philox reconstruction using known seeds, positions, distributions, and layouts.
- Test safetensors range reads, shard selection, dtype conversion, pinned revisions, cache reuse, and deterministic sampling.
- Use synthetic tensors generated from known seeds and SM counts to test candidate ranking and held-out validation.
- Test missing seeds, small tensors, unsupported formats, network failures, unavailable revisions, and malformed metadata.
- Test CPU-initialized, quantized, merged, heavily fine-tuned, and custom-initialized cases as inconclusive or appropriately limited.
- Test that a high fitting correlation without held-out support is downgraded.
- Test JSON schema stability and human-readable report rendering.
- Run formatting, static checks, unit tests, and the upstream reproduction suite in CI.

## Limitations and non-goals

- This method does not identify the hardware used for every training step.
- CPU initialization cannot be fingerprinted as a GPU layout.
- Quantization, substantial transformation, merging, and fine-tuning can destroy or weaken the signal.
- Small initialization operations may never reach the hardware-dependent execution-grid limit.
- Unknown-seed recovery can dominate runtime and may remain infeasible for a given model.
- A layout match does not uniquely prove a GPU vendor or exact product.
- The first release will not promise successful attribution for every arbitrary Hugging Face model.

## Acceptance criteria

The implementation is ready for an initial release when it can:

1. Reproduce the upstream examples with attribution-preserving code.
2. Inspect a pinned Hugging Face repository without executing remote code.
3. Select and retrieve deterministic tensor samples by range.
4. Run known-seed candidate-layout comparisons through the CLI.
5. Validate the leading candidate on held-out data and compare alternatives.
6. Produce a reproducible JSON report with provenance, evidence, limitations, and an explicit inconclusive state.
7. Pass the synthetic, upstream-regression, error-handling, and safety tests described above.

## Assumptions

- This empty workspace will be bootstrapped as the `atripathy86` fork/adaptation rather than treated as an independent reimplementation.
- The upstream MIT license, attribution, notices, and research citations remain in the resulting repository.
- Known-seed and experimentally reproducible models are the first milestone; automatic arbitrary-model seed discovery is optional and resource-intensive.
- Remote model repositories are read-only inputs, and arbitrary model code is never executed.
