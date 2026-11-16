The Unemployment Labs research describes a powerful technique: determining which GPU architecture initialized an AI model by analyzing numerical fingerprints embedded in its published weights.

You could build a Python tool that accepts an arbitrary Hugging Face model URL, downloads selected weights, and attempts to identify the GPU architecture used during initialization.

The important distinction is that it fingerprints initialization hardware, not necessarily the hardware used throughout training.

1. How the technique works

The method exploits an implementation detail in PyTorch’s CUDA random-number generation.

When PyTorch initializes model weights, it typically uses a pseudorandom number generator called Philox. The generated numbers are deterministic for a given seed, but their placement in a tensor can depend on the GPU’s hardware configuration.

Conceptual example: each cell shows the worker that supplies a weight. Changing the worker count changes the placement pattern, even with the same random seed.

The process is:

* Seed: PyTorch initializes Philox with a seed, such as 953049583.
* GPU execution: CUDA distributes the generation of random values across many GPU threads.
* Hardware-dependent layout: The number of streaming multiprocessors (SMs) influences the total number of threads.
* Weight placement: Threads write their generated values into specific positions in the tensor.
* Fingerprint: Different SM counts produce different arrangements of otherwise reproducible random numbers.

For sufficiently large initialization operations in the configuration studied, PyTorch uses up to 2,048 threads per SM.

For example:

GPU family	SM count	Approx. total threads
A100 / A800	108	221,184
H100 / H800 / H200	132	270,336
B200 / B300	148	303,104
GB200 / GB300	152	311,296

These are the layouts identified in the report, not universal identifiers for every variant of those GPUs. Unemployment Labs +1

Why does this survive training?

Some weights remain close to their initialization values, especially unused or minimally updated parameters.

Other weights retain a weak statistical correlation with their starting values.

The researchers demonstrated that a model’s final weights can still correlate with the original Philox sequence, provided the correct seed and GPU thread layout are used.

For DeepSeek-V4-Flash, they reported a correlation of approximately r = 0.066 across trained weights, which was statistically significant when aggregated across many parameters. Unemployment Labs

This makes the problem a statistical reconstruction problem, rather than simply reading hardware metadata.

2. What the PyTorch source code tells us

The relevant function in DistributionTemplates.h is calc_execution_policy().

The key calculation is:

const uint32_t block_size = 256;
uint32_t blocks_per_sm =
    maxThreadsPerMultiProcessor / block_size;
grid.x = std::min(
    multiProcessorCount * blocks_per_sm,
    grid.x
);

The kernel then initializes each thread’s Philox state:

curand_init(seed, idx, offset, &state);

And distributes the generated numbers using a grid-stride loop.

The important variables are:

* multiProcessorCount: Number of SMs.
* maxThreadsPerMultiProcessor: Hardware-specific thread capacity.
* grid.x: Number of CUDA blocks.
* idx: The thread’s unique identifier.
* seed and offset: Philox state inputs.

For large enough tensors, the grid reaches its hardware-dependent limit. That creates the fingerprint. For smaller tensors, the grid may not reach that limit, making GPU identification impossible from that operation.

GitHub

3. The researchers published working code

This is particularly useful for your project.

macrogrants/checkpoint-forensics⁠￼

It is an MIT-licensed implementation, with code to reproduce fingerprints without owning every GPU type.

File	Purpose
gpuid.py	GPU-dependent Philox reconstruction
replay.py	Tests candidate SM counts against known model weights
fetch.py	Retrieves model weight samples
models.json	Model-specific seeds, tensor information and references
gpu_check.py	Validates results against real PyTorch CUDA execution

The published replay implementation tests SM counts from 16 to 256 and uses known recovered seeds. It does not provide a complete arbitrary-model seed-discovery pipeline; the authors report that the seeds were originally brute-forced across the 32-bit search space. GitHub

You can reproduce their supplied examples with:

git clone https://github.com/macrogrants/checkpoint-forensics
cd checkpoint-forensics
pip install numpy scipy
python replay.py

This is the starting point I would use rather than rebuilding Philox from scratch.

4. Architecture for an arbitrary Hugging Face URL

Hugging Face model URL

Repository ID + pinned revision

Model inspector

Read config, model card, tensor index and initialization hints

Weight sampler

Select promising tensors and retrieve sampled values

Seed discovery

Metadata candidates, lab-specific rules, then accelerated search

Layout reconstruction

Philox, SM count, dtype, initialization and offsets

Statistical validation

Correlations, exact matches, holdout tests and competing hypotheses

GPU fingerprint report

Hardware family · Evidence strength · Limitations

Implementation stages

Stage A — Inspect the Hugging Face repository

Use huggingface_hub to read:

* config.json and the model card.
* Safetensors metadata and tensor names.
* Model architecture and parameter dimensions.
* Any published seed, initialization method, or training configuration.

Prefer pinned revisions for reproducibility. Avoid executing arbitrary remote model code.

Stage B — Select useful tensors

Prioritize embedding matrices, especially rows corresponding to unused tokens, and other parameters likely to retain initialization values.

Avoid starting with heavily trained attention matrices unless necessary.

For very large models, read selected tensor ranges rather than downloading the entire checkpoint.

Stage C — Recover the random seed

This is likely the hardest computational step.

Start with published seeds, configuration files and known seed conventions.

If unsuccessful, search the candidate seed space. A full 32-bit search contains approximately 4.29 billion candidates, so an optimized native implementation is preferable to a Python loop.

Stage D — Reconstruct candidate GPU layouts

For each seed candidate, test plausible combinations of:

* SM count and threads per SM.
* Initialization distribution (normal_, uniform_, trunc_normal_, etc.).
* Tensor shape and memory layout.
* Philox counter offsets.
* Floating-point precision and conversion behavior.

Use gpuid.py as the reference implementation, extending it for additional initialization methods.

Stage E — Validate the fingerprint

For every candidate, compute:

r=\operatorname{corr}(W_{\mathrm{observed}},W_{\mathrm{reconstructed}})

Then independently validate the best candidate on held-out weights, assess statistical significance, and compare competing SM counts.

A high correlation on untouched weights is strong evidence. A weaker correlation across many trained weights requires more careful statistical testing.

The output should support an explicit inconclusive result rather than always assigning a GPU.

5. A practical first implementation

You can already use the published gpuid.py functions to test a Hugging Face weight sample against candidate GPU layouts, provided you know the seed, tensor’s original initialization positions, and initialization method.

For a sampled tensor:

import numpy as np
from gpuid import draw, corr
def identify_gpu(weights, positions, seed):
    results = []
    for sms in range(16, 257):
        threads = 2048 * sms
        expected = draw(
            positions,
            threads,
            seed,
            dist="normal"
        )
        score = corr(weights, expected)
        results.append((sms, score))
    return sorted(
        results,
        key=lambda x: x[1],
        reverse=True
    )[:5]

This is the core comparison, not yet a complete arbitrary-URL tool. In particular, positions must represent the tensor’s actual positions in the original RNG operation, and the best score must be independently validated.

The existing fetch.py is especially useful: it already implements HTTP byte-range retrieval of selected BF16 embedding rows from pinned Hugging Face safetensors shards. GitHub +1

6. What will work and what won’t

Model situation	Expected result
PyTorch CUDA initialization, seed known, untouched weights	Strong candidate
PyTorch CUDA initialization, seed unknown	Requires expensive seed recovery
Trained weights retaining initialization signal	Possible with statistical analysis
CPU-initialized weights	GPU cannot be identified by this method
Quantized or substantially transformed weights	Often inconclusive
Custom initialization kernel	Requires additional fingerprint implementation
Fine-tuned or merged model	May identify the original base initialization

The researchers also demonstrated that specially configured AMD MI300X hardware can reproduce certain Blackwell-compatible layouts. Thus, SM layout alone does not always prove the GPU vendor. Unemployment Labs

7. Recommended development approach

I would build this as a reusable Python CLI with three modes:

* Inspect: Accept any Hugging Face URL and identify whether forensic analysis is feasible.
* Fingerprint: Automatically sample weights, discover or search for seeds, and compare hardware layouts.
* Validate: Run held-out statistical tests and produce a machine-readable evidence report.

Example intended interface:

gpu-forensics inspect \
  https://huggingface.co/org/model
gpu-forensics fingerprint \
  https://huggingface.co/org/model \
  --seed auto \
  --output report.json

These are proposed commands, not functionality already present in the published repository.

My recommendation: Fork macrogrants/checkpoint-forensics and build the generic Hugging Face ingestion and seed-discovery pipeline around its existing Philox implementation. Start by reproducing the five published examples, then validate on additional models with disclosed hardware.

The biggest engineering challenges will be automated seed recovery, reconstruction of model-specific initialization operations, and correctly reporting when the evidence is insufficient. The actual GPU fingerprint comparison is relatively straightforward once those inputs are known.
