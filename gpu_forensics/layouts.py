"""Hardware layout hypotheses for PyTorch's CUDA RNG kernels.

PyTorch's ``calc_execution_policy`` (aten/src/ATen/native/cuda/DistributionTemplates.h)
launches ``grid.x = min(multiProcessorCount * maxThreadsPerMultiProcessor / 256,
ceil(numel / 256))`` blocks of 256 threads.  The fingerprint is therefore the total
thread count ``G = 256 * grid.x``, not the SM count alone: different devices with
equal ``SMs * threadsPerSM`` produce identical layouts.
"""

from __future__ import annotations

from dataclasses import dataclass

BLOCK = 256


@dataclass(frozen=True)
class Device:
    name: str
    sms: int
    threads_per_sm: int
    note: str = ""

    @property
    def threads(self) -> int:
        return self.sms * (self.threads_per_sm // BLOCK) * BLOCK


# Published values for common accelerators.  AMD devices report compute units as
# multiProcessorCount under ROCm.  Entries are layout hypotheses, not identifiers.
DEVICES = [
    Device("V100", 80, 2048),
    Device("A100 / A800", 108, 2048),
    Device("A10", 72, 1536),
    Device("A40 / RTX A6000", 84, 1536),
    Device("L4", 58, 1536),
    Device("L40 / L40S", 142, 1536),
    Device("RTX 3090", 82, 1536),
    Device("RTX 4090", 128, 1536),
    Device("RTX 5090", 170, 1536),
    Device("RTX PRO 6000 Blackwell", 188, 1536),
    Device("H20", 78, 2048),
    Device("H100 PCIe", 114, 2048),
    Device("H100 SXM / H800 / H200", 132, 2048),
    Device("B200 / B300", 148, 2048, "as identified in the checkpoint-forensics report"),
    Device("GB200 / GB300", 152, 2048),
    Device("GB10", 48, 1536),
    Device("MI250X (per GCD)", 110, 2048, "ROCm"),
    Device("MI300X / MI325X", 304, 2048, "ROCm"),
    Device("MI300X (2-way partition)", 152, 2048, "ROCm; same layout as GB200/GB300"),
]

THREADS_PER_SM = (1024, 1536, 2048)
SM_RANGE = range(16, 321)


def effective_threads(numel: int, threads: int) -> int:
    """Total threads actually launched for a tensor of ``numel`` elements."""
    blocks = -(-numel // BLOCK)
    return min(threads // BLOCK, blocks) * BLOCK


def philox_offset_increment(numel: int, threads: int, unroll: int = 4) -> int:
    """How far one ``normal_``/``uniform_`` call advances the Philox offset."""
    g = effective_threads(numel, threads)
    return ((numel - 1) // (g * unroll) + 1) * 4


def devices_for(threads: int) -> list[Device]:
    return [d for d in DEVICES if d.threads == threads]


def describe(threads: int) -> str:
    names = [d.name for d in devices_for(threads)]
    return " / ".join(names) if names else "no catalogued device"


def candidate_threads(threads_per_sm=THREADS_PER_SM, sms=SM_RANGE) -> list[int]:
    """Every distinct ``G`` produced by the given SM counts and threads/SM values."""
    return sorted({s * (t // BLOCK) * BLOCK for s in sms for t in threads_per_sm})


def equivalent_configs(threads: int, threads_per_sm=THREADS_PER_SM) -> list[tuple[int, int]]:
    """(SMs, threads/SM) pairs that produce the same ``G``."""
    return [(threads // t, t) for t in threads_per_sm if threads % t == 0]


def catalogue_threads() -> list[int]:
    return sorted({d.threads for d in DEVICES})
