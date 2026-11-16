"""Generate ground-truth initialization fixtures with stock PyTorch on a CUDA device.

Each fixture records the device layout (SMs, threads/SM), the exact init call, any
preceding RNG calls (which advance the Philox offset), and a few rows of raw
output bits.  ``tests/test_fixtures.py`` then checks that the CPU reconstruction
in ``gpu_forensics.recipes`` reproduces them bit for bit.

    python tools/synthesize_fixtures.py --out fixtures/

Requires a CUDA build of PyTorch (not a project dependency).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

ROWS, WIDTH = 4096, 1024
PICK = [0, 1, ROWS // 2, ROWS - 2, ROWS - 1]
CASES = [
    # name, dtype, init, params, preceding RNG call sizes
    ("normal_f32", torch.float32, "normal", {"mean": 0.0, "std": 0.02}, []),
    ("normal_bf16", torch.bfloat16, "normal", {"mean": 0.0, "std": 0.006}, []),
    ("normal_f16", torch.float16, "normal", {"mean": 0.0, "std": 0.02}, []),
    ("normal_bf16_offset", torch.bfloat16, "normal", {"mean": 0.0, "std": 0.02}, [1000, 3_000_000]),
    ("uniform_f32", torch.float32, "uniform", {"low": -0.05, "high": 0.05}, []),
    ("uniform_bf16", torch.bfloat16, "uniform", {"low": -0.05, "high": 0.05}, []),
    ("trunc_normal_bf16_3sigma", torch.bfloat16, "trunc_normal", {"mean": 0.0, "std": 0.02, "bound_sigmas": 3.0}, []),
    ("trunc_normal_f32_default", torch.float32, "trunc_normal", {"mean": 0.0, "std": 0.02, "a": -2.0, "b": 2.0}, []),
]


def init(t, kind, p):
    if kind == "normal":
        t.normal_(p["mean"], p["std"])
    elif kind == "uniform":
        t.uniform_(p["low"], p["high"])
    else:
        a, b = p.get("a"), p.get("b")
        if "bound_sigmas" in p:
            a, b = p["mean"] - p["bound_sigmas"] * p["std"], p["mean"] + p["bound_sigmas"] * p["std"]
        torch.nn.init.trunc_normal_(t, mean=p["mean"], std=p["std"], a=a, b=b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="fixtures")
    parser.add_argument("--seed", type=int, default=123456789)
    args = parser.parse_args()
    props = torch.cuda.get_device_properties(0)
    tps = getattr(props, "max_threads_per_multi_processor", 2048)
    device = {"name": props.name, "sms": props.multi_processor_count, "threads_per_sm": tps,
              "threads": props.multi_processor_count * (tps // 256) * 256, "torch": torch.__version__}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    slug = props.name.replace(" ", "_")
    for name, dtype, kind, params, pre in CASES:
        torch.cuda.manual_seed(args.seed)
        for n in pre:
            torch.empty(n, device="cuda").normal_()
        t = torch.empty((ROWS, WIDTH), dtype=dtype, device="cuda")
        init(t, kind, params)
        rows = t[PICK].float().cpu().numpy() if dtype != torch.float32 else t[PICK].cpu().numpy()
        meta = {"case": name, "device": device, "seed": args.seed, "dist": kind, "params": params,
                "dtype": str(dtype).replace("torch.", ""), "shape": [ROWS, WIDTH], "rows": PICK,
                "preceding_numels": pre}
        np.save(out / f"{slug}-{name}.npy", rows.astype(np.float32))
        (out / f"{slug}-{name}.json").write_text(json.dumps(meta, indent=2))
        print(f"{name}: {device['threads']} threads, finite={math.isfinite(float(rows.sum()))}")


if __name__ == "__main__":
    main()
