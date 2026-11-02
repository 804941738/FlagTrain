# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Public-interface eager comparison; same timing boundary as benchmark/base.py."""

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import statistics
import subprocess
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
from bias_gelu_paths import generic_execution

import flag_train

SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (16384, 4096),
    (2048, 11008),
    (16, 128, 8192),
    (256, 32768),
]


def stats(rows):
    values = [r["speedup"] for r in rows]
    return dict(
        count=len(values),
        arithmetic=statistics.mean(values),
        geometric=math.exp(statistics.mean(map(math.log, values))),
        minimum=min(values),
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--shape", type=int, choices=range(7))
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--direction", choices=["forward", "backward"])
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--implementation", choices=["default", "generic"], default="default"
    )
    a = p.parse_args()
    if a.rounds < 1:
        p.error("--rounds must be positive")
    with generic_execution() if a.implementation == "generic" else nullcontext():
        run(a)


def run(a):
    rows = []
    torch.manual_seed(2026)
    for repeat in range(a.rounds):
        for si, shape in enumerate(SHAPES):
            if a.shape is not None and si != a.shape:
                continue
            for dtype_name in ["float16", "bfloat16", "float32"]:
                if a.dtype and a.dtype != dtype_name:
                    continue
                torch.cuda.reset_peak_memory_stats()
                dtype = getattr(torch, dtype_name)
                x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
                b = torch.randn(
                    shape[-1], device="cuda", dtype=dtype, requires_grad=True
                )
                g = torch.randn_like(x)
                for direction in ["forward", "backward"]:
                    if a.direction and a.direction != direction:
                        continue
                    funcs = []
                    for op in [
                        lambda x, b: F.gelu(x + b, approximate="tanh"),
                        flag_train.megatron.bias_gelu_impl,
                    ]:
                        if direction == "backward":
                            y = op(x, b)
                            funcs.append(
                                lambda y=y, x=x, b=b, g=g: torch.autograd.grad(
                                    y, (x, b), g, retain_graph=True
                                )
                            )
                        else:
                            funcs.append(lambda op=op, x=x, b=b: op(x, b))
                    times = [None, None]
                    for idx in ([0, 1] if repeat % 2 == 0 else [1, 0]):
                        times[idx] = (
                            triton.testing.do_bench(
                                funcs[idx],
                                warmup=25,
                                rep=100,
                                return_mode="median",
                                grad_to_none=(
                                    [x, b] if direction == "backward" else None
                                ),
                            )
                            * 1000
                        )
                    row = dict(
                        round=repeat,
                        shape=list(shape),
                        dtype=dtype_name,
                        direction=direction,
                        torch_us=times[0],
                        flagtrain_us=times[1],
                        speedup=times[0] / times[1],
                        memory_peak=torch.cuda.max_memory_allocated(),
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    del funcs
                del x, b, g
                if direction == "backward":
                    del y
                gc.collect()
    medians = []
    for row in rows:
        key = (row["shape"], row["dtype"], row["direction"])
        if any((r["shape"], r["dtype"], r["direction"]) == key for r in medians):
            continue
        group = [r for r in rows if (r["shape"], r["dtype"], r["direction"]) == key]
        medians.append(
            {
                **row,
                "speedup": statistics.median(r["speedup"] for r in group),
                "torch_us": statistics.median(r["torch_us"] for r in group),
                "flagtrain_us": statistics.median(r["flagtrain_us"] for r in group),
            }
        )
    summaries = {"all": stats(medians)}
    for field in ["dtype", "direction"]:
        for value in sorted(set(r[field] for r in medians)):
            summaries[value] = stats([r for r in medians if r[field] == value])
    failures = {
        d: summaries[d]
        for d in ("forward", "backward")
        if d in summaries and summaries[d]["geometric"] < 0.8
    }
    slow_cases = [r for r in medians if r["speedup"] < 0.8]
    result = dict(
        rows=rows,
        medians=medians,
        summary=summaries,
        failures=failures,
        slow_cases=slow_cases,
        below_hard_gate=[r for r in rows if r["speedup"] < 0.8],
        environment=dict(
            implementation=a.implementation,
            public_module=flag_train.megatron.bias_gelu_impl.__module__,
            commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            source_sha256=hashlib.sha256(
                b"".join(
                    p.read_bytes()
                    for p in sorted(
                        Path("src/flag_train/megatron").rglob("*bias*gelu*")
                    )
                    if p.is_file() and p.suffix == ".py"
                )
            ).hexdigest(),
            megatron_core=importlib.metadata.version("megatron-core"),
            torch=torch.__version__,
            triton=triton.__version__,
            gpu=torch.cuda.get_device_name(),
            device=torch.cuda.current_device(),
        ),
        timing=dict(warmup=25, rep=100, return_mode="median"),
    )
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps(result, indent=2))
    with path.with_suffix(".csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summaries, indent=2), flush=True)
    print(
        f"DIRECTION GEOMEAN FAILURES <0.80: {len(failures)}; "
        f'individual measurements <0.80: {len(result["below_hard_gate"])}',
        flush=True,
    )


if __name__ == "__main__":
    main()
