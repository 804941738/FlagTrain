# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Forward-only Ascend 910B comparison of the public operator against PyTorch Eager."""

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu
import triton

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
DTYPES = ("float16", "bfloat16", "float32")


def stats(rows):
    values = [r["speedup"] for r in rows]
    return dict(
        count=len(values),
        arithmetic=statistics.mean(values),
        geometric=math.exp(statistics.mean(map(math.log, values))),
        minimum=min(values),
    )


def summarize(rows, rounds):
    groups = {}
    for row in rows:
        groups.setdefault((tuple(row["shape"]), row["dtype"]), []).append(row)
    medians = [
        {
            **group[0],
            **{
                key: statistics.median(r[key] for r in group)
                for key in ("torch_us", "flagtrain_us", "speedup")
            },
        }
        for group in groups.values()
    ]
    summary = stats(medians)
    expected = {(shape, dtype) for shape in SHAPES for dtype in DTYPES}
    complete = (
        rounds >= 3
        and set(groups) == expected
        and all(
            len(group) == rounds and {r["round"] for r in group} == set(range(rounds))
            for group in groups.values()
        )
    )
    return dict(
        rows=rows,
        medians=medians,
        summary=summary,
        by_dtype={
            dtype: stats([r for r in medians if r["dtype"] == dtype])
            for dtype in sorted({r["dtype"] for r in rows})
        },
        round_summary={
            repeat: stats([r for r in rows if r["round"] == repeat])
            for repeat in range(rounds)
        },
        acceptance=dict(
            metric="arithmetic_mean_of_case_median_speedups",
            threshold=0.8,
            complete=complete,
            passed=summary["arithmetic"] >= 0.8 if complete else None,
        ),
        slow_cases=[r for r in medians if r["speedup"] < 0.8],
    )


def npu_bench(fn):
    # Events enclose a batch to amortize event resolution for microsecond kernels.
    # Each invocation includes allocation/dispatch as in the eager public API.
    for _ in range(25):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(11):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 100)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--shape", type=int, choices=range(len(SHAPES)))
    parser.add_argument("--dtype", choices=DTYPES)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")

    rows = []
    torch.manual_seed(2026)
    for repeat in range(args.rounds):
        for index, shape in enumerate(SHAPES):
            if args.shape is not None and args.shape != index:
                continue
            for dtype_name in DTYPES:
                if args.dtype and args.dtype != dtype_name:
                    continue
                torch.npu.reset_peak_memory_stats()
                dtype = getattr(torch, dtype_name)
                x = torch.randn(shape, device="npu", dtype=dtype)
                bias = torch.randn(shape[-1], device=x.device, dtype=dtype)
                functions = [
                    lambda x=x, bias=bias: F.gelu(x + bias, approximate="tanh"),
                    lambda x=x, bias=bias: flag_train.megatron.bias_gelu_impl(x, bias),
                ]
                times = [None, None]
                for side in ([0, 1] if repeat % 2 == 0 else [1, 0]):
                    times[side] = npu_bench(functions[side])
                row = dict(
                    round=repeat,
                    shape=list(shape),
                    dtype=dtype_name,
                    torch_us=times[0],
                    flagtrain_us=times[1],
                    speedup=times[0] / times[1],
                    memory_peak=torch.npu.max_memory_allocated(),
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
                del functions, x, bias
                gc.collect()

    result = summarize(rows, args.rounds)
    result.update(
        environment=dict(
            public_module=flag_train.megatron.bias_gelu_impl.__module__,
            commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            measured_sources_dirty=subprocess.run(
                [
                    "git",
                    "diff",
                    "--quiet",
                    "HEAD",
                    "--",
                    "src/flag_train/megatron/bias_gelu_impl.py",
                    "src/flag_train/utils/libentry.py",
                    "tools/bench_bias_gelu_910b.py",
                ],
                check=False,
            ).returncode
            != 0,
            source_sha256=hashlib.sha256(
                Path("src/flag_train/megatron/bias_gelu_impl.py").read_bytes()
            ).hexdigest(),
            libentry_source_sha256=hashlib.sha256(
                Path("src/flag_train/utils/libentry.py").read_bytes()
            ).hexdigest(),
            benchmark_source_sha256=hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            python=platform.python_version(),
            torch=torch.__version__,
            triton=triton.__version__,
            gpu=torch.npu.get_device_name(),
            torch_npu=torch_npu.__version__,
            flagtree=importlib.metadata.version("flagtree"),
            cann=Path(
                "/usr/local/Ascend/cann/share/info/runtime/version.info"
            ).read_text(),
            device=torch.npu.current_device(),
            ascend_visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        ),
        timing=dict(
            method="NPU Event batch latency",
            warmup=25,
            batches=11,
            iterations=100,
            return_mode="median",
            requires_grad=False,
        ),
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    for records, target in [
        (rows, path.with_suffix(".csv")),
        (result["medians"], path.with_name(path.stem + "_medians.csv")),
    ]:
        with target.open("w") as output:
            writer = csv.DictWriter(output, fieldnames=records[0])
            writer.writeheader()
            writer.writerows(records)
    print(
        json.dumps(
            {k: result[k] for k in ("summary", "round_summary", "acceptance")}, indent=2
        ),
        flush=True,
    )
    if result["acceptance"]["complete"]:
        print(
            "FORMAL FORWARD PERFORMANCE: "
            + ("PASS" if result["acceptance"]["passed"] else "FAIL"),
            flush=True,
        )
        if not result["acceptance"]["passed"]:
            raise SystemExit(1)
    else:
        print(
            "DIAGNOSTIC ONLY: formal acceptance requires all 21 cases and >=3 complete rounds",
            flush=True,
        )


if __name__ == "__main__":
    main()
