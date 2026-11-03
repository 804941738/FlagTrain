# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic host/launch/device breakdown; not formal performance acceptance."""

import argparse
import cProfile
import importlib
import json
import pstats
import time
from pathlib import Path

import torch
import torch_npu

import flag_train
from bench_bias_gelu_910b import npu_bench


SHAPES = [(64, 64), (1024, 1024), (16384, 4096)]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if flag_train.vendor_name != "ascend":
        parser.error("requires an Ascend NPU")
    args.output.mkdir(parents=True, exist_ok=True)
    module = importlib.import_module("flag_train.megatron.bias_gelu_impl")
    entry = module._bias_gelu_forward_kernel
    rows = []
    for shape in SHAPES:
        x = torch.randn(shape, device="npu", dtype=torch.float16)
        b = torch.randn(shape[-1], device=x.device, dtype=x.dtype)
        y = torch.empty_like(x)
        public = lambda: flag_train.megatron.bias_gelu_impl(x, b)
        for _ in range(25):
            public()
        torch.npu.synchronize()
        n, h = x.numel(), shape[-1]
        lanes = (n + module.CORE_NUM - 1) // module.CORE_NUM
        block = min(16384, max(128, 1 << (lanes - 1).bit_length()))
        grid = (min((n + block - 1) // block, module.CORE_NUM), 1, 1)
        even = n % block == 0 and h % block == 0
        values = (x, b, y, n, h, 1, block, True, even)
        key = entry.key([x, b, y, n], [], [h, 1, block, True, even])
        kernel = entry.kernel_cache[torch.npu.current_device()][key][0]
        # A fixed stream is used only in the diagnostic stages that isolate
        # stream-query cost. The public and LibEntry stages query it normally.
        stream = torch.npu.current_stream().npu_stream

        def context():
            with module.torch_device_fn.device(x.device):
                pass

        stages = {
            "public": public,
            "libentry_preallocated": lambda: entry[grid](
                x, b, y, n, h, 1, BLOCK=block, PERSISTENT=True,
                EVEN=even, num_warps=4, num_stages=1,
            ),
            "compiled_runner": lambda: kernel[grid](*values),
            "runner_fixed_stream": lambda: kernel[grid](*values, stream=stream),
            "launcher_fixed_stream": lambda: kernel.run(
                *grid, stream, kernel.function, kernel.packed_metadata,
                None, None, None, *values,
            ),
            "allocation": lambda: torch.empty(x.shape, dtype=x.dtype, device=x.device),
            "device_context": context,
        }
        for name, fn in stages.items():
            event_us = npu_bench(fn)
            torch.npu.synchronize()
            start = time.perf_counter()
            for _ in range(1000):
                fn()
            host_us = (time.perf_counter() - start) * 1000
            torch.npu.synchronize()
            row = dict(shape=shape, stage=name, event_us=event_us,
                       host_submission_us=host_us)
            rows.append(row)
            print(json.dumps(row), flush=True)
        profile = cProfile.Profile()
        profile.enable()
        for _ in range(200):
            public()
        profile.disable()
        torch.npu.synchronize()
        with (args.output / f"cpu_{shape[0]}.txt").open("w") as out:
            pstats.Stats(profile, stream=out).strip_dirs().sort_stats("cumtime").print_stats(40)
        p = torch_npu.profiler
        with p.profile(
            activities=[p.ProfilerActivity.CPU, p.ProfilerActivity.NPU],
            schedule=p.schedule(wait=0, warmup=1, active=5, repeat=1),
            experimental_config=p._ExperimentalConfig(
                profiler_level=p.ProfilerLevel.Level1,
                aic_metrics=p.AiCMetrics.PipeUtilization,
            ),
            on_trace_ready=p.tensorboard_trace_handler(str(args.output / f"profile_{shape[0]}")),
        ) as prof:
            for _ in range(6):
                with torch.profiler.record_function("flagtrain_public"):
                    public()
                with torch.profiler.record_function("torch_eager"):
                    torch.nn.functional.gelu(x + b, approximate="tanh")
                prof.step()
            torch.npu.synchronize()
    (args.output / "breakdown.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
