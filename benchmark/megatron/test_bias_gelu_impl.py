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
"""Forward benchmark against Megatron's default JIT, with labelled group fallback.

The NVIDIA reference collector in baseline/megatron remains Eager. This entry
imports the native GeLUFunction.apply independently: its inner bias_gelu, not
apply itself, owns the torch.compile wrapper. Compile every case before timing;
if a non-NVIDIA backend cannot compile, replay the whole group in force_eager.
"""

import importlib
import importlib.metadata
import json
import traceback
from contextlib import contextmanager, nullcontext

import pytest
import torch

import flag_train
from flag_train.runtime import torch_device_fn

from .. import base, conftest

_CORE_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (16384, 4096),
    (2048, 11008),
    (16, 128, 8192),
    (256, 32768),
]

_COMPILE_BUDGET = 128


def base_op():
    """Return the installed native interface without changing Megatron JIT state."""
    try:
        module = importlib.import_module("megatron.core.fusions.fused_bias_gelu")
    except ImportError as exc:
        raise RuntimeError(
            "bias_gelu_impl benchmark requires installed megatron-core; "
            "a Torch formula is not a Megatron performance baseline"
        ) from exc
    return module.bias_gelu_impl


@contextmanager
def _compile_budget():
    """PR #50's scoped budget, including the legacy config aliases."""
    config = torch._dynamo.config
    updates = {
        name: max(getattr(config, name), _COMPILE_BUDGET)
        for name in (
            "recompile_limit",
            "cache_size_limit",
            "accumulated_recompile_limit",
        )
        if hasattr(config, name)
    }
    updates["suppress_errors"] = False
    # These are fixed-shape benchmarks. Automatic generalization after the
    # second shape would replace the per-shape Inductor kernels with a slower
    # dynamic kernel, changing the baseline versus isolated case measurements.
    if hasattr(config, "automatic_dynamic_shapes"):
        updates["automatic_dynamic_shapes"] = False
    if hasattr(config, "fail_on_recompile_limit_hit"):
        updates["fail_on_recompile_limit_hit"] = True
    with config.patch(updates):
        yield


def _is_compile_failure(exc):
    """Fallback only for compiler/backend failures, never numerical/runtime errors."""
    from torch._dynamo.exc import BackendCompilerFailed, Unsupported

    pending, seen = [exc], set()
    while pending:
        inner = pending.pop()
        if inner is None or id(inner) in seen:
            continue
        seen.add(id(inner))
        if isinstance(inner, (AssertionError, ValueError, torch.OutOfMemoryError)):
            return False
        pending.extend(
            [
                getattr(inner, "inner_exception", None),
                inner.__cause__,
                inner.__context__,
            ]
        )
    return isinstance(exc, (BackendCompilerFailed, Unsupported))


def _execution_mode(mode, jit_wrapped=True):
    if mode == "compiled":
        # A fail_on_recompile stance changes the hot eval-frame callback and
        # adds host dispatch overhead. This config checks only cache misses.
        return torch._dynamo.config.patch(error_on_recompile=True)
    if mode == "eager" and not jit_wrapped:
        return nullcontext()
    stance = getattr(torch.compiler, "set_stance", None)
    if stance is not None:
        return stance("force_eager")
    if mode == "eager":
        raise RuntimeError(
            "Safe native Megatron Eager fallback requires torch.compiler.set_stance; "
            "no module reload or outer apply unwrapping is allowed"
        )


def _compile_counts():
    from torch._dynamo.utils import counters

    return {
        name: dict(counters[name])
        for name in ("frames", "stats", "graph_break", "unimplemented")
    }


def _probe(reference, args, compiled):
    # Both calls and synchronization are outside GenericBenchmark's timer.
    # Do not even warm up with a profiler here: on some PyTorch builds its
    # hooks affect subsequent host dispatch. GPU traces belong in another process.
    before = _compile_counts() if compiled else None
    reference(*args)
    torch_device_fn.synchronize()
    if compiled:
        from torch._dynamo.exc import Unsupported

        after = _compile_counts()
        for key in ("graph_break", "unimplemented"):
            if after[key] != before[key]:
                raise Unsupported(f"Megatron {key}: {after[key]}")
        if not after["stats"].get("unique_graphs", 0):
            raise RuntimeError("Megatron did not produce a compiled graph")
        reference(*args)
        torch_device_fn.synchronize()
        return {"compiled_graphs": after["stats"]["unique_graphs"]}
    return {}


def _preflight(bench, reference, metadata, compiled):
    for dtype in bench.to_bench_dtypes:
        for shape in bench.shapes:
            case = {"shape": list(shape), "dtype": str(dtype)}
            metadata["current_case"] = case
            args = next(bias_gelu_input_fn(shape, dtype, bench.device))
            evidence = _probe(reference, args, compiled)
            metadata["cases"].append(
                dict(case, mode="compiled" if compiled else "eager", **evidence)
            )
    metadata.pop("current_case", None)


def _prepare_baseline(bench, reference):
    module = importlib.import_module("megatron.core.fusions.fused_bias_gelu")
    compiled = hasattr(module.bias_gelu, "_torchdynamo_orig_callable")
    metadata = {
        "package": "megatron-core",
        "version": importlib.metadata.version("megatron-core"),
        "operator": "megatron.core.fusions.fused_bias_gelu.bias_gelu_impl",
        "native_interface": "GeLUFunction.apply",
        "source": module.__file__,
        "mode": "compiled" if compiled else "eager",
        "jit_wrapped": compiled,
        "fallback_reason": None,
        "cases": [],
    }
    if not compiled:
        if flag_train.vendor_name == "nvidia":
            raise RuntimeError(
                "NVIDIA requires Megatron default torch.compile; inner bias_gelu "
                "is Eager. Run a fresh process without the reference collector."
            )
        metadata["fallback_reason"] = "Megatron's internal bias_gelu JIT is disabled"
    else:
        try:
            _preflight(bench, reference, metadata, compiled=True)
            return metadata
        except Exception as exc:
            if not _is_compile_failure(exc):
                raise
            reason = f"{type(exc).__name__}: {exc}"
            if flag_train.vendor_name == "nvidia":
                raise RuntimeError(
                    f"NVIDIA requires compiled Megatron: {reason}"
                ) from exc
            metadata.update(
                mode="eager",
                fallback_reason=reason,
                compilation_traceback=traceback.format_exc(),
                compile_attempts=metadata["cases"],
                failed_case=metadata.pop("current_case", None),
                cases=[],
            )
    # Select one baseline for the entire table, including cases compiled earlier.
    context = _execution_mode("eager") if compiled else nullcontext()
    with context:
        _preflight(bench, reference, metadata, compiled=False)
    return metadata


class BiasGeluBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _CORE_SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_CORE_SHAPES)


def bias_gelu_input_fn(shape, dtype, device):
    yield (
        torch.randn(shape, dtype=dtype, device=device),
        torch.randn(shape[-1], dtype=dtype, device=device),
    )


@pytest.mark.bias_gelu_impl
def test_bias_gelu_impl_perf():
    dtypes = [torch.float16, torch.float32]
    if flag_train.runtime.device.support_bf16:
        dtypes.append(torch.bfloat16)
    bench = BiasGeluBenchmark(
        input_fn=bias_gelu_input_fn,
        op_name="bias_gelu_impl",
        torch_op=base_op(),
        dtypes=dtypes,
        native_baseline=True,
    )
    bench.set_train(flag_train.megatron.bias_gelu_impl)
    if base.Config.query or base.Config.list_cases:
        return bench.run()
    if base.Config.skip_native:
        raise RuntimeError(
            "Megatron performance validation requires the native baseline"
        )
    with _compile_budget(), torch.no_grad():
        bench.init_user_config()
        metadata = _prepare_baseline(bench, bench.torch_op)
        metadata["timing_mode"] = base.Config.mode.value
        metadata["compiler_settings"] = {
            name: getattr(torch._dynamo.config, name)
            for name in (
                "recompile_limit",
                "cache_size_limit",
                "automatic_dynamic_shapes",
                "suppress_errors",
            )
            if hasattr(torch._dynamo.config, name)
        }
        metadata["compile_counts_after_preflight"] = _compile_counts()
        print(f"\nBaseline: megatron-core {metadata['version']}")
        print(
            f"Baseline Mode: {metadata['mode']}"
            + (
                " (Megatron default torch.compile)"
                if metadata["mode"] == "compiled"
                else ""
            )
        )
        print("Operator: bias_gelu_impl (native GeLUFunction.apply)")
        if metadata["fallback_reason"]:
            print(f"Fallback Reason: {metadata['fallback_reason']}")
        print("Baseline Metadata: " + json.dumps(metadata))
        if base.Config.record_json:
            conftest.TEST_RESULTS.setdefault(bench.op_name, {})["baseline"] = metadata
        with _execution_mode(metadata["mode"], metadata["jit_wrapped"]):
            before = _compile_counts()
            bench.run()
            after = _compile_counts()
            if after != before:
                raise RuntimeError(
                    "Megatron compilation activity occurred during timing"
                )
        metadata["timing_compile_counts_unchanged"] = True
