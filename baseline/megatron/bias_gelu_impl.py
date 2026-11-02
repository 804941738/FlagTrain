# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native eager Megatron baseline; unavailable dependencies raise, never fallback.

Forward collection uses the same shapes/dtypes as the GenericBenchmark module.
Backward timing is provided by GenericBenchmark's is_backward path.
"""

import importlib

import torch

OP_NAME = "bias_gelu_impl"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (16384, 4096),
    (2048, 11008),
    (16, 128, 8192),
    (256, 32768),
]
_NATIVE = None


def native():
    global _NATIVE
    if _NATIVE is None:
        try:
            import megatron.core.jit as megatron_jit

            megatron_jit.disable_jit_fuser()
            module = importlib.import_module("megatron.core.fusions.fused_bias_gelu")
            # megatron.core may have imported the fusion before disable_jit_fuser.
            if hasattr(module.bias_gelu, "_torchdynamo_orig_callable") or hasattr(
                module.bias_gelu_back, "_torchdynamo_orig_callable"
            ):
                module = importlib.reload(module)
            _NATIVE = module.bias_gelu_impl
        except Exception as exc:
            raise RuntimeError(
                "bias_gelu_impl baseline requires installed Megatron-LM's "
                "megatron.core.fusions.fused_bias_gelu (eager); no formula fallback"
            ) from exc
    return _NATIVE


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = binding["shape"]
    return (
        torch.randn(shape, dtype=dtype, device=device),
        torch.randn(shape[-1], dtype=dtype, device=device),
    ), {}


def key_shape(binding):
    return list(binding["shape"])
