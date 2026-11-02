# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""GenericBenchmark forward/backward against installed Megatron, no fallback.

Backward includes Autograd and bias reduction on both sides. This differs from
the standalone V1 ATen backward microbenchmark. Missing Megatron is an error.
The baseline explicitly disables Megatron's fuser to measure eager ATen kernels.
"""

import pytest
import torch

import flag_train
from baseline.megatron.bias_gelu_impl import native

from .. import base

_CORE_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (16384, 4096),
    (2048, 11008),
    (16, 128, 8192),
    (256, 32768),
]


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
@pytest.mark.parametrize("is_backward", [False, True], ids=["forward", "backward"])
def test_bias_gelu_impl_perf(is_backward):
    dtypes = [torch.float16, torch.float32]
    if flag_train.runtime.device.support_bf16:
        dtypes.append(torch.bfloat16)
    bench = BiasGeluBenchmark(
        input_fn=bias_gelu_input_fn,
        op_name="bias_gelu_impl",
        torch_op=native(),
        dtypes=dtypes,
        is_backward=is_backward,
        native_baseline=True,
    )
    bench.set_train(flag_train.megatron.bias_gelu_impl)
    bench.run()
