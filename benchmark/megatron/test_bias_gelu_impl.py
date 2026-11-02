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
"""Forward-only GenericBenchmark against the installed Megatron Eager baseline."""

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
def test_bias_gelu_impl_perf():
    dtypes = [torch.float16, torch.float32]
    if flag_train.runtime.device.support_bf16:
        dtypes.append(torch.bfloat16)
    bench = BiasGeluBenchmark(
        input_fn=bias_gelu_input_fn,
        op_name="bias_gelu_impl",
        torch_op=native(),
        dtypes=dtypes,
        native_baseline=True,
    )
    bench.set_train(flag_train.megatron.bias_gelu_impl)
    bench.run()
