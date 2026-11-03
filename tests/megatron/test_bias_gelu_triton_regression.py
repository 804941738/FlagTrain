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
"""Forward-only regressions: ordinary LibEntry dispatch and tensor semantics."""

import pytest
import torch
import torch.nn.functional as F

import flag_train
from flag_train.runtime import device

from .test_bias_gelu_impl import _check

pytestmark = pytest.mark.bias_gelu_impl


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_finite_extreme_values(dtype):
    limit = torch.finfo(dtype).max
    x = torch.tensor([[-limit, limit]], device=flag_train.device, dtype=dtype)
    _check(x, torch.zeros(2, device=x.device, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 7), (0, 7)])
@pytest.mark.parametrize("bias_shape", [(), (7,), (1, 7)])
def test_forward_only_output(dtype, shape, bias_shape):
    x = torch.randn(
        shape[::-1], device=flag_train.device, dtype=dtype, requires_grad=True
    ).t()
    b = torch.randn(bias_shape, device=x.device, dtype=dtype, requires_grad=True)
    y = flag_train.megatron.bias_gelu_impl(x, b)
    assert not y.requires_grad and y.grad_fn is None
    _check(x, b)


@pytest.mark.skipif(device.vendor_name != "nvidia", reason="CUDA streams")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_nondefault_stream(dtype):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(17, 31, device=flag_train.device, dtype=dtype)
        b = torch.randn(31, device=x.device, dtype=dtype)
        actual = flag_train.megatron.bias_gelu_impl(x, b)
        expected = F.gelu(x + b, approximate="tanh")
    stream.synchronize()
    tolerance = {torch.float16: 3e-3, torch.bfloat16: 2e-2, torch.float32: 2e-5}[dtype]
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_unaligned_views(dtype):
    x = torch.randn(4097, device=flag_train.device, dtype=dtype)[1:].view(64, 64)
    b = torch.randn(65, device=x.device, dtype=dtype)[1:]
    _check(x, b)


def test_standard_libentry_dispatch(monkeypatch):
    import importlib

    from flag_train.utils.libentry import LibEntry

    module = importlib.import_module("flag_train.megatron.bias_gelu_impl")
    run = LibEntry.run
    launched = []

    def record(entry, *args, **kwargs):
        launched.append(entry)
        return run(entry, *args, **kwargs)

    monkeypatch.setattr(LibEntry, "run", record)
    x = torch.randn(4, 7, device=flag_train.device)
    b = torch.randn(7, device=x.device)
    _check(x, b)
    assert launched == [module._bias_gelu_forward_kernel]


def test_single_row_large_hidden_grid():
    hidden = (1 << 26) + 1
    x = torch.zeros(1, hidden, device=flag_train.device, dtype=torch.float16)
    b = torch.zeros(hidden, device=x.device, dtype=x.dtype)
    y = flag_train.megatron.bias_gelu_impl(x, b)
    assert y.shape == x.shape and torch.all(y == 0)


@pytest.mark.skipif(device.vendor_name != "ascend", reason="Ascend coreDim regression")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n", [65535 * 256 - 1, 65535 * 256, 65535 * 256 + 1, (1 << 26) + 1])
def test_ascend_grid_boundaries(n, dtype):
    # Nonzero random data detects missing/repeated tiles, including the last lane.
    torch.manual_seed(910)
    x = torch.randn(n, device=flag_train.device, dtype=dtype)
    b = torch.tensor(0.125, device=x.device, dtype=dtype)
    y = flag_train.megatron.bias_gelu_impl(x, b)
    tolerance = {torch.float16: 3e-3, torch.bfloat16: 2e-2, torch.float32: 2e-5}[dtype]
    for start in range(0, n, 1 << 20):
        z = (x[start:start + (1 << 20)].cpu() + b.cpu()).to(dtype).double()
        ref = (0.5 * z * (1 + torch.tanh(0.79788456 * z * (1 + 0.044715 * z * z)))).to(dtype)
        torch.testing.assert_close(y[start:start + (1 << 20)].cpu(), ref,
                                   rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(device.vendor_name != "ascend", reason="Ascend row tiling regression")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("hidden", [128, 4096, 11008])
def test_ascend_row_tiling_extremes(dtype, hidden):
    x = torch.full((2, hidden), torch.finfo(dtype).max,
                   device=flag_train.device, dtype=dtype)
    x[:, ::2] = -torch.finfo(dtype).max
    _check(x, torch.zeros(hidden, device=x.device, dtype=dtype))
