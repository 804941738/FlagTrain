# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native dispatch regressions beyond the shared operator contract suite."""

import pytest
import torch

import flag_train
from flag_train.runtime import device

pytestmark = pytest.mark.skipif(device.vendor_name != "nvidia", reason="CUDA dispatch")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_finite_extreme_gradients(dtype):
    x = torch.tensor(
        [[-1e20, 1e20]], device=flag_train.device, dtype=dtype, requires_grad=True
    )
    b = torch.zeros(2, device=x.device, dtype=dtype, requires_grad=True)
    out = flag_train.megatron.bias_gelu_impl(x, b)
    dx, db = torch.autograd.grad(out, (x, b), torch.ones_like(out))
    torch.testing.assert_close(dx, x.new_tensor([[0, 1]]))
    torch.testing.assert_close(db, b.new_tensor([0, 1]))


def test_saved_tensor_version_check():
    x = torch.randn(64, 64, device=flag_train.device, requires_grad=True)
    b = torch.randn(64, device=x.device, requires_grad=True)
    y = flag_train.megatron.bias_gelu_impl(x, b)
    with torch.no_grad():
        b.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(y.sum(), (x, b))


def test_nondefault_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(64, 64, device=flag_train.device, requires_grad=True)
        b = torch.randn(64, device=x.device, requires_grad=True)
        y = flag_train.megatron.bias_gelu_impl(x, b)
        g = torch.randn_like(y)
        actual = torch.autograd.grad(y, (x, b), g)
        reference = torch.nn.functional.gelu(x + b, approximate="tanh")
        expected = torch.autograd.grad(reference, (x, b), g)
    stream.synchronize()
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=2e-5, atol=2e-5)
