# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise the portable GPU activation on a real device, including NVIDIA.

Passing NATIVE_TANH=False selects the portable arithmetic without changing
platform detection or enabling any Ascend scheduling or mathematics.
"""

import importlib

import pytest
import torch
import triton
import triton.language as tl

import flag_train
from flag_train.runtime import device as runtime_device

impl = importlib.import_module("flag_train.megatron.bias_gelu_impl")
pytestmark = [
    pytest.mark.bias_gelu_impl,
    pytest.mark.skipif(
        runtime_device.vendor_name == "ascend", reason="GPU math fallback"
    ),
]


@triton.jit
def _portable_probe(X, B, Y, N: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < N, other=0).to(tl.float32)
    b = tl.load(B + offsets % H, offsets < N, other=0).to(tl.float32)
    z = (x + b).to(X.dtype.element_ty).to(tl.float32)
    argument = 0.79788456 * z * (1.0 + 0.044715 * z * z)
    result = impl._gpu_gelu(z, argument, False)
    tl.store(Y + offsets, result, offsets < N)


def _run(x, b):
    out = torch.empty_like(x)
    _portable_probe[(triton.cdiv(x.numel(), 256),)](
        x, b, out, N=x.numel(), H=x.shape[-1], BLOCK=256, num_warps=4
    )
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("case", ["random", "dense", "special", "rounding"])
def test_portable_gpu_gelu(dtype, case):
    torch.manual_seed(1234)
    shape = (2, 4097)
    b = torch.zeros(shape[-1], device=flag_train.device, dtype=dtype)
    if case == "random":
        x = torch.randn(shape, device=flag_train.device, dtype=dtype)
        b = torch.randn_like(b)
    elif case == "dense":
        x = torch.linspace(-20, 20, shape[-1], device=b.device).to(dtype).repeat(2, 1)
    elif case == "special":
        limit, tiny = torch.finfo(dtype).max, torch.finfo(dtype).tiny
        values = torch.tensor(
            [
                -float("inf"),
                float("inf"),
                float("nan"),
                -limit,
                limit,
                -1000,
                1000,
                -8,
                -5,
                -3,
                -tiny,
                -tiny / 128,
                -0.0,
                0.0,
                tiny / 128,
                tiny,
            ],
            device=b.device,
            dtype=dtype,
        )
        x = values.repeat((shape[0] * shape[1] + values.numel() - 1) // values.numel())
        x = x[: shape[0] * shape[1]].reshape(shape)
    else:
        x = torch.ones(shape, device=b.device, dtype=dtype)
        b.fill_(torch.finfo(dtype).eps / 2)
    actual = _run(x, b)
    z = (x.cpu() + b.cpu()).to(dtype).double()
    argument = 0.79788456 * z * (1.0 + 0.044715 * z * z)
    expected = (0.5 * z * (1.0 + torch.tanh(argument))).to(dtype)
    tolerance = {torch.float16: 3e-3, torch.bfloat16: 2e-2, torch.float32: 2e-5}[dtype]
    assert (
        actual.shape == x.shape and actual.dtype == dtype and actual.device == x.device
    )
    torch.testing.assert_close(
        actual.cpu(), expected, rtol=tolerance, atol=tolerance, equal_nan=True
    )
    if case == "rounding":
        assert torch.equal(actual, _run(x, torch.zeros_like(b)))
