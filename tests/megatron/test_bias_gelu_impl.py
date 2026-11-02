# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Forward and both gradients; formula oracle is explicitly not Megatron.

The reference rounds bias addition to the operand dtype, then evaluates the
activation/analytic derivative in higher precision. --ref cpu is supported.
An additional test compares the actual installed Megatron interface in all dtypes.
"""

import pytest
import torch

import flag_train
from flag_train.megatron.bias_gelu_impl import _backward_block

from .. import accuracy_utils as utils

pytestmark = pytest.mark.bias_gelu_impl
_SHAPES = [(), (0,), (0, 17), (3, 0), (1,), (17,), (2, 19, 7), (64, 128)]
if not utils.QUICK_MODE:
    _SHAPES += [(1024, 1024), (4096, 4096), (2, 1024, 4097)]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "numel,low_precision_block",
    [
        (0, 256),
        (1, 256),
        ((1 << 20) - 1, 256),
        (1 << 20, 1024),
        ((1 << 20) + 1, 1024),
        ((1 << 23) - 1, 1024),
        (1 << 23, 2048),
        ((1 << 23) + 1, 2048),
        (1 << 24, 2048),
    ],
)
def test_bias_gelu_backward_tiling(numel, low_precision_block, dtype):
    expected = 256 if dtype == torch.float32 else low_precision_block
    assert _backward_block(numel, dtype) == expected


def _check(x, b, needs=(True, True)):
    x = x.detach().requires_grad_(needs[0])
    b = b.detach().requires_grad_(needs[1])
    y = flag_train.megatron.bias_gelu_impl(x, b)
    assert y.shape == torch.broadcast_shapes(x.shape, b.shape)
    assert y.dtype == x.dtype and y.device == x.device
    # Slice a larger allocation to exercise a noncontiguous upstream gradient.
    g = torch.randn((*y.shape, 2), device=x.device, dtype=x.dtype)[..., 0]
    operands = [v for v in (x, b) if v.requires_grad]
    grads = torch.autograd.grad(y, operands, g)
    # Round the add before upcasting, even for the CPU reference.
    z = (utils.to_reference(x.detach()) + utils.to_reference(b.detach())).to(x.dtype)
    z = utils.to_reference(z, upcast=True)
    t = torch.tanh(0.79788456 * z * (1 + 0.044715 * z * z))
    ref = (0.5 * z * (1 + t)).to(x.dtype)
    derivative = 0.5 * (1 + t) + 0.5 * z * (1 - t * t) * (
        0.79788456 + 0.1070322243 * z * z
    )
    dz = (utils.to_reference(g) * derivative).to(x.dtype)
    tolerances = {
        torch.float16: (3e-3, 3e-3),
        torch.bfloat16: (2e-2, 2e-2),
        torch.float32: (2e-5, 2e-5),
    }
    rtol, atol = tolerances[x.dtype]
    torch.testing.assert_close(utils.to_reference(y), ref, rtol=rtol, atol=atol)
    for actual, operand in zip(grads, operands):
        expected = dz.sum_to_size(operand.shape)
        assert actual.shape == operand.shape and actual.dtype == operand.dtype
        torch.testing.assert_close(
            utils.to_reference(actual), expected, rtol=rtol, atol=atol
        )


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_gelu_impl(shape, dtype):
    utils.init_seed(0)
    x = torch.randn(shape, device=flag_train.device, dtype=dtype)
    b = torch.randn(shape[-1:] if shape else (), device=x.device, dtype=dtype)
    _check(x, b)


@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("bias_shape", [(), (7,), (1, 7), (2, 1, 7), (2, 3, 7)])
def test_bias_gelu_broadcast(dtype, bias_shape):
    utils.init_seed(1)
    x = torch.randn((2, 3, 7), device=flag_train.device, dtype=dtype)
    b = torch.randn(bias_shape, device=x.device, dtype=dtype)
    _check(x, b)
    # Broadcasting may expand the input as well as the bias.
    _check(b, x)


@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_gelu_noncontiguous(dtype):
    utils.init_seed(2)
    x = torch.randn((17, 9), device=flag_train.device, dtype=dtype).t()
    b = torch.randn((34,), device=x.device, dtype=dtype)[::2]
    assert not x.is_contiguous() and not b.is_contiguous()
    _check(x, b)
    _check(x[:, ::2], b[::2])


@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("needs", [(True, False), (False, True)])
def test_bias_gelu_one_gradient(dtype, needs):
    x = torch.randn((4, 17), device=flag_train.device, dtype=dtype)
    _check(x, x[0].clone(), needs)


@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_gelu_values_and_empty(dtype):
    values = [-1000, -20, -8, -3, -1, -1e-4, 0, 1e-4, 1, 3, 8, 20, 1000]
    x = torch.tensor(values, device=flag_train.device, dtype=dtype)
    _check(x, torch.zeros_like(x))
    _check(
        torch.empty((0, 17), device=x.device, dtype=dtype),
        torch.ones((1, 17), device=x.device, dtype=dtype),
    )
    _check(
        torch.empty((2, 0, 7), device=x.device, dtype=dtype),
        torch.ones((), device=x.device, dtype=dtype),
    )


def test_bias_gelu_bad_arguments():
    x = torch.randn((4, 8), device=flag_train.device)
    with pytest.raises(TypeError, match="tensor"):
        flag_train.megatron.bias_gelu_impl(x, None)
    with pytest.raises(ValueError, match="dtype"):
        flag_train.megatron.bias_gelu_impl(x, x.to(torch.int32))
    with pytest.raises(ValueError, match="same dtype"):
        flag_train.megatron.bias_gelu_impl(x, x.to(torch.float16))
    with pytest.raises(ValueError, match="accelerator"):
        flag_train.megatron.bias_gelu_impl(x, torch.randn(8))
    with pytest.raises(ValueError, match="broadcast"):
        flag_train.megatron.bias_gelu_impl(x, x.new_empty(3))
    with pytest.raises(ValueError, match="broadcast"):
        flag_train.megatron.bias_gelu_impl(x.new_empty(0, 8), x.new_empty(3))


@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_gelu_nonfinite(dtype):
    x = torch.tensor(
        [float("inf"), -float("inf"), float("nan")],
        device=flag_train.device,
        dtype=dtype,
        requires_grad=True,
    )
    b = torch.zeros_like(x, requires_grad=True)
    y = flag_train.megatron.bias_gelu_impl(x, b)
    assert torch.isposinf(y[0])
    assert torch.isnan(y[1:]).all()
    dx, db = torch.autograd.grad(y, (x, b), torch.ones_like(y))
    assert torch.isnan(dx).all() and torch.isnan(db).all()


@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_gelu_installed_megatron(dtype):
    module = pytest.importorskip("megatron.core.fusions.fused_bias_gelu")
    utils.init_seed(3)
    x = torch.randn(
        (16, 127), device=flag_train.device, dtype=dtype, requires_grad=True
    )
    b = torch.randn((127,), device=x.device, dtype=dtype, requires_grad=True)
    y = flag_train.megatron.bias_gelu_impl(x, b)
    ref = module.bias_gelu_impl(x, b)
    g = torch.randn_like(y)
    # Eager Megatron rounds intermediate low-precision arithmetic; the kernel
    # computes activation/derivative in fp32, so equality is tolerance based.
    rtol, atol = {
        torch.float16: (5e-3, 1e-2),
        torch.bfloat16: (4e-2, 8e-2),
        torch.float32: (2e-5, 2e-5),
    }[dtype]
    torch.testing.assert_close(y, ref, rtol=rtol, atol=atol)
    actual = torch.autograd.grad(y, (x, b), g)
    expected = torch.autograd.grad(ref, (x, b), g)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=rtol, atol=atol)
