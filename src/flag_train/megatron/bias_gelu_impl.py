# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Bias + tanh GELU with first-order autograd, matching Megatron's interface.

Operands must share a supported dtype/device and obey PyTorch broadcasting.
Scalar, vector and arbitrary broadcast biases are supported; uncommon layouts
are materialized. Autograd reduces the elementwise gradients to operand shapes.
The add is rounded to the operand dtype before fp32 activation arithmetic.
Higher-order gradients are deliberately unsupported.
"""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry, tl_extra_shim
from flag_train.utils import triton_lang_extension as tle

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _load_z(X, B, offsets, mask, H: tl.constexpr, MODE: tl.constexpr):
    x = tl.load(X + offsets, mask, other=0).to(tl.float32)
    if MODE == 0:
        b = tl.load(B).to(tl.float32)
    elif MODE == 1:
        b = tl.load(B + offsets % H, mask, other=0).to(tl.float32)
    else:
        b = tl.load(B + offsets, mask, other=0).to(tl.float32)
    return (x + b).to(X.dtype.element_ty).to(tl.float32)


@libentry()
@triton.jit
def _forward(X, B, Y, N, H: tl.constexpr, MODE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tle.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    z = _load_z(X, B, offsets, offsets < N, H, MODE)
    t = tl_extra_shim.tanh(0.79788456 * z * (1.0 + 0.044715 * z * z))
    tl.store(Y + offsets, 0.5 * z * (1.0 + t), offsets < N)


@libentry()
@triton.jit
def _backward(X, B, G, DX, N, H: tl.constexpr, MODE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tle.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    z = _load_z(X, B, offsets, mask, H, MODE)
    # Outside this interval tanh is saturated in fp32. Bounding the polynomial
    # avoids 0 * inf for large finite bf16/fp32 operands in the derivative.
    bounded = tl.minimum(tl.maximum(z, -10.0), 10.0)
    t = tl_extra_shim.tanh(0.79788456 * bounded * (1.0 + 0.044715 * bounded * bounded))
    derivative = 0.5 * (1.0 + t) + 0.5 * bounded * (1.0 - t * t) * (
        0.79788456 + 0.1070322243 * bounded * bounded
    )
    # Preserve Megatron's NaN derivative for nonfinite pre-activations.
    derivative = tl.where(
        (tl.abs(z) == float("inf")) | (z != z), float("nan"), derivative
    )
    g = tl.load(G + offsets, mask, other=0).to(tl.float32)
    tl.store(DX + offsets, g * derivative, mask)


def _backward_block(n, dtype):
    # H20 V1 cutoffs confirmed against the original source by the lead engineer.
    if dtype == torch.float32 or n < (1 << 20):
        return 256
    return 1024 if n < (1 << 23) else 2048


# Fixed configurations need no LibEntry tuning on NVIDIA. Use Triton's own
# device/stream-aware JIT cache; retain framework dispatch for other backends.
_forward_launch = _forward.fn if runtime_device.vendor_name == "nvidia" else _forward
_backward_launch = _backward.fn if runtime_device.vendor_name == "nvidia" else _backward


class _BiasGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias):
        if input.ndim and bias.ndim == 1 and bias.numel() == input.shape[-1]:
            x, b = input.contiguous(), bias.contiguous()
            shape, h, mode = input.shape, bias.numel(), 1
        else:
            return _BiasGelu._broadcast_forward(ctx, input, bias)
        return _BiasGelu._launch_forward(ctx, input, x, b, shape, h, mode)

    @staticmethod
    def _broadcast_forward(ctx, input, bias):
        x, b = torch.broadcast_tensors(input, bias)
        shape = x.shape
        x = x.contiguous()
        h = shape[-1] if shape else 1
        if bias.numel() == 1:
            b, mode = bias.contiguous(), 0
        elif (
            x.numel()
            and h
            and b.stride(-1) == 1
            and all(s == 0 for s in b.stride()[:-1])
        ):
            b, mode = b.reshape(-1, h)[0].contiguous(), 1
        elif bias.ndim == 1 and bias.numel() == h:
            b, mode = bias.contiguous(), 1
        else:
            b, mode = b.contiguous(), 2
        return _BiasGelu._launch_forward(ctx, input, x, b, shape, h, mode)

    @staticmethod
    def _launch_forward(ctx, input, x, b, shape, h, mode):
        ctx.save_for_backward(x, b)
        ctx.h, ctx.mode = h, mode
        out = torch.empty(shape, dtype=input.dtype, device=input.device)
        n = out.numel()
        if n:
            with torch_device_fn.device(input.device):
                _forward_launch[(triton.cdiv(n, 256),)](
                    x, b, out, n, h, mode, BLOCK=256, num_warps=4
                )
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, b = ctx.saved_tensors
        grad = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        n = grad.numel()
        if n:
            block = _backward_block(n, x.dtype)
            with torch_device_fn.device(x.device):
                _backward_launch[(triton.cdiv(n, block),)](
                    x,
                    b,
                    grad_output.contiguous(),
                    grad,
                    n,
                    ctx.h,
                    ctx.mode,
                    BLOCK=block,
                    num_warps=4,
                )
        # Like Megatron's GeLUFunction: the engine sum_to's each gradient to
        # its original input shape, including scalar and empty broadcasts.
        return grad if ctx.needs_input_grad[0] else None, (
            grad if ctx.needs_input_grad[1] else None
        )


def bias_gelu_impl(input, bias):
    """Return GELU(input + bias), using the tanh approximation (first order)."""
    for name, tensor in (("input", input), ("bias", bias)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.dtype not in _DTYPES:
            raise ValueError(f"{name} must have FP16, BF16 or FP32 dtype")
        if tensor.device.type == "cpu" or tensor.device.type != runtime_device.name:
            raise ValueError(f"{name} must be on the active accelerator")
    if input.dtype != bias.dtype or input.device != bias.device:
        raise ValueError("input and bias must have the same dtype and device")
    if not (input.ndim and bias.ndim == 1 and bias.numel() == input.shape[-1]):
        try:
            torch.broadcast_shapes(input.shape, bias.shape)
        except RuntimeError:
            raise ValueError(
                "input and bias must have broadcast-compatible shapes"
            ) from None
    if (
        runtime_device.vendor_name == "nvidia"
        and input.ndim
        and bias.ndim == 1
        and bias.numel() == input.shape[-1]
        and input.is_contiguous()
        and bias.is_contiguous()
        and 0 < input.numel() < (1 << 31)
    ):
        from ._bias_gelu_cuda import implementation

        native = implementation()
        if native is not None:
            return native(input, bias)
    return _BiasGelu.apply(input, bias)
