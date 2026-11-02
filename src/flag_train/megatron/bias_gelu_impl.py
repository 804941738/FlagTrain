# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Bias + tanh GELU with first-order autograd, matching Megatron's interface.

Operands must share a supported dtype/device and obey PyTorch broadcasting.
Scalar, vector and arbitrary broadcast biases are supported; uncommon layouts
are materialized. Autograd reduces the elementwise gradients to operand shapes.
The add is rounded to the operand dtype before fp32 activation arithmetic.
Higher-order gradients are deliberately unsupported.
"""

from contextlib import nullcontext

import torch
import triton
import triton.language as tl

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


_FAST_LOW_PRECISION_TANH = tl.constexpr(runtime_device.vendor_name == "nvidia")


@triton.jit
def _backward_tanh(u, DTYPE: tl.constexpr):
    if _FAST_LOW_PRECISION_TANH and DTYPE != tl.float32:
        e = tl.exp(-2.0 * tl.abs(u))
        return tl.where(u >= 0.0, (1.0 - e) / (1.0 + e), (e - 1.0) / (1.0 + e))
    else:
        return tl_extra_shim.tanh(u)


@libentry()
@triton.jit
def _backward(X, B, G, DX, N, H: tl.constexpr, MODE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tle.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    z = _load_z(X, B, offsets, mask, H, MODE)
    # Outside this interval tanh is saturated in fp32. Bounding the polynomial
    # avoids 0 * inf for large finite bf16/fp32 operands in the derivative.
    bounded = tl.minimum(tl.maximum(z, -10.0), 10.0)
    t = _backward_tanh(
        0.79788456 * bounded * (1.0 + 0.044715 * bounded * bounded), X.dtype.element_ty
    )
    derivative = 0.5 * (1.0 + t) + 0.5 * bounded * (1.0 - t * t) * (
        0.79788456 + 0.1070322243 * bounded * bounded
    )
    # Preserve Megatron's NaN derivative for nonfinite pre-activations.
    derivative = tl.where(
        (tl.abs(z) == float("inf")) | (z != z), float("nan"), derivative
    )
    g = tl.load(G + offsets, mask, other=0).to(tl.float32)
    tl.store(DX + offsets, g * derivative, mask)


@triton.jit
def _vector_backward(
    X, B, G, DX, P, R: tl.constexpr, H: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr
):
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    bias = tl.load(B + cols, cols < H, 0).to(tl.float32)
    acc = tl.full((BC,), 0, tl.float32)
    for r in range(BR):
        row = tl.program_id(0) * BR + r
        offsets = row * H + cols
        mask = (row < R) & (cols < H)
        z = (
            (tl.load(X + offsets, mask, 0).to(tl.float32) + bias)
            .to(X.dtype.element_ty)
            .to(tl.float32)
        )
        v = tl.minimum(tl.maximum(z, -10.0), 10.0)
        t = _backward_tanh(
            0.79788456 * v * (1.0 + 0.044715 * v * v), X.dtype.element_ty
        )
        d = 0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * (
            0.79788456 + 0.1070322243 * v * v
        )
        d = tl.where((tl.abs(z) == float("inf")) | (z != z), float("nan"), d)
        dx = (tl.load(G + offsets, mask, 0).to(tl.float32) * d).to(DX.dtype.element_ty)
        tl.store(DX + offsets, dx, mask)
        acc += tl.where(mask, dx.to(tl.float32), 0.0)
    tl.store(P + tl.program_id(0) * H + cols, acc, cols < H)


@triton.jit
def _reduce_bias(
    P, DB, H: tl.constexpr, PARTS: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr
):
    rows = tl.arange(0, BR)
    cols = tl.program_id(0) * BC + tl.arange(0, BC)
    acc = tl.full((BR, BC), 0, tl.float32)
    for tile in range(triton.cdiv(PARTS, BR)):
        rr = tile * BR + rows
        v = tl.load(
            P + rr[:, None] * H + cols[None, :],
            (rr[:, None] < PARTS) & (cols[None, :] < H),
            0,
        )
        acc += v
    tl.store(DB + cols, tl.sum(acc, 0), cols < H)


@triton.jit
def _small_backward(
    X, B, G, DX, P, R: tl.constexpr, H: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr
):
    cols = tl.program_id(1) * BC + tl.arange(0, BC)[None, :]
    rows = tl.program_id(0) * BR + tl.arange(0, BR)[:, None]
    bias = tl.load(B + cols, cols < H, 0).to(tl.float32)
    offsets = rows * H + cols
    mask = (rows < R) & (cols < H)
    z = (
        (tl.load(X + offsets, mask, 0).to(tl.float32) + bias)
        .to(X.dtype.element_ty)
        .to(tl.float32)
    )
    v = tl.minimum(tl.maximum(z, -10.0), 10.0)
    t = _backward_tanh(0.79788456 * v * (1.0 + 0.044715 * v * v), X.dtype.element_ty)
    d = 0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * (0.79788456 + 0.1070322243 * v * v)
    d = tl.where((tl.abs(z) == float("inf")) | (z != z), float("nan"), d)
    dx = (tl.load(G + offsets, mask, 0).to(tl.float32) * d).to(DX.dtype.element_ty)
    tl.store(DX + offsets, dx, mask)
    acc = tl.sum(tl.where(mask, dx.to(tl.float32), 0.0), 0)
    tl.store(
        P + tl.program_id(0) * H + tl.reshape(cols, (BC,)),
        acc,
        tl.reshape(cols, (BC,)) < H,
    )


def _backward_block(n, dtype):
    # H20 V1 cutoffs confirmed against the original source by the lead engineer.
    if dtype == torch.float32 or n < (1 << 20):
        return 256
    return 1024 if n < (1 << 23) else 2048


# Fixed configurations need no LibEntry tuning on NVIDIA. Use Triton's own
# device/stream-aware JIT cache; retain framework dispatch for other backends.
_forward_launch = _forward.fn if runtime_device.vendor_name == "nvidia" else _forward
_backward_launch = _backward.fn if runtime_device.vendor_name == "nvidia" else _backward


_backward_cache = {}


def _launch_backward(x, b, upstream, grad, n, h, mode, block):
    args = (x, b, upstream, grad, n, h, mode, block)
    grid = (triton.cdiv(n, block), 1, 1)
    if runtime_device.vendor_name != "nvidia":
        _backward_launch[grid](*args, num_warps=4)
        return
    # Cache compiled Triton code, never tensors or streams. CompiledKernel uses
    # the caller's current stream. Preserve pointer-alignment specialization.
    key = (
        x.device.index,
        x.dtype,
        n,
        h,
        mode,
        block,
        tuple(t.data_ptr() % 16 == 0 for t in args[:4]),
    )
    kernel = _backward_cache.get(key)
    if kernel is None:
        kernel = _backward_launch[grid](*args, num_warps=4)
        if len(_backward_cache) >= 256:
            _backward_cache.clear()
        _backward_cache[key] = kernel
    else:
        kernel[grid](*args)


_vector_cache = {}


def _cached_vector_launch(jit, grid, tensors, constants):
    key = (
        jit,
        tensors[0].device.index,
        tuple((t.dtype, t.data_ptr() % 16 == 0) for t in tensors),
        constants,
    )
    kernel = _vector_cache.get(key)
    args = (*tensors, *constants)
    if kernel is None:
        kernel = jit[grid](*args, num_warps=4)
        if len(_vector_cache) >= 256:
            _vector_cache.clear()
        _vector_cache[key] = kernel
    else:
        kernel[grid](*args)


def _backward_device_guard(x):
    # Avoid entering an already-active CUDA device context. A different device
    # and every non-NVIDIA backend still use the framework's device guard.
    if (
        runtime_device.vendor_name == "nvidia"
        and torch_device_fn.current_device() == x.device.index
    ):
        return nullcontext()
    return torch_device_fn.device(x.device)


class _BiasGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias):
        if input.ndim and bias.ndim == 1 and bias.numel() == input.shape[-1]:
            x, b = input.contiguous(), bias.contiguous()
            shape, h, mode = input.shape, bias.numel(), 1
        else:
            return _BiasGelu._broadcast_forward(ctx, input, bias)
        out = _BiasGelu._launch_forward(ctx, input, x, b, shape, h, mode)
        ctx.vector_bias = True
        return out

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
        ctx.vector_bias = False
        out = torch.empty(shape, dtype=input.dtype, device=input.device)
        n = out.numel()
        if n:
            with torch_device_fn.device(input.device):
                _forward_launch[(triton.cdiv(n, 256),)](
                    x, b, out, n, h, mode, BLOCK=256, num_warps=4
                )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # The engine disables grad mode for ordinary backward. Reject higher
        # orders explicitly without wrapping each first-order call in no_grad.
        if torch.is_grad_enabled():
            raise RuntimeError("bias_gelu_impl supports first-order gradients only")
        x, b = ctx.saved_tensors
        grad = torch.empty_like(x)
        n = grad.numel()
        if (
            runtime_device.vendor_name == "nvidia"
            and ctx.vector_bias
            and ctx.needs_input_grad[1]
            and n
            and n <= (1 << 16)
            and n // ctx.h <= 128
        ):
            h = ctx.h
            rows = n // h
            db = torch.empty_like(b)
            with _backward_device_guard(x):
                _cached_vector_launch(
                    _small_backward,
                    (1, triton.cdiv(h, 64), 1),
                    (x, b, grad_output.contiguous(), grad, db),
                    (rows, h, triton.next_power_of_2(rows), 64),
                )
            return grad if ctx.needs_input_grad[0] else None, db
        if (
            runtime_device.vendor_name == "nvidia"
            and ctx.vector_bias
            and ctx.needs_input_grad[1]
            and n > (1 << 20)
        ):
            h = ctx.h
            rows = n // h
            br, bc = 8, 512
            parts = triton.cdiv(rows, br)
            partial = torch.empty((parts, h), device=x.device, dtype=torch.float32)
            db = torch.empty_like(b)
            with _backward_device_guard(x):
                _cached_vector_launch(
                    _vector_backward,
                    (parts, triton.cdiv(h, bc), 1),
                    (x, b, grad_output.contiguous(), grad, partial),
                    (rows, h, br, bc),
                )
                _cached_vector_launch(
                    _reduce_bias,
                    (triton.cdiv(h, 32), 1, 1),
                    (partial, db),
                    (h, parts, 32, 32),
                )
            return grad if ctx.needs_input_grad[0] else None, db
        if n:
            block = _backward_block(n, x.dtype)
            with _backward_device_guard(x):
                _launch_backward(
                    x, b, grad_output.contiguous(), grad, n, ctx.h, ctx.mode, block
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
    return _BiasGelu.apply(input, bias)
