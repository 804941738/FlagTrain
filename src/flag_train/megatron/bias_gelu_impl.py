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
"""Bias + tanh GELU with first-order Autograd and standard LibEntry launches.

The addition and elementwise gradients round to the operand dtype; activation
arithmetic and bias accumulation use FP32. Vector-bias training saves the
activation derivative. General broadcasts retain an elementwise fallback.
"""

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry, tl_extra_shim
from flag_train.utils import triton_lang_extension as tle

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_NUM_WARPS = 4


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


@triton.jit
def _gelu_derivative(z, t, u, DTYPE: tl.constexpr):
    v = tl.minimum(tl.maximum(z, -10.0), 10.0)
    if DTYPE == tl.float32:
        # Logistic/sech form avoids cancellation in 1+t and 1-t*t.
        e = tl.exp(-2.0 * tl.abs(u))
        inverse = 1.0 / (1.0 + e)
        base = tl.where(v >= 0.0, inverse, e * inverse)
        d = base + 2.0 * v * e * inverse * inverse * (0.79788456 + 0.1070322243 * v * v)
        # Retain the FP32 tanh saturation contract, including large upstream g.
        d = tl.where(t == 1.0, 1.0, tl.where(t == -1.0, 0.0, d))
    else:
        d = 0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * (
            0.79788456 + 0.1070322243 * v * v
        )
    return tl.where((tl.abs(z) == float("inf")) | (z != z), float("nan"), d)


@triton.jit
def _compensated_add(acc, correction, value):
    # Keep small FP32 bias sums accurate without changing Inf/NaN propagation.
    value = value - correction
    updated = acc + value
    correction = tl.where(tl.abs(updated) < float("inf"), (updated - acc) - value, 0.0)
    return updated, correction


@libentry()
@triton.jit
def _forward(X, B, Y, N, H: tl.constexpr, MODE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tle.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    z = _load_z(X, B, offsets, offsets < N, H, MODE)
    t = tl_extra_shim.tanh(0.79788456 * z * (1.0 + 0.044715 * z * z))
    tl.store(Y + offsets, 0.5 * z * (1.0 + t), offsets < N)


@libentry()
@triton.jit
def _forward_saved(X, B, Y, D, N, H: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tle.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    z = _load_z(X, B, offsets, offsets < N, H, 1)
    u = 0.79788456 * z * (1.0 + 0.044715 * z * z)
    t = tl_extra_shim.tanh(u)
    tl.store(Y + offsets, 0.5 * z * (1.0 + t), offsets < N)
    d = _gelu_derivative(z, t, u, X.dtype.element_ty)
    tl.store(D + offsets, d, offsets < N)


@libentry()
@triton.jit
def _backward(X, B, G, DX, N, H: tl.constexpr, MODE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tle.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    z = _load_z(X, B, offsets, mask, H, MODE)
    # Outside this interval tanh is saturated in fp32. Bounding the polynomial
    # avoids 0 * inf for large finite bf16/fp32 operands in the derivative.
    bounded = tl.minimum(tl.maximum(z, -10.0), 10.0)
    u = 0.79788456 * bounded * (1.0 + 0.044715 * bounded * bounded)
    t = tl_extra_shim.tanh(u)
    derivative = _gelu_derivative(z, t, u, X.dtype.element_ty)
    g = tl.load(G + offsets, mask, other=0).to(tl.float32)
    tl.store(DX + offsets, g * derivative, mask)


@libentry()
@triton.jit
def _saved_backward(
    D,
    G,
    DX,
    P,
    R: tl.constexpr,
    H: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    pid = tle.program_id(0)
    column_tiles = triton.cdiv(H, BC)
    row_tile, column_tile = pid // column_tiles, pid % column_tiles
    rows = row_tile * BR + tl.arange(0, BR)
    cols = column_tile * BC + tl.arange(0, BC)
    offsets = rows[:, None] * H + cols[None, :]
    mask = (rows[:, None] < R) & (cols[None, :] < H)
    d = tl.load(D + offsets, mask, 0)
    g = tl.load(G + offsets, mask, 0).to(tl.float32)
    dx = (d * g).to(DX.dtype.element_ty)
    tl.store(DX + offsets, dx, mask)
    partial = tl.sum(tl.where(mask, dx.to(tl.float32), 0.0), 0)
    tl.store(P + row_tile * H + cols, partial, cols < H)


@libentry()
@triton.jit
def _saved_backward_columns(
    D,
    G,
    DX,
    DB,
    R: tl.constexpr,
    H: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    rows = tl.arange(0, BR)
    cols = tle.program_id(0) * BC + tl.arange(0, BC)
    acc = tl.full((BR, BC), 0, tl.float32)
    for tile in range(triton.cdiv(R, BR)):
        rr = tile * BR + rows
        offsets = rr[:, None] * H + cols[None, :]
        mask = (rr[:, None] < R) & (cols[None, :] < H)
        d = tl.load(D + offsets, mask, 0)
        g = tl.load(G + offsets, mask, 0).to(tl.float32)
        dx = (d * g).to(DX.dtype.element_ty)
        tl.store(DX + offsets, dx, mask)
        acc += tl.where(mask, dx.to(tl.float32), 0.0)
    tl.store(DB + cols, tl.sum(acc, 0), cols < H)


@libentry()
@triton.jit
def _saved_backward_loop(
    D,
    G,
    DX,
    P,
    R: tl.constexpr,
    H: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    BIAS: tl.constexpr,
):
    pid = tle.program_id(0)
    column_tiles = triton.cdiv(H, BC)
    row_tile, column_tile = pid // column_tiles, pid % column_tiles
    cols = column_tile * BC + tl.arange(0, BC)
    acc = tl.full((BC,), 0, tl.float32)
    correction = tl.full((BC,), 0, tl.float32)
    for r in range(BR):
        row = row_tile * BR + r
        offsets = row * H + cols
        mask = (row < R) & (cols < H)
        d = tl.load(D + offsets, mask, 0)
        g = tl.load(G + offsets, mask, 0).to(tl.float32)
        dx = (d * g).to(DX.dtype.element_ty)
        tl.store(DX + offsets, dx, mask)
        if BIAS:
            value = tl.where(mask, dx.to(tl.float32), 0.0)
            if DX.dtype.element_ty == tl.float32:
                acc, correction = _compensated_add(acc, correction, value)
            else:
                acc += value
    if BIAS:
        tl.store(P + row_tile * H + cols, acc, cols < H)


@libentry()
@triton.jit
def _reduce_bias(
    P, DB, H: tl.constexpr, PARTS: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr
):
    rows = tl.arange(0, BR)
    cols = tle.program_id(0) * BC + tl.arange(0, BC)
    acc = tl.full((BR, BC), 0, tl.float32)
    correction = tl.full((BR, BC), 0, tl.float32)
    for tile in range(triton.cdiv(PARTS, BR)):
        rr = (tile * BR + rows).to(tl.int64)
        v = tl.load(
            P + rr[:, None] * H + cols[None, :],
            (rr[:, None] < PARTS) & (cols[None, :] < H),
            0,
        )
        if DB.dtype.element_ty == tl.float32:
            acc, correction = _compensated_add(acc, correction, v)
        else:
            acc += v
    tl.store(DB + cols, tl.sum(acc, 0), cols < H)


def _backward_block(n, dtype):
    # Larger low-precision tiles amortize elementwise memory traffic.
    if dtype == torch.float32 or n < (1 << 20):
        return 256
    return 1024 if n < (1 << 23) else 2048


class _BiasGeluSaved(torch.autograd.Function):
    """Vector-bias training: cache FP32 derivatives, fuse gradient production."""

    @staticmethod
    def forward(ctx, input, bias):
        x, b = input.contiguous(), bias.contiguous()
        n, h = x.numel(), b.numel()
        out = torch.empty_like(x)
        derivative = torch.empty(x.shape, dtype=torch.float32, device=x.device)
        block = 512 if n >= (1 << 20) else 256
        with torch_device_fn.device(x.device):
            _forward_saved[(triton.cdiv(n, block),)](
                x, b, out, derivative, n, h, BLOCK=block, num_warps=_NUM_WARPS
            )
        # Retain operands as well as the derivative for Autograd version checks.
        ctx.save_for_backward(x, b, derivative)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        if torch.is_grad_enabled():
            raise RuntimeError("bias_gelu_impl supports first-order gradients only")
        x, b, derivative = ctx.saved_tensors
        n, h = x.numel(), b.numel()
        rows = n // h
        grad = torch.empty_like(x)
        upstream = grad_output.contiguous()
        need_bias = ctx.needs_input_grad[1]
        db = None
        with torch_device_fn.device(x.device):
            if rows == 1 or not need_bias:
                _saved_backward_loop[(rows * triton.cdiv(h, 1024),)](
                    derivative,
                    upstream,
                    grad,
                    grad,
                    rows,
                    h,
                    BR=1,
                    BC=1024,
                    BIAS=False,
                    num_warps=_NUM_WARPS,
                )
                if need_bias:
                    db = grad.view(b.shape)
            elif n <= (1 << 16) and rows <= 128:
                db = torch.empty_like(b)
                _saved_backward[(triton.cdiv(h, 64),)](
                    derivative,
                    upstream,
                    grad,
                    db,
                    rows,
                    h,
                    BR=triton.next_power_of_2(rows),
                    BC=64,
                    num_warps=_NUM_WARPS,
                )
            elif n <= (1 << 20) or rows <= 256:
                # Few rows fit one column-owned reduction, avoiding partials.
                bc = 8 if n <= (1 << 20) else 32
                db = torch.empty_like(b)
                _saved_backward_columns[(triton.cdiv(h, bc),)](
                    derivative,
                    upstream,
                    grad,
                    db,
                    rows,
                    h,
                    BR=256,
                    BC=bc,
                    num_warps=_NUM_WARPS,
                )
            else:
                br = 32 if rows >= 1024 else 8
                bc = 256 if x.dtype == torch.float32 else 512
                parts = triton.cdiv(rows, br)
                db = torch.empty_like(b)
                partial = torch.empty((parts, h), dtype=torch.float32, device=x.device)
                _saved_backward_loop[(parts * triton.cdiv(h, bc),)](
                    derivative,
                    upstream,
                    grad,
                    partial,
                    rows,
                    h,
                    BR=br,
                    BC=bc,
                    BIAS=True,
                    num_warps=_NUM_WARPS,
                )
                _reduce_bias[(triton.cdiv(h, 32),)](
                    partial,
                    db,
                    h,
                    parts,
                    BR=32,
                    BC=32,
                    num_warps=_NUM_WARPS,
                )
        return grad if ctx.needs_input_grad[0] else None, db


class _BiasGelu(torch.autograd.Function):
    """General broadcasting: save operands, let Autograd reduce gradient shapes."""

    @staticmethod
    def forward(ctx, input, bias):
        if input.ndim and bias.ndim == 1 and bias.numel() == input.shape[-1]:
            x, b = input.contiguous(), bias.contiguous()
            h, mode = bias.numel(), 1
        else:
            x, expanded = torch.broadcast_tensors(input, bias)
            x = x.contiguous()
            h = x.shape[-1] if x.ndim else 1
            if bias.numel() == 1:
                b, mode = bias.contiguous(), 0
            elif (
                x.numel()
                and h
                and expanded.stride(-1) == 1
                and all(s == 0 for s in expanded.stride()[:-1])
            ):
                b, mode = expanded.reshape(-1, h)[0].contiguous(), 1
            elif bias.ndim == 1 and bias.numel() == h:
                b, mode = bias.contiguous(), 1
            else:
                b, mode = expanded.contiguous(), 2
        ctx.save_for_backward(x, b)
        ctx.h, ctx.mode = h, mode
        out = torch.empty_like(x)
        n = x.numel()
        if n:
            with torch_device_fn.device(x.device):
                _forward[(triton.cdiv(n, 256),)](
                    x, b, out, n, h, mode, BLOCK=256, num_warps=_NUM_WARPS
                )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        if torch.is_grad_enabled():
            raise RuntimeError("bias_gelu_impl supports first-order gradients only")
        x, b = ctx.saved_tensors
        grad = torch.empty_like(x)
        n = x.numel()
        if n:
            block = _backward_block(n, x.dtype)
            with torch_device_fn.device(x.device):
                _backward[(triton.cdiv(n, block),)](
                    x,
                    b,
                    grad_output.contiguous(),
                    grad,
                    n,
                    ctx.h,
                    ctx.mode,
                    BLOCK=block,
                    num_warps=_NUM_WARPS,
                )
        # The engine sums both returned gradients to the original operand shapes.
        return (
            grad if ctx.needs_input_grad[0] else None,
            grad if ctx.needs_input_grad[1] else None,
        )


def bias_gelu_impl(input, bias):
    """Return tanh GELU(input + bias), with first-order gradients for both operands.

    FP16/BF16/FP32 tensors must share the active accelerator and obey PyTorch
    broadcasting. Noncontiguous inputs are materialized; empty tensors are valid.
    Invalid types raise TypeError; invalid dtype, device or shapes raise ValueError.
    """
    for name, tensor in (("input", input), ("bias", bias)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.dtype not in _DTYPES:
            raise ValueError(f"{name} must have FP16, BF16 or FP32 dtype")
        if tensor.device.type == "cpu" or tensor.device.type != runtime_device.name:
            raise ValueError(f"{name} must be on the active accelerator")
    if input.dtype != bias.dtype or input.device != bias.device:
        raise ValueError("input and bias must have the same dtype and device")
    vector_bias = input.ndim and bias.ndim == 1 and bias.numel() == input.shape[-1]
    if not vector_bias:
        try:
            torch.broadcast_shapes(input.shape, bias.shape)
        except RuntimeError:
            raise ValueError(
                "input and bias must have broadcast-compatible shapes"
            ) from None
    if (
        torch.is_grad_enabled()
        and (input.requires_grad or bias.requires_grad)
        and vector_bias
        and input.numel()
    ):
        return _BiasGeluSaved.apply(input, bias)
    return _BiasGelu.apply(input, bias)
