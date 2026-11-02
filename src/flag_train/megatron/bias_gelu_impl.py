# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Bias + tanh GELU with first-order autograd, matching Megatron's interface.

Operands must share a supported dtype/device and obey PyTorch broadcasting.
Scalar, vector and arbitrary broadcast biases are supported; uncommon layouts
are materialized. Triton or Autograd reduces the elementwise gradients to operand shapes.
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
from flag_train.utils.libentry import major_version as _entry_major
from flag_train.utils.libentry import minor_version as _entry_minor

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
            value = v - correction
            updated = acc + value
            correction = tl.where(
                tl.abs(updated) < float("inf"), (updated - acc) - value, 0.0
            )
            acc = updated
        else:
            acc += v
    tl.store(DB + cols, tl.sum(acc, 0), cols < H)


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
        _launch_fixed_entry(_backward_launch, grid, args)
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


_generic_arg_indices = {}


def _launch_fixed_entry(entry, grid, args):
    # Use LibEntry's own backend-aware specialization key and cache. Never
    # bypass tuners, heuristics, hooks, or older compiled-argument protocols.
    if not isinstance(entry.fn, triton.runtime.JITFunction) or not (
        _entry_major == 3 and 3 <= _entry_minor <= 6
    ):
        entry[grid](*args, num_warps=4)
        return
    indices = _generic_arg_indices.get(entry)
    if indices is None:
        indices = (
            tuple(entry.specialize_indices),
            tuple(entry.do_not_specialize_indices),
            tuple(p.num for p in entry.jit_function.params if p.is_constexpr),
        )
        _generic_arg_indices[entry] = indices
    spec, dns, const = indices
    key = entry.key(
        tuple(args[i] for i in spec),
        tuple(args[i] for i in dns),
        tuple(args[i] for i in const),
    )
    device = torch_device_fn.current_device()
    cache = entry._cpu_cache if device == "cpu" else entry.kernel_cache[device]
    cached = cache.get(key)
    if cached is None or cached[4]:
        entry[grid](*args, num_warps=4)
    else:
        # Rebind every launch so the installed backend resolves its stream in
        # the same way as LibEntry.run; cache invalidation remains authoritative.
        cached[0][grid](*args)


def _launch_kernel(jit, grid, tensors, constants):
    if runtime_device.vendor_name != "nvidia" or not isinstance(
        jit.fn, triton.runtime.JITFunction
    ):
        _launch_fixed_entry(jit, grid, (*tensors, *constants))
        return
    jit = jit.fn
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
def _saved_backward(
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
    rows = row_tile * BR + tl.arange(0, BR)
    cols = column_tile * BC + tl.arange(0, BC)
    offsets = rows[:, None] * H + cols[None, :]
    mask = (rows[:, None] < R) & (cols[None, :] < H)
    d = tl.load(D + offsets, mask, 0)
    g = tl.load(G + offsets, mask, 0).to(tl.float32)
    dx = (d * g).to(DX.dtype.element_ty)
    tl.store(DX + offsets, dx, mask)
    if BIAS:
        partial = tl.sum(tl.where(mask, dx.to(tl.float32), 0.0), 0)
        tl.store(P + row_tile * H + cols, partial, cols < H)


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
                value -= correction
                updated = acc + value
                correction = tl.where(
                    tl.abs(updated) < float("inf"), (updated - acc) - value, 0.0
                )
                acc = updated
            else:
                acc += value
    if BIAS:
        tl.store(P + row_tile * H + cols, acc, cols < H)


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
    BIAS: tl.constexpr,
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


class _BiasGeluSaved(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias):
        x, b = input.contiguous(), bias.contiguous()
        n, h = x.numel(), b.numel()
        out = torch.empty_like(x)
        derivative = torch.empty(x.shape, dtype=torch.float32, device=x.device)
        block = 512 if n >= (1 << 20) else 256
        with torch_device_fn.device(x.device):
            _launch_kernel(
                _forward_saved,
                (triton.cdiv(n, block), 1, 1),
                (x, b, out, derivative),
                (n, h, block),
            )
        ctx.save_for_backward(x, b, derivative)
        ctx.n, ctx.h = n, h
        return out

    @staticmethod
    def backward(ctx, grad_output):
        if torch.is_grad_enabled():
            raise RuntimeError("bias_gelu_impl supports first-order gradients only")
        x, b, derivative = ctx.saved_tensors
        grad = torch.empty_like(x)
        h, rows = ctx.h, ctx.n // ctx.h
        upstream = grad_output.contiguous()
        need_bias = ctx.needs_input_grad[1]
        if rows == 1:
            with _backward_device_guard(x):
                _launch_kernel(
                    _saved_backward_loop,
                    (triton.cdiv(h, 1024), 1, 1),
                    (derivative, upstream, grad, grad),
                    (1, h, 1, 1024, False),
                )
            return grad if ctx.needs_input_grad[0] else None, (
                grad.view(b.shape) if need_bias else None
            )
        if need_bias and ctx.n <= (1 << 20) and rows > 128:
            db = torch.empty_like(b)
            with _backward_device_guard(x):
                _launch_kernel(
                    _saved_backward_columns,
                    (triton.cdiv(h, 8), 1, 1),
                    (derivative, upstream, grad, db),
                    (rows, h, 256, 8, True),
                )
            return grad if ctx.needs_input_grad[0] else None, db
        small = ctx.n <= (1 << 16) and rows <= 128
        br = triton.next_power_of_2(rows) if small else (32 if rows >= 1024 else 8)
        bc = 64 if small else (256 if x.dtype == torch.float32 else 512)
        parts = triton.cdiv(rows, br)
        if need_bias:
            db = torch.empty_like(b)
            partial = (
                db
                if small
                else torch.empty((parts, h), dtype=torch.float32, device=x.device)
            )
        else:
            db, partial = None, grad
            br, bc, parts = 1, 1024, rows
        with _backward_device_guard(x):
            _launch_kernel(
                _saved_backward if small else _saved_backward_loop,
                (parts * triton.cdiv(h, bc), 1, 1),
                (derivative, upstream, grad, partial),
                (rows, h, br, bc, need_bias),
            )
            if need_bias and not small:
                _launch_kernel(
                    _reduce_bias,
                    (triton.cdiv(h, 32), 1, 1),
                    (partial, db),
                    (h, parts, 32, 32),
                )
        return grad if ctx.needs_input_grad[0] else None, db


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
    def backward(ctx, grad_output):
        # The engine disables grad mode for ordinary backward. Reject higher
        # orders explicitly without wrapping each first-order call in no_grad.
        if torch.is_grad_enabled():
            raise RuntimeError("bias_gelu_impl supports first-order gradients only")
        x, b = ctx.saved_tensors
        grad = torch.empty_like(x)
        n = grad.numel()
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
    if (
        torch.is_grad_enabled()
        and (input.requires_grad or bias.requires_grad)
        and input.ndim
        and bias.ndim == 1
        and bias.numel() == input.shape[-1]
        and input.numel()
    ):
        return _BiasGeluSaved.apply(input, bias)
    return _BiasGelu.apply(input, bias)
