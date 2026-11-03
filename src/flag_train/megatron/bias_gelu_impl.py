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
"""Forward-only bias + tanh GELU using the common Triton/FlagTree backend.

Bias addition rounds to the operand dtype before FP32 activation arithmetic.
The output is a plain tensor: this operator does not propagate gradients.
"""

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry, tl_extra_shim
from flag_train.utils import triton_lang_extension as tle

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_BLOCK_SIZE = 256
_NUM_WARPS = 4
_IS_ASCEND = runtime_device.vendor_name == "ascend"
if _IS_ASCEND:
    from flag_train.runtime.backend._ascend.utils import CORE_NUM


@triton.jit
def _bias_gelu_tile(
    X, B, Y, N, H: tl.constexpr, BIAS_MODE: tl.constexpr, BLOCK: tl.constexpr, tile, ROWWISE: tl.constexpr, EVEN: tl.constexpr
):
    if ROWWISE and BIAS_MODE == 1 and not (H <= BLOCK and H & (H - 1) == 0):
        # Explicitly contiguous bias columns avoid Ascend's scalar gather
        # lowering of a vector remainder. Row tails have their own mask.
        columns = tile % tl.cdiv(H, BLOCK) * BLOCK + tl.arange(0, BLOCK)
        offsets = tile // tl.cdiv(H, BLOCK) * H + columns
        mask = (offsets < N) & (columns < H)
    else:
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
    if EVEN:
        mask = True
    x = tl.load(X + offsets, mask, other=0).to(tl.float32)
    if BIAS_MODE == 0:
        b = tl.load(B).to(tl.float32)
    elif BIAS_MODE == 1:
        if ROWWISE and H <= BLOCK and H & (H - 1) == 0:
            row = tl.load(B + tl.arange(0, H)).to(tl.float32)
            b = tl.broadcast_to(row[None, :], (BLOCK // H, H)).reshape((BLOCK,))
        elif ROWWISE:
            b = tl.load(B + columns, mask, other=0).to(tl.float32)
        else:
            b = tl.load(B + offsets % H, mask, other=0).to(tl.float32)
    else:
        b = tl.load(B + offsets, mask, other=0).to(tl.float32)
    # Match the standalone dtype-rounded add, rather than fusing an FP32 add.
    z = (x + b).to(X.dtype.element_ty).to(tl.float32)
    t = tl_extra_shim.tanh(0.79788456 * z * (1.0 + 0.044715 * z * z))
    tl.store(Y + offsets, 0.5 * z * (1.0 + t), mask)


@libentry()
@triton.jit
def _bias_gelu_forward_kernel(
    X, B, Y, N, H: tl.constexpr, BIAS_MODE: tl.constexpr, BLOCK: tl.constexpr,
    PERSISTENT: tl.constexpr = False, EVEN: tl.constexpr = False,
):
    if PERSISTENT:
        # Bound physical programs to vector cores; stride over arbitrarily many
        # logical tiles. The runtime trip count avoids compile-time unrolling.
        if BIAS_MODE == 1 and H >= 128 and not (H <= BLOCK and H & (H - 1) == 0):
            tiles = (N // H) * tl.cdiv(H, BLOCK)
        else:
            tiles = tl.cdiv(N, BLOCK)
        for tile in range(tle.program_id(0), tiles, tl.num_programs(0)):
            _bias_gelu_tile(X, B, Y, N, H, BIAS_MODE, BLOCK, tile, H >= 128, EVEN)
    else:
        _bias_gelu_tile(X, B, Y, N, H, BIAS_MODE, BLOCK, tle.program_id(0), False, EVEN)


def bias_gelu_impl(input, bias):
    """Return tanh GELU(input + bias), without gradient propagation.

    Operands must share the active accelerator and an FP16/BF16/FP32 dtype.
    PyTorch broadcasting, noncontiguous inputs and empty tensors are supported.
    The result is contiguous, with the broadcast shape and operand dtype.
    Invalid types raise TypeError; invalid dtype, device or shapes raise ValueError.
    """
    for name, tensor in (("input", input), ("bias", bias)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.dtype not in _DTYPES:
            raise ValueError(f"{name} must have FP16, BF16 or FP32 dtype")
        if tensor.device.type != runtime_device.name or tensor.device.type == "cpu":
            raise ValueError(f"{name} must be on the active accelerator")
    if input.dtype != bias.dtype or input.device != bias.device:
        raise ValueError("input and bias must have the same dtype and device")

    vector_bias = input.ndim and bias.ndim == 1 and bias.numel() == input.shape[-1]
    if vector_bias:
        x, b = input, bias
    else:
        try:
            x, b = torch.broadcast_tensors(input, bias)
        except RuntimeError:
            raise ValueError(
                "input and bias must have broadcast-compatible shapes"
            ) from None
    out = torch.empty(x.shape, dtype=input.dtype, device=input.device)
    n = x.numel()
    if not n:
        return out

    h = x.shape[-1] if x.ndim else 1
    if vector_bias:
        b, mode = bias.contiguous(), 1
    elif bias.numel() == 1:
        b, mode = bias.contiguous(), 0
    elif b.stride(-1) == 1 and all(s == 0 for s in b.stride()[:-1]):
        # Preserve a shared bias row instead of copying its full broadcast.
        b, mode = b.reshape(-1, h)[0].contiguous(), 1
    else:
        b, mode = b.contiguous(), 2
    x = x.contiguous()
    block = (
        min(16384, max(128, triton.next_power_of_2(triton.cdiv(n, 40))))
    ) if _IS_ASCEND else _BLOCK_SIZE
    tiles = (n // h) * triton.cdiv(h, block) if _IS_ASCEND and mode == 1 and h >= 128 and not (h <= block and h & (h - 1) == 0) else triton.cdiv(n, block)
    grid = (min(tiles, CORE_NUM) if _IS_ASCEND else tiles,)
    with torch_device_fn.device(input.device):
        _bias_gelu_forward_kernel[grid](
            x, b, out, n, h, mode, BLOCK=block, PERSISTENT=_IS_ASCEND, num_warps=_NUM_WARPS,
            EVEN=_IS_ASCEND and n % block == 0 and (mode != 1 or h % block == 0),
            **({"num_stages": 1} if _IS_ASCEND else {}),
        )
    return out
