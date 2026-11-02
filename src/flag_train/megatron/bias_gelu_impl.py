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


@libentry()
@triton.jit
def _bias_gelu_forward_kernel(
    X, B, Y, N, H: tl.constexpr, BIAS_MODE: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tle.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X + offsets, mask, other=0).to(tl.float32)
    if BIAS_MODE == 0:
        b = tl.load(B).to(tl.float32)
    elif BIAS_MODE == 1:
        b = tl.load(B + offsets % H, mask, other=0).to(tl.float32)
    else:
        b = tl.load(B + offsets, mask, other=0).to(tl.float32)
    # Match the standalone dtype-rounded add, rather than fusing an FP32 add.
    z = (x + b).to(X.dtype.element_ty).to(tl.float32)
    t = tl_extra_shim.tanh(0.79788456 * z * (1.0 + 0.044715 * z * z))
    tl.store(Y + offsets, 0.5 * z * (1.0 + t), mask)


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
    with torch_device_fn.device(input.device):
        _bias_gelu_forward_kernel[(triton.cdiv(n, _BLOCK_SIZE),)](
            x, b, out, n, h, mode, BLOCK=_BLOCK_SIZE, num_warps=_NUM_WARPS
        )
    return out
