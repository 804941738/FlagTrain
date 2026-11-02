# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Optional native autograd dispatch for NVIDIA contiguous vector bias.

The CUDA toolkit is optional: installations without a compiler retain Triton.
Compilation is lazy and uses PyTorch's process-safe extension cache.
"""

import logging
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def implementation():
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        return None
    sources = Path(__file__).with_name("csrc")
    try:
        return load(
            name="flagtrain_bias_gelu_h20",
            sources=[str(sources / "bias_gelu.cpp"), str(sources / "bias_gelu.cu")],
            extra_cuda_cflags=["-O3"],
        ).apply
    except (OSError, RuntimeError) as exc:
        logging.getLogger(__name__).warning(
            "Optional bias GELU CUDA extension unavailable; using Triton: %s", exc
        )
        return None
