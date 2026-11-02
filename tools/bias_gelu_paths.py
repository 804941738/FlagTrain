# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit, process-local diagnostic selection; never imported by the operator."""

import importlib
from contextlib import contextmanager
from types import SimpleNamespace

import flag_train


@contextmanager
def generic_execution():
    """Use real generic launch policy on the present device, not foreign hardware.

    Run in a separate diagnostic process: mutating module bindings is not a
    thread-safe production dispatch API. Kernel mathematics are unchanged.
    """
    module = importlib.import_module("flag_train.megatron.bias_gelu_impl")
    if flag_train.megatron.bias_gelu_impl is not module.bias_gelu_impl:
        raise RuntimeError(
            "A vendor override is active; generic selection needs review"
        )
    original = module.runtime_device, module._forward_launch, module._backward_launch
    module.runtime_device = SimpleNamespace(
        vendor_name="generic", name=original[0].name
    )
    module._forward_launch = module._forward
    module._backward_launch = module._backward
    try:
        yield
    finally:
        module.runtime_device, module._forward_launch, module._backward_launch = (
            original
        )
