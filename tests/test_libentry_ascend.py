# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Ascend cache-hit dispatch: fresh operands, specialization and stream state."""

import importlib

import pytest
import torch
import triton
import triton.language as tl

import flag_train
from flag_train.utils import libentry

module = importlib.import_module("flag_train.utils.libentry")
pytestmark = pytest.mark.skipif(
    flag_train.vendor_name != "ascend", reason="Ascend dispatch"
)


@libentry()
@triton.jit
def _copy_scaled(X, Y, N, SCALE: tl.constexpr = 1, BLOCK: tl.constexpr = 128):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(
        Y + offsets, tl.load(X + offsets, offsets < N, other=0) * SCALE, offsets < N
    )


@pytest.mark.parametrize("fast", [False, True])
def test_entry_argument_and_specialization_changes(fast):
    entry = module.LibEntry(_copy_scaled.fn)
    entry._ascend_plain_jit = fast
    for dtype in [torch.float32, torch.float16]:
        for n, scale, offset in [(129, 1, 0), (257, 2, 1), (129, 3, 0), (129, 1, 0)]:
            x = torch.arange(n + offset, device="npu", dtype=dtype)[offset:]
            y = torch.full_like(x, float("nan"))
            # Both cold/miss and hot calls must bind new pointers and constexprs.
            for _ in range(2):
                entry[(triton.cdiv(n, 128),)](x, y, n, SCALE=scale)
                torch.testing.assert_close(y.cpu(), x.cpu() * scale)
            z = torch.empty_like(x)
            entry[(triton.cdiv(n, 128),)](X=x, Y=z, N=n, SCALE=scale, BLOCK=128)
            torch.testing.assert_close(z.cpu(), x.cpu() * scale)


def test_callable_grid_uses_standard_path():
    x = torch.arange(129, device="npu", dtype=torch.float32)
    y = torch.empty_like(x)
    for _ in range(2):
        _copy_scaled[lambda meta: (triton.cdiv(meta["N"], meta["BLOCK"]),)](x, y, 129)
        torch.testing.assert_close(y.cpu(), x.cpu())


def test_current_stream_is_queried_for_each_hit(monkeypatch):
    x = torch.arange(129, device="npu", dtype=torch.float32)
    y = torch.empty_like(x)
    _copy_scaled[(2,)](x, y, 129)
    torch.npu.synchronize()
    original = _copy_scaled._ascend_raw_stream
    seen = []

    def record(device):
        stream = original(device)
        seen.append(stream)
        return stream

    monkeypatch.setattr(_copy_scaled, "_ascend_raw_stream", record)
    for stream in [torch.npu.Stream(), torch.npu.Stream()]:
        with torch.npu.stream(stream):
            for _ in range(2):
                _copy_scaled[(2,)](x, y, 129)
                assert seen[-1] == stream.npu_stream
        stream.synchronize()
        torch.testing.assert_close(y.cpu(), x.cpu())


def test_launch_metadata_hooks_remain_active(monkeypatch):
    x = torch.arange(129, device="npu", dtype=torch.float32)
    y = torch.empty_like(x)
    _copy_scaled[(2,)](x, y, 129)
    kernel = next(iter(_copy_scaled.kernel_cache[0].values()))[0]
    original = kernel.launch_metadata
    seen = []

    def metadata(grid, stream, *args):
        seen.append((grid, stream, args[0], args[1]))
        return original(grid, stream, *args)

    monkeypatch.setattr(kernel, "launch_metadata", metadata)
    _copy_scaled[(2,)](x, y, 129)
    assert seen and seen[-1][0] == (2, 1, 1)
    assert seen[-1][2] is x and seen[-1][3] is y
    torch.testing.assert_close(y.cpu(), x.cpu())


def test_other_vendor_does_not_enable_fast_path(monkeypatch):
    monkeypatch.setattr(module.runtime.device, "vendor_name", "nvidia")
    entry = module.LibEntry(_copy_scaled.fn)
    assert not entry._ascend_plain_jit
    # Exercise the unchanged generic dispatch on the available NPU backend.
    # This is a control-flow regression, not NVIDIA hardware validation.
    x = torch.arange(129, device="npu", dtype=torch.float32)
    y = torch.empty_like(x)
    for _ in range(2):
        entry[(2,)](x, y, 129)
        torch.testing.assert_close(y.cpu(), x.cpu())


@libentry()
@triton.jit(do_not_specialize=["N"])
def _copy_dynamic(X, Y, N, BLOCK: tl.constexpr = 128):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + offsets, tl.load(X + offsets, offsets < N, other=0), offsets < N)


def test_dynamic_scalar_and_integer_width():
    entry = module.LibEntry(_copy_dynamic.fn)
    x = torch.arange(256, device="npu", dtype=torch.float32)
    y = torch.full_like(x, float("nan"))
    entry[(2,)](x, y, 129)
    torch.testing.assert_close(y[:129].cpu(), x[:129].cpu())
    assert torch.isnan(y[129:]).all()
    entry[(2,)](x, y, 256)
    assert len(entry.kernel_cache[0]) == 1
    torch.testing.assert_close(y.cpu(), x.cpu())
    entry[(2,)](x, y, (1 << 31) + 1)
    assert len(entry.kernel_cache[0]) == 2
    torch.testing.assert_close(y.cpu(), x.cpu())


def test_heuristic_entry_keeps_standard_dispatch():
    wrapped = triton.heuristics({"SCALE": lambda args: 2})(_copy_scaled.fn)
    entry = module.LibEntry(wrapped)
    assert not entry._ascend_plain_jit
    x = torch.arange(129, device="npu", dtype=torch.float32)
    y = torch.empty_like(x)
    for _ in range(2):
        entry[(2,)](x, y, 129)
        torch.testing.assert_close(y.cpu(), 2 * x.cpu())


def test_enter_and_exit_hooks_are_forwarded(monkeypatch):
    x = torch.arange(129, device="npu", dtype=torch.float32)
    y = torch.empty_like(x)
    _copy_scaled[(2,)](x, y, 129)
    seen = []
    monkeypatch.setattr(
        module._triton_knobs.runtime,
        "launch_enter_hook",
        lambda metadata: seen.append("enter"),
    )
    monkeypatch.setattr(
        module._triton_knobs.runtime,
        "launch_exit_hook",
        lambda metadata: seen.append("exit"),
    )
    _copy_scaled[(2,)](x, y, 129)
    torch.testing.assert_close(y.cpu(), x.cpu())
    assert seen == ["enter", "exit"]


def test_missing_nowait_capability_uses_compiled_runner(monkeypatch):
    entry = module.LibEntry(_copy_scaled.fn)
    monkeypatch.setattr(entry, "_ascend_raw_stream", None)
    x = torch.arange(129, device="npu", dtype=torch.float32)
    y = torch.empty_like(x)
    for _ in range(2):
        entry[(2,)](x, y, 129)
        torch.testing.assert_close(y.cpu(), x.cpu())
