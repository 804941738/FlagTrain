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
"""Triton numerical and autograd regressions beyond the shared operator contract suite."""

import pytest
import torch

import flag_train
from flag_train.runtime import device


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_finite_extreme_gradients(dtype):
    x = torch.tensor(
        [[-1e20, 1e20]], device=flag_train.device, dtype=dtype, requires_grad=True
    )
    b = torch.zeros(2, device=x.device, dtype=dtype, requires_grad=True)
    out = flag_train.megatron.bias_gelu_impl(x, b)
    dx, db = torch.autograd.grad(out, (x, b), torch.ones_like(out))
    torch.testing.assert_close(dx, x.new_tensor([[0, 1]]))
    torch.testing.assert_close(db, b.new_tensor([0, 1]))


def test_saved_tensor_version_check():
    x = torch.randn(64, 64, device=flag_train.device, requires_grad=True)
    b = torch.randn(64, device=x.device, requires_grad=True)
    y = flag_train.megatron.bias_gelu_impl(x, b)
    with torch.no_grad():
        b.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(y.sum(), (x, b))


@pytest.mark.skipif(device.vendor_name != "nvidia", reason="CUDA streams")
def test_nondefault_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(64, 64, device=flag_train.device, requires_grad=True)
        b = torch.randn(64, device=x.device, requires_grad=True)
        y = flag_train.megatron.bias_gelu_impl(x, b)
        g = torch.randn_like(y)
        actual = torch.autograd.grad(y, (x, b), g)
        reference = torch.nn.functional.gelu(x + b, approximate="tanh")
        expected = torch.autograd.grad(reference, (x, b), g)
    stream.synchronize()
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=2e-5, atol=2e-5)


def test_higher_order_rejected():
    x = torch.randn(4, 17, device=flag_train.device, requires_grad=True)
    b = torch.randn(17, device=x.device, requires_grad=True)
    y = flag_train.megatron.bias_gelu_impl(x, b)
    with pytest.raises(RuntimeError, match="first-order"):
        torch.autograd.grad(y, (x, b), torch.ones_like(y), create_graph=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_unaligned_views_and_upstream(dtype):
    # Contiguous views can have unaligned pointers. Reusing a graph with a
    # different upstream pointer must not reuse incompatible specialization.
    x = torch.randn(4097, device=flag_train.device, dtype=dtype)[1:].view(64, 64)
    b = torch.randn(65, device=x.device, dtype=dtype)[1:]
    x.requires_grad_()
    b.requires_grad_()
    y = flag_train.megatron.bias_gelu_impl(x, b)
    ref = torch.nn.functional.gelu(x + b, approximate="tanh")
    for offset in (0, 1, 0):
        g = torch.randn(4097, device=x.device, dtype=dtype)[
            offset : offset + 4096
        ].view(64, 64)
        actual = torch.autograd.grad(y, (x, b), g, retain_graph=True)
        expected = torch.autograd.grad(ref, (x, b), g, retain_graph=True)
        tolerance = {torch.float16: 3e-3, torch.bfloat16: 2e-2, torch.float32: 2e-5}[
            dtype
        ]
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, rtol=tolerance, atol=tolerance)


def test_standard_libentry_dispatch(monkeypatch):
    import importlib

    from flag_train.utils.libentry import LibEntry

    module = importlib.import_module("flag_train.megatron.bias_gelu_impl")
    run = LibEntry.run
    launched = []

    def record(entry, *args, **kwargs):
        launched.append(entry)
        return run(entry, *args, **kwargs)

    monkeypatch.setattr(LibEntry, "run", record)
    x = torch.randn(4, 7, device=flag_train.device, requires_grad=True)
    b = torch.randn(7, device=x.device, requires_grad=True)
    y = flag_train.megatron.bias_gelu_impl(x, b)
    g = torch.randn_like(y)
    actual = torch.autograd.grad(y, (x, b), g)
    expected = torch.autograd.grad(
        torch.nn.functional.gelu(x + b, approximate="tanh"), (x, b), g
    )
    assert launched == [module._forward_saved, module._saved_backward]
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=2e-5, atol=2e-5)


def test_single_row_large_hidden_grid():
    # This hidden size would require >65535 column programs in a 2D grid.
    hidden = (1 << 26) + 1
    x = torch.zeros(
        1, hidden, device=flag_train.device, dtype=torch.float16, requires_grad=True
    )
    b = torch.zeros(hidden, device=x.device, dtype=x.dtype, requires_grad=True)
    y = flag_train.megatron.bias_gelu_impl(x, b)
    dx, db = torch.autograd.grad(y, (x, b), torch.ones_like(y))
    assert dx.shape == x.shape and db.shape == b.shape
    assert torch.all(dx == 0.5)
    assert torch.all(db == 0.5)


def test_input_version_check():
    x = torch.randn(4, 7, device=flag_train.device, requires_grad=True)
    b = torch.randn(7, device=x.device, requires_grad=True)
    y = flag_train.megatron.bias_gelu_impl(x, b)
    with torch.no_grad():
        x.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(y, (x, b), torch.ones_like(y))


@pytest.mark.parametrize("case", ["positive_inf", "opposite_inf", "nan"])
def test_compensated_reduction_nonfinite_upstream(case):
    x = torch.zeros(
        1025, 1024, device=flag_train.device, dtype=torch.float32, requires_grad=True
    )
    b = torch.zeros(1024, device=x.device, dtype=x.dtype, requires_grad=True)
    y = flag_train.megatron.bias_gelu_impl(x, b)
    g = torch.ones_like(y)
    g[0] = float("nan") if case == "nan" else float("inf")
    if case == "opposite_inf":
        g[1] = -float("inf")
    dx, db = torch.autograd.grad(y, (x, b), g)
    if case == "positive_inf":
        assert torch.isposinf(db).all() and torch.isposinf(dx[0]).all()
    else:
        assert torch.isnan(db).all()


def test_large_row_fp32_high_precision_reduction():
    # Keep per-element FP32 rounding, but use FP64 reference accumulation:
    # a 16384-row FP32 oracle sum can itself lose >2e-5 near cancellation.
    from .. import accuracy_utils as utils

    if utils.QUICK_MODE:
        pytest.skip("large-row precision regression")
    torch.manual_seed(0)
    x = torch.randn(
        16384, 4096, device=flag_train.device, dtype=torch.float32, requires_grad=True
    )
    b = torch.randn(4096, device=x.device, dtype=x.dtype, requires_grad=True)
    g = torch.randn(16384, 4096, 2, device=x.device, dtype=x.dtype)[..., 0]
    y = flag_train.megatron.bias_gelu_impl(x, b)
    dx, db = torch.autograd.grad(y, (x, b), g)
    reference_device = (
        x.device
        if utils.fp64_is_supported and not utils.TO_CPU
        else torch.device("cpu")
    )
    with torch.no_grad():
        for start in range(0, 4096, 128):
            columns = slice(start, start + 128)
            z = (x[:, columns] + b[columns]).to(reference_device).double()
            t = torch.tanh(0.79788456 * z * (1 + 0.044715 * z * z))
            derivative = 0.5 * (1 + t) + 0.5 * z * (1 - t * t) * (
                0.79788456 + 0.1070322243 * z * z
            )
            expected_dx = (g[:, columns].to(reference_device) * derivative).float()
            expected_db = expected_dx.double().sum(0).float()
            torch.testing.assert_close(
                dx[:, columns], expected_dx.to(x.device), rtol=2e-5, atol=2e-5
            )
            torch.testing.assert_close(
                db[columns], expected_db.to(x.device), rtol=2e-5, atol=2e-5
            )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_wide_bias_column_reduction(dtype):
    from .test_bias_gelu_impl import _check

    torch.manual_seed(0)
    x = torch.randn(256, 32768, device=flag_train.device, dtype=dtype)
    b = torch.randn(32768, device=x.device, dtype=dtype)
    _check(x, b)
