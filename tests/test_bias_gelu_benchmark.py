# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Baseline policy regressions; synthetic failures never enter performance data."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from benchmark.megatron import test_bias_gelu_impl as bench

pytestmark = pytest.mark.bias_gelu_impl


@pytest.fixture
def installed(monkeypatch):
    def helper(*args):
        return args[0]

    helper._torchdynamo_orig_callable = object()
    native = object()
    module = SimpleNamespace(
        bias_gelu=helper, bias_gelu_impl=native, __file__="/test/megatron/fusion.py"
    )
    monkeypatch.setattr(
        bench,
        "importlib",
        SimpleNamespace(
            import_module=lambda name: module,
            metadata=SimpleNamespace(version=lambda name: "0.17.1"),
        ),
    )
    monkeypatch.setattr(bench.flag_train, "vendor_name", "metax")
    return module


def compiler_failure(inner=None):
    from torch._dynamo.exc import BackendCompilerFailed

    return BackendCompilerFailed(
        lambda: None, inner or RuntimeError("test backend unsupported"), None
    )


def test_native_interface_is_preserved(installed):
    assert bench.base_op() is installed.bias_gelu_impl
    assert installed.bias_gelu._torchdynamo_orig_callable is not bench.base_op()


def test_missing_megatron_is_unavailable(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError("test megatron is missing")

    monkeypatch.setattr(bench, "importlib", SimpleNamespace(import_module=missing))
    with pytest.raises(RuntimeError, match="not a Megatron") as exc:
        bench.base_op()
    assert isinstance(exc.value.__cause__, ModuleNotFoundError)


@pytest.mark.parametrize(
    "vendor", ["nvidia", "ascend", "metax", "iluvatar", "hygon", "thead"]
)
def test_compile_success_on_every_vendor(installed, monkeypatch, vendor):
    monkeypatch.setattr(bench.flag_train, "vendor_name", vendor)
    calls = []

    def preflight(b, reference, metadata, compiled):
        calls.append((reference, compiled))
        metadata["cases"].append({"mode": "compiled"})

    monkeypatch.setattr(bench, "_preflight", preflight)
    result = bench._prepare_baseline(None, installed.bias_gelu_impl)
    assert result["mode"] == "compiled" and result["fallback_reason"] is None
    assert calls == [(installed.bias_gelu_impl, True)]


@pytest.mark.parametrize("vendor", ["ascend", "metax", "iluvatar", "hygon", "thead"])
def test_partial_compile_failure_replays_entire_group(installed, monkeypatch, vendor):
    monkeypatch.setattr(bench.flag_train, "vendor_name", vendor)
    calls = []

    def preflight(b, reference, metadata, compiled):
        calls.append(compiled)
        if compiled:
            metadata["cases"].append({"shape": [64, 64], "mode": "compiled"})
            metadata["current_case"] = {"shape": [1024, 1024]}
            raise compiler_failure()
        metadata["cases"].extend({"mode": "eager"} for _ in range(21))

    monkeypatch.setattr(bench, "_preflight", preflight)
    result = bench._prepare_baseline(None, installed.bias_gelu_impl)
    assert calls == [True, False]
    assert result["mode"] == "eager" and len(result["cases"]) == 21
    assert all(case["mode"] == "eager" for case in result["cases"])
    assert result["compile_attempts"][0]["mode"] == "compiled"
    assert result["failed_case"]["shape"] == [1024, 1024]
    assert "test backend unsupported" in result["fallback_reason"]
    assert "BackendCompilerFailed" in result["compilation_traceback"]


def test_nvidia_compile_failure_cannot_fallback(installed, monkeypatch):
    monkeypatch.setattr(bench.flag_train, "vendor_name", "nvidia")

    def fail(*a, **kw):
        raise compiler_failure()

    monkeypatch.setattr(bench, "_preflight", fail)
    with pytest.raises(RuntimeError, match="NVIDIA requires compiled"):
        bench._prepare_baseline(None, installed.bias_gelu_impl)


@pytest.mark.parametrize(
    "error",
    [
        AssertionError("wrong output"),
        ValueError("bad shape"),
        RuntimeError("device error"),
    ],
)
def test_runtime_or_correctness_failures_are_not_fallback(
    installed, monkeypatch, error
):
    def fail(*a, **kw):
        raise error

    monkeypatch.setattr(bench, "_preflight", fail)
    with pytest.raises(type(error), match=str(error)):
        bench._prepare_baseline(None, installed.bias_gelu_impl)


def test_wrapped_assertion_is_not_compile_fallback():
    assert not bench._is_compile_failure(compiler_failure(AssertionError("wrong")))
    assert not bench._is_compile_failure(
        compiler_failure(torch.OutOfMemoryError("oom"))
    )
    assert not bench._is_compile_failure(
        compiler_failure(compiler_failure(ValueError("nested invalid input")))
    )


def test_preimported_eager_is_rejected_on_nvidia(installed, monkeypatch):
    del installed.bias_gelu._torchdynamo_orig_callable
    monkeypatch.setattr(bench.flag_train, "vendor_name", "nvidia")
    with pytest.raises(RuntimeError, match="fresh process"):
        bench._prepare_baseline(None, installed.bias_gelu_impl)


def test_disabled_default_jit_is_labelled_eager(installed, monkeypatch):
    del installed.bias_gelu._torchdynamo_orig_callable
    calls = []
    monkeypatch.setattr(
        bench, "_preflight", lambda b, r, m, compiled: calls.append(compiled)
    )
    result = bench._prepare_baseline(None, installed.bias_gelu_impl)
    assert calls == [False]
    assert result["mode"] == "eager" and "disabled" in result["fallback_reason"]


def test_compile_config_restored_after_error():
    config = torch._dynamo.config
    names = [
        n
        for n in [
            "recompile_limit",
            "cache_size_limit",
            "accumulated_recompile_limit",
            "suppress_errors",
            "fail_on_recompile_limit_hit",
            "automatic_dynamic_shapes",
        ]
        if hasattr(config, n)
    ]
    saved = {n: getattr(config, n) for n in names}
    with pytest.raises(ValueError, match="sentinel"):
        with bench._compile_budget():
            for n in ["recompile_limit", "cache_size_limit"]:
                if hasattr(config, n):
                    assert getattr(config, n) >= 128
            assert not config.suppress_errors
            if hasattr(config, "automatic_dynamic_shapes"):
                assert not config.automatic_dynamic_shapes
            raise ValueError("sentinel")
    assert {n: getattr(config, n) for n in names} == saved


def test_eager_fallback_uses_official_stance_and_restores(monkeypatch):
    active = []

    @contextmanager
    def stance(name):
        active.append(name)
        try:
            yield
        finally:
            active.pop()

    monkeypatch.setattr(torch.compiler, "set_stance", stance)
    with bench._execution_mode("eager"):
        assert active == ["force_eager"]
    assert active == []


def test_no_unsafe_eager_fallback_on_old_torch(monkeypatch):
    monkeypatch.delattr(torch.compiler, "set_stance")
    with pytest.raises(RuntimeError, match="requires torch.compiler.set_stance"):
        bench._execution_mode("eager")
    with bench._execution_mode("eager", jit_wrapped=False):
        pass


def test_compiled_timing_uses_cache_miss_guard_not_stance(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not change the hot eval-frame callback")

    monkeypatch.setattr(torch.compiler, "set_stance", forbidden)
    saved = torch._dynamo.config.error_on_recompile
    with bench._execution_mode("compiled"):
        assert torch._dynamo.config.error_on_recompile
    assert torch._dynamo.config.error_on_recompile == saved


@pytest.mark.parametrize("name", ["cache_size_limit", "recompile_limit"])
def test_legacy_and_current_compile_budgets(monkeypatch, name):
    config = SimpleNamespace(suppress_errors=True, **{name: 8})

    @contextmanager
    def patch(updates):
        saved = {n: getattr(config, n) for n in updates}
        for n, value in updates.items():
            setattr(config, n, value)
        try:
            yield
        finally:
            for n, value in saved.items():
                setattr(config, n, value)

    config.patch = patch
    monkeypatch.setattr(
        bench, "torch", SimpleNamespace(_dynamo=SimpleNamespace(config=config))
    )
    with bench._compile_budget():
        assert getattr(config, name) == 128
        assert not config.suppress_errors
    assert getattr(config, name) == 8 and config.suppress_errors


def test_preflight_executes_native_and_cached_replay(monkeypatch):
    calls = []
    counts = iter(
        [
            {"stats": {}, "graph_break": {}, "unimplemented": {}},
            {"stats": {"unique_graphs": 1}, "graph_break": {}, "unimplemented": {}},
        ]
    )
    monkeypatch.setattr(bench, "_compile_counts", lambda: next(counts))
    monkeypatch.setattr(bench.torch_device_fn, "synchronize", lambda: None)
    evidence = bench._probe(lambda x: calls.append(x), (17,), compiled=True)
    assert calls == [17, 17] and evidence["compiled_graphs"] == 1


@pytest.mark.parametrize("key", ["graph_break", "unimplemented"])
def test_preflight_rejects_partial_or_silent_fallback(monkeypatch, key):
    from torch._dynamo.exc import Unsupported

    before = {"stats": {}, "graph_break": {}, "unimplemented": {}}
    after = dict(before, stats={"unique_graphs": 1})
    after[key] = {"unsupported backend": 1}
    counts = iter([before, after])
    monkeypatch.setattr(bench, "_compile_counts", lambda: next(counts))
    monkeypatch.setattr(bench.torch_device_fn, "synchronize", lambda: None)
    with pytest.raises(Unsupported, match=key):
        bench._probe(lambda: None, (), compiled=True)
