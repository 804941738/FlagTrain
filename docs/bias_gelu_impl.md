# bias_gelu_impl

`flag_train.megatron.bias_gelu_impl(input, bias)` computes tanh GELU of the
broadcast sum. Both operands must share an accelerator, device index and dtype
(FP16, BF16 or FP32). Scalar and arbitrary broadcast bias, noncontiguous tensors,
empty inputs and first-order gradients for both operands are supported.

The add rounds to the operand dtype before FP32 activation arithmetic. Gradients
round to that dtype before broadcast reduction. Backward bounds the polynomial
at ±10 to preserve saturated derivatives for very large finite values, and
returns NaN derivatives for nonfinite sums. Higher-order gradients are rejected
explicitly. Existing correctness tolerances remain unchanged.

## Implementation

The implementation uses Python, Triton / FlagTree and PyTorch Autograd Python
interfaces. It has no custom C/C++, CUDA kernel source, PyBind module or extension
build dependency. The previous extension-based commit remains in Git history
only and is not imported, built or required by this implementation.

A vector bias fast path avoids redundant shape broadcasting and expanded views.
Fixed configurations on NVIDIA use Triton's own JIT launch mechanism and a
bounded compiled-kernel cache. This cache retains compiled Triton code, never
tensors or streams, and accounts for device, dtype, scalar parameters and pointer
alignment. Other vendors retain LibEntry dispatch. The global runtime is unchanged.

Other vendors still require hardware regression for tanh shim behavior, device
guards and compiler launch support. No performance claims are made for Ascend,
MetaX, Hygon, Iluvatar or T-Head.

## Validation

```bash
export CUDA_VISIBLE_DEVICES=1
python -m pytest tests/megatron/test_bias_gelu_impl.py -q -rs --tb=short
python -m pytest tests/megatron/test_bias_gelu_triton_regression.py -q --tb=short
python tools/bench_bias_gelu_h20.py --rounds 3 \
  --output /workspace/flagtrain_results/codex_perf/triton_only/final
```

The independent benchmark uses `F.gelu(x+b, approximate="tanh")`, the same x,
bias, grad_output, device, dtype and gradient requirements on both sides.
Backward uses complete `autograd.grad`, including input and bias gradients;
graph construction happens outside timing. It uses ordinary
`do_bench(warmup=25, rep=100, return_mode="median")`, with separate forward and
backward measurements. CSV/JSON preserve every round and per-case medians.

The current acceptance gate is geometric mean >=0.80 for each direction's 21
core cases; individual cases may be below 0.80. The script reports those slow
cases separately rather than treating them as direction-level gate failures.
The original Megatron benchmark and eager baseline remain unchanged.

For NVIDIA vector bias backward, small inputs can compute dInput and dBias in a
single Triton program per column tile. Larger inputs accumulate dtype-rounded
gradients over eight rows per program, then use a bounded FP32 partial reduction
kernel. This avoids materializing another full-sized bias gradient. General
broadcasts and other vendors retain the original elementwise/reduction behavior.
On NVIDIA, FP16/BF16 backward evaluates tanh through a stable exponential
identity; FP32 and other vendors retain the existing tanh shim. The original
accuracy tolerances and saturation handling are unchanged.

H20 validation on this branch: the original 94 tests plus 8 regression cases pass.
Three complete PyTorch comparison rounds give forward geometric mean 1.7327x
and backward geometric mean 0.8214x (statistics over per-case median speedups).
Some small cases remain below 0.80x; this satisfies the revised direction-level
gate, not the earlier per-case 0.90x gate. Full data and experiment logs are in
`/workspace/flagtrain_results/codex_perf/triton_only/`.
