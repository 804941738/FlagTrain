# Bias GELU (Forward only)

`flag_train.megatron.bias_gelu_impl(input, bias)` computes:

```text
z = input + bias
y = 0.5 * z * (1 + tanh(0.79788456 * z * (1 + 0.044715 * z * z)))
```

Operands must be FP16, BF16 or FP32 tensors on the same active accelerator and
must obey PyTorch broadcasting. Scalar, vector and general broadcast bias,
noncontiguous inputs and empty tensors are supported. The result is contiguous,
with the broadcast shape and operand dtype. Invalid types raise `TypeError`;
invalid dtype, device or broadcast shapes raise `ValueError`.

The result is a plain tensor with `requires_grad=False` and `grad_fn=None`, even
when an operand requires gradients. This operator does not propagate gradients,
matching the repository's Forward-only `fast_gelu` design.

## Implementation and precision

One `_bias_gelu_forward_kernel` implements scalar, shared-vector and elementwise
bias loads. A regular vector bias is read directly without creating a full-size
broadcast. For general broadcasting, Python prepares broadcast views and
materializes noncontiguous data when necessary. A bias shared across rows is
kept as one row. Empty outputs return without launching a kernel.

Bias addition rounds to the operand dtype before FP32 activation arithmetic.
The output casts to the operand dtype at the final store. The formula and
existing precision tolerances are preserved; positive infinity yields positive
infinity, while negative infinity and NaN yield NaN. Extreme finite values retain
the formula's saturated behavior.

The kernel uses `@libentry()` above `@triton.jit`, `tle.program_id` for wide
indexing and `tl_extra_shim.tanh` for backend-selected mathematics. Python uses
`torch_device_fn.device` and the standard `kernel[grid](...)` interface.
Compilation, specialization and caching belong to the installed runtime.
A fixed BLOCK=256 and num_warps=4 keep the launch policy simple; there is no
vendor branch, private cache, custom native extension or intermediate activation
workspace. Noncontiguous or general broadcast inputs may require layout copies.

NVIDIA H20, Ascend 910B, MetaX, Iluvatar, Hygon and T-Head can reuse the same
source and launch mechanism, subject to FlagTrain's existing vendor registration.
Only H20 is tested here. The other platforms still need real compilation,
correctness and performance tests for tanh accuracy, indexing, grid limits and
compiler layout support; H20 measurements do not certify those devices.

## Validation on H20

```bash
export CUDA_VISIBLE_DEVICES=1
python -m pytest tests/megatron/test_bias_gelu_impl.py \
  tests/megatron/test_bias_gelu_triton_regression.py -q -rs --tb=short
python tools/bench_bias_gelu_h20.py --rounds 3 \
  --output /workspace/flagtrain_results/codex_perf/forward_only/pytorch
python -m pytest benchmark/megatron/test_bias_gelu_impl.py \
  -m bias_gelu_impl --level core --mode kernel --record json \
  --output /workspace/flagtrain_results/codex_perf/forward_only/megatron.json -vs
```

Correctness covers all three dtypes, scalar/vector/general broadcasts, layout
conversions, empty and large tensors, extreme and nonfinite values, invalid
arguments, the absence of an output gradient node, and nondefault CUDA streams.
Numerical references include the dtype-rounded formula, PyTorch tanh GELU and
the installed Megatron Forward implementation.

The H20 tool measures the complete public interface against
`F.gelu(input + bias, approximate="tanh")` on identical inputs without gradient
requirements. It uses ordinary `triton.testing.do_bench`, 25 ms warmup, 100 ms
repetition and median latency. Seven shapes and three dtypes form 21 Forward
cases. At least three complete rounds are required for formal performance
acceptance: take each case's median of per-round speedups, then the equally
weighted **arithmetic mean** of those 21 medians, which must reach **0.80**.
A ratio of summed latencies is not this metric. Geometric means and minimum
speedups are diagnostic only. CSV/JSON preserve every latency, case median and
round aggregate. `--shape` and `--dtype` filters support diagnosis; incomplete
case sets or fewer than three rounds explicitly receive no formal PASS result.
The unchanged Megatron Eager baseline is measured separately.
