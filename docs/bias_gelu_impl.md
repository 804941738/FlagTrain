# Bias GELU

`flag_train.megatron.bias_gelu_impl(input, bias)` returns the tanh approximation
of GELU applied to the broadcast sum:

```text
z = input + bias
y = 0.5 * z * (1 + tanh(0.79788456 * z * (1 + 0.044715 * z * z)))
```

Both operands must be tensors on the same active accelerator, with the same
FP16, BF16 or FP32 dtype. The output has the broadcast shape and operand dtype.
Scalar, vector and general PyTorch broadcasting are supported, including empty
and noncontiguous tensors. Noncontiguous operands are materialized as needed.
Invalid operand types raise `TypeError`; invalid dtype, device or broadcast
shapes raise `ValueError`.

## Triton implementation

Every device kernel uses `@libentry()` above `@triton.jit` and is launched through
its standard `kernel[grid](...)` interface inside `torch_device_fn.device(...)`.
Compilation, specialization, stream selection and caching belong to FlagTrain
and its installed Triton/FlagTree backend. The operator has no vendor-specific
launch branch, private cache or dependency on LibEntry's internal cache layout.

All six target platforms reuse these kernels from the same source:

- `_forward`: inference and general broadcast forward.
- `_forward_saved`: vector-bias training forward, writing output and a FP32
  derivative checkpoint in one pass.
- `_backward`: elementwise gradient for the broadcast fallback.
- `_saved_backward`: small vector-bias gradient and local bias reduction.
- `_saved_backward_columns`: column-owned reduction with no partial workspace.
- `_saved_backward_loop`: elementwise or segmented vector-bias gradient.
- `_reduce_bias`: final reduction of FP32 partial bias gradients.

Indexing uses `tle.program_id` and wide offsets. Grids use only their first
dimension, avoiding secondary grid-dimension limits. Mathematical helpers use
standard Triton operations and `tl_extra_shim.tanh`, which the platform layer
selects through `triton_lang_helper.py`. No custom C/C++ or CUDA extension is used.

## Autograd and reduction

Two Python Autograd classes keep different saved-state lifetimes explicit.
`_BiasGeluSaved` handles nonempty vector bias when either operand needs gradients.
It caches the derivative and retains the operands for Autograd version checks.
`_BiasGelu` handles other broadcasts, empty inputs and inference; it saves operands
and reconstructs the derivative when needed. General broadcast gradients are
returned in the expanded output shape and reduced by the Autograd engine to each
original operand shape. No derivative checkpoint is allocated for inference.

Backward supports input-only, bias-only and joint gradients. Single-row bias
gradients reuse a view of dInput. Small inputs use a local reduction; medium
inputs and inputs with at most 256 rows use a column-owned reduction; other
larger inputs write partial sums and launch
a second reduction kernel. These strategies have different memory access and
parallelism requirements, so retaining them avoids forcing every shape through
one inefficient reduction. Programs write disjoint output ranges; the final
reduction runs afterwards on the same current stream, with no atomics or
cross-program synchronization.

Higher-order gradients raise `RuntimeError`. Nondefault streams and saved-tensor
in-place modification checks are covered by regression tests.

## Numerical contract

Bias addition rounds to the operand dtype before FP32 activation arithmetic.
Elementwise gradients round to the operand dtype **before** bias reduction;
partial sums accumulate in FP32 and cast only at the final bias writeback.

The shared FP32 derivative uses a stable logistic/sech expression to avoid
cancellation, retaining the FP32 tanh saturation behavior. Other dtypes reuse the
forward tanh. The derivative bounds its polynomial at ±10 so extreme finite
values retain saturated gradients without `0 * inf`. Nonfinite pre-activations
have NaN derivatives.

FP32 segmented reductions use compensated accumulation. Nonfinite accumulators
reset the compensation to preserve infinity/NaN propagation. The large-row
accuracy regression keeps per-element FP32 rounding and unchanged `2e-5`
tolerances, with FP64 reference accumulation on CPU when the backend lacks FP64.
Production arithmetic does not require FP64.

## Training memory

The derivative checkpoint costs **4 * N bytes**, where N is the output element
count. A segmented reduction additionally needs
`4 * ceil(rows / BR) * hidden` bytes of partial sums, as well as the gradient
outputs. For a 16384x4096 FP32 activation, the derivative is **256 MiB** and the
32-row partial workspace is **8 MiB**. Checkpointing trades this memory for
avoiding repeated activation evaluation during backward. Inference has no
checkpoint, and single-row reduction has no partial workspace.

## Validation on H20

```bash
export CUDA_VISIBLE_DEVICES=1
python -m pytest tests/megatron/test_bias_gelu_impl.py \
  tests/megatron/test_bias_gelu_triton_regression.py -q -rs --tb=short
python tools/bench_bias_gelu_h20.py --rounds 3 \
  --output /workspace/flagtrain_results/codex_perf/unified_validation
python -m pytest benchmark/megatron/test_bias_gelu_impl.py \
  -m bias_gelu_impl --level core --mode kernel --record json \
  --output /workspace/flagtrain_results/codex_perf/unified_megatron.json -vs
```

The independent H20 tool measures the complete public interface against
`F.gelu(input + bias, approximate="tanh")`, using the same operands and upstream
gradients. It covers seven core shapes, three dtypes and separate forward and
backward timings, with full `autograd.grad` for both operands. Backward graphs
are prepared outside timing. Ordinary `triton.testing.do_bench` uses 25 ms
warmup, 100 ms repetition and median latency. All rounds, absolute latencies,
per-item speedups and arithmetic/geometric aggregates are saved to CSV/JSON.
Single-shape, dtype and direction filters support diagnosis. The direction
geometric-mean acceptance floor is 0.80; individual slower cases are reported.
Megatron Eager remains a separate reference and is never substituted for PyTorch.

## Backend validation

NVIDIA H20, Ascend 910B, MetaX, Iluvatar, Hygon and T-Head use the same computation
and public LibEntry launch mechanism, subject to FlagTrain's existing external
vendor registration. Deployments should verify the exported function's module
when attributing results to this source.

Current tiling defaults were measured on H20. Other devices still require real
compilation, correctness and performance tests, especially for tanh/exp accuracy,
FP32 reduction, grid limits, register/shared-memory use and compiler layout
support. If necessary, tune the host shape policy without copying mathematical
kernels. H20 results do not establish performance or compatibility on other
hardware. CPU-only whole-package import is also not certified by H20 tests.
