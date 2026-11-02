# bias_gelu_impl

`flag_train.megatron.bias_gelu_impl(input, bias)` computes tanh GELU of the
broadcast sum. Both operands must share an accelerator, device index and dtype
(FP16, BF16 or FP32). Scalar and arbitrary broadcast bias, noncontiguous tensors,
empty inputs and first-order gradients for both operands are supported.

The add rounds to the operand dtype before FP32 activation arithmetic. Gradients
round to that dtype before broadcast reduction. Backward bounds the polynomial
at ±10 to preserve saturated derivatives for very large finite values, and
returns NaN derivatives for nonfinite sums. Higher-order gradients are unsupported.
No existing correctness tolerance is changed.

## Dispatch

The generic implementation uses Triton and framework device guards. A vector
bias fast path avoids redundant shape broadcasting and expanded views. NVIDIA
fixed configurations use Triton's own JIT launch cache; other vendors retain
LibEntry dispatch. The global LibEntry runtime is unchanged.

On NVIDIA, contiguous tensors with a one-dimensional bias and fewer than 2^31
elements can use an optional CUDA/C++ implementation. Native C++ Autograd avoids
a Python callback during backward. Aligned loads/stores process four elements
per thread. Large vector-bias backward can accumulate dtype-rounded gradients
in FP32 row segments, then reduce directly into the bias dtype. Alignment and
grid-size guards retain the general path when necessary. CUDA stream and device
selection use PyTorch's current stream and CUDAGuard.

The optional extension builds lazily with the existing PyTorch CUDA toolkit and
C++ compiler, using PyTorch's extension cache. First-call build latency is **not**
steady-state latency and may take minutes. Without the toolkit, or after a build
failure, the operator retains the Triton implementation (and its performance).
No compiler, Torch, Triton or driver is installed by the operator. CUDA sources
are included in package data. The CUDA extension is never imported on other
vendors. This path adds a compiler/build-cache deployment requirement on NVIDIA;
its availability should be checked before performance-sensitive training.

Ascend, Hygon, MetaX, Iluvatar and T-Head still require hardware regression of
the generic broadcast fast path. No performance claims are made for those devices.

## Validation

```bash
export CUDA_VISIBLE_DEVICES=1
# Optional: compile only for this H20 during local iteration.
export TORCH_CUDA_ARCH_LIST=9.0
python -m pytest tests/megatron/test_bias_gelu_impl.py -q -rs --tb=short
python -m pytest tests/megatron/test_bias_gelu_cuda_dispatch.py -q --tb=short
python tools/bench_bias_gelu_h20.py --rounds 3 --output /workspace/flagtrain_results/codex_perf/pytorch_final
python -m pytest benchmark/megatron/test_bias_gelu_impl.py -m bias_gelu_impl \
  --level core --mode kernel --record json \
  --output /workspace/flagtrain_results/codex_perf/megatron_final.json -vs
```

The standalone validation script uses PyTorch `F.gelu(x+b, approximate="tanh")`,
the same inputs and upstream gradient on both sides, full Autograd including
both input and bias gradients, and `do_bench(warmup=25, rep=100, return_mode="median")`.
Forward and backward are measured separately; graph construction is outside the
backward interval. It writes every round to CSV/JSON and computes medians,
arithmetic/geometric means, minima and gate failures. `--shape` selects a core
shape index; `--dtype` and `--direction` allow focused probes. The existing
Megatron benchmark remains unchanged and uses the installed eager implementation.
The H20 experiment logs and full data are stored outside the source tree under
`/workspace/flagtrain_results/codex_perf/`.
