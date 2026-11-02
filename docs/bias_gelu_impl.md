# Portable bias_gelu_impl

`flag_train.megatron.bias_gelu_impl(input, bias)` computes tanh GELU of the
broadcast sum. Both operands must share an accelerator, device index and dtype
(FP16, BF16 or FP32). Scalar and arbitrary broadcast bias, noncontiguous tensors,
empty inputs and first-order gradients for both operands are supported.

The addition rounds to the operand dtype before FP32 activation arithmetic.
Elementwise gradients round to the operand dtype **before** bias reduction.
Reductions accumulate in FP32 and cast only at final writeback. FP32 segmented
sums use compensated accumulation; nonfinite accumulators reset the compensation
to preserve infinity/NaN propagation. The derivative
bounds its polynomial at ±10 to preserve saturated gradients for extreme finite
values; nonfinite sums have NaN derivatives. Higher orders are explicitly rejected.
Existing correctness tolerances are unchanged.

## Shared computation

All six backends import this same source; no vendor-specific implementation of
this operator is present in the repository. FlagTrain's existing vendor registrar
can still install an external override, so each deployment should verify the
public function's `__module__` before attributing results to this implementation.
There are no custom C/C++ or CUDA sources, extension loaders, PyBind modules,
private compiler builds, inline assembly or CUDA-only imports in the operator.

For nonempty vector bias requiring gradients, `_BiasGeluSaved` executes:

1. `_forward_saved`: fused output plus FP32 derivative checkpoint, reusing tanh.
2. `_saved_backward`, `_saved_backward_columns` or `_saved_backward_loop`: multiply
   the checkpoint by grad_output, round dInput, and accumulate bias gradients.
3. `_reduce_bias` when partial reduction is needed.

These kernels and the `_gelu_derivative` mathematical helper are shared across
NVIDIA, Ascend, MetaX, Iluvatar, Hygon and T-Head. Math uses `tl_extra_shim.tanh`,
selected by the installed backend through `triton_lang_helper.py`; there is no
separate NVIDIA tanh formula. FP32 derivatives use an equivalent logistic/sech
expression to avoid cancellation, retaining exact FP32 tanh saturation; other
dtypes reuse the forward tanh expression. This dtype policy is shared by all
backends and introduces no FP64 production arithmetic. Indexing uses `tle.program_id` and wide offsets.
Backward grids are flattened to one dimension to avoid a large column count
exceeding a secondary grid-dimension limit. Programs own disjoint output ranges;
partial reduction runs afterwards on the same current stream, with no atomics
or inter-program spin barriers.

General broadcasts retain `_BiasGelu`: the generic forward/elementwise backward
kernels plus Autograd sum-to-shape. They reuse the same derivative helper. Empty
inputs and inference do not allocate the derivative checkpoint. Scalar biases
and general broadcast shapes retain this fallback. Noncontiguous vector inputs
are materialized before entering the shared checkpoint path.

The checkpoint costs **4*N bytes** during training. Larger backward workspaces
cost `4*ceil(rows/BR)*hidden` bytes. At 16384x4096, the checkpoint is 256 MiB and
the 32-row partial workspace is 8 MiB (previously 32 MiB with eight-row segments).
Single-row bias gradients reuse a view of dInput and need no bias workspace.
This memory/time tradeoff must be considered for large training activations.

## Shape policy and launch policy

The host shape policy is separate from kernel mathematics. Current defaults were
measured on H20; they are starting points, not proven optimal foreign-device
settings. All launches use the abstract Triton `num_warps=4` interface, with no
hardcoded NVIDIA SM count, lane shuffle or warp-specific device instructions.

| Case | Shared backward | BR | BC | Launches |
|---|---|---:|---:|---:|
| One row / input-gradient-only | Elementwise loop | 1 | 1024 | 1 |
| N<=65536, rows<=128 | Full local row reduction | next power of two(rows) | 64 | 1 |
| N<=1048576, rows>128 | Streaming column reduction | 256 | 8 | 1 |
| Other vector bias | Partial row reduction | 32 if rows>=1024, otherwise 8 | 256 for FP32, 512 otherwise | 2 |

Training forward uses BLOCK=512 for N>=1048576 and 256 otherwise. General
broadcast forward retains BLOCK=256. Further per-backend tuning can change the
host policy while retaining these same mathematical kernels.

NVIDIA uses a bounded cache of plain compiled Triton kernels, keyed by device,
dtypes, constants and alignment. It never caches tensors or stream handles.
Wrapped kernels retain framework dispatch. Generic launches use LibEntry's own
backend-aware key and authoritative per-device kernel cache; this preserves
backend specialization such as Hygon pointer/buffer rules and cache invalidation.
For plain JIT kernels with the repository's full-argument protocol (Triton
3.3–3.6), a local shortcut avoids rebuilding general argument dictionaries.
Cache misses, pre-hooks, tuners/heuristics and other argument protocols use full
LibEntry dispatch. The shared `libentry.py` is unchanged. The backend rebinds its
kernel launcher per call and resolves the current stream; foreign backends are
never sent through the NVIDIA alignment-only cache or CUDA device shortcuts.

## Reproduction on H20

```bash
export CUDA_VISIBLE_DEVICES=1
python -m pytest tests/megatron/test_bias_gelu_impl.py \
  tests/megatron/test_bias_gelu_triton_regression.py -q -rs --tb=short
python tools/check_bias_gelu_generic.py
python tools/bench_bias_gelu_h20.py --implementation default --rounds 3 \
  --output /workspace/flagtrain_results/codex_perf/backward_margin/final_native
python tools/bench_bias_gelu_h20.py --implementation generic --rounds 3 \
  --output /workspace/flagtrain_results/codex_perf/backward_margin/final_generic
```

`--implementation generic` is an explicit **diagnostic-process-only** selection
from `tools/bias_gelu_paths.py`. It temporarily selects the real generic launch
branches through module bindings, then restores them; it is not a thread-safe
production dispatch API. Production code never imports this helper or detects
benchmark names/environment. The public interface and shared mathematical kernels
remain in the measured path. H20 generic results still use the H20 compiler and
device and do not establish Ascend/MetaX/Iluvatar/Hygon/T-Head performance.

Both paths use identical seeded inputs and grad_output, the same PyTorch
`F.gelu(x+b, approximate="tanh")` baseline and full `autograd.grad` including both
gradients. Ordinary `do_bench` uses warmup=25ms, rep=100ms and median latency;
backward graph construction is outside timing. CSV/JSON retain all rounds and
absolute delays. Forward/backward are measured separately. The generic path uses
the same checkpoint and reduction policy, including workspace allocation.

The repository's earlier acceptance floor remains direction geometric mean
>=0.80; the portable-backward engineering target is generic backward >=0.80 on
H20 and NVIDIA backward >=1.00 where possible. Per-case slowdowns are reported.
All measurements and failed experiments are stored outside the source tree in
`/workspace/flagtrain_results/codex_perf/backward_margin/`.

Foreign-device compiler support for layouts, FP32 accumulation, tanh accuracy,
grid limits, register/shared-memory resources and launch protocols still requires
real hardware tests. No foreign platform has been performance-certified here.

The extended large-row accuracy regression keeps 2e-5 tolerances and per-element
FP32 rounding, but uses FP64 **reference** accumulation (CPU when the backend has
no FP64). The original 102 correctness tests and performance baseline are not
modified. A plain FP32 reference sum at 16384 rows was observed to differ from
FP64 accumulation by up to 6.1e-5; that reference error is recorded separately
from operator error. Infinity/NaN upstream gradients have explicit regressions.
