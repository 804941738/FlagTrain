// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

template <typename T, bool Backward>
__device__ __forceinline__ T evaluate(T x, T b, T g) {
  float z = float(T(float(x) + float(b)));
  float v = Backward ? fminf(fmaxf(z, -10.f), 10.f) : z;
  float u = 0.79788456f * v * (1.f + 0.044715f * v * v);
  float t = tanhf(u);
  float value;
  if (Backward) {
    float d = 0.5f * (1.f + t) +
              0.5f * v * (1.f - t * t) * (0.79788456f + 0.1070322243f * v * v);
    if (!isfinite(z))
      d = NAN;
    value = float(g) * d;
  } else {
    value = 0.5f * z * (1.f + t);
  }
  return T(value);
}

template <typename T> struct alignas(sizeof(T) * 4) Vec { T v[4]; };

template <typename T, bool Backward, bool Vector>
__global__ void bias_gelu_kernel(const T *x, const T *b, const T *g, T *out,
                                 int n, int h) {
  constexpr int width = Vector ? 4 : 1;
  int count = n / width;
  int columns = h / width;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += int64_t(blockDim.x) * gridDim.x) {
    int bi = (columns & (columns - 1)) == 0 ? (int(i) & (columns - 1))
                                            : (int(i) % columns);
    if (Vector) {
      auto xv = reinterpret_cast<const Vec<T> *>(x)[i];
      auto bv = reinterpret_cast<const Vec<T> *>(b)[bi];
      Vec<T> gv, result;
      if (Backward)
        gv = reinterpret_cast<const Vec<T> *>(g)[i];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        result.v[j] =
            evaluate<T, Backward>(xv.v[j], bv.v[j], Backward ? gv.v[j] : T(0));
      reinterpret_cast<Vec<T> *>(out)[i] = result;
    } else {
      out[i] = evaluate<T, Backward>(x[i], b[bi], Backward ? g[i] : T(0));
    }
  }
}

template <typename T, int Rows>
__global__ void fused_backward(const T *x, const T *b, const T *g, T *out,
                               float *partial, int rows, int h) {
  int col = blockIdx.x * 512 + threadIdx.x * 4;
  if (col >= h)
    return;
  auto bv = *reinterpret_cast<const Vec<T> *>(b + col);
  float accum[4] = {0, 0, 0, 0};
#pragma unroll 1
  for (int r = 0; r < Rows; ++r) {
    int row = blockIdx.y * Rows + r;
    if (row < rows) {
      int offset = row * h + col;
      auto xv = *reinterpret_cast<const Vec<T> *>(x + offset);
      auto gv = *reinterpret_cast<const Vec<T> *>(g + offset);
      Vec<T> result;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        result.v[j] = evaluate<T, true>(xv.v[j], bv.v[j], gv.v[j]);
        accum[j] += float(result.v[j]);
      }
      *reinterpret_cast<Vec<T> *>(out + offset) = result;
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j)
    partial[blockIdx.y * h + col + j] = accum[j];
}

template <typename T>
__global__ void reduce_partials(const float *partial, T *db, int parts, int h) {
  __shared__ float scratch[8][32];
  int col = blockIdx.x * 32 + threadIdx.x;
  float value = 0.f;
  for (int r = threadIdx.y; r < parts; r += 8)
    if (col < h)
      value += partial[r * h + col];
  scratch[threadIdx.y][threadIdx.x] = value;
  __syncthreads();
  if (threadIdx.y == 0 && col < h) {
    float total = 0.f;
#pragma unroll
    for (int r = 0; r < 8; ++r)
      total += scratch[r][threadIdx.x];
    db[col] = T(total);
  }
}

std::vector<at::Tensor> bias_gelu_fused_grad_cuda(at::Tensor x, at::Tensor b,
                                                  at::Tensor g) {
  const c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty_like(x);
  int h = b.numel(), rows = x.numel() / h;
  int tile_rows = rows <= 512 ? 8 : 16;
  int parts = (rows + tile_rows - 1) / tile_rows;
  auto partial = at::empty({parts, h}, x.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
      "bias_gelu_fused_grad", [&] {
        if (tile_rows == 8)
          fused_backward<scalar_t, 8>
              <<<dim3((h + 511) / 512, parts), 128, 0, stream>>>(
                  x.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
                  g.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                  partial.data_ptr<float>(), rows, h);
        else
          fused_backward<scalar_t, 16>
              <<<dim3((h + 511) / 512, parts), 128, 0, stream>>>(
                  x.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
                  g.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                  partial.data_ptr<float>(), rows, h);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto db = at::empty_like(b);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
      "bias_gelu_reduce", [&] {
        reduce_partials<scalar_t><<<(h + 31) / 32, dim3(32, 8), 0, stream>>>(
            partial.data_ptr<float>(), db.data_ptr<scalar_t>(), parts, h);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, db};
}

template <bool Backward>
at::Tensor launch(at::Tensor x, at::Tensor b, at::Tensor g) {
  const c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty_like(x);
  auto n = x.numel();
  if (n) {
    bool vector = n % 4 == 0 && b.numel() % 4 == 0 &&
                  uintptr_t(x.data_ptr()) % 16 == 0 &&
                  uintptr_t(b.data_ptr()) % 16 == 0 &&
                  (!Backward || uintptr_t(g.data_ptr()) % 16 == 0);
    int blocks = std::min<int64_t>((n / (vector ? 4 : 1) + 255) / 256, 65535);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
        "bias_gelu", [&] {
          auto go = Backward ? g.data_ptr<scalar_t>() : nullptr;
          if (vector)
            bias_gelu_kernel<scalar_t, Backward, true>
                <<<blocks, 256, 0, stream>>>(
                    x.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), go,
                    out.data_ptr<scalar_t>(), n, b.numel());
          else
            bias_gelu_kernel<scalar_t, Backward, false>
                <<<blocks, 256, 0, stream>>>(
                    x.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), go,
                    out.data_ptr<scalar_t>(), n, b.numel());
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

at::Tensor bias_gelu_cuda(at::Tensor x, at::Tensor b) {
  return launch<false>(x, b, at::Tensor());
}
at::Tensor bias_gelu_grad_cuda(at::Tensor x, at::Tensor b, at::Tensor g) {
  return launch<true>(x, b, g);
}
