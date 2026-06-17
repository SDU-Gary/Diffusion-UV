#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

namespace {

constexpr uint32_t PRIME_X = 73856093u;
constexpr uint32_t PRIME_Y = 19349663u;
constexpr uint32_t PRIME_Z = 83492791u;

__device__ __forceinline__ uint32_t positive_mod_hash(
    int ix, int iy, int iz, uint32_t table_size) {
  uint32_t x = static_cast<uint32_t>(ix) * PRIME_X;
  uint32_t y = static_cast<uint32_t>(iy) * PRIME_Y;
  uint32_t z = static_cast<uint32_t>(iz) * PRIME_Z;
  return (x ^ y ^ z) % table_size;
}

__device__ __forceinline__ void bspline_weights(float t, float* w) {
  float t2 = t * t;
  float t3 = t2 * t;
  float omt = 1.0f - t;
  w[0] = (omt * omt * omt) * (1.0f / 6.0f);
  w[1] = (3.0f * t3 - 6.0f * t2 + 4.0f) * (1.0f / 6.0f);
  w[2] = (-3.0f * t3 + 3.0f * t2 + 3.0f * t + 1.0f) * (1.0f / 6.0f);
  w[3] = t3 * (1.0f / 6.0f);
}

__device__ __forceinline__ float level_resolution(
    int level, int num_levels, int base_res, int max_res) {
  if (num_levels <= 1) {
    return static_cast<float>(base_res);
  }
  float ratio = static_cast<float>(max_res) / static_cast<float>(base_res);
  float exponent = static_cast<float>(level) / static_cast<float>(num_levels - 1);
  return static_cast<float>(base_res) * powf(ratio, exponent);
}

__global__ void bspline_hash_forward_kernel(
    const float* __restrict__ positions,
    const float* __restrict__ hash_table,
    float* __restrict__ output,
    int N,
    int L,
    int T,
    int F,
    int base_res,
    int max_res) {
  int linear = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * L * F;
  if (linear >= total) {
    return;
  }

  int f = linear % F;
  int level = (linear / F) % L;
  int n = linear / (L * F);

  const float* p = positions + n * 3;
  float res = level_resolution(level, L, base_res, max_res);
  float x = fminf(fmaxf(p[0], 0.0f), 1.0f) * res;
  float y = fminf(fmaxf(p[1], 0.0f), 1.0f) * res;
  float z = fminf(fmaxf(p[2], 0.0f), 1.0f) * res;

  int ix = static_cast<int>(floorf(x));
  int iy = static_cast<int>(floorf(y));
  int iz = static_cast<int>(floorf(z));
  float tx = x - static_cast<float>(ix);
  float ty = y - static_cast<float>(iy);
  float tz = z - static_cast<float>(iz);

  float wx[4], wy[4], wz[4];
  bspline_weights(tx, wx);
  bspline_weights(ty, wy);
  bspline_weights(tz, wz);

  float acc = 0.0f;
  for (int dx = 0; dx < 4; ++dx) {
    int gx = ix + dx - 1;
    for (int dy = 0; dy < 4; ++dy) {
      int gy = iy + dy - 1;
      float wxy = wx[dx] * wy[dy];
      for (int dz = 0; dz < 4; ++dz) {
        int gz = iz + dz - 1;
        float w = wxy * wz[dz];
        uint32_t h = positive_mod_hash(gx, gy, gz, static_cast<uint32_t>(T));
        int table_index = ((level * T + static_cast<int>(h)) * F) + f;
        acc += w * hash_table[table_index];
      }
    }
  }

  output[linear] = acc;
}

__global__ void bspline_hash_backward_hash_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ positions,
    float* __restrict__ grad_hash_table,
    int N,
    int L,
    int T,
    int F,
    int base_res,
    int max_res) {
  int linear = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * L * F;
  if (linear >= total) {
    return;
  }

  int f = linear % F;
  int level = (linear / F) % L;
  int n = linear / (L * F);
  float go = grad_output[linear];

  const float* p = positions + n * 3;
  float res = level_resolution(level, L, base_res, max_res);
  float x = fminf(fmaxf(p[0], 0.0f), 1.0f) * res;
  float y = fminf(fmaxf(p[1], 0.0f), 1.0f) * res;
  float z = fminf(fmaxf(p[2], 0.0f), 1.0f) * res;

  int ix = static_cast<int>(floorf(x));
  int iy = static_cast<int>(floorf(y));
  int iz = static_cast<int>(floorf(z));
  float tx = x - static_cast<float>(ix);
  float ty = y - static_cast<float>(iy);
  float tz = z - static_cast<float>(iz);

  float wx[4], wy[4], wz[4];
  bspline_weights(tx, wx);
  bspline_weights(ty, wy);
  bspline_weights(tz, wz);

  for (int dx = 0; dx < 4; ++dx) {
    int gx = ix + dx - 1;
    for (int dy = 0; dy < 4; ++dy) {
      int gy = iy + dy - 1;
      float wxy = wx[dx] * wy[dy];
      for (int dz = 0; dz < 4; ++dz) {
        int gz = iz + dz - 1;
        float w = wxy * wz[dz];
        uint32_t h = positive_mod_hash(gx, gy, gz, static_cast<uint32_t>(T));
        int table_index = ((level * T + static_cast<int>(h)) * F) + f;
        atomicAdd(grad_hash_table + table_index, go * w);
      }
    }
  }
}

}  // namespace

torch::Tensor bspline_hash_forward_cuda(
    torch::Tensor positions,
    torch::Tensor hash_table,
    int64_t base_res,
    int64_t max_res) {
  int N = static_cast<int>(positions.size(0));
  int L = static_cast<int>(hash_table.size(0));
  int T = static_cast<int>(hash_table.size(1));
  int F = static_cast<int>(hash_table.size(2));
  auto output = torch::empty({N, L * F}, positions.options());

  int total = N * L * F;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  bspline_hash_forward_kernel<<<blocks, threads>>>(
      positions.data_ptr<float>(),
      hash_table.data_ptr<float>(),
      output.data_ptr<float>(),
      N, L, T, F,
      static_cast<int>(base_res),
      static_cast<int>(max_res));

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor bspline_hash_backward_hash_cuda(
    torch::Tensor grad_output,
    torch::Tensor positions,
    std::vector<int64_t> hash_table_shape,
    int64_t base_res,
    int64_t max_res) {
  int N = static_cast<int>(positions.size(0));
  int L = static_cast<int>(hash_table_shape[0]);
  int T = static_cast<int>(hash_table_shape[1]);
  int F = static_cast<int>(hash_table_shape[2]);
  auto grad_hash_table = torch::zeros({L, T, F}, grad_output.options());

  int total = N * L * F;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  bspline_hash_backward_hash_kernel<<<blocks, threads>>>(
      grad_output.data_ptr<float>(),
      positions.data_ptr<float>(),
      grad_hash_table.data_ptr<float>(),
      N, L, T, F,
      static_cast<int>(base_res),
      static_cast<int>(max_res));

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_hash_table;
}
