#define MYCUDA_EXPORTS
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_header.h"

// EMA kernel: update each a[i,j,k]
__global__ void ema_update_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ z,
    float* __restrict__ a,
    float ema_lambda,
    int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = D * D * D;

    if (idx >= total) return;

    int i = idx / (D * D);
    int j = (idx / D) % D;
    int k = idx % D;

    float update_val = x[i] * y[j] * z[k];
    float old_val = a[idx];
    a[idx] = (1.0f - ema_lambda) * old_val + ema_lambda * update_val;
}

extern "C" void launch_ema_update_kernel(
    const float* x,
    const float* y,
    const float* z,
    float* a,
    float ema_lambda,
    int D
) {
    int total = D * D * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    ema_update_kernel<<<blocks, threads>>>(x, y, z, a, ema_lambda, D);
    cudaDeviceSynchronize();
}
