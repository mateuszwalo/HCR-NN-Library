#define MYCUDA_EXPORTS
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "cuda_header.h"

__global__ void mean_estimation_kernel(
    const float* __restrict__ fx,
    const float* __restrict__ fy,
    const float* __restrict__ fz,
    float* __restrict__ out,
    int D,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = D * D * D;

    if (idx >= total) return;

    int i = idx / (D * D);
    int j = (idx / D) % D;
    int k = idx % D;

    float sum = 0.0f;

    for (int n = 0; n < N; n++) {
        float xi = fx[n * D + i];
        float yj = fy[n * D + j];
        float zk = fz[n * D + k];
        sum += xi * yj * zk;
    }
    
    // normalize by number of triplets
    out[idx] = sum / N;
}

extern "C" void launch_mean_estimation_kernel(
    const float*  fx,
    const float*  fy,
    const float*  fz,
    float*  out,
    int D,
    int N
){
    int total = D * D * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    mean_estimation_kernel<<<blocks, threads>>>(fx, fy, fz, out, D, N);
    cudaDeviceSynchronize();
}