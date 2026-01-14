#define MYCUDA_EXPORTS
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "cuda_header.h"

// Kernel to compute denominator (i=0 slice)
__global__ void denominator_kernel(
    const float* __restrict__ a,
    const float* __restrict__ fy,
    const float* __restrict__ fz,
    float* denom,
    int D
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < D && k < D) {
        atomicAdd(denom, a[0 * D * D + j * D + k] * fy[j] * fz[k]);
    }
}

// Kernel to compute context[i] for all i
__global__ void context_kernel(
    const float* __restrict__ a,
    const float* __restrict__ fy,
    const float* __restrict__ fz,
    float* context,
    int D
) {
    int i = blockIdx.x;
    int j = threadIdx.y;
    int k = threadIdx.x;

    __shared__ float partial[32][32]; // assuming D <= 32 (can generalize)

    float val = 0.0f;
    if (j < D && k < D) {
        val = a[i * D * D + j * D + k] * fy[j] * fz[k];
    }
    partial[j][k] = val;

    __syncthreads();

    // Reduction along j,k
    if (j == 0 && k == 0) {
        float sum = 0.0f;
        for (int jj = 0; jj < D; jj++) {
            for (int kk = 0; kk < D; kk++) {
                sum += partial[jj][kk];
            }
        }
        context[i] = sum;
    }
}

extern "C" void launch_context_kernel(
    const float* a,
    const float* fy,
    const float* fz,
    float* context,
    int D
) {
    dim3 threads(D, D);
    dim3 blocks(D);
    context_kernel<<<blocks, threads>>>(a, fy, fz, context, D);
    cudaDeviceSynchronize();
}

extern "C" void launch_denominator_kernel(
    const float* a,
    const float* fy,
    const float* fz,
    float* denom,
    int D
) {
    dim3 threads(16, 16);
    dim3 blocks((D + 15)/16, (D + 15)/16);
    denominator_kernel<<<blocks, threads>>>(a, fy, fz, denom, D);
    cudaDeviceSynchronize();
}