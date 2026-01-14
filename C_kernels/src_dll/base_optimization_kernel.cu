#define MYCUDA_EXPORTS
#include "cuda_header.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Kernel for new_a = einsum('li, ljk -> ijk', U^T, a)
// U: [D, D], a: [D, D, D], new_a: [D, D, D]
__global__ void transform_tensor_kernel(
    const float* __restrict__ U,
    const float* __restrict__ a,
    float* __restrict__ new_a,
    int D) 
{
    int i = blockIdx.x;   // i index in output
    int j = threadIdx.y;  // j index in output
    int k = threadIdx.x;  // k index in output

    if (i < D && j < D && k < D) {
        float val = 0.0f;
        for (int l = 0; l < D; l++) {
            val += U[l * D + i] * a[l * D * D + j * D + k];
        }
        new_a[i * D * D + j * D + k] = val;
    }
}

extern "C" void launch_transform_tensor_kernel(
    const float* U,
    const float* a,
    float* new_a,
    int D)
{
    dim3 threads(D, D);
    dim3 blocks(D);
    transform_tensor_kernel<<<blocks, threads>>>(U, a, new_a, D);
    cudaDeviceSynchronize();
}