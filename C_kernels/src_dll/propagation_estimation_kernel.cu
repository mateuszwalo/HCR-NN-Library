#define MYCUDA_EXPORTS
#include "cuda_header.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

__global__ void bilinear_kernel(
    const float* __restrict__ a0,
    const float* __restrict__ a1,
    const float* __restrict__ fy,
    const float* __restrict__ fz,
    float* denom,
    float* num,
    int D
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < D && k < D) {
        float val_yz = fy[j] * fz[k];
        atomicAdd(denom, a0[j * D + k] * val_yz);
        atomicAdd(num,   a1[j * D + k] * val_yz);
    }
}

extern "C" void launch_bilinear_kernel(
    const float*  a0,
    const float*  a1,
    const float*  fy,
    const float*  fz,
    float* denom,
    float* num,
    int D
) {
    dim3 threads(16, 16);
    dim3 blocks((D + 15) / 16, (D + 15) / 16);

    bilinear_kernel<<<blocks, threads>>>(a0, a1, fy, fz, denom, num, D);
    cudaDeviceSynchronize();
}