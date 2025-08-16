#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

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

void bilinear_cuda() {
    
}