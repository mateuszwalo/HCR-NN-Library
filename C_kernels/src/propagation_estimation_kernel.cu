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

torch::Tensor propagate_expectation_cuda(
    torch::Tensor a,   // (2, D, D)
    torch::Tensor fy,  // (D,)
    torch::Tensor fz   // (D,)
) {
    int D = fy.size(0);

    auto denom = torch::zeros({1}, fy.options());
    auto num   = torch::zeros({1}, fy.options());

    dim3 threads(16, 16);
    dim3 blocks((D + 15) / 16, (D + 15) / 16);

    bilinear_kernel<<<blocks, threads>>>(
        a[0].data_ptr<float>(),
        a[1].data_ptr<float>(),
        fy.data_ptr<float>(),
        fz.data_ptr<float>(),
        denom.data_ptr<float>(),
        num.data_ptr<float>(),
        D
    );

    torch::Tensor ratio = num / (denom + 1e-8);

    torch::Tensor const_val = torch::sqrt(torch::tensor(3.0, fy.options()));
    torch::Tensor propagated = 0.5 + (0.5 / const_val) * (ratio - 1.0);

    return propagated;
}
