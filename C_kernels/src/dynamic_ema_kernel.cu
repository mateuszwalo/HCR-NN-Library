#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

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


torch::Tensor ema_update_cuda(
    torch::Tensor a,   // (D,D,D)
    torch::Tensor x,   // (D,)
    torch::Tensor y,   // (D,)
    torch::Tensor z,   // (D,)
    float ema_lambda
) {
    int D = x.size(0);
    int total = D * D * D;

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    ema_update_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        z.data_ptr<float>(),
        a.data_ptr<float>(),
        ema_lambda,
        D
    );

    return a;
}
