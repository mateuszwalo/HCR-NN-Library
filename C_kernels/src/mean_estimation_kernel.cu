#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel: each thread computes one element A[i,j,k]
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

void mean_estimation_cuda(
    torch::Tensor fx,
    torch::Tensor fy,
    torch::Tensor fz,
    torch::Tensor out
) {
    int D = fx.size(1);
    int N = fx.size(0);
    int total = D * D * D;

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    mean_estimation_kernel<<<blocks, threads>>>(
        fx.data_ptr<float>(),
        fy.data_ptr<float>(),
        fz.data_ptr<float>(),
        out.data_ptr<float>(),
        D, N
    );
}