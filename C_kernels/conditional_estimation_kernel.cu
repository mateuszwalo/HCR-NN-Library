#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

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

torch::Tensor conditional_estimation_cuda(
    torch::Tensor x_candidates, // (N, D)
    torch::Tensor fy,           // (D,)
    torch::Tensor fz,           // (D,)
    torch::Tensor a             // (D, D, D)
) {
    int N = x_candidates.size(0);
    int D = fy.size(0);

    auto denom = torch::zeros({1}, fy.options());
    auto context = torch::zeros({D}, fy.options());
    auto scores = torch::zeros({N}, fy.options());

    // Launch denominator kernel
    dim3 threads1(16, 16);
    dim3 blocks1((D + 15)/16, (D + 15)/16);
    denominator_kernel<<<blocks1, threads1>>>(
        a.data_ptr<float>(), fy.data_ptr<float>(), fz.data_ptr<float>(),
        denom.data_ptr<float>(), D
    );

    // Launch context kernel
    dim3 threads2(D, D);
    dim3 blocks2(D);
    context_kernel<<<blocks2, threads2>>>(
        a.data_ptr<float>(), fy.data_ptr<float>(), fz.data_ptr<float>(),
        context.data_ptr<float>(), D
    );

    // Normalize context by denominator
    context /= denom + 1e-8;

    // Compute scores = X @ context
    scores = torch::matmul(x_candidates, context);

    return scores;
}
