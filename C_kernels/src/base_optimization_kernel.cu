#include <torch/extension.h>
#include <vector>

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

torch::Tensor transform_tensor_cuda(torch::Tensor U, torch::Tensor a) {
    int D = U.size(0);

    auto new_a = torch::zeros_like(a);

    dim3 blocks(D);
    dim3 threads(D, D);  // (k, j) inside block

    transform_tensor_kernel<<<blocks, threads>>>(
        U.data_ptr<float>(),
        a.data_ptr<float>(),
        new_a.data_ptr<float>(),
        D
    );

    return new_a;
}

