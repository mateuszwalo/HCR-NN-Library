#define MYCUDA_EXPORTS
#include "cuda_header.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// Entropy kernel: compute -mean(sum(p^2))
__global__ void entropy_kernel(
    const float* __restrict__ activations, // (B, D)
    float* __restrict__ entropy_out,
    int B, int D
) {
    int b = blockIdx.x;
    if (b >= B) return;

    // compute softmax denominator
    float max_val = -1e20f;
    for (int d = 0; d < D; d++) {
        max_val = fmaxf(max_val, activations[b*D + d]);
    }

    float sum_exp = 0.0f;
    for (int d = 0; d < D; d++) {
        sum_exp += expf(activations[b*D + d] - max_val);
    }

    // compute sum of squared probs
    float sum_sq = 0.0f;
    for (int d = 0; d < D; d++) {
        float p = expf(activations[b*D + d] - max_val) / sum_exp;
        sum_sq += p * p;
    }

    entropy_out[b] = -sum_sq;
}

// Mutual information kernel: avoids outer product
__global__ void mi_kernel(
    const float* __restrict__ actX, // (B, D)
    const float* __restrict__ actY, // (B, D)
    float* __restrict__ mi_out,
    int B, int D
) {
    int b = blockIdx.x;
    if (b >= B) return;

    // softmax normalization for X
    float max_x = -1e20f;
    for (int d = 0; d < D; d++) max_x = fmaxf(max_x, actX[b*D + d]);
    float sum_exp_x = 0.0f;
    for (int d = 0; d < D; d++) sum_exp_x += expf(actX[b*D + d] - max_x);

    // softmax normalization for Y
    float max_y = -1e20f;
    for (int d = 0; d < D; d++) max_y = fmaxf(max_y, actY[b*D + d]);
    float sum_exp_y = 0.0f;
    for (int d = 0; d < D; d++) sum_exp_y += expf(actY[b*D + d] - max_y);

    // compute sum of squared probs for X and Y
    float sum_sq_x = 0.0f, sum_sq_y = 0.0f;
    for (int d = 0; d < D; d++) {
        float px = expf(actX[b*D + d] - max_x) / sum_exp_x;
        float py = expf(actY[b*D + d] - max_y) / sum_exp_y;
        sum_sq_x += px * px;
        sum_sq_y += py * py;
    }

    mi_out[b] = sum_sq_x * sum_sq_y;
}

extern "C" void launch_entropy_kernel(
    const float* activations,
    float* entropy_out,
    int B, int D
) {
    dim3 blocks(B);
    dim3 threads(1);
    entropy_kernel<<<blocks, threads>>>(activations, entropy_out, B, D);
    cudaDeviceSynchronize();
}

extern "C" void launch_mi_kernel(
    const float* actX,
    const float* actY,
    float* mi_out,
    int B, int D
) {
    dim3 blocks(B);
    dim3 threads(1);
    mi_kernel<<<blocks, threads>>>(actX, actY, mi_out, B, D);
    cudaDeviceSynchronize();
}