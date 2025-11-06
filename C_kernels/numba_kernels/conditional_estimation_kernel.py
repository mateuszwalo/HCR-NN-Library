import numpy as np
from numba import cuda, float32
import math

@cuda.jit
def denominator_kernel(a, fy, fz, denom, D):
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    k = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if j < D and k < D:
        # Equivalent to: atomicAdd(denom, a[0, j, k] * fy[j] * fz[k])
        cuda.atomic.add(denom, 0, a[0, j, k] * fy[j] * fz[k])

@cuda.jit
def context_kernel(a, fy, fz, context, D):
    i = cuda.blockIdx.x
    j = cuda.threadIdx.y
    k = cuda.threadIdx.x

    partial = cuda.shared.array((32, 32), dtype=float32)

    val = 0.0
    if j < D and k < D:
        val = a[i, j, k] * fy[j] * fz[k]
    partial[j, k] = val

    cuda.syncthreads()

    if j == 0 and k == 0:
        s = 0.0
        for jj in range(D):
            for kk in range(D):
                s += partial[jj, kk]
        context[i] = s

def conditional_estimation(x_candidates, fy, fz, a, return_context=False):
    x_candidates = np.ascontiguousarray(x_candidates.astype(np.float32))
    fy = np.ascontiguousarray(fy.astype(np.float32))
    fz = np.ascontiguousarray(fz.astype(np.float32))
    a = np.ascontiguousarray(a.astype(np.float32))

    N, D = x_candidates.shape[0], fy.shape[0]

    denom = np.zeros(1, dtype=np.float32)
    context = np.zeros(D, dtype=np.float32)

    da = cuda.to_device(a)
    dfy = cuda.to_device(fy)
    dfz = cuda.to_device(fz)
    ddenom = cuda.to_device(denom)
    dcontext = cuda.to_device(context)

    threads1 = (16, 16)
    blocks1 = ((D + 15) // 16, (D + 15) // 16)
    denominator_kernel[blocks1, threads1](da, dfy, dfz, ddenom, D)
    cuda.synchronize()

    threads2 = (D, D)
    blocks2 = D
    context_kernel[blocks2, threads2](da, dfy, dfz, dcontext, D)
    cuda.synchronize()

    denom = ddenom.copy_to_host()[0]
    context = dcontext.copy_to_host().reshape(-1)
    context = context / (denom + 1e-8)

    scores = x_candidates @ context  # (N, D) @ (D,) -> (N,)

    if return_context:
        return scores, context
    else:
        return scores

