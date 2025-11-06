import numpy as np
from numba import cuda, float32

@cuda.jit
def transform_tensor_kernel(U, a, new_a, D):
    i = cuda.blockIdx.x
    j = cuda.threadIdx.y
    k = cuda.threadIdx.x

    if i < D and j < D and k < D:
        val = 0.0
        for l in range(D):
            val += U[l, i] * a[l, j, k]
        new_a[i, j, k] = val


def transform_tensor(U, a):
    U = np.ascontiguousarray(U.astype(np.float32))
    a = np.ascontiguousarray(a.astype(np.float32))

    D = U.shape[0]
    new_a = np.zeros_like(a)

    dU = cuda.to_device(U)
    da = cuda.to_device(a)
    dnew_a = cuda.device_array_like(a)

    threadsperblock = (D, D)   # (k, j)
    blockspergrid = D          # i
    
    transform_tensor_kernel[blockspergrid, threadsperblock](dU, da, dnew_a, D)
    cuda.synchronize()

    new_a = dnew_a.copy_to_host()
    return new_a
