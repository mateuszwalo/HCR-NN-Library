import numpy as np
from numba import cuda, float32

@cuda.jit
def ema_update_kernel(x, y, z, a, ema_lambda, D):
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    total = D * D * D

    if idx >= total:
        return

    i = idx // (D * D)
    j = (idx // D) % D
    k = idx % D

    update_val = x[i] * y[j] * z[k]
    old_val = a[i, j, k]
    a[i, j, k] = (1.0 - ema_lambda) * old_val + ema_lambda * update_val

def ema_update(a, x, y, z, ema_lambda):

    a = np.ascontiguousarray(a.astype(np.float32))
    x = np.ascontiguousarray(x.astype(np.float32))
    y = np.ascontiguousarray(y.astype(np.float32))
    z = np.ascontiguousarray(z.astype(np.float32))

    D = x.shape[0]
    total = D * D * D

    d_a = cuda.to_device(a)
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_z = cuda.to_device(z)

    threads_per_block = 256
    blocks_per_grid = (total + threads_per_block - 1) // threads_per_block

    ema_update_kernel[blocks_per_grid, threads_per_block](d_x, d_y, d_z, d_a, ema_lambda, D)
    cuda.synchronize()

    return d_a.copy_to_host()


if __name__ == "__main__":
    D = 4
    ema_lambda = 0.1

    a = np.random.rand(D, D, D).astype(np.float32)
    x = np.random.rand(D).astype(np.float32)
    y = np.random.rand(D).astype(np.float32)
    z = np.random.rand(D).astype(np.float32)

    new_a = ema_update(a, x, y, z, ema_lambda)

    ref = (1 - ema_lambda) * a + ema_lambda * np.einsum('i,j,k->ijk', x, y, z)
    print("max abs error:", np.max(np.abs(new_a - ref)))
