import numpy as np
from numba import cuda, float32

@cuda.jit
def mean_estimation_kernel(fx, fy, fz, out, D, N):
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    total = D * D * D
    if idx >= total:
        return

    i = idx // (D * D)
    j = (idx // D) % D
    k = idx % D

    sum_val = 0.0
    for n in range(N):
        xi = fx[n, i]
        yj = fy[n, j]
        zk = fz[n, k]
        sum_val += xi * yj * zk

    out[i, j, k] = sum_val / N

def mean_estimation(fx, fy, fz):

    fx = np.ascontiguousarray(fx.astype(np.float32))
    fy = np.ascontiguousarray(fy.astype(np.float32))
    fz = np.ascontiguousarray(fz.astype(np.float32))

    N, D = fx.shape
    out = np.zeros((D, D, D), dtype=np.float32)

    d_fx = cuda.to_device(fx)
    d_fy = cuda.to_device(fy)
    d_fz = cuda.to_device(fz)
    d_out = cuda.to_device(out)

    threads_per_block = 256
    total = D * D * D
    blocks_per_grid = (total + threads_per_block - 1) // threads_per_block

    mean_estimation_kernel[blocks_per_grid, threads_per_block](d_fx, d_fy, d_fz, d_out, D, N)
    cuda.synchronize()

    return d_out.copy_to_host()

if __name__ == "__main__":
    N, D = 8, 4
    fx = np.random.rand(N, D).astype(np.float32)
    fy = np.random.rand(N, D).astype(np.float32)
    fz = np.random.rand(N, D).astype(np.float32)

    out = mean_estimation(fx, fy, fz)

    ref = np.einsum('ni,nj,nk->ijk', fx, fy, fz) / N
    print("Max abs error:", np.max(np.abs(out - ref)))
