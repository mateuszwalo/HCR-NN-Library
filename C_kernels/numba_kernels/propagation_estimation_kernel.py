import numpy as np
from numba import cuda, float32
import math
import torch

@cuda.jit
def bilinear_kernel(a0, a1, fy, fz, denom, num, D):
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    k = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if j < D and k < D:
        val_yz = fy[j] * fz[k]
        cuda.atomic.add(denom, 0, a0[j, k] * val_yz)
        cuda.atomic.add(num,   0, a1[j, k] * val_yz)

def propagate_expectation(a, fy, fz):
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(fy, torch.Tensor):
        fy = fy.detach().cpu().numpy()
    if isinstance(fz, torch.Tensor):
        fz = fz.detach().cpu().numpy()

    a = np.ascontiguousarray(a.astype(np.float32))
    fy = np.ascontiguousarray(fy.astype(np.float32))
    fz = np.ascontiguousarray(fz.astype(np.float32))

    D = fy.shape[0]

    a0 = a[0].reshape(D, D)
    a1 = a[1].reshape(D, D)

    denom = np.zeros(1, dtype=np.float32)
    num = np.zeros(1, dtype=np.float32)

    d_a0 = cuda.to_device(a0)
    d_a1 = cuda.to_device(a1)
    d_fy = cuda.to_device(fy)
    d_fz = cuda.to_device(fz)
    d_denom = cuda.to_device(denom)
    d_num = cuda.to_device(num)

    threads = (16, 16)
    blocks = ((D + 15) // 16, (D + 15) // 16)

    bilinear_kernel[blocks, threads](d_a0, d_a1, d_fy, d_fz, d_denom, d_num, D)
    cuda.synchronize()

    denom = d_denom.copy_to_host()[0]
    num = d_num.copy_to_host()[0]

    ratio = num / (denom + 1e-8)
    const_val = math.sqrt(3.0)
    propagated = 0.5 + (0.5 / const_val) * (ratio - 1.0)

    return np.float32(propagated)
