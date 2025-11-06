import numpy as np
from numba import cuda, float32
import math
import torch
import torch.nn as nn

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


@cuda.jit
def entropy_kernel(activations, entropy_out, B, D):
    b = cuda.blockIdx.x
    if b >= B:
        return

    max_val = -1e20
    for d in range(D):
        val = activations[b, d]
        if val > max_val:
            max_val = val

    sum_exp = 0.0
    for d in range(D):
        sum_exp += math.exp(activations[b, d] - max_val)

    sum_sq = 0.0
    for d in range(D):
        p = math.exp(activations[b, d] - max_val) / sum_exp
        sum_sq += p * p

    entropy_out[b] = -sum_sq

@cuda.jit
def mi_kernel(actX, actY, mi_out, B, D):
    b = cuda.blockIdx.x
    if b >= B:
        return

    max_x = -1e20
    for d in range(D):
        val = actX[b, d]
        if val > max_x:
            max_x = val

    sum_exp_x = 0.0
    for d in range(D):
        sum_exp_x += math.exp(actX[b, d] - max_x)

    max_y = -1e20
    for d in range(D):
        val = actY[b, d]
        if val > max_y:
            max_y = val

    sum_exp_y = 0.0
    for d in range(D):
        sum_exp_y += math.exp(actY[b, d] - max_y)

    sum_sq_x = 0.0
    sum_sq_y = 0.0
    for d in range(D):
        px = math.exp(actX[b, d] - max_x) / sum_exp_x
        py = math.exp(actY[b, d] - max_y) / sum_exp_y
        sum_sq_x += px * px
        sum_sq_y += py * py

    mi_out[b] = sum_sq_x * sum_sq_y

def approximate_entropy(activations):

    if isinstance(activations, torch.Tensor):
        activations = activations.detach().cpu().numpy()

    activations = np.ascontiguousarray(activations.astype(np.float32))
    B, D = activations.shape
    entropy_out = np.zeros(B, dtype=np.float32)

    d_act = cuda.to_device(activations)
    d_entropy = cuda.to_device(entropy_out)

    entropy_kernel[B, 1](d_act, d_entropy, B, D)
    cuda.synchronize()

    entropy_out = d_entropy.copy_to_host()
    return float(np.mean(entropy_out))


def approximate_mi(actX, actY):

    if isinstance(actX, torch.Tensor):
        actX = actX.detach().cpu().numpy()
    if isinstance(actY, torch.Tensor):
        actY = actY.detach().cpu().numpy()

    actX = np.ascontiguousarray(actX.astype(np.float32))
    actY = np.ascontiguousarray(actY.astype(np.float32))

    B, D = actX.shape
    mi_out = np.zeros(B, dtype=np.float32)

    dX = cuda.to_device(actX)
    dY = cuda.to_device(actY)
    d_mi = cuda.to_device(mi_out)

    mi_kernel[B, 1](dX, dY, d_mi, B, D)
    cuda.synchronize()

    mi_out = d_mi.copy_to_host()
    return float(np.mean(mi_out))

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

def mean_estimation(fx, fy, fz, D=None):

    fx = np.ascontiguousarray(fx.astype(np.float32))
    fy = np.ascontiguousarray(fy.astype(np.float32))
    fz = np.ascontiguousarray(fz.astype(np.float32))

    N, D = fx.shape if D is None else (fx.shape[0], D)
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
