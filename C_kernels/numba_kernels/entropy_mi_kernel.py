import numpy as np
from numba import cuda, float32
import math, torch

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
