import numpy as np
import math, os
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
import time
import layers_kernel
from typing import Any, Callable, List, Tuple, Literal, Optional
import numba

#Nowsza wersja optymalizacji funkcji składowych HCR.
#Częściowo testowana i gotowa do użytku w pewnym zakresie.

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in {end - start:.4f}s")
        return result
    return wrapper

#Default imports of HCRNN Components.
#Make sure that __init__.py file is correctly opening layers

__all__ = ['CDFNorm', 
           'MeanEstimation', 
           'ConditionalEstimation', 
           'PropagationEstimation', 
           'EntropyAndMutualInformation', 
           'DynamicEMA', 
           'BaseOptimization', 
           'InformationBottleneckLoss']

import torch
from torch import nn

#Layers
class CDFNorm(nn.Module):
    # Using Literal for restricted string options improves IDE autocomplete
    method: Literal['gaussian', 'empirical']
    unbiased: bool
    eps: float
    affine: bool
    track_running_stats: bool
    momentum: float
    
    # Parameters and Buffers are optionally Tensors depending on init flags
    weight: Optional[nn.Parameter]
    bias: Optional[nn.Parameter]
    running_mean: Optional[Tensor]
    running_var: Optional[Tensor]
    num_batches_tracked: Optional[Tensor]

    def __init__(
        self, 
        method: Literal['gaussian', 'empirical'] = 'gaussian', 
        unbiased: bool = True, 
        eps: float = 1e-5, 
        affine: bool = False, 
        track_running_stats: bool = True, 
        momentum: float = 0.1
    ) -> None:
        super().__init__()
        self.method = method
        self.unbiased = unbiased
        self.eps = eps
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.momentum = momentum

        if affine:
            self.weight = nn.Parameter(torch.ones(1))
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(1))
            self.register_buffer('running_var', torch.ones(1))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

    def _gaussian_transform(self, x: Tensor) -> Tensor:
        if self.training and self.track_running_stats:
            # Ensure the buffer exists before usage for strict type checking
            assert self.running_mean is not None and self.running_var is not None
            
            var, mean = torch.var_mean(x, unbiased=self.unbiased)
            with torch.no_grad():
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var, self.momentum)
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked += 1
        else:
            mean, var = self.running_mean, self.running_var

        # Standard deviation and normalization pass
        std: Tensor = torch.sqrt(var + self.eps)
        x_norm: Tensor = 0.5 * (1 + torch.erf((x - mean) / (std * math.sqrt(2.0))))

        if self.affine and self.weight is not None and self.bias is not None:
            return x_norm * self.weight + self.bias
        return x_norm

    def _empirical_transform(self, x: Tensor) -> Tensor:
        # Double argsort provides the rank of each element
        ranks: Tensor = torch.argsort(torch.argsort(x))
        x_norm: Tensor = (ranks.float() + 1) / (len(x) + 1)
        
        if self.affine and self.weight is not None and self.bias is not None:
            return x_norm * self.weight + self.bias
        return x_norm

    def forward(self, x: Tensor) -> Tensor:
        if self.method == 'gaussian':
            return self._gaussian_transform(x)
        elif self.method == 'empirical':
            return self._empirical_transform(x)
        raise ValueError(f"Unsupported normalization method: {self.method}")
    
class MeanEstimationBaseline(nn.Module):
    """
    Learnable third-order moment estimator:
        A = E[ f(x) ⊗ f(y) ⊗ f(z) ]
    """
    def __init__(self, *, feature_fn, feature_dim):
        super().__init__()
        self.feature_fn = feature_fn
        self.feature_dim = feature_dim

    def forward(self, x, y, z):
        """
        x, y, z: tensors [B, input_dim]
        returns: A [D, D, D]
        """
        fx = self.feature_fn(x)   # [B, D]
        fy = self.feature_fn(y)   # [B, D]
        fz = self.feature_fn(z)   # [B, D]

        # outer products for each batch
        # [B, D, D, D]
        outer = torch.einsum('bi,bj,bk->bijk', fx, fy, fz)

        # mean over batch
        A = outer.mean(dim=0)

        return A
 
class ConditionalEstimation(nn.Module):
    # Attribute type hints for class properties
    x_candidates: List[Any]
    y: Any
    z: Any
    a: Tensor
    feature_fn: Callable[[Any], Tensor]

    def __init__(
        self,
        *,
        x_candidates: List[Any],
        y: Any,
        z: Any,
        a: Tensor,
        feature_fn: Callable[[Any], Tensor]
    ) -> None:
        super().__init__()
        self.x_candidates = x_candidates
        self.y = y
        self.z = z
        self.a = a
        self.feature_fn = feature_fn

    def forward(self) -> List[Tensor]:
        """
        Computes conditional scores for candidate inputs.
        """
        target_dtype: torch.dtype = self.a.dtype
        target_device: torch.device = self.a.device
    
        # Transform inputs and move to target device/dtype
        fy: Tensor = self.feature_fn(self.y).to(dtype=target_dtype, device=target_device).view(-1)
        fz: Tensor = self.feature_fn(self.z).to(dtype=target_dtype, device=target_device).view(-1)
    
        # Prepare NumPy versions for CUDA kernel processing
        fy_np: np.ndarray = fy.cpu().numpy()
        fz_np: np.ndarray = fz.cpu().numpy()
        a_np: np.ndarray = self.a.cpu().numpy()
        
        # Determine feature dimension D from the y-feature
        D: int = fy_np.shape[0]
    
        # Create dummy x for shape compatibility in kernel
        dummy_x: np.ndarray = np.zeros((1, D), dtype=np.float32)
    
        # Execute custom CUDA kernel to retrieve context
        _, context_np = layers_kernel.conditional_estimation(
            dummy_x, fy_np, fz_np, a_np, return_context=True
        )
    
        # Restore context to PyTorch tensor format
        context_tensor: Tensor = torch.as_tensor(context_np, dtype=target_dtype, device=target_device)
    
        # Iteratively compute dot-product scores for all candidates
        scores: List[Tensor] = []
        for x in self.x_candidates:
            fx: Tensor = self.feature_fn(x).to(dtype=target_dtype, device=target_device).view(-1)
            score: Tensor = torch.dot(fx, context_tensor)
            scores.append(score)
    
        return scores
    
class PropagationEstimation(nn.Module):
    """
    estimate Propagation of Tensor.
    """
    def __init__(self,
                 *,
                 y,
                 z,
                 a,
                 feature_fn):
        super().__init__()
        self.y = y
        self.z = z
        self.a = a
        self.feature_fn = feature_fn

    def forward(self):
        # dopasowanie dtype/device do tensora a
        target_dtype = self.a.dtype
        target_device = self.a.device

        fy = self.feature_fn(self.y).to(dtype=target_dtype, device=target_device).view(-1)
        fz = self.feature_fn(self.z).to(dtype=target_dtype, device=target_device).view(-1)

        numerator = torch.einsum('jk,j,k->', self.a[1], fy, fz)
        denominator = torch.einsum('jk,j,k->', self.a[0], fy, fz)

        ratio = numerator / (denominator + 1e-8)

        # przesunięcie bazy
        centered_ratio = ratio - 1.0

        const = torch.sqrt(torch.tensor(3.0, dtype=ratio.dtype, device=ratio.device))
        propagated = 0.5 + (1.0 / (2.0 * const)) * centered_ratio

        return propagated
    
class EntropyAndMutualInformation(nn.Module):
    """
    Calcuate entropy and/or mutual information. 
    Two methods are specifically set in the same class.
    """
    def __init__(self, compute_mi=False):
        super().__init__()
        self.compute_mi = compute_mi

    def approximate_entropy(self, activations):

        # Normalizacja prawdopodobieństw funkcji aktywacji
        probs = F.softmax(activations, dim=1)
        entropy = -torch.sum(probs ** 2, dim=1).mean()
        return entropy

    def approximate_mutual_information(self, act_X, act_Y):
        # Normalizacja funkcji aktywacji
        probs_X = F.softmax(act_X, dim=1)
        probs_Y = F.softmax(act_Y, dim=1)

        joint_probs = torch.bmm(probs_X.unsqueeze(2), probs_Y.unsqueeze(1))
        mi = torch.sum(joint_probs ** 2, dim=(1,2)).mean()
        return mi
    
    def forward(self, x, y=None):
        if self.compute_mi and y is not None:
            return self.approximate_mutual_information(x, y)
        else:
            return self.approximate_entropy(x)

class DynamicEMA(nn.Module):
    def __init__(self, ema_lambda: float = 0.9):
        super().__init__()
        self.ema_lambda = ema_lambda
        self.register_buffer("a", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.a.numel() == 1:
            self.a = x.detach().clone()
        else:
            self.a = (self.ema_lambda * self.a.detach() +
                      (1 - self.ema_lambda) * x.detach())
        return self.a

#Optimizer

#Loss Function
class InformationBottleneckLoss(nn.Module):
    """
    Information Bottleneck criterion for neural networks.
    
    Computes:
        L_IB = I(X; T) - β * I(T; Y)
    where I(·;·) is estimated using a Hilbert-Schmidt Independence Criterion (HSIC) approximation:
        HSIC(X, Y) ≈ ||XᵀY||_F²
    """
    def __init__(self, beta: float = 1.0, normalize: bool = True) -> None:
        super().__init__()
        self.beta = beta
        self.normalize = normalize

    def _hsic(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        Compute a differentiable Hilbert-Schmidt Independence Criterion (HSIC) approximation.
        Tr((XXᵀ)(YYᵀ)) = ||XᵀY||_F²
        """
        if self.normalize:
            X = F.normalize(X, dim=-1)
            Y = F.normalize(Y, dim=-1)

        cross = torch.matmul(X.T, Y)
        hsic_value = torch.sum(cross.pow(2))
        return hsic_value / X.size(0)

    def forward(self, X: Tensor, T: Tensor, Y: Tensor) -> Tensor:
        """
        Compute the Information Bottleneck loss:
        L = I(X;T) - β * I(T;Y)
        """
        I_XT = self._hsic(X, T)
        I_TY = self._hsic(T, Y)
        loss = I_XT - self.beta * I_TY
        return loss
 