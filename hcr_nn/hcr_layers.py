import numpy as np
import math, os
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
import time
import layers_kernel
from typing import Any, Callable, List, Tuple, Literal, Optional


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
    
class MeanEstimation(nn.Module):
    # Defining attributes for clarity and static analysis
    triplets: List[Tuple[Any, Any, Any]]
    feature_fn: Callable[[Any], Tensor]
    feature_dm: int

    def __init__(
        self,
        *,
        triplets: List[Tuple[Any, Any, Any]],
        feature_fn: Callable[[Any], Tensor],
        feature_dm: int
    ) -> None:
        super().__init__()
        self.triplets = triplets
        self.feature_fn = feature_fn
        self.feature_dm = feature_dm

    def forward(self) -> Tensor:
        """
        Calculates the mean estimation using a custom CUDA kernel.
        
        Returns:
            Tensor: The result from the layers_kernel.mean_estimation.
        """
        N: int = len(self.triplets)
        D: int = self.feature_dm

        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Precompute features
        fx_list: List[Tensor] = []
        fy_list: List[Tensor] = []
        fz_list: List[Tensor] = []
        
        for (x, y, z) in self.triplets:
            fx_list.append(self.feature_fn(x))
            fy_list.append(self.feature_fn(y))
            fz_list.append(self.feature_fn(z))

        # Stack and move to device
        fx: Tensor = torch.stack(fx_list).to(device)
        fy: Tensor = torch.stack(fy_list).to(device)
        fz: Tensor = torch.stack(fz_list).to(device)

        # Run CUDA kernel
        # Note: detach().cpu().numpy() converts to a NumPy array, 
        # so 'a' type depends on what the kernel returns (likely a Tensor or Array)
        result_array: Any = layers_kernel.mean_estimation(
            fx.detach().cpu().numpy(),
            fy.detach().cpu().numpy(),
            fz.detach().cpu().numpy()
        )
        
        # Ensure the return type matches the expected Tensor hint
        return torch.as_tensor(result_array)
 
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
        
        Returns:
            List[Tensor]: A list of dot-product scores for each candidate in x_candidates.
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
    # Class attribute annotations
    y: Any
    z: Any
    a: Tensor
    feature_fn: Callable[[Any], Tensor]

    def __init__(
        self,
        *,
        y: Any,
        z: Any,
        a: Tensor,
        feature_fn: Callable[[Any], Tensor]
    ) -> None:
        super().__init__()
        self.y = y
        self.z = z
        self.a = a
        self.feature_fn = feature_fn

    def forward(self) -> Tensor:
        """
        Computes propagated expectation ratio using custom kernel operations.
        
        Returns:
            Tensor: The normalized propagation estimation result.
        """
        target_dtype: torch.dtype = self.a.dtype
        target_device: torch.device = self.a.device

        # Map inputs to features and ensure they match 'a' in dtype and device
        fy: Tensor = self.feature_fn(self.y).to(dtype=target_dtype, device=target_device).view(-1)
        fz: Tensor = self.feature_fn(self.z).to(dtype=target_dtype, device=target_device).view(-1)

        # Execution of custom CUDA kernels
        # Using .contiguous() is crucial for stable C++ extension performance
        numerator: Tensor = layers_kernel.propagate_expectation(self.a[1].contiguous(), fy, fz)
        denominator: Tensor = layers_kernel.propagate_expectation(self.a[0].contiguous(), fy, fz)

        # Numerical stability handling for ratio calculation
        ratio: Tensor = numerator / (denominator + 1e-8)

        # Normalization and transformation logic
        centered_ratio: Tensor = ratio - 1.0
        const: Tensor = torch.sqrt(torch.tensor(3.0, dtype=target_dtype, device=target_device))
        propagated: Tensor = 0.5 + (1.0 / (2.0 * const)) * centered_ratio

        return propagated
    
class EntropyAndMutualInformation(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def approximate_entropy(self, activations: Tensor) -> Tensor:
        """
        Calculates the approximate Shannon entropy of the given activations.
        
        Args:
            activations: Input tensor of shape (batch_size, num_classes).
            
        Returns:
            Tensor: A scalar tensor representing the mean entropy.
        """
        probs: Tensor = F.softmax(activations, dim=-1)
        log_probs: Tensor = torch.log(probs + 1e-8)
        # Sum over classes, mean over batch
        entropy: Tensor = -(probs * log_probs).sum(dim=-1).mean()
        return entropy

    def approximate_mutual_information(self, act_X: Tensor, act_Y: Tensor) -> Tensor:
        """
        Calculates the mutual information between two sets of activations using a CUDA kernel.
        
        Args:
            act_X: Activations for variable X.
            act_Y: Activations for variable Y.
            
        Returns:
            Tensor: The calculated mutual information.
        """
        probs_X: Tensor = F.softmax(act_X, dim=1)
        probs_Y: Tensor = F.softmax(act_Y, dim=1)
    
        # Ensure the CUDA kernel receives the probability tensors
        return layers_kernel.approximate_mi_cu(probs_X, probs_Y)
    
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
 