import numpy as np
import math, os
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
import layers_kernels
import time

"""
Wczesna wersja optymalizacji funkcji składowych HCR.
Nietestowana i niegotowa do użytku.
"""

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, ** kwargs)
        end = time.time()
        return (start, end, end - start)
    return wrapper

@timer
def printer():
    for x in 10000:
        print('hello world')

"""
Default imports of HCRNN Components.
Make sure that __init__.py file is correctly opening layers
"""

__all__ = ['CDFNorm', 
           'MeanEstimation', 
           'ConditionalEstimation', 
           'PropagationEstimation', 
           'EntropyAndMutualInformation', 
           'DynamicEMA', 
           'BaseOptimization', 
           'InformationBottleneck']

class CDFNorm(nn.Module):
    def __init__(self, method='gaussian', unbiased=True, eps=1e-5, affine=False, track_running_stats=True):
        """
        Normalizacja CDF (dystrybuanty).

        Parametry:
            method: metoda normalizacji ('gaussian' lub 'empirical')
            unbiased: czy użyć nieobciążonego estymatora wariancji
            eps: mała wartość dla stabilności numerycznej
            affine: czy zastosować transformację afiniczną
            track_running_stats: czy śledzić statystyki podczas uczenia
        """
        super().__init__()
        self.method = method
        self.unbiased = unbiased
        self.eps = eps
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.ones(1))  # Parametr skalujący
            self.bias = nn.Parameter(torch.zeros(1))    # Parametr przesunięcia

        if self.track_running_stats:
            # Rejestracja buforów dla średniej i wariancji
            self.register_buffer('running_mean', torch.zeros(1))
            self.register_buffer('running_var', torch.ones(1))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def _gaussian_transform(self, x):
        """Transformacja Gaussa - normalizacja przy użyciu CDF rozkładu normalnego."""
        if self.training and self.track_running_stats:
            # Obliczanie statystyk podczas uczenia

            var, mean = torch.var_mean(x, unbiased=self.unbiased)

            with torch.no_grad():
                # Aktualizacja średniej kroczącej
                self.running_mean.lerp_(mean, self.momentum)
                # Aktualizacja wariancji kroczącej
                self.running_var.lerp_(var, self.momentum)
                self.num_batches_tracked.add_(1)
        else:
            # Użycie zapisanych statystyk podczas ewaluacji
            mean = self.running_mean
            var = self.running_var

        # Obliczenie CDF przy użyciu funkcji błędu
        std = torch.sqrt(var + self.eps)
        x_norm = 0.5 * (1 + torch.erf((x - mean) / (std * torch.sqrt(torch.tensor(2.0, device=x.device, dtype=x.dtype)))))

        if self.affine:
            # Transformacja afiniczną
            x_norm = x_norm * self.weight + self.bias

        return x_norm
    
    ''' błędna funkcja
    def _empirical_transform(self, x):
        """Empiryczna transformacja CDF na podstawie rang."""
        x_norm = torch.zeros_like(x)
        for i in range(len(x)):
            # Obliczenie rangi dla każdego elementu
            x_norm[i] = (x < x[i]).float().mean()

        if self.affine:
            # Transformacja afiniczną
            x_norm = x_norm * self.weight + self.bias

        return x_norm
        '''
    
    def _empirical_transform(self, x):
        N = torch.numel(x)
        
        sorted_x, indices = torch.sort(x)
        ranks = torch.empty_like(indices, dtype=torch.float)
        ranks[indices] = torch.arange(1, N + 1, device=x.device, dtype=torch.float)

        x_norm = ranks / N

        if self.affine:
            x_norm = x_norm * self.weight + self.bias

        return x_norm

    def forward(self, x):
        """
        Przebieg forward normalizacji CDF.

        Parametry:
            x: tensor wejściowy

        Zwraca:
            Znormalizowany tensor w przedziale [0,1]
        """
        if self.method == 'gaussian':
            return self._gaussian_transform(x)
        elif self.method == 'empirical':
            return self._empirical_transform(x)
        else:
            raise ValueError(f"Niewspierana metoda normalizacji: {self.method}")
        
class MeanEstimation(nn.Module):
    def __init__(self,
                 *,
                 triplets,
                 feature_fn,
                 feature_dm
                 ):
        super().__init__()
        self.triplets = triplets
        self.feature_fn = feature_fn
        self.feature_dm = feature_dm

    def compute_tensor_mean(self) -> Tensor:
        """
        Parametry:
            triplets: array (x, y, z)
            feature_fn: funckaj mapująca
            feature_dm: wymiary D
        """
        N = len(self.triplets)
        D = self.feature_dm

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Precompute features
        fx_list, fy_list, fz_list = [], [], []
        for (x, y, z) in self.triplets:
            fx_list.append(self.feature_fn(x))
            fy_list.append(self.feature_fn(y))
            fz_list.append(self.feature_fn(z))

        fx = torch.stack(fx_list).to(device)
        fy = torch.stack(fy_list).to(device)
        fz = torch.stack(fz_list).to(device)

        # Run CUDA kernel
        a = layers_kernels.mean_estimation_cu(fx, fy, fz, D)

        return a
    
class ConditionalEstimation(nn.Module):
    def __init__(self,
                 *,
                 x_candidates,
                 y,
                 z,
                 a,
                 feature_fn) -> None:
        super().__init__()
        self.x_candidates = x_candidates
        self.y = y
        self.z = z
        self.a = a
        self.feature_fn = feature_fn

    def conditional_score(self):

        target_dtype = self.a.dtype
        target_device = self.a.device

        fy = self.feature_fn(self.y).to(dtype=target_dtype, device=target_device).view(-1)
        fz = self.feature_fn(self.z).to(dtype=target_dtype, device=target_device).view(-1)

        # denominator: use slice A[0]
        denominator = layers_kernels.conditional_estimation_cu(
            self.a[0:1].contiguous(), fy, fz
        )[0]

        # all i-scores at once with CUDA
        context_sums = layers_kernels.conditional_estimation_cu(
            self.a.contiguous(), fy, fz
        )  # shape [I]

        scores = []
        for x in self.x_candidates:
            fx = self.feature_fn(x).to(dtype=target_dtype, device=target_device).view(-1)
            # weighted dot product
            score = torch.dot(fx, context_sums / (denominator + 1e-8))
            scores.append(score)

        return scores
    
class PropagationEstimation(nn.Module):
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

    def propagate_expectation(self):
        target_dtype = self.a.dtype
        target_device = self.a.device

        fy = self.feature_fn(self.y).to(dtype=target_dtype, device=target_device).view(-1)
        fz = self.feature_fn(self.z).to(dtype=target_dtype, device=target_device).view(-1)

        # Use CUDA kernel instead of einsum
        numerator   = layers_kernels.propagate_expectation_cu(self.a[1].contiguous(), fy, fz)
        denominator = layers_kernels.propagate_expectation_cu(self.a[0].contiguous(), fy, fz)

        ratio = numerator / (denominator + 1e-8)

        centered_ratio = ratio - 1.0
        const = torch.sqrt(torch.tensor(3.0, dtype=ratio.dtype, device=ratio.device))
        propagated = 0.5 + (1.0 / (2.0 * const)) * centered_ratio

        return propagated
    
class EntropyAndMutualInformation(nn.Module):

    def approximate_entropy(self, activations):
        probs = F.softmax(activations, dim=1)

        return layers_kernels.approximate_entropy_cu(probs)

    def approximate_mutual_information(self, act_X, act_Y):
        probs_X = F.softmax(act_X, dim=1)
        probs_Y = F.softmax(act_Y, dim=1)
    
        return layers_kernels.approximate_mi_cu(probs_X, probs_Y)
    
class DynamicEMA(nn.Module):
    def __init__(self, x, y, z, ema_lambda) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.z = z
        self.ema_lambda = ema_lambda
        self.a = torch.zeros((len(x), len(y), len(z)), device=x.device, dtype=x.dtype)  # pusty tensor o rozmiarze sumy einstein'a

    def EMAUpdateMethod(self):
        self.a = layers_kernels.ema_update_cu(self.x, self.y, self.z, self.a, self.ema_lambda)
        return self.a
        
class BaseOptimization(nn.Module):
    def __init__(self,
                 *,
                 a, #tensor do optymalizacji
                 ) -> None:
        self. a = a

    def optimization_early(self) -> Tensor:
        M = self.a.reshape(len(self.a[0]), -1)

        # Obliczenie SVD
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)

        # Step 4: Transformacja Tensora
        new_a = layers_kernels.base_optimization_cu(U.T.contiguous(), self.a.contiguous())

        return new_a
    
class InformationBottleneck(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, X_features, Y_features):
        """
        Optymizacja:
        Obliczenie Tr( (X Xᵀ)(Y Yᵀ) )
        Bardziej wydajną formułą: Tr( (XᵀY)(YᵀX) )
        """
        
        """Implementuje równanie (15) z artykułu"""
        XY = X_features @ Y_features.T
        return torch.sum(XY * XY)

    def bottleneck_loss(self, X_features, T_features, Y_features):
        """Implementuje równanie (10) z artykułu"""
        I_XT = self(X_features, T_features)
        I_TY = self(T_features, Y_features)
        return I_XT - self.beta * I_TY
    
    