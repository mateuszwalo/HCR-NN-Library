import numpy as np
import math, os
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
import time
import layers_kernel


#Wczesna wersja optymalizacji funkcji składowych HCR.
#Nietestowana i niegotowa do użytku.

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


#Default imports of HCRNN Components.
#Make sure that __init__.py file is correctly opening layers

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
        a = layers_kernel.mean_estimation(
            fx.detach().cpu().numpy(),
            fy.detach().cpu().numpy(),
            fz.detach().cpu().numpy()
        )
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
    
        # Move to NumPy for CUDA
        fy_np = fy.cpu().numpy()
        fz_np = fz.cpu().numpy()
        a_np = self.a.cpu().numpy()
        a0_np = self.a[0:1].cpu().numpy()
        D = fy_np.shape[0]
    
        # Dummy x for shape
        dummy_x = np.zeros((1, D), dtype=np.float32)
    
        # Get denom and context
        _, context_np = layers_kernel.conditional_estimation(dummy_x, fy_np, fz_np, a_np, return_context=True)
    
        # Convert to tensor
        context_tensor = torch.tensor(context_np, dtype=target_dtype, device=target_device)
    
        # Compute conditional scores
        scores = []
        for x in self.x_candidates:
            fx = self.feature_fn(x).to(dtype=target_dtype, device=target_device).view(-1)
            score = torch.dot(fx, context_tensor)
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

        numerator   = layers_kernel.propagate_expectation(self.a[1].contiguous(), fy, fz)
        denominator = layers_kernel.propagate_expectation(self.a[0].contiguous(), fy, fz)

        ratio = numerator / (denominator + 1e-8)

        centered_ratio = ratio - 1.0
        const = torch.sqrt(torch.tensor(3.0, dtype=ratio.dtype, device=ratio.device))
        propagated = 0.5 + (1.0 / (2.0 * const)) * centered_ratio

        return propagated
    
class EntropyAndMutualInformation(nn.Module):

    def approximate_entropy(self, activations):
        probs = F.softmax(activations, dim=1)

        return layers_kernel.approximate_entropy_cu(probs)

    def approximate_mutual_information(self, act_X, act_Y):
        probs_X = F.softmax(act_X, dim=1)
        probs_Y = F.softmax(act_Y, dim=1)
    
        return layers_kernel.approximate_mi_cu(probs_X, probs_Y)
    
class DynamicEMA(nn.Module):
    def __init__(self, x, y, z, ema_lambda) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.z = z
        self.ema_lambda = ema_lambda
        self.a = torch.zeros((len(x), len(y), len(z)), device=x.device, dtype=x.dtype)

    def EMAUpdateMethod(self):
        a_np = self.a.detach().cpu().numpy()
        x_np = self.x.detach().cpu().numpy()
        y_np = self.y.detach().cpu().numpy()
        z_np = self.z.detach().cpu().numpy()

        updated = layers_kernel.ema_update(a_np, x_np, y_np, z_np, float(self.ema_lambda))
        self.a = torch.tensor(updated, device=self.x.device, dtype=self.x.dtype)
        return self.a
        
class BaseOptimization(nn.Module):
    def __init__(self, *, a: Tensor):
        super().__init__()
        self.a = a

    def optimization_early(self) -> Tensor:
        M = self.a.reshape(len(self.a[0]), -1)
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)

        new_a = layers_kernel.transform_tensor(
            U.T.contiguous().cpu().numpy(),
            self.a.cpu().numpy()
        )

        return torch.tensor(new_a, dtype=self.a.dtype, device=self.a.device)

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
