import numpy as np
import math, os
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor

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
    """
        Normalizacja CDF (dystrybuanty).

        Parametry:
            method: metoda normalizacji ('gaussian' lub 'empirical')
            unbiased: czy użyć nieobciążonego estymatora wariancji
            eps: mała wartość dla stabilności numerycznej
            affine: czy zastosować transformację afiniczną
            track_running_stats: czy śledzić statystyki podczas uczenia
    """
    def __init__(self, method='gaussian', unbiased=True, eps=1e-5, affine=False, track_running_stats=True):
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
            mean = x.mean()
            var = x.var(unbiased=self.unbiased)
            with torch.no_grad():
                # Aktualizacja średniej kroczącej
                self.running_mean = (1 - 0.1) * self.running_mean + 0.1 * mean
                # Aktualizacja wariancji kroczącej
                self.running_var = (1 - 0.1) * self.running_var + 0.1 * var
                self.num_batches_tracked += 1
        else:
            # Użycie zapisanych statystyk podczas ewaluacji
            mean = self.running_mean
            var = self.running_var

        # Obliczenie CDF przy użyciu funkcji błędu
        x_norm = 0.5 * (1 + torch.erf((x - mean) / (torch.sqrt(var + self.eps) * math.sqrt(2))))

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
    """
    Calculate mean estimation for a tensor.
    """
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
        a = np.zeros((self.feature_dm, self.feature_dm, self.feature_dm))

        for (x, y, z) in self.triplets:
            fx = self.feature_fn(x)
            fy = self.feature_fn(y)
            fz = self.feature_fn(z)

            outer = np.einsum('i,j,k->ijk', fx, fy, fz)

            a += outer

        a /= len(self.triplets)  # Normalizacja na trójkach
        return a
    
class ConditionalEstimation(nn.Module):
    """
    Calculate estimation based on condition.
    """
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

        D = len(self.a)
        fy = self.feature_fn(self.y)
        fz = self.feature_fn(self.z)

        denominator = 0
        for j in range(D):
            for k in range(D):
                denominator += self.a[0, j, k] * fy[j] * fz[k]

        scores = []
        for x in self.x_candidates:
            fx = self.feature_fn(x)

            score = 0
            for i in range(D):
                context_sum = 0
                for j in range(D):
                    for k in range(D):
                        context_sum += self.a[i, j, k] * fy[j] * fz[k]
                score += fx[i] * (context_sum / (denominator + 1e-8)) #uniknięcie dzielenia przez zero

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

    def propagate_expectation(self):
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
    
class DynamicEMA(nn.Module):
    """
    Calculate Dynamic Exponential Moving Average for a Tensor.
    """
    def __init__(self, x, y, z, ema_lambda) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.z = z
        self.ema_lambda = ema_lambda
        self.a = torch.zeros_like(torch.einsum('i,j,k->ijk', x, y, z))  # pusty tensor o rozmiarze sumy einstein'a

    def EMAUpdateMethod(self):
        def f_i(x): return x
        def f_j(y): return y
        def f_k(z): return z

        update_tensor = torch.einsum('i,j,k->ijk', f_i(self.x), f_j(self.y), f_k(self.z))

        self.a = (1 - self.ema_lambda) * self.a + self.ema_lambda * update_tensor

        return self.a
    
class BaseOptimization(nn.Module):
    """
    Optimize the base of tensor.
    """
    def __init__(self,
                 *,
                 a, #tensor do optymalizacji
                 ) -> None:
        self. a = a

    def optimization_early(self) -> Tensor:
        M = self.a.reshape(len(self.a[0]), -1)

        # Obliczenie SVD
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)

        # Transformacja Bazy, tu przykładowa funkcja, do wymiany
        def f_x(x):
            return torch.sin(x * torch.linspace(0, 1, len(self.a[2])))

        # nowa baza g_i(x) = sum_j v_ij * f_j(x)
        def g_i(x, U):
            f = f_x(x)
            return torch.matmul(U.T, f)

        # Step 4: Transformacja Tensora
        new_a = torch.einsum('li,ljk->ijk', U.T, self.a)

        return new_a
    
class InformationBottleneck(nn.Module):
    """
    Calculate information bottleneck.
    """
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, X_features, Y_features):
        """Implementuje równanie (15) z artykułu"""
        C_X = X_features @ X_features.T
        C_Y = Y_features @ Y_features.T
        return torch.trace(C_X @ C_Y)

    def bottleneck_loss(self, X_features, T_features, Y_features):
        """Implementuje równanie (10) z artykułu"""
        I_XT = self(X_features, T_features)
        I_TY = self(T_features, Y_features)
        return I_XT - self.beta * I_TY
    
def hcr_nn_info():
    """
    Print information about the package.
    """
    print('The package implementation for Hierarchical Correlation Reconstruction')
