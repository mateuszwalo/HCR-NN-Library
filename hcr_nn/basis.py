import torch
import torch.nn as nn
from torch import Tensor
from scipy.special import eval_legendre


class PolynomialBasis(nn.Module):
    """
    Shifted‐Legendre orthonormal basis on [0,1]:
      f_i(u) = sqrt(2*i+1) * P_i(2u-1)
    where P_i is the standard Legendre polynomial on [-1,1].
    """

    def __init__(self, degree: int):
        super().__init__()
        self.degree = degree

    def forward(self, u: Tensor) -> Tensor:
        # ensure correct dtype and device
        dtype = u.dtype
        device = u.device
        u_flat = u.reshape(-1)
        x = 2 * u_flat - 1  # map [0,1] -> [-1,1]
        x_np = x.detach().cpu().numpy()
        out = []
        for i in range(self.degree + 1):
            Pi_np = eval_legendre(i, x_np)
            Pi = torch.as_tensor(Pi_np, dtype=dtype, device=device)
            scale = torch.sqrt(torch.tensor(2 * i + 1, dtype=dtype, device=device))
            out.append(scale * Pi)
        F = torch.stack(out, dim=-1)  # shape (N, degree+1)
        return F.reshape(*u.shape, self.degree + 1)


class CosineBasis(nn.Module):
    """
    Orthonormal cosine basis on [0,1]:
    f_0(u)=1
    f_j(u)=sqrt(2) * cos(j*pi*u) for j>0
    """

    def __init__(self, degree: int):
        super().__init__()
        self.degree = degree

    def forward(self, u: Tensor) -> Tensor:
        dtype = u.dtype
        device = u.device
        u_flat = u.reshape(-1)
        out = [torch.ones_like(u_flat, dtype=dtype, device=device)]
        for j in range(1, self.degree + 1):
            coef = torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
            out.append(coef * torch.cos(torch.pi * j * u_flat))
        F = torch.stack(out, dim=-1)
        return F.reshape(*u.shape, self.degree + 1)


class KDEBasis(nn.Module):
    """
    Empirical KDE basis: for each training point u_k we have basis function
      f_k(u) = Gaussian(u; center=u_k, bandwidth=sigma)
    """

    def __init__(self, centers: Tensor, bandwidth: float):
        super().__init__()
        centers = centers.reshape(-1)
        if centers.numel() == 0:
            raise ValueError("centers must be non-empty")
        if bandwidth <= 0:
            raise ValueError("bandwidth must be > 0")
        self.register_buffer('centers', centers)
        self.bandwidth = float(bandwidth)

    def forward(self, u: Tensor) -> Tensor:
        dtype = u.dtype
        device = u.device
        u_flat = u.reshape(-1, 1)  # (N,1)
        centers = self.centers.reshape(1, -1)
        diff = (u_flat - centers.to(device)) / self.bandwidth
        two_pi = torch.tensor(2 * torch.pi, dtype=dtype, device=device)
        norm = self.bandwidth * torch.sqrt(two_pi)
        K = torch.exp(-0.5 * diff**2) / norm
        return K.reshape(*u.shape, -1)


def select_basis(name: str, degree: int, **kwargs) -> nn.Module:
    """
    Select basis from among polynomial, cosine and kde.
    """
    name = name.lower()
    if name == 'polynomial':
        if degree is None:
            raise ValueError("Polynomial basis requires 'degree'")
        return PolynomialBasis(degree)
    elif name == 'cosine':
        if degree is None:
            raise ValueError("Cosine basis requires 'degree'")
        return CosineBasis(degree)
    elif name == 'kde':
        centers = kwargs.get('centers')
        bandwidth = kwargs.get('bandwidth', 0.05)
        if centers is None:
            raise ValueError("KDEBasis requires 'centers' tensor")
        return KDEBasis(centers, bandwidth)
    else:
        raise ValueError(f"Unknown basis type: {name}")
