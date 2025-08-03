# hcr_nn/density.py

import torch
from torch import Tensor


def clamp_density(rho: Tensor, eps: float = 1e-6) -> Tensor:
    return torch.clamp(rho, min=eps)


def joint_density(u: Tensor,
                  coeffs: Tensor,
                  basis_vals: Tensor) -> Tensor:
    # for 2D HCR: basis_vals shape (...,2,deg+1)
    return torch.einsum('...i,ij,...j->...',
                        basis_vals[..., 0, :],
                        coeffs,
                        basis_vals[..., 1, :])


def conditional_density(u1_grid: Tensor,
                        u2_scalar: float,
                        coeffs: Tensor,
                        basis: torch.nn.Module,
                        deg: int,
                        eps: float = 1e-6) -> Tensor:
    dtype = u1_grid.dtype
    device = u1_grid.device

    # basis for u2
    u2 = torch.tensor([u2_scalar], dtype=dtype, device=device)
    f2 = basis(u2).reshape(deg+1)

    # basis for each u1 on grid
    f1 = basis(u1_grid)  # (G, deg+1)

    # joint density
    rho = torch.einsum('i,ij,gj->g', f2, coeffs, f1)
    rho = clamp_density(rho, eps)

    # normalize via trapezoidal rule
    du = 1.0 / (u1_grid.numel() - 1)
    Z = rho.sum() * du
    return rho / Z


def expected_u1_given_u2(u2_scalar: float,
                        coeffs: Tensor,
                        basis: torch.nn.Module,
                        deg: int,
                        grid_size: int = 200,
                        eps: float = 1e-6) -> float:
    dtype = coeffs.dtype
    device = coeffs.device
    u1_grid = torch.linspace(0, 1, grid_size, dtype=dtype, device=device)
    p = conditional_density(u1_grid, u2_scalar, coeffs, basis, deg, eps)
    du = 1.0 / (grid_size - 1)
    return float((u1_grid * p).sum() * du)
