# High-level HCR models: 2D conditional network (E[u1|u2]) with selectable basis.
# Compatible with hcr_nn.basis (Polynomial/Cosine/KDE) and torch.
#
# Expected input to forward():
#   - Either quantile space u in [0,1], shape (B, 2)
#   - Or raw x, if you pass a callable quantile_fn: x -> u in [0,1]
#
# Output:
#   - Tensor (B,) with E[u1 | u2] (in quantile space). You can convert
#     it do surowej skali przez swoją funkcję inv_cdf gdy potrzebujesz.

from __future__ import annotations
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor
from hcr_nn.basis import select_basis


class HCRCond2D(nn.Module):
    """
    HCR conditional model for 2D:
      - learns coefficient tensor a ∈ R^{(deg+1) x (deg+1)}
      - computes E[u1 | u2] by integrating conditional density over a grid in [0,1]

    Mat. idea:
      ρ(u1,u2) = Σ_{i,j} a_{ij} f_i(u1) f_j(u2)
      p(u1|u2) ∝ max(ρ(u1,u2), eps)
      E[u1|u2] = ∫ u1 * p(u1|u2) du1

    Notes:
      - Vectorized forward (batch x grid) → szybkie inference.
      - You can pass `quantile_fn` to transform raw x -> u in [0,1] inside forward.
    """

    def __init__(
        self,
        degree: int,
        basis_name: str = "polynomial",      # 'polynomial' | 'cosine' | 'kde'
        grid_size: int = 256,
        coeff_init: str = "xavier",          # 'xavier' | 'zeros' | 'eye' (eye only if deg>=1)
        quantile_fn: Optional[Callable[[Tensor], Tensor]] = None,
        kde_centers: Optional[Tensor] = None,
        kde_bandwidth: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.degree = int(degree)
        self.grid_size = int(grid_size)
        self.quantile_fn = quantile_fn

        # Basis selection (for both u1 and u2 we reuse the same 1D basis)
        if basis_name.lower() == "kde":
            if kde_centers is None:
                raise ValueError("KDE basis requires `kde_centers` (1D tensor of centers in [0,1]).")
            self.basis = select_basis("kde", degree=0, centers=kde_centers.to(device=device, dtype=dtype),
                                      bandwidth=float(kde_bandwidth))
            # For KDE degree is effectively (#centers-1), but we store degree only for API symmetry
            self._basis_dim = kde_centers.numel()
        else:
            self.basis = select_basis(basis_name, degree=self.degree)
            self._basis_dim = self.degree + 1

        # Coefficient tensor a_{ij} (learnable)
        a = torch.empty(self._basis_dim, self._basis_dim, dtype=dtype, device=device)
        if coeff_init == "zeros":
            nn.init.zeros_(a)
        elif coeff_init == "eye":
            # 'eye' makes sense as a neutral-ish start for polynomial/cosine when _basis_dim>=2
            a.fill_(0.)
            eye_sz = min(self._basis_dim, self._basis_dim)
            a[:eye_sz, :eye_sz] = torch.eye(eye_sz, dtype=dtype, device=device)
        elif coeff_init == "xavier":
            nn.init.xavier_uniform_(a)
        else:
            raise ValueError(f"Unknown coeff_init '{coeff_init}'")
        self.coeffs = nn.Parameter(a)  # shape (D,D)

        # Precompute u1 grid in [0,1] for expectation
        self.register_buffer("u1_grid", torch.linspace(0, 1, self.grid_size, dtype=dtype, device=device))

        # Small positive for density clamping
        self.eps = 1e-6

    @property
    def basis_dim(self) -> int:
        return self._basis_dim

    def _basis_1d(self, u: Tensor) -> Tensor:
        """
        Evaluate basis on 1D inputs u ∈ [0,1].
        Returns shape (..., basis_dim).
        """
        F = self.basis(u)  # shape (..., basis_dim) for our bases
        # Ensure dtype/device consistency with params
        return F.to(dtype=self.coeffs.dtype, device=self.coeffs.device)

    def _conditional_density(self, u2_vec: Tensor) -> Tensor:
        """
        Compute p(u1|u2) over whole batch and u1_grid.
        u2_vec: (B,) in [0,1]
        Returns: p_cond of shape (B, G) (rows integrate to 1).
        """
        B = u2_vec.shape[0]
        G = self.u1_grid.shape[0]

        # Basis for u2: (B, D)
        f2 = self._basis_1d(u2_vec)  # (B, D)

        # Basis for u1 grid: (G, D)
        f1g = self._basis_1d(self.u1_grid)  # (G, D)

        # ρ_b(g) = Σ_{i,j} f2_b[j] * a_{ij} * f1g_g[i]
        # einsum: 'bd,ij,gd->bg'
        rho = torch.einsum('bd,ij,gd->bg', f2, self.coeffs, f1g)

        # clamp and normalize along grid
        rho = torch.clamp(rho, min=self.eps)          # (B,G)
        # integrate with Δu = 1/(G-1)
        du = 1.0 / (G - 1)
        Z = (rho.sum(dim=1, keepdim=True) * du)       # (B,1)
        p = rho / Z
        return p

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward computes E[u1|u2] for a batch of inputs.
        Input x can be:
          - (B, 2) already in quantile space [0,1], if quantile_fn is None
          - (B, 2) in raw space, if quantile_fn is provided (will be transformed)
        Output: (B,) – expectation in quantile space.
        """
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError(f"HCRCond2D.forward expects (B,2), got {tuple(x.shape)}")

        x = x.to(dtype=self.coeffs.dtype, device=self.coeffs.device)

        if self.quantile_fn is not None:
            # user-provided callable that maps raw x -> u in [0,1]
            u = self.quantile_fn(x)
        else:
            u = x  # assume already quantiles

        u1 = u[:, 0]  # not used for expectation directly, but can be useful for loss
        u2 = u[:, 1]

        # p(u1|u2) for whole batch
        p = self._conditional_density(u2)          # (B,G)
        du = 1.0 / (self.grid_size - 1)
        # E[u1|u2] = Σ u1_g * p_b(g) * du
        e = (self.u1_grid.unsqueeze(0) * p).sum(dim=1) * du  # (B,)
        return e  # in quantile space [0,1]

    @torch.no_grad()
    def conditional_curve(self, u2_grid: Tensor) -> Tensor:
        """
        Helper for visualization:
        Returns E[u1|u2] for a given u2_grid (1D tensor in [0,1]).
        """
        u2_grid = u2_grid.to(dtype=self.coeffs.dtype, device=self.coeffs.device)
        p = self._conditional_density(u2_grid)  # (G2, G1)
        du = 1.0 / (self.grid_size - 1)
        e = (self.u1_grid.unsqueeze(0) * p).sum(dim=1) * du
        return e

    def extra_repr(self) -> str:
        return (f"degree={self.degree}, basis_dim={self._basis_dim}, "
                f"grid_size={self.grid_size}, eps={self.eps}")


def build_hcr_cond2d(
    degree: int,
    basis: str = "polynomial",
    grid_size: int = 256,
    coeff_init: str = "xavier",
    quantile_fn: Optional[Callable[[Tensor], Tensor]] = None,
    kde_centers: Optional[Tensor] = None,
    kde_bandwidth: float = 0.05,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> HCRCond2D:
    """
    Convenience factory.
    """
    return HCRCond2D(
        degree=degree,
        basis_name=basis,
        grid_size=grid_size,
        coeff_init=coeff_init,
        quantile_fn=quantile_fn,
        kde_centers=kde_centers,
        kde_bandwidth=kde_bandwidth,
        dtype=dtype,
        device=device,
    )
