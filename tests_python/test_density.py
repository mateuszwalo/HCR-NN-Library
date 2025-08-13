# tests/test_density.py
import pytest
import torch

# PolynomialBasis uses scipy.special.eval_legendre
pytest.importorskip("scipy")

from hcr_nn.basis import PolynomialBasis, CosineBasis, KDEBasis
from hcr_nn.density import (
    clamp_density,
    joint_density,
    conditional_density,
    expected_u1_given_u2,
)


# ------------------------- clamp_density -------------------------

def test_clamp_density_min():
    rho = torch.tensor([0.5, -1e-3, 0.0, 1e-9], dtype=torch.float64)
    clamped = clamp_density(rho, eps=1e-6)
    # All values should be at least eps
    assert torch.all(clamped >= 1e-6)
    # Values greater than eps should remain unchanged
    assert clamped[0].item() == pytest.approx(0.5)


# ------------------------- joint_density -------------------------

def test_joint_density_identity_coeffs_matches_reference():
    """
    For coeffs = I, rho(u1, u2) = sum_i f_i(u1) * f_i(u2).
    Verify einsum-based implementation against this reference.
    """
    deg = 3
    C = torch.eye(deg + 1, dtype=torch.float64)
    basis = PolynomialBasis(deg)

    N = 123
    u1 = torch.linspace(0, 1, N, dtype=torch.float64)
    u2 = torch.linspace(0, 1, N, dtype=torch.float64)

    F1 = basis(u1)  # (N, D+1)
    F2 = basis(u2)  # (N, D+1)

    # basis_vals: (N, 2, D+1) for paired (u1[n], u2[n])
    basis_vals = torch.stack([F1, F2], dim=1)  # (N, 2, D+1)
    u12 = torch.stack([u1, u2], dim=1)        # (N, 2) for shape validation

    rho = joint_density(u12, C, basis_vals)          # (N,)
    rho_ref = torch.einsum("ni,ni->n", F1, F2)       # sum_i f_i(u1)*f_i(u2)

    assert rho.shape == (N,)
    assert torch.allclose(rho, rho_ref, rtol=1e-7, atol=1e-7)


# ------------------------- conditional_density -------------------------

def test_conditional_density_zero_coeffs_is_uniform_after_norm():
    """
    With coeffs == 0, after clamping and normalization the conditional density
    should be uniform over [0,1].
    """
    deg = 2
    C = torch.zeros(deg + 1, deg + 1, dtype=torch.float64)
    basis = PolynomialBasis(deg)

    N = 501
    u1_grid = torch.linspace(0, 1, N, dtype=torch.float64)
    p = conditional_density(u1_grid, 0.37, C, basis, deg, eps=1e-6)

    du = 1.0 / (N - 1)
    area = (p.sum() * du).item()
    assert area == pytest.approx(1.0, rel=1e-6, abs=1e-6)

    # Uniform means nearly constant values across the grid
    assert torch.max(p) - torch.min(p) < 1e-6

    # For uniform on [0,1], E[u1|u2] ~ 0.5
    mu = expected_u1_given_u2(0.37, C, basis, deg, grid_size=N, eps=1e-6)
    assert mu == pytest.approx(0.5, abs=5e-3)


def test_conditional_density_identity_coeffs_integrates_to_one_and_nonnegative():
    deg = 3
    C = torch.eye(deg + 1, dtype=torch.float64)
    basis = PolynomialBasis(deg)

    N = 1001
    u1_grid = torch.linspace(0, 1, N, dtype=torch.float64)
    p = conditional_density(u1_grid, 0.5, C, basis, deg, eps=1e-12)

    du = 1.0 / (N - 1)
    area = (p.sum() * du).item()
    assert area == pytest.approx(1.0, rel=1e-4, abs=1e-4)
    assert torch.all(p >= 0)


def test_expected_u1_given_u2_in_range():
    deg = 2
    C = torch.eye(deg + 1, dtype=torch.float64)
    basis = PolynomialBasis(deg)

    mu = expected_u1_given_u2(0.2, C, basis, deg, grid_size=801, eps=1e-12)
    assert 0.0 <= mu <= 1.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_conditional_density_dtype_consistency(dtype):
    deg = 1
    C = torch.eye(deg + 1, dtype=dtype)
    basis = PolynomialBasis(deg)

    u1_grid = torch.linspace(0, 1, 257, dtype=dtype)
    p = conditional_density(u1_grid, 0.3, C, basis, deg, eps=1e-8)
    assert p.dtype == dtype


# ------------------------- (aux) basis sanity kept here for convenience -------------------------
# These are technically basis tests, but we keep them here for convenience while density depends on basis.

@pytest.mark.parametrize("degree", [0, 1, 3])
def test_polynomial_orthonormality_trapz(degree):
    """
    ∫_0^1 f_i(u) f_j(u) du ≈ δ_ij via trapezoidal rule on a dense grid.
    """
    u = torch.linspace(0, 1, 1001, dtype=torch.float64)
    F = PolynomialBasis(degree)(u)  # (1001, degree+1)
    inner = torch.trapz(F.unsqueeze(2) * F.unsqueeze(1), u, dim=0)
    eye = torch.eye(degree + 1, dtype=torch.float64)
    assert torch.allclose(inner, eye, atol=1e-2), f"Polynomial basis not orthonormal for deg={degree}"


@pytest.mark.parametrize("degree", [0, 1, 3])
def test_cosine_orthonormality_trapz(degree):
    u = torch.linspace(0, 1, 1001, dtype=torch.float64)
    F = CosineBasis(degree)(u)
    inner = torch.trapz(F.unsqueeze(2) * F.unsqueeze(1), u, dim=0)
    eye = torch.eye(degree + 1, dtype=torch.float64)
    assert torch.allclose(inner, eye, atol=1e-2), f"Cosine basis not orthonormal for deg={degree}"


def test_kde_basis_shape_and_nonnegativity():
    centers = torch.rand(50, dtype=torch.float64)
    basis = KDEBasis(centers=centers, bandwidth=0.1)
    u = torch.linspace(0, 1, 200, dtype=torch.float64)
    K = basis(u)
    assert K.shape == (200, 50)
    assert torch.all(K >= 0), "KDE basis must produce non-negative values"
