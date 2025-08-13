import torch
import numpy as np
import pytest

from basis import PolynomialBasis, CosineBasis, KDEBasis, select_basis
from density import clamp_density, joint_density, conditional_density, expected_u1_given_u2

@pytest.mark.parametrize("degree", [0, 1, 3])
def test_polynomial_orthonormality(degree):
    # Setup
    u = torch.linspace(0, 1, 1001)
    basis = PolynomialBasis(degree=degree)
    F = basis(u)  # shape (1001, degree+1)
    # Compute inner products via trapezoidal rule
    inner = torch.trapz(F.unsqueeze(2) * F.unsqueeze(1), u, dim=0)
    # Should be ~ identity
    eye = torch.eye(degree+1)
    assert torch.allclose(inner, eye, atol=1e-2), f"Polynomial basis not orthonormal for deg={degree}"

@pytest.mark.parametrize("degree", [0, 1, 3])
def test_cosine_orthonormality(degree):
    u = torch.linspace(0, 1, 1001)
    basis = CosineBasis(degree=degree)
    F = basis(u)
    inner = torch.trapz(F.unsqueeze(2) * F.unsqueeze(1), u, dim=0)
    eye = torch.eye(degree+1)
    assert torch.allclose(inner, eye, atol=1e-2), f"Cosine basis not orthonormal for deg={degree}"

def test_kde_basis():
    centers = torch.rand(50)
    basis = KDEBasis(centers=centers, bandwidth=0.1)
    u = torch.linspace(0,1,200)
    F = basis(u)
    assert F.shape == (200, 50)
    assert torch.all(F >= 0), "KDE basis must produce non-negative values"

def test_clamp_density():
    rho = torch.tensor([0.5, -0.1, 0.0, 1e-8])
    clamped = clamp_density(rho, eps=1e-3)
    assert torch.all(clamped >= 1e-3)

def test_joint_density_simple():
    # simple 2D identity coeffs
    deg = 2
    coeffs = torch.eye(deg+1)
    basis = PolynomialBasis(degree=deg)
    u = torch.linspace(0,1,100)
    # build basis_vals shape (100,2,deg+1)
    F1 = basis(u)
    F2 = basis(u)
    basis_vals = torch.stack([F1, F2], dim=1)
    rho = joint_density(torch.stack([u,u], dim=1), coeffs, basis_vals)
    assert rho.shape == (100,)
    assert torch.all(rho >= 0)

def test_conditional_and_expected():
    deg = 2
    coeffs = torch.eye(deg+1)
    basis = PolynomialBasis(degree=deg)
    u1_grid = torch.linspace(0,1,200)
    u2_scalar = 0.5
    p = conditional_density(u1_grid, u2_scalar, coeffs, basis, deg)
    # density integrates to 1
    area = torch.trapz(p, u1_grid)
    assert pytest.approx(area.item(), rel=1e-3) == 1.0
    # expected value lies in [0,1]
    exp = expected_u1_given_u2(u2_scalar, coeffs, basis, deg)
    assert 0.0 <= exp <= 1.0
