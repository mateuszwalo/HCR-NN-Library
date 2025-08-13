# tests/test_basis.py
import math
import pytest
import torch

pytest.importorskip("scipy")  # PolynomialBasis uses scipy.special.eval_legendre

from hcr_nn.basis import PolynomialBasis, CosineBasis, KDEBasis, select_basis


def _rand_u(shape, dtype=torch.float32):
    """Random uniform sampling in [0,1] — needed for orthonormality tests."""
    return torch.rand(shape, dtype=dtype)


# ---------- Shapes & dtypes ----------

@pytest.mark.parametrize("degree", [0, 1, 3])
@pytest.mark.parametrize("shape", [(10,), (4, 5), (2, 3, 7)])
def test_shapes_cosine(degree, shape):
    u = _rand_u(shape)
    basis = CosineBasis(degree)
    F = basis(u)
    assert F.shape == (*shape, degree + 1)
    assert F.dtype == u.dtype


@pytest.mark.parametrize("degree", [0, 1, 3])
@pytest.mark.parametrize("shape", [(10,), (4, 5)])
def test_shapes_polynomial(degree, shape):
    u = _rand_u(shape)
    basis = PolynomialBasis(degree)
    F = basis(u)
    assert F.shape == (*shape, degree + 1)
    assert F.dtype == u.dtype


@pytest.mark.parametrize("n_centers", [1, 5, 12])
@pytest.mark.parametrize("shape", [(8,), (3, 4)])
def test_shapes_kde(n_centers, shape):
    u = _rand_u(shape)
    centers = torch.linspace(0, 1, n_centers)
    basis = KDEBasis(centers, bandwidth=0.1)
    K = basis(u)
    assert K.shape == (*shape, n_centers)
    assert K.dtype == u.dtype


# ---------- Orthonormality on [0,1] (Monte Carlo) ----------

@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_cosine_orthonormality_mc(degree):
    N = 20000  # fast and stable on CPU
    u = _rand_u((N,))
    F = CosineBasis(degree)(u)  # (N, d+1)
    G = (F.T @ F) / N  # ~ I
    I = torch.eye(degree + 1, dtype=F.dtype)
    assert torch.allclose(G, I, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_polynomial_orthonormality_mc(degree):
    N = 20000
    u = _rand_u((N,))
    F = PolynomialBasis(degree)(u)
    G = (F.T @ F) / N
    I = torch.eye(degree + 1, dtype=F.dtype)
    # Slightly looser tolerance due to CPU/NumPy transitions and scaling
    assert torch.allclose(G, I, rtol=8e-2, atol=8e-2)


# ---------- KDE sanity checks ----------

def test_kde_nonnegative_and_center_peaks():
    centers = torch.tensor([0.2, 0.5, 0.8])
    bw = 0.05
    basis = KDEBasis(centers, bandwidth=bw)

    # Values at centers should be >= values at points far away
    u = torch.tensor([0.2, 0.5, 0.8, 0.2 + 4*bw, 0.5 + 4*bw, 0.8 + 4*bw])
    K = basis(u)  # (6, 3)

    assert torch.all(K >= 0)

    # Rows 0/1/2 are values exactly at the centers
    # Rows 3/4/5 — same columns, but shifted by 4*sigma (smaller values expected)
    assert torch.all(K[0, 0] > K[3, 0])
    assert torch.all(K[1, 1] > K[4, 1])
    assert torch.all(K[2, 2] > K[5, 2])


# ---------- Factory & validation ----------

def test_select_basis_factory():
    poly = select_basis("polynomial", degree=3)
    cos = select_basis("cosine", degree=2)
    kde = select_basis("kde", degree=0, centers=torch.linspace(0, 1, 5), bandwidth=0.1)
    assert isinstance(poly, PolynomialBasis)
    assert isinstance(cos, CosineBasis)
    assert isinstance(kde, KDEBasis)


def test_select_basis_errors():
    with pytest.raises(ValueError):
        select_basis("polynomial", degree=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        select_basis("cosine", degree=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        select_basis("kde", degree=0)  # missing centers


def test_kde_validation_errors():
    with pytest.raises(ValueError):
        KDEBasis(torch.tensor([]), bandwidth=0.1)
    with pytest.raises(ValueError):
        KDEBasis(torch.tensor([0.1, 0.2]), bandwidth=0.0)
    with pytest.raises(ValueError):
        KDEBasis(torch.tensor([0.1, 0.2]), bandwidth=-1.0)


# ---------- Grad flow (autograd should pass through inputs) ----------

def test_grad_flow_through_inputs():
    u = _rand_u((32,), dtype=torch.float64).requires_grad_(True)
    F1 = CosineBasis(3)(u).sum()
    F2 = PolynomialBasis(3)(u).sum()
    K = KDEBasis(torch.linspace(0, 1, 7), bandwidth=0.1)(u).sum()

    total = F1 + F2 + K
    total.backward()
    assert u.grad is not None
    # Gradient should not be NaN
    assert torch.all(torch.isfinite(u.grad))


# ---------- Analytical orthonormality tests ----------

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
    u = torch.linspace(0, 1, 200)
    F = basis(u)
    assert F.shape == (200, 50)
    assert torch.all(F >= 0), "KDE basis must produce non-negative values"
