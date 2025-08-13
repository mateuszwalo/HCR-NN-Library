# tests/test_models.py
import pytest
import torch
import math

pytest.importorskip("scipy")  # polynomial basis may be used elsewhere

from hcr_nn.models import HCRCond2D, build_hcr_cond2d


def _rand_u(bsz, *, dtype=torch.float32, device="cpu"):
    # Random points already in quantile space [0,1]
    return torch.rand(bsz, 2, dtype=dtype, device=device)


def _kde_centers(n=8, *, dtype=torch.float32, device="cpu"):
    return torch.linspace(0, 1, n, dtype=dtype, device=device)


# ---------- Shapes & dtypes ----------

@pytest.mark.parametrize("basis", ["cosine", "polynomial"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_shapes_and_dtype(basis, dtype):
    # polynomial requires scipy; skip if basis not available elsewhere
    m = build_hcr_cond2d(degree=3, basis=basis, grid_size=129, dtype=dtype)
    x = _rand_u(17, dtype=dtype)
    y = m(x)
    assert y.shape == (17,)
    assert y.dtype == dtype


@pytest.mark.parametrize("basis", ["kde"])
def test_forward_shapes_kde(basis):
    m = build_hcr_cond2d(
        degree=0, basis=basis, grid_size=101,
        kde_centers=_kde_centers(9, dtype=torch.float64),
        kde_bandwidth=0.1, dtype=torch.float64
    )
    x = _rand_u(5, dtype=torch.float64)
    y = m(x)
    assert y.shape == (5,)
    assert torch.isfinite(y).all()


# ---------- Validation ----------

def test_forward_raises_on_bad_shape():
    m = build_hcr_cond2d(degree=2, basis="cosine", grid_size=64)
    with pytest.raises(ValueError):
        m(torch.rand(10, 3))   # wrong trailing dim
    with pytest.raises(ValueError):
        m(torch.rand(10))      # not 2D


def test_kde_requires_centers():
    with pytest.raises(ValueError):
        build_hcr_cond2d(degree=0, basis="kde", kde_centers=None)


# ---------- Normalization & expectation sanity ----------

@pytest.mark.parametrize("basis", ["cosine", "polynomial"])
def test_conditional_density_rows_integrate_to_one(basis):
    m = build_hcr_cond2d(degree=3, basis=basis, grid_size=1025, dtype=torch.float64)
    # Use u2 vector directly (1D) by calling the private method
    u2_vec = torch.linspace(0, 1, 7, dtype=torch.float64)
    p = m._conditional_density(u2_vec)  # (7, G)
    area = torch.trapz(p, m.u1_grid, dim=1)  # integrate along grid
    assert torch.allclose(area, torch.ones_like(area), atol=1e-2, rtol=1e-4)


@pytest.mark.parametrize("basis", ["cosine", "polynomial"])
def test_zero_coeffs_give_uniform_and_expectation_half(basis):
    m = build_hcr_cond2d(degree=2, basis=basis, grid_size=501, dtype=torch.float64)
    # Force coefficients to (almost) zeros
    with torch.no_grad():
        m.coeffs.zero_()
    x = _rand_u(8, dtype=torch.float64)
    y = m(x)
    # Expectation over uniform on [0,1] ≈ 0.5
    assert torch.allclose(y, torch.full_like(y, 0.5), atol=5e-3)


# ---------- quantile_fn behavior ----------

def test_quantile_fn_equivalence_to_prequantile():
    def clamp_quantile_fn(t):
        # Make sure it returns (B,2) in [0,1]
        return t.clamp(0, 1)

    m = build_hcr_cond2d(degree=3, basis="cosine", grid_size=129, dtype=torch.float64,
                         quantile_fn=clamp_quantile_fn)
    x_raw = torch.randn(9, 2, dtype=torch.float64)  # raw space (not necessarily in [0,1])
    y_raw = m(x_raw)
    y_pre = m(x_raw.clamp(0, 1))  # already in quantile space
    assert torch.allclose(y_raw, y_pre, atol=1e-9, rtol=0)


# ---------- conditional_curve helper ----------

def test_conditional_curve_shapes_and_finiteness():
    m = build_hcr_cond2d(degree=3, basis="cosine", grid_size=123, dtype=torch.float64)
    u2_grid = torch.linspace(0, 1, 111, dtype=torch.float64)
    e = m.conditional_curve(u2_grid)
    assert e.shape == (u2_grid.numel(),)
    assert torch.isfinite(e).all()
    # values should lie in [0,1] quantile space
    assert (e >= 0).all() and (e <= 1).all()


# ---------- device movement & gradients ----------

def test_grad_flow_to_coeffs():
    m = build_hcr_cond2d(degree=3, basis="cosine", grid_size=64, dtype=torch.float32)
    x = _rand_u(12, dtype=torch.float32).requires_grad_(True)
    y = m(x)
    loss = y.sum()
    loss.backward()
    assert m.coeffs.grad is not None
    assert torch.isfinite(m.coeffs.grad).all()


@pytest.mark.parametrize("basis", ["cosine"])
def test_device_cpu(basis):
    m = build_hcr_cond2d(degree=2, basis=basis, grid_size=64, dtype=torch.float32).to("cpu")
    x = _rand_u(4, dtype=torch.float32, device="cpu")
    y = m(x)
    assert y.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_cuda_roundtrip():
    m = build_hcr_cond2d(degree=2, basis="cosine", grid_size=64, dtype=torch.float32).to("cuda")
    x = _rand_u(4, dtype=torch.float32, device="cuda")
    y = m(x)
    assert y.device.type == "cuda"
    # Move back to CPU
    m = m.to("cpu")
    x_cpu = _rand_u(4, dtype=torch.float32, device="cpu")
    y2 = m(x_cpu)
    assert y2.device.type == "cpu"
