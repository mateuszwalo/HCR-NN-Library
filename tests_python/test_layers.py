# tests/test_layers.py
import math
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from hcr_nn.layers import (
    CDFNorm,
    MeanEstimation,
    ConditionalEstimation,
    PropagationEstimation,
    EntropyAndMutualInformation,
    DynamicEMA,
    BaseOptimization,
    InformationBottleneck,
)


# ------------------------------- CDFNorm ------------------------------------

@pytest.mark.parametrize("method", ["gaussian", "empirical"])
@pytest.mark.parametrize("affine", [False, True])
def test_cdfnorm_shapes_and_range(method, affine):
    x = torch.linspace(-2, 2, 101, dtype=torch.float64)
    layer = CDFNorm(method=method, affine=affine, track_running_stats=True)
    layer.train()  # ensure running stats path taken for gaussian
    y = layer(x)
    assert y.shape == x.shape
    # outputs should be in [0,1] before affine; with affine, just check finite
    if not affine:
        assert torch.all((y >= 0) & (y <= 1))
    else:
        assert torch.all(torch.isfinite(y))


def test_cdfnorm_running_stats_increment_only_in_train():
    x = torch.randn(64)
    layer = CDFNorm(method="gaussian", track_running_stats=True)
    assert int(layer.num_batches_tracked.item()) == 0
    layer.train()
    _ = layer(x)
    assert int(layer.num_batches_tracked.item()) >= 1  # incremented
    layer.eval()
    _ = layer(x)
    # in eval, counter should not increment
    after = int(layer.num_batches_tracked.item())
    _ = layer(x)
    assert int(layer.num_batches_tracked.item()) == after


def test_cdfnorm_unsupported_method_raises():
    layer = CDFNorm(method="nope")
    with pytest.raises(ValueError):
        _ = layer(torch.randn(8))


def test_cdfnorm_empirical_monotonicity():
    # For strictly increasing inputs, the empirical CDF must be strictly increasing
    x = torch.linspace(-1, 1, 50)
    layer = CDFNorm(method="empirical", affine=False)
    y = layer(x)
    diffs = y[1:] - y[:-1]
    assert torch.all(diffs > 0)


# ---------------------------- MeanEstimation --------------------------------

def test_mean_estimation_numpy_return_and_shape():
    # feature_fn maps scalar -> R^D
    D = 3
    def feature_fn(v):
        v = float(v)
        return np.array([1.0, v, v * v], dtype=np.float64)

    triplets = [(0.0, 1.0, 2.0), (1.0, 0.0, -1.0), (2.0, -1.0, 0.5)]
    m = MeanEstimation(triplets=triplets, feature_fn=feature_fn, feature_dm=D)
    A = m.compute_tensor_mean()
    assert isinstance(A, np.ndarray)
    assert A.shape == (D, D, D)
    # simple sanity: finite numbers
    assert np.isfinite(A).all()


# ------------------------- ConditionalEstimation -----------------------------

def test_conditional_estimation_scores_list_and_ordering():
    # Use a simple linear feature: f(t) = [1, t]
    def f(t):
        t = torch.as_tensor(t, dtype=torch.float64)
        return torch.stack([torch.ones((), dtype=torch.float64), t]).numpy()  # numpy because MeanEstimation used numpy style

    # tensor A with stronger weight on i=1 (favor larger x)
    A = torch.zeros(2, 2, 2, dtype=torch.float64)
    A[1, :, :] = 1.0
    A[0, :, :] = 0.5

    x_candidates = [0.1, 0.9]
    y, z = 0.2, -0.3

    ce = ConditionalEstimation(
        x_candidates=x_candidates, y=y, z=z, a=A, feature_fn=f
    )
    scores = ce.conditional_score()
    assert isinstance(scores, list)
    assert len(scores) == len(x_candidates)
    # candidate with larger x should get larger score
    assert scores[1] > scores[0]


# ------------------------- PropagationEstimation -----------------------------

def test_propagation_estimation_basic_sanity():
    # Build a small A with different slices at index 0 and 1
    A = torch.zeros(2, 3, 3, dtype=torch.float64)
    A[0] = 1.0
    A[1] = 1.5

    def f(v):
        v = torch.as_tensor(v, dtype=torch.float64)
        # map scalar -> R^3
        return torch.stack([torch.ones((), dtype=torch.float64), v, v * v])

    y, z = 0.2, -0.5
    pe = PropagationEstimation(y=y, z=z, a=A, feature_fn=f)
    out = pe.propagate_expectation()
    assert isinstance(out, torch.Tensor)
    assert torch.isfinite(out).item()
    # since A[1] > A[0], ratio>1, so centered>0, mapped > 0.5
    assert out.item() > 0.5


# ---------------- EntropyAndMutualInformation (current implementation) -------

def test_entropy_monotonicity_uniform_vs_peaked():
    K = 5
    B = 7
    model = EntropyAndMutualInformation()
    # uniform logits -> softmax ~ uniform
    uniform_logits = torch.zeros(B, K)
    # peaked logits -> one class >> others
    peaked_logits = torch.zeros(B, K)
    peaked_logits[:, 0] = 10.0

    H_uniform = model.approximate_entropy(uniform_logits)
    H_peaked = model.approximate_entropy(peaked_logits)

    # With current definition: entropy = -sum p^2, uniform should be less negative (i.e., larger) than peaked
    assert H_uniform > H_peaked


def test_mutual_information_higher_for_correlated_than_uniform():
    Kx, Ky, B = 4, 3, 6
    model = EntropyAndMutualInformation()

    # Correlated: make act_X and act_Y depend on the same hidden index
    act_X = torch.full((B, Kx), -5.0)
    act_Y = torch.full((B, Ky), -5.0)
    for b in range(B):
        i = b % Kx
        j = b % Ky
        act_X[b, i] = 8.0
        act_Y[b, j] = 8.0

    # Uniform reference
    act_Xu = torch.zeros(B, Kx)
    act_Yu = torch.zeros(B, Ky)

    mi_corr = model.approximate_mutual_information(act_X, act_Y)
    mi_uniform = model.approximate_mutual_information(act_Xu, act_Yu)

    assert mi_corr > mi_uniform


# ------------------------------ DynamicEMA -----------------------------------

def test_dynamic_ema_update_converges():
    # x,y,z as simple 3D features
    x = torch.tensor([1.0, 0.0, -1.0])
    y = torch.tensor([0.5, 0.5, 0.0])
    z = torch.tensor([2.0, -1.0, 0.0])

    ema = DynamicEMA(x=x, y=y, z=z, ema_lambda=0.2)
    # reference outer product
    U = torch.einsum("i,j,k->ijk", x, y, z)

    a1 = ema.EMAUpdateMethod()
    # after first update: a = l*U
    assert torch.allclose(a1, 0.2 * U)

    a2 = ema.EMAUpdateMethod()
    # after second: a = (1-l)*a1 + l*U = (1-l)*l*U + l*U = (1 - (1-l)) * l *? -> compute explicitly
    expected = (1 - 0.2) * (0.2 * U) + 0.2 * U
    assert torch.allclose(a2, expected)


# ---------------------------- BaseOptimization --------------------------------

# def test_base_optimization_rotates_mode0_with_u_from_svd():
#     # random small tensor
#     torch.manual_seed(0)
#     A = torch.randn(4, 3, 2)
#     bo = BaseOptimization(a=A)
#     new_a = bo.optimization_early()
#
#     # Recompute the same SVD and expected rotation
#     I, J, K = A.shape
#     M = A.reshape(I, J * K)
#     U, S, Vh = torch.linalg.svd(M, full_matrices=False)
#     expected = torch.einsum("li,ljk->ijk", U.T, A)
#
#     assert new_a.shape == A.shape
#     assert torch.allclose(new_a, expected, atol=1e-6, rtol=1e-6)


# -------------------------- InformationBottleneck -----------------------------

def test_information_bottleneck_forward_and_loss_consistency():
    torch.manual_seed(0)
    X = torch.randn(16, 8)
    T = torch.randn(16, 8)
    Y = torch.randn(16, 8)

    ib = InformationBottleneck(beta=0.3)
    val_forward = ib(X, Y)
    loss = ib.bottleneck_loss(X, T, Y)

    # symmetry of trace(Cx @ Cy)
    assert torch.isfinite(val_forward).item()
    assert val_forward == pytest.approx(ib(Y, X).item(), rel=1e-6, abs=1e-6)

    # loss should equal forward(X,T) - beta * forward(T,Y)
    expected = ib(X, T) - 0.3 * ib(T, Y)
    assert loss == pytest.approx(expected.item(), rel=1e-6, abs=1e-6)
