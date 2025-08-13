# hcr_nn/neuron.py
from __future__ import annotations
from typing import Optional
import math
import torch
import torch.nn as nn
from torch import Tensor


__all__ = ["HCRNeuron"]


class HCRNeuron(nn.Module):
    """
    HCR neuron: applies a 1D basis expansion F(u) followed by a linear map.

    Given u in [0,1], a basis module produces F(u) in R^{B}.
    This neuron computes y = W @ F(u) + b, where W in R^{out_features x B}.

    Args:
        basis:      nn.Module mapping (...,) -> (..., B)
        out_features: number of output channels (>=1)
        bias:       whether to include bias term
    Input:
        u: Tensor of shape (...,) with values typically in [0,1]
    Output:
        y: Tensor of shape (..., out_features)
    """

    __constants__ = ["out_features", "basis_dim", "has_bias"]

    def __init__(self, basis: nn.Module, out_features: int, bias: bool = True):
        super().__init__()
        if not isinstance(out_features, int) or out_features <= 0:
            raise ValueError(f"out_features must be a positive int, got {out_features!r}.")

        self.basis = basis
        self.out_features = out_features

        # Probe basis to determine basis_dim (B)
        with torch.no_grad():
            probe = torch.zeros(1)
            B = int(self.basis(probe).shape[-1])
        self.basis_dim = B

        # Parameters: W ∈ R^{out_features x B}, b ∈ R^{out_features}
        self.weight = nn.Parameter(torch.empty(out_features, B))
        self.has_bias = bool(bias)
        if self.has_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Kaiming init for linear over features, zero bias
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # type: ignore[name-defined]
        if self.has_bias:
            nn.init.zeros_(self.bias)

    def forward(self, u: Tensor) -> Tensor:
        F = self.basis(u)  # (..., B)
        # cast params to input dtype/device for safe matmul
        W = self.weight.to(dtype=F.dtype, device=F.device)
        Y = torch.einsum("...b,ob->...o", F, W)
        if self.bias is not None:
            b = self.bias.to(dtype=F.dtype, device=F.device)
            Y = Y + b
        return Y

    def extra_repr(self) -> str:
        return f"basis_dim={self.basis_dim}, out_features={self.out_features}, bias={self.has_bias}"
