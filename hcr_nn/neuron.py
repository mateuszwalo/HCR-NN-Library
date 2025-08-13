import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

__all__ = [
    'HCRNN_Neuron'
]

class Neuron(nn.Module):

    '''
    definicja stałych określająca rozmiar wejścia i wyjścia dla liczb neuronów.
    '''

    __constants__ = ['in_features', 'out_features']
    __methods__ = ['gaussian']
    in_features: int
    out_features: int

    def __init__(self):
        pass