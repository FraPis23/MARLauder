"""Orthogonal weight init — MAPPO paper (Yu et al. 2022), Table 7:
'network initialization: Orthogonal'.

Orthogonal weights keep the forward activation norm and the backprop gradient
norm stable across network depth and GRU recurrence (W is norm-preserving:
||Wx|| == ||x||), avoiding vanishing/exploding signal that the torch-default
Kaiming-uniform init does not guard against in recurrent actor-critic nets.

Convention:
    hidden layers   gain = sqrt(2)   (ReLU/GELU compensation)
    policy output   gain ~ 0.01      (start near-uniform → healthy exploration)
    value output    gain = 1.0
"""
from __future__ import annotations

import torch.nn as nn

SQRT2 = 2.0 ** 0.5


def orthogonal_(module: nn.Module, gain: float = SQRT2) -> nn.Module:
    """In-place orthogonal init for a single Linear or GRUCell (biases → 0)."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.GRUCell):
        nn.init.orthogonal_(module.weight_ih, gain)
        nn.init.orthogonal_(module.weight_hh, gain)
        if module.bias_ih is not None:
            nn.init.zeros_(module.bias_ih)
            nn.init.zeros_(module.bias_hh)
    return module


def apply_orthogonal(root: nn.Module, gain: float = SQRT2) -> None:
    """Recursively apply orthogonal_ to every Linear / GRUCell under `root`.

    Apply this first, then override specific output heads with a smaller gain.
    """
    root.apply(lambda m: orthogonal_(m, gain))
