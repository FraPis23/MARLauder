"""Welford online mean/variance — ported from TOM/training/value_normalizer.py to torch.

Critic predicts normalized values; GAE uses denormalized; MSE target uses normalized returns.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ValueNormalizer(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("var", torch.ones(1))
        self.register_buffer("count", torch.zeros(1))
        self.eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.detach().reshape(-1).float()
        bn = float(x.numel())
        if bn == 0:
            return
        bm = x.mean()
        bv = x.var(unbiased=False)
        total = self.count + bn
        delta = bm - self.mean
        new_mean = self.mean + delta * bn / total
        m_a = self.var * self.count
        m_b = bv * bn
        m2 = m_a + m_b + delta * delta * self.count * bn / total
        self.var.copy_(m2 / total)
        self.mean.copy_(new_mean)
        self.count.copy_(total)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var.clamp(min=self.eps).sqrt())

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.var.clamp(min=self.eps).sqrt() + self.mean
