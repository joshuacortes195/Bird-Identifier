"""Exponential moving average of model weights (improves + stabilizes eval).

Small self-contained implementation (avoids depending on a specific timm EMA API).
"""

from __future__ import annotations

import copy

import torch
from torch import nn


class ModelEma(nn.Module):
    def __init__(self, model: nn.Module, decay: float = 0.9998) -> None:
        super().__init__()
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for ema_p, p in zip(
            self.module.state_dict().values(), model.state_dict().values(), strict=True
        ):
            if ema_p.dtype.is_floating_point:
                ema_p.mul_(d).add_(p.detach(), alpha=1 - d)
            else:
                ema_p.copy_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)
