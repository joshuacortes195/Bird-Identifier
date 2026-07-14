"""Optimizer + LR schedule construction.

Supports plain AdamW now and layer-wise LR decay (discriminative LRs) as a config
option for Phase 5. The scheduler is cosine with linear warmup, implemented directly
so it works identically on any platform.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    layer_decay: float | None = None  # e.g. 0.75 enables layer-wise LR decay (Phase 5)
    warmup_epochs: float = 1.0
    min_lr_ratio: float = 0.01  # final LR = lr * min_lr_ratio


def _no_decay(name: str, param: nn.Parameter) -> bool:
    return param.ndim <= 1 or name.endswith(".bias")


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    """AdamW with weight-decay disabled on norms/biases (standard for transformers/convnext)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if _no_decay(name, p) else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas)


class CosineWarmupScheduler:
    """Per-step linear warmup then cosine decay to ``lr * min_lr_ratio``.

    Kept explicit (rather than a timm/torch scheduler) so behavior is identical across
    machines and trivially unit-testable.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.01,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(0, warmup_steps)
        self.total_steps = max(total_steps, self.warmup_steps + 1)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self._step = 0
        self.step(0)

    def _scale(self, step: int) -> float:
        if step < self.warmup_steps:
            return (step + 1) / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine

    def step(self, step: int | None = None) -> None:
        if step is None:
            step = self._step
            self._step += 1
        scale = self._scale(step)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base * scale

    def get_last_lr(self) -> list[float]:
        return [g["lr"] for g in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {"_step": self._step, "base_lrs": self.base_lrs}

    def load_state_dict(self, state: dict) -> None:
        self._step = state["_step"]
        self.base_lrs = state["base_lrs"]
