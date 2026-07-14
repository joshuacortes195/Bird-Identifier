"""Model factory: a timm backbone (as a pooled feature extractor) + a pluggable head.

The backbone is created with ``num_classes=0`` so it returns pooled features; the head
(from ``configs/head/``) maps those to logits. This keeps head-swapping a config change
and gives clean access to features for Grad-CAM later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import timm
import torch
from torch import nn

from wildlife.models.heads import build_head


@dataclass
class ModelConfig:
    backbone: str = "convnextv2_base.fcmae_ft_in22k_in1k"
    pretrained: bool = True
    head: str = "linear"
    head_kwargs: dict = field(default_factory=dict)
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1


class SpeciesClassifier(nn.Module):
    """Backbone feature extractor + classification head."""

    def __init__(self, backbone: nn.Module, head: nn.Module, num_features: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.num_features = num_features

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled feature vector (N, num_features)."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    @property
    def num_classes(self) -> int:
        return self.head.num_classes


def build_model(cfg: ModelConfig, num_classes: int) -> SpeciesClassifier:
    """Build a classifier for ``num_classes`` (from the taxonomy — never hardcoded)."""
    backbone = timm.create_model(
        cfg.backbone,
        pretrained=cfg.pretrained,
        num_classes=0,  # -> global-pooled features, no classifier
        drop_rate=cfg.drop_rate,
        drop_path_rate=cfg.drop_path_rate,
    )
    num_features = backbone.num_features
    head = build_head(
        cfg.head, in_features=num_features, num_classes=num_classes, **cfg.head_kwargs
    )
    return SpeciesClassifier(backbone, head, num_features)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
