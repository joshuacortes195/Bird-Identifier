"""Pluggable classification heads (extensibility seam #4).

A ``Head`` is any module mapping pooled backbone features -> class logits. Heads are
registered by name and selected via ``configs/head/``; swapping heads is a config
change, not a refactor. ``LinearHead`` ships now. ``HierarchicalHead`` is a documented
stub (interface only) for the future coarse-router -> per-group-specialist design and
is intentionally NOT implemented in this bird build.

The head's output dimension always comes from the taxonomy's ``num_classes`` — never a
hardcoded 555.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import torch
from torch import nn

_HEAD_REGISTRY: dict[str, type[nn.Module]] = {}
T = TypeVar("T", bound=type[nn.Module])


def register_head(name: str) -> Callable[[T], T]:
    key = name.lower()

    def _wrap(cls: T) -> T:
        if key in _HEAD_REGISTRY:
            raise ValueError(f"Head '{name}' already registered by {_HEAD_REGISTRY[key]!r}")
        _HEAD_REGISTRY[key] = cls
        return cls

    return _wrap


def available_heads() -> list[str]:
    return sorted(_HEAD_REGISTRY)


def build_head(name: str, in_features: int, num_classes: int, **kwargs) -> nn.Module:
    key = name.lower()
    if key not in _HEAD_REGISTRY:
        raise KeyError(f"Unknown head '{name}'. Registered: {available_heads()}")
    return _HEAD_REGISTRY[key](in_features=in_features, num_classes=num_classes, **kwargs)


class ClassifierHead(nn.Module):
    """Base head: knows its feature width and class count."""

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

    def forward(self, feats: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError


@register_head("linear")
class LinearHead(ClassifierHead):
    """Dropout + single linear layer over pooled features."""

    def __init__(self, in_features: int, num_classes: int, *, dropout: float = 0.0) -> None:
        super().__init__(in_features, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(feats))


@register_head("hierarchical")
class HierarchicalHead(ClassifierHead):
    """STUB (future seam) — coarse supercategory router -> per-group specialist heads.

    Interface is defined so the model factory already supports selecting it via config;
    the implementation is intentionally omitted in the bird build (a single supercategory
    makes it a no-op today). Implementing it later requires no factory/training changes.
    """

    def __init__(
        self, in_features: int, num_classes: int, *, num_supercategories: int = 1, **_
    ) -> None:
        super().__init__(in_features, num_classes)
        raise NotImplementedError(
            "HierarchicalHead is a documented future seam (see docs/EXTENSIBILITY.md) and "
            "is not implemented in the bird build. Use head=linear."
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:  # pragma: no cover - stub
        raise NotImplementedError
