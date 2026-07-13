"""Dataset registry — the extensibility seam for adding taxa/datasets by name.

Datasets self-register via the ``@register_dataset("name")`` decorator; training and
eval code resolve a dataset through :func:`build_dataset` and never hardcode paths or
dataset classes. Adding a new dataset (e.g. iNaturalist for mammals later) is a new
registered class + a config file — no edits to the training loop.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from wildlife.data.base import SpeciesDataset

_REGISTRY: dict[str, type] = {}

T = TypeVar("T")


def register_dataset(name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator registering a dataset under ``name`` (case-insensitive)."""

    key = name.lower()

    def _wrap(cls: type[T]) -> type[T]:
        if key in _REGISTRY:
            raise ValueError(f"Dataset '{name}' is already registered by {_REGISTRY[key]!r}")
        _REGISTRY[key] = cls
        return cls

    return _wrap


def available_datasets() -> list[str]:
    return sorted(_REGISTRY)


def get_dataset_class(name: str) -> type:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown dataset '{name}'. Registered: {available_datasets() or '(none)'}. "
            "Did you forget to import its module so it can self-register?"
        )
    return _REGISTRY[key]


def build_dataset(name: str, **kwargs: Any) -> SpeciesDataset:
    """Instantiate a registered dataset by name."""
    return get_dataset_class(name)(**kwargs)
