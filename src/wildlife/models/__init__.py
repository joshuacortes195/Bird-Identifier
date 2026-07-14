"""wildlife.models — model factory + pluggable heads."""

from wildlife.models.factory import ModelConfig, SpeciesClassifier, build_model, count_parameters
from wildlife.models.heads import available_heads, build_head, register_head

__all__ = [
    "ModelConfig",
    "SpeciesClassifier",
    "available_heads",
    "build_head",
    "build_model",
    "count_parameters",
    "register_head",
]
