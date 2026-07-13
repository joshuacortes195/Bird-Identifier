"""wildlife.data — datasets, transforms, loaders, and the dataset registry.

Importing this package triggers dataset self-registration (see ``datasets``), so
``build_dataset("nabirds", ...)`` resolves without callers importing the class.
"""

from wildlife.data import datasets as _datasets  # noqa: F401  (registers datasets)
from wildlife.data.registry import available_datasets, build_dataset, register_dataset

__all__ = ["available_datasets", "build_dataset", "register_dataset"]
