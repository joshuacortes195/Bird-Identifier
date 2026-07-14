"""wildlife.data — datasets, transforms, loaders, and the dataset registry.

Importing this package triggers dataset self-registration (see ``datasets``), so
``build_dataset("nabirds", ...)`` resolves without callers importing the class.

Registration depends on torch/torchvision. The serving path only needs the torch-free
submodules (``taxonomy``, ``imageio``), and its container ships without torch, so the
registration import is best-effort: if torch is unavailable it's skipped, and the pure
Python submodules stay importable. ``build_dataset`` still raises clearly if called
without a registered dataset.
"""

try:
    from wildlife.data import datasets as _datasets  # noqa: F401  (registers datasets)
except ImportError:  # torch/torchvision absent (e.g. lean serve image, CPU-x86 dev box)
    pass

from wildlife.data.registry import available_datasets, build_dataset, register_dataset

__all__ = ["available_datasets", "build_dataset", "register_dataset"]
