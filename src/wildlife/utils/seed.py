"""Reproducibility helpers.

Full bitwise determinism on CUDA is not always achievable (some cuDNN kernels have
no deterministic implementation); ``seed_everything(deterministic=True)`` opts into
deterministic algorithms where possible and warns rather than guaranteeing.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42, *, deterministic: bool = False) -> int:
    """Seed python, numpy and torch RNGs. Returns the seed for logging.

    Args:
        seed: The seed value applied to all RNGs.
        deterministic: If True, request deterministic CUDA/cuDNN algorithms. This can
            slow training and is not guaranteed for every op.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # warn_only: don't hard-crash on ops lacking a deterministic kernel.
            torch.use_deterministic_algorithms(True, warn_only=True)
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:
        # torch is optional for pure-data/utility contexts.
        pass

    return seed
