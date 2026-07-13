"""Name -> source-preparer mapping used by scripts/download_data.py.

Kept torch-free so data acquisition needs no torch. The torch ``Dataset`` classes
(Phase 2) register in ``wildlife.data.registry`` and read the manifests produced here.
"""

from __future__ import annotations

from collections.abc import Callable

from wildlife.data.sources.cub import prepare_cub
from wildlife.data.sources.cub_format import ParsedDataset
from wildlife.data.sources.inat import prepare_inat
from wildlife.data.sources.nabirds import prepare_nabirds

# dataset name -> (preparer, taxonomy-config-filename)
SOURCE_PREPARERS: dict[str, Callable[..., ParsedDataset]] = {
    "nabirds": prepare_nabirds,
    "cub": prepare_cub,
    "inat": prepare_inat,  # stub — raises NotImplementedError
}

TAXONOMY_CONFIG_NAME: dict[str, str] = {
    "nabirds": "birds.yaml",  # NABirds defines the canonical bird label space
    "cub": "cub.yaml",
}


def available_sources() -> list[str]:
    return sorted(SOURCE_PREPARERS)
