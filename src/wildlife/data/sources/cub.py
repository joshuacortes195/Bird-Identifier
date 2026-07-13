"""CUB-200-2011 — small, fast smoke-test dataset (200 classes, ~11.8k images).

Used to prove the pipeline before scaling to NABirds. Same file format as NABirds.
"""

from __future__ import annotations

from pathlib import Path

from wildlife.data.sources.cub_format import ParsedDataset, parse_cub_format
from wildlife.data.sources.download import (
    DownloadError,
    ensure_hf_dataset,
    extract_archive,
    find_dataset_root,
    try_download,
)
from wildlife.utils.logging import get_logger

log = get_logger(__name__)

# Candidate direct-download mirrors (tried in order). The canonical Caltech link is
# frequently offline, so mirrors are listed first.
CUB_URLS = [
    "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz",
    "https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz",
]
# Public HF mirror fallback (no auth for public repos).
CUB_HF_REPO = "efekankavalci/CUB-200-2011"


def download_cub(raw_dir: str | Path) -> Path:
    """Ensure CUB is downloaded+extracted under ``raw_dir``; return its dataset root."""
    raw_dir = Path(raw_dir)
    extract_dir = raw_dir / "cub200"
    try:
        return find_dataset_root(extract_dir)
    except DownloadError:
        pass

    archive = raw_dir / "CUB_200_2011.tgz"
    try:
        try_download(CUB_URLS, archive)
        extract_archive(archive, extract_dir)
    except DownloadError as e:
        log.warning("Direct CUB download failed (%s); trying HF mirror.", e)
        ensure_hf_dataset(CUB_HF_REPO, extract_dir)
    return find_dataset_root(extract_dir)


def prepare_cub(raw_dir: str | Path) -> ParsedDataset:
    root = download_cub(raw_dir)
    return parse_cub_format(root, taxonomy_name="cub200", supercategory="bird")
