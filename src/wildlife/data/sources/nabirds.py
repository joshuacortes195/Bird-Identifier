"""NABirds — primary dataset (~48.5k images, 555 visual categories, with bounding boxes).

The official dataset (Cornell Lab) requires accepting an agreement, so it can't be
fetched fully unattended. This module tries public mirrors first and otherwise prints
exact manual-placement instructions. Once the files are present the parse is identical
to CUB (shared file format).
"""

from __future__ import annotations

from pathlib import Path

from wildlife.data.sources.cub_format import ParsedDataset, parse_cub_format
from wildlife.data.sources.download import (
    DownloadError,
    ensure_hf_file,
    extract_archive,
    find_dataset_root,
    try_download,
)
from wildlife.utils.logging import get_logger

log = get_logger(__name__)

# NABirds mirrors move often; these are attempted in order. Update as needed.
NABIRDS_URLS = [
    "https://thijs.ai/data/nabirds.tar.gz",
    "https://s3.amazonaws.com/fast-ai-imageclas/nabirds.tgz",
]
# Public HF single-file mirrors (verified layout: images.txt/bounding_boxes.txt/...).
# Tried in order: (repo_id, filename).
NABIRDS_HF_FILES = [
    ("QianC95/NABirds", "NABirds.tar.gz"),
    ("antokun/nabirds", "nabirds.zip"),
]

MANUAL_INSTRUCTIONS = """
NABirds could not be downloaded automatically (it requires accepting Cornell's
agreement). To proceed:

  1. Request access + download 'nabirds.tar.gz' from:
       https://dl.allaboutbirds.org/nabirds
     (or a Kaggle mirror, e.g. search 'NABirds' on kaggle.com/datasets)
  2. Place the archive at:  {archive}
     OR extract it so this file exists:  {extract_dir}/nabirds/images.txt
  3. Re-run:  python scripts/download_data.py --dataset nabirds

Everything else (parsing, splits, taxonomy) is automated once the files are present.
"""


def download_nabirds(raw_dir: str | Path) -> Path:
    raw_dir = Path(raw_dir)
    extract_dir = raw_dir / "nabirds"
    try:
        return find_dataset_root(extract_dir)
    except DownloadError:
        pass

    archive = raw_dir / "nabirds.tar.gz"
    if archive.exists():
        extract_archive(archive, extract_dir)
        return find_dataset_root(extract_dir)

    try:
        try_download(NABIRDS_URLS, archive)
        extract_archive(archive, extract_dir)
        return find_dataset_root(extract_dir)
    except DownloadError as e:
        log.warning("Direct NABirds download failed (%s); trying HF mirrors.", e)

    for repo_id, filename in NABIRDS_HF_FILES:
        try:
            local = ensure_hf_file(repo_id, filename, raw_dir / "nabirds_hf")
            extract_archive(local, extract_dir)
            return find_dataset_root(extract_dir)
        except Exception as e:  # noqa: BLE001 - try next mirror
            log.warning("HF mirror %s/%s failed: %s", repo_id, filename, e)

    raise DownloadError(MANUAL_INSTRUCTIONS.format(archive=archive, extract_dir=extract_dir))


def prepare_nabirds(raw_dir: str | Path) -> ParsedDataset:
    root = download_nabirds(raw_dir)
    # taxonomy_name="birds": NABirds *defines* the bird label space -> configs/taxonomy/birds.yaml.
    # A future dataset (iNat mammals) would extend/merge into a broader animals taxonomy.
    return parse_cub_format(root, taxonomy_name="birds", supercategory="bird")
