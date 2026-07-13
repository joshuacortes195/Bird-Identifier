"""Robust download + extract helpers with resume and archive handling.

Datasets are large and mirrors move, so downloads try a list of candidate URLs and
fall back to clear manual instructions rather than failing opaquely. Nothing here is
dataset-specific.
"""

from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from wildlife.utils.logging import get_logger

log = get_logger(__name__)


class DownloadError(RuntimeError):
    pass


def download_file(url: str, dest: str | Path, *, chunk_mb: int = 4, timeout: int = 30) -> Path:
    """Stream ``url`` to ``dest`` with a progress bar and simple resume support."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    resume_from = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

    with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
        if r.status_code in (200, 206):
            total = int(r.headers.get("content-length", 0)) + resume_from
            mode = "ab" if r.status_code == 206 and resume_from else "wb"
            with (
                tmp.open(mode) as f,
                tqdm(
                    total=total or None,
                    initial=resume_from if mode == "ab" else 0,
                    unit="B",
                    unit_scale=True,
                    desc=dest.name,
                ) as bar,
            ):
                for chunk in r.iter_content(chunk_size=chunk_mb * 1024 * 1024):
                    f.write(chunk)
                    bar.update(len(chunk))
        else:
            raise DownloadError(f"HTTP {r.status_code} for {url}")

    tmp.rename(dest)
    return dest


def try_download(urls: list[str], dest: str | Path) -> Path:
    """Try each candidate URL in turn; raise with guidance if all fail."""
    dest = Path(dest)
    if dest.exists():
        log.info("Archive already present: %s", dest)
        return dest
    errors = []
    for url in urls:
        try:
            log.info("Downloading %s", url)
            return download_file(url, dest)
        except Exception as e:  # noqa: BLE001 - try next mirror
            log.warning("Failed (%s): %s", url, e)
            errors.append(f"{url} -> {e}")
    raise DownloadError(
        "All candidate URLs failed:\n  " + "\n  ".join(errors) + f"\nPlace the archive manually at: {dest}"
    )


def extract_archive(archive: str | Path, dest: str | Path) -> Path:
    """Extract a .tar.gz/.tgz/.tar/.zip into ``dest``."""
    archive, dest = Path(archive), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    log.info("Extracting %s -> %s", archive.name, dest)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    elif archive.name.endswith((".tar.gz", ".tgz", ".tar")):
        mode = "r:gz" if archive.name.endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(archive, mode) as t:
            t.extractall(dest)  # noqa: S202 - trusted dataset archives
    else:
        raise DownloadError(f"Unknown archive type: {archive.name}")
    return dest


def find_dataset_root(search_dir: str | Path, marker: str = "images.txt") -> Path:
    """Locate the directory containing ``marker`` (archives nest under a top folder)."""
    search_dir = Path(search_dir)
    if (search_dir / marker).exists():
        return search_dir
    matches = list(search_dir.rglob(marker))
    if not matches:
        raise DownloadError(f"Could not find {marker} under {search_dir}")
    return matches[0].parent


def ensure_hf_dataset(repo_id: str, dest: str | Path, *, repo_type: str = "dataset") -> Path:
    """Snapshot-download a Hugging Face repo (no auth needed for public repos)."""
    from huggingface_hub import snapshot_download

    dest = Path(dest)
    log.info("Fetching HF %s repo '%s'", repo_type, repo_id)
    path = snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=str(dest))
    return Path(path)


def ensure_hf_file(repo_id: str, filename: str, dest: str | Path, *, repo_type: str = "dataset") -> Path:
    """Download a single file from a public HF repo. Returns the local path."""
    from huggingface_hub import hf_hub_download

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    log.info("Fetching HF %s '%s' from '%s'", repo_type, filename, repo_id)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        local_dir=str(dest),
    )
    return Path(path)


def safe_rmtree(path: str | Path) -> None:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
