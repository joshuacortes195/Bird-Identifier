"""Checkpoint IO. Every checkpoint embeds the config, git commit, and metrics so any
run is traceable back to exactly what produced it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import torch


def git_commit_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[3],
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[3],
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    ema: torch.nn.Module | None = None,
    epoch: int = 0,
    best_metric: float | None = None,
    metrics: dict | None = None,
    config: dict | None = None,
    class_names: list[str] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "metrics": metrics or {},
        "config": config or {},
        "git_commit": git_commit_hash(),
        "class_names": class_names,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None and hasattr(scaler, "state_dict"):
        payload["scaler"] = scaler.state_dict()
    if ema is not None:
        payload["ema"] = ema.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)
