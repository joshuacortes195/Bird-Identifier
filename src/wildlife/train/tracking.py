"""Experiment tracking with a pluggable backend (local CSV/JSONL now; W&B in Phase 4).

The training loop only needs ``log(row, step)``, ``log_config(cfg)`` and ``finish()``.
The local backend has no external dependency and always works; W&B is optional and
falls back to local if the package/API key is missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wildlife.utils.logging import get_logger

log = get_logger("tracking")


class LocalTracker:
    """Append metrics to JSONL + a run config JSON under outputs/runs/<run_name>/."""

    def __init__(self, run_name: str, output_dir: str = "outputs") -> None:
        self.dir = Path(output_dir) / "runs" / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.dir / "metrics.jsonl"
        self._f = self.metrics_path.open("a", encoding="utf-8")

    def log_config(self, config: dict[str, Any]) -> None:
        (self.dir / "config.json").write_text(
            json.dumps(config, indent=2, default=str), encoding="utf-8"
        )

    def log(self, row: dict[str, Any], step: int | None = None) -> None:
        rec = {"step": step, **row}
        self._f.write(json.dumps(rec, default=str) + "\n")
        self._f.flush()

    def finish(self) -> None:
        if not self._f.closed:
            self._f.close()


class WandbTracker:
    """Weights & Biases backend; delegates a local mirror too. Falls back to local-only
    if wandb import or init fails.
    """

    def __init__(self, cfg, run_name: str) -> None:
        self.local = LocalTracker(run_name, cfg.output_dir)
        self.run = None
        try:
            import wandb

            self.run = wandb.init(
                project=cfg.tracking.project,
                entity=cfg.tracking.entity,
                name=run_name,
                reinit=True,
            )
            self._wandb = wandb
        except Exception as e:  # noqa: BLE001
            log.warning("W&B unavailable (%s); using local tracking only.", e)

    def log_config(self, config: dict[str, Any]) -> None:
        self.local.log_config(config)
        if self.run is not None:
            self.run.config.update(config, allow_val_change=True)

    def log(self, row: dict[str, Any], step: int | None = None) -> None:
        self.local.log(row, step)
        if self.run is not None:
            self.run.log(row, step=step)

    def finish(self) -> None:
        self.local.finish()
        if self.run is not None:
            self.run.finish()


def build_tracker(cfg, run_name: str):
    backend = str(cfg.tracking.backend).lower()
    if backend == "none":
        return None
    if backend == "wandb":
        return WandbTracker(cfg, run_name)
    return LocalTracker(run_name, cfg.output_dir)
