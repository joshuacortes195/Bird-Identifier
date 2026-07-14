#!/usr/bin/env python
"""Run fine-grained ablations sequentially and build an ablation table (Phase 5).

Self-contained + resilient: each ablation trains, then evaluates on the NABirds test
set; results are appended to results/ablation_table.md after every run, so a crash or an
interruption never loses completed runs. Designed to run detached overnight — it does not
depend on the Claude agent staying awake.

    python scripts/run_ablations.py

Baseline (no crop, 224px) is already recorded from the earlier full run.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = str(REPO_ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(PY).exists():
    PY = sys.executable

# name -> (train overrides, eval overrides). Each isolates one lever vs the baseline.
ABLATIONS: list[tuple[str, list[str], list[str]]] = [
    (
        "bbox_crop_224",
        ["data.bbox_crop=true"],
        ["data.bbox_crop=true"],
    ),
    (
        "res288",
        ["data.transform.image_size=288", "data.batch_size=16", "train.grad_accum=4"],
        ["data.transform.image_size=288", "data.batch_size=32"],
    ),
]

COMMON = ["data=nabirds", "model=convnextv2_base", "train=baseline"]
TABLE = REPO_ROOT / "results" / "ablation_table.md"
LOG_DIR = REPO_ROOT / "outputs" / "ablations"
LOG_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_ROW = "| baseline (224px, no crop) | 0.8900 | 0.9878 | 0.869 | (reference) |"


def _write_table(rows: list[str]) -> None:
    TABLE.write_text(
        "# Ablation table — NABirds test set (ConvNeXt-V2-Base)\n\n"
        "Each row isolates one fine-grained lever vs. the baseline. Real numbers only;\n"
        "rows appear as each run finishes.\n\n"
        "| variant | top-1 | top-5 | macro-F1 | notes |\n"
        "|---------|------:|------:|---------:|-------|\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def _run(cmd: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=_utf8_env())
    return proc.returncode


def _utf8_env() -> dict:
    import os

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _parse_eval(log_path: Path) -> tuple[float, float, float] | None:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"top1=([\d.]+) top5=([\d.]+) macroF1=([\d.]+)", text)
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    return None


def main() -> int:
    rows = [BASELINE_ROW]
    _write_table(rows)

    for name, train_ov, eval_ov in ABLATIONS:
        t0 = time.time()
        run_name = f"abl_{name}"
        ckpt = REPO_ROOT / "outputs" / "checkpoints" / run_name / "best.pt"

        print(f"[ablation] TRAIN {name} ...", flush=True)
        train_cmd = [PY, "scripts/train.py", *COMMON, *train_ov, f"run_name={run_name}"]
        rc = _run(train_cmd, LOG_DIR / f"{name}.train.log")
        if rc != 0 or not ckpt.exists():
            rows.append(f"| {name} | — | — | — | TRAIN FAILED (rc={rc}) |")
            _write_table(rows)
            print(f"[ablation] {name} train failed rc={rc}", flush=True)
            continue

        print(f"[ablation] EVAL {name} ...", flush=True)
        eval_cmd = [
            PY,
            "scripts/evaluate.py",
            f"+checkpoint={ckpt}",
            *COMMON,
            *eval_ov,
        ]
        _run(eval_cmd, LOG_DIR / f"{name}.eval.log")
        parsed = _parse_eval(LOG_DIR / f"{name}.eval.log")
        mins = (time.time() - t0) / 60
        if parsed:
            t1, t5, f1 = parsed
            rows.append(f"| {name} | {t1:.4f} | {t5:.4f} | {f1:.3f} | {mins:.0f} min |")
        else:
            rows.append(f"| {name} | — | — | — | eval parse failed ({mins:.0f} min) |")
        _write_table(rows)
        print(f"[ablation] {name} done in {mins:.0f} min", flush=True)

    print("[ablation] ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
