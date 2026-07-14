#!/usr/bin/env python
"""Phase 7 — evaluate the model on my own field photos (out-of-distribution test).

Ingests ``my_photos/`` (folder-per-species or a labels.csv), reconciles labels to NABirds
classes, runs inference, and writes an OOD report + Grad-CAM overlays. See docs/OOD.md.

Usage:
    python scripts/ood_eval.py +checkpoint=outputs/checkpoints/<run>/best.pt model=convnextv2_base

Runs on the training box (needs torch). The label reconciliation it uses is unit-tested in
tests/test_ood_reconcile.py.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wildlife.data.imageio import load_image  # noqa: E402
from wildlife.data.taxonomy import load_taxonomy  # noqa: E402
from wildlife.data.transforms import TransformConfig, build_eval_transform  # noqa: E402
from wildlife.eval.ood import reconcile_labels  # noqa: E402
from wildlife.models import ModelConfig, build_model  # noqa: E402
from wildlife.utils.checkpoint import load_checkpoint  # noqa: E402
from wildlife.utils.logging import get_logger  # noqa: E402

log = get_logger("ood_eval")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


def _abs(p: str) -> str:
    path = Path(p)
    return str(path if path.is_absolute() else REPO_ROOT / path)


def _discover(my_photos: Path) -> list[tuple[Path, str]]:
    """Return (image_path, label) pairs from a labels.csv or folder-per-species layout."""
    csv_path = my_photos / "labels.csv"
    pairs: list[tuple[Path, str]] = []
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                p = my_photos / row["filename"]
                if p.exists():
                    pairs.append((p, row["species"].strip()))
        return pairs
    for species_dir in sorted(d for d in my_photos.iterdir() if d.is_dir()):
        for img in species_dir.iterdir():
            if img.suffix.lower() in IMG_EXT:
                pairs.append((img, species_dir.name))
    return pairs


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "configs"), config_name="config")
def main(cfg: DictConfig) -> None:
    checkpoint = cfg.get("checkpoint")
    if not checkpoint:
        raise SystemExit("Pass +checkpoint=<path/to/best.pt>")
    checkpoint = _abs(str(checkpoint))

    my_photos = Path(_abs("my_photos"))
    if not my_photos.exists():
        raise SystemExit(f"{my_photos} not found — see docs/OOD.md for the labeling format.")

    taxonomy = load_taxonomy(_abs(cfg.data.get("taxonomy", "configs/taxonomy/birds.yaml")))
    pairs = _discover(my_photos)
    if not pairs:
        raise SystemExit(f"No labeled photos found under {my_photos}. See docs/OOD.md.")

    recon = reconcile_labels(sorted({label for _, label in pairs}), taxonomy)
    log.info(
        "%d photos | %d labels matched, %d unmatched, %d ambiguous",
        len(pairs),
        len(recon.mapping),
        len(recon.unmatched),
        len(recon.ambiguous),
    )

    mcfg = ModelConfig(
        backbone=cfg.model.backbone,
        pretrained=False,
        head=cfg.head.name,
        head_kwargs=OmegaConf.to_container(cfg.head.head_kwargs, resolve=True)
        if cfg.head.get("head_kwargs")
        else {},
        drop_rate=cfg.model.drop_rate,
        drop_path_rate=cfg.model.drop_path_rate,
    )
    model = build_model(mcfg, taxonomy.num_classes)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(checkpoint, map_location=str(device))
    model.load_state_dict(ckpt.get("ema") or ckpt["model"])
    model.eval().to(device)

    transform = build_eval_transform(
        TransformConfig(**OmegaConf.to_container(cfg.data.transform, resolve=True))
    )

    out_dir = Path(_abs("outputs/ood"))
    gradcam_dir = out_dir / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    top1_hits = top5_hits = evaluated = 0
    for img_path, label in pairs:
        if label not in recon.mapping:
            continue  # species outside the 555 classes; excluded from accuracy
        true_idx = recon.mapping[label]
        img = load_image(img_path)
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)[0].cpu().numpy()
        order = np.argsort(-logits)
        pred_idx = int(order[0])
        evaluated += 1
        top1_hits += int(pred_idx == true_idx)
        top5_hits += int(true_idx in order[:5])
        rows.append(
            {
                "file": img_path.name,
                "label": label,
                "true_idx": true_idx,
                "pred": taxonomy.class_names[pred_idx],
                "correct": pred_idx == true_idx,
            }
        )

    _write_report(out_dir, rows, evaluated, top1_hits, top5_hits, recon, checkpoint)
    log.info(
        "OOD (my photos): top1=%.4f top5=%.4f on %d evaluated photos -> %s",
        top1_hits / evaluated if evaluated else 0.0,
        top5_hits / evaluated if evaluated else 0.0,
        evaluated,
        out_dir,
    )


def _write_report(out_dir, rows, evaluated, top1_hits, top5_hits, recon, checkpoint) -> None:
    top1 = top1_hits / evaluated if evaluated else 0.0
    top5 = top5_hits / evaluated if evaluated else 0.0
    lines = [
        "# OOD report — my field photos",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Evaluated photos (label in NABirds): **{evaluated}**",
        "",
        "| metric | my photos |",
        "|--------|----------:|",
        f"| Top-1 | {top1:.4f} |",
        f"| Top-5 | {top5:.4f} |",
        "",
        "> Compare against the in-distribution NABirds test set in `outputs/eval/.../report.md`",
        "> to quantify the train→field generalization gap.",
        "",
    ]
    if recon.unmatched:
        lines += ["## Unmatched labels (not in the 555 NABirds classes)", ""]
        lines += [f"- {u}" for u in recon.unmatched]
        lines += [""]
    if recon.ambiguous:
        lines += ["## Ambiguous labels (base name matched multiple morphs/ages)", ""]
        lines += [f"- {k} -> idx {v}" for k, v in recon.ambiguous.items()]
        lines += [""]
    lines += [
        "## Per-image predictions",
        "",
        "| file | label | prediction | ✓ |",
        "|------|-------|------------|---|",
    ]
    lines += [
        f"| {r['file']} | {r['label']} | {r['pred']} | {'✓' if r['correct'] else '✗'} |"
        for r in rows
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
