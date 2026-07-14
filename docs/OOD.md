# Phase 7 — Out-of-distribution test on my own photos ⭐

The headline generalization result: how the model does on real A7IV field photos vs. the
NABirds test set. `my_photos/` is gitignored — drop your labeled shots there and run the
script; no code changes needed.

## How to label your photos

Either layout works — the ingest auto-detects:

**A) Folder-per-species** (simplest):

```
my_photos/
  Wood Duck/            IMG_1001.jpg  IMG_1002.heic
  American Robin/       DSC_0042.jpg
  Reddish Egret/        ...           # base names are fine; morph/age is reconciled
```

**B) A labels CSV** at `my_photos/labels.csv`:

```csv
filename,species
IMG_1001.jpg,Wood Duck
IMG_1002.heic,Reddish Egret (Dark morph)
```

Notes:
- HEIC/HEIF and EXIF orientation are handled (same loader as the API).
- Species names are reconciled to NABirds classes by exact → normalized → base-name match
  (`wildlife/eval/ood.py`, unit-tested). The run reports **unmatched** and **ambiguous**
  labels so you can fix them instead of silently dropping photos.
- Photos whose species isn't among the 555 NABirds classes are expected to be unmatched —
  that's a legitimate finding, not an error.

## Run it

```bash
python scripts/ood_eval.py +checkpoint=outputs/checkpoints/<run>/best.pt \
    model=convnextv2_base data=nabirds
```

Writes `outputs/ood/`:
- `report.md` — top-1/top-5 on your photos vs. the in-distribution test set (the
  generalization gap), plus per-image predictions and the reconciliation summary.
- `gradcam/` — attention overlays on your photos (lighting/pose/blur/distance analysis).

## What I need from you

1. **Phase 5 to finish** so there's a checkpoint to load.
2. **Your labeled photos** in `my_photos/` (either layout above).

Optional stretch (in the script's plan, not yet run): few-shot fine-tune on a couple of
species the model struggles with, and measure the lift.
