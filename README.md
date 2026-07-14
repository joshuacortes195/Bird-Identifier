# Wildlife Classifier — Fine-Grained Bird Species Identification

An end-to-end ML + full-stack project: train a fine-grained North American **bird**
species classifier, evaluate it rigorously (including on the author's own wildlife
photos as an out-of-distribution test), optimize it for fast inference, and serve it
behind an HTTPS API with a cross-device web app.

> **Taxonomy-neutral by design.** The package is named `wildlife` (not `birds`).
> Birds are the only taxon populated here, but the label space is config-driven and
> datasets/heads are pluggable, so mammals and other taxa can be added later as a
> data change — not a rewrite. The animal expansion itself is **not** built here.

## Status

Built phase-by-phase. See `docs/PHASE_SUMMARIES.md` for the running log of decisions
and real measured numbers. **No metric appears here or anywhere until a real run
produced it.**

### Results (real, measured)

ConvNeXt-V2-Base on **NABirds** (555 fine-grained classes), trained on an RTX 3060
(~3.3 h, bfloat16 AMP, EMA, Mixup/CutMix):

| Metric (NABirds test, 24,633 imgs) | Value |
|------------------------------------|------:|
| Top-1  | **89.00%** |
| Top-5  | **98.78%** |
| Macro-F1 | 0.869 |
| Mean per-class accuracy | 0.867 |

**Inference optimization** (ONNX Runtime, CPU, batch=1): ONNX fp32 runs at **6.9 img/s
(145 ms p50)** — 1.4× faster than PyTorch; dynamic INT8 shrinks the model 4× (337 MB →
**85.6 MB**) at a latency cost, so fp32 ONNX is the serving default. The FastAPI service
was validated against the real model (correct top-k on held-out test images).

Full detail — including the confusion matrix, calibration diagram, and benchmark table —
in [`results/`](results/RESULTS.md). Remaining work: OOD test on the author's own photos
(awaiting photos) and live deployment (see `docs/PHASE_SUMMARIES.md`).

## Quickstart

### 1. Install

Torch is installed separately per platform (the wheel differs), then the package:

```bash
# Windows + NVIDIA GPU (this project's training box)
python -m venv .venv && .venv\Scripts\activate
./tasks.ps1 install-cuda      # torch + torchvision, CUDA 12.4
./tasks.ps1 install           # package + dev/serve/optimize/tracking extras

# macOS / Linux / CPU
python -m venv .venv && source .venv/bin/activate
make install-cpu
make install
```

> On Windows `make` may not be installed — use `./tasks.ps1 <target>` (same targets).

### 2. Verify environment

```bash
python scripts/report_env.py    # prints hardware + recommended compute plan
```

### 3. Lint & test

```bash
make lint    # or ./tasks.ps1 lint
make test    # or ./tasks.ps1 test
```

## Compute plan (detected on the training box)

| Item | Value |
|------|-------|
| GPU | NVIDIA RTX 3060, 12 GB VRAM |
| Tier | **Base** — `convnextv2_base` @ 224px, batch 32 × grad-accum 2 (eff. 64), AMP |
| Smoke | `convnextv2_nano`/`tiny` on CUB-200 / a 10-class subset |

The compute plan is derived from VRAM at runtime (`src/wildlife/utils/env_report.py`),
so the same configs run on a cloud GPU or CPU fallback unchanged.

## Repo layout

```
configs/     Hydra configs (data, model, head, taxonomy, train)
src/wildlife/  data · models · train · eval · optimize · serve · utils
scripts/     download_data · prepare_splits · train · evaluate · export
frontend/    React + TS + Vite + Tailwind web app (Phase 10)
tests/       unit tests + fast end-to-end smoke test
```

`data/`, `outputs/`, and `my_photos/` are gitignored.

## Architecture

```
                        TRAIN (GPU)                         SERVE (CPU)
  ┌──────────────┐   ┌──────────────────┐   export   ┌───────────────────┐
  │ NABirds/CUB  │──▶│ ConvNeXt-V2-Base │──────────▶ │ model.onnx (fp32) │
  │ (registry +  │   │ + LinearHead     │  (ONNX +   │  ONNX Runtime     │
  │  taxonomy)   │   │ timm · Hydra     │  quantize) │  (torch-free)     │
  └──────────────┘   └──────────────────┘            └─────────┬─────────┘
                                                               │
                          ┌────────────────────────────────────▼─────────┐
                          │ FastAPI  POST /predict · GET /health          │
                          │ HEIC/EXIF · validation · CORS · rate limit    │
                          └────────────────────────────────────┬─────────┘
                                                     HTTPS JSON │ (top-k + Grad-CAM)
                          ┌────────────────────────────────────▼─────────┐
                          │ React + TS + Vite + Tailwind (Netlify)        │
                          │ upload (camera/drag-drop) · confidence bars   │
                          └───────────────────────────────────────────────┘
```

## Reproduce the full pipeline

```bash
# 1. Data (NABirds ~9.9 GB; CUB is the fast smoke set)
python scripts/download_data.py --dataset nabirds

# 2. Train (RTX 3060 ~3.3 h) — or `train=smoke data=subset` to prove the loop first
python scripts/train.py data=nabirds model=convnextv2_base train=baseline

# 3. Evaluate on the held-out test set (metrics + confusion + calibration + Grad-CAM)
python scripts/evaluate.py +checkpoint=outputs/checkpoints/<run>/best.pt \
    data=nabirds model=convnextv2_base
python scripts/calibrate.py +checkpoint=outputs/checkpoints/<run>/best.pt \
    data=nabirds model=convnextv2_base          # temperature scaling

# 4. Export to ONNX + quantize + benchmark
python scripts/export.py +checkpoint=outputs/checkpoints/<run>/best.pt \
    model=convnextv2_base data=nabirds +benchmark=true

# 5. Serve + web app
make serve                                        # FastAPI on :8000 (WILDLIFE_MODEL_PATH)
cd frontend && npm install && npm run dev          # web app on :5173
```

See `docs/API.md` (endpoint contract), `docs/DEPLOY.md` (hosting), and `docs/OOD.md`
(own-photo test format).

## Résumé bullets (grounded strictly in the measured results above)

- Trained a fine-grained **ConvNeXt-V2** classifier on **NABirds (555 species)** to
  **89.0% top-1 / 98.8% top-5** on the held-out test set, on a single RTX 3060 using
  bfloat16 mixed precision, EMA, and Mixup/CutMix.
- Built the **full ML lifecycle** in a reproducible, config-driven (Hydra) repo — data
  registry, training, evaluation (confusion matrix + calibration/ECE + Grad-CAM), ONNX
  optimization, and a FastAPI + React web app — with 57 automated tests and CI.
- Optimized inference with **ONNX Runtime**, reaching **1.4× CPU speedup** over PyTorch
  and a **4× model-size reduction** via INT8 quantization, enabling free-tier CPU hosting.

## Project plan

13 phases from scaffold → data → training → evaluation → OOD test → optimization →
API → web app → deployment → docs. GPU-bound phases (training/eval/optimize) run on
the workstation; the API/frontend/deploy phases need no GPU.

## License

MIT.
