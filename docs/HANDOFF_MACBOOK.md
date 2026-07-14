# MacBook Handoff — continue the Wildlife/Bird Classifier

The GPU-bound work (training + test-set eval) is **done** on the Windows box. Everything
below runs on a Mac with **no GPU**. This file has (A) the manual setup steps and (B) a
copy-paste prompt for Claude Code to continue the project seamlessly.

---

## A. One-time setup on the Mac (do this first)

1. **Get the repo up to date:**
   ```bash
   # if not cloned yet:
   git clone https://github.com/joshuacortes195/Bird-Identifier.git
   cd Bird-Identifier
   # if already cloned:
   git pull origin main
   ```

2. **Drop the trained checkpoint in place.** Recreate this folder and put `best.pt` in it:
   ```bash
   mkdir -p outputs/checkpoints/nabirds-convnextv2_base-20260713_181835
   # move the downloaded best.pt into that folder:
   mv ~/Downloads/best.pt outputs/checkpoints/nabirds-convnextv2_base-20260713_181835/
   ```

3. **Create the environment (macOS — plain PyPI torch has Apple-Silicon/MPS support):**
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -U pip
   pip install torch torchvision          # macOS: NOT the cpu/cuda index — plain PyPI
   pip install -e ".[dev,serve,optimize]"
   ```

4. **Verify the checkpoint loads (no dataset needed):**
   ```bash
   python -c "import torch; c=torch.load('outputs/checkpoints/nabirds-convnextv2_base-20260713_181835/best.pt', map_location='cpu', weights_only=False); print('classes:', len(c['class_names']), '| best val_top1:', c['best_metric'], '| git:', c['git_commit'])"
   ```
   You should see `classes: 555` and the val metric ~0.9158.

5. **Open Claude Code in the repo and paste the prompt in section B.**
   (Optional: switch to a lighter model with `/model` — the heavy reasoning is done; the
   rest is web dev.)

> **Dataset is optional.** The full NABirds data (~9.9 GB) is NOT needed for ONNX export,
> the API, the frontend, or deployment. It's only needed to *re-run* dataset-level eval or
> render Grad-CAM over dataset images — re-download any time with
> `python scripts/download_data.py --dataset nabirds`.

---

## B. Paste this into Claude Code on the Mac

```
You are my senior ML engineering partner, continuing an in-progress project. Read
docs/PHASE_SUMMARIES.md and results/RESULTS.md first — they are the source of truth for
what is done and the real measured numbers. Do NOT fabricate metrics; only report numbers
from runs you actually execute.

PROJECT: A fine-grained North American bird species classifier (NABirds, 555 classes),
end-to-end ML + full-stack, culminating in a deployed web app where I upload a bird photo
and get the identification. Package is taxonomy-neutral ("wildlife", not "birds") on
purpose — do not rename it.

WHAT'S ALREADY DONE (Phases 0-6 core), all committed on main:
- Repo scaffold, tooling (ruff/pytest/CI), Hydra configs, Makefile + tasks.ps1.
- Data pipeline: NABirds + CUB downloaders, dataset REGISTRY, config-driven taxonomy
  (configs/taxonomy/birds.yaml = 555 classes; NEVER hardcode 555), transforms, dataloaders.
- Model factory over timm backbones + pluggable Head interface (LinearHead ships;
  HierarchicalHead is a documented stub — leave it stubbed).
- Training loop (AMP/bf16, EMA, mixup, cosine+warmup, checkpoint best/last, resume).
- TRAINED MODEL: ConvNeXt-V2-Base, test-set top-1 89.00%, top-5 98.78%, macro-F1 0.869,
  ECE 0.134. Checkpoint at outputs/checkpoints/nabirds-convnextv2_base-20260713_181835/best.pt
  (weights loadable via wildlife.eval.runner._load_model_from_ckpt — it prefers the EMA
  weights stored under ckpt['ema'] with a "module." prefix, and sets pretrained=False).
- Eval suite (scripts/evaluate.py): top-1/5, macro-F1, per-class, confusion matrix,
  calibration. Results in results/.

ENVIRONMENT: macOS, no CUDA. Use CPU/MPS. The training used bf16 AMP (CUDA only); on Mac
the eval/inference paths already fall back to fp32 — keep it that way. venv is at .venv.

CONVENTIONS (follow these):
- Every hyperparameter lives in configs/ (Hydra); nothing hardcoded. Type hints + docstrings.
- Pass `ruff check .` and `ruff format --check .` clean; add/keep pytest tests green.
- Commit at the END of every phase with a clear message (I don't need to approve between
  phases). Work autonomously; only stop for: (a) something only I can provide (my own
  photos, hosting accounts/credentials), or (b) a paid deploy — ask before spending money.
- Never claim results you didn't produce.

DO THESE PHASES IN ORDER:

Phase 6 finish (interpretability): Add Grad-CAM (pytorch-grad-cam) overlays on correct +
incorrect predictions targeting the ConvNeXt last stage; render a few samples to
outputs/eval/gradcam/. Add temperature scaling to fix the 0.134 ECE (fit T on the val set,
report new ECE). Write a short error analysis of the most-confused species pairs from
results/nabirds_test_metrics.json. Commit.

Phase 7 (OOD on my own photos) — BLOCKED until I add photos: Build the my_photos/ ingest
path now (EXIF/HEIC via the existing wildlife.data.imageio loader; a labeling format like
my_photos/<species>/*.jpg; reconcile my species names to NABirds categories with a small
mapping file + fuzzy match). Wire scripts/evaluate.py (or a new scripts/evaluate_ood.py) to
run inference on my_photos and report accuracy vs. the NABirds test set + per-image Grad-CAM.
Then STOP and tell me exactly how to add my photos. Do not fabricate OOD numbers.

Phase 8 (optimization): Export best.pt to ONNX (opset 17); verify PyTorch vs ONNXRuntime
parity within tolerance. Apply quantization (dynamic INT8; try static with a small
calibration set if it helps conv-heavy ConvNeXt). Benchmark latency (p50/p95) + throughput
+ model size + accuracy delta on CPU. Produce a benchmark table in results/. Save the
serving artifact to outputs/serving/.

Phase 9 (backend): FastAPI service loading the ONNX model once at startup. POST /predict
(multipart image upload) -> JSON top-k {common name, scientific name, confidence,
inference_time}; optional base64 Grad-CAM behind a flag. GET /health. Robust input handling
(content-type + size validation, HEIC/EXIF via imageio, server-side downscale, strip EXIF on
echo). CORS for the frontend origin, basic rate limiting, structured logging. Dockerfile
(already scaffolded). Keep a Gradio demo as `make demo`. Provide a curl example. Note:
scientific names aren't in NABirds — add a common->scientific mapping file (best effort;
mark unmapped as null).

Phase 10 (frontend): React + TypeScript + Vite + Tailwind in frontend/. Upload from anywhere
(file picker + drag/drop desktop; accept="image/*" + capture on mobile). Client-side
downscale before upload + preview. Results UI: top-k with confidence bars, scientific names,
Grad-CAM overlay toggle; explicit loading/empty/error states; "not confident" message below
a threshold. Responsive + a11y. Typed API client, base URL from an env var, no `any`.

Phase 11 (deploy) — needs my hosting accounts: Deploy the Docker API to a CPU host (HF
Spaces / Render / Railway / Fly.io) over HTTPS; deploy frontend to Netlify (GitHub
auto-deploy). Lock CORS to the real origin. Document the deploy in the README. ASK ME before
anything that costs money or needs my login.

Phase 12 (docs): README with problem framing, dataset, method, the real ablation/eval/OOD/
benchmark numbers, an architecture diagram (model -> API -> web app -> deploy), the live demo
link, and reproduction steps. Draft 2-3 resume bullets grounded strictly in real numbers.

EXTENSIBILITY SEAMS to preserve (do not build the animal expansion): taxonomy-neutral
package, config-driven label space, dataset registry (inat stubbed), Head interface
(LinearHead shipped, HierarchicalHead stubbed), supercategory plumbed end-to-end.

Start by reading docs/PHASE_SUMMARIES.md, confirming the checkpoint loads, then proceed with
Phase 6. Work through the phases autonomously, committing after each.
```
