# Phase Summaries — running log

One short entry per phase: key decisions, real measured numbers (only from runs that
actually executed), files changed, and what's next. No fabricated metrics.

---

## Phase 0 — Scaffold & environment ✅

**Environment detected (training box):**

| Item | Value |
|------|-------|
| OS | Windows 10 (19045), x86_64 |
| GPU | NVIDIA RTX 3060, **12 GB VRAM** (Ampere, sm_86) |
| System RAM | 16 GB |
| Python | 3.13.7 (`C:\Python313`) — repo targets 3.11+; CI runs 3.11/3.12 |
| Node / Git | Node 22.20, Git 2.47 |
| CUDA | driver 610.47 (13.3 UMD) → CUDA 12.x runtime wheels |

**Compute plan chosen:** Base tier — `convnextv2_base.fcmae_ft_in22k_in1k` @ 224px,
batch 32 × grad-accum 2 (effective 64), AMP. Smoke tier = `convnextv2_nano/tiny` on
CUB-200 / 10-class subset. Derived from VRAM at runtime, so cloud/CPU fallback runs
the same configs. (Will confirm with the user right before the Phase 5 full run.)

**Decisions:**
- Package named `wildlife` (taxonomy-neutral seam), not `birds`.
- `make` is not installed on Windows → added `tasks.ps1` mirroring every Make target.
- Torch/torchvision are **not** pinned in `pyproject.toml` — installed per-platform
  (CUDA on Windows, CPU on Mac/CI) so one repo works on both machines.
- Python 3.13 locally, but CI targets 3.11/3.12 for wheel-availability safety.

**Files:** `pyproject.toml`, `Makefile`, `tasks.ps1`, `Dockerfile`, `.gitignore`,
`.github/workflows/ci.yml`, `configs/config.yaml`, `src/wildlife/{__init__,__main__}.py`,
`src/wildlife/utils/{seed,logging,env_report}.py`, subpackage inits,
`tests/test_seed.py`, `tests/test_env_report.py`, `scripts/report_env.py`, `README.md`.

**Verification:** `ruff check` + `ruff format --check` clean; `pytest` green. See commit.

**Next:** Phase 1 — data acquisition (NABirds + CUB-200), EDA, splits, dataset
registry, `configs/taxonomy/birds.yaml`.

---

## Phase 1 — Data acquisition & EDA ✅

**Datasets acquired (real):**

| Dataset | Classes | Train | Val | Test | w/ bbox | Imbalance (train) |
|---------|--------:|------:|----:|-----:|--------:|------------------:|
| NABirds | **555** | 21,519 | 2,410 | 24,633 | 48,562 | 18.0× |
| CUB-200 | 200 | 5,394 | 600 | 5,794 | 11,788 | 1.04× |

- Official NABirds split **respected** (24,633 test matches the published split); a
  seeded, stratified 10%/class val set is carved from train.
- Class counts are **derived** from the data (555 / 200), never hardcoded.

**Sourcing:** NABirds is gated (Cornell agreement), so `download_data.py` tries direct
mirrors → public HF single-file mirrors. Working mirror: **`antokun/nabirds`**
(`nabirds.zip`, full official layout incl. `train_test_split.txt`, `bounding_boxes.txt`,
`parts/`, `sizes.txt`). `QianC95/NABirds` turned out to be a geolocation dataset — the
downloader detects the missing `images.txt` and falls through. CUB via the fast.ai S3
mirror. If all fail, clear manual-placement instructions are printed.

**Seams delivered:** dataset **registry** (`register_dataset`), config-driven **taxonomy**
(`configs/taxonomy/birds.yaml` = 555 classes, `cub.yaml` = 200; each class →
common_name → `supercategory: bird`), and an **`inat` stub** (`sources/inat.py`,
`register_dataset` comment) marking the future seam.

**EDA:** `outputs/eda/{nabirds,cub}/` — class-distribution (long tail), image-size/aspect
stats, sample grid, `report.md`. Hardest-confusable pairs deferred to Phase 6 (needs a model).

**Files:** `data/{schema,taxonomy,registry,prepare,eda}.py`,
`data/sources/{download,cub_format,cub,nabirds,inat,preparers}.py`,
`scripts/{download_data,eda,make_subset}.py`, `configs/data/{cub,nabirds,subset}.yaml`.

**Next:** Phase 2 — torch dataloaders + fine-grained transforms.

---

## Phase 2 — Data pipeline ✅

- **`SpeciesDataset`** reads manifests, returns `{image, label, supercategory}` (coarse
  taxon plumbed through, constant "bird" now). **Bbox-crop-to-subject** path is a config
  toggle (`bbox_crop`) so its effect is measurable.
- **Transforms:** train = RandomResizedCrop + flip + RandAugment + normalize + random
  erasing; eval = deterministic resize/center-crop. **Mixup/CutMix** wrapper (timm),
  batch-level, applied in the training loop.
- **Imbalance handling:** `WeightedRandomSampler` + inverse-freq class weights, both config options.
- **Datasets self-register** (`nabirds`, `cub`) and resolve via `build_dataset(name, ...)`.
- **Benchmark (real):** NABirds train loader = **741.6 imgs/sec** (batch 32, 6 workers) —
  won't bottleneck the RTX 3060. Batch grid + EDA figures rendered.
- **Tests:** 22 passing (manifest round-trip, taxonomy contiguity, eval-transform
  determinism, bbox-crop clamping/correctness, registry resolution, dataset item contract,
  collate, bbox-toggle changes output). All run in CI via a synthetic on-disk mini-dataset.

**Files:** `data/{imageio,base,datasets,transforms,loaders,visualize}.py`,
`scripts/benchmark_loader.py`, `tests/{conftest,test_schema_taxonomy,test_transforms_imageio,test_dataset_registry}.py`.

**Next:** Phase 3 — model factory + `Head` interface + training loop, proven by a
subset overfit smoke test. (GPU work begins.)

---

## Phase 3 — Baseline training loop + smoke test ✅

- **Model factory** over timm backbones as pooled feature extractors + a **pluggable
  `Head`** (registry seam #4): `LinearHead` ships; `HierarchicalHead` is a stub that
  raises `NotImplementedError`. Head output dim = `taxonomy.num_classes` (never 555).
  Configs: `model/{convnextv2_base,nano,atto}.yaml`, `head/{linear,hierarchical}.yaml`.
- **Training loop:** AMP, gradient accumulation, AdamW (no-decay on norms/bias),
  cosine LR + linear warmup, label smoothing, Mixup/CutMix, optional EMA, checkpoint
  (best+last, with config+git-hash+metrics embedded), early stopping, resume.
- **Smoke test (real):** `python scripts/train.py` overfits the 10-class subset —
  **train loss 2.30 → 0.013, val top-1 0.93, top-5 1.00** in 20 epochs on the RTX 3060.
  Proves the loop is correct before scaling.

**Bug found + fixed (AMP):** the initial fp16 `GradScaler` path silently **skipped ~60%
of optimizer steps** (persistent fp16 overflow on convnextv2), pinning loss at ln(10).
Isolated it by proving the fp32 loop learns (val 0.90), then switched AMP to **bfloat16**
on Ampere (fp32 dynamic range, no loss scaling needed). fp16+scaler retained as fallback
where bf16 is unavailable. This matters for Phase 5 — the full run relies on this path.

- **Tests:** 30 passing (added model-forward/head-output-dim, hierarchical-stub raises,
  scheduler warmup→decay, one-step loss decrease, checkpoint round-trip).

**Files:** `models/{factory,heads}.py`, `train/{loop,optim,ema,tracking}.py`,
`utils/checkpoint.py`, `scripts/train.py`, `configs/{model,head,train}/*.yaml`,
`tests/{test_model_heads,test_train_core}.py`.

**Next (GATE):** Phase 4 (tracking) is wired (local JSONL + optional W&B). Phase 5 is the
long full-NABirds run — **needs user OK on compute** before kicking off (~3h estimate).

---

## Phase 9 — Production inference API ✅ (built ahead of order)

**Why out of order:** built on a Mac (Intel, Python 3.14) *while Phase 5 trains on the
RTX 3060 PC*. Modern torch has no wheel for this box, so the GPU-bound phases (5–8) can't
run here — but the serve + web layers don't need the GPU, only the stable model
*interface*. So Phase 9 (and 10 next) were built first; Phases 6/8 torch code slots in
behind the same interfaces when the checkpoint lands. On a feature branch
(`phase-6-10-fullstack`), **not pushed to main**, so the PC's Phase 5 push stays clean.

- **Torch-free serving path.** The API runs on **ONNX Runtime + Pillow + NumPy** (no
  torch/timm/hydra), keeping the deploy image small for free-tier CPU hosts.
  `serve/preprocess.py` reimplements the eval transform (resize→center-crop→ImageNet
  norm) in pure NumPy; a torch-gated parity test pins it to `build_eval_transform`.
- **`Predictor` interface** (`serve/predictor.py`): `OnnxPredictor` (production, lazy
  `onnxruntime` import) and `StubPredictor` (deterministic, model-free — lets the frontend
  and contract be tested before a checkpoint exists). Prod refuses the stub unless
  `WILDLIFE_ALLOW_STUB=1`, so it never silently serves fake predictions.
- **FastAPI app** (`serve/app.py`): model loaded once at startup (lifespan); `POST
  /predict` (multipart) → ranked top-k JSON with common/scientific name, confidence,
  `low_confidence` flag, inference time; `GET /health`. HEIC/HEIF + EXIF via the existing
  `load_image_from_bytes`; capped/chunked upload read (413), content-type gate (415),
  undecodable reject (422), server-side downscale, per-IP rate limit (429), CORS locked to
  configured origins, structured JSON logs. Config is 12-factor via env (`serve/config.py`).
- **num_classes flows from the taxonomy** — live `/health` reports 555 for `birds.yaml`,
  never hardcoded. Grad-CAM overlay field is in the contract; returns null for ONNX/stub
  (needs the torch model from Phase 6) so the UI can hide the toggle.
- **Verified (real):** `uvicorn` boots against the 555-class taxonomy; `curl` upload
  returns sensible ranked JSON; `/health` OK; 415 on non-image. **12 API/preprocess tests
  pass** (1 torch-parity test skips here, runs on PC/CI).

**Also:** lean torch-free `Dockerfile`; `make serve-dev` (stub) + `serve` (real);
optional Gradio demo split into a `demo` extra; `docs/API.md` documents the full contract.
Small enabling change: `wildlife/data/__init__.py` dataset registration is now best-effort
so the torch-free submodules (`taxonomy`, `imageio`) import without torch.

**Files:** `serve/{preprocess,predictor,config,app,gradio_demo}.py`,
`tests/{test_serve_api,test_serve_preprocess}.py`, `Dockerfile`, `Makefile`, `tasks.ps1`,
`pyproject.toml`, `docs/API.md`, `data/__init__.py`.

**Next:** Phase 10 — the React/TS/Vite/Tailwind web app against this contract.

---

## Phase 10 — Cross-device web app ✅ (built ahead of order)

React 19 · TypeScript (strict) · Vite 6 · Tailwind 4, in `frontend/`. Talks to the Phase 9
API through a fully-typed client (no `any`); API base URL from `VITE_API_BASE_URL`.

- **Upload from anywhere:** drag-and-drop + file picker on desktop; a `capture` camera path
  on mobile (`accept="image/*"`). **Client-side prep** (`lib/image.ts`) downscales/compresses
  to ≤1600px JPEG via canvas before upload (faster on cellular); HEIC passes through untouched
  for the server to decode. Preview thumbnail shown.
- **Results UI:** best-match hero (common + scientific name, confidence bar), ranked "other
  possibilities" with bars + percentages, a low-confidence banner below threshold, a
  **"What did it see?" Grad-CAM overlay** toggle with an opacity slider (fetches
  `include_gradcam=true` on demand; shows an honest "unavailable from this backend" note until
  the torch backend from Phase 6 is wired), a species "Learn more" link, and a model/latency
  chip. Explicit loading (skeleton), error (with retry), and empty states. In-session recent
  uploads.
- **Design system** (via the ui-ux-pro-max skill): minimal single-column, content-first,
  micro-interactions. Photo-forward **dark-first theme with an emerald accent** (+ full light
  mode), Newsreader serif wordmark / Inter UI. Semantic Tailwind-v4 tokens, no-flash theme
  init, manual toggle. Accessibility: visible focus rings, `prefers-reduced-motion`,
  ≥44px targets, SVG-only icons (no emoji), confidence never by color alone.
- **Verified (real):** `npm run build` passes strict `tsc` + Vite (67 KB gzipped JS). Drove
  the full **upload → identify** flow in headless Chrome against the live stub API and captured
  screenshots at **desktop dark, desktop light, and 390px phone** — best-match, shortlist,
  low-confidence banner, Grad-CAM toggle, reset, and error states all render correctly; the
  555-class count and latency flow through from the API. CORS confirmed (a deliberate origin
  mismatch produced the expected `network_error` state, then fixed).

**Files:** `frontend/` — `package.json`, Vite/TS/Tailwind config, `index.html`,
`src/{main,App}.tsx`, `src/api/{client,types}.ts`, `src/lib/image.ts`,
`src/hooks/usePredict.ts`, `src/components/*`, `src/icons/index.tsx`, `src/index.css`,
`.env.example`, `README.md`.

**Next:** Phases 6 (eval/interpretability incl. real Grad-CAM) and 8 (ONNX export) — torch
code written against the interfaces, to run on the PC/CI once the Phase 5 checkpoint lands.

---
