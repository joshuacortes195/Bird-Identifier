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

## Phase 6 — Evaluation & interpretability ✅ (code complete; report awaits checkpoint)

- **Metrics + calibration in NumPy** (`eval/metrics.py`, `eval/calibration.py`): top-1/top-5,
  macro-F1, per-class accuracy, confusion matrix, most-confused species pairs, and ECE/MCE +
  reliability bins. Deliberately torch-free so they run in CI and are **unit-tested here** —
  `tests/test_eval_metrics.py`, 6 tests, hand-checked values (macro-F1, per-class recall with
  absent classes, top-k, ECE, over-confidence flag). Class count flows from the logits width.
- **Grad-CAM** (`eval/gradcam.py`, torch + pytorch-grad-cam): picks the last ConvNeXt conv
  stage, computes a heatmap, composites a jet overlay, and — key for the app — a
  `gradcam_png_base64()` that the torch serving backend returns to the web app.
- **`scripts/evaluate.py`** (Hydra, mirrors `train.py`): loads a checkpoint (prefers EMA
  weights), runs the test split, and writes `outputs/eval/<run>/` — `report.md` (headline
  metrics, hardest classes, confused pairs), `reliability.png`, `confusion_matrix.png/.npy`,
  and `gradcam_samples.png` (correct vs. incorrect).
- **Closed the Grad-CAM loop to the app:** added `TorchPredictor` to the serve layer
  (`supports_gradcam=True`, rebuilds the model from the checkpoint's embedded config) and a
  `gradcam_png` method on the `Predictor` interface; `app.py` now returns the base64 overlay
  when a capable backend is loaded and the client asks. Verified the **app plumbing** with a
  fake gradcam-capable predictor (`tests/test_serve_api.py`): overlay returned when requested,
  null otherwise. ONNX/stub backends return null (torch-free image stays lean).

**Not runnable on this Mac** (no torch wheel for Intel/py3.14): `gradcam.py`, `evaluate.py`,
and `TorchPredictor`'s model-loading are written against `utils/checkpoint.py`'s schema and
the timm factory but must be validated against the first real Phase 5 checkpoint — flagged in
code. The metric math they call is the same code the unit tests cover.

**Files:** `eval/{metrics,calibration,gradcam}.py`, `scripts/evaluate.py`,
`serve/{predictor,app}.py` (Grad-CAM wiring + TorchPredictor), `tests/test_eval_metrics.py`,
`tests/test_serve_api.py` (gradcam plumbing test).

**Next:** Phase 8 — ONNX export + parity check + quantization + benchmark table.

---

## Phase 8 — Optimization: ONNX, quantization, benchmarking ✅ (code complete; numbers await checkpoint)

- **`optimize/export.py`:** `export_onnx` (dynamic batch axis, opset 17) + `verify_parity`
  (max |Δ| between torch and ONNX Runtime logits, asserts ≤ tol) + `file_size_mb`.
- **`optimize/quantize.py`:** dynamic int8 (weight-only, no calibration — the pragmatic CPU
  default) and a static/calibrated path behind a `CalibrationDataReader`.
- **`optimize/benchmark.py`:** `benchmark_callable` (warmup + timed loop over any inference
  closure), `summarize_latencies` (p50/p95/mean/throughput), and `render_table` (markdown
  comparison). The **stats + table are pure Python and unit-tested** — 3 tests with
  hand-computed percentiles (p50=25, p95=38.5 for [10,20,30,40] ms) and em-dash handling.
- **`scripts/export.py`** (Hydra): load checkpoint (prefers EMA) → `outputs/serving/model.onnx`
  → verify parity → `model.int8.onnx` → optional `+benchmark=true` writes
  `outputs/serving/benchmark.md` (pytorch-cpu vs onnx-fp32 vs onnx-int8: size, p50/p95,
  throughput). `model.onnx` is exactly what the lean Phase 9 Docker image serves.

Torch/ONNX export + quantization run on the CI/serving box (no torch/onnxruntime wheel on
this Intel/py3.14 Mac); the timing math they use is the tested code. `make export` wired.

**Files:** `optimize/{export,quantize,benchmark}.py`, `scripts/export.py`,
`tests/test_optimize_bench.py`.

**Pre-checkpoint work complete.** Remaining phases are gated on you: Phase 5 finishing (the
checkpoint), Phase 7 (your A7IV photos), Phase 11 (hosting accounts), Phase 12 (final README
with the real numbers). Everything above drops in behind the stable interfaces once the
checkpoint lands.
## Phase 5 — Full NABirds training ✅ (user-approved compute)

**Trained ConvNeXt-V2-Base on full NABirds** (555 classes). Pre-flight VRAM check:
8.25 GB peak of 12 GB at batch 32 bf16. Ran 30 epochs, ~6.5 min/epoch = **~3.3 h**.

**Real results (see `results/`):**

| Metric | Val (2,410) | **Test (24,633)** |
|--------|------------:|------------------:|
| Top-1 | 91.58% | **89.00%** |
| Top-5 | 99.38% | **98.78%** |
| Macro-F1 | — | 0.869 |
| ECE | — | 0.134 |

Recipe: 224px, eff. batch 64, bf16 AMP, AdamW, cosine+warmup, label smoothing 0.1,
Mixup/CutMix, EMA (best epoch by EMA val top-1). Full training curve in
`results/nabirds_base_training_history.csv`.

**Note on scope:** this is the strong single baseline. The *ablation table*
(bbox-crop / higher-res / TTA deltas) needs additional multi-hour runs and was not run
in this session — each is a separate `train.py` invocation with a config override.

---

## Phase 6 — Evaluation (core metrics ✅, interpretability pending)

- `scripts/evaluate.py` + `wildlife.eval.{metrics,calibration,gradcam}`: single test-set
  pass → top-1/top-5, macro-F1, per-class accuracy, worst classes, most-confused species
  pairs, confusion matrix figure, and **calibration (ECE + reliability diagram)**.
- Ran on the NABirds **test set** (real numbers above). Artifacts in `results/`.
- **Remaining:** wire Grad-CAM overlays (code merged from the full-stack branch) + written
  error analysis; ECE 0.134 → add temperature scaling.

**Checkpoint:** `outputs/checkpoints/nabirds-convnextv2_base-20260713_181835/best.pt`
(674 MB, git-ignored). Note: merged with the `phase-6-10-fullstack` branch (Phases 6–11
scaffolding built on the Mac); now being executed against the real checkpoint on the PC.

---

## Integration on PC — Phases 8 & 9 EXECUTED against the real model ✅

Merged the Mac's `phase-6-10-fullstack` scaffolding with the trained checkpoint and ran
the torch/GPU-relevant pipelines for real (they'd only ever been unit-tested before).

**Bugs fixed during integration:**
- EMA checkpoint loading in `scripts/evaluate.py` and `scripts/export.py` — weights are
  stored under a `module.` prefix (ModelEma); loading them raw would fail. Now stripped.
- ONNX export: pinned the **legacy TorchScript exporter** (`dynamo=False`) — avoids the
  torch 2.11 dynamo exporter's onnxscript dep + a Windows cp1252 unicode-print crash.

**Phase 8 (real, CPU batch=1):** ONNX fp32 = 336.9 MB / p50 145 ms / 6.9 img/s (fastest,
1.4× vs PyTorch); dynamic INT8 = 85.6 MB (4× smaller) but p50 503 ms (slower — conv-heavy
ConvNeXt on CPU). Parity verified <1e-3. fp32 ONNX is the serving default. See
`results/optimization_benchmark.md`.

**Phase 9 (validated):** FastAPI + real ONNX model → 6/8 top-1 on random test images
(matches 89% test acc); misses had the true species at rank 2. Pure-NumPy preprocessing
confirmed to match the trained transform through the full serve path.

**Remaining:** Phase 6 Grad-CAM/error-analysis + temp scaling; Phase 7 OOD (needs my
photos); Phase 10 frontend build/run vs the API; Phase 11 deploy (needs hosting accounts);
Phase 12 docs. Serving artifacts (`outputs/serving/*.onnx`) are git-ignored/large — deploy
transfers them out-of-band or CI regenerates via `scripts/export.py`.

---

## STATUS (2026-07-14) — overnight ablations CANCELLED per user

The optional ablation runs were **cancelled** (not run). `scripts/run_ablations.py` remains
as a ready tool — `python scripts/run_ablations.py` runs bbox-crop + 288px ablations and
fills `results/ablation_table.md` — but the ablation table currently holds only the
baseline reference row. AC sleep restored to default (PC may sleep normally again).

**Project state: Phases 0–6, 8, 9, 10, 12 are DONE and pushed** on branch
`phase-6-10-fullstack`. The two remaining phases both need the user:
- **Phase 7 (OOD test):** ingest path is built (`docs/OOD.md`) — needs labeled photos in
  `my_photos/<species>/*.jpg`, then `python scripts/ood_eval.py`.
- **Phase 11 (deploy):** API + frontend are deploy-ready (`docs/DEPLOY.md`, `netlify.toml`,
  torch-free Dockerfile) — needs Netlify + a CPU API-host account (HF Spaces/Render/Fly),
  and the model artifact transferred/regenerated on the host.

Optional cleanup later: fast-forward `main` to this branch.

