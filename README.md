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

## Project plan

13 phases from scaffold → data → training → evaluation → OOD test → optimization →
API → web app → deployment → docs. GPU-bound phases (training/eval/optimize) run on
the workstation; the API/frontend/deploy phases need no GPU.

## License

MIT.
