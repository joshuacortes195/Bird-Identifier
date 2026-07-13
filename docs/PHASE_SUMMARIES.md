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
