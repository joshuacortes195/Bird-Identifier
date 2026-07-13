"""Environment / hardware detection and a VRAM-based compute-plan recommendation.

Printed at the start of Phase 0 (and re-runnable any time) so every machine the
repo runs on records exactly what it detected and what tier it would pick.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field


@dataclass
class ComputePlan:
    tier: str
    backbone: str
    image_size: int
    batch_size: int
    grad_accum: int
    amp: bool
    notes: list[str] = field(default_factory=list)


def _recommend_plan(vram_gb: float | None, has_cuda: bool) -> ComputePlan:
    """Pick a model tier + batch geometry from available VRAM.

    Thresholds are conservative for fine-grained training at 224px with AMP.
    """
    if not has_cuda or vram_gb is None:
        return ComputePlan(
            tier="Tiny (CPU/cloud fallback)",
            backbone="convnextv2_nano.fcmae_ft_in22k_in1k",
            image_size=224,
            batch_size=16,
            grad_accum=4,
            amp=False,
            notes=[
                "No CUDA GPU detected — training will be slow.",
                "Use the identical config on Google Colab / a cloud GPU (it runs unchanged).",
            ],
        )
    if vram_gb >= 20:
        return ComputePlan("Large", "convnextv2_large.fcmae_ft_in22k_in1k", 224, 32, 2, True)
    if vram_gb >= 10:
        return ComputePlan(
            tier="Base",
            backbone="convnextv2_base.fcmae_ft_in22k_in1k",
            image_size=224,
            batch_size=32,
            grad_accum=2,
            amp=True,
            notes=["A 384px fine-tune stage is feasible at batch ~12-16."],
        )
    if vram_gb >= 6:
        return ComputePlan("Tiny", "convnextv2_tiny.fcmae_ft_in22k_in1k", 224, 24, 2, True)
    return ComputePlan(
        tier="Nano",
        backbone="convnextv2_nano.fcmae_ft_in22k_in1k",
        image_size=224,
        batch_size=16,
        grad_accum=4,
        amp=True,
        notes=["Low VRAM — prefer the smoke/subset configs for iteration."],
    )


def collect_report() -> dict:
    """Gather OS/Python/torch/CUDA facts into a plain dict."""
    report: dict = {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "torch": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_name": None,
        "vram_gb": None,
    }
    try:
        import torch

        report["torch"] = torch.__version__
        if torch.cuda.is_available():
            report["cuda_available"] = True
            report["cuda_version"] = torch.version.cuda
            report["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            report["vram_gb"] = round(props.total_memory / (1024**3), 1)
    except ImportError:
        report["torch"] = "not installed"
    return report


def format_report(report: dict, plan: ComputePlan) -> str:
    lines = [
        "=" * 62,
        " ENVIRONMENT REPORT",
        "=" * 62,
        f" OS            : {report['os']} ({report['machine']})",
        f" Python        : {report['python']}",
        f" PyTorch       : {report['torch']}",
        f" CUDA available: {report['cuda_available']}  (cuda {report['cuda_version']})",
        f" GPU           : {report['gpu_name']}",
        f" VRAM          : {report['vram_gb']} GB" if report["vram_gb"] else " VRAM         : n/a",
        "-" * 62,
        " RECOMMENDED COMPUTE PLAN",
        "-" * 62,
        f" Tier       : {plan.tier}",
        f" Backbone   : {plan.backbone}",
        f" Image size : {plan.image_size}",
        f" Batch      : {plan.batch_size} (grad-accum {plan.grad_accum} "
        f"= effective {plan.batch_size * plan.grad_accum})",
        f" AMP        : {plan.amp}",
    ]
    for note in plan.notes:
        lines.append(f"   - {note}")
    lines.append("=" * 62)
    return "\n".join(lines)


def main() -> int:
    report = collect_report()
    plan = _recommend_plan(report["vram_gb"], report["cuda_available"])
    print(format_report(report, plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
