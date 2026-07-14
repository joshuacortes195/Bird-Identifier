#!/usr/bin/env python
"""Export the best checkpoint to ONNX, quantize, verify parity, and benchmark (Phase 8).

Writes serving artifacts to ``outputs/serving/`` (``model.onnx`` picked up by the API /
Docker image) and a benchmark table to ``outputs/serving/benchmark.md``.

Usage:
    python scripts/export.py +checkpoint=outputs/checkpoints/<run>/best.pt \\
        model=convnextv2_base [+benchmark=true]

Runs on the training/CI box (torch + onnx + onnxruntime). The benchmark statistics/table
code it calls is unit-tested in tests/test_optimize_bench.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wildlife.data.taxonomy import load_taxonomy  # noqa: E402
from wildlife.models import ModelConfig, build_model  # noqa: E402
from wildlife.optimize.benchmark import benchmark_callable, render_table  # noqa: E402
from wildlife.optimize.export import export_onnx, file_size_mb, verify_parity  # noqa: E402
from wildlife.optimize.quantize import quantize_dynamic_onnx  # noqa: E402
from wildlife.utils.checkpoint import load_checkpoint  # noqa: E402
from wildlife.utils.logging import get_logger  # noqa: E402

log = get_logger("export")


def _abs(p: str) -> str:
    path = Path(p)
    return str(path if path.is_absolute() else REPO_ROOT / path)


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "configs"), config_name="config")
def main(cfg: DictConfig) -> None:
    checkpoint = cfg.get("checkpoint")
    if not checkpoint:
        raise SystemExit("Pass +checkpoint=<path/to/best.pt>")
    checkpoint = _abs(str(checkpoint))

    taxonomy = load_taxonomy(_abs(cfg.data.get("taxonomy", "configs/taxonomy/birds.yaml")))
    num_classes = taxonomy.num_classes
    input_size = int(cfg.data.transform.image_size)

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
    model = build_model(mcfg, num_classes)

    ckpt = load_checkpoint(checkpoint, map_location="cpu")
    state = ckpt.get("ema") or ckpt["model"]
    model.load_state_dict(state)
    model.eval()
    log.info("Loaded %s | %d classes | input %d", checkpoint, num_classes, input_size)

    serving_dir = Path(_abs("outputs/serving"))
    onnx_path = serving_dir / "model.onnx"
    export_onnx(model, onnx_path, input_size=input_size)
    max_diff = verify_parity(model, onnx_path, input_size=input_size)
    log.info("ONNX exported: %s | parity max|Δ|=%.2e", onnx_path, max_diff)

    int8_path = serving_dir / "model.int8.onnx"
    quantize_dynamic_onnx(onnx_path, int8_path)
    log.info("Quantized (int8 dynamic): %s", int8_path)

    if cfg.get("benchmark"):
        _run_benchmark(model, onnx_path, int8_path, input_size, serving_dir)

    log.info("Serving artifacts in %s", serving_dir)


def _run_benchmark(model, onnx_path, int8_path, input_size, serving_dir: Path) -> None:
    import numpy as np
    import onnxruntime as ort
    import torch

    x_np = np.random.randn(1, 3, input_size, input_size).astype("float32")
    x_pt = torch.from_numpy(x_np)

    def run_torch():
        with torch.no_grad():
            model(x_pt)

    rows = [benchmark_callable(run_torch, "pytorch-cpu", size_mb=_state_size_mb(model))]
    for label, path in [("onnx-fp32", onnx_path), ("onnx-int8", int8_path)]:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        rows.append(
            benchmark_callable(
                lambda s=sess, n=name: s.run(None, {n: x_np}),
                label,
                size_mb=file_size_mb(path),
            )
        )

    table = render_table(rows)
    (serving_dir / "benchmark.md").write_text(
        "# Inference benchmark (CPU, batch=1)\n\n" + table + "\n", encoding="utf-8"
    )
    log.info("Benchmark table:\n%s", table)


def _state_size_mb(model) -> float:
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return total / (1024 * 1024)


if __name__ == "__main__":
    main()
