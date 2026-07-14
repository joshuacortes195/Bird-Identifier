"""Export a trained model to ONNX and verify PyTorch↔ONNX Runtime parity (Phase 8).

Runs on the training/CI box (torch + onnx + onnxruntime). The exported ``model.onnx`` +
the taxonomy are what the lean serving image (Phase 9) loads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def export_onnx(
    model,
    output_path: str | Path,
    *,
    input_size: int = 224,
    opset: int = 17,
) -> Path:
    """Export ``model`` (a SpeciesClassifier in eval mode) to ONNX with a dynamic batch axis."""
    import torch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        # Legacy TorchScript exporter: mature for CNNs, deterministic across platforms,
        # and avoids the dynamo exporter's onnxscript dep + unicode console prints.
        dynamo=False,
    )
    return output_path


def verify_parity(
    model, onnx_path: str | Path, *, input_size: int = 224, tol: float = 1e-3
) -> float:
    """Return the max abs difference between torch and ONNX Runtime logits on random input.

    Raises AssertionError if it exceeds ``tol``.
    """
    import onnxruntime as ort
    import torch

    x = torch.randn(2, 3, input_size, input_size)
    model.eval()
    with torch.no_grad():
        torch_out = model(x).cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]

    max_diff = float(np.abs(torch_out - onnx_out).max())
    assert max_diff <= tol, f"ONNX parity failed: max|Δ|={max_diff:.2e} > tol={tol:.2e}"
    return max_diff


def file_size_mb(path: str | Path) -> float:
    return Path(path).stat().st_size / (1024 * 1024)
