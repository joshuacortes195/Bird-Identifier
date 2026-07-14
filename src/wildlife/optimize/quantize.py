"""Post-training quantization of the ONNX model (Phase 8).

Dynamic quantization (weights → int8, activations quantized at runtime) is the pragmatic
default for CPU serving: no calibration data needed and a solid size/latency win. A static
(calibrated) path is provided behind an optional calibration reader. Runs on the CI/serving
box (onnxruntime.quantization).
"""

from __future__ import annotations

from pathlib import Path


def quantize_dynamic_onnx(model_path: str | Path, output_path: str | Path) -> Path:
    """Weight-only int8 dynamic quantization — no calibration set required."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(model_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )
    return output_path


def quantize_static_onnx(
    model_path: str | Path, output_path: str | Path, calibration_reader
) -> Path:
    """Static int8 quantization using a CalibrationDataReader (activations calibrated).

    ``calibration_reader`` must yield ``{input_name: np.ndarray}`` batches of representative
    (preprocessed) images. Better accuracy retention than dynamic when a small calibration
    set is available.
    """
    from onnxruntime.quantization import QuantType, quantize_static

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(model_path),
        model_output=str(output_path),
        calibration_data_reader=calibration_reader,
        weight_type=QuantType.QInt8,
    )
    return output_path
