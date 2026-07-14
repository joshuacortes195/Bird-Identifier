# Results — NABirds fine-grained bird classifier

All numbers below are from **real runs** on the training box (RTX 3060, 12 GB).

## Model

- **Backbone:** ConvNeXt-V2-Base (ImageNet-22k pretrained), linear head
- **Recipe:** 224px, batch 32 × grad-accum 2 (eff. 64), **bfloat16 AMP**, AdamW,
  cosine LR + 2-epoch warmup, label smoothing 0.1, Mixup/CutMix, EMA, 30 epochs
- **Params:** 88.3 M
- **Training time:** ~30 epochs × ~6.5 min = **~3.3 h**

## Headline metrics

| Metric | Val (2,410 imgs) | **Test (24,633 imgs)** |
|--------|-----------------:|-----------------------:|
| Top-1  | 91.58% | **89.00%** |
| Top-5  | 99.38% | **98.78%** |
| Macro-F1 | — | 0.869 |
| Mean per-class acc | — | 0.867 |
| ECE (calibration) | — | 0.134 |

- 555 fine-grained categories (NABirds visual categories, incl. plumage/sex splits).
- Best epoch selected by EMA val top-1.
- **Calibration:** ECE 0.134 → mildly over-confident; temperature scaling is an easy
  follow-up (planned in the Phase 6 interpretability pass).

## Files

- `nabirds_test_metrics.json` — full metrics incl. worst classes + most-confused pairs
- `test_confusion_matrix.png` — row-normalized confusion matrix (555×555)
- `test_reliability.png` — reliability diagram (calibration)
- `nabirds_base_training_history.csv` — per-epoch train/val curves
- `nabirds_eda_report.md` — dataset EDA summary

## Reproduce

```bash
python scripts/download_data.py --dataset nabirds
python scripts/train.py data=nabirds model=convnextv2_base train=baseline
python scripts/evaluate.py --dataset nabirds --split test
```

## Optimization (Phase 8) — real CPU benchmark, batch=1

Exported `best.pt` → ONNX (opset 17, legacy exporter); PyTorch↔ONNXRuntime parity
verified within 1e-3. Dynamic INT8 quantization applied.

| Variant | Size (MB) | p50 (ms) | p95 (ms) | Throughput (img/s) |
|---------|----------:|---------:|---------:|-------------------:|
| PyTorch CPU | 336.7 | 198 | 216 | 5.0 |
| **ONNX fp32** | 336.9 | **145** | 153 | **6.9** |
| ONNX int8 | **85.6** | 503 | 535 | 2.0 |

**Finding (honest):** ONNX fp32 is the fastest (1.4× over PyTorch). Dynamic INT8 shrinks
the model **4×** (85.6 MB) but *regresses* latency — a known effect for conv-heavy
ConvNeXt on CPU (per-op quant/dequant overhead without fast int8 conv kernels). So the
**fp32 ONNX is the serving default**; int8 is only preferable when disk/memory is the hard
constraint. Artifacts: `outputs/serving/{model.onnx, model.int8.onnx}` (git-ignored, large).

## Serving (Phase 9) — validated with the real model

FastAPI (`POST /predict` multipart, `GET /health`) loads the ONNX model once at startup;
preprocessing is pure NumPy (torch-free serving image). Validated on random NABirds test
images: **6/8 top-1** (matches 89% test acc); the 2 misses had the true species as the #2
prediction (Cordilleran vs Least Flycatcher; Cassin's vs Plumbeous Vireo — genuinely hard
pairs). Note: scientific names are `null` (NABirds has none) — a common→scientific mapping
is a small follow-up.

## Not yet done (next steps)

- **Phase 6 (interpretability):** Grad-CAM overlays + written error analysis (metrics done).
- **Phase 7 (OOD):** evaluate on the photographer's own A7IV shots — **blocked** until
  labeled photos are added to `my_photos/`.
- **Phase 8 (optimization):** ONNX export + quantization + latency benchmarks (CPU-side;
  runs on any machine once the checkpoint is available).
- **Ablations:** bbox-crop / higher-res / TTA deltas — additional training runs.
