# Inference benchmark (CPU, batch=1)

| variant | size (MB) | p50 (ms) | p95 (ms) | throughput (img/s) | top-1 |
|---------|----------:|---------:|---------:|-------------------:|------:|
| pytorch-cpu | 336.7 | 198.32 | 216.15 | 5.0 | — |
| onnx-fp32 | 336.9 | 145.18 | 153.47 | 6.9 | — |
| onnx-int8 | 85.6 | 503.25 | 534.95 | 2.0 | — |
