# Confidence calibration (temperature scaling)

- Temperature fit on val (no test leakage): **T = 0.717**
- Test ECE: **0.1338 → 0.0369** (improved)
- Test MCE: 0.1733 → 0.1578
- Over-confident before: False; after: True

Temperature scaling is a post-hoc, accuracy-preserving fix (it only rescales logits, so top-1/top-5 are unchanged). Apply T at serving time before softmax.
