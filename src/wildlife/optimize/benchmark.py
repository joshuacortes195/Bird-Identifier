"""Latency/throughput benchmarking + results table (Phase 8).

The measurement loop needs a runnable model (torch or ONNX Runtime), but the statistics
and the markdown table are pure Python and unit-tested (``tests/test_optimize_bench.py``).
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchStat:
    name: str
    p50_ms: float
    p95_ms: float
    mean_ms: float
    throughput_ips: float  # images/sec (batch=1)
    size_mb: float | None = None
    accuracy: float | None = None


def summarize_latencies(
    latencies_s: list[float], name: str, size_mb: float | None = None, accuracy: float | None = None
) -> BenchStat:
    """Turn a list of per-inference wall times (seconds) into a stat row."""
    if not latencies_s:
        raise ValueError("no latencies to summarize")
    ms = sorted(t * 1000.0 for t in latencies_s)
    p50 = _percentile(ms, 50)
    p95 = _percentile(ms, 95)
    mean_ms = statistics.fmean(ms)
    throughput = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    return BenchStat(
        name=name,
        p50_ms=p50,
        p95_ms=p95,
        mean_ms=mean_ms,
        throughput_ips=throughput,
        size_mb=size_mb,
        accuracy=accuracy,
    )


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Linear-interpolation percentile on an already-sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def benchmark_callable(
    run_once: Callable[[], object], name: str, *, warmup: int = 10, iters: int = 100, **extra
) -> BenchStat:
    """Time ``run_once`` (a zero-arg inference closure) ``iters`` times after ``warmup``."""
    for _ in range(warmup):
        run_once()
    latencies: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run_once()
        latencies.append(time.perf_counter() - t0)
    return summarize_latencies(latencies, name, **extra)


def render_table(rows: list[BenchStat]) -> str:
    """Markdown comparison table across variants."""
    header = (
        "| variant | size (MB) | p50 (ms) | p95 (ms) | throughput (img/s) | top-1 |\n"
        "|---------|----------:|---------:|---------:|-------------------:|------:|"
    )
    lines = [header]
    for r in rows:
        size = f"{r.size_mb:.1f}" if r.size_mb is not None else "—"
        acc = f"{r.accuracy:.4f}" if r.accuracy is not None else "—"
        lines.append(
            f"| {r.name} | {size} | {r.p50_ms:.2f} | {r.p95_ms:.2f} "
            f"| {r.throughput_ips:.1f} | {acc} |"
        )
    return "\n".join(lines)
