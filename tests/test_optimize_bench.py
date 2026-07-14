"""Benchmark statistics + table rendering (Phase 8). Pure Python, no torch/onnx needed."""

from __future__ import annotations

from wildlife.optimize.benchmark import BenchStat, render_table, summarize_latencies


def test_summarize_latencies_known_values():
    stat = summarize_latencies([0.010, 0.020, 0.030, 0.040], name="pt", size_mb=100.0, accuracy=0.9)
    # ms = [10, 20, 30, 40]; p50 interpolated = 25, p95 = 38.5, mean = 25.
    assert abs(stat.p50_ms - 25.0) < 1e-9
    assert abs(stat.p95_ms - 38.5) < 1e-9
    assert abs(stat.mean_ms - 25.0) < 1e-9
    assert abs(stat.throughput_ips - 40.0) < 1e-9
    assert stat.size_mb == 100.0
    assert stat.accuracy == 0.9


def test_summarize_single_sample():
    stat = summarize_latencies([0.05], name="x")
    assert stat.p50_ms == 50.0
    assert stat.p95_ms == 50.0
    assert abs(stat.throughput_ips - 20.0) < 1e-9


def test_render_table_formats_and_handles_missing():
    rows = [
        BenchStat("pytorch", 12.3, 20.1, 30.0, 45.0, size_mb=350.0, accuracy=0.812),
        BenchStat("onnx-int8", 4.5, 6.7, 5.0, 200.0, size_mb=None, accuracy=None),
    ]
    table = render_table(rows)
    assert "| variant |" in table
    assert "pytorch" in table and "350.0" in table and "0.8120" in table
    # Missing size/accuracy render as em dash.
    assert "| onnx-int8 | — |" in table
    assert table.count("\n") == 3  # header sep + 2 rows
