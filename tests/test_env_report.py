"""Compute-plan recommendation is driven by VRAM, not hardcoded."""

from __future__ import annotations

from wildlife.utils.env_report import _recommend_plan, collect_report, format_report


def test_no_cuda_gives_cpu_fallback():
    plan = _recommend_plan(None, has_cuda=False)
    assert "CPU" in plan.tier or "fallback" in plan.tier.lower()
    assert plan.amp is False


def test_12gb_selects_base_tier():
    plan = _recommend_plan(12.0, has_cuda=True)
    assert plan.tier == "Base"
    assert plan.batch_size * plan.grad_accum == 64


def test_high_vram_selects_large():
    assert _recommend_plan(24.0, has_cuda=True).tier == "Large"


def test_report_formats_without_error():
    report = collect_report()
    plan = _recommend_plan(report["vram_gb"], report["cuda_available"])
    text = format_report(report, plan)
    assert "ENVIRONMENT REPORT" in text
    assert "COMPUTE PLAN" in text
