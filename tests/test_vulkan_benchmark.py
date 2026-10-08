from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

path = Path(__file__).resolve().parents[1] / "scripts/verify_vulkan.py"
spec = importlib.util.spec_from_file_location("vulkan_benchmark", path)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


COMPARISON_MODES = ("baseline-opengl", "rhi-vulkan")


def results(current=10, vulkan=8):
    return [{"case": label, "mode": mode, "pipeline_seconds": value}
            for label, _ in benchmark.CASES
            for mode, value in zip(COMPARISON_MODES, (current, vulkan), strict=True)
            for _ in range(3)]


def test_historical_gate_requires_review_and_meaningful_export_gain():
    _, gate = benchmark.medians_and_gate(results(), True, COMPARISON_MODES)
    assert gate["full_song_gate_passed"]
    assert gate["export_reduction_vs_baseline_opengl"] == pytest.approx(.2)
    assert gate["historical_performance_comparison"]
    assert not benchmark.medians_and_gate(results(), False, COMPARISON_MODES)[1]["full_song_gate_passed"]
    assert not benchmark.medians_and_gate(results(vulkan=9.1), True, COMPARISON_MODES)[1]["full_song_gate_passed"]


def test_default_vulkan_verification_does_not_invent_historical_performance():
    rows = [row for row in results() if row["mode"] == "rhi-vulkan"]
    _, gate = benchmark.medians_and_gate(rows, True)
    assert benchmark.MODES == ("rhi-vulkan",)
    assert gate["full_song_gate_passed"]
    assert gate["export_reduction_vs_baseline_opengl"] is None
    assert gate["any_case_regresses_over_5_percent"] is None
    assert not gate["historical_performance_comparison"]
    assert not benchmark.medians_and_gate(rows, False)[1]["full_song_gate_passed"]


def test_one_regressing_case_blocks_full_song_even_with_large_total_gain():
    rows = results(vulkan=5)
    for row in rows:
        if row["case"] == "dense" and row["mode"] == "rhi-vulkan":
            row["pipeline_seconds"] = 10.6
    _, gate = benchmark.medians_and_gate(rows, True, COMPARISON_MODES)
    assert gate["export_reduction_vs_baseline_opengl"] > .1
    assert gate["any_case_regresses_over_5_percent"] and not gate["full_song_gate_passed"]


def test_gpu_identity_does_not_confuse_api_prefixes_or_accept_software():
    gl = {"gpu_info": {"renderer": "NVIDIA Corporation NVIDIA GeForce Example GPU/PCIe/SSE2 4.6"}}
    vk = {"gpu_info": {"renderer": "NVIDIA GeForce Example GPU"}}
    assert benchmark.gpu_identity(gl) == benchmark.gpu_identity(vk)
    assert benchmark.gpu_identity({"gpu_info": {"renderer": "Intel Arc GPU"}}) == "intel arc gpu"
    with pytest.raises(RuntimeError, match="hardware GPU"):
        benchmark.gpu_identity({"gpu_info": {"renderer": "llvmpipe"}})


def test_failed_quality_rerun_replaces_previous_success_report(tmp_path, monkeypatch):
    from stavellum.domain.models import RenderSettings
    destination = tmp_path / "benchmark.json"
    destination.write_text('{"status":"complete","results":["old"]}', encoding="utf-8")
    document = SimpleNamespace(settings=RenderSettings(), validate=lambda: None)
    monkeypatch.setattr(sys, "argv", ["verify_vulkan", "fixture.stproj", "--output", str(tmp_path)])
    monkeypatch.setattr(benchmark, "load_document", lambda _: document)
    monkeypatch.setattr(benchmark, "prepare_render_app", lambda _: None)
    monkeypatch.setattr(benchmark, "compile_scene", lambda _: SimpleNamespace(settings=document.settings))
    monkeypatch.setattr(benchmark.shutil, "which", lambda _: "fixture-encoder")
    def unavailable(*args):
        raise RuntimeError("Vulkan initialization unavailable")
    monkeypatch.setattr(benchmark, "quality_check", unavailable)
    with pytest.raises(RuntimeError, match="initialization unavailable"):
        benchmark.main()
    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["status"] == "failed" and report["results"] == []
    assert "initialization unavailable" in report["error"]
