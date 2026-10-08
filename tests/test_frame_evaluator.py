"""Production core ownership, numerical parity and error separation."""

from __future__ import annotations

import ctypes
import math
import os
import random
import subprocess
import sys
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
from frame_reference import _tile_plan, activity_levels, layout_at
from test_render import colored_activity_document, rendered_document

from stavellum._core import CoreBackendError, CoreTarget, library_path
from stavellum._frame import FrameEvaluator
from stavellum.gpu import GpuBackendError
from stavellum.layout import _CurveKey, _Track
from stavellum.render import FrameRenderer, RasterFrameRenderer
from stavellum.rhi import RhiFrameRenderer
from stavellum.scene import compile_scene


@pytest.fixture(scope="module")
def core_scene():
    document = colored_activity_document(intro_delay=2)
    document.settings.announcement_auto_hide = True
    return compile_scene(document)


def test_states_are_independent_readonly_and_cache_is_bounded(core_scene, monkeypatch):
    with FrameEvaluator(core_scene) as evaluator:
        calls = []
        original = evaluator._target.frame

        def frame(*times):
            calls.append(times)
            return original(*times)

        monkeypatch.setattr(evaluator._target, "frame", frame)
        first = evaluator.evaluate(3.01)
        assert first.audio_time == pytest.approx(1.01)
        saved = first.layout
        assert evaluator.evaluate(3.01) is first
        with pytest.raises(TypeError):
            first.layout.rows["part"] = None
        with pytest.raises(TypeError):
            first.activity["part"] = (0, 0)
        for index in range(6):
            evaluator.evaluate(index + 10)
        assert len(evaluator._states) == 5
        assert first.layout == saved
        assert evaluator.evaluate(3.01) == first
        assert len(calls) == 8
        assert evaluator.report()["scene_evaluation_count"] == 8


def test_settings_are_snapshotted_and_intro_speed_is_zero(core_scene):
    scene = replace(core_scene, settings=replace(core_scene.settings))
    with FrameEvaluator(scene) as evaluator:
        scene.settings.intro_delay_seconds = 99
        for t in (-1, 0, 1.999):
            state = evaluator.evaluate(t)
            assert state.audio_time == 0 and state.camera_speed == 0
        state = evaluator.evaluate(3)
        assert state.audio_time == 1
        assert state.camera_speed == pytest.approx(scene.camera.speed_at(1))


def test_closed_and_nonfinite_queries_are_clear_errors(core_scene):
    evaluator = FrameEvaluator(core_scene)
    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(CoreBackendError, match="有限"):
            evaluator.evaluate(value)
        with pytest.raises(CoreBackendError, match="有限"):
            evaluator._target.frame(0, value)
    evaluator.close()
    evaluator.close()
    with pytest.raises(CoreBackendError, match="关闭"):
        evaluator.evaluate(0)
    with pytest.raises(CoreBackendError, match="关闭"):
        evaluator._target.frame(0, 0)


def test_missing_core_fails_cpu_rendering_with_install_instruction(core_scene, monkeypatch, tmp_path):
    missing = tmp_path / "missing-core.dll"
    monkeypatch.setenv("STAVELLUM_CORE_DLL", str(missing))
    with pytest.raises(CoreBackendError) as failure:
        RasterFrameRenderer(core_scene)
    assert str(missing) in str(failure.value)
    assert "scripts/build_rust.py --install" in str(failure.value)
    assert not isinstance(failure.value, GpuBackendError)


def test_loader_never_chooses_development_outputs(monkeypatch):
    monkeypatch.delenv("STAVELLUM_CORE_DLL", raising=False)
    assert library_path().parts[-3:] == ("native", "core", "stavellum_core.dll")


def test_invalid_library_missing_symbols_and_abi_fail(core_scene, monkeypatch, tmp_path):
    invalid = tmp_path / "invalid.dll"
    invalid.write_bytes(b"not a DLL")
    monkeypatch.setenv("STAVELLUM_CORE_DLL", str(invalid))
    with pytest.raises(CoreBackendError, match="无法加载"):
        CoreTarget(core_scene)
    monkeypatch.setattr(ctypes, "CDLL", lambda _: SimpleNamespace())
    with pytest.raises(CoreBackendError, match="无法加载"):
        CoreTarget(core_scene)
    monkeypatch.setattr(CoreTarget, "_bind", lambda *args: None)
    monkeypatch.setattr(ctypes, "CDLL", lambda _: SimpleNamespace(spcore_abi_version=lambda: 9))
    with pytest.raises(CoreBackendError, match="ABI"):
        CoreTarget(core_scene)


def test_asset_initialization_failure_releases_core(core_scene, monkeypatch):
    closed = []
    original = FrameEvaluator.close

    def close(self):
        closed.append(self)
        original(self)

    def fail(self):
        raise ValueError("asset initialization failed")

    monkeypatch.setattr(FrameEvaluator, "close", close)
    monkeypatch.setattr(RasterFrameRenderer, "_prepare_assets", fail)
    with pytest.raises(ValueError, match="asset initialization"):
        RasterFrameRenderer(core_scene)
    assert len(closed) == 1 and closed[0]._closed


@pytest.mark.parametrize("direct", [False, True])
def test_gpu_initialization_failure_releases_owned_core(core_scene, monkeypatch, direct):
    evaluators = []
    original = FrameEvaluator.__init__

    def initialize(self, scene):
        original(self, scene)
        evaluators.append(self)

    def fail(*args):
        raise GpuBackendError("graphics initialization failed")

    monkeypatch.setattr(FrameEvaluator, "__init__", initialize)
    monkeypatch.setattr("stavellum.rhi.RhiTarget", fail)
    scene = replace(core_scene, settings=replace(core_scene.settings, render_backend="gpu"))
    with pytest.raises(GpuBackendError, match="initialization failed"):
        (RhiFrameRenderer if direct else FrameRenderer)(scene)
    assert len(evaluators) == 1 and evaluators[0]._closed
    assert not evaluators[0]._target._finalizer.alive


def test_plain_import_and_inspect_do_not_load_core(tmp_path):
    from mido import Message, MidiFile, MidiTrack

    midi = MidiFile()
    midi.tracks.append(MidiTrack([
        Message("note_on", note=60, velocity=100),
        Message("note_off", note=60, velocity=0, time=480),
    ]))
    path = tmp_path / "inspect.mid"
    midi.save(path)
    script = "import stavellum.scene, stavellum.render; from stavellum.cli import main; import sys; sys.exit(main(['inspect',sys.argv[1]]))"
    result = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=30,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                            env={**os.environ, "STAVELLUM_CORE_DLL": str(tmp_path / "missing.dll")})
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"notes": 1' in result.stdout


def test_rhi_batch_plan_and_commands_share_evaluator(core_scene, monkeypatch):
    # The GPU capability is irrelevant to the scene evaluator ownership contract.
    from test_rhi import FakeTarget

    monkeypatch.setattr("stavellum.rhi.RhiTarget", FakeTarget)
    with RasterFrameRenderer(core_scene) as assets:
        with RhiFrameRenderer(core_scene, assets=assets) as renderer:
            renderer.batch_plan(3)
            renderer.commands(3)
            assert renderer._assets._evaluator is assets._evaluator
            assert assets._evaluator.evaluation_count == 1
        assert not assets._evaluator._closed
        assets.render_frame(3)
        assert assets._evaluator.evaluation_count == 1
    assert assets._evaluator._closed


def test_core_failure_propagates_without_gpu_fallback(core_scene, monkeypatch):
    from test_rhi import FakeTarget

    monkeypatch.setattr("stavellum.rhi.RhiTarget", FakeTarget)
    scene = replace(core_scene, settings=replace(core_scene.settings, render_backend="auto"))
    with FrameRenderer(scene) as renderer:
        def fail(*args):
            raise CoreBackendError("core operation failed")

        monkeypatch.setattr(renderer._evaluator._target, "frame", fail)
        with pytest.raises(CoreBackendError, match="core operation"):
            renderer.render_frame(3)
        assert renderer.render_backend == "gpu"
        assert renderer.fallback_reasons == []
        assert renderer.frame_count == 0
    assert renderer._evaluator._closed


def test_native_handle_released_once(core_scene, monkeypatch):
    target = CoreTarget(core_scene)
    released = []
    original = target._dll.spcore_close

    def close(handle):
        released.append(handle)
        return original(handle)

    monkeypatch.setattr(target._dll, "spcore_close", close)
    target.close()
    target.close()
    assert len(released) == 1
    assert not target._finalizer.alive


def test_gpu_recovery_reports_current_shared_core_counters(core_scene, monkeypatch):
    from test_rhi import FakeTarget

    class LostTarget(FakeTarget):
        def render(self, commands):
            raise GpuBackendError("lost device")

    monkeypatch.setattr("stavellum.rhi.RhiTarget", LostTarget)
    scene = replace(core_scene, settings=replace(core_scene.settings, render_backend="auto"))
    with FrameRenderer(scene) as renderer:
        renderer.render_frame(3)
        assert renderer.render_backend == "cpu"
        evaluator = renderer._evaluator
        renderer.render_frame(4)
        assert renderer._evaluator is evaluator
        assert renderer.backend_report()["scene_evaluation_count"] == 2
        assert renderer.backend_report()["scene_evaluation_cache_hits"] == 1


@pytest.mark.parametrize("direct", [False, True])
def test_graphics_cleanup_failure_still_releases_core(core_scene, monkeypatch, direct):
    from test_rhi import FakeTarget

    class FailedCloseTarget(FakeTarget):
        def close(self):
            raise GpuBackendError("graphics cleanup failed")

    monkeypatch.setattr("stavellum.rhi.RhiTarget", FailedCloseTarget)
    scene = replace(core_scene, settings=replace(core_scene.settings, render_backend="gpu"))
    renderer = RhiFrameRenderer(scene) if direct else FrameRenderer(scene)
    evaluator = renderer._assets._evaluator if direct else renderer._evaluator
    with pytest.raises(GpuBackendError, match="cleanup failed"):
        renderer.close()
    assert evaluator._closed
    assert not evaluator._target._finalizer.alive
    renderer.close()


@pytest.mark.parametrize("piano", [False, True])
@pytest.mark.parametrize("audio_offset", [-1.25, 0, 2.5])
def test_all_layout_fields_activity_and_residency_match_python(piano, audio_offset):
    document = rendered_document(piano=True) if piano else colored_activity_document(intro_delay=2)
    if not piano:
        document.project.notes[0] = replace(document.project.notes[0], velocity=0)
    document.settings.score_start_in_audio_sec = audio_offset
    document.settings.announcement_auto_hide = True
    scene = compile_scene(document)
    rng = random.Random(572)
    times = [-1, 0, scene.settings.intro_delay_seconds]
    times += [rng.uniform(-1, scene.score_duration + 5) for _ in range(120)]
    for track in [scene.layout.zoom, *scene.layout.tops.values(), *scene.layout.opacities.values()]:
        for key in track.keys:
            times.extend((key.time - 1e-6, key.time, key.time + 1e-6))
    for part in scene.parts:
        for note in part.notes:
            for boundary in (note.start, note.start + 0.1, note.end, note.end + 0.12):
                t = scene.settings.presentation_time(boundary)
                times.extend((t - 1e-6, t, t + 1e-6))
    rng.shuffle(times)
    with RasterFrameRenderer(scene) as assets:
        for t in times:
            state = assets._evaluator.evaluate(t)
            expected = layout_at(scene.layout, t)
            assert state.world_x == pytest.approx(scene.camera_x_at(t), rel=1e-9, abs=1e-9)
            assert state.camera_speed == pytest.approx(scene.camera_speed_at(t), rel=1e-9, abs=1e-9)
            assert state.layout.scale == pytest.approx(expected.scale, rel=1e-9, abs=1e-9)
            assert state.layout.bounds == pytest.approx(expected.bounds, rel=1e-9, abs=1e-9)
            assert state.layout.region_bounds == pytest.approx(expected.region_bounds, rel=1e-9, abs=1e-9)
            for part in scene.parts:
                row, reference = state.layout.rows[part.part_id], expected.rows[part.part_id]
                for field in fields(row):
                    assert getattr(row, field.name) == pytest.approx(
                        getattr(reference, field.name), rel=1e-9, abs=1e-9)
                assert state.activity[part.part_id] == pytest.approx(
                    activity_levels(part, state.audio_time), rel=1e-9, abs=1e-9)
            for budget in (0, 1, 8 * 1024 * 1024):
                assets.cache_limit = budget
                assert assets._tile_plan(state) == _tile_plan(assets, expected, scene.camera_x_at(t))


def test_native_abi_rejects_invalid_times_and_degenerate_zoom(core_scene):
    with CoreTarget(core_scene) as core:
        rows = (ctypes.c_double * (11 * len(core_scene.parts)))()
        output = (ctypes.c_double * 16)()
        assert core._dll.spcore_frame(core._handle, math.nan, 0, output, rows, len(core_scene.parts))
        assert b"finite" in core._dll.spcore_last_error()
    layout = replace(core_scene.layout, zoom=_Track([_CurveKey(0, -1000)]))
    with CoreTarget(replace(core_scene, layout=layout)) as core:
        with pytest.raises(CoreBackendError, match="finite range"):
            core.frame(0, 0)
