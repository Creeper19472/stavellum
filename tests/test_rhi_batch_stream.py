"""Bounded Vulkan batches preserve every unyielded time and image owner."""

from __future__ import annotations

import math
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtGui import QColor, QImage
from test_export_stream import paint_marker, rgba_pixels
from test_gpu import FakeVulkanRenderer
from test_render import rendered_document
from test_rhi import FakeTarget

from stavellum.presentation.scene import compile_scene
from stavellum.rendering._rhi import library_path
from stavellum.rendering.gpu import GpuBackendError
from stavellum.rendering.render import FrameRenderer
from stavellum.rendering.rhi import RhiFrameRenderer
from stavellum.rendering.shared import _ExportTimes


class FakeBatchTarget(FakeTarget):
    def __init__(self, *args):
        super().__init__(*args)
        self.batches = []
        self.cancel_after_submit = None

    def render_batch(self, commands_by_frame):
        self.batches.append(commands_by_frame)
        images = [super(FakeBatchTarget, self).render(commands)
                  for commands in commands_by_frame]
        if self.cancel_after_submit:
            self.cancel_after_submit()
        return images

    def render(self, commands):
        return self.render_batch([commands])[0]

    def report(self):
        histogram = Counter(str(len(batch)) for batch in self.batches)
        size = self.width * self.height * 4
        peak = max((len(batch) for batch in self.batches), default=0)
        return {**super().report(), "gpu_submission_count": len(self.batches),
                "gpu_submitted_frame_count": len(self.submitted),
                "gpu_batch_size_histogram": dict(histogram), "gpu_batch_peak_size": peak,
                "readback_output_peak_bytes": peak * size,
                "readback_buffer_peak_bytes": peak * size,
                "readback_mode": "rhi-batch-sync" if peak > 1 else "rhi-sync"}


@pytest.fixture(scope="module")
def batch_scene():
    return compile_scene(rendered_document())


@pytest.fixture
def fake_batch_target(monkeypatch):
    monkeypatch.setattr("stavellum.rendering.rhi.RhiTarget", FakeBatchTarget)
    monkeypatch.setenv("STAVELLUM_RHI_BATCH_SIZE", "4")


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 5, 7, 8, 9])
def test_direct_stream_drains_short_final_batches_and_keeps_owned_images(
        batch_scene, fake_batch_target, count):
    with RhiFrameRenderer(batch_scene) as renderer:
        stream = renderer.export_frames([1.0] * count, lambda: False)
        frames = list(stream)
        assert [index for index, _ in frames] == list(range(count))
        assert [len(batch) for batch in renderer._target.batches] == (
            [4] * (count // 4) + ([count % 4] if count % 4 else []))
        assert renderer.frame_count == count
        assert stream.report()["yielded_frame_count"] == count
        assert stream.report()["frame_stream_pending_peak_bytes"] <= 4 * 640 * 360 * 4
    assert all(image.pixelColor(3, 3) == QColor("red") for _, image in frames)


@pytest.mark.parametrize(("width", "height", "expected"), [
    (1920, 1080, 4), (2560, 1440, 4), (3840, 2160, 2), (7680, 4320, 1)])
def test_batch_output_cap_never_prevents_a_single_frame(width, height, expected):
    renderer = RhiFrameRenderer.__new__(RhiFrameRenderer)
    renderer.scene = SimpleNamespace(settings=SimpleNamespace(width=width, height=height))
    renderer._requested_batch_size = 4
    renderer._target = SimpleNamespace(render_batch=lambda commands: commands)
    assert renderer.export_batch_limit == expected


@pytest.mark.parametrize("size", [1, 2, 4])
def test_diagnostic_batch_size_and_public_single_frame_remain_independent(
        batch_scene, fake_batch_target, monkeypatch, size):
    monkeypatch.setenv("STAVELLUM_RHI_BATCH_SIZE", str(size))
    with RhiFrameRenderer(batch_scene) as renderer:
        image = renderer.render_frame(1)
        assert image.format() == QImage.Format.Format_RGBA8888
        assert len(renderer._target.batches) == 1
        assert len(renderer._target.batches[0]) == 1
        stream = renderer.export_frames([1] * 5, lambda: False)
        assert next(stream)[0] == 0
        assert len(renderer._target.batches[1]) == size
        assert all(image.format() == QImage.Format.Format_ARGB32
                   for _, image in stream)


def test_same_plan_keeps_all_tile_image_ids_stable_between_prepared_frames(
        batch_scene, fake_batch_target):
    with RhiFrameRenderer(batch_scene) as renderer:
        first_plan = renderer.batch_plan(1)
        assert first_plan is not None
        images = renderer.render_batch_times([1, 1.001, 1.002, 1.003], lambda: False)
        assert len(images) == 4
        batch = renderer._target.batches[0]
        texture_sets = [{quad.texture_id for quad in commands if quad.texture_id}
                        for commands in batch]
        assert all(keys == texture_sets[0] for keys in texture_sets)
        assert renderer._assets.cache_peak_bytes <= renderer._assets.cache_limit
        assert renderer._assets.tile_cache_evictions == 0


def test_direct_stream_counters_freeze_before_renderer_reuse(batch_scene, fake_batch_target):
    with RhiFrameRenderer(batch_scene) as renderer:
        first = renderer.export_frames([1] * 5, lambda: False)
        assert len(list(first)) == 5
        first.close()
        second = renderer.export_frames([1] * 2, lambda: False)
        assert len(list(second)) == 2
        second.close()
        assert first.report()["gpu_submitted_frame_count"] == 5
        assert first.report()["gpu_batch_size_histogram"] == {"4": 1, "1": 1}
        assert second.report()["gpu_submitted_frame_count"] == 2
        assert second.report()["gpu_batch_size_histogram"] == {"4": 0, "1": 0, "2": 1}


def test_over_budget_visible_tiles_are_prepared_and_submitted_one_at_a_time(
        batch_scene, fake_batch_target):
    with RhiFrameRenderer(batch_scene) as renderer:
        renderer._assets.cache_limit = 0
        assert renderer.batch_plan(1) is None
        stream = renderer.export_frames([1, 1, 1], lambda: False)
        assert len(list(stream)) == 3
        assert [len(batch) for batch in renderer._target.batches] == [1, 1, 1]
        assert stream.report()["readback_mode"] == "rhi-sync"
        assert stream.report()["readback_mode_history"] == ["rhi-sync"]
        assert stream.report()["requested_readback_mode"] == "rhi-batch-sync"
        assert stream.report()["batch_budget_split_count"] == 3
        assert renderer._assets.cache_peak_bytes == 0


def test_tile_boundary_lookahead_is_reused_without_reconsuming_iterator():
    consumed = []

    def times():
        for time in [0, .1, .2, 1, 1.1, 99, 2, 2.1]:
            consumed.append(time)
            yield time

    gpu = SimpleNamespace(export_batch_limit=4,
                          batch_plan=lambda time: None if time == 99 else math.floor(time))
    source = _ExportTimes(times(), lambda: False)
    assert not consumed
    assert source.take_batch(gpu) == [0, .1, .2]
    assert consumed == [0, .1, .2, 1]
    assert source.take_batch(gpu) == [1, 1.1]
    assert consumed == [0, .1, .2, 1, 1.1, 99]
    assert source.take_batch(gpu) == [99]
    assert source.take_batch(gpu) == [2, 2.1]
    with pytest.raises(StopIteration):
        source.take_batch(gpu)
    assert source.tile_splits == 1 and source.budget_splits == 2


class FakeBatchGpu(FakeVulkanRenderer):
    export_batch_limit = 4
    export_readback_mode = "rhi-batch-sync"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batches = []
        self.generated = 0
        self.fail_batch = None
        self.failure_stage = "before"
        self.after_submit = None

    def batch_plan(self, time):
        return "same-resident-tiles"

    def render_batch_times(self, times, cancel):
        index = len(self.batches)
        self.batches.append(list(times))
        if index == self.fail_batch and self.failure_stage == "before":
            raise GpuBackendError("fake batch device lost")
        images = []
        for time in times:
            if cancel():
                raise InterruptedError("cancelled during batch prepare")
            images.append(self._render(time))
        self.generated += len(images)
        if index == self.fail_batch:
            raise GpuBackendError("fake batch device lost")
        if self.after_submit:
            self.after_submit()
        return images

    def backend_report(self):
        histogram = Counter(str(len(batch)) for batch in self.batches)
        peak = max((len(batch) for batch in self.batches), default=0)
        size = self.scene.settings.width * self.scene.settings.height * 4
        return {**super().backend_report(), "gpu_submission_count": len(self.batches),
                "gpu_submitted_frame_count": self.generated,
                "gpu_batch_size_histogram": dict(histogram), "gpu_batch_peak_size": peak,
                "readback_buffer_peak_bytes": peak * size,
                "readback_output_peak_bytes": peak * size,
                "readback_staging_peak_bytes": peak * size}


@pytest.fixture
def production_batch(batch_scene, monkeypatch):
    monkeypatch.setattr("stavellum.rendering.rhi.RhiFrameRenderer", FakeBatchGpu)
    renderers = []

    def make(backend="auto"):
        scene = replace(batch_scene, settings=replace(batch_scene.settings, render_backend=backend))
        renderer = FrameRenderer(scene)
        renderer.paint_frame = paint_marker
        renderers.append(renderer)
        return renderer

    yield make
    for renderer in renderers:
        renderer.close()


@pytest.mark.parametrize("stage", ["before", "after"])
@pytest.mark.parametrize("failure_batch", [0, 1])
def test_auto_batch_failure_replays_all_unyielded_times_once(
        production_batch, stage, failure_batch):
    renderer = production_batch()
    gpu = renderer._gpu
    gpu.fail_batch, gpu.failure_stage = failure_batch, stage
    consumed = []
    times = [1.25, 0, 8.5, .125, .5, 2, 9, 0, 1.25]

    def source():
        for seconds in times:
            consumed.append(seconds)
            yield seconds

    stream = renderer.export_frames(source(), lambda: False)
    frames = list(stream)
    reference = production_batch("cpu")
    expected = [rgba_pixels(reference.render_frame(time)) for time in times]
    assert consumed == times
    assert [index for index, _ in frames] == list(range(len(times)))
    assert [rgba_pixels(image) for _, image in frames] == expected
    assert renderer.frame_count == len(times)
    assert stream.pixel_format == "bgra"
    assert stream.report()["readback_mode_history"] == (
        ["rhi-batch-sync", "cpu"] if failure_batch else ["cpu"])
    assert gpu.closed and renderer._gpu is None
    assert stream.report()["cpu_format_conversion_bytes"] == (
        len(times) - failure_batch * 4) * 640 * 360 * 4


def test_gpu_batch_failure_remains_fatal_without_reyielding_previous_batch(production_batch):
    renderer = production_batch("gpu")
    gpu = renderer._gpu
    gpu.fail_batch = 1
    stream = renderer.export_frames(range(9), lambda: False)
    previous = [next(stream) for _ in range(4)]
    saved = [rgba_pixels(image) for _, image in previous]
    with pytest.raises(GpuBackendError, match="fake batch device lost"):
        next(stream)
    assert renderer.frame_count == 4 and gpu.closed
    assert renderer._export_stream is None and renderer._closed
    assert [rgba_pixels(image) for _, image in previous] == saved


def test_buffered_cancellation_keeps_last_yielded_image_and_cleanup_order(production_batch):
    renderer = production_batch()
    state = {"cancel": False}
    stream = renderer.export_frames(range(12), lambda: state["cancel"])
    _, retained = next(stream)
    saved = rgba_pixels(retained)
    assert renderer.frame_count == 1 and renderer._gpu.generated == 4
    state["cancel"] = True
    with pytest.raises(InterruptedError):
        next(stream)
    assert renderer._export_stream is stream  # the encoder owner must close it
    assert len(stream._ready) == 3
    stream.close()
    stream.close()
    renderer.close()
    assert not stream._ready and renderer.frame_count == 1
    assert rgba_pixels(retained) == saved


def test_cancellation_after_submission_never_yields_a_prepared_frame(production_batch):
    renderer = production_batch()
    state = {"cancel": False}
    renderer._gpu.after_submit = lambda: state.update(cancel=True)
    stream = renderer.export_frames(range(12), lambda: state["cancel"])
    with pytest.raises(InterruptedError):
        next(stream)
    assert renderer.frame_count == 0 and renderer._gpu.generated == 4
    assert len(stream._ready) == 4
    stream.close()
    assert not stream._ready


def test_cancellation_during_commands_never_submits_partial_batch(
        batch_scene, fake_batch_target, monkeypatch):
    with RhiFrameRenderer(batch_scene) as renderer:
        state = {"cancel": False}
        original = renderer.commands
        prepared = []

        def commands(seconds):
            result = original(seconds)
            prepared.append(seconds)
            if len(prepared) == 2:
                state["cancel"] = True
            return result

        monkeypatch.setattr(renderer, "commands", commands)
        stream = renderer.export_frames([1] * 8, lambda: state["cancel"])
        with pytest.raises(InterruptedError):
            next(stream)
        assert prepared == [1, 1] and not renderer._target.batches
        assert renderer.frame_count == 0
        stream.close()


def test_new_stream_counters_exclude_previous_attempts_and_merge_histogram_once(production_batch):
    renderer = production_batch()
    first = renderer.export_frames(range(5), lambda: False)
    assert len(list(first)) == 5
    first.close()
    first.close()
    assert first.report()["gpu_submission_count"] == 2
    assert first.report()["gpu_batch_size_histogram"] == {"4": 1, "1": 1}
    second = renderer.export_frames(range(3), lambda: False)
    assert len(list(second)) == 3
    second.close()
    assert second.report()["gpu_submission_count"] == 1
    assert second.report()["gpu_submitted_frame_count"] == 3
    assert second.report()["gpu_batch_size_histogram"] == {"4": 0, "1": 0, "3": 1}
    totals = renderer._export_report
    assert totals["gpu_submission_count"] == 3
    assert totals["gpu_batch_size_histogram"] == {"4": 1, "1": 1, "3": 1}
    assert totals["yielded_frame_count"] == 8 and renderer.frame_count == 8
    assert totals["gpu_batch_peak_size"] == 4
    assert totals["frame_stream_pending_peak_bytes"] == 4 * 640 * 360 * 4
    assert first.report()["gpu_submission_count"] == 2
    assert first.report()["gpu_batch_size_histogram"] == {"4": 1, "1": 1}


def test_closed_stream_fallback_reasons_remain_independent_of_later_streams(production_batch):
    renderer = production_batch()
    gpu = renderer._gpu
    first = renderer.export_frames([0], lambda: False)
    assert len(list(first)) == 1
    first.close()
    assert first.report()["readback_fallback_reasons"] == []

    gpu.fail_batch = len(gpu.batches)
    second = renderer.export_frames([.5, 1], lambda: False)
    assert len(list(second)) == 2
    second.close()
    assert first.report()["readback_fallback_reasons"] == []
    assert second.report()["readback_fallback_reasons"] == ["fake batch device lost"]

    report = second.report()
    report["readback_fallback_reasons"].append("caller mutation")
    assert second.report()["readback_fallback_reasons"] == ["fake batch device lost"]
    assert renderer._export_report["readback_fallback_reasons"] == ["fake batch device lost"]
    renderer.close()
    assert first.report()["readback_fallback_reasons"] == []


REAL_BATCH_STREAM = r'''
import os, sys
from dataclasses import replace
sys.path.insert(0, "tests")
from test_render import rendered_document
from PySide6.QtGui import QImage
from stavellum.domain.models import RenderSettings
from stavellum.graphics.qt import prepare_render_app
from stavellum.rendering.render import FrameRenderer
from stavellum.presentation.scene import compile_scene

os.environ.pop("STAVELLUM_RHI_COPY_READBACK", None)
os.environ["STAVELLUM_RHI_RGBA_READBACK"] = sys.argv[1]
prepare_render_app(RenderSettings(render_backend="gpu"))
document = rendered_document(piano=True)
document.settings.render_backend = "gpu"
scene = compile_scene(document)
times = [0, .125, .5, .5, 1, 1.001, 1.002, 1.003, 2, .1, 1, 1]

def pixels(image):
    image = image.convertToFormat(QImage.Format.Format_RGBA8888)
    return bytes(image.constBits())

for budget in (64, 0):
    current = replace(scene, settings=replace(scene.settings, cache_megabytes=budget))
    os.environ["STAVELLUM_RHI_BATCH_SIZE"] = "1"
    with FrameRenderer(current) as reference:
        stream = reference.export_frames(times, lambda: False)
        expected = [pixels(image) for _, image in stream]
    os.environ["STAVELLUM_RHI_BATCH_SIZE"] = "4"
    with FrameRenderer(current) as renderer:
        stream = renderer.export_frames(times, lambda: False)
        frames = list(stream)
        report = stream.report()
        assert [index for index, _ in frames] == list(range(len(times)))
        assert [pixels(image) for _, image in frames] == expected
        assert renderer.frame_count == len(times)
        assert report["memory_copy_bytes"] == 0
        assert report["gpu_batch_peak_size"] == (4 if budget else 1), report
        assert ("rhi-batch-sync" in report["readback_mode_history"]) == bool(budget), report
        retained = frames[0][1]
        saved = pixels(retained)
    assert pixels(retained) == saved

os.environ["STAVELLUM_RHI_BATCH_SIZE"] = "4"
with FrameRenderer(scene) as renderer:
    cancelled = [False]
    stream = renderer.export_frames([1] * 8, lambda: cancelled[0])
    index, retained = next(stream)
    saved = pixels(retained)
    assert index == 0 and renderer.frame_count == 1
    assert len(stream._ready) == 3
    cancelled[0] = True
    try:
        next(stream)
        raise AssertionError("buffered frame escaped cancellation")
    except InterruptedError:
        pass
    assert renderer._export_stream is stream
    stream.close()
    other = renderer.export_frames([1, 1], lambda: False)
    assert [index for index, _ in other] == [0, 1]
assert pixels(retained) == saved
'''


@pytest.mark.integration
@pytest.mark.parametrize("force_rgba", ["0", "1"])
@pytest.mark.skipif(sys.platform != "win32", reason="Windows native renderer required")
def test_actual_gpu_batch_stream_matches_single_frames_and_survives_close(force_rgba):
    if not library_path().is_file():
        pytest.skip("Native Vulkan renderer DLL has not been built")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", REAL_BATCH_STREAM, force_rgba], cwd=root,
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=45, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert result.returncode == 0, result.stdout + result.stderr
