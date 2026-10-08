"""Production frame stream ordering, ownership and Vulkan failure recovery."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QImage
from test_gpu import FakeVulkanRenderer
from test_render import rendered_document

from stavellum.presentation.scene import compile_scene
from stavellum.rendering.gpu import GpuBackendError
from stavellum.rendering.render import FrameRenderer


@pytest.fixture(scope="module")
def stream_scene():
    return compile_scene(rendered_document())


def paint_marker(painter, seconds):
    shade = round(seconds * 40) % 256
    painter.fillRect(QRectF(0, 0, 640, 360), QColor(shade, 17, 201))
    painter.fillRect(QRectF(7, 11, 31, 29), QColor("red"))
    painter.fillRect(QRectF(557, 299, 39, 31), QColor("blue"))


def rgba_pixels(image):
    normalized = image.convertToFormat(QImage.Format.Format_RGBA8888)
    return bytes(normalized.constBits())


class FakeRenderer(FakeVulkanRenderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submitted = []
        self.fail_at = None
        self.failure_stage = "before"

    def _render(self, seconds):
        index = len(self.submitted)
        self.submitted.append(seconds)
        if index == self.fail_at and self.failure_stage == "before":
            raise GpuBackendError("fake Vulkan readback failure")
        image = super()._render(seconds)
        if index == self.fail_at:
            raise GpuBackendError("fake Vulkan readback failure")
        return image


@pytest.fixture
def fake_renderer(stream_scene, monkeypatch):
    monkeypatch.setattr("stavellum.rendering.rhi.RhiFrameRenderer", FakeRenderer)
    renderers = []

    def make(*, backend="auto"):
        stream_scene.settings.render_backend = backend
        renderer = FrameRenderer(stream_scene)
        renderer.paint_frame = paint_marker
        renderers.append(renderer)
        return renderer

    yield make
    for renderer in renderers:
        renderer.close()
    stream_scene.settings.render_backend = "cpu"


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 7])
def test_vulkan_stream_preserves_all_short_and_long_sequences(fake_renderer, count):
    renderer = fake_renderer()
    times = [1.25, 0, 8.5, .125, .5, 2, 9][:count]
    stream = renderer.export_frames(times, lambda: False)
    assert stream.pixel_format == "bgra"
    frames = list(stream)
    stream.close()
    assert [index for index, _ in frames] == list(range(count))
    assert renderer.frame_count == count
    assert renderer._gpu.submitted == times
    for (_, image), seconds in zip(frames, times, strict=True):
        assert image.pixelColor(20, 20) == QColor("red")
        assert image.pixelColor(575, 315) == QColor("blue")
        assert image.pixelColor(100, 100) == QColor(round(seconds * 40) % 256, 17, 201)


@pytest.mark.parametrize("stage", ["before", "after"])
@pytest.mark.parametrize("failure_index", [0, 1, 4])
def test_auto_failure_recovers_every_unyielded_time_once(fake_renderer, stage, failure_index):
    renderer = fake_renderer()
    gpu = renderer._gpu
    gpu.failure_stage, gpu.fail_at = stage, failure_index
    times = [1.25, 0, 8.5, .125, .5, 2, 9]
    stream = renderer.export_frames(times, lambda: False)
    frames = list(stream)
    stream.close()
    reference = fake_renderer(backend="cpu")
    expected = [rgba_pixels(reference.render_frame(seconds)) for seconds in times]
    assert [index for index, _ in frames] == list(range(len(times)))
    assert [rgba_pixels(image) for _, image in frames] == expected
    assert renderer.frame_count == len(times)
    assert renderer.backend == "cpu" and gpu.closed
    assert len(renderer.fallback_reasons) == 1
    assert stream.pixel_format == "bgra"
    assert all(image.format() == QImage.Format.Format_ARGB32 for _, image in frames)
    assert stream.report()["readback_mode_history"] == (
        ["rhi-sync", "cpu"] if failure_index else ["cpu"])
    conversion_bytes = (len(times) - failure_index) * 640 * 360 * 4
    stream_report = stream.report()
    final_report = renderer.backend_report()
    assert stream_report["cpu_format_conversion_bytes"] == conversion_bytes
    assert stream_report["memory_copy_bytes"] == conversion_bytes
    assert final_report["memory_copy_bytes"] == conversion_bytes
    assert final_report["memory_copy_seconds"] == stream_report["cpu_format_conversion_seconds"]


def test_explicit_gpu_failure_is_fatal_and_closes_renderer(fake_renderer):
    renderer = fake_renderer(backend="gpu")
    gpu = renderer._gpu
    gpu.fail_at = 1
    stream = renderer.export_frames([0, .125, .5, 1], lambda: False)
    assert next(stream)[0] == 0
    with pytest.raises(GpuBackendError, match="fake Vulkan"):
        next(stream)
    stream.close()
    assert renderer.frame_count == 1 and gpu.closed
    assert not renderer.fallback_reasons
    with pytest.raises(RuntimeError, match="已经关闭"):
        renderer.render_frame(.5)


def test_drawing_error_is_not_hidden_as_device_fallback(fake_renderer):
    renderer = fake_renderer()

    def broken_paint(painter, seconds):
        raise ValueError("application paint bug")

    renderer.paint_frame = broken_paint
    stream = renderer.export_frames([0, .5, 1], lambda: False)
    with pytest.raises(ValueError, match="application paint bug"):
        next(stream)
    stream.close()
    assert not renderer.fallback_reasons and renderer.frame_count == 0
    assert renderer._export_stream is None


def test_device_error_keeps_original_reason_when_native_close_also_fails(fake_renderer, monkeypatch):
    renderer = fake_renderer(backend="gpu")
    gpu = renderer._gpu
    gpu.fail_at = 0

    def failed_cleanup():
        gpu.closed = True
        raise GpuBackendError("native cleanup failed")

    monkeypatch.setattr(gpu, "close", failed_cleanup)
    stream = renderer.export_frames([0, .5], lambda: False)
    with pytest.raises(GpuBackendError, match="fake Vulkan readback") as failure:
        next(stream)
    assert any("native cleanup failed" in note for note in failure.value.__notes__)
    assert gpu.closed and renderer._export_stream is None
    stream.close()


def test_cancelled_before_first_frame_does_not_submit_or_yield(fake_renderer):
    renderer = fake_renderer()
    stream = renderer.export_frames([0, .125, .5], lambda: True)
    with pytest.raises(InterruptedError):
        next(stream)
    stream.close()
    assert renderer.frame_count == 0 and not renderer._gpu.submitted


def test_cancelled_stream_closes_idempotently_and_keeps_returned_image(fake_renderer):
    renderer = fake_renderer()
    state = {"cancelled": False}
    stream = renderer.export_frames([0, .125, .5, 1], lambda: state["cancelled"])
    first = next(stream)
    before = rgba_pixels(first[1])
    state["cancelled"] = True
    with pytest.raises(InterruptedError):
        next(stream)
    stream.close()
    stream.close()
    assert renderer.frame_count == 1 and renderer._gpu.submitted == [0]
    assert renderer._export_stream is None
    assert rgba_pixels(first[1]) == before


def test_images_survive_stream_and_renderer_close_without_aliasing_next_frame(fake_renderer):
    renderer = fake_renderer()
    stream = renderer.export_frames([0, .125, .5, 1, 2, 3, 4], lambda: False)
    first_index, retained = next(stream)
    retained_pixels = rgba_pixels(retained)
    remainder = list(stream)
    stream.close()
    renderer.close()
    assert first_index == 0 and len(remainder) == 6
    assert rgba_pixels(retained) == retained_pixels
    retained.fill(QColor("yellow"))
    assert all(image.pixelColor(20, 20) == QColor("red") for _, image in remainder)


def test_new_stream_restarts_at_zero_after_early_close(fake_renderer):
    renderer = fake_renderer()
    first = renderer.export_frames([0, .5, 1, 2], lambda: False)
    assert next(first)[0] == 0
    with pytest.raises(RuntimeError):
        renderer.export_frames([0], lambda: False)
    first.close()
    second = renderer.export_frames([2, .5, 0], lambda: False)
    assert [index for index, _ in second] == [0, 1, 2]
    second.close()


def test_export_times_are_consumed_one_at_a_time(fake_renderer):
    renderer = fake_renderer()
    consumed = []

    def times():
        for index in range(100):
            consumed.append(index)
            yield index / 30

    stream = renderer.export_frames(times(), lambda: False)
    assert not consumed
    assert next(stream)[0] == 0 and consumed == [0]
    assert next(stream)[0] == 1 and consumed == [0, 1]
    stream.close()
    assert consumed == [0, 1]
