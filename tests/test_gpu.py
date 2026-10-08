"""Production Vulkan selection and recovery, without requiring a driver."""

from __future__ import annotations

from dataclasses import replace

import pytest
from PySide6.QtGui import QImage
from test_render import pixels, rendered_document

from stavellum.presentation.scene import compile_scene
from stavellum.rendering.gpu import GpuBackendError
from stavellum.rendering.render import FrameRenderer


@pytest.fixture(scope="module")
def cpu_scene():
    return compile_scene(rendered_document())


class FakeVulkanRenderer:
    def __init__(self, scene, *, assets):
        self.scene, self.assets = scene, assets
        self.readback_seconds = 0.0
        self.closed = False

    def _render(self, seconds):
        image = self.assets._render_cpu(seconds)
        self.readback_seconds += .01
        return image.convertToFormat(QImage.Format.Format_ARGB32)

    def render_frame(self, seconds):
        return self._render(seconds).convertToFormat(QImage.Format.Format_RGBA8888)

    def backend_report(self):
        return {"graphics_api": "vulkan", "gpu_info": {"renderer": "test Vulkan", "msaa_samples": 4},
                "readback_seconds": self.readback_seconds,
                "gpu_texture_cache_budget_bytes": self.scene.settings.cache_megabytes * 1024 * 1024}

    def close(self):
        self.closed = True


def test_unavailable_vulkan_auto_falls_back_and_explicit_gpu_fails(cpu_scene, monkeypatch):
    def unavailable(*args, **kwargs):
        raise GpuBackendError("test Vulkan device unavailable")

    monkeypatch.setattr("stavellum.rendering.rhi.RhiFrameRenderer", unavailable)
    cpu_scene.settings.render_backend = "auto"
    with FrameRenderer(cpu_scene) as renderer:
        assert not renderer.render_frame(.5).isNull()
        assert renderer.backend == "cpu"
        assert renderer.fallback_reasons == ["test Vulkan device unavailable"]
        assert renderer.backend_report()["gpu_texture_cache_budget_bytes"] == 0
    cpu_scene.settings.render_backend = "gpu"
    with pytest.raises(GpuBackendError, match="已选择 Vulkan GPU"):
        FrameRenderer(cpu_scene)
    cpu_scene.settings.render_backend = "cpu"


def test_mid_frame_vulkan_failure_retries_same_frame_once_and_retains_tiles(cpu_scene, monkeypatch):
    cpu_scene.settings.render_backend = "cpu"
    with FrameRenderer(cpu_scene) as reference:
        expected = pixels(reference.render_frame(1.5))

    class LostRenderer(FakeVulkanRenderer):
        def _render(self, seconds):
            super()._render(seconds)
            raise GpuBackendError("test Vulkan device lost after drawing")

    monkeypatch.setattr("stavellum.rendering.rhi.RhiFrameRenderer", LostRenderer)
    cpu_scene.settings.render_backend = "auto"
    with FrameRenderer(cpu_scene) as renderer:
        gpu = renderer._gpu
        calls = []
        original = renderer._crop

        def count_crop(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        monkeypatch.setattr(renderer, "_crop", count_crop)
        assert pixels(renderer.render_frame(1.5)) == expected
        initial_calls = len(calls)
        assert initial_calls > 0
        assert renderer.frame_count == 1 and renderer.backend == "cpu" and gpu.closed
        assert renderer.backend_report()["readback_seconds"] == .01
        assert renderer.fallback_reasons == ["test Vulkan device lost after drawing"]
        assert pixels(renderer.render_frame(1.5)) == expected
        assert len(calls) == initial_calls
    renderer.close()
    with pytest.raises(RuntimeError, match="已经关闭"):
        renderer.render_frame(1.5)
    cpu_scene.settings.render_backend = "gpu"
    renderer = FrameRenderer(cpu_scene)
    with pytest.raises(GpuBackendError, match="device lost"):
        renderer.render_frame(1.5)
    assert renderer.frame_count == 0
    with pytest.raises(RuntimeError, match="已经关闭"):
        renderer.render_frame(1.5)
    renderer.close()
    cpu_scene.settings.render_backend = "cpu"


def test_drawing_errors_are_not_hidden_by_auto_fallback(cpu_scene, monkeypatch):
    class BugRenderer(FakeVulkanRenderer):
        def _render(self, seconds):
            raise ValueError("application drawing error")

    monkeypatch.setattr("stavellum.rendering.rhi.RhiFrameRenderer", BugRenderer)
    cpu_scene.settings.render_backend = "auto"
    with FrameRenderer(cpu_scene) as renderer:
        with pytest.raises(ValueError, match="application drawing error"):
            renderer.render_frame(0)
        assert renderer.fallback_reasons == [] and renderer.frame_count == 0
    cpu_scene.settings.render_backend = "cpu"


def test_logo_survives_vulkan_failure_with_cached_image_and_same_absolute_time(cpu_scene, monkeypatch):
    settings = replace(cpu_scene.settings, render_backend="cpu", logo_enabled=True,
                       logo_display_mode="intro", logo_size_ratio=0.2)
    scene = replace(cpu_scene, settings=settings)
    with FrameRenderer(scene) as reference:
        expected = pixels(reference.render_frame(6.4))

    class LostRenderer(FakeVulkanRenderer):
        def _render(self, seconds):
            super()._render(seconds)
            self.logo = self.assets._logo[0]
            raise GpuBackendError("test Vulkan device lost with logo")

    monkeypatch.setattr("stavellum.rendering.rhi.RhiFrameRenderer", LostRenderer)
    with FrameRenderer(replace(scene, settings=replace(settings, render_backend="auto"))) as renderer:
        gpu = renderer._gpu
        assert pixels(renderer.render_frame(6.4)) == expected
        assert renderer.backend == "cpu" and gpu.closed
        assert renderer._logo[0] is gpu.logo
        assert pixels(renderer.render_frame(6.4)) == expected
