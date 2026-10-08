"""Real production Vulkan preview/export frames and readback ownership."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from native_support import require_vulkan_device

from stavellum.rendering._rhi import library_path

GPU_READBACK_SCRIPT = r'''
import json, os, sys
from dataclasses import replace
from PySide6.QtGui import QColor, QImage
from stavellum.rendering.gpu import GpuBackendError
from stavellum.domain.models import RenderSettings
from stavellum.graphics.qt import prepare_render_app
from stavellum.rendering.render import FrameRenderer
from stavellum.presentation.scene import compile_scene
sys.path.insert(0, "tests")
from test_render import rendered_document

os.environ.pop("STAVELLUM_RHI_COPY_READBACK", None)
os.environ.pop("STAVELLUM_RHI_RGBA_READBACK", None)
app = prepare_render_app(RenderSettings(render_backend="gpu"))
document = rendered_document()
scene = compile_scene(document)
times = [4, .5, 0, .125, 1, .5, 9.5]
def rgba(image):
    normalized = image.convertToFormat(QImage.Format.Format_RGBA8888)
    return bytes(normalized.constBits())

identities = []
for backend in ("auto", "gpu"):
    scene.settings.render_backend = backend
    with FrameRenderer(scene) as renderer:
        assert renderer.backend == "gpu", renderer.backend_report()
        assert renderer.backend_report()["graphics_api"] == "vulkan"
        identities.append(renderer.backend_report()["gpu_info"])
        expected = [rgba(renderer.render_frame(seconds)) for seconds in times]
        assert expected[1] == expected[5] and expected[0] != expected[2]
        for seconds, wanted in reversed(list(zip(times, expected))):
            assert rgba(renderer.render_frame(seconds)) == wanted
        renderer.cache.clear()
        renderer.cache_bytes = 0
        renderer.cache_limit = 0
        expected = [rgba(renderer.render_frame(seconds)) for seconds in times]
        stream = renderer.export_frames(times, lambda: False)
        assert stream.pixel_format == "bgra"
        frames = list(stream)
        stream.close()
        assert [index for index, image in frames] == list(range(len(times)))
        assert [rgba(image) for index, image in frames] == expected
        retained = frames[0][1]
        shallow = QImage(retained)
        report = renderer.backend_report()
        assert report["memory_copy_seconds"] == report["memory_copy_bytes"] == 0
        assert report["owned_readback_frame_count"] >= len(times)
        assert report["readback_seconds"] > 0
        assert report["gpu_info"]["msaa_samples"] == 4
        assert report["gpu_texture_cache_budget_bytes"] == 8 * 1024 * 1024
        other_scene = replace(scene, settings=replace(scene.settings, cache_megabytes=16))
        with FrameRenderer(other_scene) as second:
            assert second._gpu._target._handle != renderer._gpu._target._handle
            second.render_frame(.5)
            assert second.backend_report()["gpu_texture_cache_budget_bytes"] == 16 * 1024 * 1024
        assert renderer.backend_report()["gpu_texture_cache_budget_bytes"] == 8 * 1024 * 1024
    assert rgba(retained) == rgba(shallow) == expected[0]
    retained.fill(QColor("yellow"))
    assert rgba(shallow) == expected[0]

# Cancel before issuing another frame; all returned storage remains owned by Qt.
scene.settings.render_backend = "gpu"
with FrameRenderer(scene) as renderer:
    state = {"cancelled": False}
    stream = renderer.export_frames(times, lambda: state["cancelled"])
    _, retained = next(stream)
    saved = rgba(retained)
    submitted = renderer.backend_report()["gpu_submitted_frame_count"]
    state["cancelled"] = True
    try:
        next(stream)
    except InterruptedError:
        pass
    else:
        raise AssertionError("cancelled Vulkan stream produced a frame")
    stream.close(); stream.close()
    assert renderer.backend_report()["gpu_submitted_frame_count"] == submitted
assert rgba(retained) == saved

# Real Vulkan first frame followed by device loss must retry the same presentation
# time on CPU and preserve the already-selected FFmpeg BGRA input format.
scene.settings.render_backend = "auto"
with FrameRenderer(scene) as renderer:
    stream = renderer.export_frames(times, lambda: False)
    first = next(stream)
    # Frames already completed by the first batch retain their GPU pixels.
    prefix = [rgba(first[1]), *(rgba(image) for image in stream._ready)]
    def lost(*args, **kwargs):
        raise GpuBackendError("simulated Vulkan device loss")
    renderer._gpu._target.render = lost
    renderer._gpu._target.render_batch = lost
    recovered = [first, *list(stream)]
    stream.close()
    assert renderer.backend == "cpu" and renderer.frame_count == len(times)
    assert renderer.fallback_reasons == ["simulated Vulkan device loss"]
    assert [index for index, image in recovered] == list(range(len(times)))
    assert stream.pixel_format == "bgra"
    assert all(image.format() == QImage.Format.Format_ARGB32 for _, image in recovered)
scene.settings.render_backend = "cpu"
with FrameRenderer(scene) as renderer:
    expected = [rgba(renderer.render_frame(seconds)) for seconds in times]
assert [rgba(image) for _, image in recovered[:len(prefix)]] == prefix
assert [rgba(image) for _, image in recovered[len(prefix):]] == expected[len(prefix):]
assert not any(window.isVisible() for window in app.allWindows())
print(json.dumps({"gpu": identities, "random_access": True, "ownership": True,
                  "recovery": True, "cancellation": True}), flush=True)
'''


@pytest.mark.integration
@pytest.mark.skipif(sys.platform != "win32", reason="Windows native renderer required")
def test_real_default_vulkan_preview_stream_ownership_cancellation_and_recovery():
    root = Path(__file__).resolve().parents[1]
    dll = library_path()
    if not dll.exists():
        pytest.skip("Native Vulkan renderer DLL has not been built")
    require_vulkan_device()
    result = subprocess.run(
        [sys.executable, "-c", GPU_READBACK_SCRIPT], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=90, cwd=root,
        env={**os.environ, "PYTHONUTF8": "1"},
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert len(report["gpu"]) == 2 and all(identity["msaa_samples"] == 4 for identity in report["gpu"])
    assert report["random_access"] and report["ownership"]
    assert report["recovery"] and report["cancellation"]
