"""Run the existing decoded audio/indicator sync contract on actual GPU + NVENC."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from stavellum.rendering._rhi import library_path

ACCELERATED_SCRIPT = r'''
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "tests"))
from stavellum.exporting.encoding import probe_nvenc
from stavellum.rendering.gpu import GpuBackendError
from stavellum.domain.models import RenderSettings
from stavellum.graphics.qt import prepare_render_app
from stavellum.rendering.render import FrameRenderer
from stavellum.presentation.scene import compile_scene
import test_export
from test_render import rendered_document
import shutil

backend = sys.argv[2]
batch_size = int(sys.argv[3])
if batch_size == 1:
    os.environ.pop("STAVELLUM_RHI_BATCH_SIZE", None)
else:
    os.environ["STAVELLUM_RHI_BATCH_SIZE"] = str(batch_size)
settings = RenderSettings(width=640, height=360, fps=60, render_backend=backend)
prepare_render_app(settings)
try:
    document = rendered_document()
    document.settings.render_backend = backend
    with FrameRenderer(compile_scene(document)) as renderer:
        assert renderer.backend == "gpu", renderer.backend_report()
        assert renderer.backend_report()["graphics_api"] == "vulkan"
        assert not renderer.render_frame(.125).isNull()
except GpuBackendError as error:
    print(json.dumps({"skip": str(error)}))
    raise SystemExit(0)
available, reason = probe_nvenc(shutil.which("ffmpeg"), settings, lambda: False)
if not available:
    print(json.dumps({"skip": reason}))
    raise SystemExit(0)

original = test_export.click_document
def accelerated_click(*args, **kwargs):
    document = original(*args, **kwargs)
    document.settings.render_backend = backend
    document.settings.video_encoder = "h264_nvenc"
    return document
test_export.click_document = accelerated_click
directory = Path(sys.argv[1])
directory.mkdir(parents=True, exist_ok=True)
test_export.test_intro_delay_preserves_channel_sync_score_offset_and_audio_tail(
    directory, offset=.5, channels=2)
report = json.loads((directory / "intro.render.json").read_text(encoding="utf-8"))
assert report["render_backend"] == "gpu"
assert report["graphics_api"] == "vulkan"
assert report["requested_render_backend"] == backend
assert report["video_encoder"] == "h264_nvenc"
assert not report["encoder_fallback_reasons"] and not report["render_fallback_reasons"]
assert report["readback_seconds"] > 0
assert report["export_pixel_format"] == "bgra"
assert report["readback_mode"] in ("rhi-sync", "rhi-batch-sync")
if batch_size == 1:
    assert report["readback_mode_history"] == ["rhi-sync"]
    assert report["gpu_submission_count"] == report["frames"]
    assert report["gpu_batch_peak_size"] == 1
else:
    assert "rhi-batch-sync" in report["readback_mode_history"]
    assert report["gpu_submission_count"] < report["frames"]
assert sum(int(size) * count for size, count in report["gpu_batch_size_histogram"].items()) == report["frames"]
assert report["frame_queue_peak"] <= 2
assert report["total_frame_buffer_peak_bytes"] <= (2 * report["gpu_batch_peak_size"] + 4) * 640 * 360 * 4
assert report["memory_copy_seconds"] == 0
print(json.dumps({"frames": report["frames"], "gpu": report["gpu_info"]["renderer"],
                  "sync_tolerance_seconds": 1/60, "audio_channels": 2,
                  "intro_delay_seconds": .217, "audio_offset_seconds": .5,
                  "readback_mode": report["readback_mode"], "graphics_api": "vulkan",
                  "requested_render_backend": backend,
                  "tail_burst_preserved": True, "render_backend": "gpu",
                  "video_encoder": "h264_nvenc"}))
'''


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["auto", "gpu"])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.skipif(sys.platform != "win32" or not shutil.which("ffmpeg")
                    or not shutil.which("ffprobe"), reason="Windows and FFmpeg required")
def test_actual_gpu_nvenc_preserves_decoded_sync_stereo_intro_and_audio_tail(tmp_path,
                                                                          backend,
                                                                          batch_size):
    root = Path(__file__).resolve().parents[1]
    dll = library_path()
    if not dll.exists():
        pytest.skip("Native Vulkan renderer DLL has not been built")
    result = subprocess.run([sys.executable, "-c", ACCELERATED_SCRIPT, str(tmp_path), backend,
                             str(batch_size)],
                            cwd=root,
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=90, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    if "skip" in report:
        pytest.skip(report["skip"])
    assert report["frames"] == 284 and report["tail_burst_preserved"]
    assert report["graphics_api"] == "vulkan" and report["requested_render_backend"] == backend
