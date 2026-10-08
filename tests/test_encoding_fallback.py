"""Hardware retry restarts the frame stream and preserves outputs on other failures."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import wave
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage

from stavellum.domain.models import (
    NoteEvent,
    PartMapping,
    ProjectDocument,
    ProjectIR,
    RenderSettings,
    TrackInfo,
)
from stavellum.exporting import encoding, export
from stavellum.graphics import qt


@pytest.mark.parametrize(("details", "expected"), [
    ("[h264_nvenc @ abc] Cannot load nvcuda.dll", True),
    ("OpenEncodeSessionEx failed: device lost", True),
    ("Unknown encoder 'h264_nvenc'", True),
    ("Unknown encoder 'h264_nvenc'\nError opening output files: Encoder not found", True),
    ("Error while opening encoder for output stream #0:1", False),
    ("[aac @ abc] Unsupported channel layout; Error while opening encoder", False),
    ("h264_nvenc: Error opening output: Permission denied", False),
    ("h264_nvenc: Invalid data found when processing input", False),
    ("h264_nvenc: No space left on device", False),
])
def test_only_hardware_failures_qualify_for_reencoding(details, expected):
    assert encoding.hardware_failure(details) is expected


def test_cpu_encoder_skips_probe_and_hardware_quality_remains_independent(monkeypatch):
    settings = RenderSettings(video_encoder="libx264", crf=23, preset="fast",
                              nvenc_cq=17, nvenc_preset="p7")
    monkeypatch.setattr(encoding, "probe_nvenc", lambda *args: pytest.fail("CPU must skip NVENC"))
    assert encoding.select_encoder("ffmpeg", settings, lambda: False) == ("libx264", [])
    cpu = encoding.video_arguments(settings, "libx264")
    gpu = encoding.video_arguments(settings, "h264_nvenc")
    assert cpu[cpu.index("-crf") + 1] == "23" and "-cq" not in cpu
    assert gpu[gpu.index("-cq") + 1] == "17" and "-crf" not in gpu
    assert cpu[cpu.index("-preset") + 1] == "fast"
    assert gpu[gpu.index("-preset") + 1] == "p7"


def test_probe_failure_falls_back_only_when_selection_is_automatic(monkeypatch):
    monkeypatch.setattr(encoding, "probe_nvenc", lambda *args: (False, "device unavailable"))
    settings = RenderSettings(video_encoder="auto")
    assert encoding.select_encoder("ffmpeg", settings, lambda: False) == (
        "libx264", ["device unavailable"])
    settings.video_encoder = "h264_nvenc"
    with pytest.raises(RuntimeError, match="device unavailable"):
        encoding.select_encoder("ffmpeg", settings, lambda: False)


def test_cancelling_real_probe_process_closes_its_pipe_and_terminates_child(monkeypatch):
    original_popen = subprocess.Popen
    children = []
    started = []

    def sleeping_child(arguments, **kwargs):
        child = original_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        children.append(child)
        started.append(time.perf_counter())
        return child

    monkeypatch.setattr(encoding.subprocess, "Popen", sleeping_child)
    with pytest.raises(InterruptedError):
        encoding.probe_nvenc("ffmpeg", RenderSettings(),
                             lambda: bool(started) and time.perf_counter() - started[0] >= .15)
    assert len(children) == 1 and children[0].poll() is not None
    assert children[0].stderr.closed


@pytest.fixture
def encoding_job(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and FFprobe required")
    source = tmp_path / "audio.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(48000)
        audio.writeframes(b"\0\0" * 24000)
    project = ProjectIR("fixture.mid", "midi", "Retry", tracks=[TrackInfo("v", "Violin")],
                        notes=[NoteEvent("n", "v", 0, 480, 60)])
    settings = RenderSettings(width=320, height=240, fps=24, render_backend="cpu",
                              video_encoder="auto", preset="ultrafast")
    document = ProjectDocument(project, [PartMapping("v", "Violin", ["v"])], str(source), settings)
    state = SimpleNamespace(document=document, encoders=[], times=[], children=[],
                            failure="hardware", corrupt_audio=False, invalid_output=False, renderers=[])

    def compile_score(doc, **kwargs):
        return SimpleNamespace(settings=deepcopy(doc.settings), score_duration=.5, diagnostics=[],
                               scale=1, camera=SimpleNamespace(half_window_seconds=0,
                                                             certified_drift_source=0),
                               layout=SimpleNamespace(urgent_intervals=[], recovery_intervals=[]))

    class Renderer:
        render_backend = "cpu"
        fallback_reasons = []
        cache_peak_bytes = 0
        cache_limit = 1024

        def __init__(self, scene):
            self.scene = scene
            self.closed = 0
            state.renderers.append(self)

        def render_frame(self, seconds):
            state.times[-1].append(seconds)
            image = QImage(320, 240, QImage.Format.Format_RGBA8888)
            image.fill(Qt.GlobalColor.black)
            return image

        def backend_report(self):
            return {"render_backend": "cpu", "requested_render_backend": "cpu"}

        def close(self):
            self.closed += 1

    original_popen = subprocess.Popen

    def forced_encoder(arguments, **kwargs):
        arguments = list(arguments)
        if "-c:v" not in arguments:
            return original_popen(arguments, **kwargs)
        encoder = arguments[arguments.index("-c:v") + 1]
        state.encoders.append(encoder)
        state.times.append([])
        if state.corrupt_audio:
            source.write_bytes(b"corrupt WAV input")
        if state.invalid_output:
            arguments[-1] = str(tmp_path / "missing-directory" / "output.mp4")
        if encoder == "h264_nvenc":
            if state.failure == "hardware":
                arguments[arguments.index("-c:v") + 1] = "unavailable-nvenc-for-fixture"
            else:
                # Run a real CPU encoder while exercising the NVENC selection path.
                start, end = arguments.index("-c:v"), arguments.index("-c:a")
                arguments[start:end] = encoding.video_arguments(settings, "libx264")
        child = original_popen(arguments, **kwargs)
        state.children.append(child)
        return child

    monkeypatch.setattr(qt, "prepare_render_app", lambda settings: None)
    monkeypatch.setattr(export, "compile_scene", compile_score)
    monkeypatch.setattr(export, "FrameRenderer", Renderer)
    monkeypatch.setattr(encoding, "probe_nvenc", lambda *args: (True, ""))
    monkeypatch.setattr(export.subprocess, "Popen", forced_encoder)
    return state


@pytest.mark.integration
def test_real_broken_pipe_retries_once_from_zero_with_monotonic_progress(encoding_job, tmp_path):
    target = tmp_path / "retry.mp4"
    target.write_bytes(b"previous valid result")
    progress = []
    details = []
    export.export_video(encoding_job.document, target,
                        lambda fraction, message: progress.append(fraction),
                        progress_detail=details.append)
    assert encoding_job.encoders == ["h264_nvenc", "libx264"]
    assert encoding_job.times[0][0] == encoding_job.times[1][0] == 0
    assert encoding_job.times[1] == [index / 24 for index in range(12)]
    assert progress == sorted(progress) and progress[0] == 0 and progress[-1] == 1
    info = subprocess.run([shutil.which("ffprobe"), "-v", "error", "-show_streams", "-of", "json",
                           str(target)], capture_output=True, check=True)
    video = next(s for s in json.loads(info.stdout)["streams"] if s["codec_type"] == "video")
    assert int(video["nb_frames"]) == 12
    summary = json.loads(target.with_suffix(".render.json").read_text(encoding="utf-8"))
    assert summary["video_encoder"] == "libx264"
    assert len(summary["encoding_attempts"]) == 2
    assert not summary["encoding_attempts"][0]["succeeded"]
    assert summary["encoding_attempts"][1]["succeeded"]
    assert "Unknown encoder" in summary["encoder_fallback_reasons"][0]
    assert not list(tmp_path.glob("*.partial.mp4"))
    assert encoding_job.renderers[0].closed == 1
    second_attempt = [item for item in details if item.phase == "frames" and item.attempt == 2]
    assert second_attempt[0].completed == 0 and second_attempt[-1].completed == 12
    assert all(item.video_encoder == "libx264" and item.fallback_reason
               for item in second_attempt)
    assert details[-1].phase == "done" and details[-1].attempt == 2
    assert all(child.poll() is not None and child.stdin.closed and child.stderr.closed
               for child in encoding_job.children)


@pytest.mark.integration
def test_renderer_cleanup_failure_does_not_report_completion(encoding_job, tmp_path, monkeypatch):
    encoding_job.document.settings.video_encoder = "libx264"
    details, progress = [], []

    def failed_close(renderer):
        renderer.closed += 1
        raise OSError("renderer cleanup failed")

    monkeypatch.setattr(export.FrameRenderer, "close", failed_close)
    with pytest.raises(OSError, match="renderer cleanup failed"):
        export.export_video(encoding_job.document, tmp_path / "cleanup.mp4",
                            lambda fraction, message: progress.append(fraction),
                            progress_detail=details.append)
    assert not any(item.phase == "done" for item in details)
    assert all(fraction < 1 for fraction in progress)
    assert encoding_job.renderers[0].closed == 1


@pytest.mark.integration
def test_explicit_nvenc_failure_preserves_previous_result_without_retry(encoding_job, tmp_path):
    encoding_job.document.settings.video_encoder = "h264_nvenc"
    target = tmp_path / "explicit.mp4"
    target.write_bytes(b"previous valid result")
    with pytest.raises(RuntimeError, match="Unknown encoder"):
        export.export_video(encoding_job.document, target)
    assert encoding_job.encoders == ["h264_nvenc"]
    assert target.read_bytes() == b"previous valid result"
    assert not list(tmp_path.glob("*.partial.mp4"))
    assert encoding_job.renderers[0].closed == 1
    assert all(child.poll() is not None for child in encoding_job.children)


@pytest.mark.integration
def test_corrupt_audio_does_not_retry_or_replace_existing_result(encoding_job, tmp_path):
    encoding_job.failure = "none"
    encoding_job.corrupt_audio = True
    target = tmp_path / "audio-error.mp4"
    target.write_bytes(b"previous valid result")
    with pytest.raises(RuntimeError, match="Invalid data"):
        export.export_video(encoding_job.document, target)
    assert encoding_job.encoders == ["h264_nvenc"]
    assert target.read_bytes() == b"previous valid result"
    assert not list(tmp_path.glob("*.partial.mp4"))
    assert encoding_job.renderers[0].closed == 1


@pytest.mark.integration
def test_invalid_output_path_does_not_retry_or_replace_existing_result(encoding_job, tmp_path):
    encoding_job.failure = "none"
    encoding_job.invalid_output = True
    target = tmp_path / "output-error.mp4"
    target.write_bytes(b"previous valid result")
    with pytest.raises(RuntimeError, match="No such file or directory"):
        export.export_video(encoding_job.document, target)
    assert encoding_job.encoders == ["h264_nvenc"]
    assert target.read_bytes() == b"previous valid result"
    assert not list(tmp_path.glob("*.partial.mp4"))
    assert encoding_job.renderers[0].closed == 1


@pytest.mark.integration
def test_statistics_write_failure_preserves_previous_video_and_cleans_partials(
        encoding_job, tmp_path, monkeypatch):
    encoding_job.failure = "none"
    target = tmp_path / "sidecar-error.mp4"
    target.write_bytes(b"previous valid result")
    original_write = Path.write_text

    def fail_summary(path, *args, **kwargs):
        if path.suffix == ".json":
            raise OSError("statistics volume unavailable")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_summary)
    with pytest.raises(OSError, match="statistics volume unavailable"):
        export.export_video(encoding_job.document, target)
    assert encoding_job.encoders == ["h264_nvenc"]
    assert target.read_bytes() == b"previous valid result"
    assert not list(tmp_path.glob("*.partial.*"))
    assert encoding_job.renderers[0].closed == 1


@pytest.mark.integration
def test_cancelled_nvenc_attempt_never_falls_back_or_replaces_existing_result(encoding_job, tmp_path):
    encoding_job.failure = "none"
    target = tmp_path / "cancelled.mp4"
    target.write_bytes(b"previous valid result")
    cancelled = False

    def progress(fraction, message):
        nonlocal cancelled
        if fraction > .05:
            cancelled = True

    with pytest.raises(InterruptedError):
        export.export_video(encoding_job.document, target, progress, lambda: cancelled)
    assert encoding_job.encoders == ["h264_nvenc"]
    assert target.read_bytes() == b"previous valid result"
    assert not list(tmp_path.glob("*.partial.mp4"))
    assert encoding_job.renderers[0].closed == 1
    assert all(child.poll() is not None and child.stdin.closed and child.stderr.closed
               for child in encoding_job.children)
