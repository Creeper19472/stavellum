"""Encoder integration: timestamps, real audio, atomic cancellation and frame counts."""

import json
import math
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest
from native_frames import frame_layout

from stavellum.domain.models import (
    Metadata,
    NoteEvent,
    PartMapping,
    ProjectDocument,
    ProjectIR,
    RenderSettings,
    TrackInfo,
)
from stavellum.exporting.export import audio_duration, export_video
from stavellum.presentation.scene import compile_scene
from stavellum.rendering.render import FrameRenderer

pytestmark = [pytest.mark.integration, pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required")]


def click_document(directory: Path, offset: float = 0, *, channels: int = 1, intro_delay: float = 0, tail_burst: bool = False) -> ProjectDocument:
    project = ProjectIR("fixture.mid", "midi", "Click fixture", ppq=480, bpm=120, tracks=[TrackInfo("v", "Violin")], notes=[NoteEvent(f"n{i}", "v", i * 960, 120, 72 + i, 127) for i in range(4)], duration_ticks=3840)
    document = ProjectDocument(project, [PartMapping("v", "Violin", ["v"], instrument="violin", clef="treble", key_signature=0)], str(directory / "click.wav"), RenderSettings(width=640, height=360, fps=60, preset="ultrafast", score_start_in_audio_sec=offset, intro_delay_seconds=intro_delay), Metadata("Clicks"))
    document.settings.render_backend = "cpu"
    document.settings.video_encoder = "libx264"
    rate = 48000
    audio = np.zeros((round((4 + max(0, offset)) * rate), channels), dtype=np.float32)
    gains = np.linspace(1.0, .65, channels)
    times = np.arange(1200) / rate
    burst = np.sin(2 * np.pi * 2000 * times) * np.exp(-times * 150) * .7
    for i in range(4):
        index = round((i + offset) * rate)
        if index >= 0:
            count = max(0, min(len(burst), len(audio) - index))
            audio[index:index + count] += burst[:count, None] * gains
    if tail_burst:
        index = len(audio) - round(.15 * rate)
        audio[index:index + len(burst)] += burst[:, None] * gains
    with wave.open(document.audio_path, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes((audio * 32767).astype("<i2").tobytes())
    return document


def probe(path: Path) -> dict:
    result = subprocess.run([shutil.which("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], capture_output=True, check=True)
    return json.loads(result.stdout)


def indicator_brightness(scene, image: np.ndarray, time: float) -> float:
    """Sample complete interior pixels at the indicator's current position."""
    x, top, width, height = frame_layout(scene, time).rows[scene.parts[0].part_id].indicator_rect
    left = max(0, math.ceil(x + 1))
    right = min(image.shape[1], math.floor(x + width - 1))
    first = max(0, math.ceil(top + 2))
    last = min(image.shape[0], math.floor(top + height - 2))
    assert left < right and first < last
    return float(image[first:last, left:right].mean())


def test_encoded_grid_fixture_sync_and_frame_count(tmp_path):
    document = click_document(tmp_path)
    target = tmp_path / "sync.mp4"
    progress = []
    export_video(document, target, lambda fraction, message: progress.append(fraction))
    info = probe(target)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert video["nb_frames"] == "240"
    assert video["avg_frame_rate"] == "60/1"
    assert video["width"] == 640 and video["height"] == 360
    assert progress[0] == 0 and progress[-1] == 1
    assert progress == sorted(progress)

    # Decoder compensates AAC priming. Its actual click peaks must match the
    # visual MIDI-trigger frames within one frame at start, middle and end.
    decoded_audio = tmp_path / "decoded.f32"
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-i", str(target), "-vn", "-ac", "1", "-ar", "48000", "-f", "f32le", str(decoded_audio)], capture_output=True, check=True)
    audio = np.fromfile(decoded_audio, dtype="<f4")
    for expected in (0, 1, 2, 3):
        first = max(0, round((expected - .05) * 48000))
        last = min(len(audio), round((expected + .05) * 48000))
        peak_time = (first + np.argmax(np.abs(audio[first:last]))) / 48000
        assert abs(peak_time - expected) <= 1 / 60

    indices = [0, 59, 60, 119, 120, 179, 180]
    selected = "+".join(f"eq(n\\,{n})" for n in indices)
    decoded_frames = tmp_path / "decoded.rgb"
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-i", str(target), "-vf", f"select={selected}", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", str(decoded_frames)], capture_output=True, check=True)
    images = np.fromfile(decoded_frames, dtype=np.uint8).reshape(len(indices), 360, 640, 3)
    scene = compile_scene(document)
    interior = np.array([indicator_brightness(scene, image, index / 60)
                         for index, image in zip(indices, images, strict=True)])
    assert min(interior[[0, 2, 4, 6]]) > 170
    assert max(interior[[1, 3, 5]]) < 80

    summary = json.loads(target.with_suffix(".render.json").read_text(encoding="utf-8"))
    assert summary["frames"] == 240
    assert summary["cache_peak_bytes"] <= summary["cache_limit_bytes"]


def test_cancel_keeps_previous_output_and_removes_partial(tmp_path):
    document = click_document(tmp_path, intro_delay=.217)
    target = tmp_path / "existing.mp4"
    target.write_bytes(b"previous valid user result")
    should_cancel = False

    def progress(fraction, message):
        nonlocal should_cancel
        if fraction > .1:
            should_cancel = True

    with pytest.raises(InterruptedError):
        export_video(document, target, progress, lambda: should_cancel)
    assert target.read_bytes() == b"previous valid user result"
    assert not list(tmp_path.glob("*.partial.mp4"))
    assert target.with_suffix(".render.log").exists()


def test_positive_audio_offset_preserves_end_and_tail(tmp_path):
    document = click_document(tmp_path, offset=.5)
    target = tmp_path / "offset.mp4"
    export_video(document, target)
    summary = json.loads(target.with_suffix(".render.json").read_text(encoding="utf-8"))
    assert summary["frames"] == 270
    assert summary["duration_seconds"] == 4.5
    assert audio_duration(document.audio_path) == 4.5


@pytest.mark.parametrize(("offset", "channels"), [(0, 1), (.5, 2), (-.35, 2)])
def test_intro_delay_preserves_channel_sync_score_offset_and_audio_tail(tmp_path, offset, channels):
    delay = .217  # Deliberately falls between video frames at 60 fps.
    document = click_document(tmp_path, offset, channels=channels, intro_delay=delay, tail_burst=True)
    target = tmp_path / "intro.mp4"
    export_video(document, target)
    info = probe(target)
    video = next(stream for stream in info["streams"] if stream["codec_type"] == "video")
    encoded_audio = next(stream for stream in info["streams"] if stream["codec_type"] == "audio")
    source_duration = 4 + max(0, offset)
    expected_frames = math.ceil((source_duration + delay) * 60)
    assert int(video["nb_frames"]) == expected_frames
    assert encoded_audio["channels"] == channels
    summary = json.loads(target.with_suffix(".render.json").read_text(encoding="utf-8"))
    assert summary["frames"] == expected_frames
    assert summary["duration_seconds"] == expected_frames / 60
    assert summary["audio_duration_seconds"] == source_duration
    assert summary["intro_delay_seconds"] == delay
    assert summary["score_start_in_audio_sec"] == offset
    assert source_duration + delay <= summary["duration_seconds"] < source_duration + delay + 1 / 60

    decoded_audio = tmp_path / "intro.f32"
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-i", str(target), "-vn", "-ar", "48000", "-f", "f32le", str(decoded_audio)], capture_output=True, check=True)
    audio = np.fromfile(decoded_audio, dtype="<f4").reshape(-1, channels)
    # Leave 20 ms before the first possible click for AAC pre-echo.
    assert np.max(np.abs(audio[:round((delay - .02) * 48000)])) < .002
    click_times = [delay + i + offset for i in range(4) if i + offset >= 0]
    for expected in [*click_times, delay + source_duration - .15]:
        first = max(0, round((expected - .04) * 48000))
        last = min(len(audio), round((expected + .04) * 48000))
        for channel in range(channels):
            window = np.abs(audio[first:last, channel])
            assert window.max() > .2
            peak_time = (first + np.argmax(window)) / 48000
            assert abs(peak_time - expected) <= 1 / 60

    active_indices = [math.ceil(expected * 60) for expected in click_times]
    quiet_indices = [index - 1 for index in active_indices]
    intro_index = math.floor(delay * 60 / 2)
    indices = sorted({intro_index, *active_indices, *quiet_indices})
    selected = "+".join(f"eq(n\\,{index})" for index in indices)
    decoded_frames = tmp_path / "intro.rgb"
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-i", str(target), "-vf", f"select={selected}", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", str(decoded_frames)], capture_output=True, check=True)
    images = np.fromfile(decoded_frames, dtype=np.uint8).reshape(len(indices), 360, 640, 3)
    scene = compile_scene(document)
    brightness = {}
    for index, image in zip(indices, images, strict=True):
        brightness[index] = indicator_brightness(scene, image, index / 60)
    assert min(brightness[index] for index in active_indices) > 170
    assert max(brightness[index] for index in [intro_index, *quiet_indices]) < 80


def test_missing_audio_and_encoder_failure_are_reviewable(tmp_path, monkeypatch):
    document = click_document(tmp_path)
    document.audio_path = str(tmp_path / "missing.wav")
    with pytest.raises(ValueError, match="原曲音频"):
        export_video(document, tmp_path / "missing.mp4")
    document.audio_path = str(tmp_path / "click.wav")
    document.settings.preset = "invalid-preset-for-fixture"
    with pytest.raises(ValueError, match="编码预设"):
        export_video(document, tmp_path / "invalid.mp4")
    document.settings.preset = "ultrafast"
    # Exercise an actual failed encoder after validation, rather than rely on an
    # invalid user setting being passed through to FFmpeg.
    original_popen = subprocess.Popen

    def failed_encoder(arguments, **kwargs):
        arguments = list(arguments)
        arguments[arguments.index("-c:v") + 1] = "unavailable-codec-for-fixture"
        return original_popen(arguments, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", failed_encoder)
    with pytest.raises(RuntimeError, match="FFmpeg"):
        export_video(document, tmp_path / "failed.mp4")
    assert not (tmp_path / "failed.mp4").exists()
    assert not list(tmp_path.glob("*.partial.mp4"))
    assert (tmp_path / "failed.render.log").read_text(encoding="utf-8")


def test_float_wav_duration_and_empty_audio_rejection(tmp_path):
    document = click_document(tmp_path)
    float_wav = tmp_path / "floating-point.wav"
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-i", document.audio_path,
                    "-c:a", "pcm_f32le", str(float_wav)], capture_output=True, check=True)
    assert audio_duration(float_wav) == pytest.approx(4.0)
    empty_wav = tmp_path / "empty.wav"
    with wave.open(str(empty_wav), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(48000)
    with pytest.raises(ValueError, match="时长"):
        audio_duration(empty_wav)


@pytest.mark.parametrize("delay", [0, .217])
def test_preview_and_export_frame_share_the_clock(tmp_path, delay):
    document = click_document(tmp_path, offset=.5, intro_delay=delay)
    scene = compile_scene(document)
    assert scene.beat_at_time(.5) == 0
    assert scene.beat_at_time(1.5) == 2
    renderer = FrameRenderer(scene)
    presentation_time = document.settings.presentation_time(1.5)
    frame = renderer.render_frame(presentation_time)
    first = bytes(frame.constBits())
    renderer.render_frame(.1)
    frame = renderer.render_frame(presentation_time)
    assert bytes(frame.constBits()) == first
