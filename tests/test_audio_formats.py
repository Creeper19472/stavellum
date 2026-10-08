"""Common original-audio formats load in Qt and survive real MP4 export."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import wave

import numpy as np
import pytest
from PySide6.QtCore import QEventLoop, QTimer, QUrl
from PySide6.QtMultimedia import QMediaPlayer

from stavellum.domain.models import (
    Metadata,
    NoteEvent,
    PartMapping,
    ProjectDocument,
    ProjectIR,
    RenderSettings,
    TrackInfo,
)
from stavellum.exporting.audio import audio_duration
from stavellum.exporting.export import export_video
from stavellum.graphics.qt import ensure_app

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                       reason="FFmpeg and ffprobe required"),
]

FORMATS = [
    ("wav", "pcm_s16le"), ("flac", "flac"), ("mp3", "libmp3lame"),
    ("m4a", "aac"), ("aac", "aac"), ("ogg", "libvorbis"),
    ("opus", "libopus"), ("aif", "pcm_s16be"), ("aiff", "pcm_s16be"),
    ("wma", "wmav2"),
]


def run_ffmpeg(arguments):
    return subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", *arguments], capture_output=True, check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


@pytest.fixture(scope="module")
def encoders():
    result = run_ffmpeg(["-encoders"])
    return {fields[1] for line in result.stdout.decode("utf-8", "replace").splitlines()
            if len(fields := line.split()) > 1 and fields[0].startswith("A")}


@pytest.fixture(params=FORMATS, ids=[extension for extension, _ in FORMATS])
def original_audio(request, tmp_path, encoders):
    extension, codec = request.param
    if codec not in encoders:
        pytest.skip(f"Fixture encoder {codec} unavailable")
    rate = 48000
    values = np.zeros((rate, 2), dtype=np.float32)
    times = np.arange(1200) / rate
    burst = np.sin(2 * np.pi * 2000 * times) * np.exp(-times * 150) * .7
    for seconds in (.10, .50, .88):
        index = round(seconds * rate)
        values[index:index + len(burst)] = burst[:, None] * np.array([1.0, .65])
    pcm = tmp_path / "source.wav"
    with wave.open(str(pcm), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes((values * 32767).astype("<i2").tobytes())
    path = tmp_path / f"原曲 含尾音.{extension}"
    run_ffmpeg(["-i", str(pcm), "-c:a", codec, str(path)])
    return path


def test_common_formats_have_usable_duration_and_load_in_qt(original_audio):
    assert audio_duration(original_audio) == pytest.approx(1.0, abs=.1)
    app = ensure_app()
    player = QMediaPlayer()
    loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)

    def loaded(status):
        if status in (QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.InvalidMedia):
            loop.quit()

    player.mediaStatusChanged.connect(loaded)
    player.errorOccurred.connect(lambda *args: loop.quit())
    try:
        player.setSource(QUrl.fromLocalFile(str(original_audio)))
        timer.start(5000)
        loop.exec()
        assert player.error() == QMediaPlayer.Error.NoError, player.errorString()
        assert player.mediaStatus() == QMediaPlayer.MediaStatus.LoadedMedia
        assert player.hasAudio()
        assert player.duration() == pytest.approx(1000, abs=100)
        assert player.isSeekable()
    finally:
        timer.stop()
        player.setSource(QUrl())
        player.deleteLater()
        app.processEvents()


def test_common_formats_export_aac_audio_with_intro_sync_and_tail(original_audio, tmp_path):
    project = ProjectIR(
        "source.mid", "midi", "Audio formats", ppq=480, bpm=120,
        tracks=[TrackInfo("v", "Violin")],
        notes=[NoteEvent("first", "v", 96, 24, 72), NoteEvent("middle", "v", 480, 24, 74),
               NoteEvent("tail", "v", 840, 24, 76)],
        duration_ticks=960,
    )
    settings = RenderSettings(
        width=320, height=240, fps=24, preset="ultrafast", render_backend="cpu",
        video_encoder="libx264", intro_delay_seconds=.217,
    )
    document = ProjectDocument(project, [PartMapping("v", "Violin", ["v"], instrument="violin")],
                               str(original_audio), settings, Metadata("Audio formats"))
    target = tmp_path / "result.mp4"
    export_video(document, target)
    information = json.loads(subprocess.run(
        [shutil.which("ffprobe"), "-v", "error", "-show_streams", "-of", "json", str(target)],
        capture_output=True, check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ).stdout)
    sound = next(stream for stream in information["streams"] if stream["codec_type"] == "audio")
    video = next(stream for stream in information["streams"] if stream["codec_type"] == "video")
    assert sound["codec_name"] == "aac"
    assert sound["channels"] == 2
    summary = json.loads(target.with_suffix(".render.json").read_text(encoding="utf-8"))
    expected_seconds = settings.presentation_duration(audio_duration(original_audio),
                                                     summary["score_duration_seconds"])
    expected_frames = math.ceil(expected_seconds * settings.fps)
    assert int(video["nb_frames"]) == expected_frames
    assert summary["duration_seconds"] == expected_frames / settings.fps
    assert summary["audio_duration_seconds"] == pytest.approx(1.0, abs=.1)
    decoded = tmp_path / "decoded.f32"
    run_ffmpeg(["-i", str(target), "-vn", "-ar", "48000", "-f", "f32le", str(decoded)])
    samples = np.fromfile(decoded, dtype="<f4").reshape(-1, 2)
    assert np.max(np.abs(samples[:round(.19 * 48000)])) < .002
    tolerance = 1 / settings.fps if original_audio.suffix == ".flac" else .08
    for expected in (.217 + .10, .217 + .50, .217 + .88):
        first, last = round((expected - .09) * 48000), round((expected + .09) * 48000)
        for channel in range(2):
            window = np.abs(samples[first:last, channel])
            assert np.max(window) > .05, "input clicks and tail must remain audible"
            peak_time = (first + int(np.argmax(window))) / 48000
            assert abs(peak_time - expected) <= tolerance
    assert not list(tmp_path.glob("*.partial.mp4"))
