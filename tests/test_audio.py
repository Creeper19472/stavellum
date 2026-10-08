"""Audio probing rejects invalid inputs and never leaves a cancelled child running."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import time
import wave

import pytest

from stavellum.exporting import audio


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "原曲.flac"
    path.write_bytes(b"not a PCM WAV")
    return path


class ProbeProcess:
    def __init__(self, result=None, *, errors="", code=0, blocked=False, ignore_terminate=False):
        self.output = json.dumps(result) if result is not None else "invalid JSON"
        self.errors = errors
        self.code = code
        self.blocked = blocked
        self.ignore_terminate = ignore_terminate
        self.returncode = None
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.terminated = False
        self.killed = False
        self.on_communicate = None

    def communicate(self, timeout=None):
        if self.on_communicate is not None:
            self.on_communicate()
        if self.blocked and self.returncode is None:
            raise subprocess.TimeoutExpired("ffprobe", timeout)
        self.returncode = self.code if self.returncode is None else self.returncode
        return self.output, self.errors

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not self.ignore_terminate:
            self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9


def install_probe(monkeypatch, information=None, **options):
    process = ProbeProcess(information, **options)
    monkeypatch.setattr(audio.shutil, "which", lambda name: "ffprobe")
    calls = []

    def popen(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return process

    monkeypatch.setattr(audio.subprocess, "Popen", popen)
    return process, calls


def test_pcm_wav_uses_sample_count_without_ffprobe_even_when_renamed(tmp_path, monkeypatch):
    path = tmp_path / "原曲.mp3"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        stream.writeframes(b"\0\0\0\0" * 24000)
    monkeypatch.setattr(audio.shutil, "which", lambda name: pytest.fail("PCM WAV needs no probe"))
    assert audio.audio_duration(path) == .5


def test_empty_pcm_wav_is_rejected(tmp_path):
    path = tmp_path / "empty.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(48000)
    with pytest.raises(ValueError, match="时长"):
        audio.audio_duration(path)


def test_pcm_wav_with_invalid_sample_rate_is_rejected(tmp_path):
    path = tmp_path / "invalid-rate.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        stream.writeframes(b"\0\0" * 48)
    contents = bytearray(path.read_bytes())
    contents[24:28] = b"\0\0\0\0"
    path.write_bytes(contents)
    with pytest.raises(ValueError, match="时长"):
        audio.audio_duration(path)


@pytest.mark.parametrize(("stream_duration", "container_duration", "expected"), [
    ("2.25", "9.0", 2.25),
    (None, "3.5", 3.5),
    ("N/A", "3.5", 3.5),
    ("nan", "3.5", 3.5),
    ("-1", "3.5", 3.5),
])
def test_first_audio_track_duration_has_priority_with_container_fallback(
        source, monkeypatch, stream_duration, container_duration, expected):
    process, calls = install_probe(monkeypatch, {
        "streams": [{"codec_type": "audio", "duration": stream_duration}],
        "format": {"duration": container_duration},
    })
    assert audio.audio_duration(source) == expected
    arguments, options = calls[0]
    assert arguments[arguments.index("-select_streams") + 1] == "a:0"
    assert arguments[-1] == str(source.resolve())
    assert options["encoding"] == "utf-8"
    assert process.stdout.closed and process.stderr.closed


@pytest.mark.parametrize("streams", [None, [], {}, [{}], [{"codec_type": "video"}]])
def test_container_duration_does_not_make_a_silent_video_valid_audio(source, monkeypatch, streams):
    install_probe(monkeypatch, {"streams": streams, "format": {"duration": "4.0"}})
    with pytest.raises(ValueError, match="音频轨道"):
        audio.audio_duration(source)


@pytest.mark.parametrize("value", [None, "", "N/A", "NaN", "inf", "-inf", 0, -1, True])
def test_invalid_durations_are_rejected(source, monkeypatch, value):
    install_probe(monkeypatch, {
        "streams": [{"codec_type": "audio", "duration": value}],
        "format": {"duration": value},
    })
    with pytest.raises(ValueError, match="时长"):
        audio.audio_duration(source)


def test_missing_audio_and_missing_ffprobe_have_reviewable_messages(tmp_path, source, monkeypatch):
    with pytest.raises(ValueError, match="找不到原曲音频"):
        audio.audio_duration(tmp_path / "missing.flac")
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="ffprobe.*PATH"):
        audio.audio_duration(source)


def test_ffprobe_failure_preserves_the_underlying_diagnostic(source, monkeypatch):
    process, _ = install_probe(monkeypatch, code=1, errors="invalid data found in 原曲.flac\n")
    with pytest.raises(ValueError, match="无法读取原曲音频.*invalid data"):
        audio.audio_duration(source)
    assert process.stdout.closed and process.stderr.closed


@pytest.mark.parametrize("information", [None, []])
def test_malformed_probe_results_are_reviewable(source, monkeypatch, information):
    install_probe(monkeypatch, information)
    with pytest.raises(ValueError, match="无效的音频信息"):
        audio.audio_duration(source)


def test_unlaunchable_ffprobe_has_a_chinese_error(source, monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda name: "ffprobe")

    def denied(*args, **kwargs):
        raise PermissionError("access denied")

    monkeypatch.setattr(audio.subprocess, "Popen", denied)
    with pytest.raises(RuntimeError, match="无法启动 ffprobe.*access denied"):
        audio.audio_duration(source)


def test_cancellation_before_reading_never_launches_a_probe(source, monkeypatch):
    _, calls = install_probe(monkeypatch)
    with pytest.raises(InterruptedError, match="取消"):
        audio.audio_duration(source, cancel=lambda: True)
    assert calls == []


@pytest.mark.parametrize("ignore_terminate", [False, True])
def test_cancellation_stops_and_reaps_the_probe(source, monkeypatch, ignore_terminate):
    process, _ = install_probe(monkeypatch, blocked=True, ignore_terminate=ignore_terminate)
    cancelled = False

    def request_cancel():
        nonlocal cancelled
        cancelled = True

    process.on_communicate = request_cancel
    with pytest.raises(InterruptedError, match="取消"):
        audio.audio_duration(source, cancel=lambda: cancelled)
    assert process.terminated
    assert process.killed == ignore_terminate
    assert process.poll() is not None
    assert process.stdout.closed and process.stderr.closed


def test_probe_completed_during_cancellation_does_not_return_a_duration(source, monkeypatch):
    process, _ = install_probe(monkeypatch, {"streams": [{"codec_type": "audio", "duration": "1"}]})
    cancelled = False

    def request_cancel():
        nonlocal cancelled
        cancelled = True

    process.on_communicate = request_cancel
    with pytest.raises(InterruptedError, match="取消"):
        audio.audio_duration(source, cancel=lambda: cancelled)


def test_probe_timeout_reaps_a_real_child(source, monkeypatch):
    real_popen = subprocess.Popen
    children = []
    monkeypatch.setattr(audio.shutil, "which", lambda name: "ffprobe")
    monkeypatch.setattr(audio, "_PROBE_TIMEOUT_SECONDS", .03)

    def sleeping_probe(arguments, **kwargs):
        child = real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(audio.subprocess, "Popen", sleeping_probe)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="读取音频超时"):
        audio.audio_duration(source)
    assert time.monotonic() - started < 3
    assert len(children) == 1 and children[0].poll() is not None
    assert children[0].stdout.closed and children[0].stderr.closed
