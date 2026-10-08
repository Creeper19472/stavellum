"""Original audio inputs and bounded, cancellable duration probing."""

import json
import math
import shutil
import subprocess
import time
import wave
from collections.abc import Callable
from pathlib import Path

AUDIO_FILE_FILTER = (
    "常见音频 (*.wav *.flac *.mp3 *.m4a *.aac *.ogg *.opus *.aif *.aiff *.wma);;"
    "WAV 音频 (*.wav);;FLAC 音频 (*.flac);;MP3 音频 (*.mp3);;"
    "M4A / AAC 音频 (*.m4a *.aac);;Ogg / Opus 音频 (*.ogg *.opus);;"
    "AIFF 音频 (*.aif *.aiff);;WMA 音频 (*.wma);;所有文件 (*)"
)

_PROBE_TIMEOUT_SECONDS = 10.0


def _check_cancelled(cancelled: Callable[[], bool]) -> None:
    if cancelled():
        raise InterruptedError("已取消导出。")


def _probe_audio(source: Path, cancelled: Callable[[], bool]) -> dict:
    executable = shutil.which("ffprobe")
    if not executable:
        raise RuntimeError("此音频需要 ffprobe 读取，请将 FFmpeg 和 ffprobe 加入 PATH。")
    _check_cancelled(cancelled)
    arguments = [
        executable, "-v", "error", "-select_streams", "a:0", "-show_entries",
        "stream=codec_type,duration:format=duration", "-of", "json", str(source.resolve()),
    ]
    try:
        process = subprocess.Popen(
            arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError as exc:
        raise RuntimeError(f"无法启动 ffprobe：{exc}") from exc
    started = time.monotonic()
    try:
        while True:
            _check_cancelled(cancelled)
            remaining = _PROBE_TIMEOUT_SECONDS - (time.monotonic() - started)
            if remaining <= 0:
                raise RuntimeError("读取音频超时（10 秒），请检查文件是否可正常访问。")
            try:
                output, errors = process.communicate(timeout=min(.1, remaining))
            except subprocess.TimeoutExpired:
                continue
            _check_cancelled(cancelled)
            if process.returncode != 0:
                details = errors.strip() or "文件已损坏或格式不受支持。"
                raise ValueError(f"无法读取原曲音频：{details}")
            try:
                result = json.loads(output)
            except json.JSONDecodeError as exc:
                raise ValueError("ffprobe 返回了无效的音频信息。") from exc
            if not isinstance(result, dict):
                raise ValueError("ffprobe 返回了无效的音频信息。")
            return result
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()


def _valid_duration(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    return duration if math.isfinite(duration) and duration > 0 else None


def audio_duration(path: str | Path, *, cancel: Callable[[], bool] | None = None) -> float:
    """Read the first audio track's duration, retaining a fast path for PCM WAV."""
    cancelled = cancel or (lambda: False)
    _check_cancelled(cancelled)
    source = Path(path)
    if not source.is_file():
        raise ValueError("找不到原曲音频，请先选择存在的音频文件。")
    try:
        with wave.open(str(source), "rb") as audio:
            rate = audio.getframerate()
            duration = audio.getnframes() / rate if rate > 0 else None
    except (wave.Error, EOFError):
        information = _probe_audio(source, cancelled)
        streams = information.get("streams")
        if (not isinstance(streams, list) or not streams
                or not isinstance(streams[0], dict)
                or streams[0].get("codec_type") != "audio"):
            raise ValueError("所选文件不包含可用的音频轨道。")
        duration = _valid_duration(streams[0].get("duration"))
        if duration is None:
            container = information.get("format")
            duration = _valid_duration(container.get("duration")) if isinstance(container, dict) else None
    except OSError as exc:
        raise ValueError(f"无法读取原曲音频：{exc}") from exc
    _check_cancelled(cancelled)
    if duration is None or _valid_duration(duration) is None:
        raise ValueError("音频时长无效，必须为大于零的有限数值。")
    return duration
