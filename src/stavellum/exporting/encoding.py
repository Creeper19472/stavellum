"""Encoder selection and a cancellable real NVENC initialization probe."""

from __future__ import annotations

import subprocess
import time

from stavellum.domain.models import RenderSettings


def video_arguments(settings: RenderSettings, encoder: str) -> list[str]:
    if encoder == "h264_nvenc":
        return ["-c:v", encoder, "-preset", settings.nvenc_preset, "-tune", "hq",
                "-rc", "vbr", "-cq", str(settings.nvenc_cq), "-b:v", "0",
                "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", settings.preset, "-crf", str(settings.crf),
            "-pix_fmt", "yuv420p"]


def probe_nvenc(executable: str, settings: RenderSettings, cancelled) -> tuple[bool, str]:
    """Encode two real frames; an advertised encoder alone is insufficient."""
    if cancelled():
        raise InterruptedError("已取消导出。")
    arguments = [executable, "-hide_banner", "-loglevel", "error", "-nostdin",
                 "-f", "lavfi", "-i",
                 f"color=c=black:s={settings.width}x{settings.height}:r={settings.fps},format=rgba",
                 "-frames:v", "2", *video_arguments(settings, "h264_nvenc"),
                 "-an", "-f", "null", "-"]
    process = subprocess.Popen(arguments, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    started = time.perf_counter()
    try:
        while True:
            if cancelled():
                raise InterruptedError("已取消导出。")
            try:
                _, output = process.communicate(timeout=.1)
                if cancelled():
                    raise InterruptedError("已取消导出。")
                details = output.decode("utf-8", "replace").strip()
                return process.returncode == 0, details
            except subprocess.TimeoutExpired:
                if time.perf_counter() - started > 10:
                    return False, "NVENC 实际编码探测超时。"
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
        if process.stderr is not None:
            process.stderr.close()


def select_encoder(executable: str, settings: RenderSettings, cancelled) -> tuple[str, list[str]]:
    if settings.video_encoder == "libx264":
        return "libx264", []
    available, reason = probe_nvenc(executable, settings, cancelled)
    if available:
        return "h264_nvenc", []
    reason = reason or "NVENC 未能初始化。"
    if settings.video_encoder == "h264_nvenc":
        raise RuntimeError("FFmpeg NVENC 编码器不可用：" + reason)
    return "libx264", [reason]


def hardware_failure(details: str) -> bool:
    """Retry only encoder/device faults, never invalid audio or output paths."""
    value = details.lower()
    if any(token in value for token in (
        "permission denied", "no space left", "error opening input", "invalid data found",
        "no such file or directory", "error decoding", "error demuxing",
        "unsupported channel layout", "error initializing an internal resampler",
    )):
        return False
    return any(token in value for token in (
        "nvenc", "nvencode", "nvcuda", "cuda_error", "cuinit", "cuctx", "capable devices",
        "openencodesession", "initializeencoder",
        "driver does not support", "device lost", "device removed",
    ))
