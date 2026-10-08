"""Offline raw-frame export with bounded memory, cancellation and atomic completion."""

from __future__ import annotations

import json
import math
import queue
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

from stavellum.domain.models import ProjectDocument
from stavellum.domain.progress import ProgressReporter
from stavellum.graphics.musicfont import MUSIC_FALLBACK, MUSIC_FAMILY
from stavellum.presentation.scene import compile_scene
from stavellum.rendering.render import FrameRenderer

from .audio import audio_duration


class _EncodingFailure(RuntimeError):
    def __init__(self, details: str, statistics: dict):
        super().__init__("FFmpeg 编码失败：" + details)
        self.details = details
        self.statistics = statistics


class _LegacyFrameStream:
    """Keep older renderers compatible with the bounded export pipeline."""

    pixel_format = "rgba"

    def __init__(self, renderer, times, cancelled):
        self.renderer = renderer
        self.times = iter(enumerate(times))
        self.cancelled = cancelled
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        if self.cancelled():
            raise InterruptedError("已取消导出。")
        index, seconds = next(self.times)
        return index, self.renderer.render_frame(seconds)

    def close(self):
        self.closed = True

    def report(self):
        return {"readback_mode": "synchronous"}


class _FrameQueue(queue.Queue):
    def __init__(self):
        super().__init__(maxsize=2)
        self.peak = 0

    def _put(self, item):
        super()._put(item)
        self.peak = max(self.peak, self._qsize())


class _WriterFailure(Exception):
    def __init__(self, error):
        self.error = error


def _frame_buffer_estimate(statistics):
    # Pending QImages share the native output allocation; do not count it twice.
    # Summing independent phase peaks is a conservative bound, not measured RSS.
    return (statistics.get("frame_buffer_peak_bytes", 0)
            + max(statistics.get("readback_buffer_peak_bytes", 0),
                  statistics.get("frame_stream_pending_peak_bytes", 0))
            + statistics.get("readback_staging_peak_bytes", 0))


def _encode_attempt(executable, video_encoder, scene, renderer, audio_path, temporary,
                    frames, total_seconds, report, cancelled, *, detail=None, attempt=1,
                    fallback_reason=""):
    from .encoding import video_arguments

    settings = scene.settings
    times = (index / settings.fps for index in range(frames))
    factory = getattr(renderer, "export_frames", None)
    stream = (factory(times, cancelled) if factory is not None
              else _LegacyFrameStream(renderer, times, cancelled))
    pixel_format = stream.pixel_format
    audio_filter = "apad"
    if settings.intro_delay_seconds > 0:
        audio_filter = f"adelay=delays={settings.intro_delay_seconds * 1000:.9f}:all=1,apad"
    arguments = [executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                 "-f", "rawvideo", "-pixel_format", pixel_format, "-video_size",
                 f"{settings.width}x{settings.height}", "-framerate", str(settings.fps),
                 "-i", "pipe:0", "-i", str(Path(audio_path).resolve()),
                 "-map", "0:v:0", "-map", "1:a:0", *video_arguments(settings, video_encoder),
                 "-c:a", "aac", "-b:a", "256k", "-af", audio_filter, "-t",
                 f"{total_seconds:.9f}", "-movflags", "+faststart", str(temporary)]
    process = None
    errors: list[str] = []
    finished = threading.Event()
    abort = threading.Event()
    writer_done = threading.Event()
    writer_errors = []
    frame_queue = _FrameQueue()
    sentinel = object()
    state_lock = threading.Lock()
    buffered_bytes = 0
    statistics = {"video_encoder": video_encoder, "frames_written": 0,
                  "pipe_wait_seconds": 0.0, "encoder_finalize_seconds": 0.0,
                  "frame_queue_wait_seconds": 0.0, "frame_queue_peak": 0,
                  "frame_buffer_peak_bytes": 0, "export_pixel_format": pixel_format}
    started = time.perf_counter()
    reader = watcher = writer = None
    writer_started = False
    failure = None
    succeeded = False

    def collect_errors():
        assert process.stderr is not None
        for line in iter(process.stderr.readline, b""):
            errors.append(line.decode("utf-8", "replace").rstrip())
            if len(errors) > 200:
                del errors[:50]

    def watch_cancel():
        while not finished.wait(.05):
            if (cancelled() or abort.is_set()) and process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass
                return

    def fail_writer(error):
        writer_errors.append(error)
        abort.set()

    def write_frames():
        nonlocal buffered_bytes
        assert process.stdin is not None
        try:
            while not abort.is_set():
                try:
                    item = frame_queue.get(timeout=.05)
                except queue.Empty:
                    if cancelled():
                        raise InterruptedError("已取消导出。")
                    if process.poll() is not None:
                        raise BrokenPipeError("编码器在接收全部视频帧之前退出。")
                    continue
                if item is sentinel:
                    break
                _, frame, size = item
                del item
                view = None
                try:
                    view = memoryview(frame.constBits())
                    offset = 0
                    pipe_started = time.perf_counter()
                    try:
                        while offset < size:
                            if cancelled() or abort.is_set():
                                raise InterruptedError("已取消导出。")
                            written = process.stdin.write(view[offset:size])
                            if written is None or written <= 0 or written > size - offset:
                                raise BrokenPipeError("编码器管道未能完整写入视频帧。")
                            offset += written
                    finally:
                        with state_lock:
                            statistics["pipe_wait_seconds"] += time.perf_counter() - pipe_started
                    with state_lock:
                        statistics["frames_written"] += 1
                finally:
                    if view is not None:
                        view.release()
                    # The image owns constBits() storage until every partial write ends.
                    del frame
                    with state_lock:
                        buffered_bytes -= size
        except BaseException as error:
            fail_writer(error)
        finally:
            try:
                process.stdin.close()
            except BaseException as error:
                if not writer_errors:
                    fail_writer(error)
            writer_done.set()

    last_reported = 0
    phase = "frames"

    def check_pipeline(*, force_detail=False):
        nonlocal last_reported
        if cancelled():
            raise InterruptedError("已取消导出。")
        if writer_errors:
            raise _WriterFailure(writer_errors[0])
        with state_lock:
            written = statistics["frames_written"]
        if written and (not last_reported or written == frames
                        or written - last_reported >= max(1, settings.fps // 2)):
            if written != last_reported:
                report(.05 + .94 * written / frames,
                       f"{renderer.render_backend} / {video_encoder}：逐帧渲染 {written}/{frames}")
                last_reported = written
            if cancelled():
                raise InterruptedError("已取消导出。")
        if detail is not None:
            message = (f"逐帧渲染并送入编码器 {written}/{frames}"
                       if phase == "frames" else "正在完成音频编码与 MP4 封装…")
            detail.emit(phase, message, force=force_detail, attempt=attempt,
                        completed=written, total=frames, unit="frames", fps=settings.fps,
                        render_backend=renderer.render_backend, video_encoder=video_encoder,
                        fallback_reason="\n".join(filter(None, [fallback_reason,
                            *getattr(renderer, "fallback_reasons", [])])))

    def enqueue(item):
        while True:
            check_pipeline()
            queue_started = time.perf_counter()
            try:
                frame_queue.put(item, timeout=.05)
                return
            except queue.Full:
                pass
            finally:
                statistics["frame_queue_wait_seconds"] += time.perf_counter() - queue_started

    try:
        check_pipeline(force_detail=True)
        process = subprocess.Popen(arguments, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                   stdout=subprocess.DEVNULL,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        reader = threading.Thread(target=collect_errors, name="export-stderr", daemon=True)
        watcher = threading.Thread(target=watch_cancel, name="export-cancel", daemon=True)
        writer = threading.Thread(target=write_frames, name="export-writer", daemon=True)
        reader.start()
        watcher.start()
        writer.start()
        writer_started = True
        for index in range(frames):
            check_pipeline()
            try:
                stream_index, frame = next(stream)
            except StopIteration:
                raise RuntimeError("导出帧流在全部视频帧生成之前结束。") from None
            if stream_index != index:
                raise RuntimeError(f"导出帧顺序错误：需要 {index}，收到 {stream_index}。")
            size = frame.sizeInBytes()
            with state_lock:
                buffered_bytes += size
                statistics["frame_buffer_peak_bytes"] = max(
                    statistics["frame_buffer_peak_bytes"], buffered_bytes)
            enqueue((index, frame, size))
            del frame
        enqueue(sentinel)
        while not writer_done.wait(.05):
            check_pipeline()
        writer.join()
        check_pipeline(force_detail=True)
        phase = "finalize"
        check_pipeline(force_detail=True)
        finalized = time.perf_counter()
        while True:
            try:
                return_code = process.wait(timeout=.05)
                break
            except subprocess.TimeoutExpired:
                check_pipeline()
        statistics["encoder_finalize_seconds"] = time.perf_counter() - finalized
        check_pipeline()
        succeeded = return_code == 0
    except BaseException as error:
        failure = error
    finally:
        # Terminating the child releases a writer blocked inside an OS pipe write.
        # Only that writer closes stdin; RHI cleanup stays on this thread.
        if not succeeded:
            abort.set()
        finished.set()
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if writer_started:
            writer.join(timeout=5)
            if writer.is_alive():
                succeeded = False
                if failure is None or isinstance(failure, _WriterFailure):
                    failure = RuntimeError("视频管道线程未能在编码器退出后停止。")
        if reader is not None and reader.ident is not None:
            reader.join(timeout=3)
        if watcher is not None and watcher.ident is not None:
            watcher.join(timeout=1)
        if process is not None:
            pipes = [process.stderr]
            if not writer_started:
                pipes.append(process.stdin)
            for pipe in pipes:
                if pipe and not pipe.closed:
                    try:
                        pipe.close()
                    except OSError:
                        pass
        try:
            stream.close()
            statistics.update(stream.report())
        except BaseException as error:
            # A failed cleanup cannot safely qualify for another encoder attempt.
            # Preserve an original drawing error so it remains diagnosable.
            if failure is None or isinstance(failure, _WriterFailure):
                failure = error
            succeeded = False
        while True:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                break
        statistics["frame_queue_peak"] = frame_queue.peak
        statistics["total_frame_buffer_peak_bytes"] = _frame_buffer_estimate(statistics)
        statistics["frame_buffer_accounting"] = (
            "conservative sum of writer, native/pending output and Qt staging phase peaks; "
            "excludes caller-retained images and driver allocations")
        statistics["wall_seconds"] = time.perf_counter() - started
        statistics["succeeded"] = succeeded
        statistics["errors"] = list(errors)
    if failure is not None and not isinstance(failure, _WriterFailure):
        raise failure
    if not succeeded:
        pipe_error = failure.error if isinstance(failure, _WriterFailure) else None
        details = "\n".join(errors[-12:]) or str(pipe_error or "编码器异常退出。")
        raise _EncodingFailure(details, statistics) from pipe_error
    return statistics


def export_video(document: ProjectDocument, output_path: str | Path, progress=None, cancel=None,
                 *, progress_detail=None) -> str:
    from stavellum.graphics.qt import prepare_render_app

    from .encoding import hardware_failure, select_encoder

    total_started = time.perf_counter()
    detail = ProgressReporter(progress_detail)
    progress_callback = progress or (lambda fraction, message: None)
    last_progress = 0.0

    def report(fraction, message):
        nonlocal last_progress
        last_progress = max(last_progress, fraction)
        progress_callback(last_progress, message)

    cancelled = cancel or (lambda: False)
    detail.emit("check", "正在检查音频与导出设置…", force=True)
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("未找到 FFmpeg，请将 ffmpeg.exe 所在目录加入 PATH。")
    document.validate()
    if cancelled():
        raise InterruptedError("已取消导出。")
    target = Path(output_path).resolve()
    if target.suffix.lower() != ".mp4":
        raise ValueError("视频输出文件必须使用 .mp4 扩展名。")
    duration = audio_duration(document.audio_path, cancel=cancelled)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{uuid.uuid4().hex[:12]}.partial.mp4")
    summary_temporary = temporary.with_suffix(".json")
    log_temporary = temporary.with_suffix(".log")
    log_path = target.with_suffix(".render.log")
    report(0.0, "正在准备谱面与动画…")
    detail.emit("compile", "正在准备谱面与动画…", force=True)
    prepare_render_app(document.settings)
    compile_started = time.perf_counter()
    def compile_progress(fraction, message):
        report(fraction * .04, message)
        detail.emit("compile", message, force=True)

    scene = compile_scene(document, progress=compile_progress, cancel=cancelled)
    compile_seconds = time.perf_counter() - compile_started
    if cancelled():
        raise InterruptedError("已取消导出。")
    detail.emit("prepare", "正在初始化渲染器与选择编码器…", force=True)
    renderer = FrameRenderer(scene)
    settings = scene.settings
    total_seconds = settings.presentation_duration(duration, scene.score_duration)
    frames = math.ceil(total_seconds * settings.fps)
    total_seconds = frames / settings.fps
    attempts = []
    encoder_fallbacks = []
    succeeded = False
    loop_started = time.perf_counter()
    log_lines = []
    attempt_number = 1
    try:
        selected_encoder, encoder_fallbacks = select_encoder(executable, settings, cancelled)
        log_lines.extend(f"编码器后备：{reason}" for reason in encoder_fallbacks)
        report(.05, f"{renderer.render_backend} / {selected_encoder}：开始导出，共 {frames} 帧。")
        while True:
            try:
                attempt = _encode_attempt(executable, selected_encoder, scene, renderer,
                                          document.audio_path, temporary, frames, total_seconds,
                                          report, cancelled, detail=detail, attempt=attempt_number,
                                          fallback_reason="\n".join(encoder_fallbacks))
                attempts.append(attempt)
                log_lines.extend(attempt["errors"])
                break
            except _EncodingFailure as failure:
                attempts.append(failure.statistics)
                log_lines.extend(failure.statistics["errors"])
                if cancelled():
                    raise InterruptedError("已取消导出。") from None
                if (settings.video_encoder == "auto" and selected_encoder == "h264_nvenc"
                        and hardware_failure(failure.details)):
                    encoder_fallbacks.append(failure.details)
                    log_lines.append("NVENC 运行失败；从第 0 帧以 libx264 重导一次。")
                    temporary.unlink(missing_ok=True)
                    selected_encoder = "libx264"
                    attempt_number += 1
                    report(last_progress, "NVENC 运行失败，正在从第 0 帧用 CPU 编码器重新导出…")
                    continue
                raise
        if cancelled():
            raise InterruptedError("已取消导出。")
        summary = {
            "frames": frames, "fps": settings.fps, "width": settings.width, "height": settings.height,
            "duration_seconds": total_seconds, "audio_duration_seconds": duration,
            "score_duration_seconds": scene.score_duration,
            "score_start_in_audio_sec": settings.score_start_in_audio_sec,
            "intro_delay_seconds": settings.intro_delay_seconds,
            "music_font": MUSIC_FAMILY, "music_font_fallback": MUSIC_FALLBACK,
            "camera_half_window_seconds": scene.camera.half_window_seconds,
            "camera_certified_drift_at_maximum_scale_px": scene.camera.certified_drift_source * scene.scale,
            "urgent_layout_intervals": scene.layout.urgent_intervals,
            "layout_recovery_intervals": scene.layout.recovery_intervals,
            "wall_seconds": round(time.perf_counter() - loop_started, 3),
            "total_wall_seconds": round(time.perf_counter() - total_started, 3),
            "compile_seconds": compile_seconds,
            "compilation": getattr(scene, "compilation_report", {}),
            "video_encoder": selected_encoder, "requested_video_encoder": settings.video_encoder,
            "encoder_parameters": ({"cq": settings.nvenc_cq, "preset": settings.nvenc_preset}
                                   if selected_encoder == "h264_nvenc"
                                   else {"crf": settings.crf, "preset": settings.preset}),
            "encoder_fallback_reasons": encoder_fallbacks, "encoding_attempts": attempts,
            # Rendering, readback and pipe writes overlap; their durations must
            # not be summed to infer the elapsed export time.
            "pipe_wait_seconds": sum(item["pipe_wait_seconds"] for item in attempts),
            "encoder_finalize_seconds": sum(item["encoder_finalize_seconds"] for item in attempts),
            "frame_queue_wait_seconds": sum(item["frame_queue_wait_seconds"] for item in attempts),
            "frame_queue_peak": max(item["frame_queue_peak"] for item in attempts),
            "frame_buffer_peak_bytes": max(item["frame_buffer_peak_bytes"] for item in attempts),
            "export_pixel_format": attempts[-1]["export_pixel_format"],
            "cache_peak_bytes": renderer.cache_peak_bytes, "cache_limit_bytes": renderer.cache_limit,
            "source_version": document.project.source_version,
            "diagnostics": [{"severity": d.severity, "code": d.code, "message": d.message}
                            for d in scene.diagnostics],
            **renderer.backend_report(),
        }
        summary["total_frame_buffer_peak_bytes"] = _frame_buffer_estimate(summary)
        summary["frame_buffer_accounting"] = attempts[-1].get(
            "frame_buffer_accounting", "conservative sum of frame-buffer phase peaks")
        summary.setdefault("readback_mode", attempts[-1].get("readback_mode", "synchronous"))
        log_lines.extend(f"渲染后备：{reason}" for reason in renderer.fallback_reasons)
        log_lines.append(f"逐帧编码完成：{renderer.render_backend} / {selected_encoder}。")
        detail.emit("save", "正在保存视频与渲染报告…", force=True,
                    attempt=attempt_number, completed=frames, total=frames, unit="frames",
                    fps=settings.fps, render_backend=renderer.render_backend,
                    video_encoder=selected_encoder,
                    fallback_reason="\n".join([*encoder_fallbacks, *renderer.fallback_reasons]))
        summary_temporary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        log_temporary.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        if cancelled():
            raise InterruptedError("已取消导出。")
        summary_temporary.replace(target.with_suffix(".render.json"))
        log_temporary.replace(log_path)
        temporary.replace(target)
        succeeded = True
    except BaseException as error:
        log_lines.append(str(error))
        raise
    finally:
        renderer.close()
        temporary.unlink(missing_ok=True)
        summary_temporary.unlink(missing_ok=True)
        log_temporary.unlink(missing_ok=True)
        if not succeeded and not log_lines:
            log_lines.append("导出已取消或中断。")
        if not succeeded:
            try:
                log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
            except OSError:
                pass  # Logging must not mask the encoder/file/cancellation error.
    detail.emit("done", "视频导出完成。", force=True,
                attempt=attempt_number, completed=frames, total=frames, unit="frames",
                fps=settings.fps, render_backend=renderer.render_backend,
                video_encoder=selected_encoder,
                fallback_reason="\n".join([*encoder_fallbacks, *renderer.fallback_reasons]))
    report(1.0, "视频导出完成。")
    return str(target)
