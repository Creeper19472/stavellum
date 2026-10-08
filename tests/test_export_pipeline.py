"""Bounded frame ownership and encoder-pipe failures without requiring a GPU."""

from __future__ import annotations

import io
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from PySide6.QtGui import QImage

from stavellum.domain.models import RenderSettings
from stavellum.domain.progress import ProgressReporter
from stavellum.exporting import export


class RecordingPipe:
    def __init__(self, process, *, partial=7, blocked=False, failed=False):
        self.process = process
        self.partial = partial
        self.blocked = blocked
        self.failed = failed
        self.closed = False
        self.data = bytearray()
        self.threads = []
        self.entered = threading.Event()

    def write(self, data):
        self.threads.append(threading.current_thread().name)
        self.entered.set()
        if self.blocked:
            assert self.process.exited.wait(3), "encoder pipe was not released"
            raise BrokenPipeError("child terminated")
        if self.failed:
            self.process.returncode = 1
            self.process.exited.set()
            raise BrokenPipeError("early encoder exit")
        size = min(len(data), self.partial)
        self.data.extend(data[:size])
        return size

    def close(self):
        self.threads.append(threading.current_thread().name)
        self.closed = True
        if self.process.returncode is None:
            self.process.returncode = 0
            self.process.exited.set()


class Process:
    def __init__(self, **pipe_options):
        self.returncode = None
        self.exited = threading.Event()
        self.terminated = False
        self.stdin = RecordingPipe(self, **pipe_options)
        self.stderr = io.BytesIO(b"encoder diagnostic\n" if pipe_options.get("failed") else b"")

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15
        self.exited.set()

    def kill(self):
        self.terminate()

    def wait(self, timeout=None):
        if not self.exited.wait(timeout):
            raise subprocess.TimeoutExpired("child", timeout)
        return self.returncode


def frame(index):
    image = QImage(4, 2, QImage.Format.Format_RGBA8888)
    image.fill(0xFF000000 | index)
    return image


class Stream:
    pixel_format = "bgra"

    def __init__(self, times, *, drawing_error=None, before_next=None, cleanup_error=False):
        self.times = iter(enumerate(times))
        self.requested = []
        self.closed = 0
        self.drawing_error = drawing_error
        self.before_next = before_next
        self.cleanup_error = cleanup_error
        self.owner = threading.get_ident()

    def __next__(self):
        assert threading.get_ident() == self.owner
        index, seconds = next(self.times)
        if self.before_next is not None:
            self.before_next(index)
        self.requested.append(seconds)
        if index == self.drawing_error:
            raise ValueError("drawing failed")
        return index, frame(index)

    def close(self):
        assert threading.get_ident() == self.owner
        self.closed += 1
        if self.cleanup_error:
            raise OSError("stream cleanup failed")

    def report(self):
        assert self.closed
        return {"readback_mode": "rhi-sync", "gpu_submit_seconds": .125,
                "readback_buffer_peak_bytes": 192}


class Renderer:
    render_backend = "cpu"

    def __init__(self, **stream_options):
        self.stream_options = stream_options
        self.streams = []

    def export_frames(self, times, cancel):
        stream = Stream(times, **self.stream_options)
        self.streams.append(stream)
        return stream


@pytest.fixture
def job(tmp_path):
    scene = SimpleNamespace(settings=RenderSettings(width=4, height=2, fps=24))

    def run(renderer, *, count=12, cancelled=lambda: False, progress=lambda *args: None,
            detail=None, attempt=1):
        return export._encode_attempt("ffmpeg", "libx264", scene, renderer,
                                      tmp_path / "audio.wav", tmp_path / "out.mp4",
                                      count, count / 24, progress, cancelled,
                                      detail=detail, attempt=attempt)

    return run


def test_writer_handles_partial_writes_in_order_and_creates_stream_before_child(job, monkeypatch):
    renderer = Renderer()
    child = Process()
    put_indices = []
    original_queue = export._FrameQueue

    class ObservedQueue(original_queue):
        def _put(self, item):
            super()._put(item)
            if isinstance(item, tuple):
                put_indices.append(item[0])

    renderer.stream_options["before_next"] = lambda index: (
        pytest.fail("next frame consumed before previous enqueue")
        if len(put_indices) != index else None)

    def popen(arguments, **kwargs):
        assert len(renderer.streams) == 1 and not renderer.streams[0].requested
        assert arguments[arguments.index("-pixel_format") + 1] == "bgra"
        return child

    monkeypatch.setattr(export, "_FrameQueue", ObservedQueue)
    monkeypatch.setattr(export.subprocess, "Popen", popen)
    progress = []
    statistics = job(renderer, progress=lambda value, message: progress.append((value, message)))
    expected_frames = [frame(index) for index in range(12)]
    assert child.stdin.data == b"".join(bytes(image.constBits()) for image in expected_frames)
    assert set(child.stdin.threads) == {"export-writer"}
    assert renderer.streams[0].closed == 1
    assert statistics["frames_written"] == 12 and statistics["succeeded"]
    assert 1 <= statistics["frame_queue_peak"] <= 2
    assert statistics["frame_buffer_peak_bytes"] <= 4 * 4 * 2 * 4
    assert statistics["export_pixel_format"] == "bgra"
    assert statistics["readback_mode"] == "rhi-sync" and statistics["gpu_submit_seconds"] == .125
    assert progress[-1][0] == pytest.approx(.99)
    assert child.stdin.closed and child.stderr.closed


def test_detail_counts_only_complete_pipe_writes_and_enters_finalize(job, monkeypatch):
    renderer = Renderer()
    child = Process(partial=3)
    snapshots = []
    monkeypatch.setattr(export.subprocess, "Popen", lambda *args, **kwargs: child)

    def receive(snapshot):
        snapshots.append(snapshot)
        assert snapshot.completed * frame(0).sizeInBytes() <= len(child.stdin.data)
        if snapshot.phase == "finalize":
            assert child.stdin.closed and snapshot.completed == snapshot.total

    job(renderer, count=9, detail=ProgressReporter(receive), attempt=2)
    frame_progress = [item for item in snapshots if item.phase == "frames"]
    assert frame_progress[0].completed == 0
    assert frame_progress[-1].completed == frame_progress[-1].total == 9
    assert all(item.attempt == 2 and item.unit == "frames" and item.fps == 24
               and item.render_backend == "cpu" and item.video_encoder == "libx264"
               for item in snapshots)
    assert snapshots[-1].phase == "finalize"
    assert not any(item.phase == "done" for item in snapshots)


def test_blocked_pipe_reports_unchanged_counts_while_waiting(job, monkeypatch):
    renderer = Renderer()
    child = Process(blocked=True)
    snapshots = []
    cancelled = threading.Event()
    monkeypatch.setattr(export.subprocess, "Popen", lambda *args, **kwargs: child)
    timer = threading.Timer(.65, cancelled.set)
    timer.start()
    try:
        with pytest.raises(InterruptedError):
            job(renderer, count=20, cancelled=cancelled.is_set,
                detail=ProgressReporter(snapshots.append))
    finally:
        timer.cancel()
    assert len(snapshots) >= 2
    assert all(item.phase == "frames" and item.completed == 0 for item in snapshots)
    assert snapshots[-1].elapsed_seconds >= .25
    assert child.terminated and child.stdin.closed


def test_finalize_wait_keeps_its_phase_and_reports_heartbeat(job, monkeypatch):
    class SlowFinalizeProcess(Process):
        def wait(self, timeout=None):
            if timeout == .05:
                time.sleep(.05)
                if not hasattr(self, "finalize_started"):
                    self.finalize_started = time.perf_counter()
                if time.perf_counter() - self.finalize_started < .3:
                    raise subprocess.TimeoutExpired("child", timeout)
            return super().wait(timeout)

    child = SlowFinalizeProcess()
    snapshots = []
    monkeypatch.setattr(export.subprocess, "Popen", lambda *args, **kwargs: child)
    job(Renderer(), detail=ProgressReporter(snapshots.append))
    first_finalize = next(index for index, item in enumerate(snapshots)
                          if item.phase == "finalize")
    finalizing = snapshots[first_finalize:]
    assert len(finalizing) >= 2
    assert all(item.phase == "finalize" and item.completed == item.total
               for item in finalizing)


def test_legacy_renderer_remains_rgba_and_is_not_closed_by_attempt(job, monkeypatch):
    times = []
    renderer = SimpleNamespace(render_backend="cpu",
                               render_frame=lambda seconds: (times.append(seconds), frame(0))[1])
    child = Process()

    def popen(arguments, **kwargs):
        assert arguments[arguments.index("-pixel_format") + 1] == "rgba"
        return child

    monkeypatch.setattr(export.subprocess, "Popen", popen)
    statistics = job(renderer, count=3)
    assert times == [0, 1 / 24, 2 / 24]
    assert statistics["readback_mode"] == "synchronous"


@pytest.mark.parametrize("count", [3, 100])
def test_full_queue_cancel_terminates_child_and_releases_writer(job, monkeypatch, count):
    child = Process(blocked=True)
    renderer = Renderer()
    cancelled = threading.Event()
    monkeypatch.setattr(export.subprocess, "Popen", lambda *args, **kwargs: child)
    timer = threading.Timer(.15, cancelled.set)
    timer.start()
    started = time.perf_counter()
    try:
        with pytest.raises(InterruptedError):
            job(renderer, count=count, cancelled=cancelled.is_set)
    finally:
        timer.cancel()
        timer.join()
    assert time.perf_counter() - started < 3
    assert child.terminated and child.stdin.closed and child.stderr.closed
    assert len(renderer.streams[0].requested) <= 4
    assert renderer.streams[0].closed == 1
    assert not any(thread.name == "export-writer" for thread in threading.enumerate())


def test_cancelling_inside_next_terminates_child_and_stops_writer_before_stream_close(
        job, monkeypatch):
    child = Process(blocked=True)
    cancelled = threading.Event()
    events = []
    closed_state = []
    original_terminate = child.terminate
    original_pipe_close = child.stdin.close

    def terminate():
        events.append("terminate")
        original_terminate()

    def pipe_close():
        events.append("writer-close")
        original_pipe_close()

    def before_next(index):
        if index == 3:
            assert child.stdin.entered.wait(2)
            cancelled.set()
            raise InterruptedError("cancelled during next")

    class CancellationStream(Stream):
        def close(self):
            events.append("stream-close")
            closed_state.append((child.terminated, child.stdin.closed,
                                 any(thread.name == "export-writer"
                                     for thread in threading.enumerate())))
            super().close()

    class CancellationRenderer(Renderer):
        def export_frames(self, times, cancel):
            stream = CancellationStream(times, before_next=before_next)
            self.streams.append(stream)
            return stream

    child.terminate = terminate
    child.stdin.close = pipe_close
    renderer = CancellationRenderer()
    monkeypatch.setattr(export.subprocess, "Popen", lambda *args, **kwargs: child)
    with pytest.raises(InterruptedError, match="cancelled during next"):
        job(renderer, count=100, cancelled=cancelled.is_set)
    assert events.index("terminate") < events.index("writer-close") < events.index("stream-close")
    assert closed_state == [(True, True, False)]
    assert renderer.streams[0].closed == 1 and child.stderr.closed


def test_early_writer_failure_stops_producer_and_keeps_diagnostics(job, monkeypatch):
    child = Process(failed=True)
    renderer = Renderer()
    monkeypatch.setattr(export.subprocess, "Popen", lambda *args, **kwargs: child)
    with pytest.raises(export._EncodingFailure, match="encoder diagnostic") as failure:
        job(renderer, count=100)
    assert failure.value.statistics["frames_written"] == 0
    assert not failure.value.statistics["succeeded"]
    assert len(renderer.streams[0].requested) <= 4
    assert renderer.streams[0].closed == 1 and child.stdin.closed


def test_drawing_failure_is_preserved_while_blocked_writer_is_released(job, monkeypatch):
    child = Process(blocked=True)
    renderer = Renderer(drawing_error=3)
    monkeypatch.setattr(export.subprocess, "Popen", lambda *args, **kwargs: child)
    with pytest.raises(ValueError, match="drawing failed"):
        job(renderer, count=100)
    assert child.terminated and child.stdin.closed and child.stderr.closed
    assert renderer.streams[0].closed == 1


def test_child_start_failure_closes_stream(job, monkeypatch):
    renderer = Renderer()

    def unavailable(*args, **kwargs):
        raise OSError("child start failed")

    monkeypatch.setattr(export.subprocess, "Popen", unavailable)
    with pytest.raises(OSError, match="child start failed"):
        job(renderer)
    assert renderer.streams[0].closed == 1


def test_queue_wait_excludes_progress_callback_time(job, monkeypatch):
    permit_first_write = threading.Event()
    permit_next_get = threading.Event()
    clock = [0.0]
    original_queue = export._FrameQueue
    child = Process(partial=1024)
    original_write = child.stdin.write

    def first_write(data):
        if not child.stdin.data:
            assert permit_first_write.wait(2)
        return original_write(data)

    class DelayedConsumerQueue(original_queue):
        taken = 0

        def get(self, *args, **kwargs):
            if self.taken == 1:
                assert permit_next_get.wait(2)
            item = super().get(*args, **kwargs)
            self.taken += 1
            return item

        def put(self, item, *args, **kwargs):
            if isinstance(item, tuple) and item[0] == 3:
                permit_first_write.set()
            return super().put(item, *args, **kwargs)

    def progress(*args):
        # The first frame is written while enqueue retries a full queue. Account
        # for callback work on a deterministic clock without a timing threshold.
        clock[0] += 10
        permit_next_get.set()

    child.stdin.write = first_write
    monkeypatch.setattr(export, "_FrameQueue", DelayedConsumerQueue)
    monkeypatch.setattr(export.subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(export.time, "perf_counter", lambda: clock[0])
    statistics = job(Renderer(), count=4, progress=progress)
    assert permit_next_get.is_set() and statistics["frames_written"] == 4
    assert statistics["frame_queue_wait_seconds"] == 0


@pytest.mark.parametrize("drawing_error", [None, 3])
def test_cleanup_failure_prevents_retry_but_preserves_drawing_error(job, monkeypatch, drawing_error):
    child = Process(failed=drawing_error is None, blocked=drawing_error is not None)
    renderer = Renderer(drawing_error=drawing_error, cleanup_error=True)
    monkeypatch.setattr(export.subprocess, "Popen", lambda *args, **kwargs: child)
    expected = OSError if drawing_error is None else ValueError
    message = "stream cleanup failed" if drawing_error is None else "drawing failed"
    with pytest.raises(expected, match=message) as failure:
        job(renderer, count=100)
    assert not isinstance(failure.value, export._EncodingFailure)
    assert child.stdin.closed and child.stderr.closed
    assert renderer.streams[0].closed == 1


@pytest.mark.parametrize("early_exit", [False, True])
def test_real_child_releases_os_pipe_on_cancellation_or_early_exit(job, monkeypatch, early_exit):
    original_popen = subprocess.Popen
    children = []
    started = time.perf_counter()

    def child_process(*args, **kwargs):
        code = ("import sys; sys.stderr.write('real early exit\\n'); sys.exit(2)"
                if early_exit else "import time; time.sleep(30)")
        child = original_popen([sys.executable, "-c", code], **kwargs)
        children.append(child)
        return child

    # A complete image exceeds the OS pipe capacity, so write() really blocks.
    class LargeRenderer:
        render_backend = "cpu"

        def render_frame(self, seconds):
            return QImage(512, 512, QImage.Format.Format_RGBA8888)

    monkeypatch.setattr(export.subprocess, "Popen", child_process)
    expected = export._EncodingFailure if early_exit else InterruptedError
    with pytest.raises(expected):
        job(LargeRenderer(), count=100,
            cancelled=lambda: not early_exit and time.perf_counter() - started > .2)
    assert time.perf_counter() - started < 5
    assert len(children) == 1 and children[0].poll() is not None
    assert children[0].stdin.closed and children[0].stderr.closed
