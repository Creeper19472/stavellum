"""The main window keeps one export view and uses worker terminal signals."""

import os
import queue
import shutil
import threading
import time
import wave
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import QApplication

from stavellum.domain.models import (
    Metadata,
    NoteEvent,
    PartMapping,
    ProjectDocument,
    ProjectIR,
    TrackInfo,
)
from stavellum.domain.progress import ExportProgress
from stavellum.engraving import notation
from stavellum.exporting.audio import AUDIO_FILE_FILTER
from stavellum.graphics import qt
from stavellum.ui import background, gui

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class FakeJob(QObject):
    progress = Signal(float, str)
    progress_detail = Signal(object)
    succeeded = Signal(object)
    failed = Signal(str, str)
    cancelled = Signal()
    finished = Signal()

    def __init__(self, operation, payload, parent=None):
        super().__init__(parent)
        self.operation, self.payload = operation, payload
        self.running = True
        self._cancel_requested_at = None
        self.cancel_calls = 0

    def start(self):
        pass

    def cancel(self):
        self.cancel_calls += 1
        self._cancel_requested_at = time.monotonic()

    def shutdown(self):
        self.running = False


@pytest.fixture
def window(app, monkeypatch):
    monkeypatch.setattr(gui, "BackgroundJob", FakeJob)
    widget = gui.MainWindow()
    monkeypatch.setattr(widget, "compile_preview", lambda: None)
    project = ProjectIR("fixture.mid", "midi", "测试曲", tracks=[TrackInfo("v", "Violin")],
                        notes=[NoteEvent("n", "v", 0, 480, 72)], duration_ticks=480)
    widget.set_document(ProjectDocument(project, [PartMapping("v", "Violin", ["v"])],
                                       metadata=Metadata("Test")))
    yield widget
    widget._dirty = False
    widget.close()
    widget.deleteLater()
    app.processEvents()


def start_export(window, tmp_path):
    destination = str(tmp_path / "video.mp4")
    window._start_job("video", (window.document, destination), window._export_finished)
    return window._job, window._export_dialog


@pytest.mark.parametrize("operation", ["video", "parts"])
def test_export_menu_uses_existing_worker_and_progress_workflow(window, tmp_path, monkeypatch, operation):
    audio = tmp_path / "song.wav"
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(b"\x00\x00" * 8000)
    window.document.audio_path = str(audio)
    destination = str(tmp_path / "result.mp4") if operation == "video" else str(tmp_path)
    monkeypatch.setattr(gui.QFileDialog, "getSaveFileName", lambda *args: (destination, ""))
    monkeypatch.setattr(gui.QFileDialog, "getExistingDirectory", lambda *args: destination)
    action = window.export_video_action if operation == "video" else window.export_parts_action
    action.trigger()
    job = window._job
    assert job.operation == operation and job.payload[1] == destination
    assert job.payload[0].mappings == window.document.mappings
    assert window._export_dialog.isVisible()
    assert window.export_detail_button.isEnabled()
    assert not window.export_video_action.isEnabled()
    assert not window.export_parts_action.isEnabled()
    job.cancelled.emit()
    job.finished.emit()
    assert window._job is None
    assert window.export_video_action.isEnabled() and window.export_parts_action.isEnabled()


def test_hide_reopen_and_worker_success_keep_result(window, tmp_path, monkeypatch):
    monkeypatch.setattr(gui.QMessageBox, "information", lambda *args: pytest.fail("duplicate success popup"))
    job, dialog = start_export(window, tmp_path)
    assert dialog.isVisible() and window.export_detail_button.isEnabled()
    assert not window.tabs.isEnabled()
    dialog.close()
    assert not dialog.isVisible() and job.cancel_calls == 0
    job.progress.emit(1.0, "旧回调完成")
    job.progress_detail.emit(ExportProgress("done", "等待后台确认", 5, completed=100,
                                             total=100, unit="frames"))
    assert dialog.running and dialog.progress_bar.value() < 1000
    job.succeeded.emit(str(tmp_path / "video.mp4"))
    job.finished.emit()
    assert window._job is None and not dialog.running and not dialog.isVisible()
    assert dialog.progress_bar.value() == 1000 and window.progress.value() == 1000
    assert window.tabs.isEnabled()
    window._show_export_details()
    assert dialog.isVisible()


def test_cancel_is_requested_once_and_late_progress_cannot_reset_status(window, tmp_path):
    job, dialog = start_export(window, tmp_path)
    job.progress_detail.emit(ExportProgress("frames", "渲染", 1, completed=10, total=100))
    window._cancel_job()
    window._cancel_job()
    dialog.request_cancel()
    assert job.cancel_calls == 1 and not window.cancel_button.isEnabled()
    job.progress_detail.emit(ExportProgress("frames", "迟到的进度", 2, completed=20, total=100))
    assert "取消" in window.job_label.text() and "取消" in dialog.status_label.text()
    assert dialog.progress_bar.value() == 100
    job.cancelled.emit()
    job.finished.emit()
    assert not dialog.running and "已取消" in dialog.status_label.text()


def test_failure_and_start_failure_remain_available(window, tmp_path, monkeypatch):
    popups = []
    monkeypatch.setattr(gui.QMessageBox, "exec", lambda box: popups.append(box.text()))
    job, dialog = start_export(window, tmp_path)
    job.failed.emit("磁盘写入失败", "磁盘已满")
    job.finished.emit()
    assert "磁盘写入失败" in dialog.status_label.text()
    assert dialog.error_details.toPlainText() == "磁盘已满"
    assert len(popups) == 1
    monkeypatch.setattr(FakeJob, "start", lambda self: (_ for _ in ()).throw(OSError("无法启动")))
    window._start_job("parts", (window.document, str(tmp_path)), window._export_finished)
    assert window._job is None and not window._export_dialog.running
    assert "无法启动" in window._export_dialog.status_label.text()
    assert len(popups) == 2


def test_audio_selection_uses_common_formats_and_cancel_does_not_change_path(window, tmp_path, monkeypatch):
    selected = str(tmp_path / "原曲.FLAC")
    calls = []

    def choose(*args):
        calls.append(args)
        return selected, ""

    sources = []
    monkeypatch.setattr(gui.QFileDialog, "getOpenFileName", choose)
    monkeypatch.setattr(window, "_set_audio", sources.append)
    window.audio_action.trigger()
    assert window.document.audio_path == selected and sources == [selected] and window._dirty
    assert calls[0][-1] == AUDIO_FILE_FILTER and "*.flac" in calls[0][-1]
    monkeypatch.setattr(gui.QFileDialog, "getOpenFileName", lambda *args: ("", ""))
    window.audio_action.trigger()
    assert sources == [selected] and window.document.audio_path == selected


def test_preview_error_restores_silent_transport_and_keeps_export_path(window, tmp_path):
    path = str(tmp_path / "invalid.flac")
    window.document.audio_path = path
    window.player.setSource(QUrl.fromLocalFile(path))
    window._audio_error(QMediaPlayer.Error.FormatError, "不支持的解码器")
    assert window.player.source().isEmpty() and window.document.audio_path == path
    assert "可无声预览" in window.audio_label.text()
    window._playing = True
    window._play_origin = 0
    window._play_started = time.monotonic() - 1
    window._advance_playback_clock()
    assert window._position > 0


def test_background_cancel_is_idempotent_and_terminal_result_wins(monkeypatch):
    job = background.BackgroundJob("parts", None)
    job._process = SimpleNamespace()
    messages = []
    job.progress.connect(lambda *args: messages.append(args))
    job.cancel()
    job.cancel()
    assert len(messages) == 1
    job._done = True
    job.cancel()
    assert len(messages) == 1
    job._close_queue()
    cancelled = threading.Event()
    worker_messages = queue.Queue()
    monkeypatch.setattr(qt, "ensure_app", lambda **kwargs: None)

    def export(*args, **kwargs):
        kwargs["progress_detail"](ExportProgress("done", "已保存", 1))
        cancelled.set()  # A cancel request arrives after files have been committed.
        return ["saved.musicxml", "saved.pdf"]

    monkeypatch.setattr(notation, "export_parts", export)
    background._worker("parts", (None, "destination"), worker_messages, cancelled)
    assert worker_messages.get_nowait()[0] == "progress_detail"
    assert worker_messages.get_nowait() == ("result", ["saved.musicxml", "saved.pdf"])


def test_dead_worker_reports_failure_once_even_during_nested_event_loop():
    job = background.BackgroundJob("parts", None)
    job._close_queue()
    job._messages = queue.Queue()
    job._close_queue = lambda: None
    job._process = SimpleNamespace(is_alive=lambda: False, exitcode=1,
                                   join=lambda **kwargs: None)
    job._dead_since = time.monotonic() - 1
    failures = []

    def on_failure(*args):
        failures.append(args)
        if len(failures) == 1:
            job._poll()  # QMessageBox.exec() can run the polling timer again.

    job.failed.connect(on_failure)
    job._poll()
    assert len(failures) == 1 and not job.running


def test_shutdown_handles_a_process_that_failed_to_start():
    job = background.BackgroundJob("parts", None)
    job._process = job._context.Process(target=print)
    assert job._process.pid is None
    job.shutdown()
    assert not job.running


@pytest.mark.integration
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
@pytest.mark.parametrize("operation", ["video", "parts"])
def test_spawned_export_drives_window_to_saved_result(app, tmp_path, monkeypatch, operation):
    widget = gui.MainWindow()
    monkeypatch.setattr(widget, "compile_preview", lambda: None)
    project = ProjectIR("fixture.mid", "midi", "Test", tracks=[TrackInfo("v", "Violin")],
                        notes=[NoteEvent("n", "v", 0, 480, 72)], duration_ticks=480)
    document = ProjectDocument(project, [PartMapping("v", "Violin", ["v"])], metadata=Metadata("Test"))
    document.settings.width, document.settings.height = 320, 240
    document.settings.fps = 24
    document.settings.render_backend = "cpu"
    document.settings.video_encoder = "libx264"
    document.settings.preset = "ultrafast"
    audio_path = tmp_path / "audio.wav"
    with wave.open(str(audio_path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\x00\x00" * 8000)
    document.audio_path = str(audio_path)
    errors = []
    monkeypatch.setattr(widget, "_job_error", lambda *args: errors.append(args))
    widget.set_document(document)
    destination = tmp_path / "output.mp4" if operation == "video" else tmp_path / "parts"
    widget._start_job(operation, (document, str(destination)), widget._export_finished)
    job = widget._job
    snapshots = []
    job.progress_detail.connect(snapshots.append)
    try:
        deadline = time.monotonic() + 30
        while widget._job is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert widget._job is None and not errors
        assert snapshots and snapshots[-1].phase == "done"
        assert any(value.total > 0 for value in snapshots)
        assert not widget._export_dialog.running
        assert widget._export_dialog.progress_bar.value() == 1000
        assert destination.exists()
    finally:
        job.shutdown()
        widget._dirty = False
        widget.close()
        widget.deleteLater()
        app.processEvents()
