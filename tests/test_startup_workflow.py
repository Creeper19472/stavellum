"""Welcome and new-project workflows commit only completed imports."""

from __future__ import annotations

import copy
import os
import time
import wave
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
from PySide6.QtCore import QEvent, QObject, QSettings, Qt, QTimer, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from stavellum import gui
from stavellum.models import (
    Metadata,
    NoteEvent,
    PartMapping,
    ProjectDocument,
    ProjectIR,
    TrackInfo,
    load_document,
    save_document,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ORIGINAL_COMPILE_PREVIEW = gui.MainWindow.compile_preview

PROCESSING = {
    "auto_simplify_accidentals": False,
    "auto_ottava": True,
    "auto_staccato": False,
    "auto_grace": True,
    "auto_dynamics": False,
}


class FakeJob(QObject):
    progress = Signal(float, str)
    progress_detail = Signal(object)
    succeeded = Signal(object)
    failed = Signal(str, str)
    cancelled = Signal()
    finished = Signal()

    def __init__(self, operation, payload, parent=None):
        super().__init__(parent)
        self.operation = operation
        self.payload = payload
        self.running = False
        self._cancel_requested_at = None
        self.cancel_calls = 0

    def start(self):
        self.running = True

    def cancel(self):
        self.cancel_calls += 1
        self._cancel_requested_at = time.monotonic()

    def shutdown(self):
        self.running = False

    def finish(self):
        self.running = False
        self.finished.emit()

    def complete(self, result):
        self.succeeded.emit(result)
        self.finish()


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path):
    return QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat)


@pytest.fixture
def make_window(app, settings, monkeypatch):
    windows = []
    monkeypatch.setattr(gui, "BackgroundJob", FakeJob)
    monkeypatch.setattr(QMessageBox, "exec", lambda box: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Discard
    )

    def capture_preview(widget, *args):
        widget.test_previews.append((copy.deepcopy(widget.document), widget._job))

    monkeypatch.setattr(gui.MainWindow, "compile_preview", capture_preview)
    monkeypatch.setattr(gui.MainWindow, "_show_error", lambda widget, error: widget.test_errors.append(error))

    def create(path=None):
        widget = gui.MainWindow(path, settings=settings)
        widget._show_welcome()
        widget.test_previews = []
        widget.test_errors = []
        windows.append(widget)
        return widget

    yield create
    for widget in windows:
        widget._dirty = False
        monkeypatch.setattr(widget, "_confirm_replace", lambda: True)
        if widget._job is not None:
            widget._job.shutdown()
            widget._job = None
        widget.close()
        widget.deleteLater()
    app.processEvents()


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "乐曲.mid"
    path.write_bytes(b"midi fixture; parsing is performed by the fake worker")
    return path


def imported_project(source, *, timing_confirmed=True):
    return ProjectIR(
        str(source), "midi" if timing_confirmed else "flp", "新乐曲",
        tracks=[TrackInfo("violin", "Violin"), TrackInfo("piano", "Piano")],
        notes=[NoteEvent("v", "violin", 0, 480, 72), NoteEvent("p", "piano", 480, 480, 60)],
        duration_ticks=960, timing_confirmed=timing_confirmed,
    )


def existing_document(source):
    project = imported_project(source)
    project.name = "已有工程"
    return ProjectDocument(
        project, [PartMapping("old", "Old violin", ["violin"])],
        metadata=Metadata(title="已有工程"),
    )


def install_existing(window, source, app, *, dirty=True):
    document = existing_document(source)
    window.set_document(document, str(source.with_suffix(".stproj")))
    app.processEvents()
    window.test_previews.clear()
    window._dirty = dirty
    renderer = SimpleNamespace(
        scene=SimpleNamespace(settings=document.settings, score_duration=1.0), close=lambda: None
    )
    window.renderer = renderer
    return document, window.project_path, renderer


def options(source, audio="", processing=None):
    from stavellum.new_project import NewProjectOptions

    return NewProjectOptions(str(source), str(audio), dict(PROCESSING if processing is None else processing))


def assert_original(window, state):
    document, path, renderer = state
    assert window.document is document
    assert window.project_path == path
    assert window.renderer is renderer
    assert window._dirty


def test_empty_startup_and_welcome_resume_editor(make_window, source, app):
    window = make_window()
    assert window.welcome.isVisible() and not window.isVisible()
    assert window.document is None and window._job is None
    install_existing(window, source, app)
    assert window.isVisible() and not window.welcome.isVisible()
    window._show_welcome()
    assert window.welcome.isVisible() and not window.isVisible()
    window._resume_editor()
    assert window.isVisible() and not window.welcome.isVisible()


def test_window_switch_hides_origin_before_showing_target(make_window, source, app):
    window = make_window()
    events = []

    class VisibilityObserver(QObject):
        def eventFilter(self, watched, event):
            if event.type() in {QEvent.Type.Hide, QEvent.Type.Show}:
                events.append((watched, event.type()))
            return False

    observer = VisibilityObserver()
    window.installEventFilter(observer)
    window.welcome.installEventFilter(observer)
    install_existing(window, source, app)
    assert events == [(window.welcome, QEvent.Type.Hide), (window, QEvent.Type.Show)]
    events.clear()
    window._show_welcome()
    assert events == [(window, QEvent.Type.Hide), (window.welcome, QEvent.Type.Show)]
    events.clear()
    window.welcome.resume_button.click()
    assert events == [(window.welcome, QEvent.Type.Hide), (window, QEvent.Type.Show)]
    assert window.welcome.isWindow() and window.isWindow()
    assert window.centralWidget().isAncestorOf(window.editor)
    assert not window.centralWidget().isAncestorOf(window.welcome)


def test_wizard_coexists_with_editor_and_preserves_choices_on_switch(
    make_window, source, app
):
    window = make_window()
    document, saved_path, _ = install_existing(window, source, app)
    window.new_action.trigger()
    window.wizard.source_edit.setText(str(source))
    assert window.isVisible() and window.wizard.isVisible()
    assert window.wizard.isWindow()
    assert window.wizard.windowType() == Qt.WindowType.Tool
    assert not window.wizard.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
    assert window.wizard.windowModality() == Qt.WindowModality.NonModal
    assert window.wizard.parentWidget() is window
    assert all(control.isEnabled() for control in (
        window.tabs, window.save_action, window.compile_button, window.export_parts_action,
        window.play_button, window.seek, window.new_action, window.open_action,
    ))
    window.metadata_controls["title"].setText("向导打开时编辑")
    assert window.save_project()
    assert load_document(saved_path).metadata.title == "向导打开时编辑"
    window.wizard.next_button.click()
    window.wizard.processing_checks["auto_staccato"].setChecked(False)
    window.wizard.move(80, 100)
    geometry = window.wizard.geometry()
    window._new_project("different.mid")
    assert window.wizard.source_edit.text() == str(source)
    assert window.wizard.steps.currentIndex() == 1
    window._show_welcome()
    assert window.welcome.isVisible() and not window.isVisible()
    assert window.wizard.isVisible() and window.wizard.parentWidget() is window.welcome
    assert window.wizard.geometry() == geometry
    window._resume_editor()
    assert window.wizard.isVisible() and window.wizard.parentWidget() is window
    assert window.wizard.geometry() == geometry
    assert not window.wizard.processing_checks["auto_staccato"].isChecked()
    window._cancel_wizard()
    assert not window.wizard.isVisible() and window.isVisible()
    assert window.document is document


def test_other_project_load_keeps_unsubmitted_wizard(make_window, source, app, tmp_path):
    window = make_window()
    window._new_project(str(source))
    window.wizard.next_button.click()
    path = tmp_path / "other.stproj"
    document = existing_document(source)
    save_document(document, path)
    window.open_path(str(path))
    assert window.welcome.isVisible() and window.wizard.isVisible()
    window._job.complete(document)
    app.processEvents()
    assert window.isVisible() and not window.welcome.isVisible()
    assert window.wizard.isVisible() and window.wizard.parentWidget() is window
    assert window.wizard.steps.currentIndex() == 1
    window._cancel_wizard()
    assert window.isVisible() and window.document is document


@pytest.mark.parametrize("entry", ["button", "close", "escape"])
@pytest.mark.parametrize("importing", [False, True])
def test_wizard_cancel_entries_share_cleanup(make_window, source, app, entry, importing):
    window = make_window()
    state = install_existing(window, source, app)
    window._new_project(str(source))
    if importing:
        window._create_project(options(source))
    job = window._job

    def cancel():
        if entry == "button":
            window.wizard.cancel_button.click()
        elif entry == "close":
            window.wizard.close()
        else:
            QTest.keyClick(window.wizard, Qt.Key.Key_Escape)

    cancel()
    if importing:
        cancel()
        assert window.wizard.isVisible() and job.cancel_calls == 1
        assert "正在取消" in window.wizard.job_label.text()
        job.succeeded.emit(imported_project(source))
        assert_original(window, state)
        job.finish()
        app.processEvents()
    assert window.isVisible() and not window.wizard.isVisible()
    assert_original(window, state)


@pytest.mark.parametrize("result", ["failure", "cancelled"])
def test_loading_failure_or_cancel_keeps_welcome(make_window, source, app, tmp_path, result):
    window = make_window()
    path = tmp_path / "project.stproj"
    path.touch()
    window.open_path(str(path))
    job = window._job
    assert window.welcome.task_status.isVisible() and not window.isVisible()
    job.progress.emit(0.4, "读取工程内容")
    assert window.welcome.progress.value() == 400
    assert window.welcome.job_label.text() == "读取工程内容"
    if result == "failure":
        job.failed.emit("读取失败", "错误详情")
    else:
        window.welcome.cancel_task_button.click()
        assert job.cancel_calls == 1 and not window.welcome.cancel_task_button.isEnabled()
        job.cancelled.emit()
        job.succeeded.emit(existing_document(source))
    job.finish()
    app.processEvents()
    assert window.welcome.isVisible() and not window.isVisible()
    assert not window.welcome.task_status.isVisible()
    assert window.document is None and window._job is None


def test_wizard_task_progress_and_confirmation_parent(make_window, source, app, monkeypatch):
    window = make_window()
    install_existing(window, source, app)
    parents = []
    monkeypatch.setattr(QMessageBox, "question", lambda parent, *args:
                        parents.append(parent) or QMessageBox.StandardButton.Discard)
    window._new_project(str(source))
    window._create_project(options(source))
    assert parents == [window.wizard]
    assert window.wizard.task_status.isVisible()
    window._job.progress.emit(0.65, "解析音乐来源")
    assert window.wizard.progress.value() == 650
    assert window.wizard.job_label.text() == "解析音乐来源"
    window._job.failed.emit("来源读取失败", "错误详情")
    window._job.finish()
    assert not window.wizard.task_status.isVisible()
    assert window.wizard.isVisible() and window.wizard.error_label.text()


@pytest.mark.parametrize("origin", ["welcome", "editor"])
@pytest.mark.parametrize("discard", [False, True])
def test_closing_primary_window_coordinates_exit(
    make_window, source, app, monkeypatch, origin, discard
):
    window = make_window()
    state = install_existing(window, source, app)
    if origin == "welcome":
        window._show_welcome()
    window._new_project(str(source))
    primary = window.welcome if origin == "welcome" else window
    monkeypatch.setattr(QMessageBox, "question", lambda *args:
                        QMessageBox.StandardButton.Discard if discard else
                        QMessageBox.StandardButton.Cancel)
    primary.close()
    if discard:
        assert not any(widget.isVisible() for widget in (window, window.welcome, window.wizard))
        assert window.renderer is None and window._closing
    else:
        assert primary.isVisible() and window.wizard.isVisible()
        assert_original(window, state)
        assert not window._closing
def test_queued_preview_can_run_alongside_new_wizard(
    make_window, source, app, monkeypatch
):
    window = make_window()
    install_existing(window, source, app)
    gather_calls = []
    monkeypatch.setattr(
        window, "_gather_document", lambda *args, **kwargs: gather_calls.append(True) or True
    )
    monkeypatch.setattr(
        window, "compile_preview", MethodType(ORIGINAL_COMPILE_PREVIEW, window)
    )
    QTimer.singleShot(0, window.compile_preview)
    window._new_project(str(source))
    window._resume_editor()
    app.processEvents()
    assert window.wizard.isVisible()
    assert window._job.operation == "compile" and gather_calls == [True]
    assert not window.wizard.source_edit.isEnabled()
    assert not window.wizard.next_button.isEnabled()
    window._job.finish()
    assert window.wizard.source_edit.isEnabled()
    assert window.wizard.next_button.isEnabled()


def test_closing_during_import_stops_worker_and_ignores_late_results(
    make_window, source, app
):
    window = make_window()
    window._new_project(str(source))
    window._create_project(options(source))
    job = window._job
    assert job.running
    window.welcome.close()
    assert not job.running and window._job is None
    job.succeeded.emit(imported_project(source))
    window.compile_preview()
    window._show_welcome()
    app.processEvents()
    assert window.document is None
    assert not any(widget.isVisible() for widget in (window, window.welcome, window.wizard))


def test_demo_from_welcome_enters_editor_after_success(make_window, source, app):
    window = make_window()
    window.welcome.demo_button.click()
    assert window._job.operation == "demo"
    assert window.welcome.isVisible() and not window.isVisible()
    window._job.complete(existing_document(source))
    app.processEvents()
    assert window.isVisible() and not window.welcome.isVisible()


def test_cancel_wizard_during_preview_keeps_editor_job(make_window, source, app):
    window = make_window()
    install_existing(window, source, app)
    window._new_project(str(source))
    window._start_job("compile", window.document, lambda result: None)
    job = window._job
    window.wizard.cancel_button.click()
    assert not window.wizard.isVisible() and window.isVisible()
    assert window._job is job and job.cancel_calls == 0


def test_native_windows_wizard_ownership_and_stacking(make_window, source, app):
    if os.name != "nt" or app.platformName() != "windows":
        pytest.skip("requires the native Windows Qt platform")
    import ctypes
    from ctypes import wintypes

    from PySide6.QtWidgets import QWidget

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
    user32.GetWindow.restype = wintypes.HWND
    user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetWindowLongW.restype = ctypes.c_long

    def is_above(front, back):
        handle = user32.GetWindow(back, 3)  # GW_HWNDPREV: preceding window in Z order.
        while handle:
            if handle == front:
                return True
            handle = user32.GetWindow(handle, 3)
        return False

    window = make_window()
    install_existing(window, source, app)
    window._new_project(str(source))
    independent = QWidget()
    try:
        for parent in (window, window.welcome, window):
            if parent is window.welcome:
                window._show_welcome()
            else:
                window._resume_editor()
            parent.raise_()
            parent.activateWindow()
            app.processEvents()
            wizard_handle = int(window.wizard.winId())
            parent_handle = int(parent.winId())
            assert user32.GetWindow(wizard_handle, 4) == parent_handle  # GW_OWNER
            assert is_above(wizard_handle, parent_handle)
            assert not user32.GetWindowLongW(wizard_handle, -20) & 0x8  # WS_EX_TOPMOST
        independent.show()
        independent.raise_()
        independent.activateWindow()
        app.processEvents()
        assert is_above(int(independent.winId()), wizard_handle)
    finally:
        independent.close()


@pytest.mark.parametrize("extension", [".flp", ".mid", ".midi"])
def test_raw_sources_enter_wizard_without_importing_or_confirming(
    make_window, source, app, monkeypatch, extension
):
    window = make_window()
    state = install_existing(window, source, app)
    selected = source.with_suffix(extension)
    selected.touch()
    monkeypatch.setattr(window, "_confirm_replace", lambda: pytest.fail("confirmed before creation"))
    monkeypatch.setattr(gui.QFileDialog, "getOpenFileName", lambda *args: (str(selected), ""))
    window.open_action.trigger()
    assert window.wizard.isVisible()
    assert window.wizard.source_edit.text() == str(selected)
    assert window._job is None
    assert_original(window, state)


def test_startup_project_loads_directly_and_is_recorded_after_success(
    make_window, source, app, tmp_path
):
    path = tmp_path / "existing.stproj"
    document = existing_document(source)
    save_document(document, path)
    window = make_window(str(path))
    app.processEvents()
    job = window._job
    assert job.operation == "load" and job.payload == str(path)
    assert window.document is None
    assert recent_paths(window) == []
    job.complete(document)
    app.processEvents()
    assert window.isVisible() and not window.welcome.isVisible()
    assert window.project_path == str(path)
    assert not window._dirty
    assert recent_paths(window) == [str(path.resolve())]


@pytest.mark.parametrize("audio_selected", [False, True])
def test_create_applies_choices_before_first_preview_and_roundtrips(
    make_window, source, app, tmp_path, audio_selected
):
    window = make_window()
    audio = tmp_path / "原曲.flac" if audio_selected else ""
    if audio:
        audio.touch()
    window._new_project(str(source))
    window._create_project(options(source, audio))
    job = window._job
    assert job.operation == "import" and job.payload == (str(source), 0)
    assert window.document is None and window.test_previews == []
    job.succeeded.emit(imported_project(source))
    assert window.test_previews == []
    job.finish()
    app.processEvents()
    assert window.isVisible() and not window.welcome.isVisible()
    assert window.project_path == "" and window._dirty
    assert window.document.audio_path == str(audio)
    assert len(window.document.mappings) == 2
    for mapping in window.document.mappings:
        assert {name: getattr(mapping, name) for name in PROCESSING} == PROCESSING
    assert len(window.test_previews) == 1
    preview_document, preview_job = window.test_previews[0]
    assert preview_job is None and preview_document.audio_path == str(audio)
    assert {name: getattr(preview_document.mappings[1], name) for name in PROCESSING} == PROCESSING
    destination = tmp_path / "created.stproj"
    window.project_path = str(destination)
    assert window.save_project()
    restored = load_document(destination)
    assert restored.audio_path == str(audio)
    assert all({name: getattr(part, name) for name in PROCESSING} == PROCESSING for part in restored.mappings)


def test_unconfirmed_flp_timing_opens_source_page_without_auto_preview(make_window, source, app):
    window = make_window()
    window._new_project(str(source))
    window._create_project(options(source))
    window._job.complete(imported_project(source, timing_confirmed=False))
    app.processEvents()
    assert window.isVisible() and not window.welcome.isVisible()
    assert window.tabs.currentIndex() == 0
    assert not window.timing_confirmed.isChecked()
    assert window.test_previews == []
    assert window._dirty and window.project_path == ""


@pytest.mark.parametrize("answer", [QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save])
def test_cancelled_confirmation_or_failed_save_stops_creation(
    make_window, source, app, monkeypatch, answer
):
    window = make_window()
    state = install_existing(window, source, app)
    window._new_project(str(source))
    monkeypatch.setattr(QMessageBox, "question", lambda *args: answer)
    monkeypatch.setattr(window, "save_project", lambda: False)
    window._create_project(options(source))
    assert window.wizard.isVisible()
    assert window._job is None and window.test_previews == []
    assert_original(window, state)


def test_failed_import_keeps_previous_project_and_allows_retry(
    make_window, source, app
):
    window = make_window()
    state = install_existing(window, source, app)
    window._new_project(str(source))
    window.wizard.processing_checks["auto_staccato"].setChecked(False)
    selected = options(source)
    window._create_project(selected)
    first = window._job
    first.failed.emit("无效 MIDI", "解析失败")
    first.finish()
    app.processEvents()
    assert window.wizard.isVisible()
    assert window._job is None
    assert window.wizard.source_edit.text() == str(source)
    assert not window.wizard.processing_checks["auto_staccato"].isChecked()
    assert_original(window, state)
    window._create_project(selected)
    assert window._job is not None and window._job is not first
    window._job.complete(imported_project(source))
    app.processEvents()
    assert window.document is not state[0] and window.isVisible() and not window.welcome.isVisible()


def test_save_before_creation_keeps_saved_original_after_import_failure(
    make_window, source, app, monkeypatch
):
    window = make_window()
    document, saved_path, renderer = install_existing(window, source, app)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Save)
    window._new_project(str(source))
    window._create_project(options(source))
    assert window._job.operation == "import"
    assert load_document(saved_path).metadata.title == document.metadata.title
    assert recent_paths(window) == [str(Path(saved_path).resolve())]
    window._job.failed.emit("来源读取失败", "解析错误")
    window._job.finish()
    app.processEvents()
    assert window.document is document and window.renderer is renderer
    assert window.project_path == saved_path and not window._dirty
    assert window.wizard.isVisible()


@pytest.mark.parametrize("origin", ["welcome", "editor"])
def test_cancel_waits_for_cleanup_and_ignores_late_success(
    make_window, source, app, origin
):
    window = make_window()
    state = install_existing(window, source, app)
    if origin == "welcome":
        window._show_welcome()
    expected_window = window.welcome if origin == "welcome" else window
    window._new_project(str(source))
    window._create_project(options(source))
    job = window._job
    assert not window.open_action.isEnabled()
    window._cancel_wizard()
    window._cancel_wizard()
    assert job.cancel_calls == 1
    assert window.wizard.isVisible() and window._job is job
    job.cancelled.emit()
    assert window.wizard.isVisible()
    job.succeeded.emit(imported_project(source))
    assert_original(window, state)
    job.finish()
    app.processEvents()
    assert expected_window.isVisible() and not window.wizard.isVisible() and window._job is None
    assert_original(window, state)
    assert window.test_previews == []


def test_disappearing_source_prevents_import_without_replacing_document(
    make_window, source, app
):
    window = make_window()
    state = install_existing(window, source, app)
    window._new_project(str(source))
    source.unlink()
    window._create_project(options(source))
    assert window._job is None and window.wizard.error_label.text()
    assert window.wizard.isVisible()
    assert_original(window, state)


def recent_paths(window):
    return window.recent_projects.paths()


def test_recent_projects_keep_successes_only_and_missing_entry_is_safe(
    make_window, source, app, tmp_path, monkeypatch
):
    window = make_window()
    state = install_existing(window, source, app)
    destination = tmp_path / "saved.stproj"
    window.project_path = str(destination)
    assert window.save_project()
    assert recent_paths(window) == [str(destination.resolve())]
    missing = tmp_path / "missing.stproj"
    window.recent_projects.record(str(missing))
    old_state = (window.document, window.project_path, window.renderer)
    window._dirty = True
    window.open_path(str(missing))
    assert window._job is None and window.test_errors
    assert_original(window, old_state)
    assert str(missing.resolve()) in recent_paths(window)
    failed = tmp_path / "failed.stproj"
    failed.touch()
    window.open_path(str(failed))
    window._job.failed.emit("不能打开", "无效项目")
    window._job.finish()
    app.processEvents()
    assert str(failed.resolve()) not in recent_paths(window)
    monkeypatch.setattr(gui, "save_document", lambda *args: (_ for _ in ()).throw(OSError("只读")))
    window.project_path = str(tmp_path / "unsaved.stproj")
    assert not window.save_project()
    assert str(Path(window.project_path).resolve()) not in recent_paths(window)
    assert state[0] is window.document


def test_recent_projects_persist_deduplicate_and_limit_to_ten(make_window, settings, tmp_path):
    window = make_window()
    targets = [tmp_path / f"project-{index}.stproj" for index in range(12)]
    for target in targets:
        window.recent_projects.record(str(target))
    window.recent_projects.record(str(targets[5].parent / "." / targets[5].name))
    expected = [str(targets[5].resolve())] + [str(target.resolve()) for target in reversed(targets[2:]) if target != targets[5]]
    expected = expected[:10]
    assert recent_paths(window) == expected
    settings.sync()
    another = make_window()
    assert recent_paths(another) == expected


def test_real_midi_worker_imports_wizard_choices_before_preview(
    make_window, app, tmp_path, monkeypatch
):
    import mido

    from stavellum.background import BackgroundJob

    source = tmp_path / "真实来源.mid"
    midi = mido.MidiFile()
    for channel, name, pitch in ((0, "Violin", 72), (1, "Piano", 60)):
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=name))
        track.append(mido.Message("note_on", note=pitch, velocity=90, channel=channel))
        track.append(mido.Message("note_off", note=pitch, channel=channel, time=480))
        midi.tracks.append(track)
    midi.save(source)
    audio = tmp_path / "真实原曲.wav"
    with wave.open(str(audio), "wb") as file:
        file.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        file.writeframes(b"\x00\x00" * 8000)
    window = make_window()
    monkeypatch.setattr(gui, "BackgroundJob", BackgroundJob)
    assert window.welcome.isVisible() and not window.isVisible()
    window.open_path(str(source))
    assert window.wizard.isVisible()
    window._create_project(options(source, audio))
    deadline = time.monotonic() + 15
    while window._job is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert window._job is None, "the real MIDI importer did not complete"
    assert window.isVisible() and not window.welcome.isVisible(), window.wizard.error_label.text()
    assert window.document.audio_path == str(audio)
    assert len(window.document.mappings) == 2
    assert all({name: getattr(part, name) for name in PROCESSING} == PROCESSING for part in window.document.mappings)
    assert len(window.test_previews) == 1
    document, preview_job = window.test_previews[0]
    assert preview_job is None and document.audio_path == str(audio)
    assert all({name: getattr(part, name) for name in PROCESSING} == PROCESSING for part in document.mappings)
