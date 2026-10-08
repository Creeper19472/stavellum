"""The startup cover hands navigation over once and respects early cancellation."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QWidget

from stavellum import gui, qt, startup
from stavellum.startup import StartupSplash

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def splash(app):
    widget = StartupSplash()
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_enter_before_ready_is_remembered_and_finishes_once(splash):
    finished = QSignalSpy(splash.finished)
    QTest.keyClick(splash, Qt.Key.Key_Return)
    assert finished.count() == 0 and splash.isVisible()
    visible_during_handoff = []
    splash.finished.connect(lambda: visible_during_handoff.append(splash.isVisible()))
    splash.mark_ready()
    splash.mark_ready()
    QTest.mouseClick(splash, Qt.MouseButton.LeftButton)
    assert finished.count() == 1
    assert visible_during_handoff == [True]
    assert not splash.isVisible() and not splash._timer.isActive()


def test_ready_automatically_hands_over_without_input(splash, monkeypatch):
    monkeypatch.setattr(splash, "HOLD_MS", 1)
    finished = QSignalSpy(splash.finished)
    splash.mark_ready()
    assert finished.wait(500)
    assert finished.count() == 1 and not splash.isVisible()


def test_click_skips_ready_cover_immediately(splash):
    finished = QSignalSpy(splash.finished)
    splash.mark_ready()
    QTest.mouseClick(splash, Qt.MouseButton.LeftButton)
    assert finished.count() == 1 and not splash._timer.isActive()


def test_space_skips_instead_of_activating_close_button(splash, app):
    finished = QSignalSpy(splash.finished)
    splash.mark_ready()
    splash.setFocus()
    app.processEvents()
    assert app.focusWidget() is splash
    QTest.keyClick(app.focusWidget(), Qt.Key.Key_Space)
    assert finished.count() == 1 and not splash.cancelled


@pytest.mark.parametrize("close_by", ["button", "escape"])
def test_close_cancels_pending_handoff(splash, close_by):
    finished = QSignalSpy(splash.finished)
    closed = QSignalSpy(splash.close_requested)
    splash.mark_ready()
    if close_by == "button":
        QTest.mouseClick(splash.close_button, Qt.MouseButton.LeftButton)
    else:
        QTest.keyClick(splash, Qt.Key.Key_Escape)
    splash.mark_ready()
    splash._finish()
    assert splash.cancelled and closed.count() == 1
    assert finished.count() == 0 and not splash._timer.isActive()


@pytest.mark.parametrize("cancel", [False, True])
def test_run_gui_shows_cover_before_initializing_and_defers_project_open(app, monkeypatch, cancel):
    calls = []
    covers = []
    windows = []
    state = SimpleNamespace(quit=False)

    class Cover(StartupSplash):
        HOLD_MS = 1

        def __init__(self):
            super().__init__()
            covers.append(self)

        def show(self):
            super().show()
            if cancel:
                self.close()

    class Workspace(QWidget):
        def __init__(self):
            super().__init__()
            assert covers[0].isVisible()
            calls.append("initialize")
            windows.append(self)

        def _show_welcome(self):
            calls.append("welcome")
            self.show()

        def open_path(self, path):
            assert self.isVisible()
            calls.append(("open", path))

    def execute():
        assert calls == []  # No expensive workspace construction before the event loop.
        for _ in range(100):
            QTest.qWait(5)
            if state.quit or any(isinstance(call, tuple) for call in calls):
                break
        assert state.quit if cancel else calls == ["initialize", "welcome", ("open", "song.stproj")]
        return 0

    proxy = SimpleNamespace(
        setApplicationName=lambda value: None,
        setWindowIcon=lambda value: None,
        setOrganizationName=lambda value: None,
        quit=lambda: setattr(state, "quit", True),
        exec=execute,
    )
    monkeypatch.setattr(qt, "prepare_render_app", lambda settings: proxy)
    monkeypatch.setattr(gui, "MainWindow", Workspace)
    monkeypatch.setattr(startup, "StartupSplash", Cover)
    try:
        assert gui.run_gui("song.stproj") == 0
        if cancel:
            assert calls == [] and not windows
    finally:
        for widget in windows:
            widget.close()
            widget.deleteLater()
        # A successful handoff already schedules the cover for deletion.
        if cancel:
            covers[0].deleteLater()
        app.processEvents()
