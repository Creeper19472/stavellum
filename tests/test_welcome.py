"""The landing page owns navigation signals, never the opened document."""

from __future__ import annotations

import os

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication

from stavellum import __version__
from stavellum.welcome import RecentProjects, WelcomePage

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path):
    store = QSettings(str(tmp_path / "welcome.ini"), QSettings.Format.IniFormat)
    store.setFallbacksEnabled(False)
    return store


def test_recent_projects_persist_order_deduplicate_and_limit(settings, tmp_path):
    recent = RecentProjects(settings)
    for index in range(12):
        recent.record(str(tmp_path / f"piece-{index}.stproj"))
    recent.record(str(tmp_path / "piece-7.stproj"))
    expected = [7, 11, 10, 9, 8, 6, 5, 4, 3, 2]
    assert recent.paths() == [str(tmp_path / f"piece-{index}.stproj") for index in expected]
    restored = RecentProjects(QSettings(settings.fileName(), QSettings.Format.IniFormat))
    assert restored.paths() == recent.paths()
    recent.record(str(tmp_path / "unimported.mid"))
    assert restored.paths() == recent.paths()


def test_recent_projects_normalize_and_keep_missing_files(settings, tmp_path):
    missing = tmp_path / "missing.stproj"
    settings.setValue(RecentProjects.KEY, [str(missing), str(tmp_path / "." / missing.name)])
    recent = RecentProjects(settings)
    assert recent.paths() == [str(missing)]
    assert not missing.exists()
    recent.record(str(missing))
    assert recent.paths() == [str(missing)]
    if os.name == "nt":
        recent.record(str(missing).upper())
        assert len(recent.paths()) == 1


def test_recent_projects_accept_legacy_single_setting_and_invalid_data(settings, tmp_path):
    path = str(tmp_path / "single.stproj")
    recent = RecentProjects(settings)
    settings.setValue(RecentProjects.KEY, path)
    assert recent.paths() == [path]
    settings.setValue(RecentProjects.KEY, [42, "", "other.mid", path])
    assert recent.paths() == [path]


def test_welcome_actions_and_missing_recent_project(app, tmp_path):
    page = WelcomePage()
    page.show()
    app.processEvents()
    missing = str(tmp_path / "moved.stproj")
    project_spy = QSignalSpy(page.project_requested)
    new_spy = QSignalSpy(page.new_requested)
    open_spy = QSignalSpy(page.open_requested)
    resume_spy = QSignalSpy(page.resume_requested)
    assert page.empty_label.isVisible()
    assert not page.resume_button.isVisible()
    assert not page.selected_button.isEnabled()
    page.set_recent_projects([missing])
    assert "文件已移动或删除" in page.recent_list.item(0).text()
    assert page.recent_list.item(0).data(Qt.ItemDataRole.UserRole) == missing
    QTest.mouseClick(page.selected_button, Qt.MouseButton.LeftButton)
    assert project_spy.count() == 1
    assert project_spy.at(0) == [missing]
    # itemActivated handles both a double-click and keyboard Enter.
    page.recent_list.itemActivated.emit(page.recent_list.item(0))
    assert project_spy.count() == 2
    QTest.mouseClick(page.new_button, Qt.MouseButton.LeftButton)
    QTest.mouseClick(page.open_button, Qt.MouseButton.LeftButton)
    assert new_spy.count() == open_spy.count() == 1
    page.set_has_document(True)
    assert page.resume_button.isVisible()
    QTest.mouseClick(page.resume_button, Qt.MouseButton.LeftButton)
    assert resume_spy.count() == 1
    page.deleteLater()
    app.processEvents()


def test_demo_link_and_busy_guard_all_navigation(app, tmp_path):
    page = WelcomePage()
    page.set_recent_projects([str(tmp_path / "piece.stproj")])
    page.set_has_document(True)
    demo_spy = QSignalSpy(page.demo_requested)
    project_spy = QSignalSpy(page.project_requested)
    page.demo_button.click()
    assert demo_spy.count() == 1
    page.set_busy(True)
    for control in (page.selected_button, page.open_button, page.new_button,
                    page.resume_button, page.demo_button, page.recent_list):
        assert not control.isEnabled()
    page.demo_button.click()
    page.recent_list.itemActivated.emit(page.recent_list.item(0))
    page._activate_selection()
    assert demo_spy.count() == 1
    assert project_spy.count() == 0
    # The task bar lives on the same page and must stay interactive while a
    # background job runs, so only action controls are disabled.
    assert page.task_status.isEnabled() and page.cancel_task_button.isEnabled()
    page.set_busy(False)
    assert page.selected_button.isEnabled()
    assert page.resume_button.isEnabled()
    page.demo_button.click()
    assert demo_spy.count() == 2
    page.deleteLater()
    app.processEvents()


@pytest.mark.parametrize("width,height", [(900, 620), (1100, 740), (1280, 800), (1440, 900)])
def test_welcome_page_paints_dark_chrome_at_common_sizes(app, width, height):
    page = WelcomePage()
    page.resize(width, height)
    page.show()
    app.processEvents()
    assert page.size().width() == width
    assert page.size().height() == height
    assert page.new_button.geometry().right() < width
    image = page.grab().toImage()
    assert not image.isNull()
    assert image.pixelColor(2, 2).name().lower() == "#10161e"
    page.deleteLater()
    app.processEvents()


def test_start_page_carries_compact_name_and_version(app):
    page = WelcomePage()
    assert page.header.text() == "Stavellum"
    assert page.version_label.text() == f"v{__version__}"
    page.deleteLater()
    app.processEvents()


def test_all_recent_projects_are_reachable_in_small_window(app, tmp_path):
    page = WelcomePage()
    page.resize(900, 620)
    paths = [str(tmp_path / f"project-{number}.stproj") for number in range(10)]
    page.set_recent_projects(paths)
    page.show()
    app.processEvents()
    assert page.recent_list.verticalScrollBar().maximum() > 0
    page.recent_list.setFocus()
    QTest.keyClick(page.recent_list, Qt.Key.Key_End)
    app.processEvents()
    last = page.recent_list.item(9)
    assert page.recent_list.currentItem() is last
    assert page.recent_list.viewport().rect().contains(page.recent_list.visualItemRect(last).center())
    spy = QSignalSpy(page.project_requested)
    QTest.keyClick(page.recent_list, Qt.Key.Key_Return)
    assert spy.at(0) == [paths[-1]]
    page.deleteLater()
    app.processEvents()
