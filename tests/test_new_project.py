"""Wizard choices stay local until a valid, explicit create request."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from stavellum.exporting.audio import AUDIO_FILE_FILTER
from stavellum.ui import new_project
from stavellum.ui.new_project import NewProjectOptions, ProjectWizard

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def wizard(app):
    widget = ProjectWizard()
    requests = []
    widget.create_requested.connect(requests.append)
    widget.test_requests = requests
    yield widget
    widget.deleteLater()
    app.processEvents()


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "测试 原曲.MIDI"
    path.write_bytes(b"source content is parsed only after submission")
    return path


def _last_step(wizard, source):
    wizard.source_edit.setText(str(source))
    wizard.next_button.click()
    wizard.next_button.click()
    assert wizard.steps.currentIndex() == 2


def test_initial_state_and_existing_source_validation(wizard, source, tmp_path):
    assert wizard.steps.currentIndex() == 0
    assert not wizard.next_button.isEnabled()
    assert not wizard.back_button.isEnabled()
    assert {field: check.isChecked() for field, check in wizard.processing_checks.items()} == {
        field: field != "auto_ottava" for field in wizard.processing_checks
    }
    wizard.source_edit.setText(str(tmp_path / "missing.mid"))
    assert not wizard.next_button.isEnabled()
    unsupported = tmp_path / "music.txt"
    unsupported.write_text("data", encoding="utf-8")
    wizard.source_edit.setText(str(unsupported))
    assert not wizard.next_button.isEnabled()
    wizard.source_edit.setText(str(source))
    assert wizard.next_button.isEnabled()


def test_back_navigation_preserves_choices_and_skip_clears_audio(wizard, source, tmp_path):
    audio = tmp_path / "音频 原曲.mp3"
    audio.write_bytes(b"audio does not require a decode probe")
    wizard.reset(str(source))
    wizard.next_button.click()
    wizard.audio_edit.setText(str(audio))
    wizard.next_button.click()
    wizard.processing_checks["auto_ottava"].setChecked(True)
    wizard.back_button.click()
    assert wizard.audio_edit.text() == str(audio)
    wizard.back_button.click()
    assert wizard.source_edit.text() == str(source)
    wizard.next_button.click()
    wizard.skip_button.click()
    assert wizard.steps.currentIndex() == 2
    assert wizard.audio_edit.text() == ""
    assert wizard.processing_checks["auto_ottava"].isChecked()
    assert "跳过" in wizard.summary_label.text()
    wizard.create_button.click()
    assert wizard.test_requests[0].audio_path == ""


def test_create_snapshots_paths_and_processing_without_decoding(wizard, source, tmp_path):
    audio = tmp_path / "原曲.WAV"
    audio.write_bytes(b"intentionally not a valid WAV")
    _last_step(wizard, source)
    wizard.audio_edit.setText(str(audio))
    wizard.processing_checks["auto_simplify_accidentals"].setChecked(False)
    wizard.processing_checks["auto_grace"].setChecked(False)
    wizard.create_button.click()
    assert len(wizard.test_requests) == 1
    options = wizard.test_requests[0]
    assert isinstance(options, NewProjectOptions)
    assert options.source_path == str(source.resolve())
    assert options.audio_path == str(audio.resolve())
    assert options.processing == {
        "auto_simplify_accidentals": False,
        "auto_ottava": False,
        "auto_staccato": True,
        "auto_grace": False,
        "auto_dynamics": True,
    }
    wizard.processing_checks["auto_ottava"].setChecked(True)
    assert options.processing["auto_ottava"] is False


def test_reset_restores_defaults_and_removes_errors(wizard, source):
    _last_step(wizard, source)
    wizard.audio_edit.setText("old audio.wav")
    for control in wizard.processing_checks.values():
        control.setChecked(not control.isChecked())
    wizard.set_error("导入失败", "Traceback details")
    wizard.set_busy(True, cancelling=True)
    wizard.reset(str(source))
    assert wizard.steps.currentIndex() == 0
    assert wizard.source_edit.text() == str(source)
    assert wizard.audio_edit.text() == ""
    assert wizard.error_label.text() == ""
    assert wizard.error_details.toPlainText() == ""
    assert {field: check.isChecked() for field, check in wizard.processing_checks.items()} == {
        field: field != "auto_ottava" for field in wizard.processing_checks
    }
    assert wizard.source_edit.isEnabled()
    assert wizard.cancel_button.isEnabled()
    assert wizard.next_button.isEnabled()


@pytest.mark.parametrize("change", ["deleted", "directory", "unsupported"])
def test_revalidate_source_at_final_submission(wizard, source, tmp_path, change):
    _last_step(wizard, source)
    if change == "deleted":
        source.unlink()
    elif change == "directory":
        directory = tmp_path / "directory.mid"
        directory.mkdir()
        wizard.source_edit.setText(str(directory))
    else:
        unsupported = tmp_path / "renamed.txt"
        unsupported.write_bytes(b"data")
        wizard.source_edit.setText(str(unsupported))
    # Exercise submission itself even when a field edit already disabled the button.
    wizard._create()
    assert wizard.test_requests == []
    assert wizard.error_label.text()
    assert wizard.steps.currentIndex() == 2


def test_missing_audio_can_be_corrected_without_losing_choices(wizard, source, tmp_path):
    _last_step(wizard, source)
    wizard.audio_edit.setText(str(tmp_path / "missing.wav"))
    wizard.processing_checks["auto_dynamics"].setChecked(False)
    wizard.create_button.click()
    assert wizard.test_requests == []
    assert "原曲音频" in wizard.error_label.text()
    wizard.back_button.click()
    wizard.skip_button.click()
    wizard.create_button.click()
    assert len(wizard.test_requests) == 1
    assert wizard.test_requests[0].processing["auto_dynamics"] is False


def test_busy_prevents_duplicate_creation_but_allows_cancel(wizard, source):
    _last_step(wizard, source)
    cancellations = []
    wizard.cancel_requested.connect(lambda: cancellations.append(True))
    wizard.set_busy(True)
    assert not wizard.source_edit.isEnabled()
    assert not wizard.audio_edit.isEnabled()
    assert not wizard.back_button.isEnabled()
    assert not wizard.create_button.isEnabled()
    assert all(not check.isEnabled() for check in wizard.processing_checks.values())
    wizard._create()
    wizard._back()
    assert wizard.test_requests == []
    assert wizard.steps.currentIndex() == 2
    wizard.cancel_button.click()
    assert cancellations == [True]
    wizard.set_busy(True, cancelling=True)
    assert not wizard.cancel_button.isEnabled()
    wizard.cancel_button.click()
    assert cancellations == [True]
    wizard.set_busy(False)
    assert wizard.create_button.isEnabled()
    assert wizard.source_edit.isEnabled()


def test_summary_and_errors_treat_paths_and_markup_as_plain_text(wizard, source):
    _last_step(wizard, source)
    wizard.audio_edit.setText("音乐 <b>原曲 & 演奏.wav")
    assert wizard.summary_label.textFormat() == Qt.TextFormat.PlainText
    assert "<b>原曲 & 演奏.wav" in wizard.summary_label.text()
    wizard.set_error("无法导入 <a href='file'>来源</a>", "详细错误 <b>")
    assert wizard.error_label.textFormat() == Qt.TextFormat.PlainText
    wizard.error_details_button.click()
    assert wizard.error_details.toPlainText() == "详细错误 <b>"
    assert not wizard.error_details.isHidden()
    wizard.error_details_button.click()
    assert wizard.error_details.isHidden()


def test_pickers_use_source_and_existing_audio_filters(wizard, source, monkeypatch):
    calls = []

    def pick(parent, title, directory, file_filter):
        calls.append((title, file_filter))
        return str(source), ""

    monkeypatch.setattr(new_project.QFileDialog, "getOpenFileName", pick)
    wizard.source_browse_button.click()
    assert wizard.source_edit.text() == str(source)
    assert "*.flp *.mid *.midi" in calls[-1][1]
    wizard.next_button.click()
    wizard.audio_browse_button.click()
    assert calls[-1][1] == AUDIO_FILE_FILTER
    assert Path(wizard.audio_edit.text()) == source
