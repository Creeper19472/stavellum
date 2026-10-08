"""Workflow checks exercise persisted edits and real spawned process communication."""

from __future__ import annotations

import copy
import math
import os
import shutil
import time
import wave
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import QApplication, QComboBox, QGroupBox, QScrollArea

from stavellum.domain.mapping import activity_color
from stavellum.domain.models import (
    ANIMATION_DURATIONS,
    ANIMATION_PRESETS,
    Metadata,
    NoteEvent,
    PartMapping,
    ProjectDocument,
    ProjectIR,
    TrackInfo,
    load_document,
    save_document,
)
from stavellum.ui import gui as gui_module
from stavellum.ui.background import BackgroundJob
from stavellum.ui.gui import MainWindow

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def document():
    project = ProjectIR(
        "fixture.mid", "midi", "测试曲", tracks=[
            TrackInfo("violin", "Violin I"),
            TrackInfo("pizz", "Violin I Pizz"),
            TrackInfo("bell", "Bell"),
        ], notes=[
            NoteEvent("n1", "violin", 0, 480, 72),
            NoteEvent("n2", "pizz", 960, 480, 74),
            NoteEvent("n3", "bell", 1440, 480, 60),
        ], duration_ticks=4800,
    )
    return ProjectDocument(project, [
        PartMapping("v1", "Violin I", ["violin"], instrument="violin", icon="violin"),
        PartMapping("p1", "Pizz", ["pizz"], articulations={"pizz": "pizz."}, auto_ottava=True),
        PartMapping("b1", "Bell", ["bell"], instrument="bell", auto_ottava=True),
    ], metadata=Metadata(title="测试曲"))


def embedded_test_icon(window):
    from stavellum.graphics.icons import make_icon_asset

    reference, asset = make_icon_asset(
        b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><rect width="10" height="10" fill="red"/></svg>',
        "red.svg", "image/svg+xml",
    )
    window.document.icon_assets[reference[6:]] = asset
    window._set_icon(reference)
    assert window.apply_part()
    return reference, asset


def test_explicit_icons_survive_instrument_change_and_visibility_toggle(window, tmp_path):
    reference, _ = embedded_test_icon(window)
    assert window.use_icon.isChecked()
    assert window.use_icon.text() == "使用乐器图标"
    window.instrument.setCurrentIndex(window.instrument.findData("piano"))
    assert window.icon.text() == reference
    assert window.apply_part()
    assert window.document.mappings[0].icon == reference
    window.use_icon.setChecked(False)
    assert not window.choose_icon_button.isEnabled()
    assert not window.auto_icon_button.isEnabled()
    window.instrument.setCurrentIndex(window.instrument.findData("cello"))
    assert window.apply_part()
    assert not window.document.mappings[0].use_icon
    assert window.document.mappings[0].icon == reference
    window.project_path = str(tmp_path / "hidden-icon.stproj")
    assert window.save_project()
    window.set_document(load_document(window.project_path))
    assert not window.use_icon.isChecked()
    assert window.icon.text() == reference
    window.use_icon.setChecked(True)
    assert window.choose_icon_button.isEnabled()
    assert window.auto_icon_button.isEnabled()
    assert window.apply_part()
    assert window.document.mappings[0].icon == reference
    window.auto_icon_button.click()
    assert window.icon.text() == ""
    assert window.apply_part()
    assert window.document.mappings[0].icon == ""


def test_loading_project_refreshes_embedded_icon_name_and_thumbnail(window):
    reference, _ = embedded_test_icon(window)
    window.auto_icon_button.click()  # Leave the old text field visible before loading.
    assert not window.icon.isHidden()
    window.set_document(window.document)
    assert window.icon.text() == reference
    assert window.icon.isHidden() and not window.icon_name.isHidden()
    assert window.icon_name.text() == "red.svg"
    assert window.icon_preview.pixmap().toImage().pixelColor(24, 24).red() == 255
    assert not window._dirty


def test_invalid_fontawesome_input_does_not_replace_previous_mapping(window):
    reference, _ = embedded_test_icon(window)
    before = copy.deepcopy(window.document.mappings[0])
    window.icon.setText("fa:solid:missing-icon")
    assert not window.apply_part()
    assert window.document.mappings[0] == before
    assert window.document.mappings[0].icon == reference
    assert "Font Awesome" in window.test_errors[-1]


def test_embedded_icon_survives_merge_split_and_reimport(window):
    reference, asset = embedded_test_icon(window)
    window.use_icon.setChecked(False)
    assert window.apply_part()
    window.parts.item(0).setSelected(True)
    window.parts.item(1).setSelected(True)
    window._merge_parts()
    assert window.document.mappings[0].icon == reference
    assert not window.document.mappings[0].use_icon
    window._split_part()
    assert all(mapping.icon == reference for mapping in window.document.mappings[:2])
    assert all(not mapping.use_icon for mapping in window.document.mappings[:2])
    source = copy.deepcopy(window.document.project)
    window._job = SimpleNamespace(reimporting=True)
    try:
        window._source_imported(source)
    finally:
        window._job = None
    assert window.document.icon_assets == {reference[6:]: asset}
    assert all(mapping.icon == reference for mapping in window.document.mappings[:2])
    assert all(not mapping.use_icon for mapping in window.document.mappings[:2])


@pytest.mark.parametrize("accepted", [True, False])
def test_icon_picker_only_changes_project_on_accept(window, monkeypatch, accepted):
    from stavellum.graphics.icons import make_icon_asset
    from stavellum.ui import icon_picker

    selection = make_icon_asset(
        b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><circle cx="5" cy="5" r="5"/></svg>',
        "round.svg", "image/svg+xml",
    )
    before = copy.deepcopy(window.document.to_dict())
    previous_text = window.icon.text()
    dialog = SimpleNamespace(selection=selection, DialogCode=SimpleNamespace(Accepted=1),
                             exec=lambda: int(accepted), deleteLater=lambda: None)
    monkeypatch.setattr(icon_picker, "IconPicker", lambda *args: dialog)
    window._choose_icon()
    if accepted:
        assert window.icon.text() == selection[0]
        assert window.document.icon_assets == {selection[0][6:]: selection[1]}
        assert window._dirty
    else:
        assert window.icon.text() == previous_text
        assert window.document.to_dict() == before


@pytest.fixture
def window(app, document, monkeypatch, tmp_path):
    widget = MainWindow(settings=QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat))
    monkeypatch.setattr(widget, "compile_preview", lambda *args: None)
    errors = []
    monkeypatch.setattr(widget, "_show_error", errors.append)
    widget.test_errors = errors
    widget.set_document(document)
    app.processEvents()
    yield widget
    widget._dirty = False
    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_file_and_export_menus_follow_document_and_worker_state(window):
    file_menu, export_menu, preview_menu = [action.menu() for action in window.menuBar().actions()]
    assert (file_menu.title(), export_menu.title(), preview_menu.title()) == ("文件", "导出", "预览")
    assert [action for action in file_menu.actions() if not action.isSeparator()][:-1] == [
        window.new_action, window.open_action, window.demo_action, window.audio_action,
        window.save_action, window.save_as_action, window.welcome_action,
    ]
    assert export_menu.actions() == [window.export_video_action, window.export_parts_action]
    assert preview_menu.actions() == [window.rebuild_action, window.clear_cache_action]
    assert window.new_action.shortcut().toString() == "Ctrl+N"
    assert window.open_action.shortcut().toString() == "Ctrl+O"
    assert window.save_action.shortcut().toString() == "Ctrl+S"
    document_actions = (
        window.audio_action, window.save_action, window.save_as_action,
        window.export_video_action, window.export_parts_action, window.compile_button, window.rebuild_action,
    )
    opening_actions = (window.new_action, window.open_action, window.demo_action, window.welcome_action)
    assert all(action.isEnabled() for action in (*document_actions, *opening_actions))
    window._job = SimpleNamespace(_cancel_requested_at=None)
    window._update_actions()
    assert not window.clear_cache_action.isEnabled()
    assert all(not action.isEnabled() for action in (*document_actions, *opening_actions))
    window._job._cancel_requested_at = 1
    window._update_actions()
    assert all(not action.isEnabled() for action in (*document_actions, *opening_actions))
    window._job = None
    window._update_actions()
    assert window.clear_cache_action.isEnabled()
    assert all(action.isEnabled() for action in (*document_actions, *opening_actions))
    window.document = None
    window._update_actions()
    assert all(not action.isEnabled() for action in document_actions)
    assert all(action.isEnabled() for action in opening_actions)


@pytest.mark.parametrize("size,sidebar_width", [((1480, 920), 470), ((1000, 700), 420)])
def test_grouped_tabs_remain_accessible_in_narrow_sidebar(window, app, size, sidebar_width):
    from stavellum.graphics.qt import ensure_app

    ensure_app()
    window.resize(*size)
    window.show()
    window.tabs.parentWidget().setSizes([sidebar_width, size[0] - sidebar_width])
    expected_counts = [3, 5, 7]
    for index, count in enumerate(expected_counts):
        window.tabs.setCurrentIndex(index)
        app.processEvents()
        tab = window.tabs.widget(index)
        groups = tab.findChildren(QGroupBox)
        assert len(groups) == count
        assert len({group.parentWidget() for group in groups}) == 1
        scroll = tab if isinstance(tab, QScrollArea) else tab.findChild(QScrollArea)
        assert scroll.horizontalScrollBar().maximum() == 0
        for group in groups:
            scroll.ensureWidgetVisible(group)
            app.processEvents()
            assert group.isVisible()
        for control in scroll.widget().findChildren(QComboBox):
            scroll.ensureWidgetVisible(control)
            app.processEvents()
            assert control.width() > 50
            assert scroll.viewport().rect().contains(control.mapTo(scroll.viewport(), control.rect().center()))


def test_merge_split_and_order_preserve_notes_and_articulation(window):
    window.document.project.tracks[0].color = "#4080c0"
    window.document.project.tracks[1].color = "#c04080"
    original_notes = [(note.note_id, note.track_id, note.pitch, note.start_tick, note.duration_tick)
                      for note in window.document.project.notes]
    window.parts.item(0).setSelected(True)
    window.parts.item(1).setSelected(True)
    window._merge_parts()
    assert len(window.document.mappings) == 2
    assert window.document.mappings[0].track_ids == ["violin", "pizz"]
    assert window.document.mappings[0].articulations == {"pizz": "pizz."}
    assert activity_color(window.document.project, window.document.mappings[0]) == "#4080c0"
    window._split_part()
    assert [m.track_ids for m in window.document.mappings] == [["violin"], ["pizz"], ["bell"]]
    assert window.document.mappings[1].articulations == {"pizz": "pizz."}
    assert [activity_color(window.document.project, mapping) for mapping in window.document.mappings] == [
        "#4080c0", "#c04080", "#ffffff",
    ]
    window._move_part(1)
    assert [m.track_ids for m in window.document.mappings] == [["pizz"], ["violin"], ["bell"]]
    window.document.validate()
    resulting_notes = [(n.note_id, n.track_id, n.pitch, n.start_tick, n.duration_tick)
                       for n in window.document.project.notes]
    assert resulting_notes == original_notes


def test_track_reassignment_and_settings_roundtrip(window, tmp_path):
    window.part_tracks.item(1).setCheckState(Qt.CheckState.Checked)
    window.part_name.setText("Violin I — arco / pizz.")
    window.use_icon.setChecked(True)
    window.keyswitches.setText("24, 25")
    window.articulations.setPlainText('{"violin": "arco", "pizz": "pizz."}')
    window.offset.setValue(1.25)
    window.bpm.setValue(96)
    window.settings_controls["width"].setValue(1280)
    window.settings_controls["height"].setValue(720)
    window.metadata_controls["composer"].setText("程悦")
    target = tmp_path / "composition.stproj"
    window.project_path = str(target)
    assert window.save_project()
    restored = load_document(target)
    assert restored.mappings[0].track_ids == ["violin", "pizz"]
    assert restored.mappings[0].use_icon
    assert restored.mappings[0].keyswitches == [24, 25]
    assert not restored.mappings[1].enabled
    assert restored.mappings[1].track_ids == []
    assert restored.settings.score_start_in_audio_sec == 1.25
    assert (restored.settings.width, restored.settings.height) == (1280, 720)
    assert restored.project.bpm == 96
    assert restored.metadata.composer == "程悦"
    restored.validate()
    assert not window._dirty


def test_render_and_inference_options_roundtrip_and_keep_encoder_quality_independent(window, tmp_path):
    assert window.auto_simplify_accidentals.isChecked()
    assert not window.auto_ottava.isChecked()
    assert window.auto_ottava.text() == "自动八度移位"
    assert "实际演奏音高" in window.auto_ottava.toolTip()
    assert not window._dirty
    window.auto_ottava.setChecked(True)
    assert window._dirty
    window.auto_ottava.setChecked(False)
    window.auto_simplify_accidentals.setChecked(False)
    assert window._dirty
    assert window.auto_staccato.isChecked() and window.auto_grace.isChecked()
    assert window.auto_dynamics.isChecked()
    assert window.render_backend.currentData() == "auto"
    assert "RHI Vulkan" in window.render_backend.currentText()
    assert window.video_encoder.currentData() == "auto"
    window.auto_staccato.setChecked(False)
    window.auto_grace.setChecked(False)
    window.auto_dynamics.setChecked(False)
    window.render_backend.setCurrentIndex(window.render_backend.findData("gpu"))
    assert "RHI Vulkan" in window.render_backend.currentText()
    window.video_encoder.setCurrentIndex(window.video_encoder.findData("h264_nvenc"))
    window.settings_controls["crf"].setValue(23)
    window.preset.setCurrentText("fast")
    window.settings_controls["nvenc_cq"].setValue(19)
    window.nvenc_preset.setCurrentText("p7")
    assert window._dirty
    assert not window.settings_controls["crf"].isEnabled()
    assert not window.preset.isEnabled()
    assert window.settings_controls["nvenc_cq"].isEnabled()
    assert window.nvenc_preset.isEnabled()
    window.project_path = str(tmp_path / "render-options.stproj")
    assert window.save_project()
    restored = load_document(window.project_path)
    assert restored.settings.render_backend == "gpu"
    assert restored.settings.video_encoder == "h264_nvenc"
    assert (restored.settings.crf, restored.settings.preset) == (23, "fast")
    assert (restored.settings.nvenc_cq, restored.settings.nvenc_preset) == (19, "p7")
    assert not restored.mappings[0].auto_staccato and not restored.mappings[0].auto_grace
    assert not restored.mappings[0].auto_dynamics
    assert not restored.mappings[0].auto_simplify_accidentals
    assert not restored.mappings[0].auto_ottava
    window.set_document(restored)
    assert window.render_backend.currentData() == "gpu"
    assert window.nvenc_preset.currentText() == "p7"
    assert not window.auto_staccato.isChecked() and not window.auto_grace.isChecked()
    assert not window.auto_dynamics.isChecked()
    assert not window.auto_simplify_accidentals.isChecked()
    assert not window.auto_ottava.isChecked()
    assert not window._dirty
    window.video_encoder.setCurrentIndex(window.video_encoder.findData("libx264"))
    assert window.settings_controls["crf"].isEnabled()
    assert not window.settings_controls["nvenc_cq"].isEnabled()
    window.video_encoder.setCurrentIndex(window.video_encoder.findData("auto"))
    assert window.settings_controls["crf"].isEnabled()
    assert window.settings_controls["nvenc_cq"].isEnabled()
    assert window._gather_document()
    assert (window.document.settings.crf, window.document.settings.nvenc_cq) == (23, 19)


def test_inference_choices_follow_part_switch_merge_split_and_reimport(window):
    original_notes = copy.deepcopy(window.document.project.notes)
    window.auto_simplify_accidentals.setChecked(False)
    window.auto_ottava.setChecked(False)
    window.auto_staccato.setChecked(False)
    window.auto_dynamics.setChecked(False)
    assert window.apply_part()
    window.parts.setCurrentRow(1)
    assert window.auto_simplify_accidentals.isChecked()
    assert window.auto_ottava.isChecked()
    window.auto_grace.setChecked(False)
    assert window.apply_part()
    window.parts.setCurrentRow(0)
    assert not window.auto_simplify_accidentals.isChecked()
    assert not window.auto_ottava.isChecked()
    assert not window.auto_staccato.isChecked() and window.auto_grace.isChecked()
    window.parts.clearSelection()
    window.parts.item(0).setSelected(True)
    window.parts.item(1).setSelected(True)
    window._merge_parts()
    assert not window.document.mappings[0].auto_staccato
    assert window.document.mappings[0].auto_grace
    assert not window.document.mappings[0].auto_simplify_accidentals
    assert not window.document.mappings[0].auto_ottava
    window._split_part()
    assert all(not m.auto_staccato and m.auto_grace for m in window.document.mappings[:2])
    assert all(not m.auto_dynamics for m in window.document.mappings[:2])
    assert all(not m.auto_simplify_accidentals for m in window.document.mappings[:2])
    assert all(not m.auto_ottava for m in window.document.mappings[:2])
    window.document.settings.render_backend = "cpu"
    window.document.settings.video_encoder = "libx264"
    window.document.settings.nvenc_cq = 21
    window._job = SimpleNamespace(reimporting=True)
    try:
        window._source_imported(copy.deepcopy(window.document.project))
    finally:
        window._job = None
    assert all(not m.auto_staccato and m.auto_grace for m in window.document.mappings[:2])
    assert window.document.settings.render_backend == "cpu"
    assert window.document.settings.video_encoder == "libx264"
    assert window.document.settings.nvenc_cq == 21
    assert all(not m.auto_dynamics for m in window.document.mappings[:2])
    assert all(not m.auto_simplify_accidentals for m in window.document.mappings[:2])
    assert all(not m.auto_ottava for m in window.document.mappings[:2])
    assert window.document.mappings[2].auto_simplify_accidentals
    assert window.document.mappings[2].auto_ottava
    assert window.document.project.notes == original_notes


def test_reimport_enables_simplification_for_new_parts_and_preserves_existing_choice(window):
    window.auto_simplify_accidentals.setChecked(False)
    window.auto_ottava.setChecked(False)
    assert window.apply_part()
    project = copy.deepcopy(window.document.project)
    project.tracks.append(TrackInfo("flute", "Flute"))
    project.notes.append(NoteEvent("flute-note", "flute", 0, 480, 72))
    window._job = SimpleNamespace(reimporting=True)
    try:
        window._source_imported(project)
    finally:
        window._job = None
    by_track = {track: mapping for mapping in window.document.mappings for track in mapping.track_ids}
    assert not by_track["violin"].auto_simplify_accidentals
    assert not by_track["violin"].auto_ottava
    assert by_track["flute"].auto_simplify_accidentals
    assert not by_track["flute"].auto_ottava


def test_renderer_resources_close_on_replace_project_switch_failure_and_exit(window, monkeypatch):
    from stavellum.rendering import render

    class Renderer:
        render_backend = "cpu"
        fallback_reasons = ["测试 GPU 不可用"]

        def __init__(self, scene):
            self.scene = scene
            self.closed = 0
            self.fail = False

        def render_frame(self, seconds):
            if self.fail:
                raise RuntimeError("测试绘制失败")
            image = QImage(320, 240, QImage.Format.Format_RGBA8888)
            image.fill(Qt.GlobalColor.black)
            return image

        def close(self):
            self.closed += 1

    monkeypatch.setattr(render, "FrameRenderer", Renderer)
    scene = SimpleNamespace(settings=copy.deepcopy(window.document.settings),
                            score_duration=1.0, diagnostics=[])
    old = Renderer(scene)
    window.renderer = old
    window._compiled(scene)
    first = window.renderer
    assert old.closed == 1
    assert "CPU" in window.preview_status.text() and "测试 GPU 不可用" in window.preview_status.text()
    first.render_backend = "gpu"
    first.fallback_reasons = []
    window._show_renderer_status(force=True)
    assert "RHI Vulkan" in window.preview_status.text()
    window._mark_dirty()
    window._render_position()
    assert "设置已修改" in window.preview_status.text()
    window.set_document(window.document)
    assert first.closed == 1
    window._compiled(scene)
    failed = window.renderer
    failed.fail = True
    window._render_position()
    assert failed.closed == 1 and window.renderer is None
    assert "测试绘制失败" in window.preview_status.text()
    window._compiled(scene)
    final = window.renderer
    window._dirty = False
    window.close()
    assert final.closed == 1 and window.renderer is None


def test_unchanged_preview_keeps_renderer_and_saved_edits_still_update(window, tmp_path, monkeypatch):
    from stavellum.presentation.compilation_cache import preview_key
    from stavellum.presentation.scene import compile_scene

    window.render_backend.setCurrentIndex(window.render_backend.findData("cpu"))
    assert window._gather_document()
    window._pending_preview_key = preview_key(window.document)
    window._compiled(compile_scene(window.document))
    renderer, compiled = window.renderer, window._scene
    requests = []
    monkeypatch.setattr(window, "_start_job", lambda *args: requests.append(args))
    MainWindow.compile_preview(window)
    assert not requests
    assert window.renderer is renderer and window._scene is compiled
    window.metadata_controls["title"].setText("已保存的新标题")
    window.project_path = str(tmp_path / "updated.stproj")
    assert window.save_project() and not window._dirty
    MainWindow.compile_preview(window)
    assert len(requests) == 1
    assert requests[0][1].metadata.title == "已保存的新标题"
    assert window.renderer is renderer
    MainWindow.compile_preview(window, force_rebuild=True)
    assert requests[-1][1][1] == {"force_rebuild": True}


@pytest.mark.parametrize("failure", ["initialization", "first-frame"])
def test_replacement_failure_keeps_previous_preview(window, monkeypatch, failure):
    from stavellum.rendering import render

    image = QImage(320, 240, QImage.Format.Format_RGBA8888)
    image.fill(Qt.GlobalColor.black)
    old = SimpleNamespace(closed=0)
    old.close = lambda: setattr(old, "closed", old.closed + 1)
    window.renderer = old
    window._scene = "previous-scene"
    window._applied_preview_key = "previous-key"
    window.preview.set_frame(image)
    candidates = []

    class FailedRenderer:
        def __init__(self, scene):
            if failure == "initialization":
                raise RuntimeError("initialization")
            self.closed = 0
            candidates.append(self)

        def render_frame(self, time):
            raise RuntimeError("first-frame")

        def close(self):
            self.closed += 1

    monkeypatch.setattr(render, "FrameRenderer", FailedRenderer)
    incoming = SimpleNamespace(settings=window.document.settings, score_duration=1)
    with pytest.raises(RuntimeError, match=failure):
        window._compiled(incoming)
    assert window.renderer is old and old.closed == 0
    assert window._scene == "previous-scene" and window._applied_preview_key == "previous-key"
    assert all(candidate.closed == 1 for candidate in candidates)


def test_gui_prepares_vulkan_platform_before_constructing_the_window(monkeypatch):
    from stavellum.graphics import qt

    events = []
    application = SimpleNamespace(
        setApplicationName=lambda name: None, setOrganizationName=lambda name: None,
        setWindowIcon=lambda icon: None,
        exec=lambda: 0,
    )

    def prepare(settings):
        assert settings.render_backend == "auto"
        events.append("prepared")
        return application

    class Window:
        def __init__(self, project_path):
            assert events == ["prepared"]
            assert project_path == "existing.stproj"
            events.append("created")

        def _show_welcome(self):
            events.append("shown")

    monkeypatch.setattr(qt, "prepare_render_app", prepare)
    monkeypatch.setattr(gui_module, "MainWindow", Window)
    assert gui_module.run_gui("existing.stproj") == 0
    assert events == ["prepared", "created", "shown"]


@pytest.mark.parametrize("name", ["fast", "medium", "slow", "very_slow"])
def test_animation_presets_update_only_transition_durations(window, name):
    window.settings_controls["intro_delay_seconds"].setValue(2.5)
    window.settings_controls["announcement_hold_seconds"].setValue(8)
    window.settings_controls["animation_stable_seconds"].setValue(4.125)
    window.animation_preset.setCurrentIndex(window.animation_preset.findData(name))
    assert tuple(window.settings_controls[field].value() for field in ANIMATION_DURATIONS
                 ) == pytest.approx(ANIMATION_PRESETS[name])
    assert window.settings_controls["intro_delay_seconds"].value() == 2.5
    assert window.settings_controls["announcement_hold_seconds"].value() == 8
    assert window.settings_controls["animation_stable_seconds"].value() == 4.125
    assert window.bpm.value() == window.document.project.bpm
    assert window._gather_document()
    assert window.document.settings.animation_preset() == name
    window.settings_controls["overlay_enter_seconds"].setValue(1.234)
    assert window.animation_preset.currentData() == "custom"
    before = tuple(window.settings_controls[field].value() for field in ANIMATION_DURATIONS)
    window.animation_preset.setCurrentIndex(window.animation_preset.findData("custom"))
    assert tuple(window.settings_controls[field].value() for field in ANIMATION_DURATIONS) == before


@pytest.mark.parametrize("stable_seconds", [0.0, 4.125])
def test_animation_stable_time_control_persists_without_changing_speed_preset(window, tmp_path, stable_seconds):
    control = window.settings_controls["animation_stable_seconds"]
    assert (control.minimum(), control.maximum(), control.singleStep(), control.decimals()) == (0, 60, 0.1, 3)
    assert control.value() == 2.0
    assert "0 关闭" in control.toolTip()
    before = tuple(window.settings_controls[field].value() for field in ANIMATION_DURATIONS)
    control.setValue(stable_seconds)
    assert window._dirty
    assert window.animation_preset.currentData() == "medium"
    assert tuple(window.settings_controls[field].value() for field in ANIMATION_DURATIONS) == before
    window.project_path = str(tmp_path / "animation-stable.stproj")
    assert window.save_project()
    restored = load_document(window.project_path)
    assert restored.settings.animation_stable_seconds == stable_seconds
    window.set_document(restored)
    assert control.value() == stable_seconds
    assert window._gather_document()
    assert window.document.settings.animation_stable_seconds == stable_seconds


def test_loading_custom_animation_durations_does_not_apply_a_preset(window):
    settings = window.document.settings
    settings.enter_seconds = 1.234
    settings.overlay_exit_seconds = 2.345
    before = copy.deepcopy(settings)
    window.set_document(window.document)
    assert window.animation_preset.currentData() == "custom"
    assert tuple(window.settings_controls[field].value() for field in ANIMATION_DURATIONS
                 ) == tuple(getattr(before, field) for field in ANIMATION_DURATIONS)
    assert window._gather_document()
    assert window.document.settings == before


def test_intro_and_overlay_settings_persist_and_enable_relevant_hold_fields(window, tmp_path):
    assert not window.settings_controls["announcement_hold_seconds"].isEnabled()
    assert "tempo_hold_seconds" not in window.settings_controls
    assert not hasattr(window, "tempo_display_mode")
    window.announcement_auto_hide.setChecked(True)
    assert window.settings_controls["announcement_hold_seconds"].isEnabled()
    window.settings_controls["intro_delay_seconds"].setValue(1.25)
    window.settings_controls["announcement_hold_seconds"].setValue(6.5)
    window.animation_preset.setCurrentIndex(window.animation_preset.findData("slow"))
    window.project_path = str(tmp_path / "intro.stproj")
    assert window.save_project()
    restored = load_document(window.project_path)
    assert restored.settings.intro_delay_seconds == 1.25
    assert restored.settings.announcement_auto_hide
    assert restored.settings.announcement_hold_seconds == 6.5
    assert restored.settings.animation_preset() == "slow"
    window.set_document(restored)
    assert window.animation_preset.currentData() == "slow"
    assert window.announcement_auto_hide.isChecked()


def test_logo_controls_modes_presets_and_persistence(window, tmp_path):
    controls = window.settings_controls
    names = ("logo_size_ratio", "logo_opacity", "logo_enter_seconds",
             "logo_hold_seconds", "logo_exit_seconds")
    assert not window.logo_enabled.isChecked()
    assert not window.logo_display_mode.isEnabled()
    assert all(not controls[name].isEnabled() for name in names)
    window.logo_enabled.setChecked(True)
    assert window.logo_display_mode.isEnabled()
    assert controls["logo_size_ratio"].isEnabled() and controls["logo_opacity"].isEnabled()
    assert all(not controls[name].isEnabled() for name in names[2:])
    window.logo_display_mode.setCurrentIndex(window.logo_display_mode.findData("fade_in"))
    assert controls["logo_enter_seconds"].isEnabled()
    assert not controls["logo_hold_seconds"].isEnabled()
    assert not controls["logo_exit_seconds"].isEnabled()
    window.logo_display_mode.setCurrentIndex(window.logo_display_mode.findData("intro"))
    assert all(controls[name].isEnabled() for name in names)
    values = (0.12, 0.65, 0.5, 0.0, 0.4)
    for name, value in zip(names, values, strict=True):
        controls[name].setValue(value)
    window.animation_preset.setCurrentIndex(window.animation_preset.findData("slow"))
    assert tuple(controls[name].value() for name in names) == values
    assert "设置已修改" in window.preview_status.text()
    window.project_path = str(tmp_path / "logo.stproj")
    assert window.save_project()
    restored = load_document(window.project_path)
    assert restored.settings.logo_enabled
    assert restored.settings.logo_display_mode == "intro"
    assert tuple(getattr(restored.settings, name) for name in names) == values
    window.set_document(restored)
    assert window.logo_enabled.isChecked()
    assert window.logo_display_mode.currentData() == "intro"
    assert all(controls[name].isEnabled() for name in names)
    assert tuple(controls[name].value() for name in names) == values
    assert not window._dirty
    controls["logo_enter_seconds"].setValue(0.75)
    assert window.animation_preset.currentData() == "slow"
    window.logo_enabled.setChecked(False)
    assert all(not controls[name].isEnabled() for name in names)


def test_invalid_mapping_does_not_modify_document(window):
    original = window.document.to_dict()
    window.part_name.setText("Changed")
    window.percussion_map.setPlainText("[1, 2, 3]")
    assert not window.apply_part()
    assert window.document.to_dict() == original
    assert window.test_errors


def test_explicit_midi_channel_split_preserves_source_timing(window):
    window.document.project.tracks[0].color = "#4080c0"
    window.document.project.notes.append(NoteEvent("n4", "violin", 2400, 960, 79, midi_channel=3))
    original = [(n.note_id, n.start_tick, n.duration_tick, n.pitch)
                for n in window.document.project.notes]
    window.articulations.setPlainText('{"violin": "arco"}')
    window.auto_simplify_accidentals.setChecked(False)
    window.part_tracks.setCurrentRow(0)
    window._split_midi_channels()
    assert window.document.mappings[0].track_ids == ["violin-midi-0"]
    assert window.document.mappings[1].track_ids == ["violin-midi-3"]
    assert window.document.mappings[1].articulations == {"violin-midi-3": "arco"}
    assert all(not mapping.auto_simplify_accidentals for mapping in window.document.mappings[:2])
    assert [activity_color(window.document.project, mapping) for mapping in window.document.mappings[:2]] == [
        "#4080c0", "#4080c0",
    ]
    assert [(n.note_id, n.start_tick, n.duration_tick, n.pitch)
            for n in window.document.project.notes] == original
    window.document.validate()


def test_seek_requests_absolute_frame_time(window):
    class Renderer:
        def __init__(self):
            self.times = []
            self.scene = SimpleNamespace(score_duration=window.document.project.duration_seconds,
                                         settings=window.document.settings)

        def render_frame(self, seconds):
            self.times.append(seconds)
            frame = QImage(320, 240, QImage.Format.Format_RGB32)
            frame.fill(QColor("black"))
            return frame

    renderer = Renderer()
    window.renderer = renderer
    window.seek_to_milliseconds(2000)
    window.seek_to_milliseconds(500)
    window.seek_to_milliseconds(2000)
    assert renderer.times == [2.0, 0.5, 2.0]
    assert not window.preview.frame.isNull()
    assert window.seek.value() == 2000


class _ClockPlayer:
    """A controlled media clock, without relying on platform callback cadence."""

    def __init__(self, *, silent=False):
        self.silent = silent
        self.position_ms = 0
        self.duration_ms = 10000
        self.status = QMediaPlayer.MediaStatus.BufferedMedia
        self.state = QMediaPlayer.PlaybackState.StoppedState

    def source(self):
        return SimpleNamespace(isEmpty=lambda: self.silent)

    def duration(self):
        return self.duration_ms

    def position(self):
        return self.position_ms

    def mediaStatus(self):
        return self.status

    def playbackState(self):
        return self.state

    def setPosition(self, milliseconds):
        self.position_ms = milliseconds

    def play(self):
        self.state = QMediaPlayer.PlaybackState.PlayingState

    def pause(self):
        self.state = QMediaPlayer.PlaybackState.PausedState


def _start_clock_preview(window, monkeypatch, *, silent=False):
    clock = [100.0]
    frames = []
    player = _ClockPlayer(silent=silent)
    monkeypatch.setattr(gui_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(window, "player", player)
    monkeypatch.setattr(window, "renderer", SimpleNamespace(scene=SimpleNamespace(
        score_duration=window.document.project.duration_seconds,
        settings=window.document.settings,
    )))
    monkeypatch.setattr(window, "_render_position", lambda: frames.append(window._position))
    window.toggle_playback()
    window.timer.stop()
    return clock, player, frames


@pytest.mark.parametrize("fps", [24, 30, 60])
def test_preview_repaint_uses_precise_timer_and_compiled_frame_rate(window, monkeypatch, fps):
    window.document.settings.fps = fps
    clock, _, frames = _start_clock_preview(window, monkeypatch, silent=True)
    assert window.timer.timerType() == Qt.TimerType.PreciseTimer
    assert window.timer.interval() == round(1000 / fps)
    # A delayed repaint still advances by elapsed time, not by a frame count.
    clock[0] += 0.217
    window._tick()
    assert frames[-1] == pytest.approx(0.217)


def test_audio_preview_advances_between_sparse_media_callbacks(window, monkeypatch):
    clock, player, frames = _start_clock_preview(window, monkeypatch)
    for step in (0.033, 0.066, 0.099):
        clock[0] = 100 + step
        window._tick()
    assert frames == pytest.approx([0.033, 0.066, 0.099])
    clock[0] = 100.1
    player.position_ms = 100
    window._audio_position_changed(100)
    clock[0] += 0.033
    window._tick()
    assert frames[-1] == pytest.approx(0.133)


def test_media_millisecond_rounding_does_not_reverse_preview(window, monkeypatch):
    clock, player, frames = _start_clock_preview(window, monkeypatch)
    clock[0] += 0.0335
    window._tick()
    window._audio_position_changed(33)
    window._tick()
    assert frames[-1] == frames[-2]
    clock[0] += 0.02
    window._tick()
    assert frames[-1] == pytest.approx(0.0335 + 0.02 * 0.999)


def test_late_coarse_media_callback_corrects_rate_without_holding_frames(window, monkeypatch):
    clock, player, frames = _start_clock_preview(window, monkeypatch)
    for elapsed in (0.1, 0.2, 0.29):
        clock[0] = 100 + elapsed
        window._tick()
    clock[0] = 100.3
    player.position_ms = 200
    window._audio_position_changed(200)
    previous_time, previous_position = 100.29, frames[-1]
    for elapsed in (0.3, 0.333, 0.366, 0.399, 0.432):
        clock[0] = 100 + elapsed
        window._tick()
        assert frames[-1] > previous_position
        speed = (frames[-1] - previous_position) / (clock[0] - previous_time)
        assert 0.95 - 1e-7 <= speed <= 1.05 + 1e-7
        previous_time, previous_position = clock[0], frames[-1]


def test_repeated_playing_notifications_do_not_reset_to_a_stale_getter(window, monkeypatch):
    clock, player, frames = _start_clock_preview(window, monkeypatch)
    clock[0] += 0.5
    window._tick()
    player.position_ms = 400
    window._audio_status_changed(QMediaPlayer.MediaStatus.BufferedMedia)
    window._audio_playback_state_changed(QMediaPlayer.PlaybackState.PlayingState)
    window._tick()
    assert frames[-1] == pytest.approx(0.5)
    clock[0] += 0.033
    window._tick()
    assert frames[-1] == pytest.approx(0.533)


def test_stall_with_a_stale_getter_preserves_the_frame_and_continuous_resume(window, monkeypatch):
    clock, player, frames = _start_clock_preview(window, monkeypatch)
    clock[0] += 0.29
    window._tick()
    player.position_ms = 200
    player.status = QMediaPlayer.MediaStatus.StalledMedia
    window._audio_status_changed(player.status)
    window._audio_position_changed(200)
    clock[0] += 5
    window._tick()
    assert frames[-1] == pytest.approx(0.29)
    player.status = QMediaPlayer.MediaStatus.BufferedMedia
    window._audio_status_changed(player.status)
    window._tick()
    assert frames[-1] == pytest.approx(0.29)
    clock[0] += 0.033
    window._tick()
    assert frames[-1] == pytest.approx(0.29 + 0.033 * 0.95)


def test_seek_and_pause_reanchor_continuous_playback(window, monkeypatch):
    clock, player, frames = _start_clock_preview(window, monkeypatch)
    clock[0] += 0.1
    window._tick()
    window.seek_to_milliseconds(2000)
    assert frames[-1] == 2
    clock[0] += 0.125
    window._tick()
    assert frames[-1] == pytest.approx(2.125)
    window.pause_playback()
    paused = window._position
    clock[0] += 5
    window._tick()
    assert window._position == paused
    window.toggle_playback()
    window.timer.stop()
    clock[0] += 0.1
    window._tick()
    assert frames[-1] == pytest.approx(paused + 0.1)


@pytest.mark.parametrize("status", [QMediaPlayer.MediaStatus.StalledMedia,
                                  QMediaPlayer.MediaStatus.BufferingMedia,
                                  QMediaPlayer.MediaStatus.LoadingMedia])
def test_buffering_freezes_clock_and_resume_discards_elapsed_wall_time(window, monkeypatch, status):
    clock, player, frames = _start_clock_preview(window, monkeypatch)
    clock[0] += 0.2
    window._tick()
    player.position_ms = 200
    player.status = status
    window._audio_status_changed(status)
    clock[0] += 5
    window._tick()
    assert frames[-1] == pytest.approx(0.2)
    player.status = QMediaPlayer.MediaStatus.BufferedMedia
    window._audio_status_changed(player.status)
    window._tick()
    assert frames[-1] == pytest.approx(0.2)
    clock[0] += 0.033
    window._tick()
    assert frames[-1] == pytest.approx(0.233)


def test_media_pause_and_resume_wait_for_the_player_clock(window, monkeypatch):
    clock, player, frames = _start_clock_preview(window, monkeypatch)
    clock[0] += 0.2
    window._tick()
    player.position_ms = 200
    player.state = QMediaPlayer.PlaybackState.PausedState
    window._audio_playback_state_changed(player.state)
    clock[0] += 3
    window._tick()
    assert frames[-1] == pytest.approx(0.2)
    player.state = QMediaPlayer.PlaybackState.PlayingState
    window._audio_playback_state_changed(player.state)
    clock[0] += 0.05
    window._tick()
    assert frames[-1] == pytest.approx(0.25)


def test_end_of_audio_keeps_the_score_tail_clock_and_stops_at_duration(window, monkeypatch):
    clock, player, frames = _start_clock_preview(window, monkeypatch)
    player.duration_ms = 1000
    player.status = QMediaPlayer.MediaStatus.EndOfMedia
    window._audio_status_changed(player.status)
    window._audio_position_changed(0)
    clock[0] += 0.3
    window._tick()
    assert frames[-1] == pytest.approx(1.3)
    window._audio_status_changed(player.status)
    clock[0] += 0.1
    window._tick()
    assert frames[-1] == pytest.approx(1.4)
    clock[0] += 20
    window._tick()
    assert frames[-1] == pytest.approx(window.document.project.duration_seconds)
    assert not window._playing


@pytest.mark.parametrize("paused", [False, True])
def test_seek_beyond_audio_end_keeps_the_score_time_and_can_seek_back(window, monkeypatch, paused):
    clock, player, frames = _start_clock_preview(window, monkeypatch)
    player.duration_ms = 1000
    if paused:
        window.pause_playback()
    window.seek_to_milliseconds(2000)
    assert frames[-1] == 2
    if paused:
        window.toggle_playback()
        window.timer.stop()
    window._audio_position_changed(1000)
    window._audio_status_changed(QMediaPlayer.MediaStatus.BufferedMedia)
    window._audio_playback_state_changed(QMediaPlayer.PlaybackState.PausedState)
    clock[0] += 0.1
    window._tick()
    assert frames[-1] == pytest.approx(2.1)
    window.seek_to_milliseconds(500)
    clock[0] += 0.033
    window._tick()
    assert frames[-1] == pytest.approx(0.533)


def test_silent_playback_keeps_its_monotonic_clock(window, monkeypatch):
    clock, player, frames = _start_clock_preview(window, monkeypatch, silent=True)
    clock[0] += 0.2
    window._tick()
    assert frames[-1] == pytest.approx(0.2)
    window.pause_playback()
    clock[0] += 2
    window._tick()
    assert window._position == pytest.approx(0.2)


def test_intro_delay_keeps_audio_paused_and_ignores_stale_media_callbacks(window, monkeypatch):
    window.document.settings.intro_delay_seconds = 2
    clock, player, frames = _start_clock_preview(window, monkeypatch)
    assert player.position_ms == 0
    assert player.state == QMediaPlayer.PlaybackState.PausedState
    clock[0] += 1
    window._tick()
    window._audio_position_changed(9500)
    window._audio_status_changed(QMediaPlayer.MediaStatus.EndOfMedia)
    window._audio_playback_state_changed(QMediaPlayer.PlaybackState.PlayingState)
    assert not window._audio_tail_running
    assert not window._audio_clock_running
    clock[0] += 0.25
    window.pause_playback()
    assert window._position == pytest.approx(1.25)
    clock[0] += 5
    window._tick()
    assert window._position == pytest.approx(1.25)
    window.toggle_playback()
    window.timer.stop()
    clock[0] += 1
    window._tick()
    assert frames[-1] == 2
    assert player.position_ms == 0
    assert player.state == QMediaPlayer.PlaybackState.PlayingState
    clock[0] += 0.125
    window._tick()
    assert frames[-1] == pytest.approx(2.125)


def test_intro_boundary_starts_audio_from_zero_and_freezes_during_buffering(window, monkeypatch):
    window.document.settings.intro_delay_seconds = 1
    clock, player, frames = _start_clock_preview(window, monkeypatch)
    player.status = QMediaPlayer.MediaStatus.BufferingMedia
    clock[0] += 1.2
    window._tick()
    assert frames[-1] == 1
    assert player.position_ms == 0
    assert player.state == QMediaPlayer.PlaybackState.PlayingState
    clock[0] += 4
    window._tick()
    assert frames[-1] == 1
    player.status = QMediaPlayer.MediaStatus.BufferedMedia
    window._audio_status_changed(player.status)
    clock[0] += 0.1
    window._tick()
    assert frames[-1] == pytest.approx(1.1)


def test_seek_maps_presentation_time_to_audio_and_can_return_to_intro(window, monkeypatch):
    window.document.settings.intro_delay_seconds = 2
    clock, player, frames = _start_clock_preview(window, monkeypatch)
    window.seek_to_milliseconds(3250)
    assert frames[-1] == 3.25
    assert player.position_ms == 1250
    window._audio_position_changed(1250)
    clock[0] += 0.1
    window._tick()
    assert frames[-1] == pytest.approx(3.35)
    window.seek_to_milliseconds(750)
    assert player.position_ms == 0
    assert player.state == QMediaPlayer.PlaybackState.PausedState
    window._audio_position_changed(1250)
    window._audio_status_changed(QMediaPlayer.MediaStatus.EndOfMedia)
    clock[0] += 0.25
    window._tick()
    assert frames[-1] == 1
    window.seek_to_milliseconds(2500)
    assert player.position_ms == 500
    assert player.state == QMediaPlayer.PlaybackState.PlayingState


@pytest.mark.parametrize("offset", [-0.5, 0.75])
def test_compiled_settings_control_preview_delay_duration_and_score_offset(window, monkeypatch, offset):
    settings = window.document.settings
    settings.intro_delay_seconds = 2
    settings.score_start_in_audio_sec = offset
    clock, player, frames = _start_clock_preview(window, monkeypatch, silent=True)
    window.renderer.scene.settings = copy.deepcopy(settings)
    window.document.settings.intro_delay_seconds = 9
    window.document.settings.score_start_in_audio_sec = 7
    player.duration_ms = 0
    duration = 2 + window.document.project.duration_seconds + offset
    assert window._duration() == pytest.approx(duration)
    window._update_transport()
    assert window.seek.maximum() == math.ceil(duration * 1000)
    clock[0] += 2.25
    window._tick()
    assert frames[-1] == pytest.approx(2.25)
    window.seek_to_milliseconds(999999)
    assert frames[-1] == pytest.approx(duration)


def test_end_of_audio_with_intro_delay_keeps_the_score_tail_clock(window, monkeypatch):
    window.document.settings.intro_delay_seconds = 2
    clock, player, frames = _start_clock_preview(window, monkeypatch)
    player.duration_ms = 1000
    window.seek_to_milliseconds(2500)
    player.status = QMediaPlayer.MediaStatus.EndOfMedia
    window._audio_status_changed(player.status)
    window._audio_position_changed(0)
    clock[0] += 0.3
    window._tick()
    assert frames[-1] == pytest.approx(3.3)
    clock[0] += 20
    window._tick()
    assert frames[-1] == pytest.approx(2 + window.document.project.duration_seconds)
    assert not window._playing


def test_replay_waits_at_intro_boundary_until_old_end_of_media_status_is_replaced(window, monkeypatch):
    window.document.settings.intro_delay_seconds = 1
    clock, player, frames = _start_clock_preview(window, monkeypatch)
    player.duration_ms = 1000
    window.seek_to_milliseconds(round(window._duration() * 1000))
    window.pause_playback()
    player.status = QMediaPlayer.MediaStatus.EndOfMedia
    window.toggle_playback()
    window.timer.stop()
    assert window._position == 0
    assert player.position_ms == 0
    clock[0] += 1.2
    window._tick()
    assert frames[-1] == 1
    assert window._awaiting_audio_start
    window._audio_status_changed(QMediaPlayer.MediaStatus.EndOfMedia)
    window._audio_position_changed(1000)
    window._audio_playback_state_changed(QMediaPlayer.PlaybackState.PlayingState)
    clock[0] += 5
    window._tick()
    assert frames[-1] == 1
    assert not window._audio_tail_running
    player.status = QMediaPlayer.MediaStatus.BufferedMedia
    window._audio_status_changed(player.status)
    assert not window._awaiting_audio_start
    window._audio_status_changed(QMediaPlayer.MediaStatus.EndOfMedia)
    clock[0] += 0.1
    window._tick()
    assert frames[-1] == pytest.approx(1.1)


def test_delayed_end_of_media_callback_after_seek_does_not_jump_to_audio_end(window, monkeypatch):
    window.document.settings.intro_delay_seconds = 1
    clock, player, frames = _start_clock_preview(window, monkeypatch)
    window.seek_to_milliseconds(2000)
    window._audio_status_changed(QMediaPlayer.MediaStatus.EndOfMedia)
    assert not window._audio_tail_running
    assert window._position == 2
    clock[0] += 0.1
    window._tick()
    assert frames[-1] == pytest.approx(2.1)


def _wait_for_job(app, job, timeout=15):
    deadline = time.monotonic() + timeout
    while job.running and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert not job.running, "worker did not finish"


def test_spawn_worker_returns_document_and_reports_parse_errors(app, document, tmp_path):
    target = tmp_path / "session.stproj"
    save_document(document, target)
    received, failed = [], []
    job = BackgroundJob("load", str(target))
    job.succeeded.connect(received.append)
    job.failed.connect(lambda message, details: failed.append((message, details)))
    job.start()
    _wait_for_job(app, job)
    assert len(received) == 1
    assert received[0].to_dict() == load_document(target).to_dict()
    assert not failed
    target.write_text("{malformed", encoding="utf-8")
    bad = BackgroundJob("load", str(target))
    bad.failed.connect(lambda message, details: failed.append((message, details)))
    bad.start()
    _wait_for_job(app, bad)
    assert failed and "JSONDecodeError" in failed[0][1]
    job.shutdown()
    bad.shutdown()


def test_spawn_worker_cancellation_is_distinct_from_failure(app, document, tmp_path):
    target = tmp_path / "session.stproj"
    save_document(document, target)
    states = []
    job = BackgroundJob("load", str(target))
    job.succeeded.connect(lambda value: states.append("success"))
    job.failed.connect(lambda message, details: states.append("failure"))
    job.cancelled.connect(lambda: states.append("cancelled"))
    job.start()
    job.cancel()
    _wait_for_job(app, job)
    assert states == ["cancelled"]
    job.shutdown()


@pytest.mark.integration
def test_spawn_compile_returns_shared_score_geometry(app, document):
    results, errors = [], []
    job = BackgroundJob("compile", document)
    job.succeeded.connect(results.append)
    job.failed.connect(lambda message, details: errors.append((message, details)))
    job.start()
    _wait_for_job(app, job, timeout=30)
    assert not errors, errors
    assert len(results) == 1
    scene = results[0]
    assert [part.part_id for part in scene.parts] == [mapping.part_id for mapping in document.mappings]
    assert len(scene.axis.beats) == len(scene.axis.xs) >= 2
    assert scene.axis.x_at(0) < scene.axis.x_at(4)
    assert "<svg" in scene.svg or ":svg" in scene.svg
    job.shutdown()


@pytest.mark.integration
def test_desktop_generates_preview_and_repeated_seek_is_deterministic(app, document, monkeypatch):
    widget = MainWindow()
    errors = []
    monkeypatch.setattr(widget, "_show_error", errors.append)
    monkeypatch.setattr(widget, "_job_error", lambda message, details: errors.append((message, details)))
    widget.set_document(document)
    app.processEvents()
    assert widget._job is not None
    job = widget._job
    _wait_for_job(app, job, timeout=30)
    assert not errors, errors
    assert widget.renderer is not None
    assert widget.preview.frame.width() == document.settings.width
    assert widget.play_button.isEnabled()
    widget.seek_to_milliseconds(1000)
    expected = bytes(widget.preview.frame.bits())
    widget.seek_to_milliseconds(500)
    widget.seek_to_milliseconds(1000)
    assert bytes(widget.preview.frame.bits()) == expected
    widget._dirty = False
    widget.close()
    widget.deleteLater()
    app.processEvents()


@pytest.mark.integration
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_closing_running_export_preserves_output_and_cleans_worker(app, document, tmp_path):
    audio_path = tmp_path / "audio.wav"
    with wave.open(str(audio_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        stream.writeframes(b"\x00\x00" * 48000 * 3)
    document.audio_path = str(audio_path)
    document.project.duration_ticks = 480 * 1200
    document.settings.width, document.settings.height = 640, 360
    document.settings.fps = 30
    target = tmp_path / "preserved.mp4"
    target.write_bytes(b"existing user video")
    messages, errors = [], []
    job = BackgroundJob("video", (document, str(target)))
    job.progress.connect(lambda fraction, message: messages.append(message))
    job.failed.connect(lambda message, details: errors.append((message, details)))
    job.start()
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            app.processEvents()
            if errors or (any("逐帧渲染" in message for message in messages)
                          and list(tmp_path.glob("*.partial.mp4"))):
                break
            time.sleep(0.01)
        assert not errors, errors
        assert any("逐帧渲染" in message for message in messages)
        assert job.running
        process = job._process
        job.shutdown()
        assert not process.is_alive()
        assert process.exitcode is not None
        assert target.read_bytes() == b"existing user video"
        assert not list(tmp_path.glob("*.partial.mp4"))
    finally:
        job.shutdown()


@pytest.mark.integration
def test_silent_preview_transport_includes_tail_extended_by_quantization(app, document, monkeypatch):
    document.project.notes = [NoteEvent("short-note", "violin", 0, 100, 72)]
    document.project.duration_ticks = 100
    document.mappings = [document.mappings[0]]
    document.settings.score_start_in_audio_sec = 0.5
    widget = MainWindow()
    errors = []
    monkeypatch.setattr(widget, "_show_error", errors.append)
    monkeypatch.setattr(widget, "_job_error", lambda message, details: errors.append((message, details)))
    widget.set_document(document)
    app.processEvents()
    assert widget._job is not None
    _wait_for_job(app, widget._job, timeout=30)
    assert not errors, errors
    assert widget.player.duration() == 0
    scene = widget.renderer.scene
    assert scene.score_duration > document.project.duration_seconds
    assert scene.score_duration == pytest.approx(0.125)
    assert widget._duration() == pytest.approx(0.625)
    assert widget.seek.maximum() == 625
    widget.seek_to_milliseconds(10000)
    assert widget._position == pytest.approx(0.625)
    widget._dirty = False
    widget.close()
    widget.deleteLater()
    app.processEvents()


@pytest.mark.integration
def test_reimport_arco_only_preserves_corrections_and_removes_pizz_articulation(window):
    from stavellum.presentation.scene import compile_scene

    window.document.mappings = [PartMapping(
        "custom-violin", "Violin I 校正", ["violin", "pizz"], instrument="violin",
        clef="alto", key_signature=-3, use_icon=True,
        articulations={"violin": "arco", "pizz": "pizz."},
    )]
    window.document.project.tracks[0].color = "#c04080"
    source = ProjectIR("fixture.mid", "midi", "Arco only",
                       tracks=[TrackInfo("violin", "Violin I arco", "#4080c0")],
                       notes=[NoteEvent("reimported", "violin", 0, 480, 72)])
    window._job = SimpleNamespace(reimporting=True)
    try:
        window._source_imported(source)
    finally:
        window._job = None
    mapping = window.document.mappings[0]
    assert (mapping.name, mapping.clef, mapping.key_signature) == ("Violin I 校正", "alto", -3)
    assert mapping.track_ids == ["violin"]
    assert mapping.articulations == {"violin": "arco"}
    window.document.validate()
    scene = compile_scene(window.document)
    assert len(scene.parts) == 1
    assert scene.parts[0].part_id == "custom-violin"
    assert scene.parts[0].activity_color == "#4080c0"


@pytest.mark.integration
def test_reimport_suggestion_prunes_tracks_already_owned_by_retained_mapping(window):
    from stavellum.presentation.scene import compile_scene

    window.document.mappings = [window.document.mappings[0]]
    window.document.mappings[0].part_id = "part-violin"
    source = copy.deepcopy(window.document.project)
    source.tracks = [TrackInfo("violin", "Violin I arco", "#4080c0"),
                     TrackInfo("pizz", "Violin I Pizz", "#c04080")]
    source.notes = [note for note in source.notes if note.track_id in {"violin", "pizz"}]
    window._job = SimpleNamespace(reimporting=True)
    try:
        window._source_imported(source)
    finally:
        window._job = None
    assert [mapping.track_ids for mapping in window.document.mappings] == [["violin"], ["pizz"]]
    assert window.document.mappings[0].part_id == "part-violin"
    assert window.document.mappings[1].part_id != "part-violin"
    assert window.document.mappings[1].articulations == {"pizz": "pizz."}
    window.document.validate()
    assert [part.activity_color for part in compile_scene(window.document).parts] == ["#4080c0", "#c04080"]


def test_reassigning_track_prunes_technique_from_part_that_still_has_tracks(window):
    window.document.mappings[0].track_ids = ["violin", "pizz"]
    window.document.mappings[0].articulations = {"violin": "arco", "pizz": "pizz."}
    window.document.mappings[1].enabled = False
    window._refresh_parts(2)
    window.part_tracks.item(1).setCheckState(Qt.CheckState.Checked)
    assert window.apply_part()
    assert window.document.mappings[0].track_ids == ["violin"]
    assert window.document.mappings[0].articulations == {"violin": "arco"}
    window.document.validate()


def test_unchecking_source_track_prunes_stale_technique_in_repeated_apply(window):
    window.document.mappings[0].track_ids = ["violin", "pizz"]
    window.document.mappings[0].articulations = {"violin": "arco", "pizz": "pizz."}
    window.document.mappings[1].enabled = False
    window._refresh_parts(0)
    window.part_tracks.item(1).setCheckState(Qt.CheckState.Unchecked)
    assert window.apply_part(refresh=False)
    assert window.apply_part(refresh=False)
    assert window.document.mappings[0].articulations == {"violin": "arco"}
    window.document.validate()
