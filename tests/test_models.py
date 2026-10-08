"""Persistence and validation protect source timing and incomplete editing work."""

import json
import math
from copy import deepcopy
from pathlib import Path

import pytest

from stavellum.domain.mapping import activity_color
from stavellum.domain.models import (
    ANIMATION_DURATIONS,
    ANIMATION_PRESETS,
    AutomationPoint,
    Diagnostic,
    Metadata,
    NoteEvent,
    PartMapping,
    ProjectDocument,
    ProjectIR,
    RenderSettings,
    TrackInfo,
    VolumeAutomation,
    VolumeRoute,
    load_document,
    save_document,
)


def project_document():
    project = ProjectIR("source.flp", "flp", "项目", ppq=96, bpm=112.5, tracks=[TrackInfo("bow", "Violin I"), TrackInfo("pizz", "Violin I Pizz.")], notes=[NoteEvent("held", "bow", 7, 503, 72, velocity=95, midi_channel=2, source_pattern="Phrase", slide=True), NoteEvent("pluck", "pizz", 576, 0, 76, velocity=70)], diagnostics=[Diagnostic("warning", "slide", "请核对", "bow")], arrangement_names=["Main", "Alternative"], arrangement_index=1, duration_ticks=1152)
    mapping = PartMapping("violin1", "Violin I", ["bow", "pizz"], instrument="violin", icon="violin", clef="treble", key_signature=-4, transpose=2, quantization=32, triplets=False, keyswitches=[24, 25], articulations={"bow": "arco", "pizz": "pizz."}, use_icon=True)
    settings = RenderSettings(width=1280, height=720, fps=30, staff_scale=1.25, score_start_in_audio_sec=-0.125, title_x=0.05, title_font_size=52, crf=20, preset="fast")
    return ProjectDocument(project, [mapping], audio_path="original.wav", settings=settings, metadata=Metadata("曲名", "副标题", "作曲者", "编曲者"))


@pytest.mark.parametrize("legacy_enabled", [True, False])
@pytest.mark.parametrize("new_enabled", [None, True, False])
def test_legacy_icon_switch_migrates_without_changing_display_or_input(
        tmp_path, legacy_enabled, new_enabled):
    payload = project_document().to_dict()
    mapping = payload["mappings"][0]
    mapping.pop("use_icon")
    mapping["confirmed"] = legacy_enabled
    mapping["auto_ottava"] = True
    if new_enabled is not None:
        mapping["use_icon"] = new_enabled
    before = deepcopy(payload)
    document = ProjectDocument.from_dict(payload)
    assert payload == before
    assert document.mappings[0].use_icon is (legacy_enabled if new_enabled is None else new_enabled)
    assert document.mappings[0].auto_ottava is True
    destination = tmp_path / "migrated.stproj"
    save_document(document, destination)
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert "confirmed" not in saved["mappings"][0]
    assert saved["schema_version"] == 1
    assert load_document(destination).mappings == document.mappings


def test_missing_icon_switch_defaults_to_visible_and_legacy_none_stays_hidden():
    payload = project_document().to_dict()
    mapping = payload["mappings"][0]
    mapping.pop("use_icon")
    assert ProjectDocument.from_dict(payload).mappings[0].use_icon
    mapping.update(confirmed=True, icon="none")
    restored = ProjectDocument.from_dict(payload).mappings[0]
    assert not restored.use_icon
    assert restored.icon == ""


@pytest.mark.parametrize("field_name", ["confirmed", "use_icon"])
@pytest.mark.parametrize("value", [None, 0, "false"])
def test_icon_visibility_rejects_nonboolean_project_fields(field_name, value):
    payload = project_document().to_dict()
    mapping = payload["mappings"][0]
    mapping.pop("use_icon")
    mapping[field_name] = value
    with pytest.raises(ValueError, match=field_name):
        ProjectDocument.from_dict(payload)


def test_logo_settings_roundtrip_and_legacy_defaults(tmp_path):
    document = project_document()
    assert not document.settings.logo_enabled
    values = dict(logo_enabled=True, logo_display_mode="intro", logo_size_ratio=0.12,
                  logo_opacity=0.65, logo_enter_seconds=0.5, logo_hold_seconds=0.0,
                  logo_exit_seconds=0.4)
    for name, value in values.items():
        setattr(document.settings, name, value)
    document.validate()
    filename = tmp_path / "logo.stproj"
    save_document(document, filename)
    restored = load_document(filename)
    assert restored.settings == document.settings
    assert restored.schema_version == 1
    payload = document.to_dict()
    for name in values:
        del payload["settings"][name]
    legacy = ProjectDocument.from_dict(payload)
    assert {name: getattr(legacy.settings, name) for name in values} == {
        name: getattr(RenderSettings(), name) for name in values}


@pytest.mark.parametrize("name,value", [
    ("logo_enabled", 1), ("logo_display_mode", "loop"),
    ("logo_size_ratio", 0.019), ("logo_size_ratio", 0.251),
    ("logo_opacity", -0.01), ("logo_opacity", 1.01),
    ("logo_enter_seconds", 0), ("logo_exit_seconds", -1), ("logo_hold_seconds", -1),
    *[(name, value) for name in ("logo_size_ratio", "logo_opacity", "logo_enter_seconds",
                                "logo_hold_seconds", "logo_exit_seconds")
      for value in (math.inf, math.nan)],
])
def test_logo_settings_reject_invalid_values_even_when_disabled(name, value):
    settings = RenderSettings()
    setattr(settings, name, value)
    with pytest.raises(ValueError):
        settings.validate()


def test_roundtrip_preserves_settings_mapping_metadata_and_exact_raw_timing(tmp_path):
    document = project_document()
    document.project.tracks[0].color = "#4080c0"
    document.project.tracks[1].color = "#c04080"
    before = deepcopy(document.to_dict())
    filename = tmp_path / "song.sheet.json"
    save_document(document, filename)
    restored = load_document(filename)
    assert restored.settings == document.settings
    assert restored.mappings == document.mappings
    assert restored.metadata == document.metadata
    assert restored.project.notes == document.project.notes
    assert restored.project.tracks == document.project.tracks
    assert activity_color(restored.project, restored.mappings[0]) == "#4080c0"
    assert restored.project.diagnostics == document.project.diagnostics
    assert restored.project.notes[0].start_tick == 7
    assert restored.project.notes[0].duration_tick == 503
    assert restored.project.notes[0].slide is True
    assert restored.project.notes[1].duration_tick == 0
    assert document.to_dict() == before
    assert json.loads(filename.read_text(encoding="utf-8"))["schema_version"] == 1
    restored.validate()


@pytest.mark.parametrize("source_type", ["flp", "midi"])
def test_legacy_project_without_track_colors_loads_with_white_activity(tmp_path, source_type):
    document = project_document()
    document.project.source_type = source_type
    payload = document.to_dict()
    for track in payload["project"]["tracks"]:
        del track["color"]
    filename = tmp_path / "legacy.stproj"
    filename.write_text(json.dumps(payload), encoding="utf-8")
    restored = load_document(filename)
    assert [track.color for track in restored.project.tracks] == ["#ffffff", "#ffffff"]
    assert activity_color(restored.project, restored.mappings[0]) == "#ffffff"
    restored.validate()


@pytest.mark.parametrize("field_name, expected", [("auto_simplify_accidentals", True), ("auto_ottava", False)])
def test_new_part_mapping_uses_notation_defaults(field_name, expected):
    mapping = PartMapping("violin", "Violin", ["bow"])
    assert getattr(mapping, field_name) is expected
    assert getattr(project_document().mappings[0], field_name) is expected


@pytest.mark.parametrize("field_name", ["auto_simplify_accidentals", "auto_ottava"])
@pytest.mark.parametrize("enabled", [True, False])
def test_notation_simplification_setting_survives_save_without_schema_change(
        tmp_path, field_name, enabled):
    document = project_document()
    setattr(document.mappings[0], field_name, enabled)
    target = tmp_path / "spelling.stproj"
    save_document(document, target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["mappings"][0][field_name] is enabled
    assert payload["schema_version"] == 1
    restored = load_document(target)
    assert getattr(restored.mappings[0], field_name) is enabled
    assert restored.project.notes == document.project.notes
    restored.validate()


@pytest.mark.parametrize("field_name, expected", [("auto_simplify_accidentals", True), ("auto_ottava", False)])
def test_legacy_project_without_notation_settings_uses_defaults(tmp_path, field_name, expected):
    document = project_document()
    payload = document.to_dict()
    del payload["mappings"][0][field_name]
    target = tmp_path / "legacy-spelling.stproj"
    target.write_text(json.dumps(payload), encoding="utf-8")
    restored = load_document(target)
    assert getattr(restored.mappings[0], field_name) is expected
    assert restored.project.notes == document.project.notes
    assert restored.schema_version == 1
    restored.validate()


@pytest.mark.parametrize("field_name", ["auto_simplify_accidentals", "auto_ottava"])
@pytest.mark.parametrize("value", [None, 0, 1, "false", [], {}])
def test_notation_simplification_rejects_nonboolean_values(field_name, value, tmp_path):
    document = project_document()
    setattr(document.mappings[0], field_name, value)
    with pytest.raises(ValueError, match=field_name):
        document.validate()
    with pytest.raises(ValueError, match=field_name):
        save_document(document, tmp_path / "invalid.stproj")
    with pytest.raises(ValueError, match=field_name):
        ProjectDocument.from_dict(document.to_dict())


def test_relative_references_resolve_against_document_not_current_directory(tmp_path, monkeypatch):
    document = project_document()
    document.project.source_path = "sources/project.flp"
    document.audio_path = "audio/song.flac"
    folder = tmp_path / "project"
    filename = folder / "project.json"
    save_document(document, filename)
    monkeypatch.chdir(tmp_path)
    restored = load_document(filename)
    assert Path(restored.project.source_path) == (folder / "sources/project.flp").resolve()
    assert Path(restored.audio_path) == (folder / "audio/song.flac").resolve()
    assert document.project.source_path == "sources/project.flp"


def test_absolute_references_stay_absolute_and_empty_audio_stays_empty(tmp_path):
    document = project_document()
    document.project.source_path = str((tmp_path / "source.flp").resolve())
    document.audio_path = ""
    filename = tmp_path / "project.json"
    save_document(document, filename)
    restored = load_document(filename)
    assert restored.project.source_path == document.project.source_path
    assert restored.audio_path == ""


def test_unconfirmed_and_incomplete_projects_can_be_saved_for_correction(tmp_path):
    document = project_document()
    document.project.timing_confirmed = False
    document.mappings[0].use_icon = False
    document.mappings[0].key_signature = 12  # Pending correction, not structural corruption.
    filename = tmp_path / "unfinished.json"
    save_document(document, filename)
    restored = load_document(filename)
    assert not restored.project.timing_confirmed
    assert restored.mappings[0].key_signature == 12
    with pytest.raises(ValueError, match="固定速度"):
        restored.validate()


@pytest.mark.parametrize("version", [2, 0, "1", True, None])
def test_unknown_or_mistyped_schema_is_rejected(version):
    value = project_document().to_dict()
    value["schema_version"] = version
    with pytest.raises(ValueError, match="版本"):
        ProjectDocument.from_dict(value)


def test_missing_version_uses_original_schema_default():
    value = project_document().to_dict()
    value.pop("schema_version")
    assert ProjectDocument.from_dict(value).schema_version == 1


@pytest.mark.parametrize("saved_backend", [None, "auto", "gpu", "cpu"])
def test_existing_project_backend_values_keep_their_selection_without_a_schema_change(
        saved_backend, tmp_path):
    payload = project_document().to_dict()
    if saved_backend is None:
        del payload["settings"]["render_backend"]
    else:
        payload["settings"]["render_backend"] = saved_backend
    restored = ProjectDocument.from_dict(payload)
    restored.validate()
    expected = saved_backend or "auto"
    assert restored.settings.render_backend == expected
    target = tmp_path / "existing.stproj"
    save_document(restored, target)
    assert load_document(target).settings.render_backend == expected
    assert json.loads(target.read_text(encoding="utf-8"))["schema_version"] == 1


@pytest.mark.parametrize("mutation", [
    lambda data: data.update(settings=None),
    lambda data: data.update(mappings={}),
    lambda data: data.update(unknown_field="typo"),
    lambda data: data["project"].update(notes="not an array"),
    lambda data: data["project"]["notes"][0].update(start_tick=1.5),
    lambda data: data["project"]["notes"][0].update(slide="false"),
    lambda data: data["settings"].update(width="1920"),
    lambda data: data["metadata"].update(title=None),
    lambda data: data["mappings"][0].update(keyswitches=["24"]),
])
def test_malformed_structure_raises_actionable_valueerror(mutation):
    data = project_document().to_dict()
    mutation(data)
    with pytest.raises(ValueError, match="字段|对象|数组|结构"):
        ProjectDocument.from_dict(data)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("field_name", ["score_start_in_audio_sec", "enter_seconds", "staff_scale"])
def test_nonfinite_render_numbers_never_reach_encoder(field_name, value):
    document = project_document()
    setattr(document.settings, field_name, value)
    with pytest.raises(ValueError, match="有限"):
        document.validate()


def test_nonfinite_save_preserves_previous_document_and_cleans_temporary_files(tmp_path):
    document = project_document()
    filename = tmp_path / "project.json"
    save_document(document, filename)
    before = filename.read_bytes()
    document.project.bpm = math.nan
    with pytest.raises(ValueError):
        save_document(document, filename)
    assert filename.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_overflowing_numeric_json_is_reported_as_invalid_value():
    value = project_document().to_dict()
    value["project"]["bpm"] = 10 ** 1000
    with pytest.raises(ValueError, match="有限"):
        ProjectDocument.from_dict(value)


@pytest.mark.parametrize("bad_json", ['{"project": NaN}', '{"project": Infinity}', '{"project":'])
def test_invalid_json_has_clear_errors(tmp_path, bad_json):
    filename = tmp_path / "bad.json"
    filename.write_text(bad_json, encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        load_document(filename)


def test_utf8_bom_project_files_are_readable(tmp_path):
    filename = tmp_path / "bom.json"
    filename.write_text(json.dumps(project_document().to_dict(), ensure_ascii=False), encoding="utf-8-sig")
    assert load_document(filename).metadata.title == "曲名"


@pytest.mark.parametrize("field_name,value", [
    ("pitch", 128), ("velocity", -1), ("midi_channel", 16),
    ("start_tick", -1), ("duration_tick", -1), ("track_id", "missing"),
    ("note_id", ""),
])
def test_invalid_source_events_are_rejected_before_compilation(field_name, value):
    document = project_document()
    setattr(document.project.notes[0], field_name, value)
    with pytest.raises(ValueError):
        document.validate()


def test_duplicate_source_and_part_ids_and_channel_ranges_are_rejected():
    document = project_document()
    document.project.notes[1].note_id = document.project.notes[0].note_id
    with pytest.raises(ValueError, match="音符 ID"):
        document.validate()
    document = project_document()
    document.project.tracks[1].track_id = document.project.tracks[0].track_id
    with pytest.raises(ValueError, match="音轨 ID"):
        document.validate()
    document = project_document()
    document.project.tracks[0].midi_channel = 16
    with pytest.raises(ValueError, match="MIDI 通道"):
        document.validate()
    document = project_document()
    document.mappings.append(deepcopy(document.mappings[0]))
    document.mappings[-1].enabled = False
    with pytest.raises(ValueError, match="分谱 ID"):
        document.validate()


@pytest.mark.parametrize("field_name,value", [
    ("transpose", 128), ("key_signature", 8), ("keyswitches", [-1]),
    ("percussion_map", {"128": "C5"}), ("percussion_map", {"038": "C5"}),
    ("percussion_map", {"38": " "}), ("articulations", {"other-track": "pizz."}),
    ("track_ids", ["bow", "bow"]),
])
def test_invalid_mapping_overrides_are_rejected(field_name, value):
    document = project_document()
    setattr(document.mappings[0], field_name, value)
    with pytest.raises(ValueError):
        document.validate()


def test_disabled_mapping_may_be_incomplete_but_tracks_cannot_be_double_assigned():
    document = project_document()
    document.mappings.append(PartMapping("disabled", "Pending", [], key_signature=12, enabled=False))
    document.validate()
    document.mappings[-1].track_ids = ["bow"]
    document.mappings[-1].enabled = True
    with pytest.raises(ValueError, match="归属"):
        document.validate()


@pytest.mark.parametrize("field_name,value", [("crf", 52), ("preset", "invented"), ("title_font_size", 0), ("title_x", -0.1)])
def test_invalid_encoder_or_text_settings_are_rejected(field_name, value):
    document = project_document()
    setattr(document.settings, field_name, value)
    with pytest.raises(ValueError):
        document.validate()


def test_disabled_instrument_icon_does_not_block_rendering():
    document = project_document()
    document.mappings[0].use_icon = False
    document.validate()


@pytest.mark.parametrize("name", ["fast", "medium", "slow", "very_slow"])
def test_animation_preset_roundtrip_preserves_song_clock_and_hold_settings(name):
    settings = RenderSettings(intro_delay_seconds=2, announcement_hold_seconds=7,
                              animation_stable_seconds=4.125,
                              fps=30, score_start_in_audio_sec=1.25)
    settings.apply_animation_preset(name)
    assert settings.animation_preset() == name
    assert tuple(getattr(settings, field) for field in ANIMATION_DURATIONS
                 ) == ANIMATION_PRESETS[name]
    assert (settings.intro_delay_seconds, settings.announcement_hold_seconds,
            settings.animation_stable_seconds, settings.fps, settings.score_start_in_audio_sec
            ) == (2, 7, 4.125, 30, 1.25)
    settings.enter_seconds += 0.123
    assert settings.animation_preset() == "custom"
    with pytest.raises(ValueError, match="预设"):
        settings.apply_animation_preset("custom")


def test_original_projects_use_new_defaults_without_changing_custom_durations():
    value = project_document().to_dict()
    for field in ("intro_delay_seconds", "overlay_enter_seconds", "overlay_exit_seconds",
                  "announcement_auto_hide", "announcement_hold_seconds", "animation_stable_seconds"):
        value["settings"].pop(field)
    value["settings"]["enter_seconds"] = 1.234
    document = ProjectDocument.from_dict(value)
    assert document.settings.animation_preset() == "custom"
    assert document.settings.enter_seconds == 1.234
    assert document.settings.intro_delay_seconds == 0
    assert not document.settings.announcement_auto_hide
    assert document.settings.animation_stable_seconds == 2.0
    document.validate()


def test_presentation_settings_persist_with_no_schema_change(tmp_path):
    document = project_document()
    settings = document.settings
    settings.intro_delay_seconds = 2.75
    settings.announcement_auto_hide = True
    settings.announcement_hold_seconds = 8.25
    settings.animation_stable_seconds = 4.125
    settings.apply_animation_preset("very_slow")
    filename = tmp_path / "presentation.stproj"
    save_document(document, filename)
    restored = load_document(filename)
    assert restored.settings == settings
    assert restored.schema_version == 1
    restored.validate()


@pytest.mark.parametrize("stable_seconds", [0.0, 2.0, 4.125])
def test_animation_stable_time_survives_save_and_old_projects_use_two_seconds(tmp_path, stable_seconds):
    document = project_document()
    document.settings.animation_stable_seconds = stable_seconds
    document.validate()
    target = tmp_path / "stable-time.stproj"
    save_document(document, target)
    assert load_document(target).settings.animation_stable_seconds == stable_seconds
    payload = json.loads(target.read_text(encoding="utf-8"))
    del payload["settings"]["animation_stable_seconds"]
    target.write_text(json.dumps(payload), encoding="utf-8")
    restored = load_document(target)
    assert restored.settings.animation_stable_seconds == 2.0
    restored.validate()


@pytest.mark.parametrize("offset", [-2, 0, 1.5])
def test_presentation_clock_adds_intro_once_and_keeps_audio_and_score_offset_separate(offset):
    settings = RenderSettings(intro_delay_seconds=3, score_start_in_audio_sec=offset)
    assert settings.audio_time(0) == 0
    assert settings.audio_time(2) == 0
    assert settings.audio_time(4.25) == 1.25
    assert settings.presentation_time(1.25) == 4.25
    assert settings.presentation_duration(7, 10) == 3 + max(7, 10 + offset)
    assert settings.presentation_duration(15, 10) == 18


@pytest.mark.parametrize("field,value", [
    ("intro_delay_seconds", -0.1), ("announcement_hold_seconds", -0.1),
    ("overlay_enter_seconds", 0), ("overlay_exit_seconds", 0),
    ("announcement_auto_hide", "false"),
    ("animation_stable_seconds", -0.1), ("animation_stable_seconds", "2"),
    ("animation_stable_seconds", True),
])
def test_invalid_presentation_settings_are_rejected(field, value):
    document = project_document()
    setattr(document.settings, field, value)
    with pytest.raises(ValueError):
        document.validate()


@pytest.mark.parametrize("field", ["intro_delay_seconds", "announcement_hold_seconds",
                                   "overlay_enter_seconds", "overlay_exit_seconds", "animation_stable_seconds"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_presentation_settings_are_rejected(field, value):
    document = project_document()
    setattr(document.settings, field, value)
    with pytest.raises(ValueError, match="有限"):
        document.validate()


@pytest.mark.parametrize("mode", ["persistent", "intro"])
def test_old_tempo_preferences_are_retired_without_loosening_unknown_fields(mode):
    payload = project_document().to_dict()
    payload["settings"].update(tempo_display_mode=mode, tempo_hold_seconds=7.5)
    restored = ProjectDocument.from_dict(payload)
    restored.validate()
    assert "tempo_display_mode" not in restored.to_dict()["settings"]
    assert "tempo_hold_seconds" not in restored.to_dict()["settings"]
    payload["settings"]["unknown_tempo_setting"] = 1
    with pytest.raises(ValueError, match="未知字段"):
        ProjectDocument.from_dict(payload)


def automated_document():
    document = project_document()
    document.project.tracks[0].mixer_insert = 4
    document.project.volume_routes = [VolumeRoute(
        "bow", ["channel:0", "mixer:4", "mixer:0"],
        {"channel:0": 0.78125, "mixer:4": 1.0, "mixer:0": 1.0},
    )]
    document.project.volume_automations = [VolumeAutomation(
        "volume-clip", "mixer:4", "mixer_volume",
        [AutomationPoint(0, 0.2), AutomationPoint(960.5, 0.8, -0.617143, 0xff000000)],
        96, 1056, source_offset_tick=24.25, value_scale=1.25,
        source_label="Volume", minimum=0.1, maximum=0.9,
    )]
    document.mappings[0].auto_dynamics = False
    return document


def test_automation_roundtrip_preserves_curve_route_and_disabled_recognition(tmp_path):
    document = automated_document()
    document.validate()
    target = tmp_path / "automated.stproj"
    save_document(document, target)
    restored = load_document(target)
    restored.validate()
    assert restored.project.volume_automations == document.project.volume_automations
    assert restored.project.volume_routes == document.project.volume_routes
    assert restored.project.tracks[0].mixer_insert == 4
    assert not restored.mappings[0].auto_dynamics
    assert restored.project.notes == document.project.notes
    assert restored.schema_version == 1


def test_legacy_flp_automation_requires_explicit_reimport():
    payload = project_document().to_dict()
    payload["project"].pop("volume_automations")
    payload["project"].pop("volume_routes")
    payload["mappings"][0].pop("auto_dynamics")
    restored = ProjectDocument.from_dict(payload)
    assert not restored.project.volume_automations and not restored.project.volume_routes
    assert restored.mappings[0].auto_dynamics
    assert any(item.code == "legacy-volume-automation" for item in restored.project.diagnostics)


def test_automation_extent_keeps_exact_endpoints_within_the_score():
    document = automated_document()
    document.project.volume_automations[0].end_tick = 1536.25
    assert document.project.end_tick == 1537


def test_pattern_recording_source_survives_roundtrip_and_clip_is_default():
    document = automated_document()
    assert document.project.volume_automations[0].source_kind == "clip"
    document.project.volume_automations[0].source_kind = "pattern"
    restored = ProjectDocument.from_dict(document.to_dict())
    assert restored.project.volume_automations[0].source_kind == "pattern"
    payload = document.to_dict()
    payload["project"]["volume_automations"][0].pop("source_kind")
    assert ProjectDocument.from_dict(payload).project.volume_automations[0].source_kind == "clip"


def test_unsupported_curve_can_be_preserved_without_inventing_points():
    document = automated_document()
    automation = document.project.volume_automations[0]
    automation.supported = False
    automation.unsupported_reason = "曲线被截断"
    automation.points = []
    automation.raw_payload = "0100ff80"
    document.validate()
    restored = ProjectDocument.from_dict(document.to_dict())
    assert restored.project.volume_automations[0] == automation


@pytest.mark.parametrize("mutation", [
    lambda d: setattr(d.project.volume_automations[0].points[0], "tick", -0.5),
    lambda d: setattr(d.project.volume_automations[0].points[0], "value", math.nan),
    lambda d: setattr(d.project.volume_automations[0], "value_scale", 0),
    lambda d: setattr(d.project.volume_automations[0], "end_tick", 96),
    lambda d: setattr(d.project.volume_automations[0], "source_offset_tick", -1),
    lambda d: setattr(d.project.volume_automations[0], "source_kind", "unknown"),
    lambda d: setattr(d.project.volume_automations[0], "raw_payload", "unverified"),
    lambda d: d.project.volume_routes[0].initial_values.clear(),
    lambda d: setattr(d.project.volume_routes[0], "track_id", "missing"),
])
def test_invalid_automation_never_reaches_notation(mutation):
    document = automated_document()
    mutation(document)
    with pytest.raises(ValueError):
        document.validate()
