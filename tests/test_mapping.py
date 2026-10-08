import pytest

from stavellum.domain.mapping import (
    activity_color,
    identify_instrument,
    split_track_by_midi_channel,
    suggest_mappings,
)
from stavellum.domain.models import (
    AutomationPoint,
    NoteEvent,
    PartMapping,
    ProjectIR,
    TrackInfo,
    VolumeAutomation,
    VolumeRoute,
)


def _project(names):
    project = ProjectIR("sample.mid", "midi", "Sample")
    for index, name in enumerate(names):
        track_id = f"track-{index}"
        project.tracks.append(TrackInfo(track_id, name))
        project.notes.append(NoteEvent(f"note-{index}", track_id, index * 480, 480, 60))
    return project


def test_string_techniques_merge_without_collapsing_voices():
    project = _project(["Violin I", "Violin I Pizz", "Violin II Arco", "Violin II Pizzicato"])
    mappings = suggest_mappings(project)
    assert len(mappings) == 2
    assert mappings[0].track_ids == ["track-0", "track-1"]
    assert mappings[1].track_ids == ["track-2", "track-3"]
    assert mappings[0].articulations == {"track-0": "arco", "track-1": "pizz."}
    assert all(mapping.use_icon for mapping in mappings)
    assert not any(mapping.auto_ottava for mapping in mappings)


def test_identical_named_voices_are_not_automatically_merged():
    assert len(suggest_mappings(_project(["Violin", "Violin"]))) == 2
    assert len(suggest_mappings(_project(["Violin", "Violin", "Violin Pizz"]))) == 3
    assert len(suggest_mappings(_project(["Violin", "Violin Pizz", "Violin"]))) == 3


def test_unknown_plugin_does_not_identify_loaded_timbre():
    assert identify_instrument(TrackInfo("x", "Kontakt 7", plugin="Fruity Wrapper")) == "unknown"
    assert identify_instrument(TrackInfo("x", "FLEX", plugin="FLEX")) == "unknown"
    mapping = suggest_mappings(_project(["Kontakt 7"]))[0]
    assert mapping.icon == ""


def test_piano_and_percussion_defaults_and_gm_hints():
    project = _project(["钢琴", "Drums", "Channel 3"])
    project.tracks[-1].plugin = "General MIDI: Flute"
    piano, drums, flute = suggest_mappings(project)
    assert piano.grand_staff
    assert drums.percussion and drums.clef == "percussion"
    assert flute.instrument == "flute"


def test_chinese_pizzicato_merge_and_double_bass():
    mappings = suggest_mappings(_project(["小提琴 1 普通", "小提琴 1 拨奏", "低音提琴"] ))
    assert len(mappings) == 2
    assert mappings[-1].instrument == "double_bass"


def test_actual_orchestral_plural_and_technique_names():
    mappings = suggest_mappings(_project(["Celli", "Celli Pizz.", "Celeste", "Violins 1", "Violins 1 Spiccato", "Violins 2 Spiccato"]))
    assert len(mappings) == 4
    assert mappings[0].instrument == "cello"
    assert mappings[0].articulations == {"track-0": "arco", "track-1": "pizz."}
    assert mappings[1].instrument == "bell"
    assert mappings[2].track_ids == ["track-3", "track-4"]
    assert mappings[2].articulations["track-4"] == "spiccato"
    assert mappings[3].track_ids == ["track-5"]


def test_activity_color_prefers_regular_source_even_when_pizzicato_comes_first():
    project = _project(["Violin I Pizz", "Violin I", "Violin II"])
    for track, color in zip(project.tracks, ["#ff0000", "#0080ff", "#00ff00"], strict=True):
        track.color = color
    mappings = suggest_mappings(project)
    assert activity_color(project, mappings[0]) == "#0080ff"
    assert activity_color(project, mappings[1]) == "#00ff00"
    assert activity_color(project, PartMapping("pizz", "Pizz", ["track-0"])) == "#ff0000"


@pytest.mark.parametrize("technique", [
    "arco", "normal", "sustain", "sustained", "legato", "普通", "弓奏", "常规", " ＡＲＣＯ. ",
])
def test_activity_color_regular_mapping_overrides_special_source_name(technique):
    project = _project(["Violin Pizz", "Violin Staccato"])
    project.tracks[0].color = "#ff0000"
    project.tracks[1].color = "#00ff00"
    mapping = PartMapping("v", "Violin", ["track-0", "track-1"],
                          articulations={"track-1": technique})
    assert activity_color(project, mapping) == "#00ff00"


@pytest.mark.parametrize("technique", ["pizz.", "spiccato", "tremolo", "Custom Patch", ".", "．"])
def test_activity_color_unknown_explicit_technique_does_not_become_regular(technique):
    project = _project(["Violin Normal", "Violin Arco"])
    project.tracks[0].color = "#ff0000"
    project.tracks[1].color = "#00ff00"
    mapping = PartMapping("v", "Violin", ["track-0", "track-1"],
                          articulations={"track-0": technique})
    assert activity_color(project, mapping) == "#00ff00"


def test_activity_color_regular_sources_follow_mapping_order_including_unlabelled_sources():
    project = _project(["Violin", "Violin Arco"])
    project.tracks[0].color = "#ff0000"
    project.tracks[1].color = "#00ff00"
    mapping = PartMapping("v", "Violin", ["track-0", "track-1"],
                          articulations={"track-0": " ", "track-1": "arco"})
    assert activity_color(project, mapping) == "#ff0000"
    mapping.track_ids.reverse()
    assert activity_color(project, mapping) == "#00ff00"


def test_activity_color_without_regular_source_uses_first_mapping_source():
    project = _project(["Violin Pizz", "Violin Spiccato"])
    project.tracks[0].color = "#ff0000"
    project.tracks[1].color = "#00ff00"
    mapping = PartMapping("v", "Violin", ["track-1", "track-0"])
    assert activity_color(project, mapping) == "#00ff00"


@pytest.mark.parametrize(("color", "expected"), [
    ("#A0B1C2", "#a0b1c2"), ("#000000", "#000000"), ("red", "#ffffff"),
    ("#fff", "#ffffff"), ("#12345678", "#ffffff"), ("#zzzzzz", "#ffffff"),
    ("", "#ffffff"),
])
def test_activity_color_accepts_only_rgb_hex_with_white_fallback(color, expected):
    project = _project(["Violin"])
    project.tracks[0].color = color
    assert activity_color(project, suggest_mappings(project)[0]) == expected


def test_activity_color_without_available_source_is_white():
    project = _project(["Violin"])
    assert activity_color(project, PartMapping("empty", "Empty", [])) == "#ffffff"
    assert activity_color(project, PartMapping("missing", "Missing", ["absent"])) == "#ffffff"


def test_split_channels_is_explicit_and_preserves_note_timing():
    project = _project(["Kontakt"])
    project.tracks[0].color = "#4280ac"
    project.notes += [NoteEvent("note-extra", "track-0", 480, 960, 65, midi_channel=2)]
    assert len(suggest_mappings(project)) == 1
    ids = split_track_by_midi_channel(project, "track-0")
    assert ids == ["track-0-midi-0", "track-0-midi-2"]
    assert [note.track_id for note in project.notes] == ids
    assert [(note.start_tick, note.duration_tick) for note in project.notes] == [(0, 480), (480, 960)]
    assert [track.midi_channel for track in project.tracks] == [0, 2]
    assert [track.color for track in project.tracks] == ["#4280ac", "#4280ac"]


def test_split_channels_preserves_shared_channel_and_mixer_automation():
    project = _project(["Kontakt"])
    project.notes.append(NoteEvent("other", "track-0", 480, 960, 65, midi_channel=2))
    project.volume_routes = [VolumeRoute("track-0", ["channel:0", "mixer:4"],
                                       {"channel:0": .8, "mixer:4": 1.0})]
    project.volume_automations = [VolumeAutomation(
        "volume", "channel:0", "channel_volume",
        [AutomationPoint(0, .4), AutomationPoint(1440, .8)], 0, 1440,
    )]
    identifiers = split_track_by_midi_channel(project, "track-0")
    assert [route.track_id for route in project.volume_routes] == identifiers
    assert all(route.control_ids == ["channel:0", "mixer:4"] for route in project.volume_routes)
    assert len(project.volume_automations) == 1
    project.volume_routes[0].initial_values["channel:0"] = .5
    assert project.volume_routes[1].initial_values["channel:0"] == .8
