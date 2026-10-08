"""Notation inference must earn its hints while preserving source performance."""

import xml.etree.ElementTree as ET
from copy import deepcopy

import pytest

from stavellum.domain.models import NoteEvent, PartMapping, ProjectDocument, ProjectIR, TrackInfo
from stavellum.engraving.notation import build_notation
from stavellum.presentation.scene import compile_scene


def gate_note(identifier, start, duration, pitch=72, *, track="a", channel=0, velocity=100, **kwargs):
    return NoteEvent(identifier, track, start, duration, pitch, velocity=velocity,
                     midi_channel=channel, key_release_tick=start + duration, **kwargs)


def document(notes, *, duration_ticks=1920, bpm=120, **mapping_options):
    tracks = list(dict.fromkeys(note.track_id for note in notes))
    mapping = PartMapping("p", "Part", tracks, key_signature=0, **mapping_options)
    project = ProjectIR("", "midi", "Inference", bpm=bpm, notes=notes,
                        tracks=[TrackInfo(track, track) for track in tracks], duration_ticks=duration_ticks)
    return ProjectDocument(project, [mapping])


def regular_notes(**kwargs):
    return [gate_note(str(index), index * 480, 240, 72 + index, **kwargs) for index in range(4)]


def test_regular_short_gates_restore_rhythm_and_output_real_staccato_without_source_changes():
    project = document(regular_notes())
    before = deepcopy(project.to_dict())
    result = build_notation(project)
    assert [(event.start_beat, event.end_beat, event.staccato) for event in result.quantized_events] == [
        (0, 1, True), (1, 2, True), (2, 3, True), (3, 4, True)]
    xml = ET.fromstring(result.musicxml)
    assert len(xml.findall(".//notations/articulations/staccato")) == 4
    assert not xml.findall(".//note/rest")
    assert project.to_dict() == before
    diagnostic = next(item for item in result.diagnostics if item.code == "auto-note-recognition")
    assert "跳音 4 个" in diagnostic.message and diagnostic.track_id == "p"
    scene = compile_scene(project)
    assert [(note.start, note.end) for note in scene.parts[0].notes] == [
        (0, 0.25), (0.5, 0.75), (1, 1.25), (1.5, 1.75)]


def test_last_attack_does_not_extend_past_known_source_time():
    result = build_notation(document(regular_notes(), duration_ticks=0))
    assert [event.staccato for event in result.quantized_events] == [True, True, True, False]
    assert result.quantized_events[-1].end_beat == 3.5


def test_individual_part_switch_restores_short_durations_and_existing_technique_words():
    notes = regular_notes(articulation="staccato")
    project = document(notes, auto_staccato=False)
    result = build_notation(project)
    assert not any(event.staccato for event in result.quantized_events)
    assert [event.end_beat - event.start_beat for event in result.quantized_events] == [0.5] * 4
    xml = ET.fromstring(result.musicxml)
    assert not xml.findall(".//staccato")
    assert "staccato" in [element.text for element in xml.findall(".//words")]


@pytest.mark.parametrize("technique", ["staccato", "spiccato", "跳音", "跳弓"])
def test_explicit_technique_adds_point_but_does_not_recover_isolated_note_duration(technique):
    event = gate_note("n", 0, 240, articulation=technique)
    result = build_notation(document([event]))
    assert result.quantized_events[0].staccato
    assert result.quantized_events[0].end_beat == 0.5
    assert ET.fromstring(result.musicxml).find(".//staccato") is not None


@pytest.mark.parametrize("technique", ["legato", "tenuto", "pizz."])
def test_explicit_non_staccato_technique_prevents_duration_inference(technique):
    result = build_notation(document(regular_notes(articulation=technique)))
    assert not any(event.staccato for event in result.quantized_events)
    assert "明确奏法 4" in next(item.message for item in result.diagnostics if item.code == "auto-note-recognition")


def test_source_technique_overrides_mapping_label():
    result = build_notation(document(regular_notes(articulation="legato"), articulations={"a": "staccato"}))
    assert not any(event.staccato for event in result.quantized_events)


@pytest.mark.parametrize("case", ["too_few", "irregular", "long_rest", "unknown", "slide", "polyphonic", "different_chord_gates", "different_channels", "different_tracks", "percussion"])
def test_ambiguous_short_notes_remain_ordinary(case):
    notes = regular_notes()
    options = {}
    if case == "too_few":
        notes = notes[:3]
    elif case == "irregular":
        notes[-1].start_tick += 240
        notes[-1].key_release_tick += 240
    elif case == "long_rest":
        for index, source in enumerate(notes):
            source.start_tick = index * 1920
            source.key_release_tick = source.start_tick + source.duration_tick
    elif case == "unknown":
        for source in notes:
            source.key_release_tick = None
    elif case == "slide":
        for source in notes:
            source.slide = True
    elif case == "polyphonic":
        notes.append(gate_note("held", 0, 1920, 60))
    elif case == "different_chord_gates":
        notes += [gate_note(f"chord-{index}", index * 480, 360, 60) for index in range(4)]
    elif case == "different_channels":
        for index, source in enumerate(notes):
            source.midi_channel = index % 2
    elif case == "different_tracks":
        for index, source in enumerate(notes):
            source.track_id = "a" if index % 2 else "b"
    elif case == "percussion":
        options["percussion"] = True
    result = build_notation(document(notes, **options))
    assert not any(event.staccato or event.grace_main_id for event in result.quantized_events)


def test_synchronous_chord_gates_can_share_recovered_rhythm_and_staccato():
    notes = regular_notes() + [gate_note(f"bass-{index}", index * 480, 240, 60) for index in range(4)]
    result = build_notation(document(notes))
    assert all(event.staccato for event in result.quantized_events)
    assert len(ET.fromstring(result.musicxml).findall(".//staccato")) == 4


def grace_pair(*, grace_pitch=74, main_pitch=72):
    return [gate_note("grace", 450, 15, grace_pitch, velocity=65),
            gate_note("main", 480, 480, main_pitch)]


def mixed_expression_document():
    """Small score for XML/PDF/video visual checks of both inferred symbols."""
    notes = [gate_note(str(index), index * 480, 240, pitch)
             for index, pitch in enumerate((72, 74, 76, 77))]
    notes += [gate_note("grace", 2370, 15, 79, velocity=60),
              gate_note("main", 2400, 480, 77)]
    return document(notes, duration_ticks=3840, instrument="violin", articulations={"a": "arco"})


def test_mixed_visual_fixture_contains_staccato_and_pre_grace_together():
    result = build_notation(mixed_expression_document())
    xml = ET.fromstring(result.musicxml)
    assert len(xml.findall(".//staccato")) == 4
    assert len(xml.findall(".//grace")) == 1


def test_single_off_grid_pre_grace_has_real_symbol_and_does_not_pollute_time_axis():
    project = document(grace_pair())
    before = deepcopy(project.to_dict())
    result = build_notation(project)
    event = next(event for event in result.quantized_events if event.source_id == "grace")
    assert (event.start_beat, event.end_beat, event.grace_main_id) == (1, 1, "main")
    xml = ET.fromstring(result.musicxml)
    ornament = next(item for item in xml.findall(".//note") if item.find("grace") is not None)
    assert ornament.find("grace").get("slash") == "yes"
    assert ornament.find("duration") is None
    assert all(anchor.beat == pytest.approx(1) for anchor in result.anchors if anchor.kind == "note")
    assert "前倚音 1 个" in next(item.message for item in result.diagnostics if item.code == "auto-note-recognition")
    scene = compile_scene(project)
    assert scene.axis.beats == sorted(scene.axis.beats)
    assert (scene.parts[0].notes[0].start, scene.parts[0].notes[0].end) == (0.46875, 0.484375)
    assert project.to_dict() == before


def test_grace_follows_principal_across_piano_staff_split():
    result = build_notation(document(grace_pair(grace_pitch=59, main_pitch=61), instrument="piano"))
    xml = ET.fromstring(result.musicxml)
    ornament = next(item for item in xml.findall(".//note") if item.find("grace") is not None)
    assert ornament.findtext("staff") == "1"
    assert result.staff_part_ids == ["p", "p"]


@pytest.mark.parametrize("case", ["disabled", "fine_grid", "loud", "far_pitch", "same_pitch", "main_off_grid", "unknown", "long", "fast_run", "explicit_staccato"])
def test_grace_does_not_replace_plausible_regular_or_ambiguous_notes(case):
    notes = grace_pair()
    options = {}
    if case == "disabled":
        options["auto_grace"] = False
    elif case == "fine_grid":
        notes[0].duration_tick = 30
        notes[0].key_release_tick = 480
    elif case == "loud":
        notes[0].velocity = 110
    elif case == "far_pitch":
        notes[0].pitch = 79
    elif case == "same_pitch":
        notes[0].pitch = 72
    elif case == "main_off_grid":
        notes[1].start_tick = 510
        notes[1].key_release_tick = 990
    elif case == "unknown":
        notes[0].key_release_tick = None
    elif case == "long":
        notes[0].start_tick = 330
        notes[0].duration_tick = 135
    elif case == "fast_run":
        notes.insert(0, gate_note("prior-short", 400, 15, 75, velocity=65))
    elif case == "explicit_staccato":
        notes[0].articulation = "staccato"
        options["auto_staccato"] = False
    result = build_notation(document(notes, **options))
    assert not any(event.grace_main_id for event in result.quantized_events)
    assert ET.fromstring(result.musicxml).find(".//grace") is None


def test_pedal_sounding_lengths_do_not_replace_known_key_gates_for_inference():
    notes = regular_notes()
    for source in notes:
        source.duration_tick = 1920 - source.start_tick
    project = document(notes)
    result = build_notation(project)
    assert all(event.staccato for event in result.quantized_events)
    assert [event.end_beat for event in result.quantized_events] == [1, 2, 3, 4]
    assert [source.end_tick for source in notes] == [1920] * 4
