"""Octave compression must preserve the shared score's performance semantics."""

import warnings
import xml.etree.ElementTree as ET
from copy import deepcopy
from fractions import Fraction

import pytest
from music21 import chord, converter, note, pitch, spanner
from pypdf import PdfReader

from stavellum.domain.models import NoteEvent, PartMapping, ProjectDocument, ProjectIR, TrackInfo
from stavellum.engraving.notation import _toolkit, _xml, build_notation, export_parts
from stavellum.presentation.scene import compile_scene


def document(pitches, *, starts=None, durations=None, **options):
    events = [NoteEvent(str(index), "a", (starts or [i * 480 for i in range(len(pitches))])[index],
                        (durations or [480] * len(pitches))[index], midi)
              for index, midi in enumerate(pitches)]
    options.setdefault("auto_ottava", True)
    mapping = PartMapping("p", "Part", ["a"], key_signature=0, **options)
    return ProjectDocument(ProjectIR("", "midi", "Octave test",
                                     tracks=[TrackInfo("a", "A")], notes=events), [mapping])


def xml_notes(value):
    result = []
    for item in ET.fromstring(value).findall(".//note"):
        sounding = item.find("pitch")
        if sounding is None:
            continue
        value = pitch.Pitch(sounding.findtext("step"))
        value.octave = int(sounding.findtext("octave"))
        value.accidental = int(sounding.findtext("alter", "0"))
        result.append((value.ps, item.findtext("duration"), item.findtext("voice"),
                       item.findtext("staff"), item.find("grace") is not None,
                       tuple(tie.get("type") for tie in item.findall("tie"))))
    return result


@pytest.mark.parametrize("pitches,clef_name,label,shift,direction,size", [
    ([88, 89, 91, 93], "treble", "8va", 1, "down", "8"),
    ([24, 26, 28, 29], "bass", "8vb", -1, "up", "8"),
    ([100, 101, 103, 105], "treble", "15ma", 2, "down", "15"),
    ([12, 14, 16, 17], "bass", "15mb", -2, "up", "15"),
])
def test_four_octave_lines_keep_exported_pitch_duration_and_source(
    pitches, clef_name, label, shift, direction, size,
):
    source = document(pitches, clef=clef_name)
    before = deepcopy(source.to_dict())
    result = build_notation(source)
    assert source.to_dict() == before
    assert len(result.octave_spans) == 1
    span = result.octave_spans[0]
    assert (span.part_id, span.staff_index, span.start_beat, span.end_beat, span.label) == (
        "p", 0, Fraction(0), Fraction(4), label)
    assert span.start_element_id and span.end_element_id
    assert [item.pitch.ps for item in result.score.recurse().notes] == [
        midi - 12 * shift for midi in pitches]
    line = list(result.score.recurse().getElementsByClass(spanner.Ottava))[0]
    assert line.type == label and line.transposing
    assert [event.pitch for event in result.quantized_events] == pitches
    root = ET.fromstring(result.musicxml)
    marks = root.findall(".//octave-shift")
    assert [(mark.get("type"), mark.get("size")) for mark in marks] == [
        (direction, size), ("stop", size)]
    assert any(item.find("direction-type/octave-shift") is not None
               for item in root.findall(".//direction"))
    mei_octaves = [item for item in ET.fromstring(result.mei).iter()
                   if item.tag.rsplit("}", 1)[-1] == "octave"]
    assert len(mei_octaves) == 1 and mei_octaves[0].get("staff") == "1"
    assert result.element_part_ids[mei_octaves[0].get("{http://www.w3.org/XML/1998/namespace}id")] == "p"
    assert any("octave" in item.get("class", "").split()
               for item in ET.fromstring(result.display_svg).iter())
    source.mappings[0].auto_ottava = False
    baseline = build_notation(source)
    assert not baseline.octave_spans
    assert xml_notes(result.musicxml) == xml_notes(baseline.musicxml)
    assert result.quantized_events == baseline.quantized_events
    restored = converter.parseData(result.musicxml)
    assert [item.pitch.ps for item in restored.toSoundingPitch().recurse().notes] == pitches
    # Exporting the derived score twice must not transpose it again in place.
    assert xml_notes(_xml(result.score)) == xml_notes(result.musicxml)
    assert line.transposing


def test_octave_binding_includes_every_cross_bar_tie_and_uses_the_latest_end():
    source = document([90, 91, 93, 95], starts=[0, 2880, 3360, 3840],
                      durations=[2880, 480, 480, 480], clef="treble")
    result = build_notation(source)
    line = list(result.score.recurse().getElementsByClass(spanner.Ottava))[0]
    tied = [item for item in result.score.recurse().notes if item.tie]
    assert [item.tie.type for item in tied] == ["start", "stop"]
    assert all(line.hasSpannedElement(item) for item in result.score.recurse().notes)
    assert result.octave_spans[0].end_beat == 9
    source.mappings[0].auto_ottava = False
    assert xml_notes(result.musicxml) == xml_notes(build_notation(source).musicxml)


@pytest.mark.parametrize("fifths", [-7, 7])
def test_octave_written_accidentals_keep_seven_alteration_keys_and_chromatic_boundaries(fifths):
    source = document([73, 88, 89, 89, 90, 76], clef="treble")
    source.mappings[0].key_signature = fifths
    result = build_notation(source)
    assert result.octave_spans
    shifted_items = [item for item in result.score.recurse().notes if isinstance(item, note.Note)]
    assert all(item.pitch.name for item in shifted_items)
    xml = ET.fromstring(result.musicxml)
    assert xml.findtext(".//key/fifths") == str(fifths)
    # The repeated F naturals after a sharp/key change must remain explicit.
    assert any(item.findtext("accidental") == "natural" for item in xml.findall(".//note"))
    source.mappings[0].auto_ottava = False
    assert xml_notes(result.musicxml) == xml_notes(build_notation(source).musicxml)


def test_octave_grace_inherits_main_shift_and_does_not_create_a_separate_span():
    source = document([88, 89, 91, 93], clef="treble")
    for event in source.project.notes:
        event.key_release_tick = event.end_tick
    source.project.notes[1].key_release_tick = 900
    source.project.notes.append(NoteEvent("grace", "a", 930, 15, 90,
                                          velocity=65, key_release_tick=945))
    result = build_notation(source)
    assert len(result.octave_spans) == 1
    grace = next(item for item in result.score.recurse().notes if item.duration.isGrace)
    assert grace.pitch.ps == 78
    line = list(result.score.recurse().getElementsByClass(spanner.Ottava))[0]
    assert line.hasSpannedElement(grace)
    source.mappings[0].auto_ottava = False
    baseline = build_notation(source)
    assert xml_notes(result.musicxml) == xml_notes(baseline.musicxml)


def test_first_grace_and_triplets_share_octave_line_and_written_pitch_context():
    source = document([88, 89, 91, 93, 95, 93],
                      starts=[480 + index * 320 for index in range(6)],
                      durations=[320] * 6, clef="treble", triplets=True,
                      auto_staccato=False)
    for event in source.project.notes:
        event.key_release_tick = event.end_tick
    source.project.notes.append(NoteEvent("grace", "a", 450, 15, 90,
                                          velocity=65, key_release_tick=465))
    result = build_notation(source)
    span, = result.octave_spans
    line = next(iter(result.score.recurse().getElementsByClass(spanner.Ottava)))
    assert line.getFirst().duration.isGrace
    assert span.start_element_id == line.getFirst().id
    xml = ET.fromstring(result.musicxml)
    assert xml.find(".//grace") is not None and xml.find(".//time-modification") is not None
    assert len(compile_scene(source).octave_spans) == 1
    source.mappings[0].auto_ottava = False
    assert xml_notes(result.musicxml) == xml_notes(build_notation(source).musicxml)


def test_both_piano_staves_and_transposed_chords_survive_independent_part_export(tmp_path):
    source = document([100, 101, 103, 105], instrument="piano", transpose=1)
    source.project.notes.extend(NoteEvent(f"b{i}", "a", i * 480, 480, midi)
                                for i, midi in enumerate([12, 14, 16, 17]))
    source.project.notes.extend(NoteEvent(f"c{i}", "a", i * 480, 480, midi)
                                for i, midi in enumerate([103, 105, 107, 108]))
    result = build_notation(source)
    assert {(span.staff_index, span.octaves) for span in result.octave_spans} == {(0, 2), (1, -2)}
    assert any(isinstance(item, chord.Chord) for item in result.score.recurse().notes)
    directions = [item for item in ET.fromstring(result.musicxml).findall(".//direction")
                  if item.find("direction-type/octave-shift") is not None]
    assert {(item.findtext("staff"), item.get("placement")) for item in directions} == {
        ("1", "above"), ("2", "below")}
    paths = export_parts(source, tmp_path)
    printed = next(path for path in paths if path.endswith(".musicxml"))
    with open(printed, encoding="utf-8") as handle:
        assert xml_notes(handle.read()) == xml_notes(result.musicxml)
    pdf = PdfReader(next(path for path in paths if path.endswith(".pdf")))
    assert pdf.pages and len(pdf.pages[0].get_contents().get_data()) > 500
    toolkit = _toolkit(result.musicxml, continuous=False)
    assert "octave" in toolkit.renderToSVG(1)


def test_percussion_and_conflicting_staff_voices_keep_the_original_notation():
    source = document([88, 89, 91, 93], clef="treble")
    source.project.notes.append(NoteEvent("held", "a", 0, 1920, 60))
    assert not build_notation(source).octave_spans
    source.mappings[0].percussion = True
    result = build_notation(source)
    assert not result.octave_spans
    assert "打击乐" in next(item.message for item in result.diagnostics if item.code == "auto-ottava")


def test_held_voice_does_not_close_the_visible_line_before_later_melody_notes():
    source = document([91, 93, 95, 91], durations=[240] * 4, clef="treble",
                      auto_staccato=False)
    source.project.notes.append(NoteEvent("held", "a", 0, 1920, 89))
    result = build_notation(source)
    span = result.octave_spans[0]
    line = next(iter(result.score.recurse().getElementsByClass(spanner.Ottava)))
    part = result.score.parts[0]
    last = line.getLast()
    assert last.getOffsetInHierarchy(part) + last.duration.quarterLength == 4
    final_head = max(part.recurse().notes, key=lambda item: item.getOffsetInHierarchy(part))
    assert span.end_element_id == final_head.id
    assert span.end_beat == 4
    assert all(line.hasSpannedElement(item) for item in result.score.recurse().notes)
    for item in ET.fromstring(result.mei).iter():
        if item.tag.rsplit("}", 1)[-1] == "note":
            assert item.get("oct") == "5" and item.get("oct.ges") == "6"
        elif item.tag.rsplit("}", 1)[-1] == "octave":
            assert item.get("endid") == "#" + span.end_element_id
    paginated = _toolkit(result.musicxml, continuous=False, octave_score=result.score,
                         octave_spans=result.octave_spans)
    for item in ET.fromstring(paginated.getMEI()).iter():
        if item.tag.rsplit("}", 1)[-1] == "note":
            assert item.get("oct") == "5" and item.get("oct.ges") == "6"
    source.mappings[0].auto_ottava = False
    assert xml_notes(result.musicxml) == xml_notes(build_notation(source).musicxml)


def test_overlapping_piano_octave_lines_use_distinct_numbers_across_measures():
    source = document([88, 89, 91, 93] * 4, instrument="piano")
    source.project.notes.extend(NoteEvent(f"b{i}", "a", i * 480, 480, midi)
                                for i, midi in enumerate([24, 26, 28, 29] * 4))
    result = build_notation(source)
    assert {(span.staff_index, span.octaves, span.end_beat) for span in result.octave_spans} == {
        (0, 1, 16), (1, -1, 16)}
    marks = ET.fromstring(result.musicxml).findall(".//octave-shift")
    assert {mark.get("number") for mark in marks if mark.get("type") != "stop"} == {"1", "2"}
    assert {mark.get("number") for mark in marks if mark.get("type") == "stop"} == {"1", "2"}
    octaves = [item for item in ET.fromstring(result.mei).iter()
               if item.tag.rsplit("}", 1)[-1] == "octave"]
    assert len(octaves) == 2 and all(item.get("endid") for item in octaves)
    # Verovio 6.3 warns while checking the other still-active staff's span.
    # Assert the actual ownership and written/sounding pitches instead.
    root = ET.fromstring(result.mei)
    xml_id = "{http://www.w3.org/XML/1998/namespace}id"
    owners = {}
    for staff in root.iter():
        if staff.tag.rsplit("}", 1)[-1] != "staff":
            continue
        for item in staff.iter():
            if item.tag.rsplit("}", 1)[-1] == "note":
                owners[item.get(xml_id)] = staff.get("n")
                expected = ("5", "6") if staff.get("n") == "1" else ("2", "1")
                assert (item.get("oct"), item.get("oct.ges")) == expected
    for mark in octaves:
        assert owners[mark.get("startid").lstrip("#")] == mark.get("staff")
        assert owners[mark.get("endid").lstrip("#")] == mark.get("staff")
    assert len(compile_scene(source).octave_spans) == 2


def test_second_voice_opening_grace_exports_start_before_first_voice_stop():
    source = document([95, 93, 95, 91], starts=[240, 360, 480, 600],
                      durations=[120, 120, 120, 960], clef="treble",
                      auto_staccato=False)
    source.project.notes.extend([
        NoteEvent("held", "a", 240, 720, 89, midi_channel=1),
        NoteEvent("grace", "a", 207, 15, 90, velocity=65, midi_channel=1),
    ])
    for event in source.project.notes:
        event.key_release_tick = event.end_tick
    result = build_notation(source)
    span, = result.octave_spans
    assert (span.start_beat, span.end_beat) == (Fraction(1, 2), Fraction(13, 4))
    line = next(iter(result.score.recurse().getElementsByClass(spanner.Ottava)))
    assert line.getFirst().duration.isGrace
    root = ET.fromstring(result.musicxml)
    cursor, divisions = Fraction(0), Fraction(root.findtext(".//divisions"))
    directions = []
    for element in root.find("part/measure"):
        if element.tag == "note":
            if element.find("grace") is None and element.find("chord") is None:
                cursor += Fraction(element.findtext("duration", "0")) / divisions
        elif element.tag in ("backup", "forward"):
            delta = Fraction(element.findtext("duration", "0")) / divisions
            cursor += delta if element.tag == "forward" else -delta
        elif element.tag == "direction":
            mark = element.find("direction-type/octave-shift")
            if mark is not None:
                directions.append((mark.get("type"), cursor
                                   + Fraction(element.findtext("offset", "0")) / divisions))
    assert directions == [("down", Fraction(1, 2)), ("stop", Fraction(13, 4))]
    with warnings.catch_warnings(record=True) as caught:
        restored = converter.parseData(result.musicxml)
    assert not any("octave-shift" in str(item.message) for item in caught)
    assert len(list(restored.recurse().getElementsByClass(spanner.Ottava))) == 1
    assert len(compile_scene(source).octave_spans) == 1
    for element in ET.fromstring(result.mei).iter():
        if element.tag.rsplit("}", 1)[-1] == "note":
            assert (element.get("oct"), element.get("oct.ges")) == ("5", "6")
    source.mappings[0].auto_ottava = False
    assert xml_notes(result.musicxml) == xml_notes(build_notation(source).musicxml)
