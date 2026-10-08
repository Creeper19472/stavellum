import xml.etree.ElementTree as ET
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import pytest
from music21 import chord, clef, converter, dynamics, key, note, spanner, stream
from pypdf import PdfReader

from stavellum.domain.models import (
    AutomationPoint,
    Metadata,
    NoteEvent,
    PartMapping,
    ProjectDocument,
    ProjectIR,
    TrackInfo,
    VolumeAutomation,
    VolumeRoute,
)
from stavellum.engraving.notation import (
    _spelled_pitch,
    _toolkit,
    _write_pdf,
    build_notation,
    export_parts,
)
from stavellum.graphics.svg import SVG_NS, normalize_svg


def document(notes, mappings=None, *, ppq=480, duration_ticks=0):
    tracks = list(dict.fromkeys(event.track_id for event in notes)) or ["track"]
    project = ProjectIR("", "midi", "test", ppq=ppq, tracks=[TrackInfo(t, t) for t in tracks], notes=notes, duration_ticks=duration_ticks)
    if mappings is None:
        mappings = [PartMapping(t, t, [t], key_signature=0) for t in tracks]
    return ProjectDocument(project, mappings, metadata=Metadata("Notation Test", composer="Composer"))


@pytest.mark.parametrize("fifths", range(-7, 8))
def test_simplified_spelling_preserves_all_midi_pitches_and_prefers_key_scale(fifths):
    signature = key.KeySignature(fifths)
    scale = {value.pitchClass: value for value in signature.asKey().getPitches()}
    for midi in range(128):
        value = _spelled_pitch(midi, signature)
        assert value.ps == midi
        assert value.midi == midi
        alteration = value.accidental.alter if value.accidental else 0
        assert abs(alteration) <= 1
        if midi % 12 in scale:
            expected = scale[midi % 12]
            assert value.step == expected.step
            assert alteration == (expected.accidental.alter if expected.accidental else 0)


@pytest.mark.parametrize(("fifths", "midi", "expected"), [
    (7, 0, "B#-2"), (7, 60, "B#3"), (7, 72, "B#4"),
    (6, 65, "E#4"), (7, 77, "E#5"),
    (-6, 11, "C-0"), (-6, 71, "C-5"), (-7, 59, "C-4"), (-7, 64, "F-4"),
])
def test_simplified_enharmonic_spelling_uses_the_correct_octave(fifths, midi, expected):
    value = _spelled_pitch(midi, key.KeySignature(fifths))
    assert value.nameWithOctave == expected
    assert value.ps == midi


@pytest.mark.parametrize(("fifths", "midi", "expected"), [
    (7, 60, "C4"), (6, 65, "F4"), (-6, 71, "B4"), (-7, 64, "E4"),
])
def test_disabling_simplification_retains_legacy_spelling(fifths, midi, expected):
    assert _spelled_pitch(midi, key.KeySignature(fifths),
                          simplify_accidentals=False).nameWithOctave == expected


@pytest.mark.parametrize(("fifths", "midi", "expected"), [
    (7, 64, "E4"), (6, 60, "C4"), (-7, 65, "F4"), (-6, 60, "C4"),
    (1, 63, "D#4"), (-1, 61, "D-4"), (0, 63, "E-4"),
])
def test_chromatic_pitches_keep_conventional_spelling(fifths, midi, expected):
    signature = key.KeySignature(fifths)
    assert _spelled_pitch(midi, signature).nameWithOctave == expected
    assert _spelled_pitch(midi, signature,
                          simplify_accidentals=False).nameWithOctave == expected


@pytest.mark.parametrize(("fifths", "pitches", "restoration"), [
    (7, [60, 59, 60, 65, 64, 65], "sharp"),
    (-7, [71, 72, 71, 64, 65, 64], "flat"),
])
def test_simplification_keeps_necessary_naturals_and_restores_key_accidentals(
    fifths, pitches, restoration,
):
    sources = [NoteEvent(str(index), "a", index * 480, 480, midi)
               for index, midi in enumerate(pitches)]
    mapping = PartMapping("a", "A", ["a"], key_signature=fifths)
    result = build_notation(document(sources, [mapping]))
    xml = ET.fromstring(result.musicxml)
    assert [element.text for element in xml.findall(".//accidental")] == [
        "natural", restoration, "natural", restoration]
    assert [item.pitch.ps for item in result.score.parts[0].recurse().notes] == pitches
    assert xml.findtext(".//key/fifths") == str(fifths)


@pytest.mark.parametrize("enabled", [True, False])
def test_simplification_is_shared_by_transposed_chords_graces_ties_and_both_piano_staves(enabled):
    sources = [
        NoteEvent("bass", "bass", 0, 2880, 46),
        NoteEvent("chord-low", "upper", 0, 480, 58),
        NoteEvent("chord-high", "upper", 0, 480, 63),
        NoteEvent("grace", "ornament", 930, 15, 63, velocity=65, key_release_tick=945),
        NoteEvent("main", "ornament", 960, 480, 64, key_release_tick=1440),
    ]
    mapping = PartMapping("p", "Piano", ["bass", "upper", "ornament"],
                          instrument="piano", key_signature=7, transpose=2,
                          auto_simplify_accidentals=enabled)
    source = document(sources, [mapping])
    before = deepcopy(source.to_dict())
    result = build_notation(source)
    upper, lower = result.score.parts
    cluster = next(item for item in upper.recurse().notes if isinstance(item, chord.Chord))
    assert [value.nameWithOctave for value in cluster.pitches] == (
        ["B#3", "E#4"] if enabled else ["C4", "F4"])
    ornament = next(item for item in upper.recurse().notes if item.duration.isGrace)
    assert ornament.nameWithOctave == ("E#4" if enabled else "F4")
    bass = list(lower.recurse().notes)
    assert [item.nameWithOctave for item in bass] == ["B#2" if enabled else "C3"] * 2
    assert [item.tie.type for item in bass] == ["start", "stop"]
    assert [item.pitch.ps for item in bass] == [48, 48]
    assert {event.source_id: event.pitch for event in result.quantized_events} == {
        "bass": 48, "chord-low": 60, "chord-high": 65, "grace": 65, "main": 66}
    assert next(event for event in result.quantized_events
                if event.source_id == "grace").grace_main_id == "main"
    xml = ET.fromstring(result.musicxml)
    assert xml.findtext(".//key/fifths") == "7"
    xml_ornament = next(item for item in xml.findall(".//note") if item.find("grace") is not None)
    assert xml_ornament.findtext("pitch/step") == ("E" if enabled else "F")
    assert xml_ornament.findtext("pitch/alter", "0") == ("1" if enabled else "0")
    assert xml_ornament.findtext("staff") == "1"
    svg_accidentals = [element for element in ET.fromstring(result.display_svg).iter()
                       if "accid" in element.get("class", "").split()
                       and any(child.tag in (f"{{{SVG_NS}}}use", f"{{{SVG_NS}}}path")
                               for child in element.iter())]
    assert bool(xml.findall(".//accidental")) is not enabled
    assert bool(svg_accidentals) is not enabled
    assert len(svg_accidentals) == len(xml.findall(".//accidental"))
    assert source.to_dict() == before


@pytest.mark.parametrize("continuous", [True, False])
def test_all_engraving_modes_use_leland_with_bravura_fallback(continuous):
    result = build_notation(document([NoteEvent("n", "a", 0, 480, 60)]))
    toolkit = _toolkit(result.musicxml, continuous=continuous)
    assert toolkit.getOptions()["font"] == "Leland"
    assert toolkit.getOptions()["fontFallback"] == "Bravura"
    root = ET.fromstring(toolkit.renderToSVG(1))
    paths = [node for node in root.iter() if node.tag == f"{{{SVG_NS}}}path"]
    assert paths
    # Check the outline itself: score symbols are paths, so the SVG does not
    # need to contain a CSS font-family declaration for this music font.
    reference = ET.parse(Path(toolkit.getResourcePath()) / "Leland" / "E0A4.xml")
    notehead_path = reference.find("path").get("d")
    assert any(path.get("d") == notehead_path for path in paths)


def test_polyphonic_voices_keep_shared_time_and_tie_across_bar():
    source = [NoteEvent("long", "a", 0, 2880, 60), NoteEvent("triplet", "a", 160, 160, 72)]
    result = build_notation(document(source))
    measures = list(result.score.parts[0].getElementsByClass(stream.Measure))
    assert len(measures) == 2
    assert [voice.id for voice in measures[0].voices] == [1, 2]
    long_notes = [n for n in result.score.parts[0].recurse().notes if isinstance(n, note.Note) and n.pitch.midi == 60]
    assert [n.tie.type for n in long_notes] == ["start", "stop"]
    assert any(anchor.kind == "note" and anchor.beat == pytest.approx(1 / 3) for anchor in result.anchors)
    assert max(anchor.beat for anchor in result.anchors) < 8
    assert source[1].start_tick == 160
    assert source[0].duration_tick == 2880


def test_chords_rests_dotted_notes_and_fixed_clef():
    result = build_notation(document([NoteEvent("c", "a", 240, 720, 60), NoteEvent("e", "a", 240, 720, 64), NoteEvent("g", "a", 240, 720, 67)]))
    notes = list(result.score.parts[0].recurse().notes)
    assert len(notes) == 1 and isinstance(notes[0], chord.Chord)
    assert notes[0].quarterLength == 1.5
    assert list(result.score.parts[0].recurse().getElementsByClass(clef.Clef))
    assert list(result.score.parts[0].recurse().getElementsByClass(note.Rest))
    assert "<defs>" in result.display_svg
    assert len([e for e in ET.fromstring(result.display_svg).iter() if e.get("class") == "system"]) == 1


def test_quantization_keyswitch_transpose_and_articulation_merge():
    notes = [NoteEvent("key", "normal", 0, 120, 24), NoteEvent("a", "normal", 10, 450, 60), NoteEvent("p", "pizz", 960, 480, 64)]
    mapping = PartMapping("violin1", "Violin I", ["normal", "pizz"], key_signature=-3, transpose=2, keyswitches=[24], articulations={"normal": "arco", "pizz": "pizz"})
    result = build_notation(document(notes, [mapping]))
    assert [event.pitch for event in result.quantized_events] == [62, 66]
    assert result.quantized_events[0].start_beat == 0
    assert result.quantized_events[1].articulation == "pizz."
    assert any(d.code == "quantization-drift" for d in result.diagnostics)
    assert "arco" in result.musicxml and "pizz." in result.musicxml
    assert notes[1].start_tick == 10 and notes[1].pitch == 60


def test_duplicate_merge_and_voice_identity_remain_separate():
    notes = [NoteEvent("n1", "one", 0, 480, 60), NoteEvent("n2", "pizz", 0, 480, 60), NoteEvent("n3", "two", 0, 480, 60)]
    mappings = [PartMapping("i", "Violin I", ["one", "pizz"], key_signature=0), PartMapping("ii", "Violin II", ["two"], key_signature=0)]
    result = build_notation(document(notes, mappings))
    assert result.staff_part_ids == ["i", "ii"]
    assert len(result.quantized_events) == 2
    assert any(d.code == "duplicate-notes" for d in result.diagnostics)


def test_piano_grandstaff_roundtrips_as_one_musicxml_instrument():
    notes = [NoteEvent("bass", "p", 0, 480, 40), NoteEvent("high", "p", 0, 480, 72)]
    result = build_notation(document(notes, [PartMapping("piano", "Piano", ["p"], instrument="piano", key_signature=0)]))
    assert result.staff_part_ids == ["piano", "piano"]
    xml = ET.fromstring(result.musicxml)
    assert len(xml.findall("part")) == 1
    assert xml.findtext("part/measure/attributes/staves") == "2"
    assert len([e for e in xml.findall(".//words") if (e.text or "").startswith("BPM ")]) == 1
    assert [c.sign for c in result.score.parts[0].recurse().getElementsByClass(clef.Clef)] == ["G"]
    assert [c.sign for c in result.score.parts[1].recurse().getElementsByClass(clef.Clef)] == ["F"]


def test_percussion_mapping_and_invalid_mapping():
    notes = [NoteEvent("s", "drum", 0, 480, 38), NoteEvent("h", "drum", 0, 480, 42)]
    mapping = PartMapping("drums", "Drums", ["drum"], percussion=True, percussion_map={"38": "C5", "42": "hi-hat"})
    result = build_notation(document(notes, [mapping]))
    xml = ET.fromstring(result.musicxml)
    assert xml.find(".//unpitched") is not None
    assert "x" in [e.text for e in xml.findall(".//notehead")]
    assert isinstance(list(result.score.parts[0].recurse().notes)[0], chord.ChordBase)
    mapping.percussion_map["38"] = "an invalid pitch"
    with pytest.raises(ValueError, match="打击乐映射"):
        build_notation(document(notes, [mapping]))


@pytest.mark.integration
def test_xml_and_paginated_pdf_export_roundtrip(tmp_path):
    notes = [NoteEvent(str(i), "a", i * 480, 480, 60 + i % 12) for i in range(512)]
    project = document(notes)
    progress = []
    details = []
    paths = export_parts(project, tmp_path, lambda fraction, message: progress.append(fraction),
                         progress_detail=details.append)
    assert len(paths) == 2 and progress[-1] == 1
    assert all(0 <= value <= 1 for value in progress)
    restored = converter.parse(paths[0])
    assert len(list(restored.recurse().notes)) == 512
    pdf = PdfReader(paths[1])
    assert len(pdf.pages) >= 2
    assert all(len(page.get_contents().get_data()) > 500 for page in pdf.pages)
    page_texts = [" ".join(page.extract_text().split()) for page in pdf.pages]
    assert all("Notation Test" in text for text in page_texts)
    assert "Composer" in pdf.pages[0].extract_text()
    assert all(f"{i + 1} / {len(pdf.pages)}" in text for i, text in enumerate(page_texts))
    assert all("â€" not in text for text in page_texts)
    assert not list(tmp_path.glob("*.tmp"))
    assert details[0].phase == "compile" and details[-1].phase == "done"
    assert details[-1].completed == details[-1].total == 1
    pdf_details = [item for item in details if item.phase == "pdf"]
    assert all(item.completed == 0 and item.unit == "parts" for item in pdf_details)
    assert pdf_details[-1].page == pdf_details[-1].pages == len(pdf.pages)
    assert any(item.phase == "save" and item.completed == 1 for item in details)


def test_pdf_page_progress_follows_drawing_and_supports_cancellation(tmp_path):
    class Toolkit:
        rendered = []

        def getPageCount(self):
            return 3

        def renderToSVG(self, page):
            self.rendered.append(page)
            return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
                    '<rect x="10" y="10" width="50" height="50" fill="black"/></svg>')

    toolkit = Toolkit()
    pages = []
    cancelled = False

    def progress(page, total):
        nonlocal cancelled
        pages.append((page, total))
        assert page == len(toolkit.rendered)
        cancelled = page == 1

    target = tmp_path / "cancelled.pdf"
    with pytest.raises(InterruptedError, match="取消分谱"):
        _write_pdf(toolkit, target, "Test", lambda: cancelled, page_progress=progress)
    assert pages == [(0, 3), (1, 3)]
    assert toolkit.rendered == [1]
    # A cancelled page callback still closes the painter and releases the file.
    target.unlink()


def test_cancel_before_part_export_leaves_no_partial_files(tmp_path):
    project = document([NoteEvent("n", "a", 0, 480, 60)])
    with pytest.raises(InterruptedError):
        export_parts(project, tmp_path, cancel=lambda: True)
    assert not list(tmp_path.iterdir())


def test_unusual_ppq_triplets_are_fraction_exact_and_all_rest_part_supported():
    project = document([NoteEvent("n", "a", 32, 32, 60)], ppq=96, duration_ticks=1536)
    result = build_notation(project)
    assert result.quantized_events[0].start_beat == pytest.approx(float(Fraction(1, 3)))
    assert len(list(result.score.parts[0].getElementsByClass(stream.Measure))) == 4
    empty = document([], duration_ticks=1920)
    result = build_notation(empty)
    assert any(d.code == "empty-part" for d in result.diagnostics)
    assert any(a.kind == "rest" for a in result.anchors)


def test_sanitized_part_ids_do_not_create_duplicate_svg_ids():
    notes = [NoteEvent("a", "a", 0, 480, 60), NoteEvent("b", "b", 0, 480, 72)]
    mappings = [PartMapping("violin 1", "I", ["a"], key_signature=0), PartMapping("violin_1", "II", ["b"], key_signature=0)]
    result = build_notation(document(notes, mappings))
    ids = [e.get("id") for e in ET.fromstring(result.display_svg).iter() if e.get("id")]
    assert len(ids) == len(set(ids))


def test_svg_text_keeps_position_style_and_avoids_private_font_glyphs():
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">
      <text font-size="0px"><tspan x="500" y="40" text-anchor="middle" font-style="italic"><tspan font-size="30px">标题</tspan></tspan></text>
      <text x="80" y="100" font-size="0px"><tspan font-family="Leipzig" font-size="60px">&#xeca5;</tspan><tspan font-size="30px"> = 120</tspan></text>
    </svg>'''
    root = ET.fromstring(normalize_svg(svg, foreground="black"))
    texts = root.findall(f"{{{SVG_NS}}}text")
    assert (texts[0].get("x"), texts[0].get("y"), texts[0].get("text-anchor")) == ("500", "40", "middle")
    assert texts[0].get("font-style") == "italic"
    assert texts[0].get("font-size") == "30px"
    assert not texts[0].findall(f"{{{SVG_NS}}}tspan/{{{SVG_NS}}}tspan")
    assert "".join(texts[1].itertext()) == "BPM 120"
    assert texts[1].get("font-size") == "30px"


def test_display_has_no_tempo_text_or_duplicate_arco_labels():
    notes = [NoteEvent("long", "bow", 0, 2880, 72), NoteEvent("short", "bow", 160, 160, 76), NoteEvent("pizz", "pizz", 1920, 480, 74)]
    mapping = PartMapping("v", "Violin", ["bow", "pizz"], articulations={"pizz": "pizz."}, key_signature=0)
    result = build_notation(document(notes, [mapping]))
    texts = [element.text for element in ET.fromstring(result.display_svg).iter() if element.tag.rsplit("}", 1)[-1] == "tspan" and element.text]
    assert "BPM" not in " ".join(texts)
    assert texts == ["arco", "arco / pizz.", "arco"]
    assert all(direction.get("placement") == "above" for direction in ET.fromstring(result.musicxml).findall(".//direction") if direction.find("direction-type/words") is not None and not direction.findtext("direction-type/words", "").startswith("BPM "))


def test_playing_directions_use_edwin_italic_in_musicxml_and_normalized_svg():
    notes = [NoteEvent("a", "bow", 0, 480, 72, articulation="arco"),
             NoteEvent("p", "bow", 1920, 480, 74, articulation="cresc.")]
    result = build_notation(document(notes))
    words = [word for word in ET.fromstring(result.musicxml).findall(".//words")
             if not (word.text or "").startswith("BPM ")]
    assert words
    assert all(word.get("font-family") == "Edwin" and word.get("font-style") == "italic"
               for word in words)
    texts = [element for element in ET.fromstring(normalize_svg(result.display_svg)).iter()
             if element.tag == f"{{{SVG_NS}}}tspan" and element.text]
    assert {element.text for element in texts} >= {"arco", "cresc."}
    assert all(element.get("font-family") == "Edwin" and element.get("font-style") == "italic"
               for element in texts)


def test_svg_text_inherits_italic_styles_and_preserves_explicit_normal_and_cjk():
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 200">
      <g style="font-style: italic; font-family: Arial">
        <text x="20" y="80" font-size="24">
          <tspan>cresc. </tspan><tspan font-style="normal">sempre </tspan>
          <tspan style="font-style: oblique"><tspan>dim. 渐弱</tspan></tspan>
        </text>
      </g>
    </svg>'''
    root = ET.fromstring(normalize_svg(svg))
    spans = [element for element in root.iter() if element.tag == f"{{{SVG_NS}}}tspan"]
    assert [(span.text, span.get("font-family"), span.get("font-style")) for span in spans] == [
        ("cresc. ", "Edwin", "italic"), ("sempre ", "Edwin", "normal"),
        ("dim. ", "Edwin", "italic"), ("渐弱", "Source Han Serif SC", "normal"),
    ]
    assert not root.findall(f".//{{{SVG_NS}}}tspan/{{{SVG_NS}}}tspan")


def test_interleaved_arco_pizz_phrase_does_not_stack_a_label_on_each_note():
    notes = [NoteEvent("sustain", "bow", 0, 3840, 60)]
    notes.extend(NoteEvent(f"pluck-{i}", "pizz", 120 + i * 480, 120, 72) for i in range(7))
    mapping = PartMapping("c", "Cello", ["bow", "pizz"], articulations={"bow": "arco", "pizz": "pizz."}, key_signature=0)
    result = build_notation(document(notes, [mapping]))
    words = [word.text for word in ET.fromstring(result.musicxml).findall(".//direction-type/words") if not (word.text or "").startswith("BPM ")]
    assert words == ["arco", "arco / pizz.", "arco"]
    assert any(d.code == "mixed-articulations" for d in result.diagnostics)


def test_mei_ownership_maps_system_ties_directions_and_tuplets_to_correct_part():
    notes = [NoteEvent("held-v", "v", 0, 2880, 72, articulation="arco"), NoteEvent("triplet", "v", 160, 160, 76), NoteEvent("held-c", "c", 0, 3840, 40)]
    result = build_notation(document(notes))
    seen = {"tie": [], "dir": [], "tuplet": []}
    for element in ET.fromstring(result.display_svg).iter():
        for category in seen:
            if category in element.get("class", "").split() and "bounding-box" not in element.get("class", ""):
                seen[category].append(result.element_part_ids.get(element.get("id")))
    assert set(seen["tie"]) == {"v", "c"}
    assert set(seen["dir"]) == {"v"}
    assert set(seen["tuplet"]) == {"v"}


def dynamic_document(*, grand=False):
    source = document([NoteEvent("held", "a", 0, 3840, 72)],
                      [PartMapping("a", "A", ["a"], instrument="piano" if grand else "violin", key_signature=0)])
    source.project.volume_routes = [VolumeRoute("a", ["channel:1"], {"channel:1": 1})]
    source.project.volume_automations = [VolumeAutomation("fade", "channel:1", "channel_volume",
                                                         [AutomationPoint(0, 0.2), AutomationPoint(840, 0.8)],
                                                         125.125, 965.125)]
    return source


def test_volume_hairpins_have_precise_anchors_inside_held_notes_and_keep_source_timing():
    source = dynamic_document()
    result = build_notation(source)
    anchors = list(result.score.parts[0].recurse().getElementsByClass(spanner.SpannerAnchor))
    assert [anchor.getOffsetInHierarchy(result.score.parts[0]) for anchor in anchors] == [
        Fraction("125.125") / 480, Fraction("965.125") / 480]
    wedges = list(result.score.parts[0].getElementsByClass(dynamics.DynamicWedge))
    assert len(wedges) == 1 and isinstance(wedges[0], dynamics.Crescendo)
    assert [event.duration_tick for event in source.project.notes] == [3840]
    xml = ET.fromstring(result.musicxml)
    assert [wedge.get("type") for wedge in xml.findall(".//wedge")] == ["crescendo", "stop"]
    assert all(direction.get("placement") == "below" for direction in xml.findall(".//direction")
               if direction.find("direction-type/wedge") is not None)
    hairpins = [element for element in ET.fromstring(result.display_svg).iter() if element.get("class") == "hairpin"]
    assert len(hairpins) == 1 and result.element_part_ids[hairpins[0].get("id")] == "a"


def test_grandstaff_volume_hairpin_appears_once_below_the_lower_staff():
    result = build_notation(dynamic_document(grand=True))
    assert not list(result.score.parts[0].getElementsByClass(dynamics.DynamicWedge))
    assert len(list(result.score.parts[1].getElementsByClass(dynamics.DynamicWedge))) == 1
    directions = [direction for direction in ET.fromstring(result.musicxml).findall(".//direction")
                  if direction.find("direction-type/wedge") is not None]
    assert len(directions) == 2
    assert all(direction.get("placement") == "below" and direction.findtext("staff") == "2"
               for direction in directions)
    assert len([element for element in ET.fromstring(result.display_svg).iter()
                if element.get("class") == "hairpin"]) == 1


def test_disabling_volume_recognition_preserves_the_automation_without_markings():
    source = dynamic_document()
    source.mappings[0].auto_dynamics = False
    result = build_notation(source)
    assert not ET.fromstring(result.musicxml).findall(".//wedge")
    assert len(source.project.volume_automations) == 1
    assert not any(diagnostic.code == "auto-volume-dynamics" for diagnostic in result.diagnostics)


def test_long_decimal_automation_offsets_are_stored_as_original_fractions():
    source = dynamic_document()
    source.project.volume_automations[0].start_tick = 125.3333333333
    result = build_notation(source)
    anchors = list(result.score.parts[0].recurse().getElementsByClass(spanner.SpannerAnchor))
    assert anchors[0].offset == Fraction("125.3333333333") / 480


def test_a_hairpin_end_on_the_barline_belongs_to_the_previous_measure():
    source = dynamic_document()
    source.project.notes[0].duration_tick = 5760
    source.project.volume_automations = [VolumeAutomation("boundary", "channel:1", "channel_volume",
                                                         [AutomationPoint(0, 0.2), AutomationPoint(1920, 0.8)],
                                                         1920, 3840)]
    result = build_notation(source)
    anchors = list(result.score.parts[0].recurse().getElementsByClass(spanner.SpannerAnchor))
    assert [(anchor.activeSite.number, anchor.offset) for anchor in anchors] == [(2, 0), (2, 4)]
    assert [anchor.getOffsetInHierarchy(result.score.parts[0]) for anchor in anchors] == [4, 8]
    measures = ET.fromstring(result.musicxml).findall("part/measure")
    assert [wedge.get("type") for wedge in measures[1].findall(".//wedge")] == ["crescendo", "stop"]
    assert not measures[2].findall(".//wedge")


def test_deduplicated_notes_still_check_both_audible_sources_for_volume_conflicts():
    source = document([NoteEvent("one", "a", 0, 1920, 72), NoteEvent("two", "b", 0, 1920, 72)],
                      [PartMapping("merged", "Merged", ["a", "b"], key_signature=0)])
    source.project.volume_routes = [VolumeRoute(track, [f"channel:{index}"], {f"channel:{index}": 1})
                                   for index, track in enumerate(("a", "b"), 1)]
    source.project.volume_automations = [VolumeAutomation("fade", "channel:1", "channel_volume",
                                                         [AutomationPoint(0, 0.2), AutomationPoint(1920, 0.8)],
                                                         0, 1920)]
    result = build_notation(source)
    assert len(result.quantized_events) == 1
    assert not ET.fromstring(result.musicxml).findall(".//wedge")
    assert any(diagnostic.code == "volume-dynamics-conflict" for diagnostic in result.diagnostics)
