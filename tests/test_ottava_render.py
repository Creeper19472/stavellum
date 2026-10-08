"""Octave reminders follow engraved geometry in both CPU and Vulkan composition."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import replace
from fractions import Fraction

import pytest
from music21 import chord, clef, key, meter, note, spanner, stream
from native_frames import frame_layout
from PySide6.QtGui import QImage
from PySide6.QtSvg import QSvgRenderer
from test_rhi import FakeTarget

from stavellum.domain.models import (
    NoteEvent,
    PartMapping,
    ProjectDocument,
    ProjectIR,
    RenderSettings,
    TrackInfo,
)
from stavellum.engraving import notation
from stavellum.engraving.ottava import OctaveSpan
from stavellum.presentation.scene import _bounds, compile_scene
from stavellum.rendering.raster import RasterFrameRenderer
from stavellum.rendering.rhi import RhiFrameRenderer


@pytest.fixture
def octave_scene(monkeypatch):
    """Use genuine engraving with explicit spans, independent of recognition thresholds."""
    def build(octaves=1, *, neighbor=False, move_line=False, chords=False):
        tracks = [TrackInfo("neighbor", "Neighbor")] if neighbor else []
        tracks.append(TrackInfo("octave", "Octave"))
        midi = 84 if octaves > 0 else 36
        source = [NoteEvent(f"octave-{i}", "octave", i * 480, 480, midi) for i in range(32)]
        if neighbor:
            source += [NoteEvent(f"neighbor-{i}", "neighbor", i * 480, 480, 60) for i in range(32)]
        document = ProjectDocument(
            ProjectIR("", "midi", "Octave geometry", tracks=tracks, notes=source),
            [PartMapping(track.track_id, track.name, [track.track_id],
                         clef="treble" if octaves > 0 else "bass", key_signature=0, auto_ottava=True)
             for track in tracks],
            settings=RenderSettings(width=640, height=360, render_backend="cpu"),
        )
        score = stream.Score()
        for track in tracks:
            part = stream.Part(id=track.track_id)
            part.append(clef.TrebleClef() if octaves > 0 else clef.BassClef())
            part.append(key.KeySignature(0))
            part.append(meter.TimeSignature("4/4"))
            for i in range(32):
                written = midi - 12 * octaves if track.track_id == "octave" else 60
                item = chord.Chord([written, written + 3]) if chords else note.Note(written)
                item.quarterLength = 1
                part.append(item)
            part.makeNotation(inPlace=True)
            for i, item in enumerate(part.recurse().notes):
                item.id = f"probe_{track.track_id}_{i}"
                if isinstance(item, chord.Chord):
                    for member_index, member in enumerate(item.notes):
                        member.id = f"{item.id}_n{member_index}"
            score.insert(0, part)
        part = score.parts[-1]
        members = list(part.recurse().notes)
        endpoints = [item.notes[0] if isinstance(item, chord.Chord) else item
                     for item in (members[0], members[-1])]
        span = OctaveSpan("octave", 0, Fraction(0), Fraction(32), octaves,
                          str(endpoints[0].id), str(endpoints[-1].id))
        part.insert(0, spanner.Ottava(*members, type=span.label, transposing=True,
                                    placement="above" if octaves > 0 else "below"))
        value = notation._xml(score)
        toolkit = notation._toolkit(notation._display_xml(value))
        mei = toolkit.getMEI()
        svg = toolkit.renderToSVG(1)
        if move_line:
            root = ET.fromstring(svg)
            octave = next(element for element in root.iter() if element.get("class") == "octave")
            octave.set("transform", "translate(0,-3000)")
            svg = ET.tostring(root, encoding="unicode")
        staff_ids = [track.track_id for track in tracks]
        anchors = [notation.NotationAnchor(float(entry["qstamp"]), element)
                   for entry in toolkit.renderToTimemap() for element in entry.get("on", [])]
        result = notation.NotationResult(svg, score, [], staff_ids, value, anchors,
                                         element_part_ids=notation._element_owners(mei, staff_ids),
                                         octave_spans=[span], mei=mei)
        monkeypatch.setattr(notation, "build_notation", lambda _document: result)
        return compile_scene(document, use_cache=False)
    return build


@pytest.mark.parametrize("octaves", [1, -1, 2, -2])
def test_all_octave_labels_compile_separate_number_and_line_geometry(octave_scene, octaves):
    scene = octave_scene(octaves)
    mark, = scene.octave_spans
    assert mark.label == {1: "8va", -1: "8vb", 2: "15ma", -2: "15mb"}[octaves]
    assert mark.label_right < mark.end_x
    part = scene.parts[0]
    assert part.source_top < mark.label_y < part.source_bottom
    renderer = QSvgRenderer(scene.svg.encode())
    octave = next(element for element in ET.fromstring(scene.svg).iter()
                  if element.get("class") == "octave")
    bounds = _bounds(renderer, octave)
    assert part.source_top < bounds.top() < bounds.bottom() < part.source_bottom


def test_octave_bounds_keep_mei_owner_when_line_reaches_neighbor(octave_scene):
    scene = octave_scene(neighbor=True, move_line=True)
    parts = {part.part_id: part for part in scene.parts}
    renderer = QSvgRenderer(scene.svg.encode())
    octave = next(element for element in ET.fromstring(scene.svg).iter()
                  if element.get("class") == "octave")
    bounds = _bounds(renderer, octave)
    assert bounds.top() < parts["neighbor"].staff_centers[0]
    assert scene.element_part_ids[octave.get("id")] == "octave"
    assert parts["octave"].source_top < bounds.top()
    with RasterFrameRenderer(scene) as assets:
        assert assets.body_renderers["octave"].elementExists(octave.get("id"))
        assert not assets.body_renderers["neighbor"].elementExists(octave.get("id"))


def test_chord_member_endpoints_match_verovios_chord_and_last_note_anchors(octave_scene):
    scene = octave_scene(chords=True)
    mark, = scene.octave_spans
    assert mark.label == "8va" and mark.label_right < mark.end_x


def test_piano_octave_lines_compile_for_both_staves_of_one_logical_part():
    document = ProjectDocument(
        ProjectIR("", "midi", "Piano octaves", tracks=[TrackInfo("p", "Piano")],
                  notes=[NoteEvent(f"{i}_{midi}", "p", i * 480, 480, midi)
                         for i in range(16) for midi in (88, 24)]),
        [PartMapping("p", "Piano", ["p"], instrument="piano", key_signature=0, auto_ottava=True)],
        settings=RenderSettings(render_backend="cpu"),
    )
    scene = compile_scene(document)
    assert [(span.staff_index, span.label) for span in scene.octave_spans] == [(0, "8va"), (1, "8vb")]
    part, = scene.parts
    assert part.source_top < scene.octave_spans[0].label_y < part.staff_centers[0]
    assert part.staff_centers[1] < scene.octave_spans[1].label_y < part.source_bottom


def test_left_reminder_waits_for_whole_number_and_ends_with_line(octave_scene):
    scene = octave_scene()
    mark, = scene.octave_spans
    part = scene.parts[0]
    layout = frame_layout(scene, 5)
    offset = (scene.play_x - scene.body_left) / layout.scale
    with RasterFrameRenderer(scene) as assets:
        for left, present in [(mark.label_right - 0.01, False),
                              (mark.label_right + 0.01, True),
                              (mark.end_x - 0.01, True), (mark.end_x + 0.01, False)]:
            assert bool(assets._octave_overlays(part, layout, left + offset)) is present
        assert not assets._octave_overlays(part, layout, scene.camera_x_at(0))
        scene.octave_spans = []
        assert not assets._octave_overlays(part, layout, mark.label_right + 1 + offset)


def test_reminder_scales_with_row_and_disappears_when_hidden(octave_scene):
    scene = octave_scene()
    part = scene.parts[0]
    mark, = scene.octave_spans
    layout = frame_layout(scene, 5)
    left = (mark.label_right + mark.end_x) / 2
    with RasterFrameRenderer(scene) as assets:
        def overlays(frame_layout):
            world = left + (scene.play_x - scene.body_left) / frame_layout.scale
            return assets._octave_overlays(part, frame_layout, world)
        image, rect = overlays(layout)[0]
        smaller = replace(layout, scale=layout.scale / 2)
        other, reduced = overlays(smaller)[0]
        assert image.cacheKey() == other.cacheKey()
        assert reduced.width() == pytest.approx(rect.width() / 2)
        assert reduced.height() == pytest.approx(rect.height() / 2)
        assert reduced.left() - scene.body_left == pytest.approx((rect.left() - scene.body_left) / 2)
        hidden = replace(layout, rows={part.part_id: replace(layout.rows[part.part_id], opacity=0)})
        assert overlays(hidden) == []
        assert any(bytes(image.constBits())[3::4])
        assert 0 in bytes(image.constBits())[3::4]


def test_cpu_reminder_pixels_are_identical_after_random_seeking(octave_scene):
    scene = octave_scene()
    times = (0, 2, 4, 8, 12, 15)
    with RasterFrameRenderer(scene) as assets:
        def pixels(time):
            frame = assets.render_frame(time)
            return bytes(frame.constBits())

        expected = {time: pixels(time) for time in times}
        assert any(assets._octave_overlays(scene.parts[0], frame_layout(scene, time),
                                           scene.camera_x_at(time)) for time in times)
        for time in (12, 0, 15, 4, 2, 8, 0):
            assert pixels(time) == expected[time]
        mark, = scene.octave_spans
        labels = [replace(mark, label=label) for label in ("8va", "8vb", "15ma", "15mb")]
        scene.octave_spans = labels
        left = (mark.label_right + mark.end_x) / 2
        layout = frame_layout(scene, 5)
        assets._octave_overlays(scene.parts[0], layout,
                                left + (scene.play_x - scene.body_left) / layout.scale)
        assert len(assets._octave_labels) == 4


def test_rhi_uses_same_cached_reminder_asset_position_mask_and_opacity(octave_scene, monkeypatch):
    scene = octave_scene()
    part = scene.parts[0]
    mark, = scene.octave_spans
    layout = frame_layout(scene, 5)
    layout = replace(layout, rows={part.part_id: replace(layout.rows[part.part_id], opacity=0.4)})
    world = (mark.label_right + mark.end_x) / 2 + (scene.play_x - scene.body_left) / layout.scale
    monkeypatch.setattr("stavellum.rendering.rhi.RhiTarget", FakeTarget)
    with RasterFrameRenderer(scene) as assets, RhiFrameRenderer(scene, assets=assets) as renderer:
        state = assets._evaluator.evaluate(5)
        monkeypatch.setattr(assets._evaluator, "evaluate",
                            lambda _time: replace(state, layout=layout, world_x=world))
        image, rect = assets._octave_overlays(part, layout, world)[0]
        assert image.format() == QImage.Format.Format_RGBA8888
        commands = renderer.commands(5)
        label, = [quad for quad in commands if quad.texture_id == image.cacheKey()]
        assert (label.x, label.y, label.w, label.h) == pytest.approx(rect.getRect())
        assert label.a == pytest.approx(0.4)
        mask = assets._octave_mask(rect, layout)
        assert mask.left() == scene.body_left and mask.top() < rect.top()
        masks = [quad for quad in commands if quad.texture_id == 0 and quad.r == quad.g == quad.b == 0
                 and (quad.x, quad.y, quad.w, quad.h) == pytest.approx(mask.getRect())]
        assert len(masks) == 1 and masks[0].a == 1
        expected = (label.x, label.y, label.w, label.h, label.a)
        for time in (10, 0, 5):
            repeated, = [quad for quad in renderer.commands(time)
                         if quad.texture_id == image.cacheKey()]
            assert (repeated.x, repeated.y, repeated.w, repeated.h, repeated.a) == expected
        layout = replace(layout, rows={part.part_id: replace(layout.rows[part.part_id], opacity=0)})
        assert not any(quad.texture_id == image.cacheKey() for quad in renderer.commands(5))
