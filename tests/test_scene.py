"""Behavior checks against real engraved geometry, including sparse and dense bars."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET

import pytest
from native_frames import activity_level, activity_levels, frame_layout
from PySide6.QtSvg import QSvgRenderer

from stavellum.models import (
    Metadata,
    NoteEvent,
    PartMapping,
    ProjectDocument,
    ProjectIR,
    RenderSettings,
    TrackInfo,
)
from stavellum.qt import ensure_app
from stavellum.scene import (
    ActiveNote,
    ScenePart,
    _anchor_x,
    _axis_from_anchors,
    _svg_index,
    compile_scene,
    part_state,
)

pytestmark = pytest.mark.integration


def activity_part(notes):
    return ScenePart("activity", "Activity", "", 0, 100, [50], notes)


@pytest.mark.parametrize(("seconds", "expected"), [
    (0.9, (0, 0)), (1.0, (1, 1)), (1.05, (1, 0.5)), (1.1, (1, 0)),
    (2.9, (1, 0)), (3.0, (1, 0)), (3.06, (0.5, 0)), (3.12, (0, 0)),
])
def test_activity_long_note_has_a_100ms_onset_flash_and_120ms_release(seconds, expected):
    part = activity_part([ActiveNote(1, 3, 127)])
    assert activity_levels(part, seconds) == pytest.approx(expected)
    assert activity_level(part, seconds) == pytest.approx(expected[0])


def test_activity_short_note_flash_fades_with_its_release_tail():
    part = activity_part([ActiveNote(0, 0.02, 127)])
    level, attack = activity_levels(part, 0.05)
    assert level == pytest.approx(0.896484375)
    assert attack == pytest.approx(level / 2)
    assert 0 < attack < level < 1
    assert activity_levels(part, 0.1)[1] == 0
    assert activity_levels(part, 0.14) == (0, 0)


def test_activity_repeated_and_overlapping_notes_retrigger_flash_without_adding_velocities():
    part = activity_part([ActiveNote(0, 4, 100), ActiveNote(1, 2, 64), ActiveNote(1, 2, 80),
                          ActiveNote(3, 4, 64)])
    assert activity_levels(part, 0.5) == pytest.approx((100 / 127, 0))
    assert activity_levels(part, 1) == pytest.approx((100 / 127, 80 / 127))
    assert activity_levels(part, 1.05) == pytest.approx((100 / 127, 40 / 127))
    assert activity_levels(part, 2.5) == pytest.approx((100 / 127, 0))
    assert activity_levels(part, 3) == pytest.approx((100 / 127, 64 / 127))


def test_activity_future_notes_and_zero_velocity_do_not_light_or_flash():
    part = activity_part([ActiveNote(1, 2, 0), ActiveNote(3, 4, 127)])
    for seconds in (0, 1, 1.05, 2, 2.06, 2.99):
        assert activity_levels(part, seconds) == (0, 0)


def test_activity_zero_length_note_uses_release_tail_and_flash():
    part = activity_part([ActiveNote(1, 1, 127)])
    assert activity_levels(part, 0.99) == (0, 0)
    assert activity_levels(part, 1) == (1, 1)
    level, attack = activity_levels(part, 1.05)
    assert 0 < attack < level < 1
    assert attack == pytest.approx(level / 2)
    assert activity_levels(part, 1.12) == (0, 0)


def test_activity_levels_are_identical_after_random_seeking_and_note_reordering():
    notes = [ActiveNote(0, 0.02, 80), ActiveNote(0.3, 1.5, 127), ActiveNote(0.4, 0.8, 90)]
    part = activity_part(notes)
    seconds = (0.01, 0.05, 0.14, 0.3, 0.35, 0.4, 0.45, 0.51, 1.5, 1.56, 1.62)
    expected = {time: activity_levels(part, time) for time in seconds}
    for time in (1.56, 0.35, 0.01, 1.62, 0.4, 0.14, 1.5, 0.51, 0.05, 0.45, 0.3, 0.35):
        assert activity_levels(part, time) == expected[time]
        assert activity_levels(activity_part(list(reversed(notes))), time) == expected[time]
        level, attack = expected[time]
        assert 0 <= attack <= level <= 1


def score_document(
    *, target_notes=((0, 0.25), (32, 1)), dense=True, bars=16, offset=0.0,
) -> ProjectDocument:
    """A continuously playing carrier makes the target's rests visible in context."""
    notes = []
    step = 0.25 if dense else 2
    for index in range(int(bars * 4 / step)):
        notes.append(NoteEvent(f"carrier-{index}", "carrier", round(index * step * 480),
                               round(step * 480), [60, 64, 67, 72][index % 4], 80))
    notes.extend(NoteEvent(f"target-{index}", "target", round(start * 480),
                           round(duration * 480), 79, 100)
                 for index, (start, duration) in enumerate(target_notes))
    return ProjectDocument(
        ProjectIR("test.mid", "midi", "Visibility", tracks=[
            TrackInfo("carrier", "Carrier"), TrackInfo("target", "Target"),
        ], notes=notes, duration_ticks=bars * 4 * 480),
        [PartMapping("carrier", "Carrier", ["carrier"], clef="treble", key_signature=0),
         PartMapping("target", "Target", ["target"], clef="treble", key_signature=0)],
        settings=RenderSettings(width=640, height=360, fps=30,
                                score_start_in_audio_sec=offset),
        metadata=Metadata(title="Test"),
    )


def bar_on_screen(scene, bar: int, seconds: float) -> tuple[float, float]:
    """Observe where a real engraved bar lands in the shared scrolling viewport."""
    left, right = scene.measure_bounds[bar]
    now = scene.camera_x_at(seconds)
    return tuple(scene.play_x + (x - now) * frame_layout(scene, seconds).scale for x in (left, right))


def first_bar_entry(scene, bar: int, maximum: float) -> float:
    # Search the displayed geometry, rather than derive visibility from its spans.
    if bar_on_screen(scene, bar, 0)[0] <= scene.body_right:
        return 0.0
    low, high = 0.0, maximum
    assert bar_on_screen(scene, bar, high)[0] < scene.body_right
    for _ in range(40):
        middle = (low + high) / 2
        if bar_on_screen(scene, bar, middle)[0] > scene.body_right:
            low = middle
        else:
            high = middle
    return high


@pytest.fixture(scope="module")
def dense_scene():
    ensure_app()
    return compile_scene(score_document())


def test_natural_width_axis_roundtrips_and_tracks_share_columns(dense_scene):
    axis = dense_scene.axis
    for beat in (0, 0.25, 2.125, 4, 19.375, 32, 63.875, 64):
        assert axis.beat_at(axis.x_at(beat)) == pytest.approx(beat, abs=1e-7)
    assert all(right > left for left, right in zip(axis.xs, axis.xs[1:]))
    assert len({part.part_id for part in dense_scene.parts}) == 2
    # Dense rhythm gets a wider engraved measure than a half-note measure.
    sparse = compile_scene(score_document(dense=False))
    dense_width = dense_scene.measure_bounds[2][1] - dense_scene.measure_bounds[2][0]
    sparse_width = sparse.measure_bounds[2][1] - sparse.measure_bounds[2][0]
    assert dense_width > sparse_width * 1.5


def test_whole_bar_rest_center_does_not_anchor_bar_start():
    document = score_document(target_notes=((24, 1),), dense=False, bars=8)
    document.project.notes = [note for note in document.project.notes if note.track_id == "target"]
    scene = compile_scene(document)
    svg = ET.fromstring(scene.svg)
    measure = [node for node in svg.iter() if "measure" in node.get("class", "").split()][1]
    rest = next(node for node in measure.iter() if "mRest" in node.get("class", "").split())
    renderer = QSvgRenderer(scene.svg.encode("utf-8"))
    rest_id = rest.attrib["id"]
    bounds = renderer.transformForElement(rest_id).mapRect(renderer.boundsOnElement(rest_id))
    # An empty bar starts before its centered rest, so the playback line reaches
    # the center later than the bar's first beat.
    assert scene.axis.x_at(4) < bounds.left()
    assert scene.axis.beat_at(bounds.center().x()) > 4.5
    assert scene.tempo_mark.x == scene.axis.x_at(0)


def test_empty_opening_keeps_the_first_staff_until_its_tempo_scrolls_out():
    document = score_document(target_notes=((96, .25),), bars=32)
    document.project.notes = [event for event in document.project.notes
                              if event.track_id != "carrier" or event.start_tick >= 64 * 480]
    scene = compile_scene(document)
    assert scene.tempo_mark.owner_id == "carrier"
    assert frame_layout(scene, 0).rows["carrier"].opacity == 1
    assert all(event.start_tick >= 64 * 480 for event in document.project.notes)
    exit_time = scene.layout.tempo_exit_time
    assert frame_layout(scene, exit_time - 1e-5).rows["carrier"].opacity == 1
    assert frame_layout(scene, exit_time + 1).rows["carrier"].opacity == 0
    assert scene.layout.padding_at("carrier", exit_time + 1) == 0


def test_bar_fallbacks_cannot_displace_a_later_actual_note():
    axis = _axis_from_anchors({168: 214566, 169: 215888, 170.75: 218194},
                             {168.5: 217894, 172: 219994})
    assert 168.5 not in axis.beats
    assert axis.x_at(169) == 215888
    assert axis.beat_at(215888) == 169


def test_conflicting_actual_onsets_are_reported_instead_of_silently_dropped():
    with pytest.raises(ValueError, match="音符拍位发生横向冲突"):
        _axis_from_anchors({0: 100, 1: 90, 2: 300}, {4: 500})


def test_long_rest_geometry_never_removes_other_parts_note_onsets(monkeypatch):
    from stavellum import notation as notation_module

    document = score_document(target_notes=((0, .5), (3, .5)), bars=2)
    engraved = notation_module.build_notation(document)
    monkeypatch.setattr(notation_module, "build_notation", lambda _: engraved)
    scene = compile_scene(document)
    root = ET.fromstring(scene.svg)
    index, transforms = _svg_index(root)
    renderer = QSvgRenderer(scene.svg.encode("utf-8"))
    expected = {}
    for anchor in engraved.anchors:
        if anchor.kind == "note":
            beat = round(anchor.beat, 8)
            x = _anchor_x(index[anchor.element_id], transforms, renderer)
            expected[beat] = min(x, expected.get(beat, x))
    assert any(anchor.kind == "rest" for anchor in engraved.anchors)
    for beat, x in expected.items():
        assert scene.axis.x_at(beat) == pytest.approx(x)


def test_camera_uses_presentation_time_and_freezes_during_intro():
    document = score_document(bars=2)
    document.settings.intro_delay_seconds = 2
    scene = compile_scene(document)
    assert scene.camera_x_at(0) == scene.camera_x_at(1.5)
    assert scene.camera_x_at(2) == scene.camera.x_at(0)
    assert scene.camera_x_at(2.75) == scene.camera.x_at(.75)
    assert scene.camera_speed_at(1.5) == 0
    assert scene.camera_min_speed(1.5, 2.5) == 0
    assert scene.camera_min_speed(2, 3) > 0


def test_current_beat_stays_in_left_corridor_at_every_zoom():
    scene = compile_scene(score_document())
    left, right = scene.play_corridor
    for index in range(2000):
        time = index / 60
        exact = scene.axis.x_at(scene.beat_at_time(scene.settings.audio_time(time)))
        displayed = scene.play_x + (exact - scene.camera_x_at(time)) * frame_layout(scene, time).scale
        assert left - 1e-6 <= displayed <= right + 1e-6


def test_tiny_score_body_has_a_safe_zero_width_camera_corridor():
    document = score_document(bars=2)
    document.settings.score_right = document.settings.score_left + document.settings.header_width + .001
    scene = compile_scene(document)
    assert scene.body_left <= scene.play_x <= scene.body_right
    assert scene.play_corridor[0] == scene.play_corridor[1]
    assert scene.camera.half_window_seconds == 0


def test_part_state_preserves_the_layout_curves_derivatives_after_interruption():
    scene = compile_scene(score_document(bars=2))
    part = scene.parts[0]
    curve = scene.layout.opacities[part.part_id]
    assert part.opacity_curve is curve
    curve.transition(.1, .6, 0)
    curve.transition(.3, .8, 1)
    part.opacity_keys = curve.opacity_keys()
    assert curve.sample(.3).velocity != 0
    for time in (.15, .299, .3, .301, .45, .65, .8, 1):
        assert part_state(part, time, scene.settings)[0] == frame_layout(scene, time).rows[part.part_id].opacity


def test_single_instrument_remains_visible_through_silence_and_tail():
    document = score_document(target_notes=((40, 1),))
    document.mappings = [document.mappings[1]]
    scene = compile_scene(document)
    assert len(scene.parts) == 1
    for seconds in (0, 3, 12, 20, 31, 90):
        assert part_state(scene.parts[0], seconds, scene.settings) == (1.0, 1.0)


def test_sustained_note_retains_part_until_its_last_bar_passes():
    scene = compile_scene(score_document(target_notes=((0, 48),)))
    target = scene.parts[1]
    for seconds in (0.5, 6, 12, 23):
        assert part_state(target, seconds, scene.settings) == (1.0, 1.0)
        assert activity_level(target, seconds) > 0
    assert part_state(target, 30, scene.settings) == (0.0, 0.0)


def test_initially_hidden_part_waits_for_measure_entry_despite_lookahead():
    scene = compile_scene(score_document(target_notes=((32, 1),)))
    target = scene.parts[1]
    entry = first_bar_entry(scene, 8, maximum=16)
    before = entry - scene.settings.enter_seconds - 0.1
    assert before > 0
    # Four-bar lookahead already includes the note while the part is hidden.
    assert scene.beat_at_time(before) + 16 > 32
    assert bar_on_screen(scene, 8, before)[0] > scene.body_right
    assert part_state(target, before, scene.settings) == (0.0, 0.0)
    # The complete slot is prepared after actual entry, then the row fades in.
    assert part_state(target, entry + scene.settings.reflow_seconds / 2, scene.settings) == (0.0, 0.0)
    fade = next((first, last) for first, last in zip(target.opacity_curve.keys, target.opacity_curve.keys[1:])
                if first.value == 0 and last.value > 0)
    assert fade[0].time >= entry
    during = (fade[0].time + fade[1].time) / 2
    assert 0 < part_state(target, during, scene.settings)[0] < 1
    after = fade[1].time + 0.01
    assert part_state(target, after, scene.settings) == (1.0, 1.0)


def test_already_visible_part_hides_when_visible_bars_and_lookahead_are_empty(dense_scene):
    target = dense_scene.parts[1]
    assert part_state(target, 0, dense_scene.settings) == (1.0, 1.0)
    assert bar_on_screen(dense_scene, 0, 4)[1] < dense_scene.body_left
    assert dense_scene.beat_at_time(4) + 16 < 32
    assert part_state(target, 4, dense_scene.settings) == (0.0, 0.0)


def test_already_visible_part_keeps_four_bar_lookahead_outside_viewport():
    scene = compile_scene(score_document(target_notes=((0, 0.25), (20, 1))))
    target = scene.parts[1]
    seconds = 2.5
    assert bar_on_screen(scene, 0, seconds)[1] < scene.body_left
    assert bar_on_screen(scene, 5, seconds)[0] > scene.body_right
    assert scene.beat_at_time(seconds) < 20 < scene.beat_at_time(seconds) + 16
    assert activity_level(target, seconds) == 0
    assert part_state(target, seconds, scene.settings) == (1.0, 1.0)


def test_visible_distant_measure_takes_priority_over_four_bar_limit():
    document = score_document(target_notes=((0, 0.25), (24, 1)), dense=False)
    document.settings.staff_scale = 0.5  # A wide actual viewport extends beyond four bars.
    scene = compile_scene(document)
    target = scene.parts[1]
    seconds = 3.0
    now_beat = scene.beat_at_time(seconds)
    assert now_beat + 16 < 24
    assert bar_on_screen(scene, 0, seconds)[1] < scene.body_left
    left, right = bar_on_screen(scene, 6, seconds)
    assert left < scene.body_right and right > scene.body_left
    assert activity_level(target, seconds) == 0
    assert part_state(target, seconds, scene.settings) == (1.0, 1.0)


def test_piano_double_staff_is_one_visible_instrument():
    document = score_document(target_notes=())
    document.mappings = [PartMapping("piano", "Piano", ["carrier"], instrument="piano",
                                    grand_staff=True, key_signature=0)]
    document.project.notes = [NoteEvent("low", "carrier", 32 * 480, 480, 36),
                              NoteEvent("high", "carrier", 32 * 480, 480, 84)]
    scene = compile_scene(document)
    assert len(scene.parts) == 1
    assert len(scene.parts[0].staff_centers) == 2
    assert scene.parts[0].staff_centers[0] < scene.parts[0].staff_centers[1]
    assert len(scene.parts[0].notes) == 2
    assert part_state(scene.parts[0], 0, scene.settings) == (1.0, 1.0)


@pytest.mark.parametrize("offset", [-2.0, 3.0])
def test_audio_offset_translates_activity_and_visibility(dense_scene, offset):
    document = score_document(offset=offset)
    shifted = compile_scene(document)
    assert shifted.beat_at_time(offset) == 0
    assert shifted.parts[1].notes[1].start == pytest.approx(16 + offset)
    for seconds in (4, 10, 14, 16, 16.05, 16.1, 20):
        assert part_state(shifted.parts[1], seconds + offset, shifted.settings) == pytest.approx(
            part_state(dense_scene.parts[1], seconds, dense_scene.settings), abs=1e-7
        )
        assert activity_level(shifted.parts[1], seconds + offset) == pytest.approx(
            activity_level(dense_scene.parts[1], seconds)
        )
        assert activity_levels(shifted.parts[1], seconds + offset) == pytest.approx(
            activity_levels(dense_scene.parts[1], seconds)
        )


def test_negative_offset_discards_visibility_windows_that_end_before_audio_start():
    scene = compile_scene(score_document(target_notes=((0, 0.25),), offset=-20))
    target = scene.parts[1]
    assert target.notes[0].end < 0
    assert not target.spans
    assert activity_level(target, 0) == 0
    assert part_state(target, 0, scene.settings) == (0.0, 0.0)


def test_slide_is_retained_with_diagnostic_and_explicit_keyswitch_is_filtered():
    document = score_document(target_notes=((32, 1),))
    document.project.notes[0].slide = True
    original = copy.deepcopy(document.project.notes)
    document.mappings[1].keyswitches = [79]
    scene = compile_scene(document)
    assert len(scene.parts[0].notes) == len([n for n in original if n.track_id == "carrier"])
    assert activity_level(scene.parts[0], 0.05) > 0
    assert any("slide" in diagnostic.code.casefold() for diagnostic in scene.diagnostics)
    assert not scene.parts[1].notes
    assert not scene.parts[1].spans
    for seconds in (0, 12, 16, 20):
        assert part_state(scene.parts[1], seconds, scene.settings) == (0.0, 0.0)
        assert activity_level(scene.parts[1], seconds) == 0
    assert document.project.notes == original


def crosspart_document():
    """Extremes in different bars create overlapping whole-song vertical extents."""
    return ProjectDocument(
        ProjectIR("test.mid", "midi", "Cross-part ledger lines", tracks=[
            TrackInfo("upper", "Upper"), TrackInfo("lower", "Lower"),
        ], notes=[
            NoteEvent("upper-normal", "upper", 0, 480, 72, articulation="arco"),
            NoteEvent("upper-low", "upper", 1920, 480, 24, articulation="pizz."),
            NoteEvent("lower-high", "lower", 0, 480, 108, articulation="pizz."),
            NoteEvent("lower-normal", "lower", 1920, 480, 48, articulation="arco"),
        ], duration_ticks=7680),
        [PartMapping("upper", "Upper", ["upper"], clef="treble", key_signature=0),
         PartMapping("lower", "Lower", ["lower"], clef="bass", key_signature=0)],
    )


def owned_elements(scene):
    result = []

    def visit(element, owner=None):
        owner = scene.element_part_ids.get(element.get("id", ""), owner)
        if owner:
            result.append((element, owner))
        for child in element:
            visit(child, owner)

    visit(ET.fromstring(scene.svg))
    return result


def test_crosspart_extremes_preserve_complete_owned_glyph_bounds():
    scene = compile_scene(crosspart_document())
    upper, lower = scene.parts
    assert upper.source_bottom > lower.source_top + 500
    renderer = QSvgRenderer(scene.svg.encode("utf-8"))
    parts = {part.part_id: part for part in scene.parts}
    directions = set()
    notes = set()
    for element, owner in owned_elements(scene):
        classes = set(element.get("class", "").split())
        if not classes & {"note", "dir", "tie", "staff"}:
            continue
        identity = element.attrib["id"]
        bounds = renderer.transformForElement(identity).mapRect(renderer.boundsOnElement(identity))
        assert not bounds.isEmpty()
        assert bounds.top() >= parts[owner].source_top
        assert bounds.bottom() <= parts[owner].source_bottom
        if "dir" in classes:
            directions.add(owner)
        if "note" in classes:
            notes.add(owner)
    assert directions == notes == {"upper", "lower"}
