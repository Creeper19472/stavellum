"""Layout invariants observable across admissions, fades and random seeks."""

from __future__ import annotations

import copy
import math
import pickle
import subprocess
import sys

import pytest
from native_frames import activity_level, frame_layout

from stavellum.layout import _CurveKey, _Track
from stavellum.models import Metadata, RenderSettings
from stavellum.scene import (
    ActiveNote,
    CompiledScene,
    ScenePart,
    TimeAxis,
    compile_visibility,
    part_state,
)


def layout_scene(*, count=10, bpm=120, late=(), grand=False, stable_seconds=2.0):
    settings = RenderSettings(animation_stable_seconds=stable_seconds)
    seconds = 60 / bpm
    parts = []
    for index in range(count):
        is_grand = grand and index == count - 1
        height = 2400 if is_grand else 1000 + index * 20
        centers = [500, 1900] if is_grand else [500 + index * 10]
        notes = [ActiveNote(0, (64 if index == 0 else 1) * seconds, 100)]
        parts.append(ScenePart(str(index), str(index), "piano" if is_grand else "violin", 0, height, centers, notes))
    for index, beat in enumerate(late):
        parts.append(ScenePart(f"late-{index}", "Late", "violin", 0, 1000, [500],
                               [ActiveNote(beat * seconds, (beat + 1) * seconds, 100)]))
    scene = CompiledScene("", (0, 0, 64000, 40000), TimeAxis([0, 64], [0, 64000]),
                          parts, settings, Metadata(), bpm, 4, 64 * seconds,
                          2 * 12 / 180, 0, 200, [], [(index * 4000, (index + 1) * 4000) for index in range(16)])
    compile_visibility(scene)
    return scene


def shown(layout):
    return [row for row in layout.rows.values() if row.opacity > 1e-8]


def test_hidden_rows_recenter_and_survivors_enlarge_to_the_common_cap():
    scene = layout_scene()
    # A slow zoom can outlast a short excerpt. Keep the survivor playing until
    # the rate-limited geometry finishes instead of imposing a five-second zoom.
    scene.parts[0].notes[0].end = 1000
    compile_visibility(scene)
    initial = frame_layout(scene, 0)
    sparse = frame_layout(scene, scene.layout.times[-1])
    assert len(shown(initial)) == 10
    assert len(shown(sparse)) == 1
    assert sparse.scale > initial.scale * 2
    assert sparse.scale == pytest.approx(2 * 12 / 180)
    # Only the opening group contains the tempo; the surviving staff later
    # recenters using its own content and enlarged activity indicator.
    center = (scene.settings.score_top + scene.settings.score_bottom) * scene.settings.height / 2
    for layout in (initial, sparse):
        assert sum(layout.bounds) / 2 == pytest.approx(center, abs=0.5)
    assert center == pytest.approx(0.44 * scene.settings.height)


def test_visible_rows_and_scaled_indicators_never_overlap_or_leave_the_region():
    scene = layout_scene(count=8, late=(24, 28), grand=True)
    scene.parts[0].notes[0].end = 1000
    compile_visibility(scene)
    sizes = set()
    region_top = scene.settings.score_top * scene.settings.height
    region_bottom = scene.settings.score_bottom * scene.settings.height
    for index in range(2200):
        layout = frame_layout(scene, index / 120)
        rows = sorted(shown(layout), key=lambda row: row.top)
        assert layout.scale <= scene.scale
        for row in rows:
            sizes.add(row.indicator_rect[2:])
            assert row.bounds[0] >= region_top - 1e-6
            assert row.bounds[1] <= region_bottom + 1e-6
            assert row.icon_size * 19 / 16 < row.indicator_rect[3]
        assert all(first.bounds[1] <= second.bounds[0] + 1e-6 for first, second in zip(rows, rows[1:]))
    assert len(sizes) > 1


def test_incoming_row_waits_for_its_actual_bar_then_for_a_complete_slot():
    scene = layout_scene(count=1, late=(32,))
    incoming = scene.parts[-1]
    alpha_keys = incoming.opacity_keys
    fade_start = next(first[0] for first, last in zip(alpha_keys, alpha_keys[1:]) if first[1] == 0 and last[1] > 0)
    preparation_start = fade_start - scene.settings.reflow_seconds
    layout = frame_layout(scene, preparation_start)
    world = scene.camera_x_at(preparation_start)
    actual_entry_x = scene.play_x + (scene.measure_bounds[8][0] - world) * layout.scale
    assert actual_entry_x == pytest.approx(scene.body_right, abs=0.001)
    assert part_state(incoming, preparation_start - 0.01, scene.settings) == (0, 0)
    assert part_state(incoming, fade_start - 0.001, scene.settings) == (0, 0)
    assert 0 < part_state(incoming, fade_start + 0.01, scene.settings)[0] < 1
    assert frame_layout(scene, fade_start + scene.settings.enter_seconds).rows[incoming.part_id].opacity == 1


def test_fast_admissions_are_queued_without_losing_their_note_onsets():
    scene = layout_scene(count=1, bpm=900, late=(24, 28, 32))
    for part in scene.parts[1:]:
        row = frame_layout(scene, part.notes[0].start + 0.001).rows[part.part_id]
        assert row.opacity == pytest.approx(1.0)
    for index in range(500):
        rows = sorted(shown(frame_layout(scene, index / 240)), key=lambda row: row.top)
        assert all(first.bounds[1] <= second.bounds[0] + 1e-6 for first, second in zip(rows, rows[1:]))


def test_timeline_is_pickleable_seekable_and_independent_of_export_fps():
    scene = layout_scene(count=6, late=(32,))
    serialized = pickle.loads(pickle.dumps(scene))
    expected = {time: frame_layout(scene, time) for time in (0, 2.25, 5, 12.1, 13, 18, 40)}
    for time in (40, 0, 13, 2.25, 18, 5, 12.1, 0):
        assert frame_layout(serialized, time) == expected[time]
    alternative = copy.deepcopy(scene)
    alternative.settings.fps = 25
    compile_visibility(alternative)
    assert alternative.layout.keyframes == scene.layout.keyframes
    assert len(scene.layout.keyframes) < 40


def test_rectangles_match_between_single_and_grand_staff_even_after_zoom():
    scene = layout_scene(count=2, grand=True)
    first = frame_layout(scene, 0)
    assert first.rows["0"].indicator_rect[2:] == first.rows["1"].indicator_rect[2:]
    assert first.rows["0"].indicator_rect[2:] == pytest.approx((195 * first.scale, 720 * first.scale))
    sparse = frame_layout(scene, 5)
    assert sparse.rows["0"].indicator_rect[2:] == sparse.rows["1"].indicator_rect[2:]
    assert sparse.rows["0"].indicator_rect[2:] == pytest.approx((195 * sparse.scale, 720 * sparse.scale))


def test_sparse_rows_enlarge_the_lamp_and_icon_continuously():
    scene = layout_scene(count=10)
    scene.parts[0].notes[0].end = 1000
    compile_visibility(scene)
    dense = frame_layout(scene, 0)
    sparse = frame_layout(scene, scene.layout.times[-1])
    first, last = dense.rows["0"], sparse.rows["0"]
    assert last.indicator_rect[3] == pytest.approx(96)
    assert last.indicator_rect[2] == pytest.approx(26)
    assert last.icon_size == pytest.approx(60)
    assert last.indicator_rect[3] > first.indicator_rect[3] * 2
    for key in scene.layout.zoom.keys[1:-1]:
        before, after = frame_layout(scene, key.time - 1e-6), frame_layout(scene, key.time + 1e-6)
        assert before.rows["0"].indicator_rect == pytest.approx(after.rows["0"].indicator_rect, abs=0.001)


@pytest.mark.parametrize("count", [2, 10])
def test_indicator_dimensions_follow_manual_staff_zoom_without_overflow(count):
    scene = layout_scene(count=count)
    zoomed = copy.deepcopy(scene)
    zoomed.settings.staff_scale = 2
    zoomed.scale *= 2
    compile_visibility(zoomed)
    first, enlarged = frame_layout(scene, 0), frame_layout(zoomed, 0)
    assert enlarged.rows["0"].indicator_rect[2] / first.rows["0"].indicator_rect[2] == pytest.approx(
        enlarged.scale / first.scale * zoomed.layout.indicator_source_width / scene.layout.indicator_source_width)
    assert enlarged.bounds[0] >= enlarged.region_bounds[0]
    assert enlarged.bounds[1] <= enlarged.region_bounds[1]


def test_custom_score_region_keeps_indicator_right_anchor_and_dense_bounds():
    scene = layout_scene(count=16)
    scene.parts[0].notes[0].end = 1000
    scene.settings.score_top = 0.1
    scene.settings.score_bottom = 0.5
    compile_visibility(scene)
    sizes = set()
    for time in (0, 1, 3, 5):
        layout = frame_layout(scene, time)
        sizes.update(row.indicator_rect[2:] for row in shown(layout))
        for row in shown(layout):
            x, _, width, _ = row.indicator_rect
            assert x + width == pytest.approx(scene.settings.score_left * scene.settings.width - 11)
    assert frame_layout(scene, 0).rows["0"].indicator_rect[3] < 48
    for time in (0, 5):
        if time == 0 or time >= scene.layout.zoom.keys[-1].time:
            assert sum(frame_layout(scene, time).bounds) / 2 == pytest.approx(0.3 * scene.settings.height, abs=0.5)


def test_geometry_is_c2_across_retargeting_and_fades_keep_complete_slots():
    scene = layout_scene(count=8, late=(24,))
    for track in [scene.layout.zoom, *scene.layout.tops.values()]:
        for key in track.keys[1:-1]:
            before, after = track.sample(key.time - 1e-8), track.sample(key.time + 1e-8)
            assert before.value == pytest.approx(after.value, abs=0.003)
            assert before.velocity == pytest.approx(after.velocity, abs=0.005)
            assert before.acceleration == pytest.approx(after.acceleration, abs=0.05)
        assert track.keys[-1].velocity == 0
        assert track.keys[-1].acceleration == 0
    # A fade keeps the complete source-space slot during any simultaneous zoom.
    alpha = scene.parts[1].opacity_keys
    first, last = next((first, last) for first, last in zip(alpha, alpha[1:]) if first[1] == 1 and last[1] == 0)
    halfway = frame_layout(scene, (first[0] + last[0]) / 2)
    assert halfway.rows["1"].opacity == pytest.approx(0.5)
    for time in (first[0], (first[0] + last[0]) / 2, last[0] - 1e-6):
        frame = frame_layout(scene, time)
        row = frame.rows["1"]
        assert (row.bounds[1] - row.bounds[0]) / frame.scale == pytest.approx(scene.parts[1].source_height)


def test_retargeting_preserves_already_rendered_values_and_derivatives():
    track = _Track([_CurveKey(0, math.log(0.04))])
    track.transition(1, 20, math.log(0.13))
    previous = {time: track.sample(time) for time in (0, 1, 1.25, 4, 5.5)}
    at_interrupt = track.sample(6)
    track.transition(6, 9, math.log(0.07))
    for time, value in previous.items():
        actual = track.sample(time)
        assert actual.value == pytest.approx(value.value, abs=1e-12)
        assert actual.velocity == pytest.approx(value.velocity, abs=1e-12)
        assert actual.acceleration == pytest.approx(value.acceleration, abs=1e-12)
    assert track.sample(6) == at_interrupt


@pytest.mark.parametrize("offset", [-1.25, 0, 1.25])
def test_announcement_region_and_tempo_expand_during_a_frozen_intro(offset):
    scene = layout_scene(count=8, grand=True)
    settings = scene.settings
    settings.intro_delay_seconds = 10
    settings.score_start_in_audio_sec = offset
    settings.announcement_auto_hide = True
    settings.overlay_enter_seconds = 0.25
    settings.announcement_hold_seconds = 0.5
    settings.overlay_exit_seconds = 0.5
    for part in scene.parts:
        for note in part.notes:
            note.start += offset
            note.end += offset
    compile_visibility(scene)
    hide_end = 1.25
    before = frame_layout(scene, hide_end)
    middle = frame_layout(scene, hide_end + settings.reflow_seconds / 2)
    expanded = frame_layout(scene, hide_end + settings.reflow_seconds)
    assert before.region_bounds[1] == settings.score_bottom * settings.height
    assert before.region_bounds[1] < middle.region_bounds[1] < expanded.region_bounds[1]
    assert expanded.region_bounds == pytest.approx((0.05 * settings.height, 0.95 * settings.height))
    assert before.region_bounds[0] > middle.region_bounds[0] > expanded.region_bounds[0]
    assert before.scale < expanded.scale <= scene.scale
    assert sum(expanded.bounds) / 2 == pytest.approx(sum(expanded.region_bounds) / 2, abs=0.5)
    assert scene.camera_x_at(0) == scene.camera_x_at(hide_end + settings.reflow_seconds)
    assert scene.camera_speed_at(hide_end + settings.reflow_seconds) == 0
    sizes = set()
    for index in range(300):
        layout = frame_layout(scene, index / 120)
        assert scene.layout.padding_at("0", index / 120) == scene.tempo_mark.padding
        assert layout.bounds[0] >= layout.region_bounds[0] - 1e-6
        assert layout.bounds[1] <= layout.region_bounds[1] + 1e-6
        sizes.update(row.indicator_rect[2:] for row in shown(layout))
    assert len(sizes) > 1


def test_persistent_announcement_and_large_custom_region_keep_their_boundaries():
    scene = layout_scene(count=1)
    assert frame_layout(scene, 100).region_bounds == frame_layout(scene, 0).region_bounds
    scene.settings.score_bottom = 0.98
    scene.settings.announcement_auto_hide = True
    compile_visibility(scene)
    assert frame_layout(scene, 100).region_bounds[1] == pytest.approx(0.98 * scene.settings.height)


def test_normal_zoom_disturbance_is_bounded_relative_to_actual_camera_pan():
    scene = layout_scene(count=10)
    scene.parts[0].notes[0].end = 1000
    compile_visibility(scene)
    radius = max(scene.play_x - scene.body_left, scene.body_right - scene.play_x)
    moving = scene.layout.zoom.keys
    assert moving[-1].time - moving[1].time > scene.settings.reflow_seconds
    checked = 0
    for index in range(3000):
        time = index / 60
        if any(start <= time <= end for start, end in scene.layout.urgent_intervals):
            continue
        pan = scene.camera_speed_at(time) * frame_layout(scene, time).scale
        zoom = abs(scene.layout.zoom.sample(time).velocity) * radius
        assert zoom <= 0.5 * pan + 1e-7
        checked += int(zoom > 1e-7)
    assert checked > 100


def test_new_admission_interrupts_a_long_zoom_and_keeps_its_onset_visible():
    # Disabling stabilization retains the legacy retarget/recovery behavior.
    scene = layout_scene(count=8, late=(24, 28), stable_seconds=0)
    interruptions = [key for key in scene.layout.zoom.keys if abs(key.velocity) > 1e-8]
    assert interruptions
    for part in scene.parts[-2:]:
        note_time = scene.settings.presentation_time(part.notes[0].start)
        assert frame_layout(scene, note_time).rows[part.part_id].opacity == pytest.approx(1.0)
    for index in range(6000):
        frame = frame_layout(scene, index / 240)
        rows = sorted(shown(frame), key=lambda row: row.top)
        assert all(first.bounds[1] <= second.bounds[0] + 1e-6 for first, second in zip(rows, rows[1:]))


def test_deadline_zoom_recovers_without_unbounded_duration_or_losing_continuity():
    scene = layout_scene(count=1, late=(24, 28, 32, 36), stable_seconds=0)
    # A tall incoming part forces a deadline shrink. That wider viewport admits
    # more tall parts while the first shrink still has nonzero derivatives.
    for part in scene.parts[1:]:
        part.source_bottom = 3000
    compile_visibility(scene)
    recovery = scene.layout.recovery_intervals
    assert recovery
    assert all(0 < end - start <= scene.settings.reflow_seconds + 1e-8 for start, end in recovery)
    assert all(math.isfinite(key.time) for key in scene.layout.zoom.keys)
    assert scene.layout.times[-1] < 150
    for start, end in recovery:
        before, after = scene.layout.zoom.sample(start - 1e-7), scene.layout.zoom.sample(start + 1e-7)
        assert before.velocity == pytest.approx(after.velocity, abs=1e-5)
        assert before.acceleration == pytest.approx(after.acceleration, abs=1e-3)
    for part in scene.parts[1:]:
        assert frame_layout(scene, part.notes[0].start).rows[part.part_id].opacity == pytest.approx(1)
    radius = max(scene.play_x - scene.body_left, scene.body_right - scene.play_x)
    for index in range(4000):
        time = index / 120
        frame = frame_layout(scene, time)
        rows = sorted(shown(frame), key=lambda row: row.top)
        assert frame.bounds[0] >= frame.region_bounds[0] - 1e-6
        assert frame.bounds[1] <= frame.region_bounds[1] + 1e-6
        assert all(first.bounds[1] <= second.bounds[0] + 1e-6 for first, second in zip(rows, rows[1:]))
        if not any(start <= time <= end for start, end in scene.layout.urgent_intervals):
            assert abs(scene.layout.zoom.sample(time).velocity) * radius <= 0.5 * scene.camera_speed_at(time) * frame.scale + 1e-7


def test_tempo_keeps_silent_first_row_only_until_its_tail_leaves_the_score():
    scene = layout_scene(count=2)
    scene.parts[0].notes = [ActiveNote(8, 8.2, 100)]
    scene.parts[1].notes[0].end = 1000
    compile_visibility(scene)
    first = frame_layout(scene, 0)
    assert first.rows["0"].opacity == 1
    assert scene.tempo_mark.owner_id == "0"
    assert scene.tempo_mark.x == scene.axis.x_at(0)
    exit_time = scene.layout.tempo_exit_time
    assert math.isfinite(exit_time)
    assert frame_layout(scene, exit_time - 1e-5).rows["0"].opacity == 1
    tail = scene.play_x + (scene.tempo_mark.x + scene.tempo_mark.width - scene.camera_x_at(exit_time)) * frame_layout(scene, exit_time).scale
    assert tail == pytest.approx(scene.body_left, abs=0.001)
    assert scene.layout.padding_at("0", exit_time - 1e-5) == scene.tempo_mark.padding
    assert scene.layout.padding_at("0", exit_time) == 0
    middle = frame_layout(scene, 8)
    assert middle.rows["0"].opacity == 1
    last = frame_layout(scene, 16)
    assert last.rows["0"].opacity == 0
    for track in [scene.layout.zoom, *scene.layout.tops.values()]:
        before, after = track.sample(exit_time - 1e-7), track.sample(exit_time + 1e-7)
        assert before.value == pytest.approx(after.value, abs=0.001)
        assert before.velocity == pytest.approx(after.velocity, abs=0.001)
        assert before.acceleration == pytest.approx(after.acceleration, abs=0.01)


def test_activity_is_velocity_brightness_with_a_smooth_release_and_overlap():
    part = ScenePart("part", "Part", "", 0, 1000, [500], [ActiveNote(0, 0.2, 127), ActiveNote(0.25, 0.5, 64)])
    assert activity_level(part, 0.1) == 1
    assert activity_level(part, 0.23) > 0.85  # A smooth release does not drop linearly.
    assert activity_level(part, 0.25) > 64 / 127
    assert activity_level(part, 0.4) == pytest.approx(64 / 127)
    assert 0 < activity_level(part, 0.6199) < 1e-6
    assert activity_level(part, 0.62) == 0


@pytest.mark.parametrize("width,height", [(1920, 1080), (2560, 1440), (3840, 2160),
                                          (3440, 1440), (2160, 2160)])
def test_hidden_announcement_recenters_the_whole_frame_at_any_resolution(width, height):
    scene = layout_scene(count=8, grand=True)
    settings = scene.settings
    settings.width, settings.height = width, height
    settings.intro_delay_seconds = 10
    settings.announcement_auto_hide = True
    settings.overlay_enter_seconds = 0.25
    settings.announcement_hold_seconds = 0.5
    settings.overlay_exit_seconds = 0.5
    scene.scale *= height / 1080
    scene.camera = None
    compile_visibility(scene)
    start = 1.25
    finish = start + settings.reflow_seconds
    assert frame_layout(scene, start).region_bounds == pytest.approx(
        (settings.score_top * height, settings.score_bottom * height))
    settled = frame_layout(scene, finish)
    assert settled.region_bounds == pytest.approx((0.05 * height, 0.95 * height))
    assert sum(settled.bounds) / 2 == pytest.approx(height / 2, abs=0.5)
    for index in range(81):
        time = start - 0.02 + (settings.reflow_seconds + 0.04) * index / 80
        frame = frame_layout(scene, time)
        rows = sorted(shown(frame), key=lambda row: row.top)
        assert frame.bounds[0] >= frame.region_bounds[0] - 1e-6
        assert frame.bounds[1] <= frame.region_bounds[1] + 1e-6
        assert all(first.bounds[1] <= last.bounds[0] + 1e-6 for first, last in zip(rows, rows[1:]))
    for boundary in (start, finish):
        # The region and visible geometry keep zero velocity/acceleration at
        # the ends of the frozen-intro expansion.
        delta = 1e-6
        before = scene.layout.region_at(boundary - delta)
        now = scene.layout.region_at(boundary)
        after = scene.layout.region_at(boundary + delta)
        for low, middle, high in zip(before, now, after, strict=True):
            assert (high - low) / (2 * delta) == pytest.approx(0.0, abs=0.005)
            assert (high - 2 * middle + low) / delta**2 == pytest.approx(0.0, abs=0.5 * height / 1080)


@pytest.mark.parametrize("top,bottom", [(0, 0.65), (0.03, 0.65), (0.23, 0.98)])
def test_expanded_region_respects_custom_margin_while_staying_symmetric(top, bottom):
    scene = layout_scene(count=2)
    scene.settings.score_top, scene.settings.score_bottom = top, bottom
    scene.settings.announcement_auto_hide = True
    compile_visibility(scene)
    margin = min(0.05, top, 1 - bottom)
    assert frame_layout(scene, 100).region_bounds == pytest.approx(
        (margin * scene.settings.height, (1 - margin) * scene.settings.height))


@pytest.mark.parametrize("stable,remains_visible", [(5.4, False), (5.5, True)])
def test_short_hidden_interval_is_merged_at_the_configured_completion_threshold(stable, remains_visible):
    scene = layout_scene(count=2, stable_seconds=stable)
    scene.parts[0].notes = [ActiveNote(0, 100, 100)]
    scene.parts[1].notes = [ActiveNote(0, 20, 100), ActiveNote(30, 30.2, 100)]
    compile_visibility(scene)
    # The old row would finish fading at 20.5 and its returning bar would
    # enter at 25.926: a 5.426-second hidden state, measured after completion.
    assert frame_layout(scene, 21).rows["1"].opacity == float(remains_visible)
    assert frame_layout(scene, 30).rows["1"].opacity == 1
    if remains_visible:
        assert all(frame_layout(scene, time).rows["1"].opacity == 1 for time in (20.6, 22, 24, 26, 30))


def test_a_newly_visible_row_holds_for_two_seconds_after_its_fade_completes():
    scene = layout_scene(count=1, bpm=900, late=(32,))
    keys = scene.parts[-1].opacity_keys
    complete = next(time for time, value in keys if time > 0 and value == 1)
    exit_start = next(first[0] for first, last in zip(keys, keys[1:]) if first[1] == 1 and last[1] == 0)
    assert exit_start - complete == pytest.approx(2.0)
    assert frame_layout(scene, scene.parts[-1].notes[0].start).rows["late-0"].opacity == 1


def test_stabilization_skips_long_transient_zoom_and_preserves_required_onsets():
    legacy = layout_scene(count=8, late=(24, 28), stable_seconds=0)
    stable = layout_scene(count=8, late=(24, 28))
    assert any(abs(key.velocity) > 1e-8 for key in legacy.layout.zoom.keys)
    assert not any(abs(key.velocity) > 1e-8 for key in stable.layout.zoom.keys)
    # In this short passage, the long enlargement would immediately be undone.
    # Its existing slots and scale remain available to both later phrases.
    assert frame_layout(stable, 5).scale == pytest.approx(frame_layout(stable, 0).scale)
    for part in stable.parts[-2:]:
        assert frame_layout(stable, part.notes[0].start).rows[part.part_id].opacity == 1
    frame = frame_layout(stable, 14)
    rows = sorted(shown(frame), key=lambda row: row.top)
    assert rows and all(first.bounds[1] <= last.bounds[0] + 1e-6 for first, last in zip(rows, rows[1:]))


@pytest.mark.parametrize("width,height", [(1920, 1080), (2560, 1440), (3840, 2160)])
def test_skipping_a_long_zoom_still_centers_the_stationary_visible_group(width, height):
    scene = layout_scene(count=8, late=(24, 28))
    scene.settings.width, scene.settings.height = width, height
    scene.settings.announcement_auto_hide = True
    scene.scale *= height / 1080
    scene.camera = None
    compile_visibility(scene)
    # Hidden old slots must not leave the survivor at the top of the frame
    # merely because the full enlargement cannot finish before the next state.
    frame = frame_layout(scene, 24)
    assert [part for part, row in frame.rows.items() if row.opacity > 1e-8] == ["0"]
    assert frame.scale < scene.scale
    assert sum(frame.bounds) / 2 == pytest.approx(height / 2, abs=0.5)
    assert frame_layout(scene, 26) == frame


def test_reservations_coalesce_fast_admissions_without_skipping_music():
    scene = layout_scene(count=1, late=(24, 28, 32, 36))
    for part in scene.parts[1:]:
        part.source_bottom = 3000
    compile_visibility(scene)
    changes = [(first, last) for first, last in zip(scene.layout.zoom.keys, scene.layout.zoom.keys[1:])
               if abs(first.value - last.value) > 1e-8]
    assert len(changes) == 1
    assert changes[0][1].time <= scene.parts[1].notes[0].start
    for part in scene.parts[1:]:
        assert frame_layout(scene, part.notes[0].start).rows[part.part_id].opacity == 1
    for index in range(480):
        frame = frame_layout(scene, index / 24)
        rows = sorted(shown(frame), key=lambda row: row.top)
        assert frame.bounds[0] >= frame.region_bounds[0] - 1e-6
        assert frame.bounds[1] <= frame.region_bounds[1] + 1e-6
        assert all(first.bounds[1] <= last.bounds[0] + 1e-6 for first, last in zip(rows, rows[1:]))
    cached = {time: frame_layout(scene, time) for time in (0, 5, 8.5, 10, 12, 16, 19)}
    for time in (19, 0, 16, 8.5, 5, 12, 10):
        assert frame_layout(scene, time) == cached[time]
    alternative = copy.deepcopy(scene)
    alternative.settings.fps = 25
    compile_visibility(alternative)
    assert alternative.layout.keyframes == scene.layout.keyframes


def test_region_reflow_during_first_fade_preserves_already_visible_history():
    scenes = []
    for hide_time in (12.45, 13.1):
        scene = layout_scene(count=1, late=(32,))
        scene.settings.announcement_auto_hide = True
        scene.settings.overlay_enter_seconds = 0.25
        scene.settings.overlay_exit_seconds = 0.5
        scene.settings.announcement_hold_seconds = hide_time - 0.75
        compile_visibility(scene)
        scenes.append(scene)
    interrupted, reference = scenes
    for time in (12.3, 12.35, 12.4):
        assert 0 < frame_layout(reference, time).rows["late-0"].opacity < 1
        assert frame_layout(interrupted, time) == frame_layout(reference, time)
    for index in range(120):
        frame = frame_layout(interrupted, 12.25 + index / 120)
        rows = sorted(shown(frame), key=lambda row: row.top)
        assert frame.bounds[0] >= frame.region_bounds[0] - 1e-6
        assert frame.bounds[1] <= frame.region_bounds[1] + 1e-6
        assert all(first.bounds[1] <= last.bounds[0] + 1e-6 for first, last in zip(rows, rows[1:]))


def test_deadline_preparation_keeps_notes_visible_without_viewport_admission():
    scene = layout_scene(count=1, late=(32,))
    # An intentionally distant engraved bar makes the music deadline the only
    # way to discover this row; actual notes must remain a hard requirement.
    scene.measure_bounds[8] = (100000, 104000)
    compile_visibility(scene)
    note = scene.parts[-1].notes[0]
    assert frame_layout(scene, note.start).rows["late-0"].opacity == 1
    assert frame_layout(scene, note.end - 1e-6).rows["late-0"].opacity == 1


def test_music_after_an_empty_gap_gets_a_valid_slot_before_fading_in():
    scene = layout_scene(count=1, late=(40,))
    scene.parts[0].notes = [ActiveNote(0, 0.2, 100)]
    scene.parts[-1].source_bottom = 10000
    compile_visibility(scene)
    assert shown(frame_layout(scene, 5)) == []
    note = scene.parts[-1].notes[0]
    frame = frame_layout(scene, note.start)
    assert frame.rows["late-0"].opacity == 1
    assert frame.bounds[0] >= frame.region_bounds[0] - 1e-6
    assert frame.bounds[1] <= frame.region_bounds[1] + 1e-6


def test_sub_frame_admission_deadlines_do_not_stall_the_event_heap():
    # Bound this regression in a child process: a stale heap head used to
    # prevent the scheduler from ever reaching another frame or event.
    code = """
import runpy, sys
sys.path.insert(0, "tests")
from native_frames import frame_layout
from stavellum.scene import compile_visibility
scene = runpy.run_path('tests/test_layout.py')['layout_scene'](
    count=1, bpm=60000, late=(24, 28, 32, 36))
scene.settings.enter_seconds = 0.000001
scene.settings.exit_seconds = 0.000001
scene.settings.reflow_seconds = 0.000001
for part in scene.parts[1:]:
    part.source_bottom = 3000
compile_visibility(scene)
for part in scene.parts[1:]:
    assert abs(frame_layout(scene, part.notes[0].start).rows[part.part_id].opacity - 1) < 1e-8
assert scene.layout.times[-1] < 3
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, timeout=20)
    assert completed.returncode == 0, completed.stdout + completed.stderr
