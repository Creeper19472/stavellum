import math

import pytest

from stavellum.domain.models import (
    AutomationPoint,
    NoteEvent,
    PartMapping,
    ProjectIR,
    TrackInfo,
    VolumeAutomation,
    VolumeRoute,
)
from stavellum.engraving.dynamics import (
    UnsupportedCurve,
    infer_dynamics,
    interpolate_point,
    volume_gain,
)


def automation(target="channel:1", values=(0.4, 0.9), *, start=0, end=1920,
               points=None, offset=0, scale=1):
    return VolumeAutomation(f"{target}-{start}", target,
                            "channel_volume" if target.startswith("channel:") else "mixer_volume",
                            points or [AutomationPoint(0, values[0]), AutomationPoint(end - start, values[1])],
                            start, end, source_offset_tick=offset, value_scale=scale)


def project(clips, *, tracks=("a",), notes=None, controls=None):
    routes = [VolumeRoute(track, (controls or {}).get(track, [f"channel:{index + 1}"]),
                          {target: 1.0 for target in (controls or {}).get(track, [f"channel:{index + 1}"])})
              for index, track in enumerate(tracks)]
    return ProjectIR("", "flp", "Dynamics", tracks=[TrackInfo(track, track) for track in tracks],
                     notes=notes or [NoteEvent(track, track, 0, 1920, 60 + index) for index, track in enumerate(tracks)],
                     volume_automations=clips, volume_routes=routes)


def infer(source, tracks=None):
    return infer_dynamics(source, PartMapping("part", "Part", tracks or [track.track_id for track in source.tracks]))


def test_native_fl_gain_and_channel_mixer_products():
    assert volume_gain(0) == 0
    assert volume_gain(1) == pytest.approx(1)
    assert 20 * math.log10(volume_gain(1.25)) == pytest.approx(5.6, abs=0.1)
    channel = automation(values=(0.5, 0.8))
    single, _ = infer(project([channel]))
    combined, _ = infer(project([channel, automation("mixer:1", values=(0.5, 0.8), scale=1.25)],
                                controls={"a": ["channel:1", "mixer:1"]}))
    assert len(single) == len(combined) == 1
    assert combined[0].change_db > single[0].change_db * 1.9


def test_the_right_hand_point_defines_the_original_fl_segment_shape():
    first = AutomationPoint(0, 0.2, tension=0.4, metadata=0x01000000)
    last = AutomationPoint(480, 0.8, metadata=0x02000000)
    assert interpolate_point(first, last, 0.5) == pytest.approx(0.5)
    last.metadata = 123
    with pytest.raises(UnsupportedCurve):
        interpolate_point(first, last, 0.5)


@pytest.mark.parametrize("start,end,tension,metadata,observed", [
    (0.25, 0.75, 0.2438096553, 0x01000000, (0.544754, 0.671713, 0.726428)),
    (0.75, 0.25, -0.6171427965, 0xFF000000, (0.749264, 0.743043, 0.690746)),
    (0.25, 0.75, 0.5, 0x01000000, (0.661449, 0.734653, 0.747680)),
])
def test_single_curve_matches_fl_studio_native_render_observations(start, end, tension, metadata, observed):
    # FL Studio rendered a steady sine with channel-volume automation. These
    # values were recovered from 20 ms RMS windows with the native gain law.
    first = AutomationPoint(0, start)
    last = AutomationPoint(480, end, tension, metadata)
    assert [interpolate_point(first, last, x) for x in (0.25, 0.5, 0.75)] == pytest.approx(observed, abs=2e-4)


def test_curved_single_fade_preserves_source_endpoint_times():
    source = project([automation(points=[AutomationPoint(0, 0.25),
                                         AutomationPoint(1920, 0.75, 0.5, 0x01000000)])])
    spans, diagnostics = infer(source)
    assert [(span.start_tick, span.end_tick) for span in spans] == [(0, 1920)]
    assert not any(item.code == "volume-dynamics-unsupported" for item in diagnostics)


def test_native_hold_points_keep_the_prior_value_until_the_jump_without_a_hairpin():
    first = AutomationPoint(0, 0.25)
    last = AutomationPoint(1920, 0.75, 0.2438096553, 0x01000002)
    assert [interpolate_point(first, last, x) for x in (0, 0.25, 0.5, 0.999, 1)] == [
        0.25, 0.25, 0.25, 0.25, 0.75]
    spans, diagnostics = infer(project([automation(points=[first, last])]))
    assert not spans
    assert not any(item.code == "volume-dynamics-unsupported" for item in diagnostics)


def test_single_fade_keeps_exact_endpoints_inside_a_sustained_note():
    source = project([automation(start=125.125, end=1743.875)], notes=[NoteEvent("held", "a", 0, 1920, 72)])
    spans, diagnostics = infer(source)
    assert len(spans) == 1
    assert (spans[0].start_tick, spans[0].end_tick, spans[0].direction) == (125.125, 1743.875, "crescendo")
    assert any(item.code == "auto-volume-dynamics" for item in diagnostics)


def test_only_note_overlap_is_required_and_does_not_trim_the_fade():
    source = project([automation()], notes=[NoteEvent("middle", "a", 600, 300, 72)])
    spans, _ = infer(source)
    assert [(span.start_tick, span.end_tick) for span in spans] == [(0, 1920)]


def test_same_direction_small_segments_merge_before_the_one_db_threshold():
    source = project([automation(points=[AutomationPoint(0, 0.8), AutomationPoint(600, 0.82),
                                         AutomationPoint(1200, 0.84), AutomationPoint(1920, 0.86)])])
    spans, _ = infer(source)
    assert len(spans) == 1 and spans[0].start_tick == 0 and spans[0].end_tick == 1920
    assert spans[0].change_db > 1


@pytest.mark.parametrize("clip", [automation(values=(0.8, 0.81)), automation(end=60)])
def test_minor_or_sub_grid_changes_are_suppressed(clip):
    assert infer(project([clip]))[0] == []


def test_triplets_use_the_smallest_enabled_grid_without_snapping_source_endpoints():
    source = project([automation(start=0.125, end=90.125)])
    mapping = PartMapping("part", "Part", ["a"], triplets=True)
    spans, _ = infer_dynamics(source, mapping)
    assert [(span.start_tick, span.end_tick) for span in spans] == [(0.125, 90.125)]
    mapping.triplets = False
    assert infer_dynamics(source, mapping)[0] == []


def test_turns_and_flat_sections_split_hairpins():
    source = project([automation(points=[AutomationPoint(0, 0.2), AutomationPoint(600, 0.8),
                                         AutomationPoint(1200, 0.8), AutomationPoint(1920, 0.3)])])
    spans, _ = infer(source)
    assert [(span.start_tick, span.end_tick, span.direction) for span in spans] == [
        (0, 600, "crescendo"), (1200, 1920, "diminuendo")]


def test_resets_between_clips_do_not_become_one_continuous_hairpin():
    source = project([automation(values=(0.2, 0.8), end=960),
                      automation(values=(0.2, 0.8), start=960, end=1920)])
    spans, _ = infer(source)
    assert [(span.start_tick, span.end_tick) for span in spans] == [(0, 960), (960, 1920)]


def test_source_crop_preserves_the_original_curve_and_min_max():
    clip = automation(start=240, end=720, offset=240,
                      points=[AutomationPoint(0, 0), AutomationPoint(960, 1)])
    clip.minimum, clip.maximum = 0.2, 0.9
    source = project([clip])
    spans, _ = infer(source)
    assert [(span.start_tick, span.end_tick) for span in spans] == [(240, 720)]
    assert clip.points[0].tick == 0 and clip.points[1].tick == 960


def test_active_sources_must_agree_in_direction_but_can_have_different_baselines():
    source = project([automation(values=(0.2, 0.5)), automation("channel:2", values=(0.6, 0.9))], tracks=("a", "b"))
    spans, _ = infer(source)
    assert len(spans) == 1 and spans[0].direction == "crescendo"
    source.volume_automations[1].points.reverse()
    source.volume_automations[1].points = [AutomationPoint(0, 0.9), AutomationPoint(1920, 0.6)]
    spans, diagnostics = infer(source)
    assert not spans
    assert any(item.code == "volume-dynamics-conflict" for item in diagnostics)


def test_inactive_opposing_source_does_not_block_or_report_a_conflict():
    source = project([automation(), automation("channel:2", values=(0.9, 0.4))], tracks=("a", "b"),
                     notes=[NoteEvent("a", "a", 0, 1920, 72), NoteEvent("b", "b", 2400, 480, 60)])
    spans, diagnostics = infer(source)
    assert len(spans) == 1 and spans[0].direction == "crescendo"
    assert not any(item.code == "volume-dynamics-conflict" for item in diagnostics)


def test_flat_active_source_conflicts_and_unknown_routing_is_reported():
    source = project([automation()], tracks=("a", "b"))
    spans, diagnostics = infer(source)
    assert not spans and any(item.code == "volume-dynamics-conflict" for item in diagnostics)
    source.volume_routes[1].supported = False
    source.volume_routes[1].unsupported_reason = "动态多路径"
    spans, diagnostics = infer(source)
    assert not spans and any(item.code == "volume-dynamics-unsupported" for item in diagnostics)


def test_overlapping_controls_skip_only_the_ambiguous_interval():
    source = project([automation(), automation(start=600, end=1200)])
    spans, diagnostics = infer(source)
    assert [(span.start_tick, span.end_tick) for span in spans] == [(0, 600), (1200, 1920)]
    assert any(item.code == "volume-dynamics-unsupported" for item in diagnostics)


def test_disabling_recognition_keeps_source_automation_intact():
    source = project([automation()])
    mapping = PartMapping("part", "Part", ["a"], auto_dynamics=False)
    assert infer_dynamics(source, mapping) == ([], [])
    assert len(source.volume_automations[0].points) == 2


def test_a_muted_serial_control_prevents_false_mixer_hairpins():
    source = project([automation("mixer:1")], controls={"a": ["channel:1", "mixer:1"]})
    source.volume_routes[0].initial_values["channel:1"] = 0
    assert infer(source)[0] == []


def test_the_composed_gain_floor_suppresses_near_silence_changes():
    source = project([automation(values=(0.00001, 0.00002))])
    assert infer(source)[0] == []


def test_an_unsupported_finished_clip_is_not_a_trusted_baseline():
    unknown = automation(values=(0.4, 0.6), end=240)
    unknown.supported, unknown.unsupported_reason = False, "LFO"
    source = project([unknown, automation("mixer:1", start=480, end=1920)],
                     controls={"a": ["channel:1", "mixer:1"]})
    spans, diagnostics = infer(source)
    assert not spans and any(item.code == "volume-dynamics-unsupported" for item in diagnostics)


def test_opposing_serial_controls_use_the_composed_gain_including_its_interior_turn():
    source = project([automation(values=(0.2, 1)), automation("mixer:1", values=(1, 0.2))],
                     controls={"a": ["channel:1", "mixer:1"]})
    spans, _ = infer(source)
    assert [(span.start_tick, span.end_tick, span.direction) for span in spans] == [
        (0, 960, "crescendo"), (960, 1920, "diminuendo")]


def test_conflicting_active_sources_only_trim_the_conflicting_interval():
    source = project([automation(), automation("channel:2", values=(0.9, 0.4))], tracks=("a", "b"),
                     notes=[NoteEvent("a", "a", 0, 1920, 72), NoteEvent("b", "b", 600, 600, 60)])
    spans, diagnostics = infer(source)
    assert [(span.start_tick, span.end_tick) for span in spans] == [(0, 600), (1200, 1920)]
    assert all(span.direction == "crescendo" for span in spans)
    assert any(item.code == "volume-dynamics-conflict" for item in diagnostics)


def recorded_automation(values=(0.2, 0.35, 0.5, 0.65, 0.8), *, target="channel:1", step=30,
                        start=0, end=None, offset=0):
    points = [AutomationPoint(index * step, value, metadata=2) for index, value in enumerate(values)]
    clip = automation(target, points=points, start=start,
                      end=end if end is not None else start + points[-1].tick, offset=offset)
    clip.source_kind = "pattern"
    return clip


def test_dense_recorded_event_steps_form_a_hairpin_without_changing_the_hold_curve():
    clip = recorded_automation()
    source = project([clip])
    before = [(point.tick, point.value, point.tension, point.metadata) for point in clip.points]
    spans, _ = infer(source)
    assert [(span.start_tick, span.end_tick, span.direction) for span in spans] == [(0, 120, "crescendo")]
    assert [(point.tick, point.value, point.tension, point.metadata) for point in clip.points] == before
    assert interpolate_point(clip.points[0], clip.points[1], 0.5) == 0.2
    assert source.notes[0].velocity == 100


@pytest.mark.parametrize("clip", [recorded_automation(values=(0.2, 0.4, 0.6, 0.8)),
                                  recorded_automation(step=60)])
def test_isolated_or_sparse_recorded_steps_do_not_become_hairpins(clip):
    assert infer(project([clip]))[0] == []


def test_dense_hold_points_in_an_automation_clip_are_still_discrete_jumps():
    clip = recorded_automation()
    clip.source_kind = "clip"
    assert infer(project([clip]))[0] == []


def test_native_dense_event_cadence_is_detected_without_extending_to_pattern_end():
    ticks = [2, *range(6, 195)]
    points = [AutomationPoint(tick, 0.25 + index / (len(ticks) - 1) * 0.5, metadata=2)
              for index, tick in enumerate(ticks)]
    clip = automation(points=points, start=2, end=384, offset=2)
    clip.source_kind = "pattern"
    source = project([clip])
    source.ppq = 96
    spans, _ = infer(source)
    assert [(span.start_tick, span.end_tick) for span in spans] == [(2, 194)]


def test_cropped_and_repeated_recorded_curves_keep_their_placement_and_reset_boundaries():
    cropped = recorded_automation(start=100, end=180, offset=20)
    repeated = recorded_automation(start=200, end=320)
    spans, _ = infer(project([cropped, repeated]))
    assert [(span.start_tick, span.end_tick) for span in spans] == [(100, 180), (200, 320)]
    first = recorded_automation()
    second = recorded_automation(start=120, end=240)
    spans, _ = infer(project([first, second]))
    assert [(span.start_tick, span.end_tick) for span in spans] == [(0, 120), (120, 240)]


def test_dense_recorded_trends_use_the_composed_gain_of_channel_and_mixer_controls():
    channel = recorded_automation(values=tuple(0.2 + index * 0.1 for index in range(9)))
    mixer = automation("mixer:1", values=(1, 0.2), end=240)
    source = project([channel, mixer], controls={"a": ["channel:1", "mixer:1"]})
    spans, _ = infer(source)
    assert [(span.start_tick, span.end_tick, span.direction) for span in spans] == [
        (0, 120, "crescendo"), (120, 240, "diminuendo")]
    source.volume_routes[0].initial_values["channel:1"] = 0
    source.volume_automations = [recorded_automation(target="mixer:1")]
    assert infer(source)[0] == []


def test_dense_recorded_trends_still_require_consensus_between_active_sources():
    source = project([recorded_automation()], tracks=("a", "b"))
    spans, diagnostics = infer(source)
    assert not spans and any(item.code == "volume-dynamics-conflict" for item in diagnostics)
