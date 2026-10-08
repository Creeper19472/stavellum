"""Timing and compilation fidelity checks; fixtures are built from explicit events."""

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import mido
import pytest

from stavellum.importers import ImportFailure, import_project
from stavellum.importers.flp import _playlist_record_size, _read_events
from stavellum.importers.flp_automation import decode_pattern_controls, place_pattern_controls


def _event(event_id, value):
    if event_id < 192:
        if isinstance(value, int):
            value = value.to_bytes(1 if event_id < 64 else 2 if event_id < 128 else 4, "little")
        return bytes([event_id]) + value
    if isinstance(value, str):
        value = (value + "\x00").encode("ascii" if event_id == 199 else "utf-16-le")
    remaining = len(value)
    prefix = bytearray()
    while remaining >= 128:
        prefix.append((remaining & 127) | 128)
        remaining >>= 7
    prefix.append(remaining)
    return bytes([event_id]) + prefix + value


def _note(start=0, length=96, pitch=60, channel=0, color=0, flags=0, fine_pitch=120):
    return struct.pack("<IHHIHH8B", start, flags, channel, length, pitch, 0,
                       fine_pitch, 0, 64, color, 64, 100, 128, 128)


def _clip(position=0, length=384, pattern=1, offset=0xFFFFFFFF, flags=0x40,
          track=0, size=88, channel=None):
    index = 20480 + pattern if channel is None else channel
    core = struct.pack("<IHHIHHHH4sII", position, 20480, index, length,
                       499 - track, 0, 120, flags, bytes([64, 100, 128, 128]),
                       offset, 0xFFFFFFFF)
    tail = bytearray(size - 32)
    if size >= 80:
        struct.pack_into("<d", tail, 32, 1.0)
    return core + tail


def _flp(path, clips, notes=None, version="26.1.1.5539", ppq=96,
         cut=True, extra_events=b"", second_arrangement=None, channel_kind=4):
    parts = [_event(199, version), _event(156, 120000), _event(17, 4), _event(18, 4), _event(30, int(cut))]
    if int(version.split(".")[0]) >= 24:
        # Measured FL26 header: three-byte AC and separate C0 build text.
        parts += [_event(172, b"\x01\x01\x00"), _event(192, f"FL Studio {version}")]
    parts += [_event(224, bytes(21))]  # Not a Pattern! Header container regression.
    parts += [_event(65, 1), _event(224, notes if notes is not None else _note()), _event(193, "Phrase")]
    parts += [_event(64, 0), _event(21, channel_kind), _event(0, 1), _event(203, "Violin I")]
    parts += [_event(201, "Fruity Wrapper"), extra_events]
    parts += [_event(99, 0), _event(241, "Main"), _event(233, b"".join(clips))]
    track_data = struct.pack("<IIIB", 1, 0xffffff, 0, 1)
    parts += [_event(238, track_data)]
    if second_arrangement is not None:
        parts += [_event(99, 1), _event(241, "Alternative"), _event(233, b"".join(second_arrangement)), _event(238, track_data)]
    payload = b"".join(parts)
    path.write_bytes(struct.pack("<4sIhHH", b"FLhd", 6, 0, 1, ppq) + b"FLdt" + struct.pack("<I", len(payload)) + payload)
    return path


def _automation_points(points, lfo=False):
    header = bytearray(21)
    struct.pack_into("<I", header, 0, 1)
    header[8] = int(lfo)
    struct.pack_into("<I", header, 17, len(points))
    return bytes(header) + b"".join(struct.pack("<ddfI", *point) for point in points)


def _volume_flp(path, *, ppq=96, target=0, clips=None, minimum=0, maximum=12800,
                points=None, mapping=469, lfo=False, branch=False, branch_automation=False,
                initial=None, old=False, source_enabled=True):
    if points is None:
        points = [(0, .25, 0, 0), (4, .75, 0, 0x02000000)]
    if clips is None:
        auto = bytearray(_clip(channel=1, length=4 * ppq, size=32 if old else 88))
        struct.pack_into("<ff", auto, 24, -1.0, -1.0)
        clips = [_clip(length=4 * ppq, size=32 if old else 88), bytes(auto)]
    extra = (_event(22 if old else 104, 1) + _event(219, struct.pack("<iIi", 6400, 10000, 0))
             + _event(64, 1) + _event(21, 5) + _event(0, int(source_enabled)) + _event(203, "Volume")
             + _event(219, struct.pack("<iIi", minimum, maximum, 0))
             + _event(234, _automation_points(points, lfo))
             + _event(227, struct.pack("<HHIIII", 0, 1, 0, target, 8, mapping)))
    if branch_automation:
        extra += _event(227, struct.pack("<HHIIII", 0, 1, 0, 0x70801FC0, 8, 469))
    mixer = b""
    for insert in range(3 if branch else 2):
        routes = bytes([0]) if insert == 0 else bytes([1, 0, int(branch and insert == 1)])
        mixer += _event(236, struct.pack("<III", 0, 8, 0)) + _event(235, routes)
    if old:
        mixer += _event(225, b"".join(struct.pack("<IBBHi", 0, 192, 0, insert << 6, 12800)
                                     for insert in range(2)))
    else:
        mixer += _event(225, b"".join(struct.pack("<IIi", 0, 0x70001FC0 + (insert << 22), 12800)
                                     for insert in range(3 if branch else 2)))
    if initial is not None:
        mixer += _event(216, struct.pack("<IIi", 0, target, initial))
    source = _flp(path, clips, ppq=ppq, notes=_note(length=ppq * 4),
                  extra_events=extra, version="20.7.0.1702" if old else "26.1.1.5539")
    data = bytearray(source.read_bytes())
    struct.pack_into("<H", data, 10, 2)
    struct.pack_into("<I", data, 18, len(data) - 22 + len(mixer))
    source.write_bytes(data + mixer)
    return source


@pytest.mark.parametrize("target,expected,scale", [
    (0, "channel:0", 1), (0x70401FC0, "mixer:1", 1.25), (0x70001FC0, "mixer:0", 1.25),
])
def test_flp_native_volume_targets_restore_curve_and_serial_path(tmp_path, target, expected, scale):
    project = import_project(_volume_flp(tmp_path / "volume.flp", target=target, initial=9000))
    assert project.timing_confirmed
    assert len(project.volume_automations) == 1
    automation = project.volume_automations[0]
    assert (automation.target_id, automation.value_scale, automation.supported) == (expected, scale, True)
    assert [(point.tick, point.value, point.metadata) for point in automation.points] == [
        (0, .25, 0), (384, .75, 0x02000000)]
    route = project.volume_routes[0]
    assert route.supported
    assert route.control_ids == ["channel:0", "mixer:1", "mixer:0"]
    assert route.initial_values[expected] == 9000 / 12800
    assert project.tracks[0].mixer_insert == 1


def test_flp_old_mixer_rec_and_channel_route_layout(tmp_path):
    project = import_project(_volume_flp(tmp_path / "old.flp", old=True, target=0x204020C0))
    assert project.timing_confirmed
    assert project.volume_automations[0].target_id == "mixer:1"
    assert project.volume_routes[0].supported


@pytest.mark.parametrize("ppq", [96, 480])
def test_flp_volume_crop_repeat_preserves_original_points_and_tension(tmp_path, ppq):
    auto = bytearray(_clip(channel=1, position=2 * ppq, length=2 * ppq))
    struct.pack_into("<ff", auto, 24, 1.5, 3.5)
    repeated = bytearray(auto)
    struct.pack_into("<I", repeated, 0, 6 * ppq)
    muted = bytearray(auto)
    struct.pack_into("<H", muted, 18, 0x2040)
    source = _volume_flp(tmp_path / "cropped.flp", ppq=ppq,
        clips=[_clip(length=8 * ppq), bytes(auto), bytes(repeated), bytes(muted)],
        points=[(0, .2, 0, 0), (4, .8, -.6171428, 0xFF000000)],
        minimum=3200, maximum=9600)
    before = source.read_bytes()
    project = import_project(source)
    first, second = project.volume_automations
    assert first.automation_id != second.automation_id
    assert (first.start_tick, first.end_tick, first.source_offset_tick) == (2 * ppq, 4 * ppq, 1.5 * ppq)
    assert (first.minimum, first.maximum) == (.25, .75)
    assert first.points[-1].tick == 4 * ppq
    assert first.points[-1].tension == pytest.approx(-.6171428)
    assert first.points == second.points
    assert source.read_bytes() == before


@pytest.mark.parametrize("mapping,lfo", [(100, False), (469, True)])
def test_flp_unverified_volume_mapping_keeps_clock_confirmed(tmp_path, mapping, lfo):
    project = import_project(_volume_flp(tmp_path / "unsupported.flp", mapping=mapping, lfo=lfo))
    assert project.timing_confirmed
    assert not project.volume_automations[0].supported
    assert any(d.code == "volume_automation_unsupported" for d in project.diagnostics)


def test_flp_unknown_or_tempo_target_remains_clock_unconfirmed(tmp_path):
    project = import_project(_volume_flp(tmp_path / "tempo.flp", target=0x40000005))
    assert not project.timing_confirmed
    assert project.volume_automations == []


def test_flp_static_parallel_routes_factor_common_volume_controls(tmp_path):
    project = import_project(_volume_flp(tmp_path / "parallel.flp", branch=True))
    route = project.volume_routes[0]
    assert route.supported
    assert route.control_ids == ["channel:0", "mixer:1", "mixer:0"]


def test_flp_automated_noncommon_mixer_branch_is_diagnosed(tmp_path):
    project = import_project(_volume_flp(tmp_path / "branch.flp", branch=True, branch_automation=True))
    assert project.timing_confirmed
    assert not project.volume_routes[0].supported
    assert "非公共" in project.volume_routes[0].unsupported_reason
    assert any(d.code == "volume_route_unsupported" for d in project.diagnostics)


def test_flp_muted_volume_source_does_not_affect_score(tmp_path):
    project = import_project(_volume_flp(tmp_path / "disabled.flp", source_enabled=False))
    assert project.timing_confirmed
    assert not project.volume_automations


def _rewrite_flp_events(source, mutate, channel_count=None):
    events, version, ppq, count = _read_events(source.read_bytes())
    rewritten = mutate([(event.event_id, event.value) for event in events])
    payload = b"".join(_event(eid, value) for eid, value in rewritten)
    source.write_bytes(struct.pack("<4sIhHH", b"FLhd", 6, 0, channel_count or count, ppq)
                       + b"FLdt" + struct.pack("<I", len(payload)) + payload)
    return source


def test_flp_serial_bus_and_master_controls_follow_explicit_path(tmp_path):
    source = _volume_flp(tmp_path / "bus.flp", branch=True, target=0x70801FC0)

    def mutate(events):
        insert = -1
        result = []
        for eid, value in events:
            if eid == 236:
                insert += 1
            if eid == 235 and insert == 1:
                value = bytes([0, 0, 1])
            result.append((eid, value))
        return result

    project = import_project(_rewrite_flp_events(source, mutate))
    assert project.volume_routes[0].supported
    assert project.volume_routes[0].control_ids == ["channel:0", "mixer:1", "mixer:2", "mixer:0"]
    assert project.volume_automations[0].target_id == "mixer:2"


def test_flp_shared_mixer_target_applies_to_both_note_sources(tmp_path):
    source = _volume_flp(tmp_path / "shared.flp", target=0x70401FC0)

    def mutate(events):
        result = []
        pattern = False
        for eid, value in events:
            if eid == 65:
                pattern = True
            elif eid == 64:
                pattern = False
            if eid == 224 and pattern:
                value += _note(channel=2, pitch=72)
            if eid == 99:
                result.extend([(64, struct.pack("<H", 2)), (21, bytes([4])), (0, bytes([1])),
                               (203, "Second Voice\0".encode("utf-16-le")),
                               (104, struct.pack("<H", 1)), (219, struct.pack("<iIi", 6400, 10000, 0))])
            result.append((eid, value))
        return result

    project = import_project(_rewrite_flp_events(source, mutate, channel_count=3))
    assert len(project.tracks) == 2
    assert all(route.supported and "mixer:1" in route.control_ids for route in project.volume_routes)
    assert {route.control_ids[0] for route in project.volume_routes} == {"channel:0", "channel:2"}


def test_flp_silent_send_does_not_make_branch_automation_ambiguous(tmp_path):
    source = _volume_flp(tmp_path / "silent-send.flp", branch=True, branch_automation=True)

    def mutate(events):
        return [(eid, value + struct.pack("<IIi", 0, 0x70402002, 0) if eid == 225 else value)
                for eid, value in events]

    project = import_project(_rewrite_flp_events(source, mutate))
    assert project.volume_routes[0].supported
    assert "mixer:2" not in project.volume_routes[0].control_ids


def test_flp_volume_initialization_overrides_current_knob_even_when_saved_first(tmp_path):
    source = _volume_flp(tmp_path / "initial.flp", target=0x70401FC0, initial=10000)

    def mutate(events):
        init = [event for event in events if event[0] == 216]
        return [events[0]] + init + [event for event in events[1:] if event[0] != 216]

    project = import_project(_rewrite_flp_events(source, mutate))
    assert project.volume_routes[0].initial_values["mixer:1"] == 10000 / 12800


def test_flp_unknown_plugin_binding_alongside_volume_retains_clock_guard(tmp_path):
    source = _volume_flp(tmp_path / "mixed-targets.flp")

    def mutate(events):
        return events + [(227, struct.pack("<HHIIII", 0, 1, 0, 0x00008002, 8, 469))]

    project = import_project(_rewrite_flp_events(source, mutate))
    assert len(project.volume_automations) == 1
    assert not project.timing_confirmed


def test_flp_corrupt_volume_points_preserve_unsupported_target_for_diagnostics(tmp_path):
    source = _volume_flp(tmp_path / "bad-points.flp", points=[(0, .2, 0, 0), (4, float("nan"), 0, 0)])
    project = import_project(source)
    assert project.timing_confirmed
    automation = project.volume_automations[0]
    assert not automation.supported and not automation.points
    assert "无效" in automation.unsupported_reason


@pytest.mark.parametrize("target,expected", [(17, "channel-offset:0"), (0x40000000, "global:main-volume")])
def test_flp_uncalibrated_native_volume_masks_composed_dynamics_without_clock_warning(tmp_path, target, expected):
    source = _volume_flp(tmp_path / "uncalibrated-volume.flp")

    def mutate(events):
        return events + [(227, struct.pack("<HHIIII", 0, 1, 0, target, 8, 469))]

    project = import_project(_rewrite_flp_events(source, mutate))
    assert project.timing_confirmed
    unsupported = next(automation for automation in project.volume_automations if automation.target_id == expected)
    assert not unsupported.supported and "增益" in unsupported.unsupported_reason
    assert not project.volume_routes[0].supported
    assert not any(d.code == "automation_timing_unverified" for d in project.diagnostics)


def _add_pattern_controls(source, records):
    payload = b"".join(struct.pack("<III", *record) for record in records)

    def mutate(events):
        result = []
        pattern = False
        for eid, value in events:
            if eid == 65:
                pattern = True
            elif eid == 64:
                pattern = False
            result.append((eid, value))
            if eid == 224 and pattern:
                result.append((223, payload))
        return result

    return _rewrite_flp_events(source, mutate)


@pytest.mark.parametrize("target,maximum,scale", [(0, 12800, 1), (0x70401FC0, 16000, 1.25)])
def test_flp_pattern_volume_uses_native_integer_rec_and_absolute_ticks(tmp_path, target, maximum, scale):
    source = _volume_flp(tmp_path / "pattern-volume.flp", clips=[_clip(length=768)])
    project = import_project(_add_pattern_controls(source, [
        (2, target, maximum // 4), (98, target, maximum // 2), (194, target, 3 * maximum // 4)]))
    assert project.timing_confirmed
    assert len(project.volume_automations) == 2
    first, repeated = project.volume_automations
    assert first.source_kind == repeated.source_kind == "pattern"
    assert (first.start_tick, first.end_tick, first.source_offset_tick) == (2, 384, 2)
    assert (repeated.start_tick, repeated.end_tick, repeated.source_offset_tick) == (386, 768, 2)
    assert first.value_scale == scale
    assert [(point.tick, point.value, point.metadata) for point in first.points] == [
        (2, .25, 2), (98, .5, 2), (194, .75, 2)]


def test_flp_pattern_crop_does_not_rewrite_original_steps(tmp_path):
    source = _volume_flp(tmp_path / "pattern-crop.flp", clips=[_clip(length=192, offset=48)])
    project = import_project(_add_pattern_controls(source, [(2, 0, 3219), (98, 0, 6438), (194, 0, 9657)]))
    automation = project.volume_automations[0]
    assert (automation.start_tick, automation.end_tick, automation.source_offset_tick) == (0, 192, 48)
    assert [(point.tick, point.value) for point in automation.points] == [
        (2, 3219 / 12800), (98, 6438 / 12800), (194, 9657 / 12800)]


def test_flp_pattern_controller_extent_prevents_premature_note_repetition(tmp_path):
    source = _volume_flp(tmp_path / "pattern-extent.flp", clips=[_clip(length=1536)])
    project = import_project(_add_pattern_controls(source, [(2, 0, 10000), (500, 0, 12000)]))
    assert [note.start_tick for note in project.notes] == [0, 768]
    assert [automation.end_tick for automation in project.volume_automations] == [768, 1536]


def test_flp_pattern_unknown_control_keeps_clock_guard_alongside_native_volume(tmp_path):
    source = _volume_flp(tmp_path / "pattern-unknown.flp", clips=[_clip()])
    project = import_project(_add_pattern_controls(source, [(0, 0, 10000), (0, 0x40000005, 120000)]))
    assert not project.timing_confirmed
    assert len(project.volume_automations) == 1
    assert project.volume_automations[0].target_id == "channel:0"


def test_flp_truncated_secondary_binding_does_not_hide_possible_clock_control(tmp_path):
    source = _volume_flp(tmp_path / "truncated-binding.flp")
    project = import_project(_rewrite_flp_events(source,
        lambda events: events + [(227, struct.pack("<HH", 0, 1))]))
    assert not project.timing_confirmed
    assert len(project.volume_automations) == 1


def test_flp_static_zero_gain_paths_are_excluded_before_common_control_factoring(tmp_path):
    source = _volume_flp(tmp_path / "silent-paths.flp", branch=True)

    def mutate(events):
        result = []
        for eid, value in events:
            if eid == 225:
                payload = bytearray(value)
                struct.pack_into("<i", payload, 8, 0)  # Master static zero silences both branches.
                value = bytes(payload)
            result.append((eid, value))
        return result

    project = import_project(_rewrite_flp_events(source, mutate))
    assert not project.volume_routes[0].supported
    assert "静态音量" in project.volume_routes[0].unsupported_reason


@pytest.mark.parametrize("rec", [8, 0x70402000])
def test_flp_played_routing_automation_disables_volume_composition(tmp_path, rec):
    source = _volume_flp(tmp_path / "dynamic-route.flp")
    project = import_project(_rewrite_flp_events(source,
        lambda events: events + [(227, struct.pack("<HHIIII", 0, 1, 0, rec, 8, 469))]))
    assert not project.volume_routes[0].supported
    assert "路由" in project.volume_routes[0].unsupported_reason


def test_flp_pattern_send_events_disable_static_volume_composition(tmp_path):
    source = _volume_flp(tmp_path / "pattern-send.flp", clips=[_clip()])
    project = import_project(_add_pattern_controls(source, [(0, 0, 10000), (0, 0x70402000, 8000)]))
    assert not project.volume_routes[0].supported
    assert "路由" in project.volume_routes[0].unsupported_reason


def test_flp_pattern_invalid_value_cannot_be_cleared_by_later_valid_record(tmp_path):
    source = _volume_flp(tmp_path / "pattern-invalid.flp", clips=[_clip()])
    records = [(0, 0, 20000), (12, 0, 5000)]
    project = import_project(_add_pattern_controls(source, records))
    automation = project.volume_automations[0]
    assert not automation.supported and automation.points == []
    assert "整数范围" in automation.unsupported_reason
    assert bytes.fromhex(automation.raw_payload) == b"".join(struct.pack("<III", *record) for record in records)


def test_flp_unsupported_clip_preserves_original_point_container(tmp_path):
    points = [(0, .2, 0, 0), (4, float("nan"), 0, 0)]
    project = import_project(_volume_flp(tmp_path / "raw-points.flp", points=points))
    assert bytes.fromhex(project.volume_automations[0].raw_payload) == _automation_points(points)


def _native_volume_evidence():
    return json.loads((Path(__file__).parent / "fixtures/flp_volume_calibration.json").read_text(encoding="utf-8"))


def test_flp_smooth_interpolation_matches_independent_native_wav_observations():
    from stavellum.domain.models import AutomationPoint
    from stavellum.engraving.dynamics import interpolate_point

    evidence = _native_volume_evidence()
    for case in evidence["automation_clip"]["single_curve"]["cases"]:
        left = AutomationPoint(0, case["start_value"])
        right = AutomationPoint(1, case["end_value"], case["right_point_tension"], case["right_point_metadata"])
        for observation in case["observations"]:
            assert interpolate_point(left, right, observation["fraction"]) == pytest.approx(
                observation["observed_value"], abs=.0002)
    assert not evidence["provenance"]["private_projects_used"]


def test_flp_native_e223_fixture_uses_integer_values_and_interleaved_absolute_ticks():
    evidence = _native_volume_evidence()["pattern_events"]["interleaved_channel_mixer"]
    payloads = [bytes.fromhex(value) for value in evidence["raw_hex"]]
    controls, clock_known, end = decode_pattern_controls(payloads, {1})
    assert clock_known and end == 245
    targets = {control.binding.target_id: control for control in controls}
    assert [(point.tick, point.value) for point in targets["channel:1"].points] == [
        (2, 3219 / 12800), (98, 6438 / 12800), (194, 9657 / 12800)]
    assert [(point.tick, point.value) for point in targets["mixer:1"].points] == [
        (51, 4024 / 16000), (148, 8048 / 16000), (244, 12072 / 16000)]
    assert all(point.metadata == 2 for control in controls for point in control.points)
    assert all(control.raw_payload == "".join(evidence["raw_hex"]) for control in controls)


def test_flp_native_sparse_pattern_platforms_are_steps_and_dense_draw_becomes_hairpin():
    from stavellum.domain.models import NoteEvent, PartMapping, ProjectIR, TrackInfo, VolumeRoute
    from stavellum.engraving.dynamics import infer_dynamics

    evidence = _native_volume_evidence()
    native = evidence["pattern_events"]
    platforms = native["playback"]["observations"]
    assert platforms[0]["rms"] == platforms[1]["rms"] == platforms[2]["rms"]
    assert platforms[3]["rms"] == platforms[4]["rms"] == platforms[5]["rms"]
    assert platforms[6]["rms"] == platforms[7]["rms"] == platforms[8]["rms"]
    project = ProjectIR("native-calibration", "flp", "Native controls", ppq=96,
        tracks=[TrackInfo("native-channel", "Sampler")],
        notes=[NoteEvent("native-note", "native-channel", 0, 384, 60)],
        volume_routes=[VolumeRoute("native-channel", ["channel:1"], {"channel:1": 10000 / 12800})])
    mapping = PartMapping("native-part", "Sampler", ["native-channel"])
    for source_kind, expected in [("sparse_channel", False), ("dense_draw", True)]:
        payloads = [bytes.fromhex(value) for value in native[source_kind]["raw_hex"]]
        controls, confirmed, _ = decode_pattern_controls(payloads, {1})
        assert confirmed
        project.volume_automations = place_pattern_controls(controls, 0, 384, 0, 384, "native", "Native sample")
        spans, diagnostics = infer_dynamics(project, mapping)
        assert bool(spans) == expected
        if expected:
            assert len(spans) == 1 and spans[0].direction == "crescendo"
        assert not any(diagnostic.severity in ("warning", "error") for diagnostic in diagnostics)


def test_fl26_repeats_crop_muted_and_arrangement_are_exact(tmp_path):
    source = _flp(tmp_path / "source.flp", [
        _clip(position=0), _clip(position=384),
        _clip(position=768, length=96, offset=48),
        _clip(position=1152, flags=0x2040),
    ], notes=_note(start=0, length=192) + _note(start=288, length=96),
        second_arrangement=[_clip(position=1920)])
    before = source.read_bytes()
    project = import_project(source)
    assert [(n.start_tick, n.duration_tick) for n in project.notes] == [
        (0, 192), (288, 96), (384, 192), (672, 96), (768, 96)]
    assert [n.key_release_tick for n in project.notes] == [192, 384, 576, 768, None]
    assert project.arrangement_names == ["Main", "Alternative"]
    alternative = import_project(source, arrangement_index=1)
    assert [(n.start_tick, n.duration_tick) for n in alternative.notes] == [(1920, 192), (2208, 96)]
    assert any(d.code == "muted_clips" for d in project.diagnostics)
    assert any(d.code == "header_note_container" for d in project.diagnostics)
    assert source.read_bytes() == before


@pytest.mark.parametrize("ppq", [96, 480, 960])
def test_step_notes_retain_trigger_and_diagnose_sample_gate(tmp_path, ppq):
    source = _flp(tmp_path / "steps.flp", [_clip(length=ppq * 4)], notes=_note(length=0), ppq=ppq)
    project = import_project(source)
    assert project.notes[0].duration_tick == ppq // 4
    assert project.notes[0].key_release_tick is None
    assert any(d.code == "step_trigger_duration" for d in project.diagnostics)


def test_flp_note_color_retained_without_automatic_routing(tmp_path):
    source = _flp(tmp_path / "colors.flp", [_clip()], notes=_note(color=3) + _note(start=96, color=7))
    project = import_project(source)
    assert len(project.tracks) == 1
    assert project.tracks[0].midi_channel is None
    assert [note.midi_channel for note in project.notes] == [3, 7]


@pytest.mark.parametrize("color_bytes,expected", [
    (None, "#ffffff"),
    ("12345600", "#123456"),
    ("12345680", "#123456"),
    ("123456ff", "#123456"),
])
def test_flp_channel_rack_rgb_preserves_color_without_alpha(tmp_path, color_bytes, expected):
    extra = _event(128, bytes.fromhex(color_bytes)) if color_bytes is not None else b""
    source = _flp(tmp_path / "rack-color.flp", [_clip()], extra_events=extra)
    before = source.read_bytes()
    project = import_project(source)
    assert project.tracks[0].color == expected
    assert source.read_bytes() == before


def test_flp_channel_rack_colors_remain_independent_for_articulation_tracks(tmp_path):
    extra = (_event(128, bytes.fromhex("12345600"))
             + _event(64, 1) + _event(21, 4) + _event(0, 1)
             + _event(203, "Violin I pizzicato") + _event(128, bytes.fromhex("e0a16000")))
    source = _flp(tmp_path / "articulation-colors.flp", [_clip()],
        notes=_note(channel=0) + _note(start=96, channel=1), extra_events=extra)
    _rewrite_flp_events(source, lambda events: events, channel_count=2)
    project = import_project(source)
    assert [(track.track_id, track.color) for track in project.tracks] == [
        ("flp-channel-0", "#123456"), ("flp-channel-1", "#e0a160")]


def test_flp_playlist_color_does_not_override_channel_rack_color(tmp_path):
    source = _flp(tmp_path / "playlist-color.flp", [_clip()],
        extra_events=_event(128, bytes.fromhex("12345600")))

    def mutate(events):
        return [(eid, value[:4] + bytes.fromhex("fedcba00") + value[8:] if eid == 238 else value)
                for eid, value in events]

    project = import_project(_rewrite_flp_events(source, mutate))
    assert project.tracks[0].color == "#123456"


def test_flp_clip_extension_loops_and_noncut_gate_spills(tmp_path):
    looping = import_project(_flp(tmp_path / "loop.flp", [_clip(length=768)], notes=_note(length=96)))
    assert [note.start_tick for note in looping.notes] == [0, 384]
    gate = import_project(_flp(tmp_path / "gate.flp", [_clip(length=96)], notes=_note(length=192), cut=False))
    assert gate.notes[0].duration_tick == 192


def test_flp_crop_before_note_requires_play_truncated_option(tmp_path):
    source = _flp(tmp_path / "crop.flp", [_clip(length=96, offset=48)], notes=_note(length=192), cut=False)
    with pytest.raises(ImportFailure, match="没有可制谱音符"):
        import_project(source)


@pytest.mark.parametrize("size,version", [(32, "20.7.0.1702"), (60, "21.0.0.3000"), (80, "24.2.99.4720"), (88, "26.1.1.5539")])
def test_version_gated_playlist_layouts(tmp_path, size, version):
    source = _flp(tmp_path / "version.flp", [_clip(size=size)], version=version)
    assert len(import_project(source).notes) == 1


def test_80_88_common_multiple_uses_version_not_divisibility():
    assert _playlist_record_size("25.2.5.1234", bytes(17600)) == 80
    assert _playlist_record_size("26.1.1.5539", bytes(17600)) == 88


def test_unknown_or_truncated_playlist_never_silently_drops(tmp_path):
    source = _flp(tmp_path / "unknown.flp", [_clip(size=80)])
    with pytest.raises(ImportFailure) as info:
        import_project(source)
    assert info.value.diagnostics[0].code == "unsupported_playlist_layout"
    source = _flp(tmp_path / "truncated.flp", [_clip()])
    source.write_bytes(source.read_bytes()[:-1])
    with pytest.raises(ImportFailure) as info:
        import_project(source)
    assert info.value.diagnostics[0].code == "invalid_flp_header"


def test_layer_slide_and_arpeggiator_reported(tmp_path):
    parameters = bytes(40) + struct.pack("<I", 1)
    source = _flp(tmp_path / "special.flp", [_clip()], notes=_note(flags=8),
                  channel_kind=3, extra_events=_event(215, parameters))
    project = import_project(source)
    codes = {diagnostic.code for diagnostic in project.diagnostics}
    assert {"slide_notes", "layer_channel", "arpeggiator"} <= codes


@pytest.mark.parametrize("follow_master", [True, False, None])
def test_flp_main_pitch_reports_affected_channels_without_rewriting_notes(tmp_path, follow_master):
    parameters = bytearray(12)
    parameters[11] = bool(follow_master)
    extra = _event(80, struct.pack("<h", 100))
    if follow_master is not None:
        extra += _event(215, parameters)
    source = _flp(tmp_path / "master-pitch.flp", [_clip()], extra_events=extra)
    before = source.read_bytes()
    project = import_project(source)
    assert project.notes[0].pitch == 60
    assert source.read_bytes() == before
    warnings = [d for d in project.diagnostics if d.code == "source_pitch_unverified"]
    assert bool(warnings) is (follow_master is not False)
    if warnings:
        assert "+100 音分" in warnings[0].message
        assert warnings[0].track_id == project.tracks[0].track_id
    assert any(d.code == "project_main_pitch" for d in project.diagnostics)


def test_flp_channel_root_fine_stretch_and_note_pitch_are_diagnosed(tmp_path):
    parameters = bytearray(104)
    parameters[11] = True
    parameters[63] = True
    struct.pack_into("<i", parameters, 100, -200)
    levels = struct.pack("<iIi", 0, 12800, 100)
    extra = (_event(219, levels) + _event(135, 61) +
             _event(142, struct.pack("<i", -25)) + _event(215, parameters))
    project = import_project(_flp(tmp_path / "channel-pitch.flp", [_clip()],
        notes=_note(fine_pitch=123), extra_events=extra))
    warning = next(d for d in project.diagnostics if d.code == "source_pitch_unverified")
    assert project.notes[0].pitch == 60
    assert all(value in warning.message for value in (
        "通道音高 +100", "键盘微调 -25", "采样伸缩音高 -200", "根音 61", "Add to key", "音符微调 +30"))


def test_flp_default_pitch_settings_need_no_warning(tmp_path):
    parameters = bytearray(104)
    parameters[11] = True
    project = import_project(_flp(tmp_path / "default-pitch.flp", [_clip()], extra_events=(
        _event(219, struct.pack("<iIi", 0, 12800, 0)) + _event(135, 60) +
        _event(142, 0) + _event(215, parameters))))
    assert not any(d.code == "source_pitch_unverified" for d in project.diagnostics)


def test_empty_pattern_references_and_audio_are_explained(tmp_path):
    source = _flp(tmp_path / "audio.flp", [_clip(channel=0), _clip(pattern=2)])
    with pytest.raises(ImportFailure) as info:
        import_project(source)
    codes = {diagnostic.code for diagnostic in info.value.diagnostics}
    assert {"flp_layout", "audio_clip", "implicit_empty_patterns", "no_arranged_notes"} <= codes


def test_pattern_event_automation_requires_timing_confirmation(tmp_path):
    source = _flp(tmp_path / "automation.flp", [_clip()])
    data = source.read_bytes()
    events, version, ppq, channels = _read_events(data)
    stream = b"".join(_event(event.event_id, event.value)
                      + (_event(223, struct.pack("<III", 0, 0x40000005, 120000))
                         if event.event_id == 224 and len(event.value) == 24 else b"")
                      for event in events)
    source.write_bytes(data[:18] + struct.pack("<I", len(stream)) + stream)
    project = import_project(source)
    assert not project.timing_confirmed


def _midi(path, *, tempo=500000, meter=(4, 4), extra=(), ppq=480):
    midi = mido.MidiFile(type=1, ticks_per_beat=ppq)
    midi.tracks.append(mido.MidiTrack([
        mido.MetaMessage("set_tempo", tempo=tempo),
        mido.MetaMessage("time_signature", numerator=meter[0], denominator=meter[1]),
    ]))
    midi.tracks.append(mido.MidiTrack([
        mido.MetaMessage("track_name", name="Violin I"),
        mido.Message("program_change", channel=0, program=40),
        mido.Message("note_on", note=60, velocity=100, channel=0),
        mido.Message("note_off", note=60, channel=0, time=ppq), *extra,
    ]))
    midi.save(path)
    return path


@pytest.mark.parametrize("ppq", [96, 480, 960])
def test_midi_preserves_ticks_and_gm_identity(tmp_path, ppq):
    project = import_project(_midi(tmp_path / "notes.mid", ppq=ppq))
    assert project.ppq == ppq
    assert project.notes[0].duration_tick == ppq
    assert project.notes[0].key_release_tick == ppq
    assert project.tracks[0].plugin == "General MIDI: Violin"
    assert project.duration_seconds == 0.5


@pytest.mark.parametrize("message,code", [
    (mido.MetaMessage("set_tempo", tempo=600000, time=1), "variable_tempo"),
    (mido.MetaMessage("time_signature", numerator=3, denominator=4, time=1), "variable_meter"),
])
def test_midi_variable_timing_rejected(tmp_path, message, code):
    with pytest.raises(ImportFailure) as info:
        import_project(_midi(tmp_path / "variable.mid", extra=[message]))
    assert info.value.diagnostics[0].code == code


def test_midi_same_redundant_timing_is_allowed(tmp_path):
    source = _midi(tmp_path / "same.mid", extra=[mido.MetaMessage("set_tempo", tempo=500000, time=1)])
    assert import_project(source).bpm == 120


def test_midi_sustain_and_multiple_channels(tmp_path):
    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack([
        mido.MetaMessage("track_name", name="Piano"),
        mido.Message("note_on", note=60, velocity=90, channel=0),
        mido.Message("control_change", control=64, value=127, channel=0),
        mido.Message("note_off", note=60, channel=0, time=240),
        mido.Message("note_on", note=67, velocity=80, channel=1),
        mido.Message("note_off", note=67, channel=1, time=240),
        mido.Message("control_change", control=64, value=0, channel=0, time=240),
    ]))
    midi.save(tmp_path / "sustain.mid")
    project = import_project(tmp_path / "sustain.mid")
    assert [(n.start_tick, n.duration_tick, n.midi_channel) for n in project.notes] == [(0, 720, 0), (240, 240, 1)]
    assert [n.key_release_tick for n in project.notes] == [240, 480]
    assert len(project.tracks) == 2
    assert project.tracks[0].track_id != project.tracks[1].track_id


def test_midi_channel_controller_state_spans_tracks(tmp_path):
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks += [mido.MidiTrack([
        mido.Message("program_change", program=40, channel=2),
        mido.Message("control_change", control=64, value=127, channel=2),
        mido.Message("control_change", control=64, value=0, channel=2, time=960),
    ]), mido.MidiTrack([
        mido.Message("note_on", note=60, velocity=100, channel=2),
        mido.Message("note_off", note=60, channel=2, time=480),
    ])]
    midi.save(tmp_path / "global-controller.mid")
    project = import_project(tmp_path / "global-controller.mid")
    assert project.notes[0].duration_tick == 960
    assert project.notes[0].key_release_tick == 480
    assert project.tracks[0].plugin == "General MIDI: Violin"


@pytest.mark.parametrize("control", [64, 120, 121])
def test_real_midi_release_survives_pedal_reset_and_all_sound_off(tmp_path, control):
    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack([
        mido.Message("control_change", control=64, value=127),
        mido.Message("note_on", note=60, velocity=90),
        mido.Message("note_off", note=60, time=120),
        mido.Message("control_change", control=control, value=0, time=600),
    ]))
    source = tmp_path / f"release-{control}.mid"
    midi.save(source)
    event = import_project(source).notes[0]
    assert event.duration_tick == 720
    assert event.key_release_tick == 120


@pytest.mark.parametrize("ending", [120, 123, "file_end", "pedaled_all_notes_off"])
def test_forced_midi_endings_do_not_invent_real_key_release(tmp_path, ending):
    messages = [mido.Message("note_on", note=60, velocity=90)]
    if ending == "file_end":
        messages.append(mido.MetaMessage("end_of_track", time=480))
    elif ending == "pedaled_all_notes_off":
        messages += [mido.Message("control_change", control=64, value=127),
                     mido.Message("control_change", control=123, value=0, time=480),
                     mido.Message("control_change", control=64, value=0, time=480)]
    else:
        messages.append(mido.Message("control_change", control=ending, value=0, time=480))
    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack(messages))
    source = tmp_path / "forced-end.mid"
    midi.save(source)
    event = import_project(source).notes[0]
    assert event.duration_tick == (960 if ending == "pedaled_all_notes_off" else 480)
    assert event.key_release_tick is None


def test_overlapping_same_pitch_midi_and_velocity_zero_retain_individual_gates(tmp_path):
    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack([
        mido.Message("note_on", note=60, velocity=90),
        mido.Message("note_on", note=60, velocity=80, time=120),
        mido.Message("note_off", note=60, time=120),
        mido.Message("note_on", note=60, velocity=0, time=120),
    ]))
    source = tmp_path / "overlap.mid"
    midi.save(source)
    events = import_project(source).notes
    assert [(event.start_tick, event.duration_tick, event.key_release_tick) for event in events] == [
        (0, 240, 240), (120, 240, 360)]


@pytest.mark.parametrize("offset,length,cut,gate", [
    (0xFFFFFFFF, 384, True, 672),
    (0xFFFFFFFF, 96, True, None),
    (48, 384, True, None),
    (48, 96, True, None),
    (0xFFFFFFFF, 96, False, 672),
])
def test_flp_real_gate_is_known_only_when_neither_clip_boundary_crops_note(tmp_path, offset, length, cut, gate):
    source = _flp(tmp_path / "clip-gate.flp", [_clip(position=480, offset=offset, length=length)],
                  notes=_note(length=192), cut=cut)
    event = import_project(source).notes[0]
    assert event.key_release_tick == gate


def test_local_verification_compiles_and_compares_normalized_ppq(tmp_path):
    source = _flp(tmp_path / "local.flp", [_clip()])
    reference = _midi(tmp_path / "reference.mid", ppq=480)
    before = source.read_bytes()
    result = subprocess.run([sys.executable, "scripts/verify_flp.py", str(source),
        "--compile", "--reference-midi", str(reference)], capture_output=True, text=True,
        encoding="utf-8", env=os.environ | {"PYTHONIOENCODING": "utf-8"}, check=False)
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(result.stdout)
    assert report["source_unchanged"]
    assert report["reference"]["exact_pitch_and_timing_match"]
    assert report["notation"]["quantized_notes"] == 1
    assert str(source) not in result.stdout
    assert source.read_bytes() == before


def test_reference_comparison_reports_actual_mismatch(tmp_path):
    source = _flp(tmp_path / "wrong-pitch.flp", [_clip()], notes=_note(pitch=62))
    reference = _midi(tmp_path / "reference.mid")
    result = subprocess.run([sys.executable, "scripts/verify_flp.py", str(source),
        "--reference-midi", str(reference)], capture_output=True, text=True, encoding="utf-8",
        env=os.environ | {"PYTHONIOENCODING": "utf-8"}, check=False)
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert not report["reference"]["exact_pitch_and_timing_match"]
    assert report["reference"]["missing_count"] == report["reference"]["extra_count"] == 1


def test_real_image_line_fl24_template_if_installed():
    source = Path("C:/Program Files/Image-Line/FL Studio 2026/Data/Templates/Utility/Create a chord progression/Create a chord progression.flp")
    if not source.exists():
        pytest.skip("Image-Line reference template not installed")
    project = import_project(source)
    assert project.source_version.startswith("24.")
    assert len(project.notes) == 51
    assert any(d.code == "flp_layout" and "4 个 Playlist" in d.message for d in project.diagnostics)


@pytest.mark.parametrize("name,records", [
    ("empty-fl2026-emptypatternsinplaylist.flp", 4),
    ("empty-fl2026-emptypatternsinplaylistresizedtohalf.flp", 4),
    ("empty-fl2026-sampleplacement.flp", 1),
    ("empty-fl2026-sampleplacementresized.flp", 1),
])
def test_real_public_fl26_framing_and_resized_clips_if_available(name, records):
    # Public real FL 26 saves: https://github.com/demberto/PyFLP/pull/205
    # https://github.com/user-attachments/files/31938992/empty-fl2025-fl2026.zip
    source = Path(".cache/importers/public") / name
    if not source.exists():
        pytest.skip("Public reference saves not downloaded; synthetic regression tests remain active")
    with pytest.raises(ImportFailure) as info:
        import_project(source)
    assert any(d.code == "flp_layout" and f"{records} 个 Playlist" in d.message for d in info.value.diagnostics)
    assert info.value.diagnostics[-1].code == "no_arranged_notes"
