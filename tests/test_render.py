"""Real RGBA rendering checks independent of real-time playback and encoder timing."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import replace

import pytest
from native_frames import frame_layout
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer
from test_scene import crosspart_document, owned_elements, score_document

from stavellum.domain.models import (
    Metadata,
    NoteEvent,
    PartMapping,
    ProjectDocument,
    ProjectIR,
    RenderSettings,
    TrackInfo,
)
from stavellum.graphics.qt import ensure_app
from stavellum.presentation.scene import compile_scene
from stavellum.rendering.render import FrameRenderer
from stavellum.rendering.shared import logo_opacity

pytestmark = pytest.mark.integration


def rendered_document(*, piano=False):
    notes = [NoteEvent(f"note-{index}", "track", index * 240, 240,
                       [60, 64, 67, 72, 84, 36][index % 6], 80 + index % 30)
             for index in range(128)]
    return ProjectDocument(
        ProjectIR("test.mid", "midi", "Render", tracks=[TrackInfo("track", "Piano" if piano else "Violin")],
                  notes=notes),
        [PartMapping("part", "Piano" if piano else "Violin", ["track"],
                     instrument="piano" if piano else "violin", grand_staff=piano,
                     clef="auto" if piano else "treble", key_signature=0,
                     icon="piano" if piano else "violin", use_icon=True)],
        settings=RenderSettings(width=640, height=360, fps=30, cache_megabytes=8,
                                render_backend="cpu"),
        metadata=Metadata(title="逐帧测试", subtitle="原始 RGBA", composer="Composer"),
    )


def colored_activity_document(*, velocity=127, intro_delay=0):
    """Special techniques precede the regular source in a merged FLP part."""
    return ProjectDocument(
        ProjectIR("colors.flp", "flp", "Colored activity", tracks=[
            TrackInfo("pizz", "Violin pizzicato", color="#ef6020"),
            TrackInfo("arco", "Violin arco", color="#2496e0"),
            TrackInfo("solo", "Viola", color="#c44d82"),
        ], notes=[
            NoteEvent("pizz-note", "pizz", 0, 240, 72, velocity),
            NoteEvent("arco-note", "arco", 480, 240, 72, velocity),
            NoteEvent("solo-note", "solo", 0, 960, 60, 127),
        ], duration_ticks=1920),
        [PartMapping("part", "Violin", ["pizz", "arco"], instrument="violin",
                     clef="treble", key_signature=0, icon="violin"),
         PartMapping("solo", "Viola", ["solo"], instrument="viola",
                     clef="alto", key_signature=0, icon="viola")],
        settings=RenderSettings(width=640, height=360, render_backend="cpu",
                                intro_delay_seconds=intro_delay),
    )


@pytest.fixture(scope="module")
def scene():
    ensure_app()
    return compile_scene(rendered_document())


def pixels(frame: QImage) -> bytes:
    assert frame.format() == QImage.Format.Format_RGBA8888
    return bytes(frame.constBits())


@pytest.mark.parametrize("mode,times,expected", [
    ("persistent", [0, 0.5, 1, 6, 6.4, 6.8, 20], [0.8] * 7),
    ("fade_in", [0, 0.5, 1, 6, 20], [0, 0.4, 0.8, 0.8, 0.8]),
    ("intro", [0, 0.5, 1, 6, 6.4, 6.8, 20], [0, 0.4, 0.8, 0.8, 0.4, 0, 0]),
])
def test_logo_animation_uses_presentation_time_and_ignores_other_animation_settings(
        mode, times, expected):
    settings = RenderSettings(logo_enabled=True, logo_display_mode=mode,
                              intro_delay_seconds=10, overlay_enter_seconds=10,
                              overlay_exit_seconds=10, announcement_auto_hide=True)
    assert [logo_opacity(time, settings) for time in times] == pytest.approx(expected)
    settings.logo_enabled = False
    assert [logo_opacity(time, settings) for time in times] == [0] * len(times)


def test_logo_zero_hold_and_zero_opacity():
    settings = RenderSettings(logo_enabled=True, logo_display_mode="intro", logo_hold_seconds=0)
    assert logo_opacity(1, settings) == 0.8
    assert logo_opacity(1.4, settings) == pytest.approx(0.4)
    assert logo_opacity(1.8, settings) == 0
    settings.logo_opacity = 0
    assert logo_opacity(1, settings) == 0


@pytest.mark.parametrize("width,height", [(320, 240), (1920, 1080), (3840, 2160), (480, 800)])
def test_logo_geometry_preserves_source_aspect_and_frame_margins(scene, width, height):
    from stavellum.graphics.branding import logo_image

    settings = replace(scene.settings, width=width, height=height,
                       logo_enabled=True, logo_size_ratio=0.25)
    with FrameRenderer(replace(scene, settings=settings)) as renderer:
        image, rect, opacity = renderer.logo_overlay(0)
        source = logo_image(dark=True)
        assert not image.isNull()
        assert opacity == 0.8
        assert rect.width() / rect.height() == pytest.approx(source.width() / source.height())
        assert max(rect.width(), rect.height()) == pytest.approx(min(width, height) * 0.25)
        assert width - rect.right() == pytest.approx(min(width, height) * 0.025)
        assert height - rect.bottom() == pytest.approx(min(width, height) * 0.025)
        assert QRectF(0, 0, width, height).contains(rect)


def test_logo_pixels_are_seekable_cached_and_absent_when_disabled(scene, monkeypatch):
    from stavellum.rendering import raster as render

    calls = []
    load_image = render.logo_image

    def counted_logo(*, dark=False):
        calls.append(dark)
        return load_image(dark=dark)

    monkeypatch.setattr(render, "logo_image", counted_logo)
    settings = replace(scene.settings, logo_enabled=True, logo_display_mode="intro",
                       logo_size_ratio=0.2, intro_delay_seconds=10)
    enabled_scene = replace(scene, settings=settings)
    with FrameRenderer(enabled_scene) as renderer, FrameRenderer(scene) as disabled:
        disabled.render_frame(0)
        assert not calls and disabled._logo is None
        samples = [0, 0.5, 1, 6.4, 6.8, 20]
        expected = {time: pixels(renderer.render_frame(time)) for time in samples}
        assert calls == [True]
        first_image = renderer._logo[0]
        for time in reversed(samples):
            assert pixels(renderer.render_frame(time)) == expected[time]
        assert renderer._logo[0] is first_image
        with FrameRenderer(replace(enabled_scene, settings=replace(settings, logo_enabled=False))) as baseline:
            for time in samples:
                reference = pixels(baseline.render_frame(time))
                if time in (0, 6.8, 20):
                    assert expected[time] == reference
                else:
                    assert expected[time] != reference
        # Streamed export samples must use the same absolute-time state.
        assert [pixels(image) for _, image in renderer.export_frames([1, 0, 6.8], lambda: False)] == [
            expected[1], expected[0], expected[6.8]]
    assert renderer._logo is None


def lamp_interior(frame: QImage, rectangle) -> QImage:
    x, y, width, height = rectangle
    left, top = math.ceil(x + 1.5), math.ceil(y + 1.5)
    right, bottom = math.floor(x + width - 1.5), math.floor(y + height - 1.5)
    return frame.copy(left, top, max(1, right - left), max(1, bottom - top))


def lamp_rectangle(frame: QImage, rectangle) -> QImage:
    """Include the former border while excluding partly covered edge pixels."""
    x, y, width, height = rectangle
    left, top = math.ceil(x), math.ceil(y)
    right, bottom = math.floor(x + width), math.floor(y + height)
    return frame.copy(left, top, max(1, right - left), max(1, bottom - top))


def assert_uniform_rgb(image: QImage, expected):
    rgba = pixels(image)
    for channel, value in enumerate(expected):
        levels = list(rgba[channel::4])
        assert max(levels) - min(levels) <= 1
        assert min(levels) == pytest.approx(value, abs=1)
        assert max(levels) == pytest.approx(value, abs=1)
    assert min(rgba[3::4]) == 255


def assert_uniform_lamp_rgb(frame: QImage, rectangle, expected):
    assert_uniform_rgb(lamp_interior(frame, rectangle), expected)


def lamp_rgb(renderer, scene, time, part_id="part"):
    frame = renderer.render_frame(time)
    rectangle = frame_layout(scene, time).rows[part_id].indicator_rect
    x, y, width, height = rectangle
    rgb = frame.pixelColor(math.floor(x + width / 2), math.floor(y + height / 2)).getRgb()[:3]
    assert_uniform_lamp_rgb(frame, rectangle, rgb)
    return rgb


def fixed_header(frame, scene):
    s = scene.settings
    return frame.copy(round(s.score_left * s.width), round(s.score_top * s.height),
                      round(scene.body_left - s.score_left * s.width),
                      round((s.score_bottom - s.score_top) * s.height))


def test_random_access_frames_equal_sequential_frames_byte_for_byte(scene):
    sequential = FrameRenderer(scene)
    expected = {seconds: pixels(sequential.render_frame(seconds))
                for seconds in (0, 0.125, 0.5, 1, 4, 9.5, 12)}
    random_access = FrameRenderer(scene)
    for seconds in (12, 0.5, 4, 0, 9.5, 0.125, 1, 0.5, 12):
        assert pixels(random_access.render_frame(seconds)) == expected[seconds]
    assert expected[0] != expected[1]


def test_fixed_header_is_stationary_and_absent_from_scrolling_body(scene):
    renderer = FrameRenderer(scene)
    # Compare after the opening tempo's space has been reclaimed; subsequent
    # music scrolls while the header remains in the same column and row.
    first = renderer.render_frame(12)
    later = renderer.render_frame(13)
    assert pixels(fixed_header(first, scene)) == pixels(fixed_header(later, scene))
    # Both foreground and background exist: an all-black crop cannot satisfy this test.
    rgb = pixels(fixed_header(first, scene))
    assert max(rgb[0::4]) > 0 and min(rgb[0::4]) == 0
    groups = [node.attrib["id"] for node in ET.fromstring(scene.svg).iter()
              if "id" in node.attrib and set(node.get("class", "").split()) & {"clef", "keySig", "meterSig"}]
    assert groups
    assert all(renderer.header_renderer.elementExists(ident) for ident in groups)
    assert all(not renderer.body_renderer.elementExists(ident) for ident in groups)


def test_tile_cache_remains_bounded_and_eviction_preserves_pixels(scene):
    renderer = FrameRenderer(scene)
    # A deliberately small budget forces eviction using a short score, rather than
    # requiring a many-hour input to exercise memory bounds.
    tile_bytes = (renderer.TILE_PIXELS + 2 * renderer.TILE_BLEED) * math.ceil(scene.parts[0].source_height * scene.scale) * 4
    renderer.cache_limit = 2 * tile_bytes
    original = pixels(renderer.render_frame(0))
    initial_keys = set(renderer.cache)
    for seconds in (2, 4, 8, 12, 16, 24, 30):
        renderer.render_frame(seconds)
        assert renderer.cache_bytes <= renderer.cache_limit
        assert sum(image.sizeInBytes() for image in renderer.cache.values()) == renderer.cache_bytes
    assert renderer.cache_peak_bytes <= renderer.cache_limit
    assert initial_keys and not initial_keys.intersection(renderer.cache)
    assert pixels(renderer.render_frame(0)) == original


def test_double_staff_renders_both_staff_bands_as_one_part():
    scene = compile_scene(rendered_document(piano=True))
    frame = FrameRenderer(scene).render_frame(0)
    assert len(scene.parts) == 1
    part = scene.parts[0]
    layout = frame_layout(scene, 0)
    assert len(part.staff_centers) == 2
    for center in part.staff_centers:
        y = round(layout.rows[part.part_id].top + (center - part.source_top) * layout.scale)
        band = frame.copy(round(scene.body_right) - 10, y - 2, 6, 5)
        assert max(pixels(band)[0::4]) > 0


def test_part_renderers_exclude_neighbors_without_changing_time_geometry():
    scene = compile_scene(crosspart_document())
    original_axis = (scene.axis.beats.copy(), scene.axis.xs.copy(), scene.measure_bounds.copy())
    renderer = FrameRenderer(scene)
    whole = QSvgRenderer(scene.svg.encode("utf-8"))
    checked = set()
    for element, owner in owned_elements(scene):
        classes = set(element.get("class", "").split())
        if not classes & {"staff", "note", "dir"}:
            continue
        identity = element.attrib["id"]
        for part in scene.parts:
            body = renderer.body_renderers[part.part_id]
            assert body.elementExists(identity) == (owner == part.part_id)
        if classes & {"note", "dir"}:
            own = renderer.body_renderers[owner]
            expected = whole.transformForElement(identity).mapRect(whole.boundsOnElement(identity))
            actual = own.transformForElement(identity).mapRect(own.boundsOnElement(identity))
            assert actual.getRect() == pytest.approx(expected.getRect())
        checked.update((owner, kind) for kind in classes & {"staff", "dir"})
    assert checked == {("upper", "staff"), ("upper", "dir"),
                       ("lower", "staff"), ("lower", "dir")}
    renderer.render_frame(0)
    renderer.render_frame(2)
    assert (scene.axis.beats, scene.axis.xs, scene.measure_bounds) == original_axis


def test_extreme_ledger_lines_remain_visible_after_individual_part_cropping():
    scene = compile_scene(crosspart_document())
    renderer = FrameRenderer(scene)
    full_svg = QSvgRenderer(scene.svg.encode("utf-8"))
    frame = renderer.render_frame(0)
    parts = {part.part_id: part for part in scene.parts}
    layout = frame_layout(scene, 0)
    row_positions = {identity: row.top for identity, row in layout.rows.items()}
    samples = []
    for element, owner in owned_elements(scene):
        if "staff" not in element.get("class", "").split():
            continue
        transform = full_svg.transformForElement(element.attrib["id"])
        for group in element.iter():
            classes = set(group.get("class", "").split())
            if not classes & {"ledgerLines"} or not (list(group) and classes & {"above", "below"}):
                continue
            # Sample the outermost real ledger line, far enough from the staff
            # that a duplicated neighboring staff line cannot satisfy the test.
            path = list(group)[-1]
            x1, y1, x2, _ = map(float, re.findall(r"[-+]?(?:\d*\.\d+|\d+)", path.attrib["d"]))
            point = transform.map(QPointF((x1 + x2) / 2, y1))
            x = scene.play_x + (point.x() - scene.camera_x_at(0)) * layout.scale
            y = row_positions[owner] + (point.y() - parts[owner].source_top) * layout.scale
            assert scene.body_left < x < scene.body_right
            assert abs(point.y() - parts[owner].staff_centers[0]) > 1500
            sample = frame.copy(round(x) - 2, round(y) - 2, 5, 5)
            assert max(pixels(sample)[0::4]) > 80
            samples.append(owner)
    assert set(samples) == {"upper", "lower"}


@pytest.mark.parametrize("grand_staff", [False, True])
def test_barline_paths_belong_to_one_part_without_neighbor_fragments(grand_staff):
    document = crosspart_document()
    if grand_staff:
        document.mappings[0].instrument = "piano"
        document.mappings[0].grand_staff = True
        document.mappings[0].clef = "auto"
    scene = compile_scene(document)
    renderer = FrameRenderer(scene)
    full = QSvgRenderer(scene.svg.encode("utf-8"))
    groups = [element for element in ET.fromstring(scene.svg).iter()
              if "barLine" in element.get("class", "").split()]
    assert groups
    owners = set()
    for group in groups:
        paths = [element for element in group.iter() if element.tag.endswith("}path")]
        assert paths
        for path in paths:
            identity = path.get("id", "")
            assert identity in scene.element_part_ids
            owner = scene.element_part_ids[identity]
            owners.add(owner)
            for part in scene.parts:
                assert renderer.body_renderers[part.part_id].elementExists(identity) == (part.part_id == owner)
                assert renderer.header_renderers[part.part_id].elementExists(identity) == (part.part_id == owner)
            own = renderer.body_renderers[owner]
            before = full.transformForElement(identity).mapRect(full.boundsOnElement(identity))
            after = own.transformForElement(identity).mapRect(own.boundsOnElement(identity))
            assert after.getRect() == pytest.approx(before.getRect())
    assert owners == {"upper", "lower"}
    if grand_staff:
        assert len(scene.parts[0].staff_centers) == 2


def test_activity_indicator_lights_the_whole_interior_at_weak_velocity():
    document = rendered_document()
    document.project.notes = [NoteEvent("weak", "track", 0, 120, 72, 40)]
    scene = compile_scene(document)
    renderer = FrameRenderer(scene)
    rectangle = frame_layout(scene, 0).rows["part"].indicator_rect
    active = lamp_interior(renderer.render_frame(0), rectangle)
    levels = list(pixels(active)[0::4])
    assert max(levels) - min(levels) <= 1
    assert min(levels) == pytest.approx(255 * 40 / 127, abs=1)
    held = lamp_rgb(renderer, scene, .11)
    assert held == pytest.approx((209 * 40 / 127,) * 3, abs=1)
    rectangle = frame_layout(scene, 0.5).rows["part"].indicator_rect
    resting = lamp_interior(renderer.render_frame(0.5), rectangle)
    assert max(pixels(resting)[0::4]) == 0
    outlined = renderer.render_frame(0.5).copy(QRectF(*rectangle).toAlignedRect())
    assert max(pixels(outlined)[0::4]) > 100


@pytest.mark.parametrize("velocity", [127, 40])
def test_colored_activity_uses_regular_source_across_techniques_flash_hold_and_release(velocity):
    scene = compile_scene(colored_activity_document(velocity=velocity))
    assert {part.part_id: part.activity_color for part in scene.parts} == {
        "part": "#2496e0", "solo": "#c44d82",
    }
    with FrameRenderer(scene) as renderer:
        onset = lamp_rgb(renderer, scene, 0)
        held = lamp_rgb(renderer, scene, .125)
        assert max(onset) == pytest.approx(255 * velocity / 127, abs=1)
        assert max(held) == pytest.approx(209 * velocity / 127, abs=1)
        assert max(onset) > max(held) > 0
        assert QColor(*held).hueF() == pytest.approx(QColor("#2496e0").hueF(), abs=.01)
        assert QColor(*onset).hueF() == pytest.approx(QColor("#2496e0").hueF(), abs=.01)
        # Changing to arco repeats the flash with the same merged-part color.
        assert lamp_rgb(renderer, scene, .5) == pytest.approx(onset, abs=1)
        assert lamp_rgb(renderer, scene, .625) == pytest.approx(held, abs=1)
        solo_held = lamp_rgb(renderer, scene, .125, "solo")
        assert QColor(*solo_held).hueF() == pytest.approx(QColor("#c44d82").hueF(), abs=.01)
        assert solo_held != held
        for time in (0, .125, .25, .5, .625, .75, .31, .81):
            frame = renderer.render_frame(time)
            rectangle = frame_layout(scene, time).rows["part"].indicator_rect
            assert_uniform_rgb(lamp_rectangle(frame, rectangle), lamp_rgb(renderer, scene, time))
        for time in (.25, .75):
            assert lamp_rgb(renderer, scene, time) == pytest.approx(held, abs=1)
        # The existing 120 ms release is half-bright at its midpoint and black
        # once it has ended, without borrowing the independent part's activity.
        for time in (.31, .81):
            assert lamp_rgb(renderer, scene, time) == pytest.approx(
                tuple(value / 2 for value in held), abs=1)
        for time in (.371, .4, .871, .9):
            frame = renderer.render_frame(time)
            assert_uniform_lamp_rgb(frame, frame_layout(scene, time).rows["part"].indicator_rect,
                                    (0, 0, 0))
        quiet = renderer.render_frame(.4)
        rectangle = frame_layout(scene, .4).rows["part"].indicator_rect
        outlined = quiet.copy(QRectF(*rectangle).toAlignedRect())
        assert max(pixels(outlined)[0::4]) > 100


@pytest.mark.parametrize("source", ["#5c656a", "#619532", "#953432"])
def test_typical_fl_channel_colors_are_brighter_while_held_and_flash_for_100ms(source):
    document = rendered_document()
    document.project.tracks[0].color = source
    document.project.notes = [NoteEvent("held", "track", 0, 960, 72, 127)]
    scene = compile_scene(document)
    with FrameRenderer(scene) as renderer:
        onset = lamp_rgb(renderer, scene, 0)
        middle = lamp_rgb(renderer, scene, .05)
        held = lamp_rgb(renderer, scene, .1)
        assert max(onset) > max(middle) > max(held) > max(QColor(source).getRgb()[:3])
        for time in (.125, .5, .9):
            assert lamp_rgb(renderer, scene, time) == held
        for rgb in (onset, middle, held):
            assert QColor(*rgb).hueF() == pytest.approx(QColor(source).hueF(), abs=.015)
        for time, rgb in ((0, onset), (.05, middle), (.1, held), (.9, held)):
            frame = renderer.render_frame(time)
            assert_uniform_rgb(lamp_rectangle(frame, frame_layout(scene, time).rows["part"].indicator_rect), rgb)
        quiet = renderer.render_frame(1.121)
        rectangle = frame_layout(scene, 1.121).rows["part"].indicator_rect
        assert_uniform_lamp_rgb(quiet, rectangle, (0, 0, 0))
        assert max(pixels(lamp_rectangle(quiet, rectangle))[0::4]) > 150


def test_repeated_and_overlapping_notes_flash_without_interrupting_held_light():
    document = rendered_document()
    document.project.tracks[0].color = "#619532"
    document.project.notes = [
        NoteEvent("held", "track", 0, 960, 72, 127),
        NoteEvent("repeat", "track", 240, 240, 72, 127),
        NoteEvent("overlap", "track", 480, 240, 76, 40),
    ]
    scene = compile_scene(document)
    with FrameRenderer(scene) as renderer:
        peak = lamp_rgb(renderer, scene, 0)
        held = lamp_rgb(renderer, scene, .125)
        assert lamp_rgb(renderer, scene, .249) == held
        assert lamp_rgb(renderer, scene, .25) == peak
        assert max(held) < max(lamp_rgb(renderer, scene, .30)) < max(peak)
        assert lamp_rgb(renderer, scene, .351) == held
        assert max(held) < max(lamp_rgb(renderer, scene, .5)) < max(peak)
        assert lamp_rgb(renderer, scene, .625) == held
        assert lamp_rgb(renderer, scene, .9) == held


def test_short_notes_flash_then_release_and_restore_outline_after_120ms():
    document = rendered_document()
    document.project.tracks[0].color = "#953432"
    document.project.notes = [NoteEvent("short", "track", 0, 48, 72, 127)]
    scene = compile_scene(document)
    with FrameRenderer(scene) as renderer:
        brightness = [max(lamp_rgb(renderer, scene, time)) for time in (0, .025, .05, .11)]
        assert all(first > second > 0 for first, second in zip(brightness, brightness[1:]))
        tail = renderer.render_frame(.11)
        rectangle = frame_layout(scene, .11).rows["part"].indicator_rect
        assert_uniform_rgb(lamp_rectangle(tail, rectangle), lamp_rgb(renderer, scene, .11))
        quiet = renderer.render_frame(.171)
        rectangle = frame_layout(scene, .171).rows["part"].indicator_rect
        assert_uniform_lamp_rgb(quiet, rectangle, (0, 0, 0))
        assert max(pixels(lamp_rectangle(quiet, rectangle))[0::4]) > 150


def test_colored_activity_is_dark_during_intro_and_random_seeks_preserve_pixels():
    scene = compile_scene(colored_activity_document(velocity=40, intro_delay=2))
    times = (0, 1, 1.999, 2, 2.025, 2.05, 2.1, 2.125, 2.31, 2.4, 2.5, 2.55, 2.6, 2.81, 2.9)
    with FrameRenderer(scene) as sequential:
        expected = {time: pixels(sequential.render_frame(time)) for time in times}
        for time in (0, 1, 1.999):
            frame = sequential.render_frame(time)
            for row in frame_layout(scene, time).rows.values():
                assert_uniform_lamp_rgb(frame, row.indicator_rect, (0, 0, 0))
        assert max(lamp_rgb(sequential, scene, 2)) > max(lamp_rgb(sequential, scene, 2.1)) > 0
    with FrameRenderer(scene) as random_access:
        for time in reversed(times + (2.5, 2.025, 0, 2.55, 2.1)):
            assert pixels(random_access.render_frame(time)) == expected[time]


@pytest.mark.parametrize("offset", [-.125, .75])
def test_colored_flashes_follow_audio_offset_without_replaying_elapsed_onsets(offset):
    baseline = compile_scene(colored_activity_document(velocity=40, intro_delay=2))
    document = colored_activity_document(velocity=40, intro_delay=2)
    document.settings.score_start_in_audio_sec = offset
    shifted = compile_scene(document)
    with FrameRenderer(baseline) as reference, FrameRenderer(shifted) as renderer:
        for time in (0, 1.999):
            for row in frame_layout(shifted, time).rows.values():
                assert_uniform_lamp_rgb(renderer.render_frame(time), row.indicator_rect, (0, 0, 0))
        for audio_time in (0, .05, .125, .25, .375, .5, .75, .9, 1.1):
            for part_id in ("part", "solo"):
                actual = lamp_rgb(renderer, shifted, 2 + audio_time, part_id)
                wanted = lamp_rgb(reference, baseline, 2 + audio_time - offset, part_id)
                assert actual == pytest.approx(wanted, abs=1)
        if offset < 0:
            assert lamp_rgb(renderer, shifted, 2) == lamp_rgb(reference, baseline, 2.125)
            assert max(lamp_rgb(renderer, shifted, 2)) < max(lamp_rgb(reference, baseline, 2))


def test_recompiling_refreshes_activity_colors_after_source_and_mapping_edits():
    document = colored_activity_document()
    original = compile_scene(document)
    document.project.tracks[1].color = "#8844cc"
    recolored = compile_scene(document)
    assert original.parts[0].activity_color == "#2496e0"
    assert recolored.parts[0].activity_color == "#8844cc"
    document.mappings[0].articulations = {"pizz": "normal", "arco": "staccato"}
    remapped = compile_scene(document)
    assert remapped.parts[0].activity_color == "#ef6020"
    document.mappings = [
        PartMapping("pizz", "Violin pizzicato", ["pizz"], clef="treble", key_signature=0),
        PartMapping("arco", "Violin arco", ["arco"], clef="treble", key_signature=0),
    ]
    split = compile_scene(document)
    assert {part.part_id: part.activity_color for part in split.parts} == {
        "pizz": "#ef6020", "arco": "#8844cc",
    }


def test_indicator_sizes_match_across_staves_and_grow_with_changing_layout():
    document = score_document()
    document.mappings[0].instrument = "piano"
    document.mappings[0].grand_staff = True
    document.mappings[0].clef = "auto"
    scene = compile_scene(document)
    assert len(scene.parts[0].staff_centers) == 2
    sizes = set()
    scales = set()
    for seconds in (0, 1, 3, 4, 8, 12, 14, 16, 18, 24):
        layout = frame_layout(scene, seconds)
        scales.add(round(layout.scale, 8))
        for row in layout.rows.values():
            x, y, width, height = row.indicator_rect
            assert width == pytest.approx(195 * layout.scale)
            assert height == pytest.approx(720 * layout.scale)
            assert x + width == pytest.approx(scene.layout.indicator_right)
            sizes.add((width, height))
    assert len(sizes) > 1
    assert len(scales) > 1


def test_zooming_random_access_frames_and_cache_remain_deterministic():
    document = score_document()
    # Exercise interrupted zooms even when a short target would be suppressed
    # by the default stable-state filter; dense default layouts are checked in
    # test_render_cache.py independently.
    document.settings.animation_stable_seconds = 0
    document.settings.score_bottom = 0.40
    scene = compile_scene(document)
    sequential = FrameRenderer(scene)
    times = (0, 0.1, 1, 3, 4, 10, 14, 15, 16, 18)
    expected = {seconds: pixels(sequential.render_frame(seconds)) for seconds in times}
    random_access = FrameRenderer(scene)
    for seconds in reversed(times):
        assert pixels(random_access.render_frame(seconds)) == expected[seconds]
        assert random_access.cache_bytes <= random_access.cache_limit
    assert len({frame_layout(scene, seconds).scale for seconds in times}) > 1


def test_small_output_single_indicator_is_large_enough_to_show_activity():
    document = rendered_document()
    document.settings.width, document.settings.height = 320, 240
    document.project.notes = [NoteEvent("bright", "track", 0, 120, 72, 127)]
    scene = compile_scene(document)
    rectangle = frame_layout(scene, 0).rows["part"].indicator_rect
    assert rectangle[2] >= 5 and rectangle[3] >= 20
    renderer = FrameRenderer(scene)
    active = lamp_interior(renderer.render_frame(0), rectangle)
    quiet = lamp_interior(renderer.render_frame(0.5), frame_layout(scene, 0.5).rows["part"].indicator_rect)
    assert min(pixels(active)[0::4]) == 255
    assert max(pixels(quiet)[0::4]) == 0


def test_opening_pause_freezes_score_and_keeps_first_note_dark():
    document = rendered_document()
    document.settings.intro_delay_seconds = 2.125
    scene = compile_scene(document)
    renderer = FrameRenderer(scene)

    def score(frame):
        top = math.ceil(frame_layout(scene, 0).rows["part"].top)
        return pixels(frame.copy(0, top, document.settings.width,
                                 math.floor(scene.settings.score_bottom * document.settings.height) - top))

    first = renderer.render_frame(0)
    assert score(first) == score(renderer.render_frame(1.125))
    assert score(first) == score(renderer.render_frame(2.124))
    rectangle = frame_layout(scene, 0).rows["part"].indicator_rect
    assert max(pixels(lamp_interior(first, rectangle))[0::4]) == 0
    started = renderer.render_frame(2.125)
    assert max(pixels(lamp_interior(started, rectangle))[0::4]) > 100
    assert score(first) != score(renderer.render_frame(2.25))
    expected = {time: pixels(renderer.render_frame(time)) for time in (0, 1, 2.125, 3.25, 12)}
    random = FrameRenderer(scene)
    for time in (12, 1, 3.25, 0, 2.125, 1):
        assert pixels(random.render_frame(time)) == expected[time]


def test_metadata_group_fades_during_pause_then_optionally_disappears():
    document = rendered_document()
    document.settings.intro_delay_seconds = 10
    document.settings.announcement_auto_hide = True
    document.settings.announcement_hold_seconds = 0.5
    scene = compile_scene(document)
    renderer = FrameRenderer(scene)

    def metadata(time):
        frame = renderer.render_frame(time)
        return pixels(frame.copy(0, round(document.settings.height * 0.70),
                                 math.floor(scene.settings.score_left * scene.settings.width - 110 * scene.settings.width / 1920),
                                 round(document.settings.height * 0.30)))

    assert max(metadata(0)[0::4]) == 0
    middle, full = metadata(0.5), metadata(1)
    assert 0 < max(middle[0::4]) < max(full[0::4])
    assert metadata(1.5) == full
    assert 0 < max(metadata(1.9)[0::4]) < max(full[0::4])
    assert max(metadata(2.3)[0::4]) == 0
    document.settings.announcement_auto_hide = False
    scene = compile_scene(document)
    renderer = FrameRenderer(scene)
    assert metadata(8) == full


def test_hidden_announcement_reclaims_space_during_intro_and_random_seeks_match():
    document = rendered_document(piano=True)
    document.settings.intro_delay_seconds = 10
    document.settings.announcement_auto_hide = True
    document.settings.announcement_hold_seconds = 0.5
    document.settings.score_bottom = 0.42
    scene = compile_scene(document)
    renderer = FrameRenderer(scene)
    hide_end = 2.3
    initial = frame_layout(scene, hide_end)
    enlarged = frame_layout(scene, 4)
    assert initial.region_bounds[1] == pytest.approx(0.42 * scene.settings.height)
    assert enlarged.region_bounds[1] == pytest.approx(0.95 * scene.settings.height)
    assert enlarged.scale > initial.scale
    assert scene.camera_x_at(0) == scene.camera_x_at(4)
    frame = renderer.render_frame(4)
    row = enlarged.rows["part"]
    for center in scene.parts[0].staff_centers:
        y = round(row.top + (center - scene.parts[0].source_top) * enlarged.scale)
        assert max(pixels(frame.copy(round(scene.body_right) - 10, y - 2, 6, 5))[0::4]) > 100
    assert max(pixels(lamp_interior(frame, row.indicator_rect))[0::4]) == 0
    expected = {time: pixels(renderer.render_frame(time)) for time in (0, 2.3, 2.475, 4, 10, 11)}
    random = FrameRenderer(scene)
    for time in (11, 2.475, 0, 10, 4, 2.3, 4):
        assert pixels(random.render_frame(time)) == expected[time]


@pytest.mark.parametrize("bpm", [120, 112.5])
def test_tempo_appears_at_first_beat_then_scrolls_out_without_timed_fades(bpm, monkeypatch):
    import stavellum.rendering.raster as render_module

    document = score_document()
    document.project.bpm = bpm
    document.settings.intro_delay_seconds = 2
    document.settings.overlay_enter_seconds = 10
    document.settings.overlay_exit_seconds = 10
    scene = compile_scene(document)
    # The painter-internals assertions below are defined by the CPU raster;
    # pin it so an installed native backend cannot change what is captured.
    document.settings.render_backend = "cpu"
    scene = replace(scene, settings=replace(scene.settings, render_backend="cpu"))
    renderer = FrameRenderer(scene)
    labels = []
    original = render_module.draw_text_rect

    def capture(painter, text, size, rectangle, **kwargs):
        labels.append((text, size, rectangle.getRect(), painter.transform().m11(),
                       painter.transform().dx(), painter.transform().dy(), painter.opacity()))
        return original(painter, text, size, rectangle, **kwargs)

    monkeypatch.setattr(render_module, "draw_text_rect", capture)

    def tempo(time):
        frame = QImage(scene.settings.width, scene.settings.height, QImage.Format.Format_RGBA8888)
        frame.fill(Qt.GlobalColor.transparent)
        painter = QPainter(frame)
        renderer._draw_tempo(painter, time, frame_layout(scene, time), scene.camera_x_at(time))
        painter.end()
        return pixels(frame)

    full = tempo(0)
    assert len(labels) == 1 and labels[0][0] == f"= {bpm}"
    assert max(full[3::4]) > 100
    assert tempo(1.9) == full
    exit_time = scene.layout.tempo_exit_time
    assert math.isfinite(exit_time) and exit_time > 2
    for time in (0, 2, (2 + exit_time) / 2):
        assert max(tempo(time)[3::4]) > 0
        layout = frame_layout(scene, time)
        assert labels[-1][1] == 270
        assert labels[-1][3] == pytest.approx(layout.scale)
        assert labels[-1][4] == pytest.approx(scene.play_x + (scene.axis.x_at(0) - scene.camera_x_at(time)) * layout.scale)
        owner = layout.rows[scene.tempo_mark.owner_id]
        assert labels[-1][5] == pytest.approx(owner.top - scene.tempo_mark.padding * layout.scale)
        assert labels[-1][6] == 1
    for time in (exit_time, exit_time + 0.1, 4, 12, 18):
        assert max(tempo(time)[3::4]) == 0


@pytest.mark.parametrize("piano", [False, True])
@pytest.mark.parametrize("offset", [-0.25, 0.5])
def test_staff_extensions_turn_gray_only_right_of_real_terminal_bar(piano, offset):
    document = rendered_document(piano=piano)
    document.settings.score_start_in_audio_sec = offset
    document.settings.intro_delay_seconds = 0.375
    scene = compile_scene(document)
    renderer = FrameRenderer(scene)
    assert scene.terminal_x is not None
    # Invert the complete displayed geometry, including the opening reflow.
    positions = (scene.body_right + 20, (scene.body_left + scene.body_right) / 2,
                 scene.body_left - 20)
    for terminal_position in positions:
        low, high = scene.settings.intro_delay_seconds, scene.score_duration + abs(offset) + 20
        for _ in range(60):
            middle = (low + high) / 2
            displayed = scene.play_x + (scene.terminal_x - scene.camera_x_at(middle)) * frame_layout(scene, middle).scale
            if displayed > terminal_position:
                low = middle
            else:
                high = middle
        time = (low + high) / 2
        frame = renderer.render_frame(time)
        part = scene.parts[0]
        row = frame_layout(scene, time).rows[part.part_id]
        scale = frame_layout(scene, time).scale
        for center in part.staff_centers:
            y = round(row.top + (center - part.source_top) * scale)
            for x in (round(scene.body_left + 8), round(scene.body_right - 8)):
                # A band maximum tolerates anti-aliasing on a subpixel staff center.
                band = frame.copy(x - 2, y - 2, 5, 5)
                intensity = max(pixels(band)[0::4])
                if x > terminal_position:
                    assert 25 < intensity <= 103
                else:
                    assert intensity > 103
        assert pixels(FrameRenderer(scene).render_frame(time)) == pixels(frame)
