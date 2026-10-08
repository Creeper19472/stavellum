"""Dense frame working sets stay bounded without rerasterizing their hot tiles."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
from native_frames import frame_layout, frame_state, raster_state
from test_render import pixels, rendered_document

from stavellum.models import NoteEvent, TrackInfo
from stavellum.render import FrameRenderer
from stavellum.scene import compile_scene

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", params=[(2560, 1440), (3840, 2160)])
def dense_scene(request):
    document = rendered_document()
    document.settings.width, document.settings.height = request.param
    document.settings.cache_megabytes = 64
    document.project.tracks = [TrackInfo(f"track-{index}", f"Part {index}")
                               for index in range(7)]
    document.project.notes = [
        NoteEvent(f"note-{part}-{index}", f"track-{part}", index * 240, 240,
                  [36, 60, 84, 72][index % 4], 100)
        for part in range(7) for index in range(16)
    ]
    document.mappings = [replace(document.mappings[0], part_id=f"part-{index}",
                                 name=f"Part {index}", track_ids=[f"track-{index}"])
                         for index in range(7)]
    return compile_scene(document)


def test_dense_high_resolution_warm_frames_stop_rasterizing(dense_scene):
    with FrameRenderer(dense_scene) as renderer:
        time = 1.0
        layout = frame_layout(dense_scene, time)
        plan = renderer._tile_plan(frame_state(dense_scene, time))
        assert plan.level > 0
        assert plan.raster_scale >= layout.scale
        assert plan.working_bytes <= renderer.cache_limit
        first = pixels(renderer.render_frame(time))
        misses, raster_seconds = renderer.tile_cache_misses, renderer.svg_raster_seconds
        assert misses > 0 and raster_seconds > 0
        for _ in range(3):
            assert pixels(renderer.render_frame(time)) == first
            assert renderer.tile_cache_misses == misses
            assert renderer.svg_raster_seconds == raster_seconds
        assert renderer.tile_cache_hits >= 3 * len(plan.resident_keys)
        assert renderer.cache_peak_bytes <= renderer.cache_limit
        assert renderer.visible_tile_working_peak_bytes == plan.working_bytes
        # Headers retain their maximum raster resolution while body tiles use
        # the smaller layer; changing cache history never changes either image.
        for part in dense_scene.parts:
            assert renderer.headers[part.part_id].height() == math.ceil(
                part.source_height * dense_scene.scale)
        for other_time in (3.0, 0.0, 0.5, 1.0):
            renderer.render_frame(other_time)
        assert pixels(renderer.render_frame(time)) == first
        report = renderer.backend_report()
        assert report["tile_cache_hits"] == renderer.tile_cache_hits
        assert report["tile_cache_misses"] == renderer.tile_cache_misses
        assert report["svg_raster_seconds"] == renderer.svg_raster_seconds
        assert report["visible_tile_working_peak_bytes"] <= renderer.cache_limit


def test_over_budget_frame_preserves_its_resident_subset(dense_scene):
    with FrameRenderer(dense_scene) as renderer:
        layout = frame_layout(dense_scene, 1)
        level = frame_state(dense_scene, 1).tile_level
        tile = renderer._tile(dense_scene.parts[0], -30, level)
        renderer.cache_limit = 2 * tile.sizeInBytes()
        plan = renderer._tile_plan(frame_state(dense_scene, 1))
        visible_count = len(layout.rows) * (plan.last_index - plan.first_index + 1)
        assert 0 < len(plan.resident_keys) < visible_count
        assert plan.working_bytes > renderer.cache_limit
        # Include prior frames/layers in the cache before the overloaded frame.
        first = pixels(renderer.render_frame(1))
        resident_images = {key: renderer.cache[key].cacheKey() for key in plan.resident_keys}
        misses, hits = renderer.tile_cache_misses, renderer.tile_cache_hits
        assert pixels(renderer.render_frame(1)) == first
        assert renderer.tile_cache_misses - misses == visible_count - len(plan.resident_keys)
        assert renderer.tile_cache_hits - hits == len(plan.resident_keys)
        assert {key: renderer.cache[key].cacheKey() for key in plan.resident_keys} == resident_images
        assert renderer.cache_bytes <= renderer.cache_limit
        assert renderer.tile_cache_evictions > 0
        assert renderer._frame_resident_keys is None
        # Different history selects the same quality and pinned subset.
        with FrameRenderer(dense_scene) as clean:
            clean.cache_limit = renderer.cache_limit
            assert clean._tile_plan(frame_state(dense_scene, 1)).resident_keys == plan.resident_keys
            assert pixels(clean.render_frame(1)) == first


def test_raster_layer_boundaries_never_upsample_or_depend_on_cache(dense_scene):
    with FrameRenderer(dense_scene):
        for exponent in range(7):
            boundary = math.ldexp(dense_scene.scale, -exponent)
            for scale in (math.nextafter(boundary, 0), boundary,
                          math.nextafter(boundary, math.inf)):
                if scale > dense_scene.scale:
                    continue
                state = raster_state(dense_scene, scale)
                scale = state.layout.scale
                raster_scale = state.tile_raster_scale
                assert raster_scale >= scale
                assert raster_scale / 2 < scale


def test_single_oversized_tile_bypasses_cache_and_keeps_random_access(dense_scene):
    with FrameRenderer(dense_scene) as renderer:
        renderer.cache_limit = 1
        first = pixels(renderer.render_frame(1))
        assert not renderer.cache and renderer.cache_bytes == 0
        misses = renderer.tile_cache_misses
        renderer.render_frame(0)
        assert pixels(renderer.render_frame(1)) == first
        assert renderer.tile_cache_misses > misses
        assert renderer.tile_cache_hits == renderer.tile_cache_evictions == 0


def test_frame_error_releases_tile_pins(dense_scene, monkeypatch):
    with FrameRenderer(dense_scene) as renderer:
        original = renderer._tile

        def failed(*args, **kwargs):
            raise RuntimeError("drawing failed")

        monkeypatch.setattr(renderer, "_tile", failed)
        with pytest.raises(RuntimeError, match="drawing failed"):
            renderer.render_frame(1)
        assert renderer._frame_resident_keys is None
        monkeypatch.setattr(renderer, "_tile", original)
        renderer.render_frame(1)
        assert renderer.cache_bytes <= renderer.cache_limit
