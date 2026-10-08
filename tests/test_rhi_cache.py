"""RHI uses the shared bounded tile plan before crossing the native boundary."""

from __future__ import annotations

import hashlib
import math

import pytest
from native_frames import frame_layout, frame_state
from test_render_cache import dense_scene as dense_scene
from test_rhi import FakeTarget

from stavellum.rendering.rhi import RhiFrameRenderer

pytestmark = pytest.mark.integration


class InspectingTarget(FakeTarget):
    def __init__(self, *args):
        super().__init__(*args)
        self.fingerprints = {}

    def texture(self, image):
        key = image.cacheKey()
        if key not in self.fingerprints:
            self.fingerprints[key] = (image.width(), image.height(),
                                      hashlib.sha256(image.constBits()).digest())
        return key


@pytest.fixture
def fake_native(monkeypatch):
    monkeypatch.setattr("stavellum.rendering.rhi.RhiTarget", InspectingTarget)


def command_pixels(renderer, seconds):
    """Include actual asset pixels and geometry, independent of QImage cache keys."""
    return [(renderer._target.fingerprints[quad.texture_id] if quad.texture_id else None,
             *(getattr(quad, name) for name, _ in quad._fields_ if name != "texture_id"))
            for quad in renderer.commands(seconds)]


def test_dense_high_resolution_warm_commands_stop_rasterizing(dense_scene, fake_native):
    with RhiFrameRenderer(dense_scene) as renderer:
        assets = renderer._assets
        layout = frame_layout(dense_scene, 1)
        plan = assets._tile_plan(frame_state(dense_scene, 1))
        assert plan.level > 0 and plan.raster_scale >= layout.scale
        assert plan.working_bytes <= assets.cache_limit
        first = command_pixels(renderer, 1)
        misses, raster_seconds = assets.tile_cache_misses, assets.svg_raster_seconds
        assert misses > 0 and raster_seconds > 0
        for _ in range(3):
            assert command_pixels(renderer, 1) == first
            assert assets.tile_cache_misses == misses
            assert assets.svg_raster_seconds == raster_seconds
        assert assets.tile_cache_hits >= 3 * len(plan.resident_keys)
        assert assets.cache_peak_bytes <= assets.cache_limit
        assert assets.visible_tile_working_peak_bytes == plan.working_bytes
        assert all(level == plan.level for _, level, _ in assets.cache)
        commands = renderer.commands(1)
        for part in dense_scene.parts:
            header = assets.headers[part.part_id]
            assert header.height() == math.ceil(part.source_height * dense_scene.scale)
            quad = next(command for command in commands if command.texture_id == header.cacheKey())
            header_ratio = layout.scale / dense_scene.scale
            assert (quad.w, quad.h) == pytest.approx(
                (header.width() * header_ratio, header.height() * header_ratio))
        report = renderer.backend_report()
        for name in ("tile_cache_hits", "tile_cache_misses", "tile_cache_evictions",
                     "svg_raster_seconds", "visible_tile_working_peak_bytes"):
            assert report[name] == getattr(assets, name)
        for seconds in (3, 0, .5, 1):
            renderer.commands(seconds)
        assert command_pixels(renderer, 1) == first
        assert assets._frame_resident_keys is None


def test_over_budget_commands_keep_hot_assets_and_pixels_across_history(dense_scene, fake_native):
    with RhiFrameRenderer(dense_scene) as renderer:
        assets = renderer._assets
        layout = frame_layout(dense_scene, 1)
        level = frame_state(dense_scene, 1).tile_level
        tile = assets._tile(dense_scene.parts[0], -30, level)
        assets.cache_limit = 2 * tile.sizeInBytes()
        plan = assets._tile_plan(frame_state(dense_scene, 1))
        visible_count = len(layout.rows) * (plan.last_index - plan.first_index + 1)
        assert 0 < len(plan.resident_keys) < visible_count
        assert plan.working_bytes > assets.cache_limit
        first = command_pixels(renderer, 1)
        hot = {key: assets.cache[key].cacheKey() for key in plan.resident_keys}
        misses, hits = assets.tile_cache_misses, assets.tile_cache_hits
        assert command_pixels(renderer, 1) == first
        assert assets.tile_cache_misses - misses == visible_count - len(plan.resident_keys)
        assert assets.tile_cache_hits - hits == len(plan.resident_keys)
        assert {key: assets.cache[key].cacheKey() for key in plan.resident_keys} == hot
        assert assets.cache_bytes <= assets.cache_limit and assets.tile_cache_evictions > 0
        assert assets._frame_resident_keys is None
        with RhiFrameRenderer(dense_scene) as clean:
            clean._assets.cache_limit = assets.cache_limit
            assert command_pixels(clean, 1) == first


@pytest.mark.parametrize("failure", ["raster", "upload"])
def test_command_failure_releases_tile_pins(dense_scene, fake_native, monkeypatch, failure):
    with RhiFrameRenderer(dense_scene) as renderer:
        owner, name = ((renderer._assets, "_tile") if failure == "raster"
                       else (renderer._target, "texture"))
        original = getattr(owner, name)

        def fail(*args):
            raise RuntimeError("asset failed")

        monkeypatch.setattr(owner, name, fail)
        with pytest.raises(RuntimeError, match="asset failed"):
            renderer.commands(1)
        assert renderer._assets._frame_resident_keys is None
        monkeypatch.setattr(owner, name, original)
        renderer.commands(1)
        assert renderer._assets.cache_bytes <= renderer._assets.cache_limit
