"""Independent pre-migration Python arithmetic for native parity tests."""

import math

from stavellum.presentation.curves import ease
from stavellum.presentation.timeline import FrameLayout, RowLayout
from stavellum.rendering.raster import TilePlan

ACTIVITY_ATTACK_SECONDS = 0.1
ACTIVITY_RELEASE_SECONDS = 0.12

def activity_levels(part, time: float) -> tuple[float, float]:
    """Sustained activity and onset emphasis at absolute audio time."""
    # Envelopes indicate MIDI events, not measured audio loudness.
    level = attack = 0.0
    for note in part.notes:
        if time < note.start or note.velocity <= 0:
            continue
        if time < note.end:
            note_level = note.velocity / 127
        elif time < note.end + ACTIVITY_RELEASE_SECONDS:
            note_level = note.velocity / 127 * (1 - ease((time - note.end) / ACTIVITY_RELEASE_SECONDS))
        else:
            continue
        level = max(level, note_level)
        age = time - note.start
        if age < ACTIVITY_ATTACK_SECONDS:
            attack = max(attack, note_level * (1 - ease(age / ACTIVITY_ATTACK_SECONDS)))
    return level, attack

def layout_at(self, time: float) -> FrameLayout:
    scale = self.scale_at(time)
    rows = {}
    visible_bounds = []
    for part_id, track in self.tops.items():
        top = track.sample(time).value
        alpha = max(0.0, min(1.0, self.opacities[part_id].sample(time).value))
        height, staff_midpoint = self.part_dimensions[part_id]
        indicator_height = self.indicator_source_height * scale
        indicator_width = self.indicator_source_width * scale
        indicator_top = top + staff_midpoint * scale - indicator_height / 2
        rectangle = (self.indicator_right - indicator_width, indicator_top,
                     indicator_width, indicator_height)
        bounds = (min(top - self.padding_at(part_id, time) * scale, indicator_top),
                  max(top + height * scale, indicator_top + indicator_height))
        rows[part_id] = RowLayout(top, alpha, rectangle, bounds, self.icon_source_size * scale)
        if alpha > 1e-9:
            visible_bounds.append(bounds)
    region = self.region_at(time)
    center = sum(region) / 2
    bounds = ((min(item[0] for item in visible_bounds), max(item[1] for item in visible_bounds))
              if visible_bounds else (center, center))
    return FrameLayout(scale, rows, bounds, region)

def _raster_level(self, display_scale: float) -> int:
    """Choose the cheapest power-of-two raster that never needs upsampling."""
    level = max(0, math.floor(math.log2(self.scene.scale / display_scale)))
    # Guard roundoff at exact layer boundaries without rounding up into a
    # raster that is smaller than the requested display resolution.
    while level and math.ldexp(self.scene.scale, -level) < display_scale:
        level -= 1
    while math.ldexp(self.scene.scale, -level - 1) >= display_scale:
        level += 1
    return level

def _tile_plan(self, layout: FrameLayout, world_x: float) -> TilePlan:
    level = _raster_level(self, layout.scale)
    raster_scale = math.ldexp(self.scene.scale, -level)
    left = world_x - (self.scene.play_x - self.scene.body_left) / layout.scale
    right = world_x + (self.scene.body_right - self.scene.play_x) / layout.scale
    first = math.floor(left * raster_scale / self.TILE_PIXELS)
    last = math.floor(right * raster_scale / self.TILE_PIXELS)
    candidates = []
    working_bytes = 0
    for order, part in enumerate(self.scene.parts):
        row = layout.rows.get(part.part_id)
        if row is None or row.opacity <= 1e-6:
            continue
        size = ((self.TILE_PIXELS + 2 * self.TILE_BLEED)
                * max(1, math.ceil(part.source_height * raster_scale)) * 4)
        for index in range(first, last + 1):
            distance = abs((index + 0.5) * self.TILE_PIXELS - world_x * raster_scale)
            candidates.append((distance, order, index, (part.part_id, level, index), size))
            working_bytes += size
    remaining = self.cache_limit
    resident = set()
    for _, _, _, key, size in sorted(candidates):
        if size <= remaining:
            resident.add(key)
            remaining -= size
    return TilePlan(level, raster_scale, first, last, working_bytes, frozenset(resident))
