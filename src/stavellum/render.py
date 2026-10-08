"""Shared preview/export renderer; every image is a pure function of scene and time."""

from __future__ import annotations

import math
import time as clock
import xml.etree.ElementTree as ET
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen
from PySide6.QtSvg import QSvgRenderer

from ._frame import FrameEvaluator, FrameState
from .branding import logo_image
from .gpu import GpuBackendError
from .icons import draw_image_icon, resolve_icon
from .layout import FrameLayout, ease
from .models import RenderSettings
from .musicfont import metronome_renderer
from .qt import prepare_render_app
from .scene import CompiledScene, ScenePart
from .typography import _layout, draw_text, draw_text_rect

TileKey = tuple[str, int, int]


@dataclass(frozen=True)
class TilePlan:
    """One frame's raster level and bounded, history-independent resident set."""

    level: int
    raster_scale: float
    first_index: int
    last_index: int
    working_bytes: int
    resident_keys: frozenset[TileKey]


class _ExportTimes:
    """Keep at most one lookahead time when a tile boundary splits a batch."""

    def __init__(self, times, cancel):
        self._times, self.cancel = iter(times), cancel
        self._lookahead = deque()
        self.budget_splits = 0
        self.tile_splits = 0

    def check_cancel(self):
        if self.cancel():
            raise InterruptedError("已取消导出。")

    def next_time(self):
        self.check_cancel()
        return self._lookahead.popleft() if self._lookahead else next(self._times)

    def take_batch(self, renderer):
        times = [self.next_time()]
        limit = renderer.export_batch_limit
        if limit == 1:
            return times
        plan = renderer.batch_plan(times[0])
        self.check_cancel()
        if plan is None:
            self.budget_splits += 1
            return times
        while len(times) < limit:
            try:
                seconds = self.next_time()
            except StopIteration:
                break
            following = renderer.batch_plan(seconds)
            self.check_cancel()
            if following != plan:
                self._lookahead.append(seconds)
                if following is None:
                    self.budget_splits += 1
                else:
                    self.tile_splits += 1
                break
            times.append(seconds)
        return times

    def close(self):
        self._lookahead.clear()
        self._times = iter(())


def _export_native_delta(current, baseline):
    counters = ("gpu_submit_seconds", "fence_wait_seconds", "memory_copy_seconds",
                "scene_evaluation_count", "scene_evaluation_seconds", "scene_evaluation_cache_hits",
                "memory_copy_bytes", "gpu_submitted_frame_count", "owned_readback_frame_count",
                "copied_readback_frame_count", "inplace_format_conversion_seconds",
                "inplace_format_conversion_bytes", "gpu_submission_count",
                "synchronous_readback_seconds")
    values = {key: max(0, current.get(key, 0) - baseline.get(key, 0)) for key in counters}
    before = baseline.get("gpu_batch_size_histogram", {})
    values["gpu_batch_size_histogram"] = {
        size: max(0, count - before.get(size, 0))
        for size, count in current.get("gpu_batch_size_histogram", {}).items()}
    return values


def overlay_opacity(time: float, settings: RenderSettings, *, hide: bool, hold: float) -> float:
    """Absolute presentation-time opacity, including the fully visible hold."""
    opacity = ease(time / settings.overlay_enter_seconds)
    if hide:
        opacity *= 1 - ease((time - settings.overlay_enter_seconds - hold)
                           / settings.overlay_exit_seconds)
    return opacity


def activity_lamp_color(part: ScenePart, activity: tuple[float, float]) -> QColor | None:
    """Brighten the Rack hue, with headroom for a brief note-on highlight."""
    level, attack = activity
    alpha = round(level * 255)
    if alpha <= 0:
        return None
    flash = min(1.0, attack / level)
    hue, saturation, value, _ = QColor(part.activity_color).getHsvF()
    held_value = min(0.82, value * 1.5 + 0.10)
    peak_value = min(1.0, held_value + 0.22)
    color = QColor.fromHsvF(
        hue, saturation * (1 - 0.15 * flash),
        held_value + (peak_value - held_value) * flash,
    )
    color.setAlpha(alpha)
    return color


def logo_opacity(time: float, settings: RenderSettings) -> float:
    """Absolute presentation-time logo state, independent of playback history."""
    if not settings.logo_enabled:
        return 0.0
    opacity = settings.logo_opacity
    if settings.logo_display_mode != "persistent":
        opacity *= ease(time / settings.logo_enter_seconds)
    if settings.logo_display_mode == "intro":
        if time >= settings.logo_enter_seconds + settings.logo_hold_seconds + settings.logo_exit_seconds:
            return 0.0
        opacity *= 1 - ease((time - settings.logo_enter_seconds - settings.logo_hold_seconds)
                           / settings.logo_exit_seconds)
    return max(0.0, min(settings.logo_opacity, opacity))


class RasterFrameRenderer:
    """CPU raster composition and cached assets shared by the Vulkan compositor."""

    TILE_PIXELS = 1024
    TILE_BLEED = 2

    def __init__(self, scene: CompiledScene):
        prepare_render_app(scene.settings)
        self.scene = scene
        self.requested_backend = getattr(scene.settings, "render_backend", "auto")
        self.render_backend = "cpu"
        self.fallback_reasons: list[str] = []
        self.gpu_info: dict[str, str | int] = {}
        self.gpu_texture_cache_budget_bytes = 0
        self._closed = False
        self.render_seconds = 0.0
        self.readback_seconds = 0.0
        self.frame_count = 0
        self._export_stream: ExportFrameStream | None = None
        self._export_report = {
            "gpu_submit_seconds": 0.0,
            "fence_wait_seconds": 0.0,
            "memory_copy_seconds": 0.0,
            "gpu_submitted_frame_count": 0,
            "readback_buffer_peak_bytes": 0,
            "readback_mode": "unused",
            "readback_mode_history": [],
            "readback_fallback_reasons": [],
        }
        self._evaluator = FrameEvaluator(scene)
        try:
            self._prepare_assets()
        except BaseException:
            self._evaluator.close()
            raise

    def _prepare_assets(self) -> None:
        scene = self.scene
        self.tempo_renderer = metronome_renderer()
        root = ET.fromstring(scene.svg)
        self.header_renderers: dict[str, QSvgRenderer] = {}
        self.body_renderers: dict[str, QSvgRenderer] = {}
        for part in scene.parts:
            part_root = deepcopy(root)
            for parent in part_root.iter():
                for child in list(parent):
                    owner = scene.element_part_ids.get(child.get("id", ""))
                    if owner is not None and owner != part.part_id:
                        parent.remove(child)
            self.header_renderers[part.part_id] = QSvgRenderer(ET.tostring(part_root, encoding="utf-8"))
            for parent in part_root.iter():
                for child in list(parent):
                    if set(child.get("class", "").split()).intersection({"clef", "keySig", "meterSig", "label", "labelAbbr", "mNum"}):
                        parent.remove(child)
            self.body_renderers[part.part_id] = QSvgRenderer(ET.tostring(part_root, encoding="utf-8"))
        # Convenient inspection handles for single-part scenes.
        self.header_renderer = self.header_renderers[scene.parts[0].part_id]
        self.body_renderer = self.body_renderers[scene.parts[0].part_id]
        self.cache: OrderedDict[TileKey, QImage] = OrderedDict()
        self._frame_resident_keys: frozenset[TileKey] | None = None
        self.headers: dict[str, QImage] = {}
        self._gpu_icons: dict[str, QImage] = {}
        self._gpu_tempo: QImage | None = None
        self._octave_labels: dict[str, QImage] = {}
        self._logo: tuple[QImage, QRectF] | None = None
        self.cache_bytes = 0
        self.cache_peak_bytes = 0
        self.cache_limit = scene.settings.cache_megabytes * 1024 * 1024
        self.tile_cache_hits = 0
        self.tile_cache_misses = 0
        self.tile_cache_evictions = 0
        self.svg_raster_seconds = 0.0
        self.visible_tile_working_peak_bytes = 0
        self.font_sizes = {}
        for key, size in (("title", scene.settings.title_font_size), ("subtitle", scene.settings.subtitle_font_size), ("credits", scene.settings.credits_font_size)):
            self.font_sizes[key] = max(8, round(size * scene.settings.height / 1080))

    @property
    def backend(self) -> str:
        return self.render_backend

    def backend_report(self) -> dict:
        """Report measured CPU cache/timing separately from configured GPU memory."""
        return {
            **self._export_report,
            **self._evaluator.report(),
            "requested_render_backend": self.requested_backend,
            "render_backend": self.render_backend,
            "gpu_info": dict(self.gpu_info),
            "render_fallback_reasons": list(self.fallback_reasons),
            "render_seconds": self.render_seconds,
            "readback_seconds": self.readback_seconds,
            "rendered_frame_count": self.frame_count,
            "cache_peak_bytes": self.cache_peak_bytes,
            "cache_limit_bytes": self.cache_limit,
            "tile_cache_hits": self.tile_cache_hits,
            "tile_cache_misses": self.tile_cache_misses,
            "tile_cache_evictions": self.tile_cache_evictions,
            "svg_raster_seconds": self.svg_raster_seconds,
            "visible_tile_working_peak_bytes": self.visible_tile_working_peak_bytes,
            "gpu_texture_cache_budget_bytes": self.gpu_texture_cache_budget_bytes,
            "gpu_texture_cache_measured": False,
        }

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._export_stream is not None:
                self._export_stream.close()
        finally:
            self._evaluator.close()
            self.cache.clear()
            self.headers.clear()
            self._gpu_icons.clear()
            self._gpu_tempo = None
            self._octave_labels.clear()
            self._logo = None
            self.cache_bytes = 0
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def export_frames(self, times: Iterable[float], cancel: Callable[[], bool]):
        """Own a bounded, ordered export stream; public single frames remain RGBA."""
        if self._closed:
            raise RuntimeError("逐帧绘图器已经关闭。")
        if self._export_stream is not None:
            raise RuntimeError("同一个绘图器只能同时打开一个导出帧流。")
        stream = ExportFrameStream(self, times, cancel)
        self._export_stream = stream
        return stream

    def _crop(self, part: ScenePart, source_x: float, pixel_width: int, renderer: QSvgRenderer,
              *, raster_scale: float | None = None) -> QImage:
        scale = self.scene.scale if raster_scale is None else raster_scale
        started = clock.perf_counter()
        image = QImage(pixel_width, max(1, math.ceil(part.source_height * scale)), QImage.Format.Format_RGBA8888)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        painter.scale(scale, scale)
        painter.translate(-source_x, -part.source_top)
        try:
            renderer.render(painter, QRectF(*self.scene.view_box))
        finally:
            painter.end()
            self.svg_raster_seconds += clock.perf_counter() - started
        return image

    def _tile_plan(self, state: FrameState) -> TilePlan:
        layout, world_x = state.layout, state.world_x
        level, raster_scale = state.tile_level, state.tile_raster_scale
        first, last = state.tile_first, state.tile_last
        candidates = []
        for order, part in enumerate(self.scene.parts):
            row = layout.rows.get(part.part_id)
            if row is None or row.opacity <= 1e-6:
                continue
            size = ((self.TILE_PIXELS + 2 * self.TILE_BLEED)
                    * max(1, math.ceil(part.source_height * raster_scale)) * 4)
            for index in range(first, last + 1):
                distance = abs((index + 0.5) * self.TILE_PIXELS - world_x * raster_scale)
                candidates.append((distance, order, index, (part.part_id, level, index), size))
        remaining = self.cache_limit
        resident = set()
        for _, _, _, key, size in sorted(candidates):
            if size <= remaining:
                resident.add(key)
                remaining -= size
        return TilePlan(level, raster_scale, first, last, state.tile_working_bytes, frozenset(resident))

    @contextmanager
    def _pinned_tiles(self, keys: frozenset[TileKey]):
        previous = self._frame_resident_keys
        self._frame_resident_keys = keys
        try:
            yield
        finally:
            self._frame_resident_keys = previous

    def _tile(self, part: ScenePart, index: int, level: int = 0) -> QImage:
        # Keep maximum resolution as the default for direct tile callers.
        key = (part.part_id, level, index)
        cached = self.cache.get(key)
        if cached is not None:
            self.tile_cache_hits += 1
            self.cache.move_to_end(key)
            return cached
        self.tile_cache_misses += 1
        raster_scale = math.ldexp(self.scene.scale, -level)
        source_x = (index * self.TILE_PIXELS - self.TILE_BLEED) / raster_scale
        image = self._crop(part, source_x, self.TILE_PIXELS + 2 * self.TILE_BLEED,
                           self.body_renderers[part.part_id], raster_scale=raster_scale)
        size = image.sizeInBytes()
        if size > self.cache_limit or (self._frame_resident_keys is not None
                                      and key not in self._frame_resident_keys):
            # Draw an uncached image once; do not displace this frame's hot set.
            return image
        while self.cache and self.cache_bytes + size > self.cache_limit:
            victim = next((candidate for candidate in self.cache
                           if self._frame_resident_keys is None
                           or candidate not in self._frame_resident_keys), None)
            if victim is None:
                return image
            removed = self.cache.pop(victim)
            self.cache_bytes -= removed.sizeInBytes()
            self.tile_cache_evictions += 1
        if size <= self.cache_limit:
            self.cache[key] = image
            self.cache_bytes += size
            self.cache_peak_bytes = max(self.cache_peak_bytes, self.cache_bytes)
        return image

    def _header(self, part: ScenePart) -> QImage:
        if part.part_id not in self.headers:
            width = math.ceil((self.scene.header_right - self.scene.header_left) * self.scene.scale)
            self.headers[part.part_id] = self._crop(part, self.scene.header_left, max(1, width), self.header_renderers[part.part_id])
        return self.headers[part.part_id]

    def _octave_image(self, label: str) -> QImage:
        if label not in self._octave_labels:
            text = _layout(f"({label})", 240 * self.scene.scale, italic=True)
            bounds = text.boundingRect()
            image = QImage(max(1, math.ceil(bounds.width()) + 4),
                           max(1, math.ceil(bounds.height()) + 4),
                           QImage.Format.Format_RGBA8888)
            image.fill(Qt.GlobalColor.transparent)
            painter = QPainter(image)
            try:
                painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
                painter.setPen(QColor("white"))
                text.draw(painter, QPointF(2 - bounds.left(), 2 - bounds.top()))
            finally:
                painter.end()
            self._octave_labels[label] = image
        return self._octave_labels[label]

    def _octave_overlays(self, part: ScenePart, layout: FrameLayout,
                         world_x: float) -> list[tuple[QImage, QRectF]]:
        row = layout.rows.get(part.part_id)
        if row is None or row.opacity <= 1e-6:
            return []
        scene = self.scene
        left = world_x - (scene.play_x - scene.body_left) / layout.scale
        ratio = layout.scale / scene.scale
        overlays = []
        for span in scene.octave_spans:
            # Geometry, rather than the currently playing beat, also works during seeking.
            if span.part_id != part.part_id or not span.label_right < left < span.end_x:
                continue
            image = self._octave_image(span.label)
            height = image.height() * ratio
            rect = QRectF(scene.body_left + 24 * layout.scale,
                          row.top + (span.label_y - part.source_top) * layout.scale - height / 2,
                          image.width() * ratio, height)
            overlays.append((image, rect))
        return overlays

    def _octave_mask(self, rect: QRectF, layout: FrameLayout) -> QRectF:
        # Edwin's line leading differs from the engraved SMuFL number. Cover
        # that small vertical gap as well as the dash's antialiased stroke.
        padding = max(2.0, 64 * layout.scale)
        mask = rect.adjusted(-padding, -padding, padding, padding)
        mask.setLeft(self.scene.body_left)
        return mask

    def render_frame(self, time: float) -> QImage:
        """Render presentation time; engraving and activity retain audio time."""
        if self._closed:
            raise RuntimeError("逐帧绘图器已经关闭。")
        started = clock.perf_counter()
        try:
            image = self._render_cpu(time)
            self.frame_count += 1
            return image
        finally:
            self.render_seconds += clock.perf_counter() - started

    def _render_cpu(self, time: float) -> QImage:
        s = self.scene.settings
        image = QImage(s.width, s.height, QImage.Format.Format_RGBA8888)
        image.fill(QColor("black"))
        painter = QPainter(image)
        try:
            self.paint_frame(painter, time)
        finally:
            painter.end()
        return image

    def paint_frame(self, painter: QPainter, time: float) -> None:
        """Draw the absolute-time scene onto a CPU image."""
        state = self._evaluator.evaluate(time)
        world_x, layout = state.world_x, state.layout
        plan = self._tile_plan(state)
        self.visible_tile_working_peak_bytes = max(self.visible_tile_working_peak_bytes,
                                                 plan.working_bytes)
        with self._pinned_tiles(plan.resident_keys):
            self._paint_score(painter, state, plan)
        self._draw_metadata(painter, time)
        self._draw_tempo(painter, time, layout, world_x)
        overlay = self.logo_overlay(time)
        if overlay is not None:
            image, rect, opacity = overlay
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.setOpacity(opacity)
            painter.drawImage(rect, image)
            painter.restore()

    def logo_overlay(self, time: float) -> tuple[QImage, QRectF, float] | None:
        """Share one raster asset and screen-space geometry with the GPU compositor."""
        s = self.scene.settings
        opacity = logo_opacity(time, s)
        if opacity <= 0:
            return None
        if self._logo is None:
            source = logo_image(dark=True)  # Both compositors use a black canvas.
            if source.isNull():
                return None
            short_side = min(s.width, s.height)
            scale = short_side * s.logo_size_ratio / max(source.width(), source.height())
            width, height = source.width() * scale, source.height() * scale
            margin = short_side * 0.025
            rect = QRectF(s.width - margin - width, s.height - margin - height, width, height)
            image = source.scaled(QSize(math.ceil(width), math.ceil(height)),
                                  Qt.AspectRatioMode.IgnoreAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation)
            self._logo = image, rect
        image, rect = self._logo
        return image, rect, opacity

    def _paint_score(self, painter: QPainter, state: FrameState, plan: TilePlan) -> None:
        scene = self.scene
        s = scene.settings
        time, layout, world_x = state.presentation_time, state.layout, state.world_x
        in_intro = time < s.intro_delay_seconds
        painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        scale = layout.scale
        ratio = scale / plan.raster_scale
        header_ratio = scale / scene.scale
        for part in scene.parts:
            row = layout.rows.get(part.part_id)
            if row is not None and row.opacity > 1e-6:
                row_y = row.top
                painter.save()
                painter.setOpacity(row.opacity)
                region_top, region_bottom = layout.region_bounds
                painter.setClipRect(QRectF(scene.body_left, region_top - 2,
                                          scene.body_right - scene.body_left,
                                          region_bottom - region_top + 4))
                # Staff lines continue through silent lead-in and the reverb tail.
                terminal_x = (scene.play_x + (scene.terminal_x - world_x) * scale
                              if scene.terminal_x is not None else scene.body_right)
                split = min(scene.body_right, max(scene.body_left, terminal_x))
                for center in part.staff_centers:
                    for step in range(-2, 3):
                        line_y = row_y + (center - part.source_top + step * 180) * scale
                        if split > scene.body_left:
                            painter.setPen(QPen(QColor("white"), max(0.55, 13 * scale)))
                            painter.drawLine(QPointF(scene.body_left, line_y), QPointF(split, line_y))
                        if split < scene.body_right:
                            painter.setPen(QPen(QColor("#666666"), max(0.55, 13 * scale)))
                            painter.drawLine(QPointF(split, line_y), QPointF(scene.body_right, line_y))
                for tile_index in range(plan.first_index, plan.last_index + 1):
                    tile = self._tile(part, tile_index, plan.level)
                    destination = scene.play_x + (tile_index * self.TILE_PIXELS / plan.raster_scale - world_x) * scale
                    painter.save()
                    painter.setClipRect(QRectF(destination, row_y, self.TILE_PIXELS * ratio, tile.height() * ratio), Qt.ClipOperation.IntersectClip)
                    painter.drawImage(QRectF(destination - self.TILE_BLEED * ratio, row_y, tile.width() * ratio, tile.height() * ratio), tile)
                    painter.restore()
                for image, rect in self._octave_overlays(part, layout, world_x):
                    # Erase only the small piece of static dash underneath the reminder.
                    painter.setOpacity(1.0)
                    painter.fillRect(self._octave_mask(rect, layout), QColor("black"))
                    painter.setOpacity(row.opacity)
                    painter.drawImage(rect, image)
                painter.restore()
                painter.save()
                painter.setOpacity(row.opacity)
                header = self._header(part)
                painter.drawImage(QRectF(scene.body_left - header.width() * header_ratio - 15 * s.width / 1920, row_y, header.width() * header_ratio, header.height() * header_ratio), header)
                rectangle = QRectF(*row.indicator_rect)
                color = None if in_intro else activity_lamp_color(part, state.activity[part.part_id])
                if color is not None:
                    painter.fillRect(rectangle, color)
                else:
                    stroke = max(1, round(s.height / 1080))
                    if rectangle.adjusted(stroke, stroke, -stroke, -stroke).isEmpty():
                        # A one- or two-pixel lamp has no room for a hollow outline.
                        painter.fillRect(rectangle, QColor("#404040"))
                    else:
                        painter.setPen(QPen(QColor("#dedede"), stroke))
                        painter.setBrush(Qt.BrushStyle.NoBrush)
                        painter.drawRect(rectangle.adjusted(stroke / 2, stroke / 2, -stroke / 2, -stroke / 2))
                if part.icon and part.icon != "unknown":
                    center = QPointF(rectangle.left() - scene.layout.icon_source_gap * scale - row.icon_size / 2,
                                     rectangle.center().y())
                    if self.render_backend == "gpu":
                        self._draw_gpu_icon(painter, part.icon, center, row.icon_size)
                    else:
                        self._draw_icon(painter, part.icon, center, row.icon_size)
                painter.restore()

    def render(self, time: float) -> QImage:
        return self.render_frame(time)

    def _draw_gpu_icon(self, painter: QPainter, kind: str, center: QPointF, size: float) -> None:
        # Rasterize fixed pictograms once, like score glyphs, so random-access
        # Vulkan frames use identical texture pixels.
        max_size = self.scene.layout.icon_source_size * self.scene.scale
        if kind not in self._gpu_icons:
            extent = max(1, math.ceil(max_size * 1.25)) + 4
            image = QImage(extent, extent, QImage.Format.Format_RGBA8888)
            image.fill(Qt.GlobalColor.transparent)
            icon_painter = QPainter(image)
            try:
                icon_painter.setRenderHints(QPainter.RenderHint.Antialiasing)
                self._draw_icon(icon_painter, kind, QPointF(extent / 2, extent / 2), max_size)
            finally:
                icon_painter.end()
            self._gpu_icons[kind] = image
        image = self._gpu_icons[kind]
        ratio = size / max_size
        painter.drawImage(QRectF(center.x() - image.width() * ratio / 2,
                                center.y() - image.height() * ratio / 2,
                                image.width() * ratio, image.height() * ratio), image)

    def _draw_metadata(self, painter: QPainter, time: float) -> None:
        s = self.scene.settings
        opacity = overlay_opacity(time, s, hide=s.announcement_auto_hide,
                                  hold=s.announcement_hold_seconds)
        if opacity <= 0:
            return
        painter.save()
        painter.setOpacity(opacity)
        m = self.scene.metadata
        unit = s.height / 1080
        x, y = s.title_x * s.width, s.title_y * s.height
        painter.setPen(QColor("white"))
        draw_text(painter, m.title, self.font_sizes["title"], QPointF(x, y))
        if m.subtitle:
            draw_text(painter, m.subtitle, self.font_sizes["subtitle"], QPointF(x, y + 39 * unit))
        if m.composer or m.arranger:
            line_y = y + 61 * unit
            painter.setPen(QPen(QColor("#8e8e8e"), 0.8 * unit))
            painter.drawLine(QPointF(x, line_y), QPointF(min(x + 620 * s.width / 1920, s.width * 0.94), line_y))
            painter.setPen(QColor("white"))
            row_y = y + 113 * unit
            for label, value in (("作曲", m.composer), ("编曲", m.arranger)):
                if value:
                    draw_text(painter, label, self.font_sizes["credits"], QPointF(x, row_y))
                    draw_text(painter, value, self.font_sizes["credits"], QPointF(x + 124 * s.width / 1920, row_y))
                    row_y += 35 * unit
        painter.restore()

    def _draw_tempo(self, painter: QPainter, time: float, layout: FrameLayout, world_x: float) -> None:
        mark = self.scene.tempo_mark
        if mark is None or time >= self.scene.layout.tempo_exit_time:
            return
        row = layout.rows[mark.owner_id]
        x = self.scene.play_x + (mark.x - world_x) * layout.scale
        y = row.top - mark.padding * layout.scale
        # Lay out once in staff-space units, then transform the complete group.
        # A fixed font size avoids integer-pixel jumps during animated zoom.
        painter.save()
        top, bottom = layout.region_bounds
        painter.setClipRect(QRectF(self.scene.body_left, top,
                                  self.scene.body_right - self.scene.body_left,
                                  bottom - top))
        painter.setOpacity(1.0)
        if self.render_backend == "gpu":
            if self._gpu_tempo is None:
                image = QImage(math.ceil(mark.width * self.scene.scale) + 4,
                               math.ceil(mark.height * self.scene.scale) + 4,
                               QImage.Format.Format_RGBA8888)
                image.fill(Qt.GlobalColor.transparent)
                cached_painter = QPainter(image)
                try:
                    cached_painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
                    cached_painter.translate(2, 2)
                    cached_painter.scale(self.scene.scale, self.scene.scale)
                    self._paint_tempo_group(cached_painter)
                finally:
                    cached_painter.end()
                self._gpu_tempo = image
            image = self._gpu_tempo
            ratio = layout.scale / self.scene.scale
            painter.drawImage(QRectF(x - 2 * ratio, y - 2 * ratio,
                                     image.width() * ratio, image.height() * ratio), image)
            painter.restore()
            return
        painter.translate(x, y)
        painter.scale(layout.scale, layout.scale)
        self._paint_tempo_group(painter)
        painter.restore()

    def _paint_tempo_group(self, painter: QPainter) -> None:
        mark = self.scene.tempo_mark
        self.tempo_renderer.render(painter, QRectF(0, 0, mark.glyph_width, mark.height))
        painter.setPen(QColor("white"))
        label = QRectF(mark.glyph_width + mark.label_gap, 0,
                       mark.width - mark.glyph_width - mark.label_gap, mark.height)
        draw_text_rect(painter, mark.label, mark.font_size, label,
                       alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    def _draw_icon(self, painter: QPainter, kind: str, center: QPointF, size: float) -> None:
        """Use bundled SVGs where matched, with original pictograms as fallback."""
        if kind in ("none", "unknown", ""):
            return
        renderer = resolve_icon(kind, self.scene.icon_assets)
        if renderer is not None:
            draw_image_icon(painter, renderer, QRectF(center.x() - size / 2, center.y() - size / 2,
                                                    size, size))
            return
        painter.save()
        painter.translate(center)
        painter.scale(size / 32, size / 32)
        painter.setPen(QPen(QColor("white"), 1.8))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if kind in ("bell", "celesta", "glockenspiel"):
            path = QPainterPath()
            path.moveTo(-12, 9)
            path.cubicTo(-7, 5, -11, -9, -2, -12)
            path.cubicTo(7, -14, 8, 4, 13, 7)
            path.closeSubpath()
            painter.drawPath(path)
            painter.drawEllipse(QRectF(-2, 9, 5, 4))
        elif kind in ("violin", "viola", "cello", "bass", "double_bass", "guitar"):
            painter.drawEllipse(QRectF(-9, -2, 16, 15))
            painter.drawEllipse(QRectF(-6, -10, 11, 12))
            painter.drawLine(QPointF(0, -19), QPointF(-1, 9))
            painter.drawLine(QPointF(13, -16), QPointF(8, 15))
            painter.drawLine(QPointF(-5, 5), QPointF(4, 5))
        elif kind == "piano":
            painter.drawRoundedRect(QRectF(-16, -10, 32, 21), 2, 2)
            for key in range(-12, 16, 6):
                painter.drawLine(QPointF(key, -9), QPointF(key, 10))
            for key in (-10, -4, 8):
                painter.fillRect(QRectF(key, -9, 3, 12), QColor("white"))
        elif kind in ("trumpet", "trombone", "horn", "tuba"):
            painter.drawEllipse(QRectF(-12, -4, 20, 14))
            painter.drawLine(QPointF(-15, -7), QPointF(4, -7))
            path = QPainterPath()
            path.moveTo(4, -7)
            path.lineTo(15, -13)
            path.lineTo(15, 2)
            path.lineTo(4, -4)
            painter.drawPath(path)
        elif kind in ("percussion", "drums"):
            painter.drawEllipse(QRectF(-13, -5, 26, 10))
            painter.drawLine(QPointF(-13, 0), QPointF(-11, 12))
            painter.drawLine(QPointF(13, 0), QPointF(11, 12))
            painter.drawArc(QRectF(-11, 7, 22, 10), 180 * 16, 180 * 16)
            painter.drawLine(QPointF(-12, -14), QPointF(6, -3))
            painter.drawLine(QPointF(13, -13), QPointF(-5, -2))
        elif kind == "harp":
            path = QPainterPath()
            path.moveTo(-12, -15)
            path.cubicTo(2, -18, 11, -8, 13, 14)
            path.lineTo(-10, 14)
            path.closeSubpath()
            painter.drawPath(path)
            for x in (-5, 0, 5):
                painter.drawLine(QPointF(x, -8), QPointF(x, 13))
        else:
            # Woodwinds share a recognizable tube; identity remains in the project editor.
            painter.drawLine(QPointF(-11, 13), QPointF(10, -14))
            painter.drawLine(QPointF(-7, 16), QPointF(14, -11))
            for x, y in ((-5, 8), (0, 1), (5, -6)):
                painter.drawEllipse(QRectF(x - 1, y - 1, 2, 2))
        painter.restore()


class FrameRenderer(RasterFrameRenderer):
    """Use Vulkan for auto/GPU settings and Raster for explicit CPU or fallback."""

    def __init__(self, scene: CompiledScene):
        super().__init__(scene)
        self._gpu = None
        self._last_gpu_report = {}
        if self.requested_backend != "cpu":
            from .rhi import RhiFrameRenderer

            try:
                self._gpu = RhiFrameRenderer(scene, assets=self)
            except GpuBackendError as exc:
                self.render_backend = "cpu"
                if self.requested_backend == "gpu":
                    self.close()
                    raise GpuBackendError(f"已选择 Vulkan GPU 渲染，但 Vulkan 不可用：{exc}") from exc
                self.fallback_reasons.append(str(exc))
            except BaseException:
                super().close()
                raise
            else:
                self.render_backend = "gpu"
                native = self._gpu.backend_report()
                self.gpu_info = dict(native.get("gpu_info", {}))
                self.gpu_texture_cache_budget_bytes = native.get(
                    "gpu_texture_cache_budget_bytes", self.cache_limit)

    def backend_report(self) -> dict:
        native = self._last_gpu_report if self._gpu is None else self._gpu.backend_report()
        base = super().backend_report()
        report = {**base, **native}
        # The shared core continues evaluating CPU frames after GPU recovery.
        report.update(self._evaluator.report())
        for key in ("requested_render_backend", "render_backend", "render_fallback_reasons",
                    "render_seconds", "rendered_frame_count", "readback_mode",
                    "readback_mode_history", "readback_fallback_reasons"):
            report[key] = base[key]
        report["memory_copy_seconds"] = (native.get("memory_copy_seconds", 0.0)
                                         + base.get("cpu_format_conversion_seconds", 0.0))
        report["memory_copy_bytes"] = (native.get("memory_copy_bytes", 0)
                                       + base.get("cpu_format_conversion_bytes", 0))
        return {**report, "graphics_api": "vulkan" if self.render_backend == "gpu" else "raster",
                "readback_seconds": native.get("readback_seconds", 0.0),
                "requested_readback_mode": (native.get("requested_readback_mode", "rhi-sync")
                                            if self.requested_backend != "cpu" else "cpu")}

    def _gpu_failure(self, error: GpuBackendError) -> None:
        gpu, self._gpu = self._gpu, None
        if gpu is not None:
            self._last_gpu_report = gpu.backend_report()
            try:
                gpu.close()
            except GpuBackendError as cleanup_error:
                error.add_note(f"Vulkan 清理失败：{cleanup_error}")
        if self.requested_backend == "gpu":
            self.close()
            raise error
        self.render_backend = "cpu"
        self.fallback_reasons.append(str(error))

    def _render(self, seconds: float, *, native: bool, count: bool = True) -> QImage:
        if self._closed:
            raise RuntimeError("逐帧绘图器已经关闭。")
        started = clock.perf_counter()
        try:
            if self._gpu is not None:
                try:
                    image = (self._gpu._render(seconds) if native
                             else self._gpu.render_frame(seconds))
                except GpuBackendError as error:
                    self._gpu_failure(error)
                    image = self._render_cpu(seconds)
            else:
                image = self._render_cpu(seconds)
            if count:
                self.frame_count += 1
            return image
        finally:
            self.render_seconds += clock.perf_counter() - started

    def render_frame(self, time: float) -> QImage:
        return self._render(time, native=False)

    def _render_export(self, time: float) -> QImage:
        return self._render(time, native=True, count=False)

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._export_stream is not None:
                self._export_stream.close()
        finally:
            gpu = self._gpu
            self._gpu = None
            try:
                if gpu is not None:
                    try:
                        self._last_gpu_report = gpu.backend_report()
                    finally:
                        gpu.close()
            finally:
                super().close()


class ExportFrameStream(Iterator[tuple[int, QImage]]):
    """Deliver independently owned frames in order to the bounded writer queue."""

    def __init__(self, renderer: RasterFrameRenderer, times: Iterable[float], cancel):
        self.renderer, self.cancel = renderer, cancel
        self._times = _ExportTimes(times, cancel)
        self._ready = deque()
        self._cpu_times = deque()
        self._index = 0
        self._closed = False
        self._gpu_target = getattr(renderer, "_gpu", None)
        self.pixel_format = "bgra" if self._gpu_target is not None else "rgba"
        self._baseline = self._target_report()
        self._final_target_report = None
        self._modes = []
        self._ready_mode = "unused"
        self._fallback_start = len(renderer.fallback_reasons)
        self._final_fallback_reasons = None
        self._cpu_copy_seconds = 0.0
        self._cpu_copy_bytes = 0
        self._pending_peak = 0

    def _fill_batch(self, gpu):
        started = clock.perf_counter()
        try:
            times = self._times.take_batch(gpu)
            try:
                images = gpu.render_batch_times(times, self.cancel)
            except GpuBackendError as error:
                # Nothing in this batch has been delivered. Keep the original
                # times so auto recovery never consumes the iterator twice.
                self.renderer._gpu_failure(error)
                self._cpu_times.extend(times)
            else:
                self._ready.extend(images)
                self._ready_mode = "rhi-batch-sync" if len(images) > 1 else "rhi-sync"
                self._pending_peak = max(self._pending_peak,
                                         sum(image.sizeInBytes() for image in self._ready))
        finally:
            self.renderer.render_seconds += clock.perf_counter() - started

    def _target_report(self):
        if getattr(self, "_final_target_report", None) is not None:
            return self._final_target_report
        native = self._gpu_target.backend_report() if self._gpu_target is not None else {}
        return {**native, **self.renderer._evaluator.report()}

    def __next__(self):
        if self._closed:
            raise StopIteration
        try:
            self._times.check_cancel()
            gpu = getattr(self.renderer, "_gpu", None)
            if not self._ready and not self._cpu_times and hasattr(gpu, "render_batch_times"):
                self._fill_batch(gpu)
            self._times.check_cancel()
            if self._ready:
                image = self._ready.popleft()
                mode = self._ready_mode
            else:
                seconds = (self._cpu_times.popleft() if self._cpu_times
                           else self._times.next_time())
                if isinstance(self.renderer, FrameRenderer):
                    image = self.renderer._render_export(seconds)
                else:
                    image = self.renderer.render_frame(seconds)
                mode = "rhi-sync" if getattr(self.renderer, "_gpu", None) is not None else "cpu"
            target = (QImage.Format.Format_ARGB32 if self.pixel_format == "bgra"
                      else QImage.Format.Format_RGBA8888)
            if image.format() != target:
                started = clock.perf_counter()
                image = image.convertToFormat(target)
                self._cpu_copy_seconds += clock.perf_counter() - started
                self._cpu_copy_bytes += image.sizeInBytes()
            self._times.check_cancel()
            if not self._modes or self._modes[-1] != mode:
                self._modes.append(mode)
            index = self._index
            self._index += 1
            if isinstance(self.renderer, FrameRenderer):
                self.renderer.frame_count += 1
            return index, image
        except InterruptedError:
            # The export owner stops FFmpeg and joins the writer before closing.
            raise
        except BaseException:
            self.close()
            raise

    def report(self):
        current = self._target_report()
        values = _export_native_delta(current, self._baseline)
        values["memory_copy_seconds"] += self._cpu_copy_seconds
        values["memory_copy_bytes"] += self._cpu_copy_bytes
        peaks = {key: current.get(key, 0) for key in
                 ("gpu_batch_peak_size", "readback_output_peak_bytes", "readback_staging_peak_bytes")}
        fallback_reasons = (self.renderer.fallback_reasons[self._fallback_start:]
                            if self._final_fallback_reasons is None else self._final_fallback_reasons)
        compute = {key: value for key, value in current.items() if key.startswith("scene_compute_")}
        return {**values, **compute, "readback_mode": self._modes[-1] if self._modes else "unused",
                "cpu_format_conversion_seconds": self._cpu_copy_seconds,
                "cpu_format_conversion_bytes": self._cpu_copy_bytes,
                "readback_mode_history": list(self._modes),
                "readback_fallback_reasons": list(fallback_reasons),
                "readback_buffer_peak_bytes": current.get("readback_buffer_peak_bytes", 0),
                **peaks,
                "frame_stream_pending_peak_bytes": self._pending_peak,
                "yielded_frame_count": self._index,
                "batch_budget_split_count": self._times.budget_splits,
                "batch_tile_split_count": self._times.tile_splits}

    def close(self):
        if self._closed:
            return
        self._final_target_report = self._target_report()
        self._final_fallback_reasons = self.renderer.fallback_reasons[self._fallback_start:]
        self._closed = True
        self._ready.clear()
        self._cpu_times.clear()
        self._times.close()
        report = self.report()
        totals = self.renderer._export_report
        for key, value in report.items():
            if key in ("readback_mode_history", "readback_fallback_reasons"):
                totals[key].extend(value)
            elif key == "readback_mode":
                totals[key] = value
            elif key.endswith("_peak_bytes") or key == "gpu_batch_peak_size":
                totals[key] = max(totals.get(key, 0), value)
            elif key == "gpu_batch_size_histogram":
                histogram = totals.setdefault(key, {})
                for size, count in value.items():
                    histogram[size] = histogram.get(size, 0) + count
            elif key.startswith("scene_compute_"):
                totals[key] = value
            else:
                totals[key] = totals.get(key, 0) + value
        if self.renderer._export_stream is self:
            self.renderer._export_stream = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def render_frame(scene: CompiledScene, time: float) -> QImage:
    """Convenience entrypoint. Reuse FrameRenderer for preview and video loops."""
    with FrameRenderer(scene) as renderer:
        return renderer.render_frame(time)
