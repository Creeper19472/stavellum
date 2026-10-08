"""QRhi Vulkan compositor shared by preview, frame output and video export."""

from __future__ import annotations

import math
import os
import threading
import time as clock
from collections import deque
from dataclasses import replace

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter

from ._rhi import Quad, RhiTarget
from .gpu import GpuBackendError
from .qt import prepare_render_app
from .render import (
    RasterFrameRenderer,
    _export_native_delta,
    _ExportTimes,
    activity_lamp_color,
    overlay_opacity,
)


def clipped_quad(texture_id, destination, clip, color=(1.0, 1.0, 1.0, 1.0)):
    """Clip floating geometry while preserving the complete texture mapping."""
    x, y, width, height = destination
    left, top, clip_width, clip_height = clip
    if width <= 0 or height <= 0:
        return None
    x0, y0 = max(x, left), max(y, top)
    x1, y1 = min(x + width, left + clip_width), min(y + height, top + clip_height)
    if x1 <= x0 or y1 <= y0:
        return None
    return Quad(texture_id, x0, y0, x1 - x0, y1 - y0,
                (x0 - x) / width, (y0 - y) / height,
                (x1 - x) / width, (y1 - y) / height, *color)


class RhiFrameRenderer:
    """Build small cached assets on CPU; compose every output frame on GPU."""

    def __init__(self, scene, api: str = "vulkan", *, assets=None):
        if api != "vulkan":
            raise ValueError("RHI API must be vulkan")
        started = clock.perf_counter()
        prepare_render_app(replace(scene.settings, render_backend="gpu"))
        self.scene = scene
        self.api = api
        self.render_backend = "gpu"
        self.fallback_reasons = []
        self._thread = threading.get_ident()
        # Batching is opt-in until it improves the complete encoding pipeline.
        requested_batch = os.environ.get("STAVELLUM_RHI_BATCH_SIZE", "1")
        if requested_batch not in ("1", "2", "4"):
            raise ValueError("STAVELLUM_RHI_BATCH_SIZE must be 1, 2 or 4")
        self._requested_batch_size = int(requested_batch)
        # The production wrapper supplies its own cached SVG assets.
        self._owns_assets = assets is None
        self._assets = (RasterFrameRenderer(replace(scene, settings=replace(scene.settings, render_backend="cpu")))
                        if assets is None else assets)
        self._assets.render_backend = "gpu"  # select existing cached icon/tempo helpers
        self._closed = False
        self._stream = None
        self.frame_count = 0
        self.render_seconds = 0.0
        self.asset_prepare_seconds = 0.0
        self.command_build_seconds = 0.0
        self._metadata = None
        self._metadata_position = (0, 0)
        try:
            self._target = RhiTarget(scene.settings.width, scene.settings.height,
                                     scene.settings.cache_megabytes, api)
        except BaseException:
            if self._owns_assets:
                self._assets.close()
            raise
        self.initialization_seconds = clock.perf_counter() - started
        self._last_report = {}

    @property
    def export_batch_limit(self):
        # Keep only a bounded number of independently owned CPU outputs. A
        # single frame exceeding the cap still has to be rendered on its own.
        pixels = self.scene.settings.width * self.scene.settings.height * 4
        capacity = max(1, (64 * 1024 * 1024) // pixels)
        if not hasattr(self._target, "render_batch") or getattr(self._target, "_copy_readback", False):
            return 1
        return min(self._requested_batch_size, capacity)

    @property
    def export_readback_mode(self):
        return "rhi-batch-sync" if self.export_batch_limit > 1 else "rhi-sync"

    def batch_plan(self, seconds):
        """Admit only frames whose entire visible tile set can stay resident."""
        self._check()
        state = self._assets._evaluator.evaluate(seconds)
        plan = self._assets._tile_plan(state)
        if plan.working_bytes > self._assets.cache_limit:
            return None
        return plan.level, plan.resident_keys

    def _check(self):
        if self._closed:
            raise RuntimeError("RHI renderer is closed")
        if threading.get_ident() != self._thread:
            raise RuntimeError("RHI renderer must be used on its creating thread")

    @property
    def backend(self):
        return self.render_backend

    @property
    def cache_peak_bytes(self):
        return self._assets.cache_peak_bytes

    @property
    def cache_limit(self):
        return self._assets.cache_limit

    def _asset(self, function, *args):
        started = clock.perf_counter()
        try:
            return function(*args)
        finally:
            self.asset_prepare_seconds += clock.perf_counter() - started

    def _metadata_image(self):
        if self._metadata is not None:
            return self._metadata
        # Rasterize only the static metadata layer once, then retain its crop.
        # QPicture does not include QTextLayout glyph bounds reliably.
        from PySide6.QtGui import QBitmap, QRegion
        s = self.scene.settings
        image = QImage(s.width, s.height, QImage.Format.Format_RGBA8888)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        try:
            painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
            settings = self._assets.scene.settings
            self._assets.scene = replace(self._assets.scene, settings=replace(
                settings, announcement_auto_hide=False, overlay_enter_seconds=1e-9))
            try:
                self._assets._draw_metadata(painter, 1.0)
            finally:
                self._assets.scene = replace(self._assets.scene, settings=settings)
        finally:
            painter.end()
        bounds = QRegion(QBitmap.fromImage(image.createAlphaMask())).boundingRect()
        bounds = bounds.adjusted(-2, -2, 2, 2).intersected(image.rect())
        image = image.copy(bounds) if not bounds.isEmpty() else QImage(1, 1, QImage.Format.Format_RGBA8888)
        if bounds.isEmpty():
            image.fill(Qt.GlobalColor.transparent)
        self._metadata_position = (bounds.x(), bounds.y())
        self._metadata = image
        return image

    def _icon_image(self, kind):
        assets = self._assets
        if kind not in assets._gpu_icons:
            # Helper uses QPainter only for this fixed-size pictogram, once.
            size = self.scene.layout.icon_source_size * self.scene.scale
            probe = QImage(1, 1, QImage.Format.Format_RGBA8888)
            painter = QPainter(probe)
            try:
                assets._draw_gpu_icon(painter, kind, QPointF(), size)
            finally:
                painter.end()
        return assets._gpu_icons[kind]

    def _tempo_image(self):
        assets, mark = self._assets, self.scene.tempo_mark
        if assets._gpu_tempo is None:
            image = QImage(math.ceil(mark.width * self.scene.scale) + 4,
                           math.ceil(mark.height * self.scene.scale) + 4, QImage.Format.Format_RGBA8888)
            image.fill(Qt.GlobalColor.transparent)
            painter = QPainter(image)
            try:
                painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
                painter.translate(2, 2)
                painter.scale(self.scene.scale, self.scene.scale)
                assets._paint_tempo_group(painter)
            finally:
                painter.end()
            assets._gpu_tempo = image
        return assets._gpu_tempo

    def commands(self, time):
        self._check()
        scene, s, assets = self.scene, self.scene.settings, self._assets
        started = clock.perf_counter()
        previous_assets = self.asset_prepare_seconds
        commands = []
        canvas = (0, 0, s.width, s.height)

        def solid(rect, value, opacity, clip=canvas):
            color = QColor(value)
            alpha = opacity * color.alphaF()
            quad = clipped_quad(0, rect, clip, (color.redF() * alpha, color.greenF() * alpha,
                                               color.blueF() * alpha, alpha))
            if quad is not None:
                commands.append(quad)

        def texture(image, rect, opacity, clip=canvas):
            quad = clipped_quad(1, rect, clip, (opacity,) * 4)
            if quad is not None:
                quad.texture_id = self._target.texture(image)
                commands.append(quad)

        state = assets._evaluator.evaluate(time)
        world_x, layout = state.world_x, state.layout
        plan = assets._tile_plan(state)
        assets.visible_tile_working_peak_bytes = max(assets.visible_tile_working_peak_bytes,
                                                    plan.working_bytes)
        scale = layout.scale
        ratio = scale / plan.raster_scale
        header_ratio = scale / scene.scale
        top, bottom = layout.region_bounds
        body_clip = (scene.body_left, top - 2, scene.body_right - scene.body_left, bottom - top + 4)
        with assets._pinned_tiles(plan.resident_keys):
            for part in scene.parts:
                row = layout.rows.get(part.part_id)
                if row is None or row.opacity <= 1e-6:
                    continue
                y, opacity = row.top, row.opacity
                terminal = (scene.play_x + (scene.terminal_x - world_x) * scale
                            if scene.terminal_x is not None else scene.body_right)
                split = min(scene.body_right, max(scene.body_left, terminal))
                stroke = max(0.55, 13 * scale)
                for center in part.staff_centers:
                    for step in range(-2, 3):
                        line_y = y + (center - part.source_top + step * 180) * scale
                        for x0, x1, color in ((scene.body_left, split, "white"), (split, scene.body_right, "#666666")):
                            if x1 > x0:
                                solid((x0 - stroke / 2, line_y - stroke / 2, x1 - x0 + stroke, stroke), color, opacity, body_clip)
                for index in range(plan.first_index, plan.last_index + 1):
                    image = self._asset(assets._tile, part, index, plan.level)
                    x = scene.play_x + (index * assets.TILE_PIXELS / plan.raster_scale - world_x) * scale
                    core = QRectF(x, y, assets.TILE_PIXELS * ratio, image.height() * ratio)
                    clip = core.intersected(QRectF(*body_clip)).intersected(QRectF(*canvas))
                    texture(image, (x - assets.TILE_BLEED * ratio, y, image.width() * ratio,
                                    image.height() * ratio), opacity, clip.getRect())
                for image, rect in self._asset(assets._octave_overlays, part, layout, world_x):
                    solid(assets._octave_mask(rect, layout).getRect(), "black", 1.0, body_clip)
                    texture(image, rect.getRect(), opacity, body_clip)
                image = self._asset(assets._header, part)
                texture(image, (scene.body_left - image.width() * header_ratio - 15 * s.width / 1920,
                                y, image.width() * header_ratio, image.height() * header_ratio), opacity)
                rect = QRectF(*row.indicator_rect)
                color = (None if time < s.intro_delay_seconds
                         else activity_lamp_color(part, state.activity[part.part_id]))
                if color is not None:
                    solid(rect.getRect(), color, opacity)
                else:
                    border = max(1, round(s.height / 1080))
                    if rect.adjusted(border, border, -border, -border).isEmpty():
                        solid(rect.getRect(), "#404040", opacity)
                    else:
                        x, lamp_y, w, h = rect.getRect()
                        for outline in ((x, lamp_y, w, border), (x, lamp_y + h - border, w, border),
                                        (x, lamp_y + border, border, h - 2 * border),
                                        (x + w - border, lamp_y + border, border, h - 2 * border)):
                            solid(outline, "#dedede", opacity)
                if part.icon and part.icon != "unknown":
                    image = self._asset(self._icon_image, part.icon)
                    maximum = scene.layout.icon_source_size * scene.scale
                    icon_ratio = row.icon_size / maximum
                    center_x = rect.left() - scene.layout.icon_source_gap * scale - row.icon_size / 2
                    texture(image, (center_x - image.width() * icon_ratio / 2,
                                    rect.center().y() - image.height() * icon_ratio / 2,
                                    image.width() * icon_ratio, image.height() * icon_ratio), opacity)
        opacity = overlay_opacity(time, s, hide=s.announcement_auto_hide, hold=s.announcement_hold_seconds)
        if opacity > 0:
            image = self._asset(self._metadata_image)
            texture(image, (*self._metadata_position, image.width(), image.height()), opacity)
        mark = scene.tempo_mark
        if mark is not None and time < scene.layout.tempo_exit_time:
            row = layout.rows[mark.owner_id]
            image = self._asset(self._tempo_image)
            x = scene.play_x + (mark.x - world_x) * scale
            y = row.top - mark.padding * scale
            texture(image, (x - 2 * header_ratio, y - 2 * header_ratio,
                            image.width() * header_ratio, image.height() * header_ratio),
                    1.0, (scene.body_left, top, scene.body_right - scene.body_left, bottom - top))
        overlay = self._asset(assets.logo_overlay, time)
        if overlay is not None:
            image, rect, opacity = overlay
            texture(image, rect.getRect(), opacity)
        self.command_build_seconds += clock.perf_counter() - started - (self.asset_prepare_seconds - previous_assets)
        return commands

    def _render(self, time):
        self._check()
        started = clock.perf_counter()
        try:
            image = self._target.render(self.commands(time))
            self.frame_count += 1
            return image
        finally:
            self.render_seconds += clock.perf_counter() - started

    def render_frame(self, time):
        return self._render(time).convertToFormat(QImage.Format.Format_RGBA8888)

    def render_batch_times(self, times, cancel):
        """Prepare on the owner thread, then wait once for a bounded batch."""
        self._check()
        started = clock.perf_counter()
        try:
            commands = []
            for seconds in times:
                if cancel():
                    raise InterruptedError("已取消导出。")
                commands.append(self.commands(seconds))
            if cancel():
                raise InterruptedError("已取消导出。")
            images = ([self._target.render(commands[0])] if len(commands) == 1
                      else self._target.render_batch(commands))
            if len(images) != len(times):
                raise GpuBackendError("Vulkan batch returned an unexpected frame count")
            self.frame_count += len(images)
            return images
        finally:
            self.render_seconds += clock.perf_counter() - started

    def export_frames(self, times, cancel):
        self._check()
        if self._stream is not None:
            raise RuntimeError("Only one export stream can be open per RHI renderer")
        self._stream = RhiFrameStream(self, times, cancel)
        return self._stream

    def backend_report(self):
        native = self._last_report if self._closed else self._target.report()
        return {**native, **self._assets._evaluator.report(), "requested_render_backend": "gpu", "render_backend": "gpu",
                "render_fallback_reasons": [], "render_seconds": self.render_seconds,
                "rendered_frame_count": self.frame_count, "initialization_seconds": self.initialization_seconds,
                "asset_prepare_seconds": self.asset_prepare_seconds, "command_build_seconds": self.command_build_seconds,
                "cache_peak_bytes": self._assets.cache_peak_bytes, "cache_limit_bytes": self._assets.cache_limit,
                "tile_cache_hits": self._assets.tile_cache_hits,
                "tile_cache_misses": self._assets.tile_cache_misses,
                "tile_cache_evictions": self._assets.tile_cache_evictions,
                "svg_raster_seconds": self._assets.svg_raster_seconds,
                "visible_tile_working_peak_bytes": self._assets.visible_tile_working_peak_bytes,
                "requested_readback_mode": self.export_readback_mode,
                "export_batch_limit": self.export_batch_limit,
                "batch_output_limit_bytes": 64 * 1024 * 1024,
                "readback_mode_history": [native.get("readback_mode", "rhi-sync")],
                "readback_fallback_reasons": []}

    def close(self):
        if self._closed:
            return
        self._check()
        try:
            if self._stream is not None:
                self._stream.close()
            self._last_report = self._target.report()
        finally:
            try:
                self._target.close()
            finally:
                if self._owns_assets:
                    self._assets.close()
                self._metadata = None
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class RhiFrameStream:
    pixel_format = "bgra"

    def __init__(self, renderer, times, cancel):
        self.renderer, self.cancel = renderer, cancel
        self._times = _ExportTimes(times, cancel)
        self._ready = deque()
        self._index = 0
        self._closed = False
        self._pending_peak = 0
        self._ready_mode = "unused"
        self._modes = []
        self._baseline = renderer.backend_report()
        self._final_native_report = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        self.renderer._check()
        try:
            self._times.check_cancel()
            if not self._ready:
                times = self._times.take_batch(self.renderer)
                images = self.renderer.render_batch_times(times, self.cancel)
                self._ready.extend(images)
                self._ready_mode = "rhi-batch-sync" if len(images) > 1 else "rhi-sync"
                self._pending_peak = max(self._pending_peak,
                                         sum(image.sizeInBytes() for image in self._ready))
            self._times.check_cancel()
            image = self._ready.popleft()
            if not self._modes or self._modes[-1] != self._ready_mode:
                self._modes.append(self._ready_mode)
            index = self._index
            self._index += 1
            return index, image
        except InterruptedError:
            # The export owner first stops FFmpeg and joins its writer.
            raise
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self._closed:
            self.renderer._check()
            self._final_native_report = self.renderer.backend_report()
            self._closed = True
            self._ready.clear()
            self._times.close()
            if self.renderer._stream is self:
                self.renderer._stream = None

    def report(self):
        current = self.renderer.backend_report()
        final = current if self._final_native_report is None else self._final_native_report
        return {**current, **_export_native_delta(final, self._baseline),
                "yielded_frame_count": self._index,
                "readback_mode": self._modes[-1] if self._modes else "unused",
                "readback_mode_history": list(self._modes),
                "frame_stream_pending_peak_bytes": self._pending_peak,
                "batch_budget_split_count": self._times.budget_splits,
                "batch_tile_split_count": self._times.tile_splits}
