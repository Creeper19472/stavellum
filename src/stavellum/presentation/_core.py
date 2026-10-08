"""Explicit ctypes ABI for the Rust per-frame scene evaluator."""

from __future__ import annotations

import ctypes
import math
import os
import weakref
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import CompiledScene


class CoreBackendError(RuntimeError):
    """A scene evaluator failure; never a graphics fallback signal."""


def _release(dll, handle):
    dll.spcore_close(handle)


class CurveKey(ctypes.Structure):
    _fields_ = [("time", ctypes.c_double), ("value", ctypes.c_double),
                ("velocity", ctypes.c_double), ("acceleration", ctypes.c_double)]


class Note(ctypes.Structure):
    _fields_ = [("start", ctypes.c_double), ("end", ctypes.c_double),
                ("velocity", ctypes.c_int32)]


class LayoutConsts(ctypes.Structure):
    _fields_ = [(name, ctypes.c_double) for name in (
        "indicator_right", "indicator_source_width", "indicator_source_height",
        "icon_source_size", "icon_source_gap", "region_top", "region_bottom",
        "expanded_top", "expanded_bottom", "expansion_start", "expansion_duration",
        "tempo_padding", "tempo_exit_time", "scene_scale", "play_x",
        "body_left", "body_right", "cache_limit")] + [("tempo_owner", ctypes.c_int32)]


class CoreRow(ctypes.Structure):
    _fields_ = [(name, ctypes.c_double) for name in (
        "top", "opacity", "indicator_x", "indicator_y", "indicator_w", "indicator_h",
        "bounds_top", "bounds_bottom", "icon_size", "activity_level", "activity_attack")]


class CoreFrame(ctypes.Structure):
    _fields_ = [(name, ctypes.c_double) for name in (
        "world_x", "scale", "region_top", "region_bottom", "bounds_top", "bounds_bottom",
        "camera_speed", "tile_raster_scale", "tile_working_bytes")] + [
        ("tile_level", ctypes.c_int32), ("tile_first", ctypes.c_int32),
        ("tile_last", ctypes.c_int32), ("part_count", ctypes.c_int32)]


def library_path() -> Path:
    override = os.environ.get("STAVELLUM_CORE_DLL")
    if override:
        return Path(override).resolve()
    return Path(files("stavellum")) / "native/core/stavellum_core.dll"


class CoreTarget:
    """Serialize a compiled scene once; evaluate frames with one FFI call."""

    def __init__(self, scene: CompiledScene, *, cache_limit: float | None = None):
        self._part_count = len(scene.parts)
        self._handle = None
        self.path = library_path()
        try:
            dll = ctypes.CDLL(str(self.path))
            self._bind(dll)
        except (OSError, AttributeError) as error:
            raise CoreBackendError(
                f"无法加载场景计算库 {self.path}；请运行 "
                f"uv run python scripts/build_rust.py --install：{error}"
            ) from error
        self._dll = dll
        if dll.spcore_abi_version() != 1:
            raise CoreBackendError(
                f"场景计算库 ABI 不匹配：{self.path}；请运行 "
                "uv run python scripts/build_rust.py --install"
            )
        self._compile(scene, cache_limit)
        self._finalizer = weakref.finalize(self, _release, dll, self._handle)

    @staticmethod
    def _bind(dll):
        pointer = ctypes.c_void_p
        for name, result, arguments in (
            ("spcore_abi_version", ctypes.c_uint32, ()),
            ("spcore_compile", pointer, (
                pointer, pointer, ctypes.c_int64,
                ctypes.c_double, ctypes.c_double, ctypes.c_double,
                ctypes.c_int64, pointer, ctypes.c_int64,
                pointer, pointer, ctypes.c_int64,
                pointer, pointer, ctypes.c_int64,
                pointer, pointer, pointer, pointer)),
            ("spcore_frame", ctypes.c_int, (
                pointer, ctypes.c_double, ctypes.c_double, pointer, pointer, ctypes.c_int32)),
            ("spcore_close", ctypes.c_int, (pointer,)),
            ("spcore_last_error", ctypes.c_char_p, ()),
        ):
            function = getattr(dll, name)
            function.restype, function.argtypes = result, arguments

    def _compile(self, scene, cache_limit):
        dll = self._dll
        layout = scene.layout
        camera = scene.camera
        if layout is None or camera is None:
            raise CoreBackendError("谱面尚未编译排版和相机时间轴。")
        parts = scene.parts
        # Track key order matches compile_layout's insertion order.
        tops_arrays = [self._keys(layout.tops[part.part_id]) for part in parts]
        opacity_arrays = [self._keys(layout.opacities[part.part_id]) for part in parts]
        zoom_array = self._keys(layout.zoom)
        note_arrays = []
        for part in parts:
            note_arrays.append((Note * len(part.notes))(*(
                Note(note.start, note.end, note.velocity) for note in part.notes)))
        dimensions = (ctypes.c_double * (2 * len(parts)))(*(
            value for part in parts
            for value in (part.source_height,
                          (part.staff_centers[0] + part.staff_centers[-1]) / 2 - part.source_top)))
        def counts(arrays):
            return (ctypes.c_int64 * len(arrays))(*(len(array) for array in arrays))
        beat_array = (ctypes.c_double * len(scene.axis.beats))(*scene.axis.beats)
        x_array = (ctypes.c_double * len(scene.axis.xs))(*scene.axis.xs)
        note_storage = self._concatenate(note_arrays, Note)
        owner = next((index for index, part in enumerate(parts)
                      if part.part_id == layout.tempo_owner_id), -1)
        self._consts = LayoutConsts(
            layout.indicator_right, layout.indicator_source_width,
            layout.indicator_source_height, layout.icon_source_size,
            layout.icon_source_gap, layout.region_top, layout.region_bottom,
            layout.expanded_top, layout.expanded_bottom,
            float("nan") if layout.expansion_start is None else layout.expansion_start,
            layout.expansion_duration, layout.tempo_padding, layout.tempo_exit_time,
            scene.scale, scene.play_x, scene.body_left, scene.body_right,
            scene.settings.cache_megabytes * 1024 * 1024 if cache_limit is None
            else cache_limit, owner)
        tops_flat = self._concatenate(tops_arrays, CurveKey)
        opacity_flat = self._concatenate(opacity_arrays, CurveKey)
        self._handle = dll.spcore_compile(
            beat_array, x_array, len(scene.axis.beats),
            camera.beats_per_second, camera.score_offset, camera.half_window_seconds,
            len(parts), zoom_array, len(zoom_array),
            tops_flat, counts(tops_arrays), len(tops_flat),
            opacity_flat, counts(opacity_arrays), len(opacity_flat),
            dimensions, note_storage, counts(note_arrays),
            ctypes.byref(self._consts))
        if not self._handle:
            self._error()

    @staticmethod
    def _keys(track):
        return (CurveKey * len(track.keys))(*(
            CurveKey(key.time, key.value, key.velocity, key.acceleration)
            for key in track.keys))

    @staticmethod
    def _concatenate(arrays, kind):
        total = sum(len(array) for array in arrays)
        flat = (kind * total)()
        stride = ctypes.sizeof(kind)
        position = 0
        for array in arrays:
            ctypes.memmove(ctypes.addressof(flat) + position * stride, array,
                           stride * len(array))
            position += len(array)
        return flat

    def _error(self):
        message = self._dll.spcore_last_error()
        detail = message.decode("utf-8", errors="replace") if message else "Native core operation failed"
        raise CoreBackendError(f"场景计算失败（{self.path}）：{detail}")

    def frame(self, presentation_time: float, audio_time: float):
        """Return (CoreFrame, CoreRow[part_count]) freshly read from native."""
        if not self._handle:
            raise CoreBackendError("场景计算器已经关闭。")
        if not math.isfinite(presentation_time) or not math.isfinite(audio_time):
            raise CoreBackendError("场景计算时间必须为有限数值。")
        rows = (CoreRow * self._part_count)()
        frame = CoreFrame()
        if self._dll.spcore_frame(self._handle, presentation_time, audio_time,
                                  ctypes.byref(frame), rows, len(rows)):
            self._error()
        return frame, rows

    def close(self):
        if getattr(self, "_handle", None):
            if self._dll.spcore_close(self._handle):
                self._error()
            self._handle = None
            self._finalizer.detach()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()
