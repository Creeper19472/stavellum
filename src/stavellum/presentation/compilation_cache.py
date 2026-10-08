"""Disposable, content-addressed engraving and timeline data shared by all processes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
import zlib
from dataclasses import asdict, fields, is_dataclass
from functools import cache
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from types import UnionType
from typing import get_args, get_origin, get_type_hints

from PySide6.QtCore import QStandardPaths

FORMAT_VERSION = 1
DISK_BUDGET_BYTES = 512 * 1024 * 1024
MAX_ENTRY_BYTES = 256 * 1024 * 1024
TIMELINE_SETTINGS = (
    "width", "height", "staff_scale", "score_left", "score_right", "score_top", "score_bottom",
    "header_width", "score_start_in_audio_sec", "enter_seconds", "exit_seconds", "reflow_seconds",
    "animation_stable_seconds", "intro_delay_seconds", "overlay_enter_seconds",
    "overlay_exit_seconds", "announcement_hold_seconds", "announcement_auto_hide",
)


def _json(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


@cache
def engine_fingerprint() -> str:
    """Version source resources as well as packages during editable development."""
    package = files("stavellum")
    resources = [f"{name}.py" for name in (
        "presentation/compilation_cache", "presentation/scene", "presentation/types",
        "presentation/curves", "presentation/timeline", "presentation/visibility",
        "presentation/axis", "presentation/camera", "presentation/layout",
        "engraving/notation", "engraving/inference", "engraving/dynamics", "engraving/ottava",
        "domain/models", "domain/mapping", "graphics/svg", "graphics/musicfont",
        "graphics/typography", "graphics/font_registry", "graphics/qt", "graphics/icons",
    )] + [f"fonts/{name}" for name in (
        "Edwin-Roman.otf", "Edwin-Italic.otf", "SourceHanSerifSC-Regular.otf",
    )]
    return _digest({
        "format": FORMAT_VERSION,
        "dependencies": {name: version(name) for name in ("music21", "verovio", "PySide6")},
        "resources": {name: hashlib.sha256(package.joinpath(name).read_bytes()).hexdigest()
                      for name in resources},
    })


def geometry_key(document) -> str:
    project = document.project
    mappings = []
    for mapping in document.mappings:
        values = asdict(mapping)
        for name in ("name", "icon", "use_icon"):
            values.pop(name)
        mappings.append(values)
    return _digest({
        "engine": engine_fingerprint(),
        "project": {name: getattr(project, name) for name in
                    ("ppq", "bpm", "numerator", "denominator", "duration_ticks")},
        "notes": [asdict(note) for note in project.notes],
        "automations": [asdict(item) for item in project.volume_automations],
        "routes": [asdict(item) for item in project.volume_routes],
        # Disabled mappings can affect _global_key's keyswitch exclusion.
        "mappings": mappings,
    })


def timeline_key(engraving_key, settings) -> str:
    return _digest({"geometry": engraving_key,
                    "settings": {name: getattr(settings, name) for name in TIMELINE_SETTINGS}})


def preview_key(document) -> str:
    """Capture the applied snapshot independently of the project's save/dirty state."""
    return _digest(document.to_dict())


def cache_directory() -> Path:
    override = os.environ.get("STAVELLUM_CACHE_DIR")
    if override:
        return Path(override).resolve()
    location = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.GenericCacheLocation)
    if not location:
        raise OSError("用户缓存目录不可用")
    return Path(location) / "Stavellum" / "compiled"


class CompilationCache:
    def __init__(self, directory: Path | None = None, *, budget_bytes=DISK_BUDGET_BYTES):
        try:
            self.directory = Path(directory) if directory is not None else cache_directory()
        except OSError:
            self.directory = None
        self.budget_bytes = budget_bytes

    def _path(self, layer, key):
        if layer not in {"geometry", "timeline"} or len(key) != 64 or any(
                char not in "0123456789abcdef" for char in key):
            raise ValueError("缓存地址无效")
        return self.directory / f"{layer}-{key}.json.z"

    def read(self, layer, key, decode):
        if self.directory is None:
            return None
        try:
            path = self._path(layer, key)
            if path.stat().st_size > MAX_ENTRY_BYTES:
                return None
            decompressor = zlib.decompressobj()
            raw = decompressor.decompress(path.read_bytes(), MAX_ENTRY_BYTES + 1)
            if len(raw) > MAX_ENTRY_BYTES or not decompressor.eof or decompressor.unused_data:
                return None
            envelope = json.loads(raw, parse_constant=lambda value: _invalid(value))
            if (set(envelope) != {"version", "key", "payload", "checksum"}
                    or envelope["version"] != FORMAT_VERSION or envelope["key"] != key
                    or envelope["checksum"] != _digest(envelope["payload"])):
                return None
            result = decode(envelope["payload"])
            try:
                os.utime(path, None)
            except OSError:
                pass
            return result
        except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError,
                RecursionError, zlib.error, ET.ParseError):
            return None

    def write(self, layer, key, payload, cancelled=lambda: False):
        if self.directory is None:
            return
        temporary = None
        try:
            raw = _json({"version": FORMAT_VERSION, "key": key, "payload": payload,
                         "checksum": _digest(payload)})
            if len(raw) > MAX_ENTRY_BYTES:
                return
            compressed = zlib.compress(raw)
            if len(compressed) > min(self.budget_bytes, MAX_ENTRY_BYTES):
                return
            self.directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix=".compile-",
                                             suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(compressed)
            if cancelled():
                raise InterruptedError("已取消谱面编译。")
            temporary.replace(self._path(layer, key))
            self._prune()
        except OSError as error:
            if isinstance(error, InterruptedError):
                raise
            # The compilation remains useful even when storage is unavailable.
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def _prune(self):
        entries = []
        for path in self.directory.glob("*-*.json.z"):
            try:
                stat = path.stat()
                entries.append((stat.st_mtime_ns, path, stat.st_size))
            except OSError:
                pass
        total = sum(size for _, _, size in entries)
        for _, path, size in sorted(entries):
            if total <= self.budget_bytes:
                break
            try:
                path.unlink(missing_ok=True)
                total -= size
            except OSError:
                pass

    def clear(self) -> bool:
        if self.directory is None:
            return False
        succeeded = True
        # Never recurse or remove arbitrary files from an override directory.
        for pattern in ("geometry-*.json.z", "timeline-*.json.z", ".compile-*.tmp"):
            for path in self.directory.glob(pattern):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    succeeded = False
        return succeeded


def _invalid(value):
    raise ValueError(f"无效缓存数据：{value}")


@cache
def _types(record_type):
    return get_type_hints(record_type)


def _decode(expected, value):
    """Decode only caller-selected dataclass contracts; never load classes from file data."""
    origin = get_origin(expected)
    arguments = get_args(expected)
    if origin is UnionType:
        for candidate in arguments:
            try:
                return _decode(candidate, value)
            except (ValueError, TypeError):
                pass
        return _invalid("union")
    if origin in (list, tuple):
        if not isinstance(value, list):
            return _invalid("array")
        if origin is list:
            return [_decode(arguments[0], item) for item in value]
        if len(arguments) != len(value):
            return _invalid("tuple size")
        return tuple(_decode(item_type, item) for item_type, item in zip(arguments, value, strict=True))
    if origin is dict:
        if not isinstance(value, dict):
            return _invalid("object")
        return {_decode(arguments[0], key): _decode(arguments[1], item) for key, item in value.items()}
    if is_dataclass(expected):
        names = {item.name for item in fields(expected) if item.init}
        if not isinstance(value, dict) or set(value) != names:
            return _invalid("record fields")
        return expected(**{name: _decode(_types(expected)[name], item) for name, item in value.items()})
    if expected is float:
        if type(value) not in (int, float) or not math.isfinite(value):
            return _invalid("number")
    elif expected in (str, int, bool, type(None)):
        if type(value) is not expected:
            return _invalid("primitive")
    else:
        return _invalid("unsupported type")
    return value


def decode_geometry(data, document):
    from .axis import TimeAxis
    from .types import EngravedGeometry

    geometry = _decode(EngravedGeometry, data)
    expected = [mapping.part_id for mapping in document.mappings if mapping.enabled]
    if [part.part_id for part in geometry.parts] != expected:
        return _invalid("part order")
    ids = set(expected)
    root = ET.fromstring(geometry.svg)
    if (root.tag.rsplit("}", 1)[-1] != "svg" or min(geometry.view_box[2:]) <= 0
            or geometry.header_right <= geometry.header_left or geometry.end_beat < 0
            or not geometry.measure_bounds or geometry.terminal_x < geometry.header_right):
        return _invalid("geometry")
    TimeAxis(geometry.beats, geometry.xs)
    if any(right <= left for left, right in geometry.measure_bounds):
        return _invalid("measure bounds")
    for part in geometry.parts:
        if (part.source_bottom <= part.source_top or not part.staff_centers
                or any(not part.source_top < center < part.source_bottom for center in part.staff_centers)):
            return _invalid("staff bounds")
    if (any(owner not in ids for owner in geometry.element_part_ids.values())
            or any(item.part_id not in ids for item in geometry.diagnostics)
            or any(item.part_id not in ids or item.staff_index < 0
                   or item.staff_index >= len(geometry.parts[expected.index(item.part_id)].staff_centers)
                   or item.font_size <= 0 for item in geometry.octave_spans)):
        return _invalid("part references")
    return geometry


def encode_timeline(scene):
    layout = asdict(scene.layout)
    if math.isinf(layout["tempo_exit_time"]):
        layout["tempo_exit_time"] = None
    camera = asdict(scene.camera)
    camera.pop("axis")
    return {"layout": layout, "camera": camera, "tempo_mark": asdict(scene.tempo_mark)}


def decode_timeline(data, scene):
    from .camera import CameraTimeline
    from .timeline import LayoutTimeline
    from .types import TempoMark

    if not isinstance(data, dict) or set(data) != {"layout", "camera", "tempo_mark"}:
        return _invalid("timeline fields")
    layout_data = dict(data["layout"])
    infinite_exit = layout_data["tempo_exit_time"] is None
    if infinite_exit:
        layout_data["tempo_exit_time"] = 0.0
    layout = _decode(LayoutTimeline, layout_data)
    if infinite_exit:
        layout.tempo_exit_time = math.inf
    camera_values = data["camera"]
    camera_fields = {item.name for item in fields(CameraTimeline)} - {"axis"}
    if not isinstance(camera_values, dict) or set(camera_values) != camera_fields:
        return _invalid("camera fields")
    camera = CameraTimeline(scene.axis, **{name: _decode(float, value)
                                          for name, value in camera_values.items()})
    tempo = _decode(TempoMark, data["tempo_mark"])
    ids = {part.part_id for part in scene.parts}
    if (any(set(mapping) != ids for mapping in (layout.tops, layout.opacities, layout.part_dimensions))
            or tempo.owner_id != scene.parts[0].part_id or layout.tempo_owner_id != tempo.owner_id
            or camera.beats_per_second != scene.bpm / 60
            or camera.score_offset != scene.settings.score_start_in_audio_sec
            or min(camera.half_window_seconds, camera.certified_drift_source) < 0
            or not layout.keyframes or layout.expansion_duration <= 0
            or layout.region_bottom <= layout.region_top
            or min(tempo.width, tempo.glyph_width, tempo.height, tempo.font_size) <= 0):
        return _invalid("timeline references")
    for track in (layout.zoom, *layout.tops.values(), *layout.opacities.values()):
        if not track.keys or any(last.time <= first.time for first, last in zip(track.keys, track.keys[1:])):
            return _invalid("curve order")
    if (any(frame.scale <= 0 or set(frame.tops) != ids or set(frame.opacities) != ids
            for frame in layout.keyframes)
            or any(last.time <= first.time for first, last in zip(layout.keyframes, layout.keyframes[1:]))):
        return _invalid("keyframes")
    # Rebuild derived lookup arrays without running the camera/layout compiler.
    layout.times = [frame.time for frame in layout.keyframes]
    return layout, camera, tempo

