"""Compile shared engraving geometry and absolute-time visibility before rendering."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field, replace

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QTransform
from PySide6.QtSvg import QSvgRenderer

from .axis import TimeAxis
from .camera import CameraTimeline, compile_camera
from .icons import resolve_icon
from .layout import LayoutTimeline, _Track, compile_layout, ease
from .mapping import activity_color
from .models import Diagnostic, IconAsset, Metadata, ProjectDocument, RenderSettings
from .musicfont import metronome_renderer
from .qt import ensure_app
from .svg import normalize_svg
from .typography import _layout


@dataclass(slots=True)
class ActiveNote:
    start: float
    end: float
    velocity: int


@dataclass(slots=True)
class VisibleSpan:
    start: float
    end: float
    initial: bool = False


@dataclass(slots=True)
class ScenePart:
    part_id: str
    name: str
    icon: str
    source_top: float
    source_bottom: float
    staff_centers: list[float]
    notes: list[ActiveNote]
    spans: list[VisibleSpan] = field(default_factory=list)
    opacity_keys: list[tuple[float, float]] = field(default_factory=list)
    opacity_curve: _Track | None = None
    activity_color: str = "#ffffff"

    @property
    def source_height(self) -> float:
        return self.source_bottom - self.source_top


@dataclass(slots=True, frozen=True)
class TempoMark:
    """A single score-owned mark at the first beat, measured in source units."""

    owner_id: str
    x: float
    label: str
    width: float
    glyph_width: float
    height: float = 360.0
    gap: float = 135.0
    label_gap: float = 90.0
    font_size: float = 270.0

    @property
    def padding(self) -> float:
        return self.height + self.gap


@dataclass(slots=True, frozen=True)
class SceneOctaveSpan:
    """Source geometry for a reminder when an octave line enters from the left."""

    part_id: str
    staff_index: int
    label: str
    label_right: float
    end_x: float
    label_y: float
    font_size: float = 240.0


@dataclass(slots=True)
class PartGeometry:
    part_id: str
    source_top: float
    source_bottom: float
    staff_centers: list[float]


@dataclass(slots=True)
class EngravingDiagnostic:
    severity: str
    code: str
    body: str
    part_id: str


@dataclass(slots=True)
class EngravedGeometry:
    svg: str
    view_box: tuple[float, float, float, float]
    beats: list[float]
    xs: list[float]
    parts: list[PartGeometry]
    header_left: float
    header_right: float
    measure_bounds: list[tuple[float, float]]
    element_part_ids: dict[str, str]
    octave_spans: list[SceneOctaveSpan]
    terminal_x: float
    end_beat: float
    diagnostics: list[EngravingDiagnostic]


@dataclass(slots=True)
class CompiledScene:
    svg: str
    view_box: tuple[float, float, float, float]
    axis: TimeAxis
    parts: list[ScenePart]
    settings: RenderSettings
    metadata: Metadata
    bpm: float
    bar_beats: float
    score_duration: float
    scale: float
    header_left: float
    header_right: float
    diagnostics: list[Diagnostic]
    measure_bounds: list[tuple[float, float]]
    element_part_ids: dict[str, str] = field(default_factory=dict)
    layout: LayoutTimeline | None = None
    terminal_x: float | None = None
    camera: CameraTimeline | None = None
    tempo_mark: TempoMark | None = None
    octave_spans: list[SceneOctaveSpan] = field(default_factory=list)
    icon_assets: dict[str, IconAsset] = field(default_factory=dict)
    compilation_report: dict = field(default_factory=dict)

    @property
    def body_left(self) -> float:
        return (self.settings.score_left + self.settings.header_width) * self.settings.width

    @property
    def play_x(self) -> float:
        return sum(self.play_corridor) / 2

    @property
    def play_corridor(self) -> tuple[float, float]:
        width = self.body_right - self.body_left
        unit = self.settings.width / 1920
        right = min(0.10 * width, 120 * unit)
        left = min(16 * unit, right)
        return self.body_left + left, self.body_left + right

    @property
    def body_right(self) -> float:
        return self.settings.score_right * self.settings.width

    def beat_at_time(self, time: float) -> float:
        return (time - self.settings.score_start_in_audio_sec) * self.bpm / 60

    def camera_x_at(self, presentation_time: float) -> float:
        if self.camera is None:
            raise RuntimeError("谱面尚未编译相机时间轴。")
        return self.camera.x_at(self.settings.audio_time(presentation_time))

    def camera_speed_at(self, presentation_time: float) -> float:
        if self.camera is None:
            raise RuntimeError("谱面尚未编译相机时间轴。")
        if presentation_time < self.settings.intro_delay_seconds:
            return 0.0
        return self.camera.speed_at(self.settings.audio_time(presentation_time))

    def camera_min_speed(self, start: float, end: float) -> float:
        if self.camera is None:
            raise RuntimeError("谱面尚未编译相机时间轴。")
        if min(start, end) < self.settings.intro_delay_seconds:
            return 0.0
        return self.camera.min_speed(self.settings.audio_time(start), self.settings.audio_time(end))


def smoothstep(value: float) -> float:
    return ease(value)


def part_state(part: ScenePart, time: float, settings: RenderSettings) -> tuple[float, float]:
    """Opacity and occupancy at presentation time, independent of frame history."""
    if part.opacity_curve is not None:
        opacity = max(0.0, min(1.0, part.opacity_curve.sample(time).value))
        return opacity, 1.0 if opacity > 0 else 0.0
    if part.opacity_keys:
        index = max(0, bisect_right(part.opacity_keys, (time, math.inf)) - 1)
        start, first = part.opacity_keys[index]
        end, last = part.opacity_keys[min(index + 1, len(part.opacity_keys) - 1)]
        fraction = ease((time - start) / (end - start)) if end > start else 0.0
        opacity = first + (last - first) * fraction
        # A fading row retains a complete slot; FrameLayout owns reflow geometry.
        return opacity, 1.0 if opacity > 0 else 0.0
    opacity = occupancy = 0.0
    for span in part.spans:
        if time < span.start or time > span.end + max(settings.exit_seconds, settings.reflow_seconds):
            continue
        enter = 1.0 if span.initial else smoothstep((time - span.start) / settings.enter_seconds)
        occupy_enter = 1.0 if span.initial else smoothstep((time - span.start) / settings.reflow_seconds)
        leave = 1 - smoothstep((time - span.end) / settings.exit_seconds)
        occupy_leave = 1 - smoothstep((time - span.end) / settings.reflow_seconds)
        opacity = max(opacity, enter * leave)
        occupancy = max(occupancy, occupy_enter * occupy_leave)
    return opacity, occupancy


def _classes(el: ET.Element) -> set[str]:
    return set(el.get("class", "").split())


def _transform(text: str) -> QTransform:
    result = QTransform()
    for command, values in re.findall(r"([A-Za-z]+)\s*\(([^)]*)\)", text):
        numbers = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", values)]
        if command == "translate":
            result.translate(numbers[0], numbers[1] if len(numbers) > 1 else 0)
        elif command == "scale":
            result.scale(numbers[0], numbers[1] if len(numbers) > 1 else numbers[0])
        elif command == "matrix" and len(numbers) == 6:
            result = QTransform(*numbers) * result
        elif command == "rotate":
            if len(numbers) == 3:
                result.translate(numbers[1], numbers[2])
            result.rotate(numbers[0])
            if len(numbers) == 3:
                result.translate(-numbers[1], -numbers[2])
    return result


def _svg_index(root: ET.Element) -> tuple[dict[str, ET.Element], dict[int, QTransform]]:
    index = {}
    transforms = {}

    def walk(el: ET.Element, parent: QTransform) -> None:
        current = _transform(el.get("transform", "")) * parent
        transforms[id(el)] = current
        if el.get("id"):
            index[el.get("id")] = el
        for child in el:
            walk(child, current)

    walk(root, QTransform())
    return index, transforms


def _bounds(renderer: QSvgRenderer, el: ET.Element) -> QRectF:
    ident = el.get("id", "")
    if not ident or not renderer.elementExists(ident):
        return QRectF()
    return renderer.transformForElement(ident).mapRect(renderer.boundsOnElement(ident))


def _anchor_x(el: ET.Element, transforms: dict[int, QTransform], renderer: QSvgRenderer) -> float:
    heads = [n for n in el.iter() if "notehead" in _classes(n)]
    search = heads[0] if heads else el
    for use in search.iter():
        if use.tag.endswith("}use"):
            return transforms[id(use)].map(QPointF(float(use.get("x", "0")), float(use.get("y", "0")))).x()
    return _bounds(renderer, el).left()


def _identify_octave_labels(root: ET.Element) -> None:
    """Give octave glyphs IDs so Qt can measure the number separately from its line."""
    for element in root.iter():
        if "octave" not in _classes(element):
            continue
        identifier = element.get("id")
        if not identifier:
            continue
        for index, child in enumerate(element.iter()):
            if child.tag.rsplit("}", 1)[-1] in {"use", "text"} and not child.get("id"):
                child.set("id", f"{identifier}_label_{index}")


def _scene_octave_spans(notation, index: dict[str, ET.Element],
                        renderer: QSvgRenderer) -> list[SceneOctaveSpan]:
    spans = getattr(notation, "octave_spans", ())
    mei = getattr(notation, "mei", "")
    if not spans or not mei:
        return []
    xml_id = "{http://www.w3.org/XML/1998/namespace}id"
    mei_root = ET.fromstring(mei)
    mei_index = {element.get(xml_id): element for element in mei_root.iter() if element.get(xml_id)}
    mei_parents = {child: parent for parent in mei_root.iter() for child in parent}
    octaves = [element for element in mei_root.iter()
               if element.tag.rsplit("}", 1)[-1] == "octave"]

    def references(reference: str, note_id: str) -> bool:
        identifier = reference.lstrip("#")
        if identifier == note_id:
            return True
        element = mei_index.get(identifier)
        # Verovio may anchor an octave stop to a chord's last note, even though
        # the exported span stores its first note. Both identify the same onset.
        parent = mei_parents.get(element)
        if parent is not None and parent.tag.rsplit("}", 1)[-1] == "chord":
            element = parent
        return element is not None and any(child.get(xml_id) == note_id for child in element.iter())

    result = []
    for span in spans:
        if not span.start_element_id or not span.end_element_id:
            continue
        staves = [i + 1 for i, part_id in enumerate(notation.staff_part_ids) if part_id == span.part_id]
        if not 0 <= span.staff_index < len(staves):
            continue
        expected_staff = str(staves[span.staff_index])
        element = next((element for element in octaves
                        if expected_staff in element.get("staff", "").split()
                        and element.get("dis") == ("8" if abs(span.octaves) == 1 else "15")
                        and element.get("dis.place") == ("above" if span.octaves > 0 else "below")
                        and references(element.get("startid", ""), span.start_element_id)
                        and references(element.get("endid", ""), span.end_element_id)), None)
        engraved = index.get(element.get(xml_id, "")) if element is not None else None
        if engraved is None:
            continue
        labels = [_bounds(renderer, child) for child in engraved.iter()
                  if child.tag.rsplit("}", 1)[-1] in {"use", "text"}]
        labels = [rect for rect in labels if not rect.isEmpty()]
        rect = _bounds(renderer, engraved)
        if not labels or rect.isEmpty():
            continue
        label = labels[0]
        for other in labels[1:]:
            label = label.united(other)
        result.append(SceneOctaveSpan(span.part_id, span.staff_index, span.label,
                                      label.right(), rect.right(), label.center().y()))
    return result


def compile_visibility(scene: CompiledScene) -> None:
    """Visibility and zoom share absolute presentation time with the camera."""
    from .compilation_cache import restore_visibility

    glyph = metronome_renderer()
    glyph_width = 360 * glyph.viewBoxF().width() / glyph.viewBoxF().height()
    label = "= " + f"{scene.bpm:.6f}".rstrip("0").rstrip(".")
    text_layout = _layout(label, 270)
    label_width = text_layout.lineAt(0).naturalTextWidth()
    scene.tempo_mark = TempoMark(scene.parts[0].part_id, scene.axis.x_at(0), label,
                                glyph_width + 90 + label_width + 2, glyph_width)
    if scene.camera is None:
        left, right = scene.play_corridor
        scene.camera = compile_camera(scene.axis, scene.bpm, scene.settings.score_start_in_audio_sec,
                                      (right - left) / (2 * scene.scale))
    scene.layout = compile_layout(scene)
    restore_visibility(scene)


def _axis_from_anchors(notes: dict[float, float], synthetic: dict[float, float]) -> TimeAxis:
    """Never let rests or approximate bar positions displace an actual onset."""
    trusted = sorted(notes.items())
    for (first_beat, first_x), (last_beat, last_x) in zip(trusted, trusted[1:]):
        if last_x <= first_x:
            raise ValueError(f"刻谱音符拍位发生横向冲突：第 {first_beat:g} 与 {last_beat:g} 拍"
                             f"位置为 {first_x:g}、{last_x:g}；请核对声部与量化设置。")
    beats = [beat for beat, _ in trusted]
    selected = dict(notes)
    previous_x = -math.inf
    for beat, x in sorted((notes | {key: value for key, value in synthetic.items() if key not in notes}).items()):
        if beat in notes:
            previous_x = x
            continue
        index = bisect_left(beats, beat)
        next_x = trusted[index][1] if index < len(trusted) else math.inf
        if previous_x < x < next_x:
            selected[beat] = x
            previous_x = x
    if len(selected) < 2:
        raise ValueError("无法建立单调的谱面时间轴。")
    ordered = sorted(selected.items())
    return TimeAxis([beat for beat, _ in ordered], [x for _, x in ordered])


def _engrave_geometry(document: ProjectDocument) -> EngravedGeometry:
    from .notation import build_notation

    notation = build_notation(document)
    svg = normalize_svg(notation.display_svg)
    root = ET.fromstring(svg)
    _identify_octave_labels(root)
    svg = ET.tostring(root, encoding="unicode")
    renderer = QSvgRenderer(svg.encode("utf-8"))
    if not renderer.isValid():
        raise ValueError("Qt 无法读取刻谱结果。")
    index, transforms = _svg_index(root)
    measures = [e for e in root.iter() if "measure" in _classes(e)]
    if not measures:
        raise ValueError("谱面没有小节，无法建立播放时间轴。")
    first_staves = [e for e in measures[0] if "staff" in _classes(e)]
    if len(first_staves) != len(notation.staff_part_ids):
        raise ValueError("刻谱谱表数量与乐器映射不符，已停止以防缺失声部。")
    staff_rects = [_bounds(renderer, el) for el in first_staves]
    owners = dict(notation.element_part_ids)
    for staff, part_id in zip(first_staves, notation.staff_part_ids, strict=True):
        owners[staff.attrib["id"]] = part_id
    centers = []
    for staff in first_staves:
        paths = [e for e in staff if e.tag.endswith("}path")]
        ys = []
        for path in paths[:5]:
            coordinates = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", path.get("d", ""))
            if len(coordinates) >= 2:
                ys.append(transforms[id(path)].map(QPointF(float(coordinates[0]), float(coordinates[1]))).y())
        centers.append(sum(ys) / len(ys) if ys else _bounds(renderer, staff).center().y())
    for measure in measures[1:]:
        staves = [e for e in measure if "staff" in _classes(e)]
        if len(staves) != len(first_staves):
            raise ValueError("某小节缺少谱表，已停止以防不完整显示。")
        for i, staff in enumerate(staves):
            staff_rects[i] = staff_rects[i].united(_bounds(renderer, staff))
            owners[staff.attrib["id"]] = notation.staff_part_ids[i]

    # Verovio groups every staff's bar-line paths under one system-level node.
    # Give those paths individual identities so tall ledger-line bands do not
    # pick up fragments from another staff. The bridge between piano staves
    # maps to the same logical instrument as its two endpoints.
    for measure_index, measure in enumerate(measures):
        for bar in (e for e in measure if "barLine" in _classes(e)):
            for path_index, path in enumerate(bar.iter()):
                if not path.tag.endswith("}path"):
                    continue
                coordinates = [float(value) for value in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", path.get("d", ""))]
                if len(coordinates) < 4 or len(coordinates) % 2:
                    continue
                points = [transforms[id(path)].map(QPointF(x, y)) for x, y in zip(coordinates[::2], coordinates[1::2], strict=True)]
                midpoint = (min(point.y() for point in points) + max(point.y() for point in points)) / 2
                nearest = min(range(len(centers)), key=lambda i: abs(midpoint - centers[i]))
                identifier = path.get("id") or f"sp_render_bar_{measure_index}_{bar.attrib['id']}_{path_index}"
                path.set("id", identifier)
                owners[identifier] = notation.staff_part_ids[nearest]

    # MEI ownership keeps directions with their instrument even when their
    # bounding box extends toward a neighbouring staff. Fall back to geometry
    # only for decorations without an explicit staff or note reference.
    for element in root.iter():
        if _classes(element).intersection({"tie", "slur", "dir", "dynam", "hairpin", "tuplet", "octave"}):
            rect = _bounds(renderer, element)
            if not rect.isEmpty():
                owner = owners.get(element.get("id", ""))
                candidates = [i for i, part_id in enumerate(notation.staff_part_ids) if part_id == owner] or list(range(len(centers)))
                nearest = min(candidates, key=lambda i: abs(rect.center().y() - centers[i]))
                owners[element.attrib["id"]] = notation.staff_part_ids[nearest]
                staff_rects[nearest] = staff_rects[nearest].united(rect)

    octave_spans = _scene_octave_spans(notation, index, renderer)
    for span in octave_spans:
        staves = [i for i, part_id in enumerate(notation.staff_part_ids) if part_id == span.part_id]
        staff_index = staves[span.staff_index]
        # Reminders use the same source scale and may be slightly taller than SMuFL digits.
        label_height = _layout(f"({span.label})", span.font_size, italic=True).boundingRect().height() + 48
        staff_rects[staff_index] = staff_rects[staff_index].united(
            QRectF(span.label_right, span.label_y - label_height / 2, 1, label_height))

    headers = [e for e in measures[0].iter() if _classes(e).intersection({"clef", "keySig", "meterSig"})]
    header_rects = [_bounds(renderer, e) for e in headers if not _bounds(renderer, e).isEmpty()]
    if not header_rects:
        raise ValueError("刻谱结果缺少固定谱号。")
    header_left = min(r.left() for r in header_rects) - 65
    header_right = max(r.right() for r in header_rects) + 90

    notes_at: dict[float, list[float]] = defaultdict(list)
    for anchor in notation.anchors:
        el = index.get(anchor.element_id)
        if el is None or anchor.kind != "note":
            continue
        notes_at[round(anchor.beat, 8)].append(_anchor_x(el, transforms, renderer))
    bar_beats = document.project.bar_beats
    points = {beat: min(xs) for beat, xs in notes_at.items()}
    # Longer rests can be centered far beyond their onset and even later notes.
    # Their engraving remains intact; only notes and safe bar points set the clock.
    bar_ends = []
    for measure in measures:
        bars = [e for e in measure if "barLine" in _classes(e)]
        rects = [_bounds(renderer, e) for e in bars]
        bar_ends.append(max((r.center().x() for r in rects if not r.isEmpty()), default=_bounds(renderer, measure).right()))
    measure_bounds = []
    synthetic = {}
    for i, right in enumerate(bar_ends):
        left = bar_ends[i - 1] if i else header_right + 110
        measure_bounds.append((left, right))
        beat = float(i * bar_beats)
        synthetic[beat] = left + min(220, (right - left) * 0.14)
    final_beat = len(measures) * bar_beats
    synthetic[final_beat] = max(bar_ends[-1], max(points.values(), default=header_right) + 180)
    axis = _axis_from_anchors(points, synthetic)

    groups: dict[str, list[int]] = {}
    for i, part_id in enumerate(notation.staff_part_ids):
        groups.setdefault(part_id, []).append(i)
    parts = []
    for part_id, staff_indices in groups.items():
        content_top = min(staff_rects[i].top() for i in staff_indices)
        content_bottom = max(staff_rects[i].bottom() for i in staff_indices)
        # Padding is source-space and scales with the engraved glyphs.
        parts.append(PartGeometry(
            part_id, content_top - 100, content_bottom + 100,
            [centers[i] for i in staff_indices],
        ))
    terminal_bounds = [_bounds(renderer, element) for element in measures[-1]
                       if "barLine" in _classes(element)]
    terminal_x = max((bounds.right() for bounds in terminal_bounds if not bounds.isEmpty()),
                     default=bar_ends[-1])
    # Import diagnostics belong to the current document, not the cached engraving.
    names = {m.part_id: m.name for m in document.mappings}
    diagnostics = []
    for item in notation.diagnostics[len(document.project.diagnostics):]:
        prefix = names[item.track_id] + "："
        if not item.message.startswith(prefix):
            raise ValueError("记谱诊断缺少声部名称前缀。")
        diagnostics.append(EngravingDiagnostic(item.severity, item.code,
                                               item.message[len(prefix):], item.track_id))
    return EngravedGeometry(
        ET.tostring(root, encoding="unicode"), tuple(renderer.viewBoxF().getRect()),
        axis.beats, axis.xs, parts, header_left, header_right, measure_bounds, owners,
        octave_spans, terminal_x,
        max((e.end_beat for e in notation.quantized_events), default=0), diagnostics,
    )


def _assemble_scene(document: ProjectDocument, geometry: EngravedGeometry) -> CompiledScene:
    mappings = {m.part_id: m for m in document.mappings if m.enabled}
    offset = document.settings.score_start_in_audio_sec
    tick_seconds = 60 / document.project.bpm / document.project.ppq
    parts = []
    for part in geometry.parts:
        mapping = mappings[part.part_id]
        notes = [ActiveNote(n.start_tick * tick_seconds + offset,
                            n.end_tick * tick_seconds + offset, n.velocity)
                 for n in document.project.notes
                 if n.track_id in mapping.track_ids and n.pitch not in mapping.keyswitches]
        parts.append(ScenePart(
            part.part_id, mapping.name,
            (mapping.icon or mapping.instrument) if mapping.use_icon and mapping.icon != "none" else "",
            part.source_top, part.source_bottom, list(part.staff_centers), notes,
            activity_color=activity_color(document.project, mapping),
        ))
    # Whole-song extents can overlap when ledger lines occur in different bars.
    # Keep each extent intact; renderers filter other instruments by identity.
    s = replace(document.settings)
    default_scale = (12 / 180) * (s.height / 1080) * s.staff_scale
    header_scale = (s.header_width * s.width - 20 * s.width / 1920) / (geometry.header_right - geometry.header_left)
    # Upper zoom limit for layout and fixed assets; body tiles independently
    # choose a discrete raster level that covers their current display scale.
    scale = min(2 * default_scale, header_scale)
    score_duration = max(document.project.duration_seconds, geometry.end_beat * 60 / document.project.bpm)
    diagnostics = [replace(item) for item in document.project.diagnostics]
    diagnostics.extend(Diagnostic(item.severity, item.code,
                                  mappings[item.part_id].name + "：" + item.body, item.part_id)
                       for item in geometry.diagnostics)
    scene = CompiledScene(
        geometry.svg, geometry.view_box, TimeAxis(list(geometry.beats), list(geometry.xs)),
        parts, s, replace(document.metadata), document.project.bpm, document.project.bar_beats,
        score_duration, scale, geometry.header_left, geometry.header_right, diagnostics,
        list(geometry.measure_bounds), dict(geometry.element_part_ids),
    )
    scene.icon_assets = {digest: replace(asset) for digest, asset in document.icon_assets.items()
                         if any(part.icon == f"asset:{digest}" for part in parts)}
    scene.octave_spans = list(geometry.octave_spans)
    scene.terminal_x = geometry.terminal_x
    return scene


def compile_scene(document: ProjectDocument, *, use_cache: bool = True,
                  force_rebuild: bool = False, progress=None, cancel=None) -> CompiledScene:
    """Restore independent engraving and timeline layers, then bind current display data."""
    import time

    from .compilation_cache import (
        CompilationCache,
        decode_geometry,
        decode_timeline,
        encode_timeline,
        geometry_key,
        restore_visibility,
        timeline_key,
    )

    started = time.perf_counter()
    cancelled = cancel or (lambda: False)

    def report(fraction, message):
        if cancelled():
            raise InterruptedError("已取消谱面编译。")
        if progress:
            progress(fraction, message)

    report(0.0, "正在准备预览…")
    document.validate()
    ensure_app()
    for mapping in document.mappings:
        if mapping.enabled:
            resolve_icon(mapping.icon, document.icon_assets)
    cache = CompilationCache() if use_cache else None
    key = geometry_key(document) if cache else ""
    geometry = None
    lookup_started = time.perf_counter()
    if cache and not force_rebuild:
        report(0.05, "正在读取已有谱面…")
        geometry = cache.read("geometry", key, lambda data: decode_geometry(data, document))
    geometry_hit = geometry is not None
    lookup_seconds = time.perf_counter() - lookup_started
    engraving_started = time.perf_counter()
    if geometry is None:
        report(0.1, "正在重新制谱…")
        geometry = _engrave_geometry(document)
        report(0.5, "谱面已生成…")
        if cache:
            from dataclasses import asdict

            cache.write("geometry", key, asdict(geometry), cancelled)
    engraving_seconds = time.perf_counter() - engraving_started if not geometry_hit else 0.0
    scene = _assemble_scene(document, geometry)
    time_key = timeline_key(key, document.settings) if cache else ""
    timeline = None
    lookup_started = time.perf_counter()
    if cache and not force_rebuild:
        timeline = cache.read("timeline", time_key, lambda data: decode_timeline(data, scene))
    lookup_seconds += time.perf_counter() - lookup_started
    timeline_hit = timeline is not None
    timeline_started = time.perf_counter()
    if timeline is None:
        report(0.6, "正在更新动画布局…")
        compile_visibility(scene)
        report(0.9, "动画布局已更新…")
        if cache:
            cache.write("timeline", time_key, encode_timeline(scene), cancelled)
    else:
        scene.layout, scene.camera, scene.tempo_mark = timeline
        restore_visibility(scene)
    timeline_seconds = time.perf_counter() - timeline_started if not timeline_hit else 0.0
    minimum_scale = min(key.scale for key in scene.layout.keyframes)
    if minimum_scale * 180 < 6 * scene.settings.height / 1080:
        scene.diagnostics.append(Diagnostic("warning", "DENSE_LAYOUT", "同时显示的谱表较多，建议减少乐器或增加谱区高度。"))
    report(1.0, "已复用谱面与动画布局" if geometry_hit and timeline_hit else "预览编译完成")
    scene.compilation_report = {
        "geometry_cache_hit": geometry_hit, "timeline_cache_hit": timeline_hit,
        "cache_lookup_seconds": lookup_seconds, "engraving_seconds": engraving_seconds,
        "timeline_seconds": timeline_seconds, "total_compile_seconds": time.perf_counter() - started,
    }
    return scene
