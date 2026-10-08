"""Serializable application contracts. Musical source timing is never quantized in place."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import tempfile
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from functools import cache
from pathlib import Path
from types import UnionType
from typing import Any, get_args, get_origin, get_type_hints


@cache
def _record_types(record_type: type) -> dict[str, Any]:
    return get_type_hints(record_type)


def _check_type(value: Any, expected: Any, location: str) -> None:
    """Check file structure without requiring the project to be ready to render."""
    origin = get_origin(expected)
    if origin is UnionType:
        for candidate in get_args(expected):
            try:
                _check_type(value, candidate, location)
                return
            except ValueError:
                pass
        raise ValueError(f"项目字段 {location} 的类型无效。")
    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"项目字段 {location} 必须为数组。")
        element_type = get_args(expected)[0]
        for index, element in enumerate(value):
            _check_type(element, element_type, f"{location}[{index}]")
        return
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"项目字段 {location} 必须为对象。")
        key_type, value_type = get_args(expected)
        for key_value, element in value.items():
            _check_type(key_value, key_type, f"{location} 的键")
            _check_type(element, value_type, f"{location}[{key_value!r}]")
        return
    if expected is float:
        try:
            finite = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"项目字段 {location} 必须为有限数值。")
    elif expected in (int, str, bool, type(None)):
        if type(value) is not expected:
            raise ValueError(f"项目字段 {location} 的类型无效。")
    elif is_dataclass(expected):
        if not isinstance(value, expected):
            raise ValueError(f"项目字段 {location} 的结构无效。")
        _check_record(value, location)


def _check_record(record: Any, location: str) -> None:
    for name, expected in _record_types(type(record)).items():
        _check_type(getattr(record, name), expected, f"{location}.{name}")


def _restore(record_type: type, values: Any, location: str):
    if not isinstance(values, dict):
        raise ValueError(f"项目字段 {location} 必须为对象。")
    try:
        record = record_type(**values)
    except TypeError as exc:
        raise ValueError(f"项目字段 {location} 包含缺失或未知字段：{exc}") from exc
    _check_record(record, location)
    return record


def _array(value: Any, location: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"项目字段 {location} 必须为数组。")
    return value


@dataclass(slots=True)
class Diagnostic:
    severity: str
    code: str
    message: str
    track_id: str = ""


@dataclass(slots=True)
class NoteEvent:
    note_id: str
    track_id: str
    start_tick: int
    duration_tick: int
    pitch: int
    velocity: int = 100
    midi_channel: int = 0
    articulation: str = ""
    source_pattern: str = ""
    slide: bool = False
    key_release_tick: int | None = None

    @property
    def end_tick(self) -> int:
        return self.start_tick + self.duration_tick


@dataclass(slots=True)
class TrackInfo:
    track_id: str
    name: str
    color: str = "#ffffff"
    plugin: str = ""
    midi_channel: int | None = None
    source_kind: str = "notes"
    mixer_insert: int | None = None


@dataclass(slots=True)
class AutomationPoint:
    """Original source-local tick and normalized control value, before Min/Max."""

    tick: float
    value: float
    tension: float = 0.0
    metadata: int = 0


@dataclass(slots=True)
class VolumeAutomation:
    """One arranged automation clip or Pattern recording linked to a volume control."""

    automation_id: str
    target_id: str
    target_kind: str
    points: list[AutomationPoint]
    start_tick: float
    end_tick: float
    source_offset_tick: float = 0.0
    minimum: float = 0.0
    maximum: float = 1.0
    value_scale: float = 1.0
    source_label: str = ""
    supported: bool = True
    unsupported_reason: str = ""
    source_kind: str = "clip"
    raw_payload: str = ""


@dataclass(slots=True)
class VolumeRoute:
    """Confirmed serial or common gain controls; initials use FL raw / 12800."""

    track_id: str
    control_ids: list[str]
    initial_values: dict[str, float]
    supported: bool = True
    unsupported_reason: str = ""


@dataclass(slots=True)
class ProjectIR:
    source_path: str
    source_type: str
    name: str
    ppq: int = 480
    bpm: float = 120.0
    numerator: int = 4
    denominator: int = 4
    tracks: list[TrackInfo] = field(default_factory=list)
    notes: list[NoteEvent] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    source_version: str = ""
    arrangement_names: list[str] = field(default_factory=list)
    arrangement_index: int = 0
    duration_ticks: int = 0
    timing_confirmed: bool = True
    volume_automations: list[VolumeAutomation] = field(default_factory=list)
    volume_routes: list[VolumeRoute] = field(default_factory=list)

    @property
    def end_tick(self) -> int:
        return max(self.duration_ticks, max((n.end_tick for n in self.notes), default=0),
                   math.ceil(max((clip.end_tick for clip in self.volume_automations), default=0)))

    @property
    def bar_beats(self) -> float:
        return self.numerator * 4 / self.denominator

    @property
    def duration_seconds(self) -> float:
        return self.end_tick / self.ppq * 60 / self.bpm


@dataclass(slots=True)
class PartMapping:
    part_id: str
    name: str
    track_ids: list[str]
    instrument: str = "unknown"
    icon: str = ""
    clef: str = "auto"
    key_signature: int | None = None
    transpose: int = 0
    quantization: int = 16
    triplets: bool = True
    grand_staff: bool = False
    percussion: bool = False
    percussion_map: dict[str, str] = field(default_factory=dict)
    keyswitches: list[int] = field(default_factory=list)
    articulations: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    use_icon: bool = True
    auto_staccato: bool = True
    auto_grace: bool = True
    auto_dynamics: bool = True
    auto_simplify_accidentals: bool = True
    auto_ottava: bool = False


ANIMATION_DURATIONS = (
    "enter_seconds", "exit_seconds", "reflow_seconds",
    "overlay_enter_seconds", "overlay_exit_seconds",
)
ANIMATION_PRESETS = {
    name: tuple(duration * factor for duration in (0.35, 0.25, 0.35, 1.0, 0.8))
    for name, factor in (("fast", 0.5), ("medium", 1.0), ("slow", 2.0), ("very_slow", 4.0))
}


@dataclass(slots=True)
class RenderSettings:
    width: int = 1920
    height: int = 1080
    fps: int = 60
    staff_scale: float = 1.0
    score_left: float = 0.17
    score_right: float = 0.88
    score_top: float = 0.23
    score_bottom: float = 0.65
    header_width: float = 0.11
    title_x: float = 0.035
    title_y: float = 0.76
    title_font_size: int = 44
    subtitle_font_size: int = 22
    credits_font_size: int = 22
    score_start_in_audio_sec: float = 0.0
    enter_seconds: float = 0.35
    exit_seconds: float = 0.25
    reflow_seconds: float = 0.35
    animation_stable_seconds: float = 2.0
    intro_delay_seconds: float = 0.0
    overlay_enter_seconds: float = 1.0
    overlay_exit_seconds: float = 0.8
    announcement_auto_hide: bool = False
    announcement_hold_seconds: float = 5.0
    logo_enabled: bool = False
    logo_display_mode: str = "persistent"
    logo_size_ratio: float = 0.09
    logo_opacity: float = 0.8
    logo_enter_seconds: float = 1.0
    logo_hold_seconds: float = 5.0
    logo_exit_seconds: float = 0.8
    cache_megabytes: int = 64
    crf: int = 18
    preset: str = "medium"
    # Preserve project values: auto prefers RHI Vulkan, gpu requires it, cpu opts out.
    render_backend: str = "auto"
    video_encoder: str = "auto"
    nvenc_cq: int = 18
    nvenc_preset: str = "p5"

    def animation_preset(self) -> str:
        values = tuple(getattr(self, name) for name in ANIMATION_DURATIONS)
        return next((name for name, durations in ANIMATION_PRESETS.items()
                     if all(math.isclose(value, duration, abs_tol=1e-6)
                            for value, duration in zip(values, durations, strict=True))), "custom")

    def apply_animation_preset(self, name: str) -> None:
        if name not in ANIMATION_PRESETS:
            raise ValueError("动画速度预设无效。")
        for field_name, duration in zip(ANIMATION_DURATIONS, ANIMATION_PRESETS[name], strict=True):
            setattr(self, field_name, duration)

    def audio_time(self, presentation_time: float) -> float:
        return max(0.0, presentation_time - self.intro_delay_seconds)

    def presentation_time(self, audio_time: float) -> float:
        return self.intro_delay_seconds + audio_time

    def presentation_duration(self, audio_duration: float, score_duration: float) -> float:
        return self.intro_delay_seconds + max(0.0, audio_duration,
                                             score_duration + self.score_start_in_audio_sec)

    def validate(self) -> None:
        _check_record(self, "settings")
        if self.width < 320 or self.height < 240 or self.width % 2 or self.height % 2:
            raise ValueError("视频尺寸必须为偶数，且至少为 320×240。")
        if self.fps not in (24, 25, 30, 50, 60):
            raise ValueError("帧率必须为 24、25、30、50 或 60。")
        if not (0 <= self.score_left < self.score_left + self.header_width < self.score_right <= 1):
            raise ValueError("谱区横向边界和固定头部宽度无效。")
        if not (0 <= self.score_top < self.score_bottom <= 1):
            raise ValueError("谱区纵向边界无效。")
        if not (0.3 <= self.staff_scale <= 3.0):
            raise ValueError("谱面缩放必须在 0.3 到 3.0 之间。")
        if self.cache_megabytes < 8:
            raise ValueError("图块缓存至少需要 8 MiB。")
        if min(getattr(self, name) for name in ANIMATION_DURATIONS) <= 0:
            raise ValueError("动画时长必须为正数。")
        if self.animation_stable_seconds < 0:
            raise ValueError("动画后最短稳定时间必须为非负数。")
        if min(self.intro_delay_seconds, self.announcement_hold_seconds) < 0:
            raise ValueError("开场停顿和文字停留时间必须为非负数。")
        if self.logo_display_mode not in ("persistent", "fade_in", "intro"):
            raise ValueError("Logo 显示方式无效。")
        if not 0.02 <= self.logo_size_ratio <= 0.25:
            raise ValueError("Logo 大小比例必须位于 0.02 到 0.25。")
        if not 0 <= self.logo_opacity <= 1:
            raise ValueError("Logo 不透明度必须位于 0 到 1。")
        if min(self.logo_enter_seconds, self.logo_exit_seconds) <= 0:
            raise ValueError("Logo 淡入和淡出时长必须为正数。")
        if self.logo_hold_seconds < 0:
            raise ValueError("Logo 停留时间必须为非负数。")
        if not 0 <= self.title_x <= 1 or not 0 <= self.title_y <= 1:
            raise ValueError("标题位置必须位于画面内。")
        if min(self.title_font_size, self.subtitle_font_size, self.credits_font_size) <= 0:
            raise ValueError("文字字号必须为正整数。")
        if not 0 <= self.crf <= 51:
            raise ValueError("H.264 CRF 必须位于 0 到 51。")
        if self.preset not in ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"):
            raise ValueError("H.264 编码预设无效。")
        if self.render_backend not in ("auto", "cpu", "gpu"):
            raise ValueError("帧渲染后端必须为 auto、cpu 或 gpu。")
        if self.video_encoder not in ("auto", "libx264", "h264_nvenc"):
            raise ValueError("视频编码器必须为 auto、libx264 或 h264_nvenc。")
        if not 1 <= self.nvenc_cq <= 51:
            raise ValueError("NVENC CQ 必须位于 1 到 51；较小的值更清晰。")
        if self.nvenc_preset not in tuple(f"p{index}" for index in range(1, 8)):
            raise ValueError("NVENC 编码预设必须为 p1 到 p7。")


@dataclass(slots=True)
class Metadata:
    title: str = "Untitled"
    subtitle: str = ""
    composer: str = ""
    arranger: str = ""


@dataclass(slots=True)
class IconAsset:
    """An embedded, render-ready image; no local paths or Qt objects."""

    name: str
    media_type: str
    data: str
    source: str = "custom"

    def decoded(self) -> bytes:
        try:
            return base64.b64decode(self.data, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError(f"图标 {self.name} 的 Base64 数据无效。") from exc


def _check_icon_assets(document: ProjectDocument) -> None:
    for digest, asset in document.icon_assets.items():
        if asset.media_type not in ("image/svg+xml", "image/png", "image/jpeg", "image/webp"):
            raise ValueError(f"图标 {asset.name} 的图片格式不受支持。")
        data = asset.decoded()
        if not data or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"图标 {asset.name} 的内容校验失败。")
    for mapping in document.mappings:
        if mapping.icon.startswith("asset:") and mapping.icon[6:] not in document.icon_assets:
            raise ValueError(f"分谱 {mapping.name} 引用了缺失的图标资源。")


@dataclass(slots=True)
class ProjectDocument:
    project: ProjectIR
    mappings: list[PartMapping]
    audio_path: str = ""
    settings: RenderSettings = field(default_factory=RenderSettings)
    metadata: Metadata = field(default_factory=Metadata)
    schema_version: int = 1
    icon_assets: dict[str, IconAsset] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        referenced = {mapping.icon[6:] for mapping in self.mappings if mapping.icon.startswith("asset:")}
        value["icon_assets"] = {key: asset for key, asset in value["icon_assets"].items() if key in referenced}
        if not value["icon_assets"]:
            value.pop("icon_assets")
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProjectDocument:
        if not isinstance(value, dict):
            raise ValueError("项目文件的最外层必须为对象。")
        version = value.get("schema_version", 1)
        if type(version) is not int or version != 1:
            raise ValueError("不支持此项目文件版本。")
        unknown = set(value) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"项目文件包含未知字段：{', '.join(sorted(unknown))}")
        project_values = value.get("project")
        if not isinstance(project_values, dict):
            raise ValueError("项目文件缺少有效的 project 对象。")
        p = dict(project_values)
        legacy_flp = p.get("source_type") == "flp"
        legacy_volume = legacy_flp and "volume_automations" not in p
        legacy_note_ids = [
            item.get("note_id", "") for item in _array(p.get("notes", []), "project.notes")
            if isinstance(item, dict) and "key_release_tick" not in item
        ]
        for name, record_type in (("tracks", TrackInfo), ("notes", NoteEvent),
                                  ("diagnostics", Diagnostic), ("volume_routes", VolumeRoute)):
            p[name] = [_restore(record_type, item, f"project.{name}[{index}]") for index, item in enumerate(_array(p.get(name, []), f"project.{name}"))]
        automations = []
        for index, item in enumerate(_array(p.get("volume_automations", []), "project.volume_automations")):
            location = f"project.volume_automations[{index}]"
            if not isinstance(item, dict):
                raise ValueError(f"项目字段 {location} 必须为对象。")
            automation = dict(item)
            automation["points"] = [
                _restore(AutomationPoint, point, f"{location}.points[{point_index}]")
                for point_index, point in enumerate(_array(automation.get("points", []), f"{location}.points"))
            ]
            automations.append(_restore(VolumeAutomation, automation, location))
        p["volume_automations"] = automations
        if legacy_volume:
            p["diagnostics"].append(Diagnostic(
                "info", "legacy-volume-automation",
                "旧 FLP 项目未保存音量自动化；重新导入源工程可补齐力度渐变，已保存的分谱设置仍可保留。",
            ))
        if legacy_flp and legacy_note_ids and not any(
            item.code == "step_trigger_duration" for item in p["diagnostics"]
        ):
            legacy_ids = set(legacy_note_ids)
            for event in p["notes"]:
                if event.note_id in legacy_ids:
                    event.key_release_tick = event.end_tick
            p["diagnostics"].append(Diagnostic(
                "info", "legacy-flp-gate",
                "旧 FLP 项目按保存时长补充门控信息；旧裁切信息无法恢复，可重新导入以获得更准确的自动识别。",
            ))
        settings_values = value.get("settings", {})
        if not isinstance(settings_values, dict):
            raise ValueError("项目字段 settings 必须为对象。")
        settings_values = dict(settings_values)
        settings_values.pop("tempo_display_mode", None)
        settings_values.pop("tempo_hold_seconds", None)
        asset_values = value.get("icon_assets", {})
        if not isinstance(asset_values, dict):
            raise ValueError("项目字段 icon_assets 必须为对象。")
        mappings = []
        for index, item in enumerate(_array(value.get("mappings", []), "mappings")):
            location = f"mappings[{index}]"
            if not isinstance(item, dict):
                raise ValueError(f"项目字段 {location} 必须为对象。")
            mapping_values = dict(item)
            if "confirmed" in mapping_values:
                legacy_use_icon = mapping_values.pop("confirmed")
                _check_type(legacy_use_icon, bool, f"{location}.confirmed")
                mapping_values.setdefault("use_icon", legacy_use_icon)
            mapping = _restore(PartMapping, mapping_values, location)
            if mapping.icon == "none":
                mapping.icon = ""
                mapping.use_icon = False
            mappings.append(mapping)
        document = cls(
            project=_restore(ProjectIR, p, "project"),
            mappings=mappings,
            audio_path=value.get("audio_path", ""),
            settings=_restore(RenderSettings, settings_values, "settings"),
            metadata=_restore(Metadata, value.get("metadata", {}), "metadata"),
            icon_assets={key: _restore(IconAsset, item, f"icon_assets[{key!r}]")
                         for key, item in asset_values.items()},
        )
        _check_record(document, "document")
        _check_icon_assets(document)
        return document

    def validate(self) -> None:
        _check_record(self, "document")
        _check_icon_assets(self)
        if self.schema_version != 1:
            raise ValueError("不支持此项目文件版本。")
        self.settings.validate()
        p = self.project
        if p.ppq <= 0 or not 1 <= p.bpm <= 999 or p.numerator <= 0 or p.denominator not in (1, 2, 4, 8, 16, 32):
            raise ValueError("PPQ、速度或拍号无效。")
        if not p.timing_confirmed:
            raise ValueError("请确认工程使用固定速度和拍号，或导入已验证的 MIDI。")
        if p.duration_ticks < 0 or p.arrangement_index < 0 or (p.arrangement_names and p.arrangement_index >= len(p.arrangement_names)):
            raise ValueError("工程长度或编曲索引无效。")
        tracks: set[str] = set()
        for track in p.tracks:
            if not track.track_id or track.track_id in tracks:
                raise ValueError("音轨 ID 不能为空或重复。")
            tracks.add(track.track_id)
            if track.midi_channel is not None and not 0 <= track.midi_channel <= 15:
                raise ValueError("音轨 MIDI 通道必须位于 0 到 15。")
            if track.mixer_insert is not None and not 0 <= track.mixer_insert <= 127:
                raise ValueError("混音器轨道必须位于 0 到 127。")
        route_ids: set[str] = set()
        for route in p.volume_routes:
            if route.track_id not in tracks or route.track_id in route_ids:
                raise ValueError("音量路由引用了不存在或重复的音轨。")
            route_ids.add(route.track_id)
            if len(route.control_ids) != len(set(route.control_ids)) or any(not control for control in route.control_ids):
                raise ValueError("音量路由的控制目标不能为空或重复。")
            if set(route.initial_values) != set(route.control_ids) or any(not 0 <= value <= 1.25 for value in route.initial_values.values()):
                raise ValueError("音量路由须为每个控制目标保存有效的 FL 音量初值。")
        automation_ids: set[str] = set()
        for automation in p.volume_automations:
            if not automation.automation_id or automation.automation_id in automation_ids or not automation.target_id:
                raise ValueError("音量自动化 ID 不能为空或重复，控制目标不能为空。")
            automation_ids.add(automation.automation_id)
            if automation.target_kind not in ("channel_volume", "mixer_volume"):
                raise ValueError("音量自动化目标类型无效。")
            if automation.source_kind not in ("clip", "pattern"):
                raise ValueError("音量自动化来源类型无效。")
            if (not isinstance(automation.raw_payload, str) or len(automation.raw_payload) % 2
                    or any(character not in "0123456789abcdefABCDEF" for character in automation.raw_payload)):
                raise ValueError("音量自动化原始记录必须使用十六进制保存。")
            if automation.start_tick < 0 or automation.end_tick <= automation.start_tick or automation.source_offset_tick < 0:
                raise ValueError("音量自动化起止或裁切位置无效。")
            if not all(0 <= value <= 1 for value in (automation.minimum, automation.maximum)) or not 0 < automation.value_scale <= 1.25:
                raise ValueError("音量自动化范围和控制倍率无效。")
            if automation.supported and not automation.points:
                raise ValueError("音量自动化须保留控制点。")
            previous_tick = -1.0
            for point in automation.points:
                if point.tick < 0 or point.tick < previous_tick or not 0 <= point.value <= 1 or not 0 <= point.metadata <= 0xffffffff:
                    raise ValueError("音量自动化控制点必须按非负拍位排序，并保留有效的归一化值和曲线元数据。")
                previous_tick = point.tick
        note_ids: set[str] = set()
        for note_event in p.notes:
            if not note_event.note_id or note_event.note_id in note_ids:
                raise ValueError("音符 ID 不能为空或重复。")
            note_ids.add(note_event.note_id)
            if note_event.track_id not in tracks:
                raise ValueError(f"音符 {note_event.note_id} 引用了不存在的音轨。")
            if note_event.start_tick < 0 or note_event.duration_tick < 0:
                raise ValueError("原始音符的起点和时长必须为非负 tick 整数。")
            if note_event.key_release_tick is not None and not (
                note_event.start_tick <= note_event.key_release_tick <= note_event.end_tick
            ):
                raise ValueError("音符的按键释放时刻必须位于原始起止时间内。")
            if not 0 <= note_event.pitch <= 127 or not 0 <= note_event.velocity <= 127 or not 0 <= note_event.midi_channel <= 15:
                raise ValueError("音符音高、力度必须位于 0 到 127，MIDI 通道必须位于 0 到 15。")
        used: set[str] = set()
        ids: set[str] = set()
        for m in self.mappings:
            if not m.part_id or m.part_id in ids:
                raise ValueError("分谱 ID 不能为空或重复。")
            ids.add(m.part_id)
            if not m.enabled:
                continue
            if not m.track_ids or any(t not in tracks for t in m.track_ids):
                raise ValueError(f"分谱 {m.name} 包含不存在的音轨。")
            if len(m.track_ids) != len(set(m.track_ids)):
                raise ValueError(f"分谱 {m.name} 包含重复的音轨。")
            if used.intersection(m.track_ids):
                raise ValueError("同一个音轨不能同时归属多个已启用分谱。")
            used.update(m.track_ids)
            if m.quantization not in (8, 16, 32, 64):
                raise ValueError("量化格点必须为八分、十六分、三十二分或六十四分音符。")
            if m.clef not in ("auto", "treble", "bass", "alto", "tenor", "percussion"):
                raise ValueError("谱号设置无效。")
            if m.key_signature is not None and not -7 <= m.key_signature <= 7:
                raise ValueError("调号必须在七个降号与七个升号之间。")
            if not -127 <= m.transpose <= 127:
                raise ValueError("移调必须为 -127 到 127 之间的半音整数。")
            if any(not 0 <= pitch <= 127 for pitch in m.keyswitches):
                raise ValueError("Keyswitch 音高必须位于 0 到 127。")
            for source_pitch, display_pitch in m.percussion_map.items():
                if not source_pitch.isascii() or not source_pitch.isdecimal() or not 0 <= int(source_pitch) <= 127 or source_pitch != str(int(source_pitch)) or not display_pitch.strip():
                    raise ValueError("打击乐映射必须使用 0 到 127 的音高字符串作为键，并指定非空谱面位置。")
            if any(track_id not in m.track_ids for track_id in m.articulations):
                raise ValueError(f"分谱 {m.name} 的奏法设置引用了未归属该分谱的音轨。")
        if not used:
            raise ValueError("至少需要一个已启用的分谱。")


def save_document(document: ProjectDocument, path: str | Path) -> None:
    # Persist incomplete editing work; rendering readiness is checked separately.
    _check_record(document, "document")
    _check_icon_assets(document)
    if document.schema_version != 1:
        raise ValueError("不支持此项目文件版本。")
    payload = json.dumps(document.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_document(path: str | Path) -> ProjectDocument:
    target = Path(path).resolve()
    def invalid_constant(value: str):
        raise ValueError(f"项目 JSON 包含非有限数值 {value}。")
    try:
        value = json.loads(target.read_text(encoding="utf-8-sig"), parse_constant=invalid_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"项目 JSON 无效（第 {exc.lineno} 行，第 {exc.colno} 列）：{exc.msg}") from exc
    document = ProjectDocument.from_dict(value)
    for owner, attr in ((document, "audio_path"), (document.project, "source_path")):
        reference = getattr(owner, attr)
        if reference and not Path(reference).is_absolute():
            setattr(owner, attr, str((target.parent / reference).resolve()))
    return document
