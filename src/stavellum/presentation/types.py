"""Shared scene and engraving contracts, independent of compilation."""

from __future__ import annotations

from dataclasses import dataclass, field

from stavellum.domain.models import Diagnostic, IconAsset, Metadata, RenderSettings

from .axis import TimeAxis
from .camera import CameraTimeline
from .curves import _Track
from .timeline import LayoutTimeline


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
