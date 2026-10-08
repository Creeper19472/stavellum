"""Read-only FLP adapter with checked event framing and versioned Playlist records.

PyFLP 2.2.1 supplies the decoded note/channel/track structures. Its top-level
parser cannot handle Python 3.14's empty Enum or modern Playlist layouts, so this
adapter reads framing itself, without patching PyFLP or rewriting source files.
Layouts were checked against Image-Line templates and the real FL 26.1.5 saves
attached to https://github.com/demberto/PyFLP/pull/205. Unknown layouts fail closed.
"""

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

from pyflp.arrangement import TrackEvent
from pyflp.channel import LevelsEvent, ParametersEvent
from pyflp.pattern import NotesEvent

from stavellum.domain.models import Diagnostic, NoteEvent, ProjectIR, TrackInfo

from .errors import ImportFailure
from .flp_automation import (
    build_routes,
    decode_automation,
    decode_pattern_controls,
    place_automation,
    place_pattern_controls,
)

_HEADER = struct.Struct("<4sIhHH")
_CLIP_CORE = struct.Struct("<IHHIHHHH4sII")


@dataclass(slots=True)
class _RawEvent:
    event_id: int
    value: bytes
    offset: int


@dataclass(slots=True)
class _Channel:
    iid: int
    name: str = ""
    plugin: str = ""
    color: str = "#ffffff"
    enabled: bool = True
    kind: int = 0
    sample_path: str = ""
    arpeggiator: bool = False
    pitch_shift: int = 0
    fine_tune: int = 0
    root_note: int = 60
    stretching_pitch: int = 0
    main_pitch: bool | None = None
    add_root: bool | None = None
    volume: int = 10000


@dataclass(slots=True)
class _Pattern:
    iid: int
    name: str = ""
    notes: list = field(default_factory=list)
    length: int = 0
    has_controllers: bool = False
    controllers: list[bytes] = field(default_factory=list)
    channel_loops: bool = False
    meters: list[tuple[int, int, int]] = field(default_factory=list)


@dataclass(slots=True)
class _Arrangement:
    name: str = "Arrangement"
    playlist: bytes | None = None
    tracks_enabled: dict[int, bool] = field(default_factory=dict)
    meters: list[tuple[int, int, int]] = field(default_factory=list)


def _failure(code: str, message: str) -> ImportFailure:
    return ImportFailure(message + " 可改用已烘焙的 MIDI 导入。", [Diagnostic("error", code, message)])


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 35, 7):
        if offset >= len(data):
            raise _failure("invalid_flp_stream", "FLP 事件长度被截断。")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
    raise _failure("invalid_flp_stream", "FLP 事件长度超过允许范围。")


def _read_events(data: bytes) -> tuple[list[_RawEvent], str, int, int]:
    if len(data) < 22:
        raise _failure("invalid_flp_header", "FLP 文件头不完整。")
    magic, size, file_format, channel_count, ppq = _HEADER.unpack_from(data)
    if magic != b"FLhd" or size != 6 or data[14:18] != b"FLdt" or file_format != 0:
        raise _failure("invalid_flp_header", "输入不是完整的 FL Studio Song 工程。")
    if not 1 <= ppq <= 9600 or struct.unpack_from("<I", data, 18)[0] != len(data) - 22:
        raise _failure("invalid_flp_header", "FLP PPQ 或事件块长度无效。")
    offset = 22
    version = ""
    major = 0
    events = []
    while offset < len(data):
        event_start = offset
        event_id = data[offset]
        offset += 1
        if event_id < 64:
            length = 1
        elif event_id < 128:
            length = 2
        elif event_id < 192:
            # Real FL 26 saves contain a three-byte 0xAC followed by text event
            # 0xC0, not a varint-prefixed DWORD. Preserve older four-byte files.
            length = 3 if event_id == 172 and major >= 24 else 4
        else:
            length, offset = _read_varint(data, offset)
        if length > len(data) - offset:
            raise _failure("invalid_flp_stream", f"FLP 事件 {event_id}（字节 {event_start}）被截断。")
        value = data[offset:offset + length]
        offset += length
        if event_id == 199:
            try:
                version = value.rstrip(b"\x00").decode("ascii")
                numbers = tuple(int(part) for part in version.split("."))
            except (ValueError, UnicodeError) as exc:
                raise _failure("unsupported_flp_version", "FLP 版本字段无效。") from exc
            if not numbers or not 11 <= numbers[0] <= 26:
                raise _failure("unsupported_flp_version", f"尚未验证 FL Studio {version} 的格式。")
            major = numbers[0]
        elif not version:
            raise _failure("unsupported_flp_version", "FLP 首事件没有版本信息，无法安全选择格式。")
        events.append(_RawEvent(event_id, value, event_start))
    return events, version, ppq, channel_count


def _text(value: bytes, version: str) -> str:
    encoding = "utf-16-le" if tuple(int(n) for n in version.split(".")[:2]) >= (11, 5) else "ascii"
    return value.decode(encoding).rstrip("\x00")


def _playlist_record_size(version: str, payload: bytes) -> int:
    numbers = tuple(int(n) for n in version.split("."))
    major, minor = numbers[:2] if len(numbers) > 1 else (numbers[0], 0)
    if major >= 26:
        size = 88
    elif major >= 24:
        size = 80
    elif major >= 21 or (major == 20 and minor >= 99):
        size = 60
        # Early FL 21.2 saves can already have the next layout. Never choose
        # by divisibility when both sizes divide the same payload.
        if minor >= 2 and len(payload) % 60 and not len(payload) % 80:
            size = 80
    else:
        size = 32
    if len(payload) % size:
        raise _failure("unsupported_playlist_layout",
                       f"FL {version} 的 Playlist 长度 {len(payload)} 不符合已验证的 {size} 字节记录。")
    return size


def _decode_content(events: list[_RawEvent], version: str):
    channels: dict[int, _Channel] = {}
    patterns: dict[int, _Pattern] = {}
    arrangements: list[_Arrangement] = []
    channel = None
    pattern = None
    arrangement = None
    marker: list[int] | None = None
    marker_target = None
    numerator = denominator = 4
    bpm = 120.0
    main_pitch = 0
    play_cut_notes = True
    diagnostics = []
    for event in events:
        eid, value = event.event_id, event.value
        number = int.from_bytes(value, "little") if len(value) <= 4 else 0
        if eid == 156:
            bpm = number / 1000
        elif eid == 66:
            bpm = float(number)
        elif eid == 93:
            bpm += number / 1000
        elif eid == 17:
            numerator = number
        elif eid == 18:
            denominator = number
        elif eid == 30:
            play_cut_notes = bool(number)
        elif eid == 80:
            main_pitch = struct.unpack("<h", value)[0]
        elif eid == 64:
            channel = channels.setdefault(number, _Channel(number))
            pattern = None
        elif eid == 65:
            pattern = patterns.setdefault(number, _Pattern(number))
            channel = None
        elif eid == 99:
            arrangement = _Arrangement()
            arrangements.append(arrangement)
            channel = pattern = None
        elif eid == 236:
            channel = None  # Mixer effects reuse PluginID.Name / InternalName.
        elif eid == 241 and arrangement is not None:
            arrangement.name = _text(value, version) or "Arrangement"
        elif eid == 233:
            if arrangement is None:
                raise _failure("missing_arrangement", "Playlist 出现在编曲身份之前，无法可靠归属。")
            if arrangement.playlist is not None:
                raise _failure("unsupported_playlist_layout", "一个编曲含多个 Playlist 事件。")
            arrangement.playlist = value
        elif eid == 238 and arrangement is not None:
            if len(value) < 13:
                raise _failure("invalid_track_data", "Playlist 音轨状态被截断。")
            track = TrackEvent.STRUCT.parse(value)
            # iid is one-based; Playlist stores the zero-based reverse index.
            arrangement.tracks_enabled[int(track["iid"]) - 1] = bool(track["enabled"])
        elif eid == 148:
            if marker is not None and marker_target is not None:
                marker_target.meters.append(tuple(marker))
            marker = [number, 0, 0]
            marker_target = arrangement if arrangement is not None else pattern
        elif eid in (33, 34) and marker is not None:
            marker[1 if eid == 33 else 2] = number
        elif channel is not None:
            if eid in (192, 203):
                channel.name = _text(value, version)
            elif eid == 201:
                channel.plugin = _text(value, version)
            elif eid == 0:
                channel.enabled = bool(number)
            elif eid == 21:
                channel.kind = number
            elif eid == 128:
                channel.color = "#" + value[:3].hex()
            elif eid == 196:
                channel.sample_path = _text(value, version)
            elif eid == 215:
                params = ParametersEvent.STRUCT.parse(value)
                channel.arpeggiator = bool(params.get("arp.direction"))
                channel.main_pitch = params.get("keyboard.main_pitch")
                channel.add_root = params.get("keyboard.add_root")
                channel.stretching_pitch = int(params.get("stretching.pitch") or 0)
            elif eid == 219:
                channel.pitch_shift = int(LevelsEvent.STRUCT.parse(value).get("pitch_shift") or 0)
                if len(value) >= 8:
                    channel.volume = struct.unpack_from("<I", value, 4)[0]
            elif eid == 135:
                channel.root_note = number
            elif eid == 142:
                channel.fine_tune = struct.unpack("<i", value)[0]
        elif pattern is not None:
            if eid == 193:
                pattern.name = _text(value, version)
            elif eid == 224:
                if len(value) % 24:
                    raise _failure("unsupported_note_layout", f"Pattern {pattern.iid} 的音符记录不是 24 字节。")
                pattern.notes.extend(NotesEvent.STRUCT.parse(value))
            elif eid == 164:
                pattern.length = number
            elif eid == 223 and value:
                pattern.has_controllers = True
                pattern.controllers.append(value)
            elif eid == 161 and struct.unpack("<i", value)[0] == -3:
                pattern.channel_loops = True
        elif eid == 224 and value:
            # FL 25+ has unrelated header E0 data before the first real pattern.
            diagnostics.append(Diagnostic("info", "header_note_container",
                "已跳过 Pattern 身份之前的工程头部 E0 容器，不将其视为音乐音符。"))
    if marker is not None and marker_target is not None:
        marker_target.meters.append(tuple(marker))
    return channels, patterns, arrangements, bpm, numerator, denominator, play_cut_notes, main_pitch, diagnostics


def import_flp(path: str | Path, arrangement_index: int = 0) -> ProjectIR:
    source = Path(path).resolve()
    try:
        events, version, ppq, expected_channels = _read_events(source.read_bytes())
        (channels, patterns, arrangements, bpm, numerator, denominator,
         play_cut_notes, main_pitch, diagnostics) = _decode_content(events, version)
    except ImportFailure:
        raise
    except Exception as exc:
        raise _failure("flp_decode_failed", f"FLP 字段解码失败：{exc}") from exc
    if len(channels) != expected_channels:
        raise _failure("missing_channel_data",
                       f"FLP 头声明 {expected_channels} 个通道，实际读取 {len(channels)} 个，拒绝不完整导入。")
    if not arrangements or not 0 <= arrangement_index < len(arrangements):
        raise _failure("missing_arrangement", "未找到所选 Arrangement，不能把所有 Pattern 直接拼接。")
    if not (1 <= bpm <= 999 and numerator > 0 and denominator in (1, 2, 4, 8, 16, 32)):
        raise _failure("invalid_timing", "FLP 速度或拍号不在首版支持范围内。")
    arrangement = arrangements[arrangement_index]
    if arrangement.playlist is None:
        raise _failure("missing_playlist", "所选 Arrangement 没有 Playlist 数据。")
    payload = arrangement.playlist
    record_size = _playlist_record_size(version, payload)
    diagnostics.append(Diagnostic("info", "flp_layout",
        f"FL {version}：已完整读取 {len(events)} 个事件、{len(payload) // record_size} 个 Playlist 记录（{record_size} 字节）。"))
    project = ProjectIR(str(source), "flp", source.stem, ppq, bpm, numerator, denominator,
                        diagnostics=diagnostics, source_version=version,
                        arrangement_names=[arr.name for arr in arrangements],
                        arrangement_index=arrangement_index)
    automation_data = decode_automation(events, set(channels), ppq)
    diagnostics.extend(automation_data.diagnostics)
    bar_ticks = ppq * numerator * 4 / denominator
    max_track = 499 if tuple(int(n) for n in version.split(".")[:3]) >= (12, 9, 1) else 198
    used_channels: set[int] = set()
    used_patterns: set[int] = set()
    pattern_controls = {}
    zero_notes = 0
    slide_channels: set[int] = set()
    color_channels: set[int] = set()
    fine_pitch_channels: dict[int, set[int]] = {}
    muted_clips = 0
    empty_patterns: set[int] = set()
    for clip_number, start in enumerate(range(0, len(payload), record_size)):
        raw = payload[start:start + record_size]
        (position, pattern_base, item_index, length, reverse_track, group,
         marker, flags, constants, source_offset, end_offset) = _CLIP_CORE.unpack_from(raw)
        if pattern_base != 20480 or reverse_track > max_track or marker != 120:
            raise _failure("invalid_playlist_record", f"Playlist 第 {clip_number + 1} 个记录的身份或音轨无效。")
        if not arrangement.tracks_enabled.get(max_track - reverse_track, True) or flags & 0x2000:
            muted_clips += 1
            continue
        project.duration_ticks = max(project.duration_ticks, position + length)
        if not length:
            continue
        if item_index <= pattern_base:
            if item_index not in channels:
                raise _failure("missing_clip_channel", f"Playlist 引用了不存在的通道 {item_index}。")
            ch = channels[item_index]
            if ch.kind == 5:
                automations, clock_known = place_automation(
                    automation_data, ch, raw, ppq, position, length,
                    f"flp-auto-{arrangement_index}-{clip_number}")
                project.volume_automations.extend(automations)
                for automation in automations:
                    if not automation.supported:
                        diagnostics.append(Diagnostic("warning", "volume_automation_unsupported",
                            f"Automation Clip「{ch.name}」：{automation.unsupported_reason}。"))
                if not clock_known:
                    project.timing_confirmed = False
                    diagnostics.append(Diagnostic("warning", "automation_timing_unverified",
                        f"Automation Clip「{ch.name}」含未解码目标，请确认无速度自动化；如有请使用固定速度 MIDI。"))
            else:
                diagnostics.append(Diagnostic("warning", "audio_clip",
                    f"Audio Clip「{ch.name or ch.sample_path}」没有可制谱音符；声音由原曲音频保留，需要补充 MIDI。"))
            continue
        pattern_id = item_index - pattern_base
        if not 1 <= pattern_id <= 999:
            raise _failure("invalid_pattern_reference", f"Playlist 引用了无效 Pattern {pattern_id}。")
        pattern = patterns.get(pattern_id)
        if pattern is None:
            # FL doesn't serialize unnamed empty patterns. Record the implicit
            # empty identity instead of inventing notes or dropping the clip.
            empty_patterns.add(pattern_id)
            continue
        used_patterns.add(pattern_id)
        if pattern.channel_loops:
            raise _failure("unsupported_channel_loop", f"Pattern {pattern_id} 使用独立通道循环，尚不能可靠展开。")
        if record_size >= 80:
            scale = struct.unpack_from("<d", raw, 64)[0]
            if not math.isclose(scale, 1.0, abs_tol=1e-9):
                raise _failure("unsupported_pattern_stretch", f"Pattern {pattern_id} 含未验证的时间伸缩。")
        offset = 0 if source_offset == 0xFFFFFFFF else source_offset
        if offset > 0x7FFFFFFF:
            raise _failure("unsupported_pattern_offset", f"Pattern {pattern_id} 含未验证的裁切偏移。")
        # A step-sequencer trigger has no gate duration. A sixteenth-note glyph
        # expresses its trigger; the WAV is the authoritative sample decay.
        pattern_end = max((int(note["position"]) + (int(note["length"]) or max(1, ppq // 4))
                           for note in pattern.notes), default=0)
        if pattern_id not in pattern_controls:
            pattern_controls[pattern_id] = decode_pattern_controls(
                pattern.controllers, set(channels), automation_data)
        controllers, clock_known, controller_end = pattern_controls[pattern_id]
        period_end = max(pattern_end, controller_end)
        period = pattern.length or max(1, round(math.ceil(period_end / bar_ticks) * bar_ticks))
        project.volume_automations.extend(place_pattern_controls(controllers, position, length, offset,
            period, f"flp-pattern-auto-{arrangement_index}-{clip_number}",
            f"Pattern {pattern_id}: {pattern.name}"))
        first_repeat = max(0, offset // period - 1)
        final_repeat = math.ceil((offset + length) / period)
        for repeat in range(first_repeat, final_repeat):
            for source_note_index, note in enumerate(pattern.notes):
                rack_id = int(note["rack_channel"])
                if rack_id not in channels:
                    raise _failure("missing_note_channel", f"Pattern {pattern_id} 引用了不存在的 Rack 通道 {rack_id}。")
                channel = channels[rack_id]
                if not channel.enabled:
                    continue
                start_tick = int(note["position"]) + repeat * period
                duration = int(note["length"])
                if not duration:
                    duration = max(1, ppq // 4)
                end_tick = start_tick + duration
                if start_tick >= offset + length or end_tick <= offset:
                    continue
                if not play_cut_notes and start_tick < offset:
                    continue
                clipped_start = max(start_tick, offset)
                clipped_end = min(end_tick, offset + length) if play_cut_notes else end_tick
                if clipped_end <= clipped_start:
                    continue
                used_channels.add(rack_id)
                zero_notes += not int(note["length"])
                slide = bool(int(note["flags"]) & 8)
                if slide:
                    slide_channels.add(rack_id)
                midi_color = int(note["midi_channel"])
                if midi_color & 15:
                    color_channels.add(rack_id)
                fine_pitch = int(note["fine_pitch"])
                if fine_pitch != 120:
                    fine_pitch_channels.setdefault(rack_id, set()).add((fine_pitch - 120) * 10)
                project.notes.append(NoteEvent(
                    f"flp-{arrangement_index}-{clip_number}-{repeat}-{source_note_index}",
                    f"flp-channel-{rack_id}", position + clipped_start - offset,
                    clipped_end - clipped_start, int(note["key"]),
                    min(127, max(0, int(note["velocity"]))), midi_color & 15,
                    source_pattern=f"{pattern_id}:{pattern.name}", slide=slide,
                    key_release_tick=(position + end_tick - offset
                                      if int(note["length"]) > 0 and clipped_start == start_tick and clipped_end == end_tick
                                      else None)))
    for pattern_id in used_patterns:
        pattern = patterns[pattern_id]
        if pattern.has_controllers and not pattern_controls[pattern_id][1]:
            project.timing_confirmed = False
            diagnostics.append(Diagnostic("warning", "pattern_automation_unverified",
                f"Pattern {pattern_id} 含事件自动化；目标未完整解码，请确认速度固定或补充 MIDI。"))
    for automation in project.volume_automations:
        if automation.source_kind == "pattern" and not automation.supported:
            diagnostics.append(Diagnostic("warning", "volume_automation_unsupported",
                f"{automation.source_label}：{automation.unsupported_reason}。"))
    for target in [arrangement] + [patterns[p] for p in used_patterns]:
        for marker_position, num, den in target.meters:
            if marker_position >= 134217728 and num and den and (num, den) != (numerator, denominator):
                raise _failure("variable_meter", "工程含拍号变化，首版仅支持固定拍号。")
    for rack_id in channels:
        if rack_id not in used_channels:
            continue
        channel = channels[rack_id]
        track_id = f"flp-channel-{rack_id}"
        project.tracks.append(TrackInfo(track_id, channel.name or channel.plugin or f"Channel {rack_id + 1}",
                                        channel.color, channel.plugin,
                                        mixer_insert=automation_data.channel_inserts.get(rack_id)))
        pitch_settings = []
        if main_pitch and channel.main_pitch is not False:
            status = "作用于该通道" if channel.main_pitch else "是否作用于该通道未验证"
            pitch_settings.append(f"主音高 {main_pitch:+d} 音分（{status}）")
        if channel.pitch_shift:
            pitch_settings.append(f"通道音高 {channel.pitch_shift:+d} 音分")
        if channel.fine_tune:
            pitch_settings.append(f"键盘微调 {channel.fine_tune:+d} 音分")
        if channel.stretching_pitch:
            pitch_settings.append(f"采样伸缩音高 {channel.stretching_pitch:+d} 音分")
        if channel.root_note != 60:
            pitch_settings.append(f"根音 {channel.root_note}（默认 60）")
        if channel.add_root:
            pitch_settings.append("键盘 Add to key 已开启")
        if rack_id in fine_pitch_channels:
            offsets = ", ".join(f"{cents:+d}" for cents in sorted(fine_pitch_channels[rack_id]))
            pitch_settings.append(f"音符微调 {offsets} 音分")
        if pitch_settings:
            diagnostics.append(Diagnostic("warning", "source_pitch_unverified",
                f"「{channel.name}」含 {'；'.join(pitch_settings)}。谱面保留钢琴卷帘音高，"
                "实际发声音高需核对；整半音可确认后设置分谱移调，微调／插件音高请补充 MIDI。", track_id))
        if channel.kind == 3:
            diagnostics.append(Diagnostic("warning", "layer_channel",
                f"「{channel.name}」是 Layer：谱面保留触发音符，子通道音高映射需补充 MIDI。", track_id))
        if channel.arpeggiator:
            diagnostics.append(Diagnostic("warning", "arpeggiator",
                f"「{channel.name}」启用通道琶音器；谱面保留输入音符，实际琶音请补充 MIDI。", track_id))
        if rack_id in slide_channels:
            diagnostics.append(Diagnostic("warning", "slide_notes",
                f"「{channel.name}」含 Slide；谱面保留滑音目标音高，听感以原曲音频为准。", track_id))
        if rack_id in color_channels:
            diagnostics.append(Diagnostic("info", "note_color_channel",
                "已保留音符颜色编号；颜色不自动等同 MIDI 路由，可确认后按 MIDI 通道拆分。", track_id))
        if channel.plugin.casefold() in ("fruity wrapper", "patcher"):
            diagnostics.append(Diagnostic("info", "plugin_generated_notes",
                "插件内部琶音、音色及 Keyswitch 无法仅从名称可靠识别，请检查分谱或补充 MIDI。", track_id))
    if zero_notes:
        diagnostics.append(Diagnostic("warning", "step_trigger_duration",
            f"{zero_notes} 个零长度步进触发音符以十六分音符记谱；采样尾音由原曲音频保留。"))
    if main_pitch:
        diagnostics.append(Diagnostic("info", "project_main_pitch",
            f"工程保存的主音高为 {main_pitch:+d} 音分；已逐通道检查 keyboard.main_pitch，"
            "禁用主音高响应的通道不受此设置影响。"))
    if muted_clips:
        diagnostics.append(Diagnostic("info", "muted_clips", f"已跳过 {muted_clips} 个静音 Clip／禁用 Playlist 音轨。"))
    if empty_patterns:
        diagnostics.append(Diagnostic("info", "implicit_empty_patterns",
            "Playlist 中未单独序列化的空 Pattern：" + ", ".join(map(str, sorted(empty_patterns)))))
    project.volume_routes = build_routes(automation_data, channels,
        {rack_id: f"flp-channel-{rack_id}" for rack_id in used_channels}, project.volume_automations)
    for route in project.volume_routes:
        if not route.supported and project.volume_automations:
            diagnostics.append(Diagnostic("warning", "volume_route_unsupported",
                route.unsupported_reason + "。", route.track_id))
    project.notes.sort(key=lambda note: (note.start_tick, note.track_id, note.pitch, note.note_id))
    if not project.notes:
        raise ImportFailure("所选 Arrangement 没有可制谱音符；请放置含音符的 Pattern 或补充 MIDI。", diagnostics + [
            Diagnostic("error", "no_arranged_notes", "完整读取 Playlist 后没有找到音乐音符。")])
    return project
