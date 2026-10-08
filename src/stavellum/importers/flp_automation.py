"""Native FL volume controls, preserved automation curves and audible routing.

The current REC constants and control law are documented by Image-Line's
bundled Shared/Python/Lib/midi.py and utils.py. Older MixerParams records use
the PyFLP 2.2.1 layout. Unknown mappings are retained with an explicit reason.
"""

import math
import struct
from dataclasses import dataclass, field

from stavellum.domain.models import AutomationPoint, Diagnostic, VolumeAutomation, VolumeRoute


@dataclass(slots=True)
class AutomationSource:
    points: list[AutomationPoint] = field(default_factory=list)
    minimum: float = 0.0
    maximum: float = 1.0
    reason: str = ""
    raw_payload: str = ""


@dataclass(slots=True)
class PatternControl:
    binding: Binding
    points: list[AutomationPoint]
    raw_payload: str


@dataclass(slots=True)
class Binding:
    target_id: str
    target_kind: str
    value_scale: float
    reason: str = ""


@dataclass(slots=True)
class AutomationData:
    sources: dict[int, AutomationSource] = field(default_factory=dict)
    bindings: dict[int, list[Binding]] = field(default_factory=dict)
    unknown_bindings: set[int] = field(default_factory=set)
    unidentified_binding: bool = False
    initial_values: dict[str, float] = field(default_factory=dict)
    channel_inserts: dict[int, int] = field(default_factory=dict)
    mixer_routes: dict[int, list[int]] = field(default_factory=dict)
    mixer_enabled: dict[int, bool] = field(default_factory=dict)
    send_values: dict[tuple[int, int], int] = field(default_factory=dict)
    dynamic_routes: set[int] = field(default_factory=set)
    routing_bindings: dict[int, set[int]] = field(default_factory=dict)
    channel_routing_bindings: dict[int, set[int]] = field(default_factory=dict)
    dynamic_channels: set[int] = field(default_factory=set)
    diagnostics: list[Diagnostic] = field(default_factory=list)


def volume_target(rec: int, channel_ids: set[int]) -> Binding | None:
    """Decode only native volume REC ids; plugin parameter ids are excluded."""
    high, low = rec >> 16, rec & 0xFFFF
    if low == 0 and high in channel_ids:
        return Binding(f"channel:{high}", "channel_volume", 1.0)
    if low == 17 and high in channel_ids:
        return Binding(f"channel-offset:{high}", "channel_volume", 1.0,
                       "通道音量偏移控制的增益映射尚未校准")
    if rec == 0x40000000:
        return Binding("global:main-volume", "mixer_volume", 1.0,
                       "主面板总音量控制的增益映射尚未校准")
    if low == 0x1FC0 and 0x7000 <= high < 0xF000 and not high & 0x3F:
        return Binding(f"mixer:{(high - 0x7000) >> 6}", "mixer_volume", 1.25)
    if low == 0x20C0 and 0x2000 <= high < 0x4000 and not high & 0x3F:
        return Binding(f"mixer:{(high - 0x2000) >> 6}", "mixer_volume", 1.25)
    return None


def _send_target(rec: int) -> tuple[int, int] | None:
    high, low = rec >> 16, rec & 0xFFFF
    if 0x7000 <= high < 0xF000 and not high & 0x3F and 0x2000 <= low < 0x2200:
        return (high - 0x7000) >> 6, low - 0x2000
    if 0x2000 <= high < 0x4000 and not high & 0x3F and 0x2040 <= low < 0x20C0:
        return (high - 0x2000) >> 6, low - 0x2040
    return None


def decode_points(payload: bytes, ppq: int) -> AutomationSource:
    result = AutomationSource(raw_payload=payload.hex())
    if len(payload) < 21:
        result.reason = "自动化点容器被截断"
        return result
    count = struct.unpack_from("<I", payload, 17)[0]
    if not count or count > (len(payload) - 21) // 24:
        result.reason = "自动化点数量或长度无效"
        return result
    position = 0.0
    for index in range(count):
        delta, value, tension, metadata = struct.unpack_from("<ddfI", payload, 21 + 24 * index)
        position += delta * ppq
        if (not all(math.isfinite(number) for number in (delta, value, tension, position))
                or delta < 0 or not 0 <= value <= 1):
            result.points.clear()
            result.reason = "自动化点位置、取值或张力无效"
            return result
        result.points.append(AutomationPoint(position, value, tension, metadata))
    # E234 byte 8 is the LFO switch, not the amount at offset 4. A nonzero
    # amount is also saved when the LFO is disabled.
    if payload[8]:
        result.reason = "自动化启用 LFO，尚不能转为确定的音量曲线"
    return result


def decode_automation(events, channel_ids: set[int], ppq: int) -> AutomationData:
    result = AutomationData()
    channel_id = None
    insert = -1
    initialized = {}
    for event in events:
        eid, value = event.event_id, event.value
        if eid == 64:
            channel_id = int.from_bytes(value, "little")
        elif eid in (65, 99):
            channel_id = None
        elif eid == 236:
            channel_id = None
            insert += 1
            if len(value) >= 8:
                result.mixer_enabled[insert] = bool(struct.unpack_from("<I", value, 4)[0] & 8)
        elif eid == 235 and insert >= 0:
            result.mixer_routes[insert] = [destination for destination, enabled in enumerate(value) if enabled]
        elif eid == 225:
            if len(value) % 12:
                result.diagnostics.append(Diagnostic("warning", "volume_mixer_params",
                    "Mixer 控制记录长度未验证，相关路径不会推断音量。"))
                continue
            for offset in range(0, len(value), 12):
                rec, raw = struct.unpack_from("<Ii", value, offset + 4)
                target = volume_target(rec, channel_ids)
                if target is not None and target.target_kind == "mixer_volume":
                    result.initial_values[target.target_id] = raw / 12800
                elif (send := _send_target(rec)) is not None:
                    result.send_values[send] = raw
                elif rec >> 16 < 0x2000:
                    # Legacy MixerParams packs a byte parameter + channel_data,
                    # rather than a full REC id. Slot parameters are unrelated.
                    parameter = rec & 0xFF
                    old_insert = (rec >> 22) & 0x7F
                    if parameter == 192:
                        result.initial_values[f"mixer:{old_insert}"] = raw / 12800
                    elif 64 <= parameter < 192:
                        result.send_values[old_insert, parameter - 64] = raw
        elif eid == 216:
            if len(value) % 12:
                result.diagnostics.append(Diagnostic("warning", "volume_initial_controls",
                    "初始控制记录长度未验证，无法恢复全部音量初值。"))
                continue
            for offset in range(0, len(value), 12):
                rec, raw = struct.unpack_from("<Ii", value, offset + 4)
                target = volume_target(rec, channel_ids)
                if target is not None:
                    initialized[target.target_id] = raw / 12800
        elif eid == 227:
            if len(value) < 12:
                if len(value) >= 4:
                    result.unknown_bindings.add(struct.unpack_from("<H", value, 2)[0])
                else:
                    result.unidentified_binding = True
                result.diagnostics.append(Diagnostic("warning", "automation_binding",
                    "自动化链接记录被截断，无法确认目标。"))
                continue
            source_id = struct.unpack_from("<H", value, 2)[0]
            rec = struct.unpack_from("<I", value, 8)[0]
            target = volume_target(rec, channel_ids)
            if target is None:
                result.unknown_bindings.add(source_id)
                if (send := _send_target(rec)) is not None:
                    result.routing_bindings.setdefault(source_id, set()).add(send[0])
                elif rec & 0xFFFF == 8 and rec >> 16 in channel_ids:
                    result.channel_routing_bindings.setdefault(source_id, set()).add(rec >> 16)
                continue
            if len(value) != 20 or value[4:8] != bytes(4) or value[12:] != bytes.fromhex("08000000d5010000"):
                target.reason = "自动化链接使用未验证的控制器或映射公式"
            if target not in result.bindings.setdefault(source_id, []):
                result.bindings[source_id].append(target)
        elif channel_id is not None:
            if eid == 104:
                result.channel_inserts[channel_id] = int.from_bytes(value, "little")
            elif eid == 22 and channel_id not in result.channel_inserts:
                result.channel_inserts[channel_id] = int.from_bytes(value, "little", signed=True)
            elif eid == 234:
                previous = result.sources.get(channel_id)
                result.sources[channel_id] = decode_points(value, ppq)
                if previous:
                    result.sources[channel_id].minimum = previous.minimum
                    result.sources[channel_id].maximum = previous.maximum
            elif eid == 219 and len(value) >= 8:
                minimum, maximum = struct.unpack_from("<iI", value)
                source = result.sources.setdefault(channel_id, AutomationSource())
                source.minimum, source.maximum = minimum / 12800, maximum / 12800
    result.initial_values.update(initialized)
    return result


def place_automation(data: AutomationData, channel, raw: bytes, ppq: int,
                     position: int, length: int, identity: str) -> tuple[list[VolumeAutomation], bool]:
    """Return placed volume controls and whether all possible clock targets are known."""
    bindings = data.bindings.get(channel.iid, [])
    clock_known = (bool(bindings) and channel.iid not in data.unknown_bindings
                   and not data.unidentified_binding)
    if not channel.enabled:
        return [], True
    data.dynamic_routes.update(data.routing_bindings.get(channel.iid, set()))
    data.dynamic_channels.update(data.channel_routing_bindings.get(channel.iid, set()))
    source = data.sources.get(channel.iid, AutomationSource(reason="缺少自动化点数据"))
    offset, end_offset = struct.unpack_from("<ff", raw, 24)
    reason = source.reason
    if not all(math.isfinite(number) and (number == -1 or number >= 0) for number in (offset, end_offset)):
        reason = "自动化 Clip 裁切偏移未验证"
    if len(raw) >= 80 and not math.isclose(struct.unpack_from("<d", raw, 64)[0], 1, abs_tol=1e-9):
        reason = "自动化 Clip 时间伸缩尚未验证"
    if not 0 <= source.minimum <= 1 or not 0 <= source.maximum <= 1:
        reason = "自动化 Min/Max 超出已验证的控制范围"
    automations = []
    safe_offset = offset if math.isfinite(offset) and offset >= 0 else 0.0
    minimum = source.minimum if 0 <= source.minimum <= 1 else 0.0
    maximum = source.maximum if 0 <= source.maximum <= 1 else 1.0
    for index, binding in enumerate(bindings):
        unsupported = binding.reason or reason
        automations.append(VolumeAutomation(
            f"{identity}-{index}", binding.target_id, binding.target_kind, list(source.points),
            position, position + length, safe_offset * ppq,
            minimum, maximum, binding.value_scale, channel.name,
            not unsupported, unsupported, raw_payload=source.raw_payload))
    return automations, clock_known


def decode_pattern_controls(payloads: list[bytes], channel_ids: set[int],
                            data: AutomationData | None = None):
    """FL-native E223 records are absolute tick, REC target, integer value.

    PyFLP's Float32 value declaration misreads the native control integers. Keep
    the exact step samples; any inference of a densely recorded musical ramp
    belongs to the dynamics layer, rather than changing their playback curve.
    """
    controls = {}
    bindings = {}
    clock_known = True
    end = 0
    for payload in payloads:
        if len(payload) % 12:
            clock_known = False
            continue
        for position, rec, raw in struct.iter_unpack("<III", payload):
            end = max(end, position + 1)
            binding = volume_target(rec, channel_ids)
            if binding is None:
                clock_known = False
                if data is not None:
                    if (send := _send_target(rec)) is not None:
                        data.dynamic_routes.add(send[0])
                    elif rec & 0xFFFF == 8 and rec >> 16 in channel_ids:
                        data.dynamic_channels.add(rec >> 16)
                continue
            native_maximum = 12800 * binding.value_scale
            point = AutomationPoint(position, raw / native_maximum, metadata=2)
            if not 0 <= point.value <= 1:
                binding.reason = "Pattern 音量事件超出已校准的原生整数范围"
            previous = bindings.get(binding.target_id)
            if previous is not None and previous.reason:
                binding.reason = previous.reason
            bindings[binding.target_id] = binding
            # At the same tick, retain the last native event for that control.
            controls.setdefault(binding.target_id, {})[position] = point
    decoded = []
    for target, points in controls.items():
        binding = bindings[target]
        values = [point for _, point in sorted(points.items())]
        if any(not 0 <= point.value <= 1 for point in values):
            values = []
        decoded.append(PatternControl(binding, values, "".join(payload.hex() for payload in payloads)))
    return decoded, clock_known, end


def place_pattern_controls(controls, position: int, length: int, offset: int,
                           period: int, identity: str, label: str) -> list[VolumeAutomation]:
    """Crop and repeat samples without advancing the first controller event."""
    result = []
    first_repeat = max(0, offset // period)
    final_repeat = math.ceil((offset + length) / period)
    for repeat in range(first_repeat, final_repeat):
        repeat_start = repeat * period
        repeat_end = repeat_start + period
        for target_index, control in enumerate(controls):
            binding, points = control.binding, control.points
            first_event = points[0].tick + repeat_start if points else repeat_start
            left = max(offset, repeat_start, first_event)
            right = min(offset + length, repeat_end)
            if left >= right:
                continue
            result.append(VolumeAutomation(
                f"{identity}-{repeat}-{target_index}", binding.target_id, binding.target_kind,
                list(points), position + left - offset, position + right - offset,
                left - repeat_start, value_scale=binding.value_scale,
                source_label=label, supported=not binding.reason,
                unsupported_reason=binding.reason, source_kind="pattern", raw_payload=control.raw_payload))
    return result


def build_routes(data: AutomationData, channels, track_ids: dict[int, str],
                 automations: list[VolumeAutomation]) -> list[VolumeRoute]:
    """Factor common gain controls out of all audible paths to Mixer Master.

    A static parallel branch does not prevent identifying its shared multiplier.
    An automated non-common branch cannot be composed without its signal mix.
    """
    active_controls = {automation.target_id for automation in automations}
    routes = []

    def paths(current: int, visited: frozenset[int]) -> list[list[int]]:
        if current in visited or current not in data.mixer_enabled:
            raise ValueError("Mixer 路由含循环或缺少插入轨状态")
        if not data.mixer_enabled[current]:
            return []
        if data.initial_values.get(f"mixer:{current}") == 0 and f"mixer:{current}" not in active_controls:
            return []
        if current in data.dynamic_routes:
            raise ValueError("Mixer Send 路由含未支持的自动化")
        if current == 0:
            return [[0]]
        destinations = [destination for destination in data.mixer_routes.get(current, [])
                        if data.send_values.get((current, destination), 12800) > 0]
        if not destinations:
            raise ValueError("Mixer 路径没有可确认的 Master 输出")
        found = []
        for destination in destinations:
            found.extend([[current] + suffix for suffix in paths(destination, visited | {current})])
            if len(found) > 512:
                raise ValueError("Mixer 路径数量超出可验证范围")
        return found

    for channel_id, track_id in track_ids.items():
        channel = channels[channel_id]
        channel_control = f"channel:{channel_id}"
        raw = getattr(channel, "volume", 10000)
        initial = {channel_control: data.initial_values.get(channel_control, raw / 12800)}
        controls = [channel_control]
        reason = ""
        insert = data.channel_inserts.get(channel_id)
        try:
            if not 0 <= initial[channel_control] <= 1.25:
                initial[channel_control] = 0.0
                raise ValueError("通道初始音量超出已验证的控制范围")
            if f"channel-offset:{channel_id}" in active_controls:
                raise ValueError("该通道同时有未校准的音量偏移自动化，无法确认合成渐变")
            if "global:main-volume" in active_controls:
                raise ValueError("主面板总音量含未校准自动化，无法确认合成渐变")
            if channel_id in data.dynamic_channels:
                raise ValueError("通道的 Mixer 输出含未支持的路由自动化")
            if insert is None or insert < 0:
                raise ValueError("通道的 Mixer 输出未确认")
            found = paths(insert, frozenset())
            if not found:
                raise ValueError("Mixer 输出路径的静态音量为零或已静音")
            common = set(found[0]).intersection(*found[1:])
            branch = set().union(*map(set, found)) - common
            if any(f"mixer:{number}" in active_controls for number in branch):
                raise ValueError("音量自动化位于非公共 Mixer 分支，无法推断混合后的渐变")
            for number in found[0]:
                if number not in common:
                    continue
                target = f"mixer:{number}"
                if target not in data.initial_values:
                    raise ValueError("Mixer 初始音量未确认")
                if not 0 <= data.initial_values[target] <= 1.25:
                    raise ValueError("Mixer 初始音量超出已验证的控制范围")
                controls.append(target)
                initial[target] = data.initial_values[target]
        except ValueError as exc:
            reason = str(exc)
        routes.append(VolumeRoute(track_id, controls, initial, not reason, reason))
    return routes
