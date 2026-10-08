"""Standard MIDI import. Preserve source ticks, channels, and sustain lengths."""

from collections import defaultdict, deque
from pathlib import Path

import mido

from stavellum.domain.models import Diagnostic, NoteEvent, ProjectIR, TrackInfo

from .errors import ImportFailure

# Unambiguous GM patches are hints; user confirmation still controls icons.
GM_NAMES = {
    0: "Piano", 1: "Piano", 2: "Piano", 3: "Piano", 6: "Harpsichord",
    8: "Celesta", 9: "Glockenspiel", 10: "Music box", 11: "Vibraphone",
    12: "Marimba", 13: "Xylophone", 14: "Tubular bells", 16: "Organ",
    24: "Guitar", 25: "Guitar", 26: "Guitar", 27: "Guitar", 28: "Guitar",
    29: "Guitar", 30: "Guitar", 31: "Guitar", 32: "Bass", 33: "Bass",
    34: "Bass", 35: "Bass", 40: "Violin", 41: "Viola", 42: "Cello",
    43: "Double bass", 46: "Harp", 47: "Timpani", 56: "Trumpet",
    57: "Trombone", 58: "Tuba", 60: "Horn", 64: "Saxophone",
    65: "Saxophone", 66: "Saxophone", 67: "Saxophone", 68: "Oboe",
    69: "English horn", 70: "Bassoon", 71: "Clarinet", 72: "Piccolo",
    73: "Flute", 74: "Recorder",
}


def _fixed_timing(midi: mido.MidiFile) -> tuple[float, int, int]:
    tempos: list[tuple[int, int]] = []
    meters: list[tuple[int, tuple[int, int]]] = []
    for track in midi.tracks:
        tick = 0
        for message in track:
            tick += message.time
            if message.type == "set_tempo":
                tempos.append((tick, message.tempo))
            elif message.type == "time_signature":
                meters.append((tick, (message.numerator, message.denominator)))
    initial_tempos = {value for tick, value in tempos if tick == 0}
    initial_meters = {value for tick, value in meters if tick == 0}
    if len(initial_tempos) > 1 or len(initial_meters) > 1:
        raise ImportFailure("MIDI 起点含相互冲突的速度或拍号。", [
            Diagnostic("error", "conflicting_timing", "MIDI 起点的速度或拍号不一致。")
        ])
    tempo = next(iter(initial_tempos), 500_000)
    meter = next(iter(initial_meters), (4, 4))
    if tempo <= 0:
        raise ImportFailure("MIDI 速度必须为正数。")
    if any(value != tempo for _, value in tempos):
        raise ImportFailure("首版仅支持固定速度；请提供固定速度 MIDI。", [
            Diagnostic("error", "variable_tempo", "检测到 MIDI 速度变化，未导入。")
        ])
    if any(value != meter for _, value in meters):
        raise ImportFailure("首版仅支持固定拍号；请提供固定拍号 MIDI。", [
            Diagnostic("error", "variable_meter", "检测到 MIDI 拍号变化，未导入。")
        ])
    if meter[0] <= 0 or meter[1] not in (1, 2, 4, 8, 16, 32):
        raise ImportFailure("MIDI 拍号不在首版支持范围内。")
    return float(mido.tempo2bpm(tempo)), *meter


def import_midi(path: str | Path) -> ProjectIR:
    source = Path(path).resolve()
    try:
        midi = mido.MidiFile(source)
    except (OSError, ValueError, EOFError) as exc:
        raise ImportFailure(f"无法读取 MIDI：{exc}") from exc
    if midi.type == 2:
        raise ImportFailure("MIDI Type 2 含独立时间轴；请导出 Type 0 或 Type 1。")
    if midi.ticks_per_beat <= 0:
        raise ImportFailure("首版不支持 SMPTE 时间码 MIDI，请使用 PPQ 时间基准。")
    bpm, numerator, denominator = _fixed_timing(midi)
    project = ProjectIR(str(source), "midi", source.stem, midi.ticks_per_beat,
                        bpm, numerator, denominator, arrangement_names=[source.stem])
    sequence = 0
    bend_tracks: set[str] = set()
    names: dict[int, str] = {}
    timeline = []
    for track_index, track in enumerate(midi.tracks):
        name = next((m.name for m in track if m.type == "track_name"), "")
        instrument_name = next((m.name for m in track if m.type == "instrument_name"), "")
        names[track_index] = name or instrument_name
        tick = 0
        for message_index, message in enumerate(track):
            tick += message.time
            timeline.append((tick, track_index, message_index, message))
        project.duration_ticks = max(project.duration_ticks, tick)
    # Channel state is shared by every SMF track. CC64/program changes in a
    # dedicated controller track must affect notes in the musical track too.
    timeline.sort(key=lambda item: item[:3])
    programs = defaultdict(int)
    active: dict[tuple[int, int], deque[NoteEvent]] = defaultdict(deque)
    released: dict[int, list[NoteEvent]] = defaultdict(list)
    pedals = defaultdict(bool)
    tracks: dict[tuple[int, int, int], TrackInfo] = {}

    def finish(note: NoteEvent, end: int) -> None:
        note.duration_tick = max(1, end - note.start_tick)
        if end == note.start_tick:
            project.diagnostics.append(Diagnostic(
                "warning", "zero_length_midi", "零长度 MIDI 音符保留为一个源时钟刻度。", note.track_id))
        project.notes.append(note)

    for tick, track_index, _, message in timeline:
        if not hasattr(message, "channel"):
            continue
        channel = message.channel
        if message.type == "program_change":
            programs[channel] = message.program
        elif message.type == "note_on" and message.velocity > 0:
            program = programs[channel]
            identity = (track_index, channel, program)
            if identity not in tracks:
                identifier = f"midi-{track_index}-{channel}-{program}"
                patch = "Percussion" if channel == 9 else GM_NAMES.get(program, f"GM {program + 1}")
                display = names[track_index] or patch
                tracks[identity] = TrackInfo(identifier, display,
                    plugin=f"General MIDI: {patch}", midi_channel=channel)
            sequence += 1
            note = NoteEvent(f"midi-note-{sequence}", tracks[identity].track_id,
                             tick, 0, message.note, message.velocity, channel)
            active[(channel, message.note)].append(note)
        elif message.type in ("note_off", "note_on"):
            notes = active[(channel, message.note)]
            if notes:
                note = notes.popleft()
                # Retain the real key release even while CC64 keeps sounding.
                # Controller-forced endings and file-end repairs stay unknown.
                note.key_release_tick = tick
                if pedals[channel]:
                    released[channel].append(note)
                else:
                    finish(note, tick)
            else:
                project.diagnostics.append(Diagnostic("warning", "orphan_note_off",
                    f"音轨 {track_index + 1} 含未配对的 Note Off（{message.note}）。"))
        elif message.type == "control_change":
            if message.control == 64:
                pedals[channel] = message.value >= 64
                if not pedals[channel]:
                    for note in released.pop(channel, []):
                        finish(note, tick)
            elif message.control in (120, 123):
                for key in list(active):
                    if key[0] == channel:
                        for note in active.pop(key):
                            if message.control == 123 and pedals[channel]:
                                released[channel].append(note)
                            else:
                                finish(note, tick)
                if message.control == 120:
                    for note in released.pop(channel, []):
                        finish(note, tick)
            elif message.control == 121:
                pedals[channel] = False
                for note in released.pop(channel, []):
                    finish(note, tick)
        elif message.type == "pitchwheel":
            bend_tracks.add(f"音轨 {track_index + 1} / MIDI {channel + 1}")
    dangling = [note for queue in active.values() for note in queue]
    dangling += [note for queue in released.values() for note in queue]
    for note in dangling:
        finish(note, project.duration_ticks)
    if dangling:
        project.diagnostics.append(Diagnostic("warning", "unclosed_notes",
            f"{len(dangling)} 个未结束音符在 MIDI 文件末尾结束。"))
    track_counts = defaultdict(int)
    for index, _, _ in tracks:
        track_counts[index] += 1
    for (index, channel, program), track_info in sorted(tracks.items()):
        if track_counts[index] > 1:
            track_info.name += f" · MIDI {channel + 1} / {GM_NAMES.get(program, f'GM {program + 1}')}"
        project.tracks.append(track_info)
    for label in sorted(bend_tracks):
        project.diagnostics.append(Diagnostic("warning", "pitch_bend",
            f"{label} 含弯音；谱面保留基础音高，弯音听感由原曲音频保留。"))
    project.notes.sort(key=lambda note: (note.start_tick, note.track_id, note.pitch, note.note_id))
    if not project.notes:
        raise ImportFailure("MIDI 中没有可制谱音符。")
    return project
