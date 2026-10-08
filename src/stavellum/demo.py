"""Original three-instrument fixture with technique changes, rests and held notes."""

from __future__ import annotations

import math
import wave
from pathlib import Path

import mido
import numpy as np

from stavellum.domain.mapping import suggest_mappings
from stavellum.domain.models import (
    Metadata,
    NoteEvent,
    ProjectDocument,
    ProjectIR,
    TrackInfo,
    save_document,
)


def create_demo_document(output_dir: str | Path | None = None, bars: int = 16, tail_seconds: float = 1.25) -> ProjectDocument:
    if bars < 4 or bars > 1000:
        raise ValueError("示例长度必须为 4 到 1000 小节。")
    directory = Path(output_dir or "artifacts/demo").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    project = ProjectIR(str(directory / "demo.mid"), "midi", "Stavellum Demo", ppq=480, bpm=120, tracks=[
        TrackInfo("violin-arco", "Violin I arco", "#c0c0c0", midi_channel=0),
        TrackInfo("violin-pizz", "Violin I Pizz", "#c0c0c0", midi_channel=1),
        TrackInfo("cello", "Cello", "#b0b0b0", midi_channel=2),
        TrackInfo("bell", "Bell", "#dedede", midi_channel=3),
    ], duration_ticks=bars * 4 * 480)

    def add(track: str, beat: float, length: float, pitch: int, velocity: int) -> None:
        channel = {"violin-arco": 0, "violin-pizz": 1, "cello": 2, "bell": 3}[track]
        project.notes.append(NoteEvent(f"demo-{len(project.notes)}", track, round(beat * 480), max(1, round(length * 480)), pitch, velocity, channel))

    motif = [(0, .75, 72, 108), (.75, .25, 74, 75), (1, .5, 75, 105), (1.75, .75, 77, 95), (2.5, .25, 79, 82), (2.75, .75, 80, 112), (3.5, .5, 79, 76)]
    for bar in range(bars):
        position = bar % 16
        base = bar * 4
        if position in (0, 1, 2, 3, 12, 13, 14, 15):
            track = "violin-pizz" if position in (2, 3, 14, 15) else "violin-arco"
            if position == 13:
                for step, pitch in enumerate((72, 75, 79)):
                    add(track, base + step / 3, 1 / 3, pitch, 96)
                add(track, base + 1, 3, 80, 110)
            else:
                for start, length, pitch, velocity in motif:
                    add(track, base + start, length, pitch - (2 if position % 2 else 0), velocity)
        if position in (0, 1, 2, 3, 4, 5, 15):
            if position == 0:
                add("cello", base, min(10, (bars - bar) * 4), 48, 78)
            elif position not in (1, 2):
                add("cello", base, 2, 43 if position % 2 else 46, 70)
                add("cello", base + 2, 2, 48 if position % 2 else 50, 82)
        if position in (2, 3, 10, 11, 12, 13):
            add("bell", base, 2, 55, 68)
            add("bell", base + 2, 2, 58 if position % 2 else 60, 76)
    project.notes.sort(key=lambda n: (n.start_tick, n.track_id, n.pitch))
    mappings = suggest_mappings(project)
    for mapping in mappings:
        mapping.key_signature = -3
        if mapping.instrument in ("cello", "bell"):
            mapping.clef = "bass"
    document = ProjectDocument(project, mappings, str(directory / "demo.wav"), metadata=Metadata("Stavellum", "FL Studio 工程 → 滚动五线谱 · 逐帧渲染演示", "Original demo", ""))
    _write_midi(project, directory / "demo.mid")
    _write_audio(project, directory / "demo.wav", tail_seconds)
    save_document(document, directory / "demo.stproj")
    return document


def _write_midi(project: ProjectIR, target: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=project.ppq)
    conductor = mido.MidiTrack()
    conductor.extend([mido.MetaMessage("track_name", name="Conductor", time=0), mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(project.bpm), time=0), mido.MetaMessage("time_signature", numerator=project.numerator, denominator=project.denominator, time=0), mido.MetaMessage("end_of_track", time=project.end_tick)])
    midi.tracks.append(conductor)
    programs = {"violin-arco": 40, "violin-pizz": 45, "cello": 42, "bell": 14}
    for track in project.tracks:
        output = mido.MidiTrack()
        output.append(mido.MetaMessage("track_name", name=track.name))
        output.append(mido.Message("program_change", channel=track.midi_channel or 0, program=programs[track.track_id]))
        events = []
        for note in project.notes:
            if note.track_id == track.track_id:
                events.extend([(note.start_tick, 1, mido.Message("note_on", channel=note.midi_channel, note=note.pitch, velocity=note.velocity)), (note.end_tick, 0, mido.Message("note_off", channel=note.midi_channel, note=note.pitch, velocity=0))])
        previous = 0
        for tick, _, event in sorted(events, key=lambda event: (event[0], event[1])):
            event.time = tick - previous
            output.append(event)
            previous = tick
        output.append(mido.MetaMessage("end_of_track", time=max(0, project.end_tick - previous)))
        midi.tracks.append(output)
    midi.save(str(target))


def _write_audio(project: ProjectIR, target: Path, tail_seconds: float) -> None:
    sample_rate = 48000
    duration = project.duration_seconds + max(0.0, tail_seconds)
    sample_count = math.ceil(duration * sample_rate)
    # Write bounded blocks; even a five-minute fixture does not need a song-sized array.
    notes = [(n.start_tick / project.ppq * 60 / project.bpm, n.end_tick / project.ppq * 60 / project.bpm, n) for n in project.notes]
    with wave.open(str(target), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        for first in range(0, sample_count, sample_rate):
            last = min(sample_count, first + sample_rate)
            times = np.arange(first, last, dtype=np.float64) / sample_rate
            sound = np.zeros(last - first, dtype=np.float64)
            for start, end, note in notes:
                release = 0.45 if note.track_id == "bell" else 0.09
                if end + release <= times[0] or start > times[-1]:
                    continue
                local = times - start
                envelope = np.clip(local / .008, 0, 1) * np.clip((end + release - times) / release, 0, 1)
                frequency = 440 * 2 ** ((note.pitch - 69) / 12)
                phase = local * (2 * np.pi * frequency)
                tone = np.sin(phase) * .7 + np.sin(phase * 2) * .2 + np.sin(phase * 3) * .1
                if "pizz" in note.track_id:
                    envelope *= np.exp(-np.maximum(local, 0) * 4)
                if note.track_id == "bell":
                    envelope *= np.exp(-np.maximum(local, 0) * 1.5)
                sound += tone * envelope * (note.velocity / 127) * .17
            # Quiet beat clicks make sync independently measurable.
            beat_seconds = 60 / project.bpm
            click_age = np.mod(times, beat_seconds)
            clicks = np.sin(click_age * 2 * np.pi * 2000) * np.exp(-click_age * 200) * (click_age < .025)
            sound += clicks * .045 * (times < project.duration_seconds)
            output.writeframesraw((np.clip(sound, -.95, .95) * 32767).astype("<i2").tobytes())
