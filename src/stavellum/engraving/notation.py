"""Quantize copies of source notes, engrave a common score, and export printable parts.

All offsets are quarter-note beats.  The source performance remains untouched.  A
continuous score is engraved once; the video renderer crops that same SVG for each
instrument, so independently engraved parts can never drift horizontally.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
import zlib
from collections import defaultdict
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path

import verovio
from music21 import (
    articulations,
    chord,
    clef,
    dynamics,
    expressions,
    instrument,
    key,
    layout,
    metadata,
    meter,
    musicxml,
    note,
    percussion,
    pitch,
    spanner,
    stream,
    tempo,
)

from stavellum.domain.models import Diagnostic, NoteEvent, PartMapping, ProjectDocument
from stavellum.domain.progress import ProgressReporter
from stavellum.graphics.musicfont import MUSIC_FALLBACK, MUSIC_FAMILY

from .dynamics import DynamicSpan, infer_dynamics
from .inference import recognize_notes
from .ottava import OctaveNote, OctaveSpan, infer_ottavas


@dataclass(frozen=True, slots=True)
class NotationAnchor:
    beat: float
    element_id: str
    kind: str = "note"


@dataclass(frozen=True, slots=True)
class QuantizedEvent:
    source_id: str
    part_id: str
    start_beat: float
    end_beat: float
    pitch: int
    velocity: int
    articulation: str = ""
    staccato: bool = False
    grace_main_id: str = ""


@dataclass(slots=True)
class NotationResult:
    display_svg: str
    score: stream.Score
    diagnostics: list[Diagnostic]
    staff_part_ids: list[str]
    musicxml: str
    anchors: list[NotationAnchor] = field(default_factory=list)
    quantized_events: list[QuantizedEvent] = field(default_factory=list)
    element_part_ids: dict[str, str] = field(default_factory=dict)
    octave_spans: list[OctaveSpan] = field(default_factory=list)
    mei: str = ""


@dataclass(frozen=True, slots=True)
class _Event:
    source: NoteEvent
    start: Fraction
    end: Fraction
    midi: int
    articulation: str
    staccato: bool = False
    grace_main_id: str = ""


_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _identifier(value: str) -> str:
    # XML IDs must begin with a letter; add a digest-like character-code suffix
    # only when names would otherwise lose all identifying characters.
    safe = _SAFE.sub("_", value).strip("_")
    if not safe:
        safe = "_".join(f"{ord(c):x}" for c in value) or "part"
    return f"sp_{safe}_{zlib.crc32(value.encode('utf-8')):08x}"


def _nearest(value: Fraction, step: Fraction) -> Fraction:
    return round(value / step) * step


def _quantize_position(value: Fraction, step: Fraction, triplets: bool) -> Fraction:
    straight = _nearest(value, step)
    if not triplets:
        return straight
    ternary = _nearest(value, step * Fraction(2, 3))
    # Recognize clearly ternary timing instead of snapping slightly humanized
    # straight notes to a much denser combined lattice.
    if abs(value - ternary) <= Fraction(1, 32) and abs(value - ternary) < abs(value - straight):
        return ternary
    return straight


def _articulation(event: NoteEvent, mapping: PartMapping) -> str:
    value = event.articulation or mapping.articulations.get(event.track_id, "")
    value = value.strip().lower()
    if value in ("pizz", "pizzicato", "pizz."):
        return "pizz."
    if value in ("normal", "arco", "sustain", "legato"):
        return "arco"
    return value


def _events(document: ProjectDocument, mapping: PartMapping, diagnostics: list[Diagnostic]) -> list[_Event]:
    step = Fraction(4, mapping.quantization)
    sources = [source for source in document.project.notes
               if source.track_id in mapping.track_ids and source.pitch not in mapping.keyswitches
               and 0 <= (source.pitch if mapping.percussion else source.pitch + mapping.transpose) <= 127]
    recognition = recognize_notes(sources, mapping, document.project.ppq, document.project.bpm,
                                  document.project.end_tick,
                                  lambda value: _quantize_position(value, step, mapping.triplets))
    keyswitch_count = sum(source.track_id in mapping.track_ids and source.pitch in mapping.keyswitches
                          for source in document.project.notes)
    if keyswitch_count:
        recognition.skipped["Keyswitch"] += keyswitch_count
    sources_by_id = {source.note_id: source for source in sources}
    result: list[_Event] = []
    seen: set[tuple[Fraction, Fraction, int]] = set()
    retained: dict[tuple[Fraction, Fraction, int], str] = {}
    aliases: dict[str, str] = {}
    max_shift = 0.0
    shifted = duplicates = slides = out_of_range = 0
    for source in document.project.notes:
        if source.track_id not in mapping.track_ids or source.pitch in mapping.keyswitches:
            continue
        raw_start = Fraction(max(0, source.start_tick), document.project.ppq)
        raw_end = Fraction(max(0, source.end_tick), document.project.ppq)
        start = _quantize_position(raw_start, step, mapping.triplets)
        end = _quantize_position(raw_end, step, mapping.triplets)
        interpretation = recognition.notes.get(source.note_id)
        grace_main_id = interpretation.grace_main_id if interpretation else ""
        staccato = interpretation.staccato if interpretation else False
        if grace_main_id:
            main = sources_by_id[grace_main_id]
            start = end = _quantize_position(Fraction(main.start_tick, document.project.ppq), step, mapping.triplets)
        elif interpretation is not None and interpretation.rhythmic_end is not None:
            end = interpretation.rhythmic_end
        elif end <= start:
            end = start + step
        midi = source.pitch if mapping.percussion else source.pitch + mapping.transpose
        if not 0 <= midi <= 127:
            out_of_range += 1
            continue
        token = (start, end, midi)
        if token in seen:
            duplicates += 1
            aliases[source.note_id] = retained[token]
            continue
        seen.add(token)
        retained[token] = source.note_id
        drift = max(abs(float(start - raw_start)), abs(float(end - raw_end)))
        if drift > 1 / document.project.ppq:
            shifted += 1
            max_shift = max(max_shift, drift * 60 / document.project.bpm)
        slides += bool(source.slide)
        result.append(_Event(source, start, end, midi, _articulation(source, mapping), staccato, grace_main_id))
    result = [replace(event, grace_main_id=aliases.get(event.grace_main_id, event.grace_main_id))
              if event.grace_main_id else event for event in result]
    if shifted:
        diagnostics.append(Diagnostic("info", "quantization-drift", f"{mapping.name}：{shifted} 个音符调整了记谱时间，最大偏差 {max_shift * 1000:.1f} ms；原始演奏时间未改变。", mapping.part_id))
    if duplicates:
        diagnostics.append(Diagnostic("warning", "duplicate-notes", f"{mapping.name}：合并后 {duplicates} 个相同音高、起止的重复音符合并为一个。", mapping.part_id))
    if slides:
        diagnostics.append(Diagnostic("warning", "slide-notation", f"{mapping.name}：{slides} 个 Slide 以基础音高记谱；请补充烘焙 MIDI 核对实际演奏。", mapping.part_id))
    if out_of_range:
        diagnostics.append(Diagnostic("warning", "transpose-range", f"{mapping.name}：{out_of_range} 个移调后超出 MIDI 音域的音符未写入谱面。", mapping.part_id))
        recognition.skipped["移调后超出音域"] += out_of_range
    skipped = "、".join(f"{reason} {count}" for reason, count in sorted(recognition.skipped.items())) or "无"
    diagnostics.append(Diagnostic("info", "auto-note-recognition",
                                  f"{mapping.name}：跳音 {sum(event.staccato for event in result)} 个，前倚音 {sum(bool(event.grace_main_id) for event in result)} 个；跳过推断：{skipped}。原始演奏时间未改变。",
                                  mapping.part_id))
    if any(event.articulation == "pizz." for event in result):
        result = [replace(event, articulation="arco") if not event.articulation else event for event in result]
    ordered = sorted(result, key=lambda e: (e.start, e.end, e.midi, e.source.note_id))
    previous: dict[int, _Event] = {}
    overlapping = 0
    for event in ordered:
        other = previous.get(event.midi)
        if other is not None and other.end > event.start and other.source.track_id != event.source.track_id:
            overlapping += 1
        if other is None or other.end < event.end:
            previous[event.midi] = event
    if overlapping:
        diagnostics.append(Diagnostic("warning", "merge-overlap", f"{mapping.name}：合并音轨中有 {overlapping} 个同音高重叠事件，已分配独立声部，请核对奏法。", mapping.part_id))
    return ordered


def _global_key(document: ProjectDocument) -> int:
    analysis = stream.Stream()
    pitched_tracks = {track_id for mapping in document.mappings if mapping.enabled and not mapping.percussion for track_id in mapping.track_ids}
    excluded = {track_id: set(mapping.keyswitches) for mapping in document.mappings for track_id in mapping.track_ids}
    for event in document.project.notes:
        if event.track_id not in pitched_tracks or event.pitch in excluded.get(event.track_id, set()) or event.duration_tick <= 0 or not 0 <= event.pitch <= 127:
            continue
        item = note.Note(event.pitch)
        item.quarterLength = Fraction(event.duration_tick, document.project.ppq)
        analysis.insert(Fraction(event.start_tick, document.project.ppq), item)
    if not analysis.notes:
        return 0
    try:
        return max(-7, min(7, analysis.analyze("key").sharps))
    except (ValueError, stream.StreamException):
        return 0


def _key_signature(mapping: PartMapping, inferred: int) -> key.KeySignature:
    if mapping.key_signature is not None:
        return key.KeySignature(mapping.key_signature)
    if mapping.percussion:
        return key.KeySignature(0)
    signature = key.KeySignature(inferred)
    if mapping.transpose:
        signature = signature.transpose(mapping.transpose)
        if not -7 <= signature.sharps <= 7:
            signature = key.KeySignature(key.Key(signature.asKey().tonic.getEnharmonic()).sharps)
    return signature


def _choose_clef(mapping: PartMapping, events: list[_Event], staff_index: int, grand: bool):
    if mapping.percussion or mapping.clef == "percussion":
        return clef.PercussionClef()
    if grand:
        return clef.TrebleClef() if staff_index == 0 else clef.BassClef()
    choices = {"treble": clef.TrebleClef, "bass": clef.BassClef, "alto": clef.AltoClef, "tenor": clef.TenorClef}
    if mapping.clef in choices:
        return choices[mapping.clef]()
    if mapping.instrument.lower() in ("viola",):
        return clef.AltoClef()
    average = sum(e.midi * float(e.end - e.start) for e in events) / max(sum(float(e.end - e.start) for e in events), 0.001)
    return clef.BassClef() if events and average < 60 else clef.TrebleClef()


def _spelled_pitch(midi: int, signature: key.KeySignature, *,
                   simplify_accidentals: bool = True) -> pitch.Pitch:
    value = pitch.Pitch()
    value.midi = midi
    if simplify_accidentals:
        for altered in signature.alteredPitches:
            if altered.pitchClass != value.pitchClass:
                continue
            candidate = deepcopy(altered)
            candidate.octave = value.octave
            # B# belongs to the octave below C; C-flat to the octave above B.
            # Match the sounding pitch before makeNotation decides what to show.
            candidate.octave += (midi - int(candidate.ps)) // 12
            return candidate
    if signature.sharps < 0 and value.accidental is not None and value.accidental.alter > 0:
        value = value.getEnharmonic()
    elif signature.sharps > 0 and value.accidental is not None and value.accidental.alter < 0:
        value = value.getEnharmonic()
    return value


_DRUM_NAMES = {
    "kick": ("F", 4, instrument.BassDrum), "bass drum": ("F", 4, instrument.BassDrum),
    "snare": ("C", 5, instrument.SnareDrum), "hi-hat": ("G", 5, instrument.HiHatCymbal),
    "hihat": ("G", 5, instrument.HiHatCymbal), "ride": ("F", 5, instrument.RideCymbals),
    "crash": ("A", 5, instrument.CrashCymbals), "tom": ("A", 4, instrument.UnpitchedPercussion),
    "floor tom": ("G", 4, instrument.UnpitchedPercussion),
    "low tom": ("A", 4, instrument.UnpitchedPercussion),
    "mid tom": ("D", 5, instrument.UnpitchedPercussion),
    "high tom": ("E", 5, instrument.UnpitchedPercussion),
    "cowbell": ("E", 5, instrument.Cowbell),
}
_GM_DRUM_NAMES = {35: "kick", 36: "kick", 37: "snare", 38: "snare", 40: "snare", 41: "floor tom", 42: "hi-hat", 43: "floor tom", 44: "hi-hat", 45: "low tom", 46: "hi-hat", 47: "mid tom", 48: "high tom", 49: "crash", 50: "high tom", 51: "ride", 53: "ride", 56: "cowbell", 57: "crash", 59: "ride"}


def _drum(midi: int, mapping: PartMapping) -> note.Unpitched:
    display = mapping.percussion_map.get(str(midi), _GM_DRUM_NAMES.get(midi, "tom"))
    item = note.Unpitched()
    if display.lower() in _DRUM_NAMES:
        step, octave, instrument_type = _DRUM_NAMES[display.lower()]
        item.displayStep, item.displayOctave = step, octave
        item.storedInstrument = instrument_type()
    else:
        try:
            display_pitch = pitch.Pitch(display)
            item.displayStep, item.displayOctave = display_pitch.step, display_pitch.octave or 4
            item.storedInstrument = instrument.UnpitchedPercussion()
        except (pitch.PitchException, pitch.AccidentalException, ValueError) as exc:
            raise ValueError(f"打击乐映射 {midi}={display!r} 无效；请输入 C5 等谱面音高或 kick/snare/hi-hat 等名称。") from exc
    if display.lower() in ("hi-hat", "hihat", "ride", "crash"):
        item.notehead = "x"
    return item


def _assign_voices(events: list[_Event]) -> list[list[list[_Event]]]:
    groups: dict[tuple[Fraction, Fraction, str, bool], list[_Event]] = defaultdict(list)
    for event in events:
        if event.grace_main_id:
            continue
        groups[(event.start, event.end, event.articulation, event.staccato)].append(event)
    voices: list[list[list[_Event]]] = []
    ends: list[Fraction] = []
    for (start, end, _articulation_name, _staccato), group in sorted(groups.items()):
        available = next((i for i, last in enumerate(ends) if last <= start), None)
        if available is None:
            available = len(voices)
            voices.append([])
            ends.append(Fraction(0))
        voices[available].append(group)
        ends[available] = end
    return voices or [[]]


def _articulation_labels(events: list[_Event], bar_length: Fraction) -> list[tuple[Fraction, str]]:
    """Mark technique phrases, not individual staccato note onsets.

    A technique stays in force over short rests.  Interleaved arco and pizz
    channels therefore form a mixed phrase instead of adding a stack of labels
    on every attack.  Gaps longer than a bar begin a new phrase.
    """
    techniques: dict[str, list[tuple[Fraction, Fraction]]] = defaultdict(list)
    for event in events:
        if event.articulation:
            techniques[event.articulation].append((event.start, event.end))
    changes: dict[Fraction, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for name, intervals in techniques.items():
        phrases: list[tuple[Fraction, Fraction]] = []
        for start, end in sorted(intervals):
            if phrases and start <= phrases[-1][1] + bar_length:
                phrases[-1] = (phrases[-1][0], max(end, phrases[-1][1]))
            else:
                phrases.append((start, end))
        for start, end in phrases:
            changes[start][name] += 1
            changes[end][name] -= 1
    active: dict[str, int] = defaultdict(int)
    labels: list[tuple[Fraction, str]] = []
    previous = ""
    for beat, deltas in sorted(changes.items()):
        for name, delta in deltas.items():
            active[name] += delta
        label = " / ".join(sorted(name for name, count in active.items() if count > 0))
        if label and label != previous:
            labels.append((beat, label))
            previous = label
    return labels


def _build_staff(document: ProjectDocument, mapping: PartMapping, events: list[_Event], signature: key.KeySignature, end: Fraction, staff_index: int, grand: bool,
                 *, octave_spans: list[OctaveSpan] | None = None) -> stream.Part:
    part_type = stream.PartStaff if grand else stream.Part
    part = part_type(id=f"{_identifier(mapping.part_id)}_s{staff_index}")
    part.partName = mapping.name
    part.partAbbreviation = ""
    generic = instrument.Instrument()
    generic.instrumentName = mapping.instrument
    generic.partId = str(part.id)
    part.insert(0, generic)
    staff_clef = _choose_clef(mapping, events, staff_index, grand)
    part.insert(0, staff_clef)
    part.insert(0, deepcopy(signature))
    part.insert(0, meter.TimeSignature(f"{document.project.numerator}/{document.project.denominator}"))
    part.insert(0, tempo.MetronomeMark(number=document.project.bpm))
    spelled = {event.source.note_id: _spelled_pitch(
        event.midi, signature,
        simplify_accidentals=mapping.auto_simplify_accidentals and not mapping.percussion,
    ) for event in events}
    shifts = infer_ottavas([
        OctaveNote(event.source.note_id, event.start, event.end,
                   spelled[event.source.note_id].diatonicNoteNum, event.grace_main_id)
        for event in events
    ], staff_clef.lowestLine, mapping.part_id, staff_index) if (
        mapping.auto_ottava and not mapping.percussion
        and not isinstance(staff_clef, clef.PercussionClef)
    ) else []

    def written_pitch(event: _Event) -> pitch.Pitch:
        value = deepcopy(spelled[event.source.note_id])
        for shift in shifts:
            if shift.start_beat <= event.start < shift.end_beat:
                value.octave -= shift.octaves
                break
        return value

    bar_length = Fraction(document.project.numerator * 4, document.project.denominator)
    for start, label in _articulation_labels(events, bar_length):
        expression = expressions.TextExpression(label)
        expression.placement = "above"
        expression.style.fontFamily = "Edwin"
        expression.style.fontStyle = "italic"
        part.insert(start, expression)
    voices = _assign_voices(events)
    graces: dict[str, list[_Event]] = defaultdict(list)
    for event in events:
        if event.grace_main_id:
            graces[event.grace_main_id].append(event)
    for voice_index, groups in enumerate(voices):
        voice = stream.Voice(id=voice_index + 1)
        last_end = Fraction(0)
        for group in groups:
            start, stop = group[0].start, group[0].end
            if start > last_end:
                voice.insert(last_end, note.Rest(quarterLength=start - last_end))
            for member in group:
                for grace in graces.get(member.source.note_id, []):
                    ornament = note.Note(written_pitch(grace), type="eighth").getGrace()
                    ornament.volume.velocity = grace.source.velocity
                    voice.insert(start, ornament)
            if mapping.percussion:
                members = [_drum(e.midi, mapping) for e in group]
                item = members[0] if len(members) == 1 else percussion.PercussionChord(members)
            elif len(group) == 1:
                item = note.Note(written_pitch(group[0]))
            else:
                item = chord.Chord([written_pitch(e) for e in group])
            item.quarterLength = stop - start
            item.volume.velocity = max(e.source.velocity for e in group)
            if group[0].staccato:
                item.articulations.append(articulations.Staccato())
            voice.insert(start, item)
            last_end = stop
        if last_end < end:
            voice.insert(last_end, note.Rest(quarterLength=end - last_end))
        part.insert(0, voice)
    part.makeNotation(inPlace=True)
    # music21 starts voice numbering at zero.  Verovio treats voice 0 as an
    # invalid layer and shifts subsequent voices, corrupting shared time maps.
    for measure_index, measure in enumerate(part.getElementsByClass(stream.Measure)):
        for voice_index, voice in enumerate(measure.voices):
            voice.id = voice_index + 1
        for event_index, item in enumerate(measure.recurse().notesAndRests):
            item.id = f"{part.id}_m{measure_index}_e{event_index}"
            if isinstance(item, chord.ChordBase):
                for note_index, member in enumerate(item.notes):
                    member.id = f"{item.id}_n{note_index}"
    final_shifts = _bind_ottavas(part, shifts, signature)
    if octave_spans is not None:
        octave_spans.extend(final_shifts)
    return part


def _bind_ottavas(part: stream.Part, shifts: list[OctaveSpan],
                  signature: key.KeySignature) -> list[OctaveSpan]:
    """Bind only after makeNotation has created every tied note fragment.

    The score stores written pitches, while MusicXML restores the sounding
    octave on its own copy through these transposing spanners.
    """
    if not shifts:
        return []
    items = sorted((
        (Fraction(item.getOffsetInHierarchy(part)).limit_denominator(1_000_000), item)
        for item in part.recurse().notes
    ), key=lambda entry: (entry[0], not entry[1].duration.isGrace))

    def element_id(item) -> str:
        return str(item.notes[0].id if isinstance(item, chord.ChordBase) else item.id)

    result = []
    for shift in shifts:
        selected = [(beat, item) for beat, item in items
                    if shift.start_beat <= beat < shift.end_beat]
        if not selected:
            raise ValueError("八度移位区间没有对应的记谱音符。")
        first = selected[0][1]
        # MusicXML ends at the latest rhythmic release. The derived graphical
        # endpoint instead follows the last notehead, even when an earlier
        # voice sustains beyond that final attack.
        _, last = max(selected, key=lambda entry: (
            entry[0] + Fraction(entry[1].quarterLength), entry[0]))
        _, final_head = max(selected, key=lambda entry: (
            entry[0], not entry[1].duration.isGrace, Fraction(entry[1].quarterLength)))
        members = [first] + [item for _, item in selected
                             if item is not first and item is not last]
        if last is not first:
            members.append(last)
        ottava = spanner.Ottava(*members, type=shift.label, transposing=True,
                               placement="above" if shift.octaves > 0 else "below")
        part.insert(0, ottava)
        result.append(replace(shift, start_element_id=element_id(first),
                              end_element_id=element_id(final_head)))
    _ottava_boundary_accidentals(items, shifts, signature)
    return result


def _ottava_boundary_accidentals(items, shifts: list[OctaveSpan],
                                 signature: key.KeySignature) -> None:
    """Restate chromatic pitches when the octave interpretation changes."""
    boundaries = {span.start_beat for span in shifts} | {span.end_beat for span in shifts}
    for boundary in sorted(boundaries):
        following = next((beat for beat, item in items
                          if beat >= boundary and not item.duration.isGrace), None)
        if following is None:
            continue
        for beat, item in items:
            if beat != following:
                continue
            for value in item.pitches:
                alteration = value.accidental.alter if value.accidental else 0
                key_accidental = signature.accidentalByStep(value.step)
                expected = key_accidental.alter if key_accidental else 0
                if alteration != expected:
                    if value.accidental is None:
                        value.accidental = pitch.Accidental("natural")
                    value.accidental.displayStatus = True


def _xml(score: stream.Score) -> str:
    value = musicxml.m21ToXml.GeneralObjectExporter(score).parse().decode("utf-8")
    root = ET.fromstring(value)
    _order_ottava_stops(root)
    _number_ottavas(root)
    has_tempo = False
    for measure in root.iter("measure"):
        for direction in list(measure.findall("direction")):
            for direction_type in direction.findall("direction-type"):
                metronome = direction_type.find("metronome")
                if metronome is None:
                    continue
                if has_tempo:
                    # The fixed global tempo belongs above the first staff.
                    # Grand-staff and merged score exports would duplicate it.
                    measure.remove(direction)
                    break
                has_tempo = True
                # Verovio writes this as a private SMuFL text glyph in its SVG.
                # Qt has no embedded Leipzig font, so a plain label avoids a
                # missing-glyph square; the sibling <sound> keeps timing.
                bpm = metronome.findtext("per-minute", "120")
                direction_type.remove(metronome)
                ET.SubElement(direction_type, "words").text = f"BPM {bpm}"
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode")


def _order_ottava_stops(root: ET.Element) -> None:
    """Keep starts before stops when a line joins different XML voices.

    A later voice can contain the opening grace while an earlier voice holds
    the final release. music21 then writes the stop before backing up to the
    start. Move only those premature stops to the measure's end and preserve
    their musical time with a MusicXML direction offset.
    """
    for part in root.findall("part"):
        active: set[tuple[str, str]] = set()
        divisions = Fraction(1)
        for measure in part.findall("measure"):
            cursor = Fraction(0)
            premature = []
            started = set()
            for element in list(measure):
                if element.tag == "attributes":
                    divisions = Fraction(element.findtext("divisions", str(divisions)))
                elif element.tag == "note":
                    if element.find("grace") is None and element.find("chord") is None:
                        cursor += Fraction(element.findtext("duration", "0")) / divisions
                elif element.tag in ("backup", "forward"):
                    delta = Fraction(element.findtext("duration", "0")) / divisions
                    cursor += delta if element.tag == "forward" else -delta
                elif element.tag == "direction":
                    position = cursor + Fraction(element.findtext("offset", "0")) / divisions
                    for mark in element.findall("direction-type/octave-shift"):
                        token = (element.findtext("staff", "1"), mark.get("number", "1"))
                        if mark.get("type") in ("up", "down"):
                            active.add(token)
                            started.add(token)
                        elif mark.get("type") == "stop":
                            if token in active:
                                active.remove(token)
                            else:
                                premature.append((element, token, position))
            for direction, token, position in premature:
                if token not in started or token not in active:
                    continue
                offset = direction.find("offset")
                if offset is None:
                    offset = ET.Element("offset")
                    index = len(direction.findall("direction-type"))
                    direction.insert(index, offset)
                value = (position - cursor) * divisions
                offset.text = str(value.numerator) if value.denominator == 1 else f"{float(value):.12g}"
                measure.remove(direction)
                measure.append(direction)
                active.remove(token)


def _number_ottavas(root: ET.Element) -> None:
    """Keep simultaneous octave lines distinct after PartStaff XML merging.

    music21 assigns spanner numbers independently to each staff. Verovio tracks
    octave-shift numbers across the entire MusicXML part, so overlapping piano
    lines must be renumbered together, including their stop/continue directions.
    """
    for part in root.findall("part"):
        active: dict[tuple[str, str], str] = {}
        for direction in part.findall("measure/direction"):
            staff = direction.findtext("staff", "1")
            for mark in direction.findall("direction-type/octave-shift"):
                token = (staff, mark.get("number", "1"))
                if mark.get("type") in ("up", "down"):
                    used = set(active.values())
                    number = next(str(index) for index in range(1, len(used) + 2)
                                  if str(index) not in used)
                    active[token] = number
                number = active.get(token)
                if number is not None:
                    mark.set("number", number)
                if mark.get("type") == "stop":
                    active.pop(token, None)


def _insert_dynamics(part: stream.Part, spans: list[DynamicSpan], ppq: int) -> None:
    """Anchor exact automation times, including endpoints inside a sustained note."""
    measures = list(part.getElementsByClass(stream.Measure))
    if not measures:
        return
    for span_index, span in enumerate(spans):
        anchors = []
        for endpoint, tick in enumerate((span.start_tick, span.end_tick)):
            beat = Fraction(str(tick)) / ppq
            measure = next((measure for measure in reversed(measures)
                            if measure.offset < beat or (endpoint == 0 and measure.offset == beat)), measures[0])
            anchor = spanner.SpannerAnchor()
            anchor.id = f"{part.id}_dynamic_{span_index}_{endpoint}"
            offset = beat - Fraction(str(measure.offset))
            measure.insert(offset, anchor)
            # music21.insert converts Fraction to float before storing it.
            # Its public setter retains the original source rational value.
            measure.setElementOffset(anchor, offset)
            anchors.append(anchor)
        wedge = (dynamics.Crescendo if span.direction == "crescendo" else dynamics.Diminuendo)(*anchors)
        wedge.placement = "below"
        part.insert(0, wedge)


def _dynamic_activity(document: ProjectDocument, mapping: PartMapping,
                      events: list[_Event]) -> list[tuple[str, float, float]]:
    """Retain all sounding sources, including notes deduplicated in the printed chord."""
    kept = {event.source.note_id: event for event in events}
    step = Fraction(4, mapping.quantization)
    activity = []
    for source in document.project.notes:
        if (source.track_id not in mapping.track_ids or source.pitch in mapping.keyswitches
                or not 0 <= (source.pitch if mapping.percussion else source.pitch + mapping.transpose) <= 127):
            continue
        if source.note_id in kept:
            event = kept[source.note_id]
            start, end = event.start, event.end
        else:
            start = _quantize_position(Fraction(max(0, source.start_tick), document.project.ppq), step, mapping.triplets)
            end = _quantize_position(Fraction(max(0, source.end_tick), document.project.ppq), step, mapping.triplets)
            if end <= start:
                end = start + step
        if end > start:
            activity.append((source.track_id, float(start * document.project.ppq), float(end * document.project.ppq)))
    return activity


def _display_xml(value: str) -> str:
    root = ET.fromstring(value)
    for tag in ("part-name", "part-abbreviation"):
        for element in root.iter(tag):
            element.text = ""
    for direction in root.iter("direction"):
        for direction_type in list(direction):
            words = direction_type.find("words")
            if words is not None and (words.text or "").startswith("BPM "):
                direction.remove(direction_type)
    return ET.tostring(root, encoding="unicode")


def _toolkit(value: str, continuous: bool = True, *,
             octave_score: stream.Score | None = None,
             octave_spans: list[OctaveSpan] | None = None):
    toolkit = verovio.toolkit()
    options = {"condense": "none", "xmlIdSeed": 1, "svgViewBox": True, "svgBoundingBoxes": True, "svgFormatRaw": True,
               "font": MUSIC_FAMILY, "fontFallback": MUSIC_FALLBACK}
    if continuous:
        options.update({"breaks": "none", "header": "none", "footer": "none", "adjustPageWidth": True, "adjustPageHeight": True})
    else:
        options.update({"breaks": "auto", "pageWidth": 2100, "pageHeight": 2970, "scale": 100, "header": "none", "footer": "none"})
    if not toolkit.setOptions(options) or not toolkit.loadData(value):
        raise ValueError(f"Verovio 无法加载谱面：{toolkit.getLog()}")
    if octave_score is not None and octave_spans:
        _align_ottava_mei(toolkit, octave_score, octave_spans)
    return toolkit


def _align_ottava_mei(toolkit, score: stream.Score, spans: list[OctaveSpan]) -> None:
    """Use the common written score to repair sequential MusicXML import.

    Verovio tracks octave shifts in XML document order. A backup into another
    voice after a stop can leave that voice unshifted, even though its notes lie
    within the staff's octave interval. MEI encodes written and sounding octaves
    independently, so restore the former and keep the original oct.ges intact.
    The graphical endpoint is the final notehead; the MusicXML stop retains the
    full rhythmic end, which may belong to an earlier sustained voice.
    """
    written = {}
    for item in score.recurse().notes:
        members = item.notes if isinstance(item, chord.ChordBase) else [item]
        for member in members:
            if isinstance(member, note.Note):
                written[str(member.id)] = str(member.pitch.octave)
    root = ET.fromstring(toolkit.getMEI())
    xml_id = "{http://www.w3.org/XML/1998/namespace}id"
    parents = {child: parent for parent in root.iter() for child in parent}
    nodes = {element.get(xml_id): element for element in root.iter() if element.get(xml_id)}
    changed = False
    for identifier, octave in written.items():
        element = nodes.get(identifier)
        if element is None or element.tag.rsplit("}", 1)[-1] != "note":
            continue
        sounding = element.get("oct.ges", element.get("oct"))
        if element.get("oct") != octave:
            element.set("oct", octave)
            if sounding == octave:
                element.attrib.pop("oct.ges", None)
            elif sounding is not None:
                element.set("oct.ges", sounding)
            changed = True

    def chord_reference(identifier: str) -> str:
        identifier = identifier.lstrip("#")
        element = nodes.get(identifier)
        parent = parents.get(element)
        if parent is not None and parent.tag.rsplit("}", 1)[-1] == "chord":
            return parent.get(xml_id, identifier)
        return identifier

    octaves = [element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "octave"]
    for span in spans:
        target = next((element for element in octaves
                       if chord_reference(element.get("startid", ""))
                       == chord_reference(span.start_element_id)), None)
        if target is None:
            raise ValueError("Verovio 的八度线缺少对应的起始音符。")
        if chord_reference(target.get("endid", "")) != chord_reference(span.end_element_id):
            target.set("endid", "#" + chord_reference(span.end_element_id))
            changed = True
    if changed:
        # Verovio expects unprefixed MEI tags. Avoid changing ElementTree's
        # global namespace registry, which is also used for SVG rendering.
        namespace = "http://www.music-encoding.org/ns/mei"
        for element in root.iter():
            element.tag = element.tag.removeprefix("{" + namespace + "}")
        root.set("xmlns", namespace)
        if not toolkit.loadData(ET.tostring(root, encoding="unicode")):
            raise ValueError(f"Verovio 无法加载八度移位谱面：{toolkit.getLog()}")


def _element_owners(mei: str, staff_part_ids: list[str]) -> dict[str, str]:
    """Recover decoration ownership from MEI, never from vertical proximity.

    Ties and slurs are system children with startid links; directions carry a
    staff number.  Their descendant SVG text/rend IDs inherit that ownership.
    """
    root = ET.fromstring(mei)
    xml_id = "{http://www.w3.org/XML/1998/namespace}id"
    owners: dict[str, str] = {}
    pending: list[ET.Element] = []

    def staff_owner(numbers: str) -> str:
        parts = {staff_part_ids[int(number) - 1] for number in numbers.split() if number.isdecimal() and 1 <= int(number) <= len(staff_part_ids)}
        return next(iter(parts)) if len(parts) == 1 else ""

    def walk(element: ET.Element, inherited: str = "") -> None:
        tag = element.tag.rsplit("}", 1)[-1]
        owner = inherited
        if tag in ("staff", "staffDef"):
            owner = staff_owner(element.get("n", "")) or inherited
        elif element.get("staff"):
            owner = staff_owner(element.get("staff", "")) or inherited
        elif tag == "staffGrp":
            group_parts = {staff_owner(child.get("n", "")) for child in element.iter() if child.tag.rsplit("}", 1)[-1] == "staffDef"}
            if len(group_parts) == 1:
                owner = next(iter(group_parts))
        identifier = element.get(xml_id)
        if identifier and owner:
            owners[identifier] = owner
        elif identifier and (element.get("startid") or element.get("corresp")):
            pending.append(element)
        for child in element:
            walk(child, owner)

    walk(root)
    for element in pending:
        references = (element.get("startid") or element.get("corresp") or "").split()
        owner = next((owners.get(reference.lstrip("#"), "") for reference in references if owners.get(reference.lstrip("#"))), "")
        if owner:
            for child in element.iter():
                identifier = child.get(xml_id)
                if identifier:
                    owners[identifier] = owner
    return owners


def build_notation(document: ProjectDocument) -> NotationResult:
    """Return one fully engraved continuous score and IDs for its musical time columns."""
    document.validate()
    diagnostics = list(document.project.diagnostics)
    score = stream.Score(id="stavellum_score")
    score.metadata = metadata.Metadata()
    score.metadata.title = document.metadata.title
    score.metadata.composer = document.metadata.composer
    if document.metadata.arranger:
        score.metadata.add("arranger", document.metadata.arranger)
    if document.metadata.subtitle:
        score.metadata.alternativeTitle = document.metadata.subtitle
    inferred_key = _global_key(document)
    active = [mapping for mapping in document.mappings if mapping.enabled]
    prepared = [(mapping, _events(document, mapping, diagnostics)) for mapping in active]
    bar_length = Fraction(document.project.numerator * 4, document.project.denominator)
    raw_end = Fraction(document.project.end_tick, document.project.ppq)
    last = max([raw_end] + [e.end for _, events in prepared for e in events])
    end = max(bar_length, math.ceil(last / bar_length) * bar_length)
    staff_part_ids: list[str] = []
    quantized: list[QuantizedEvent] = []
    octave_spans: list[OctaveSpan] = []
    for mapping, events in prepared:
        signature = _key_signature(mapping, inferred_key)
        grand = mapping.grand_staff or mapping.instrument.lower() == "piano"
        main_pitches = {event.source.note_id: event.midi for event in events if not event.grace_main_id}
        staff_events = [[e for e in events if main_pitches.get(e.grace_main_id, e.midi) >= 60],
                        [e for e in events if main_pitches.get(e.grace_main_id, e.midi) < 60]] if grand and not mapping.percussion else [events]
        staff_parts = []
        first_shift = len(octave_spans)
        for staff_index, members in enumerate(staff_events):
            part = _build_staff(document, mapping, members, signature, end, staff_index, len(staff_events) == 2,
                                octave_spans=octave_spans)
            score.insert(0, part)
            staff_parts.append(part)
            staff_part_ids.append(mapping.part_id)
        counts = {label: sum(span.label == label for span in octave_spans[first_shift:])
                  for label in ("8va", "8vb", "15ma", "15mb")}
        if not mapping.auto_ottava:
            octave_message = "自动八度移位已关闭。"
        elif mapping.percussion or mapping.clef == "percussion":
            octave_message = "打击乐谱表跳过八度移位。"
        else:
            octave_message = "、".join(f"{label} {count} 段" for label, count in counts.items())
            octave_message += "；其余乐段保留原位（常规音域、连续起音不足、移位收益不足或同谱表音域冲突）。"
        diagnostics.append(Diagnostic("info", "auto-ottava", f"{mapping.name}：{octave_message}",
                                      mapping.part_id))
        dynamic_spans, dynamic_diagnostics = infer_dynamics(document.project, mapping,
                                                            _dynamic_activity(document, mapping, events))
        diagnostics.extend(dynamic_diagnostics)
        _insert_dynamics(staff_parts[-1], dynamic_spans, document.project.ppq)
        if len(staff_parts) == 2:
            score.insert(0, layout.StaffGroup(staff_parts, name=mapping.name, abbreviation="", symbol="brace", barTogether=True))
        quantized.extend(QuantizedEvent(e.source.note_id, mapping.part_id, float(e.start), float(e.end), e.midi, e.source.velocity, e.articulation, e.staccato, e.grace_main_id) for e in events)
        if mapping.key_signature is None and not mapping.percussion:
            diagnostics.append(Diagnostic("info", "key-inferred", f"{mapping.name}：自动调号为 {signature.sharps}；可在记谱设置中覆盖。", mapping.part_id))
        if mapping.percussion and any(str(e.midi) not in mapping.percussion_map for e in events):
            diagnostics.append(Diagnostic("info", "percussion-default", f"{mapping.name}：未配置的打击乐音高按 GM 鼓组映射显示，可手动覆盖。", mapping.part_id))
        unknown_drums = sorted({e.midi for e in events if mapping.percussion and str(e.midi) not in mapping.percussion_map and e.midi not in _GM_DRUM_NAMES})
        if unknown_drums:
            diagnostics.append(Diagnostic("warning", "percussion-unmapped", f"{mapping.name}：音高 {', '.join(map(str, unknown_drums))} 没有已知鼓组映射，暂置于中间鼓位置；请配置谱面位置。", mapping.part_id))
        if not events:
            diagnostics.append(Diagnostic("info", "empty-part", f"{mapping.name}：过滤后没有音符，保留休止谱表。", mapping.part_id))
        if any(" / " in label for _, label in _articulation_labels(events, bar_length)):
            diagnostics.append(Diagnostic("info", "mixed-articulations", f"{mapping.name}：部分乐句同时包含多种奏法，已合并标注；可拆分声部以便分别阅读。", mapping.part_id))
    value = _xml(score)
    toolkit = _toolkit(_display_xml(value), octave_score=score, octave_spans=octave_spans)
    svg = toolkit.renderToSVG(1)
    if toolkit.getPageCount() != 1:
        raise ValueError("连续总谱未形成一个完整横排系统。")
    anchors: list[NotationAnchor] = []
    mei = toolkit.getMEI()
    xml_id = "{http://www.w3.org/XML/1998/namespace}id"
    grace_ids = {element.get(xml_id) for element in ET.fromstring(mei).iter()
                 if element.tag.rsplit("}", 1)[-1] == "note" and element.get("grace")}
    for entry in toolkit.renderToTimemap({"includeRests": True}):
        beat = float(entry["qstamp"])
        anchors.extend(NotationAnchor(beat, element, "note") for element in entry.get("on", []) if element not in grace_ids)
        anchors.extend(NotationAnchor(beat, element, "rest") for element in entry.get("restsOn", []))
    # Every bar has a geometric anchor even if it contains only whole-bar rests.
    measures = [e for e in ET.fromstring(svg).iter() if e.attrib.get("class", "").split() == ["measure"]]
    anchors.extend(NotationAnchor(i * float(bar_length), element.attrib["id"], "measure") for i, element in enumerate(measures) if "id" in element.attrib)
    owners = _element_owners(mei, staff_part_ids)
    return NotationResult(svg, score, diagnostics, staff_part_ids, value, sorted(anchors, key=lambda a: (a.beat, a.kind, a.element_id)), quantized, owners,
                          octave_spans, mei)


def _cancelled(cancel) -> bool:
    if cancel is None:
        return False
    return bool(cancel.is_set() if hasattr(cancel, "is_set") else cancel())


def _write_pdf(toolkit, target: Path, title: str, cancel=None, *, subtitle: str = "", credits: str = "",
               page_progress: Callable[[int, int], None] | None = None) -> None:
    from PySide6.QtCore import QByteArray, QMarginsF, QRectF
    from PySide6.QtGui import QColor, QPageLayout, QPageSize, QPainter, QPdfWriter
    from PySide6.QtSvg import QSvgRenderer

    from stavellum.graphics.qt import ensure_app
    from stavellum.graphics.svg import normalize_svg
    from stavellum.graphics.typography import draw_text_rect, text_height

    ensure_app()
    writer = QPdfWriter(str(target))
    writer.setTitle(title)
    writer.setCreator("Stavellum / Verovio")
    writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    writer.setPageMargins(QMarginsF(10, 10, 10, 10), QPageLayout.Unit.Millimeter)
    writer.setResolution(150)
    painter = QPainter(writer)
    if not painter.isActive():
        raise OSError(f"无法写入 PDF：{target}")
    pages = toolkit.getPageCount()
    footer_height = 38
    try:
        if page_progress:
            page_progress(0, pages)
        for page in range(1, pages + 1):
            if _cancelled(cancel):
                raise InterruptedError("已取消分谱导出。")
            if page > 1 and not writer.newPage():
                raise OSError("无法创建下一页 PDF。")
            svg = normalize_svg(toolkit.renderToSVG(page), foreground="black")
            renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
            if not renderer.isValid():
                raise ValueError("Qt 无法读取分谱 SVG。")
            painter.setPen(QColor("black"))
            title_size = 26 if page == 1 else 21
            title_height = max(43, text_height(title, title_size, writer.width(),
                                               device=writer, wrap=True))
            draw_text_rect(painter, title, title_size,
                           QRectF(0, 0, writer.width(), title_height), wrap=True)
            header_bottom = title_height
            if page == 1 and subtitle:
                detail_height = text_height(subtitle, 16, writer.width(), device=writer, wrap=True)
                draw_text_rect(painter, subtitle, 16,
                               QRectF(0, header_bottom + 2, writer.width(), detail_height), wrap=True)
                header_bottom += detail_height + 2
            if page == 1 and credits:
                detail_height = text_height(credits, 16, writer.width(), device=writer, wrap=True)
                draw_text_rect(painter, credits, 16,
                               QRectF(0, header_bottom + 2, writer.width(), detail_height), wrap=True)
                header_bottom += detail_height + 2
            draw_text_rect(painter, f"{page} / {pages}", 16,
                           QRectF(0, writer.height() - footer_height, writer.width(), footer_height))
            header_height = max(100, header_bottom + 9)
            box = renderer.viewBoxF()
            factor = min(writer.width() / box.width(), (writer.height() - header_height - footer_height) / box.height())
            width, height = box.width() * factor, box.height() * factor
            renderer.render(painter, QRectF((writer.width() - width) / 2, header_height, width, height))
            if page_progress:
                page_progress(page, pages)
    finally:
        painter.end()


def export_parts(document: ProjectDocument, output_dir: str | Path, progress: Callable[[float, str], None] | None = None, cancel=None,
                  *, progress_detail=None) -> list[str]:
    """Write each logical instrument as MusicXML and a paginated vector PDF."""
    detail = ProgressReporter(progress_detail)
    active = [mapping for mapping in document.mappings if mapping.enabled]
    detail.emit("compile", "正在准备分谱与刻谱…", force=True,
                total=len(active), unit="parts")
    if _cancelled(cancel):
        raise InterruptedError("已取消分谱导出。")
    result = build_notation(document)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for index, mapping in enumerate(active):
        if _cancelled(cancel):
            raise InterruptedError("已取消分谱导出。")
        detail.emit("xml", f"正在生成 {mapping.name} 的 MusicXML…", force=True,
                    completed=index, total=len(active), unit="parts", part_name=mapping.name,
                    page=0, pages=0)
        selected = [deepcopy(part) for part, part_id in zip(result.score.parts, result.staff_part_ids, strict=True) if part_id == mapping.part_id]
        part_score = stream.Score()
        part_score.metadata = deepcopy(result.score.metadata)
        part_score.metadata.title = f"{document.metadata.title} — {mapping.name}"
        for part in selected:
            part_score.insert(0, part)
        if len(selected) == 2:
            part_score.insert(0, layout.StaffGroup(selected, name=mapping.name, abbreviation="", symbol="brace", barTogether=True))
        value = _xml(part_score)
        # Index and ID prevent collisions when two sections have the same name.
        filename = f"{index + 1:02d}_{_identifier(mapping.part_id)}"
        xml_path = destination / f"{filename}.musicxml"
        pdf_path = destination / f"{filename}.pdf"
        xml_tmp, pdf_tmp = xml_path.with_suffix(".musicxml.tmp"), pdf_path.with_suffix(".pdf.tmp")
        try:
            xml_tmp.write_text(value, encoding="utf-8")
            if progress:
                progress(index / len(active), f"正在导出 {mapping.name}")
            detail.emit("pdf", f"正在为 {mapping.name} 排版并生成 PDF…", force=True,
                        completed=index, total=len(active), unit="parts", part_name=mapping.name)
            toolkit = _toolkit(value, continuous=False, octave_score=part_score,
                               octave_spans=[span for span in result.octave_spans
                                             if span.part_id == mapping.part_id])
            credits = "    ".join(f"{label}：{value}" for label, value in (("作曲", document.metadata.composer), ("编曲", document.metadata.arranger)) if value)

            def page_progress(page, pages):
                detail.emit("pdf", f"正在生成 {mapping.name} 的 PDF：第 {page}/{pages} 页",
                            force=page in (0, pages), completed=index, total=len(active),
                            unit="parts", part_name=mapping.name, page=page, pages=pages)

            _write_pdf(toolkit, pdf_tmp, part_score.metadata.title, cancel,
                       subtitle=document.metadata.subtitle, credits=credits,
                       page_progress=page_progress)
            if _cancelled(cancel):
                raise InterruptedError("已取消分谱导出。")
            detail.emit("save", f"正在保存 {mapping.name} 的 MusicXML 与 PDF…", force=True,
                        completed=index, total=len(active), unit="parts", part_name=mapping.name)
            xml_tmp.replace(xml_path)
            pdf_tmp.replace(pdf_path)
            paths.extend([str(xml_path), str(pdf_path)])
            detail.emit("save", f"已导出 {mapping.name}", force=True,
                        completed=index + 1, total=len(active), unit="parts", part_name=mapping.name)
            if progress:
                progress((index + 1) / len(active), f"已导出 {mapping.name}")
        finally:
            xml_tmp.unlink(missing_ok=True)
            pdf_tmp.unlink(missing_ok=True)
    detail.emit("done", "分谱导出完成。", force=True,
                completed=len(active), total=len(active), unit="parts", part_name="",
                page=0, pages=0)
    return paths
