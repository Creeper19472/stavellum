"""Conservative notation hints derived from source gates, without changing playback."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction

from stavellum.domain.models import NoteEvent, PartMapping


@dataclass(frozen=True, slots=True)
class NoteInterpretation:
    staccato: bool = False
    grace_main_id: str = ""
    rhythmic_end: Fraction | None = None


@dataclass(slots=True)
class Recognition:
    notes: dict[str, NoteInterpretation] = field(default_factory=dict)
    skipped: Counter[str] = field(default_factory=Counter)


def source_technique(source: NoteEvent, mapping: PartMapping) -> str:
    return (source.articulation or mapping.articulations.get(source.track_id, "")).strip().casefold().rstrip(".")


def _gate(source: NoteEvent, ppq: int) -> Fraction | None:
    if source.key_release_tick is None or source.key_release_tick <= source.start_tick:
        return None
    return Fraction(source.key_release_tick - source.start_tick, ppq)


def _regular_short_note(source: NoteEvent, ppq: int, triplets: bool) -> bool:
    """Exact fine-grid notes have a plausible ordinary rhythmic interpretation."""
    if source.key_release_tick is None:
        return False
    positions = (Fraction(source.start_tick, ppq), Fraction(source.key_release_tick, ppq))
    for division in (32, 64):
        step = Fraction(4, division)
        grids = (step, step * Fraction(2, 3)) if triplets else (step,)
        for grid in grids:
            if all(abs(value - round(value / grid) * grid) <= Fraction(1, ppq)
                   for value in positions):
                return True
    return False


def recognize_notes(
    sources: list[NoteEvent], mapping: PartMapping, ppq: int, bpm: float,
    end_tick: int, quantize: Callable[[Fraction], Fraction],
) -> Recognition:
    """Use independent source/channel lanes; ambiguity always keeps ordinary notes."""
    result = Recognition()
    if not mapping.auto_staccato and not mapping.auto_grace:
        result.skipped["识别已关闭"] = len(sources)
        return result
    step = Fraction(4, mapping.quantization)
    tick = Fraction(1, ppq)
    reasons: dict[str, str] = {}
    excluded: set[str] = set()
    explicit: set[str] = set()
    lanes: dict[tuple[str, int], dict[int, list[NoteEvent]]] = defaultdict(lambda: defaultdict(list))
    for source in sources:
        technique = source_technique(source, mapping)
        reason = ""
        if mapping.percussion:
            reason = "打击乐"
        elif source.slide:
            reason = "Slide"
        elif technique in ("pizz", "pizzicato", "legato", "tenuto", "连奏", "保持音"):
            reason = "明确奏法"
        elif _gate(source, ppq) is None:
            reason = "未知或零门控"
        if technique in ("staccato", "spiccato", "跳音", "跳弓"):
            explicit.add(source.note_id)
            if mapping.auto_staccato and not mapping.percussion and not source.slide:
                result.notes[source.note_id] = NoteInterpretation(staccato=True)
        if reason:
            reasons[source.note_id] = reason
            excluded.add(source.note_id)
        lanes[(source.track_id, source.midi_channel)][source.start_tick].append(source)

    for onsets in lanes.values():
        groups = [onsets[start] for start in sorted(onsets)]
        blocked: set[int] = set()
        last_gate_end = -1
        for index, group in enumerate(groups):
            start = group[0].start_tick
            releases = [source.key_release_tick for source in group]
            if (last_gate_end > start + 1 or any(end is None for end in releases)
                    or (all(end is not None for end in releases) and max(releases) - min(releases) > 1)):
                blocked.add(index)
                for source in group:
                    reasons.setdefault(source.note_id, "复调或不同门控和弦")
            last_gate_end = max(last_gate_end, *(source.key_release_tick if source.key_release_tick is not None else source.end_tick for source in group))

        grace_ids: set[str] = set()
        if mapping.auto_grace:
            for index, group in enumerate(groups[:-1]):
                following = groups[index + 1]
                if len(group) != 1 or len(following) != 1 or index in blocked or index + 1 in blocked:
                    continue
                candidate, main = group[0], following[0]
                if candidate.note_id in excluded or main.note_id in excluded or candidate.note_id in explicit:
                    continue
                gate, main_gate = _gate(candidate, ppq), _gate(main, ppq)
                onset_gap = Fraction(main.start_tick - candidate.start_tick, ppq)
                release_gap = Fraction(main.start_tick - candidate.key_release_tick, ppq)
                raw_start, main_start = Fraction(candidate.start_tick, ppq), Fraction(main.start_tick, ppq)
                if not (gate <= Fraction(1, 4) and float(gate) * 60 / bpm <= 0.12
                        and 0 < onset_gap <= Fraction(1, 2) and float(onset_gap) * 60 / bpm <= 0.18
                        and -tick <= release_gap and float(release_gap) * 60 / bpm <= 0.08
                        and 1 <= abs(candidate.pitch - main.pitch) <= 2
                        and candidate.velocity <= main.velocity
                        and main_gate >= max(step, 3 * gate)
                        and abs(main_start - quantize(main_start)) <= min(Fraction(1, 32), step / 8)
                        and abs(raw_start - quantize(raw_start)) >= step / 4):
                    continue
                if _regular_short_note(candidate, ppq, mapping.triplets):
                    reasons.setdefault(candidate.note_id, "规则短音保留")
                    continue
                if index and len(groups[index - 1]) == 1:
                    previous = groups[index - 1][0]
                    previous_gate = _gate(previous, ppq)
                    if previous_gate is not None and previous_gate <= Fraction(1, 4) and candidate.start_tick - previous.start_tick <= ppq / 2:
                        reasons.setdefault(candidate.note_id, "连续短音保留")
                        continue
                grace_ids.add(candidate.note_id)
                result.notes[candidate.note_id] = NoteInterpretation(grace_main_id=main.note_id)

        if mapping.auto_staccato:
            ordinary = [group for group in groups if not any(source.note_id in grace_ids for source in group)]
            starts = [quantize(Fraction(group[0].start_tick, ppq)) for group in ordinary]
            # Recompute overlap for ordinary groups so a removed grace never starts a run.
            ordinary_blocked = {id(groups[index]) for index in blocked}

            def eligible(group: list[NoteEvent], slot: Fraction) -> bool:
                return (id(group) not in ordinary_blocked
                        and not any(source.note_id in excluded for source in group)
                        and all(Fraction(35, 100) <= _gate(source, ppq) / slot <= Fraction(65, 100) for source in group))

            for index in range(len(ordinary) - 3):
                slot = starts[index + 1] - starts[index]
                if not Fraction(1, 4) <= slot <= 2:
                    continue
                if any(starts[index + j + 1] - starts[index + j] != slot for j in range(3)):
                    continue
                if not all(eligible(ordinary[index + j], slot) for j in range(3)):
                    continue
                for offset in range(3):
                    for source in ordinary[index + offset]:
                        result.notes[source.note_id] = NoteInterpretation(staccato=True, rhythmic_end=starts[index + offset + 1])
                # The final attack follows a demonstrated pattern, but never extends
                # past known score time or into the next attack after this run.
                terminal = index + 3
                terminal_end = starts[terminal] + slot
                limit = starts[terminal + 1] if terminal + 1 < len(starts) else Fraction(end_tick, ppq)
                if terminal_end <= limit and eligible(ordinary[terminal], slot):
                    for source in ordinary[terminal]:
                        result.notes[source.note_id] = NoteInterpretation(staccato=True, rhythmic_end=terminal_end)

    for source in sources:
        interpretation = result.notes.get(source.note_id)
        if interpretation is None:
            reasons.setdefault(source.note_id, "节奏或门控证据不足")
        elif source.note_id in reasons and interpretation.grace_main_id:
            reasons.pop(source.note_id)
    result.skipped.update(reasons.values())
    return result
