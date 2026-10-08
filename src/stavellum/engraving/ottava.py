"""Choose octave lines for readable display without changing source pitches.

Thresholds here are conservative application heuristics, not universal engraving
rules. All positions are diatonic staff steps, and all times are quarter-note beats.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from fractions import Fraction
from itertools import groupby
from math import isclose


@dataclass(frozen=True, slots=True)
class OctaveNote:
    source_id: str
    start: Fraction
    end: Fraction
    diatonic: int
    grace_main_id: str = ""


@dataclass(frozen=True, slots=True)
class OctaveSpan:
    part_id: str
    staff_index: int
    start_beat: Fraction
    end_beat: Fraction
    octaves: int
    start_element_id: str = ""
    end_element_id: str = ""

    @property
    def label(self) -> str:
        return {1: "8va", -1: "8vb", 2: "15ma", -2: "15mb"}[self.octaves]


_SHIFTS = (0, 1, -1, 2, -2)
_LINE_COSTS = (0, 12, 12, 20, 20)


def _ledger(diatonic: int, lowest_line: int) -> int:
    return max(0, lowest_line - diatonic, diatonic - lowest_line - 8) // 2


def _directional_ledger(diatonic: int, lowest_line: int, direction: int) -> int:
    distance = diatonic - lowest_line - 8 if direction > 0 else lowest_line - diatonic
    return max(0, distance) // 2


@dataclass(frozen=True, slots=True)
class _Atom:
    notes: tuple[OctaveNote, ...]
    start: Fraction
    end: Fraction
    main_start: Fraction | None
    main_end: Fraction | None
    onset_count: int
    note_count: int
    costs: tuple[float, ...]
    ledgers: tuple[int, ...]
    severe: tuple[int, ...]
    single_severe: tuple[int, ...]
    safe: tuple[bool, ...]


def _make_atom(columns: list[tuple[Fraction, tuple[OctaveNote, ...]]],
               lowest_line: int) -> _Atom:
    notes = tuple(item for _, members in columns for item in members)
    main_notes = tuple(item for item in notes if not item.grace_main_id)
    costs = [0.0] * len(_SHIFTS)
    ledgers = [0] * len(_SHIFTS)
    severe = [0] * len(_SHIFTS)
    single_severe = [0] * len(_SHIFTS)
    safe = [True] * len(_SHIFTS)
    onset_count = 0
    for _, members in columns:
        mains = tuple(item for item in members if not item.grace_main_id)
        if not mains:
            continue
        onset_count += 1
        for index, shift in enumerate(_SHIFTS):
            values = [_ledger(item.diatonic - 7 * shift, lowest_line) for item in mains]
            costs[index] += sum(value * value for value in values) / len(values)
            if shift:
                direction = 1 if shift > 0 else -1
                severe[index] += any(
                    _directional_ledger(item.diatonic, lowest_line, direction) >= 3
                    for item in mains)
                single_severe[index] += any(
                    _directional_ledger(item.diatonic - 7 * direction,
                                        lowest_line, direction) >= 3
                    for item in mains)
    # Graces participate in every safety check, but cannot create evidence for
    # an octave line by inflating onset counts, duration, or readability costs.
    for index, shift in enumerate(_SHIFTS):
        # The mean ledger reduction is per note, including every grace, rather
        # than per main onset. Ornaments cannot dilute this acceptance rule.
        ledgers[index] = sum(_ledger(item.diatonic - 7 * shift, lowest_line)
                             for item in notes)
        if shift:
            safe[index] = all(
                (shifted := _ledger(item.diatonic - 7 * shift, lowest_line)) <= 2
                and shifted <= _ledger(item.diatonic, lowest_line) + 1
                for item in notes)
    return _Atom(
        notes, min(item.start for item in notes), max(item.end for item in notes),
        min((item.start for item in main_notes), default=None),
        max((item.end for item in main_notes), default=None),
        onset_count, len(notes), tuple(costs), tuple(ledgers),
        tuple(severe), tuple(single_severe), tuple(safe))


def _atoms(notes: list[OctaveNote], lowest_line: int) -> list[_Atom]:
    ordered = sorted(notes, key=lambda item: (
        item.start, not bool(item.grace_main_id), item.source_id, item.diatonic, item.end))
    columns = [(start, tuple(members)) for start, members in groupby(
        ordered, key=lambda item: item.start)]
    starts = [start for start, _ in columns]
    protected_end = list(range(len(columns)))
    source_columns = {item.source_id: index for index, (_, members) in enumerate(columns)
                      for item in members if not item.grace_main_id}
    for index, (_, members) in enumerate(columns):
        for item in members:
            # A sustained note forbids changing octave before its end. Notes
            # touching at end/start do not overlap and may use different lines.
            protected_end[index] = max(
                protected_end[index], bisect_left(starts, item.end) - 1)
            main_index = source_columns.get(item.grace_main_id)
            if item.grace_main_id and main_index is not None:
                first, last = sorted((index, main_index))
                protected_end[first] = max(protected_end[first], last)
    result = []
    start = 0
    while start < len(columns):
        end = protected_end[start]
        cursor = start
        while cursor <= end:
            end = max(end, protected_end[cursor])
            cursor += 1
        result.append(_make_atom(columns[start:end + 1], lowest_line))
        start = end + 1
    return result


def _infer_block(atoms: list[_Atom], part_id: str, staff_index: int) -> list[OctaveSpan]:
    """Solve all eligible intervals jointly in O(4 * atomic_groups squared)."""
    count = len(atoms)
    onsets = [0]
    note_counts = [0]
    costs = [[0.0] for _ in _SHIFTS]
    ledgers = [[0] for _ in _SHIFTS]
    severe = [[0] for _ in _SHIFTS]
    single_severe = [[0] for _ in _SHIFTS]
    last_unsafe = [[-1] for _ in _SHIFTS]
    main_ends: list[Fraction | None] = [None]
    for atom_index, atom in enumerate(atoms):
        onsets.append(onsets[-1] + atom.onset_count)
        note_counts.append(note_counts[-1] + atom.note_count)
        main_ends.append(atom.main_end if atom.main_end is not None else main_ends[-1])
        for index in range(len(_SHIFTS)):
            costs[index].append(costs[index][-1] + atom.costs[index])
            ledgers[index].append(ledgers[index][-1] + atom.ledgers[index])
            severe[index].append(severe[index][-1] + atom.severe[index])
            single_severe[index].append(single_severe[index][-1] + atom.single_severe[index])
            last_unsafe[index].append(
                atom_index if not atom.safe[index] else last_unsafe[index][-1])
    main_starts: list[Fraction | None] = [None] * (count + 1)
    for index in range(count - 1, -1, -1):
        main_starts[index] = (atoms[index].main_start if atoms[index].main_start is not None
                              else main_starts[index + 1])
    duration_starts = [start if start is not None else atoms[-1].end + 1
                       for start in main_starts[:-1]]
    # At least four onsets at 60% require three extreme onsets. These tests
    # eliminate impossible states without enumerating any candidate intervals.
    candidates = [index for index, shift in enumerate(_SHIFTS[1:], 1)
                  if severe[index][-1] >= 3
                  and (abs(shift) == 1 or single_severe[index][-1] >= 3)]
    if not candidates:
        return []
    # Convert interval inequalities to comparisons of prefix balances, keeping
    # Fraction arithmetic and repeated sums out of the quadratic inner loop.
    severe_balance = [[5 * value - 3 * onsets[position]
                       for position, value in enumerate(values)] for values in severe]
    single_balance = [[5 * value - 3 * onsets[position]
                       for position, value in enumerate(values)] for values in single_severe]
    benefit_balance = [[ledgers[0][position] - value - note_counts[position]
                        for position, value in enumerate(values)] for values in ledgers]

    best_costs = [0.0] * (count + 1)
    # Equal-cost results prefer no lines, then fewer lines, smaller octave shifts,
    # and fewer shifted main onsets. Iteration order provides a stable last tie.
    best_ties = [(0, 0, 0)] * (count + 1)
    previous = [(0, 0)] * (count + 1)
    for end in range(1, count + 1):
        best_costs[end] = best_costs[end - 1] + atoms[end - 1].costs[0]
        best_ties[end] = best_ties[end - 1]
        previous[end] = (end - 1, 0)
        last_start = bisect_right(onsets, onsets[end] - 4, hi=end) - 1
        main_end = main_ends[end]
        if last_start < 0 or main_end is None:
            continue
        last_start = min(last_start, bisect_right(duration_starts, main_end - 1, hi=end) - 1)
        for index in candidates:
            shift = _SHIFTS[index]
            first_start = last_unsafe[index][end] + 1
            shifted_cost = costs[index]
            interval_cost = shifted_cost[end] + _LINE_COSTS[index]
            extreme_limit = severe_balance[index][end]
            single_limit = single_balance[index][end]
            benefit_limit = benefit_balance[index][end]
            for start in range(last_start, first_start - 1, -1):
                if severe_balance[index][start] > extreme_limit:
                    continue
                if abs(shift) == 2 and single_balance[index][start] > single_limit:
                    continue
                if benefit_balance[index][start] > benefit_limit:
                    continue
                cost = best_costs[start] - shifted_cost[start] + interval_cost
                old_ties = best_ties[start]
                ties = (old_ties[0] + 1, old_ties[1] + abs(shift),
                        old_ties[2] + onsets[end] - onsets[start])
                equal = isclose(cost, best_costs[end], rel_tol=1e-12, abs_tol=1e-9)
                if (cost < best_costs[end] and not equal) or (equal and ties < best_ties[end]):
                    best_costs[end], best_ties[end], previous[end] = cost, ties, (start, shift)

    selected = []
    end = count
    while end:
        start, shift = previous[end]
        if shift:
            selected.append((start, end, shift))
        end = start
    selected.reverse()
    merged: list[tuple[int, int, int]] = []
    for start, end, shift in selected:
        if merged and merged[-1][1] == start and merged[-1][2] == shift:
            merged[-1] = (merged[-1][0], end, shift)
        else:
            merged.append((start, end, shift))
    result = []
    for start, end, shift in merged:
        members = tuple(item for atom in atoms[start:end] for item in atom.notes)
        last = max(members, key=lambda item: (item.end, item.start, item.source_id))
        result.append(OctaveSpan(
            part_id, staff_index, atoms[start].start, atoms[end - 1].end, shift,
            atoms[start].notes[0].source_id, last.source_id))
    return result


def infer_ottavas(notes: list[OctaveNote], lowest_line: int,
                  part_id: str, staff_index: int) -> list[OctaveSpan]:
    """Infer 8va/8vb/15ma/15mb spans using global interval optimization.

    Positive octaves lower the displayed pitches by ``7 * octaves`` staff steps.
    Chords, sustained overlaps, and a grace with its main note are indivisible.
    Silence longer than one beat always ends a block; shorter silence may be
    included in a line, but leading/trailing silence is never added to a span.
    """
    if not notes:
        return []
    atoms = _atoms(notes, lowest_line)
    result = []
    start = 0
    for index in range(1, len(atoms)):
        if atoms[index].start - atoms[index - 1].end > 1:
            result.extend(_infer_block(atoms[start:index], part_id, staff_index))
            start = index
    result.extend(_infer_block(atoms[start:], part_id, staff_index))
    return result
