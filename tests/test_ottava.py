from dataclasses import FrozenInstanceError
from fractions import Fraction
from random import Random

import pytest

from stavellum.engraving.ottava import OctaveNote, OctaveSpan, infer_ottavas


def sequence(pitches, *, start=Fraction(0), step=Fraction(1), duration=None, prefix="n"):
    if isinstance(pitches, int):
        pitches = [pitches] * 4
    duration = step if duration is None else duration
    return [OctaveNote(f"{prefix}{index}", start + index * step,
                       start + index * step + duration, pitch)
            for index, pitch in enumerate(pitches)]


def infer(notes, *, lowest_line=0):
    return infer_ottavas(notes, lowest_line, "piano", 1)


@pytest.mark.parametrize(("diatonic", "octaves", "label"), [
    (14, 1, "8va"), (-6, -1, "8vb"), (24, 2, "15ma"), (-16, -2, "15mb"),
])
def test_all_four_directions_preserve_source_notes(diatonic, octaves, label):
    notes = sequence(diatonic)
    before = list(notes)
    spans = infer(notes)
    assert spans == [OctaveSpan("piano", 1, Fraction(0), Fraction(4), octaves, "n0", "n3")]
    assert spans[0].label == label
    assert notes == before
    with pytest.raises(FrozenInstanceError):
        notes[0].diatonic = 0
    with pytest.raises(FrozenInstanceError):
        spans[0].octaves = 0


def test_staff_position_is_relative_to_the_actual_lowest_line():
    assert infer(sequence(29), lowest_line=15)[0].octaves == 1
    assert infer(sequence(9), lowest_line=15)[0].octaves == -1


@pytest.mark.parametrize("pitches", [[], [0, 2, 4, 6, 8, 6, 4, 2], [14], [8, 8, 24, 8, 8]])
def test_empty_ordinary_and_isolated_extreme_notes_do_not_get_lines(pitches):
    assert infer(sequence(pitches)) == []


def test_requires_four_distinct_main_onsets():
    assert infer(sequence([14] * 3, step=Fraction(1, 2))) == []
    # Twelve chord notes still provide only three onsets.
    notes = [OctaveNote(f"c{onset}-{voice}", Fraction(onset), Fraction(onset + 1), 14)
             for onset in range(3) for voice in range(4)]
    assert infer(notes) == []


def test_requires_one_beat_without_rounding_fractional_boundaries():
    assert infer(sequence(14, step=Fraction(1, 4)))[0].end_beat == 1
    assert infer(sequence(14, step=Fraction(1, 8))) == []
    short = sequence(14, step=Fraction(1, 4), duration=Fraction(1, 4) - Fraction(1, 1000))
    assert infer(short) == []


def test_three_ledger_boundary_is_inclusive_and_two_ledgers_do_not_trigger():
    assert infer(sequence(13)) == []
    assert infer(sequence(14))[0].octaves == 1
    assert infer(sequence(-5)) == []
    assert infer(sequence(-6))[0].octaves == -1


def test_sixty_percent_onset_boundary_is_inclusive():
    spans = infer(sequence([14, 8, 14, 8, 14]))
    assert spans == [OctaveSpan("piano", 1, Fraction(0), Fraction(5), 1, "n0", "n4")]
    # Holding the middle-register note makes the whole passage indivisible,
    # so a qualifying shorter subinterval cannot hide a failing majority.
    notes = sequence([14, 8, 14, 8, 14, 8])
    notes.append(OctaveNote("held", Fraction(0), Fraction(6), 8))
    assert infer(notes) == []


def test_short_return_to_normal_register_stays_inside_one_line():
    spans = infer(sequence([14, 14, 14, 5, 14, 14]))
    assert spans == [OctaveSpan("piano", 1, Fraction(0), Fraction(6), 1, "n0", "n5")]


def test_unsafe_return_to_normal_register_divides_the_regions():
    spans = infer(sequence([14, 14, 14, 14, 2, 14, 14, 14, 14]))
    assert spans == [
        OctaveSpan("piano", 1, Fraction(0), Fraction(4), 1, "n0", "n3"),
        OctaveSpan("piano", 1, Fraction(5), Fraction(9), 1, "n5", "n8"),
    ]


@pytest.mark.parametrize("gap", [Fraction(1, 2), Fraction(1)])
def test_short_rest_can_be_crossed_and_adjacent_equal_shifts_are_merged(gap):
    notes = sequence(14) + sequence(14, start=Fraction(4) + gap, prefix="b")
    assert infer(notes) == [OctaveSpan(
        "piano", 1, Fraction(0), Fraction(8) + gap, 1, "n0", "b3")]


def test_long_rest_breaks_lines_without_extending_over_terminal_rests():
    gap = Fraction(1001, 1000)
    notes = sequence(14, start=Fraction(3)) + sequence(14, start=Fraction(7) + gap, prefix="b")
    assert infer(notes) == [
        OctaveSpan("piano", 1, Fraction(3), Fraction(7), 1, "n0", "n3"),
        OctaveSpan("piano", 1, Fraction(7) + gap, Fraction(11) + gap, 1, "b0", "b3"),
    ]


def test_long_rest_does_not_combine_two_subthreshold_runs():
    notes = sequence([14] * 2) + sequence([14] * 2, start=Fraction(4), prefix="b")
    assert infer(notes) == []


def test_two_octaves_are_not_used_when_one_octave_already_clears_the_region():
    # 15ma would have lower raw layout cost, but fails the additional rule:
    # under 8va none of the columns still require three ledger lines.
    assert infer(sequence(19))[0].octaves == 1
    assert infer(sequence(-11))[0].octaves == -1
    assert infer(sequence(21))[0].octaves == 2
    assert infer(sequence(-13))[0].octaves == -2


def test_chords_shift_as_a_whole_and_use_onset_mean_cost():
    notes = sequence(14) + sequence(8, prefix="low")
    spans = infer(notes)
    assert len(spans) == 1 and spans[0].octaves == 1
    assert spans[0].start_beat == 0 and spans[0].end_beat == 4
    assert infer(sequence(14) + sequence(2, prefix="unsafe")) == []


def test_mean_improvement_of_one_ledger_per_note_is_required():
    notes = sequence([14] * 8)
    for voice in range(3):
        notes += sequence([8] * 8, prefix=f"voice{voice}-")
    # Each onset is extreme, and a line would reduce the objective. Diluting
    # the benefit across these chord tones still makes the interval ineligible.
    assert infer(notes) == []


def test_equal_cost_prefers_the_unshifted_score():
    notes = sequence(14) + sequence(8, prefix="a") + sequence(8, prefix="b")
    # Four onset means of (3**2 + 0 + 0) / 3 total 12: exactly one 8va line.
    assert infer(notes) == []


def test_held_normal_note_prevents_starting_an_unsafe_line_midway_through_it():
    notes = sequence(14, start=Fraction(1))
    notes.append(OctaveNote("held", Fraction(0), Fraction(5), 2))
    assert infer(notes) == []


def test_safe_held_note_is_included_in_the_indivisible_passage():
    notes = sequence(14, start=Fraction(1))
    notes.append(OctaveNote("held", Fraction(0), Fraction(5), 5))
    spans = infer(notes)
    assert spans == [OctaveSpan("piano", 1, Fraction(0), Fraction(5), 1, "held", "n3")]


def test_overlapping_polyphony_prevents_hiding_an_unsafe_sustained_note():
    notes = sequence([14] * 6, duration=Fraction(3, 2))
    notes.append(OctaveNote("other-voice", Fraction(2), Fraction(3), 2))
    assert infer(notes) == []


def test_touching_note_ends_are_not_treated_as_sustained_overlap():
    notes = [OctaveNote("low", Fraction(0), Fraction(1), 2)]
    notes += sequence(14, start=Fraction(1))
    assert infer(notes) == [OctaveSpan("piano", 1, Fraction(1), Fraction(5), 1, "n0", "n3")]


def test_graces_do_not_inflate_onset_count_or_create_duration():
    notes = sequence([14] * 3)
    notes.append(OctaveNote("grace", Fraction(0), Fraction(0), 14, "n0"))
    assert infer(notes) == []
    notes = sequence(14, step=Fraction(1, 8))
    notes.append(OctaveNote("early-grace", Fraction(-1), Fraction(-1), 14, "n0"))
    assert infer(notes) == []


def test_grace_is_shifted_and_kept_with_its_main_note():
    notes = sequence(14)
    notes.append(OctaveNote("grace", Fraction(0), Fraction(0), 15, "n0"))
    assert infer(notes) == [OctaveSpan("piano", 1, Fraction(0), Fraction(4), 1, "grace", "n3")]


@pytest.mark.parametrize("graces_per_onset,accepted", [(2, True), (3, False)])
def test_graces_count_in_the_mean_ledger_reduction_without_adding_onsets(
    graces_per_onset, accepted,
):
    notes = sequence(14)
    notes += [OctaveNote(f"grace{index}-{ornament}", item.start, item.start, 8, item.source_id)
              for index, item in enumerate(notes) for ornament in range(graces_per_onset)]
    # The four main notes save twelve ledger lines in total; the graces save
    # none. Twelve total notes meet the mean of one, while sixteen fail it.
    assert bool(infer(notes)) is accepted


def test_unsafe_grace_prevents_its_main_onset_from_being_shifted():
    notes = sequence([14] * 5)
    notes.append(OctaveNote("grace", Fraction(0), Fraction(0), 2, "n0"))
    assert infer(notes) == [OctaveSpan("piano", 1, Fraction(1), Fraction(5), 1, "n1", "n4")]


def test_very_extreme_graces_participate_in_the_two_ledger_safety_cap():
    notes = sequence(14)
    notes += [OctaveNote(f"grace{index}", item.start, item.start, 29, item.source_id)
              for index, item in enumerate(notes)]
    assert infer(notes) == []


def test_sequence_optimization_can_bridge_two_runs_that_cannot_qualify_separately():
    notes = sequence([14, 14, 14, 8, 8, 14, 14, 14])
    assert infer(notes) == [OctaveSpan("piano", 1, Fraction(0), Fraction(8), 1, "n0", "n7")]


def test_global_optimization_ends_one_octave_early_to_enable_a_better_two_octave_run():
    notes = sequence([14, 14, 14, 19, 19, 19, 21, 21, 21])
    # A greedy 8va would consume the first six onsets, leaving only three for
    # 15ma. Ending after four gives a globally cheaper valid five-onset 15ma.
    assert infer(notes) == [
        OctaveSpan("piano", 1, Fraction(0), Fraction(4), 1, "n0", "n3"),
        OctaveSpan("piano", 1, Fraction(4), Fraction(9), 2, "n4", "n8"),
    ]


def test_output_is_deterministic_for_input_order_and_does_not_mutate_it():
    notes = sequence(14) + sequence(8, prefix="chord")
    expected = infer(notes)
    rng = Random(42)
    for _ in range(10):
        rng.shuffle(notes)
        before = list(notes)
        assert infer(notes) == expected
        assert notes == before
