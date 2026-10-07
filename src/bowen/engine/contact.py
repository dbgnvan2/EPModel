"""The felt contact state on each tie, and the two-sided appraisal function.

Purpose: hold, per person per tie, how much contact the person receives and how
         much they are being acted on, and turn the deviation from their felt
         contact optimum — in either direction — into anxiety.
Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.1, #M4.C.1a, #M4.C.1b, #M4.C.1c, #M1.B.4
Tests:   tests/bowen/test_contact.py

Spec revision 11 re-derived `M4.C.1` from Kerr's KS03.2 and KS06.3: anxiety rises
with deviation from a felt contact optimum on **either** side — too little
contact, too much impingement. The forms below are Phase C plan decision D2.
Every constant is invented ([I]) and lives in ``config/bowen/constants.md``.

    steepness(fl)  = SCALE_MAX / max(fl, floor)                      (M4.C.1a: falls as fl rises)
    band(fl)       = contact_band_max × fl / SCALE_MAX               (M4.C.1a: widens as fl rises)
    optimum        = min(1, bond_energy / SCALE_MAX
                            × (1 + anxiety_togetherness_gain × excess / SCALE_MAX))   (M4.C.1b)
    too little     = max(0, optimum − felt_contact − band)
    too much       = max(0, felt_impingement − band)
    resting contact = interactive_resting_contact × optimum on an interactive tie, 0 otherwise

``excess`` is acute anxiety above the chronic floor. Felt contact relaxes toward
its resting value each tick and felt impingement relaxes toward zero, so a
severed tie's "too little" side **builds over time** rather than arriving in
full at the act (`M4.C.1c`, revision-12 finding 6). The two drives are the same
for every person and are never learned (`M4.C.1`).
"""

from __future__ import annotations

from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.objects import SCALE_MAX, Person, Relationship
from src.bowen.engine.params import EngineParams


class ContactNotInitialised(RuntimeError):
    """A tie member has no felt contact state: ``initialise_contact`` was not called."""


def steepness(person: Person, params: EngineParams) -> float:
    """Purpose: how steeply anxiety rises with deviation; falls as functional level rises.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.1a
    Tests:   tests/bowen/test_contact.py::test_m4c1a_steepness_falls_and_band_widens_with_level
    """
    return SCALE_MAX / max(person.functional_level, params.functional_level_floor)


def band(person: Person, params: EngineParams) -> float:
    """Purpose: the deviation tolerated without anxiety; widens as functional level rises.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.1a
    Tests:   tests/bowen/test_contact.py::test_m4c1a_steepness_falls_and_band_widens_with_level
    """
    return params.contact_band_max * person.functional_level / SCALE_MAX


def excess(person: Person) -> float:
    return max(0.0, person.acute_anxiety - person.chronic_anxiety)


def optimum(person: Person, tie: Relationship, params: EngineParams) -> float:
    """Purpose: the person's felt contact optimum on a tie, moved toward closeness by anxiety.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.1, #M4.C.1b
    Tests:   tests/bowen/test_contact.py::test_m4c1b_anxiety_moves_the_optimum_toward_closeness
    """
    base = tie.bond_energy / SCALE_MAX
    return min(1.0, base * (1.0 + params.anxiety_togetherness_gain * excess(person) / SCALE_MAX))


def resting_contact(person: Person, tie: Relationship, params: EngineParams) -> float:
    if not tie.interactive:
        return 0.0
    return params.interactive_resting_contact * optimum(person, tie, params)


def _felt(table: dict[PersonId, float], person: Person) -> float:
    try:
        return table[person.id]
    except KeyError:
        raise ContactNotInitialised(f"{person.id} has no felt contact state on this tie") from None


def felt_contact(person: Person, tie: Relationship) -> float:
    return _felt(tie.felt_contact, person)


def felt_impingement(person: Person, tie: Relationship) -> float:
    return _felt(tie.felt_impingement, person)


def deviation_at(person: Person, tie: Relationship, params: EngineParams, contact: float, impingement: float) -> float:
    """Purpose: the two-sided deviation at given felt values, without reading or writing the tie's state.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.1
    Tests:   tests/bowen/test_contact.py::test_m4c1_deviation_counts_on_both_sides
    """
    little = max(0.0, optimum(person, tie, params) - contact - band(person, params))
    much = max(0.0, impingement - band(person, params))
    return little + much


def too_little(person: Person, tie: Relationship, params: EngineParams, spike: float = 0.0) -> float:
    """Purpose: the "too little contact" side of the deviation, plus any TRIGGER spike on the tie.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.1, #M4.C.1c, #M4.A.2
    Tests:   tests/bowen/test_contact.py::test_m4c1_deviation_counts_on_both_sides

    A TRIGGER re-opens the tie's absence in proportion to how much the tie matters:
    the spike adds ``spike × optimum`` (M4.A.2), [I].
    """
    gap = optimum(person, tie, params) - felt_contact(person, tie) - band(person, params)
    return max(0.0, gap) + spike * optimum(person, tie, params)


def too_much(person: Person, tie: Relationship, params: EngineParams) -> float:
    """Purpose: the "too much impingement" side of the deviation.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.1
    Tests:   tests/bowen/test_contact.py::test_m4c1_deviation_counts_on_both_sides
    """
    return max(0.0, felt_impingement(person, tie) - band(person, params))


def deviation(person: Person, tie: Relationship, params: EngineParams) -> float:
    return deviation_at(person, tie, params, felt_contact(person, tie), felt_impingement(person, tie))


def clamp_unit(value: float) -> float:
    return min(1.0, max(0.0, value))


def initialise_contact(people: dict[PersonId, Person], ties: dict, params: EngineParams) -> None:
    """Purpose: start every tie member at their resting contact, with no impingement.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.1c
    Tests:   tests/bowen/test_contact.py::test_m4c1c_initial_contact_is_resting_and_a_cut_tie_starts_empty

    A tie that is cut off at t0 starts with no contact, so its "too little" side
    is at its full size from the first tick. [I]: the spec does not say how long
    a tie declared cut off has been cut off.
    """
    for tie_id in sorted(ties):
        tie = ties[tie_id]
        for member in tie_id.members():
            person = people[member]
            tie.felt_contact[member] = resting_contact(person, tie, params)
            tie.felt_impingement[member] = 0.0


def relax_contact(people: dict[PersonId, Person], ties: dict, params: EngineParams) -> list[tuple]:
    """Purpose: tick step 9 — contact relaxes toward resting, impingement toward zero.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.1c, #M1.B.4
    Tests:   tests/bowen/test_contact.py::test_m4c1c_too_little_builds_over_time_on_a_severed_tie

    Returns ``(tie, field, person, change)`` rows for the log.
    """
    changes = []
    for tie_id in sorted(ties):
        tie = ties[tie_id]
        for member in tie_id.members():
            person = people[member]
            contact = felt_contact(person, tie)
            target = resting_contact(person, tie, params) if person.alive else contact
            moved = params.contact_relaxation_rate * (target - contact)
            tie.felt_contact[member] = clamp_unit(contact + moved)
            imp = felt_impingement(person, tie)
            shed = params.impingement_relaxation_rate * imp
            tie.felt_impingement[member] = clamp_unit(imp - shed)
            if moved:
                changes.append((tie_id, "felt_contact", member, moved))
            if shed:
                changes.append((tie_id, "felt_impingement", member, -shed))
    return changes
