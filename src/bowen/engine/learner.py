"""The learner — the automatic channel repeats what relieved (Phase C step 7, plan D4).

Purpose: credit each automatic act with the felt change in anxiety over a short
         horizon after it — the actor's own and, weighted, its target's and
         witnesses' — habituated for repetition, and move the act's learned value
         toward that signal.
Spec:    docs/bowen_agent_model_spec_v2.md#M4.D.6, #M4.D.6a, #M4.D.6b, #M4.D.6d, #M4.D.6e, #M4.G.3, #M4.B.3
Tests:   tests/bowen/test_learner.py

Every form is the project's, graded [I]; the constants are in ``config/bowen/constants.md``.

**What is credited.** At step 8 every automatic act the policy emitted becomes an
*eligible act* of its actor, under the learned-value key the policy chose it by
(band, triangle position, act, target or triad). The self-directed channel —
`I-POSITION`, `STAY-IN-CONTACT`, the `M5.B` family moves, ``WITHHOLD`` — is never
registered, so it is never reinforced (`M4.D.6d`); nor are scripted or fallback outcomes.

**The signal** (`M4.D.6a`, `M4.D.6e`). At the end of every tick (step 9, after
consolidation) each person's change in acute anxiety over the tick, ``Δ``, is known. An
eligible act of age ``k`` (0 in the tick it was sent) gains

    credit_discount ** k × ( −Δ(actor) + cross_person_weight × bond-weighted mean −Δ(target, witnesses) )

except that a ``TRIANGLE``'s recruited third is left out of the mean (owner decision 2026-10-09). *Revision 26 (Phase
D B9; Claude's recommendation, approved 2026-10-16):* the mean is weighted by the actor's bond energy on its tie to
each other person, so whose calming the act is credited with follows the tie: a child's tie to a parent outweighs a
thin tie to a witness, which is "her calming is the reinforcement" (`M4.D.6e`) without naming a parent in any rule.
Another person with no tie to the actor carries no weight.

Relief is positive. The cross-person term is how a child learns that acting as the
parent's image predicts calms the parent (`FE07.4`). Nothing outside ``credit_horizon``
ticks is credited: an act's account closes when its age reaches ``credit_horizon − 1``,
so a cost that arrives later reaches the policy only as felt anxiety credited to whatever
acts are inside the horizon then (`M4.B.3`). Note that decay toward the chronic floor is part
of ``Δ``: an anxious person's every act shares in it.

**Habituation** (`M4.G.3`). When the account closes, a positive signal — relief — is scaled by
``habituation_rate ** n``, where ``n`` is how many identical acts (same kind, same target)
the actor sent in the ``habituation_window`` ticks before this one. A cost is not habituated.

**The update** (plan D4): ``value ← value + learning_rate × (signal − value)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.bowen.engine.act import Selection
from src.bowen.engine.contact import anxiety_band
from src.bowen.engine.draws import DrawKey
from src.bowen.engine.events import EventId, Mechanism, Role as DeliveryRole
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.log_records import DecidedBy, EffectRecord
from src.bowen.engine.objects import SCALE_MAX, Role
from src.bowen.engine.params import EngineParams
from src.bowen.engine.recompute import TRIANGLE_KIND
from src.bowen.engine.state import RunState
from src.bowen.engine.value_keys import POSITIONS, act_of, learned_value_key
from src.bowen.engine.counting import count


@dataclass
class EligibleAct:
    """One automatic act still inside its credit horizon."""

    tick: int
    key: str
    others: tuple[PersonId, ...]   # its target and witnesses
    repetitions: int
    signal: float = 0.0


def repetitions(state: RunState, actor: PersonId, kind: str, targets: tuple[PersonId, ...], params: EngineParams) -> int:
    """How many identical acts the actor sent in the habituation window before this tick."""
    return count(
        1 for e in state.store.sent_by(actor)
        if e.mechanism is Mechanism.MOVE and e.kind == kind and e.targets == targets
        and state.tick - params.habituation_window <= e.timestamp < state.tick
    )


def register_acts(state: RunState, selections: tuple[Selection, ...], params: EngineParams) -> None:
    """Purpose: make each emitted automatic act of the policy an eligible act of its actor.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.D.6, #M4.D.6d
    Tests:   tests/bowen/test_learner.py::test_m4d6d_self_directed_channel_is_never_reinforced,
             tests/bowen/test_learner.py::test_m4d6e_triangle_credit_excludes_recruited_third
    """
    for selection in sorted(selections, key=lambda s: (s.actor, s.index)):
        if selection.decided_by is not DecidedBy.POLICY or not selection.value_key:
            continue
        event = state.store.event(EventId(state.tick, str(selection.actor), selection.index))
        # Owner decision 2026-10-09: a TRIANGLE is credited by the seeker's own relief; the third it recruits is
        # not in the cross-person term (M4.D.6e's term is the projection account, not a recruit's distress).
        recruited = set(event.targets) if event.kind == TRIANGLE_KIND else set()
        others = tuple(p for p in (*event.targets, *event.witnesses) if state.people[p].alive and p not in recruited)
        state.people[selection.actor].eligible_acts.append(EligibleAct(
            tick=state.tick, key=selection.value_key, others=others,
            repetitions=repetitions(state, selection.actor, event.kind, event.targets, params),
        ))


def cross_person_relief(state: RunState, actor: PersonId, others, relief: dict[PersonId, float]) -> float:
    """Purpose: the others' relief after an act, weighted by the actor's bond energy on its tie to each (revision 26).
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.D.6e
    Tests:   tests/bowen/test_learner.py::test_m4d6e_the_cross_person_term_weighs_the_closer_tie_more
    """
    weights = {}
    for q in others:
        tie = state.tie_between(actor, q)
        if q in relief and tie is not None and tie.bond_energy > 0:
            weights[q] = tie.bond_energy
    total = math.fsum(weights.values())
    return math.fsum(w * relief[q] for q, w in weights.items()) / total if total else 0.0


def learn(state: RunState, start_acute: dict[PersonId, float], params: EngineParams) -> list[EffectRecord]:
    """Purpose: credit this tick's felt change to every eligible act, and close the accounts that are due.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.D.6a, #M4.D.6b, #M4.D.6e, #M4.G.3, #M4.B.3
    Tests:   tests/bowen/test_learner.py::test_m4d6_an_act_followed_by_relief_is_reinforced
    """
    relief = {pid: start_acute[pid] - p.acute_anxiety for pid, p in state.people.items() if pid in start_acute}
    changes = []
    for pid in sorted(state.people):
        person = state.people[pid]
        if not person.alive:
            person.eligible_acts = []
            continue
        still_open = []
        for act in person.eligible_acts:
            age = state.tick - act.tick
            felt = relief.get(pid, 0.0) + params.cross_person_weight * cross_person_relief(state, pid, act.others, relief)
            act.signal += params.credit_discount ** age * felt
            if age < params.credit_horizon - 1:
                still_open.append(act)
                continue
            signal = act.signal * params.habituation_rate ** act.repetitions if act.signal > 0 else act.signal
            value = person.learned_values.get(act.key, 0.0)
            delta = params.learning_rate * (signal - value)
            person.learned_values[act.key] = value + delta
            if delta:
                changes.append((pid, f"learned_value[{act.key}]", delta))
        person.eligible_acts = still_open
    return [EffectRecord(state.tick, "learning", None, people=tuple(changes))] if changes else []


def soothing_review(state: RunState, params: EngineParams) -> list[EffectRecord]:
    """Purpose: now and then a person asks "is soothing helping me?" — and if it did not, soothing loses its appeal.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.D.6 (revision 19)
    Tests:   tests/bowen/test_learner.py::test_m4d6_a_soothing_review_lowers_soothing_that_did_not_help

    The owner, 2026-10-11: "If soothing outnumbers coping, that's often because of lower basic level. Some people
    learn, eventually, that soothing isn't really the answer. … from time to time … an agent should ask — is soothing
    helping me?" Every ``soothing_review_every`` weeks each person asks with probability
    ``(basic_level / 100 + systems_perspective) / 2`` (one keyed draw). Looking back over that stretch, soothing
    helped if their load ``practice_window`` weeks after soothing was lower than when they soothed. If it did not,
    every positive learned value of an avoiding or soothing act is cut by ``soothing_review_penalty`` of itself.
    The automatic channel otherwise learns only from immediate relief (`M4.D.6`), which soothing always gives.
    """
    if (state.tick + 1) % params.soothing_review_every:
        return []
    rows = []
    for pid in sorted(state.people):
        person = state.people[pid]
        weeks = list(state.estimator_history.get(pid, ()))[-params.soothing_review_every - params.practice_window:]
        if not person.alive or not weeks:
            continue
        signals = (person.basic_level / SCALE_MAX, person.systems_perspective or 0.0)
        asks = min(1.0, math.fsum(signals) / len(signals))
        draw = state.draws.uniform(DrawKey.make("soothing_review", tick=state.tick, actor=pid, purpose="ask", index=0))
        if draw >= asks:
            continue
        by_tick = {w.tick: w.load for w in weeks}
        pairs = [(w.load, by_tick[w.tick + params.practice_window]) for w in weeks
                 if w.soothed and w.tick + params.practice_window in by_tick]
        if not pairs or math.fsum(after for _, after in pairs) < math.fsum(before for before, _ in pairs):
            continue  # soothing helped, or there was none to judge
        for key in sorted(person.learned_values):
            value = person.learned_values[key]
            if value > 0 and act_of(key) in state.kinds.soothing:
                person.learned_values[key] = value * (1.0 - params.soothing_review_penalty)
                rows.append((pid, f"learned_value[{key}]", person.learned_values[key] - value))
    return [EffectRecord(state.tick, "soothing_review", None, people=tuple(rows))] if rows else []


# --- M4.D.6f, revision 29: learning from what follows a witnessed cutoff -----------------------------------------

CUTOFF_KIND = "CUTOFF"  # M5.A.1 fixes the repertoire, so the name is the spec's, not editorial content


@dataclass
class Witnessed:
    """A cutoff one person witnessed, with how well each of their family was functioning then."""

    witness: PersonId
    tick: int
    then: tuple[tuple[PersonId, float], ...]


def _family_ties(state: RunState, pid: PersonId):
    """The witness's ties to living family members (an external agent is excluded)."""
    for tie in state.ties_of(pid):
        other = state.people[tie.id.other(pid)]
        if other.alive and other.role is Role.MEMBER:
            yield other.id, tie


def note_witnessed_cutoffs(state: RunState, deliveries) -> None:
    """Purpose: step 2 — a person delivered a CUTOFF as a witness keeps a record of how their family was functioning.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.D.6f
    Tests:   tests/bowen/test_cutoff_and_focus.py::test_m4d6f_a_family_that_functions_better_after_a_cutoff_teaches_cutoff
    """
    for delivery in deliveries:
        if delivery.role is not DeliveryRole.WITNESS or state.store.event(delivery.event_id).kind != CUTOFF_KIND:
            continue
        witness = state.people[delivery.recipient]
        if not witness.alive or witness.role is not Role.MEMBER:
            continue
        then = tuple((pid, state.people[pid].functional_level) for pid, _ in _family_ties(state, witness.id))
        state.witnessed.append(Witnessed(witness.id, state.tick, then))


def learn_from_witnessed(state: RunState, params: EngineParams) -> list[EffectRecord]:
    """Purpose: step 9 — ``witness_window`` weeks on, the witness learns from what followed a cutoff, not from the act.
    Spec:    docs/bowen_agent_model_spec_v2.md#M4.D.6f
    Tests:   tests/bowen/test_cutoff_and_focus.py::test_m4d6f_a_family_that_functions_worse_teaches_against_it

    The change seen is the bond-weighted mean change in functional level of the family the witness is tied to: the
    cutter, the person cut off and the others, read as it is (not through belief). Every learned value of ``CUTOFF``
    in the witness's current band, in both positions, toward each family member they are tied to, takes a step of
    ``witnessed_cutoff_gain`` toward it, the form of `M4.D.6a`'s update, so it stays bounded. The kind of cutoff is not
    read: one that lets the cutter function up teaches that cutoff is good (the owner, 2026-10-13).
    """
    due = [w for w in state.witnessed if state.tick - w.tick >= params.witness_window]
    if not due:
        return []
    state.witnessed = [w for w in state.witnessed if state.tick - w.tick < params.witness_window]
    rows = []
    for record in due:
        person = state.people[record.witness]
        if not person.alive:
            continue
        now = dict(_family_ties(state, record.witness))
        weights = {pid: now[pid].bond_energy for pid, _ in record.then if pid in now}
        total = math.fsum(weights.values())
        if total <= 0:
            continue
        seen = math.fsum(weights[pid] / total * (state.people[pid].functional_level - level)
                   for pid, level in record.then if pid in weights)
        band_now = anxiety_band(person, params)
        for other in sorted(now):
            for position in POSITIONS:
                key = learned_value_key(band_now, position, CUTOFF_KIND, other.value)
                value = person.learned_values.get(key, 0.0)
                person.learned_values[key] = value + params.witnessed_cutoff_gain * (seen - value)
                rows.append((record.witness, f"learned_value[{key}]", person.learned_values[key] - value))
    return [EffectRecord(state.tick, "witnessed_cutoff", None, people=tuple(rows))] if rows else []
