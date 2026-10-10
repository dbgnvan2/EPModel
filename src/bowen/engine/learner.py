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

    credit_discount ** k × ( −Δ(actor) + cross_person_weight × mean −Δ(target, witnesses) )

except that a ``TRIANGLE``'s recruited third is left out of the mean (owner decision 2026-10-09).

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

from dataclasses import dataclass

from src.bowen.engine.act import Selection
from src.bowen.engine.events import EventId, Mechanism
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.log_records import DecidedBy, EffectRecord
from src.bowen.engine.params import EngineParams
from src.bowen.engine.recompute import TRIANGLE_KIND
from src.bowen.engine.state import RunState


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
    return sum(
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
            others = [relief[q] for q in act.others if q in relief]
            felt = relief.get(pid, 0.0) + (
                params.cross_person_weight * sum(others) / len(others) if others else 0.0
            )
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
