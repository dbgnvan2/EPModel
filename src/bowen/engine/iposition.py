"""The `I-POSITION` state machine and the success state (Phase C step 8).

Purpose: run a person's differentiating sequence across weeks — prepare, define, meet
         the opposition, abort or hold, peak, resolve, follow up — from the person's
         own state and what is delivered to them; and run the assertion form when the
         person lacks the perspective or is angry.
Spec:    docs/bowen_agent_model_spec_v2.md#M5.D.1, #M5.D.2, #M5.D.2a, #M5.D.3, #M5.D.4, #M5.D.4a, #M5.D.5, #M5.D.6, #M5.D.7, #M5.D.7a, #M5.D.8, #M5.D.9, #M5.E.3, #M5.E.7, #M5.E.8, #M5.F.2a, #M5.F.4, #M1.C.5
Tests:   tests/bowen/test_iposition.py

Every form is the project's, graded [I]; the constants are in ``config/bowen/constants.md``.
The machine writes no outcome: each transition reads the mover's own excess anxiety,
functional level, efficacy and felt state on the tie, and the moves delivered to them.
Whether a sequence aborts, stalls, peaks or completes is a result.

**Starting** (`act.py`). The policy selects `I-POSITION` toward ``T``. If the mover's
``systems_perspective`` is below ``assertion_perspective_threshold``, or the mover is
angry — its "too much" side on the tie above ``anger_threshold`` (`M5.D.4a`: anger is
an intensity on the negative side of the appraisal, not a state) — it executes at once
as the **assertion form** (`M5.F.4`): an `I-POSITION` event flagged ``assertion``, which
delivers ``assertion_gain`` more impingement (it raises reactivity on the tie) and
raises the claimant's outward axis by ``assertion_evidence_gain`` (`M5.F.2a`: claiming
the position is evidence against it). Otherwise a sequence starts in ``PREPARE`` and
nothing is emitted that week (`M5.D.2a`).

**Each week, before selection** (``advance_sequences``):

* ``PREPARE`` — rehearsal lowers both of the mover's axes by ``rehearsal_rate`` of
  themselves (`M1.A.9`'s private-rehearsal input). It **fails** if the mover's excess
  anxiety rises past ``defence_threshold``. After ``prepare_ticks`` weeks ``DEFINE`` is
  due; if the issue comes up first — an impinging move from ``T`` arrives — ``DEFINE`` is
  due at once, less prepared. Less preparation leaves the axes higher and so the
  capacity to hold lower, which is how an unprepared sequence is less likely to peak.
* ``DEFINE`` (an outcome) — the genuine `I-POSITION` event to ``T``. On delivery it
  withdraws ``debit_gain`` of ``T``'s felt contact on the tie (`M5.E.7`: the move
  withdraws energy the other was receiving; *a contact proxy, since `life_energy` is not
  built*). The others' reaction is their own learned response (`M5.E.0`).
* ``OPPOSITION`` — waits for the first impinging move delivered to the mover from
  anyone. With none in ``opposition_window`` weeks the move **did not land** (`M5.E.3`),
  not a success. At the first opposition the mover holds if its excess anxiety is within
  its capacity, ``hold_gain × functional_level × efficacy``; otherwise ``ABORT`` is due.
* ``ABORT`` (an outcome) — defend, counterattack or go silent (`PURSUE`, `CONFLICT`,
  `DISTANCE`, [I]), drawn from the mover's own automatic values with a keyed draw; and
  the mover returns to the prior balance — the tie's functioning balance and the mover's
  two axes as they were when the sequence started (`M5.D.3`).
* ``HOLD`` — an angry mover **stalls**: no peak comes, and after ``stall_limit`` angry
  weeks the sequence lapses quietly (`M5.D.4`). A calm mover who receives an impinging
  move meets the **peak**: within capacity it resolves, beyond it ``ABORT`` is due. With
  no attack in ``hold_window`` weeks the sequence settles without a peak.
* ``RESOLVE`` — the opposition pulls up to the mover's level: ``T``'s functional level
  closes ``pull_up_rate`` of the gap to the mover's, if the mover's is higher (`M5.D.5`;
  solid self, outside `M6.I.4`). ``FOLLOW_UP`` is due the next week (`M5.D.6`).
* ``FOLLOW_UP`` (an outcome) — `STAY-IN-CONTACT` to ``T``. If it cannot be made (the tie
  is not live, or ``T`` has died) the pull-up is reverted and nothing completes. Made, the
  exchange is **complete** (`M5.D.7`): the mover's functional level rises by the small
  ``exchange_gain`` (`M5.D.7a`; ``basic_level`` is never written), every triangle holding
  the pair takes a permanent ``triangle_floor_decrement`` (`M1.C.5`), and the tie is left
  more solid (`M5.E.8`): both parties' axes fall by ``respect_gain`` of themselves and the
  tie's functioning habit returns to zero. The family's undifferentiation budget falls by
  ``exchange_budget_reduction`` (`M6.I.1`'s one logged sink, ``sinks.reduce_budget``).

**One outcome a week** (`M5.D.9`): on a week a step is due, the person's outcome is that
step; on other weeks the person selects normally, and an act toward ``T`` is made as
`STAY-IN-CONTACT`. A step that cannot be made that week is deferred one week and logged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.bowen.engine.contact import excess, too_much
from src.bowen.engine.events import Mechanism, Role
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.moves import DEFAULT_AREA
from src.bowen.engine.objects import SCALE_MAX, Person, TieState
from src.bowen.engine.outside_ness import axes, efficacy
from src.bowen.engine.params import EngineParams
from src.bowen.engine.sinks import reduce_budget
from src.bowen.engine.state import RunState

PREPARE, DEFINE, OPPOSITION, ABORT, HOLD, PEAK, RESOLVE, FOLLOW_UP = (
    "PREPARE", "DEFINE", "OPPOSITION", "ABORT", "HOLD", "PEAK", "RESOLVE", "FOLLOW_UP",
)
STATES = (PREPARE, DEFINE, OPPOSITION, ABORT, HOLD, PEAK, RESOLVE, FOLLOW_UP)  # M5.D.2, in order
OUTCOME_STEPS = frozenset({DEFINE, ABORT, FOLLOW_UP})
# M5.D.3's three abort branches, as acts: defend, counterattack, go silent. [I]
ABORT_ACTS = ("PURSUE", "CONFLICT", "DISTANCE")
I_POSITION, STAY_IN_CONTACT = "I-POSITION", "STAY-IN-CONTACT"


@dataclass
class Sequence:
    """One person's I-POSITION sequence in progress (M5.D)."""

    target: PersonId
    stage: str
    started: int
    stage_since: int
    prepared: int = 0
    stalls: int = 0
    due: str | None = None
    due_at: int | None = None
    start_balance: float = 0.0
    start_axes: tuple[float, float] = (0.0, 0.0)
    pull_up: float = 0.0
    deferrals: int = 0
    history: list[str] = field(default_factory=list)


def angry(state: RunState, mover: PersonId, target: PersonId, params: EngineParams) -> bool:
    """Purpose: the mover's anger on the tie — its too-much side above threshold (M5.D.4a).
    Spec:    docs/bowen_agent_model_spec_v2.md#M5.D.4, #M5.D.4a
    Tests:   tests/bowen/test_iposition.py::test_m5d4_an_angry_mover_stalls_rather_than_aborts
    """
    tie = state.tie_between(mover, target)
    return too_much(state.people[mover], tie, params) > params.anger_threshold


def assertion_form(state: RunState, mover: PersonId, target: PersonId, params: EngineParams) -> bool:
    """Purpose: whether an I-POSITION executes as the assertion form (M5.F.4, M5.D.4a).
    Spec:    docs/bowen_agent_model_spec_v2.md#M5.F.4, #M5.D.4a
    Tests:   tests/bowen/test_iposition.py::test_m5f4_low_perspective_executes_the_assertion_form
    """
    perspective = state.people[mover].systems_perspective or 0.0
    return perspective < params.assertion_perspective_threshold or angry(state, mover, target, params)


def capacity(person: Person, params: EngineParams) -> float:
    """Purpose: how much excess anxiety the mover can sit with and hold course.
    Spec:    docs/bowen_agent_model_spec_v2.md#M5.D.3, #M5.D.2a
    Tests:   tests/bowen/test_iposition.py::test_m5d2a_unprepared_sequence_holds_with_less_capacity
    """
    return params.hold_gain * person.functional_level * efficacy(person)


def begin(state: RunState, mover: PersonId, target: PersonId) -> EffectRecord:
    person = state.people[mover]
    tie = state.tie_between(mover, target)
    person.iposition_state = Sequence(
        target=target, stage=PREPARE, started=state.tick, stage_since=state.tick,
        start_balance=tie.functioning_balance.get(DEFAULT_AREA, 0.0), start_axes=axes(person), history=[PREPARE],
    )
    return _record(state, mover, "begin", PREPARE)


def _record(state: RunState, mover: PersonId, what: str, stage: str) -> EffectRecord:
    return EffectRecord(state.tick, "iposition", None, people=((mover, f"iposition:{what}:{stage}", 1.0),))


def _impinging_delivered(state: RunState, person: PersonId, sender: PersonId | None = None) -> bool:
    for delivery in state.store.delivered_to(person, Role.TARGET):
        if delivery.delivered_tick != state.tick:
            continue
        event = state.store.event(delivery.event_id)
        if event.mechanism is not Mechanism.MOVE or (sender is not None and event.sender != sender):
            continue
        if state.kinds.components_of(event.kind)[1] > 0 or event.assertion:
            return True
    return False


def _set(seq: Sequence, stage: str, tick: int) -> None:
    seq.stage, seq.stage_since = stage, tick
    seq.history.append(stage)


def end(state: RunState, mover: PersonId, outcome: str) -> EffectRecord:
    state.people[mover].iposition_state = None
    return _record(state, mover, "end", outcome)


def _live(state: RunState, mover: PersonId, target: PersonId) -> bool:
    tie = state.tie_between(mover, target)
    return (tie is not None and tie.interactive and tie.tie_state is not TieState.CUT_OFF
            and state.people[target].alive)


def advance_sequences(state: RunState, params: EngineParams) -> list[EffectRecord]:
    """Purpose: step 7, before selection — move every sequence on from this week's state and deliveries.
    Spec:    docs/bowen_agent_model_spec_v2.md#M5.D.2, #M5.D.2a, #M5.D.3, #M5.D.4, #M5.E.3, #M5.D.5, #M5.D.6
    Tests:   tests/bowen/test_iposition.py::test_m5d2_states_run_in_order
    """
    records = []
    for pid in sorted(state.people):
        person = state.people[pid]
        seq = person.iposition_state
        if seq is None:
            continue
        if not person.alive:
            records.append(end(state, pid, "mover_died"))
            continue
        if seq.due is not None:
            continue  # an outcome is still owed this week (deferred, M5.D.9)
        age = state.tick - seq.stage_since
        if seq.stage == PREPARE:
            if excess(person) > params.defence_threshold:
                records.append(end(state, pid, "prepare_failed"))
                continue
            if _impinging_delivered(state, pid, seq.target):
                seq.due = DEFINE  # the issue came up before the preparation was done
                records.append(_record(state, pid, "forced", DEFINE))
                continue
            outward, inward = axes(person)
            person.outside_ness_outward = outward * (1 - params.rehearsal_rate)
            person.outside_ness_inward = inward * (1 - params.rehearsal_rate)
            seq.prepared += 1
            if seq.prepared >= params.prepare_ticks:
                seq.due = DEFINE
        elif seq.stage == OPPOSITION:
            if age < 1:
                continue
            if _impinging_delivered(state, pid):
                if excess(person) <= capacity(person, params):
                    _set(seq, HOLD, state.tick)
                    records.append(_record(state, pid, "enter", HOLD))
                else:
                    seq.due = ABORT
            elif age >= params.opposition_window:
                records.append(end(state, pid, "did_not_land"))
        elif seq.stage == HOLD:
            if age < 1:
                continue
            if angry(state, pid, seq.target, params):
                seq.stalls += 1
                if seq.stalls >= params.stall_limit:
                    records.append(end(state, pid, "stalled"))
            elif _impinging_delivered(state, pid):
                _set(seq, PEAK, state.tick)
                if excess(person) <= capacity(person, params):
                    records.append(resolve(state, pid, params))
                else:
                    seq.due = ABORT
            elif age >= params.hold_window:
                records.append(end(state, pid, "settled_without_peak"))
        elif seq.stage == RESOLVE and seq.due_at is not None and state.tick >= seq.due_at:
            seq.due = FOLLOW_UP
    return records


def resolve(state: RunState, mover: PersonId, params: EngineParams) -> EffectRecord:
    """Purpose: the opposition pulls up to the mover's level, not to a mean (M5.D.5).
    Spec:    docs/bowen_agent_model_spec_v2.md#M5.D.5, #M5.D.6
    Tests:   tests/bowen/test_iposition.py::test_m5d5_opposition_pulls_up_to_the_movers_level
    """
    seq = state.people[mover].iposition_state
    other = state.people[seq.target]
    gap = max(0.0, state.people[mover].functional_level - other.functional_level)
    seq.pull_up = params.pull_up_rate * gap
    other.functional_level += seq.pull_up
    _set(seq, RESOLVE, state.tick)
    seq.due_at = state.tick + 1  # M5.D.6: the fast tick after RESOLVE
    return EffectRecord(state.tick, "iposition", None, people=(
        (mover, f"iposition:enter:{RESOLVE}", 1.0), (seq.target, "functional_level", seq.pull_up)))


def restore_prior_balance(state: RunState, mover: PersonId) -> None:
    """M5.D.3: an abort returns the mover to the balance it started from."""
    person = state.people[mover]
    seq = person.iposition_state
    tie = state.tie_between(mover, seq.target)
    tie.functioning_balance[DEFAULT_AREA] = seq.start_balance
    person.outside_ness_outward, person.outside_ness_inward = seq.start_axes


def step_done(state: RunState, mover: PersonId, params: EngineParams) -> list[EffectRecord]:
    """Purpose: after the week's outcome was a sequence step, move the sequence on.
    Spec:    docs/bowen_agent_model_spec_v2.md#M5.D.3, #M5.D.7, #M1.C.5, #M5.E.8
    Tests:   tests/bowen/test_iposition.py::test_m5d7_completed_exchange_raises_level_and_lowers_the_triangle
    """
    seq = state.people[mover].iposition_state
    step, seq.due, seq.deferrals = seq.due, None, 0
    if step == DEFINE:
        _set(seq, OPPOSITION, state.tick)
        return [_record(state, mover, "enter", OPPOSITION)]
    if step == ABORT:
        restore_prior_balance(state, mover)
        return [end(state, mover, "aborted")]
    if step == FOLLOW_UP:
        return complete(state, mover, params)
    raise ValueError(f"{step} is not an outcome step")


def complete(state: RunState, mover: PersonId, params: EngineParams) -> list[EffectRecord]:
    """Purpose: a completed exchange — small level gain, permanent triangle decrement, a more solid tie.
    Spec:    docs/bowen_agent_model_spec_v2.md#M5.D.7, #M5.D.7a, #M1.C.5, #M5.E.8
    Tests:   tests/bowen/test_iposition.py::test_m5e8_completion_leaves_the_tie_more_solid
    """
    person = state.people[mover]
    target = person.iposition_state.target
    person.functional_level = min(SCALE_MAX, person.functional_level + params.exchange_gain)
    lowered = []
    for tri_id in sorted(state.triangles):
        if mover in tri_id.members and target in tri_id.members:
            triangle = state.triangles[tri_id]
            triangle.intensity_floor = min(1.0, triangle.intensity_floor + params.triangle_floor_decrement)
            lowered.append((tri_id, "intensity_floor", f"{triangle.intensity_floor}"))
    for pid in (mover, target):
        p = state.people[pid]
        outward, inward = axes(p)
        p.outside_ness_outward = outward * (1 - params.respect_gain)
        p.outside_ness_inward = inward * (1 - params.respect_gain)
    state.tie_between(mover, target).functioning_habit[DEFAULT_AREA] = 0.0
    person.iposition_state = None
    done = EffectRecord(state.tick, "iposition", None, triangles=tuple(lowered), people=(
        (mover, "iposition:end:completed", 1.0), (mover, "functional_level", params.exchange_gain)))
    budget = reduce_budget(state, params, None)  # M6.I.1: the budget's one sink
    return [done] + ([budget] if budget is not None else [])


def skip_follow_up(state: RunState, mover: PersonId) -> EffectRecord:
    """Purpose: a follow-up that cannot be made reverts the gain (M5.D.6)."""
    seq = state.people[mover].iposition_state
    state.people[seq.target].functional_level -= seq.pull_up
    return end(state, mover, "follow_up_skipped")


def tie_of(seq: Sequence, mover: PersonId) -> TieId:
    return TieId.of(mover, seq.target)


# --- the week's outcome for a person in a sequence (M5.D.9) --------------------------------


def _abort_act(state: RunState, mover: PersonId, target: PersonId, params: EngineParams) -> str:
    """The abort branch, drawn from the mover's own automatic values for each act toward the target.

    The value of an act is the mean of the mover's learned values for it toward the target
    across bands and positions, 0 where none is learned. One keyed uniform (M3.D.4b's
    move-selection class, purpose "abort").
    """
    import math

    from src.bowen.engine.draws import DrawKey

    learned = state.people[mover].learned_values
    weights = []
    for kind in ABORT_ACTS:
        values = [v for k, v in learned.items() if k.endswith(f"|{kind}|{target.value}")]
        mean = sum(values) / len(values) if values else 0.0
        weights.append(math.exp(mean / params.policy_temperature))
    key = DrawKey.make("move_selection", tick=state.tick, actor=mover, purpose="abort", index=0)
    return ABORT_ACTS[state.draws.categorical(key, weights)]


def sequence_selections(state: RunState, params: EngineParams) -> tuple[list, list[EffectRecord]]:
    """Purpose: the outcomes owed this week by persons whose sequence has a step due.
    Spec:    docs/bowen_agent_model_spec_v2.md#M5.D.9, #M5.D.6, #M5.D.3
    Tests:   tests/bowen/test_iposition.py::test_m5d9_a_due_step_is_the_weeks_outcome

    A step that cannot be made because the tie is not live is deferred one week and
    logged; a second deferral ends the sequence. A follow-up that cannot be made reverts
    the gain (M5.D.6).
    """
    from src.bowen.engine.act import Selection
    from src.bowen.engine.log_records import DecidedBy

    selections, records = [], []
    for pid in sorted(state.people):
        seq = state.people[pid].iposition_state
        if seq is None or seq.due is None or not state.people[pid].alive:
            continue
        if not _live(state, pid, seq.target):
            if seq.due == FOLLOW_UP:
                records.append(skip_follow_up(state, pid))
            elif seq.deferrals >= 1:
                records.append(end(state, pid, f"{seq.due.lower()}_impossible"))
            else:
                seq.deferrals += 1
                records.append(_record(state, pid, "deferred", seq.due))
            continue
        kind = {DEFINE: I_POSITION, FOLLOW_UP: STAY_IN_CONTACT}.get(seq.due) or _abort_act(state, pid, seq.target, params)
        selections.append(Selection(
            actor=pid, kind=kind, targets=(seq.target,), intensity=params.policy_intensity,
            decided_by=DecidedBy.SEQUENCE, sequence_step=seq.due,
        ))
    return selections, records


def redirect_to_sequence_tie(state: RunState, selection):
    """Purpose: on a week with no step due, an act toward the sequence's other is made as STAY-IN-CONTACT.
    Spec:    docs/bowen_agent_model_spec_v2.md#M5.D.9
    Tests:   tests/bowen/test_iposition.py::test_m5d9_acts_on_the_sequence_tie_count_as_stay_in_contact
    """
    import dataclasses

    seq = state.people[selection.actor].iposition_state
    if seq is None or selection.targets != (seq.target,) or selection.kind in (STAY_IN_CONTACT, "WITHHOLD"):
        return selection
    if not _live(state, selection.actor, seq.target):
        return selection  # across a severed tie the act stays what it is (only REDUCE_CUTOFF is legal there)
    return dataclasses.replace(selection, kind=STAY_IN_CONTACT, value_key="", withheld=None)
