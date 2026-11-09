"""The social cues' acceptance criteria, `M11.C.50`–`M11.C.54` on the Phase C families (spec revision 55, T-202).

Purpose: each criterion as two arms on identical seeds, **both with ``social_cues_enabled`` on** (`M4.C.1f`): an
         experimental override the criterion declares, so the criteria test the cues while every other record runs
         with them off. Each arm differs from the other in one declared condition; every value is in
         ``config/bowen/criteria.md``, `[I]`. `M11.C.55`, on the twelve-person family, is in ``criteria_d2``.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.50, #M11.C.51, #M11.C.52, #M11.C.53, #M11.C.54
Tests:   tests/bowen/test_criteria_sc.py

Every readout counts what the people selected themselves (the policy's moves, not a scripted one), or reads a
state the run reached. No person's id appears here (`M2.3a`): the Phase C people are named by their roles in
``config/bowen/criterion_roles.md``, and the dyad's members by the fixture's letter ids.
"""

from __future__ import annotations

import math

from src.bowen.engine.contact import too_little
from src.bowen.engine.counting import count
from src.bowen.engine.events import Channel, Event, EventId, Mechanism, SourcePosition
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.log_records import EmittedRecord
from src.bowen.engine.signals import CUES
from src.bowen.ensemble import criteria as _criteria
from src.bowen.ensemble.criteria import ROLES, SETTINGS, act, result, scenario
from src.bowen.ensemble.runner import Criterion as C
from src.bowen.ensemble.runner import Readout
from src.bowen.io.load import load_constants
from src.bowen.readouts.roles import supply_counts
from src.bowen.scenario.params import engine_params

P = PersonId
CUES_ON = {"social_cues_enabled": 1}  # both arms of every criterion here (M4.C.1f)
SENDER, RECEIVER, CHILD = ROLES["phase_c", "parent_1"], ROLES["phase_c", "parent_2"], ROLES["phase_c", "third"]
_ENGINE = engine_params(load_constants())


def _vector(**values: float) -> tuple[float, ...]:
    return tuple(float(values.get(c.value, 0.0)) for c in CUES)


def _own_moves(records, sender, target, kinds=None, since=0, until=None):
    """The moves ``sender`` selected itself toward ``target`` in ``[since, until)`` (a scripted move is not one)."""
    out = []
    for r in records:
        if not isinstance(r, EmittedRecord):
            continue
        e = r.event
        if (e.mechanism is Mechanism.MOVE and e.sender == sender and target in e.targets
                and e.channel is not Channel.SCRIPTED and e.timestamp >= since
                and (until is None or e.timestamp < until) and (kinds is None or e.kind in kinds)):
            out.append(e)
    return out


def _held_excess(person: PersonId, excess: float, weeks: range):
    """A ``before`` hook: ``person``'s acute anxiety set to its chronic level plus ``excess`` each week in ``weeks``."""
    def before(state):
        if state.tick in weeks:
            p = state.people[person]
            p.acute_anxiety = p.chronic_anxiety + excess
    return before


# --- M11.C.50: a calmer delivery is met with more contact (KS08.4) ------------------------------------------------


def arm_c50(arm, seed, settings):
    t0, window, after = settings["t0"], settings["window"], settings["after"]
    excess = settings["high_excess"] if arm == "baseline" else settings["low_excess"]
    weeks = range(t0, t0 + window)
    forced = {}
    for t in weeks:
        forced.update(act(SENDER.value, settings["scripted_kind"], RECEIVER.value, t))
    state, records = scenario("phase_c", seed, t0 + window + after, forced=forced, overrides=CUES_ON,
                              before=_held_excess(SENDER, excess, weeks))
    contact = [e for e in _own_moves(records, RECEIVER, SENDER, since=t0 + window)
               if state.kinds.components_of(e.kind)[0] > 0]
    return result({"contact_toward_sender": float(len(contact))}, records)


# --- M11.C.51: a believed disapproval is read into neutral moves --------------------------------------------------


def arm_c51(arm, seed, settings):
    belief = settings["believed_disapproval"] if arm == "treatment" else 0.0

    def setup(state):
        state.people[RECEIVER].cue_beliefs = {SENDER.value: _vector(approval=belief)}

    state, records = scenario("phase_c", seed, settings["weeks"], setup=setup, overrides=CUES_ON)
    moves = _own_moves(records, RECEIVER, SENDER)
    hostile = count(e.kind in ("DISTANCE", "CONFLICT") for e in moves)
    # A seed with no move toward the partner reads 0 and is counted (report only), so it shows (learning-qa).
    return result({"distance_conflict_share": hostile / len(moves) if moves else 0.0,
                   "no_moves": float(not moves)}, records)


# --- M11.C.52: self-attribution by channel damps an escalation (KS17.3) -------------------------------------------


def arm_c52(arm, seed, settings):
    a, b = P("a"), P("b")
    weeks = settings["weeks"]
    total = []

    def setup(state):
        for pid in (a, b):
            p = state.people[pid]
            p.acute_anxiety = p.chronic_anxiety + settings["start_excess"]
        state.people[a].systems_perspective = settings["perspective"] if arm == "treatment" else 0.0

    def watch(state):
        total.append(state.people[a].acute_anxiety + state.people[b].acute_anxiety)

    state, records = scenario("dyad", seed, weeks, setup=setup, watch=watch, overrides=CUES_ON)
    return result({"escalation": math.fsum(total)}, records)


# --- M11.C.53: a preoccupied spouse is felt as less attentive (KS07.6; a proxy for life_energy) --------------------


def _work_stressor(target: PersonId, t0: int, weeks: int, kind: str, intensity: float) -> Event:
    return Event(id=EventId(t0, "criterion", 0), kind=kind, mechanism=Mechanism.EXOGENOUS_STRESSOR, sender=None,
                 targets=(target,), intensity=intensity, timestamp=t0, duration=weeks, exogenous=True,
                 source_position=SourcePosition.NONE, channel=Channel.EXOGENOUS)


def arm_c53(arm, seed, settings):
    t0, span = settings["t0"], settings["stressor_weeks"]
    spells = (_work_stressor(SENDER, t0, span, settings["stressor_kind"], settings["stressor_intensity"]),) \
        if arm == "treatment" else ()
    marital = TieId.of(SENDER, RECEIVER)
    felt = []

    def watch(state):
        # The week just run (state.tick − 1) is one in which the stressed spouse's moves of weeks t0 to t0 + span − 1
        # arrive, a week after they are sent: weeks t0 + 1 to t0 + span, as the second readout counts (correctness pass).
        if t0 + 1 < state.tick <= t0 + span + 1:
            person, tie = state.people[RECEIVER], state.ties[marital]
            felt.append(too_little(person, tie, _ENGINE))

    state, records = scenario("phase_c", seed, t0 + span + 1, spells=spells, watch=watch, overrides=CUES_ON)
    pursuit = _own_moves(records, RECEIVER, SENDER, kinds=("PURSUE", "CONFLICT"), since=t0 + 1, until=t0 + span + 1)
    return result({"too_little_on_marital_tie": math.fsum(felt) / len(felt) if felt else 0.0,
                   "pursue_conflict_toward_stressed": float(len(pursuit))}, records)


# --- M11.C.54: a parent's false belief about a child lowers that child's functioning (M7.E.1d, M9.9) ---------------


def arm_c54(arm, seed, settings):
    believed = settings["believed_threat"] if arm == "treatment" else 0.0

    def setup(state):
        state.people[SENDER].cue_beliefs = {CHILD.value: _vector(expectations=believed, distress=believed)}

    state, records = scenario("phase_c", seed, settings["weeks"], setup=setup, overrides=CUES_ON)
    supplied = supply_counts(records)[CHILD, SENDER]
    return result({"child_functional_level": state.people[CHILD].functional_level,
                   "child_supply_to_parent": float(supplied)}, records)


CRITERIA = {
    "M11.C.50": C("M11.C.50", "composite", ("baseline", "treatment"), (Readout("contact_toward_sender", +1),),
                  arm_c50, settings=SETTINGS["M11.C.50"]),
    "M11.C.51": C("M11.C.51", "composite", ("baseline", "treatment"),
                  (Readout("distance_conflict_share", +1), Readout("no_moves", +1, report_only=True)),
                  arm_c51, ("DISTANCE", "CONFLICT"), settings=SETTINGS["M11.C.51"]),
    "M11.C.52": C("M11.C.52", "composite", ("baseline", "treatment"), (Readout("escalation", -1),), arm_c52,
                  settings=SETTINGS["M11.C.52"]),
    "M11.C.53": C("M11.C.53", "composite", ("baseline", "treatment"),
                  (Readout("too_little_on_marital_tie", +1), Readout("pursue_conflict_toward_stressed", +1)),
                  arm_c53, ("PURSUE", "CONFLICT"), settings=SETTINGS["M11.C.53"]),
    "M11.C.54": C("M11.C.54", "composite", ("baseline", "treatment"),
                  (Readout("child_functional_level", -1), Readout("child_supply_to_parent", 0, report_only=True)),
                  arm_c54, settings=SETTINGS["M11.C.54"]),
}

# Added here, not by ``criteria``: either module may be imported first (see the end of ``criteria.py``).
_criteria.CRITERIA.update(CRITERIA)
