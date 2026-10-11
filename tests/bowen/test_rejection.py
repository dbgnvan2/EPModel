"""Cutoff as rejection, and the reconnection pull (Phase D plan B4; owner decisions P1, P2).

Purpose: prove that a cutoff leaves its tie open and is held as the cutter's rejection; that what the other sends is
         refused, at the sender's cost, and raises rejection; that the move the cutter accepts lowers it; and that a
         cut tie's "too little" side grows for both members until it reopens. Each test compares two arms and asserts
         a direction (M0.4).
Spec:    docs/bowen_agent_model_spec_v2.md#M1.B.3c, #M4.C.1c, #M5.B.3, #M1.B.4
Tests:   this file
"""

from __future__ import annotations

from src.bowen.engine.act import Selection, act
from src.bowen.engine.contact import relax_contact, resting_contact, too_little
from src.bowen.engine.event_effects import apply_delivered_cutoffs, apply_structural_event
from src.bowen.engine.events import Channel, Delivery, Event, EventId, Mechanism, Role, SourcePosition
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.log_records import DecidedBy
from src.bowen.engine.moves import apply_move_effects
from src.bowen.engine.objects import TieState
from src.bowen.engine.rejection import fade_rejection, raise_rejection, refuse
from src.bowen.engine.state import new_run_state
from src.bowen.engine.visibility import HouseholdConductanceVisibility
from src.bowen.io.load import load_constants, load_event_kinds, load_family
from src.bowen.scenario.params import engine_params

P = PersonId
RAVI, MARTA, ANA, BRUNO = P("ravi"), P("marta"), P("ana"), P("bruno")
MARITAL = TieId.of(RAVI, MARTA)
ANA_BRUNO = TieId.of(ANA, BRUNO)
PARAMS = engine_params(load_constants())
VIS = HouseholdConductanceVisibility(PARAMS.per_hop_fidelity)


def fresh():
    family = load_family()
    state = new_run_state(dict(family.people), dict(family.ties), family.family, load_event_kinds())
    return initialise_run(state, PARAMS)


def move(kind, sender, target, intensity=100.0, tick=0, index=0):
    return Event(
        id=EventId(tick, str(sender.value), index), kind=kind, mechanism=Mechanism.MOVE, sender=sender,
        targets=(target,), intensity=intensity, timestamp=tick, duration=1, exogenous=False,
        source_position=SourcePosition.NONE, channel=Channel.SCRIPTED,
    )


def reconciliation(tie, tick=0):
    return Event(
        id=EventId(tick, "script:0", 0), kind="RECONCILIATION", mechanism=Mechanism.RECONCILIATION, sender=None,
        targets=(), intensity=1.0, timestamp=tick, duration=1, exogenous=True,
        source_position=SourcePosition.NONE, channel=Channel.EXOGENOUS, on_tie=tie,
    )


def delivered(state, event):
    state.store.record_event(event)
    return (Delivery(event.timestamp + 1, event.id, event.targets[0], Role.TARGET,
                     emitted_tick=event.timestamp, latency=1),)


def test_m1b3c_a_delivered_cutoff_leaves_the_tie_open_as_the_cutters_rejection():
    state = fresh()
    batch = delivered(state, move("CUTOFF", MARTA, RAVI))
    apply_delivered_cutoffs(state, batch, PARAMS)
    tie = state.ties[MARITAL]
    assert tie.interactive and tie.tie_state is TieState.CUT_OFF
    assert tie.rejection.get(MARTA, 0.0) > 0 and tie.rejection.get(RAVI, 0.0) == 0


def test_m1b3c_a_refused_approach_never_reaches_the_cutter_and_costs_the_sender():
    """The same PURSUE, delivered to a person who rejects its sender and to one who does not."""
    arms = {}
    for rejecting in (True, False):
        state = fresh()
        if rejecting:
            raise_rejection(state.ties[MARITAL], MARTA, 0.8)
        before = {p: state.people[p].acute_anxiety for p in (RAVI, MARTA)}
        held = state.ties[MARITAL].rejection.get(MARTA, 0.0)
        accepted, records = refuse(state, delivered(state, move("PURSUE", RAVI, MARTA)), PARAMS)
        arms[rejecting] = (accepted, records, before, held, state)
    accepted, records, before, held, state = arms[True]
    assert accepted == () and [r.mechanism for r in records] == ["refusal"]
    assert state.people[RAVI].acute_anxiety > before[RAVI]          # the sender bears the refusal
    # nothing reaches the cutter: the delivery is gone from the batch (accepted == () above)
    assert state.ties[MARITAL].rejection[MARTA] > held              # and the refusal raises rejection
    open_accepted, open_records, open_before, _, open_state = arms[False]
    assert len(open_accepted) == 1 and open_records == []
    assert open_state.people[RAVI].acute_anxiety == open_before[RAVI]
    assert open_state.ties[MARITAL].rejection.get(MARTA, 0.0) == 0


def test_m1b3c_repeated_refusals_raise_rejection_to_its_ceiling():
    state = fresh()
    raise_rejection(state.ties[MARITAL], MARTA, PARAMS.rejection_refusal_threshold + 0.05)
    seen = [state.ties[MARITAL].rejection[MARTA]]
    for tick in range(30):
        refuse(state, delivered(state, move("CONFLICT", RAVI, MARTA, tick=tick)), PARAMS)
        seen.append(state.ties[MARITAL].rejection[MARTA])
    assert all(b >= a for a, b in zip(seen, seen[1:])) and seen[1] > seen[0]
    assert seen[-1] == 1.0


def test_m1b3c_rejection_refuses_only_above_its_threshold():
    """Revision 16 (graded rejection): the same PURSUE is refused just above the threshold and received just below."""
    outcomes = {}
    for offset in (+0.01, -0.01):
        state = fresh()
        raise_rejection(state.ties[MARITAL], MARTA, PARAMS.rejection_refusal_threshold + offset)
        accepted, records = refuse(state, delivered(state, move("PURSUE", RAVI, MARTA)), PARAMS)
        outcomes[offset] = (len(accepted), [r.mechanism for r in records])
        assert state.ties[MARITAL].tie_state is TieState.CUT_OFF   # cut off either way; only refusal differs
    assert outcomes == {+0.01: (0, ["refusal"]), -0.01: (1, [])}


def test_m1b3c_unrenewed_rejection_fades_and_the_tie_reopens():
    """Revision 16: a single cutoff, not renewed, stops refusing and then wears off; the open tie never was cut."""
    state, open_ = fresh(), fresh()
    apply_delivered_cutoffs(state, delivered(state, move("CUTOFF", MARTA, RAVI)), PARAMS)
    held, weeks = [state.ties[MARITAL].rejection[MARTA]], 0
    assert MARITAL not in {t for r in fade_rejection(open_, PARAMS) for t, _, _ in r.ties}
    while state.ties[MARITAL].rejected():
        records = fade_rejection(state, PARAMS)
        held.append(state.ties[MARITAL].rejection[MARTA])
        weeks += 1
        assert weeks <= 1 / PARAMS.rejection_fade + 1
    assert all(b < a for a, b in zip(held, held[1:]))
    assert state.ties[MARITAL].tie_state is TieState.ORDINARY
    assert any(f == "tie_state_ordinary" for r in records for _, f, _ in r.ties)


def test_m1b3c_cutoff_and_reduce_cutoff_are_never_refused():
    state = fresh()
    raise_rejection(state.ties[MARITAL], MARTA, 0.9)
    for index, kind in enumerate(("CUTOFF", "REDUCE_CUTOFF")):
        accepted, records = refuse(state, delivered(state, move(kind, RAVI, MARTA, index=index)), PARAMS)
        assert len(accepted) == 1 and records == []


def test_m5b3_accepted_contact_lowers_both_rejections_and_reopens_the_tie_at_zero():
    state = fresh()
    tie = state.ties[MARITAL]
    raise_rejection(tie, MARTA, PARAMS.rejection_relief * 1.5)
    raise_rejection(tie, RAVI, PARAMS.rejection_relief)
    apply_move_effects(state, delivered(state, move("REDUCE_CUTOFF", RAVI, MARTA)), PARAMS)
    assert tie.rejection[RAVI] == 0 and 0 < tie.rejection[MARTA] < PARAMS.rejection_relief * 1.5
    assert tie.tie_state is TieState.CUT_OFF                        # Marta still rejects
    apply_move_effects(state, delivered(state, move("REDUCE_CUTOFF", RAVI, MARTA, tick=1)), PARAMS)
    assert tie.rejection[MARTA] == 0 and tie.tie_state is TieState.ORDINARY


def test_m4c1c_a_cut_tie_pulls_both_members_until_it_reopens():
    """P2: no pull term — while the tie is cut, the "too little" side grows for both; reopening relieves it."""
    cut, open_ = fresh(), fresh()
    raise_rejection(cut.ties[MARITAL], MARTA, 0.5)
    for member in MARITAL.members():
        assert resting_contact(cut.people[member], cut.ties[MARITAL], PARAMS) == 0
        assert resting_contact(open_.people[member], open_.ties[MARITAL], PARAMS) > 0
    start = {m: too_little(cut.people[m], cut.ties[MARITAL], PARAMS) for m in MARITAL.members()}
    for _ in range(10):
        relax_contact(cut.people, cut.ties, PARAMS)
        relax_contact(open_.people, open_.ties, PARAMS)
    for member in MARITAL.members():
        grown = too_little(cut.people[member], cut.ties[MARITAL], PARAMS)
        assert grown > start[member]                                 # the rejected member too
        assert grown > too_little(open_.people[member], open_.ties[MARITAL], PARAMS)
    tick = 0
    while cut.ties[MARITAL].rejection.get(MARTA, 0.0) > 0:
        apply_move_effects(cut, delivered(cut, move("REDUCE_CUTOFF", RAVI, MARTA, tick=tick)), PARAMS)
        tick += 1
    pulled = {m: too_little(cut.people[m], cut.ties[MARITAL], PARAMS) for m in MARITAL.members()}
    for _ in range(10):
        relax_contact(cut.people, cut.ties, PARAMS)
    for member in MARITAL.members():
        assert too_little(cut.people[member], cut.ties[MARITAL], PARAMS) < pulled[member]


def test_m1b4_reconciliation_clears_rejection():
    state = fresh()
    raise_rejection(state.ties[MARITAL], MARTA, 0.8)
    apply_structural_event(state, reconciliation(MARITAL), PARAMS)
    tie = state.ties[MARITAL]
    assert not any(tie.rejection.values()) and tie.tie_state is TieState.ORDINARY


def test_m1b3c_a_declared_cutoff_starts_open_with_both_members_rejecting():
    state = fresh()
    tie = state.ties[ANA_BRUNO]
    assert tie.interactive and tie.tie_state is TieState.CUT_OFF
    assert tie.rejection == {ANA: PARAMS.declared_rejection, BRUNO: PARAMS.declared_rejection}
    assert all(tie.felt_contact[m] == 0 for m in ANA_BRUNO.members())  # the pull is full from week 0


def test_m1f1a_a_policy_act_is_stamped_with_its_kinds_channel():
    """B3: a self-directed act is logged self-directed; the rest automatic; a scripted act says so."""
    state = fresh()
    stamped = {}
    for index, (kind, decided) in enumerate((("STAY-IN-CONTACT", DecidedBy.POLICY), ("PURSUE", DecidedBy.POLICY),
                                             ("REDUCE_CUTOFF", DecidedBy.SEQUENCE), ("PURSUE", DecidedBy.SCRIPTED))):
        target = BRUNO if kind == "REDUCE_CUTOFF" else MARTA
        actor = ANA if kind == "REDUCE_CUTOFF" else RAVI
        act(state, Selection(actor, kind, (target,), 100.0, index=index, decided_by=decided), VIS)
        stamped[(kind, decided)] = state.store.event(EventId(state.tick, str(actor), index)).channel
    assert stamped == {
        ("STAY-IN-CONTACT", DecidedBy.POLICY): Channel.SELF_DIRECTED,
        ("PURSUE", DecidedBy.POLICY): Channel.AUTOMATIC,
        ("REDUCE_CUTOFF", DecidedBy.SEQUENCE): Channel.SELF_DIRECTED,
        ("PURSUE", DecidedBy.SCRIPTED): Channel.SCRIPTED,
    }


# --- review fixes, 2026-10-11 ------------------------------------------------------------------------------------


class _Quiet:
    """A source with no scheduled events and no selections: only what is already queued moves."""

    def scheduled(self, tick):
        return ()

    def selections(self, tick, active, state):
        return ()


def test_m1b3c_a_refused_move_is_never_recorded_as_delivered():
    """Review finding: a refused move still sat in the store, so an I-POSITION's opposition check saw it arrive."""
    from src.bowen.engine.log_records import CollectingEmitter, EffectRecord
    from src.bowen.engine.tick import run_tick
    from src.bowen.scenario.assemble import assemble
    from src.bowen.scenario.scripted_source import ScriptedSource

    seen = {}
    for rejecting in (True, False):
        parts = assemble(load_constants(), load_event_kinds(), load_family(), ScriptedSource("q", 2, (), ()), seed=0)
        state = parts.state
        if rejecting:
            raise_rejection(state.ties[MARITAL], MARTA, 0.8)
        act(state, Selection(RAVI, "CONFLICT", (MARTA,), 100.0, decided_by=DecidedBy.SCRIPTED), parts.visibility)
        emitter = CollectingEmitter()
        for _ in range(2):
            run_tick(state, _Quiet(), parts.params, parts.visibility, parts.activation, emitter)
        reached = [d for d in state.store.delivered_to(MARTA) if state.store.event(d.event_id).sender == RAVI]
        refusals = [r for r in emitter.records if isinstance(r, EffectRecord) and r.mechanism == "refusal"]
        seen[rejecting] = (len(reached), len(refusals))
    assert seen == {True: (0, 1), False: (1, 0)}


def test_m1b3c_i_position_is_not_legal_across_a_cut_off_tie():
    """Review finding: a sequence cannot run across a cut-off tie, so the policy must not start one there."""
    from src.bowen.engine.observe import observe
    from src.bowen.policy.policy import legal_outcomes

    state = fresh()
    raise_rejection(state.ties[MARITAL], MARTA, 0.1)
    legal = {o.label for o in legal_outcomes(observe(state, RAVI, PARAMS), load_event_kinds(), PARAMS)}
    open_legal = {o.label for o in legal_outcomes(observe(fresh(), RAVI, PARAMS), load_event_kinds(), PARAMS)}
    assert "I-POSITION>marta" in open_legal and "I-POSITION>marta" not in legal
    assert "PURSUE>marta" in legal  # the tie itself stays open


def test_m16c1_refusal_and_reconciliation_render_their_own_anxiety_lines():
    """Review finding: the refusal template had taken over reconciliation's 'distancing returns' line."""
    from src.bowen.engine.log_records import EffectRecord
    from src.bowen.render.trace import _Index, _Names, _what_it_did

    names = _Names(load_family().display_names)
    pursue, reunion = move("PURSUE", RAVI, MARTA), reconciliation(MARITAL)
    effects = {
        pursue.id: [EffectRecord(0, "refusal", pursue.id, acute_anxiety=((RAVI, 1.5),),
                                 ties=((MARITAL, f"rejection:{MARTA.value}", 0.1),))],
        reunion.id: [EffectRecord(0, "reconciliation", reunion.id, acute_anxiety=((RAVI, 2.0), (MARTA, 2.0)),
                                  ties=((MARITAL, "interactive", 1.0),))],
    }
    index = _Index({pursue.id: pursue, reunion.id: reunion}, {}, effects)
    refused, reunited = _what_it_did(pursue, index, names, None), _what_it_did(reunion, index, names, None)
    assert "the sender bears the refusal" in refused and "distancing" not in refused
    assert "anxiety bound by distancing returns" in reunited and "refusal" not in reunited


def test_m1f1a_the_channel_names_agree():
    """act.SELF_CHANNEL duplicates policy.SELF because act cannot import policy; they must not drift."""
    from src.bowen.engine.act import SELF_CHANNEL
    from src.bowen.policy.policy import SELF

    assert SELF_CHANNEL == SELF and SELF in set(load_event_kinds().channels.values())


def test_m5b3_reduce_cutoff_leaves_a_worry_edge_a_worry_edge():
    """Review finding: REDUCE_CUTOFF used to set interactive, which would undo an INSTITUTIONALIZE (M4.A.4)."""
    state = fresh()
    tie = state.ties[MARITAL]
    raise_rejection(tie, MARTA, PARAMS.rejection_relief)
    tie.interactive = False
    apply_move_effects(state, delivered(state, move("REDUCE_CUTOFF", RAVI, MARTA)), PARAMS)
    assert not tie.interactive and not tie.rejected()


def test_m1b3c_a_distant_tie_is_distant_again_when_the_cutoff_ends():
    """Review finding: an ended cutoff returned a hardened distant tie (M4.G.1) to ordinary."""
    state = fresh()
    tie = state.ties[MARITAL]
    tie.tie_state = TieState.DISTANT
    raise_rejection(tie, MARTA, PARAMS.rejection_fade)
    assert tie.tie_state is TieState.CUT_OFF
    fade_rejection(state, PARAMS)
    assert tie.tie_state is TieState.DISTANT


def test_m1b3c_the_invariant_catches_a_cut_off_tie_with_no_rejection():
    from src.bowen.engine.invariants import InvariantViolation, assert_invariants, snapshot

    state = fresh()
    before = snapshot(state)
    state.ties[MARITAL].tie_state = TieState.CUT_OFF  # without rejection held
    try:
        assert_invariants(state, before, frozenset(state.ties), ("standing_load", "deliver"), PARAMS.invariant_tolerance)
    except InvariantViolation as error:
        assert "M1.B.3c" in str(error)
    else:
        raise AssertionError("a cut-off tie with no rejection passed the invariants")
