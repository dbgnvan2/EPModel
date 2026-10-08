"""The belief store about ties a person is not party to (Phase C step 4, P2a).

Purpose: test that beliefs exist for exactly the ties a person is not party to, move
         only with what is delivered to that person, weigh fidelity, ignore batch
         order, and are free to differ from the tie's true state.
Spec:    docs/bowen_agent_model_spec_v2.md#M9.8, #M9.1, #M4.B.2, #M1.F.4, #M1.F.8, #M16.A.5a
Tests:   this file

Unit tests of a set mechanism with an invented smoothing rule; none is a finding (`M11.5`).
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from src.bowen.engine.beliefs import TieBelief, true_counterpart, update_beliefs
from src.bowen.engine.events import Channel, Delivery, Event, EventId, Mechanism, Role, SourcePosition
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.state import new_run_state
from src.bowen.io.load import load_constants, load_event_kinds, load_family
from src.bowen.scenario.params import engine_params

P = PersonId
RAVI, MARTA, NADIA, PIA, SOFIA = P("ravi"), P("marta"), P("nadia"), P("pia"), P("sofia")
MARITAL = TieId.of(RAVI, MARTA)
PARAMS = engine_params(load_constants())


def fresh():
    family = load_family()
    state = new_run_state(dict(family.people), dict(family.ties), family.family, load_event_kinds())
    return initialise_run(state, PARAMS)


def move(kind, intensity, sender=RAVI, target=MARTA, index=0, fidelity=1.0):
    return Event(
        id=EventId(0, str(sender.value), index), kind=kind, mechanism=Mechanism.MOVE, sender=sender,
        targets=(target,), intensity=intensity, timestamp=0, duration=1, exogenous=False,
        source_position=SourcePosition.NONE, channel=Channel.SCRIPTED, fidelity=fidelity,
    )


def delivered(state, events, recipient=NADIA, role=Role.WITNESS):
    """The perceived set step 3 hands the belief update: these events, delivered to ``recipient``."""
    items = []
    for event in events:
        state.store.record_event(event)
        items.append((Delivery(1, event.id, recipient, role, emitted_tick=0, latency=1), event))
    return {recipient: tuple(items)}


def test_m98_every_person_believes_about_each_tie_it_is_not_party_to():
    state = fresh()
    for pid, person in state.people.items():
        assert set(person.tie_beliefs) == {t for t in state.ties if pid not in t.members()}
    # The prior reads no true state: every belief starts the same, whatever the tie.
    assert {(b.tension, b.contact) for p in state.people.values() for b in p.tie_beliefs.values()} == {
        (0.0, PARAMS.interactive_resting_contact)
    }


def test_m98_belief_written_only_from_delivered_events():
    """A witnessed conflict raises the witness's believed tension; nobody else's belief moves."""
    state = fresh()
    before = {pid: {t: (b.tension, b.contact) for t, b in p.tie_beliefs.items()} for pid, p in state.people.items()}
    update_beliefs(state, delivered(state, [move("CONFLICT", 150.0)]), PARAMS)
    assert state.people[NADIA].tie_beliefs[MARITAL].tension > before[NADIA][MARITAL][0]
    for pid, person in state.people.items():
        for tie, belief in person.tie_beliefs.items():
            if (pid, tie) != (NADIA, MARITAL):
                assert (belief.tension, belief.contact) == before[pid][tie], (pid, tie)


def test_m98_belief_can_differ_from_the_true_state():
    """A tense tie seen only in its calm exchanges is believed calm: a misperceived alliance is possible."""
    state = fresh()
    tie = state.ties[MARITAL]
    for m in MARITAL.members():
        tie.felt_impingement[m] = 1.0  # truly tense
    for week in range(20):
        update_beliefs(state, delivered(state, [move("STAY-IN-CONTACT", 60.0, index=week)]), PARAMS)
    belief, truth = state.people[NADIA].tie_beliefs[MARITAL], true_counterpart(tie)
    assert belief.tension - truth.tension < -0.5  # the signed discrepancy M16.A.5a asks for


def test_m98_true_state_without_delivery_leaves_belief_unchanged():
    """No drift toward truth: the store is written by deliveries, never by the tie."""
    state = fresh()
    for m in MARITAL.members():
        state.ties[MARITAL].felt_impingement[m] = 1.0
    update_beliefs(state, {}, PARAMS)
    assert state.people[NADIA].tie_beliefs[MARITAL] == TieBelief(0.0, PARAMS.interactive_resting_contact)


def test_m98_a_party_to_the_tie_holds_no_belief_about_it():
    """The target reads its own tie directly; nothing is written to its store."""
    state = fresh()
    records = update_beliefs(state, delivered(state, [move("CONFLICT", 150.0)], recipient=MARTA, role=Role.TARGET), PARAMS)
    assert MARITAL not in state.people[MARTA].tie_beliefs and records == []


def test_m1f4_lower_fidelity_moves_belief_less():
    clear, degraded = fresh(), fresh()
    update_beliefs(clear, delivered(clear, [move("CONFLICT", 150.0)]), PARAMS)
    update_beliefs(degraded, delivered(degraded, [move("CONFLICT", 150.0, fidelity=0.4)]), PARAMS)
    assert 0 < degraded.people[NADIA].tie_beliefs[MARITAL].tension < clear.people[NADIA].tie_beliefs[MARITAL].tension


def test_m1f8_batch_order_does_not_change_beliefs():
    forward, reverse = fresh(), fresh()
    events = [move("CONFLICT", 150.0, index=0), move("STAY-IN-CONTACT", 60.0, index=1, fidelity=0.5)]
    update_beliefs(forward, delivered(forward, events), PARAMS)
    update_beliefs(reverse, delivered(reverse, list(reversed(events))), PARAMS)
    assert forward.people[NADIA].tie_beliefs == reverse.people[NADIA].tie_beliefs


# --- M4.B.2, statically: the update reads no true state -----------------------------------

TRUE_STATE = {
    "ties", "tie_between", "ties_of", "felt_contact", "felt_impingement", "conductance", "bond_energy",
    "acute_anxiety", "chronic_anxiety", "outside_ness_outward", "outside_ness_inward", "investment",
    "functioning_balance", "distance_bound_anxiety", "tie_state",
}


def true_state_reads(source: str) -> set[str]:
    return {n.attr for n in ast.walk(ast.parse(textwrap.dedent(source))) if isinstance(n, ast.Attribute)} & TRUE_STATE


def test_m4b2_belief_update_reads_no_true_state():
    assert true_state_reads(inspect.getsource(update_beliefs)) == set()


def test_m4b2_check_catches_a_belief_update_reading_the_tie():
    """Mutation: an update that reads the tie's true impingement must be caught."""
    mutant = inspect.getsource(update_beliefs).replace(
        "clamp_unit(max(0.0, impingement) * strength)",
        "state.ties[TieId.of(event.sender, target)].felt_impingement[target]",
    )
    assert mutant != inspect.getsource(update_beliefs)
    assert true_state_reads(mutant) == {"ties", "felt_impingement"}


def names_used(source: str) -> set[str]:
    tree = ast.parse(textwrap.dedent(source))
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}


def test_m98_true_counterpart_is_not_called_by_the_update():
    # Named through the imported function, so renaming it breaks this test rather than emptying it.
    assert true_counterpart.__name__ not in names_used(inspect.getsource(update_beliefs))


def test_m98_check_catches_an_update_calling_true_counterpart():
    mutant = inspect.getsource(update_beliefs).replace(
        "clamp_unit(max(0.0, impingement) * strength)", "true_counterpart(tie).tension")
    assert mutant != inspect.getsource(update_beliefs)
    assert true_counterpart.__name__ in names_used(mutant)
