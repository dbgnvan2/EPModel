"""Activation and visibility.

Purpose: prove the two components are separate, activation is the declared
         synchronous regime, and visibility computes witnesses, deliveries and
         fidelity from tie and household state.
Spec:    docs/bowen_agent_model_spec_v2.md#M3.E.1, #M3.E.2, #M1.F.1b, #M4.E.1a, #M1.F.4, #M3.C.2, #M8.5
Tests:   this file
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from src.bowen.engine import activation, visibility
from src.bowen.engine.activation import SynchronousActivation, activation_for
from src.bowen.engine.events import Channel, Event, EventId, Mechanism, Role, SourcePosition
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.visibility import HouseholdConductanceVisibility, MissingTie
from src.bowen.io.load import load_constants, load_family

P = PersonId
RAVI, MARTA, NADIA, PIA, ANA, SOFIA, BRUNO = (P(n) for n in ("ravi", "marta", "nadia", "pia", "ana", "sofia", "bruno"))


@pytest.fixture
def family():
    return load_family()


@pytest.fixture
def vis():
    return HouseholdConductanceVisibility(load_constants()["per_hop_fidelity"])


def move(sender, targets, tick=1, kind="CONFLICT", **overrides) -> Event:
    values = dict(
        id=EventId(tick, str(sender), 0), kind=kind, mechanism=Mechanism.MOVE, sender=sender,
        targets=tuple(sorted(targets)), intensity=5.0, timestamp=tick, duration=1, exogenous=False,
        source_position=SourcePosition.NONE, channel=Channel.SCRIPTED,
    )
    values.update(overrides)
    return Event(**values)


# --- M3.E.1 / M3.E.2 --------------------------------------------------------------------


def test_m3e1_activation_and_visibility_are_separate_components():
    for component in (SynchronousActivation, HouseholdConductanceVisibility):
        assert component.component_id and component.version
    assert SynchronousActivation.component_id != HouseholdConductanceVisibility.component_id
    assert "src.bowen.engine.visibility" not in _imports(activation)
    assert "src.bowen.engine.activation" not in _imports(visibility)


def _imports(module) -> set[str]:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
    return found


def test_m3e2_synchronous_activation_selects_every_living_person(family):
    component = activation_for(load_constants().activation_regime)
    assert component.grade == "[I]"
    assert component.active(0, family.people.values()) == tuple(sorted(family.people))
    people = dict(family.people)
    people[BRUNO] = dataclasses.replace(people[BRUNO], alive=False)
    assert BRUNO not in component.active(0, people.values())
    with pytest.raises(ValueError):
        activation_for("random_sequential")


# --- M1.F.1b / M4.E.1a: witnesses ----------------------------------------------------


def test_m1f1b_witnesses_computed_from_household_and_conductance(family, vis):
    result = vis.resolve(move(RAVI, (MARTA,)), family.people, family.ties)
    # Nadia and Pia share the household and have live ties to both; Ana, Sofia and Bruno do not share it.
    assert result.event.witnesses == (NADIA, PIA)


def test_m1f1b_a_different_household_is_not_a_witness(family, vis):
    people = dict(family.people)
    people[PIA] = dataclasses.replace(people[PIA], household_id="away")
    result = vis.resolve(move(RAVI, (MARTA,)), people, family.ties)
    assert result.event.witnesses == (NADIA,)


def test_m1f1b_zero_conductance_tie_does_not_witness(family, vis):
    ties = dict(family.ties)
    for anchor in (RAVI, MARTA):
        tie_id = TieId.of(anchor, PIA)
        ties[tie_id] = dataclasses.replace(ties[tie_id], conductance=0.0)
    assert vis.resolve(move(RAVI, (MARTA,)), family.people, ties).event.witnesses == (NADIA,)


def test_m1f1b_private_route_has_no_witnesses(family, vis):
    routed = move(RAVI, (MARTA,), route=(ANA,), fidelity=vis.fidelity_for((ANA,)))
    assert vis.resolve(routed, family.people, family.ties).event.witnesses == ()


def test_m4e1a_witnesses_filled_by_visibility_not_script(family, vis):
    event = move(RAVI, (MARTA,))
    assert event.witnesses == () and not event.witnesses_computed
    resolved = vis.resolve(event, family.people, family.ties).event
    assert resolved.witnesses_computed


def test_m85_visibility_calls_the_predicate_and_separates_aligned_third_parties(family, vis):
    """A household member fused into the sender is not a neutral witness (M8.5, M8.6)."""
    result = vis.resolve(move(RAVI, (MARTA,)), family.people, family.ties, fusion={NADIA: RAVI})
    assert result.event.witnesses == (PIA,)
    assert result.peripheral_candidates == (NADIA,)


def test_m85_fusion_between_sender_and_target_does_not_unseat_a_neutral_witness(family, vis):
    result = vis.resolve(move(RAVI, (MARTA,)), family.people, family.ties, fusion={RAVI: MARTA})
    assert result.event.witnesses == (NADIA, PIA)


# --- deliveries: M3.C.2 --------------------------------------------------------------


def test_m3c2_target_latency_comes_from_the_tie(family, vis):
    result = vis.resolve(move(RAVI, (SOFIA,), tick=4), family.people, family.ties)
    [delivery] = [d for d in result.deliveries if d.role is Role.TARGET]
    assert (delivery.recipient, delivery.latency, delivery.delivered_tick) == (SOFIA, 2, 6)


def test_m3c2_witness_hears_with_the_earliest_target(family, vis):
    result = vis.resolve(move(RAVI, (MARTA,), tick=4), family.people, family.ties)
    ticks = {d.recipient: d.delivered_tick for d in result.deliveries}
    assert ticks == {MARTA: 5, NADIA: 5, PIA: 5}
    assert {d.role for d in result.deliveries if d.recipient in (NADIA, PIA)} == {Role.WITNESS}


def test_m3c2_exogenous_event_arrives_in_its_own_tick(family, vis):
    job_loss = Event(
        id=EventId(12, "script:1", 0), kind="JOB_LOSS", mechanism=Mechanism.EXOGENOUS_STRESSOR,
        sender=None, targets=(RAVI,), intensity=40.0, timestamp=12, duration=34, exogenous=True,
        source_position=SourcePosition.NONE, channel=Channel.EXOGENOUS,
    )
    result = vis.resolve(job_loss, family.people, family.ties)
    target = [d for d in result.deliveries if d.role is Role.TARGET][0]
    assert (target.delivered_tick, target.latency) == (12, 0)
    assert result.event.witnesses == (MARTA, NADIA, PIA)


def test_m4a2_trigger_reaches_no_one(family, vis):
    trigger = Event(
        id=EventId(10, "script:2", 0), kind="TRIGGER", mechanism=Mechanism.TRIGGER, sender=None,
        targets=(), intensity=8.0, timestamp=10, duration=1, exogenous=True,
        source_position=SourcePosition.NONE, channel=Channel.EXOGENOUS, on_tie=TieId.of(ANA, BRUNO),
    )
    result = vis.resolve(trigger, family.people, family.ties)
    assert result.deliveries == () and result.event.witnesses == ()


def test_m3c2_a_move_needs_a_tie(family, vis):
    with pytest.raises(MissingTie):
        vis.resolve(move(NADIA, (SOFIA,)), family.people, family.ties)


# --- M1.F.4 -----------------------------------------------------------------------------


def test_m1f4_fidelity_degrades_per_private_hop(vis):
    assert vis.fidelity_for(()) == 1.0
    assert vis.fidelity_for((ANA,)) == pytest.approx(0.8)
    assert vis.fidelity_for((ANA, SOFIA)) == pytest.approx(0.64)
    assert vis.fidelity_for((ANA, SOFIA)) < vis.fidelity_for((ANA,)) < vis.fidelity_for(())
