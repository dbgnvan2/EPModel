"""The fast tick's order and the slow-tick hook.

Purpose: prove M3.D.1's order, the two ordering requirements M3.D.2 and M3.D.3,
         one record per fast tick, and the slow tick firing every 52.
Spec:    docs/bowen_agent_model_spec_v2.md#M3.A.1, #M3.B.1, #M3.C.1, #M3.D.1, #M3.D.2, #M3.D.3
Tests:   this file
"""

from __future__ import annotations

import pytest

from src.bowen.engine.act import Selection
from src.bowen.engine.activation import SynchronousActivation
from src.bowen.engine.events import Channel, Event, EventId, SourcePosition
from src.bowen.engine.identifiers import PersonId, TieId, TriangleId
from src.bowen.engine.log_records import CollectingEmitter, DeliveredRecord, EffectRecord, TickRecord
from src.bowen.engine.state import new_run_state
from src.bowen.engine.tick import STEPS, SelectionFromInactivePerson, run, run_tick
from src.bowen.engine.visibility import HouseholdConductanceVisibility
from src.bowen.io.load import load_constants, load_event_kinds, load_family
from src.bowen.scenario.params import engine_params

P = PersonId
RAVI, MARTA, NADIA, PIA, ANA, SOFIA, BRUNO = (P(n) for n in ("ravi", "marta", "nadia", "pia", "ana", "sofia", "bruno"))
PARAMS = engine_params(load_constants())
VIS = HouseholdConductanceVisibility(PARAMS.per_hop_fidelity)
ACT = SynchronousActivation()


class FixedSource:
    """A test source: events and selections keyed by tick."""

    def __init__(self, events=None, selections=None, on_select=None):
        self._events = events or {}
        self._selections = selections or {}
        self._on_select = on_select

    def scheduled(self, tick):
        return tuple(self._events.get(tick, ()))

    def selections(self, tick, active, state):
        if self._on_select:
            self._on_select(tick, state)
        return tuple(self._selections.get(tick, ()))


def fresh():
    family = load_family()
    return new_run_state(dict(family.people), dict(family.ties), family.family, load_event_kinds())


def test_m3d1_steps_run_in_spec_order():
    assert run_tick(fresh(), FixedSource(), PARAMS, VIS, ACT, CollectingEmitter()) == STEPS
    assert STEPS == (
        "standing_load", "deliver", "perceive", "appraise", "involvement",
        "triangles", "select", "act", "consolidate",
    )


def test_m3a1_one_record_per_fast_tick():
    emitter = CollectingEmitter()
    state = run(fresh(), FixedSource(), PARAMS, VIS, ACT, emitter, ticks=40)
    ticks = [r.tick for r in emitter.records if isinstance(r, TickRecord)]
    assert ticks == list(range(40)) and state.tick == 40


def test_m3b1_slow_tick_fires_every_52_fast_ticks():
    emitter = CollectingEmitter()
    run(fresh(), FixedSource(), PARAMS, VIS, ACT, emitter, ticks=105)
    fired = [r.tick for r in emitter.records if isinstance(r, EffectRecord) and r.mechanism == "slow_tick"]
    assert fired == [51, 103]


def test_m3d2_standing_load_precedes_delivery():
    """Within every tick, the standing-load record comes before any delivered record."""
    selections = {0: (Selection(RAVI, "CONFLICT", (MARTA,), 5.0),)}
    emitter = CollectingEmitter()
    run(fresh(), FixedSource(selections=selections), PARAMS, VIS, ACT, emitter, ticks=3)
    tick = None
    seen_standing = False
    delivered_ticks = []
    for record in emitter.records:
        if isinstance(record, TickRecord):
            tick, seen_standing = record.tick, False
        elif isinstance(record, EffectRecord) and record.mechanism == "standing_load":
            seen_standing = True
        elif isinstance(record, DeliveredRecord):
            assert seen_standing, f"tick {tick}: delivery before standing load"
            delivered_ticks.append(tick)
    assert delivered_ticks == [1, 1, 1]  # Marta, plus witnesses Nadia and Pia, one tick later


def test_m3d3_involvement_and_triangles_precede_select():
    """At selection the triangle state already reflects this tick's appraisal."""
    seen = {}

    def on_select(tick, state):
        seen[tick] = state.triangles[TriangleId.of(RAVI, MARTA, NADIA)].active

    job_loss = Event(
        id=EventId(2, "script:1", 0), kind="JOB_LOSS", mechanism=load_event_kinds().mechanism_of("JOB_LOSS"),
        sender=None, targets=(RAVI,), intensity=400.0, timestamp=2, duration=34, exogenous=True,
        source_position=SourcePosition.NONE, channel=Channel.EXOGENOUS,
    )
    run(fresh(), FixedSource(events={2: (job_loss,)}, on_select=on_select), PARAMS, VIS, ACT, CollectingEmitter(), ticks=3)
    assert seen[1] is False and seen[2] is True


def test_m3c1_delivery_uses_edge_latency():
    """Latency 1 (Ravi–Marta) and latency 2 (Ravi–Sofia), through the whole loop."""
    selections = {0: (Selection(RAVI, "CONFLICT", (MARTA,), 5.0),), 1: (Selection(RAVI, "PURSUE", (SOFIA,), 5.0),)}
    emitter = CollectingEmitter()
    run(fresh(), FixedSource(selections=selections), PARAMS, VIS, ACT, emitter, ticks=4)
    targets = {
        (r.delivery.recipient, r.delivery.emitted_tick): r.delivery.delivered_tick
        for r in emitter.records
        if isinstance(r, DeliveredRecord) and r.delivery.role.value == "target"
    }
    assert targets == {(MARTA, 0): 1, (SOFIA, 1): 3}


def test_m3e1_only_activated_people_may_select():
    state = fresh()
    state.people[BRUNO].alive = False
    source = FixedSource(selections={0: (Selection(BRUNO, "PURSUE", (ANA,), 1.0),)})
    with pytest.raises(SelectionFromInactivePerson):
        run_tick(state, source, PARAMS, VIS, ACT, CollectingEmitter())


def test_m4a2_trigger_in_the_loop_spikes_the_next_tick():
    trigger = Event(
        id=EventId(10, "script:2", 0), kind="TRIGGER", mechanism=load_event_kinds().mechanism_of("TRIGGER"),
        sender=None, targets=(), intensity=1.0, timestamp=10, duration=1, exogenous=True,
        source_position=SourcePosition.NONE, channel=Channel.EXOGENOUS, on_tie=TieId.of(ANA, BRUNO),
    )

    def loads(source):
        emitter = CollectingEmitter()
        run(fresh(), source, PARAMS, VIS, ACT, emitter, ticks=13)
        return {
            r.tick: dict(r.acute_anxiety)[ANA]
            for r in emitter.records
            if isinstance(r, EffectRecord) and r.mechanism == "standing_load"
        }

    quiet, spiked = loads(FixedSource()), loads(FixedSource(events={10: (trigger,)}))
    assert spiked[10] == quiet[10] and spiked[11] > quiet[11] and spiked[12] == quiet[12]
