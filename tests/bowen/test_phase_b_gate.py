"""Phase B exit criteria that run the model: G1 and G3.

Purpose: the scripted 40-week run completes, and a TRIGGER on a dormant
         family-of-origin tie moves anxiety with no contact.
Spec:    docs/bowen_agent_model_spec_v2.md#M13 (Phase B Done when), #M4.A.2, #M10.B.4, #M0.4
Tests:   this file

Every test here is an acceptance test. M10.B.4 requires the constants to be
frozen before the acceptance suite first runs against them, so each one first
checks the freeze.
"""

from __future__ import annotations

import pytest

from src.bowen.engine.events import Role
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.log_records import CollectingEmitter, DeliveredRecord, EffectRecord, InvariantRecord, TickRecord
from src.bowen.engine.tick import run
from src.bowen.io.load import (
    CONFIG_DIR, load_constants, load_event_kinds, load_family, load_frozen_constants, load_script, read_text,
)
from src.bowen.scenario.assemble import assemble
from src.bowen.scenario.config_parse import parse_table_document

ANA, BRUNO = PersonId("ana"), PersonId("bruno")


@pytest.fixture(autouse=True)
def constants_are_frozen():
    assert load_constants().frozen_at is not None, "M10.B.4: freeze the constants before an acceptance test runs"


def run_script(source=None):
    constants, kinds, family = load_constants(), load_event_kinds(), load_family()
    script = source or load_script(kinds=kinds, family=family)
    parts = assemble(constants, kinds, family, script)
    emitter = CollectingEmitter()
    run(parts.state, parts.source, parts.params, parts.visibility, parts.activation, emitter, ticks=script.ticks)
    return parts.state, emitter.records


def test_m10b4_constants_frozen_before_suite():
    """Every current value equals the frozen snapshot, or its change is logged."""
    current, frozen = load_constants(), load_frozen_constants()
    assert current.frozen_at == frozen.frozen_at is not None
    log = parse_table_document(
        read_text(CONFIG_DIR / "constants_changes.md"),
        columns=("key", "frozen_value", "new_value", "date", "criterion", "post_hoc"),
        metadata_keys=frozenset(), source="constants_changes.md",
    )
    logged = {row["key"].strip("`") for row in log.rows}
    changed = {k for k in current.values if k not in frozen.values or current[k] != frozen[k]}
    changed |= {k for k in frozen.values if k not in current.values}  # retired since the freeze
    assert changed <= logged, f"changed after freeze without a log row: {sorted(changed - logged)}"


def test_m13_phase_b_scripted_40_week_trace_runs():
    """G1: 40 tick records, every scripted delivery made, invariants passing every tick."""
    state, records = run_script()
    script = load_script()
    assert [r.tick for r in records if isinstance(r, TickRecord)] == list(range(40))
    assert state.tick == 40 and state.queue.pending() == 0
    delivered = {(r.delivery.event_id.tick, r.delivery.event_id.origin) for r in records
                 if isinstance(r, DeliveredRecord) and r.delivery.role is Role.TARGET}
    assert {(t, str(s.actor)) for t, s in script.moves} <= delivered
    invariant_ticks = [r.tick for r in records if isinstance(r, InvariantRecord)]
    assert invariant_ticks == list(range(40))


def _ana_standing(records):
    return {r.tick: dict(r.acute_anxiety)[ANA] for r in records
            if isinstance(r, EffectRecord) and r.mechanism == "standing_load"}


def test_m4a2_g3_trigger_on_dormant_tie_moves_anxiety_without_contact():
    """G3: two arms, one script, differing only in the TRIGGER on Ana–Bruno at week 10.

    Direction only (M0.4): Ana's anxiety is higher at week 11 in the trigger arm,
    and in neither arm is any event delivered across the cut-off tie.
    """
    script = load_script()
    with_trigger, records_with = run_script(script)
    without, records_without = run_script(script.without("TRIGGER"))

    # Before the trigger lands the arms are identical.
    assert _ana_standing(records_with)[10] == _ana_standing(records_without)[10]
    assert _ana_standing(records_with)[11] > _ana_standing(records_without)[11]
    assert with_trigger.people[ANA].acute_anxiety >= without.people[ANA].acute_anxiety

    tie = TieId.of(ANA, BRUNO)
    for state in (with_trigger, without):
        # "On the tie": sent across it, or about it (a TRIGGER names its tie).
        crossing = [
            d for d in state.store.delivered_to(ANA) + state.store.delivered_to(BRUNO)
            if {state.store.event(d.event_id).sender, d.recipient} == set(tie.members())
            or state.store.event(d.event_id).on_tie == tie
        ]
        assert crossing == []
        assert not state.ties[tie].interactive
