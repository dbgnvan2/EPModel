"""Run-log records: what is recorded, and how it serialises.

Purpose: prove the M16.A record shapes exist and serialise canonically.
Spec:    docs/bowen_agent_model_spec_v2.md#M16.A, #M16.B.1, #M3.D.5
Tests:   this file
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json

import pytest

from src.bowen.engine.events import Channel, Delivery, Event, EventId, Mechanism, Role, SourcePosition
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.log_records import (
    BeliefWriteRecord,
    CollectingEmitter,
    DecidedBy,
    DeliveredRecord,
    EffectRecord,
    EmittedRecord,
    InvariantRecord,
    InvariantStatus,
    LogHeader,
    SelectionRecord,
    serialize,
    to_dict,
)

RAVI, MARTA = PersonId("ravi"), PersonId("marta")


def header(**overrides) -> LogHeader:
    values = dict(
        seed=7, config_hash="abc", spec_revision="2.0 revision 10", instance_id="phase_b_reduced",
        activation_component="synchronous", activation_version="1",
        visibility_component="household_conductance", visibility_version="1",
        activation_regime="every person, every fast tick", constants_frozen_at=dt.date(2026, 10, 6),
        constant_changed_after_freeze=(("fast_tick_weeks", False),),
    )
    values.update(overrides)
    return LogHeader(**values)


def event() -> Event:
    return Event(
        id=EventId(1, "ravi", 0), kind="CONFLICT", mechanism=Mechanism.MOVE, sender=RAVI,
        targets=(MARTA,), intensity=5.0, timestamp=1, duration=1, exogenous=False,
        source_position=SourcePosition.NONE, channel=Channel.SCRIPTED,
    )


def test_m16a1_header_is_self_describing():
    body = to_dict(header())
    for key in ("seed", "config_hash", "spec_revision", "instance_id"):            # M16.A.1
        assert body[key] not in (None, "")
    for key in ("activation_component", "activation_version", "visibility_component",
                "visibility_version", "activation_regime"):                          # M16.A.1a
        assert body[key]
    assert body["record_type"] == "header"


def test_m16a7_header_records_constant_change_flags():
    body = to_dict(header())
    assert body["constant_changed_after_freeze"] == [["fast_tick_weeks", False]]
    assert body["constants_frozen_at"] == "2026-10-06"


def test_m16a2_event_records_emitted_and_delivered_times():
    e = event()
    d = Delivery(delivered_tick=3, event_id=e.id, recipient=MARTA, role=Role.TARGET, emitted_tick=1, latency=2)
    body = to_dict(DeliveredRecord(d))["delivery"]
    assert (body["emitted_tick"], body["delivered_tick"], body["latency"]) == (1, 3, 2)
    emitted = to_dict(EmittedRecord(e))["event"]
    for name in ("sender", "targets", "witnesses", "kind", "intensity", "timestamp", "duration",
                 "exogenous", "source_position", "route", "fidelity", "channel"):
        assert name in emitted


def test_m16a3_selection_record_present_for_scripted_moves():
    record = SelectionRecord(tick=1, actor=RAVI, decided_by=DecidedBy.SCRIPTED, event_id=EventId(1, "ravi", 0))
    body = to_dict(record)
    assert body["decided_by"] == "scripted"                                           # M16.A.3c
    for name in ("propensities", "draw", "beliefs_used", "legal_set"):                # M16.A.3, .3a, .3b
        assert name in body
    assert {d.value for d in DecidedBy} == {"scripted", "policy", "tie_break", "fallback"}


def test_m16a4_effects_recorded_beside_cause():
    record = EffectRecord(
        tick=2, mechanism="base_appraisal", cause=EventId(1, "ravi", 0),
        acute_anxiety=((MARTA, 0.25),), ties=((TieId.of(RAVI, MARTA), "bond_energy", 0.0),),
    )
    body = to_dict(record)
    assert body["cause"] == "1:ravi:0"
    assert body["acute_anxiety"] == [["marta", 0.25]]
    assert body["ties"] == [["marta~ravi", "bond_energy", 0.0]]
    inv = to_dict(InvariantRecord(tick=2, results=(("M6.I.6", InvariantStatus.DISABLED),)))
    assert inv["results"] == [["M6.I.6", "disabled"]]


def test_m16a5_belief_writes_tagged_apart():
    record = BeliefWriteRecord(tick=3, holder=MARTA, subject="tie:marta~ravi", value=0.4, true_value=0.7)
    assert to_dict(record)["record_type"] == "belief_write"
    assert record.discrepancy() == pytest.approx(-0.3)                                # M16.A.5a
    assert BeliefWriteRecord(3, MARTA, "x", 0.4, None).discrepancy() is None
    types = {cls.record_type for cls in (LogHeader, EmittedRecord, DeliveredRecord, SelectionRecord,
                                          EffectRecord, InvariantRecord, BeliefWriteRecord)}
    assert len(types) == 7                                                             # every tag distinct


def test_m16a6_serialisation_is_canonical():
    record = EffectRecord(tick=2, mechanism="standing_load", cause=None, acute_anxiety=((MARTA, 0.1),))
    line = serialize(record)
    assert line == serialize(dataclasses.replace(record))
    assert " " not in line and "\n" not in line
    assert list(json.loads(line)) == sorted(json.loads(line))
    with pytest.raises(ValueError):
        serialize(EffectRecord(tick=2, mechanism="x", cause=None, acute_anxiety=((MARTA, float("nan")),)))


def test_m16b1_emitter_collects_without_writing():
    emitter = CollectingEmitter()
    emitter.emit(header())
    assert emitter.records == [header()]


def test_m16a1a_header_records_activation_and_visibility_components():
    """The header names both components, their versions and the activation regime (M3.E.1, M3.E.2)."""
    from src.bowen.run import run_phase_b

    header = run_phase_b(3).records[0]
    assert (header.activation_component, header.activation_version) == ("synchronous_activation", "1")
    assert (header.visibility_component, header.visibility_version) == ("household_conductance_visibility", "1")
    assert header.activation_regime == "synchronous"


def test_m16a5a_belief_discrepancy_is_computable_per_record():
    over = BeliefWriteRecord(tick=1, holder=MARTA, subject="tie:marta~ravi", value=0.9, true_value=0.5)
    under = BeliefWriteRecord(tick=1, holder=MARTA, subject="tie:marta~ravi", value=0.2, true_value=0.5)
    assert over.discrepancy() == pytest.approx(0.4) and under.discrepancy() == pytest.approx(-0.3)
    assert to_dict(over)["true_value"] == 0.5  # both values travel in the log, so a reader can recompute it
