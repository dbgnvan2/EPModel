"""The supply readout (`M11.G.5`, spec revision 55): who supplied whom, computed after a run.

Purpose: the counts per ordered pair and the mean perceived cue per pair, on hand-built records and one short run.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.G.5
Tests:   this file
"""

from __future__ import annotations

from src.bowen.engine.events import Channel, Event, EventId, Mechanism, SourcePosition
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.log_records import EmittedRecord
from src.bowen.engine.signals import Cue
from src.bowen.ensemble.criteria import scenario
from src.bowen.readouts.roles import ReadingTap, load_supply_kinds, main_supplier, supply_counts

P = PersonId
A, B, C = P("a"), P("b"), P("c")


def _move(kind, sender, target, tick, index=0):
    return EmittedRecord(Event(
        id=EventId(tick, sender.value, index), kind=kind, mechanism=Mechanism.MOVE, sender=sender, targets=(target,),
        intensity=1.0, timestamp=tick, duration=1, exogenous=False, source_position=SourcePosition.NONE,
        channel=Channel.AUTOMATIC,
    ))


def test_m11g5_supply_kinds_come_from_config():
    assert {"STAY-IN-CONTACT", "OVERFUNCTION"} <= load_supply_kinds()


def test_m11g5_supply_counts_per_ordered_pair():
    records = [_move("STAY-IN-CONTACT", A, B, 0), _move("OVERFUNCTION", A, B, 1), _move("CONFLICT", A, B, 2),
               _move("STAY-IN-CONTACT", B, A, 3), _move("STAY-IN-CONTACT", C, B, 9)]
    counts = supply_counts(records)
    assert counts[A, B] == 2 and counts[B, A] == 1 and counts[C, B] == 1
    assert supply_counts(records, since=1, until=9)[A, B] == 1
    assert main_supplier(counts, B, (A, C)) == A
    assert main_supplier(counts, B, (C,)) == C
    assert main_supplier(counts, A, (C,)) is None


def test_m11g5_mean_received_cue_per_pair():
    tap = ReadingTap()
    state, _ = scenario("dyad", 0, 6, watch=tap, overrides={"social_cues_enabled": 1})
    means = tap.mean_received(Cue.ATTENTION)
    assert means and all(-1.0 <= v <= 1.0 for v in means.values())
    assert (A, B) in means or (B, A) in means  # the dyad fixture also holds the parents of a and b
    off = ReadingTap()
    scenario("dyad", 0, 6, watch=off)
    assert off.mean_received(Cue.ATTENTION) == {}
