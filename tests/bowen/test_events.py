"""The event record, the delivery queue, the event store, and event kinds.

Purpose: prove the M1.F record constraints, latency-based scheduling, canonical
         same-tick batches, per-person store queries, and config-sourced kinds.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.F, #M3.C.2, #M16.A.2, #M16.B.3, #M10.B.1, #M5.A.1
Tests:   this file
"""

from __future__ import annotations

import dataclasses
import random
import re
from pathlib import Path

import pytest

from src.bowen.engine.event_store import EventStore
from src.bowen.engine.events import (
    BinderKind,
    BinderRef,
    Channel,
    Delivery,
    Event,
    EventId,
    EventQueue,
    LateDelivery,
    Mechanism,
    Role,
    SourcePosition,
)
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.io.load import load_event_kinds
from src.bowen.scenario.config_parse import ConfigError
from src.bowen.scenario.event_kinds import parse_event_kinds

REPO = Path(__file__).resolve().parents[2]
RAVI, MARTA, NADIA, PIA = (PersonId(n) for n in ("ravi", "marta", "nadia", "pia"))
ANA, BRUNO = PersonId("ana"), PersonId("bruno")


def move(tick=1, sender=RAVI, targets=(MARTA,), index=0, **overrides) -> Event:
    values = dict(
        id=EventId(tick, str(sender), index),
        kind="CONFLICT",
        mechanism=Mechanism.MOVE,
        sender=sender,
        targets=tuple(sorted(targets)),
        intensity=5.0,
        timestamp=tick,
        duration=1,
        exogenous=False,
        source_position=SourcePosition.NONE,
        channel=Channel.SCRIPTED,
    )
    values.update(overrides)
    return Event(**values)


def delivery(event: Event, recipient: PersonId, latency: int = 1, role: Role = Role.TARGET) -> Delivery:
    return Delivery(
        delivered_tick=event.timestamp + latency,
        event_id=event.id,
        recipient=recipient,
        role=role,
        emitted_tick=event.timestamp,
        latency=latency,
    )


# --- M1.F.1: the record ---------------------------------------------------------


def test_m1f1_event_carries_all_fields():
    required = {
        "sender", "targets", "witnesses", "kind", "intensity", "timestamp", "duration",
        "exogenous", "source_position", "route", "fidelity", "channel",  # M1.F.1, M1.F.1a
    }
    assert required <= {f.name for f in dataclasses.fields(Event)}
    e = move()
    assert e.channel is Channel.SCRIPTED and e.route == () and e.fidelity == 1.0


def test_m1f1b_sender_cannot_choose_witnesses():
    with pytest.raises(ValueError, match="visibility component"):
        move(witnesses=(NADIA,))
    computed = move().with_witnesses((PIA, NADIA))
    assert computed.witnesses == (NADIA, PIA) and computed.witnesses_computed


def test_m1f1b_witness_cannot_be_target_or_sender():
    with pytest.raises(ValueError, match="witness"):
        move().with_witnesses((MARTA,))


def test_m1f1a_mixed_channel_carries_a_weight():
    with pytest.raises(ValueError, match="weight"):
        move(channel=Channel.MIXED)
    assert move(channel=Channel.MIXED, channel_weight=0.3).channel_weight == 0.3
    with pytest.raises(ValueError, match="only for a mixed"):
        move(channel_weight=0.3)


def test_m1f1_targets_are_canonical():
    with pytest.raises(ValueError, match="canonical"):
        dataclasses.replace(move(targets=(MARTA, NADIA)), targets=(NADIA, MARTA))


def test_m1f4_fidelity_is_bounded():
    for bad in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="fidelity"):
            move(fidelity=bad)


def test_m1f6_scripted_stressors_are_spells():
    """A stressor has a start (its tick) and a whole-tick duration; there is no probability field."""
    spell = Event(
        id=EventId(12, "script:3", 0), kind="JOB_LOSS", mechanism=Mechanism.EXOGENOUS_STRESSOR,
        sender=None, targets=(RAVI,), intensity=40.0, timestamp=12, duration=34, exogenous=True,
        source_position=SourcePosition.NONE, channel=Channel.EXOGENOUS,
    )
    assert spell.duration == 34
    assert not any("prob" in f.name for f in dataclasses.fields(Event))
    with pytest.raises(ValueError, match="duration"):
        dataclasses.replace(spell, duration=0)


def test_m1f7_exogenous_flag_counts_separately():
    with pytest.raises(ValueError, match="exogenous=True"):
        Event(
            id=EventId(1, "script:1", 0), kind="JOB_LOSS", mechanism=Mechanism.EXOGENOUS_STRESSOR,
            sender=None, targets=(RAVI,), intensity=1.0, timestamp=1, duration=1, exogenous=False,
            source_position=SourcePosition.NONE, channel=Channel.EXOGENOUS,
        )
    with pytest.raises(ValueError, match="endogenous"):
        move(exogenous=True)
    store = EventStore()
    store.record_event(move())
    store.record_event(
        Event(
            id=EventId(1, "script:1", 0), kind="JOB_LOSS", mechanism=Mechanism.EXOGENOUS_STRESSOR,
            sender=None, targets=(RAVI,), intensity=1.0, timestamp=1, duration=1, exogenous=True,
            source_position=SourcePosition.NONE, channel=Channel.EXOGENOUS,
        )
    )
    assert store.count_by_origin() == {"exogenous": 1, "endogenous": 1}


def test_m1f9_binder_unavailable_names_its_binder():
    base = dict(
        id=EventId(5, "script:2", 0), kind="BINDER_UNAVAILABLE", sender=None, targets=(MARTA, RAVI),
        intensity=0.0, timestamp=5, duration=1, exogenous=True,
        source_position=SourcePosition.NONE, channel=Channel.EXOGENOUS,
    )
    with pytest.raises(ValueError, match="binder"):
        Event(mechanism=Mechanism.BINDER_UNAVAILABLE, **base)
    named = Event(
        mechanism=Mechanism.BINDER_UNAVAILABLE,
        binder=BinderRef(BinderKind.TIE_DISTANCE, str(TieId.of(RAVI, MARTA))),
        **base,
    )
    assert named.binder.kind is BinderKind.TIE_DISTANCE
    with pytest.raises(ValueError, match="binder"):
        move(binder=BinderRef(BinderKind.TIE_DISTANCE, "x"))


def test_m4a2_trigger_names_a_tie_and_needs_no_target_contact():
    trigger = Event(
        id=EventId(10, "script:5", 0), kind="TRIGGER", mechanism=Mechanism.TRIGGER, sender=None,
        targets=(), intensity=8.0, timestamp=10, duration=1, exogenous=True,
        source_position=SourcePosition.NONE, channel=Channel.EXOGENOUS, on_tie=TieId.of(ANA, BRUNO),
    )
    assert trigger.on_tie == TieId.of(ANA, BRUNO) and trigger.targets == ()
    with pytest.raises(ValueError, match="act on a tie"):
        dataclasses.replace(trigger, on_tie=None)


# --- M3.C.2 and M16.A.2: scheduling ---------------------------------------------


def test_m3c2_delivery_tick_is_emitted_plus_latency():
    e = move(tick=4)
    d = delivery(e, MARTA, latency=3)
    assert (d.emitted_tick, d.delivered_tick, d.latency) == (4, 7, 3)
    with pytest.raises(ValueError, match="plus latency"):
        Delivery(delivered_tick=5, event_id=e.id, recipient=MARTA, role=Role.TARGET, emitted_tick=4, latency=3)


def test_m3c1_queue_releases_each_delivery_at_its_tick():
    queue = EventQueue()
    e = move(tick=1, targets=(MARTA, NADIA))
    queue.schedule(delivery(e, MARTA, latency=1))
    queue.schedule(delivery(e, NADIA, latency=3))
    assert [d.recipient for d in queue.release(2)] == [MARTA]
    assert queue.release(3) == ()
    assert [d.recipient for d in queue.release(4)] == [NADIA]


def test_m1f8_batch_order_does_not_depend_on_scheduling_order():
    events = [move(tick=1, sender=s, targets=(t,)) for s, t in ((RAVI, MARTA), (MARTA, NADIA), (NADIA, PIA), (PIA, RAVI))]
    deliveries = [delivery(e, e.targets[0]) for e in events]
    batches = set()
    rng = random.Random(0)  # test-side shuffling only; the engine never uses stdlib random
    for _ in range(10):
        shuffled = deliveries[:]
        rng.shuffle(shuffled)
        queue = EventQueue()
        for d in shuffled:
            queue.schedule(d)
        batches.add(queue.release(2))
    assert len(batches) == 1


def test_m3c1_queue_refuses_late_or_skipped_deliveries():
    queue = EventQueue()
    e = move(tick=1)
    queue.release(1)
    queue.release(2)
    with pytest.raises(LateDelivery):
        queue.schedule(delivery(e, MARTA, latency=1))
    queue2 = EventQueue()
    queue2.schedule(delivery(e, MARTA, latency=1))
    with pytest.raises(LateDelivery, match="never released"):
        queue2.release(5)


# --- M16.B.3: the store ----------------------------------------------------------


def test_m16b3_event_store_answers_per_person_queries():
    store = EventStore()
    a = move(tick=1, sender=RAVI, targets=(MARTA,)).with_witnesses((NADIA,))
    b = move(tick=2, sender=MARTA, targets=(PIA,))
    for e in (a, b):
        store.record_event(e)
    store.record_delivery(delivery(a, MARTA))
    store.record_delivery(delivery(a, NADIA, role=Role.WITNESS))
    store.record_delivery(delivery(b, PIA))
    assert store.view_of(NADIA) == (a,)
    assert store.view_of(MARTA) == (a, b)       # received a, sent b
    assert [d.event_id for d in store.delivered_to(NADIA, Role.WITNESS)] == [a.id]
    assert store.sent_by(RAVI) == (a,)


def test_m16b3_event_store_has_no_off_switch():
    """M16.B.3: the store is not disableable — no flag, no parameter."""
    assert EventStore.__init__.__code__.co_argcount == 1
    assert not [n for n in vars(EventStore) if re.search(r"enable|disable|active", n)]


def test_m16b3_store_refuses_orphan_and_duplicate_records():
    store = EventStore()
    e = move()
    with pytest.raises(ValueError, match="unrecorded"):
        store.record_delivery(delivery(e, MARTA))
    store.record_event(e)
    with pytest.raises(ValueError, match="twice"):
        store.record_event(e)


# --- M10.B.1: kinds come from config ------------------------------------------------


def test_m10b1_event_kinds_come_from_config():
    kinds = load_event_kinds()
    assert kinds.mechanism_of("TRIGGER") is Mechanism.TRIGGER
    assert kinds.mechanism_of("JOB_LOSS") is Mechanism.EXOGENOUS_STRESSOR
    with pytest.raises(ValueError, match="unknown event kind"):
        kinds.mechanism_of("JOB_LOSSS")


def test_m5a1_configured_moves_are_exactly_the_spec_repertoire():
    spec = (REPO / "docs" / "bowen_agent_model_spec_v2.md").read_text(encoding="utf-8")
    core_line = re.search(r"^\*\*M5\.A\.1\*\*.*$", spec, re.M).group(0)
    core = set(re.findall(r"`([A-Z][A-Z_-]+)`", core_line))
    added = set(re.findall(r"^\*\*M5\.B\.\d+a?\*\* `([A-Z_]+)`", spec, re.M))
    added |= set(re.findall(r"`(SPLIT|FRAME_AMBIGUITY|DISPLACE)`", re.search(r"^\*\*M5\.B\.4\*\*.*$", spec, re.M).group(0)))
    assert len(core) == 9 and len(added) == 7  # PROVOKE, M5.B.6, added at spec revision 12
    assert load_event_kinds().moves() == core | added


def test_m10b2_event_kinds_parse_strictly():
    header = "| kind | mechanism | inside_sign | outside_sign | contact | impingement | accommodates | spec |\n|---|---|---|---|---|---|---|---|\n"
    with pytest.raises(ConfigError, match="unknown mechanism"):
        parse_event_kinds(header + "| `X` | magic | +1 | +1 | 0 | 0 | no | `M1.F.1` |\n")
    with pytest.raises(ConfigError, match="duplicate"):
        parse_event_kinds(header + "| `X` | move | +1 | +1 | 0 | 0 | no | `M5.A.1` |\n| `X` | move | +1 | +1 | 0 | 0 | no | `M5.A.1` |\n")
    with pytest.raises(ConfigError, match="upper case"):
        parse_event_kinds(header + "| `job_loss` | exogenous_stressor | +1 | +1 | 0 | 0 | no | `M1.F.6` |\n")


def test_m1f2_signs_parse_and_default_to_plus_one():
    header = "| kind | mechanism | inside_sign | outside_sign | contact | impingement | accommodates | spec |\n|---|---|---|---|---|---|---|---|\n"
    kinds = parse_event_kinds(header + "| `NAME_IT` | move | +1 | -1 | 0 | 0 | no | `M1.F.2` |\n")
    assert kinds.sign("NAME_IT", SourcePosition.OUTSIDE) == -1
    assert kinds.sign("NAME_IT", SourcePosition.INSIDE) == 1
    assert kinds.sign("NAME_IT", SourcePosition.NONE) == 1
    with pytest.raises(ConfigError, match="must be \\+1 or -1"):
        parse_event_kinds(header + "| `X` | move | 0 | +1 | 0 | 0 | no | `M1.F.2` |\n")
    repo = load_event_kinds()
    assert all(repo.sign(k, p) == 1 for k in repo.mechanisms for p in SourcePosition)


def test_m4c1_event_kind_components_parse_and_are_bounded():
    """Each kind's contact and impingement components come from config, each in [-1, 1] (M4.C.1, plan D2)."""
    header = "| kind | mechanism | inside_sign | outside_sign | contact | impingement | accommodates | spec |\n|---|---|---|---|---|---|---|---|\n"
    kinds = parse_event_kinds(header + "| `POKE` | move | +1 | +1 | 0.3 | 0.6 | no | `M5.A.1` |\n")
    assert kinds.components_of("POKE") == (0.3, 0.6)
    assert kinds.components_of("ABSENT") == (0.0, 0.0)
    with pytest.raises(ConfigError, match="impingement must be in"):
        parse_event_kinds(header + "| `POKE` | move | +1 | +1 | 0.3 | 1.5 | no | `M5.A.1` |\n")
    with pytest.raises(ConfigError, match="contact must be a number"):
        parse_event_kinds(header + "| `POKE` | move | +1 | +1 | lots | 0 | no | `M5.A.1` |\n")


def test_m4c1_conflict_carries_both_components():
    """KS04.13: conflict keeps contact and enforces distance at once — the configured kind carries both."""
    contact, impingement = load_event_kinds().components_of("CONFLICT")
    assert contact > 0 and impingement > 0
