"""The event record, its deliveries, and the delivery queue.

Purpose: carry every field `M1.F` requires on an event, schedule each recipient's
         delivery by its tie's latency, and hand out same-tick deliveries as one
         canonically ordered batch.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.F, #M3.C.1, #M3.C.2, #M16.A.2
Tests:   tests/bowen/test_events.py

Event *kinds* are editorial and are named in ``config/bowen/event_kinds.md``
(M10.B.1). What the engine can do with an event is fixed here as a
``Mechanism``; the config maps each kind to one.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping

from src.bowen.engine.identifiers import PersonId, TieId


class Mechanism(enum.Enum):
    """What the engine does with an event. Kinds map onto these in config."""

    MOVE = "move"
    TRIGGER = "trigger"                        # M4.A.2 — acts through the standing term, no contact
    RECONCILIATION = "reconciliation"          # M4.A.3
    INSTITUTIONALIZE = "institutionalize"      # M4.A.4
    BINDER_UNAVAILABLE = "binder_unavailable"  # M1.F.9
    EXOGENOUS_STRESSOR = "exogenous_stressor"  # M1.F.6 — a spell, never a per-tick draw
    ENDOGENOUS_SYMPTOM = "endogenous_symptom"  # M7.D.1 — emitted when symptom load crosses threshold (M1.F.7)


TIE_MECHANISMS = frozenset({Mechanism.TRIGGER, Mechanism.RECONCILIATION})


@dataclass(frozen=True)
class EventKinds:
    """The kind vocabulary, as loaded from config.

    ``signs`` maps (kind, source position) to +1 or -1 (M1.F.2). A missing
    entry, and every event sent from no position, is +1.

    ``components`` maps a kind to its (contact, impingement) components per unit
    of scaled intensity (M4.C.1, revision 11; Phase C plan D2). A kind with no
    entry moves neither.
    """

    mechanisms: Mapping[str, Mechanism]
    signs: Mapping[tuple[str, "SourcePosition"], int] = field(default_factory=lambda: MappingProxyType({}))
    components: Mapping[str, tuple[float, float]] = field(default_factory=lambda: MappingProxyType({}))
    # M1.A.9a, FE03.1: the kinds that give way to the other (the inward axis).
    accommodating: frozenset[str] = frozenset()
    # M4.D.1a: the policy channel each move belongs to — "automatic", "self", or absent
    # (not selectable by the policy). M4.D.3a: each automatic kind's capacity layer,
    # 0 the oldest. Both editorial, from config (Phase C step 6).
    channels: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    layers: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "mechanisms", MappingProxyType(dict(self.mechanisms)))
        if any(v not in (1, -1) for v in self.signs.values()):
            raise ValueError("a source-position sign is +1 or -1")
        object.__setattr__(self, "signs", MappingProxyType(dict(self.signs)))
        if any(not (-1.0 <= c <= 1.0 and -1.0 <= i <= 1.0) for c, i in self.components.values()):
            raise ValueError("a contact or impingement component is in [-1, 1]")
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))
        if any(c not in ("automatic", "self") for c in self.channels.values()):
            raise ValueError("a policy channel is 'automatic' or 'self'")
        if set(self.layers) != {k for k, c in self.channels.items() if c == "automatic"}:
            raise ValueError("every automatic kind, and only those, has a capacity layer (M4.D.3a)")
        object.__setattr__(self, "channels", MappingProxyType(dict(self.channels)))
        object.__setattr__(self, "layers", MappingProxyType(dict(self.layers)))

    def sign(self, kind: str, position: "SourcePosition") -> int:
        if position is SourcePosition.NONE:
            return 1
        return self.signs.get((kind, position), 1)

    def components_of(self, kind: str) -> tuple[float, float]:
        return self.components.get(kind, (0.0, 0.0))

    def mechanism_of(self, kind: str) -> Mechanism:
        try:
            return self.mechanisms[kind]
        except KeyError:
            raise ValueError(f"unknown event kind {kind!r}") from None

    def in_channel(self, channel: str) -> tuple[str, ...]:
        return tuple(sorted(k for k, c in self.channels.items() if c == channel))

    def moves(self) -> frozenset[str]:
        return frozenset(k for k, m in self.mechanisms.items() if m is Mechanism.MOVE)


class SourcePosition(enum.Enum):
    """M1.F.2 — the sender's triangle position, which can flip an event's sign."""

    NONE = "none"
    INSIDE = "inside"
    OUTSIDE = "outside"


class Channel(enum.Enum):
    """M1.F.1a — which channel selected the move. Scripted events in Phase B say so."""

    AUTOMATIC = "automatic"
    SELF_DIRECTED = "self_directed"
    MIXED = "mixed"
    SCRIPTED = "scripted"
    EXOGENOUS = "exogenous"
    ENDOGENOUS = "endogenous"   # an event the engine emits, not a selection (M7.D.1)


class Attention(enum.Enum):
    """M4.C.8 — the channel an event directs the receiver's attention at, if any."""

    NONE = "none"
    FEELING = "feeling"
    INTELLECT = "intellect"


class BinderKind(enum.Enum):
    """M1.F.9 — the four binders a `binder_unavailable` event can name."""

    TIE_DISTANCE = "tie_distance"
    TRIANGLE_POSITION = "triangle_position"
    SYMPTOM_CHANNEL = "symptom_channel"
    EXTERNAL_RESPONSIBILITY = "external_responsibility"


@dataclass(frozen=True)
class BinderRef:
    kind: BinderKind
    target: str


@dataclass(frozen=True, order=True)
class EventId:
    """Structural identity: the emitting tick, who or what emitted it, and an index.

    ``origin`` is a person id for an emitted move, or ``script:<entry>`` for a
    scripted input. No run-time counter is involved (as for M1.A.20).
    """

    tick: int
    origin: str
    index: int

    def __str__(self) -> str:
        return f"{self.tick}:{self.origin}:{self.index}"


class Role(enum.Enum):
    TARGET = "target"
    WITNESS = "witness"


@dataclass(frozen=True)
class Event:
    """Purpose: one event with every field M1.F.1 and M1.F.1a require.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.F.1, #M1.F.1a, #M1.F.6, #M1.F.7, #M1.F.9
    Tests:   tests/bowen/test_events.py::test_m1f1_event_carries_all_fields
    """

    id: EventId
    kind: str
    mechanism: Mechanism
    sender: PersonId | None
    targets: tuple[PersonId, ...]
    intensity: float
    timestamp: int
    duration: int
    exogenous: bool
    source_position: SourcePosition
    channel: Channel
    route: tuple[PersonId, ...] = ()
    fidelity: float = 1.0
    witnesses: tuple[PersonId, ...] = ()
    witnesses_computed: bool = False
    channel_weight: float | None = None
    binder: BinderRef | None = None
    on_tie: TieId | None = None
    # M4.C.8: what the event directs the receiver's attention at.
    attention: Attention = Attention.NONE
    # M1.A.18a, M1.A.19: an evaluation of the target, -1 blame, +1 praise, 0 none. Two-sided by design.
    valence: int = 0
    # M5.F.4, M5.D.4a: an I-POSITION executed as the assertion form — raises reactivity on the
    # tie and counts against the claimant (M5.F.2a). Set by the engine at act, never by a policy.
    assertion: bool = False

    def __post_init__(self) -> None:
        if self.valence not in (-1, 0, 1):
            raise ValueError("valence is -1 (blame), 0 or +1 (praise)")
        if self.timestamp != self.id.tick:
            raise ValueError("timestamp must equal the emitting tick in the event id")
        if self.intensity < 0:
            raise ValueError("intensity must be non-negative")
        if isinstance(self.duration, bool) or not isinstance(self.duration, int) or self.duration < 1:
            raise ValueError("duration is a whole number of ticks, at least one (M1.F.6)")
        if not 0.0 < self.fidelity <= 1.0:
            raise ValueError("fidelity is in (0, 1]")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("duplicate target")
        if self.targets != tuple(sorted(self.targets)):
            raise ValueError("targets must be in canonical (sorted) order")
        if self.witnesses and not self.witnesses_computed:
            # M1.F.1b, M4.E.1a: the sender's policy or script never chooses witnesses.
            raise ValueError("witnesses are computed by the visibility component, not set by the sender")
        if set(self.witnesses) & (set(self.targets) | ({self.sender} if self.sender else set())):
            raise ValueError("a witness cannot also be a target or the sender")
        if (self.mechanism is Mechanism.BINDER_UNAVAILABLE) != (self.binder is not None):
            raise ValueError("a binder_unavailable event names exactly one binder; no other event does (M1.F.9)")
        if (self.mechanism in TIE_MECHANISMS) != (self.on_tie is not None):
            raise ValueError("TRIGGER and RECONCILIATION act on a tie; no other event names one")
        if self.mechanism is Mechanism.EXOGENOUS_STRESSOR and not self.exogenous:
            raise ValueError("an exogenous stressor must carry exogenous=True (M1.F.7)")
        if self.mechanism is Mechanism.MOVE:
            if self.sender is None or not self.targets:
                raise ValueError("a move has a sender and at least one target")
            if self.exogenous:
                raise ValueError("a move is endogenous")
        if self.channel is Channel.MIXED:
            if self.channel_weight is None or not 0.0 < self.channel_weight < 1.0:
                raise ValueError("a mixed channel carries a self-directed weight in (0, 1)")
        elif self.channel_weight is not None:
            raise ValueError("channel_weight is only for a mixed channel")

    def with_witnesses(self, witnesses: tuple[PersonId, ...]) -> "Event":
        """Only the visibility component calls this (M1.F.1b)."""
        return replace(self, witnesses=tuple(sorted(witnesses)), witnesses_computed=True)


@dataclass(frozen=True, order=True)
class Delivery:
    """Purpose: one recipient's copy of an event, with both timestamps (M16.A.2).
    Spec:    docs/bowen_agent_model_spec_v2.md#M16.A.2, #M3.C.1, #M3.C.2
    Tests:   tests/bowen/test_events.py::test_m3c2_delivery_tick_is_emitted_plus_latency
    """

    delivered_tick: int
    event_id: EventId
    recipient: PersonId
    role: Role = field(compare=False)
    emitted_tick: int = field(compare=False)
    latency: int = field(compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.latency, bool) or not isinstance(self.latency, int) or self.latency < 0:
            raise ValueError("latency is a whole number of ticks")
        if self.delivered_tick != self.emitted_tick + self.latency:
            raise ValueError("delivered tick must be emitted tick plus latency (M3.C.2)")


class LateDelivery(RuntimeError):
    """A delivery was scheduled for a tick that has already been delivered."""


class EventQueue:
    """Purpose: hold scheduled deliveries and release each tick's as one batch.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.F.8, #M3.D.1
    Tests:   tests/bowen/test_events.py::test_m1f8_batch_order_does_not_depend_on_scheduling_order

    The batch is returned in a canonical order (event id, then recipient), so
    the order in which deliveries were scheduled can never reach a mechanism.
    Whether *applying* a batch is order-independent is the consumer's
    obligation (step 8); this queue guarantees only that it hands out the same
    batch however it was filled.
    """

    def __init__(self) -> None:
        self._pending: dict[int, list[Delivery]] = {}
        self._released_through = -1

    def schedule(self, delivery: Delivery) -> None:
        if delivery.delivered_tick <= self._released_through:
            raise LateDelivery(
                f"delivery for tick {delivery.delivered_tick} after tick {self._released_through} was released"
            )
        self._pending.setdefault(delivery.delivered_tick, []).append(delivery)

    def release(self, tick: int) -> tuple[Delivery, ...]:
        if tick <= self._released_through:
            raise ValueError(f"tick {tick} was already released")
        stale = [t for t in self._pending if t < tick]
        if stale:
            raise LateDelivery(f"deliveries for ticks {sorted(stale)} were never released")
        self._released_through = tick
        return tuple(sorted(self._pending.pop(tick, [])))

    def pending(self) -> int:
        return sum(len(v) for v in self._pending.values())
