"""The four core objects — Person, Relationship, Triangle, Family — as state.

Purpose: hold every state variable `M1` names, with the structural constraints
         that can be enforced at construction.
Spec:    docs/bowen_agent_model_spec_v2.md#M1, #M14.A
Tests:   tests/bowen/test_objects.py; tests/bowen/test_register.py

State only. Mechanisms live in their own modules and are listed, with what they
write, in the spec's `M14.A` register; every field below has a row there
(`test_m14a_register_matches_object_fields`). A field whose writer is built in
a later phase defaults to ``None`` or an empty container until that phase.

What is deliberately absent, because the spec forbids storing it:

* reactivity — derived from level and anxiety, never stored (M1.A.1);
* a family membership set — membership is a threshold over
  ``involvement_weight`` (M1.A.12);
* any material stock or resource pool (M1.A.15);
* a fourth sink for distance (M1.D.2);
* any physical distance or contact-frequency input to conductance (M1.B.2).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from src.bowen.engine.identifiers import PersonId, TieId, TriangleId

SCALE_MIN = 0.0
SCALE_MAX = 100.0


class Role(enum.Enum):
    """M1.A.17 — an external agent is a Person with a role, not a separate type."""

    MEMBER = "member"
    EXTERNAL = "external"


class Sex(enum.Enum):
    """M1.A.21 — declared data; no Phase B mechanism reads it."""

    FEMALE = "female"
    MALE = "male"


class SymptomChannel(enum.Enum):
    """M1.A.11 — exactly three channels. *Mental*, not *emotional* (M1.A.0, KS23.11)."""

    PHYSICAL = "physical"
    MENTAL = "mental"
    SOCIAL = "social"


# M1.A.19 — the reactive state's three detectors. The third is two-sided: praise
# is as much a loss as blame (K07.1), so it counts evaluations of either sign.
REACTIVE_DETECTORS = ("critical_urge", "inside_other_problem", "evaluating")


class StructuralTier(enum.Enum):
    """M1.A.13 — exactly three tiers. A finer ranking must not exist."""

    SHOCK_WAVE_LIKELY = "shock_wave_likely"
    NEUTRAL = "neutral"
    RELIEF = "relief"


class TieState(enum.Enum):
    """M1.B.3's four states, which differ in events and energy independently.

    ``ORDINARY`` is a tie in none of the four. It is the project's addition: the
    four are what the model must be able to tell apart, not a claim that every
    tie is one of them.
    """

    ORDINARY = "ordinary"
    CUT_OFF = "cut_off"
    DISTANT = "distant"
    RESOLVED_LOW_CONTACT = "resolved_low_contact"
    OPEN_CONFLICT = "open_conflict"


class Sink(enum.Enum):
    """M1.D.1 — exactly three sinks. Distance is not one of them (M1.D.2)."""

    MARITAL_CONFLICT = "marital_conflict"
    SPOUSE_DYSFUNCTION = "spouse_dysfunction"
    CHILD_PROJECTION = "child_projection"


@dataclass(frozen=True)
class SiblingPosition:
    """Static birth-order data (M1.A.14). Never read by the propensity vector."""

    rank: int
    sibship_size: int

    def __post_init__(self) -> None:
        if not 1 <= self.rank <= self.sibship_size:
            raise ValueError(f"rank {self.rank} outside a sibship of {self.sibship_size}")


@dataclass(frozen=True)
class LeadershipOffice:
    """M1.D.4 — an occupant and a sphere, which is a move-permission scope."""

    occupant: PersonId | None
    sphere: frozenset[str]


def _check_scale(name: str, value: float) -> None:
    # M1.A.2: on 0–100, and out-of-range input raises rather than being clipped.
    if not SCALE_MIN <= value <= SCALE_MAX:
        raise ValueError(f"{name}={value} is outside {SCALE_MIN}–{SCALE_MAX}; values are not clipped")


@dataclass
class Person:
    """Purpose: one human being, or one external professional (M1.A).
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.A
    Tests:   tests/bowen/test_objects.py::test_m1a2_basic_level_out_of_range_raises_not_clips
    """

    id: PersonId
    role: Role
    sex: Sex
    household_id: str
    basic_level: float
    functional_level: float
    chronic_anxiety: float
    sibling_position: SiblingPosition | None
    financially_dependent: bool
    acute_anxiety: float = 0.0
    channel_prior: SymptomChannel | None = None
    symptom_active: dict[SymptomChannel, bool] = field(
        default_factory=lambda: {channel: False for channel in SymptomChannel}
    )
    programmed_reactivity: float | None = None
    outside_ness_outward: float | None = None
    outside_ness_inward: float | None = None
    life_energy_ratio: float | None = None
    symptom_load: dict[SymptomChannel, float] = field(
        default_factory=lambda: {channel: 0.0 for channel in SymptomChannel}
    )
    involvement_weight: float = 0.0
    structural_importance: StructuralTier | None = None
    functional_sibling_position: SiblingPosition | None = None
    beliefs: dict[str, object] = field(default_factory=dict)
    # M9.8, revision 12 P2a: belief about each tie the person is not party to.
    # Written by ``beliefs.initialise_beliefs`` and step 3's ``beliefs.update_beliefs``.
    tie_beliefs: dict | None = None
    # M4.D.6: the automatic channel's learned value per (anxiety band, act, target or triangle),
    # keyed as text. Absent means the shared initial value 0 — equal for every legal act
    # (plan D3). Written by the learner from Phase C step 7.
    learned_values: dict[str, float] = field(default_factory=dict)
    # M4.D.6, plan D4: the person's automatic acts still inside the credit horizon,
    # each accumulating its signal. Written by ``learner.register_acts`` and ``learner.learn``.
    eligible_acts: list = field(default_factory=list)
    # M5.D: the person's I-POSITION sequence in progress, or None. Written by ``iposition``.
    iposition_state: object | None = None
    systems_perspective: float | None = None
    reactive_state: dict[str, float] | None = None
    pseudo_self: float | None = None
    alive: bool = True

    def __post_init__(self) -> None:
        _check_scale("basic_level", self.basic_level)
        _check_scale("functional_level", self.functional_level)
        if set(self.symptom_load) != set(SymptomChannel) or set(self.symptom_active) != set(SymptomChannel):
            raise ValueError("symptom_load and symptom_active must have exactly the three M1.A.11 channels")
        if self.reactive_state is not None and set(self.reactive_state) != set(REACTIVE_DETECTORS):
            raise ValueError(f"reactive_state must hold exactly the M1.A.19 detectors {REACTIVE_DETECTORS}")
        if not self.household_id:
            raise ValueError("household_id is required (M1.A.22)")

    @property
    def swing(self) -> float:
        """M1.A.5a — functional level decomposes as basic level plus a swing term."""
        return self.functional_level - self.basic_level


@dataclass
class Relationship:
    """Purpose: the tie between exactly two persons; the unit of coupling state (M1.B).
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.B
    Tests:   tests/bowen/test_objects.py::test_m1b1_tie_connects_exactly_two_persons
    """

    id: TieId
    conductance: float
    bond_energy: float
    latency: int
    tie_state: TieState = TieState.ORDINARY
    interactive: bool = True
    # M4.C.1, revision 11 (Phase C plan D2): per member, in [0, 1]. Written by
    # ``contact.initialise_contact``, delivered events and step 9's relaxation.
    felt_contact: dict[PersonId, float] = field(default_factory=dict)
    felt_impingement: dict[PersonId, float] = field(default_factory=dict)
    distance_bound_anxiety: float = 0.0
    functioning_balance: dict[str, float] = field(default_factory=dict)
    # M1.B.6, L05.3: the hardened configuration a flipped balance reverts toward unless sustained.
    functioning_habit: dict[str, float] = field(default_factory=dict)
    investment: dict[PersonId, float] = field(default_factory=dict)
    areas_of_joint_activity: set[str] = field(default_factory=set)
    taboo_set: set[str] = field(default_factory=set)
    dyad_age: int = 0
    basic_togetherness: float | None = None
    functional_togetherness: float | None = None

    def __post_init__(self) -> None:
        # M3.C.2: a whole number of fast ticks, at least one.
        if isinstance(self.latency, bool) or not isinstance(self.latency, int) or self.latency < 1:
            raise ValueError(f"latency must be an integer number of ticks >= 1, got {self.latency!r}")
        if self.conductance < 0 or self.bond_energy < 0:
            raise ValueError("conductance and bond_energy must be non-negative")


@dataclass
class Triangle:
    """Purpose: a persistent three-person configuration that binds anxiety (M1.C).
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.C
    Tests:   tests/bowen/test_objects.py::test_m1c3_topology_is_stored_apart_from_activity
    """

    members: TriangleId
    inside_pair: TieId | None = None
    outside: PersonId | None = None
    active: bool = False
    bound_anxiety: float = 0.0
    activation_memory: float = 0.0
    intensity_floor: float = 0.0

    def __post_init__(self) -> None:
        if (self.inside_pair is None) != (self.outside is None):
            raise ValueError("inside_pair and outside are set together or not at all")
        if self.inside_pair is not None:
            trio = set(self.members.members)
            if set(self.inside_pair.members()) | {self.outside} != trio:
                raise ValueError("inside pair and outside member must be the triangle's three members")

    @property
    def id(self) -> TriangleId:
        return self.members


@dataclass
class Family:
    """Purpose: family-level state — the budget and its three sinks (M1.D).
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.D
    Tests:   tests/bowen/test_objects.py::test_m1d1_exactly_three_sinks
    """

    undifferentiation_budget: float
    access_vector: dict[str, float] = field(default_factory=dict)
    leadership_office: LeadershipOffice | None = None
    sink_allocations: dict[Sink, float] = field(
        default_factory=lambda: {sink: 0.0 for sink in Sink}
    )
    overflow: float = 0.0
    differentiation_capacity: bool | None = None
    tolerance: dict[PersonId, float] = field(default_factory=dict)
    ambient_anxiety: float = 0.0
    # M1.D.1, M1.D.3: the structure sink allocation reads — the nuclear marital tie, the
    # nuclear parent–child ties, and the nuclear adults' family-of-origin ties. Declared
    # data, set from the family definition (Phase C step 10).
    marital_tie: TieId | None = None
    parent_child_ties: tuple[TieId, ...] = ()
    origin_ties: tuple[TieId, ...] = ()

    def __post_init__(self) -> None:
        if set(self.sink_allocations) != set(Sink):
            raise ValueError("sink_allocations must have exactly the three M1.D.1 sinks")
