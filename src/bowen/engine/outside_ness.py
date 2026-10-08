"""Outside-ness — the two-axis hidden actor state of `M1.A.9a`, and how it moves.

Purpose: hold each person's outward and inward impingement, derive their efficacy
         from the pair, and drift both from what the person does and hears.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.9, #M1.A.9a, #M4.C.5, #M5.F.1, #M5.F.2
Tests:   tests/bowen/test_outside_ness.py

**What the two fields hold.** ``outside_ness_outward`` is **outward impingement** —
acting on the other: forceful, dogmatic, the "selfish" counterfeit.
``outside_ness_inward`` is **inward impingement** — being acted on: pleasing,
accommodating, hearing the other as critical, the "selfless" counterfeit. Both are
in [0, 1], and the differentiated position is **both low** (`M1.A.9a`; `FE03.1`).
Efficacy — the *outside-ness* `M5.C`'s gate and explainer §3.6 speak of — is
derived, never stored: ``1 − max(outward, inward)``. The maximum, because the
definition is a conjunction of two negatives: either failure fails it.

**At t0** both axes are ``initial_impingement_scale × (1 − basic_level / 100)``
(derived from level, `M10.A`), [I]: the spec gives no starting value.

**Each tick** (step 9) each axis drifts at ``outside_ness_rate`` toward a target read
from the tick's behaviour, [I]:

* outward ← the impingement the person's own moves delivered this tick, from each
  kind's impingement component × intensity / intensity_scale, clamped to [0, 1].
  The assault term below is left out, so the axis does not feed on itself.
* inward ← `M4.C.5`'s perception-side reading — the rise in the "too much" side the
  person took from what reached them this tick (**reading the other's event as
  critical is itself the evidence**) — plus their own accommodating acts
  (``accommodates`` in ``event_kinds.md``), clamped to [0, 1].

**Act identity** (`M5.F.1`, `M5.F.2`), applied in ``appraise.py`` through the event,
never by a receiver reading the field (`M4.B.2a`): a sender's outward impingement
adds ``assault_gain × outward`` to the impingement the move delivers, and its inward
impingement hollows the contact it delivers by ``1 − hollow_gain × inward``. The
same move from a forceful sender can land as an assault, from a compliant one as
empty words — "either hollow meaningless words or a hostile assault" (Ch21 · L21.2).

**Of `M1.A.9`'s three inputs at three time scales**, this builds the behavioural
drift. Private rehearsal is `PREPARE` (step 8); practice on a peripheral system
arrives with the external agent and peripheral ties (step 9).
"""

from __future__ import annotations

from src.bowen.engine.contact import clamp_unit
from src.bowen.engine.events import Mechanism
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.objects import SCALE_MAX, Person
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState


class OutsideNessNotInitialised(RuntimeError):
    """A person has no outside-ness state: ``initialise_outside_ness`` was not called."""


def axes(person: Person) -> tuple[float, float]:
    if person.outside_ness_outward is None or person.outside_ness_inward is None:
        raise OutsideNessNotInitialised(f"{person.id} has no outside-ness state")
    return person.outside_ness_outward, person.outside_ness_inward


def efficacy(person: Person) -> float:
    """Purpose: outside-ness as efficacy — both impingement axes low — derived, never stored.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.9a, #M5.C.1
    Tests:   tests/bowen/test_outside_ness.py::test_m1a9a_efficacy_needs_both_axes_low
    """
    outward, inward = axes(person)
    return clamp_unit(1.0 - max(outward, inward))


def initialise_outside_ness(people: dict[PersonId, Person], params: EngineParams) -> None:
    """Purpose: start both axes from basic level.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.9, #M10.A.1
    Tests:   tests/bowen/test_outside_ness.py::test_m1a9_axes_start_from_basic_level
    """
    for pid in sorted(people):
        person = people[pid]
        start = params.initial_impingement_scale * (1.0 - person.basic_level / SCALE_MAX)
        person.outside_ness_outward = person.outside_ness_inward = clamp_unit(start)


def outward_target(state: RunState, person: PersonId, params: EngineParams) -> float:
    delivered = 0.0
    for event in state.store.events():
        if event.timestamp == state.tick and event.sender == person and event.mechanism is Mechanism.MOVE:
            _, impingement = state.kinds.components_of(event.kind)
            delivered += max(0.0, impingement) * event.intensity / params.intensity_scale
    return clamp_unit(delivered)


def inward_target(state: RunState, person: PersonId, reading: float, params: EngineParams) -> float:
    gave_way = sum(
        event.intensity / params.intensity_scale
        for event in state.store.events()
        if event.timestamp == state.tick and event.sender == person
        and event.mechanism is Mechanism.MOVE and event.kind in state.kinds.accommodating
    )
    return clamp_unit(reading + gave_way)


def update_outside_ness(
    state: RunState, readings: dict[PersonId, float], params: EngineParams
) -> list[EffectRecord]:
    """Purpose: drift both axes toward this tick's behaviour, step 9.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.9, #M4.C.5
    Tests:   tests/bowen/test_outside_ness.py::test_m4c5_hearing_criticism_raises_inward_impingement
    """
    changes = []
    for pid in sorted(state.people):
        person = state.people[pid]
        if not person.alive:
            continue
        outward, inward = axes(person)
        d_out = params.outside_ness_rate * (outward_target(state, pid, params) - outward)
        d_in = params.outside_ness_rate * (inward_target(state, pid, readings.get(pid, 0.0), params) - inward)
        person.outside_ness_outward = clamp_unit(outward + d_out)
        person.outside_ness_inward = clamp_unit(inward + d_in)
        if d_out:
            changes.append((pid, "outside_ness_outward", d_out))
        if d_in:
            changes.append((pid, "outside_ness_inward", d_in))
    return [EffectRecord(state.tick, "outside_ness", None, people=tuple(changes))] if changes else []
