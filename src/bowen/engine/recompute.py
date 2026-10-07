"""Tick steps 5 and 6 — involvement, membership and triangle activity.

Purpose: recompute, before selection, the quantities the move gates and the
         propensity vector read (M3.D.3).
Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.12, #M1.C.3, #M1.C.4, #M8.5, #M3.D.3, #M5.A.1a
Tests:   tests/bowen/test_mechanisms.py

The spec requires these to exist and be recomputed; it gives no formulas. The
ones below are the project's, graded [I]:

* ``involvement_weight`` = Σ bond_energy / 100 over a person's ties.
  Membership is ``involvement_weight ≥ involvement_membership_threshold``
  (M1.A.12: derived as a threshold, never a stored set).
* A triangle is **active** when a ``TRIANGLE`` act among its three members falls
  within the last ``triangle_activity_window`` ticks **and** all three members
  hold live positions by the M8 predicate (M8.5). This is a **readout of recent
  acts**, not a threshold on tension: spec revision 11 amended `M1.C.3` so that
  quiet in a calm system is a result — with little tension a triangle gives little
  relief and is not reinforced — and forbade a calm-system threshold in this step.
  *Changed at Phase C step 1; Phase B used ``tension_activation_threshold``.*
* While active, its inside pair is the sender and target of the latest
  ``TRIANGLE`` act within the window, and the third member is outside.
  ``activation_memory`` counts the ticks it has been active (M1.C.4's record,
  which the learner reads from Phase C step 7).

"""

from __future__ import annotations

from src.bowen.engine.events import Mechanism
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.live_positions import Occupant, positions_live
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState

# M5.A.1 fixes the repertoire; the name is the spec's.
TRIANGLE_KIND = "TRIANGLE"


def involvement(state: RunState, person: PersonId) -> float:
    return sum(t.bond_energy for t in state.ties_of(person)) / 100.0


def is_member(state: RunState, person: PersonId, params: EngineParams) -> bool:
    """Purpose: family membership, derived from involvement each time it is asked.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.12
    Tests:   tests/bowen/test_mechanisms.py::test_m1a12_membership_is_derived_from_involvement
    """
    return state.people[person].involvement_weight >= params.involvement_membership_threshold


def recompute_involvement(state: RunState) -> None:
    """Purpose: tick step 5.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.A.12
    Tests:   tests/bowen/test_mechanisms.py::test_m1a12_membership_is_derived_from_involvement
    """
    for pid in sorted(state.people):
        state.people[pid].involvement_weight = involvement(state, pid)


def _latest_triangle_move(state: RunState, members: tuple[PersonId, PersonId, PersonId], window: int):
    latest = None
    for event in state.store.events():
        if event.kind != TRIANGLE_KIND or event.mechanism is not Mechanism.MOVE or event.sender is None:
            continue
        involved = {event.sender, *event.targets}
        recent = state.tick - window < event.timestamp <= state.tick
        if len(event.targets) == 1 and involved <= set(members) and recent:
            latest = event
    return latest


def recompute_triangles(state: RunState, params: EngineParams) -> list[EffectRecord]:
    """Purpose: tick step 6 — which triangles are active, and their positions.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.C.3, #M1.C.4, #M8.5
    Tests:   tests/bowen/test_mechanisms.py::test_m1c3_activity_is_a_readout_of_recent_triangle_acts
    """
    records = []
    for tri_id in sorted(state.triangles):
        triangle = state.triangles[tri_id]
        members = tri_id.members
        live = positions_live(Occupant(m, present=state.people[m].alive) for m in members)
        move = _latest_triangle_move(state, members, params.triangle_activity_window)
        active = move is not None and live == 3
        inside, outside = None, None
        if active:
            inside = TieId.of(move.sender, move.targets[0])
            outside = next(m for m in members if m not in inside.members())
        changes = []
        if active != triangle.active:
            changes.append((tri_id, "active", str(active).lower()))
        if inside != triangle.inside_pair:
            changes.append((tri_id, "inside_pair", str(inside) if inside else "none"))
        triangle.active = active
        triangle.inside_pair, triangle.outside = inside, outside
        if active:
            triangle.activation_memory += 1
        if changes:
            records.append(EffectRecord(state.tick, "triangle_recompute", None, triangles=tuple(changes)))
    return records
