"""The family-evaluation readout — `M11.G`'s components 1–8, and nothing of 9 or 10.

Purpose: read a finished run's state and records into the eight components `M11.G.1` names, with adaptiveness
         derived against stress (`M11.G.2`), belief and ground truth reported separately (`M9.5`, `M11.G.4`), and
         `M11.F`'s framing carried with it.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.G.1, #M11.G.2, #M11.G.3, #M11.G.4, #M9.5, #M11.D.20, #M11.D.22
Tests:   tests/bowen/test_evaluation.py

Outside the engine (`M11.D.22`): it reads after the fact and changes nothing. The forms are `[I]`.

1. **Symptom** — each person's active channels, load per channel and severity (load ÷ threshold).
2. **Binding pattern** — the sink allocation's shares (truth) and the sink the spouses' attributions name (belief).
3. **Intensity** — the budget's occupancy (the allocations and overflow over the budget) and the members' mean
   excess anxiety at the end.
4. **The multigenerational picture** — mean basic level by generation. The projection line is not read; it is
   reported as not built.
5. **Cutoff with the families of origin** — each origin tie's bond energy and whether it is cut off.
6. **Stress** — the exogenous and endogenous events in the window, counted and summed by intensity, separately.
7. **Adaptiveness** — derived: the members' automatic acts in the window per unit of exogenous stress intensity.
   Reactivity alone, and stress alone, are reported beside it; with no stress it is undefined (``None``), never 0.
8. **The network** — the access vector, and the people outside the nuclear household on an interactive tie to a
   member of it.

Components 9 and 10 (therapeutic focus, prognosis) are never emitted (`M11.G.3`).
"""

from __future__ import annotations

import math

from src.bowen.engine.contact import excess
from src.bowen.engine.events import Mechanism
from src.bowen.engine.log_records import EmittedRecord
from src.bowen.engine.objects import Role
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState
from src.bowen.engine.symptoms import threshold
from src.bowen.readouts.attribution import sink_from_beliefs
from src.bowen.engine.counting import count

COMPONENTS = ("symptom", "binding_pattern", "intensity", "multigenerational", "cutoff", "stress", "adaptiveness",
              "network")
FORBIDDEN = ("therapeutic_focus", "prognosis")  # M11.G.3


def evaluate(state: RunState, records, params: EngineParams, framing: str, window: int, generations) -> dict:
    """Purpose: the eight-component family evaluation of a finished run, over the last ``window`` weeks.
    ``generations`` is the family definition's generation per person (`M2.A`; not part of the run's state).
    Spec:    docs/bowen_agent_model_spec_v2.md#M11.G.1, #M11.G.2, #M11.G.3, #M11.G.4
    Tests:   tests/bowen/test_evaluation.py::test_m11g1_the_evaluation_has_exactly_components_one_to_eight
    """
    # Revision 28: an inactive family-of-origin holder is not counted as a member (M7.C.1h).
    members = sorted(p for p, person in state.people.items()
                     if person.role is Role.MEMBER and person.alive and not person.inactive)
    since = state.tick - window
    family = state.family
    spouses = tuple(sorted(family.marital_tie.members())) if family.marital_tie else ()
    children = frozenset(m for t in family.parent_child_ties for m in t.members()) - set(spouses)
    moves = [r.event for r in records if isinstance(r, EmittedRecord) and r.event.timestamp >= since]

    symptom = {}
    for pid in members:
        person = state.people[pid]
        limit = threshold(person, params)
        symptom[pid.value] = {c.value: {"active": person.symptom_active[c], "load": person.symptom_load[c],
                                        "severity": person.symptom_load[c] / limit} for c in person.symptom_load}

    allocated = {s.value: v for s, v in family.sink_allocations.items()}
    total = math.fsum(allocated.values())
    believed = sink_from_beliefs({p: dict(state.people[p].attribution) for p in spouses}, spouses, children) \
        if len(spouses) == 2 else None

    by_generation: dict[int, list[float]] = {}
    for pid in members:
        if generations.get(pid) is not None:
            by_generation.setdefault(generations[pid], []).append(state.people[pid].basic_level)

    exogenous = [e for e in moves if e.exogenous and e.mechanism is Mechanism.EXOGENOUS_STRESSOR]
    endogenous = [e for e in moves if not e.exogenous and e.mechanism is Mechanism.ENDOGENOUS_SYMPTOM]
    stress = math.fsum(e.intensity for e in exogenous)
    reactive = count(1 for e in moves if e.mechanism is Mechanism.MOVE and e.sender in members
                   and state.kinds.channels.get(e.kind) == "automatic")

    household = {p for p in members if state.people[p].household_id == _nuclear(state, spouses)}
    outside = sorted({o.value for p in household for t in state.ties_of(p) if t.interactive
                      for o in t.id.members() if o not in household and state.people[o].role is Role.MEMBER})

    return {
        "framing": framing,
        "window_weeks": window,
        "symptom": symptom,
        "binding_pattern": {
            "truth": {k: v / total if total else 0.0 for k, v in allocated.items()},
            "belief": believed.value if believed is not None else None,
        },
        "intensity": {
            "budget_occupancy": (total + family.overflow) / family.undifferentiation_budget
            if family.undifferentiation_budget else None,
            "mean_excess": math.fsum(excess(state.people[p]) for p in members) / len(members) if members else 0.0,
        },
        "multigenerational": {
            "basic_level_by_generation": {g: math.fsum(v) / len(v) for g, v in sorted(by_generation.items())},
            "projection_line": "not built: children are born in the run from Phase D part II, but the line is not yet read",
        },
        "cutoff": {str(t): {"bond_energy": state.ties[t].bond_energy, "cut_off": state.ties[t].rejected()}
                   for t in family.origin_ties if t in state.ties},
        "stress": {"exogenous_events": len(exogenous), "exogenous_intensity": stress,
                   "endogenous_events": len(endogenous)},
        "adaptiveness": {"automatic_acts": reactive, "exogenous_intensity": stress,
                         "reactivity_per_stress": reactive / stress if stress > 0 else None},
        "network": {"access_vector": dict(family.access_vector), "outside_ties": outside},
    }


def _nuclear(state: RunState, spouses) -> str | None:
    return state.people[spouses[0]].household_id if spouses else None


def ranking_by_symptom_load(evaluation: dict) -> list[str]:
    """Purpose: members ranked by total symptom load, as read from the evaluation (for `M11.D.20`).
    Spec:    docs/bowen_agent_model_spec_v2.md#M11.D.20
    Tests:   tests/bowen/test_evaluation.py::test_m11d20_readout_ordering_matches_run_log_ordering
    """
    totals = {p: math.fsum(c["load"] for c in channels.values()) for p, channels in evaluation["symptom"].items()}
    return sorted(totals, key=lambda p: (-totals[p], p))
