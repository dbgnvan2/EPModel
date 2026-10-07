"""Assemble a runnable model from parsed configuration.

Purpose: build the run state, source, parameters and components for one run,
         from objects already parsed — no file access here.
Spec:    docs/bowen_agent_model_spec_v2.md#M3.D.4, #M3.E.1, #M16.B.1
Tests:   tests/bowen/test_phase_b_gate.py
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bowen.engine.activation import SynchronousActivation, activation_for
from src.bowen.engine.contact import initialise_contact
from src.bowen.engine.events import EventKinds
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState, new_run_state
from src.bowen.engine.visibility import HouseholdConductanceVisibility
from src.bowen.scenario.constants import Constants
from src.bowen.scenario.family import FamilyInstance
from src.bowen.scenario.params import engine_params
from src.bowen.scenario.scripted_source import ScriptedSource


@dataclass
class Assembled:
    state: RunState
    source: ScriptedSource
    params: EngineParams
    visibility: HouseholdConductanceVisibility
    activation: SynchronousActivation


def assemble(
    constants: Constants, kinds: EventKinds, family: FamilyInstance, source: ScriptedSource, seed: int = 0
) -> Assembled:
    """A fresh state every call, so two runs share nothing (M11.D.6)."""
    params = engine_params(constants)
    state = new_run_state(dict(family.people), dict(family.ties), family.family, kinds, seed=seed)
    initialise_contact(state.people, state.ties, params)
    return Assembled(
        state=state,
        source=source,
        params=params,
        visibility=HouseholdConductanceVisibility(params.per_hop_fidelity),
        activation=activation_for(constants.activation_regime),
    )
