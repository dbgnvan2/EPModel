"""Initial run state that needs the engine's parameters.

Purpose: set every derived starting value a run needs before its first tick, in one call.
Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.1c, #M1.A.9, #M9.8
Tests:   tests/bowen/test_contact.py; tests/bowen/test_outside_ness.py; tests/bowen/test_beliefs.py

``new_run_state`` copies the declaration and holds no parameters. Felt contact
(`M4.C.1c`), outside-ness (`M1.A.9`) and the belief prior (`M9.8`) start from values derived through the
parameters, so they are set here. Each reader raises if its state is missing,
so a run that skipped this fails at once rather than running on defaults.
"""

from __future__ import annotations

from src.bowen.engine.beliefs import initialise_beliefs
from src.bowen.engine.contact import initialise_contact
from src.bowen.engine.moves import initialise_functioning
from src.bowen.engine.outside_ness import initialise_outside_ness
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState


def initialise_run(state: RunState, params: EngineParams) -> RunState:
    initialise_contact(state.people, state.ties, params)
    initialise_outside_ness(state.people, params)
    initialise_beliefs(state.people, state.ties, params)
    initialise_functioning(state.people, state.ties)
    return state
