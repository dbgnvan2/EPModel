"""Build the engine's parameter object from the parsed constants register.

Purpose: the one bridge from configuration to engine parameters.
Spec:    docs/bowen_agent_model_spec_v2.md#M0.3, #M10.1
Tests:   tests/bowen/test_mechanisms.py::test_m03_engine_params_come_from_the_register
"""

from __future__ import annotations

import dataclasses

from src.bowen.engine.params import EngineParams
from src.bowen.scenario.constants import Constants


def engine_params(constants: Constants) -> EngineParams:
    return EngineParams(**{f.name: constants[f.name] for f in dataclasses.fields(EngineParams)})
