"""The run-log header: enough to replay and attribute a run.

Purpose: build M16.A.1's self-describing header from the parsed configuration.
Spec:    docs/bowen_agent_model_spec_v2.md#M16.A.1, #M16.A.1a, #M16.A.7, #M10.B.4
Tests:   tests/bowen/test_log.py::test_m16a1_header_hashes_the_resolved_config

``config_hash`` is SHA-256 over the *resolved* configuration — parsed values,
not file text — so editing a comment in a config file leaves it unchanged and
changing any value changes it. The header carries no wall-clock time: two runs
at one seed must produce byte-identical logs, header included (M16.T.5).
"""

from __future__ import annotations

import hashlib
import json

from src.bowen.engine.activation import SynchronousActivation
from src.bowen.engine.events import EventKinds
from src.bowen.engine.log_records import LogHeader, plain
from src.bowen.engine.visibility import HouseholdConductanceVisibility
from src.bowen.scenario.constants import Constants
from src.bowen.scenario.family import FamilyInstance
from src.bowen.scenario.scripted_source import ScriptedSource


def resolved_config(constants: Constants, kinds: EventKinds, family: FamilyInstance, script: ScriptedSource) -> dict:
    return {
        "constants": {k: [c.value, c.grade] for k, c in sorted(constants.values.items())},
        "activation_regime": constants.activation_regime,
        "event_kinds": {k: m.value for k, m in sorted(kinds.mechanisms.items())},
        "signs": sorted([k, p.value, s] for (k, p), s in kinds.signs.items()),
        "family": {
            "instance_id": family.instance_id,
            "nuclear_household": family.nuclear_household,
            "people": [plain(family.people[p]) for p in sorted(family.people)],
            "ties": [plain(family.ties[t]) for t in sorted(family.ties)],
            "relations": sorted([str(t), r.value] for t, r in family.relations.items()),
            "budget": family.family.undifferentiation_budget,
        },
        "script": {
            "script_id": script.script_id,
            "ticks": script.ticks,
            "events": [plain(e) for e in script.events],
            "moves": [[t, plain(s)] for t, s in script.moves],
        },
    }


def config_hash(config: dict) -> str:
    text = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_header(
    *,
    seed: int,
    spec_revision: str,
    constants: Constants,
    frozen: Constants,
    kinds: EventKinds,
    family: FamilyInstance,
    script: ScriptedSource,
    activation: SynchronousActivation,
    visibility: HouseholdConductanceVisibility,
) -> LogHeader:
    """Purpose: the header record that opens every log.
    Spec:    docs/bowen_agent_model_spec_v2.md#M16.A.1, #M16.A.1a, #M16.A.7
    Tests:   tests/bowen/test_log.py::test_m16a7_header_flags_constants_changed_after_freeze
    """
    changed = tuple(
        (key, key not in frozen.values or constants[key] != frozen[key]) for key in sorted(constants.values)
    )
    return LogHeader(
        seed=seed,
        config_hash=config_hash(resolved_config(constants, kinds, family, script)),
        spec_revision=spec_revision,
        instance_id=family.instance_id,
        activation_component=activation.component_id,
        activation_version=activation.version,
        visibility_component=visibility.component_id,
        visibility_version=visibility.version,
        activation_regime=activation.regime,
        constants_frozen_at=constants.frozen_at,
        constant_changed_after_freeze=changed,
    )
