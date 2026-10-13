"""The family-evaluation readout (`M11.G`, Phase D B10).

Purpose: test that the readout emits exactly components 1-8, derives adaptiveness against stress, keeps belief and
         truth apart, carries the framing, and orders members as the run log does.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.G.1, #M11.G.2, #M11.G.3, #M11.G.4, #M11.D.20
Tests:   this file
"""

from __future__ import annotations

import pytest

from src.bowen.engine.log_records import EffectRecord
from src.bowen.ensemble.criteria import scenario
from src.bowen.ensemble.criteria_d import nodal
from src.bowen.io.load import load_constants
from src.bowen.readouts.evaluation import COMPONENTS, FORBIDDEN, evaluate, ranking_by_symptom_load
from src.bowen.render.trace import framing
from src.bowen.scenario.params import engine_params

PARAMS = engine_params(load_constants())
WEEKS = 104


def run(spells=True):
    from src.bowen.engine.identifiers import PersonId

    state, records = scenario("four", 3, WEEKS, spells=nodal(WEEKS, (PersonId("f"), PersonId("m"))) if spells else ())
    from src.bowen.ensemble.criteria import FAMILIES
    from src.bowen.io.load import load_family

    generations = load_family(FAMILIES["four"]).generations
    return state, records, evaluate(state, records, PARAMS, framing(records), window=52, generations=generations)


STATE, RECORDS, EVALUATION = run()


def test_m11g1_the_evaluation_has_exactly_components_one_to_eight():
    assert set(COMPONENTS) <= set(EVALUATION) and len(COMPONENTS) == 8


def test_m11g3_components_nine_and_ten_are_never_emitted():
    text = repr(EVALUATION).lower()
    for word in (*FORBIDDEN, "therapeutic", "prognosis", "treatment plan"):
        assert word not in text, word


def test_m11g2_adaptiveness_is_derived_against_stress_and_undefined_without_it():
    adapt = EVALUATION["adaptiveness"]
    assert adapt["exogenous_intensity"] > 0
    assert adapt["reactivity_per_stress"] == pytest.approx(adapt["automatic_acts"] / adapt["exogenous_intensity"])
    _, _, calm = run(spells=False)
    assert calm["adaptiveness"]["exogenous_intensity"] == 0 and calm["adaptiveness"]["reactivity_per_stress"] is None


def test_m11g4_belief_and_truth_are_reported_apart_and_the_framing_travels():
    pattern = EVALUATION["binding_pattern"]
    assert set(pattern) == {"truth", "belief"}
    assert abs(sum(pattern["truth"].values()) - 1.0) < 1e-9 or sum(pattern["truth"].values()) == 0
    assert "M11.F.9" in EVALUATION["framing"] and "M11.F.10" in EVALUATION["framing"]
    assert "not built" in EVALUATION["multigenerational"]["projection_line"]
    assert set(EVALUATION["multigenerational"]["basic_level_by_generation"]) == {1, 2, 3}


def test_m11d20_readout_ordering_matches_run_log_ordering():
    """M11.D.20: members ranked by symptom load agree between the readout and the run log's last accumulation record."""
    last = next(r for r in reversed(RECORDS) if isinstance(r, EffectRecord) and r.mechanism == "symptom_accumulation")
    from_log = sorted(((pid.value, load) for pid, _, load in last.people), key=lambda x: (-x[1], x[0]))
    members = set(EVALUATION["symptom"])
    assert [p for p, _ in from_log if p in members] == ranking_by_symptom_load(EVALUATION)
