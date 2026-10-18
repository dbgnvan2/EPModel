"""The persistence sink and the log header.

Purpose: the sink changes nothing about a run, logs are byte-identical with
         their header, and the header identifies the resolved configuration.
Spec:    docs/bowen_agent_model_spec_v2.md#M16.T.3, #M16.T.5, #M16.A.1, #M16.A.7, #M16.B.3
Tests:   this file
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType

from src.bowen.engine.activation import SynchronousActivation
from src.bowen.engine.log_records import LogHeader, serialize
from src.bowen.engine.state import canonical_state
from src.bowen.engine.visibility import HouseholdConductanceVisibility
from src.bowen.io.load import (
    load_constants, load_event_kinds, load_family, load_frozen_constants, load_script, load_spec_revision,
)
from src.bowen.io.sinks import JsonlFileSink
from src.bowen.run import run_phase_b
from src.bowen.scenario.header import build_header, config_hash, resolved_config


def test_m16t3_sink_does_not_change_results(tmp_path):
    """G10: same seed, sink attached and detached, identical final state.

    Passes vacuously in Phase B — no policy reads the log — and M16.T.3 requires
    it re-run at the end of Phases C and D.
    """
    detached = run_phase_b(7)
    path = tmp_path / "run.jsonl"
    with JsonlFileSink(path) as sink:
        attached = run_phase_b(7, extra=sink)
    assert canonical_state(attached.state) == canonical_state(detached.state)
    written = path.read_text(encoding="utf-8").splitlines()
    assert written == [serialize(r) for r in attached.records]  # the sink is faithful, too


def test_m16t5_same_seed_same_log_with_header(tmp_path):
    """G11: two persisted runs at one seed are byte-identical files, header first."""
    paths = [tmp_path / "a.jsonl", tmp_path / "b.jsonl"]
    for path in paths:
        with JsonlFileSink(path) as sink:
            run_phase_b(11, extra=sink)
    a, b = (p.read_bytes() for p in paths)
    assert a == b
    assert a.splitlines()[0].startswith(b'{"activation_component"')
    assert b'"record_type":"header"' in a.splitlines()[0]


def header(constants=None, frozen=None, script=None):
    kinds, family = load_event_kinds(), load_family()
    return build_header(
        seed=1, spec_revision=load_spec_revision(), constants=constants or load_constants(),
        frozen=frozen or load_frozen_constants(), kinds=kinds, family=family,
        script=script or load_script(kinds=kinds, family=family),
        activation=SynchronousActivation(), visibility=HouseholdConductanceVisibility(0.8),
    )


def test_m16a1_header_hashes_the_resolved_config():
    h = header()
    assert isinstance(h, LogHeader) and len(h.config_hash) == 64
    assert h.spec_revision == "2.0, revision 32"
    assert h == header()  # stable
    # A changed value changes the hash; the script's events are part of the config.
    assert header(script=load_script().without("TRIGGER")).config_hash != h.config_hash


def test_m16a1_header_hash_ignores_comments_but_not_values():
    constants = load_constants()
    kinds, family, script = load_event_kinds(), load_family(), load_script()
    base = config_hash(resolved_config(constants, kinds, family, script))
    bumped = dataclasses.replace(
        constants,
        values=MappingProxyType({**constants.values,
                                 "acute_decay_rate": dataclasses.replace(constants.values["acute_decay_rate"], value=0.25)}),
    )
    assert config_hash(resolved_config(bumped, kinds, family, script)) != base
    # Review 2026-10-11: which kinds are nodal changes the estimator's breadth, so it is part of the config.
    fewer = dataclasses.replace(kinds, nodal=kinds.nodal - {"MOVE"},
                                nodal_reach={k: r for k, r in kinds.nodal_reach.items() if k != "MOVE"})
    assert config_hash(resolved_config(constants, fewer, family, script)) != base
    for changed in (dataclasses.replace(kinds, soothing=kinds.soothing - {"DISTANCE"}),
                    dataclasses.replace(kinds, steering={k: d for k, d in kinds.steering.items() if k != "TRIANGLE"})):
        assert config_hash(resolved_config(constants, changed, family, script)) != base


def test_m16a7_header_flags_constants_changed_after_freeze():
    constants, frozen = load_constants(), load_frozen_constants()
    since_freeze = {k for k in constants.values if k not in frozen.values or constants[k] != frozen[k]}
    baseline = {k for k, flag in header().constant_changed_after_freeze if flag}
    # Re-frozen for Phase C at step 14. Changed since, logged in constants_changes.md: the pattern readouts'
    # window and threshold (owner decision D1, 2026-10-10; no criterion reads them). pattern_pole was retired.
    # Phase D B4 (2026-10-11) added the seven rejection constants (M1.B.3c, revisions 14 and 16).
    assert baseline == since_freeze == {"pattern_window", "pattern_min_acts", "pattern_projection_hold",
                                        "rejection_cutoff_gain", "rejection_refusal_gain", "rejection_relief",
                                        "refusal_cost_gain", "declared_rejection", "rejection_refusal_threshold",
                                        "rejection_fade", "estimator_window", "estimator_breadth",
                                        "estimator_sd_weight", "estimator_rate", "tie_wear_rate",
                                        "wear_offset_per_act", "perspective_channel_gain", "self_practice_gain",
                                        "stressor_persistence", "response_up_gain", "response_down_gain",
                                        "response_relaxation_rate", "stress_floor_below_basic", "practice_window",
                                        "perspective_steer_gain", "system_adjust_gain", "soothing_review_every",
                                        "soothing_review_penalty", "usual_level_rate",
                                        "constitution_strength", "channel_protection", "lock_in_gain", "lock_in_turn",
                                        "attribution_rate", "focus_rate", "belief_appraisal_gain", "marital_distance_fraction",
                                        "pseudo_self_relaxation_rate", "exchange_window",
                                        # spec revision 28, the life course (Phase D part II)
                                        "acting_age_years", "development_start_age_years", "development_end_age_years",
                                        "development_band", "development_rate", "development_band_projection",
                                        "development_band_support", "witnessed_gain", "launch_age_years",
                                        "working_age_years", "find_work_rate", "find_work_pressure_scale",
                                        "grief_scale", "grief_min_weeks", "grief_extra_weeks", "income_loss_intensity",
                                        "income_shortfall_share", "mortality_a", "mortality_b",
                                        "budget_per_undifferentiation", "coach_start_age_years",
                                        "coach_retirement_age_years", "new_tie_investment_share", "birth_focus_gain",
                                        "death_transfer_share",
                                        # spec revision 29, cutoff and focus (plan §4.8)
                                        "reactive_cutoff_gain", "witnessed_cutoff_gain", "witness_window",
                                        "perceived_weakness_gain", "focus_steer_gain",
                                        # spec revision 30, origin cutoff, danger and focus (plan §4.10)
                                        "origin_cutoff_anxiety_gain", "origin_fusion_gain", "danger_window",
                                        "danger_threshold", "danger_gain", "focus_reinforce_gain",
                                        "force_anxiety_gain", "focus_compound_gain",
                                        # spec revision 32, the nodal-event spike (plan §4.12)
                                        "nodal_spike_intensity"}
    changed = dataclasses.replace(
        constants,
        values=MappingProxyType({**constants.values,
                                 "route_damping": dataclasses.replace(constants.values["route_damping"], value=0.4)}),
    )
    flags = dict(header(constants=changed).constant_changed_after_freeze)
    assert {k for k, flag in flags.items() if flag} == baseline | {"route_damping"}


def test_m16b1_a_failed_run_leaves_no_complete_looking_log(tmp_path):
    """Review finding (2026-10-06): a run that raised left a well-formed, shorter log."""
    path = tmp_path / "run.jsonl"

    class Boom(RuntimeError):
        pass

    try:
        with JsonlFileSink(path) as sink:
            run_phase_b(7, extra=sink)
            raise Boom
    except Boom:
        pass
    assert not path.exists()
    assert (tmp_path / "run.jsonl.partial").exists()
