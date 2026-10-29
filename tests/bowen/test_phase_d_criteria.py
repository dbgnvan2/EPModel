"""The Phase D estimator criteria (Phase D plan step 6; spec revision 17).

Purpose: one gate test per criterion, each running its paired arms through the ensemble runner and requiring PASS
         (``ensemble``-marked, so excluded from the default run), and fast checks that each arm runs and differs only
         in its declared channel.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.17, #M11.C.18, #M11.C.20, #M11.C.33, #M11.C.11, #M11.C.12,
         #M11.C.22, #M11.C.30, #M11.C.31, #M11.C.2, #M11.C.14, #M11.C.21, #M11.C.23, #M11.C.24, #M11.C.26, #M11.C.34,
         #M11.C.36, #M11.C.39, #M11.C.40, #M11.C.43, #M11.C.46, #M11.C.7, #M10.B.5, #M11.2, #M13.4
Tests:   this file
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.bowen.engine.log_records import EmittedRecord
from src.bowen.ensemble.criteria_d import CRITERIA_D, MOVER, c33_inputs, nodal
from src.bowen.ensemble.runner import PASS, run_criterion
from src.bowen.io.load import load_constants, load_event_kinds

C = load_constants()
RULES = {k: C[k] for k in ("ensemble_block", "ensemble_cap", "ensemble_precision", "ensemble_margin",
                           "ensemble_alpha", "fallback_flag_rate", "equivalence_margin")}
WORKERS = os.cpu_count() or 1


def gate(cid: str) -> None:
    verdict = run_criterion(CRITERIA_D[cid], RULES, workers=WORKERS)
    assert verdict.outcome == PASS, f"{cid}: {verdict.outcome} {verdict.readouts}"


@pytest.mark.ensemble
def test_m11c18_estimator_separates_cutoff_from_iposition():
    gate("M11.C.18")


@pytest.mark.ensemble
def test_m11c20_estimator_separates_borrowed_from_basic():
    gate("M11.C.20")


@pytest.mark.ensemble
def test_m11c33_dependence_gate_blocks_basic_level_rise():
    gate("M11.C.33")


@pytest.mark.ensemble
def test_m11c17_coach_helps_and_is_not_required():
    gate("M11.C.17")


def test_m11c18_the_nodal_load_fills_every_window_with_three_kinds():
    """Every 156-week estimator window of the declared load holds the three kinds (estimator_breadth)."""
    weeks, window = CRITERIA_D["M11.C.18"].settings["weeks"], C["estimator_window"] * C["slow_tick_fast_ticks"]
    events = nodal(weeks, (MOVER,))
    assert {e.kind for e in events} <= load_event_kinds().nodal
    for end in range(window - 1, weeks, C["slow_tick_fast_ticks"]):
        kinds = {e.kind for e in events if end - window < e.timestamp <= end}
        assert len(kinds) >= C["estimator_breadth"], end


def test_m11c33_arms_differ_only_at_the_slow_tick():
    """Review 2026-10-11: up to the first slow tick that can raise basic_level (end of week 155), the mover makes
    the same choices in both arms, week by week — the dependence is visible to nothing but the slow tick."""
    from src.bowen.engine.log_records import SelectionRecord
    from src.bowen.ensemble.criteria import scenario

    weeks = C["estimator_window"] * C["slow_tick_fast_ticks"]
    settings = CRITERIA_D["M11.C.33"].settings
    for seed in (0, 3, 5):
        picks = {}
        for arm in ("baseline", "treatment"):
            _, records = scenario("triad", seed, weeks, **c33_inputs(arm, weeks, settings["act_every"]))
            # what the mover could choose, and what it chose, each week
            picks[arm] = [(r.tick, tuple(label for label, _ in r.propensities), r.event_id) for r in records
                          if isinstance(r, SelectionRecord) and r.actor == MOVER]
        assert picks["baseline"] == picks["treatment"], seed
        assert any(e is not None and "I-POSITION" in str(labels) for _, labels, e in picks["baseline"])


def test_m11c20_reports_how_much_scripted_support_a_held_position_blocked():
    """Revision 21 (owner decision 2026-10-12): the borrowed arm keeps the rule, and reports the share of its
    scripted support that moved nothing because the mover held a position."""
    from src.bowen.engine.events import Channel, Event, EventId, Mechanism, SourcePosition
    from src.bowen.engine.log_records import EffectRecord, EmittedRecord
    from src.bowen.ensemble.criteria_d import MOVER, OTHER, support_blocked

    def support(i):
        return Event(id=EventId(i, "m", 0), kind="UNDERFUNCTION", mechanism=Mechanism.MOVE, sender=OTHER,
                     targets=(MOVER,), intensity=100.0, timestamp=i, duration=1, exogenous=False,
                     source_position=SourcePosition.NONE, channel=Channel.SCRIPTED)

    events = [support(i) for i in range(4)]
    records = [EmittedRecord(e) for e in events]
    records.append(EffectRecord(1, "position_held", events[1].id, people=((MOVER, "position_held", 1.0),)))
    assert support_blocked(records) == 0.25
    assert support_blocked([]) == 0.0
    assert "support_blocked" in {r.name for r in CRITERIA_D["M11.C.20"].readouts if r.report_only}


# --- Phase D step 7, the symptom channels (B7, spec revision 24) ---


@pytest.mark.ensemble
def test_m11c11_removal_three_phase_shape():
    gate("M11.C.11")


@pytest.mark.ensemble
def test_m11c12_symptom_relief_raises_conflict():
    gate("M11.C.12")


@pytest.mark.ensemble
def test_m11c22_symptom_lock_in_is_non_monotone():
    gate("M11.C.22")


@pytest.mark.ensemble
def test_m11c30_channels_are_mutually_protective():
    gate("M11.C.30")


@pytest.mark.ensemble
def test_m11c31_binder_removal_releases_bound_anxiety():
    gate("M11.C.31")


def test_m11c22_the_held_severity_is_what_the_arm_declares():
    """Each week the child's symptom is held at the declared severity, active only when it is above zero."""
    from src.bowen.ensemble.criteria import scenario
    from src.bowen.ensemble.criteria_d import CHILD, ENGINE, held_symptom
    from src.bowen.engine.symptoms import threshold

    settings = CRITERIA_D["M11.C.22"].settings
    assert 0 < settings["mild"] < C["lock_in_turn"] < settings["severe"]  # one limb either side of the turn
    for severity in (0.0, settings["mild"], settings["severe"]):
        seen = []

        def watch(state):
            person = state.people[CHILD]
            seen.append((person.symptom_active[person.channel_prior],
                         person.symptom_load[person.channel_prior] / threshold(person, ENGINE)))

        scenario("four", 1, 6, before=held_symptom(CHILD, severity), watch=watch)
        # held before each tick; the week's leak and feed then move it a little
        assert all(active == (severity > 0) for active, _ in seen)
        assert all(abs(x - severity) <= max(0.2 * severity, 0.5) for _, x in seen)


def test_m11c31_the_budget_rises_by_exactly_what_the_binder_held():
    """M11.C.31's equality limb, which the runner (directions only) cannot assert: the budget rises at the release by
    exactly what the tie's distance held then, and by nothing else in that tick."""
    from src.bowen.engine.identifiers import TieId
    from src.bowen.ensemble.criteria_d import FATHER, MOTHER, arm_c31

    settings = CRITERIA_D["M11.C.31"].settings
    t0, tie = settings["t0"], TieId.of(FATHER, MOTHER)
    held, budget = {}, {}

    def watch(state):
        held[state.tick] = state.ties[tie].distance_bound_anxiety
        budget[state.tick] = state.family.undifferentiation_budget

    import src.bowen.ensemble.criteria_d as module

    original = module.scenario
    module.scenario = lambda *a, **k: original(*a, **{**k, "watch": watch})
    try:
        arm_c31("treatment", 2, settings)
    finally:
        module.scenario = original
    # watch runs after each tick, when state.tick has already moved on: key t + 1 is the end of week t
    assert held[t0 + 1] == 0.0 and held[t0] > 0
    assert budget[t0 + 1] - budget[t0] == pytest.approx(held[t0])


def test_m11c30_the_arms_differ_only_in_the_load_carried_elsewhere():
    """Until an onset, the two arms of M11.C.30 make the same choices: the held mental load is read by nothing but the
    integrator."""
    from src.bowen.engine.log_records import SelectionRecord
    from src.bowen.ensemble import criteria_d as module

    picks = {}
    original = module.scenario
    for arm in ("baseline", "treatment"):
        captured = {}

        def capture(*a, **k):
            state, records = original(*a, **k)
            captured["records"] = records
            return state, records

        module.scenario = capture
        try:
            module.arm_c30(arm, 4, {**CRITERIA_D["M11.C.30"].settings, "weeks": 12})
        finally:
            module.scenario = original
        picks[arm] = [(r.tick, r.actor, r.event_id) for r in captured["records"] if isinstance(r, SelectionRecord)]
    assert picks["baseline"] == picks["treatment"]


# --- Phase D steps 8-10: beliefs, projection, and criteria on existing mechanisms (spec revisions 25-27) ---


@pytest.mark.ensemble
def test_m11c26_sink_recoverable_from_belief_configuration():
    gate("M11.C.26")


@pytest.mark.ensemble
def test_m11c36_policy_reads_belief_not_true_state():
    gate("M11.C.36")


@pytest.mark.ensemble
def test_m11c14_no_technique_lowers_load_under_marital_distance():
    """Revision 31 (R31.B4): each technique, against a no-technique arm, does not lower the child's load by more than
    the equivalence bound (one-sided)."""
    gate("M11.C.14[firmness]")
    gate("M11.C.14[talking it out]")


def test_m11c14_variants_differ_from_their_baseline_only_by_the_technique():
    """R31.B4, with the owner's option (a) of 2026-10-16: the baseline has both parents hold back (a targetless
    ``WITHHOLD``) on the technique's weeks, every one made; each treatment scripts its technique from both parents to C1;
    the report-only comparison is talking it out less firmness, so its baseline reads firmness's load."""
    from src.bowen.ensemble.criteria_d import C14_TECHNIQUES, _c14_run

    for technique, kind in C14_TECHNIQUES.items():
        criterion = CRITERIA_D[f"M11.C.14[{technique}]"]
        assert criterion.requires_moves == (kind,)
        assert criterion.readouts[0].bound_side == +1 and criterion.readouts[0].direction == 0
        base = criterion.arm("baseline", 0, criterion.settings)
        treat = criterion.arm("treatment", 0, criterion.settings)
        assert base.moves["(scripted acts)"] == treat.moves["(scripted acts)"] > 0  # the same weeks displaced
        assert base.moves["(scripted act: made)"] == base.moves["(scripted acts)"]
        if technique == "talking it out":
            s = criterion.settings
            firmness, _ = _c14_run(0, s["weeks"], s["act_every"], s["start_severity"], C14_TECHNIQUES["firmness"])
            assert base.readouts["talking_less_firmness"] == firmness
            assert treat.readouts["talking_less_firmness"] == treat.readouts["child_symptom_load"]
    assert "M11.C.14" not in CRITERIA_D  # the revision 15 two-sided comparison is withdrawn


@pytest.mark.ensemble
def test_m11c2_symptom_concentrates_on_projection_target():
    gate("M11.C.2")


@pytest.mark.ensemble
def test_m11c39_pattern_outlasts_spell_only_with_learning():
    gate("M11.C.39")


@pytest.mark.ensemble
def test_m11c40_leaked_reinforcement_raises_self_directed_entropy():
    """Clause 1 gates: the automatic channel's entropy falls more with learning. Clause 2 (``automatic_less_self``)
    is reported only (the owner, 2026-10-16): a mutant letting reinforcement reach the self-directed channel moved it
    +0.015 ± 0.049, so the readout cannot see `M4.D.6d` removed. Clause 2 is proved by
    tests/bowen/test_learner.py::test_m4d6d_self_directed_channel_is_never_reinforced."""
    gate("M11.C.40")


@pytest.mark.ensemble
def test_m11c43_learned_buffer_removal_costs_more():
    gate("M11.C.43")


@pytest.mark.ensemble
def test_m11c34_system_reaction_to_symptom_discriminates_level():
    gate("M11.C.34")


@pytest.mark.ensemble
def test_m11c21_reciprocity_inverts_on_projector_collapse():
    gate("M11.C.21")


@pytest.mark.ensemble
def test_m11c23_expressed_emotion_dominates_suppression():
    gate("M11.C.23")


@pytest.mark.ensemble
def test_m11c24_whisper_of_nature_reverses_on_dissolution():
    gate("M11.C.24")


def test_m11c26_the_control_rotates_the_labels_of_the_same_runs():
    """The baseline reads the same three runs against the next driving's sink: a perfect reading scores 1 in the
    treatment and 0 in the control; a reading that names one sink for every driving scores 1/3 in both; an unread
    belief scores nothing (review 2026-10-12: the earlier check could not fail)."""
    from src.bowen.engine.objects import Sink
    from src.bowen.ensemble.criteria_d import score_c26

    driven = [Sink.MARITAL_CONFLICT, Sink.SPOUSE_DYSFUNCTION, Sink.CHILD_PROJECTION]
    assert score_c26(driven, driven, driven, 0) == (1.0, 1.0)
    assert score_c26(driven, driven, driven, 1) == (0.0, 0.0)
    same = [Sink.MARITAL_CONFLICT] * 3
    assert score_c26(same, same, driven, 0)[0] == score_c26(same, same, driven, 1)[0] == pytest.approx(1 / 3)
    assert score_c26([None] * 3, [None] * 3, driven, 0) == (0.0, 0.0)


def test_m10b5_learning_is_switched_off_only_by_a_declared_override():
    """M10.B.5: the learning-off arm's override is the criterion's, not a config default, and it stops learning."""
    from src.bowen.ensemble.criteria import scenario
    from src.bowen.ensemble.criteria_d import LEARNING_OFF

    assert C["learning_rate"] > 0
    state, _ = scenario("triad", 1, 30, overrides=LEARNING_OFF)
    assert all(v == 0 for p in state.people.values() for v in p.learned_values.values())
    state, _ = scenario("triad", 1, 30)
    assert any(v != 0 for p in state.people.values() for v in p.learned_values.values())


def test_m11c40_clause_2_reported_and_proved_by_the_unit_test():
    """The owner's option (a), 2026-10-16: clause 1 alone gates; clause 2 is a report-only readout."""
    readouts = {r.name: r for r in CRITERIA_D["M11.C.40"].readouts}
    assert not readouts["automatic_entropy_fall"].report_only
    assert readouts["automatic_less_self"].report_only
    learner_tests = (Path(__file__).parent / "test_learner.py").read_text(encoding="utf-8")
    assert "def test_m4d6d_self_directed_channel_is_never_reinforced(" in learner_tests


@pytest.mark.ensemble
def test_m11c46_triangle_relief_scales_with_third_help():
    gate("M11.C.46")


def test_m11c46_arms_differ_only_in_the_thirds_bonds_and_the_triangle_lands():
    """`M11.C.46`'s arms (version 1, 2026-10-17): in the run itself, C's bonds with F and M are swapped between the
    arms and every other tie's bond is the same; and the scripted TRIANGLE lands as a transfer in both arms, so the
    readout is never a default 0."""
    from unittest import mock

    import src.bowen.ensemble.criteria_d as module
    from src.bowen.engine.identifiers import PersonId, TieId

    criterion, real, states, relief = CRITERIA_D["M11.C.46"], module.scenario, {}, {}
    for arm in ("baseline", "treatment"):
        def spy(*args, **kwargs):
            states[arm], records = real(*args, **kwargs)
            return states[arm], records
        with mock.patch.object(module, "scenario", side_effect=spy):
            relief[arm] = module.arm_c46(arm, 0, criterion.settings).readouts["seeker_relief"]
    assert relief["baseline"] > 0 and relief["treatment"] > 0
    f, m, c = PersonId("f"), PersonId("m"), PersonId("c")
    s, base, treat = criterion.settings, states["baseline"], states["treatment"]
    assert (base.ties[TieId.of(f, c)].bond_energy, base.ties[TieId.of(m, c)].bond_energy) == (s["bond_far"], s["bond_near"])
    assert (treat.ties[TieId.of(f, c)].bond_energy, treat.ties[TieId.of(m, c)].bond_energy) == (s["bond_near"], s["bond_far"])
    others = set(base.ties) - {TieId.of(f, c), TieId.of(m, c)}
    assert others and all(base.ties[t].bond_energy == treat.ties[t].bond_energy for t in others)



@pytest.mark.ensemble
def test_m11c7_topology_not_coach_skill():
    for cid in [c for c in CRITERIA_D if c.startswith("M11.C.7[")]:
        gate(cid)


def test_m11c7_arms_differ_only_in_their_topology():
    """`M11.C.7`'s three topologies (version 1, 2026-10-17): only the one-spouse arm makes the second spouse's tie to the
    coach non-interactive; only the sided arm scripts the second spouse's SPLIT toward the coach, every one placed; no
    spouse SPLITs toward the coach on their own in any arm; and the sided coach's contact never lands while the joint
    coach's does (review 2026-10-17: without the last, a run where nothing lands would pass)."""
    from unittest import mock

    import src.bowen.ensemble.criteria_d as module
    from src.bowen.engine.identifiers import TieId
    from src.bowen.ensemble.criteria import ROLES

    coach, first, second = ROLES["phase_c", "coach"], ROLES["phase_c", "parent_1"], ROLES["phase_c", "parent_2"]
    real, seen = module.scenario, {}
    for topology in module.C7_TOPOLOGIES:
        def spy(*args, **kwargs):
            state, records = real(*args, **kwargs)
            seen[topology] = (state, kwargs["forced"], real.last_source, records)
            return state, records
        settings = {**CRITERIA_D["M11.C.7[joint over sided]"].settings, "weeks": 104, "window": 8,
                    "topologies": (topology, topology)}
        with mock.patch.object(module, "scenario", side_effect=spy):
            out = module.arm_c7("baseline", 0, settings)
        seen[topology] += (out.readouts["landed_contacts"],)
    assert [t for t in seen if not seen[t][0].ties[TieId.of(coach, second)].interactive] == ["one spouse"]
    assert [t for t in seen if seen[t][1]] == ["sided"]
    assert all(s.kind == "SPLIT" and s.targets == (coach,) for s in seen["sided"][1].values())
    assert set(seen["sided"][2].status.values()) == {"placed"}
    scripted = {(actor, week) for week, actor in seen["sided"][1]}
    for topology, (_, _, _, records, _) in seen.items():
        own = [r for r in records if isinstance(r, EmittedRecord) and r.event.kind == "SPLIT"
               and r.event.targets == (coach,) and r.event.sender in (first, second)
               and (r.event.sender, r.event.timestamp) not in scripted]
        assert own == [], topology
    assert seen["sided"][4] == 0 and seen["joint"][4] > 0


def test_forced_withheld_label_withholds_that_act_toward_that_person_only():
    """`Forced.withheld`'s `KIND>target` form (version 1, 2026-10-17): over the same seed, Marta's own SPLITs toward Dr
    Halim stop when the label is withheld and occur without it; a label naming someone with no tie raises (P2)."""
    from src.bowen.ensemble.criteria import ROLES, scenario

    coach, second = ROLES["phase_c", "coach"], ROLES["phase_c", "parent_2"]

    def splits(seed, withheld):
        _, records = scenario("phase_c", seed, 104, withheld=withheld)
        return sum(1 for r in records if isinstance(r, EmittedRecord) and r.event.kind == "SPLIT"
                   and r.event.sender == second and r.event.targets == (coach,))

    # The first seed whose run has such a SPLIT (P23: not one fixed seed, whose run any engine change can move).
    seed = next((s for s in range(12) if splits(s, None) > 0), None)
    assert seed is not None, "no seed in 12 has Marta SPLIT toward Dr Halim in 104 weeks"
    assert splits(seed, {t: {second: frozenset({f"SPLIT>{coach}"})} for t in range(104)}) == 0
    with pytest.raises(ValueError, match="no tie"):
        scenario("phase_c", 0, 2, withheld={0: {second: frozenset({"SPLIT>nobody"})}})


def test_m11c21_t145_the_readout_starts_at_the_end_of_week_t0_minus_1(monkeypatch):
    """T-145 (revision 41, the owner 2026-10-28): `M11.C.21`'s rise is read from C1's level at the end of week
    t0 − 1 (the watch at `s.tick == t0`, after that tick ran), so what returns to C1 in week t0 is counted."""
    import types

    from src.bowen.ensemble import criteria_d

    def child_at(level):
        return {criteria_d.CHILD: types.SimpleNamespace(functional_level=level)}

    def fake(family, seed, weeks, *, watch=None, **kwargs):
        for tick in range(1, weeks + 1):
            watch(types.SimpleNamespace(tick=tick, people=child_at(30.0 + tick)))
        return types.SimpleNamespace(people=child_at(40.0)), []

    monkeypatch.setattr(criteria_d, "scenario", fake)
    monkeypatch.setattr(criteria_d, "result", lambda readouts, records: readouts)
    settings = {"t0": 3, "after": 2, "drop": 10.0, "act_every": 1}
    assert criteria_d.arm_c21("treatment", 0, settings)["target_rise"] == pytest.approx(40.0 - 33.0)
