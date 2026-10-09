"""Diagnostic probes for the failing Phase C criteria: measurements, never verdicts.

Purpose: each probe runs a criterion's own arm and measures what the plan's diagnosis needs (D1-D6 of
         docs/plan_phase_c_failing_criteria.md), printing one JSON row per arm and seed. Probes are reported in
         docs/phase_c_diagnostic_record.md and never gate or prove anything.
Spec:    docs/plan_phase_c_failing_criteria.md#D1 (and D2-D6 as they are added)
Tests:   tests/bowen/test_ensemble_record.py::test_d0_diagnostic_record_is_current

    python3 tools/probes.py --probe NAME [--seeds N]

A probe samples state as each tick starts by patching ``criteria.run_tick``, as tools/level_occupancy.py does, and
calls the criterion's arm function, so the arms are the criterion's own (``c44_act_counts`` re-runs the arm's
scenario to read its records, and must be kept in step with ``arm_spell_triad``). Sampling at tick start misses the
last tick and reads anxiety after the previous tick's decay, so peaks are slightly understated. A forced arm's rows
say whether its scripted act was made (``made``): an act that is not legal that week is skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def sampling(criteria, sample):
    """A context that calls ``sample(state)`` at the start of every tick of every scenario the arm runs."""
    class Patch:
        def __enter__(self):
            self.original = criteria.run_tick

            def sampled(state, *args, **kwargs):
                sample(state)
                return self.original(state, *args, **kwargs)

            criteria.run_tick = sampled

        def __exit__(self, *exc):
            criteria.run_tick = self.original

    return Patch()


def made(criteria) -> str:
    """Whether the last scenario's scripted acts were made: "made", "skipped" (not made: not legal that week, or the
    actor was not selecting), "partly" (some of each), "not reached" (neither: the run ended first), or "none" (the
    arm scripts no act). Every arm calls ``scenario`` once, which sets ``last_source``; an arm that called it twice
    would report its last call."""
    source = getattr(criteria.scenario, "last_source", None)
    if source is None or not source.forced:
        return "none"
    missed = source.skipped + getattr(source, "owed_step", 0)
    if source.made == 0 and missed == 0:
        return "not reached"
    if missed == 0:
        return "made"
    return "skipped" if source.made == 0 else "partly"


def c29_third_person(seeds: int) -> list[dict]:
    """D1: does the third person in M11.C.29's triad ever rise above their chronic floor, and how close does their
    symptom load come to onset? Also the two parents, for comparison."""
    import src.bowen.ensemble.criteria as criteria
    from src.bowen.engine.contact import excess
    from src.bowen.engine.identifiers import PersonId
    from src.bowen.io.load import load_constants

    gain = load_constants()["symptom_threshold_gain"]
    criterion = criteria.CRITERIA["M11.C.29"]
    rows = []
    for arm in criterion.arms:
        for seed in range(seeds):
            seen: dict[str, dict] = {}

            def sample(state):
                for pid in ("f", "m", "c"):
                    p = state.people[PersonId(pid)]
                    s = seen.setdefault(pid, {"weeks_above_floor": 0, "peak_excess": 0.0, "peak_acute": 0.0,
                                              "chronic": p.chronic_anxiety, "peak_load": 0.0,
                                              "threshold_at_peak_load": 0.0})
                    s["weeks_above_floor"] += excess(p) > 0
                    s["peak_excess"] = max(s["peak_excess"], excess(p))
                    s["peak_acute"] = max(s["peak_acute"], p.acute_anxiety)
                    load = p.symptom_load[p.channel_prior] if p.channel_prior is not None else 0.0
                    if load >= s["peak_load"]:
                        s["peak_load"], s["threshold_at_peak_load"] = load, gain * p.functional_level

            with sampling(criteria, sample):
                criterion.arm(arm, seed, criterion.settings)
            for pid, s in seen.items():
                rows.append({"arm": arm, "seed": seed, "person": pid, **s})
    return rows


def triangle_relief(seeds: int) -> list[dict]:
    """D3: what the learner credits a TRIANGLE with, against every other automatic act, in M11.C.42's and M11.C.45's
    arms: the closed signal (felt relief over the credit horizon, before habituation) and the learned values left at
    the end. Run under the current roles and under `triangle-roles-swapped` (step 5's alliance reading)."""
    import src.bowen.engine.tick as tick
    import src.bowen.ensemble.criteria as criteria

    rows = []
    original = tick.learn
    for cid in ("M11.C.42", "M11.C.45"):
        criterion = criteria.CRITERIA[cid]
        for arm in criterion.arms:
            for seed in range(seeds):
                closed: list[tuple[str, float]] = []
                final_values: dict[tuple[str, str], float] = {}

                def learn(state, start_acute, params):
                    due = [a for p in state.people.values() for a in p.eligible_acts
                           if state.tick - a.tick >= params.credit_horizon - 1]
                    out = original(state, start_acute, params)
                    closed.extend((a.key.split("|")[2], a.signal) for a in due)
                    for pid, p in state.people.items():
                        final_values.update({(pid.value, k): v for k, v in p.learned_values.items()})
                    return out

                tick.learn = learn
                try:
                    criterion.arm(arm, seed, criterion.settings)
                finally:
                    tick.learn = original
                for group, pick in (("TRIANGLE", lambda k: k == "TRIANGLE"), ("other", lambda k: k != "TRIANGLE")):
                    signals = [s for k, s in closed if pick(k)]
                    values = [v for (_, k), v in final_values.items() if pick(k.split("|")[2])]
                    rows.append({"criterion": cid, "arm": arm, "acts": group, "seed": seed, "count": len(signals),
                                 "mean_signal": sum(signals) / len(signals) if signals else None,
                                 "share_relieved": sum(s > 0 for s in signals) / len(signals) if signals else None,
                                 "mean_learned_value": sum(values) / len(values) if values else None})
    return rows


def c27_deviation_terms(seeds: int) -> list[dict]:
    """D4: M11.C.27's pair deviation at the end of each arm, split into its two sides per member: "too little"
    contact (optimum less contact, beyond the band) and "too much" impingement (beyond the band), with the terms."""
    import src.bowen.ensemble.criteria as criteria
    from src.bowen.engine.contact import band, felt_contact, felt_impingement, optimum
    from src.bowen.engine.identifiers import PersonId, TieId

    rows = []
    original = criteria._pair_deviation
    for cid, criterion in criteria.CRITERIA.items():
        if not cid.startswith("M11.C.27["):
            continue
        for arm in criterion.arms:
            for seed in range(seeds):
                captured = {}

                def pair_deviation(state):
                    params = criteria._params()
                    tie = state.ties[TieId.of(PersonId("f"), PersonId("m"))]
                    for m in tie.id.members():
                        person = state.people[m]
                        o, c, i, b = optimum(person, tie, params), felt_contact(person, tie), \
                            felt_impingement(person, tie), band(person, params)
                        captured[m.value] = {"optimum": o, "contact": c, "impingement": i, "band": b,
                                             "too_little": max(0.0, o - c - b), "too_much": max(0.0, i - b)}
                    return original(state)

                criteria._pair_deviation = pair_deviation
                try:
                    criterion.arm(arm, seed, criterion.settings)
                finally:
                    criteria._pair_deviation = original
                status = made(criteria)
                for member, terms in captured.items():
                    rows.append({"cell": cid, "arm": arm, "act": status, "member": member, "seed": seed, **terms})
    return rows


def c44_act_counts(seeds: int) -> list[dict]:
    """D5: the outside-ward and inside-ward act counts behind M11.C.44's ratio, and TRIANGLE counts, per arm."""
    import src.bowen.ensemble.criteria as criteria
    from src.bowen.engine.events import Mechanism
    from src.bowen.engine.identifiers import PersonId
    from src.bowen.engine.log_records import EmittedRecord

    criterion = criteria.CRITERIA["M11.C.44"]
    triad = {PersonId("f"), PersonId("m"), PersonId("c")}
    rows = []
    for arm in criterion.arms:
        for seed in range(seeds):
            _, records = criteria.scenario("triad", seed, criterion.settings["weeks"],
                                           spells=criteria.spell("triad", criterion.settings["weeks"])
                                           if arm == "treatment" else ())
            kinds = [r.event.kind for r in records if isinstance(r, EmittedRecord)
                     and r.event.mechanism is Mechanism.MOVE and r.event.sender in triad]
            rows.append({"arm": arm, "seed": seed, "moves": len(kinds),
                         "outside_acts": sum(k in criteria.OUTSIDE_ACTS for k in kinds),
                         "inside_acts": sum(k in criteria.INSIDE_ACTS for k in kinds),
                         "triangles": kinds.count("TRIANGLE"), "pursues": kinds.count("PURSUE"),
                         "distances": kinds.count("DISTANCE"), "cutoffs": kinds.count("CUTOFF")})
    return rows


C4_NODAL_WEEKS = (30, 60, 120, 240)   # [I], declared here: the criterion's own (30), then doubling
C5_WEEKS = (60, 120, 260)             # [I], declared here: the criterion's own (60), then about 2 and 5 years


def horizons(seeds: int) -> list[dict]:
    """D6: M11.C.4's two readouts with its nodal event moved later, and M11.C.5's with a longer run, per seed:
    the treatment minus baseline difference. Only the horizon changes; every other setting is the criterion's."""
    import src.bowen.ensemble.criteria as criteria

    rows = []
    for cid, key, values in (("M11.C.4", "nodal", C4_NODAL_WEEKS), ("M11.C.5", "weeks", C5_WEEKS)):
        criterion = criteria.CRITERIA[cid]
        for value in values:
            settings = {**criterion.settings, key: value}
            for seed in range(seeds):
                base = criterion.arm("baseline", seed, settings).readouts
                base_act = made(criteria)
                treat = criterion.arm("treatment", seed, settings).readouts
                acts = f"treatment {made(criteria)}, baseline {base_act}"
                for readout in base:
                    rows.append({"criterion": cid, "horizon": f"{key}={value}", "acts": acts, "readout": readout,
                                 "seed": seed, "difference": treat[readout] - base[readout]})
    return rows


def c29_third_person_calm(seeds: int) -> list[dict]:
    """D1, follow-up: c29_third_person with the declared spell removed from both arms, to see whether the third
    person then stays below the onset threshold, so that a time course could show."""
    import src.bowen.ensemble.criteria as criteria

    original = criteria.spell
    criteria.spell = lambda *args, **kwargs: ()
    try:
        return c29_third_person(seeds)
    finally:
        criteria.spell = original


def spell_effect(seeds: int) -> list[dict]:
    """D5, follow-up: is M11.C.44/.45's calm arm calm? Each triad member's mean acute anxiety and weeks above the
    chronic floor, per arm."""
    import src.bowen.ensemble.criteria as criteria
    from src.bowen.engine.contact import excess
    from src.bowen.engine.identifiers import PersonId

    criterion = criteria.CRITERIA["M11.C.45"]
    rows = []
    for arm in criterion.arms:
        for seed in range(seeds):
            seen: dict[str, list[float]] = {}

            def sample(state):
                for pid in ("f", "m", "c"):
                    p = state.people[PersonId(pid)]
                    seen.setdefault(pid, []).append((p.acute_anxiety, excess(p) > 0))

            with sampling(criteria, sample):
                criterion.arm(arm, seed, criterion.settings)
            for pid, samples in seen.items():
                rows.append({"arm": arm, "person": pid, "seed": seed,
                             "mean_acute": sum(a for a, _ in samples) / len(samples),
                             "share_weeks_above_floor": sum(b for _, b in samples) / len(samples)})
    return rows


FORCED = ("M11.C.3", "M11.C.4", "M11.C.5", "M11.C.27[", "M11.C.29", "M11.C.32", "M11.C.35", "M11.C.42")


def scripted_acts(seeds: int) -> list[dict]:
    """Every criterion whose arms script an act: in how many seeds the act was made, or skipped because it was not
    legal that week (for example, the policy had already cut the tie the act crosses)."""
    import src.bowen.ensemble.criteria as criteria

    rows = []
    for cid, criterion in criteria.CRITERIA.items():
        if not any(cid == f or (f.endswith("[") and cid.startswith(f)) for f in FORCED):
            continue
        for arm in criterion.arms:
            for seed in range(seeds):
                criteria.scenario.last_source = None
                criterion.arm(arm, seed, criterion.settings)
                source = criteria.scenario.last_source
                rows.append({"criterion": cid, "arm": arm, "act": made(criteria), "seed": seed,
                             "made": source.made if source else 0, "skipped": source.skipped if source else 0,
                             "not_selecting": source.owed_step if source else 0})
    return rows


def _scan_arm(cid: str, arm: str, seed: int) -> dict | None:
    """One arm and seed of ``scripted_weeks``: None if the arm scripts no act."""
    import src.bowen.ensemble.criteria as criteria
    from src.bowen.engine.observe import observe
    from src.bowen.policy.policy import legal_outcomes

    criterion = criteria.CRITERIA[cid]
    t0 = criterion.settings["t0"]
    seen: dict = {}

    class Scan(criteria.Forced):
        """The policy's own run (no act is made, no tie held open), recording each tick whether each scripted act
        would be."""

        def __init__(self, policy, forced, unavailable=None, held_open=None):
            super().__init__(policy, {}, unavailable, None)
            self.script = {(t - t0, s.actor, s.kind, s.targets[0]) for (t, _), s in forced.items()}
            seen["script"] = self.script

        def selections(self, tick, active, state):
            chosen = super().selections(tick, active, state)
            for offset, actor, kind, target in self.script:
                legal = actor in active and f"{kind}>{target}" in {
                    o.label for o in legal_outcomes(observe(state, actor, self.policy.params), state.kinds,
                                                    self.policy.params)}
                seen[(tick, offset)] = seen.get((tick, offset), True) and legal
            return chosen

    original_forced, original_scenario = criteria.Forced, criteria.scenario

    def short(family, seed, weeks, **kwargs):  # every act's latest possible week is t0 + its offset
        offsets = [t - t0 for t, _ in (kwargs.get("forced") or {})]
        return original_scenario(family, seed, weeks, until=min(weeks, t0 + max(offsets, default=0) + 1), **kwargs)

    criteria.Forced, criteria.scenario = Scan, short
    try:
        criterion.arm(arm, seed, criterion.settings)
    finally:
        criteria.Forced, criteria.scenario = original_forced, original_scenario
    script = seen.get("script")
    if not script:
        return None
    first = next((w for w in range(t0 + 1) if not all(seen.get((w + o, o), False) for o, *_ in script)), t0 + 1)
    acts = ", ".join(sorted(f"{a}:{k}>{t}" + (f"@+{o}" if o else "") for o, a, k, t in script))
    return {"criterion": cid, "arm": arm, "acts": acts, "declared": f"t0={t0}", "seed": seed,
            "first_week_not_made": first}


def scripted_weeks(seeds: int) -> list[dict]:
    """Step S (docs/DECISIONS — PHASE C FAILING.md): for every criterion that scripts an act, the first week w at
    or before its declared t0 at which, scripted at w, the arm's acts would not all be made (illegal, `M4.D.1e`, or
    the actor inactive), per seed; t0 + 1 if none. The latest week made in every seed from week 0 on is the minimum
    over both arms' seeds, less one. The run is the policy's own, without step S's held-open ties: this is the
    measurement that showed only week 0 works (so the ties are now held open). For an arm that scripts several acts
    (`M11.C.29`'s disguise), the later acts are checked in that unforced run, not after the earlier ones were made.
    Whether the restated arms make their acts is ``scripted_acts``, and the ensemble record's skip column."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    import src.bowen.ensemble.criteria as criteria

    jobs = [(cid, arm, seed) for cid, c in criteria.CRITERIA.items()
            if any(cid == f or (f.endswith("[") and cid.startswith(f)) for f in FORCED)
            for arm in c.arms for seed in range(seeds)]
    with ProcessPoolExecutor(os.cpu_count() or 1) as pool:
        rows = list(pool.map(_scan_arm, *zip(*jobs), chunksize=25))
    return [r for r in rows if r is not None]


REST_WEEKS, REST_BURN_IN = 80, 20                    # [I], declared here: M11.C.44/.45's horizon; weeks left out
REST_LEVELS = (None, 20.0, 60.0, 80.0, 95.0)         # [I]: the fixture's own levels, then every member set to each
REST_TOLERANCE = 5.0                                 # [I]: "a few points" above the floor, the constants' own design
REST_SOURCES = {"standing_load": "ties", "appraisal": "appraisal", "competing_urges": "urges",
                "cutoff": "relief", "distance_binding": "relief", "calm_contact": "relief"}  # every other: "other"


def rest_state(seeds: int) -> list[dict]:
    """D7 (X1): is there a calm state? Each member's excess over the chronic floor with no spell and no scripted act,
    on both fixtures, at the fixture's own levels and with every member set to each of REST_LEVELS, and where that
    excess comes from. A source's share is its mean input per week times (1 − decay) / decay, the steady excess that
    input alone would hold (consolidate sheds ``acute_decay_rate`` of the excess each week); the self term is
    `M4.A.5`'s, separated from the ties' standing load. The parts need not sum to the mean excess: decay clamps at
    the floor and the window is finite. Weeks before REST_BURN_IN are left out."""
    import src.bowen.ensemble.criteria as criteria
    from src.bowen.engine.contact import excess
    from src.bowen.engine.log_records import EffectRecord
    from src.bowen.engine.standing_load import self_term
    from src.bowen.io.load import load_constants
    from src.bowen.scenario.params import engine_params

    params = engine_params(load_constants())
    factor = (1 - params.acute_decay_rate) / params.acute_decay_rate
    rows = []
    for family in ("triad", "phase_c"):
        for level in REST_LEVELS:
            def setup(state, level=level):
                if level is None:
                    return
                for p in state.people.values():
                    if p.role.value == "member":
                        p.basic_level = p.functional_level = level
                        p.pseudo_self = p.swing
                criteria._reinitialise(state)

            for seed in range(seeds):
                samples: dict = {}

                def sample(state):
                    if state.tick >= REST_BURN_IN:
                        for p in state.people.values():
                            if p.role.value == "member" and p.alive:
                                samples.setdefault(p.id, []).append(excess(p))

                with sampling(criteria, sample):
                    state, records = criteria.scenario(family, seed, REST_WEEKS, setup=setup)
                weeks = REST_WEEKS - REST_BURN_IN
                inputs: dict = {}
                for r in records:
                    if isinstance(r, EffectRecord) and r.tick >= REST_BURN_IN and r.mechanism != "acute_decay":
                        for pid, v in r.acute_anxiety:
                            key = (pid, REST_SOURCES.get(r.mechanism, "other"))
                            inputs[key] = inputs.get(key, 0.0) + v
                for pid, values in sorted(samples.items()):
                    person = state.people[pid]
                    own = self_term(person, params)
                    part = {s: inputs.get((pid, s), 0.0) / weeks * factor for s in
                            ("ties", "appraisal", "urges", "relief", "other")}
                    part["ties"] -= own * factor
                    rows.append({"fixture": family, "level": "own" if level is None else f"{level:g}",
                                 "person": str(pid), "seed": seed, "floor": person.chronic_anxiety,
                                 "mean_excess": sum(values) / len(values),
                                 "share_weeks_within_tolerance": sum(v <= REST_TOLERANCE for v in values) / len(values),
                                 "from_self": own * factor, **{f"from_{s}": v for s, v in part.items()}})
    return rows


ACT_EFFECT_ACTS = (("TRIANGLE", "c"), ("DISTANCE", "m"), ("CONFLICT", "m"), ("PURSUE", "m"))  # [I], declared here


def act_effects(seeds: int) -> list[dict]:
    """Q3: what a TRIANGLE does to the sender, the other in the pair and the third, against other automatic acts.
    M11.C.3's design (triad, no spell, its declared t0, its ties held open until then): f makes the act, against
    STAY-IN-CONTACT toward m in the baseline, on the same seed; per person, treatment minus baseline acute anxiety
    at the end of each week from t0 through the learner's horizon (`credit_horizon`). This is the act's own effect, which the learner's credit (`d3-…`) is not:
    the credit also holds decay and everything else that happened in those weeks."""
    import src.bowen.ensemble.criteria as criteria
    from src.bowen.engine.identifiers import PersonId
    from src.bowen.io.load import load_constants

    horizon = int(load_constants()["credit_horizon"])
    t0 = criteria.SETTINGS["M11.C.3"]["t0"]
    held = criteria.held(t0, "f-m", "f-c")  # every act here crosses f-m or f-c
    people = tuple(PersonId(p) for p in ("f", "m", "c"))
    rows = []
    for seed in range(seeds):
        _, base = criteria.scenario("triad", seed, t0 + horizon + 1, held_open=held,
                                    forced=criteria.act("f", "STAY-IN-CONTACT", "m", t0))
        base_made = made(criteria)
        for kind, target in ACT_EFFECT_ACTS:
            _, treat = criteria.scenario("triad", seed, t0 + horizon + 1, held_open=held,
                                         forced=criteria.act("f", kind, target, t0))
            status = made(criteria) if base_made == "made" else f"baseline {base_made}"
            for week in range(horizon + 1):
                b, t = criteria._acute_at(base, t0 + week, people), criteria._acute_at(treat, t0 + week, people)
                rows.append({"act": f"{kind}>{target}", "made": status, "week": f"t0+{week}", "seed": seed,
                             **{f"{p}_difference": t[p] - b[p] for p in people}})
    return rows


PROBES = {"c29_third_person": c29_third_person, "triangle_relief": triangle_relief,
          "c27_deviation_terms": c27_deviation_terms, "c44_act_counts": c44_act_counts, "horizons": horizons,
          "c29_third_person_calm": c29_third_person_calm, "spell_effect": spell_effect, "scripted_acts": scripted_acts,
          "scripted_weeks": scripted_weeks, "rest_state": rest_state, "act_effects": act_effects}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, choices=sorted(PROBES))
    parser.add_argument("--seeds", type=int, required=True)
    args = parser.parse_args()
    for row in PROBES[args.probe](args.seeds):
        print(json.dumps(row, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
