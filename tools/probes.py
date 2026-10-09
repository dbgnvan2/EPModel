"""Diagnostic probes for the failing Phase C criteria: measurements, never verdicts.

Purpose: each probe runs a criterion's own arm and measures what the plan's diagnosis needs (D1-D6 of
         docs/plan_phase_c_failing_criteria.md), printing one JSON row per arm and seed. Probes are reported in
         docs/phase_c_diagnostic_record.md and never gate or prove anything.
Spec:    docs/plan_phase_c_failing_criteria.md#D1 (and D2-D6 as they are added)
Tests:   tests/bowen/test_ensemble_record.py::test_d0_diagnostic_record_is_current

    python3 tools/probes.py --probe NAME [--seeds N]

A probe samples state as each tick starts by patching ``criteria.run_tick``, as tools/level_occupancy.py does, and
calls the criterion's arm function, so the arms are the criterion's own.
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
                final_values: dict[str, float] = {}

                def learn(state, start_acute, params):
                    due = [a for p in state.people.values() for a in p.eligible_acts
                           if state.tick - a.tick >= params.credit_horizon - 1]
                    out = original(state, start_acute, params)
                    closed.extend((a.key.split("|")[2], a.signal) for a in due)
                    for p in state.people.values():
                        final_values.update(p.learned_values)
                    return out

                tick.learn = learn
                try:
                    criterion.arm(arm, seed, criterion.settings)
                finally:
                    tick.learn = original
                for group, pick in (("TRIANGLE", lambda k: k == "TRIANGLE"), ("other", lambda k: k != "TRIANGLE")):
                    signals = [s for k, s in closed if pick(k)]
                    values = [v for k, v in final_values.items() if pick(k.split("|")[2])]
                    rows.append({"criterion": cid, "arm": arm, "acts": group, "seed": seed, "count": len(signals),
                                 "mean_signal": sum(signals) / len(signals) if signals else 0.0,
                                 "share_relieved": sum(s > 0 for s in signals) / len(signals) if signals else 0.0,
                                 "mean_learned_value": sum(values) / len(values) if values else 0.0})
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
                for member, terms in captured.items():
                    rows.append({"cell": cid, "arm": arm, "member": member, "seed": seed, **terms})
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
                treat = criterion.arm("treatment", seed, settings).readouts
                for readout in base:
                    rows.append({"criterion": cid, "horizon": f"{key}={value}", "readout": readout, "seed": seed,
                                 "difference": treat[readout] - base[readout]})
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


PROBES = {"c29_third_person": c29_third_person, "triangle_relief": triangle_relief,
          "c27_deviation_terms": c27_deviation_terms, "c44_act_counts": c44_act_counts, "horizons": horizons,
          "c29_third_person_calm": c29_third_person_calm, "spell_effect": spell_effect}


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
