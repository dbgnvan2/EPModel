"""The Phase D part II criteria: the life course (spec revision 28; plan §4.4).

Purpose: define `M11.C.15`, `M11.C.6`, `M11.C.10` and `M11.C.2` on the twelve-person family as paired arms.
         `M11.C.37` is a conservation check with no second arm (an `M11.4` exception, as `M11.C.31`'s ledger limb is)
         and is proved by tests/bowen/test_life_course.py::test_m11c37_anxiety_conserved_across_death and its mutant.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.15, #M11.C.6, #M11.C.10, #M11.C.2, #M6.3, #M7.E.1f
Tests:   tests/bowen/test_phase_d2_criteria.py

Every number is in ``config/bowen/criteria.md`` and every person's role in ``config/bowen/criterion_roles.md``; the
arm designs are Claude's recommendation, pending owner review, declared before any ran.

- **`M11.C.15`, death destabilises as recovery does.** The nuclear-four fixture; the mother carries a symptom held at
  ``severity`` (below lock-in's turn, so the family is calmed around it, `M7.D.2a`) until ``change_week``. Control:
  it continues. Recovery: it remits then (``binder_unavailable`` on her channel). Death: she dies then (`M6.3`).
  Grief and lost income are set to zero in every arm (`M10.B.5`). `M6.3b`'s transfer stays at its rule, 1 (owner
  decision 2026-10-13): at the death her own anxiety goes to the living, with what her symptom, ties and triangles
  held (`M6.3c`); in recovery, the symptom's hold is released and she stays. The death arm loses her as a source of
  load, which the recovery arm keeps; that difference is the comparison's, not a confound removed. Read: the living others' mean excess over the ``after`` weeks. The treatment arm runs both
  changes and reports each against the same control: both MUST be higher.
- **The twelve-person family over ``years``** (`M11.C.6`, `M11.C.10`, `M11.C.2`): the reference family with its life
  course, under the policy, with the declared nodal load on the two parents (as part I's `M11.C.2`). One run per seed,
  the family as declared, serves `M11.C.6` and `M11.C.2` (each compares children within it); `M11.C.10` runs its own
  two arms (spec revision 29).
  - `M11.C.6`: each child's basic level at 20 less its anchor at 5 (`M7.E.1f`). The treatment reports the child who
    received the most projection-type acts from the parents before turning 20; the baseline, the child who received
    the fewest. The most-projected child's change MUST be lower. One transmission (generation 2 to 3): the run is too
    short for three (plan §4.4).
  - `M11.C.10` (revision 29, the owner: "hold the cutoff for the run and read tie states"): the cut-off tie-weeks held
    over the run by the third generation and the children born to them (partners excluded: they did not grow up in
    this family) — each week, the ties on which one of them holds rejection (`M1.B.3c`). Treatment: the parents'
    generation's declared cutoff (its two ends are roles in config) held for the run, renewed by a scripted reactive
    ``CUTOFF`` every ``renew_every`` weeks. A renewal is skipped when either end has died or the act is not legal that
    week; each skip and the weeks the cutter held rejection are counted in the arm's bookkeeping. Baseline: that tie ordinary at ``t0`` and held open for the run. More MUST
    follow the held cutoff.
  - `M11.C.2` (revision 29, the owner: "Child load is by age"): projection-type acts received per year while the child
    is under ``projection_until_age``; symptom load as the child's mean over ages ``load_from_age`` to
    ``load_to_age``. The most-projected child's load MUST be higher.
"""

from __future__ import annotations

import dataclasses
import functools

from src.bowen.engine.act import Selection
from src.bowen.engine.contact import excess, resting_contact
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.life_course import LifeCourse, LifeSettings, TieValues, die
from src.bowen.engine.log_records import EmittedRecord
from src.bowen.engine.objects import Sex, TieState
from src.bowen.ensemble.criteria import FAMILIES, ROLES, SETTINGS, result, scenario, scripted_bookkeeping
from src.bowen.ensemble.criteria_d import ENGINE, held_symptom, nodal, remission
from src.bowen.ensemble.runner import ArmResult, Criterion, Readout
from src.bowen.io.load import load_family

P = PersonId
FATHER, MOTHER, CHILD, PARTNER = P("f"), P("m"), P("c1"), P("c2")  # the nuclear-four fixture's ids
# M11.C.15: no grief and no lost income (M10.B.5, an override declared by the criterion). M6.3b's transfer stays at 1
# (owner decision 2026-10-13: "keep the transfer in C.15, share at 1").
NO_GRIEF_OR_INCOME = {"grief_scale": 0.0, "income_loss_intensity": 0.0}
YEAR = ENGINE.slow_tick_fast_ticks


def _mortal(state) -> None:
    """Give a fixture a life course with nothing in it, so a death has somewhere to put its events (`M6.3`)."""
    none = TieValues(0.0, 0.0, 1)
    state.life = LifeCourse(settings=LifeSettings((), none, none, none, none, None, None, Sex.FEMALE), parents={})


def arm_c15(arm, seed, settings):
    """M11.C.15 (see the module docstring)."""
    at, after = settings["change_week"], settings["after"]
    others = (FATHER, CHILD, PARTNER)
    params = dataclasses.replace(ENGINE, **NO_GRIEF_OR_INCOME)
    channel = load_family(FAMILIES["four"]).people[MOTHER].channel_prior

    def run(change):
        hold = held_symptom(MOTHER, settings["severity"], until=at if change else None)
        sums = []

        def before(state):
            hold(state)
            if change == "death" and state.tick == at:
                die(state, MOTHER, params)

        def watch(state):
            if at < state.tick <= at + after:  # the weeks at .. at + after - 1, just run
                sums.append(sum(excess(state.people[p]) for p in others) / len(others))

        spells = (remission(at, MOTHER, channel),) if change == "recovery" else ()
        _, records = scenario("four", seed, at + after, spells=spells, setup=_mortal, before=before, watch=watch,
                              overrides=NO_GRIEF_OR_INCOME)
        return sum(sums) / len(sums), records

    if arm == "baseline":
        control, records = run(None)
        return result({"death_rise": control, "recovery_rise": control}, records)
    death, _ = run("death")
    recovery, records = run("recovery")
    return result({"death_rise": death, "recovery_rise": recovery}, records)


# --- the twelve-person family over 40 years --------------------------------------------------------------------


def _uncut(state) -> None:
    """M11.C.10's baseline: the parents' generation starts with no cutoff (the tie ordinary, nobody rejecting); the
    run holds it open as well (``life_run``)."""
    tie = state.tie_between(ROLES["phase_d", "cut_off_a"], ROLES["phase_d", "cut_off_b"])
    tie.rejection = {m: 0.0 for m in tie.id.members()}
    tie.tie_state = tie.uncut_state = TieState.ORDINARY
    for m in tie.id.members():
        tie.felt_contact[m] = resting_contact(state.people[m], tie, ENGINE)


def _renewals(weeks: int, every: int) -> dict:
    """M11.C.10's treatment: the cutter renews the cutoff every ``every`` weeks, a scripted reactive CUTOFF."""
    a, b = ROLES["phase_d", "cut_off_a"], ROLES["phase_d", "cut_off_b"]
    return {(w, a): Selection(actor=a, kind="CUTOFF", targets=(b,), intensity=ENGINE.policy_intensity)
            for w in range(0, weeks, every)}


def cut_off_tie_weeks(state, children) -> int:
    """Purpose: M11.C.10's weekly reading — the ties on which generation 3 or a child of theirs holds rejection.
    Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.10
    Tests:   tests/bowen/test_phase_d2_criteria.py::test_m11c10_counts_only_rejection_held_by_generation_3_and_their_children

    A tie counts once a week however many of them hold rejection on it; rejection held only by the parents' generation
    (the declared pair, the partners' parents) does not count.
    """
    held_by = set(children) | {p for p, ps in state.life.parents.items() if set(ps) & set(children)}
    return sum(1 for t in state.ties.values() if any(v > 0 and m in held_by for m, v in t.rejection.items()))


@functools.lru_cache(maxsize=2)
def life_run(seed: int, years: int, arm: str = "declared", renew_every: int = 0, ages: tuple = ()) -> dict:
    """Purpose: one run of the twelve-person family, summarised for the three criteria that read it; cached per
    process, which spares only a repeated call with the same arguments (criteria run one after another, and each
    worker has its own cache).
    Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.6, #M11.C.10, #M11.C.2
    Tests:   tests/bowen/test_phase_d2_criteria.py::test_m11c6_the_children_are_read_at_twenty

    ``arm`` is "declared" (the family as declared, for `M11.C.6` and `M11.C.2`), or `M11.C.10`'s "held" (the cutoff
    renewed every ``renew_every`` weeks) or "open" (the tie ordinary and held open). ``ages`` is `M11.C.2`'s
    (projection_until_age, load_from_age, load_to_age).
    """
    from src.bowen.readouts.patterns import PROJECTION_ACTS

    weeks = years * YEAR
    parents = (ROLES["phase_d", "parent_1"], ROLES["phase_d", "parent_2"])
    pair = (ROLES["phase_d", "cut_off_a"], ROLES["phase_d", "cut_off_b"])
    family = load_family(FAMILIES["phase_d"])
    children = tuple(sorted(p for p in family.people if set(family.parents_of(p)) == set(parents)))
    half = weeks // 2
    until, load_from, load_to = ages or (None, None, None)
    at_twenty, turned, loads = {}, {}, {c: 0.0 for c in children}
    under, aged_load, aged_weeks = {}, {c: 0.0 for c in children}, {c: 0 for c in children}
    cut_weeks, held = [0], [0]
    end = ENGINE.development_end_age_years

    def watch(state):
        for c in children:
            person = state.people[c]
            if c not in at_twenty and person.age_years is not None and person.age_years >= end:
                at_twenty[c], turned[c] = person.basic_level - state.life.anchors[c], state.tick
            if state.tick > half:
                loads[c] += sum(person.symptom_load.values()) / (weeks - half)
            if until is not None and person.age_years is not None:
                if c not in under and person.age_years >= until:
                    under[c] = state.tick
                if person.alive and load_from <= person.age_years < load_to:
                    aged_load[c] += sum(person.symptom_load.values())
                    aged_weeks[c] += 1
        if arm != "declared":
            cut_weeks[0] += cut_off_tie_weeks(state, children)
            held[0] += state.tie_between(*pair).rejection.get(pair[0], 0.0) > 0

    state, records = scenario(
        "phase_d", seed, weeks, spells=nodal(weeks, parents), watch=watch,
        setup=_uncut if arm == "open" else None,
        forced=_renewals(weeks, renew_every) if arm == "held" else None,
        held_open=((pair,), range(weeks)) if arm == "open" else None)
    moves = [r.event for r in records if isinstance(r, EmittedRecord)]
    books = scripted_bookkeeping(records)  # a skipped renewal is counted, never silent (learning-qa 2026-10-14)
    if arm != "declared":
        books[0]["(weeks the cutter held rejection on the declared tie)"] += held[0]

    def received(c, until_tick):
        return sum(1 for e in moves if e.kind in PROJECTION_ACTS and e.sender in parents and c in e.targets
                   and e.timestamp < until_tick)

    return {
        "children": children,
        "development": at_twenty,
        "childhood_acts": {c: received(c, turned.get(c, weeks)) for c in children},
        "early_acts": {c: received(c, half) for c in children},
        "late_load": loads,
        # M11.C.2 by age: acts per year observed under ``until``; load over the ages read, where any week was.
        "acts_per_year": {c: received(c, under.get(c, weeks)) / max(1, under.get(c, weeks)) * YEAR for c in children},
        "aged_load": {c: aged_load[c] / aged_weeks[c] for c in children if aged_weeks[c]},
        "cut_weeks": float(cut_weeks[0]),
        "books": books,
    }


def _pick(arm, acts: dict, values: dict) -> float:
    """The treatment reads the child who received the most, the baseline the fewest; a tie reads 0 in both."""
    ranked = sorted((c for c in acts if c in values), key=lambda c: acts[c])
    if not ranked or acts[ranked[0]] == acts[ranked[-1]]:
        return 0.0
    return values[ranked[-1] if arm == "treatment" else ranked[0]]


def _life_result(readouts: dict, summary: dict) -> ArmResult:
    return ArmResult(readouts, *summary["books"])


def arm_c6(arm, seed, settings):
    """M11.C.6 (see the module docstring)."""
    summary = life_run(seed, settings["years"], "declared")
    return _life_result({"development": _pick(arm, summary["childhood_acts"], summary["development"])}, summary)


def arm_c2_twelve(arm, seed, settings):
    """M11.C.2 on the twelve-person family, read by age (see the module docstring)."""
    ages = settings["projection_until_age"], settings["load_from_age"], settings["load_to_age"]
    summary = life_run(seed, settings["years"], "declared", 0, ages)
    return _life_result({"symptom_load": _pick(arm, summary["acts_per_year"], summary["aged_load"])}, summary)


def arm_c10(arm, seed, settings):
    """M11.C.10 (see the module docstring)."""
    summary = life_run(seed, settings["years"], "open" if arm == "baseline" else "held", settings["renew_every"])
    return _life_result({"cut_off_tie_weeks": summary["cut_weeks"]}, summary)


C = Criterion
CRITERIA_D2: dict[str, Criterion] = {
    "M11.C.15": C("M11.C.15", "composite", ("baseline", "treatment"),
                  (Readout("death_rise", +1), Readout("recovery_rise", +1)), arm_c15, settings=SETTINGS["M11.C.15"]),
    "M11.C.6": C("M11.C.6", "composite", ("baseline", "treatment"), (Readout("development", -1),), arm_c6,
                 settings=SETTINGS["M11.C.6"]),
    "M11.C.10": C("M11.C.10", "composite", ("baseline", "treatment"), (Readout("cut_off_tie_weeks", +1),), arm_c10,
                  ("CUTOFF",), settings=SETTINGS["M11.C.10"]),
    "M11.C.2[twelve-person family]": C("M11.C.2", "composite", ("baseline", "treatment"),
                                       (Readout("symptom_load", +1),), arm_c2_twelve, settings=SETTINGS["M11.C.2"]),
}
