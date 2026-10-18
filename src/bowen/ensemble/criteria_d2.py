"""The Phase D part II criteria: the life course (spec revision 28; plan §4.4).

Purpose: define `M11.C.15`, `M11.C.6`, `M11.C.10` and `M11.C.2` on the twelve-person family as paired arms.
         `M11.C.37` is a conservation check with no second arm (an `M11.4` exception, as `M11.C.31`'s ledger limb is)
         and is proved by tests/bowen/test_life_course.py::test_m11c37_anxiety_conserved_across_death and its mutant.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.15, #M11.C.6, #M11.C.10, #M11.C.2, #M6.3, #M7.E.1f
Tests:   tests/bowen/test_phase_d2_criteria.py

Every number is in ``config/bowen/criteria.md`` and every person's role in ``config/bowen/criterion_roles.md``; the
arm designs are Claude's recommendation, declared before any ran, approved by the owner 2026-10-16.

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
  - `M11.C.6` (restated at revision 32, post hoc, Claude's recommendation pending owner review): one declared child,
    ``child``, in two arms of the same seed, run until the child is 20. In the treatment the parents act as the policy
    chooses; in the baseline their projection-type acts toward the child (`PROJECTION_ACTS`) are withheld, every week.
    Read: the child's basic level at 20 less its anchor at 5 (`M7.E.1f`), which MUST be lower under projection. The
    first form compared the most- and least-projected children of one run by raw counts over different spans, so it
    picked by birth order and run year rather than by projection (trace of 2026-10-17, plan §4.12). One transmission
    (generation 2 to 3).
  - `M11.C.10` (revision 29, the owner: "hold the cutoff for the run and read tie states"; revision 30: "partner and own
    parents", "200 instead", "b as second variant"): the tie-weeks over the run on which the third generation hold
    rejection (`M1.B.3c`) toward their partner or their own parents, over exactly ``seeds`` seeds; rejection of their
    own children is reported only. Its mechanism is `M1.D.8a`. The second variant, `M11.C.10[family of origin]`,
    holds each parent's cutoff of their own parents the same way. Treatment: the parents'
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
from collections import Counter

from src.bowen.engine.act import Selection
from src.bowen.engine.events import Channel, Event, EventId, Mechanism, SourcePosition
from src.bowen.engine.contact import excess, resting_contact
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.life_course import LifeCourse, LifeSettings, TieValues, die
from src.bowen.engine.log_records import EmittedRecord
from src.bowen.engine.objects import Sex, TieState
from src.bowen.engine.rejection import raise_rejection
from src.bowen.ensemble.criteria import FAMILIES, PARENTS, ROLES, SETTINGS, _reinitialise, result, scenario, scripted_bookkeeping
from src.bowen.ensemble.criteria_d import ENGINE, held_symptom, nodal, remission
from src.bowen.ensemble.runner import ArmResult, Criterion, Readout
from src.bowen.io.load import load_family

P = PersonId
FATHER, MOTHER, CHILD, PARTNER = P("f"), P("m"), P("c1"), P("c2")  # the nuclear-four fixture's ids
# M11.C.15: no grief and no lost income (M10.B.5, an override declared by the criterion). M6.3b's transfer stays at 1
# (owner decision 2026-10-13: "keep the transfer in C.15, share at 1").
NO_GRIEF_OR_INCOME = {"grief_scale": 0.0, "income_loss_intensity": 0.0}
YEAR = ENGINE.slow_tick_fast_ticks
PARENTS_DANGER = PARENTS["danger"]  # the danger fixture's spouses (config/bowen/fixtures/danger.md)


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


def _declared_pair() -> tuple:
    return ((ROLES["phase_d", "cut_off_a"], ROLES["phase_d", "cut_off_b"]),)


def _origin_pairs() -> tuple:
    """M11.C.10[family of origin], revision 30: each parent and each of their own parents, the parent cutting off."""
    family = load_family(FAMILIES["phase_d"])
    return tuple((p, g) for p in (ROLES["phase_d", "parent_1"], ROLES["phase_d", "parent_2"])
                 for g in sorted(family.parents_of(p)))


def _uncut(pairs):
    """M11.C.10's baseline: each tie starts with no cutoff (ordinary, nobody rejecting); the run holds it open as well
    (``life_run``)."""
    def setup(state) -> None:
        for a, b in pairs:
            tie = state.tie_between(a, b)
            tie.rejection = {m: 0.0 for m in tie.id.members()}
            tie.tie_state = tie.uncut_state = TieState.ORDINARY
            for m in tie.id.members():
                tie.felt_contact[m] = resting_contact(state.people[m], tie, ENGINE)
    return setup


def _cut(pairs):
    """M11.C.10's treatment starts each named cutoff held, as the family declares its own (the cut_off_a role): the
    cutter rejecting at ``declared_rejection``. Measured 2026-10-14: a fresh cutoff fades out in about 10 weeks, before
    a 13-week renewal."""
    def setup(state) -> None:
        for a, b in pairs:
            tie = state.tie_between(a, b)
            if tie.rejection.get(a, 0.0) < ENGINE.declared_rejection:
                raise_rejection(tie, a, ENGINE.declared_rejection - tie.rejection.get(a, 0.0))
    return setup


def _renewals(weeks: int, every: int, pairs=None) -> dict:
    """M11.C.10's treatment: each cutter renews their cutoff every ``every`` weeks, a scripted reactive CUTOFF. A cutter
    with more than one tie to hold renews them in successive weeks (one act a week each)."""
    forced, used = {}, {}
    for a, b in pairs or _declared_pair():
        offset = used.get(a, 0)
        used[a] = offset + 1
        for w in range(offset, weeks, every):
            forced[w, a] = Selection(actor=a, kind="CUTOFF", targets=(b,), intensity=ENGINE.policy_intensity)
    return forced


def partner_of(state, pid):
    """The person's partner: the other member of a family's marital tie that holds them, or None."""
    return next((t.other(pid) for f in state.families() if (t := f.marital_tie) is not None and pid in t.members()),
                None)


def cut_off_tie_weeks(state, children) -> tuple[int, int]:
    """Purpose: M11.C.10's weekly reading — the ties on which generation 3 hold rejection toward their partner or their
    own parents; and, reported only, toward their own children.
    Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.10
    Tests:   tests/bowen/test_phase_d2_criteria.py::test_m11c10_reads_rejection_toward_partners_and_parents

    Revision 30 (the owner, 2026-10-14: "partner and own parents", "less to one's children"). A tie counts once a
    week; rejection held by anyone else, the declared pair among them, does not count.
    """
    tested = reported = 0
    for child in children:
        if not state.people[child].alive:
            continue
        toward = {partner_of(state, child), *state.life.parents.get(child, ())} - {None}
        own = {p for p, ps in state.life.parents.items() if child in ps}
        for tie in state.ties_of(child):
            if tie.rejection.get(child, 0.0) <= 0.0:
                continue
            other = tie.id.other(child)
            tested += other in toward
            reported += other in own
    return tested, reported


ARMS_C10 = {"open": (_declared_pair, False), "held": (_declared_pair, True),
            "origin_open": (_origin_pairs, False), "origin_held": (_origin_pairs, True)}


@functools.lru_cache(maxsize=2)
def life_run(seed: int, years: int, arm: str = "declared", renew_every: int = 0, ages: tuple = ()) -> dict:
    """Purpose: one run of the twelve-person family, summarised for the three criteria that read it; cached per
    process, which spares only a repeated call with the same arguments (criteria run one after another, and each
    worker has its own cache).
    Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.6, #M11.C.10, #M11.C.2
    Tests:   tests/bowen/test_phase_d2_criteria.py::test_m11c6_the_children_are_read_at_twenty

    ``arm`` is "declared" (the family as declared, for `M11.C.6` and `M11.C.2`), or one of `M11.C.10`'s
    (``ARMS_C10``): "held" (the declared cutoff renewed every ``renew_every`` weeks) or "open" (that tie ordinary and
    held open), and the family-of-origin variant's "origin_held" and "origin_open" (each parent's ties to their own
    parents). ``ages`` is `M11.C.2`'s (projection_until_age, load_from_age, load_to_age).
    """
    from src.bowen.readouts.patterns import PROJECTION_ACTS

    weeks = years * YEAR
    parents = (ROLES["phase_d", "parent_1"], ROLES["phase_d", "parent_2"])
    pairs, holding = (ARMS_C10[arm][0](), ARMS_C10[arm][1]) if arm in ARMS_C10 else ((), False)
    family = load_family(FAMILIES["phase_d"])
    children = tuple(sorted(p for p in family.people if set(family.parents_of(p)) == set(parents)))
    half = weeks // 2
    until, load_from, load_to = ages or (None, None, None)
    at_twenty, turned, loads, died = {}, {}, {c: 0.0 for c in children}, {}
    under, aged_load, aged_weeks = {}, {c: 0.0 for c in children}, {c: 0 for c in children}
    cut_weeks, child_weeks, held = [0], [0], [0]
    end = ENGINE.development_end_age_years

    def watch(state):
        for c in children:
            person = state.people[c]
            if not person.alive and c not in died:
                died[c] = state.tick
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
        if pairs:
            tested, reported = cut_off_tie_weeks(state, children)
            cut_weeks[0] += tested
            child_weeks[0] += reported
            held[0] += sum(state.tie_between(a, b).rejection.get(a, 0.0) > 0 for a, b in pairs)

    state, records = scenario(
        "phase_d", seed, weeks, spells=nodal(weeks, parents), watch=watch,
        setup=(_cut(pairs) if holding else _uncut(pairs)) if pairs else None,
        forced=_renewals(weeks, renew_every, pairs) if holding else None,
        held_open=(pairs, range(weeks)) if pairs and not holding else None)
    moves = [r.event for r in records if isinstance(r, EmittedRecord)]
    books = scripted_bookkeeping(records)  # a skipped renewal is counted, never silent (learning-qa 2026-10-14)
    if pairs:
        books[0]["(cutter-weeks holding rejection on the cutoffs the arm names)"] += held[0]

    def received(c, until_tick):
        return sum(1 for e in moves if e.kind in PROJECTION_ACTS and e.sender in parents and c in e.targets
                   and e.timestamp < until_tick)

    def observed(c):  # weeks under ``until`` the child was alive for (review 2026-10-14: a death shortens them)
        return min(under.get(c, weeks), died.get(c, weeks))

    return {
        "children": children,
        "development": at_twenty,
        "childhood_acts": {c: received(c, turned.get(c, weeks)) for c in children},
        "early_acts": {c: received(c, half) for c in children},
        "late_load": loads,
        # M11.C.2 by age: acts per year observed under ``until``; load over the ages read, where any week was.
        "acts_per_year": {c: received(c, observed(c)) / max(1, observed(c)) * YEAR for c in children},
        "aged_load": {c: aged_load[c] / aged_weeks[c] for c in children if aged_weeks[c]},
        "cut_weeks": float(cut_weeks[0]),
        "child_cut_weeks": float(child_weeks[0]),
        "books": books,
    }


def _pick(arm, acts: dict, values: dict) -> float:
    """The treatment reads the child who received the most, the baseline the fewest; a tie reads 0 in both."""
    ranked = sorted((c for c in acts if c in values), key=lambda c: acts[c])
    if not ranked or acts[ranked[0]] == acts[ranked[-1]]:
        return 0.0
    return values[ranked[-1] if arm == "treatment" else ranked[0]]


def _life_result(readouts: dict, summary: dict) -> ArmResult:
    # A copy: the summary is cached, and an arm that adds to its books must not change the next arm's (learning-qa).
    return ArmResult(readouts, *(Counter(c) for c in summary["books"]))


def arm_c6(arm, seed, settings):
    """Purpose: `M11.C.6` as restated at revision 32 (see the module docstring): one child's development with the
             parents' projection-type acts toward them as declared, against the same seed with those acts withheld.
    Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.6, #M7.E.1f
    Tests:   tests/bowen/test_phase_d2_criteria.py::test_m11c6_arms_differ_only_in_projection_toward_the_child

    A child who dies before 20 reads 0 in that arm and is counted."""
    from src.bowen.readouts.patterns import PROJECTION_ACTS

    child = PersonId(settings["child"])
    parents = (ROLES["phase_d", "parent_1"], ROLES["phase_d", "parent_2"])
    start = load_family(FAMILIES["phase_d"]).people[child].age_years
    end = ENGINE.development_end_age_years
    weeks = (end - start + 1) * YEAR
    withheld = None
    if arm == "baseline":
        labels = frozenset(f"{kind}>{child}" for kind in PROJECTION_ACTS)
        withheld = {t: {p: labels for p in parents} for t in range(weeks)}
    seen = {}

    def watch(state):
        person = state.people[child]
        if "development" not in seen and person.age_years is not None and person.age_years >= end:
            seen["development"] = person.basic_level - state.life.anchors[child]

    state, records = scenario("phase_d", seed, weeks, spells=nodal(weeks, parents), watch=watch, withheld=withheld)
    acts = sum(1 for r in records if isinstance(r, EmittedRecord) and r.event.kind in PROJECTION_ACTS
               and r.event.sender in parents and child in r.event.targets)
    out = result({"development": seen.get("development", 0.0), "projection_acts": float(acts)}, records)
    if "development" not in seen:
        out.moves["(the child died before 20)"] += 1
    return out


def arm_c2_twelve(arm, seed, settings):
    """M11.C.2 on the twelve-person family, read by age (see the module docstring)."""
    ages = settings["projection_until_age"], settings["load_from_age"], settings["load_to_age"]
    summary = life_run(seed, settings["years"], "declared", 0, ages)
    result_ = _life_result({"symptom_load": _pick(arm, summary["acts_per_year"], summary["aged_load"])}, summary)
    unread = len(summary["children"]) - len(summary["aged_load"])
    if unread:  # a child never seen at the ages read is left out of the pick; counted, not silent (review 2026-10-14)
        result_.moves["(children never seen at the ages read)"] += unread
    return result_


def arm_c10(arm, seed, settings):
    """M11.C.10 (see the module docstring)."""
    summary = life_run(seed, settings["years"], "open" if arm == "baseline" else "held", settings["renew_every"])
    return _life_result({"cut_off_tie_weeks": summary["cut_weeks"], "child_cut_weeks": summary["child_cut_weeks"]},
                        summary)


def arm_c10_origin(arm, seed, settings):
    """M11.C.10[family of origin], revision 30 (see the module docstring)."""
    summary = life_run(seed, settings["years"], "origin_open" if arm == "baseline" else "origin_held",
                       settings["renew_every"])
    return _life_result({"cut_off_tie_weeks": summary["cut_weeks"], "child_cut_weeks": summary["child_cut_weeks"]},
                        summary)


def arm_c47(arm, seed, settings):
    """Purpose: M11.C.47 (revision 30) — on the danger fixture, one spouse at a low level under repeated stressors
    (treatment) or at a higher level with none (baseline); read the danger the rest of the household feels from them and
    the principled part of its cutoff of them.
    Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.47, #M1.B.3e, #M4.D.1h
    Tests:   tests/bowen/test_origin_danger_focus.py::test_m11c47_arms_differ_only_in_one_persons_level_and_stress

    The arms differ only in the first spouse: their level, and the stressors they meet. ``felt_danger``: the mean over
    weeks of the largest danger anyone else in the household feels from them. ``principled_cutoff``: the mean over
    weeks of the largest product, over the same people, of the rejection held toward them and the danger felt from them,
    the principled part of that cutoff (`M1.B.3e`). A week with no rejection held adds 0, so an arm in which no one cuts
    them off reads 0: no principled cutoff, not a missing value. ``held_weeks``, reported only: the weeks in which anyone
    else at home holds rejection toward them, so a pass resting on a few seeds against a baseline with no cutoff shows
    (learning-qa, 2026-10-14).
    """
    weeks, low = settings["weeks"], arm == "treatment"
    person = PARENTS_DANGER[0]
    level = settings["level_low"] if low else settings["level_high"]

    def setup(state) -> None:
        p = state.people[person]
        p.basic_level = p.functional_level = level
        p.pseudo_self = p.swing
        _reinitialise(state)

    every = settings["stress_every"]
    spells = tuple(Event(id=EventId(t, "stress", 0), kind=settings["stress_kind"],
                         mechanism=Mechanism.EXOGENOUS_STRESSOR, sender=None, targets=(person,),
                         intensity=settings["stress_intensity"], timestamp=t, duration=1, exogenous=True,
                         source_position=SourcePosition.NONE, channel=Channel.EXOGENOUS)
                   for t in range(every, weeks, every)) if low else ()
    felt, principled, held = [], [], []

    def watch(state) -> None:
        household = state.people[person].household_id
        others = [o for o in state.people if o != person and state.people[o].household_id == household
                  and TieId.of(person, o) in state.ties]
        ties = [(o, state.ties[TieId.of(person, o)]) for o in others]
        felt.append(max((t.felt_danger.get(o, 0.0) for o, t in ties), default=0.0))
        principled.append(max((t.rejection.get(o, 0.0) * t.felt_danger.get(o, 0.0) for o, t in ties), default=0.0))
        held.append(any(t.rejection.get(o, 0.0) > 0.0 for o, t in ties))

    _, records = scenario("danger", seed, weeks, spells=spells, setup=setup, watch=watch)
    return result({"felt_danger": sum(felt) / len(felt), "principled_cutoff": sum(principled) / len(principled),
                   "held_weeks": sum(held)}, records)


C = Criterion
CRITERIA_D2: dict[str, Criterion] = {
    "M11.C.15": C("M11.C.15", "composite", ("baseline", "treatment"),
                  (Readout("death_rise", +1), Readout("recovery_rise", +1)), arm_c15, settings=SETTINGS["M11.C.15"]),
    "M11.C.6": C("M11.C.6", "composite", ("baseline", "treatment"),
                 (Readout("development", -1), Readout("projection_acts", +1, report_only=True)), arm_c6,
                 settings=SETTINGS["M11.C.6"]),
    "M11.C.10": C("M11.C.10", "composite", ("baseline", "treatment"),
                  (Readout("cut_off_tie_weeks", +1), Readout("child_cut_weeks", +1, report_only=True)), arm_c10,
                  ("CUTOFF",), settings=SETTINGS["M11.C.10"], seeds=SETTINGS["M11.C.10"]["seeds"]),
    "M11.C.10[family of origin]": C("M11.C.10", "composite", ("baseline", "treatment"),
                                    (Readout("cut_off_tie_weeks", +1), Readout("child_cut_weeks", +1, report_only=True)),
                                    arm_c10_origin, ("CUTOFF",), settings=SETTINGS["M11.C.10"],
                                    seeds=SETTINGS["M11.C.10"]["seeds"]),
    "M11.C.47": C("M11.C.47", "premise", ("baseline", "treatment"),
                  (Readout("felt_danger", +1), Readout("principled_cutoff", +1), Readout("held_weeks", +1, report_only=True)),
                  arm_c47, settings=SETTINGS["M11.C.47"]),
    "M11.C.2[twelve-person family]": C("M11.C.2", "composite", ("baseline", "treatment"),
                                       (Readout("symptom_load", +1),), arm_c2_twelve, settings=SETTINGS["M11.C.2"]),
}
