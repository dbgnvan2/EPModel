"""The Phase D estimator criteria as paired arms (Phase D plan step 6; spec revision 17).

Purpose: define `M11.C.18`, `M11.C.20`, `M11.C.33` and `M11.C.17` — the first criteria that read the basic-level
         estimator (`M1.A.4k`) — each as two arms, under a declared nodal load of three kinds so that every estimator
         window can meet its breadth.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.17, #M11.C.18, #M11.C.20, #M11.C.33, #M1.A.4k, #M7.A.1a, #M17.D.3
Tests:   tests/bowen/test_phase_d_criteria.py

Every number an arm uses is in ``config/bowen/criteria.md`` and every person's role in
``config/bowen/criterion_roles.md``; the arm designs below are `[I]`. Two were restated after ad hoc ensemble runs, by
owner decision at spec revision 19 (2026-10-11), and are not pre-registered: `M11.C.20` now differs in systems
perspective as well as the scripted act, and `M11.C.17` runs ten years and reads the coached people.
The readout of each is the rise in estimated ``basic_level`` over the run (``M1.A.4a``: nothing else writes it).

- **`M11.C.18`, the binder against the real thing.** F, with systems perspective, is brought to the same start in
  both arms; the baseline arm scripts ``CUTOFF`` toward M every ``act_every`` weeks, the treatment ``I-POSITION``.
  The I-POSITION arm's rise MUST be larger.
- **`M11.C.20`, a borrowed gain against a basic one.** The baseline's gain is borrowed: M gives way to F
  (``UNDERFUNCTION``, which moves functioning to F) for ``support_weeks``, then the support is withdrawn. The
  treatment's is F's own: ``I-POSITION`` sequences over the same weeks, with systems perspective (only in this arm:
  owner decision 2026-10-11). The basic arm's rise MUST be larger.
- **`M11.C.33`, the dependence gate on the slow clock.** F completes the same scripted sequences in both arms. In the
  baseline F is financially dependent on each slow-tick week only, and in both arms F's ``I-POSITION`` is withheld on
  those weeks, so the week-to-week gate (``M5.C.1``) changes nothing and the arms differ only in what the slow tick
  reads (``M7.A.1a``). The independent arm's rise MUST be larger.
- **`M11.C.17`, a coach helps and is not required** (rewritten at revision 17). The Phase C family over ten years;
  the coach-free arm has the coach's ties made non-interactive from week 0. The readout is the mean rise of the
  people the coach is in contact with (owner decision 2026-10-11: coaching lands slowly and reaches only them); the
  whole family's mean is reported. The coached people's rise MUST be larger. That a coach-free rise needs self-directed behaviour under load is the estimator's own rule
  (``M1.A.4h``), proved by ``tests/bowen/test_slow_tick.py::test_m1a4h_carrying_load_counts_and_discharging_it_does_not``.
"""

from __future__ import annotations

from src.bowen.engine.act import Selection
from src.bowen.engine.events import Channel, Event, EventId, Mechanism, SourcePosition
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.objects import Role
from src.bowen.ensemble.criteria import FAMILIES, ROLES, SETTINGS, result, scenario
from src.bowen.ensemble.runner import Criterion, Readout
from src.bowen.io.load import load_constants, load_family

P = PersonId
NODAL = SETTINGS["nodal"]
MOVER, OTHER = P("f"), P("m")  # the triad fixture's spouses (config/bowen/fixtures/triad.md)


def nodal(weeks: int, targets) -> tuple[Event, ...]:
    """The declared nodal load: every ``every`` weeks, one event to each target, cycling through the three kinds."""
    kinds = (NODAL["kind_1"], NODAL["kind_2"], NODAL["kind_3"])
    every = NODAL["every"]
    return tuple(
        Event(id=EventId(t, "nodal", i), kind=kinds[(t // every) % len(kinds)], mechanism=Mechanism.EXOGENOUS_STRESSOR,
              sender=None, targets=(person,), intensity=NODAL["intensity"], timestamp=t, duration=1, exogenous=True,
              source_position=SourcePosition.NONE, channel=Channel.EXOGENOUS)
        for t in range(every, weeks, every) for i, person in enumerate(targets))


def _scripted(actor, kind, target, weeks, every, skip=lambda t: False) -> dict:
    return {(t, actor): Selection(actor=actor, kind=kind, targets=(target,), intensity=SETTINGS["scripted_act"]["intensity"])
            for t in range(0, weeks, every) if not skip(t)}


def _with_perspective(state) -> None:
    state.people[MOVER].systems_perspective = 1.0


def _rise(state, family: str, who) -> float:
    start = load_family(FAMILIES[family]).people
    return sum(state.people[p].basic_level - start[p].basic_level for p in who) / len(who)


def arm_c18(arm, seed, settings):
    weeks = settings["weeks"]
    kind = "CUTOFF" if arm == "baseline" else "I-POSITION"
    state, records = scenario("triad", seed, weeks, spells=nodal(weeks, (MOVER,)), setup=_with_perspective,
                              forced=_scripted(MOVER, kind, OTHER, weeks, settings["act_every"]))
    return result({"rise": _rise(state, "triad", (MOVER,))}, records)


def arm_c20(arm, seed, settings):
    weeks, support = settings["weeks"], settings["support_weeks"]
    if arm == "baseline":  # borrowed: the partner gives way, then stops
        forced = _scripted(OTHER, "UNDERFUNCTION", MOVER, support, settings["support_every"])
    else:                  # basic: the mover's own differentiating sequences over the same weeks
        forced = _scripted(MOVER, "I-POSITION", OTHER, support, settings["act_every"])
    # Owner decision 2026-10-11: perspective only in the basic arm, where it makes the sequences differentiating. In
    # the borrowed arm the mover simply leans on the partner; with perspective it coped under load and built on the
    # support, which is part differentiation (the borrowed arm then rose more, −8.1 ± 0.7).
    state, records = scenario("triad", seed, weeks, spells=nodal(weeks, (MOVER,)),
                              setup=_with_perspective if arm == "treatment" else None, forced=forced)
    return result({"rise": _rise(state, "triad", (MOVER,))}, records)


YEAR = load_constants()["slow_tick_fast_ticks"]


def _slow_tick_week(t: int) -> bool:
    return (t + 1) % YEAR == 0


def c33_inputs(arm, weeks, act_every) -> dict:
    """The scenario arguments of one M11.C.33 arm over ``weeks`` (shared with its test)."""

    def dependent_at_the_slow_tick(state):
        state.people[MOVER].financially_dependent = arm == "baseline" and _slow_tick_week(state.tick)

    forced = _scripted(MOVER, "I-POSITION", OTHER, weeks, act_every,
                       skip=_slow_tick_week)  # never on a slow-tick week, so M5.C.1 never blocks it
    # On slow-tick weeks the dependent mover cannot choose I-POSITION (M5.C.1); withhold it in both arms that week,
    # so the legal sets match and the arms differ only in what the slow tick reads.
    withheld = {t: {MOVER: frozenset({"I-POSITION"})} for t in range(weeks) if _slow_tick_week(t)}
    return dict(spells=nodal(weeks, (MOVER,)), setup=_with_perspective, forced=forced,
                before=dependent_at_the_slow_tick, withheld=withheld)


def arm_c33(arm, seed, settings):
    weeks = settings["weeks"]
    state, records = scenario("triad", seed, weeks, **c33_inputs(arm, weeks, settings["act_every"]))
    return result({"rise": _rise(state, "triad", (MOVER,))}, records)


def arm_c17(arm, seed, settings):
    weeks = settings["weeks"]
    coach = ROLES["phase_c", "coach"]

    def coach_free(state):
        if arm == "baseline":
            for tie in state.ties_of(coach):
                tie.interactive = False

    family = load_family(FAMILIES["phase_c"])
    members = tuple(sorted(p for p, person in family.people.items() if person.role is Role.MEMBER))
    # Every member carries the load (review 2026-10-11: with only the parents loaded, the members in other
    # households could never meet the estimator's breadth, and the mean was diluted by them in both arms).
    coached = tuple(sorted(tie.id.other(coach) for tie in family.ties.values() if coach in tie.id.members()))
    state, records = scenario("phase_c", seed, weeks, spells=nodal(weeks, members), setup=coach_free)
    return result({"coached_rise": _rise(state, "phase_c", coached),
                   "family_rise": _rise(state, "phase_c", members)}, records)


C = Criterion
CRITERIA_D: dict[str, Criterion] = {
    "M11.C.18": C("M11.C.18", "check", ("baseline", "treatment"), (Readout("rise", +1),), arm_c18,
                  ("CUTOFF", "I-POSITION"), settings=SETTINGS["M11.C.18"]),
    "M11.C.20": C("M11.C.20", "check", ("baseline", "treatment"), (Readout("rise", +1),), arm_c20,
                  ("UNDERFUNCTION", "I-POSITION"), settings=SETTINGS["M11.C.20"]),
    "M11.C.33": C("M11.C.33", "premise", ("baseline", "treatment"), (Readout("rise", +1),), arm_c33,
                  ("I-POSITION",), settings=SETTINGS["M11.C.33"]),
    "M11.C.17": C("M11.C.17", "composite", ("baseline", "treatment"),
                  (Readout("coached_rise", +1), Readout("family_rise", +1, report_only=True)), arm_c17,
                  settings=SETTINGS["M11.C.17"]),
}
