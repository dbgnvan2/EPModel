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
from src.bowen.engine.log_records import EffectRecord, EmittedRecord
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
    # The basic arm has no scripted support, so it reports 0: the runner's difference is then minus the borrowed
    # arm's share (report only).
    blocked = support_blocked(records) if arm == "baseline" else 0.0
    return result({"rise": _rise(state, "triad", (MOVER,)), "support_blocked": blocked}, records)


def support_blocked(records) -> float:
    """Report only (revision 21, owner decision 2026-10-12): the share of the partner's ``UNDERFUNCTION`` toward the
    mover that moved nothing because the mover held a position (`M1.B.5`). The borrowed arm's borrowing is what is
    left; this shows how much that is."""
    events = {r.event.id: r.event for r in records if isinstance(r, EmittedRecord)}
    support = [e.id for e in events.values() if e.kind == "UNDERFUNCTION" and e.sender == OTHER and MOVER in e.targets]
    held = {r.cause for r in records if isinstance(r, EffectRecord) and r.mechanism == "position_held"}
    return sum(1 for e in support if e in held) / len(support) if support else 0.0


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
    "M11.C.20": C("M11.C.20", "check", ("baseline", "treatment"),
                  (Readout("rise", +1), Readout("support_blocked", -1, report_only=True)), arm_c20,
                  ("UNDERFUNCTION", "I-POSITION"), settings=SETTINGS["M11.C.20"]),
    "M11.C.33": C("M11.C.33", "premise", ("baseline", "treatment"), (Readout("rise", +1),), arm_c33,
                  ("I-POSITION",), settings=SETTINGS["M11.C.33"]),
    "M11.C.17": C("M11.C.17", "composite", ("baseline", "treatment"),
                  (Readout("coached_rise", +1), Readout("family_rise", +1, report_only=True)), arm_c17,
                  settings=SETTINGS["M11.C.17"]),
}


# --- Phase D step 7: the symptom channels (B7, spec revision 24; Claude's recommendation, pending owner review) ---
#
# Every arm runs the nuclear-four fixture (config/bowen/fixtures/nuclear_four.md), except M11.C.12 on the dyad, as the
# plan's §3 has it. A held severity is a declared
# condition, set before every tick as M11.C.33's dependence is: driving a symptom by stressors does not reach a chosen
# severity (ad hoc, 2026-10-12: the person's usual level catches up and the engine's own episodes dominate).

from src.bowen.scenario.params import engine_params

ENGINE = engine_params(load_constants())
CHILD, PARTNER = P("c1"), P("c2")
FATHER, MOTHER = P("f"), P("m")


def _structural(mechanism: Mechanism, kind: str, t: int, *, targets=(), on_tie=None, binder=None, n=0) -> Event:
    return Event(id=EventId(t, kind.lower(), n), kind=kind, mechanism=mechanism, sender=None, targets=targets,
                 intensity=0.0, timestamp=t, duration=1, exogenous=True, source_position=SourcePosition.NONE,
                 channel=Channel.EXOGENOUS, on_tie=on_tie, binder=binder)


def removal(t: int, who: PersonId) -> Event:
    return _structural(Mechanism.INSTITUTIONALIZE, "INSTITUTIONALIZE", t, targets=(who,))


def returning(t: int, who: PersonId, family: str = "four") -> tuple[Event, ...]:
    """The removed member comes back: every one of their ties reconciled (`M4.A.3`)."""
    ties = sorted(tie.id for tie in load_family(FAMILIES[family]).ties.values() if who in tie.id.members())
    return tuple(_structural(Mechanism.RECONCILIATION, "RECONCILIATION", t, on_tie=tie, n=i)
                 for i, tie in enumerate(ties))


def remission(t: int, who: PersonId, channel) -> Event:
    from src.bowen.engine.events import BinderKind, BinderRef

    return _structural(Mechanism.BINDER_UNAVAILABLE, "BINDER_UNAVAILABLE", t,
                       binder=BinderRef(BinderKind.SYMPTOM_CHANNEL, f"{who.value}:{channel.value}"))


def held_symptom(who: PersonId, severity: float, until: int | None = None):
    """A ``before`` hook: ``who`` carries an active symptom of ``severity`` (load ÷ threshold) in their prior channel,
    until week ``until`` (exclusive)."""
    from src.bowen.engine.symptoms import threshold

    def hold(state):
        if until is not None and state.tick >= until:
            return
        person = state.people[who]
        person.symptom_load[person.channel_prior] = severity * threshold(person, ENGINE)
        person.symptom_active[person.channel_prior] = severity > 0
    return hold


def _mean_excess(state, who) -> float:
    from src.bowen.engine.contact import excess

    return sum(excess(state.people[p]) for p in who) / len(who)


def arm_c30(arm, seed, settings):
    """M11.C.30: C1's family focus is on the social channel from week 0 in both arms; in the treatment C1 also
    carries ``held_load`` of the threshold in the mental channel (below onset, so nothing else differs)."""
    from src.bowen.engine.objects import SymptomChannel
    from src.bowen.engine.symptoms import threshold

    weeks = settings["weeks"]

    def focus(state):
        person = state.people[CHILD]
        person.constitution_strength, person.channel_focus = 0.0, {SymptomChannel.SOCIAL: 1.0}

    def carry(state):
        if arm == "treatment":
            person = state.people[CHILD]
            person.symptom_load[SymptomChannel.MENTAL] = settings["held_load"] * threshold(person, ENGINE)

    social = []
    state, records = scenario("four", seed, weeks, spells=nodal(weeks, (CHILD,)), setup=focus, before=carry,
                              watch=lambda s: social.append(s.people[CHILD].symptom_load[SymptomChannel.SOCIAL]))
    return result({"social_load": sum(social) / len(social)}, records)


def _removal_effect(seed, at, after, severity):
    """The family's mean excess over ``after`` weeks from removal, removed minus kept, with C1 held at ``severity``."""
    means = {}
    for removed in (False, True):
        window = []
        state, records = scenario(
            "four", seed, at + after, spells=(removal(at, CHILD),) if removed else (),
            before=held_symptom(CHILD, severity),
            watch=lambda s: window.append(_mean_excess(s, (FATHER, MOTHER, PARTNER))) if s.tick > at else None)
        means[removed] = sum(window) / len(window)
    return means[True] - means[False], records


def arm_c22(arm, seed, settings):
    """M11.C.22 (restated at revision 24): the effect of removing C1 is read against removing C1 with no symptom. In
    the baseline both limbs are that control; in the treatment one limb holds C1 mildly symptomatic (at the peak),
    the other severely (past the turn)."""
    at, after = settings["removal_week"], settings["after"]
    if arm == "baseline":
        effect, records = _removal_effect(seed, at, after, 0.0)
        return result({"mild_removal_effect": effect, "severe_removal_effect": effect}, records)
    mild, _ = _removal_effect(seed, at, after, settings["mild"])
    severe, records = _removal_effect(seed, at, after, settings["severe"])
    return result({"mild_removal_effect": mild, "severe_removal_effect": severe}, records)


def _conflict_between(records, a, b, start) -> int:
    return sum(1 for r in records if isinstance(r, EmittedRecord) and r.event.kind == "CONFLICT"
               and r.event.timestamp >= start and r.event.sender in (a, b) and ({a, b} - {r.event.sender}) <= set(r.event.targets))


SPOUSE_A, SPOUSE_B = P("a"), P("b")  # the dyad fixture's spouses (config/bowen/fixtures/dyad.md)


def arm_c12(arm, seed, settings):
    """M11.C.12, on the dyad (plan §3): A carries a mild symptom (held at ``severity``) until ``remission_week``; in the
    treatment it remits then (``binder_unavailable`` on A's channel, the functioning balance untouched); in the
    baseline it continues. Read: the spouses' ``CONFLICT`` acts from the remission week on."""
    at, after = settings["remission_week"], settings["after"]
    channel = load_family(FAMILIES["dyad"]).people[SPOUSE_A].channel_prior
    if arm == "treatment":
        hold, spells = held_symptom(SPOUSE_A, settings["severity"], until=at), (remission(at, SPOUSE_A, channel),)
    else:
        hold, spells = held_symptom(SPOUSE_A, settings["severity"]), ()
    state, records = scenario("dyad", seed, at + after, spells=spells, before=hold)
    return result({"marital_conflict": float(_conflict_between(records, SPOUSE_A, SPOUSE_B, at))}, records)


def arm_c31(arm, seed, settings):
    """M11.C.31: the spouses' tie holds ``held`` of distance-bound anxiety in a calm family (no stressor); in the
    treatment a ``binder_unavailable`` names it at ``t0``. Read: the budget's rise from the start."""
    from src.bowen.engine.events import BinderKind, BinderRef
    from src.bowen.engine.identifiers import TieId

    tie = TieId.of(FATHER, MOTHER)

    def bind(state):
        state.ties[tie].distance_bound_anxiety = settings["held"]

    start = load_family(FAMILIES["four"]).family.undifferentiation_budget
    spells = ()
    if arm == "treatment":
        spells = (_structural(Mechanism.BINDER_UNAVAILABLE, "BINDER_UNAVAILABLE", settings["t0"],
                              binder=BinderRef(BinderKind.TIE_DISTANCE, f"{FATHER.value}~{MOTHER.value}")),)
    state, records = scenario("four", seed, settings["weeks"], spells=spells, setup=bind)
    return result({"budget_rise": state.family.undifferentiation_budget - start}, records)


def arm_c11(arm, seed, settings):
    """M11.C.11: C1 is removed at ``removal_week``. In the baseline C1 stays away; in the treatment C1 returns after
    ``away`` weeks. Each arm reports one disturbance: the family's mean excess over ``window`` weeks after the change
    minus over ``window`` weeks before it — removal in the baseline, return in the treatment."""
    at, away, window = settings["removal_week"], settings["away"], settings["window"]
    back = at + away
    spells = (removal(at, CHILD),) + (returning(back, CHILD) if arm == "treatment" else ())
    change = at if arm == "baseline" else back
    trace = {}
    # watch runs after each tick, when state.tick has moved on: key the trace by the week just run (review 2026-10-12:
    # keyed by state.tick, both windows sat one week early)
    state, records = scenario("four", seed, back + window, spells=spells,
                              watch=lambda s: trace.__setitem__(s.tick - 1, _mean_excess(s, (FATHER, MOTHER, PARTNER))))
    before = sum(trace[t] for t in range(change - window, change)) / window
    after = sum(trace[t] for t in range(change, change + window)) / window
    return result({"disturbance": after - before}, records)


CRITERIA_D.update({
    "M11.C.30": C("M11.C.30", "premise", ("baseline", "treatment"), (Readout("social_load", -1),), arm_c30,
                  settings=SETTINGS["M11.C.30"]),
    "M11.C.22": C("M11.C.22", "premise", ("baseline", "treatment"),
                  (Readout("mild_removal_effect", +1), Readout("severe_removal_effect", -1)), arm_c22,
                  settings=SETTINGS["M11.C.22"]),
    "M11.C.12": C("M11.C.12", "composite", ("baseline", "treatment"), (Readout("marital_conflict", +1),), arm_c12,
                  ("CONFLICT",), settings=SETTINGS["M11.C.12"]),
    "M11.C.31": C("M11.C.31", "check", ("baseline", "treatment"), (Readout("budget_rise", +1),), arm_c31,
                  settings=SETTINGS["M11.C.31"]),
    "M11.C.11": C("M11.C.11", "mixed", ("baseline", "treatment"), (Readout("disturbance", +1),), arm_c11,
                  settings=SETTINGS["M11.C.11"]),
})
