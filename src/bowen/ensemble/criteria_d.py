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

from src.bowen.engine.act import WITHHOLD, Selection
from src.bowen.engine.events import Channel, Event, EventId, Mechanism, SourcePosition
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.log_records import EffectRecord, EmittedRecord
from src.bowen.engine.objects import Role
from src.bowen.ensemble.criteria import FAMILIES, ROLES, SETTINGS, result, scenario
from src.bowen.ensemble.runner import Criterion, Readout
from src.bowen.io.load import load_constants, load_event_kinds, load_family

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


# --- Phase D step 7: the symptom channels (B7, spec revision 24; Claude's recommendation, approved 2026-10-16) ---
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


# --- Phase D step 8: beliefs as a channel (B8, spec revision 25; Claude's recommendation, approved 2026-10-16) ---

DRIVINGS = ("conflict", "dysfunction", "projection")


def _driven(kind_of: str, weeks: int, every: int) -> dict:
    """The scripted acts that drive the nuclear four toward one sink: the spouses fight; M gives way and F takes
    over; or both parents focus on C1."""
    if kind_of == "conflict":
        return {**_scripted(FATHER, "CONFLICT", MOTHER, weeks, every), **_scripted(MOTHER, "CONFLICT", FATHER, weeks, every)}
    if kind_of == "dysfunction":
        return {**_scripted(MOTHER, "UNDERFUNCTION", FATHER, weeks, every),
                **_scripted(FATHER, "OVERFUNCTION", MOTHER, weeks, every)}
    return {**_scripted(FATHER, "OVERFUNCTION", CHILD, weeks, every), **_scripted(MOTHER, "PURSUE", CHILD, weeks, every)}


def arm_c26(arm, seed, settings):
    """M11.C.26: the nuclear four is driven toward each sink in turn (``weeks``, a scripted act every ``act_every``).
    Read from the belief store only, the spouses' attributions name a sink, and the sink allocation names one; the
    treatment compares each driving's readings with that driving, the baseline with the next driving's (a matched
    control: the same runs, the labels rotated). Readouts: the share of drivings whose sink the beliefs recover, and
    the share whose belief pattern the sink allocation recovers."""
    from src.bowen.readouts.attribution import sink_from_allocations, sink_from_beliefs
    from src.bowen.engine.objects import Sink

    weeks, every = settings["weeks"], settings["act_every"]
    names = {"conflict": Sink.MARITAL_CONFLICT, "dysfunction": Sink.SPOUSE_DYSFUNCTION, "projection": Sink.CHILD_PROJECTION}
    children = frozenset({CHILD, PARTNER})
    read_beliefs, read_sinks, records = [], [], []
    for driving in DRIVINGS:
        state, records = scenario("four", seed, weeks, forced=_driven(driving, weeks, every))
        attributions = {p: dict(state.people[p].attribution) for p in (FATHER, MOTHER)}
        read_beliefs.append(sink_from_beliefs(attributions, (FATHER, MOTHER), children))
        read_sinks.append(sink_from_allocations(dict(state.family.sink_allocations)))
    beliefs, sinks = score_c26(read_beliefs, read_sinks, [names[d] for d in DRIVINGS], 0 if arm == "treatment" else 1)
    # Review 2026-10-12: here the largest allocation always names the driving's own sink, so reading the belief
    # pattern from the allocation is the same comparison; it is reported, not tested.
    return result({"belief_to_sink": beliefs, "sink_to_belief": sinks}, records)


def score_c26(read_beliefs, read_sinks, driven, shift: int) -> tuple[float, float]:
    """The shares of drivings whose sink the beliefs name, and whose belief pattern the sink allocation names, each
    read against the driving ``shift`` places on (0: its own; 1: the next one's, the label-rotated control)."""
    n = len(driven)
    truth = [driven[(i + shift) % n] for i in range(n)]
    rotated = list(read_sinks[shift:]) + list(read_sinks[:shift])
    return (sum(b is not None and b == t for b, t in zip(read_beliefs, truth)) / n,
            sum(b is not None and b == s for b, s in zip(read_beliefs, rotated)) / n)


def arm_c36(arm, seed, settings):
    """M11.C.36: on the triad, F and M exchange ``CONFLICT`` every ``act_every`` weeks, witnessed by C. The arms
    differ only in C's belief about the F–M tie at the start: in the treatment C believes it tense
    (``believed_tension``). Read: C's appraisal of the pair's events over the first ``weeks`` weeks."""
    from src.bowen.engine.identifiers import TieId

    weeks, every, child = settings["weeks"], settings["act_every"], P("c")
    forced = {**_scripted(OTHER, "CONFLICT", MOVER, weeks, every), **_scripted(MOVER, "CONFLICT", OTHER, weeks, every)}

    def believe(state):
        if arm == "treatment":
            state.people[child].tie_beliefs[TieId.of(MOVER, OTHER)].tension = settings["believed_tension"]

    state, records = scenario("triad", seed, weeks, setup=believe, forced=forced)
    pair = {r.event.id for r in records if isinstance(r, EmittedRecord) and r.event.sender in (MOVER, OTHER)
            and set(r.event.targets) <= {MOVER, OTHER}}
    response = sum(v for r in records if isinstance(r, EffectRecord) and r.mechanism == "appraisal"
                   and r.cause in pair for p, v in r.acute_anxiety if p == child)
    return result({"witness_response": response}, records)


# M11.C.14, restated at revision 31 (the owner, 2026-10-16, review option B): each technique against no technique.
C14_TECHNIQUES = {"firmness": "OVERFUNCTION", "talking it out": "STAY-IN-CONTACT"}


def _c14_run(seed, weeks, every, severity, kind):
    """C1's mean symptom load over ``weeks`` on the nuclear four, the spouses' tie held at distance (felt contact 0 for
    both, each week), C1 starting with an active symptom (``severity`` of the threshold); every ``every`` weeks both
    parents send C1 ``kind``, or, when ``kind`` is None, hold back (``WITHHOLD``, no target). The owner, 2026-10-16
    (option a): the baseline holds back on the same weeks, so both arms lose the same weeks of the parents' own acts
    (a scripted act is the whole week's act). A hold is coping in the model (`M1.A.5e`), as talking it out is."""
    from src.bowen.engine.identifiers import TieId
    from src.bowen.engine.symptoms import threshold

    if kind is None:
        forced = {(t, parent): Selection(actor=parent, kind=WITHHOLD, targets=(), intensity=0.0)
                  for t in range(0, weeks, every) for parent in (FATHER, MOTHER)}
    else:
        forced = {**_scripted(FATHER, kind, CHILD, weeks, every), **_scripted(MOTHER, kind, CHILD, weeks, every)}
    marital = TieId.of(FATHER, MOTHER)

    def start(state):
        person = state.people[CHILD]
        person.symptom_load[person.channel_prior] = severity * threshold(person, ENGINE)
        person.symptom_active[person.channel_prior] = True

    def distant(state):
        for m in marital.members():
            state.ties[marital].felt_contact[m] = 0.0

    loads = []
    state, records = scenario("four", seed, weeks, setup=start, before=distant, forced=forced,
                              watch=lambda s: loads.append(sum(s.people[CHILD].symptom_load.values())))
    return sum(loads) / len(loads), records


def arm_c14(arm, seed, settings):
    """M11.C.14[technique], revision 31: the baseline has both parents hold back on the technique's weeks; the
    treatment has both parents use the
    variant's technique toward C1 (firmness ``OVERFUNCTION``, talking it out ``STAY-IN-CONTACT``). Read: C1's mean
    symptom load, one-sided (no technique lowers it by more than the equivalence bound, `M11.4`). The talking-it-out
    variant also reports the former comparison, talking it out against firmness (``talking_less_firmness``): its
    baseline reads firmness's load there, so the difference is talking it out's load less firmness's. Reported, never
    tested."""
    technique, given = settings["technique"], (settings["weeks"], settings["act_every"], settings["start_severity"])
    firmness = None
    if technique == "talking it out" and arm == "baseline":
        # Run first: ``result`` books the scripted acts of the last scenario run, which must be this arm's own.
        firmness, _ = _c14_run(seed, *given, C14_TECHNIQUES["firmness"])
    load, records = _c14_run(seed, *given, None if arm == "baseline" else C14_TECHNIQUES[technique])
    readouts = {"child_symptom_load": load}
    if technique == "talking it out":
        readouts["talking_less_firmness"] = load if arm == "treatment" else firmness
    return result(readouts, records)


CRITERIA_D.update({
    "M11.C.26": C("M11.C.26", "premise", ("baseline", "treatment"),
                  (Readout("belief_to_sink", +1), Readout("sink_to_belief", +1, report_only=True)), arm_c26,
                  settings=SETTINGS["M11.C.26"]),
    "M11.C.36": C("M11.C.36", "check", ("baseline", "treatment"), (Readout("witness_response", +1),), arm_c36,
                  ("CONFLICT",), settings=SETTINGS["M11.C.36"]),
})
for _technique, _kind in C14_TECHNIQUES.items():
    _cid = f"M11.C.14[{_technique}]"
    _readouts = (Readout("child_symptom_load", 0, bound_side=+1),)
    if _technique == "talking it out":
        _readouts += (Readout("talking_less_firmness", 0, report_only=True),)
    CRITERIA_D[_cid] = C(_cid, "premise", ("baseline", "treatment"), _readouts, arm_c14, (_kind,),
                         settings={**SETTINGS["M11.C.14"], "technique": _technique})


# --- Phase D step 9: projection in the model (B9, spec revision 26; Claude's recommendation, approved 2026-10-16) ---


def arm_c2(arm, seed, settings):
    """M11.C.2, a composite: the Phase C family under the policy for ``weeks``, with the declared nodal load on the
    parents. Each run counts the projection-type acts (``PROJECTION_ACTS``) each child received from the parents and
    each child's mean symptom load. The acts are counted in the first half of the run and the load read in the second
    (review 2026-10-12: read over the same weeks, load could attract acts as well as follow them). The treatment
    reports the later load of the child who received the most; the baseline, from the same run, that of the child who
    received the fewest (a within-run control). Ties report 0 in both arms."""
    from src.bowen.readouts.patterns import PROJECTION_ACTS

    weeks = settings["weeks"]
    family = load_family(FAMILIES["phase_c"])
    parents = (ROLES["phase_c", "parent_1"], ROLES["phase_c", "parent_2"])
    children = tuple(sorted({m for tie in family.family.parent_child_ties for m in tie.members()} - set(parents)))
    loads = {c: 0.0 for c in children}
    half = weeks // 2

    def watch(state):
        if state.tick > half:  # the end of weeks half .. weeks - 1
            for c in children:
                loads[c] += sum(state.people[c].symptom_load.values()) / (weeks - half)

    state, records = scenario("phase_c", seed, weeks, spells=nodal(weeks, parents), watch=watch)
    received = {c: sum(1 for r in records if isinstance(r, EmittedRecord) and r.event.kind in PROJECTION_ACTS
                       and r.event.sender in parents and c in r.event.targets and r.event.timestamp < half)
                for c in children}
    ranked = sorted(children, key=lambda c: received[c])
    if received[ranked[0]] == received[ranked[-1]]:
        return result({"symptom_load": 0.0}, records)
    pick = ranked[-1] if arm == "treatment" else ranked[0]
    return result({"symptom_load": loads[pick]}, records)


CRITERIA_D.update({
    "M11.C.2": C("M11.C.2", "composite", ("baseline", "treatment"), (Readout("symptom_load", +1),), arm_c2,
                 settings=SETTINGS["M11.C.2"]),
})


# --- Phase D step 10, criteria on existing mechanisms (Claude's recommendation, approved 2026-10-16) ---

# M10.B.5: an override declared by the criterion, never a config default. Revision 29 adds the second way the automatic
# channel learns, from what follows a witnessed cutoff (M4.D.6f), so learning off stops both.
LEARNING_OFF = {"learning_rate": 0.0, "witnessed_cutoff_gain": 0.0}


def _spell_on(who, start: int, stop: int, every: int, intensity: float) -> tuple[Event, ...]:
    """A declared exogenous spell: one stressor event to each of ``who`` every ``every`` weeks from start to stop."""
    return tuple(
        Event(id=EventId(t, "spell", i), kind=NODAL["kind_3"], mechanism=Mechanism.EXOGENOUS_STRESSOR, sender=None,
              targets=(p,), intensity=intensity, timestamp=t, duration=1, exogenous=True,
              source_position=SourcePosition.NONE, channel=Channel.EXOGENOUS)
        for t in range(start, stop, every) for i, p in enumerate(who))


def _acts(records, kinds, senders, start, stop) -> int:
    return sum(1 for r in records if isinstance(r, EmittedRecord) and r.event.kind in kinds
               and r.event.sender in senders and start <= r.event.timestamp < stop)


def arm_c39(arm, seed, settings):
    """M11.C.39: on the triad, a heavy spell on both spouses for ``spell_weeks``; then ``after`` calm weeks. The
    baseline has learning switched off (`M10.B.5`). Read: the spouses' ``DISTANCE`` and ``CUTOFF`` acts in the calm
    weeks (the pattern's persistence); reported, the first calm week after which the spouses make none for
    ``quiet`` weeks (time to remission)."""
    spell_weeks, after, quiet = settings["spell_weeks"], settings["after"], settings["quiet"]
    spouses = (MOVER, OTHER)
    state, records = scenario("triad", seed, spell_weeks + after,
                              spells=_spell_on(spouses, 0, spell_weeks, settings["every"], settings["intensity"]),
                              overrides=LEARNING_OFF if arm == "baseline" else None)
    persisted = _acts(records, ("DISTANCE", "CUTOFF"), spouses, spell_weeks, spell_weeks + after)
    weekly = [_acts(records, ("DISTANCE", "CUTOFF"), spouses, t, t + 1) for t in range(spell_weeks, spell_weeks + after)]
    remission = next((i for i in range(len(weekly)) if not any(weekly[i:i + quiet])), after)
    return result({"pattern_after_spell": float(persisted), "time_to_remission": float(remission)}, records)


def arm_c40(arm, seed, settings):
    """M11.C.40: the Phase C family under its declared spell for ``weeks``; the baseline with learning off. Read: the
    fall in automatic-channel repertoire entropy (`M16.A.9`) from the first ``window`` weeks to the last; reported,
    the same fall for the self-directed channel."""
    from src.bowen.ensemble.criteria import spell

    weeks, window = settings["weeks"], settings["window"]
    state, records = scenario("phase_c", seed, weeks, spells=spell("phase_c", weeks),
                              overrides=LEARNING_OFF if arm == "baseline" else None)
    windows = [_entropy(records, kinds, a, b) for kinds in (AUTOMATIC_KINDS, SELF_KINDS)
               for a, b in ((0, window), (weeks - window, weeks))]
    # Re-sweep 2026-10-12: a window with no acts has no entropy; a channel with one is read as no fall, and counted.
    automatic = windows[0] - windows[1] if None not in windows[:2] else 0.0
    self_ = windows[2] - windows[3] if None not in windows[2:] else 0.0
    # Clause 2 as a readout (review 2026-10-12): its between-arm difference is the automatic channel's difference less
    # the self-directed channel's, which MUST be positive.
    return result({"automatic_entropy_fall": automatic, "automatic_less_self": automatic - self_,
                   "self_entropy_fall": self_, "empty_windows": float(windows.count(None))}, records)


def _entropy(records, kinds, start, stop) -> float:
    import math
    from collections import Counter

    counts = Counter(r.event.kind for r in records if isinstance(r, EmittedRecord)
                     and r.event.mechanism is Mechanism.MOVE and r.event.kind in kinds and start <= r.event.timestamp < stop)
    total = sum(counts.values())
    if total == 0:
        return None
    if len(kinds) < 2:
        return 0.0
    return -sum(n / total * math.log(n / total) for n in counts.values()) / math.log(len(kinds))


AUTOMATIC_KINDS = load_event_kinds().in_channel("automatic")
SELF_KINDS = load_event_kinds().in_channel("self")


def arm_c43(arm, seed, settings):
    """M11.C.43: on the triad, F carries a spell for ``spell_weeks`` while M, calmer, is available (treatment) or
    absent from F's choices (baseline); then ``gap`` quiet weeks; then M becomes unavailable to F for ``window``
    weeks. Read: the rise in F's mean acute anxiety over that window against the ``window`` weeks before it."""
    spell_weeks, gap, window = settings["spell_weeks"], settings["gap"], settings["window"]
    lost = spell_weeks + gap
    weeks = lost + window
    absent = {t: ((MOVER,), OTHER) for t in range(lost, weeks)}
    if arm == "baseline":
        absent.update({t: ((MOVER,), OTHER) for t in range(spell_weeks)})
    trace = {}
    state, records = scenario("triad", seed, weeks, unavailable=absent,
                              spells=_spell_on((MOVER,), 0, spell_weeks, settings["every"], settings["intensity"]),
                              watch=lambda s: trace.__setitem__(s.tick - 1, s.people[MOVER].acute_anxiety))
    before = sum(trace[t] for t in range(lost - window, lost)) / window
    after = sum(trace[t] for t in range(lost, weeks)) / window
    return result({"loss_rise": after - before}, records)


def arm_c34(arm, seed, settings):
    """M11.C.34: the nuclear four at two family levels (``level_low`` below the fixture in the baseline,
    ``level_high`` above it in the treatment), the same symptom onset in C1 at week ``t0``. Read: the other members'
    automatic acts addressed to C1 in the ``window`` weeks from onset, less the same count in the same run with no
    onset (review 2026-10-12: counting all their acts measured the level, not the reaction)."""
    from src.bowen.ensemble.criteria import lowered
    from src.bowen.engine.symptoms import threshold

    t0, window = settings["t0"], settings["window"]
    by = settings["level_low"] if arm == "baseline" else -settings["level_high"]

    def onset(state):
        # Load set so that after this week's leak it still crosses the threshold, with the symptom inactive: the onset
        # fires at step 9 of week t0, as an endogenous event to C1's ties (review 2026-10-12: holding it active
        # skipped the onset, and no one reacted to one).
        if state.tick == t0:
            person = state.people[CHILD]
            for c in person.symptom_active:
                person.symptom_active[c] = False
            person.symptom_load[person.channel_prior] = (settings["severity"] * threshold(person, ENGINE)
                                                         / (1.0 - ENGINE.symptom_leak_rate))

    others = {FATHER, MOTHER, PARTNER}

    def toward_child(records):
        return sum(1 for r in records if isinstance(r, EmittedRecord) and r.event.mechanism is Mechanism.MOVE
                   and r.event.sender in others and CHILD in r.event.targets and r.event.kind in AUTOMATIC_KINDS
                   and t0 <= r.event.timestamp < t0 + window)

    _, quiet = scenario("four", seed, t0 + window, setup=lowered(by))
    state, records = scenario("four", seed, t0 + window, setup=lowered(by), before=onset)
    return result({"automatic_reaction": float(toward_child(records) - toward_child(quiet))}, records)


def arm_c21(arm, seed, settings):
    """M11.C.21: the nuclear four. In both arms both parents over-function toward C1 every ``act_every`` weeks for the
    whole run: they are projecting parents, and C1 has lent functioning when they collapse (the owner, 2026-10-16: build
    the lent balance first). At week ``t0`` both parents' functional level drops by ``drop`` in the treatment (none in the
    baseline), C1 untouched. Read: C1's functional level at the end of the run less at the end of week ``t0`` (the
    week of the drop)."""
    t0, after, drop, every = settings["t0"], settings["after"], settings["drop"], settings["act_every"]
    weeks = t0 + after
    forced = {**_scripted(FATHER, "OVERFUNCTION", CHILD, weeks, every),
              **_scripted(MOTHER, "OVERFUNCTION", CHILD, weeks, every)}

    def collapse(state):
        if arm == "treatment" and state.tick == t0:
            for parent in (FATHER, MOTHER):
                person = state.people[parent]
                person.functional_level = max(0.0, person.functional_level - drop)
                person.stress_swing -= drop  # the fall is stress's, in its ledger (revision 20: one ledger each)

    at = {}
    state, records = scenario("four", seed, weeks, before=collapse, forced=forced,
                              watch=lambda s: at.setdefault("t0", s.people[CHILD].functional_level) if s.tick == t0 + 1 else None)
    return result({"target_rise": state.people[CHILD].functional_level - at["t0"]}, records)


def arm_c23(arm, seed, settings):
    """M11.C.23 (revision 15's definitions): C1 starts with an active symptom (``start_severity``) that remits at
    ``t0``. Both parents act toward C1 every ``act_every`` weeks: ``CONFLICT`` (hostility and blame) in the
    treatment, which also suppresses C1's symptom accumulation by ``suppression`` (medication); ``STAY-IN-CONTACT``
    in the baseline, with no suppression. Read: C1's relapses, onsets after the remission."""
    from src.bowen.engine.objects import SymptomChannel
    from src.bowen.engine.symptoms import threshold

    t0, weeks, every = settings["t0"], settings["weeks"], settings["act_every"]
    kind = "CONFLICT" if arm == "treatment" else "STAY-IN-CONTACT"
    forced = {**_scripted(FATHER, kind, CHILD, weeks, every), **_scripted(MOTHER, kind, CHILD, weeks, every)}
    channel = load_family(FAMILIES["four"]).people[CHILD].channel_prior

    def start(state):
        person = state.people[CHILD]
        person.symptom_load[channel] = settings["start_severity"] * threshold(person, ENGINE)
        person.symptom_active[channel] = True
        if arm == "treatment":
            person.symptom_suppression = settings["suppression"]

    state, records = scenario("four", seed, weeks, setup=start, forced=forced, spells=(remission(t0, CHILD, channel),))
    relapses = sum(1 for r in records if isinstance(r, EffectRecord) and r.mechanism == "symptom_onset"
                   and r.tick > t0 and r.cause is not None and r.cause.origin == f"symptom:{CHILD.value}")
    return result({"relapses": float(relapses)}, records)


CRITERIA_D.update({
    "M11.C.39": C("M11.C.39", "composite", ("baseline", "treatment"),
                  (Readout("pattern_after_spell", +1), Readout("time_to_remission", +1, report_only=True)), arm_c39,
                  settings=SETTINGS["M11.C.39"]),
    "M11.C.40": C("M11.C.40", "premise", ("baseline", "treatment"),
                  # Clause 2 is reported, not tested (the owner, 2026-10-16, option a): the entropy readout cannot see
                  # M4.D.6d removed in this fixture, so tests/bowen/test_learner.py proves it directly.
                  (Readout("automatic_entropy_fall", +1), Readout("automatic_less_self", +1, report_only=True),
                   Readout("self_entropy_fall", 0, report_only=True), Readout("empty_windows", 0, report_only=True)),
                  arm_c40,
                  settings=SETTINGS["M11.C.40"]),
    "M11.C.43": C("M11.C.43", "composite", ("baseline", "treatment"), (Readout("loss_rise", +1),), arm_c43,
                  settings=SETTINGS["M11.C.43"]),
    "M11.C.34": C("M11.C.34", "mixed", ("baseline", "treatment"), (Readout("automatic_reaction", -1),), arm_c34,
                  settings=SETTINGS["M11.C.34"]),
    # Reclassified as a composite (the owner, 2026-10-17): its failure is accepted as a finding (M11.5).
    "M11.C.21": C("M11.C.21", "composite", ("baseline", "treatment"), (Readout("target_rise", +1),), arm_c21,
                  settings=SETTINGS["M11.C.21"]),
    # Reclassified as a composite (the owner, 2026-10-16): it compares the sizes of two mechanisms (M11.5).
    "M11.C.23": C("M11.C.23", "composite", ("baseline", "treatment"), (Readout("relapses", +1),), arm_c23,
                  ("CONFLICT", "STAY-IN-CONTACT"), settings=SETTINGS["M11.C.23"]),
})


def arm_c24(arm, seed, settings):
    """M11.C.24 (with revision 27): on the dyad, A takes charge and B gives way every ``act_every`` weeks until
    ``diverge_weeks``, so their functional levels diverge from a matched basic level. Then, in the treatment, the tie
    is dissolved (non-interactive from that week on); in the baseline the exchange goes on. Read, over ``after`` weeks
    from the dissolution: B's functional level change (MUST rise) and A's (MUST fall)."""
    from src.bowen.engine.identifiers import TieId

    split, after, every = settings["diverge_weeks"], settings["after"], settings["act_every"]
    weeks = split + after
    span = weeks if arm == "baseline" else split
    forced = {**_scripted(SPOUSE_A, "OVERFUNCTION", SPOUSE_B, span, every),
              **_scripted(SPOUSE_B, "UNDERFUNCTION", SPOUSE_A, span, every)}
    tie = TieId.of(SPOUSE_A, SPOUSE_B)

    def dissolve(state):
        if arm == "treatment" and state.tick >= split:
            state.ties[tie].interactive = False

    at = {}

    def watch(state):
        if state.tick == split:  # the end of the last week before dissolution
            at.update({p: state.people[p].functional_level for p in (SPOUSE_A, SPOUSE_B)})

    state, records = scenario("dyad", seed, weeks, before=dissolve, forced=forced, watch=watch)
    return result({"under_change": state.people[SPOUSE_B].functional_level - at[SPOUSE_B],
                   "over_change": state.people[SPOUSE_A].functional_level - at[SPOUSE_A]}, records)


CRITERIA_D.update({
    "M11.C.24": C("M11.C.24", "premise", ("baseline", "treatment"),
                  (Readout("under_change", +1), Readout("over_change", -1)), arm_c24,
                  ("OVERFUNCTION", "UNDERFUNCTION"), settings=SETTINGS["M11.C.24"]),
})
