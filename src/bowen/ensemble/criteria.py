"""The Phase C acceptance criteria as paired arms (plan §3, step 14).

Purpose: define each Phase C `M11.C` criterion as two arms differing in one declared
         channel (`M17.D.3`: initial state, a scripted act at one tick, or the stressor
         schedule), with its readouts, its direction and its class (`M11.5`).
Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.1, #M11.C.3, #M11.C.4, #M11.C.5, #M11.C.16, #M11.C.19, #M11.C.25, #M11.C.27, #M11.C.29, #M11.C.32, #M11.C.35, #M11.C.38, #M11.C.41, #M11.C.42, #M11.C.44, #M11.C.45, #M11.3, #M11.5, #M17.D.3
Tests:   tests/bowen/test_phase_c_gate.py

Written after the constants were frozen (`M10.B.4`). Every arm runs the full engine under
the policy and the learner; a scripted act replaces one person's outcome at one week. All
choices are `[I]` and were declared before any criterion ran: the readout definitions in
this module, and every number an arm uses — horizons, scripted-act weeks, levels, starting
states and the declared spell (`M11.3`) — in ``config/bowen/criteria.md``. "Calm" is no spell.

**Not built in Phase C** (reported as blockers, not as passes or failures): `M11.C.7` needs
`M8.2`/`M8.3`'s position predicates and declares no direction for its topology arms;
`M11.C.13` needs incidents located in a community the model does not have; `M11.C.14` needs
`M5.C.1`'s marital-distance gate, which is not built.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections import Counter

from src.bowen.engine.act import Selection
from src.bowen.engine.events import Channel, Event, EventId, Mechanism, SourcePosition
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.log_records import DecidedBy, EffectRecord, EmittedRecord, SelectionRecord
from src.bowen.engine.objects import SymptomChannel
from src.bowen.engine.tick import run_tick
from src.bowen.ensemble.runner import ArmResult, Criterion, Readout
from src.bowen.io.load import CONFIG_DIR, load_constants, load_event_kinds, load_family, load_policy_rules
from src.bowen.readouts.counterfeit import INWARD, OUTWARD, read_counterfeit
from src.bowen.scenario.assemble import assemble
from src.bowen.scenario.config_parse import ConfigError, parse_table_document
from src.bowen.scenario.policy_source import PolicySource
from src.bowen.scenario.scripted_source import ScriptedSource

P = PersonId
FAMILIES = {
    "phase_c": CONFIG_DIR / "family_phase_c.md",
    "dyad": CONFIG_DIR / "fixtures" / "dyad.md",
    "triad": CONFIG_DIR / "fixtures" / "triad.md",
    "four": CONFIG_DIR / "fixtures" / "nuclear_four.md",
}


def load_roles(path=CONFIG_DIR / "criterion_roles.md") -> dict[tuple[str, str], PersonId]:
    """Who plays each role in a criterion, from config (`M2.3a`: no person's id in Python source)."""
    document = parse_table_document(path.read_text(encoding="utf-8"), columns=("family", "role", "person"),
                                    metadata_keys=frozenset(), source=str(path))
    roles = {}
    for row, line in zip(document.rows, document.row_lines):
        key = (row["family"], row["role"])
        if key in roles:
            raise ConfigError(f"{path}:{line}: duplicate role {key}")
        roles[key] = P(row["person"])
    return roles


ROLES = load_roles()

# Keys the criteria table adds to an arm's settings itself, for expanded cells; never config.
INJECTED = frozenset({"twosome", "change", "levels", "arms"})
# Every setting a criterion reads, by section; config/bowen/criteria.md must hold exactly these.
# tests/bowen/test_ensemble_record.py::test_criteria_required_matches_what_the_arms_read ties this to the code.
REQUIRED = {
    "spell": ("kind", "intensity", "every", "from", "light_every", "light_parents"),
    "scripted_act": ("intensity",),
    "M11.C.1": ("weeks", "level_baseline", "level_treatment"),
    "M11.C.3": ("t0", "run_after", "latency"),
    "M11.C.4": ("t0", "nodal", "run_after_nodal"),
    "M11.C.5": ("weeks", "t0"),
    "M11.C.16": ("weeks", "level_baseline", "level_treatment", "window"),
    "M11.C.19": ("weeks", "low_axis", "high_axis"),
    "M11.C.25": ("weeks", "no_pole"),
    "M11.C.27": ("t0", "latency", "report_weeks", "unstable_impingement"),
    "M11.C.29": ("weeks", "t0", "disguise_span", "disguise_every"),
    "M11.C.32": ("weeks", "t0", "angry_impingement"),
    "M11.C.35": ("t0", "run_after", "conductance_high", "conductance_low"),
    "M11.C.38": ("weeks", "level_1", "level_2", "level_3", "level_4"),
    "M11.C.41": ("weeks", "level_higher", "level_lower"),
    "M11.C.42": ("t0", "weeks"),
    "M11.C.44": ("weeks",),
    "M11.C.45": ("weeks",),
}


def _value(text: str):
    """An integer, a float, or (for an event kind) the word itself."""
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    try:
        return float(text)
    except ValueError:
        return text


def load_settings(path=CONFIG_DIR / "criteria.md") -> dict[str, dict]:
    """Purpose: each criterion's declared settings from config, strictly (Hermes gate finding 4, M11.D.2)."""
    document = parse_table_document(path.read_text(encoding="utf-8"), columns=("criterion", "setting", "value"),
                                    metadata_keys=frozenset(), source=str(path))
    settings: dict[str, dict] = {}
    for row, line in zip(document.rows, document.row_lines):
        section, name = row["criterion"], row["setting"]
        if name not in REQUIRED.get(section, ()):
            raise ConfigError(f"{path}:{line}: {section} has no setting {name!r}")
        if name in settings.setdefault(section, {}):
            raise ConfigError(f"{path}:{line}: duplicate setting {section} {name}")
        settings[section][name] = _value(row["value"])
    missing = [f"{s} {n}" for s, names in REQUIRED.items() for n in names if n not in settings.get(s, {})]
    if missing:
        raise ConfigError(f"{path}: missing settings {missing}")
    return settings


SETTINGS = load_settings()
SPELL = SETTINGS["spell"]
PARENTS = {"phase_c": (ROLES["phase_c", "parent_1"], ROLES["phase_c", "parent_2"]), "dyad": (P("a"), P("b")), "triad": (P("f"), P("m")),
           "four": (P("f"), P("m"))}
AUTOMATIC = ("PURSUE", "DISTANCE", "CONFLICT", "OVERFUNCTION", "UNDERFUNCTION", "TRIANGLE", "CUTOFF")


def spell(family: str, weeks: int, every: int | None = None, parents: int | None = None) -> tuple[Event, ...]:
    events = []
    for t in range(SPELL["from"], weeks, every or SPELL["every"]):
        for i, parent in enumerate(PARENTS[family][:parents]):
            events.append(Event(
                id=EventId(t, "spell", i), kind=SPELL["kind"], mechanism=Mechanism.EXOGENOUS_STRESSOR, sender=None,
                targets=(parent,), intensity=SPELL["intensity"], timestamp=t, duration=1, exogenous=True,
                source_position=SourcePosition.NONE, channel=Channel.EXOGENOUS,
            ))
    return tuple(events)


class Forced:
    """The policy, with one person's outcome replaced at declared weeks (a scripted act).

    A scripted act is made only if it is legal that week (`M4.D.1e`): the policy may already
    have cut the tie it crosses. An act that is not legal is skipped and counted, in both arms
    alike, so a seed where it could not be made contributes no difference rather than a biased one.

    ``unavailable`` maps a week to ``(actors, absent)``: that week ``absent`` cannot be the target of
    any act by ``actors``, nor complete a triad for them (`M11.C.42`'s declared absence, `M17.D.3`).
    Each actor's selection is drawn again, under the same keyed draw, from its legal set without
    ``absent``. Every other week, and every other person, is the policy's own.

    ``held_open`` is ``(pairs, weeks)``: in those weeks neither member of a pair may cut the tie between
    them, so ``CUTOFF`` across it is not in their legal set and each one's selection is drawn again, under
    the same keyed draw, without it (step S of ``docs/DECISIONS — PHASE C FAILING.md``: a scripted act's
    ties are held open until it is made, so it is legal in every seed).
    """

    def __init__(self, policy: PolicySource, forced: dict, unavailable: dict | None = None,
                 held_open: tuple | None = None):
        self.policy, self.forced, self.unavailable = policy, forced, unavailable or {}
        self.held_open = held_open
        self.made, self.skipped, self.owed_step = 0, 0, 0  # placement counts; ``outcomes`` is what was made
        self.status: dict = {}  # (week, actor) -> "placed", "not legal", "owed a step" or "dead"

    def scheduled(self, tick):
        return self.policy.scheduled(tick)

    def selections(self, tick, active, state):
        from src.bowen.engine.observe import observe
        from src.bowen.policy.policy import legal_outcomes

        chosen = list(self.policy.selections(tick, active, state))
        # One redraw per actor, with every constraint on it this week: a declared absence and the held-open ties.
        absent = dict.fromkeys(self.unavailable[tick][0], self.unavailable[tick][1]) if tick in self.unavailable else {}
        pairs = self.held_open[0] if self.held_open and tick in self.held_open[1] else ()
        excluded = {a: frozenset(f"CUTOFF>{o}" for pair in pairs if a in pair for o in pair if o != a)
                    for pair in pairs for a in pair}
        for actor in sorted((set(absent) | set(excluded)) & set(active)):
            if state.people[actor].alive:
                redrawn = self._without(state, actor, absent.get(actor), excluded.get(actor, frozenset()))
                chosen = [s for s in chosen if s.actor != actor] + [redrawn]
        for (t, actor), selection in self.forced.items():
            if t != tick:
                continue
            if actor not in active:  # dead, or owed an I-POSITION step this week (`M5.D.9`): nothing to replace
                self.owed_step += 1
                self.status[t, actor] = "owed a step" if state.people[actor].alive else "dead"
                continue
            legal = {o.label for o in legal_outcomes(observe(state, actor, self.policy.params), state.kinds,
                                                     self.policy.params)}
            if f"{selection.kind}>{selection.targets[0]}" not in legal:
                self.skipped += 1
                self.status[t, actor] = "not legal"
                continue
            self.made += 1
            self.status[t, actor] = "placed"
            chosen = [s for s in chosen if s.actor != actor] + [selection]
        return tuple(chosen)

    def outcomes(self, records) -> Counter:
        """What became of each scripted act, from the run's records: "made" only if the act emitted was the one
        scripted. A placed act can still be rewritten after selection, e.g. to STAY-IN-CONTACT toward the actor's
        open I-POSITION sequence's other (`M5.D.9`); a week the run never reached is "not reached"."""
        emitted = {(r.event.sender, r.event.timestamp, r.event.kind, r.event.targets[:1]) for r in records
                   if isinstance(r, EmittedRecord) and r.event.mechanism is Mechanism.MOVE}
        # An I-POSITION is made either as an act or as the start of a sequence, which emits nothing that week (M5.D.9).
        begun = {(pid, r.tick) for r in records if isinstance(r, EffectRecord) and r.mechanism == "iposition"
                 for pid, flag, _ in r.people if flag.startswith("iposition:begin")}
        counts = Counter()
        for (t, actor), selection in self.forced.items():
            status = self.status.get((t, actor), "not reached")
            if status == "placed":
                same = ((actor, t, selection.kind, selection.targets[:1]) in emitted
                        or (selection.kind == "I-POSITION" and (actor, t) in begun))
                status = "made" if same else "rewritten"
            counts[status] += 1
        return counts


    def _without(self, state, actor, absent=None, excluded: frozenset[str] = frozenset()):
        """The actor's selection drawn again without ``absent`` (a person) or the ``excluded`` outcomes."""
        from types import MappingProxyType

        from src.bowen.engine.observe import observe
        from src.bowen.policy.policy import decide

        obs = observe(state, actor, self.policy.params)
        if absent is not None:
            obs = dataclasses.replace(
                obs,
                ties=tuple(v for v in obs.ties if v.other != absent),
                triangle_for=MappingProxyType({t: tri for t, tri in obs.triangle_for.items()
                                               if t != absent and absent not in tri.members}),
            )
        decision = decide(obs, state.kinds, self.policy.params, self.policy.rules, state.draws, excluded)
        return dataclasses.replace(decision.selection, urge=decision.urge)


def scenario(family: str, seed: int, weeks: int, *, spells=(), forced=None, setup=None, until=None,
             unavailable=None, held_open=None, watch=None):
    """Run one arm; return (state, records). ``until`` stops early at that week (exclusive); ``watch(state)``, if
    given, is called after every tick and must only read."""
    constants, kinds = load_constants(), load_event_kinds()
    fam = load_family(FAMILIES[family])
    events = ScriptedSource(f"{family}-arm", weeks, tuple(spells), ())
    parts = assemble(constants, kinds, fam, events, seed=seed)
    if setup:
        setup(parts.state)
    source = Forced(PolicySource(events, parts.params, load_policy_rules()), forced or {}, unavailable, held_open)
    records = []
    scenario.last_source = source
    sink = type("Sink", (), {"emit": lambda self, r: records.append(r)})()
    for _ in range(until if until is not None else weeks):
        run_tick(parts.state, source, parts.params, parts.visibility, parts.activation, sink)
        if watch:
            watch(parts.state)
    return parts.state, records


def bookkeeping(records) -> tuple[Counter, Counter, Counter]:
    moves, fallbacks, selections = Counter(), Counter(), Counter()
    for r in records:
        if isinstance(r, EmittedRecord) and r.event.mechanism is Mechanism.MOVE:
            moves[r.event.kind] += 1
        elif isinstance(r, SelectionRecord):
            selections[str(r.actor)] += 1
            if r.decided_by is DecidedBy.FALLBACK:
                fallbacks[str(r.actor)] += 1
            if r.withheld:
                moves["WITHHOLD"] += 1
    return moves, fallbacks, selections


def result(readouts, records) -> ArmResult:
    moves, fallbacks, selections = bookkeeping(records)
    source = getattr(scenario, "last_source", None)
    if source is not None and source.forced:
        for status, n in source.outcomes(records).items():
            moves[f"(scripted act: {status})"] += n
        moves["(scripted acts)"] += len(source.forced)
    return ArmResult(readouts, moves, fallbacks, selections)


def lowered(by: float):
    def setup(state):
        for p in state.people.values():
            if p.role.value == "member":
                p.basic_level = max(0.0, p.basic_level - by)
                p.functional_level = max(0.0, p.functional_level - by)
                p.pseudo_self = p.swing
        _reinitialise(state)
    return setup


def _reinitialise(state):
    """Re-derive what initialisation set from level (contact, outside-ness, beliefs, functioning)."""
    from src.bowen.engine.initialise import initialise_run
    from src.bowen.scenario.params import engine_params

    initialise_run(state, engine_params(load_constants()))


def first_onset(records, horizon: int) -> float:
    for r in records:
        if isinstance(r, EffectRecord) and r.mechanism == "symptom_onset":
            return float(r.tick)
    return float(horizon + 1)  # censored: no onset within the horizon


def held(until: int, *pairs: str) -> tuple:
    """Step S (post hoc, 2026-10-09): the ties ``pairs`` (e.g. "f-m") held open from week 0 until ``until``
    (exclusive), the same in both arms. See ``Forced`` and ``config/bowen/criteria.md``."""
    return tuple(tuple(P(m) for m in pair.split("-")) for pair in pairs), range(0, until)


def act(actor, kind, target, tick):
    intensity = SETTINGS["scripted_act"]["intensity"]
    return {(tick, P(actor)): Selection(actor=P(actor), kind=kind, targets=(P(target),), intensity=intensity)}


# --- M11.C.1 and M11.C.38: level and time to threshold ---------------------------------------


def arm_c1(arm, seed, settings):
    weeks = settings["weeks"]
    by = settings["levels"][0 if arm == "baseline" else 1]
    state, records = scenario("phase_c", seed, weeks, spells=spell("phase_c", weeks), setup=lowered(by))
    return result({"time_to_threshold": first_onset(records, weeks)}, records)


# --- M11.C.3: a triangle relieves the seeker and costs the third, within the tick ------------


def _acute_at(records, tick, people):
    """Each person's acute anxiety at the end of ``tick``: start-of-run values plus logged changes."""
    totals = Counter()
    for r in records:
        if isinstance(r, EffectRecord) and r.tick <= tick:
            for pid, v in r.acute_anxiety:
                if pid in people:
                    totals[pid] += v
    return totals


def arm_c3(arm, seed, settings):
    t0, weeks = settings["t0"], settings["t0"] + settings["run_after"]
    forced = act("f", "TRIANGLE", "c", t0) if arm == "treatment" else act("f", "STAY-IN-CONTACT", "m", t0)
    state, records = scenario("triad", seed, weeks, forced=forced, held_open=held(t0, "f-m", "f-c"))
    delivered = t0 + settings["latency"]  # the act's effect lands after the tie's latency
    change = _acute_at(records, delivered, (P("f"), P("m"), P("c")))
    # Restated 2026-10-09 (owner decision, docs/PROPOSAL — TRIANGLE RELIEF AND CREDIT.md): a triangle relieves the
    # seeker, not the pair, so the readout is the seeker's change; the partner's is reported.
    return result({"seeker_anxiety": change[P("f")], "third_anxiety": change[P("c")],
                   "partner_anxiety": change[P("m")]}, records)


# --- M11.C.4: cutoff trades now against later --------------------------------------------------


def arm_c4(arm, seed, settings):
    t0, nodal = settings["t0"], settings["nodal"]
    weeks = nodal + settings["run_after_nodal"]
    nodal_event = (Event(id=EventId(nodal, "nodal", 0), kind=SPELL["kind"], mechanism=Mechanism.EXOGENOUS_STRESSOR,
                         sender=None, targets=(P("a"),), intensity=SPELL["intensity"], timestamp=nodal, duration=1,
                         exogenous=True, source_position=SourcePosition.NONE, channel=Channel.EXOGENOUS),)
    forced = act("a", "CUTOFF", "b", t0) if arm == "treatment" else act("a", "STAY-IN-CONTACT", "b", t0)
    state, records = scenario("dyad", seed, weeks, spells=nodal_event, forced=forced, held_open=held(t0, "a-b"))
    now = _acute_at(records, t0 + 1, (P("a"),))[P("a")] - _acute_at(records, t0 - 1, (P("a"),))[P("a")]
    later = _acute_at(records, nodal + 1, (P("a"), P("b")))
    return result({"actor_relief_now": now, "family_anxiety_at_nodal": later[P("a")] + later[P("b")]}, records)


# --- M11.C.5: the change-back reaction ----------------------------------------------------------


def arm_c5(arm, seed, settings):
    weeks = settings["weeks"]

    def perspective(state):
        state.people[mover].systems_perspective = 1.0

    mover, target, third = (ROLES["phase_c", r] for r in ("mover", "target", "third"))
    forced = act(mover.value, "I-POSITION", target.value, settings["t0"]) if arm == "treatment" else {}
    state, records = scenario("phase_c", seed, weeks, spells=spell("phase_c", weeks), forced=forced, setup=perspective,
                              held_open=held(settings["t0"], f"{mover.value}-{target.value}"))
    reaction = sum(v for r in records if isinstance(r, EffectRecord) and r.tick > settings["t0"]
                   for p, v in r.acute_anxiety if p == target and v > 0)
    load = sum(state.people[third].symptom_load.values())
    return result({"target_reaction": reaction, "third_person_symptom_load": load}, records)


# --- M11.C.16: repertoire concentration depends on level ----------------------------------------


def _automatic_entropy(records, since):
    kinds = Counter(r.event.kind for r in records if isinstance(r, EmittedRecord)
                    and r.event.mechanism is Mechanism.MOVE and r.event.kind in AUTOMATIC
                    and r.event.timestamp >= since)
    total = sum(kinds.values())
    if total == 0:
        return 0.0, 0.0
    shares = [n / total for n in kinds.values()]
    return -sum(s * math.log(s) for s in shares) / math.log(len(AUTOMATIC)), max(shares)


def arm_c16(arm, seed, settings):
    weeks = settings["weeks"]
    by = settings["levels"][0 if arm == "baseline" else 1]
    state, records = scenario("phase_c", seed, weeks, spells=spell("phase_c", weeks), setup=lowered(by))
    entropy, top = _automatic_entropy(records, weeks - settings["window"])
    return result({"repertoire_entropy": entropy, "top_move_share": top}, records)


# --- M11.C.19: the two counterfeits, told apart by axis -----------------------------------------


def arm_c19(arm, seed, settings):
    low, high = settings["low_axis"], settings["high_axis"]
    axes = (low, high) if arm == "baseline" else (high, low)  # accommodator, then declarer: equal magnitude

    def setup(state):
        state.people[P("a")].outside_ness_outward, state.people[P("a")].outside_ness_inward = axes

    state, records = scenario("dyad", seed, settings["weeks"], setup=setup)
    reading = read_counterfeit(state.people[P("a")], dataclasses.replace(_params()))
    return result({"outward_failed": float(OUTWARD in reading.failed), "inward_failed": float(INWARD in reading.failed)},
                  records)


def _params():
    from src.bowen.scenario.params import engine_params

    return engine_params(load_constants())


# --- M11.C.25: the dominant pole is independent of sex ------------------------------------------


def arm_c25(arm, seed, settings):
    def setup(state):
        if arm == "treatment":  # swap the two spouses' sexes; nothing else differs
            a, b = state.people[P("a")], state.people[P("b")]
            a.sex, b.sex = b.sex, a.sex

    state, records = scenario("dyad", seed, settings["weeks"], spells=spell("dyad", settings["weeks"]), setup=setup)
    # The readout is a's dominance, not the female's. The arms differ only in which spouse is female,
    # so if sex plays no part a's dominance is the same in both arms. A readout of "the female is
    # dominant" could not fail: a sex bias raises it in both arms alike and the paired difference
    # stays at zero. Corrected at step 15, after the sex-term mutant survived (post-hoc, logged in
    # docs/phase_c_completion_report.md).
    balance = state.ties[TieId.of(P("a"), P("b"))].functioning_balance["joint"]
    if balance == 0:
        return result({"first_dominant": settings["no_pole"]}, records)  # no pole: neither spouse holds it
    return result({"first_dominant": float(balance > 0)}, records)  # the tie's first member, a, over-functioning


# --- M11.C.27: the twosome 2x2 ------------------------------------------------------------------


def _pair_deviation(state):
    from src.bowen.engine.contact import deviation

    tie = state.ties[TieId.of(P("f"), P("m"))]
    params = _params()
    return sum(deviation(state.people[m], tie, params) for m in tie.id.members())


def arm_c27(arm, seed, settings):
    t0 = settings["t0"]
    unstable = settings["twosome"] == "unstable"

    def setup(state):
        if unstable:
            tie = state.ties[TieId.of(P("f"), P("m"))]
            for m in tie.id.members():
                tie.felt_impingement[m] = settings["unstable_impingement"]

    if arm == "baseline":
        forced = act("f", "STAY-IN-CONTACT", "m", t0)
    elif settings["change"] == "add_third":
        forced = act("f", "TRIANGLE", "c", t0)
    else:  # remove one: f severs contact with m for the cell's window
        forced = act("f", "CUTOFF", "m", t0)
    ties = ("f-m", "f-c") if settings["change"] == "add_third" else ("f-m",)
    # Read week by week (owner decision, 2026-10-09: the effect is almost immediate, and a long horizon is
    # distorted by other effects). ``deviation[w]`` is the pair's deviation at the end of week w.
    deviation = []
    weeks = t0 + settings["report_weeks"] + 1
    state, records = scenario("triad", seed, weeks, forced=forced, setup=setup, held_open=held(t0, *ties),
                              watch=lambda s: deviation.append(_pair_deviation(s)))
    readouts = {"pair_deviation": deviation[t0 + settings["latency"]]}
    readouts |= {f"pair_deviation_week_{k}": deviation[t0 + k] for k in range(1, settings["report_weeks"] + 1)}
    return result(readouts, records)


# --- M11.C.29: relief and differentiation, by the third person's time course -------------------


def arm_c29(arm, seed, settings):
    weeks, t0 = settings["weeks"], settings["t0"]

    def perspective(state):
        state.people[P("f")].systems_perspective = 1.0

    span, every = settings["disguise_span"], settings["disguise_every"]
    if arm == "treatment":  # a genuine I-POSITION
        forced = act("f", "I-POSITION", "m", t0)
    else:  # distance in disguise: withdrawals over the same weeks
        forced = {k: v for t in range(t0, t0 + span, every) for k, v in act("f", "DISTANCE", "m", t).items()}
    # The readout, restated post hoc on 2026-10-09 (X1 of docs/DECISIONS — PHASE C FAILING.md): the weeks the third
    # person's symptom is active, from onset (`M1.A.6`) until the load falls below the re-arm fraction of the
    # threshold. The declared readout counted weeks with any symptom accumulation, which is positive whenever acute
    # anxiety is above the chronic floor; the model rests a few points above it by design (D7), so that count sat at
    # its maximum in both arms.
    active = []

    def watch(state):
        third = state.people[P("c")]
        active.append(third.channel_prior is not None and third.symptom_active[third.channel_prior])

    state, records = scenario("triad", seed, weeks, spells=spell("triad", weeks), forced=forced, setup=perspective,
                              held_open=held(t0 + span, "f-m"), watch=watch)
    return result({"budget": state.family.undifferentiation_budget, "third_symptom_weeks": float(sum(active))},
                  records)


# --- M11.C.32: the mover's anger stalls and degrades the move -----------------------------------


def arm_c32(arm, seed, settings):
    t0 = settings["t0"]

    def setup(state):
        state.people[P("a")].systems_perspective = 1.0

    def anger(state):
        setup(state)
        if arm == "treatment":
            state.ties[TieId.of(P("a"), P("b"))].felt_impingement[P("a")] = settings["angry_impingement"]

    state, records = scenario("dyad", seed, settings["weeks"], forced=act("a", "I-POSITION", "b", t0), setup=anger)
    assertion = sum(1 for r in records if isinstance(r, EmittedRecord) and r.event.assertion)
    peaks = sum(1 for r in records if isinstance(r, EffectRecord) and r.mechanism == "iposition"
                for _, f, _ in r.people if f in ("iposition:enter:RESOLVE",))
    return result({"assertion_form": float(assertion), "reached_peak": float(peaks)}, records)


# --- M11.C.35: a witness appraises from its own position ----------------------------------------


def arm_c35(arm, seed, settings):
    t0 = settings["t0"]

    def setup(state):
        high, low = settings["conductance_high"], settings["conductance_low"]
        state.ties[TieId.of(P("c"), P("f"))].conductance = high if arm == "treatment" else low

    forced = act("f", "CONFLICT", "m", t0)
    state, records = scenario("triad", seed, t0 + settings["run_after"], forced=forced, setup=setup,
                              held_open=held(t0, "f-m"))
    event_id = EventId(t0, "f", 0)
    witness = sum(v for r in records if isinstance(r, EffectRecord) and r.mechanism == "appraisal"
                  and r.cause == event_id for p, v in r.acute_anxiety if p == P("c"))
    return result({"witness_appraisal": witness}, records)


# --- M11.C.41: level and stress each exacerbate the pattern -------------------------------------


def arm_c41(arm, seed, settings):
    weeks = settings["weeks"]
    level, stress = settings["arms"][0 if arm == "baseline" else 1]
    light = spell("phase_c", weeks, every=SPELL["light_every"], parents=SPELL["light_parents"])
    spells = spell("phase_c", weeks) if stress == "heavy" else light
    state, records = scenario("phase_c", seed, weeks, spells=spells, setup=lowered(level))
    acute = sum(p.acute_anxiety for p in state.people.values() if p.role.value == "member")
    return result({"mean_acute": acute, "reactive_over_chance": reactive_over_chance(records)}, records)


def reactive_over_chance(records) -> float:
    """Per selection, whether the act chosen was reactive, less the reactive share of that selection's legal set;
    the mean over every selection whose legal set offered a reactive act.

    Restated post hoc on 2026-10-09 (Q2 of ``docs/DECISIONS — PHASE C FAILING.md``, approved by the owner). The
    declared readout, reactive moves over all moves, conflicts with `M4.D.3a`: a lower level closes layers 1 and 2,
    which hold five of the seven reactive acts, so the share fell at a lower level even as acute anxiety rose. The
    first restatement (reactive selected over reactive offered) was confounded the other way, found by review
    before it was relied on: a chooser picking uniformly at random scores 1/N on it, so it rose whenever the legal
    set shrank. A uniform chooser scores 0 on this one at any set size, so it reads a preference for reactive acts
    beyond what the legal set makes likely. Only the policy's own draws count: a ``FALLBACK`` is always a hold
    (`M4.D.1f`), not a choice, and a fallback rate that differed between arms would otherwise shift the readout
    (re-sweep, 2026-10-09).
    """
    kinds = {r.event.id: r.event.kind for r in records if isinstance(r, EmittedRecord)}
    excess, n = 0.0, 0
    for r in records:
        if isinstance(r, SelectionRecord) and r.decided_by is DecidedBy.POLICY and r.legal_set:
            offered = sum(1 for label in r.legal_set if label.split(">")[0] in AUTOMATIC)
            if offered:
                excess += (kinds.get(r.event_id) in AUTOMATIC) - offered / len(r.legal_set)
                n += 1
    return excess / max(1, n)


# --- M11.C.42: a triangle that relieved is reused ----------------------------------------------


def arm_c42(arm, seed, settings):
    # Decided 2026-10-08 from the spec's text (docs/phase_c_completion_report.md §9): in the baseline the third
    # member is unavailable to the pair at t0, so no triangle forms; the readout counts the pair's triangles.
    t0, weeks = settings["t0"], settings["weeks"]
    pair, third = (P("f"), P("m")), P("c")
    if arm == "treatment":
        forced, unavailable = act("f", "TRIANGLE", "c", t0), None
    else:
        forced, unavailable = {}, {t0: (pair, third)}
    state, records = scenario("triad", seed, weeks, spells=spell("triad", weeks), forced=forced,
                              setup=settings.get("setup"), unavailable=unavailable, held_open=held(t0, "f-c"))
    reused = sum(1 for r in records if isinstance(r, EmittedRecord) and r.event.sender in pair
                 and r.event.kind == "TRIANGLE" and r.event.targets == (third,) and r.event.timestamp > t0)
    return result({"triangle_reuse": float(reused)}, records)


# --- M11.C.44 and M11.C.45: position value and triangle quiet, calm against spell ---------------

OUTSIDE_ACTS, INSIDE_ACTS = ("DISTANCE", "CUTOFF"), ("TRIANGLE", "PURSUE")  # [I], declared here


def arm_spell_triad(arm, seed, settings):
    weeks = settings["weeks"]
    spells = spell("triad", weeks) if arm == "treatment" else ()
    state, records = scenario("triad", seed, weeks, spells=spells)
    triad = {P("f"), P("m"), P("c")}
    acts_ = [r.event for r in records if isinstance(r, EmittedRecord) and r.event.mechanism is Mechanism.MOVE
             and r.event.sender in triad]
    outside = sum(1 for e in acts_ if e.kind in OUTSIDE_ACTS)
    inside = sum(1 for e in acts_ if e.kind in INSIDE_ACTS)
    # M11.C.45's "rate of TRIANGLE selection": one selection per person per week (M4.D.1), so the rate is
    # TRIANGLE selections per person-week (decided 2026-10-08, docs/phase_c_completion_report.md §9).
    triangles = sum(1 for e in acts_ if e.kind == "TRIANGLE") / (len(triad) * weeks)
    return result({"outside_inside_ratio": (outside + 1) / (inside + 1), "triangle_rate": triangles}, records)


# --- the gate --------------------------------------------------------------------------------------

C = Criterion
CRITERIA: dict[str, Criterion] = {
    "M11.C.1": C("M11.C.1", "premise", ("baseline", "treatment"), (Readout("time_to_threshold", -1),), arm_c1,
                 settings={**SETTINGS["M11.C.1"], "levels": (SETTINGS["M11.C.1"]["level_baseline"], SETTINGS["M11.C.1"]["level_treatment"])}),
    "M11.C.3": C("M11.C.3", "premise", ("baseline", "treatment"),
                 (Readout("seeker_anxiety", -1), Readout("third_anxiety", +1),
                                                 Readout("partner_anxiety", 0, report_only=True)), arm_c3, ("TRIANGLE",),
                 settings=SETTINGS["M11.C.3"]),
    "M11.C.4": C("M11.C.4", "premise", ("baseline", "treatment"),
                 (Readout("actor_relief_now", -1), Readout("family_anxiety_at_nodal", +1)), arm_c4, ("CUTOFF",),
                 settings=SETTINGS["M11.C.4"]),
    "M11.C.5": C("M11.C.5", "composite", ("baseline", "treatment"),
                 (Readout("target_reaction", +1), Readout("third_person_symptom_load", +1)), arm_c5,
                 settings=SETTINGS["M11.C.5"]),
    "M11.C.16": C("M11.C.16", "composite", ("baseline", "treatment"),
                  (Readout("repertoire_entropy", -1), Readout("top_move_share", +1, report_only=True)), arm_c16,
                  settings={**SETTINGS["M11.C.16"], "levels": (SETTINGS["M11.C.16"]["level_baseline"], SETTINGS["M11.C.16"]["level_treatment"])}),
    "M11.C.19": C("M11.C.19", "check", ("baseline", "treatment"),
                  (Readout("outward_failed", +1, (0, 1)), Readout("inward_failed", -1, (0, 1))), arm_c19,
                  settings=SETTINGS["M11.C.19"]),
    "M11.C.25": C("M11.C.25", "premise", ("baseline", "treatment"), (Readout("first_dominant", 0, (0, 1)),),
                  arm_c25, settings=SETTINGS["M11.C.25"]),
    "M11.C.29": C("M11.C.29", "premise", ("baseline", "treatment"),
                  (Readout("budget", -1), Readout("third_symptom_weeks", -1)), arm_c29,
                  settings=SETTINGS["M11.C.29"]),
    "M11.C.32": C("M11.C.32", "premise", ("baseline", "treatment"),
                  (Readout("assertion_form", +1), Readout("reached_peak", -1)), arm_c32, ("I-POSITION",),
                  settings=SETTINGS["M11.C.32"]),
    "M11.C.35": C("M11.C.35", "check", ("baseline", "treatment"), (Readout("witness_appraisal", +1),), arm_c35,
                  settings=SETTINGS["M11.C.35"]),
    "M11.C.42": C("M11.C.42", "composite", ("baseline", "treatment"), (Readout("triangle_reuse", +1),), arm_c42,
                  ("TRIANGLE",), settings=SETTINGS["M11.C.42"]),
    "M11.C.44": C("M11.C.44", "composite", ("baseline", "treatment"), (Readout("outside_inside_ratio", +1),),
                  arm_spell_triad, settings=SETTINGS["M11.C.44"]),
    "M11.C.45": C("M11.C.45", "composite", ("baseline", "treatment"), (Readout("triangle_rate", +1),),
                  arm_spell_triad, ("TRIANGLE",), settings=SETTINGS["M11.C.45"]),
}
# M11.C.27: four cells, each a two-arm direction (M11.4, P9).
for twosome, change, direction in (("stable", "add_third", +1), ("stable", "remove_one", +1),
                                   ("unstable", "add_third", -1), ("unstable", "remove_one", -1)):
    cid = f"M11.C.27[{twosome},{change}]"
    weekly = tuple(Readout(f"pair_deviation_week_{k}", direction, report_only=True)
                   for k in range(1, SETTINGS["M11.C.27"]["report_weeks"] + 1))
    CRITERIA[cid] = C(cid, "composite", ("baseline", "treatment"), (Readout("pair_deviation", direction), *weekly),
                      arm_c27, settings={**SETTINGS["M11.C.27"], "twosome": twosome, "change": change})
# M11.C.38: four basic levels, a monotone ordering, as three adjacent two-arm directions (M11.4).
_levels = [v for k, v in sorted(SETTINGS["M11.C.38"].items()) if k.startswith("level_")]
for lo, hi in zip(_levels, _levels[1:]):
    cid = f"M11.C.38[-{lo:g} vs -{hi:g}]"
    CRITERIA[cid] = C(cid, "premise", ("baseline", "treatment"), (Readout("time_to_threshold", -1),), arm_c1,
                      settings={"weeks": SETTINGS["M11.C.38"]["weeks"], "levels": (lo, hi)})
# M11.C.41: the 2x2 of level and stress, four two-arm directions on two readouts each.
_hi, _lo = SETTINGS["M11.C.41"]["level_higher"], SETTINGS["M11.C.41"]["level_lower"]
for name, base, treat in (("light: lower level", (_hi, "light"), (_lo, "light")),
                          ("heavy: lower level", (_hi, "heavy"), (_lo, "heavy")),
                          ("higher level: heavier stress", (_hi, "light"), (_hi, "heavy")),
                          ("lower level: heavier stress", (_lo, "light"), (_lo, "heavy"))):
    cid = f"M11.C.41[{name}]"
    CRITERIA[cid] = C(cid, "mixed", ("baseline", "treatment"),
                      (Readout("mean_acute", +1), Readout("reactive_over_chance", +1)), arm_c41,
                      settings={"weeks": SETTINGS["M11.C.41"]["weeks"], "arms": (base, treat)})

NOT_BUILT = {
    "M11.C.7": "needs M8.2/M8.3's position predicates; the criterion declares no direction for its topology arms",
    "M11.C.13": "needs incidents located in a community; the model has none",
    "M11.C.14": "needs M5.C.1's marital-distance gate, which is not built",
}
