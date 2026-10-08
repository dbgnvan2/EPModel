"""The Phase C acceptance criteria as paired arms (plan §3, step 14).

Purpose: define each Phase C `M11.C` criterion as two arms differing in one declared
         channel (`M17.D.3`: initial state, a scripted act at one tick, or the stressor
         schedule), with its readouts, its direction and its class (`M11.5`).
Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.1, #M11.C.3, #M11.C.4, #M11.C.5, #M11.C.16, #M11.C.19, #M11.C.25, #M11.C.27, #M11.C.29, #M11.C.32, #M11.C.35, #M11.C.38, #M11.C.41, #M11.C.42, #M11.C.44, #M11.C.45, #M11.3, #M11.5, #M17.D.3
Tests:   tests/bowen/test_phase_c_gate.py

Written after the constants were frozen (`M10.B.4`). Every arm runs the full engine under
the policy and the learner; a scripted act replaces one person's outcome at one week. All
choices below — horizons, the spell, readout definitions — are `[I]` and declared here
before any criterion ran.

**The declared spell** (`M11.3`): a `JOB_LOSS` stressor of intensity 120 to each parent every
4 weeks from week 4 to the horizon. "Calm" is no spell.

**Not built in Phase C** (reported as blockers, not as passes or failures): `M11.C.7` needs
`M8.2`/`M8.3`'s position predicates and declares no direction for its topology arms;
`M11.C.13` needs incidents located in a community the model does not have; `M11.C.14` needs
`M5.C.1`'s marital-distance gate, which is not built.
"""

from __future__ import annotations

import dataclasses
import math
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
PARENTS = {"phase_c": (ROLES["phase_c", "parent_1"], ROLES["phase_c", "parent_2"]), "dyad": (P("a"), P("b")), "triad": (P("f"), P("m")),
           "four": (P("f"), P("m"))}
SPELL_INTENSITY, SPELL_EVERY, SPELL_FROM = 120.0, 4, 4
AUTOMATIC = ("PURSUE", "DISTANCE", "CONFLICT", "OVERFUNCTION", "UNDERFUNCTION", "TRIANGLE", "CUTOFF")


def spell(family: str, weeks: int, start: int = SPELL_FROM) -> tuple[Event, ...]:
    events = []
    for t in range(start, weeks, SPELL_EVERY):
        for i, parent in enumerate(PARENTS[family]):
            events.append(Event(
                id=EventId(t, "spell", i), kind="JOB_LOSS", mechanism=Mechanism.EXOGENOUS_STRESSOR, sender=None,
                targets=(parent,), intensity=SPELL_INTENSITY, timestamp=t, duration=1, exogenous=True,
                source_position=SourcePosition.NONE, channel=Channel.EXOGENOUS,
            ))
    return tuple(events)


class Forced:
    """The policy, with one person's outcome replaced at declared weeks (a scripted act).

    A scripted act is made only if it is legal that week (`M4.D.1e`): the policy may already
    have cut the tie it crosses. An act that is not legal is skipped and counted, in both arms
    alike, so a seed where it could not be made contributes no difference rather than a biased one.
    """

    def __init__(self, policy: PolicySource, forced: dict):
        self.policy, self.forced = policy, forced
        self.made, self.skipped = 0, 0

    def scheduled(self, tick):
        return self.policy.scheduled(tick)

    def selections(self, tick, active, state):
        from src.bowen.engine.observe import observe
        from src.bowen.policy.policy import legal_outcomes

        chosen = list(self.policy.selections(tick, active, state))
        for (t, actor), selection in self.forced.items():
            if t != tick or actor not in active:
                continue
            legal = {o.label for o in legal_outcomes(observe(state, actor, self.policy.params), state.kinds,
                                                     self.policy.params)}
            if f"{selection.kind}>{selection.targets[0]}" not in legal:
                self.skipped += 1
                continue
            self.made += 1
            chosen = [s for s in chosen if s.actor != actor] + [selection]
        return tuple(chosen)


def scenario(family: str, seed: int, weeks: int, *, spells=(), forced=None, setup=None, until=None):
    """Run one arm; return (state, records). ``until`` stops early at that week (exclusive)."""
    constants, kinds = load_constants(), load_event_kinds()
    fam = load_family(FAMILIES[family])
    events = ScriptedSource(f"{family}-arm", weeks, tuple(spells), ())
    parts = assemble(constants, kinds, fam, events, seed=seed)
    if setup:
        setup(parts.state)
    source = Forced(PolicySource(events, parts.params, load_policy_rules()), forced or {})
    records = []
    scenario.last_source = source
    sink = type("Sink", (), {"emit": lambda self, r: records.append(r)})()
    for _ in range(until if until is not None else weeks):
        run_tick(parts.state, source, parts.params, parts.visibility, parts.activation, sink)
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
        moves["(scripted act made)"] += source.made
        moves["(scripted act not legal, skipped)"] += source.skipped
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


def act(actor, kind, target, tick, intensity=100.0):
    return {(tick, P(actor)): Selection(actor=P(actor), kind=kind, targets=(P(target),), intensity=intensity)}


# --- M11.C.1 and M11.C.38: level and time to threshold ---------------------------------------


def arm_c1(arm, seed, settings):
    weeks = settings["weeks"]
    by = settings["levels"][0 if arm == "baseline" else 1]
    state, records = scenario("phase_c", seed, weeks, spells=spell("phase_c", weeks), setup=lowered(by))
    return result({"time_to_threshold": first_onset(records, weeks)}, records)


# --- M11.C.3: a triangle relieves the pair and costs the third, within the tick --------------


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
    t0, weeks = settings["t0"], settings["t0"] + 2
    forced = act("f", "TRIANGLE", "c", t0) if arm == "treatment" else act("f", "STAY-IN-CONTACT", "m", t0)
    state, records = scenario("triad", seed, weeks, forced=forced)
    delivered = t0 + 1  # latency 1: the act's effect lands in the next tick
    change = _acute_at(records, delivered, (P("f"), P("m"), P("c")))
    return result({"pair_anxiety": change[P("f")] + change[P("m")], "third_anxiety": change[P("c")]}, records)


# --- M11.C.4: cutoff trades now against later --------------------------------------------------


def arm_c4(arm, seed, settings):
    t0, nodal = settings["t0"], settings["nodal"]
    weeks = nodal + 2
    nodal_event = (Event(id=EventId(nodal, "nodal", 0), kind="JOB_LOSS", mechanism=Mechanism.EXOGENOUS_STRESSOR,
                         sender=None, targets=(P("a"),), intensity=SPELL_INTENSITY, timestamp=nodal, duration=1,
                         exogenous=True, source_position=SourcePosition.NONE, channel=Channel.EXOGENOUS),)
    forced = act("a", "CUTOFF", "b", t0) if arm == "treatment" else act("a", "STAY-IN-CONTACT", "b", t0)
    state, records = scenario("dyad", seed, weeks, spells=nodal_event, forced=forced)
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
    state, records = scenario("phase_c", seed, weeks, spells=spell("phase_c", weeks), forced=forced, setup=perspective)
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
    entropy, top = _automatic_entropy(records, weeks - 52)
    return result({"repertoire_entropy": entropy, "top_move_share": top}, records)


# --- M11.C.19: the two counterfeits, told apart by axis -----------------------------------------


def arm_c19(arm, seed, settings):
    axes = (0.2, 0.8) if arm == "baseline" else (0.8, 0.2)  # accommodator, then declarer: equal magnitude

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
        return result({"first_dominant": 0.5}, records)  # no pole
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
                tie.felt_impingement[m] = 0.8

    if arm == "baseline":
        forced = act("f", "STAY-IN-CONTACT", "m", t0)
    elif settings["change"] == "add_third":
        forced = act("f", "TRIANGLE", "c", t0)
    else:  # remove one: f severs contact with m for the cell's window
        forced = act("f", "CUTOFF", "m", t0)
    state, records = scenario("triad", seed, t0 + 3, forced=forced, setup=setup)
    return result({"pair_deviation": _pair_deviation(state)}, records)


# --- M11.C.29: relief and differentiation, by the third person's time course -------------------


def arm_c29(arm, seed, settings):
    weeks, t0 = settings["weeks"], settings["t0"]

    def perspective(state):
        state.people[P("f")].systems_perspective = 1.0

    if arm == "treatment":  # a genuine I-POSITION
        forced = act("f", "I-POSITION", "m", t0)
    else:  # distance in disguise: withdrawals over the same weeks
        forced = {k: v for t in range(t0, t0 + 8, 2) for k, v in act("f", "DISTANCE", "m", t).items()}
    state, records = scenario("triad", seed, weeks, spells=spell("triad", weeks), forced=forced, setup=perspective)
    symptom_weeks = sum(1 for r in records if isinstance(r, EffectRecord) and r.mechanism == "symptom_accumulation"
                        and any(p == P("c") and v > 0 for p, _, v in r.people))
    return result({"budget": state.family.undifferentiation_budget, "third_symptom_weeks": float(symptom_weeks)},
                  records)


# --- M11.C.32: the mover's anger stalls and degrades the move -----------------------------------


def arm_c32(arm, seed, settings):
    t0 = settings["t0"]

    def setup(state):
        state.people[P("a")].systems_perspective = 1.0

    def anger(state):
        setup(state)
        if arm == "treatment":
            state.ties[TieId.of(P("a"), P("b"))].felt_impingement[P("a")] = 1.0

    state, records = scenario("dyad", seed, settings["weeks"], forced=act("a", "I-POSITION", "b", t0), setup=anger)
    assertion = sum(1 for r in records if isinstance(r, EmittedRecord) and r.event.assertion)
    peaks = sum(1 for r in records if isinstance(r, EffectRecord) and r.mechanism == "iposition"
                for _, f, _ in r.people if f in ("iposition:enter:RESOLVE",))
    return result({"assertion_form": float(assertion), "reached_peak": float(peaks)}, records)


# --- M11.C.35: a witness appraises from its own position ----------------------------------------


def arm_c35(arm, seed, settings):
    t0 = settings["t0"]

    def setup(state):
        state.ties[TieId.of(P("c"), P("f"))].conductance = 1.0 if arm == "treatment" else 0.4

    forced = act("f", "CONFLICT", "m", t0)
    state, records = scenario("triad", seed, t0 + 2, forced=forced, setup=setup)
    event_id = EventId(t0, "f", 0)
    witness = sum(v for r in records if isinstance(r, EffectRecord) and r.mechanism == "appraisal"
                  and r.cause == event_id for p, v in r.acute_anxiety if p == P("c"))
    return result({"witness_appraisal": witness}, records)


# --- M11.C.41: level and stress each exacerbate the pattern -------------------------------------


def arm_c41(arm, seed, settings):
    weeks = settings["weeks"]
    level, stress = settings["arms"][0 if arm == "baseline" else 1]
    spells = spell("phase_c", weeks) if stress == "heavy" else spell("phase_c", weeks)[::4]
    state, records = scenario("phase_c", seed, weeks, spells=spells, setup=lowered(level))
    acute = sum(p.acute_anxiety for p in state.people.values() if p.role.value == "member")
    emitted = [r.event.kind for r in records if isinstance(r, EmittedRecord) and r.event.mechanism is Mechanism.MOVE
               and r.event.sender is not None]
    reactive = sum(1 for k in emitted if k in AUTOMATIC) / max(1, len(emitted))
    return result({"mean_acute": acute, "reactive_share": reactive}, records)


# --- M11.C.42: a triangle that relieved is reused ----------------------------------------------


def arm_c42(arm, seed, settings):
    t0, weeks = settings["t0"], settings["weeks"]
    forced = act("f", "TRIANGLE", "c", t0) if arm == "treatment" else act("f", "STAY-IN-CONTACT", "m", t0)
    state, records = scenario("triad", seed, weeks, spells=spell("triad", weeks), forced=forced,
                              setup=settings.get("setup"))
    reused = sum(1 for r in records if isinstance(r, EmittedRecord) and r.event.sender == P("f")
                 and r.event.kind == "TRIANGLE" and r.event.targets == (P("c"),) and r.event.timestamp > t0)
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
    triangles = sum(1 for e in acts_ if e.kind == "TRIANGLE") / max(1, len(acts_))
    return result({"outside_inside_ratio": (outside + 1) / (inside + 1), "triangle_rate": triangles}, records)


# --- the gate --------------------------------------------------------------------------------------

C = Criterion
CRITERIA: dict[str, Criterion] = {
    "M11.C.1": C("M11.C.1", "premise", ("baseline", "treatment"), (Readout("time_to_threshold", -1),), arm_c1,
                 settings={"weeks": 104, "levels": (0.0, 10.0)}),
    "M11.C.3": C("M11.C.3", "premise", ("baseline", "treatment"),
                 (Readout("pair_anxiety", -1), Readout("third_anxiety", +1)), arm_c3, ("TRIANGLE",),
                 settings={"t0": 12}),
    "M11.C.4": C("M11.C.4", "premise", ("baseline", "treatment"),
                 (Readout("actor_relief_now", -1), Readout("family_anxiety_at_nodal", +1)), arm_c4, ("CUTOFF",),
                 settings={"t0": 8, "nodal": 30}),
    "M11.C.5": C("M11.C.5", "composite", ("baseline", "treatment"),
                 (Readout("target_reaction", +1), Readout("third_person_symptom_load", +1)), arm_c5,
                 settings={"weeks": 60, "t0": 4}),
    "M11.C.16": C("M11.C.16", "composite", ("baseline", "treatment"),
                  (Readout("repertoire_entropy", -1), Readout("top_move_share", +1, report_only=True)), arm_c16,
                  settings={"weeks": 104, "levels": (0.0, 15.0)}),
    "M11.C.19": C("M11.C.19", "check", ("baseline", "treatment"),
                  (Readout("outward_failed", +1, (0, 1)), Readout("inward_failed", -1, (0, 1))), arm_c19,
                  settings={"weeks": 1}),
    "M11.C.25": C("M11.C.25", "premise", ("baseline", "treatment"), (Readout("first_dominant", 0, (0, 1)),),
                  arm_c25, settings={"weeks": 52}),
    "M11.C.29": C("M11.C.29", "premise", ("baseline", "treatment"),
                  (Readout("budget", -1), Readout("third_symptom_weeks", -1)), arm_c29,
                  settings={"weeks": 80, "t0": 4}),
    "M11.C.32": C("M11.C.32", "premise", ("baseline", "treatment"),
                  (Readout("assertion_form", +1), Readout("reached_peak", -1)), arm_c32, ("I-POSITION",),
                  settings={"weeks": 30, "t0": 0}),
    "M11.C.35": C("M11.C.35", "check", ("baseline", "treatment"), (Readout("witness_appraisal", +1),), arm_c35,
                  settings={"t0": 6}),
    "M11.C.42": C("M11.C.42", "composite", ("baseline", "treatment"), (Readout("triangle_reuse", +1),), arm_c42,
                  ("TRIANGLE",), settings={"t0": 10, "weeks": 40}),
    "M11.C.44": C("M11.C.44", "composite", ("baseline", "treatment"), (Readout("outside_inside_ratio", +1),),
                  arm_spell_triad, settings={"weeks": 80}),
    "M11.C.45": C("M11.C.45", "composite", ("baseline", "treatment"), (Readout("triangle_rate", +1),),
                  arm_spell_triad, ("TRIANGLE",), settings={"weeks": 80}),
}
# M11.C.27: four cells, each a two-arm direction (M11.4, P9).
for twosome, change, direction in (("stable", "add_third", +1), ("stable", "remove_one", +1),
                                   ("unstable", "add_third", -1), ("unstable", "remove_one", -1)):
    cid = f"M11.C.27[{twosome},{change}]"
    CRITERIA[cid] = C(cid, "composite", ("baseline", "treatment"), (Readout("pair_deviation", direction),), arm_c27,
                      settings={"t0": 8, "twosome": twosome, "change": change})
# M11.C.38: four basic levels, a monotone ordering, as three adjacent two-arm directions (M11.4).
for lo, hi in ((0.0, 5.0), (5.0, 10.0), (10.0, 15.0)):
    cid = f"M11.C.38[-{lo:g} vs -{hi:g}]"
    CRITERIA[cid] = C(cid, "premise", ("baseline", "treatment"), (Readout("time_to_threshold", -1),), arm_c1,
                      settings={"weeks": 104, "levels": (lo, hi)})
# M11.C.41: the 2x2 of level and stress, four two-arm directions on two readouts each.
for name, base, treat in (("light: lower level", (0.0, "light"), (10.0, "light")),
                          ("heavy: lower level", (0.0, "heavy"), (10.0, "heavy")),
                          ("higher level: heavier stress", (0.0, "light"), (0.0, "heavy")),
                          ("lower level: heavier stress", (10.0, "light"), (10.0, "heavy"))):
    cid = f"M11.C.41[{name}]"
    CRITERIA[cid] = C(cid, "mixed", ("baseline", "treatment"),
                      (Readout("mean_acute", +1), Readout("reactive_share", +1)), arm_c41,
                      settings={"weeks": 80, "arms": (base, treat)})

NOT_BUILT = {
    "M11.C.7": "needs M8.2/M8.3's position predicates; the criterion declares no direction for its topology arms",
    "M11.C.13": "needs incidents located in a community; the model has none",
    "M11.C.14": "needs M5.C.1's marital-distance gate, which is not built",
}
