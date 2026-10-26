"""The stock a cut-off tie holds (version 2 group G1, spec revision 36).
Purpose: test each rule of `M1.D.2b` as a direction on the Phase B family: the threat, the cap, decay to the floor,
         bridging by negative and positive contact, the faded cutoff that waits for a positive contact, and the
         invariant's new direction. Unit tests of set mechanisms with invented constants; none is a finding (`M11.5`).
Spec:    docs/bowen_agent_model_spec_v2.md#M1.D.2b, #M1.B.3c, #M6.4, #M1.D.2a
Tests:   this file
"""

from __future__ import annotations

import dataclasses
import math
import statistics

import pytest

from src.bowen.engine.cutoff_stock import (bridge_ties, cutoff_threat, decay_cutoff_stock, enter_declared_cutoffs,
                                           total_threat)
from src.bowen.engine.event_effects import apply_structural_event
from src.bowen.engine.events import Channel, Delivery, Event, EventId, Mechanism, Role, SourcePosition
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.invariants import InvariantViolation, assert_invariants, snapshot
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.moves import apply_move_effects
from src.bowen.engine.objects import TieState
from src.bowen.engine.rejection import fade_rejection, raise_rejection
from src.bowen.engine.state import new_run_state
from src.bowen.io.load import load_constants, load_event_kinds, load_family
from src.bowen.scenario.params import engine_params

P = PersonId
RAVI, MARTA, NADIA, PIA = P("ravi"), P("marta"), P("nadia"), P("pia")
MARITAL = TieId.of(RAVI, MARTA)
PARAMS = engine_params(load_constants())


def fresh():
    family = load_family()
    state = new_run_state(dict(family.people), dict(family.ties), family.family, load_event_kinds())
    return initialise_run(state, PARAMS)


def move(kind, sender, target, intensity=100.0, tick=0, index=0):
    return Event(
        id=EventId(tick, str(sender.value), index), kind=kind, mechanism=Mechanism.MOVE, sender=sender,
        targets=(target,), intensity=intensity, timestamp=tick, duration=1, exogenous=False,
        source_position=SourcePosition.NONE, channel=Channel.SCRIPTED,
    )


def batch_of(state, *events):
    batch = []
    for event in events:
        state.store.record_event(event)
        batch.append(Delivery(event.timestamp + 1, event.id, event.targets[0], Role.TARGET,
                              emitted_tick=event.timestamp, latency=1))
    return tuple(batch)


def refusal_of(state, event):
    state.store.record_event(event)
    return EffectRecord(state.tick, "refusal", event.id, acute_anxiety=((event.sender, 1.0),),
                        ties=((MARITAL, f"rejection:{event.targets[0]}", 0.1),))


def cut(state, stock=100.0, entered=True):
    """Marta cuts Ravi off; the tie holds ``stock``. ``entered``: step 4 has already entered the cut."""
    tie = state.ties[MARITAL]
    raise_rejection(tie, MARTA, 1.0)
    tie.distance_bound_anxiety = stock
    if entered:
        bridge_ties(state, (), (), PARAMS)
    return tie


# --- the threat (A1) ---------------------------------------------------------------------------


def test_m1d2b_threat_weighs_only_while_cut_off():
    """The same stock weighs on both members while the tie is cut off and on nobody while it is not."""
    state = fresh()
    tie = cut(state)
    before = {p: state.people[p].acute_anxiety for p in (RAVI, MARTA, NADIA)}
    records = cutoff_threat(state, PARAMS)
    assert [r.mechanism for r in records] == ["cutoff_threat"]
    assert state.people[RAVI].acute_anxiety > before[RAVI] and state.people[MARTA].acute_anxiety > before[MARTA]
    assert state.people[NADIA].acute_anxiety == before[NADIA]
    assert tie.distance_bound_anxiety == 100.0  # the threat does not draw the stock down
    open_state = fresh()
    open_state.ties[MARITAL].distance_bound_anxiety = 100.0  # the same stock, the tie ordinary
    before = {p: open_state.people[p].acute_anxiety for p in (RAVI, MARTA)}
    assert cutoff_threat(open_state, PARAMS) == []
    assert all(open_state.people[p].acute_anxiety == before[p] for p in before)


def test_m1d2b_threat_scales_with_bond_energy_and_spares_the_dead():
    low, high = fresh(), fresh()
    cut(low); cut(high)
    low.ties[MARITAL].bond_energy, high.ties[MARITAL].bond_energy = 20.0, 80.0
    base = {p: low.people[p].acute_anxiety for p in (RAVI, MARTA)}
    cutoff_threat(low, PARAMS); cutoff_threat(high, PARAMS)
    assert high.people[RAVI].acute_anxiety - base[RAVI] > low.people[RAVI].acute_anxiety - base[RAVI] > 0
    dead = fresh()
    cut(dead)
    dead.people[RAVI].alive = False
    ravi_before = dead.people[RAVI].acute_anxiety
    records = cutoff_threat(dead, PARAMS)
    assert dead.people[RAVI].acute_anxiety == ravi_before
    assert [pid for pid, _ in records[0].acute_anxiety] == [MARTA]


def test_m1d2b_the_weight_is_sized_to_one_divorce_over_five_years():
    """§3.7's sizing check: a tie cut at the cap at bond energy 100, never bridged, delivers about the Holmes–Rahe
    divorce (73 LCU × 6.9 = 504 intensity units) over the five years to its floor. ``[I]``: a sizing, not a source."""
    delivered = total_threat(PARAMS, PARAMS.cutoff_stock_cap, 100.0, PARAMS.cutoff_decay_weeks)
    assert abs(delivered - 504.0) / 504.0 < 0.01


# --- the cap (A2) --------------------------------------------------------------------------------


def test_m1d2b_stock_capped_and_distancer_keeps_the_rest():
    """A tie cut off holding more than the cap is cut to it at the cut; distancing into a full tie leaves the excess with
    the distancer."""
    state = fresh()
    tie = cut(state, stock=PARAMS.cutoff_stock_cap + 500.0, entered=False)
    records = bridge_ties(state, (), (), PARAMS)
    assert tie.distance_bound_anxiety == PARAMS.cutoff_stock_cap == tie.cutoff_peak
    assert [(r.mechanism, r.ties[0][2]) for r in records] == [("cutoff_cap_dissipation", -500.0)]
    ravi = state.people[RAVI]
    ravi.acute_anxiety = ravi.chronic_anxiety + 1000.0
    before = ravi.acute_anxiety
    assert apply_move_effects(state, batch_of(state, move("DISTANCE", RAVI, MARTA)), PARAMS) == []
    assert ravi.acute_anxiety == before and tie.distance_bound_anxiety == PARAMS.cutoff_stock_cap
    tie.distance_bound_anxiety = PARAMS.cutoff_stock_cap - 10.0  # room for ten: ten binds, the rest stays
    apply_move_effects(state, batch_of(state, move("DISTANCE", RAVI, MARTA, tick=1)), PARAMS)
    assert tie.distance_bound_anxiety == pytest.approx(PARAMS.cutoff_stock_cap)
    assert ravi.acute_anxiety == pytest.approx(before - 10.0)


# --- decay (A3) ----------------------------------------------------------------------------------


def test_m1d2b_decays_to_floor_and_stays():
    state = fresh()
    tie = cut(state, stock=1000.0)
    for _ in range(PARAMS.cutoff_decay_weeks):
        decay_cutoff_stock(state, PARAMS)
    floor = PARAMS.cutoff_floor_share * 1000.0
    assert tie.distance_bound_anxiety == pytest.approx(floor, rel=1e-9)
    for _ in range(PARAMS.cutoff_decay_weeks):
        assert decay_cutoff_stock(state, PARAMS) == []
    assert tie.distance_bound_anxiety == pytest.approx(floor, rel=1e-9)


def test_m1d2b_decay_only_on_a_cut_off_tie():
    state = fresh()
    state.ties[MARITAL].distance_bound_anxiety = 1000.0  # ordinary: distance-bound stock does not decay (M1.D.2a (a))
    assert decay_cutoff_stock(state, PARAMS) == [] and state.ties[MARITAL].distance_bound_anxiety == 1000.0


# --- bridging (A4) -------------------------------------------------------------------------------


def contacts(state, negative=False, positive=False, refused=False):
    events, refusals = [], []
    if negative:
        events.append(move("CONFLICT", RAVI, MARTA, index=0))
    if positive:
        events.append(move("STAY-IN-CONTACT", RAVI, MARTA, index=1))
    if refused:
        refusals.append(refusal_of(state, move("TRIANGLE", RAVI, MARTA, index=2)))
    return batch_of(state, *events), tuple(refusals)


@pytest.mark.parametrize("negative, positive, refused, expected", [
    (True, False, False, "peak"),      # a negative contact renews the stock to its peak
    (False, False, True, "peak"),      # a refusal is a negative contact whatever was refused
    (False, True, False, "lower"),     # a positive contact lets most of it go
    (True, True, False, "peak"),       # both in one week: the negative wins (the owner's Q6)
    (False, False, False, "same"),     # no contact: nothing
])
def test_m1d2b_bridging_negative_and_positive(negative, positive, refused, expected):
    state = fresh()
    tie = cut(state, stock=400.0)
    tie.distance_bound_anxiety = 200.0  # decayed since the peak
    batch, refusals = contacts(state, negative, positive, refused)
    records = bridge_ties(state, batch, refusals, PARAMS)
    if expected == "peak":
        assert tie.distance_bound_anxiety == 400.0
        assert [r.mechanism for r in records] == ["cutoff_threat_renewed"]
    elif expected == "lower":
        assert tie.distance_bound_anxiety == pytest.approx(200.0 * (1 - PARAMS.positive_bridge_dissipation))
        assert [r.mechanism for r in records] == ["cutoff_bridge_dissipation"]
    else:
        assert tie.distance_bound_anxiety == 200.0 and records == []
    assert tie.tie_state is TieState.CUT_OFF  # rejection is still held: no contact reopens it


def test_m1d2b_positive_contact_does_not_reopen_while_rejection_is_held():
    state = fresh()
    tie = cut(state)
    batch, refusals = contacts(state, positive=True)
    bridge_ties(state, batch, refusals, PARAMS)
    assert tie.rejected() and tie.tie_state is TieState.CUT_OFF


# --- the faded cutoff (A7) -----------------------------------------------------------------------


def test_m1d2b_faded_cutoff_waits_for_a_positive_contact():
    """Rejection fades to nothing; the tie stays cut off, keeps its stock and keeps weighing, until a positive contact
    is delivered — then it returns to its state before the cut, keeping what the contact left."""
    state = fresh()
    tie = cut(state)
    for _ in range(int(math.ceil(1.0 / PARAMS.rejection_fade)) + 1):
        fade_rejection(state, PARAMS)
    assert not tie.rejected() and tie.tie_state is TieState.CUT_OFF and tie.distance_bound_anxiety == 100.0
    assert cutoff_threat(state, PARAMS)
    batch, refusals = contacts(state, negative=True)
    bridge_ties(state, batch, refusals, PARAMS)
    assert tie.tie_state is TieState.CUT_OFF  # a negative contact does not reopen it
    batch, refusals = contacts(state, positive=True)
    records = bridge_ties(state, batch, refusals, PARAMS)
    assert tie.tie_state is TieState.ORDINARY and tie.cutoff_peak is None
    assert tie.distance_bound_anxiety == pytest.approx(100.0 * (1 - PARAMS.positive_bridge_dissipation))
    assert (MARITAL, "tie_state_ordinary", 1.0) in records[-1].ties
    assert cutoff_threat(state, PARAMS) == []


def test_m1d2b_a_distant_tie_reopens_distant():
    state = fresh()
    tie = state.ties[MARITAL]
    tie.tie_state = TieState.DISTANT
    tie.reference_level = tie.distance_bound_anxiety = 30.0
    cut(state, stock=30.0)
    assert tie.uncut_state is TieState.DISTANT
    tie.rejection.clear()
    batch, refusals = contacts(state, positive=True)
    bridge_ties(state, batch, refusals, PARAMS)
    # entering DISTANT again records the stock it reopens with as the reference level (M1.D.2c)
    assert tie.tie_state is TieState.DISTANT and tie.distance_bound_anxiety == pytest.approx(6.0)
    assert tie.reference_level == tie.distance_bound_anxiety


def test_m1d2c_stock_bound_during_the_cut_sets_the_reference_on_reopening():
    """Dirty state (P8): a distant tie (R = 30) is cut, binds far more while cut, and reopens distant. The reference
    level is the stock it reopens with, not the stale 30: a positive contact then removes a third of it and a negative
    adds one back, rather than removing 10 and adding nothing."""
    state = fresh()
    tie = state.ties[MARITAL]
    tie.tie_state = TieState.DISTANT
    tie.reference_level = tie.distance_bound_anxiety = 30.0
    cut(state, stock=30.0)
    tie.distance_bound_anxiety = 317.0  # bound by DISTANCE acts during the cut, under the cap
    tie.rejection.clear()
    bridge_ties(state, batch_of(state, move("STAY-IN-CONTACT", RAVI, MARTA, tick=1)), (), PARAMS)
    reopened = tie.distance_bound_anxiety
    assert tie.tie_state is TieState.DISTANT and reopened == pytest.approx(317.0 * (1 - PARAMS.positive_bridge_dissipation))
    assert tie.reference_level == reopened
    bridge_ties(state, batch_of(state, move("STAY-IN-CONTACT", RAVI, MARTA, tick=2)), (), PARAMS)
    assert tie.distance_bound_anxiety == pytest.approx(reopened * 2 / 3)
    bridge_ties(state, batch_of(state, move("CONFLICT", RAVI, MARTA, tick=3)), (), PARAMS)
    assert tie.distance_bound_anxiety == pytest.approx(reopened)


def test_m1d2b_a_declared_stock_above_the_cap_is_refused():
    """A family that declares a cut-off tie holding more than the cap is a file error: refused at initialisation,
    never dissipated with no record to carry it."""
    state = fresh()
    tie = state.ties[MARITAL]
    raise_rejection(tie, MARTA, 1.0)
    tie.cutoff_peak = None
    tie.distance_bound_anxiety = PARAMS.cutoff_stock_cap + 1.0
    with pytest.raises(ValueError, match="cutoff_stock_cap"):
        enter_declared_cutoffs(state, PARAMS)
    tie.distance_bound_anxiety = PARAMS.cutoff_stock_cap
    enter_declared_cutoffs(state, PARAMS)
    assert tie.cutoff_peak == PARAMS.cutoff_stock_cap


# --- re-entry: a dirty-state test (P8) -----------------------------------------------------------


def test_m1d2b_a_second_cutoff_enters_afresh():
    """A tie cut, reopened and cut again gets a new peak from its stock at the second cut, not the first's."""
    state = fresh()
    tie = cut(state, stock=900.0)
    assert tie.cutoff_peak == 900.0
    tie.rejection.clear()
    batch, refusals = contacts(state, positive=True)
    bridge_ties(state, batch, refusals, PARAMS)
    assert tie.tie_state is TieState.ORDINARY and tie.cutoff_peak is None
    left = tie.distance_bound_anxiety
    raise_rejection(tie, RAVI, 1.0)
    assert tie.tie_state is TieState.CUT_OFF and tie.cutoff_peak is None
    bridge_ties(state, (), (), PARAMS)
    assert tie.cutoff_peak == pytest.approx(left) and left < 900.0
    batch, refusals = contacts(state, negative=True)
    bridge_ties(state, batch, refusals, PARAMS)
    assert tie.distance_bound_anxiety == pytest.approx(left)  # renewed to the second peak, not the first


# --- reconciliation (A14) ------------------------------------------------------------------------


def test_m1d2b_reconciliation_returns_the_stock_and_clears_the_peak():
    state = fresh()
    tie = cut(state, stock=100.0)
    before = {p: state.people[p].acute_anxiety for p in (RAVI, MARTA)}
    event = Event(id=EventId(0, "script", 9), kind="RECONCILIATION", mechanism=Mechanism.RECONCILIATION, sender=None,
                  targets=tuple(sorted((RAVI, MARTA))), intensity=0.0, timestamp=0, duration=1, exogenous=True,
                  source_position=SourcePosition.NONE, channel=Channel.SCRIPTED, on_tie=MARITAL)
    state.store.record_event(event)
    apply_structural_event(state, event, PARAMS)
    assert tie.tie_state is TieState.ORDINARY and tie.cutoff_peak is None and tie.distance_bound_anxiety == 0.0
    assert state.people[RAVI].acute_anxiety - before[RAVI] == state.people[MARTA].acute_anxiety - before[MARTA] == 50.0


# --- the contact classes are configuration ---------------------------------------------------------


def test_m1d2b_contact_classes_come_from_config():
    """Which acts bridge negatively and positively is read from ``contact_classes.md``, not written in the code."""
    kinds = load_event_kinds()
    assert "CONFLICT" in kinds.negative_contacts and "CUTOFF" in kinds.negative_contacts
    assert "STAY-IN-CONTACT" in kinds.positive_contacts and "REDUCE_CUTOFF" in kinds.positive_contacts
    assert kinds.rebuffs == frozenset({"DISTANCE"})
    state = fresh()
    tie = cut(state, stock=400.0)
    tie.distance_bound_anxiety = 200.0
    swapped = dataclasses.replace(kinds, negative_contacts=frozenset({"STAY-IN-CONTACT"}),
                                  positive_contacts=frozenset({"CONFLICT"}))
    state = dataclasses.replace(state, kinds=swapped) if dataclasses.is_dataclass(state) else state
    object.__setattr__(state, "kinds", swapped)
    batch, refusals = contacts(state, negative=True)  # CONFLICT, now a positive contact under the swapped classes
    bridge_ties(state, batch, refusals, PARAMS)
    assert state.ties[MARITAL].distance_bound_anxiety == pytest.approx(200.0 * (1 - PARAMS.positive_bridge_dissipation))


def test_m1d2b_contact_classes_refuse_a_broken_table(tmp_path):
    """Milestone review MR-23: every refusal of ``load_contact_classes`` is reached by a broken copy of the committed
    table, in the style of ``test_config.py::test_s1_3b_an_undeclared_act_fails_to_load``: a set missing, listed twice
    or not one of the three; an act that is not a declared move; a set with no acts; an act in two sets."""
    from src.bowen.io.load import CONFIG_DIR, load_contact_classes
    from src.bowen.scenario.config_parse import ConfigError

    kinds = load_event_kinds()
    text = (CONFIG_DIR / "contact_classes.md").read_text(encoding="utf-8")
    rebuff = "| rebuff | DISTANCE |"
    broken = {
        "missing": ("\n".join(l for l in text.splitlines() if not l.startswith("| rebuff |")), "exactly"),
        "twice": (text + "| rebuff | DISTANCE | again |\n", "exactly"),
        "unknown set": (text.replace(rebuff, "| refusal | DISTANCE |"), "exactly"),
        "misspelt": (text.replace(rebuff, "| rebuff | DISTANSE |"), "not a declared move"),
        "empty": (text.replace(rebuff, "| rebuff |  |"), "no acts"),
        "overlap": (text.replace(rebuff, "| rebuff | CONFLICT |"), r"\['CONFLICT'\] in both negative and rebuff"),
    }
    assert rebuff in text and all(b != text for b, _ in broken.values())
    assert load_contact_classes(kinds)["rebuff"] == frozenset({"DISTANCE"})  # the committed table loads
    for name, (body, message) in broken.items():
        path = tmp_path / f"{name}.md"
        path.write_text(body, encoding="utf-8")
        with pytest.raises(ConfigError, match=message):
            load_contact_classes(kinds, path)


# --- the invariant's direction (M1.B.3c) ------------------------------------------------------------


def test_m1b3c_the_invariant_allows_a_cut_off_tie_with_no_rejection_but_not_the_reverse():
    state = fresh()
    tie = cut(state)
    before = snapshot(state)
    tie.rejection.clear()  # cut off, nothing held: allowed since revision 36

    def check():
        assert_invariants(state, before, frozenset(state.ties), ("standing_load", "deliver"), PARAMS.invariant_tolerance)

    check()
    tie.tie_state = TieState.ORDINARY
    tie.rejection[MARTA] = 0.5  # rejection held on a tie that is not cut off: never
    with pytest.raises(InvariantViolation, match="M1.B.3c"):
        check()
    tie.tie_state = TieState.CUT_OFF
    tie.cutoff_peak = tie.distance_bound_anxiety - 1.0  # stock above the peak: never
    with pytest.raises(InvariantViolation, match="M1.B.3c"):
        check()


# --- A12: the seed-7 release (version 1's week-74 transfer) no longer happens ------------------------------------


def _bridged_and_gained(records):
    """(ties that left `CUT_OFF`, people who gained anxiety in the same record), over a run's records."""
    reopened, gained = [], []
    for r in records:
        if isinstance(r, EffectRecord) and r.mechanism in ("cutoff_bridge_dissipation", "reduce_cutoff"):
            reopened += [t for t, f, _ in r.ties if f.startswith("tie_state_")]
            gained += [p for p, v in r.acute_anxiety if v > 0]
    return reopened, gained


def test_m1d2b_seed7_no_one_gains_the_stock_of_a_bridged_tie():
    """A12, a fixed-seed regression check (P23): in version 1 the seed-7 phase_c run released a reopened tie's stock
    to a third party at week 74 (`reduce_cutoff`'s transfer, D7). Under revision 36 a bridged tie's stock dissipates
    or stays in the tie; no person gains it. The run must contain a reopening, or the check reads nothing."""
    from src.bowen.ensemble.criteria import scenario, spell

    _, records = scenario("phase_c", 7, 104, spells=spell("phase_c", 104))
    reopened, gained = _bridged_and_gained(records)
    assert reopened, "seed 7 no longer reopens a cut-off tie in 104 weeks; pick a run that does"
    assert gained == []
    assert not any(isinstance(r, EffectRecord) and r.mechanism == "reduce_cutoff" and r.acute_anxiety for r in records)


# --- A6: the third party's cost is the outlet's loss (§3.5, D7) --------------------------------------------------


class ArmBroken(Exception):
    """An A6 arm was not built as declared. Not an ``AssertionError``, so the strict xfail on the direction comparison
    (``raises=AssertionError``) cannot absorb a broken arm (milestone review MR-24)."""


def _third_party_anxiety(seed: int, bridged: bool, weeks: int = 26, bridge_week: int = 2) -> float:
    """Marta has cut Ravi off (rejection faded, the tie still cut, holding stock); the couple's positive contacts are
    withheld and their tie held open every week, so the policy neither reopens nor re-cuts the tie on its own. In the bridged arm Ravi's STAY-IN-CONTACT is
    scripted at ``bridge_week`` and accepted. Returns the mean acute anxiety of the two children after that week.
    Raises ``ArmBroken`` when the arm is not what it declares: the tie not cut with its stock at the start, the
    scripted act not placed, or the tie not ``ORDINARY`` (bridged) or still ``CUT_OFF`` (left cut) after the week."""
    from src.bowen.ensemble.criteria import act, scenario

    def setup(state):
        tie = state.ties[MARITAL]
        raise_rejection(tie, MARTA, 1.0)
        tie.rejection = {}  # faded: the tie waits for a positive contact (M1.D.2b)
        tie.distance_bound_anxiety = 0.5 * PARAMS.cutoff_stock_cap
        bridge_ties(state, (), (), PARAMS)
        if not (tie.tie_state is TieState.CUT_OFF and tie.cutoff_peak == tie.distance_bound_anxiety):
            raise ArmBroken(f"seed {seed}: the tie is {tie.tie_state} with peak {tie.cutoff_peak} at the start")

    positives = ("STAY-IN-CONTACT", "REDUCE_CUTOFF")
    withheld = {t: {RAVI: [f"{k}>marta" for k in positives], MARTA: [f"{k}>ravi" for k in positives]}
                for t in range(weeks)}
    forced = act("ravi", "STAY-IN-CONTACT", "marta", bridge_week) if bridged else {}
    seen, after_bridge = [], []

    def watch(state):
        if state.tick == bridge_week + 2:  # read after the week the scripted act was delivered and bridged (or not)
            after_bridge.append(state.ties[MARITAL].tie_state)
        if state.tick > bridge_week + 1:
            seen.append(statistics.fmean(state.people[p].acute_anxiety for p in (NADIA, PIA)))

    scenario("phase_c", seed, weeks, setup=setup, forced=forced, withheld=withheld, watch=watch,
             held_open=(((RAVI, MARTA),), range(weeks)))  # neither may cut the tie again: one difference per seed
    if bridged:  # the policy may cut the tie again later; the arm's difference is the bridging itself
        placed = scenario.last_source.status.get((bridge_week, RAVI))
        if placed != "placed" or after_bridge != [TieState.ORDINARY]:
            raise ArmBroken(f"seed {seed}, bridged: the act is {placed!r}, the tie after the week {after_bridge}")
    elif after_bridge != [TieState.CUT_OFF]:
        raise ArmBroken(f"seed {seed}, left cut: the tie after the week {after_bridge}")
    return statistics.fmean(seen)


# 30 seeds, every one's arms built (seed 30's bridge does not take: the act is placed and the tie stays cut). Eight
# seeds passed by noise at stage 3 step 8 (+1.04 ± 1.03), so the test asserts beyond the half-width (the owner,
# 2026-10-26).
A6_SEEDS = range(30)


def test_m5b3_third_party_cost_arms_are_built_as_declared():
    """MR-24: the A6 arms are checked outside the strict xfail, so a broken arm fails here instead of reading as the
    expected failure. Every seed of both arms: the tie starts cut with its stock, and after the week it is ``ORDINARY``
    with the scripted act placed (bridged) or still ``CUT_OFF`` (left cut)."""
    for seed in A6_SEEDS:
        for bridged in (False, True):
            _third_party_anxiety(seed, bridged=bridged)  # raises ArmBroken on a broken arm


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="A6 not shown: over 30 seeds the children's mean anxiety after the bridging is 62.78 against "
                   "62.78 with the tie left cut, difference +0.002 ± 0.94 (2 se; stage 3 step 8, 2026-10-26; 62.5 "
                   "against 63.3 over 8 seeds on 2026-10-22). No rule is written to make it pass (§3.5); reported to "
                   "the owner in the revision 36 report.")
def test_m5b3_third_party_cost_is_the_lost_outlet():
    """A6 (§3.5): bridging a cut-off tie raises a third party's own anxiety through the outlet's loss — the expected
    route is triangle position (`M1.C.1`, `outsider_positional_gain`), not a transfer of the stock (D7). Two arms per
    seed, the same seed and legal sets, differing in one scripted accepted contact; the children's mean anxiety after
    it is compared seed by seed, and the mean difference must clear its 2-se half-width. No rule writes this cost; the direction did not hold, so the test is a strict
    expected failure: the day the model shows the cost, it fails as "unexpectedly passing" and the mark comes off.
    Only the direction comparison is expected to fail (``raises=AssertionError``); a broken arm raises ``ArmBroken``,
    which the mark does not absorb, and is checked on its own by
    ``test_m5b3_third_party_cost_arms_are_built_as_declared`` (MR-24)."""
    cut = [_third_party_anxiety(s, bridged=False) for s in A6_SEEDS]
    bridged = [_third_party_anxiety(s, bridged=True) for s in A6_SEEDS]
    differences = [b - c for b, c in zip(bridged, cut)]
    half_width = 2 * statistics.stdev(differences) / len(differences) ** 0.5  # paired, per seed
    assert statistics.fmean(differences) - half_width > 0, (statistics.fmean(differences), half_width)
