"""The external agent, landed contact and the delayed view (Phase C step 9).

Purpose: test M1.E's agent, its repertoire and thin tie, when its contact lands, and
         M16.D's delayed view, including that the event store is load-bearing.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.E.1, #M1.E.6, #M1.E.7, #M1.E.7c, #M1.E.7e, #M1.E.8, #M5.B.4, #M5.B.5, #M16.D.1, #M16.D.2, #M16.T.4, #M16.T.6
Tests:   this file

Unit tests of set mechanisms with invented constants; none is a finding (`M11.5`).
"""

from __future__ import annotations

import dataclasses
from unittest import mock

from src.bowen.engine.event_store import EventStore
from src.bowen.engine.events import Channel, Delivery, Event, EventId, Mechanism, Role, SourcePosition
from src.bowen.engine.external import landing_probability
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.visibility import HouseholdConductanceVisibility
from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.objects import Role as PersonRole
from src.bowen.engine.objects import SymptomChannel
from src.bowen.engine.observe import observe
from src.bowen.engine.state import canonical_state, new_run_state
from src.bowen.engine.tick import run
from src.bowen.io.load import CONFIG_DIR, load_constants, load_event_kinds, load_family, load_policy_rules, load_script
from src.bowen.policy.policy import EXTERNAL_REPERTOIRE, FAMILY_TO_EXTERNAL, legal_outcomes
from src.bowen.scenario.assemble import assemble
from src.bowen.scenario.params import engine_params
from src.bowen.scenario.policy_source import PolicySource
from src.bowen.scenario.scripted_source import ScriptedSource

P = PersonId
RAVI, MARTA, NADIA, HALIM = P("ravi"), P("marta"), P("nadia"), P("halim")
PARAMS = engine_params(load_constants())
KINDS = load_event_kinds()
PHASE_C = CONFIG_DIR / "family_phase_c.md"


def fresh():
    family = load_family(PHASE_C)
    state = new_run_state(dict(family.people), dict(family.ties), family.family, KINDS)
    return initialise_run(state, PARAMS)


def failing(state, pid=RAVI):
    state.people[pid].symptom_active[SymptomChannel.PHYSICAL] = True


def sent(state, sender, target, tick, index=0, route=()):
    event = Event(
        id=EventId(tick, str(sender.value), index), kind="STAY-IN-CONTACT", mechanism=Mechanism.MOVE, sender=sender,
        targets=(target,), intensity=50.0, timestamp=tick, duration=1, exogenous=False,
        source_position=SourcePosition.NONE, channel=Channel.SCRIPTED, route=route,
    )
    state.store.record_event(event)
    state.store.record_delivery(Delivery(tick + 1, event.id, target, Role.TARGET, emitted_tick=tick, latency=1))
    return event


class Collect:
    def __init__(self):
        self.records = []

    def emit(self, record):
        self.records.append(record)


def phase_c_run(weeks=80, seed=1, family_path=PHASE_C):
    constants, family = load_constants(), load_family(family_path)
    script = load_script(kinds=KINDS, family=load_family())
    events = ScriptedSource("events", weeks, script.events, ())
    parts = assemble(constants, KINDS, family, events, seed=seed)
    out = Collect()
    run(parts.state, PolicySource(events, parts.params, load_policy_rules()), parts.params, parts.visibility,
        parts.activation, out, ticks=weeks)
    return parts.state, out.records


# --- the agent and its repertoire --------------------------------------------------------


def test_m1e1_external_agent_is_a_person_with_a_restricted_repertoire_and_real_ties():
    state = fresh()
    assert state.people[HALIM].role is PersonRole.EXTERNAL and state.ties_of(HALIM)
    halim = legal_outcomes(observe(state, HALIM, PARAMS), KINDS, PARAMS)
    assert halim and {o.kind for o in halim} - {"WITHHOLD"} <= EXTERNAL_REPERTOIRE


def test_m5b4_family_to_external_moves_exist_and_go_only_to_an_external_agent():
    ravi = legal_outcomes(observe(fresh(), RAVI, PARAMS), KINDS, PARAMS)
    toward_halim = {o.kind for o in ravi if o.target == HALIM}
    assert FAMILY_TO_EXTERNAL <= toward_halim
    assert not any(o.kind in FAMILY_TO_EXTERNAL for o in ravi if o.target != HALIM)


def test_m1e7e_the_coach_tie_accumulates_no_investment():
    state, _ = phase_c_run(weeks=40)
    for tie_id in (TieId.of(HALIM, RAVI), TieId.of(HALIM, MARTA)):
        assert all(v == 0.0 for v in state.ties[tie_id].investment.values())
    assert any(v > 0 for v in state.ties[TieId.of(RAVI, MARTA)].investment.values())


def test_m1e6_the_agents_presence_changes_the_configuration():
    with_agent, _ = phase_c_run(weeks=40)
    without, _ = phase_c_run(weeks=40, family_path=CONFIG_DIR / "family_reduced.md")
    assert with_agent.people[RAVI].acute_anxiety != without.people[RAVI].acute_anxiety


# --- landing ------------------------------------------------------------------------------


def test_m1e7_contact_lands_only_when_binders_fail():
    state = fresh()
    coach, ravi = state.people[HALIM], state.people[RAVI]
    assert landing_probability(state, coach, ravi, PARAMS) == 0.0  # pain is necessary
    failing(state)
    assert landing_probability(state, coach, ravi, PARAMS) > 0.0


def test_m1e7_the_landing_rate_is_low():
    state = fresh()
    failing(state)
    assert landing_probability(state, state.people[HALIM], state.people[RAVI], PARAMS) <= 2 * PARAMS.landing_rate


def test_m1e7_coach_caught_in_the_anxiety_transmits_less():
    """M1.E.7c's third form: the agent's non-participation is what lands."""
    clear, caught = fresh(), fresh()
    for state in (clear, caught):
        failing(state)
    caught.people[HALIM].outside_ness_outward = 0.9
    assert landing_probability(caught, caught.people[HALIM], caught.people[RAVI], PARAMS) < landing_probability(
        clear, clear.people[HALIM], clear.people[RAVI], PARAMS)


def test_m1e8_more_contact_past_a_low_rate_lands_less():
    def p_after(contacts):
        state = fresh()
        failing(state)
        state.tick = PARAMS.contact_window
        for i in range(contacts):
            sent(state, HALIM, RAVI, state.tick - i, index=i)
        return landing_probability(state, state.people[HALIM], state.people[RAVI], PARAMS)
    low, high = PARAMS.contact_optimum, 4 * PARAMS.contact_optimum
    assert p_after(high) < p_after(low)
    assert high * p_after(high) < low * p_after(low)  # more coaching is worse in total, not just per contact


def test_m1e7c_delayed_self_observation_makes_landing_likelier():
    plain, with_log = fresh(), fresh()
    for state in (plain, with_log):
        failing(state)
        state.tick = PARAMS.delayed_view_weeks + 5
    sent(with_log, RAVI, MARTA, tick=1)  # Ravi's own move, older than the delay
    assert landing_probability(with_log, with_log.people[HALIM], with_log.people[RAVI], PARAMS) > landing_probability(
        plain, plain.people[HALIM], plain.people[RAVI], PARAMS)


def test_m1e7_perspective_rises_only_on_a_landed_contact_or_own_practice():
    """Every rise in perspective is logged as a landed contact or as the person's own practice (revision 18); none
    is spontaneous."""
    def sources(records):
        totals = {}
        for r in records:
            if isinstance(r, EffectRecord) and r.mechanism in ("landed_contact", "self_practice"):
                for pid, field, v in r.people:
                    if field != "systems_perspective":
                        continue
                    totals.setdefault(r.mechanism, {}).setdefault(pid, 0.0)
                    totals[r.mechanism][pid] += v
        return totals

    with mock.patch("src.bowen.engine.external.binders_failing", lambda person, params: True):
        state, records = phase_c_run(weeks=120, seed=2)
    found = sources(records)
    for pid, person in state.people.items():
        if person.role is PersonRole.EXTERNAL:
            continue
        logged = found.get("landed_contact", {}).get(pid, 0.0) + found.get("self_practice", {}).get(pid, 0.0)
        assert abs((person.systems_perspective or 0.0) - logged) < 1e-9, pid
    assert found.get("landed_contact"), "no contact landed in 120 weeks with every binder failing"
    without_agent, records = phase_c_run(weeks=120, seed=2, family_path=CONFIG_DIR / "family_reduced.md")
    practice = sources(records).get("self_practice", {})
    assert all(abs((p.systems_perspective or 0.0) - practice.get(pid, 0.0)) < 1e-9
               for pid, p in without_agent.people.items())  # without a coach, only practice moves it


# --- the delayed view (M16.D) --------------------------------------------------------------


def test_m16t4_delayed_view_is_scoped_and_lagged():
    state = fresh()
    old_own = sent(state, RAVI, MARTA, tick=1, index=0)
    old_received = sent(state, MARTA, RAVI, tick=2, index=0)
    new_own = sent(state, RAVI, MARTA, tick=40, index=0)
    others = sent(state, MARTA, NADIA, tick=3, index=0)
    private_hop = sent(state, MARTA, NADIA, tick=4, index=1, route=(RAVI,))
    # Ravi overhears the routed hop: still another agent's private hop, so never in his view.
    state.store.record_delivery(Delivery(5, private_hop.id, RAVI, Role.WITNESS, emitted_tick=4, latency=1))
    new_received = sent(state, MARTA, RAVI, tick=39, index=1)
    view = state.store.delayed_view(RAVI, now=40, delay=PARAMS.delayed_view_weeks)
    assert old_own in view and old_received in view
    assert new_own not in view and new_received not in view
    assert others not in view and private_hop not in view


def test_m16d2_default_delay_is_six_months_and_zero_is_only_an_arm():
    assert PARAMS.delayed_view_weeks == 26
    state = fresh()
    newest = sent(state, RAVI, MARTA, tick=10)
    assert newest in state.store.delayed_view(RAVI, now=10, delay=0)  # an experimental arm may set it


def test_m16t6_event_store_is_load_bearing():
    """With the store emptied by an override declared here — never by config — runs diverge.

    Both arms hold every family member's binders failing (a second declared override), so
    that contacts can land at all and the only difference between the arms is the store.
    """
    def arms(seeds):
        return [canonical_state(phase_c_run(weeks=100, seed=s)[0]) for s in seeds]

    with mock.patch("src.bowen.engine.external.binders_failing", lambda person, params: True):
        live = arms(range(8))
        with mock.patch.object(EventStore, "delayed_view", lambda self, person, now, delay: ()):
            emptied = arms(range(8))
    assert any(a != b for a, b in zip(live, emptied))


def test_m1e7_a_self_directed_act_that_worked_builds_perspective():
    """Revision 19: perspective is self-discovered. A self-directed act under load counts only when it worked — the
    person held their position (no soothing toward that person) and their reactivity on the tie came down over
    practice_window weeks. Feeling better at once does not count; an automatic act never counts."""
    from src.bowen.engine.act import Selection
    from src.bowen.engine.external import self_practice

    ravi, marta = P("ravi"), P("marta")
    tie_id = TieId.of(ravi, marta)
    window = PARAMS.practice_window

    def judged(after):
        state = fresh()
        state.ties[tie_id].felt_impingement[ravi] = 0.6           # reactive on the tie when he acts
        self_practice(state, (Selection(ravi, "STAY-IN-CONTACT", (marta,), 100.0),), {ravi: PARAMS.anxiety_band_low + 5.0}, PARAMS)
        assert (state.people[ravi].systems_perspective or 0) == 0  # nothing at once
        after(state)
        state.tick = window
        self_practice(state, (), {}, PARAMS)
        return state.people[ravi].systems_perspective or 0

    def calmed(state):
        state.ties[tie_id].felt_impingement[ravi] = 0.0

    def calmed_but_soothed(state):
        calmed(state)
        state.tick = 1
        from src.bowen.engine.act import act
        act(state, Selection(ravi, "DISTANCE", (marta,), 100.0), HouseholdConductanceVisibility(PARAMS.per_hop_fidelity))

    assert judged(calmed) == PARAMS.self_practice_gain
    assert judged(lambda state: None) == 0                         # the tie did not calm
    assert judged(calmed_but_soothed) == 0                         # he did not hold his position
    automatic = fresh()
    self_practice(automatic, (Selection(ravi, "PURSUE", (marta,), 100.0),), {ravi: PARAMS.anxiety_band_low + 5.0}, PARAMS)
    assert automatic.practice == []
    assert PARAMS.self_practice_gain < PARAMS.perspective_gain


def test_m1e7_when_a_held_stance_works_the_other_functions_up():
    """Revision 19 (the owner: "if the person doesn't change back, the system adjusts — the other agents realize that
    they aren't going to win and improve their functioning"): only when the stance worked."""
    from src.bowen.engine.act import Selection
    from src.bowen.engine.external import self_practice

    ravi, marta = P("ravi"), P("marta")
    tie_id = TieId.of(ravi, marta)

    def marta_after(calmed):
        state = fresh()
        state.ties[tie_id].felt_impingement[ravi] = 0.6
        self_practice(state, (Selection(ravi, "WITHHOLD", (marta,), 100.0),), {ravi: PARAMS.anxiety_band_low + 5.0}, PARAMS)
        before = state.people[marta].functional_level
        if calmed:
            state.ties[tie_id].felt_impingement[ravi] = 0.0
        state.tick = PARAMS.practice_window
        self_practice(state, (), {}, PARAMS)
        return state.people[marta].functional_level - before

    assert marta_after(True) == PARAMS.system_adjust_gain > 0 == marta_after(False)


def test_m1e7_the_system_adjusts_once_per_stance_and_only_under_load():
    """Review 2026-10-11: weekly acts toward one person open overlapping windows; the other adjusts once per stance,
    not once per act. And an act below the calm band is not practice under load."""
    from src.bowen.engine.act import Selection
    from src.bowen.engine.external import self_practice

    ravi, marta = P("ravi"), P("marta")
    tie_id = TieId.of(ravi, marta)
    state = fresh()
    state.ties[tie_id].felt_impingement[ravi] = 0.6
    loaded = {ravi: PARAMS.anxiety_band_low + 5.0}
    for week in range(3):                                          # three acts in a row
        state.tick = week
        self_practice(state, (Selection(ravi, "STAY-IN-CONTACT", (marta,), 100.0),), loaded, PARAMS)
    state.ties[tie_id].felt_impingement[ravi] = 0.0
    before = state.people[marta].functional_level
    for week in range(3, 3 + PARAMS.practice_window):
        state.tick = week
        self_practice(state, (), {}, PARAMS)
    assert state.people[marta].functional_level - before == PARAMS.system_adjust_gain
    calm = fresh()
    self_practice(calm, (Selection(ravi, "STAY-IN-CONTACT", (marta,), 100.0),), {ravi: PARAMS.anxiety_band_low / 2},
                  PARAMS)
    assert calm.practice == []
