"""Cutoff and focus — spec revision 29 (plan §4.8, steps R2 to R5).

Purpose: pin each revision-29 mechanism to its requirement: a reactive cutoff's cost to its holder and a principled
         cutoff's exemption (M1.B.3d), learning from what follows a witnessed cutoff (M4.D.6f), a parent's focus on a
         child's real or perceived weakness (M1.B.8a), and focus steering projection-type acts (M4.D.1g).
Spec:    docs/bowen_agent_model_spec_v2.md#M1.B.3d, #M4.D.6f, #M1.B.8a, #M4.D.1g, #M4.C.1c, #M6.4
Tests:   this file
"""

from __future__ import annotations

import dataclasses

import pytest

from src.bowen.engine.contact import deviation, too_little
from src.bowen.engine.events import Channel, Delivery, Event, EventId, Mechanism, Role, SourcePosition
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.invariants import LEDGER, SOURCE
from src.bowen.engine.learner import learn_from_witnessed, note_witnessed_cutoffs
from src.bowen.engine.observe import observe
from src.bowen.engine.reactive import weakness_focus
from src.bowen.engine.sinks import PROJECTION_KINDS
from src.bowen.engine.rejection import raise_rejection, reactive_cutoff_cost
from src.bowen.engine.state import new_run_state
from src.bowen.io.load import CONFIG_DIR, load_constants, load_event_kinds, load_family, read_text
from src.bowen.scenario.config_parse import ConfigError
from src.bowen.scenario.family import build_family
from src.bowen.policy.policy import channel_weights, legal_outcomes
from src.bowen.scenario.params import engine_params

P = PersonId
FAMILY_C = CONFIG_DIR / "family_phase_c.md"
RAVI, MARTA, ANA, BRUNO = P("ravi"), P("marta"), P("ana"), P("bruno")
MARITAL, ANA_BRUNO = TieId.of(RAVI, MARTA), TieId.of(ANA, BRUNO)
PARAMS = engine_params(load_constants())


def fresh(params=PARAMS):
    family = load_family(FAMILY_C)
    state = new_run_state(dict(family.people), dict(family.ties), family.family, load_event_kinds())
    return initialise_run(state, params)


def cost(state, params=PARAMS):
    rows = {pid: v for r in reactive_cutoff_cost(state, params) for pid, v in r.acute_anxiety}
    return rows


# --- R2: reactive and principled cutoff (M1.B.3d) ------------------------------------------------------------------


def test_m1b3d_reactive_cutoff_raises_the_cutters_anxiety():
    held, open_ = fresh(), fresh()
    raise_rejection(held.ties[MARITAL], RAVI, 1.0)
    paid = cost(held)
    expected = PARAMS.reactive_cutoff_gain * 1.0 * (100 - held.people[RAVI].basic_level) / 100
    assert paid[RAVI] == pytest.approx(expected) and expected > 0
    assert MARTA not in paid  # only the person who holds the rejection pays
    assert RAVI not in cost(open_)  # the arm without the cutoff pays nothing on that tie


def test_m1b3d_a_less_differentiated_cutter_pays_more():
    low, high = fresh(), fresh()
    low.people[RAVI].basic_level, high.people[RAVI].basic_level = 25.0, 60.0
    for state in (low, high):
        raise_rejection(state.ties[MARITAL], RAVI, 1.0)
    assert cost(low)[RAVI] > cost(high)[RAVI]


def test_m1b3d_principled_cutoff_raises_nothing_and_feels_no_loss():
    reactive, principled = fresh(), fresh()
    for state in (reactive, principled):
        tie = state.ties[MARITAL]
        raise_rejection(tie, RAVI, 1.0)
        tie.felt_contact = {RAVI: 0.0, MARTA: 0.0}
    principled.ties[MARITAL].principled = (RAVI,)
    assert RAVI in cost(reactive) and RAVI not in cost(principled)
    assert too_little(reactive.people[RAVI], reactive.ties[MARITAL], PARAMS) > 0
    assert too_little(principled.people[RAVI], principled.ties[MARITAL], PARAMS) == 0
    assert deviation(principled.people[RAVI], principled.ties[MARITAL], PARAMS) < deviation(
        reactive.people[RAVI], reactive.ties[MARITAL], PARAMS)


def test_m1b3d_only_the_declared_member_is_exempt():
    state = fresh()
    tie = state.ties[MARITAL]
    raise_rejection(tie, RAVI, 1.0)
    raise_rejection(tie, MARTA, 1.0)
    tie.felt_contact = {RAVI: 0.0, MARTA: 0.0}
    tie.principled = (RAVI,)
    assert MARTA in cost(state) and RAVI not in cost(state)
    assert too_little(state.people[MARTA], tie, PARAMS) > 0  # the other still feels the cut


def test_m1b3d_only_the_family_config_declares_a_principled_cutoff():
    """Nothing but the family config sets a principled cutoff: the declared family has none, and a config line that
    names a tie that is not cut off, or a person not on it, is refused."""
    assert all(not t.principled for t in load_family(FAMILY_C).ties.values())
    text = read_text(FAMILY_C)
    with_line = text.replace("undifferentiation_budget: 90\n", "undifferentiation_budget: 90\nprincipled_cutoffs: ana>bruno\n")
    family = build_family(with_line, load_constants())
    assert family.ties[ANA_BRUNO].principled == (ANA,)
    for bad in ("ravi>marta", "ravi>bruno"):
        broken = text.replace("undifferentiation_budget: 90\n", f"undifferentiation_budget: 90\nprincipled_cutoffs: {bad}\n")
        with pytest.raises(ConfigError, match="principled"):
            build_family(broken, load_constants())


def test_m1b3d_a_declared_principled_cutoff_is_held_by_its_holder_only():
    """The declaration names who cut off, so only they start rejecting; an ordinary declared cutoff names nobody, so both
    do. The person cut off then pays no reactive cost for a cutoff they did not make."""
    text = read_text(FAMILY_C).replace("undifferentiation_budget: 90\n",
                                       "undifferentiation_budget: 90\nprincipled_cutoffs: ana>bruno\n")
    family = build_family(text, load_constants())
    state = initialise_run(new_run_state(dict(family.people), dict(family.ties), family.family, load_event_kinds()),
                           PARAMS)
    assert state.ties[ANA_BRUNO].rejection.get(ANA, 0.0) > 0 and state.ties[ANA_BRUNO].rejection.get(BRUNO, 0.0) == 0
    assert BRUNO not in cost(state) and ANA not in cost(state)
    ordinary = fresh()
    assert all(ordinary.ties[ANA_BRUNO].rejection[m] > 0 for m in (ANA, BRUNO))


def principled_ana(state):
    """Ana holds a principled cutoff of Bruno, as the family config would declare it."""
    tie = state.ties[ANA_BRUNO]
    tie.rejection[BRUNO] = 0.0
    tie.principled = (ANA,)
    return tie


def test_m1b3d_a_policy_cutoff_after_a_principled_one_is_reactive():
    """Ana's own policy cuts Bruno off while her declared cutoff is still held: from then on she pays the reactive cost
    and feels the loss, like anyone whose cutoff the policy selected."""
    from src.bowen.engine.event_effects import apply_delivered_cutoffs
    declared, renewed = fresh(), fresh()
    for state in (declared, renewed):
        principled_ana(state)
    event = Event(id=EventId(0, "ana", 0), kind="CUTOFF", mechanism=Mechanism.MOVE, sender=ANA, targets=(BRUNO,),
                  intensity=100.0, timestamp=0, duration=1, exogenous=False, source_position=SourcePosition.NONE,
                  channel=Channel.AUTOMATIC)
    renewed.store.record_event(event)
    apply_delivered_cutoffs(renewed, (Delivery(0, event.id, BRUNO, Role.TARGET, 0, 0),), PARAMS)
    assert ANA not in cost(declared) and ANA in cost(renewed)
    assert renewed.ties[ANA_BRUNO].principled == ()


def test_m1b3d_a_faded_principled_cutoff_does_not_return_as_principled():
    """The principled kind belongs to the declared rejection: once it has faded to zero, a later rejection is reactive."""
    from src.bowen.engine.rejection import lower_rejection
    state = fresh()
    tie = principled_ana(state)
    lower_rejection(tie, ANA, 1.0)
    assert tie.principled == ()
    raise_rejection(tie, ANA, 0.5)  # say a refusal raises it again
    assert ANA in cost(state)


def test_m1b3d_a_reunion_ends_the_principled_kind():
    """A reconciliation clears the tie's rejection, the declared one included, so its kind goes with it."""
    from src.bowen.engine.event_effects import apply_structural_event
    state = fresh()
    tie = principled_ana(state)
    event = Event(id=EventId(0, "reunion", 0), kind="RECONCILIATION", mechanism=Mechanism.RECONCILIATION, sender=None,
                  targets=(), intensity=0.0, timestamp=0, duration=1, exogenous=True,
                  source_position=SourcePosition.NONE, channel=Channel.SCRIPTED, on_tie=ANA_BRUNO)
    apply_structural_event(state, event, PARAMS)
    assert tie.rejection.get(ANA, 0.0) == 0 and tie.principled == ()


def test_m64_reactive_cutoff_rise_is_logged_by_name():
    state = fresh()
    raise_rejection(state.ties[MARITAL], RAVI, 1.0)
    records = reactive_cutoff_cost(state, PARAMS)
    assert [r.mechanism for r in records] == ["reactive_cutoff"]
    assert LEDGER["reactive_cutoff"] == SOURCE  # a source in M6.4's table, so the invariants accept it


def test_m4c1c_no_too_little_term_on_a_principled_cutters_tie():
    """The standing load and the reconnection pull are the same "too little" term (M4.C.1c): with the cutoff held as
    principled it is zero for its holder, and it returns once the rejection is gone."""
    state = fresh()
    tie = state.ties[MARITAL]
    tie.felt_contact = {RAVI: 0.0, MARTA: 0.0}
    tie.principled = (RAVI,)
    raise_rejection(tie, RAVI, 1.0)
    assert too_little(state.people[RAVI], tie, PARAMS) == 0
    tie.rejection[RAVI] = 0.0
    assert too_little(state.people[RAVI], tie, PARAMS) > 0


# --- R3: learning from what follows a witnessed cutoff (M4.D.6f) ----------------------------------------------------

NADIA, PIA = P("nadia"), P("pia")


def witnessed_cutoff(state, cutter=RAVI, cut=MARTA, witness=NADIA):
    """A CUTOFF from ``cutter`` to ``cut`` delivered to ``witness`` as a witness, this week."""
    event = Event(id=EventId(state.tick, cutter.value, 0), kind="CUTOFF", mechanism=Mechanism.MOVE, sender=cutter,
                  targets=(cut,), intensity=100.0, timestamp=state.tick, duration=1, exogenous=False,
                  source_position=SourcePosition.NONE, channel=Channel.SCRIPTED)
    state.store.record_event(event)
    note_witnessed_cutoffs(state, (Delivery(state.tick, event.id, witness, Role.WITNESS, state.tick, 0),))


def read_after(state, change: dict, params=PARAMS):
    """Move functional levels by ``change`` and run to the end of the window."""
    for pid, d in change.items():
        state.people[pid].functional_level += d
    state.tick += params.witness_window
    return learn_from_witnessed(state, params)


def cutoff_values(state, witness=NADIA):
    return {k: v for k, v in state.people[witness].learned_values.items() if k.split("|")[2] == "CUTOFF"}


def nadias_family(state):
    return [t.id.other(NADIA) for t in state.ties_of(NADIA) if state.people[t.id.other(NADIA)].role.value == "member"]


def test_m4d6f_a_family_that_functions_better_after_a_cutoff_teaches_cutoff():
    state = fresh()
    witnessed_cutoff(state)
    read_after(state, {p: +5.0 for p in nadias_family(state)})
    values = cutoff_values(state)
    assert values and all(v > 0 for v in values.values())
    assert {k.split("|")[3] for k in values} == {p.value for p in nadias_family(state)}  # toward each family tie


def test_m4d6f_a_family_that_functions_worse_teaches_against_it():
    """Adversarial: the same witnessed cutoff, opposite outcomes, opposite learning."""
    better, worse = fresh(), fresh()
    for state, d in ((better, +5.0), (worse, -5.0)):
        witnessed_cutoff(state)
        read_after(state, {p: d for p in nadias_family(state)})
    assert all(v > 0 for v in cutoff_values(better).values())
    assert all(v < 0 for v in cutoff_values(worse).values())


def test_m4d6f_a_principled_cutoff_that_lets_the_cutter_function_up_teaches_it_is_good():
    state = fresh()
    raise_rejection(state.ties[MARITAL], RAVI, 1.0)
    state.ties[MARITAL].principled = (RAVI,)
    witnessed_cutoff(state)
    read_after(state, {RAVI: +10.0})
    assert all(v > 0 for v in cutoff_values(state).values())


def test_m4d6f_closer_ties_weigh_more():
    """The cutter functions up and the person cut off functions down by as much: the witness learns for cutoff when
    its tie to the cutter is the closer one, and against it when its tie to the person cut off is."""
    to_cutter, to_cut = fresh(), fresh()
    to_cutter.ties[TieId.of(NADIA, RAVI)].bond_energy, to_cutter.ties[TieId.of(NADIA, MARTA)].bond_energy = 80.0, 20.0
    to_cut.ties[TieId.of(NADIA, RAVI)].bond_energy, to_cut.ties[TieId.of(NADIA, MARTA)].bond_energy = 20.0, 80.0
    for state in (to_cutter, to_cut):
        witnessed_cutoff(state)
        read_after(state, {RAVI: +5.0, MARTA: -5.0})
    assert all(v > 0 for v in cutoff_values(to_cutter).values())
    assert all(v < 0 for v in cutoff_values(to_cut).values())


def test_m4d6f_the_kind_is_not_read():
    reactive, principled = fresh(), fresh()
    for state in (reactive, principled):
        raise_rejection(state.ties[MARITAL], RAVI, 1.0)
    principled.ties[MARITAL].principled = (RAVI,)
    for state in (reactive, principled):
        witnessed_cutoff(state)
        read_after(state, {RAVI: +4.0, MARTA: -1.0})
    assert cutoff_values(reactive) == cutoff_values(principled)


def test_m4d6f_nothing_is_learned_before_the_window_ends():
    state = fresh()
    witnessed_cutoff(state)
    state.tick += PARAMS.witness_window - 1
    assert learn_from_witnessed(state, PARAMS) == [] and cutoff_values(state) == {}


def test_m4d6f_the_value_takes_a_bounded_step_toward_the_change_seen():
    """The update is value + gain × (seen − value): a value already at the change seen does not move, so repeated
    witnessing of the same outcome converges rather than growing without bound."""
    state = fresh()
    family = nadias_family(state)
    witnessed_cutoff(state)
    read_after(state, {p: +5.0 for p in family})
    first = cutoff_values(state)
    assert all(v == pytest.approx(PARAMS.witnessed_cutoff_gain * 5.0) for v in first.values())
    for key in first:
        state.people[NADIA].learned_values[key] = 5.0
    witnessed_cutoff(state)
    read_after(state, {p: +5.0 for p in family})
    assert all(v == pytest.approx(5.0) for v in cutoff_values(state).values())


def test_m4d6f_only_a_witnessed_cutoff_is_noted():
    """A CUTOFF delivered to its target, or another kind delivered to a witness, leaves no record."""
    state = fresh()
    event = Event(id=EventId(state.tick, "ravi", 0), kind="CUTOFF", mechanism=Mechanism.MOVE, sender=RAVI,
                  targets=(MARTA,), intensity=100.0, timestamp=state.tick, duration=1, exogenous=False,
                  source_position=SourcePosition.NONE, channel=Channel.SCRIPTED)
    other = dataclasses.replace(event, id=EventId(state.tick, "ravi", 1), kind="DISTANCE")
    for e in (event, other):
        state.store.record_event(e)
    note_witnessed_cutoffs(state, (Delivery(state.tick, event.id, MARTA, Role.TARGET, state.tick, 0),
                                   Delivery(state.tick, other.id, NADIA, Role.WITNESS, state.tick, 0)))
    assert state.witnessed == []
    note_witnessed_cutoffs(state, (Delivery(state.tick, event.id, NADIA, Role.WITNESS, state.tick, 0),))
    assert [w.witness for w in state.witnessed] == [NADIA]


def test_m4d6f_a_child_at_home_witnesses_a_parents_cutoff():
    """Plan §4.8's risk: Ravi cuts off his brother Dario, who lives elsewhere; the children at home witness it."""
    from src.bowen.ensemble.criteria import FAMILIES
    from src.bowen.scenario.assemble import assemble
    from src.bowen.scenario.scripted_source import ScriptedSource
    built = assemble(load_constants(), load_event_kinds(), load_family(FAMILIES["phase_d"]),
                     ScriptedSource("x", 10, (), ()), seed=1)
    event = Event(id=EventId(0, "ravi", 0), kind="CUTOFF", mechanism=Mechanism.MOVE, sender=RAVI, targets=(P("dario"),),
                  intensity=100.0, timestamp=0, duration=1, exogenous=False, source_position=SourcePosition.NONE,
                  channel=Channel.SCRIPTED)
    seen = built.visibility.resolve(event, built.state.people, built.state.ties)
    witnesses = {d.recipient for d in seen.deliveries if d.role is Role.WITNESS}
    assert {P("leo"), NADIA, PIA} <= witnesses


# --- R4: focus on a child's weakness, real or perceived (M1.B.8a) ---------------------------------------------------

RAVI_NADIA, RAVI_PIA = TieId.of(RAVI, NADIA), TieId.of(RAVI, PIA)


def anxious_parents(state, above=10.0):
    """Both parents at their usual level of 0, carrying ``above`` this week (the load at selection)."""
    for pid in (RAVI, MARTA):
        state.people[pid].usual_excess = 0.0
    return {RAVI: above, MARTA: above}


def focus_gained(state, load, params=PARAMS):
    before = {t: state.ties[t].investment.get(RAVI, 0.0) for t in (RAVI_NADIA, RAVI_PIA)}
    weakness_focus(state, load, params)
    return {t: state.ties[t].investment.get(RAVI, 0.0) - before[t] for t in before}


def test_m1b8a_perceived_weakness_adds_focus_only_while_anxious():
    calm, anxious = fresh(), fresh()
    for state in (calm, anxious):
        state.people[NADIA].functional_level, state.people[PIA].functional_level = 30.0, 40.0
    gained_calm = focus_gained(calm, anxious_parents(calm, above=0.0))
    gained = focus_gained(anxious, anxious_parents(anxious))
    assert gained_calm == {RAVI_NADIA: 0.0, RAVI_PIA: 0.0}
    expected = PARAMS.perceived_weakness_gain * 10.0 / 100 * 10.0  # load above usual 10, shortfall 10
    assert gained[RAVI_NADIA] == pytest.approx(expected) and gained[RAVI_PIA] == 0.0


def test_m1b8a_equal_children_get_equal_focus():
    """Adversarial: no child is named — children functioning alike gain nothing, however anxious the parent."""
    state = fresh()
    state.people[NADIA].functional_level = state.people[PIA].functional_level = 35.0
    assert focus_gained(state, anxious_parents(state, above=50.0)) == {RAVI_NADIA: 0.0, RAVI_PIA: 0.0}


def test_m1b8a_a_declared_weakness_adds_focus_while_a_parent_is_anxious():
    """A real weakness is part of what the anxious parent sees, from the first week (t0), even among equal children; a
    calm parent's focus does not move for it."""
    calm, anxious = fresh(), fresh()
    for state in (calm, anxious):
        state.people[NADIA].functional_level = state.people[PIA].functional_level = 35.0
        state.people[PIA].weakness = 0.2
    assert focus_gained(calm, anxious_parents(calm, above=0.0)) == {RAVI_NADIA: 0.0, RAVI_PIA: 0.0}
    gained = focus_gained(anxious, anxious_parents(anxious))
    assert gained[RAVI_PIA] == pytest.approx(PARAMS.perceived_weakness_gain * 10.0 / 100 * 20.0)
    assert gained[RAVI_NADIA] == 0.0


def test_m1b8a_the_pressure_is_the_load_above_usual_not_the_load():
    """Adversarial: a parent whose usual level is already high is not more anxious for carrying it; only the load above
    their usual level counts."""
    state = fresh()
    state.people[NADIA].functional_level, state.people[PIA].functional_level = 30.0, 40.0
    for pid in (RAVI, MARTA):
        state.people[pid].usual_excess = 20.0
    gained = focus_gained(state, {RAVI: 30.0, MARTA: 30.0})
    assert gained[RAVI_NADIA] == pytest.approx(PARAMS.perceived_weakness_gain * 10.0 / 100 * 10.0)


def test_m1b8a_an_only_child_gains_no_perceived_focus():
    state = fresh()
    state.people[PIA].financially_dependent = False  # Pia no longer at home: Nadia is the only dependent child
    state.people[NADIA].functional_level, state.people[PIA].functional_level = 20.0, 60.0
    assert focus_gained(state, anxious_parents(state))[RAVI_NADIA] == 0.0


def test_m7e1c_a_declared_weakness_is_read_from_the_family_config():
    text = read_text(FAMILY_C)
    family = build_family(text.replace("undifferentiation_budget: 90\n", "undifferentiation_budget: 90\nweaknesses: pia=0.3\n"),
                          load_constants())
    assert family.people[PIA].weakness == 0.3 and family.people[NADIA].weakness == 0.0
    for bad in ("pia=1.5", "nobody=0.2", "pia"):
        with pytest.raises(ConfigError, match="weakness"):
            build_family(text.replace("undifferentiation_budget: 90\n", f"undifferentiation_budget: 90\nweaknesses: {bad}\n"),
                         load_constants())


# --- R5: focus steers projection-type acts (M4.D.1g) ---------------------------------------------------------------


def toward(person=RAVI, focus=0.0, params=PARAMS):
    """Ravi's selection weights with ``focus`` of his attention on his tie to Nadia, the rest on his other ties."""
    state = fresh(params)
    for tie in state.ties_of(person):
        tie.investment[person] = 1.0
    state.ties[RAVI_NADIA].investment[person] = 1.0 + focus
    obs = observe(state, person, params)
    outcomes = legal_outcomes(obs, load_event_kinds(), params)
    weights = channel_weights(outcomes, obs, load_event_kinds(), params, steered=True)
    return {o.label: w for o, w in zip(outcomes, weights)}, {o.label: o for o in outcomes}


def test_m4d1g_focus_raises_the_chance_an_act_lands_on_that_tie():
    plain_, labels = toward()
    focused, _ = toward(focus=20.0)
    steered = [k for k, o in labels.items() if o.kind in PROJECTION_KINDS and o.target == NADIA and o.channel == "automatic"]
    assert steered and all(focused[k] > plain_[k] for k in steered)


def test_m4d1g_focus_does_not_raise_how_often_a_kind_is_chosen():
    """Adversarial: focus moves weight between targets of a projection kind and leaves the kind's total weight as it
    was, even when its targets' learned values differ, so a count of projection acts does not rise because a parent is
    focused."""
    def weights(focus):
        state = fresh()
        for tie in state.ties_of(RAVI):
            tie.investment[RAVI] = 1.0
        state.ties[RAVI_NADIA].investment[RAVI] = 1.0 + focus
        obs = observe(state, RAVI, PARAMS)
        outcomes = legal_outcomes(obs, load_event_kinds(), PARAMS)
        for o in outcomes:  # unequal learned values: the focused tie's acts already pay best
            if o.target == NADIA and o.channel == "automatic":
                state.people[RAVI].learned_values[o.value_key] = 2.0
        obs = observe(state, RAVI, PARAMS)
        return dict(zip((o.label for o in outcomes), channel_weights(outcomes, obs, load_event_kinds(), PARAMS,
                                                                     steered=True))), outcomes
    plain_, outcomes = weights(0.0)
    focused, _ = weights(20.0)
    checked = 0
    for kind in PROJECTION_KINDS:
        mine = [o.label for o in outcomes if o.kind == kind and o.channel == "automatic"]
        if len(mine) > 1:
            checked += 1
            assert sum(focused[k] for k in mine) == pytest.approx(sum(plain_[k] for k in mine))
            assert any(focused[k] < plain_[k] for k in mine)  # someone else gets less
    assert checked > 0  # not vacuous: at least one projection kind has more than one target


def test_m4d1g_the_self_channel_is_not_steered():
    plain_, labels = toward()
    focused, _ = toward(focus=20.0)
    own = [k for k, o in labels.items() if o.channel != "automatic"]
    assert own and all(focused[k] == plain_[k] for k in own)


def test_m4d1g_withdrawing_kinds_are_not_steered():
    """Adversarial: a tie holding most of the attention draws no extra CUTOFF or DISTANCE."""
    plain_, labels = toward()
    focused, _ = toward(focus=20.0)
    withdrawing = [k for k, o in labels.items() if o.kind in ("CUTOFF", "DISTANCE") and o.target == NADIA]
    assert withdrawing and all(focused[k] == plain_[k] for k in withdrawing)


def test_m4d1g_the_steered_kinds_are_the_projection_row():
    from src.bowen.readouts.patterns import PROJECTION_ACTS
    assert set(PROJECTION_KINDS) == set(PROJECTION_ACTS)


def test_m1b8a_focus_raises_the_focused_childs_load():
    """The loop closes through M4.D.1g: with the parents' attention on Nadia, more projection-type acts land on her and
    she carries more, against the same seeds with steering off. Investment alone moves no load (M1.B.8)."""
    from src.bowen.engine.contact import excess
    from src.bowen.engine.log_records import EmittedRecord
    from src.bowen.ensemble.criteria import scenario

    def focus_on_nadia(state):
        for parent in (RAVI, MARTA):
            state.ties[TieId.of(parent, NADIA)].investment[parent] = 500.0

    def arm(gain):
        acts, load = 0, 0.0
        for seed in range(8):
            seen = []
            _, records = scenario("phase_c", seed, 26, setup=focus_on_nadia, overrides={"focus_steer_gain": gain},
                                  watch=lambda s: seen.append(excess(s.people[NADIA])))
            acts += sum(1 for r in records if isinstance(r, EmittedRecord) and r.event.kind in PROJECTION_KINDS
                        and r.event.sender in (RAVI, MARTA) and NADIA in r.event.targets)
            load += sum(seen) / len(seen)
        return acts, load

    on, off = arm(PARAMS.focus_steer_gain), arm(0.0)
    assert on[0] > off[0] and on[1] > off[1]


def test_m4d6f_the_key_written_is_the_key_the_policy_reads():
    """One key format: what learn_from_witnessed writes for a CUTOFF toward Ravi is what value_key builds for the
    witness's CUTOFF outcome toward Ravi, in the band and position they are in."""
    from src.bowen.policy.policy import triangle_position, value_key
    state = fresh()
    witnessed_cutoff(state)
    read_after(state, {p: +5.0 for p in nadias_family(state)})
    obs = observe(state, NADIA, PARAMS)
    position = triangle_position(obs, PARAMS)[0]
    assert value_key(obs, "CUTOFF", RAVI, PARAMS) in cutoff_values(state)
    assert value_key(obs, "CUTOFF", RAVI, PARAMS, position=position) in cutoff_values(state)


@pytest.mark.parametrize("name, value", [("witnessed_cutoff_gain", 1.5), ("witnessed_cutoff_gain", -0.1),
                                         ("witness_window", 0), ("reactive_cutoff_gain", -0.1),
                                         ("perceived_weakness_gain", -0.1), ("focus_steer_gain", -1.5)])
def test_m4d6f_m1b3d_m1b8a_m4d1g_out_of_range_constants_are_refused(name, value):
    with pytest.raises(ValueError, match=name):
        dataclasses.replace(PARAMS, **{name: value})
