"""LX.2, LX.3, LX.17, LX.18, LX.19: the LLM source inside the engine.

Purpose: prove LLMSource gives one selection per active living family member a week and leaves the external agent on
         the policy (LX.2), that every choice is legal and an illegal one is a flagged hold (LX.3), that A0 through the
         experiment is the ensemble's own run (LX.17), that a replayed run is the same with logging on and off (LX.18),
         and that the offer gate's draw moves no other draw (LX.19).
Spec:    docs/plan_v3_llm_agent_experiment.md#10
Tests:   this file
"""

from conftest import REPO_ROOT, StubChooser, StubTransport, final_state, stub_run

from experiments.llm_agents.cache import ResponseCache
from experiments.llm_agents.client import CachedChooser
from experiments.llm_agents.config import load_llm, load_text
from experiments.llm_agents.source import FAILED, OFFER_PURPOSE, Answer, LLMSource, build, execute
from src.bowen.engine.act import WITHHOLD
from src.bowen.engine.draws import DrawKey, DrawService
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.log_records import DecidedBy, SelectionRecord, serialize
from src.bowen.engine.objects import Role
from src.bowen.ensemble.criteria import scenario, spell

WEEKS = 208


class Recording(LLMSource):
    """LLMSource that keeps, each week, who was active and what it returned."""

    log: list

    def selections(self, tick, active, state):
        out = super().selections(tick, active, state)
        self.log.append((tick, tuple(active), {p: state.people[p] for p in active}, out))
        return out


def _recorded(arm="A1", seed=1, weeks=60, chooser=None):
    run = build(arm, seed, weeks, chooser or StubChooser())
    run.source.__class__ = Recording
    run.source.log = []
    return execute(run)


def test_lx2_one_selection_per_active_member():
    run = _recorded()
    interval = run.params.session_interval_weeks
    for tick, active, people, out in run.source.log:
        actors = [s.actor for s in out]
        assert len(actors) == len(set(actors)), f"week {tick}: two selections for one person"
        members = {p for p in active if people[p].alive and people[p].role is not Role.EXTERNAL}
        external = {p for p in active if people[p].alive and people[p].role is Role.EXTERNAL}
        assert members <= set(actors), f"week {tick}: a member without a selection"
        assert set(actors) <= set(active)
        assert (external <= set(actors)) is (tick % interval == 0) or not external


def test_lx2_external_agent_keeps_the_policy():
    run, _ = stub_run("A1", 1, WEEKS)
    halim = [p for p, person in run.state.people.items() if person.role is Role.EXTERNAL]
    assert halim, "phase_c has an external agent"
    sel = [r for r in run.records if isinstance(r, SelectionRecord) and r.actor in halim]
    assert sel, "the external agent had sessions"
    assert {r.decided_by for r in sel} <= {DecidedBy.POLICY, DecidedBy.FALLBACK, DecidedBy.TIE_BREAK}
    assert all(r.tick % run.params.session_interval_weeks == 0 for r in sel)
    # The members are the chooser's: every one of their selections is SCRIPTED or a flagged hold, never the policy.
    members = [r for r in run.records if isinstance(r, SelectionRecord) and r.actor not in halim
               and r.decided_by is not DecidedBy.SEQUENCE]
    assert members and {r.decided_by for r in members} <= {DecidedBy.SCRIPTED, DecidedBy.FALLBACK}
    assert all(r.propensities == () for r in members), "no policy propensities were computed for a member"


def test_lx3_choice_is_always_legal():
    run, _ = stub_run("A2", 1, WEEKS)
    by_week = {}
    for r in run.records:
        if isinstance(r, SelectionRecord):
            by_week[(r.tick, r.actor.value)] = r
    made = 0
    for c in run.source.choices:
        if c.flag is not None:
            continue
        r = by_week[(c.week, c.person)]
        label = f"{r.withheld}>{c.target}" if c.kind == WITHHOLD else f"{c.kind}>{c.target}"
        assert label in r.legal_set, f"week {c.week} {c.person}: {label} not in its legal set"
        made += 1
    assert made > 1000


def test_lx3_illegal_choice_fails_to_parse():
    class Illegal:
        def choose(self, week, token, prompt):
            return Answer(str(len(prompt.ids) + 1))   # one past the last option

    run = execute(build("A1", 1, 4, Illegal()))
    assert run.source.choices and all(c.flag == FAILED for c in run.source.choices)
    holds = [r for r in run.records if isinstance(r, SelectionRecord) and r.fallback_rule == FAILED]
    assert holds and all(r.decided_by is DecidedBy.FALLBACK and r.event_id is None for r in holds)
    # The parse check is the client's too: an id outside the round's enum is not a choice (test_client, LX.14).


def test_lx17_a0_matches_the_ensemble():
    for seed in (1, 2):
        ours = execute(build("A0", seed, WEEKS))
        state, records = scenario("phase_c", seed, WEEKS, spells=spell("phase_c", WEEKS))
        assert final_state(ours.state) == final_state(state)
        assert [serialize(r) for r in ours.records] == [serialize(r) for r in records]
    # The comparison bites: another seed is another run.
    other, _ = scenario("phase_c", 3, WEEKS, spells=spell("phase_c", WEEKS))
    assert final_state(ours.state) != final_state(other)


def test_lx18_logging_is_a_pure_observer(tmp_path):
    cfg, text = load_llm(), load_text()
    model = cfg.get("model_main")

    def chooser(cache, transport):
        return CachedChooser(run_id="lx18", model=model, arm="A1", seed=1, cfg=cfg, cache=cache,
                             reask_text=text["reask"], transport=transport)

    cache = ResponseCache(tmp_path / "lx18" / "responses.jsonl.gz", runs=tmp_path)
    live = execute(build("A1", 1, WEEKS, chooser(cache, StubTransport())))
    on = execute(build("A1", 1, WEEKS, chooser(cache, None)))
    off = execute(build("A1", 1, WEEKS, chooser(cache, None)), keep=False)
    assert off.records == [] and len(on.records) > 10000
    assert final_state(on.state) == final_state(off.state) == final_state(live.state)


def test_lx19_offer_draw_is_isolated():
    # Own purpose: no engine code draws under it.
    for path in (REPO_ROOT / "src" / "bowen").rglob("*.py"):
        assert OFFER_PURPOSE not in path.read_text(encoding="utf-8"), path
    # Keyed draws: the policy's own keys give the same values whether or not offer draws were made first.
    actor = PersonId("ravi")
    policy_keys = [DrawKey.make("move_selection", tick=t, actor=actor, purpose=p, index=0)
                   for t in range(20) for p in ("select", "withheld")]
    offer_keys = [DrawKey.make("move_selection", tick=t, actor=actor, purpose=OFFER_PURPOSE, index=i)
                  for t in range(20) for i in range(10)]
    plain = DrawService(7, debug=True)
    before = [plain.uniform(k) for k in policy_keys]
    mixed = DrawService(7, debug=True)
    for k in offer_keys:
        mixed.uniform(k)
    after = [mixed.uniform(k) for k in policy_keys]   # debug mode: a shared key would raise RepeatedDrawKey
    assert before == after
    assert set(offer_keys).isdisjoint(policy_keys)
    # In a run, every policy draw the external agent read is the keyed value itself: the family's offer draws
    # around it moved none of them.
    a1, _ = stub_run("A1", 1, WEEKS)
    halim = PersonId("halim")
    drawn = [(r.tick, r.draw) for r in a1.records
             if isinstance(r, SelectionRecord) and r.actor == halim and r.draw is not None]
    assert drawn, "the external agent drew"
    service = DrawService(1)
    for tick, value in drawn:
        assert service.uniform(DrawKey.make("move_selection", tick=tick, actor=halim, purpose="select", index=0)) == value


def test_lx7_history_change_is_to_the_next_round():
    # The change shown after an act is the tension at the person's next round less the tension when chosen, even when
    # that next round fails to parse. Measured to the next *successful* choice, a failed week would fold its change
    # into the earlier act's.
    class FailsWeekOne(StubChooser):
        def choose(self, week, person_token, prompt):
            return Answer(None, reasks=1, reason="stub") if week == 1 else super().choose(week, person_token, prompt)

    run = execute(build("A2", 1, 3, FailsWeekOne()))
    by = {(c.week, c.person): c for c in run.source.choices}
    checked = 0
    for pid, mem in run.source.memory.items():
        first = next((h for h in mem.history if h[0] == 1), None)
        if first is None or (1, pid.value) not in by:
            continue
        assert by[(1, pid.value)].flag == FAILED
        assert first[3] == by[(1, pid.value)].tension
        checked += 1
    assert checked >= 3, "enough members chose in week 0 and failed in week 1 to make the check bite"


# --- §13: the option order (LX.22, LX.24) ---------------------------------------------------------------------------

def _text_run(order):
    from conftest import DrawRecorder, TextChooser

    run = build("A1", 1, WEEKS, TextChooser(), option_order=order)
    recorder = DrawRecorder(run.state.draws)
    run.state.draws = recorder
    return execute(run), recorder


def test_lx22_order_requests_no_engine_draw():
    shuffled, s_draws = _text_run("shuffled")
    fixed, f_draws = _text_run("fixed")
    assert s_draws.keys, "the recorder saw the run's draws"
    assert s_draws.keys == f_draws.keys, "the shuffle asked the engine's draw service for nothing"


def test_lx22_order_changes_no_outcome():
    shuffled, _ = _text_run("shuffled")
    fixed, _ = _text_run("fixed")
    assert final_state(shuffled.state) == final_state(fixed.state)

    def picks(run):
        return [(c.week, c.token, c.option, c.kind, c.target, c.flag) for c in run.source.choices]
    assert picks(shuffled) == picks(fixed)
    # The ids differ: the same options were listed at other places (the check is not trivially of one order).
    assert [c.option_id for c in shuffled.source.choices] != [c.option_id for c in fixed.source.choices]


def test_lx24_id_maps_to_the_option_shown():
    run, chooser = stub_run("A1", 1, WEEKS)
    assert run.source.labeller.order == "shuffled"
    token_of = {p.value: t for p, t in run.source.tokens.items()}
    chosen = 0
    for c in run.source.choices:
        if c.flag is not None:
            continue
        prompt = chooser.prompts[(c.week, c.token)]
        shown = prompt.specs[int(c.option_id) - 1]
        assert dict(prompt.options)[c.option_id] == run.source.labeller.option(shown) == c.option
        assert c.spec == shown, f"week {c.week} {c.token}: id {c.option_id} ran {c.spec}, not the option shown"
        assert c.kind == (WITHHOLD if shown.hold else shown.kind) and token_of[c.target] == shown.target
        chosen += 1
    assert chosen > WEEKS
    # What the engine ran is what was shown: each member's selection record names the shown act.
    shown_by = {(c.week, c.person): prompt_spec for c in run.source.choices if c.flag is None
                for prompt_spec in (chooser.prompts[(c.week, c.token)].specs[int(c.option_id) - 1],)}
    checked = 0
    for r in run.records:
        if isinstance(r, SelectionRecord) and (r.tick, r.actor.value) in shown_by:
            spec = shown_by[(r.tick, r.actor.value)]
            if spec.hold:
                assert r.withheld == spec.kind
                checked += 1
            elif r.event_id is not None:
                # during an I-POSITION sequence the engine makes an act toward the sequence's other as STAY-IN-CONTACT
                # (src/bowen/engine/iposition.py, M5.D.9), in either order; so that one substitution is allowed
                kind = run.state.store.event(r.event_id).kind
                assert kind in (spec.kind, "STAY-IN-CONTACT"), f"tick {r.tick} {r.actor}: ran {kind}, shown {spec}"
                checked += kind == spec.kind
    assert checked > chosen // 2
