"""The persistence sink and the log header.

Purpose: the sink changes nothing about a run, logs are byte-identical with
         their header, and the header identifies the resolved configuration.
Spec:    docs/bowen_agent_model_spec_v2.md#M16.T.3, #M16.T.5, #M16.A.1, #M16.A.7, #M16.B.3
Tests:   this file
"""

from __future__ import annotations

import copy
import dataclasses
import re
from pathlib import Path
from types import MappingProxyType

from src.bowen.engine.activation import SynchronousActivation
from src.bowen.engine.log_records import LogHeader, serialize
from src.bowen.engine.state import canonical_state
from src.bowen.engine.visibility import HouseholdConductanceVisibility
from src.bowen.io.load import (
    load_constants, load_event_kinds, load_family, load_frozen_constants, load_script, load_spec_revision,
)
from src.bowen.io.sinks import JsonlFileSink
from src.bowen.run import run_phase_b
from src.bowen.scenario.header import build_header, config_hash, resolved_config


REPO = Path(__file__).resolve().parents[2]


def test_m16t3_sink_does_not_change_results(tmp_path):
    """G10: same seed, sink attached and detached, identical final state.

    Passes vacuously in Phase B — no policy reads the log — and M16.T.3 requires
    it re-run at the end of Phases C and D.
    """
    detached = run_phase_b(7)
    path = tmp_path / "run.jsonl"
    with JsonlFileSink(path) as sink:
        attached = run_phase_b(7, extra=sink)
    assert canonical_state(attached.state) == canonical_state(detached.state)
    written = path.read_text(encoding="utf-8").splitlines()
    assert written == [serialize(r) for r in attached.records]  # the sink is faithful, too


def test_m16t3_s1_2_canonical_state_sees_the_stressors():
    """S1.2: two states that differ only in a lasting stressor give different strings, so the comparison above cannot
    miss a sink that changed a stressor still weighing after the last compared week. Before stage 1 the canonical state
    left ``state.stressors`` out."""
    from src.bowen.engine.state import ActiveStressor

    state = run_phase_b(7).state
    target = sorted(state.people)[0]
    strings = set()
    for intensity in (10.0, 11.0):
        state.stressors[:] = [ActiveStressor(event_id="s1-2", target=target, intensity=intensity, first_tick=1,
                                             last_tick=None)]
        strings.add(canonical_state(state))
    state.stressors.clear()
    strings.add(canonical_state(state))
    assert len(strings) == 3


def test_m16t3_canonical_state_covers_every_field():
    """Every RunState field is in the canonical state under its own name, or in ``NOT_CANONICAL`` with a reason, so a
    field added later cannot be missed; and a change to each field the stage 1 review found left out changes the string
    (review 2026-10-19)."""
    import json

    from src.bowen.engine.events import Delivery, EventId, Role
    from src.bowen.engine.state import NOT_CANONICAL, RunState

    state = run_phase_b(7).state
    names = {f.name for f in dataclasses.fields(RunState)}
    assert set(NOT_CANONICAL) <= names and all(NOT_CANONICAL.values())
    assert set(json.loads(canonical_state(state))) == names - set(NOT_CANONICAL)
    a, b = sorted(state.people)[:2]
    changes = {
        "queue": lambda s: s.queue.schedule(Delivery(delivered_tick=s.tick + 5, event_id=EventId(s.tick, a.value, 0),
                                                     recipient=b, role=Role.TARGET, emitted_tick=s.tick, latency=5)),
        "practice": lambda s: s.practice.append((a, b, s.tick, 0.5)),
        "adjusted": lambda s: s.adjusted.__setitem__((a, b), s.tick + 1),
        "stances": lambda s: s.stances.__setitem__((a, b), s.tick + 1),
        "witnessed": lambda s: s.witnessed.append((a, s.tick, ())),
        "estimator_history": lambda s: s.estimator_history[a].append(s.estimator_history[a][-1]),
    }
    before = canonical_state(state)
    for name, change in changes.items():  # one after another: each must change the string from the one before it
        change(state)
        after = canonical_state(state)
        assert after != before, f"a change to state.{name} does not change the canonical state"
        before = after


def test_m16t5_same_seed_same_log_with_header(tmp_path):
    """G11: two persisted runs at one seed are byte-identical files, header first."""
    paths = [tmp_path / "a.jsonl", tmp_path / "b.jsonl"]
    for path in paths:
        with JsonlFileSink(path) as sink:
            run_phase_b(11, extra=sink)
    a, b = (p.read_bytes() for p in paths)
    assert a == b
    assert a.splitlines()[0].startswith(b'{"activation_component"')
    assert b'"record_type":"header"' in a.splitlines()[0]


def header(constants=None, frozen=None, script=None):
    kinds, family = load_event_kinds(), load_family()
    return build_header(
        seed=1, spec_revision=load_spec_revision(), constants=constants or load_constants(),
        frozen=frozen or load_frozen_constants(), kinds=kinds, family=family,
        script=script or load_script(kinds=kinds, family=family),
        activation=SynchronousActivation(), visibility=HouseholdConductanceVisibility(0.8),
    )


def test_m16a1_header_hashes_the_resolved_config():
    h = header()
    assert isinstance(h, LogHeader) and len(h.config_hash) == 64
    front = (REPO / "docs" / "bowen_agent_model_spec_v2.md").read_text(encoding="utf-8").split("\n---", 1)[0]
    stated = re.search(r"^version:[ \t]*(.+?)[ \t]*$", front, re.M).group(1)  # read here, not via the loader
    assert h.spec_revision == stated and re.fullmatch(r"2\.0, revision \d+", h.spec_revision)
    assert h == header()  # stable
    # A changed value changes the hash; the script's events are part of the config.
    assert header(script=load_script().without("TRIGGER")).config_hash != h.config_hash


def test_m16a1_header_hash_ignores_comments_but_not_values():
    constants = load_constants()
    kinds, family, script = load_event_kinds(), load_family(), load_script()
    base = config_hash(resolved_config(constants, kinds, family, script))
    bumped = dataclasses.replace(
        constants,
        values=MappingProxyType({**constants.values,
                                 "acute_decay_rate": dataclasses.replace(constants.values["acute_decay_rate"], value=0.25)}),
    )
    assert config_hash(resolved_config(bumped, kinds, family, script)) != base
    # Review 2026-10-11: which kinds are nodal changes the estimator's breadth, so it is part of the config.
    fewer = dataclasses.replace(kinds, nodal=kinds.nodal - {"MOVE"},
                                nodal_reach={k: r for k, r in kinds.nodal_reach.items() if k != "MOVE"})
    assert config_hash(resolved_config(constants, fewer, family, script)) != base
    for changed in (dataclasses.replace(kinds, soothing=kinds.soothing - {"DISTANCE"}),
                    dataclasses.replace(kinds, steering={k: d for k, d in kinds.steering.items() if k != "TRIANGLE"})):
        assert config_hash(resolved_config(constants, changed, family, script)) != base


def test_m16a1_t218_every_config_input_moves_the_hash():
    """T-218 (`M16.A.1`): every loaded input that can change a run is in `config_hash` — each event-kind field, the
    life course, the policy rules. Built from the dataclasses' own fields, so a new field cannot be left out."""
    from src.bowen.io.load import load_policy_rules
    from src.bowen.policy.rules import PolicyRules

    constants, kinds, family, script = load_constants(), load_event_kinds(), load_family(), load_script()
    policy = load_policy_rules()
    base = config_hash(resolved_config(constants, kinds, family, script, policy))
    for field in dataclasses.fields(kinds):
        value = getattr(kinds, field.name)
        if hasattr(value, "items"):
            changed = dict(list(value.items())[1:]) if value else {"X": "Y"}
        else:
            changed = frozenset(sorted(value, key=str)[1:]) if value else frozenset({"X"})
        other = copy.copy(kinds)  # past the loader's validation: only the hash is under test
        object.__setattr__(other, field.name, changed)
        assert config_hash(resolved_config(constants, other, family, script, policy)) != base, field.name
    for field in dataclasses.fields(family):
        if field.name != "instance_id":
            other = copy.copy(family)
            object.__setattr__(other, field.name, "changed")
            assert config_hash(resolved_config(constants, kinds, other, script, policy)) != base, field.name
    other_policy = object.__new__(PolicyRules)  # one legal value per rule today, so set a second one directly
    object.__setattr__(other_policy, "tie_break", "changed")
    object.__setattr__(other_policy, "fallback", policy.fallback)
    assert config_hash(resolved_config(constants, kinds, family, script, other_policy)) != base


def test_m16a7_header_flags_constants_changed_after_freeze():
    constants, frozen = load_constants(), load_frozen_constants()
    since_freeze = {k for k in constants.values if k not in frozen.values or constants[k] != frozen[k]}
    baseline = {k for k, flag in header().constant_changed_after_freeze if flag}
    # The flagged set is computed from the frozen and current registers, not pinned by hand (T-222, 2026-11-11).
    assert baseline == since_freeze and baseline
    changed = dataclasses.replace(
        constants,
        values=MappingProxyType({**constants.values,
                                 "route_damping": dataclasses.replace(constants.values["route_damping"], value=0.4)}),
    )
    flags = dict(header(constants=changed).constant_changed_after_freeze)
    assert {k for k, flag in flags.items() if flag} == baseline | {"route_damping"}


def test_m16b1_a_failed_run_leaves_no_complete_looking_log(tmp_path):
    """Review finding (2026-10-06): a run that raised left a well-formed, shorter log."""
    path = tmp_path / "run.jsonl"

    class Boom(RuntimeError):
        pass

    try:
        with JsonlFileSink(path) as sink:
            run_phase_b(7, extra=sink)
            raise Boom
    except Boom:
        pass
    assert not path.exists()
    assert (tmp_path / "run.jsonl.partial").exists()


def test_m16a1_t218_equal_config_values_give_an_equal_hash():
    """T-218: the hash reads values, so a read-only mapping in another key order hashes as the same values do."""
    from src.bowen.engine.log_records import plain
    value = plain(MappingProxyType({"b": 1, "a": 2}))
    assert type(value) is dict and list(value) == ["a", "b"]  # sorted, as a dict is, not the repr of a proxy
