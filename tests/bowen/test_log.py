"""The persistence sink and the log header.

Purpose: the sink changes nothing about a run, logs are byte-identical with
         their header, and the header identifies the resolved configuration.
Spec:    docs/bowen_agent_model_spec_v2.md#M16.T.3, #M16.T.5, #M16.A.1, #M16.A.7, #M16.B.3
Tests:   this file
"""

from __future__ import annotations

import dataclasses
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
    assert h.spec_revision == "2.0, revision 10"
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


def test_m16a7_header_flags_constants_changed_after_freeze():
    constants = load_constants()
    assert not any(flag for _, flag in header().constant_changed_after_freeze)
    changed = dataclasses.replace(
        constants,
        values=MappingProxyType({**constants.values,
                                 "route_damping": dataclasses.replace(constants.values["route_damping"], value=0.4)}),
    )
    flags = dict(header(constants=changed).constant_changed_after_freeze)
    assert flags["route_damping"] is True and sum(flags.values()) == 1


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
