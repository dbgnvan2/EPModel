"""The Phase C engineering gates (Phase C step 13).

Purpose: test M11.D.2 (no magic literals), M11.D.8/9 (spec references resolve, exact count),
         M11.D.15 (placebo arm), M11.D.16 (order invariance and permutation equivariance),
         M11.D.19 (every move reachable in a triad), M11.D.21 (no absorbing bounds) and
         M16.T.3 rerun with a policy.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.D.2, #M11.D.8, #M11.D.9, #M11.D.15, #M11.D.16, #M11.D.19, #M11.D.21, #M16.T.3
Tests:   this file
"""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path
from unittest import mock

import numpy as np

from src.bowen.engine import tick as tick_module
from src.bowen.engine.contact import clamp_unit
from src.bowen.engine.draws import DrawKey
from src.bowen.engine.event_store import EventStore
from src.bowen.engine.events import EventQueue
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.initialise import initialise_run
from src.bowen.engine.observe import observe
from src.bowen.engine.state import canonical_state, new_run_state
from src.bowen.engine.tick import run
from src.bowen.io.load import CONFIG_DIR, load_constants, load_event_kinds, load_family, load_policy_rules, load_script
from src.bowen.policy.policy import legal_outcomes, propensities
from src.bowen.scenario.assemble import assemble
from src.bowen.scenario.params import engine_params
from src.bowen.scenario.policy_source import PolicySource
from src.bowen.scenario.scripted_source import ScriptedSource

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "docs" / "bowen_agent_model_spec_v2.md"
PARAMS = engine_params(load_constants())
KINDS = load_event_kinds()
P = PersonId


class Collect:
    def __init__(self):
        self.records = []

    def emit(self, record):
        self.records.append(record)


class Null:
    def emit(self, record):
        pass


def policy_run(seed=1, weeks=40, family_path=CONFIG_DIR / "family_phase_c.md", emitter=None, wrap=None):
    family = load_family(family_path)
    script = load_script(kinds=KINDS, family=load_family())
    # The reference family's scripted spells name its people; a fixture runs without them.
    spells = script.events if family_path.name.startswith("family_") else ()
    events = ScriptedSource("events", weeks, spells, ())
    parts = assemble(load_constants(), KINDS, family, events, seed=seed)
    source = PolicySource(events, parts.params, load_policy_rules())
    run(parts.state, wrap(source) if wrap else source, parts.params, parts.visibility, parts.activation,
        emitter or Null(), ticks=weeks)
    return parts.state


# --- M11.D.2 ---------------------------------------------------------------------------------

# Structural numbers, each with its reason. Anything else in the engine or the policy is a magic
# literal and belongs in config/bowen/constants.md.
STRUCTURAL = {
    ("draws.py", 2), ("draws.py", 64), ("draws.py", 2.0), ("draws.py", 53), ("draws.py", 32), ("draws.py", 11),
    # ^ the documented key-to-counter function and 53-bit uniforms (M3.D.4a)
    ("identifiers.py", 3), ("live_positions.py", 3), ("state.py", 3), ("recompute.py", 3),
    # ^ a triangle has three members (M1.C.1)
    ("invariants.py", 3),            # exactly three sinks (M1.D.1)
    ("params.py", 2),                # a hardening run needs at least two moves (M4.G.1)
    ("policy.py", 2),                # entropy needs at least two outcomes (M4.D.1d)
    ("objects.py", 100.0),           # SCALE_MAX, the 0-100 scale, defined once
}
SCANNED = ("src/bowen/engine", "src/bowen/policy")


def magic_literals(paths) -> list[str]:
    found = []
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                if node.value in (0, 1, -1) or (path.name, node.value) in STRUCTURAL:
                    continue
                found.append(f"{path.name}:{node.lineno}: {node.value}")
    return found


def test_m11d2_no_magic_literals_in_engine():
    paths = sorted(p for d in SCANNED for p in (REPO / d).glob("*.py"))
    assert magic_literals(paths) == []


def test_m11d2_check_catches_a_literal_in_the_policy(tmp_path):
    mutant = tmp_path / "policy.py"
    mutant.write_text((REPO / "src/bowen/policy/policy.py").read_text() + "\nSLIP = 0.3\n")
    assert magic_literals([mutant]) == [f"policy.py:{len(mutant.read_text().splitlines())}: 0.3"]


# --- M11.D.8 / M11.D.9 --------------------------------------------------------------------------

EXPECTED_SPEC_REFERENCES = 577  # M11.D.9: an exact count, not a floor; update it when references change


def spec_anchors() -> set[str]:
    text = SPEC.read_text()
    ids = set(re.findall(r"^(?:\| )?(?:[-*]\s+)?\*\*(M\d+(?:\.[A-Za-z0-9]+)*)\*\*", text, re.M))
    headings = set(re.findall(r"^#{2,4} (M\d+(?:\.[A-Za-z0-9]+)*)\b", text, re.M))
    return ids | headings


def spec_references(root: Path) -> list[str]:
    refs = []
    for path in sorted(root.rglob("*.py")):
        for line in path.read_text().splitlines():
            if line.strip().startswith("Spec:"):
                refs += re.findall(r"#(M[0-9A-Za-z.]+)", line)
    return [r.rstrip(".") for r in refs]


def test_m11d8_spec_references_resolve():
    refs = spec_references(REPO / "src" / "bowen")
    anchors = spec_anchors()
    assert sorted({r for r in refs if r not in anchors}) == []
    assert len(refs) == EXPECTED_SPEC_REFERENCES


def test_m11d9_a_narrowed_scan_is_caught():
    assert len(spec_references(REPO / "src" / "bowen" / "engine")) != EXPECTED_SPEC_REFERENCES
    assert "M99.Z.1" not in spec_anchors()


# --- M11.D.15 ----------------------------------------------------------------------------------


def test_m11d15_placebo_arm_is_byte_identical():
    """An extra mechanism at zero magnitude — it draws, it changes nothing — leaves the run byte for byte."""
    original = tick_module.advance_sequences

    def placebo(state, params):
        for pid in sorted(state.people):
            state.draws.uniform(DrawKey.make("move_selection", tick=state.tick, actor=pid, purpose="placebo", index=0))
        return original(state, params)

    for seed in (1, 2, 3):
        baseline = canonical_state(policy_run(seed=seed))
        with mock.patch.object(tick_module, "advance_sequences", placebo):
            arm = canonical_state(policy_run(seed=seed))
        assert arm == baseline


# --- M11.D.16 ----------------------------------------------------------------------------------


def test_m11d16_batch_order_permutation_is_invariant():
    """Reversing the select order and every same-tick delivery batch leaves the final state identical."""

    class Reversed:
        def __init__(self, source):
            self.source = source

        def scheduled(self, tick):
            return tuple(reversed(self.source.scheduled(tick)))

        def selections(self, tick, active, state):
            return tuple(reversed(self.source.selections(tick, tuple(reversed(active)), state)))

    release = EventQueue.release
    for seed in (1, 2):
        baseline = canonical_state(policy_run(seed=seed))
        with mock.patch.object(EventQueue, "release", lambda self, t: tuple(reversed(release(self, t)))):
            permuted = canonical_state(policy_run(seed=seed, wrap=Reversed))
        assert permuted == baseline


def test_m11d16_symmetric_family_is_permutation_equivariant():
    """In distribution, not byte for byte (P7): keyed draws are keyed by identifier."""
    dyad = CONFIG_DIR / "fixtures" / "dyad.md"
    a, b = [], []
    for seed in range(40):
        state = policy_run(seed=seed, weeks=30, family_path=dyad)
        a.append(state.people[P("a")].acute_anxiety)
        b.append(state.people[P("b")].acute_anxiety)
    # A and B are identical but for their identifiers (and sex, which no mechanism reads): their
    # distributions agree. The tolerance is three standard errors of the difference in means.
    se = np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    assert abs(np.mean(a) - np.mean(b)) < 3 * se + 1e-9


# --- M11.D.19 ----------------------------------------------------------------------------------


def test_m11d19_every_move_is_reachable_in_a_triad():
    """Dense sampling of the triad's state box: each core move and WITHHOLD legal with real propensity somewhere."""
    family = load_family(CONFIG_DIR / "fixtures" / "triad.md")
    core = {"PURSUE", "DISTANCE", "CONFLICT", "OVERFUNCTION", "UNDERFUNCTION", "TRIANGLE", "CUTOFF",
            "I-POSITION", "STAY-IN-CONTACT", "WITHHOLD"}
    rng = np.random.default_rng(19)
    seen: dict[str, float] = {}
    for _ in range(300):
        state = new_run_state(dict(family.people), dict(family.ties), family.family, KINDS)
        initialise_run(state, PARAMS)
        for p in state.people.values():
            p.functional_level = float(rng.uniform(5, 95))
            p.acute_anxiety = p.chronic_anxiety + float(rng.uniform(0, 40))
            p.outside_ness_outward, p.outside_ness_inward = float(rng.uniform(0, 1)), float(rng.uniform(0, 1))
            p.systems_perspective = float(rng.uniform(0, 1))
        for tie in state.ties.values():
            for m in tie.id.members():
                tie.felt_contact[m] = float(rng.uniform(0, 1))
                tie.felt_impingement[m] = float(rng.uniform(0, 1))
        for pid in (P("f"), P("m")):
            outcomes, probs = propensities(observe(state, pid, PARAMS), KINDS, PARAMS)
            for o, p in zip(outcomes, probs):
                seen[o.kind] = max(seen.get(o.kind, 0.0), p)
    not_observed = sorted(k for k in core if seen.get(k, 0.0) < 1e-3)
    assert not_observed == [], f"not observed with real propensity (not 'unreachable'): {not_observed}"


# --- M11.D.21 ----------------------------------------------------------------------------------


def test_m11d21_state_bounds_are_not_absorbing():
    """A person started at each bound can leave it."""
    from src.bowen.engine.consolidate import consolidate
    from src.bowen.engine.contact import relax_contact
    from src.bowen.engine.outside_ness import update_outside_ness

    family = load_family()
    state = new_run_state(dict(family.people), dict(family.ties), family.family, KINDS)
    initialise_run(state, PARAMS)
    ravi, marta = P("ravi"), P("marta")
    marital = state.ties[TieId.of(ravi, marta)]
    marital.felt_contact[ravi] = 0.0          # contact at its floor
    marital.felt_impingement[ravi] = 1.0      # impingement at its ceiling
    state.people[ravi].outside_ness_outward = 1.0
    state.people[ravi].acute_anxiety = state.people[ravi].chronic_anxiety + 50.0
    relax_contact(state.people, state.ties, PARAMS)
    update_outside_ness(state, {}, PARAMS)
    consolidate(state, PARAMS)
    assert marital.felt_contact[ravi] > 0.0 and marital.felt_impingement[ravi] < 1.0
    assert state.people[ravi].outside_ness_outward < 1.0
    # functional level at 0 can rise (pseudo-self taken in an exchange)
    from src.bowen.engine.events import Channel, Delivery, Event, EventId, Mechanism, Role, SourcePosition
    from src.bowen.engine.moves import apply_move_effects

    state.people[ravi].functional_level = 0.0
    event = Event(id=EventId(state.tick, "ravi", 7), kind="OVERFUNCTION", mechanism=Mechanism.MOVE, sender=ravi,
                  targets=(marta,), intensity=100.0, timestamp=state.tick, duration=1, exogenous=False,
                  source_position=SourcePosition.NONE, channel=Channel.SCRIPTED)
    state.store.record_event(event)
    apply_move_effects(state, (Delivery(state.tick + 1, event.id, marta, Role.TARGET, emitted_tick=state.tick, latency=1),), PARAMS)
    assert state.people[ravi].functional_level > 0.0


# --- M16.T.3, rerun with a policy ----------------------------------------------------------------


def test_m16t3_sink_does_not_change_results_under_the_policy():
    quiet = canonical_state(policy_run(seed=5))
    logged = canonical_state(policy_run(seed=5, emitter=Collect()))
    assert logged == quiet
