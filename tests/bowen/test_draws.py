"""Counter-based, event-keyed draws.

Purpose: prove draws are pure functions of (seed, key), keys carry only
         structural identity, the class table matches the spec, and repeated
         keys are caught.
Spec:    docs/bowen_agent_model_spec_v2.md#M3.D.4, #M3.D.4a, #M3.D.4b, #M3.D.4c
Tests:   this file
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from src.bowen.engine.draws import (
    DRAW_CLASSES,
    DrawKey,
    DrawKeyError,
    DrawService,
    Keying,
    RepeatedDrawKey,
)
from src.bowen.engine.identifiers import PersonId

REPO = Path(__file__).resolve().parents[2]
ENGINE = REPO / "src" / "bowen" / "engine"
RAVI, MARTA, NADIA = PersonId("ravi"), PersonId("marta"), PersonId("nadia")


def select(tick: int = 3, actor: PersonId = RAVI, index: int = 0) -> DrawKey:
    return DrawKey.make("move_selection", tick=tick, actor=actor, purpose="select", index=index)


def fidelity(partner: PersonId) -> DrawKey:
    return DrawKey.make(
        "per_hop_fidelity", tick=3, actor=RAVI, partner=partner, purpose="hop", index=0
    )


# --- M3.D.4a: a pure function of seed and key ----------------------------------


def test_m3d4a_draw_is_pure_function_of_seed_and_key():
    a, b = DrawService(7), DrawService(7)
    assert a.uniform(select()) == b.uniform(select())
    assert DrawService(8).uniform(select()) != a.uniform(select())
    assert a.uniform(select(index=1)) != a.uniform(select(index=0))


def test_m3d4a_query_order_does_not_change_values():
    """The property a stateful generator lacks: what was drawn before cannot shift a draw."""
    first, second = DrawService(7), DrawService(7)
    keys = [select(tick=t) for t in range(5)]
    forward = [first.uniform(k) for k in keys]
    backward = [second.uniform(k) for k in reversed(keys)][::-1]
    assert forward == backward


def test_m3d4a_an_extra_draw_in_one_arm_shifts_no_other_draw():
    """Arms differing by one extra draw still agree on every shared key (the coupling M0.4 needs)."""
    baseline, treated = DrawService(11), DrawService(11)
    treated.uniform(DrawKey.make("tie_break_fallback", tick=3, actor=RAVI, purpose="tie", index=0))
    assert [baseline.uniform(select(tick=t)) for t in range(4)] == [
        treated.uniform(select(tick=t)) for t in range(4)
    ]


def test_m3d4a_uniforms_are_in_unit_interval_and_fixed_count():
    service = DrawService(1)
    values = service.uniforms(select(), 5)
    assert len(values) == 5 and all(0.0 <= v < 1.0 for v in values)


def test_m3d4a_draws_agree_across_processes():
    """Byte identity across processes, with Python's hash salt deliberately varied (M3.D.5)."""
    program = (
        "import sys; sys.path.insert(0, %r)\n"
        "from src.bowen.engine.draws import DrawKey, DrawService\n"
        "from src.bowen.engine.identifiers import PersonId\n"
        "k = DrawKey.make('move_selection', tick=3, actor=PersonId('ravi'), purpose='select', index=0)\n"
        "print(repr(DrawService(7).uniforms(k, 3)))\n" % str(REPO)
    )
    outputs = set()
    for salt in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=salt)
        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, timeout=60, env=env
        )
        assert result.returncode == 0, result.stderr
        outputs.add(result.stdout)
    assert len(outputs) == 1
    assert outputs.pop().strip() == repr(DrawService(7).uniforms(select(), 3))


def test_m3d4a_key_rejects_state_quantities():
    with pytest.raises(DrawKeyError, match="non-negative int"):
        DrawKey.make("move_selection", tick=3.0, actor=RAVI, purpose="select", index=0)
    with pytest.raises(DrawKeyError, match="non-negative int"):
        DrawKey.make("move_selection", tick=True, actor=RAVI, purpose="select", index=0)
    with pytest.raises(DrawKeyError, match="PersonId"):
        DrawKey.make("move_selection", tick=3, actor="ravi", purpose="select", index=0)
    with pytest.raises(DrawKeyError, match="label"):
        DrawKey.make("move_selection", tick=3, actor=RAVI, purpose=0.42, index=0)
    with pytest.raises(DrawKeyError, match="needs exactly"):
        DrawKey.make("move_selection", tick=3, actor=RAVI, purpose="select", index=0, anxiety=7)
    with pytest.raises(DrawKeyError, match="needs exactly"):
        DrawKey.make("move_selection", tick=3, actor=RAVI, purpose="select")


def test_m3d4a_no_builtin_hash_or_global_rng_in_engine():
    """Static: no salted hash(), no stdlib random, no global NumPy RNG anywhere in the engine."""
    offenders = []
    for path in sorted(ENGINE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hash":
                offenders.append(f"{path.name}: hash()")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
                if "random" in names:
                    offenders.append(f"{path.name}: imports random")
            if isinstance(node, ast.Attribute) and node.attr in {"seed", "default_rng", "RandomState"}:
                offenders.append(f"{path.name}: np.random.{node.attr}")
    assert offenders == []


# --- M3.D.4b: every draw class declares its key ---------------------------------


def _spec_draw_table() -> list[tuple[str, str, str]]:
    text = (REPO / "docs" / "bowen_agent_model_spec_v2.md").read_text(encoding="utf-8")
    start = text.index("| Draw class | Where | Proposed key | Proposed keying |")
    rows = []
    for line in text[start:].splitlines()[2:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        where = re.findall(r"`(M[\d.A-Za-z]+)`", cells[1])[0]
        plain = re.sub(r"⟦.*?⟧", "", cells[3])
        keying = re.split(r"—|,", plain)[0].strip().split(" by design")[0]
        rows.append((where, cells[2], keying))
    return rows


def test_m3d4b_table_matches_the_spec():
    ours = [(c.where, ", ".join(c.components), c.keying.value) for c in DRAW_CLASSES.values()]
    assert ours == _spec_draw_table()


def test_m3d4b_every_draw_class_declares_its_key():
    with pytest.raises(DrawKeyError, match="undeclared draw class"):
        DrawKey.make("hunch", tick=1)


def test_m3d4b_slot_keys_carry_no_partner_and_dyad_keys_do():
    for cls in DRAW_CLASSES.values():
        if cls.keying is Keying.SLOT:
            assert "partner" not in cls.components
        if cls.keying is Keying.DYAD:
            assert "partner" in cls.components


def test_m3d4b_dyad_keying_makes_a_different_partner_a_different_event():
    service = DrawService(5)
    assert service.uniform(fidelity(MARTA)) != service.uniform(fidelity(NADIA))


def test_m3d4b_tie_break_is_distinct_from_selection_at_the_same_slot():
    service = DrawService(5)
    tie = DrawKey.make("tie_break_fallback", tick=3, actor=RAVI, purpose="select", index=0)
    assert service.uniform(tie) != service.uniform(select())


# --- M3.D.4c: each key at most once ---------------------------------------------


def test_m3d4c_repeated_key_raises_in_debug():
    service = DrawService(3, debug=True)
    service.uniform(select())
    with pytest.raises(RepeatedDrawKey):
        service.uniform(select())


def test_m3d4c_repeated_key_returns_the_cached_value_outside_debug():
    service = DrawService(3)
    assert service.uniform(select()) == service.uniform(select())
    with pytest.raises(ValueError, match="n=1"):
        service.uniforms(select(), 2)


# --- seeds and inverse-transform sampling ----------------------------------------


@pytest.mark.parametrize("bad", [-1, 2**64, 1.0, True])
def test_m3d4_seed_must_be_a_64_bit_int(bad):
    with pytest.raises(ValueError, match="seed"):
        DrawService(bad)


def test_m3d4a_categorical_uses_one_uniform_and_respects_zero_weights():
    service = DrawService(9)
    picks = {service.categorical(select(tick=t), [0.0, 1.0, 0.0]) for t in range(50)}
    assert picks == {1}
    service2 = DrawService(9)
    key = select(tick=99)
    expected_u = DrawService(9).uniform(key)
    index = service2.categorical(key, [1.0, 1.0])
    assert index == (0 if expected_u < 0.5 else 1)
