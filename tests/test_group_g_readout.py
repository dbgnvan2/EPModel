"""The group G readout tool renders its section from the measurements it is given (milestone review MR-9).

Purpose: `tools/group_g_readout.py` writes `docs/review/group_g_readout.md`, whose "before" section
         `config/bowen/constants.md` cites for `cutoff_stock_cap`. These tests feed its renderer hand-made
         measurements, so no run is made, and check that every cell and the machine-readable comment come from them,
         and that writing one label's section keeps the other's.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.D.2b, #M11.C.48
Tests:   this file
"""

from __future__ import annotations

import ast

import tools.group_g_readout as tool


def _results() -> list[dict]:
    """Two seeds per family. Phase C: seed 1 cuts twice (stocks 40 and 90) with one first cutoff; seed 2 never cuts.
    Phase D: one cut, one first cutoff per seed."""
    return [
        {"family": "phase_c", "seed": 1, "cuts": [("ravi-marta", 12, 40.0, 3.0), ("marta-nadia", 30, 90.0, 2.5)],
         "firsts": [("ravi-marta", 12, 4, 2, 5)], "excess": 10.0, "symptom_share": 0.2},
        {"family": "phase_c", "seed": 2, "cuts": [], "firsts": [], "excess": 20.0, "symptom_share": 0.4},
        {"family": "phase_d", "seed": 1, "cuts": [("a-b", 100, 7.0, 1.0)], "firsts": [("a-b", 100, 6, 1, 3)],
         "excess": 1.0, "symptom_share": 0.0},
        {"family": "phase_d", "seed": 2, "cuts": [("a-c", 200, 9.0, 1.5)], "firsts": [("a-c", 200, 2, 3, 7)],
         "excess": 3.0, "symptom_share": 0.5},
    ]


def _row(section: str, family: str) -> list[str]:
    line = next(line for line in section.splitlines() if line.startswith(f"| {family} |"))
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def test_g1_readout_section_is_rendered_from_measurements():
    """Every cell of each family's row and the machine-readable comment are computed from the measurements given.
    Mutation: the largest stock taken as the first cut rather than the largest (seed 1's 40, not 90)."""
    section = tool.render_section("after", _results(), "e" * 64, 2)
    assert section.startswith(f"## after — engine hash `{'e' * 12}`, 2 seeds per family, ")
    c = _row(section, "phase_c")
    assert c[:3] == ["phase_c", str(tool.RUNS["phase_c"]), "2"]
    assert c[3] == "90 (`marta-nadia`, week 30, seed 1, bond 2.5)"
    assert c[4:13] == ["65", "1 of 2", "1", "4", "4", "2", "5", "15", "0.3"]
    d = _row(section, "phase_d")
    assert d[3] == "9 (`a-c`, week 200, seed 2, bond 1.5)"
    assert d[4:13] == ["8", "2 of 2", "2", "4", "4", "2", "5", "2", "0.25"]
    comment = next(line for line in section.splitlines() if line.startswith("<!-- after measurements: "))
    data = ast.literal_eval(comment.removeprefix("<!-- after measurements: ").removesuffix(" -->"))
    assert data == {"engine": "e" * 64, "seeds": 2,
                    "largest_stock": {"phase_c": 90.0, "phase_d": 9.0}, "mean_run": {"phase_c": 4, "phase_d": 4},
                    "excess": {"phase_c": 15.0, "phase_d": 2.0}, "symptom_share": {"phase_c": 0.30000000000000004,
                                                                                   "phase_d": 0.25}}


def test_g1_readout_page_keeps_the_other_labels_section():
    """Writing "after" keeps the committed "before" section, in label order under the header; a family with no cut
    shows "no cut" and dashes, not a number."""
    before = tool.render_section("before", [dict(r, cuts=[], firsts=[]) for r in _results()], "b" * 64, 2)
    assert _row(before, "phase_c")[3:5] == ["no cut", "—"]
    page = tool.render_page("before", before, "")
    after = tool.render_section("after", _results(), "e" * 64, 2)
    both = tool.render_page("after", after, page)
    assert both.startswith(tool.HEADER)
    assert tool.sections(both) == {"before": before, "after": after}
    assert both.index("## before — ") < both.index("## after — ")
