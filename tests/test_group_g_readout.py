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
         "firsts": [("ravi-marta", 12, 4, 2, 5, True)], "excess": 10.0, "symptom_share": 0.2},
        {"family": "phase_c", "seed": 2, "cuts": [], "firsts": [], "excess": 20.0, "symptom_share": 0.4},
        {"family": "phase_d", "seed": 1, "cuts": [("a-b", 100, 7.0, 1.0)], "firsts": [("a-b", 100, 6, 1, 3, True)],
         "excess": 1.0, "symptom_share": 0.0},
        {"family": "phase_d", "seed": 2, "cuts": [("a-c", 200, 9.0, 1.5)], "firsts": [("a-c", 200, 2, 3, 7, True)],
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
    assert c[4:14] == ["65", "1 of 2", "1", "0", "4", "4", "2", "5", "15", "0.3"]
    d = _row(section, "phase_d")
    assert d[3] == "9 (`a-c`, week 200, seed 2, bond 1.5)"
    assert d[4:14] == ["8", "2 of 2", "2", "0", "4", "4", "2", "5", "2", "0.25"]
    comment = next(line for line in section.splitlines() if line.startswith("<!-- after measurements: "))
    data = ast.literal_eval(comment.removeprefix("<!-- after measurements: ").removesuffix(" -->"))
    assert data == {"engine": "e" * 64, "seeds": 2,
                    "largest_stock": {"phase_c": 90.0, "phase_d": 9.0}, "mean_run": {"phase_c": 4, "phase_d": 4},
                    "unmatched": {"phase_c": 0, "phase_d": 0},
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


def test_m11c48_t166_readout_leaves_unmatched_first_cutoffs_out_of_the_run_columns_and_counts_them():
    """T-166: a first cutoff with no uncut tie in its week (``matched`` False) has no baseline, so the run columns,
    `M11.C.48`'s two arms, leave it out (T-123's rule) and the "unmatched" column counts it. Here Phase D's seed 2
    gains an unmatched first cutoff with a long run (30) and a baseline of 0: read as `M11.C.48` read it before T-123,
    it would lift the mean run to 12.67, lower the matched baseline to 1.33 and lift the longest uncut run to 20.
    Mutation: the run columns read every first cutoff again (the pre-T-123 bias) — caught by the means."""
    results = _results()
    results[3] = dict(results[3], firsts=results[3]["firsts"] + [("a-d", 300, 30, 0, 50, False)])
    section = tool.render_section("after", results, "e" * 64, 2)
    d = _row(section, "phase_d")
    assert d[5:12] == ["2 of 2", "3", "1", "4", "4", "2", "5"]
    comment = next(line for line in section.splitlines() if line.startswith("<!-- after measurements: "))
    data = ast.literal_eval(comment.removeprefix("<!-- after measurements: ").removesuffix(" -->"))
    assert data["mean_run"]["phase_d"] == 4 and data["unmatched"] == {"phase_c": 0, "phase_d": 1}
