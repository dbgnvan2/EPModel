"""V7's mechanical part: every output kind still carries the framing `docs/REVIEW_version_1_outputs.md` lists.

Purpose: fail when a regenerated trace, export or record loses the line that says it is not about a real family
         (`M11.F.9`), or when "virtual family" appears in a document other than the two that frame it (`M11.F.10`).
         Whether the framing is enough is the owner's review (V7), not this test.
Spec:    docs/version_1_definition.md#1 (V7); docs/bowen_agent_model_spec_v2.md#M11.F.9, #M11.F.10
Tests:   this file
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"


def test_v7_every_rendered_trace_carries_the_framing_block():
    traces = sorted((DOCS / "review").glob("*_trace_*.md"))
    assert traces, "no rendered traces found"
    for trace in traces:
        text = trace.read_text(encoding="utf-8")
        assert "It does not concern any real family or person, and nothing here is advice (`M11.F.9`)" in text, trace.name
        assert "`M11.F.10`" in text, trace.name


def test_v7_the_famdiag_export_says_it_is_not_a_real_family():
    for export in sorted((DOCS / "review").glob("*famdiag*.json")):
        assert "not a real family (spec M11.F.9)" in export.read_text(encoding="utf-8"), export.name


def test_v7_every_ensemble_record_says_a_premise_pass_is_not_a_finding_about_families():
    for name in ("phase_c_ensemble_record.md", "phase_d_ensemble_record.md", "phase_d2_ensemble_record.md"):
        text = " ".join((DOCS / name).read_text(encoding="utf-8").split())
        assert "it is not a finding about families" in text, name


def test_v7_virtual_family_appears_only_where_it_is_framed():
    """`M11.F.10`: the term only beside its framing. The spec defines the rule; the explainer's §1.1 is the framing."""
    allowed = {"bowen_agent_model_spec_v2.md", "model_explainer.md", "REVIEW_version_1_outputs.md"}
    using = sorted(p.name for p in DOCS.glob("*.md") if "virtual family" in p.read_text(encoding="utf-8").lower())
    assert set(using) <= allowed, set(using) - allowed
    assert "virtual family" not in (REPO / "README.md").read_text(encoding="utf-8").lower()
