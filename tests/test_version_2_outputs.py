"""Version 2's readout pages carry `M11.F.9`'s framing (W8).

Purpose: W8 of `docs/version_2_plan.md` §4, as V7 for version 1: the review pages version 2 added that render a run of
         the reference family each say it is not a real family, so a regenerated page that loses the line fails here.
         Whether the framing is enough is the owner's call (`docs/REVIEW_version_2_outputs.md`).
Spec:    docs/bowen_agent_model_spec_v2.md#M11.F.9
Tests:   this file
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FRAMING = "Not a real family (`M11.F.9`)."

# The pages version 2 added that render runs of the reference family, each written by its tool in `tools/`.
READOUT_PAGES = (
    "docs/review/group_g_readout.md",
    "docs/review/stage_3_displaced_coping_probe.md",
    "docs/review/stage_3_iposition_probe.md",
    "docs/review/stage_3_projection_trace.md",
    "docs/review/t176_negative_run_trace.md",
)


def unframed(pages: dict[str, str]) -> list[str]:
    """W8: the pages whose text does not carry the framing line (whitespace folded)."""
    return sorted(name for name, text in pages.items() if FRAMING not in " ".join(text.split()))


def test_w8_every_version_2_readout_page_carries_the_framing():
    pages = {p: (REPO / p).read_text(encoding="utf-8") for p in READOUT_PAGES}
    assert unframed(pages) == []


def test_w8_a_page_without_the_framing_is_reported():
    """W8 (adversarial): a page that names the clause without the sentence, or says it of something else, fails."""
    pages = {"a.md": "Not a real family (`M11.F.9`).\n", "b.md": "See `M11.F.9`.\n",
             "c.md": "Not a real person (`M11.F.9`).\n", "d.md": "Not a real\nfamily (`M11.F.9`). More.\n"}
    assert unframed(pages) == ["b.md", "c.md"]
