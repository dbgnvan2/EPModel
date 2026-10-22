"""A status claim about a document agrees with that document's own status line.

Purpose: D5b. The checklist, README.md and other documents' status lines say whether a document is open (a draft, for
         review) or settled (approved, decided, built). That is a copy of the document's own status line, and it goes
         stale when the document moves on. Both directions fail: a claim that says open about a settled document
         (D5b.3), and a claim that says settled about an open one (D5b.4, owner decision Q1, 2026-10-20).
Spec:    docs/plan_doc_claim_tests.md#D5b
Tests:   this file
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from doc_text import claims_data, own_status, phrase_pattern, sentences, tracked_files

REPO = Path(__file__).resolve().parents[1]
CHECKLIST = "docs/SPEC_REVIEW_CHECKLIST.md"

_WORDS = claims_data()["status_words"]
_UNCLASSIFIED = claims_data()["status_unclassified"]
_OPEN = phrase_pattern(_WORDS["open"])
_SETTLED = phrase_pattern(_WORDS["settled"])
# A settled word after a negator and any fillers: "not built", "not yet approved", "never approved", "nothing decided".
_NEGATED = re.compile(
    r"\b(?:" + "|".join(map(re.escape, _WORDS["negators"])) + r")\s+(?:(?:"
    + "|".join(map(re.escape, _WORDS["negation_fillers"])) + r")\s+)*(?:" + _SETTLED.pattern + ")", re.IGNORECASE)
_NEITHER_LABEL = re.compile("|".join(re.escape(label) for label in _UNCLASSIFIED["labels"]), re.IGNORECASE)
NO_REVIEW_STATUS = {p: why for p, why in _UNCLASSIFIED["documents"].items() if not p.startswith("_")}
_FIRST_SENTENCE = re.compile(r"(?<=[.!?])\s")
# A checklist entry: a bold label ending in a colon, then the entry up to the next blank line.
_ENTRY = re.compile(r"^\*\*([^*\n]{3,80}?):\*\*\s*(.*?)(?=\n\s*\n|\Z)", re.M | re.S)
_LINK = re.compile(r"\]\(<([^>]+)>\)|\]\(([^)\s]+)\)")
_SPAN = re.compile(r"`([^`\n]+\.md)`")


def classify(text: str | None) -> str | None:
    """D5b.1: "open", "settled" or None, read from the first sentence. The first open or settled word decides, so
    "APPROVED, awaiting the build" is settled and "Draft for owner approval" open; a negated settled word is not one."""
    first = _FIRST_SENTENCE.split((text or "").strip(), 1)[0]
    unnegated = _NEGATED.sub(lambda m: " " * len(m.group(0)), first)
    hits = [(m.start(), "open") for m in _OPEN.finditer(first)]
    hits += [(m.start(), "settled") for m in _SETTLED.finditer(unnegated)]
    return min(hits)[1] if hits else None


@dataclass
class StatusScan:
    """D5b: each claim compared (where, what it says, the document it names), those that disagree, and what was
    skipped without being listed in `tests/doc_claims.json`'s `status_unclassified`."""
    compared: list[tuple[str, str, str]] = field(default_factory=list)
    disagree: list[str] = field(default_factory=list)
    unaccounted: list[str] = field(default_factory=list)


def status_scan(texts: dict[str, str]) -> StatusScan:
    """D5b.2–D5b.4: every claim in the checklist's labels, README.md's sentences and the `docs/` status lines that
    names a `docs/` document with a status line of its own, compared with that status line."""
    statuses = {p: s for p, text in texts.items() if p.startswith("docs/") and (s := own_status(text))}
    claims = []  # (where, claim text, the document it names)
    scan = StatusScan()
    for match in _ENTRY.finditer(texts.get(CHECKLIST, "")):
        label = match.group(1)
        if not classify(label) and not _NEITHER_LABEL.match(label):
            scan.unaccounted.append(f"{CHECKLIST}: the label {label!r} is neither open nor settled, and is not listed")
        targets = _targets(CHECKLIST, match.group(2), links_only=True)
        claims += [(CHECKLIST, label, targets[0])] if targets else []
    for where, text in [("README.md", texts.get("README.md", ""))] + sorted(statuses.items()):
        if where == CHECKLIST:
            continue
        claims += [(where, s, target) for s in sentences(text) for target in _targets(where, s)]
    for where, claim_text, target in claims:
        said, actual = classify(claim_text), classify(statuses.get(target))
        if target == where or not said:
            continue
        if not actual:
            if target not in NO_REVIEW_STATUS:
                scan.unaccounted.append(f"{where} calls {target} {said} ({claim_text[:90]!r}); its own status "
                                        f"{statuses.get(target)!r} is neither open nor settled, and it is not listed")
            continue
        scan.compared.append((where, said, target))
        if said != actual:
            scan.disagree.append(f"{where} calls {target} {said} ({claim_text[:90]!r}); its own status is {actual}: "
                                 f"{statuses[target][:110]!r}")
    return scan


def _targets(where: str, text: str, links_only: bool = False) -> list[str]:
    """The `docs/` Markdown documents a text names, in order: link targets, and backticked paths unless links_only.

    A link is relative to the naming document's directory; a backticked path from `docs/` is relative to the repo.
    """
    base = posixpath.dirname(where)
    found = []
    for match in re.finditer(_LINK.pattern + (r"" if links_only else "|" + _SPAN.pattern), text):
        link = match.group(1) or match.group(2)
        span = None if links_only else match.group(3)
        if link and not re.match(r"[a-z]+:|#", link):
            found.append(posixpath.normpath(posixpath.join(base, link.split("#")[0])))
        elif span:
            found.append(span if span.startswith("docs/") else posixpath.normpath(posixpath.join(base, span)))
    return [p for p in found if p.startswith("docs/") and p.endswith(".md")]


def _real_texts() -> dict[str, str]:
    return {p: (REPO / p).read_text(encoding="utf-8")
            for p in tracked_files() if p == "README.md" or (p.startswith("docs/") and p.endswith(".md"))}


def test_d5b_status_claims_agree_with_each_documents_own_status():
    """D5b.3, D5b.4: the real repo. Nothing is skipped unless the data file lists it."""
    scan = status_scan(_real_texts())
    assert scan.disagree == []
    assert scan.unaccounted == []


def test_d5b_the_scan_is_not_vacuous():
    """D5b.5: the checklist's entry for the version 2 plan, settled on both sides, is among the pairs compared, and
    the checklist and README each compare at least one pair. Until revision 35 (2026-10-22) the spec's status line
    was a third source, naming the SP-1 proposal as approved; that text moved into the body of
    docs/spec_revision_log.md, where the scan does not read it, so its cross-document claims are now compared with
    nothing and no third source is asserted (test_d5b_readme_and_status_lines_are_read keeps the status-line path
    exercised)."""
    compared = status_scan(_real_texts()).compared
    assert (CHECKLIST, "settled", "docs/version_2_plan.md") in compared
    assert any(where == "README.md" for where, _, _ in compared)


def test_d5b_the_unclassified_lists_are_current():
    """D5b.2: each listed label still starts a checklist label, and each listed document is tracked and still has no
    open or settled status, so the lists cannot outlive what they excuse."""
    texts = _real_texts()
    labels = [m.group(1) for m in _ENTRY.finditer(texts[CHECKLIST])]
    assert [w for w in _UNCLASSIFIED["labels"] if not any(label.lower().startswith(w.lower()) for label in labels)] == []
    assert [p for p in NO_REVIEW_STATUS if p not in texts or classify(own_status(texts[p]))] == []


def test_d5b_an_unknown_wording_is_not_skipped_silently():
    """D5b.2 (adversarial): a label, or a document's own status, in words outside both lists is reported, not passed."""
    texts = {
        CHECKLIST: "**Under consideration, 2026-10-21:** [a.md](a.md) — a plan.\n\n"
                   "**Approved 2026-10-21:** [b.md](b.md) — a plan.\n",
        "docs/a.md": "# A\n\n**Status: APPROVED 2026-10-21.**\n",
        "docs/b.md": "# B\n\n**Status: UNDER REVIEW since 2026-10-21.**\n",
    }
    scan = status_scan(texts)
    assert scan.disagree == [] and len(scan.unaccounted) == 2
    assert "Under consideration" in scan.unaccounted[0] and "docs/b.md" in scan.unaccounted[1]


def test_d5b_a_reviewed_document_called_for_review_is_reported():
    """D5b.3: the checklist's label for the review of revisions 24–30 as it stood on 2026-10-20."""
    texts = {
        CHECKLIST: "**For review, 2026-10-15:** [REVIEW_spec_revisions_24-30.md](REVIEW_spec_revisions_24-30.md) — "
                   "the owner's review form.\n",
        "docs/REVIEW_spec_revisions_24-30.md": "# Review\n\n**Status: REVIEWED BY THE OWNER 2026-10-16.** All 85 "
                                               "entries decided.\n",
    }
    disagree = status_scan(texts).disagree
    assert len(disagree) == 1 and "REVIEW_spec_revisions_24-30.md" in disagree[0] and "open" in disagree[0]


def test_d5b_a_decided_label_on_a_draft_is_reported():
    """D5b.4: the checklist says decided; the proposal's own status line still says draft."""
    name = "PROPOSAL — CUTOFF ANXIETY, BRIDGING AND REPAIR.md"
    texts = {
        CHECKLIST: f"**Decided 2026-10-19, not built:** [<{name}>](<{name}>) — G1.\n",
        f"docs/{name}": "# G1\n\n**Status: DRAFT 2026-10-19, for owner review.** Q1–Q11 in §8, all decided.\n",
    }
    disagree = status_scan(texts).disagree
    assert len(disagree) == 1 and name in disagree[0] and "settled" in disagree[0]


def test_d5b_readme_and_status_lines_are_read():
    """D5b.2: a README.md sentence and another document's status line are claims too; agreeing ones give nothing."""
    texts = {
        "README.md": "The plan (`docs/a.md`) is a draft for review.\n\nThe record (`docs/b.md`) is approved.\n",
        "docs/a.md": "# A\n\n**Status: APPROVED 2026-10-19.**\n",
        "docs/b.md": "# B\n\n**Status: APPROVED 2026-10-19.**\n",
        "docs/c.md": "# C\n\n**Status: DRAFT, for owner review. Builds on [a.md](a.md), which is approved.**\n",
    }
    scan = status_scan(texts)
    assert ("README.md", "settled", "docs/b.md") in scan.compared
    assert ("docs/c.md", "settled", "docs/a.md") in scan.compared
    assert len(scan.disagree) == 1 and "README.md" in scan.disagree[0] and "docs/a.md" in scan.disagree[0]


@pytest.mark.parametrize("text, expected", [
    ("not approved text", "open"),
    ("a report, not an approval", None),
    ("Decided 2026-10-19, not built", "settled"),
    ("nothing decided, nothing built", None),
    ("DRAFT 2026-10-18, for owner review", "open"),
    ("APPROVED 2026-10-20, not built. The owner approved the plan, still a draft for review.", "settled"),
    ("Generated, 2026-10-19", None),
    ("REVIEWED BY THE OWNER 2026-10-16", "settled"),
    ("approval pending", None),
    # The first status word decides, and any negator counts (the learning-qa review, 2026-10-20).
    ("Not yet approved", "open"),
    ("Never approved", None),
    ("Not complete", None),
    ("APPROVED 2026-10-19, awaiting the build", "settled"),
    ("Approved 2026-10-19, replacing the draft of 2026-10-18", "settled"),
    ("Draft for owner approval, 2026-10-19", "open"),
    ("For owner approval", "open"),
    ("DECIDED by the owner 2026-10-19 (Q1–Q11 in §8); drafted the same day.", "settled"),
])
def test_d5b_classification_handles_negation(text, expected):
    """D5b.1 (adversarial): negation, case, the first sentence only, and words that look like a status and are not."""
    assert classify(text) == expected
