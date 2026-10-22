"""The current-rules view of the spec is current, complete, and removes only what C2 allows.

Purpose: docs/spec_rules/ is generated from the spec; a stale or over-trimmed view would read as the contract while
         saying less than it (P6). The real-spec checks prove no body line was dropped; the fixtures prove each
         removal on text that cannot pass by luck.
Spec:    docs/plan_model_ergonomics.md#C1, #C2, #C3, #C4, #C5
Tests:   this file
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("spec_rules_view_tool", REPO / "tools" / "spec_rules_view.py")
TOOL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TOOL)
INDEX = TOOL.index_tool


def _spec_text() -> str:
    return TOOL.SPEC.read_text(encoding="utf-8")


def _on_disk() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(TOOL.OUT.iterdir()) if p.is_file()}


def _module_files(files: dict[str, str]) -> list[tuple[str, str]]:
    return sorted((n, c) for n, c in files.items() if re.fullmatch(r"M\d\d\.md", n))


def test_c5_view_is_current_and_holds_nothing_else():
    expected = TOOL.render_all(_spec_text())
    actual = _on_disk()
    assert sorted(actual) == sorted(expected), "regenerate: python3 tools/spec_rules_view.py (and remove what it does not write)"
    stale = [name for name in expected if actual[name] != expected[name]]
    assert stale == [], f"regenerate: python3 tools/spec_rules_view.py; stale: {stale}"


def test_c1_every_file_opens_with_the_non_normative_line():
    rev = TOOL.revision(_spec_text())
    opener = TOOL.OPENER.format(revision=rev)
    for name, content in _on_disk().items():
        assert opener in content.split("\n\n", 2)[0] + "\n\n" + content.split("\n\n", 2)[1], name
    assert f"revision {rev}" in opener and "Do not edit" in opener


def test_c5_every_spec_id_appears_in_exactly_one_view_file():
    ids = INDEX.coverage_tool.spec_ids(_spec_text())
    seen: dict[str, list[str]] = {}
    for name, content in _module_files(_on_disk()):
        for line in content.split("\n"):
            found = INDEX.DEF_ROW.match(line) or INDEX.DEF_LINE.match(line)
            if found:
                seen.setdefault(found.group(1), []).append(name)
    assert sorted(seen) == sorted(ids)
    assert {i: f for i, f in seen.items() if len(f) != 1} == {}


def test_c3_every_body_line_of_the_spec_survives_in_order():
    """C2 and C3 on the real spec: every line before the revision cut that is not a block quote, with its tags
    removed, is in the view, in document order — so nothing but the four removals happened. Mutation: drop any line
    the tool should keep."""
    text = _spec_text()
    kept = []
    for line in TOOL.cut_history(TOOL.strip_status(text)).split("\n"):
        if TOOL.QUOTE_LINE.match(line):
            continue
        stripped = TOOL.TAG.sub("", line).rstrip()
        if stripped:
            kept.append(stripped)
    view = []
    for _, content in _module_files(_on_disk()):
        body = content.split("\n", 2)[2]  # after the opener and its blank line
        view += [line for line in body.split("\n") if line]
    assert view == kept
    assert any(line.startswith("**M1.A.4k**") and "*Revision 19 (amended at revision 23" in line for line in view), \
        "a kept history sentence (a stays row of docs/spec_revision_35_dispositions.md)"
    assert not any("⟦" in line or line.startswith(">") for line in view)
    assert not any(line.startswith("status:") for line in view) and "version: 2.0, revision" in "\n".join(view)


FIXTURE = """---
tags: [spec]
version: 2.0, revision 7
status: APPROVED, a long line ⟦rev7⟧
date: 2026-01-01
---

# A spec ⟦rev1 · tag in a title⟧

## 0. Status

### 0.1 Scope

Kept. ⟦rev2⟧

## M1 — Objects

**M1.A.1** — *a rule.* Second sentence; it was changed at revision 3. *Revision 4:* the floor is the usual level. ⟦rev4 · owner⟧
> "a quotation, removed"
> continued

Previously the rule read otherwise (revision 2); re-derived at revision 5.

## M11 — Criteria

| **M11.C.1** | A row ⟦rev11⟧ | `t` |

## M11.F — Framing

**M11.F.1** Framing stays with M11. Earlier revisions said otherwise.

### Revision 7 — history, dropped

- M99.9 was added here; the heading is history, not a module, and nothing is defined after the cut.

### Revision 6 — more history
"""


def test_c2_removals_are_mechanical_only():
    """Mutation: keep the status line, keep a tag, keep a quote line, cut at the wrong heading, or file M11.F apart."""
    files = TOOL.modules(FIXTURE)
    assert list(files) == ["M00.md", "M01.md", "M11.md"]
    assert files["M00.md"] == ("---\ntags: [spec]\nversion: 2.0, revision 7\ndate: 2026-01-01\n---\n\n# A spec\n\n"
                               "## 0. Status\n\n### 0.1 Scope\n\nKept.\n")
    assert files["M01.md"] == ("## M1 — Objects\n\n**M1.A.1** — *a rule.* Second sentence; it was changed at revision 3. "
                               "*Revision 4:* the floor is the usual level.\n\nPreviously the rule read otherwise "
                               "(revision 2); re-derived at revision 5.\n")
    assert files["M11.md"] == ("## M11 — Criteria\n\n| **M11.C.1** | A row | `t` |\n\n## M11.F — Framing\n\n"
                               "**M11.F.1** Framing stays with M11. Earlier revisions said otherwise.\n")
    assert TOOL.revision(FIXTURE) == 7
    import pytest
    for tail in ("\n**M99.9** Defined after the cut.\n", "\n## M12 — A module after the cut\n"):
        with pytest.raises(ValueError, match="follows the history cut"):
            TOOL.cut_history(FIXTURE + tail)
    rendered = TOOL.render_all(FIXTURE)
    assert set(rendered) == {"M00.md", "M01.md", "M11.md", "README.md", "history_in_body.md"}
    assert rendered["M01.md"].startswith(TOOL.OPENER.format(revision=7) + "\n\n## M1")


def test_c4_history_sentences_are_listed_by_phrase():
    """A capitalised `Revision N` and the phrases match, a lower-case `revision N` after a preposition too; a bare
    cross-reference `(revision 2)` does not. Mutation: `Revision \\d+` matched in any case, or a phrase dropped."""
    AT = "\\b(?:at|in|since|by|from|until|before|after) revision \\d+"
    found = {e["id"]: e for e in TOOL.history_in_body(FIXTURE)}
    assert list(found) == ["M1.A.1", "M11.F.1"]
    sentences = found["M1.A.1"]["sentences"]
    assert sentences == [
        ("Second sentence; it was changed at revision 3.", ["was changed", AT]),
        ("*Revision 4:* the floor is the usual level.", ["Revision \\d+"]),
        ("Previously the rule read otherwise (revision 2); re-derived at revision 5.", ["re-derived at", "previously", AT]),
    ]
    assert found["M11.F.1"]["sentences"] == [("Earlier revisions said otherwise.", ["Earlier revisions"])]
    assert "M99.9" not in found, "text after the revision cut is not read"
    assert TOOL.history_in_body("**M1.A.2** Cites `M1.A.1` (revision 22) and nothing else.\n") == []
    lower = TOOL.history_in_body("**M1.A.2** The floor, corrected at revision 7, is the usual level.\n")
    assert lower and lower[0]["sentences"][0][1] == [AT]


def test_c4_the_real_list_quotes_the_spec_and_states_its_count():
    text = _spec_text()
    page = (TOOL.OUT / "history_in_body.md").read_text(encoding="utf-8")
    found = TOOL.history_in_body(text)
    assert re.search(r"\*\*(\d+) IDs\*\*", page).group(1) == str(len(found))
    assert len(re.findall(r"^## M", page, re.M)) == len(found)
    flat = re.sub(r"\s+", " ", TOOL.TAG.sub("", text))
    for entry in found:
        for sentence, _ in entry["sentences"]:
            assert re.sub(r"\s+", " ", sentence) in flat, f"{entry['id']}: a listed sentence is not in the spec"
    # After revision 35 (Part E) every listed sentence is one "stays" row of the dispositions record, matched
    # sentence by sentence (an ID with a "stays" row may also have moved rows), and the counts are the record's.
    stays = [(row["id"], row["key"]) for row in _disposition_rows() if row["disposition"] == "stays"]
    listed = [(e["id"], _norm(s)) for e in found for s, _ in e["sentences"]]
    unmatched = [(i, s[:80]) for i, s in listed
                 if not any(i == ri and all(k in s.rstrip("*. ") for k in key) for ri, key in stays)]
    assert unmatched == []
    assert len(listed) == len(stays) and len({i for i, _ in listed}) == len({i for i, _ in stays})


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _present(piece: str, text: str, window: int = 36) -> bool:
    """Whether an uncut piece of a recorded sentence is in `text`: whole when short, else by any run of `window`
    characters, since the record and the log annotate the same sentence differently at its ends."""
    if len(piece) < window:
        return piece in text
    return any(piece[i:i + window] in text for i in range(len(piece) - window + 1))


def _disposition_rows() -> list[dict]:
    """The rows of docs/spec_revision_35_dispositions.md: number, ID, the sentence as it stood (cut to ~120 characters,
    `…` marking each cut), its disposition, and `key`, the uncut pieces normalised for matching."""
    record = (REPO / "docs" / "spec_revision_35_dispositions.md").read_text(encoding="utf-8")
    rows = []
    for line in record.split("\n"):
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line)]
        if len(cells) >= 6 and cells[1].isdigit():
            sentence = cells[3].replace("\\|", "|")
            key = [_norm(piece).strip("*. ") for piece in sentence.split("…") if len(_norm(piece).strip("*. ")) >= 12]
            rows.append({"number": int(cells[1]), "id": cells[2].strip("`"), "sentence": sentence,
                         "disposition": cells[4], "key": key})
    return rows


def test_e2_every_dispositions_row_is_true_of_the_spec_and_the_log():
    """Part E2 (revision 35): each row of the dispositions record holds now. A moved or rewritten sentence is in
    docs/spec_revision_log.md under its ID; a moved sentence is no longer in the spec; a kept sentence is. The owner
    reverses rows by hand, so a sentence lost on the way would otherwise go unnoticed (review 2026-10-22)."""
    rows = _disposition_rows()
    assert len(rows) == 218 and [r["number"] for r in rows] == list(range(1, 219))
    assert {r["disposition"] for r in rows} == {"stays", "moves", "rewritten"}
    text = _spec_text()
    spec = _norm(TOOL.TAG.sub("", text))
    blocks = {b["id"]: _norm(TOOL.TAG.sub("", b["text"])) for b in TOOL.index_tool.spec_blocks(text)}
    log = _norm(TOOL.TAG.sub("", (REPO / "docs" / "spec_revision_log.md").read_text(encoding="utf-8")))
    wrong = []
    for row in rows:
        where = f'{row["number"]} {row["id"]} ({row["disposition"]})'
        body = blocks.get(row["id"], spec)  # the ID's own block: a short piece can be boilerplate elsewhere
        in_log = any(_present(piece, log) for piece in row["key"])
        in_body = any(_present(piece, body) for piece in row["key"])
        if row["disposition"] in ("moves", "rewritten") and not in_log:
            wrong.append(where + ": not in the log")
        if row["disposition"] == "moves" and all(piece in body for piece in row["key"]):  # whole, not by window
            wrong.append(where + ": still in the spec")
        if row["disposition"] == "stays" and not in_body:
            wrong.append(where + ": not in the spec")
    assert wrong == []
