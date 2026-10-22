"""Generate docs/spec_index.jsonl and docs/theory/ledger_index.jsonl — one JSON object per line, one line per ID.

Purpose: let a session answer "what does ID X require now, what implements and tests it, what does the source say
         behind it" from one `grep '"M4.C.3a"'` instead of loading the 500 KB spec (docs/plan_model_ergonomics.md,
         Part B). Everything here is read from the spec, the ledger, the code and the tests; nothing is hand-written.
Spec:    docs/plan_model_ergonomics.md#B1, #B4
Tests:   tests/bowen/test_spec_index.py

    python3 tools/spec_index.py

The spec index (B1) has, per requirement ID in document order: ``id``; ``module`` and ``heading`` (the enclosing
``##`` and ``###`` headings); ``lines`` (the 1-based, inclusive span of its block, from the line defining the ID to the
line before the next ID or heading); ``first_sentence`` (generated, cut at 300 characters); ``withdrawn`` (the block
says WITHDRAWN); ``grade_tags`` (every ``[T]``/``[M]``/``[D]``/``[#]``/``[I]``/``[X]`` in the block, no single grade
inferred); ``sources`` (source IDs cited in the block, by ``tools/spec_index_rules.json``'s patterns);
``ledger_mentions`` (ledger entries whose text cites the ID); ``code`` (``path::qualname`` of every module, class or
function in ``src/`` whose docstring's ``Spec:`` line names the ID); ``tests``, ``status`` and ``status_note``
(``tools/spec_coverage.py``'s ``coverage()``, verbatim).

The ledger index (B4) has, per entry: ``id``, ``title``, ``status`` (the first bracketed label in the heading, as
written), ``labels`` (all of them), ``chapter`` (the enclosing ``##`` heading), ``file``, ``lines`` and ``spec_ids``
(the spec IDs the entry cites). Withdrawn and merged entries stay in, with their status, so a search can filter them.
The entries are ``_LEDGER.md``'s (the ``L`` chapters and the consolidated ``FE-``/``KS-`` entries) and, since
2026-10-22 (TODO, plan §11 trial: ``FE03.6`` was unreachable), the per-finding ``FE``/``KS``/``K``/``T`` entries of the
extraction files that ``spec_index_rules.json`` ``extraction_sources`` names: one record per ``## … <ID> —`` heading,
``chapter`` the file's ``# `` title, ``status`` empty (those headings carry stars, not a status). An ID defined twice
across the files, or an ID-like heading the ``entry_id`` pattern does not read, fails the run (P2).

``code_citations()`` also reports every ID a ``Spec:`` line names that the spec does not define (B3); the test fails
on any.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SPEC = REPO / "docs" / "bowen_agent_model_spec_v2.md"
LEDGER = REPO / "docs" / "theory" / "_LEDGER.md"
SRC = REPO / "src"
RULES = REPO / "tools" / "spec_index_rules.json"
OVERRIDES = REPO / "tools" / "spec_coverage_overrides.json"
SPEC_INDEX = REPO / "docs" / "spec_index.jsonl"
LEDGER_INDEX = REPO / "docs" / "theory" / "ledger_index.jsonl"
FIRST_SENTENCE_CHARS = 300

_coverage_spec = importlib.util.spec_from_file_location("spec_coverage_tool", REPO / "tools" / "spec_coverage.py")
coverage_tool = importlib.util.module_from_spec(_coverage_spec)
_coverage_spec.loader.exec_module(coverage_tool)

# A line that defines an ID (tests/test_spec_consistency.py's two forms): a bold ID at the line's start, with or
# without a list marker, or a bold ID in a table row's first cell.
DEF_LINE = re.compile(r"^(?:[-*]\s+)?\*\*(M\d+(?:\.[A-Za-z0-9]+)*)\*\*")
DEF_ROW = re.compile(r"^\| \*\*(M\d+(?:\.[A-Za-z0-9]+)*)\*\* \|")
HEADING = re.compile(r"^(#{1,6}) (.*)$")
RULE_LINE = re.compile(r"^-{3,}\s*$")  # a horizontal rule ends a block and belongs to none
GRADE_TAG = re.compile(r"\[(T|M|D|#|I|X)\]")
# A spec ID cited in prose: not inside a longer dotted ID on either side.
CITED_SPEC_ID = re.compile(r"(?<![A-Za-z0-9.])(M\d+(?:\.[A-Za-z0-9]+)*)(?![A-Za-z0-9]|\.[A-Za-z0-9])")
# The end of a sentence: a terminator, any closing emphasis or bracket, then a space and a capital, emphasis, code or
# bracket (tests/doc_text.py's rule, widened for a sentence that ends inside emphasis).
SENTENCE_END = re.compile(r"(?<=[.!?])[*_)\]\"”']*\s+(?=[A-Z*`(\[→“\"])")
# A ledger entry's heading: `##` or `###`, then its ID — a paper finding (L01.1), a consolidated entry (KS-A, KS-A.1,
# FE-B) or a resolution (U1) — then a separator. A chapter heading (`## Ch01 — …`) is not an entry.
LEDGER_ENTRY = re.compile(r"^(#{2,3}) ((?:[A-Z]{1,2}\d{2}\.\d+[a-z]?|[A-Z]{2}-[A-Z](?:\.\d+)?|U\d+))\b\s*(.*)$")
# A heading that looks like an entry (an upper-case token with digits, then a separator) but LEDGER_ENTRY does not
# read: counted and refused, so a new ID form cannot be folded into its neighbour silently (P2; L10.11a, 2026-10-22).
ENTRY_LIKE = re.compile(r"^#{2,3} [A-Z]{1,3}(?:-[A-Z])?\d*(?:\.\d+)?[a-z]?\s+[—·–-]")
LEDGER_LABEL = re.compile(r"`\[([^\]`]*)\]`")
# A per-finding heading in an extraction file (`## ⭐⭐⭐ FE03.1 — title`, `## ⚠ T6.4 — title`): `##`, optional marks
# (no letters, digits or brackets), the ID, a separator. The ID forms come from spec_index_rules.json.
EXTRACTION_MARKS = r"^## ([^\w\s`\[]*)\s*"
EXTRACTION_LIKE = re.compile(EXTRACTION_MARKS + r"([A-Z]{1,3}\d+(?:\.\d+)?[a-z]?)\s+[—·–-]")
# The `Spec:` block of a docstring: from the label to the next labelled line, a blank line or the end.
SPEC_BLOCK = re.compile(r"^[ \t]*Spec:[ \t]*(.*?)(?=^[ \t]*[A-Z][A-Za-z ]{0,20}:[ \t]|\n[ \t]*\n|\Z)", re.M | re.S)
# One citation in a `Spec:` block: an optional document, then `#` and the ID. A bare `#ID` continues the last document.
CITATION = re.compile(r"([\w./\\-]+\.md)?#([A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*)")


def rules() -> dict:
    return json.loads(RULES.read_text(encoding="utf-8"))


def source_pattern(loaded: dict | None = None) -> re.Pattern:
    patterns = {k: v for k, v in (loaded or rules())["source_id_patterns"].items() if not k.startswith("_")}
    return re.compile("|".join(f"(?:{p})" for p in patterns.values()))


# ----------------------------------------------------------------------------------------------------------------
# The spec
# ----------------------------------------------------------------------------------------------------------------

def spec_blocks(text: str) -> list[dict]:
    """Every requirement's block, in document order: id, module, heading, lines [start, end], row (a table row), and
    the block's text. A block runs from the line defining the ID to the line before the next ID or heading, trailing
    blank lines dropped."""
    lines = text.split("\n")
    blocks: list[dict] = []
    module = heading = ""
    open_block: dict | None = None

    def close(at: int) -> None:
        nonlocal open_block
        if open_block is None:
            return
        end = at
        while end > open_block["lines"][0] and not lines[end - 1].strip():
            end -= 1
        open_block["lines"] = [open_block["lines"][0], end]
        open_block["text"] = "\n".join(lines[open_block["lines"][0] - 1:end])
        blocks.append(open_block)
        open_block = None

    for number, line in enumerate(lines, start=1):
        found = HEADING.match(line)
        if found:
            close(number - 1)
            level, title = found.groups()
            if len(level) <= 2:
                module, heading = title.strip(), ""
            elif len(level) == 3:
                heading = title.strip()
            continue
        if RULE_LINE.match(line):
            close(number - 1)
            continue
        row = DEF_ROW.match(line)
        defined = row or DEF_LINE.match(line)
        if defined:
            close(number - 1)
            open_block = {"id": defined.group(1), "module": module, "heading": heading, "lines": [number, number],
                          "row": bool(row), "text": line}
            if row:
                close(number)
    close(len(lines))
    return blocks


def first_sentence(block: dict) -> str:
    """The requirement's first sentence after its ID (a table row: its first cell after the ID), generated, cut at
    FIRST_SENTENCE_CHARS. Where the first sentence is only an italic title (`**ID** — *the title.* The rule …`), the
    sentence after it comes too, so the field reads as the rule and not as its label (review 2026-10-22)."""
    first = block["text"].split("\n", 1)[0]
    if block["row"]:
        cells = [c.strip() for c in first.strip().strip("|").split("|")]
        body = cells[1] if len(cells) > 1 else ""
    else:
        body = DEF_LINE.sub("", first, count=1)
        body = re.sub(r"^\s*[—–-]?\s*", "", body)
    parts = SENTENCE_END.split(body, maxsplit=2)
    sentence = " ".join(parts[:2]) if body.startswith("*") and len(parts) > 1 else parts[0]
    return re.sub(r"\*+", "", sentence).strip()[:FIRST_SENTENCE_CHARS]  # emphasis markers dropped, code spans kept


def grade_tags(text: str) -> list[str]:
    seen: list[str] = []
    for tag in GRADE_TAG.findall(text):
        if tag not in seen:
            seen.append(tag)
    return seen


def sources(text: str, pattern: re.Pattern) -> list[str]:
    seen: list[str] = []
    for found in pattern.finditer(text):
        if found.group(0) not in seen:
            seen.append(found.group(0))
    return seen


def cited_spec_ids(text: str, defined: set[str]) -> list[str]:
    """The defined spec IDs the text cites, in order of first appearance."""
    seen: list[str] = []
    for found in CITED_SPEC_ID.finditer(text):
        if found.group(1) in defined and found.group(1) not in seen:
            seen.append(found.group(1))
    return seen


# ----------------------------------------------------------------------------------------------------------------
# The ledger
# ----------------------------------------------------------------------------------------------------------------

def ledger_entries(text: str) -> list[dict]:
    """Every ledger entry, in document order: id, title, status, labels, chapter, lines [start, end], and the entry's
    text. An entry runs from its heading to the line before the next entry heading or the next `#`/`##` heading; a
    `###` heading with no ID (a subsection of a consolidated entry) stays inside."""
    lines = text.split("\n")
    entries: list[dict] = []
    chapter = ""
    open_entry: dict | None = None

    def close(at: int) -> None:
        nonlocal open_entry
        if open_entry is None:
            return
        end = at
        while end > open_entry["lines"][0] and not lines[end - 1].strip():
            end -= 1
        open_entry["lines"] = [open_entry["lines"][0], end]
        open_entry["text"] = "\n".join(lines[open_entry["lines"][0] - 1:end])
        entries.append(open_entry)
        open_entry = None

    for number, line in enumerate(lines, start=1):
        found = HEADING.match(line)
        if not found:
            continue
        level, title = found.groups()
        entry = LEDGER_ENTRY.match(line)
        if not entry and ENTRY_LIKE.match(line) and not line.startswith(("## Ch", "### Ch")):
            raise ValueError(f"ledger line {number} looks like an entry heading LEDGER_ENTRY does not read: {line!r}")
        if entry:
            close(number - 1)
            marks, entry_id, rest = entry.groups()
            labels = LEDGER_LABEL.findall(rest)
            title = LEDGER_LABEL.sub("", rest)
            title = re.sub(r"\s{2,}", " ", re.sub(r"^\s*[—·–-]\s*", "", title)).strip()
            open_entry = {"id": entry_id, "title": title, "status": labels[0] if labels else "", "labels": labels,
                          "chapter": chapter if len(marks) == 3 else "", "lines": [number, number], "text": line}
            if len(marks) == 2:
                chapter = line[len(marks) + 1:].strip()
        elif len(level) <= 2:
            close(number - 1)
            chapter = title.strip() if len(level) == 2 else ""
    close(len(lines))
    return entries


# ----------------------------------------------------------------------------------------------------------------
# The code
# ----------------------------------------------------------------------------------------------------------------

def _docstring_nodes(tree: ast.Module):
    """(qualname, docstring) for the module and every class and function in it, with docstrings only."""
    stack: list[tuple[ast.AST, str]] = [(tree, "")]
    while stack:
        node, prefix = stack.pop()
        doc = ast.get_docstring(node, clean=False)
        if doc:
            yield prefix, doc
        body = getattr(node, "body", [])
        for child in reversed(body):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                stack.append((child, f"{prefix}.{child.name}" if prefix else child.name))


def spec_citations(doc: str) -> list[str]:
    """The IDs a docstring's `Spec:` block cites against the spec, in order. A bare `#ID` continues the document the
    previous citation named; a citation against another document (an implementation plan's S1.5a) is not one."""
    found: list[str] = []
    for block in SPEC_BLOCK.findall(doc):
        document = ""
        for named, cited in CITATION.findall(block):
            if named:
                document = named
            if document.endswith(SPEC.name) and cited not in found:
                found.append(cited)
    return found


def code_citations() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """(ID -> [path::qualname] for every citation in src/, where -> [IDs] for the citations the spec does not define)."""
    defined = set(coverage_tool.spec_ids(SPEC.read_text(encoding="utf-8")))
    cites: dict[str, list[str]] = {}
    unknown: dict[str, list[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        for qualname, doc in _docstring_nodes(ast.parse(path.read_text(encoding="utf-8"))):
            where = f"{rel}::{qualname}" if qualname else rel
            for cited in spec_citations(doc):
                if cited in defined:
                    cites.setdefault(cited, []).append(where)
                else:
                    unknown.setdefault(where, []).append(cited)
    return cites, unknown


# ----------------------------------------------------------------------------------------------------------------
# The two indexes
# ----------------------------------------------------------------------------------------------------------------

def withdrawn(block: dict) -> bool:
    """True when the requirement's defining line says WITHDRAWN. Read from that line only: a body that says another
    ID is withdrawn (M11.4's exceptions table) is not withdrawn itself (review 2026-10-22)."""
    return "WITHDRAWN" in block["text"].split("\n", 1)[0]


def spec_index(overrides: dict | None = None) -> list[dict]:
    text = SPEC.read_text(encoding="utf-8")
    ids = coverage_tool.spec_ids(text)
    blocks = spec_blocks(text)
    if [b["id"] for b in blocks] != ids:
        raise ValueError("the block parser and spec_coverage.spec_ids disagree on the IDs")
    pattern = source_pattern()
    defined = set(ids)
    mentions: dict[str, list[str]] = {}
    for entry in all_entries():
        for spec_id in cited_spec_ids(entry["text"], defined):
            mentions.setdefault(spec_id, []).append(entry["id"])
    cites, _ = code_citations()
    loaded = json.loads(OVERRIDES.read_text(encoding="utf-8")) if overrides is None else overrides
    rows = {row["id"]: row for row in coverage_tool.coverage(loaded)}
    records = []
    for block in blocks:
        row = rows[block["id"]]
        records.append({
            "id": block["id"],
            "module": block["module"],
            "heading": block["heading"],
            "lines": block["lines"],
            "first_sentence": first_sentence(block),
            "withdrawn": withdrawn(block),
            "grade_tags": grade_tags(block["text"]),
            "sources": sources(block["text"], pattern),
            "ledger_mentions": mentions.get(block["id"], []),
            "code": cites.get(block["id"], []),
            "tests": row["evidence"],
            "status": row["status"],
            "status_note": row["note"],
        })
    return records


def extraction_entry_pattern(loaded: dict | None = None) -> re.Pattern:
    return re.compile(EXTRACTION_MARKS + f"({(loaded or rules())['extraction_sources']['entry_id']})\\b\\s*(.*)$")


def extraction_files(loaded: dict | None = None) -> list[Path]:
    found: list[Path] = []
    for glob in (loaded or rules())["extraction_sources"]["globs"]:
        found += sorted(REPO.glob(glob))
    return found


def extraction_entries(text: str, pattern: re.Pattern | None = None) -> list[dict]:
    """Every per-finding entry of one extraction file, in order: id, title, status (always empty: these headings
    carry grade labels, not a status word), labels, chapter (the file's `# ` title), lines [start, end] and text. An
    entry runs from its `##` heading to the line before the next `#` or `##` heading; `###` headings stay inside. A
    `##` heading with an ID-like token the pattern does not read is refused (P2), as in the ledger."""
    pattern = extraction_entry_pattern() if pattern is None else pattern
    lines = text.split("\n")
    chapter = next((line[2:].strip() for line in lines if line.startswith("# ")), "")
    entries: list[dict] = []
    open_entry: dict | None = None

    def close(at: int) -> None:
        nonlocal open_entry
        if open_entry is None:
            return
        end = at
        while end > open_entry["lines"][0] and not lines[end - 1].strip():
            end -= 1
        open_entry["lines"] = [open_entry["lines"][0], end]
        open_entry["text"] = "\n".join(lines[open_entry["lines"][0] - 1:end])
        entries.append(open_entry)
        open_entry = None

    for number, line in enumerate(lines, start=1):
        if not line.startswith("#") or line.startswith("###"):
            continue
        close(number - 1)
        entry = pattern.match(line)
        if not entry:
            if EXTRACTION_LIKE.match(line):
                raise ValueError(f"extraction line {number} looks like an entry heading the pattern does not read: {line!r}")
            continue
        _, entry_id, rest = entry.groups()
        labels = LEDGER_LABEL.findall(rest)
        title = re.sub(r"\s{2,}", " ", re.sub(r"^\s*[—·–-]\s*", "", LEDGER_LABEL.sub("", rest))).strip()
        open_entry = {"id": entry_id, "title": title, "status": "", "labels": labels, "chapter": chapter,
                      "lines": [number, number], "text": line}
    close(len(lines))
    return entries


def all_entries() -> list[dict]:
    """The ledger's entries, then every extraction file's, each with `file` (repo-relative). IDs are unique across
    the whole set, or the index is refused."""
    entries = [dict(e, file=LEDGER.relative_to(REPO).as_posix()) for e in ledger_entries(LEDGER.read_text(encoding="utf-8"))]
    pattern = extraction_entry_pattern()
    for path in extraction_files():
        entries += [dict(e, file=path.relative_to(REPO).as_posix())
                    for e in extraction_entries(path.read_text(encoding="utf-8"), pattern)]
    seen: dict[str, str] = {}
    for e in entries:
        if e["id"] in seen:
            raise ValueError(f"entry {e['id']} is defined in both {seen[e['id']]} and {e['file']}")
        seen[e["id"]] = e["file"]
    return entries


def ledger_index() -> list[dict]:
    defined = set(coverage_tool.spec_ids(SPEC.read_text(encoding="utf-8")))
    return [{"id": e["id"], "title": e["title"], "status": e["status"], "labels": e["labels"], "chapter": e["chapter"],
             "file": e["file"], "lines": e["lines"], "spec_ids": cited_spec_ids(e["text"], defined)}
            for e in all_entries()]


def render(records: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def main(argv: list[str]) -> int:
    SPEC_INDEX.write_text(render(spec_index()), encoding="utf-8")
    LEDGER_INDEX.write_text(render(ledger_index()), encoding="utf-8")
    _, unknown = code_citations()
    print(f"written: {SPEC_INDEX.relative_to(REPO)}, {LEDGER_INDEX.relative_to(REPO)}")
    for where, cited in sorted(unknown.items()):
        print(f"B3: {where} cites IDs the spec does not define: {', '.join(cited)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
