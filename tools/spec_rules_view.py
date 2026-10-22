"""Generate docs/spec_rules/ — the spec's current rules, one file per module, with its history taken out.

Purpose: a session that needs the rules of one module reads a view a fraction of the spec's size instead of the
         whole 500 KB contract (docs/plan_model_ergonomics.md, Part C). The spec stays the source of truth; the view
         is not normative, and where they differ the spec wins.
Spec:    docs/plan_model_ergonomics.md#C1, #C2, #C3, #C4
Tests:   tests/bowen/test_spec_rules_view.py

    python3 tools/spec_rules_view.py

What the view removes, and nothing else (C2): the front matter's ``status:`` line (``version:`` stays); every
``⟦…⟧`` revision tag; every block-quote line (the source quotations; the ``→ file · ID`` pointers stay); and the
revision-history sections, from the first ``### Revision N`` heading to the end of the file, where the spec has
any (at revision 35 they moved to ``docs/spec_revision_log.md``, so the cut finds nothing and removes nothing). No
sentence of a requirement body is removed (C3), the ``*Revision N:*`` sentences included: some carry the current rule.

Files: ``M00.md`` (the title, the front matter and §0), ``M01.md`` … ``M17.md`` (one per ``## M<n>`` module; ``M11.F``
and ``M11.G`` are in ``M11.md``), ``README.md`` (the files, their sizes, and the view's size against the spec's) and
``history_in_body.md`` (C4: every ID whose body holds a sentence matching a history phrase; the phrases are in
``tools/spec_index_rules.json``. It was the input to Part E, which at revision 35 moved or rewrote 183 of the 218
sentences it listed; what it lists now is rule text that matches a phrase, each row of
``docs/spec_revision_35_dispositions.md`` marked "stays").
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "docs" / "bowen_agent_model_spec_v2.md"
OUT = REPO / "docs" / "spec_rules"
RULES = REPO / "tools" / "spec_index_rules.json"

_index_spec = importlib.util.spec_from_file_location("spec_index_tool", REPO / "tools" / "spec_index.py")
index_tool = importlib.util.module_from_spec(_index_spec)
_index_spec.loader.exec_module(index_tool)

TAG = re.compile(r"[ \t]*⟦[^⟧]*⟧")
QUOTE_LINE = re.compile(r"^>")
REVISION_HEADING = re.compile(r"^### Revision \d+")
MODULE_HEADING = re.compile(r"^## (?:M(\d+)\b|(0)\.)")
FRONT_REVISION = re.compile(r"^version:.*?revision (\d+)", re.M)
STATUS_LINE = re.compile(r"^status:.*\n", re.M)
OPENER = ("*Generated from the spec at revision {revision} by `tools/spec_rules_view.py`. Not normative: where this "
          "and the spec differ, the spec wins. Do not edit.*")


def revision(text: str) -> int:
    found = FRONT_REVISION.search(text.split("\n---", 1)[0])
    if found is None:
        raise ValueError("the spec's front matter states no revision")
    return int(found.group(1))


def cut_history(text: str) -> str:
    """The spec up to the first revision-history heading (C2's fourth removal)."""
    lines = text.split("\n")
    for number, line in enumerate(lines):
        if REVISION_HEADING.match(line):
            after = [l for l in lines[number + 1:] if l.startswith("## ") or index_tool.DEF_LINE.match(l)]
            if after:
                raise ValueError(f"a module heading or a requirement follows the history cut at line {number + 1}: "
                                 f"{after[0]!r}")
            return "\n".join(lines[:number])
    return text


def strip_status(text: str) -> str:
    """The front matter without its ``status:`` line (C2's first removal); ``version:`` and the rest stay."""
    head, sep, rest = text.partition("\n---")
    return STATUS_LINE.sub("", head, count=1) + sep + rest if sep else text


def clean(text: str) -> str:
    """C2's removals on any span of spec text: tags, block-quote lines, then runs of blank lines collapsed to one."""
    lines = [TAG.sub("", line).rstrip() for line in text.split("\n") if not QUOTE_LINE.match(line)]
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    while out and not out[-1]:
        out.pop()
    return "\n".join(out) + "\n"


def modules(text: str) -> dict[str, str]:
    """File name -> the cleaned text of its module, in document order. ``M00.md`` holds everything before the first
    ``## M`` heading (the title, the front matter and §0)."""
    body = clean(strip_status(cut_history(text)))
    parts: dict[str, list[str]] = {}
    current = "M00.md"
    for line in body.split("\n"):
        found = MODULE_HEADING.match(line)
        if found:
            current = f"M{int(found.group(1) or found.group(2)):02d}.md"
        parts.setdefault(current, []).append(line)
    return {name: "\n".join(lines).strip("\n") + "\n" for name, lines in parts.items()}


def history_phrases(loaded: dict | None = None) -> list[tuple[str, re.Pattern]]:
    rules = (loaded or json.loads(RULES.read_text(encoding="utf-8")))["history_phrases"]
    return [(p["pattern"], re.compile(p["pattern"], 0 if p["case_sensitive"] else re.IGNORECASE))
            for p in rules["phrases"]]


def sentences(text: str) -> list[str]:
    """A block's sentences, each inside its own paragraph, list item or table row (tests/doc_text.py's rule)."""
    out: list[str] = []
    for paragraph in re.split(r"\n\s*\n|\n(?=\s*(?:\||[-*] |\d+\. ))", text):
        flat = re.sub(r"\s*\n\s*", " ", paragraph).strip()
        out += [s.strip() for s in index_tool.SENTENCE_END.split(flat) if s.strip()]
    return out


def history_in_body(text: str, phrases: list[tuple[str, re.Pattern]] | None = None) -> list[dict]:
    """C4: every ID whose cleaned block holds a sentence matching a history phrase, with those sentences."""
    phrases = history_phrases() if phrases is None else phrases
    found = []
    for block in index_tool.spec_blocks(cut_history(text)):
        cleaned = clean(block["text"])
        hits = [(s, [name for name, pattern in phrases if pattern.search(s)]) for s in sentences(cleaned)]
        hits = [(s, names) for s, names in hits if names]
        if hits:
            found.append({"id": block["id"], "lines": block["lines"], "sentences": hits})
    return found


def render_history(text: str, found: list[dict], rev: int) -> str:
    lines = [
        "# History written into requirement bodies",
        "",
        OPENER.format(revision=rev),
        "",
        f"Every ID whose body holds a sentence matching a history phrase (`tools/spec_index_rules.json`, "
        f"`history_phrases`), with the sentence and the phrase it matched: **{len(found)} IDs**. The view keeps every "
        "one of these sentences (C3). This list was the input to Part E: revision 35 (2026-10-22) moved or rewrote "
        "the narratives, with one row per sentence in `docs/spec_revision_35_dispositions.md` for the owner's "
        "review, so what is listed now is rule text that happens to match a phrase (its row says why it stays). "
        "Line numbers are the spec's.",
        "",
    ]
    for entry in found:
        start, end = entry["lines"]
        where = f"line {start}" if start == end else f"lines {start}–{end}"
        lines.append(f"## {entry['id']} ({where})")
        lines.append("")
        for sentence, names in entry["sentences"]:
            lines.append(f"- {sentence} ⟨{', '.join(names)}⟩")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def render_readme(text: str, files: dict[str, str], rev: int, history: list[dict]) -> str:
    spec_bytes = len(text.encode("utf-8"))
    spec_words = len(text.split())
    view_bytes = sum(len(v.encode("utf-8")) for v in files.values())
    view_words = sum(len(v.split()) for v in files.values())
    tags = len(TAG.findall(text))
    quotes = sum(1 for line in text.split("\n") if QUOTE_LINE.match(line))
    cut_lines = len(cut_history(text).split("\n"))
    history_clause = (f"and the revision-history sections, from the spec's line {cut_lines + 1} to its end"
                      if cut_lines < len(text.split("\n")) else
                      "and the revision-history sections, of which the spec has none since revision 35 moved them to "
                      "`docs/spec_revision_log.md`")
    lines = [
        "# The spec's current rules, by module",
        "",
        OPENER.format(revision=rev),
        "",
        "One file per `## M` module of `docs/bowen_agent_model_spec_v2.md`, with §0 as `M00.md`. The view removes,",
        "mechanically and nothing else: the front matter's `status:` line; every `⟦…⟧` revision tag "
        f"({tags} of them); every block-quote line, the source quotations ({quotes} lines; the `→ file · ID` "
        f"pointers stay); {history_clause}. No sentence of "
        "a requirement body is removed: `history_in_body.md` lists the "
        f"{len(history)} IDs whose bodies hold a sentence matching a history phrase (after revision 35 moved the "
        "narratives to `docs/spec_revision_log.md`, Part E of `docs/plan_model_ergonomics.md`, these are rule text).",
        "",
        "To find an ID's module, grep `docs/spec_index.jsonl` for it: the record's `module` field names the `##` "
        "heading, and `lines` the spec's lines.",
        "",
        "| File | Module | Bytes | Words |",
        "|---|---|---|---|",
    ]
    for name, content in files.items():
        heading = next((line[3:] for line in content.split("\n") if line.startswith("## ")), "the title, the front "
                       "matter and §0")
        lines.append(f"| `{name}` | {heading} | {len(content.encode('utf-8')):,} | {len(content.split()):,} |")
    lines += [
        "",
        f"**Size.** The view's {len(files)} module files hold {view_bytes:,} bytes ({view_words:,} words) against the "
        f"spec's {spec_bytes:,} bytes ({spec_words:,} words): {view_bytes / spec_bytes:.0%} of the spec by bytes, "
        f"{view_words / spec_words:.0%} by words.",
    ]
    return "\n".join(lines) + "\n"


def render_all(text: str) -> dict[str, str]:
    """Every file the view writes, name -> content."""
    rev = revision(text)
    opener = OPENER.format(revision=rev) + "\n\n"
    files = {name: opener + content for name, content in modules(text).items()}
    history = history_in_body(text)
    files["history_in_body.md"] = render_history(text, history, rev)
    files["README.md"] = render_readme(text, {n: c for n, c in files.items() if n != "history_in_body.md"}, rev,
                                       history)
    return files


def main(argv: list[str]) -> int:
    files = render_all(SPEC.read_text(encoding="utf-8"))
    OUT.mkdir(exist_ok=True)
    stale = sorted(p.name for p in OUT.iterdir() if p.name not in files)
    for name, content in files.items():
        (OUT / name).write_text(content, encoding="utf-8")
    print(f"written: {len(files)} files to {OUT.relative_to(REPO)}")
    if stale:
        print(f"not written by this tool, left in place: {', '.join(stale)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
