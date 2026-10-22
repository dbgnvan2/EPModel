"""No new quotation of the two speech-recognition sources (SP-1).

Purpose: the 1979 tapes and the Kerr–Bowen interviews are ASR transcripts, and the rule is to cite them, never quote
         them. A quoted span of four or more words in a block (a paragraph, list item or table row) that cites one of
         them fails, unless its file is exempt or the span is allowlisted with its written source. Quote marks are
         paired inside each block, so a stray or short quote mark cannot swallow a quotation in the next one. The
         transcripts are not in the repository, so this test reads citations, not transcripts. It misses a quotation
         whose citation sits in another block or that has none, one set as a block quote without quote marks, one
         whose marks do not pair (a curly opening closed by a straight one), and one after a stray quote mark in its
         own block; one longer than 800 characters; one in straight single quotes or guillemets; one beside a citation
         form not in the rules (`KB-04`, `T6.8`, "tape five"); an HTML entity in a Markdown file is not decoded; and only
         tracked `.md` and `.html` files are read. In HTML a block is a blank-line chunk, which can hold several
         elements, and a Markdown block quote is one block however many paragraphs it has, so a citation anywhere in it
         flags every quotation in it (the proposal's §5, form A; §9).
Spec:    docs/PROPOSAL — ASR QUOTATIONS (SP-1).md#SP1.2, #SP1.4, #SP1.5
Tests:   this file
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

from doc_text import blocks, tracked_files

REPO = Path(__file__).resolve().parents[1]
RULES_FILE = REPO / "tests" / "asr_quotations.json"

# A quoted span: straight or curly double quotes, or curly single quotes, up to 800 characters, no blank line inside.
# A curly single quote followed by a letter is an apostrophe (family’s), so it does not close the span.
# Spans of any length are paired, so a short quoted word does not shift the pairing; min_words filters them after.
# In HTML, tags are blanked first (same length, newlines kept), so attribute quotes never pair into a "span", and
# entities are then decoded, so &ldquo;…&rdquo; and &quot;…&quot; are read as quote marks.
_TAG = re.compile(r"<[^>]*>")
_QUOTE = re.compile(r'"((?:[^"\n]|\n(?!\s*\n)){1,800}?)"|“((?:[^”\n]|\n(?!\s*\n)){1,800}?)”'
                    r"|‘((?:[^’\n]|’(?=\w)|\n(?!\s*\n)){1,800}?)’(?!\w)")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    span: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.span[:90]!r}"


def load_rules(path: Path = RULES_FILE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def words(text: str) -> list[str]:
    """The span's words, lower case, with emphasis marks and punctuation removed."""
    text = re.sub(r"[*_`]", "", text.lower().replace("’", "'").replace("‘", "'"))
    return re.sub(r"[^a-z0-9' ]+", " ", text).replace("'", "").split()


def flatten(text: str) -> str:
    """One line, single-spaced, without the blockquote markers that start a quotation's continuation lines."""
    return " ".join(re.sub(r"\n[ \t]*(?:>[ \t]?)+", "\n", text).split())


def findings(path: str, text: str, rules: dict) -> list[Finding]:
    """SP1.2: every quoted span of `min_words` or more words whose block carries an ASR citation."""
    cite = re.compile("|".join(rules["asr_citations"]), re.IGNORECASE)
    if path.endswith(".html"):
        text = html.unescape(_TAG.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text))
    found = []
    for head, tail in blocks(text):
        block = text[head:tail]
        if not cite.search(block):
            continue
        for match in _QUOTE.finditer(block):
            span = match.group(1) or match.group(2) or match.group(3)
            if len(words(span)) >= rules["min_words"]:
                found.append(Finding(path, text.count("\n", 0, head + match.start()) + 1, flatten(span)))
    return found


def is_exempt(path: str, rules: dict) -> bool:
    return any(path == key or (key.endswith("/") and path.startswith(key)) for key in rules["exempt"])


def scanned_files() -> list[str]:
    return [p for p in tracked_files(REPO) if p.endswith((".md", ".html"))]


def classify(path: str, text: str, rules: dict) -> tuple[list[Finding], list[Finding]]:
    """One file's findings: (not allowlisted, allowlisted); none for an exempt file."""
    if is_exempt(path, rules):
        return [], []
    allowed = {(entry["file"], flatten(entry["span"])) for entry in rules["allowlist"]}
    open_, covered = [], []
    for finding in findings(path, text, rules):
        (covered if (path, finding.span) in allowed else open_).append(finding)
    return open_, covered


def repo_findings(rules: dict) -> tuple[list[Finding], list[Finding]]:
    """Every finding outside the exempt files: (not allowlisted, allowlisted)."""
    open_, covered = [], []
    for path in scanned_files():
        o, c = classify(path, (REPO / path).read_text(encoding="utf-8", errors="replace"), rules)
        open_ += o
        covered += c
    return open_, covered


def test_sp1_2_no_quotation_beside_asr_citation():
    open_, _ = repo_findings(load_rules())
    assert not open_, (
        f"{len(open_)} quoted span(s) of four or more words sit in a block that cites a 1979 tape or a Kerr–Bowen "
        "interview. Those sources are ASR transcripts: paraphrase and cite them (SP-1). A written-source quotation "
        "goes on tests/asr_quotations.json's allowlist with its source.\n" + "\n".join(map(str, open_)))


_MUTANT = """# A made-up note

The KB04 interview is cited here, and a new quotation sits beside it: "this sentence was never spoken by anyone".

A written source may be quoted: "a made-up line standing for a book" (`kerr_book/ks12.md` · KS12.3).

Three words: "too short here" (KB04).

"a quotation in a paragraph of its own"

The citation is in the next paragraph instead (KB04).

| a table row with "a quotation in its own row" |
| the next row cites KB04 |
"""

_MUTANT_HTML = ('<p class="note">Cited: KB04. <span class="m">x</span> and "a quoted line in html text".</p>\n\n'
                '<p>Tape 5, and &ldquo;an entity encoded quotation of tape text&rdquo; here.</p>\n')

# A short quoted word, then a stray inch mark in the next item: neither may shift the pairing onto the real quotation.
_MUTANT_PAIRING = """The flag "on" is set; "a real quotation of the tape text" (KB04).

- a board 5" wide
- the next item cites KB04 and quotes "another real quotation of tape text" here
- the 1979 tapes, and ‘a single curly quotation of tape text’ here
- KB04: ‘the family’s reaction to the loss of a member is the main thing’
"""


def test_sp1_2_mutant_new_quotation_fails():
    """The checker the repository test uses catches a new quoted span beside a `KB04` citation, and passes a written
    quotation beside a written citation, a span under four words, and (the known limit) a span whose citation is in
    another paragraph or table row."""
    rules = load_rules()
    assert [f.span for f in findings("mutant.md", _MUTANT, rules)] == ["this sentence was never spoken by anyone"]
    assert [f.span for f in findings("mutant.html", _MUTANT_HTML, rules)] == [
        "a quoted line in html text", "an entity encoded quotation of tape text"]
    assert [f.span for f in findings("mutant.md", _MUTANT_PAIRING, rules)] == [
        "a real quotation of the tape text", "another real quotation of tape text",
        "a single curly quotation of tape text", "the family’s reaction to the loss of a member is the main thing"]


# One sample per citation form, matched by that form alone; a form added to the rules needs a sample here.
_CITATION_SAMPLES = {
    r"\bKB\d{2}\b": "KB04", r"\bK\d{2}\.\d+": "K07.4", r"\bTape\s*\d": "Tape 5", "Kerr[–—-]Bowen": "the Kerr–Bowen series",
    "_KB_PASS2": "_KB_PASS2.md", "_LECTURES_1979": "_LECTURES_1979.md", "_KERR_INTERVIEWS": "_KERR_INTERVIEWS.md",
    "1979 lecture": "the 1979 lecture series", r"\bTapes\s*\d": "Tapes 5 and 6", "1979 tapes?": "the 1979 tapes",
    "video lectures?": "the video lectures", r"\bKB\s\d{2}\b": "KB 04", "Kerr interview": "a Kerr interview",
}


def test_sp1_2_every_citation_form_is_read():
    """Each citation form in the rules is needed: its sample, beside a quotation, is a finding, and with that form
    removed from the rules it is not."""
    rules = load_rules()
    assert sorted(rules["asr_citations"]) == sorted(_CITATION_SAMPLES)
    for form, sample in _CITATION_SAMPLES.items():
        text = f'Cited as {sample}: "a quotation of the tape text here".\n'
        assert [f.span for f in findings("mutant.md", text, rules)] == ["a quotation of the tape text here"], form
        without = rules | {"asr_citations": [c for c in rules["asr_citations"] if c != form]}
        assert findings("mutant.md", text, without) == [], form


def test_sp1_2_mutant_unlisted_quotation_in_an_allowlisted_file_fails():
    """The routing the repository test uses: in a file with allowlist entries, a listed span is covered and any other
    span is open; an exempt file reports nothing."""
    rules = load_rules()
    entry = next(e for e in rules["allowlist"] if e["file"] == "docs/bowen_agent_model_spec_v2.md")
    text = f'The KB04 interview, and "{entry["span"]}" and "a new quotation of the interview text".\n'
    open_, covered = classify(entry["file"], text, rules)
    assert [f.span for f in open_] == ["a new quotation of the interview text"]
    assert [f.span for f in covered] == [flatten(entry["span"])]
    assert classify("docs/theory/kb/kb04.md", text, rules) == ([], [])


def test_sp1_4_rules_entries_are_live():
    """SP1.4: every exempt path is tracked, every entry gives its reason, and every allowlisted span is still a
    finding in its file (a span that was paraphrased away, or whose file became exempt, is a stale entry)."""
    rules = load_rules()
    tracked = set(tracked_files(REPO))
    problems = []
    for key, reason in rules["exempt"].items():
        if not (key in tracked or (key.endswith("/") and any(p.startswith(key) for p in tracked))):
            problems.append(f"exempt path not tracked: {key}")
        if not reason.strip():
            problems.append(f"exempt path without a reason: {key}")
    _, covered = repo_findings(rules)
    live = {(f.path, f.span) for f in covered}
    for entry in rules["allowlist"]:
        if not entry.get("source", "").strip():
            problems.append(f"allowlist entry without a source: {entry['file']}: {entry['span'][:60]!r}")
        if (entry["file"], flatten(entry["span"])) not in live:
            problems.append(f"stale allowlist entry (no longer a finding): {entry['file']}: {entry['span'][:60]!r}")
    assert not problems, "\n".join(problems)


def frozen_records(rules: dict) -> list[str]:
    """The exempt records kept unedited under the frozen-record note (their reason says so)."""
    return [key for key, reason in rules["exempt"].items() if "frozen-record note" in reason]


def missing_note(text: str, rules: dict) -> bool:
    """True when the frozen-record note is not in the text, read without blockquote marks, emphasis or line breaks."""
    plain = flatten(re.sub(r"^>\s?", "", text, flags=re.M).replace("*", ""))
    return flatten(rules["frozen_record_note"]) not in plain


def test_sp1_5_frozen_records_carry_note():
    """SP1.5: each record exempt as frozen carries the note that its ASR quotations are not the speakers' words; a
    record without the note fails (the second assertion's copy)."""
    rules = load_rules()
    records = frozen_records(rules)
    assert sorted(records) == ["docs/DECISIONS FOR APPROVAL.md", "docs/proposal_rev11_emergence.md"]
    missing = [path for path in records if missing_note((REPO / path).read_text(encoding="utf-8"), rules)]
    assert not missing, f"frozen record(s) without the SP-1 note: {missing}"
    assert missing_note("# A record\n\n> *Quotations in this record are ASR text.*\n", rules)
