"""Documents that restate a fact held somewhere else must still agree with it.

Purpose: D1, the spec's revision as README.md, CLAUDE.md, `_STATUS.md` and the checklist restate it; D2, the repo paths
         README.md and CLAUDE.md name. Each check is a function over a checkout that returns its findings, so the
         self-tests can run it on a copy that carries the drift it exists to catch.
Spec:    docs/plan_doc_claim_tests.md#D1, #D2
Tests:   this file
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from doc_text import claims_data, phrase_pattern, sentences, tracked_files

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------------------------------- D1

SPEC = "docs/bowen_agent_model_spec_v2.md"
_FRONT_REVISION = re.compile(r"^version:\s*2\.0,\s*revision\s+(\d+)\s*$", re.M)
# A range ("revisions 24–30", "revisions 24 to 27") counts as its upper end.
_REVISION = re.compile(r"\brevisions?\s+(\d+)(?:\s*(?:–|-|to)\s*(\d+))?", re.I)

# Each place that restates the current revision: its file, the region that holds the restatement, and the exact form
# of the claim where the place has one (the checklist's "the spec (v2.0, revision N"). Without an exact form, the
# largest revision the region names is its claim.
PLACES = {
    "the checklist's spec entry": (
        "docs/SPEC_REVIEW_CHECKLIST.md",
        re.compile(r"^- \[[ x]\] \[bowen_agent_model_spec_v2\.md\]\(bowen_agent_model_spec_v2\.md\).*?(?=\n- |\n\n|\Z)",
                   re.M | re.S),
        re.compile(r"the spec \(v2\.0, revision (\d+)")),
    "CLAUDE.md's spec row": ("CLAUDE.md", re.compile(r"^\| `docs/bowen_agent_model_spec_v2\.md`.*$", re.M), None),
    "_STATUS.md's ▶ STATE line": ("docs/theory/_STATUS.md", re.compile(r"^> ## ▶ STATE.*$", re.M), None),
    "README.md's Status section": ("README.md", re.compile(r"^## Status\n.*?(?=^## |\Z)", re.M | re.S), None),
}


def spec_revision(spec_text: str) -> int | None:
    """D1.1: the revision in the spec's front matter, or None when the front matter does not state one."""
    match = _FRONT_REVISION.search(spec_text.split("\n---", 1)[0])
    return int(match.group(1)) if match else None


def revision_findings(repo: Path) -> dict[str, str]:
    """D1.2–D1.7: each place whose largest named revision is not the spec's, with what it says.

    A place names past revisions too ("revision 28 built 2026-10-13"), and none can be larger than the current one, so
    the largest it names is its claim about the current revision.
    """
    current = spec_revision((repo / SPEC).read_text(encoding="utf-8"))
    if current is None:
        return {SPEC: "its front matter names no revision, so there is nothing to compare the restatements with"}
    findings = {}
    for place, (path, region, exact) in PLACES.items():
        match = region.search((repo / path).read_text(encoding="utf-8"))
        if match is None:
            findings[place] = f"not found in {path}; the restatement is gone"
            continue
        if exact:
            named = [int(n) for n in exact.findall(match.group(0))]
        else:
            named = [int(high or low) for low, high in _REVISION.findall(match.group(0))]
        if not named:
            findings[place] = f"names no revision; the restatement is gone (the spec is at {current})"
        elif max(named) != current:
            findings[place] = f"names revision {max(named)}; the spec is at {current}"
    return findings


def _copy(repo: Path, tmp_path: Path, paths) -> Path:
    for rel in paths:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / rel, tmp_path / rel)
    return tmp_path


def _d1_copy(tmp_path: Path) -> Path:
    return _copy(REPO, tmp_path, [SPEC] + [path for path, _, _ in PLACES.values()])


def _edit(path: Path, old: str, new: str, count: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"fixture edit not applicable: {old!r} is not in {path.name}"
    path.write_text(text.replace(old, new, count), encoding="utf-8")


def test_d1_spec_revision_restatements_match_the_front_matter():
    """D1.1–D1.5: the four places agree with the spec's front matter."""
    assert spec_revision((REPO / SPEC).read_text(encoding="utf-8")) is not None, "the spec's front matter names no revision"
    assert revision_findings(REPO) == {}


def test_d1_a_moved_spec_revision_is_reported_in_every_place(tmp_path):
    """The drift the test exists for: the spec moves to the next revision and none of the copies do."""
    copy = _d1_copy(tmp_path)
    current = spec_revision((REPO / SPEC).read_text(encoding="utf-8"))
    _edit(copy / SPEC, f"version: 2.0, revision {current}\n", f"version: 2.0, revision {current + 1}\n")
    findings = revision_findings(copy)
    assert sorted(findings) == sorted(PLACES)
    assert all(f"the spec is at {current + 1}" in message for message in findings.values())


def test_d1_the_checklist_drift_of_2026_10_19_is_reported(tmp_path):
    """At `7aa9fc5` the checklist's spec entry said revision 27 while the spec was at 33."""
    copy = _d1_copy(tmp_path)
    entry = PLACES["the checklist's spec entry"]
    text = (copy / entry[0]).read_text(encoding="utf-8")
    old = entry[1].search(text).group(0)
    new = re.sub(r"the spec \(v2\.0, revision \d+; version 1 is\s+revision 32\)", "the spec (v2.0, revision 27)", old)
    assert new != old, "the checklist's spec entry no longer has the wording this fixture edits"
    (copy / entry[0]).write_text(text.replace(old, new), encoding="utf-8")
    assert list(revision_findings(copy)) == ["the checklist's spec entry"]
    assert "names revision 27" in revision_findings(copy)["the checklist's spec entry"]


def test_d1_the_checklist_is_read_by_its_exact_claim(tmp_path):
    """D1.2 (adversarial): the spec moves on and the checklist's entry mentions the new revision in its description but
    still claims the old one. The largest number in the entry is right; its claim is not, so it is reported."""
    copy = _d1_copy(tmp_path)
    current = spec_revision((REPO / SPEC).read_text(encoding="utf-8"))
    _edit(copy / SPEC, f"version: 2.0, revision {current}\n", f"version: 2.0, revision {current + 1}\n")
    _edit(copy / PLACES["the checklist's spec entry"][0], " out of scope (M12),",
          f" out of scope (M12, which revision {current + 1} widens),")
    assert f"names revision {current}" in revision_findings(copy).get("the checklist's spec entry", "")


def test_d1_a_place_that_stops_naming_a_revision_fails(tmp_path):
    """D1.6: `_STATUS.md`'s state line as it was at `7aa9fc5`, which names no revision."""
    copy = _d1_copy(tmp_path)
    status = copy / "docs/theory/_STATUS.md"
    text = status.read_text(encoding="utf-8")
    line = PLACES["_STATUS.md's ▶ STATE line"][1].search(text).group(0)
    old_line = ("> ## ▶ STATE — the corpus is closed and the spec is current. **Phase C is closed with its gate failing "
                "(owner decision, 2026-10-09).**")
    status.write_text(text.replace(line, old_line), encoding="utf-8")
    findings = revision_findings(copy)
    assert list(findings) == ["_STATUS.md's ▶ STATE line"]
    assert "names no revision" in findings["_STATUS.md's ▶ STATE line"]


def test_d1_history_does_not_trip_and_a_future_revision_does(tmp_path):
    """D1.7 (adversarial): a past revision in the row is history; a revision past the spec's is a wrong claim."""
    copy = _d1_copy(tmp_path)
    current = spec_revision((REPO / SPEC).read_text(encoding="utf-8"))
    row = "| `docs/bowen_agent_model_spec_v2.md` |"
    _edit(copy / "CLAUDE.md", row, row + " revision 28 built 2026-10-13;")
    assert revision_findings(copy) == {}
    _edit(copy / "CLAUDE.md", row, row + f" revision {current + 1} planned;")
    findings = revision_findings(copy)
    assert list(findings) == ["CLAUDE.md's spec row"]
    assert f"names revision {current + 1}" in findings["CLAUDE.md's spec row"]


def test_d1_an_unreadable_front_matter_fails():
    """D1.1: the front matter is the source; without it there is nothing to compare."""
    assert spec_revision("---\ntags: [spec]\nversion: 2.0\n---\nrevision 33 in the body\n") is None


# ---------------------------------------------------------------------------------------------------- D2

PATH_DOCS = ("README.md", "CLAUDE.md")
_ROOTS = ("src/", "tools/", "tests/", "docs/", "config/")
_SKIP = set("*<>{}$")
_SPAN = re.compile(r"`([^`\n]+)`")
_FENCE = re.compile(r"^```[^\n]*\n(.*?)^```", re.M | re.S)
_LINK = re.compile(r"\]\(<([^>]+)>\)|\]\(([^)\s]+)\)")


@dataclass
class PathScan:
    """D2: what the scan read in each document — paths found, paths missing and marked retired, paths missing."""
    checked: dict[str, set[str]] = field(default_factory=dict)
    retired: dict[str, set[str]] = field(default_factory=dict)
    missing: dict[str, list[tuple[str, str]]] = field(default_factory=dict)


def path_scan(texts: dict[str, str], files: list[str]) -> PathScan:
    """D2.1–D2.3: every repo path the documents name, checked against the tracked files.

    Prose is read sentence by sentence (backtick spans and link targets); a fenced block line by line. A missing path
    passes only when `tests/doc_claims.json` lists it as retired for that document and its own sentence or line carries
    a retirement marker.
    """
    tracked = set(files)
    dirs = {"/".join(p.split("/")[:i]) for p in files for i in range(1, p.count("/") + 1)}
    names = {p.rsplit("/", 1)[-1] for p in files}
    # The five project folders, and every other top-level folder git tracks (`.github/`, `papers/`).
    roots = _ROOTS + tuple(sorted({p.split("/")[0] + "/" for p in files if "/" in p} - set(_ROOTS)))
    claims = claims_data()
    marker = phrase_pattern(claims["retired_markers"]["phrases"])
    listed = {doc: set(paths) for doc, paths in claims["retired_paths"].items() if not doc.startswith("_")}

    def exists(path: str) -> bool:
        if "/" not in path:
            return path in names  # a bare `name.py`: any tracked file of that name (D2.2)
        return path.rstrip("/") in tracked or path.rstrip("/") in dirs

    def candidates(span: str) -> list[str]:
        span = span.strip()
        if span.startswith(roots) and exists(span):
            return [span]
        found, tokens = [], span.split()
        for i, token in enumerate(tokens):
            token = re.sub(r":\d+$", "", token.strip(",;:()[]'\"").split("::")[0])
            if i and tokens[i - 1] == "-m" and re.fullmatch(r"(?:src|tools|tests)(?:\.\w+)+", token):
                module = token.replace(".", "/")
                found.append(module if module in dirs and module + ".py" not in tracked else module + ".py")
            elif token and not _SKIP & set(token) and (token.startswith(roots) or re.fullmatch(r"[\w./-]+\.py", token)):
                found.append(token)
        return found

    scan = PathScan()
    for doc, text in texts.items():
        units = [(line, False) for block in _FENCE.findall(text) for line in block.splitlines()]
        units += [(s, True) for s in sentences(_FENCE.sub("\n\n", text))]
        for unit, prose in units:
            spans = (_SPAN.findall(unit) + [a or b for a, b in _LINK.findall(unit)]) if prose else [unit]
            for span in spans:
                if prose and re.match(r"[a-z]+:|#", span):
                    continue
                for path in candidates(span.split("#")[0] if prose else span):
                    scan.checked.setdefault(doc, set()).add(path)
                    if exists(path):
                        continue
                    # Retired only when listed by name for this document and its own sentence or line says so: a
                    # marker alone can be about something else ("frozen at the tag `version-1`").
                    if path in listed.get(doc, set()) and marker.search(unit):
                        scan.retired.setdefault(doc, set()).add(path)
                    else:
                        scan.missing.setdefault(doc, []).append((path, unit[:160]))
    return scan


def _real_scan() -> PathScan:
    return path_scan({doc: (REPO / doc).read_text(encoding="utf-8") for doc in PATH_DOCS}, tracked_files())


def test_d2_repo_paths_in_readme_and_claude_md_exist():
    """D2.1–D2.3: a path README.md or CLAUDE.md names is tracked, or its own sentence says it is gone."""
    missing = _real_scan().missing
    assert missing == {}, f"{missing}; a new file is read from git, so stage it (git add) first"


def test_d2_the_old_run_command_is_reported():
    """README.md at `423bbed`, after the grid engine was removed at stage 1 step 2, still named its entry point twice."""
    readme = ("## Layout\n\n```\nsrc/main.py          orchestration, UI, I/O\n```\n\n"
              "## Running\n\n```bash\npython3 src/main.py\n```\n")
    files = [p for p in tracked_files() if p != "src/main.py"]
    missing = path_scan({"README.md": readme}, files).missing
    assert [path for path, _ in missing["README.md"]] == ["src/main.py", "src/main.py"]


def test_d2_a_retired_path_passes_only_in_its_own_sentence():
    """D2.3 (adversarial): the retirement marker counts only in the sentence that names the path."""
    files = [p for p in tracked_files() if p != "src/main.py"]
    assert path_scan({"README.md": "`src/main.py` was retired.\n"}, files).missing == {}
    missing = path_scan({"README.md": "The engine was retired. Run `src/main.py`.\n"}, files).missing
    assert [path for path, _ in missing["README.md"]] == ["src/main.py"]
    # A neighbouring table row or list item is its own unit, so its marker does not count either.
    table = "| `docs/model_config.md` | the old engine's settings |\n| `src/engine.py` | retired |\n"
    missing = path_scan({"CLAUDE.md": table}, files).missing
    assert [path for path, _ in missing["CLAUDE.md"]] == ["docs/model_config.md"]


def test_d2_paths_with_spaces_and_module_commands_resolve():
    """D2.1: a span that is one path with spaces in it, and a `-m` module checked as its file."""
    files = tracked_files()
    text = ("Read `docs/DECISIONS — FAMILY EVALUATION.md`.\n\n"
            "```\npython3 -m src.bowen.run --seed 7\n```\n\n"
            "Then `python3 -m src.bowen.gone`.\n")
    scan = path_scan({"README.md": text}, files)
    assert {"docs/DECISIONS — FAMILY EVALUATION.md", "src/bowen/run.py"} <= scan.checked["README.md"]
    assert [path for path, _ in scan.missing["README.md"]] == ["src/bowen/gone.py"]


def test_d2_patterns_placeholders_and_test_ids_are_read_as_intended():
    """D2.1: a glob or placeholder is not a path; a `::test` suffix is dropped before the check."""
    files = tracked_files()
    text = ("`tests/bowen/test_*.py`, `docs/<name>.md` and `$HOME/src/x.py` are patterns.\n\n"
            "`tests/test_repo_layout.py::test_s1_6_the_grid_engine_is_retired` is a test id.\n")
    scan = path_scan({"README.md": text}, files)
    assert scan.missing == {}
    assert scan.checked["README.md"] == {"tests/test_repo_layout.py"}


def test_d2_a_marker_about_something_else_does_not_excuse_a_path():
    """D2.3 (adversarial): a missing path that is not listed as retired fails even beside a retirement marker."""
    files = tracked_files()
    missing = path_scan({"CLAUDE.md": "`docs/a.md` is frozen at the tag `version-1`.\n"}, files).missing
    assert [path for path, _ in missing["CLAUDE.md"]] == ["docs/a.md"]
    # A listed path that its sentence no longer marks as gone fails too.
    files = [p for p in files if p != "src/main.py"]
    missing = path_scan({"README.md": "Run `src/main.py` to start.\n"}, files).missing
    assert [path for path, _ in missing["README.md"]] == ["src/main.py"]


def test_d2_the_retired_list_is_current():
    """D2.3: each path listed as retired is untracked and still named, marked, in its document."""
    listed = {doc: set(paths) for doc, paths in claims_data()["retired_paths"].items()
              if not doc.startswith("_")}
    scan, files = _real_scan(), set(tracked_files())
    assert {doc: sorted(p for p in paths if p in files) for doc, paths in listed.items()} == {d: [] for d in listed}
    assert {doc: sorted(paths - scan.retired.get(doc, set())) for doc, paths in listed.items()} == {d: [] for d in listed}


def test_d2_the_scan_is_not_vacuous():
    """D2.4: the scan reads both documents, including a path under a top-level folder read from git (`.github/`), and
    it sees README.md's retired entry point as missing and marked."""
    scan = _real_scan()
    assert "tests/test_repo_layout.py" in scan.checked.get("CLAUDE.md", set())
    assert "tools/record_cache.py" in scan.checked.get("README.md", set())
    assert ".github/workflows/tests.yml" in scan.checked.get("README.md", set())
    assert "src/main.py" in scan.retired.get("README.md", set())
