"""Documents that restate a fact held somewhere else must still agree with it.

Purpose: D2, the repo paths README.md and CLAUDE.md name. Each check is a function over a checkout that returns its
         findings, so the self-tests can run it on a copy that carries the drift it exists to catch. D1, the spec's
         revision as four documents restated it, was retired 2026-11-11 (T-225): those documents point to
         `docs/STATUS.md`, which `tools/status.py` generates.
Spec:    docs/plan_doc_claim_tests.md#D2
Tests:   this file
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from doc_text import claims_data, phrase_pattern, sentences, tracked_files

REPO = Path(__file__).resolve().parents[1]

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
    text = ("Read `docs/archive/DECISIONS — FAMILY EVALUATION.md`.\n\n"
            "```\npython3 -m src.bowen.run --seed 7\n```\n\n"
            "Then `python3 -m src.bowen.gone`.\n")
    scan = path_scan({"README.md": text}, files)
    assert {"docs/archive/DECISIONS — FAMILY EVALUATION.md", "src/bowen/run.py"} <= scan.checked["README.md"]
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
