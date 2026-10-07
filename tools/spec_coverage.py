"""Generate docs/spec_coverage.md — every spec ID with done / partial / not done.

Purpose: the coverage report M14.1 requires as part of a completion report.
Spec:    docs/bowen_agent_model_spec_v2.md#M14.1, #M14.2
Tests:   tests/bowen/test_spec_coverage.py::test_m141_spec_coverage_is_current

    python3 tools/spec_coverage.py tools/spec_coverage_phase_b.json docs/spec_coverage.md

How a status is decided, in this order:

1. **Override.** ``spec_coverage_phase_b.json`` sets the status, evidence and a
   note for an ID — every *partial*, every ID verified by a test not named for
   it, and every *not done* whose reason is more specific than its phase.
2. **A test named for the ID.** The suite passes, so the ID is *done*, and the
   evidence is each such test's file and name (test names embed the ID they
   verify, spec §0.4).
3. **Otherwise not done**, with the phase that builds it: an ``M11.C``
   criterion's own phase cell, or the module's phase from ``M13``.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "docs" / "bowen_agent_model_spec_v2.md"

_consistency_spec = importlib.util.spec_from_file_location("spec_consistency", REPO / "tests" / "test_spec_consistency.py")
_consistency = importlib.util.module_from_spec(_consistency_spec)
_consistency_spec.loader.exec_module(_consistency)

# The phase that builds each module's untested IDs (M13), first match wins.
MODULE_PHASES = [
    (r"^M1\.", "C or D — the phase that builds its mechanism (plan Appendix B)"),
    (r"^M5\.", "C"), (r"^M6\.", "C"), (r"^M7\.", "D"), (r"^M9\.", "D"),
    (r"^M4\.(B\.2|C|D)", "C"), (r"^M4\.G\.3", "C"),
    (r"^M8\.[678]", "C"), (r"^M2\.", "D — the twelve-person reference family"),
    (r"^M10\.", "C or D"), (r"^M11\.G", "D"), (r"^M11\.E", "C, D or E (M11.E)"),
    (r"^M11\.D", "C"), (r"^M11\.", "C — acceptance-test rules"),
    (r"^M12\.", "every phase — prohibitions, checked by review"),
    (r"^M15\.", "E"), (r"^M16\.D", "C"), (r"^M16\.E", "F"), (r"^M16\.", "C"),
    (r"^M17\.A\.[134]$", "C (M13.4)"), (r"^M17\.", "E"), (r"^M14\.", "every phase"),
    (r"^M13\.", "C"), (r"^M3\.", "C"), (r"^M0\.", "C"),
]


def spec_ids(text: str) -> list[str]:
    seen, ordered = set(), []
    for match in re.finditer(r"^(?:\| )?(?:[-*]\s+)?\*\*(M\d+(?:\.[A-Za-z0-9]+)*)\*\*", text, re.M):
        if match.group(1) not in seen:
            seen.add(match.group(1))
            ordered.append(match.group(1))
    assert seen == set(_consistency.defined_ids(text)), "ID extraction disagrees with the consistency guard"
    return ordered


def named_tests() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted((REPO / "tests").rglob("test_*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                found.setdefault(node.name, []).append(path.relative_to(REPO).as_posix())
    return found


def tests_for(spec_id: str, tests: dict[str, list[str]]) -> list[str]:
    """Every file holding a test named for the ID — a name in two files is two pieces of evidence."""
    token = spec_id.lower().replace(".", "")
    return sorted(
        f"{path}::{name}" for name, files in tests.items() if name.startswith(f"test_{token}_") for path in files
    )


def unmapped_tests(ids: list[str], tests: dict[str, list[str]]) -> list[str]:
    """Tests whose name embeds no spec ID; they count for nothing unless an override cites them."""
    tokens = {i.lower().replace(".", "") for i in ids}
    return sorted(
        f"{path}::{name}"
        for name, files in tests.items()
        if not any(name.startswith(f"test_{t}_") for t in tokens)
        for path in files
    )


def criterion_phases(text: str) -> dict[str, str]:
    return {cid: (_consistency._phase_of(cells[2]) or "?") for cid, cells in _consistency._criterion_rows(text)
            if len(cells) >= 3}


def module_phase(spec_id: str) -> str:
    for pattern, phase in MODULE_PHASES:
        if re.match(pattern, spec_id):
            return phase
    return "unassigned"


def coverage(overrides: dict) -> list[dict]:
    text = SPEC.read_text(encoding="utf-8")
    ids, tests, criteria = spec_ids(text), named_tests(), criterion_phases(text)
    unknown = sorted(set(overrides) - set(ids))
    if unknown:
        raise ValueError(f"overrides name IDs the spec does not define: {unknown}")
    rows = []
    for spec_id in ids:
        named = tests_for(spec_id, tests)
        if spec_id in overrides:
            o = overrides[spec_id]
            evidence = o.get("evidence", []) + [t for t in named if t not in o.get("evidence", [])]
            rows.append({"id": spec_id, "status": o["status"], "evidence": evidence, "note": o.get("note", "")})
        elif named:
            rows.append({"id": spec_id, "status": "done", "evidence": named, "note": ""})
        else:
            phase = criteria.get(spec_id) or module_phase(spec_id)
            rows.append({"id": spec_id, "status": "not done", "evidence": [], "note": f"Phase {phase}"})
    for row in rows:
        if row["status"] == "done" and not row["evidence"]:
            raise ValueError(f"{row['id']} is marked done with no evidence")
        if row["status"] not in {"done", "partial", "not done"}:
            raise ValueError(f"{row['id']}: unknown status {row['status']!r}")
    return rows


def render(rows: list[dict], overrides_name: str, unmapped: list[str] | None = None) -> str:
    counts = {s: sum(r["status"] == s for r in rows) for s in ("done", "partial", "not done")}
    lines = [
        "# Spec coverage — after Phase B",
        "",
        f"Generated by `tools/spec_coverage.py {overrides_name}` (spec `M14.1`). Every ID in",
        "`docs/bowen_agent_model_spec_v2.md`, in document order. **done** carries the test that proves it;",
        "**partial** says what is missing; **not done** names the phase that builds it. Phase B built",
        "the objects, the clock, the standing load, the base appraisal, the event record, the M8",
        "predicate, the scripted source and the run log, so most of the spec is, correctly, not done.",
        "",
        f"**{len(rows)} IDs: {counts['done']} done, {counts['partial']} partial, {counts['not done']} not done.**",
        "",
        "| ID | Status | Evidence | Note |",
        "|---|---|---|---|",
    ]
    for r in rows:
        evidence = "<br>".join(f"`{e}`" for e in r["evidence"]) or "—"
        lines.append(f"| {r['id']} | {r['status']} | {evidence} | {r['note'] or '—'} |")
    if unmapped:
        cited = {e for r in rows for e in r["evidence"]}
        lines += [
            "",
            f"## Tests named for no spec ID ({len(unmapped)})",
            "",
            "Their names embed no requirement ID, so they count only where an override cites them "
            "(marked *cited*). The rest check spec documents, tooling or helpers.",
            "",
        ]
        lines += [f"- `{t}`{' — *cited*' if t in cited else ''}" for t in unmapped]
    return "\n".join(lines) + "\n"


def report(overrides: dict, overrides_name: str) -> str:
    rows = coverage(overrides)
    return render(rows, overrides_name, unmapped_tests(spec_ids(SPEC.read_text(encoding="utf-8")), named_tests()))


def main(argv: list[str]) -> int:
    overrides_path, out = Path(argv[1]), Path(argv[2])
    text = report(json.loads(overrides_path.read_text(encoding="utf-8")), overrides_path.name)
    out.write_text(text, encoding="utf-8")
    print(f"coverage written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
