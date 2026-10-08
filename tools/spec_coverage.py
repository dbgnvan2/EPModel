"""Generate docs/spec_coverage.md — every spec ID with done / partial / not done.

Purpose: the coverage report M14.1 requires as part of a completion report.
Spec:    docs/bowen_agent_model_spec_v2.md#M14.1, #M14.2
Tests:   tests/bowen/test_spec_coverage.py::test_m141_spec_coverage_is_current

    python3 tools/spec_coverage.py tools/spec_coverage_phase_b.json docs/spec_coverage.md

How a status is decided, in this order:

1. **Override.** ``spec_coverage_phase_b.json`` sets the status, evidence and a
   note for an ID — every *partial*, every ID verified by a test not named for
   it, and every *not done* whose reason is more specific than its phase. An
   override's ``exclude_evidence`` lists tests named for the ID that must not be
   shown as evidence, because they verify a superseded form of its text.
2. **A test named for the ID.** The suite passes, so the ID is *done*, and the
   evidence is each such test's file and name (test names embed the ID they
   verify, spec §0.4).
3. **Otherwise not done**, with the phase that builds it: an ``M11.C``
   criterion's own phase cell, or the module's phase from ``M13``.

An ``M11.C`` criterion's test is ensemble-marked and outside the default suite, so rule 2's
"the suite passes" does not hold for it. Its status comes from the committed ensemble record
(``docs/phase_c_ensemble_record.md``): **done** only when every entry for it passes *and* at least
one deletion, named or sign-inverted mutant turned each entry red (``docs/phase_c_mutation_record.md``;
CLAUDE.md: a direction counts as coverage only once proved failing by mutation), **partial** with
each entry's verdict otherwise, and **not done** when the record lists it as not built.
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
ENSEMBLE_RECORD = REPO / "docs" / "phase_c_ensemble_record.md"
MUTATION_RECORD = REPO / "docs" / "phase_c_mutation_record.md"


def mutation_reds() -> set[str]:
    """Criterion entries that at least one deletion, named or sign-inverted mutant turned red."""
    if not MUTATION_RECORD.exists():
        return set()
    text = MUTATION_RECORD.read_text(encoding="utf-8")
    rows = json.loads(re.search(r"```json\n(.*?)\n```", text, re.S).group(1))
    return {r["criterion"] for r in rows if r["kind"] != "representation" and r["result"] == "red"}


def ensemble_verdicts() -> tuple[dict[str, list[tuple[str, str]]], set[str]]:
    """Each criterion's entries and verdicts from the ensemble record, and the ids it lists as not built."""
    if not ENSEMBLE_RECORD.exists():
        return {}, set()
    text = ENSEMBLE_RECORD.read_text(encoding="utf-8")
    rows = json.loads(re.search(r"```json\n(.*?)\n```", text, re.S).group(1))
    verdicts: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        verdicts.setdefault(row["criterion"].split("[")[0], []).append((row["criterion"], row["outcome"]))
    not_built = set(re.findall(r"^- `(M11\.C\.\d+)` —", text, re.M))
    return verdicts, not_built

_consistency_spec = importlib.util.spec_from_file_location("spec_consistency", REPO / "tests" / "test_spec_consistency.py")
_consistency = importlib.util.module_from_spec(_consistency_spec)
_consistency_spec.loader.exec_module(_consistency)

# The phase that builds each module's untested IDs (M13), first match wins.
MODULE_PHASES = [
    (r"^M1\.", "C or D — the phase that builds its mechanism (plan Appendix B)"),
    (r"^M5\.", "C"), (r"^M6\.", "C"), (r"^M7\.", "D"), (r"^M9\.", "D"),
    (r"^M4\.(B\.[23]|C|D)", "C"), (r"^M4\.G\.3", "C"),
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
    verdicts, not_built = ensemble_verdicts()
    reds = mutation_reds()
    unknown = sorted(set(overrides) - set(ids))
    if unknown:
        raise ValueError(f"overrides name IDs the spec does not define: {unknown}")
    rows = []
    for spec_id in ids:
        named = tests_for(spec_id, tests)
        if spec_id in overrides:
            o = overrides[spec_id]
            # ``exclude_evidence`` drops a test named for the ID that does not verify
            # the current text — one that pins a superseded revision's behaviour.
            excluded = set(o.get("exclude_evidence", []))
            evidence = o.get("evidence", []) + [
                t for t in named if t not in o.get("evidence", []) and t not in excluded
            ]
            rows.append({"id": spec_id, "status": o["status"], "evidence": evidence, "note": o.get("note", "")})
        elif spec_id in not_built:
            rows.append({"id": spec_id, "status": "not done", "evidence": named,
                         "note": "not built in Phase C — docs/phase_c_ensemble_record.md says why"})
        elif named and spec_id in verdicts:
            entries = verdicts[spec_id]
            passed = all(outcome == "PASS" for _, outcome in entries)
            proved = all(cid in reds for cid, _ in entries)
            note = "ensemble record: " + "; ".join(f"{cid} {outcome}" for cid, outcome in entries)
            if passed and not proved:
                note += "; passes, but no mutant has turned every entry red (docs/phase_c_mutation_record.md)"
            passed = passed and proved
            rows.append({"id": spec_id, "status": "done" if passed else "partial", "evidence": named, "note": note})
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
        "# Spec coverage — after Phase C",
        "",
        f"Generated by `tools/spec_coverage.py {overrides_name}` (spec `M14.1`). Every ID in",
        "`docs/bowen_agent_model_spec_v2.md`, in document order. **done** carries the test that proves it;",
        "**partial** says what is missing; **not done** names the phase that builds it. An `M11.C`",
        "criterion is **done** only when every entry passes in `docs/phase_c_ensemble_record.md` and a",
        "mutant has turned each entry red in `docs/phase_c_mutation_record.md`; the D9 sweep",
        "(`docs/phase_c_sweep_record.md`) is reported, not gated. Phases D and E build most of the rest.",
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
