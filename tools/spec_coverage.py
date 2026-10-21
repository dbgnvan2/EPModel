"""Generate docs/spec_coverage.md — every spec ID with done / partial / not done.

Purpose: the coverage report M14.1 requires as part of a completion report.
Spec:    docs/bowen_agent_model_spec_v2.md#M14.1, #M14.2
Tests:   tests/bowen/test_spec_coverage.py (test_m141_spec_coverage_is_current, and one test for each of Part A's A1-A5)

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
3. **Otherwise not done**, with a note that says no test is named for it and which
   phase built, or builds, its module (an ``M11.C`` criterion's own phase cell, or the
   module's phase from ``M13``), so that an untested ID is not read as unbuilt (plan A4).

An ``M11.C`` criterion's test is ensemble-marked and outside the default suite, so rule 2's
"the suite passes" does not hold for it. Its status comes from the committed records
(``docs/plan_spec_coverage_phase_d.md`` Part A), in this order:

* **A verdict in an ensemble record** (Phase C, Phase D part I, Phase D part II): **done** only when
  every entry for it passes, at least one deletion, named or sign-inverted mutant in a mutation
  record turned each entry red (CLAUDE.md: a direction counts as coverage only once proved failing
  by mutation), and its spec row does not say its own mutation clause is not met; **partial**,
  with each entry's verdict, otherwise. A verdict wins over a not-built list (``M11.C.14``).
* **Run as a test, not an ensemble** (``tools/phase_d_record.py``'s ``NOT_RUN_2``): **done**, with
  that test as evidence.
* **Not built** (``src/bowen/ensemble/criteria.py``'s and ``tools/phase_d_record.py``'s
  ``NOT_BUILT``): **not done**, with the reason.

Where the owner recorded a version 1 disposition for a criterion (``docs/version_1_dispositions.md``,
read by the frozen ``tools/version_1_status.py``'s parser), the note quotes it; the status stays what
the records show. The page's header names each record read and the hash it states (plan A3).
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

from tools.record_json import record_json  # noqa: E402

SPEC = REPO / "docs" / "bowen_agent_model_spec_v2.md"
ENSEMBLE_RECORDS = (
    REPO / "docs" / "phase_c_ensemble_record.md",
    REPO / "docs" / "phase_d_ensemble_record.md",
    REPO / "docs" / "phase_d2_ensemble_record.md",
)
MUTATION_RECORDS = (
    REPO / "docs" / "phase_c_mutation_record.md",
    REPO / "docs" / "phase_d_mutation_record.md",
)
RED = ("red", "reversed")
# The phases version 1 built (the tag `version-1`, docs/version_1_definition.md). An ID with no test named for it in a
# module of one of these phases is untested by name, not unbuilt (plan A4).
BUILT_PHASES = frozenset("ABCD")
# A criterion row's mutation cell that says this records that its own mutation clause is not met (M11.C.14, M11.C.16,
# M11.C.46 at revision 33); such a criterion is not done however its mutants fared. Spec rows are amended by appending,
# so a later "met" beside an old "not met" leaves the criterion partial until the row is reworded (the safe side).
UNMET_CLAUSE = re.compile(r"\b(?:not (?:yet )?met|unmet|isn't met)\b", re.IGNORECASE)
UNASSIGNED = "no test named for it; no phase in MODULE_PHASES matches its module"
# A criterion's `M11.5` class, from the spec's class table.
CLASS_ROW = re.compile(r"^\| (M11\.C\.\d+) \| (Premise|Composite|Check|Mixed)\b", re.M)
# Classes whose pass version 1 took as final without a recorded mutant (docs/version_1_definition.md V3).
NO_RECORDED_MUTANT_NEEDED = ("premise", "check")


def record_hash(path: Path) -> str:
    """The code or engine hash a record states it was generated under (plan A3); a record that states none raises."""
    found = re.search(r"^(?:code|engine)_hash: ([0-9a-f]{12,})$", path.read_text(encoding="utf-8"), re.M)
    if found is None:
        raise ValueError(f"{path.name} states no code_hash or engine_hash")
    return found.group(1)


def mutation_reds() -> set[str]:
    """Criterion entries that at least one deletion, named or sign-inverted mutant turned red (or reversed), in any
    mutation record."""
    return {r["criterion"] for path in MUTATION_RECORDS for r in record_json(path)
            if r["kind"] != "representation" and r["result"] in RED}


def ensemble_verdicts() -> dict[str, list[tuple[str, str, str]]]:
    """Spec ID -> (entry key, verdict, record file) for each entry in the ensemble records. A key is the ID, or the ID
    with its variant in brackets; a key with a verdict in two records is refused."""
    verdicts: dict[str, list[tuple[str, str, str]]] = {}
    seen: dict[str, str] = {}
    for path in ENSEMBLE_RECORDS:
        for row in record_json(path):
            key = row["criterion"]
            if key in seen:
                raise ValueError(f"{key} has a verdict in both {seen[key]} and {path.name}")
            seen[key] = path.name
            verdicts.setdefault(key.split("[")[0], []).append((key, row["outcome"], path.name))
    return verdicts


def not_built() -> dict[str, str]:
    """Criterion ID -> why it has no arm, from Phase C's and Phase D's not-built lists (the source of truth, not the
    records' prose)."""
    from src.bowen.ensemble.criteria import NOT_BUILT as PHASE_C
    from tools.phase_d_record import NOT_BUILT as PHASE_D

    out = {cid: f"{why} (`src/bowen/ensemble/criteria.py`)" for cid, why in PHASE_C.items()}
    out.update({cid: f"{why} (`tools/phase_d_record.py`)" for cid, why in PHASE_D.items()})
    return out


def run_as_test() -> dict[str, tuple[str, str]]:
    """Criterion ID -> (the test that proves it, why), for a criterion run as a test rather than an ensemble
    (`M11.C.37`). The test's existence is checked by `test_m141_every_done_names_an_existing_test`."""
    from tools.phase_d_record import NOT_RUN_2

    out = {}
    for cid, why in NOT_RUN_2.items():
        found = re.search(r"`(tests/[\w/]+\.py::test_\w+)`", why)
        if found is None:
            raise ValueError(f"{cid}: tools/phase_d_record.py's NOT_RUN_2 names no test")
        out[cid] = (found.group(1), why)
    return out


def unmet_clauses(text: str) -> set[str]:
    """Criteria whose spec row's mutation cell says its mutation clause is not met."""
    return {cid for cid, cells in _consistency._criterion_rows(text) if len(cells) >= 4 and UNMET_CLAUSE.search(cells[3])}


def disposition_notes(spec_id: str, dispositions: dict[str, dict]) -> list[str]:
    """The owner's version 1 dispositions for the ID or any of its variants, quoted (plan A2)."""
    return [f"version 1 disposition for `{key}` ({d['decided']}), {d['kind']}: {d['reason']}"
            for key, d in dispositions.items() if key.split("[")[0] == spec_id]

_consistency_spec = importlib.util.spec_from_file_location("spec_consistency", REPO / "tests" / "test_spec_consistency.py")
_consistency = importlib.util.module_from_spec(_consistency_spec)
_consistency_spec.loader.exec_module(_consistency)

# The phase that builds each module's untested IDs (M13), first match wins: (pattern, phase, detail).
MODULE_PHASES = [
    (r"^M1\.", "C or D", "the phase that builds its mechanism, plan Appendix B"),
    (r"^M5\.", "C", ""), (r"^M6\.", "C", ""), (r"^M7\.", "D", ""), (r"^M9\.", "D", ""),
    (r"^M4\.(B\.[23]|C|D)", "C", ""), (r"^M4\.G\.3", "C", ""),
    (r"^M8\.[678]", "C", ""), (r"^M2\.", "D", "the twelve-person reference family"),
    (r"^M10\.", "C or D", ""), (r"^M11\.G", "D", ""), (r"^M11\.E", "C, D or E", "M11.E"),
    (r"^M11\.D", "C", ""), (r"^M11\.", "C", "acceptance-test rules"),
    (r"^M12\.", "", "a prohibition, checked by review in every phase"),
    (r"^M15\.", "E", ""), (r"^M16\.D", "C", ""), (r"^M16\.E", "F", ""), (r"^M16\.", "C", ""),
    (r"^M17\.A\.[134]$", "C", "M13.4"), (r"^M17\.", "E", ""), (r"^M14\.", "", "every phase"),
    (r"^M13\.", "C", ""), (r"^M3\.", "C", ""), (r"^M0\.", "C", ""),
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


def module_phase(spec_id: str) -> tuple[str, str] | None:
    for pattern, phase, detail in MODULE_PHASES:
        if re.match(pattern, spec_id):
            return phase, detail
    return None


def untested_note(phase: str, detail: str) -> str:
    """Plan A4: an ID with no test named for it, with the phase that built, or builds, its module."""
    letters = set(re.findall(r"\b[A-F]\b", phase))
    if not letters:
        return f"no test named for it ({detail or 'its criterion row names no phase'})"
    where = f"Phase {phase}" + (f" ({detail})" if detail else "")
    if letters <= BUILT_PHASES:
        return f"no test named for it; its module was built in {where}"
    if letters.isdisjoint(BUILT_PHASES):
        return f"no test named for it; {where} builds it and is not built yet"
    return f"no test named for it; {where} builds it; Phase {', '.join(sorted(letters - BUILT_PHASES))} is not built yet"


def coverage(overrides: dict) -> list[dict]:
    text = SPEC.read_text(encoding="utf-8")
    ids, tests, criteria = spec_ids(text), named_tests(), criterion_phases(text)
    verdicts, unbuilt, as_test, reds = ensemble_verdicts(), not_built(), run_as_test(), mutation_reds()
    unmet = unmet_clauses(text)
    classes = {cid: cls.lower() for cid, cls in CLASS_ROW.findall(text)}
    from tools.version_1_status import dispositions

    decided = dispositions()
    unknown = sorted(set(overrides) - set(ids))
    if unknown:
        raise ValueError(f"overrides name IDs the spec does not define: {unknown}")
    # Nothing a record or the dispositions name may be dropped unread (P2): a key whose ID the spec does not define,
    # or a red mutant whose entry no ensemble record holds, is refused.
    entries = {key for found in verdicts.values() for key, _, _ in found}
    orphans = sorted((set(verdicts) | {k.split("[")[0] for k in decided}) - set(ids)) + sorted(reds - entries)
    if orphans:
        raise ValueError(f"records or dispositions name what no spec ID or ensemble entry holds: {orphans}")
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
        elif spec_id in verdicts:
            # A recorded verdict wins over a not-built list (M11.C.14: not built in Phase C, run in Phase D).
            entries = verdicts[spec_id]
            passed = all(outcome == "PASS" for _, outcome, _ in entries)
            unproved = [key for key, _, _ in entries if key not in reds]
            notes = ["ensemble: " + "; ".join(f"{key} {outcome} (`docs/{record}`)" for key, outcome, record in entries)]
            if passed and unproved:
                notes.append(f"passes, but no mutant in a mutation record has turned {', '.join(unproved)} red")
                if classes.get(spec_id) in NO_RECORDED_MUTANT_NEEDED:
                    notes.append(f"a {classes[spec_id]}: `docs/version_1_status.md` counts the pass as final without "
                                 "one (V3 asks it of composite and mixed criteria only)")
            if spec_id in unmet:
                notes.append("its spec row says its own mutation clause is not met")
            if not named:
                notes.append("no test is named for it")
            done = passed and not unproved and spec_id not in unmet and bool(named)
            rows.append({"id": spec_id, "status": "done" if done else "partial", "evidence": named,
                         "note": "; ".join(notes)})
        elif spec_id in as_test:
            test, why = as_test[spec_id]
            rows.append({"id": spec_id, "status": "done", "evidence": sorted(set(named) | {test}),
                         "note": f"run as a test, not an ensemble: {why} (`tools/phase_d_record.py`)"})
        elif spec_id in unbuilt:
            rows.append({"id": spec_id, "status": "not done", "evidence": named, "note": f"not built: {unbuilt[spec_id]}"})
        elif named and spec_id.startswith("M11.C."):
            # An M11.C test is ensemble-marked, outside the default suite: without a recorded verdict nothing says it
            # passes (review 2026-10-11: the Phase D criteria were read as done while three of them fail).
            rows.append({"id": spec_id, "status": "partial", "evidence": named,
                         "note": "built; no ensemble record holds a verdict for it"})
        elif named:
            rows.append({"id": spec_id, "status": "done", "evidence": named, "note": ""})
        else:
            phase = (criteria[spec_id], "") if spec_id in criteria else module_phase(spec_id)
            rows.append({"id": spec_id, "status": "not done", "evidence": [],
                         "note": untested_note(*phase) if phase else UNASSIGNED})
    for row in rows:
        quoted = disposition_notes(row["id"], decided)
        if quoted:
            row["note"] = "; ".join(([row["note"]] if row["note"] else []) + quoted)
    for row in rows:
        if row["status"] == "done" and not row["evidence"]:
            raise ValueError(f"{row['id']} is marked done with no evidence")
        if row["status"] not in {"done", "partial", "not done"}:
            raise ValueError(f"{row['id']}: unknown status {row['status']!r}")
    return rows


def render(rows: list[dict], overrides_name: str, unmapped: list[str] | None = None) -> str:
    counts = {s: sum(r["status"] == s for r in rows) for s in ("done", "partial", "not done")}
    lines = [
        "# Spec coverage",
        "",
        f"Generated by `tools/spec_coverage.py {overrides_name}` (spec `M14.1`). Every ID in",
        "`docs/bowen_agent_model_spec_v2.md`, in document order. **done** carries the test that proves it;",
        "**partial** says what is missing; **not done** says it is not built, or that no test is named for it",
        "and which phase built, or builds, its module. An `M11.C` criterion is **done** only when every entry",
        "passes in an ensemble record, a mutant in a mutation record has turned each entry red, and its spec",
        "row does not say its own mutation clause is not met; or when it is run as a test, not an ensemble.",
        "That last check reads the row's own annotation only, not which mutant the clause names: a criterion",
        "proved only by a joint mutant is done (`docs/phase_c_completion_report.md`).",
        "Where the owner recorded a version 1 disposition (`docs/version_1_dispositions.md`), the note quotes",
        "it; the status stays what the records show. The D9 sweep (`docs/phase_c_sweep_record.md`) is",
        "reported, not gated.",
        "",
        "Records read, each with the hash it states:",
        "",
        *[f"- `{p.relative_to(REPO).as_posix()}` ({record_hash(p)[:12]})" for p in ENSEMBLE_RECORDS + MUTATION_RECORDS],
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
