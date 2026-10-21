"""The spec coverage report is current.

Purpose: fail when docs/spec_coverage.md no longer matches what the generator
         produces from the spec, the tests and the overrides — a stale coverage
         report is a status claim that does not reconcile with its artifact (P6).
Spec:    docs/bowen_agent_model_spec_v2.md#M14.1, #M14.2
Tests:   this file
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from unittest.mock import patch

from tools.record_json import record_json

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("spec_coverage_tool", REPO / "tools" / "spec_coverage.py")
TOOL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TOOL)
OVERRIDES = REPO / "tools" / "spec_coverage_phase_b.json"


def test_m141_spec_coverage_is_current():
    expected = TOOL.report(json.loads(OVERRIDES.read_text(encoding="utf-8")), OVERRIDES.name)
    actual = (REPO / "docs" / "spec_coverage.md").read_text(encoding="utf-8")
    assert actual == expected, "regenerate: python3 tools/spec_coverage.py tools/spec_coverage_phase_b.json docs/spec_coverage.md"


def test_m141_every_done_names_an_existing_test():
    rows = TOOL.coverage(json.loads(OVERRIDES.read_text(encoding="utf-8")))
    for row in rows:
        if row["status"] != "done":
            continue
        assert row["evidence"], row["id"]
        for item in row["evidence"]:
            path, _, name = item.partition("::")
            source = (REPO / path).read_text(encoding="utf-8")
            assert not name or f"def {name}(" in source, f"{row['id']}: {item} does not exist"


def test_m141_no_id_resolves_to_an_unassigned_phase():
    """A new ID that no MODULE_PHASES pattern matches was reported "Phase unassigned"."""
    rows = TOOL.coverage(json.loads(OVERRIDES.read_text(encoding="utf-8")))
    unassigned = [r["id"] for r in rows if r["note"].startswith(TOOL.UNASSIGNED)]
    assert unassigned == [], f"IDs with no building phase: {unassigned}"


def test_m141_excluded_evidence_is_not_shown():
    """A test pinning a superseded revision must not be merged back in as evidence."""
    overrides = json.loads(OVERRIDES.read_text(encoding="utf-8"))
    named = TOOL.tests_for("M4.C.1", TOOL.named_tests())
    assert named, "no test is named for M4.C.1, so this guard checks nothing"
    excluded = named[0]
    overrides["M4.C.1"] = {"status": "partial", "note": "guard fixture", "exclude_evidence": [excluded]}
    rows = {r["id"]: r for r in TOOL.coverage(overrides)}
    assert excluded not in rows["M4.C.1"]["evidence"]
    assert set(named[1:]) <= set(rows["M4.C.1"]["evidence"])  # only the excluded one is dropped


def _rows() -> dict[str, dict]:
    return {r["id"]: r for r in TOOL.coverage(json.loads(OVERRIDES.read_text(encoding="utf-8")))}


def test_m141_phase_d_criteria_read_from_their_records():
    """Plan A1 (docs/plan_spec_coverage_phase_d.md): every entry in every ensemble record, Phase D's included, is in its
    criterion's row with its verdict and record. Mutation: the tool reads the Phase C record only."""
    rows = _rows()
    missing = []
    for path in TOOL.ENSEMBLE_RECORDS:
        for entry in record_json(path):
            key = entry["criterion"]
            row = rows[key.split("[")[0]]
            if f"{key} {entry['outcome']} (`docs/{path.name}`)" not in row["note"]:
                missing.append(f"{key} ({path.name})")
    assert missing == []
    assert any(p.name == "phase_d2_ensemble_record.md" for p in TOOL.ENSEMBLE_RECORDS)
    assert rows["M11.C.47"]["status"] == "partial", "a Phase D part II criterion read as unrecorded or done"


def test_m141_accepted_finding_shows_its_disposition():
    """Plan A2: a criterion the owner accepted as a finding, or put out of scope, carries the disposition's reason in its
    note, and its status stays what the records show (never done). Mutation: the disposition is not read."""
    from tools.version_1_status import dispositions

    rows = _rows()
    decided = {k: d for k, d in dispositions().items() if d["kind"] in ("accepted finding", "out of scope")}
    assert decided, "no accepted finding or out-of-scope disposition, so this checks nothing"
    for key, d in decided.items():
        row = rows[key.split("[")[0]]
        assert d["reason"] in row["note"], key
        assert row["status"] != "done", key


def test_m141_header_names_each_record_read():
    """Plan A3: the page names every ensemble and criterion mutation record in docs/ with the hash it states, so a new
    record the tool does not read fails here instead of leaving the page silently short."""
    on_disk = {p for p in (REPO / "docs").glob("*_ensemble_record.md")}
    on_disk |= {p for p in (REPO / "docs").glob("*_mutation_record.md") if '"criterion"' in p.read_text(encoding="utf-8")}
    assert on_disk == set(TOOL.ENSEMBLE_RECORDS + TOOL.MUTATION_RECORDS)
    page = (REPO / "docs" / "spec_coverage.md").read_text(encoding="utf-8")
    header = page.split("| ID | Status |", 1)[0]
    for path in on_disk:
        assert f"`docs/{path.name}` ({TOOL.record_hash(path)[:12]})" in header, path.name
    assert "after Phase C" not in header


def test_m141_untested_id_is_not_read_as_unbuilt():
    """Plan A4: an ID with no test named for it says so, and says whether its module's phase is built, instead of a
    bare phase that reads as "not built yet". Mutation: the note is the bare phase."""
    rows = _rows()
    untested = [r for r in rows.values() if r["note"].startswith("no test named for it")]
    assert untested, "no untested ID, so this checks nothing"
    bare = [r["id"] for r in rows.values() if r["status"] == "not done" and re.match(r"Phase [A-F]\b", r["note"])]
    assert bare == []
    assert all(r["status"] == "not done" and not r["evidence"] for r in untested)
    assert "its module was built in Phase D (the twelve-person reference family)" in rows["M2.2"]["note"]
    assert TOOL.untested_note("E", "") == "no test named for it; Phase E builds it and is not built yet"


def test_m141_unmet_mutation_clause_keeps_a_passing_criterion_partial():
    """Plan A5: `M11.C.16` has no hand-written override; its row comes from the record, and because its spec row says
    its own mutation clause is not met, a pass with red mutants is still partial. Mutation: the clause is not read."""
    assert "M11.C.16" not in json.loads(OVERRIDES.read_text(encoding="utf-8"))
    row = _rows()["M11.C.16"]
    assert "M11.C.16 PASS (`docs/phase_c_ensemble_record.md`)" in row["note"]
    assert row["status"] == "partial"
    assert "its spec row says its own mutation clause is not met" in row["note"]


def test_m141_unmet_clause_marker_reads_exactly_the_annotated_rows():
    """Plan A5 (review 2026-10-20): the spec rows whose mutation cell says the clause is not met are pinned, so a new
    annotation, in whatever words, forces a look; wordings that say "not met" without those two words are read too,
    and wordings that say "met" are not."""
    text = TOOL.SPEC.read_text(encoding="utf-8")
    assert TOOL.unmet_clauses(text) == {"M11.C.14", "M11.C.16", "M11.C.46"}
    for unmet in ("*Not met, 2026-10-08*", "**Mutation clause not met:**", "not yet met", "the clause is unmet",
                  "it isn't met"):
        assert TOOL.UNMET_CLAUSE.search(unmet), unmet
    for met in ("met only jointly", "the clause is met", "now met; was failing", "a metric"):
        assert not TOOL.UNMET_CLAUSE.search(met), met


def test_m141_criterion_run_as_a_test_is_done_with_that_test():
    """Plan A1: `M11.C.37` has no ensemble entry; it is proved by the test `tools/phase_d_record.py` names."""
    test, _ = TOOL.run_as_test()["M11.C.37"]
    row = _rows()["M11.C.37"]
    assert row["status"] == "done"
    assert test in row["evidence"]
    assert row["note"].startswith("run as a test, not an ensemble")


def test_m141_a_key_in_two_records_is_refused():
    """Plan A1: an entry with a verdict in two ensemble records would have two verdicts; the tool refuses it."""
    record = TOOL.ENSEMBLE_RECORDS[0]
    with patch.object(TOOL, "ENSEMBLE_RECORDS", (record, record)):
        try:
            TOOL.ensemble_verdicts()
        except ValueError as refused:
            assert "has a verdict in both" in str(refused)
        else:
            raise AssertionError("a key in two records was accepted")


def test_m141_an_orphan_key_is_refused_not_dropped():
    """P2 (review 2026-10-20): a red mutant whose entry no ensemble record holds would be read by nothing."""
    with patch.object(TOOL, "mutation_reds", return_value=TOOL.mutation_reds() | {"M11.C.1[no such variant]"}):
        try:
            TOOL.coverage(json.loads(OVERRIDES.read_text(encoding="utf-8")))
        except ValueError as refused:
            assert "M11.C.1[no such variant]" in str(refused)
        else:
            raise AssertionError("an orphan mutation key was dropped silently")
