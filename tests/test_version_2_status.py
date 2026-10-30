"""The version 2 status page (`docs/version_2_status.md`) and the stage 2 conditions W1 and W4.

Purpose: prove that every `M11.C` criterion has a row with a status (W1, S2.8), that the sweep column is a fraction
         and never a verdict (W4, `M11.5`, S2.9), that a verdict changed since the tag `version-1` is not final and a
         verdict that is the tag's carries version 1's disposition (W1, S2.10), and that the page is what the tool
         renders (S2.11).
Spec:    docs/implementation_plan_version_2.md#2.3 (S2.8–S2.11); docs/bowen_agent_model_spec_v2.md#M11.5
Tests:   this file

These read the committed records and dispositions; nothing here runs the engine. Every patched module attribute is
restored by ``mock.patch``.
"""

from __future__ import annotations

import json

import re
from unittest import mock

import pytest

import tools.version_1_status as v1
import tools.version_2_status as tool
import open_batch  # tests/ is on the path: pytest imports its conftest by basename

STATUS_WORDS = ("PASS", "FAIL", "pass", "fail")


@pytest.fixture(autouse=True)
def _registry_as_found():
    """While stage 3's batch is open (D31) the records and this page are the batch's as found, so the tool reads the
    criteria the batch found: a criterion added since (step 8's `M11.C.49`) is in no record until step 10."""
    if not open_batch.OPEN:
        yield
        return
    with open_batch.registry_as_found(globals()):
        yield


def _sweep_cells(text: str) -> dict[str, str]:
    table = text.split("## Every criterion", 1)[1]
    cells = {}
    for line in table.splitlines():
        if line.startswith("| `"):
            parts = [c.strip() for c in line.strip().strip("|").split("|")]
            cells[parts[0].strip("`")] = parts[7]
    return cells


def test_w1_s2_8_every_criterion_has_a_status():
    """Every `M11.C` id in the spec's criteria table has a row (a built one per record row, an unbuilt one otherwise),
    every row has a status, and the rendered page shows each."""
    rows = tool.statuses()
    spec = v1.spec_criteria()
    bases = {s.key.split("[", 1)[0] for s in rows}
    assert set(spec) <= bases, sorted(set(spec) - bases)
    assert all(s.status for s in rows)
    assert len({s.key for s in rows}) == len(rows)
    text = tool.render(rows)
    for s in rows:
        assert f"| `{s.key}` |" in text
    assert "- W1 (every criterion has a final status):" in text
    not_final = {s.key for s in rows if not s.final}
    assert not_final == {s.key for s in rows if "W1" in s.blocks}


def test_w4_s2_9_sweep_column_is_a_fraction_not_a_verdict(tmp_path):
    """A swept criterion's cell is *n of S held*; a directional criterion not swept says so; a premise or check and
    an unbuilt criterion get a dash; no cell of the column is a pass or a fail word. The reader shows an errored
    count beside the fraction and never folds it into S; a criterion the record shows as declared, not run has no
    cell from the record."""
    with mock.patch.object(tool, "sweep_fractions", return_value={"M11.C.16": "5 of 8 held"}), \
            mock.patch.object(tool, "sweep_refused", return_value={}), \
            mock.patch.object(tool, "sweep_stale", return_value=None):  # a current record; the stale one is tested below
        rows = tool.statuses()
        text = tool.render(rows)
    by_key = {s.key: s for s in rows}
    assert by_key["M11.C.16"].sweep == "5 of 8 held" and "W4" not in by_key["M11.C.16"].blocks
    passing_unswept = [s for s in rows if s.verdict == "PASS" and s.cls in v1.DIRECTIONAL and s.key != "M11.C.16"]
    assert passing_unswept and all(s.sweep == "not swept" and "W4" in s.blocks for s in passing_unswept)
    assert all(s.sweep == "—" for s in rows if s.cls in v1.PREMISE_LIKE)
    assert all(s.sweep == "—" for s in rows if s.verdict == "not built")
    cells = _sweep_cells(text)
    assert cells["M11.C.16"] == "5 of 8 held" and len(cells) == len(rows)
    assert not any(re.search(r"\b(PASS|FAIL|pass|fail)\b", cell) for cell in cells.values())
    assert "- W4 (every passing directional criterion has a sweep result): **not met**" in text
    record = tmp_path / "sweep.md"
    record.write_text('# r\n\nengine_hash: abc\n\n```json\n[{"criterion": "M11.C.16", "held": 3, "counted": 7, '
                      '"errored": 1, "samples": 8}, {"criterion": "M11.C.6", "not_run": "declared only", '
                      '"constants": [], "rows": []}]\n```\n', encoding="utf-8")
    assert tool.sweep_fractions(record) == {"M11.C.16": "3 of 7 held (1 errored, not counted)"}
    assert tool.sweep_fractions(tmp_path / "missing.md") == {}


def test_w4_a_refused_criterion_is_shown_refused_and_still_counts_for_w4(tmp_path):
    """T-159 (owner 2026-10-29): a criterion the sweep refused at the rerun (`M17.A.1`) has no fraction. Its cell says
    it was refused, never *n of S held* nor *not swept*, and W4 still counts it. Mutation: the refusal read as a
    fraction (W4 met on its row); its cell left as *not swept*. Built on `M11.C.16`, a passing directional criterion,
    since `M11.C.48` fails at spec revision 43."""
    record = tmp_path / "sweep.md"
    record.write_text('# r\n\nengine_hash: abc\n\n```json\n[{"criterion": "M11.C.16", "refused": {"sample": 2, '
                      '"seeds": 200, "own_seeds": 100}, "samples": 3, "constants": [], "rows": []}]\n```\n',
                      encoding="utf-8")
    assert tool.sweep_fractions(record) == {}
    assert tool.sweep_refused(record) == {"M11.C.16": "refused at the rerun (`M17.A.1`), sample 2"}
    assert tool.sweep_refused(tmp_path / "missing.md") == {}
    with mock.patch.object(tool, "SWEEP_RECORD", record), mock.patch.object(tool, "sweep_stale", return_value=None):
        rows = {s.key: s for s in tool.statuses()}
    assert rows["M11.C.16"].verdict == "PASS"  # a passing directional criterion, so W4 asks for a sweep result
    assert rows["M11.C.16"].sweep == "refused at the rerun (`M17.A.1`), sample 2" and "W4" in rows["M11.C.16"].blocks
    # a refusal on an earlier engine is marked as such, as a fraction is (correctness pass, T-159)
    with mock.patch.object(tool, "SWEEP_RECORD", record), mock.patch.object(tool, "sweep_stale", return_value="old"):
        stale = {s.key: s for s in tool.statuses()}["M11.C.16"]
    assert stale.sweep == f"refused at the rerun (`M17.A.1`), sample 2 ({tool.STALE_CELL})" and "W4" in stale.blocks


def test_w1_s2_10_a_changed_verdict_is_not_final():
    """A row whose verdict is not the tag's says it changed, is not final and blocks W1, and version 1's disposition
    is not applied to it; a row whose verdict is the tag's carries the disposition; a criterion the tag's page does
    not have is new, not changed."""
    tag = tool.tag_verdicts()
    assert tag["M11.C.16"] == "PASS" and tag["M11.C.42"] == "FAIL"
    rows = {s.key: s for s in tool.statuses()}
    c42 = rows["M11.C.42"]
    assert c42.verdict == "FAIL" and c42.at_tag == "FAIL" and c42.status == "fail accepted as a finding"
    assert c42.final and c42.note and "owner 2026-10-17" in c42.note
    new = [s for s in rows.values() if s.at_tag == "—" and s.verdict != "not built"]
    assert new and not any(tool.CHANGED in s.status for s in new)
    flipped = dict(tag, **{"M11.C.16": "FAIL", "M11.C.42": "PASS"})
    with mock.patch.object(tool, "tag_verdicts", return_value=flipped):
        changed = {s.key: s for s in tool.statuses()}
    for key in ("M11.C.16", "M11.C.42"):
        s = changed[key]
        assert s.status.endswith(tool.CHANGED) and not s.final and "W1" in s.blocks
        assert s.at_tag == flipped[key]
    note = changed["M11.C.42"].note  # the tag's disposition is shown as not applied, never applied
    assert note.startswith("version 1 disposition, not applied (the verdict changed): accepted finding") and "owner 2026-10-17" in note
    assert tool.CHANGED in changed["M11.C.42"].status and not changed["M11.C.42"].final
    text = tool.render(list(changed.values()))
    assert "| FAIL | PASS |" in text and tool.CHANGED in text


def test_w1_s2_10_the_tag_page_is_read_whole(tmp_path):
    """Every row of the frozen page is read, an empty class cell included (`M11.C.8`), and a row the pattern cannot
    read stops the tool instead of vanishing (P2)."""
    tag = tool.tag_verdicts()
    frozen = tool.TAG_PAGE.read_text(encoding="utf-8").split("## Every criterion", 1)[1]
    assert len(tag) == sum(1 for line in frozen.splitlines() if line.startswith("| `")) == 60
    assert tag["M11.C.8"] == "not built"
    page = tmp_path / "page.md"
    page.write_text("# p\n\n## Every criterion\n\n| a |\n|---|\n| `M11.C.1` |  | — | not built | x |\n| `M11.C.2` | premise | Phase C | PASS | x |\n"
                    "| `M11.C.3` broken row |\n", encoding="utf-8")
    import pytest
    with pytest.raises(SystemExit, match="3 criterion rows"):
        tool.tag_verdicts(page)
    page.write_text("# p\n\n## Every criterion\n\n| a |\n|---|\n| `M11.C.1` |  | — | not built | x |\n| `M11.C.2` | premise | Phase C | PASS | x |\n", encoding="utf-8")
    assert tool.tag_verdicts(page) == {"M11.C.1": "not built", "M11.C.2": "PASS"}


def test_w1_s2_10_a_tag_row_with_no_row_now_and_a_misplaced_disposition_stop_the_tool():
    """A key the tag page had and the records no longer have is shown, not dropped (P2); a disposition that names a
    criterion whose rows are all variants must name each variant."""
    import pytest
    tag = tool.tag_verdicts()
    with mock.patch.object(tool, "tag_verdicts", return_value=dict(tag, **{"M11.C.999": "PASS"})), \
            pytest.raises(SystemExit, match="M11.C.999"):
        tool.statuses()
    rows = tool.record_rows()
    variant_only = sorted({k.split("[", 1)[0] for k, _, _ in rows if "[" in k} - {k for k, _, _ in rows})
    assert variant_only  # the check needs a criterion whose rows are all variants
    decided = dict(tool.v1.dispositions())
    decided[variant_only[0]] = {"kind": "accepted finding", "reason": "x", "decided": "test", "owner": True}
    with mock.patch.object(tool.v1, "dispositions", return_value=decided), pytest.raises(SystemExit, match=variant_only[0]):
        tool.statuses()


def test_s3_24_each_verdict_is_shown_beside_revision_38s():
    """S3.24: every row shows the verdict this page gave at revision 38 (`9ebf327`, read from git) beside its verdict
    now and the tag's; a criterion added since (`M11.C.49`) says so; the summary line lists exactly the rows whose
    verdict moved; a revision 38 row with no row now stops the tool (P2); the column never sets a status."""
    import pytest
    earlier = tool.revision_38_verdicts()
    assert len(earlier) == 62 and "M11.C.49" not in earlier
    rows = tool.statuses()
    assert {s.key: s.at_revision_38 for s in rows if s.key in earlier} == earlier
    assert next(s for s in rows if s.key == "M11.C.49").at_revision_38 == "new since revision 38"
    text = tool.render(rows)
    moved = [s for s in rows if s.at_revision_38 != s.verdict]
    line = next(x for x in text.splitlines() if x.startswith("Verdicts not the one this page gave at revision 38"))
    assert f": {len(moved)} of {len(rows)} rows" in line and all(f"`{s.key}`" in line for s in moved)
    header = next(x for x in text.splitlines() if x.startswith("| Criterion |"))
    assert [c.strip() for c in header.strip("|").split("|")][4:6] == ["At `version-1`", "At revision 38"]
    with mock.patch.object(tool, "revision_38_verdicts", return_value=dict(earlier, **{"M11.C.999": "PASS"})), \
            pytest.raises(SystemExit, match="M11.C.999"):
        tool.statuses()
    flipped = {k: ("FAIL" if v == "PASS" else "PASS") for k, v in earlier.items()}
    with mock.patch.object(tool, "revision_38_verdicts", return_value=flipped):
        assert [(s.status, s.final, s.blocks) for s in tool.statuses()] == [(s.status, s.final, s.blocks) for s in rows]


def test_w4_a_fraction_from_an_earlier_engine_is_marked_and_not_counted(tmp_path):
    """P8/P6 (review of 2026-10-26): a sweep record whose printed hashes are not the current engine's is not read as
    current. Its fractions are shown with *earlier engine* and every passing directional criterion blocks W4; with the
    current hashes printed the same record is read as current; a record printing no hash is refused."""
    import pytest
    from tools.phase_d_mutation_record import engine_hash as phase_d_engine_hash
    from tools.record_cache import engine_hash

    rows = [{"criterion": "M11.C.16", "held": 5, "counted": 8, "errored": 0, "rows": []}]
    body = "```json\n" + json.dumps(rows) + "\n```\n"
    record = tmp_path / "sweep.md"
    record.write_text(f"# s\n\nengine_hash: {'0' * 64}\nphase_d_engine_hash: {phase_d_engine_hash()}\n\n{body}",
                      encoding="utf-8")
    assert "engine_hash `000000000000`" in tool.sweep_stale(record)
    with mock.patch.object(tool, "SWEEP_RECORD", record):
        rows_now = {s.key: s for s in tool.statuses()}
        note = tool.sweep_stale(record)
    with mock.patch.object(tool, "sweep_stale", return_value=note):  # render names the record by its repo path
        text = tool.render(list(rows_now.values()))
    c16 = rows_now["M11.C.16"]
    assert c16.sweep == f"5 of 8 held ({tool.STALE_CELL})"
    assert all("W4" in s.blocks for s in rows_now.values() if s.verdict == "PASS" and s.cls in tool.v1.DIRECTIONAL)
    assert "The sweep record's fractions were measured on engine_hash `000000000000`" in text
    record.write_text(f"# s\n\nengine_hash: {engine_hash()}\nphase_d_engine_hash: {phase_d_engine_hash()}\n\n{body}",
                      encoding="utf-8")
    assert tool.sweep_stale(record) is None
    with mock.patch.object(tool, "SWEEP_RECORD", record):
        assert next(s for s in tool.statuses() if s.key == "M11.C.16").sweep == "5 of 8 held"
    record.write_text(f"# s\n\n{body}", encoding="utf-8")
    with pytest.raises(SystemExit, match="no engine_hash line"):
        tool.sweep_stale(record)


def test_w3_a_pass_whose_named_mutant_survived_is_not_labelled_unmutated():
    """Milestone review MR-28: a directional pass with no red mutant was labelled "pass, no recorded mutation" even
    where its named mutant ran and survived (`M11.C.11`, `M11.C.15`, `M11.C.48`, `M11.C.48[phase_d]`,
    `docs/phase_d_mutation_record.md`). Such a pass says the mutant survived, names it, and still blocks W1 and W3; a
    pass with no mutant in any record keeps the old label. Since spec revision 43 `M11.C.11`, `M11.C.15` and
    `M11.C.17` each have a red mutant beside the survivor, so they are proved; with the red ones taken away the
    survivor is what the page shows. Mutation: survived mutants not read (all three unmutated)."""
    keys = ("M11.C.11", "M11.C.15", "M11.C.17")
    rows = {s.key: s for s in tool.statuses()}
    for key in keys:
        assert rows[key].status == "pass" and rows[key].final and "proved by" in rows[key].evidence, key
    reds = {k: v for k, v in tool.v1.red_mutants().items() if k not in keys}
    with mock.patch.object(tool.v1, "red_mutants", return_value=reds):
        rows = {s.key: s for s in tool.statuses()}
        with mock.patch.object(tool, "survived_mutants", return_value={}):
            unmutated = {s.key: s for s in tool.statuses()}
    for key in keys:
        s = rows[key]
        assert s.status == "pass, named mutant survived", (key, s.status)
        assert not s.final and {"W1", "W3"} <= s.blocks
        assert "survived" in s.evidence, (key, s.evidence)
    assert rows["M11.C.17"].evidence.endswith("; survived: `landed-contact-changes-nothing` (Phase D mutation record)")
    assert unmutated["M11.C.17"].status == "pass, no recorded mutation"
    assert not unmutated["M11.C.17"].final and {"W1", "W3"} <= unmutated["M11.C.17"].blocks


def test_s2_11_status_page_is_current():
    """The committed page is what the tool renders from the committed records (P6)."""
    assert tool.PAGE.read_text(encoding="utf-8") == tool.render(tool.statuses())


def test_s2_11_records_read_shows_every_hash_a_record_keys_by():
    """The sweep record keys its Phase D rows by a second hash, ``phase_d_engine_hash``; the page's *Records read*
    lists every hash line the record carries, each at its value there (review, 2026-10-23)."""
    record = tool.SWEEP_RECORD.read_text(encoding="utf-8")
    hashes = dict(re.findall(r"^(engine_hash|phase_d_engine_hash): (\w+)", record, re.M))
    assert set(hashes) == {"engine_hash", "phase_d_engine_hash"}
    line = next(l for l in tool.render(tool.statuses()).splitlines() if l.startswith("- Sweep record"))
    assert all(f"{kind} `{value[:12]}`" in line for kind, value in hashes.items())


# Version 2's dispositions (docs/version_2_definition_plan.md §3).

def _v2(key, kind, verdict, decided="owner 2026-10-29"):
    return {key: {"kind": kind, "verdict": verdict, "owner": decided.startswith("owner "), "decided": decided,
                  "reason": "test reason"}}


PENDING_TEXT = "Claude's recommendation 2026-10-29, pending owner review"


def test_w1_v2_dispositions_parse_strictly(tmp_path):
    """The committed file parses; a wrong cell count, an unknown kind, a verdict other than PASS or FAIL, an accepted
    finding on a PASS, a `decided` in neither form, a repeated key and a row after a break each raise."""
    real = tool.v2_dispositions()
    assert real and all(d["verdict"] in ("PASS", "FAIL") for d in real.values())
    head = tool.V2_HEADER + "\n|---|---|---|---|---|\n"
    good = "| `M11.C.49` | accepted finding | FAIL | owner 2026-10-29 | why |\n"
    path = tmp_path / "v2.md"
    path.write_text("# t\n\n" + head + good, encoding="utf-8")
    assert tool.v2_dispositions(path)["M11.C.49"] == {"kind": "accepted finding", "verdict": "FAIL", "owner": True,
                                                       "decided": "owner 2026-10-29", "reason": "why"}
    assert tool.v2_dispositions(tmp_path / "absent.md") == {}
    bad = (head + "| `M11.C.49` | accepted finding | FAIL | owner 2026-10-29 |\n",
           head + good.replace("accepted finding", "accepted"),
           head + good.replace("| FAIL |", "| pass |"),
           head + good.replace("| FAIL |", "| PASS |"),
           head + good.replace("owner 2026-10-29", "the owner"),
           head + good + good,
           head + good + "\n" + good.replace("49", "48"),
           "# no table\n")
    for text in bad:
        path.write_text(text, encoding="utf-8")
        with pytest.raises(SystemExit):
            tool.v2_dispositions(path)


def test_w1_v2_disposition_takes_precedence_over_version_1():
    """`M11.C.42` fails as at the tag and carries version 1's owner-accepted finding; a version 2 row on it that is
    Claude's recommendation takes its place, so the row is no longer final."""
    before = {s.key: s for s in tool.statuses()}["M11.C.42"]
    assert before.final and before.status == "fail accepted as a finding"
    with mock.patch.object(tool, "v2_dispositions", return_value=_v2("M11.C.42", "accepted finding", "FAIL", PENDING_TEXT)):
        after = {s.key: s for s in tool.statuses()}["M11.C.42"]
    assert after.status == "fail, acceptance pending the owner's review" and not after.final and "W1" in after.blocks
    assert after.note.startswith("test reason") and PENDING_TEXT in after.note


def test_w1_v2_verdict_reviewed_by_the_owner_makes_a_changed_verdict_final():
    """`M11.C.16` (a premise that passes) with the tag's verdict flipped to FAIL is held as changed; the owner's
    `verdict reviewed` on PASS makes it final, and Claude's recommendation of the same leaves it held."""
    flipped = dict(tool.tag_verdicts(), **{"M11.C.16": "FAIL"})
    with mock.patch.object(tool, "tag_verdicts", return_value=flipped):
        held = {s.key: s for s in tool.statuses()}["M11.C.16"]
        with mock.patch.object(tool, "v2_dispositions", return_value=_v2("M11.C.16", "verdict reviewed", "PASS")):
            owner = {s.key: s for s in tool.statuses()}["M11.C.16"]
        with mock.patch.object(tool, "v2_dispositions",
                               return_value=_v2("M11.C.16", "verdict reviewed", "PASS", PENDING_TEXT)):
            claude = {s.key: s for s in tool.statuses()}["M11.C.16"]
    assert held.status.endswith(tool.CHANGED) and not held.final
    assert owner.status == f"pass; {tool.REVIEWED}" and owner.final and "W1" not in owner.blocks
    assert owner.note.startswith("version 2: verdict reviewed (PASS) — test reason")
    assert claude.status.endswith(tool.CHANGED) and not claude.final and "W1" in claude.blocks
    assert PENDING_TEXT in claude.note


def test_w1_v2_verdict_reviewed_names_the_verdict_it_reviewed():
    """A review decided on FAIL does not clear a row that now passes: the row stays held as changed, and the page shows
    the disposition as not applied, with both verdicts."""
    flipped = dict(tool.tag_verdicts(), **{"M11.C.16": "FAIL"})
    with mock.patch.object(tool, "tag_verdicts", return_value=flipped), \
            mock.patch.object(tool, "v2_dispositions", return_value=_v2("M11.C.16", "verdict reviewed", "FAIL")):
        s = {s.key: s for s in tool.statuses()}["M11.C.16"]
    assert s.status.endswith(tool.CHANGED) and not s.final and "W1" in s.blocks
    assert s.note.startswith("version 2 disposition, not applied (decided on FAIL, now PASS): verdict reviewed")


def test_w1_v2_a_disposition_on_a_key_with_no_record_row_stops_the_tool():
    with mock.patch.object(tool, "v2_dispositions", return_value=_v2("M11.C.9", "note", "PASS")):
        with pytest.raises(SystemExit, match="names rows no record has"):
            tool.statuses()
