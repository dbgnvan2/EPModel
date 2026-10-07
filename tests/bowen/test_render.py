"""The deterministic renderer.

Purpose: the rendered trace carries what M16.C.2 requires on every event line,
         is deterministic, can show one person's view, and carries the header
         and M11.F's framing without reading as a clinical record.
Spec:    docs/bowen_agent_model_spec_v2.md#M16.T.1, #M16.C.1, #M16.C.2, #M16.C.4, #M16.C.5, #M11.F.9, #M11.F.10
Tests:   this file
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.log_records import EmittedRecord
from src.bowen.io.load import load_family
from src.bowen.render import trace
from src.bowen.render.trace import TABLE_HEAD, UnrenderableRecord, render
from src.bowen.run import run_phase_b

NAMES = load_family().display_names
RECORDS = run_phase_b(7).records
TEXT = render(RECORDS, NAMES)


def event_rows(text: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        if re.match(r"^\| \d+ \| ", line) and "(system)" not in line:
            rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


def test_m16t1_trace_renders_scripted_run():
    """G2: one line per event, each carrying week, actor, move, target, witnesses and what it did."""
    assert TABLE_HEAD in TEXT
    emitted = [r.event for r in RECORDS if isinstance(r, EmittedRecord)]
    rows = event_rows(TEXT)
    assert len(rows) == len(emitted)
    for row, event in zip(rows, emitted):
        week, who, move, toward, witnesses, did = row
        assert int(week) == event.timestamp and move == event.kind
        assert all(cell for cell in row)
        # Effects beside causes: every appraised event names who changed and by how much.
        if event.kind not in ("TRIGGER",):
            assert re.search(r"anxiety \w+ [+-]\d+\.\d", did), row
        for w in event.witnesses:
            assert NAMES[w] in witnesses


def test_m16t1_effects_are_reported_in_reader_units():
    conflict = next(r for r in event_rows(TEXT) if r[2] == "CONFLICT")
    assert "Marta +5.0" in conflict[5] and "(witness)" in conflict[5] and "arrives week 2" in conflict[5]
    trigger = next(r for r in event_rows(TEXT) if r[2] == "TRIGGER")
    assert "no event crosses the tie" in trigger[5] and trigger[3] == "Ana–Bruno"
    assert "Marta–Ravi tie now distant" in TEXT


def test_m16c1_renderer_is_deterministic():
    assert render(RECORDS, NAMES) == TEXT
    assert render(run_phase_b(7).records, NAMES) == TEXT
    tree = ast.parse(Path(trace.__file__).read_text(encoding="utf-8"))
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not imported & {"anthropic", "openai", "transformers", "langchain", "requests", "random"}


def test_m16c4_single_agent_view():
    """A person's view holds exactly the events they sent, received or witnessed."""
    for pid in (PersonId("sofia"), PersonId("ana"), PersonId("nadia")):
        expected = [
            r.event for r in RECORDS
            if isinstance(r, EmittedRecord)
            and (r.event.sender == pid or pid in r.event.targets or pid in r.event.witnesses)
        ]
        rows = event_rows(render(RECORDS, NAMES, view=pid))
        assert [(int(r[0]), r[2]) for r in rows] == [(e.timestamp, e.kind) for e in expected], pid
    sofia = event_rows(render(RECORDS, NAMES, view=PersonId("sofia")))
    assert [r[2] for r in sofia] == ["PURSUE"]
    assert "Marta" not in sofia[0][5]  # only Sofia's own effect is shown in her view
    ana = render(RECORDS, NAMES, view=PersonId("ana"))
    assert "TRIGGER" not in ana  # a TRIGGER reaches no one, so it is in no one's view
    assert "(system)" not in ana


def test_m16c5_trace_carries_header_and_framing():
    for needle in ("seed | 7", "config hash", "spec revision | 2.0, revision 10", "phase_b_reduced",
                   "constants frozen | 2026-10-06", "M11.F.9", "M11.F.10", "invented", "virtual family"):
        assert needle in TEXT, needle
    assert TEXT.index("What this is") < TEXT.index(TABLE_HEAD)


def test_m16c5_trace_does_not_resemble_a_clinical_record():
    lowered = TEXT.lower()
    for word in ("patient", "diagnosis", "case history", "presenting problem", "treatment plan",
                 "prognosis", "clinical note", "session notes"):
        assert word not in lowered, word


def test_m16c5_a_log_without_its_header_is_refused():
    with pytest.raises(ValueError, match="header"):
        render(RECORDS[1:], NAMES)


# --- review findings (2026-10-06): nothing the renderer does not understand may vanish ---


def test_m16c2_an_unknown_mechanism_raises_instead_of_vanishing():
    from src.bowen.engine.log_records import EffectRecord

    stray = EffectRecord(tick=3, mechanism="sink_allocation", cause=None)
    with pytest.raises(UnrenderableRecord, match="sink_allocation"):
        render(RECORDS[:5] + [stray], NAMES)
    first_event = next(r.event for r in RECORDS if isinstance(r, EmittedRecord))
    caused = EffectRecord(tick=3, mechanism="new_thing", cause=first_event.id)
    with pytest.raises(UnrenderableRecord, match="new_thing"):
        render(RECORDS + [caused], NAMES)


def test_m16c2_policy_selections_and_belief_writes_are_not_silently_skipped():
    from src.bowen.engine.log_records import BeliefWriteRecord, DecidedBy, SelectionRecord

    policy = SelectionRecord(tick=1, actor=PersonId("ravi"), decided_by=DecidedBy.POLICY, event_id=None)
    with pytest.raises(UnrenderableRecord, match="Phase C"):
        render(RECORDS[:3] + [policy], NAMES)
    belief = BeliefWriteRecord(tick=1, holder=PersonId("ravi"), subject="x", value=1.0, true_value=0.0)
    with pytest.raises(UnrenderableRecord, match="Phase D"):
        render(RECORDS[:3] + [belief], NAMES)


def test_m16c2_invariant_sentence_reconciles_with_the_records():
    from src.bowen.engine.log_records import InvariantRecord

    dropped = [r for r in RECORDS if not (isinstance(r, InvariantRecord) and r.tick == 7)]
    text = render(dropped, NAMES)
    assert "recorded for 39 of them" in text and "at the end of every week" not in text


def test_m16c4_a_view_of_no_one_is_refused():
    with pytest.raises(ValueError, match="not in this family"):
        render(RECORDS, NAMES, view=PersonId("nobody"))


def test_m1f6_the_trace_says_a_stressor_acts_once():
    job_loss = next(r for r in event_rows(TEXT) if r[2] == "JOB_LOSS")
    assert "one-time effect; a 34-week spell, recorded" in job_loss[5]
