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
    # Reader units: a signed figure to one decimal. The value itself comes from invented constants.
    assert re.search(r"Marta \+\d+\.\d", conflict[5]) and "(witness)" in conflict[5] and "arrives week 2" in conflict[5]
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
    # Sofia sends nothing and witnesses nothing; she is the target of what reaches her across her ties.
    assert "PURSUE" in [r[2] for r in sofia] and all(r[1] != "Sofia" for r in sofia)
    assert "Marta" not in sofia[0][5]  # only Sofia's own effect is shown in her view
    ana = render(RECORDS, NAMES, view=PersonId("ana"))
    assert "TRIGGER" not in ana  # a TRIGGER reaches no one, so it is in no one's view
    assert "(system)" not in ana


def test_m16c5_trace_carries_header_and_framing():
    for needle in ("seed | 7", "config hash", "spec revision | 2.0, revision 32", "phase_b_reduced",
                   "constants frozen | 2026-10-07", "M11.F.9", "M11.F.10", "invented", "virtual family"):
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

    stray = EffectRecord(tick=3, mechanism="unheard_of_weekly_thing", cause=None)
    with pytest.raises(UnrenderableRecord, match="unheard_of_weekly_thing"):
        render(RECORDS[:5] + [stray], NAMES)
    first_event = next(r.event for r in RECORDS if isinstance(r, EmittedRecord))
    caused = EffectRecord(tick=3, mechanism="new_thing", cause=first_event.id)
    with pytest.raises(UnrenderableRecord, match="new_thing"):
        render(RECORDS + [caused], NAMES)


def test_m16c2_policy_selections_and_belief_writes_are_not_silently_skipped():
    from src.bowen.engine.log_records import BeliefWriteRecord, DecidedBy, SelectionRecord

    # From Phase C step 6 a policy selection renders; one that emitted nothing and is neither
    # a WITHHOLD nor a fallback has no meaning and must raise, not vanish.
    policy = SelectionRecord(tick=1, actor=PersonId("ravi"), decided_by=DecidedBy.POLICY, event_id=None)
    with pytest.raises(UnrenderableRecord, match="emitted nothing"):
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


def test_m1f6_the_trace_says_a_stressor_weighs_for_its_duration():
    """Revision 18: a 34-week job loss weighs for its weeks, and its row says so."""
    job_loss = next(r for r in event_rows(TEXT) if r[2] == "JOB_LOSS")
    assert "a 34-week spell" in job_loss[5] and "still weighing this week" in job_loss[5]


def test_m16c2_withheld_and_fallback_outcomes_get_their_own_rows():
    """M4.D.1b, M4.D.1f: an outcome that emits nothing still appears in the trace."""
    from src.bowen.engine.log_records import DecidedBy, SelectionRecord

    held = SelectionRecord(tick=1, actor=PersonId("ravi"), decided_by=DecidedBy.POLICY, event_id=None,
                           withheld="PURSUE", withheld_toward=PersonId("marta"))
    fallback = SelectionRecord(tick=1, actor=PersonId("bruno"), decided_by=DecidedBy.FALLBACK, event_id=None,
                               fallback_rule="hold")
    text = render(RECORDS[:3] + [held, fallback], NAMES)
    assert "| WITHHOLD | Marta |" in text and "held back PURSUE" in text
    assert "fallback (hold): no legal act" in text


def test_m16c1_phase_c_trace_renders():
    """The Phase C review trace (plan §9) renders a policy run of the Phase C family, and does not carry the Phase B
    framing that no one chose anything."""
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "tools" / "phase_c_trace.py"
    spec = importlib.util.spec_from_file_location("tools.phase_c_trace", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    text = tool.trace_text(7)
    assert "| family instance | phase_c |" in text
    assert "no one in it\n> chose anything" not in text and "selected by the policy" in text
    assert "| 103 |" in text and "| 104 |" not in text  # exactly M11.C.1's 104 weeks (0-103)


def _phase_c_tool():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "tools" / "phase_c_trace.py"
    spec = importlib.util.spec_from_file_location("tools.phase_c_trace", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


_OUTPUTS = {}


def _phase_c_outputs(seed: int = 7) -> tuple[str, str]:
    """The trace and chart of one seed-7 run, rendered once for the chart tests."""
    if seed not in _OUTPUTS:
        _OUTPUTS[seed] = _phase_c_tool().outputs(seed)
    return _OUTPUTS[seed]


def _chart_data(chart: str) -> dict:
    import json

    match = re.search(r"const RUN = (.*?);\n", chart)
    assert match, "the chart carries its data as `const RUN = {...};`"
    return json.loads(match.group(1))


def test_m16c1_chart_series_reads_state_each_week():
    """Each week's sample is a copy of that week's values: a later change to the state does not rewrite an earlier
    week (adversarial: a sampler that kept references would show week 1's values in week 0). A dead person has no
    value that week."""
    from types import SimpleNamespace

    from tools.run_chart import WeeklySeries

    ravi = SimpleNamespace(alive=True, acute_anxiety=50.0, functional_level=40.0, chronic_anxiety=44.0, basic_level=39.0)
    ana = SimpleNamespace(alive=True, acute_anxiety=60.0, functional_level=35.0, chronic_anxiety=38.0, basic_level=37.0)
    state = SimpleNamespace(people={PersonId("ravi"): ravi, PersonId("ana"): ana})
    series = WeeklySeries()
    series.observe(state)
    ravi.acute_anxiety, ravi.functional_level = 70.0, 30.0
    ana.alive = False
    series.observe(state)
    assert series.weeks[0]["ravi"] == {"anxiety": 50.0, "functioning": 40.0, "chronic": 44.0, "basic": 39.0}
    assert series.weeks[1]["ravi"]["anxiety"] == 70.0 and series.weeks[1]["ravi"]["functioning"] == 30.0
    assert "ana" in series.weeks[0] and "ana" not in series.weeks[1]


def test_m16c1_chart_marks_outside_events_on_their_targets():
    """The chart's event marks are the trace's "(from outside)" rows of the same run, person by person and week by
    week — two surfaces of one run that must agree — and no event a family member sent is marked."""
    trace, chart = _phase_c_outputs()
    names = {p["name"]: p for p in _chart_data(chart)["people"]}
    from_trace = {}
    for week, target in re.findall(r"^\| (\d+) \| \(from outside\) \| [A-Z_]+ \| ([^|]+?) \|", trace, re.M):
        from_trace.setdefault(target, []).append(int(week))
    assert from_trace, "the seed-7 Phase C run has events from outside (the declared spell, M11.3)"
    from_chart = {name: [week for week, _ in p["events"]] for name, p in names.items() if p["events"]}
    assert from_chart == from_trace


def test_m16c1_committed_trace_and_chart_are_the_current_render():
    """The seed-7 trace and chart under docs/review/ are what tools/phase_c_trace.py renders now (P6: a committed
    output is checked against the thing it claims to show). An engine change turns this red until they are
    regenerated: python3 tools/phase_c_trace.py."""
    trace, chart = _phase_c_outputs()
    review = Path(__file__).resolve().parents[2] / "docs" / "review"
    assert (review / "phase_c_trace_seed7.md").read_text(encoding="utf-8") == trace, "stale: rerun tools/phase_c_trace.py"
    assert (review / "phase_c_chart_seed7.html").read_text(encoding="utf-8") == chart, "stale: rerun tools/phase_c_trace.py"


def test_m16c1_chart_names_the_absent_and_draws_the_undeclared():
    """P2: nobody the run held is dropped without a word. A declared person alive in no sampled week is named on the
    page as not drawn; a sampled person the family file does not declare (one born during the run) is drawn under
    their id. Adversarial: the run is real; only the samples are edited, so the rest of the page is unchanged."""
    import src.bowen.ensemble.criteria as criteria
    from tools.run_chart import WeeklySeries, render_chart

    family = load_family(criteria.FAMILIES["phase_c"])
    series = WeeklySeries()
    run = _phase_c_tool().records(7, after_tick=series.observe)
    gone = next(iter(family.display_names))
    for week in series.weeks:
        del week[str(gone)]
        week["newborn"] = dict(week[next(iter(week))])
    data = _chart_data(render_chart(run, series, family))
    assert data["absent"] == [family.display_names[gone]]
    assert family.display_names[gone] not in [p["name"] for p in data["people"]]
    newborn = next(p for p in data["people"] if p["id"] == "newborn")
    assert newborn["group"] == "Not in the family file" and len(newborn["anxiety"]) == 104
    assert data["people"][-1]["id"] == "newborn"  # drawn after the declared people


def test_m16c1_chart_script_json_cannot_end_its_element():
    """A display name holding markup cannot end the data's <script> element or open a comment in it: no <, > or &
    survives, and the JSON still reads back to the same value. The fill is one pass, so a value that contains a
    placeholder's name is not filled again, and a template missing a placeholder is refused."""
    import json

    from tools.run_chart import _fill, _script_json

    value = {"name": "</script><!--<script>&amp;"}
    text = _script_json(value)
    assert not set("<>&") & set(text) and json.loads(text) == value
    assert _fill("a __TITLE__ b __FRAMING__ c __DATA__", {"TITLE": "__DATA__", "FRAMING": "f", "DATA": "d"}) == \
        "a __DATA__ b f c d"
    with pytest.raises(ValueError, match="placeholders"):
        _fill("a __TITLE__ __DATA__", {"TITLE": "t", "FRAMING": "f", "DATA": "d"})


def test_m16c1_phase_c_chart_renders():
    """The chart is written from the same run as the trace: one panel per person, a value for every one of
    M11.C.1's 104 weeks, the trace's own M11.F framing, every placeholder filled, and the trace it sits beside
    unchanged by the sampling (the sampler only observes)."""
    trace, chart = _phase_c_outputs()
    assert trace == _phase_c_tool().trace_text(7)
    data = _chart_data(chart)
    assert data["seed"] == 7 and data["instance"] == "phase_c" and data["weeks"] == 104
    people = data["people"]
    assert sorted(p["name"] for p in people) == sorted(load_family_phase_c().display_names.values())
    for p in people:
        for key in ("anxiety", "functioning", "chronic", "basic"):
            assert len(p[key]) == 104 and all(v is not None for v in p[key]), (p["name"], key)
    assert max(v for p in people for v in p["anxiety"]) <= data["ymax"]
    assert "It does not concern any real family or person, and nothing here is advice (M11.F.9)" in chart
    assert "M11.F.10" in chart and "selected by the policy" in chart
    assert "__TITLE__" not in chart and "__FRAMING__" not in chart and "__DATA__" not in chart


def load_family_phase_c():
    import src.bowen.ensemble.criteria as criteria

    return load_family(criteria.FAMILIES["phase_c"])


def test_m16c1_chart_a_name_cannot_close_the_script():
    """Adversarial: a display name holding "</script>" must stay inside the data, or the page's script ends early."""
    import dataclasses

    from tools.run_chart import WeeklySeries, render_chart

    family = load_family_phase_c()
    hostile = dict(family.display_names)
    first = next(iter(hostile))
    hostile[first] = "</script><b>x"
    series = WeeklySeries()
    series.weeks = [{str(first): {"anxiety": 1.0, "functioning": 2.0, "chronic": 0.5, "basic": 2.0}}]
    chart = render_chart(RECORDS[:1], series, dataclasses.replace(family, display_names=hostile))
    assert chart.count("</script>") == 1
    assert _chart_data(chart)["people"][0]["name"] == "</script><b>x"


def test_m16c1_chart_script_parses(tmp_path):
    """The page's script is valid JavaScript (global rule: syntax-check embedded code)."""
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the script's syntax is not checked here")
    _, chart = _phase_c_outputs()
    script = tmp_path / "chart.js"
    script.write_text(re.search(r"<script>(.*)</script>", chart, re.S).group(1), encoding="utf-8")
    result = subprocess.run([node, "--check", str(script)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_b2_framing_says_who_chose_the_acts():
    """The sentence is read from the run's own selections: any selection not SCRIPTED means the people chose (Phase D
    plan B2; review 2026-10-10 found FALLBACK missed); a run with only scripted selections, or none, is scripted."""
    from src.bowen.engine.identifiers import PersonId
    from src.bowen.engine.log_records import DecidedBy, SelectionRecord
    from src.bowen.render.trace import POLICY, SCRIPTED, framing

    def selection(by):
        return SelectionRecord(tick=0, actor=PersonId("ravi"), decided_by=by, event_id=None)

    assert POLICY in framing([selection(DecidedBy.FALLBACK)]) and POLICY in framing([selection(DecidedBy.POLICY)])
    assert SCRIPTED in framing([]) and SCRIPTED in framing([selection(DecidedBy.SCRIPTED)])
    assert POLICY in framing([selection(DecidedBy.SCRIPTED), selection(DecidedBy.SEQUENCE)])


def test_m16c2_a_held_position_renders():
    """Revision 21 (review 2026-10-12): the seed-7 trace emits no position_held, so its sentence is pinned here."""
    from src.bowen.engine.log_records import EffectRecord

    first_event = next(r.event for r in RECORDS if isinstance(r, EmittedRecord))
    target = first_event.targets[0]
    held = EffectRecord(tick=first_event.timestamp, mechanism="position_held", cause=first_event.id,
                        people=((target, "position_held", 1.0),))
    text = render(RECORDS + [held], NAMES)
    assert f"{NAMES.get(target, target.value)} holds a position: no functioning changes hands" in text


def test_m16c2_lock_in_renders():
    """Revision 24 (M7.D.2a, M7.D.2d): a remission's return of a symptom's hold is shown beside the event; the weekly
    hold, giving back and spill are summarised once, not listed (review 2026-10-12)."""
    from src.bowen.engine.log_records import EffectRecord

    marta, ravi = PersonId("marta"), PersonId("ravi")
    first = next(r.event for r in RECORDS if isinstance(r, EmittedRecord))
    sentences = {1.0: "Marta's symptom holds 1.0", -1.0: "Marta's symptom gives back 1.0", 0.0: "past its turn"}
    for change, needle in sentences.items():
        caused = EffectRecord(first.timestamp, "lock_in", first.id, acute_anxiety=((ravi, -change or 2.0),),
                              people=((marta, "symptom_bound", change),))
        assert needle in render(RECORDS + [caused], NAMES), needle
    weekly = EffectRecord(first.timestamp, "lock_in", None, acute_anxiety=((ravi, -1.0),),
                          people=((marta, "symptom_bound", 1.0),))
    text = render(RECORDS + [weekly], NAMES)
    assert "Marta's symptom holds" not in text and "lock-in" in text


def test_m16c2_a_remitted_symptom_renders_as_ending():
    """Revision 24 (re-sweep 2026-10-12): a symptom channel's release credits nothing to the budget, so its row says the
    symptom ends rather than "0.0 returned to the family budget"."""
    from src.bowen.engine.log_records import EffectRecord

    first = next(r.event for r in RECORDS if isinstance(r, EmittedRecord))
    zero = EffectRecord(first.timestamp, "binder_unavailable", first.id, sinks=(("undifferentiation_budget", 0.0),))
    text = render(RECORDS + [zero], NAMES)
    assert "the symptom ends" in text and "0.0 returned to the family budget" not in text
