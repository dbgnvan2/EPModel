"""The run visualization: one run exported as a fam-diag diagram (plan V5).

Purpose: check that every pattern line traces back to the run's own acts, that nothing is drawn from a single
         act, that the file has what fam-diag's import needs, and that it says it is not a real family.
Spec:    docs/PROPOSAL — RUN VISUALIZATION MAPPING.md §8; docs/bowen_agent_model_spec_v2.md#M5.A.1a, #M11.F.9
Tests:   this file

fam-diag is a separate repository, so its line styles are copied here as the contract
(fam-diag ``src/frontend/src/types/index.ts``, ``EmotionalLine``). Whether fam-diag draws the file correctly
is checked by opening it, not here (plan V7).
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

from src.bowen.engine.events import Mechanism
from src.bowen.engine.identifiers import PersonId, TieId
from src.bowen.engine.log_records import EmittedRecord
from src.bowen.engine.objects import TieState
from src.bowen.io.load import load_constants, load_family
from src.bowen.readouts.patterns import APPROACH, PATTERNS, Reading, ReadoutParams

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("tools.famdiag_export", REPO / "tools" / "famdiag_export.py")
tool = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("tools.famdiag_export", tool)  # its dataclasses resolve their module by name
_spec.loader.exec_module(tool)

RP = ReadoutParams.from_constants(load_constants())
FAMDIAG_STYLES = {
    "fusion": {"fusion-dotted-wide", "fusion-dotted-tight", "fusion-solid-wide", "fusion-solid-tight", "fusion-triple"},
    "projection": {f"projection-{i}" for i in range(1, 6)},
    "distance": {"distance-dotted-wide", "distance-dotted-tight", "distance-dashed-wide", "distance-dashed-tight",
                 "distance-long"},
    "cutoff": {"cutoff"},
    "conflict": {"conflict-dotted-wide", "conflict-dotted-tight", "conflict-solid-wide", "conflict-solid-tight",
                 "conflict-double"},
}
FAMDIAG_ENDINGS = {"none", "arrow-p1-to-p2", "arrow-p2-to-p1", "arrow-bidirectional"}
SUPPORTING = {"conflict": {"CONFLICT"}, "fusion": {"PURSUE"},
              "over_underfunctioning": {"OVERFUNCTION", "UNDERFUNCTION", "PURSUE"},
              "projection": {"OVERFUNCTION", "PURSUE", "TRIANGLE"}, "distance": {"DISTANCE"},
              # Revision 17: cutoff is read from the other's refused approaches, not from CUTOFF acts (words). Found
              # 2026-10-12 when seed 3's run first read a cutoff whose block held no CUTOFF act.
              "cutoff": set(APPROACH)}


@pytest.fixture(scope="module")
def mapping():
    return tool.load_mapping()


@pytest.fixture(scope="module")
def run(mapping):
    return tool.simulate(7, mapping)


@pytest.fixture(scope="module")
def diagram(run, mapping):
    return tool.build_diagram(run, mapping, 7)


def acts_on(run, tie, block, mapping):
    """The run's acts between the tie's two people in one block, read straight from the records."""
    a, b = tie.members()
    lo, hi = block * mapping.block_weeks, (block + 1) * mapping.block_weeks
    return [r.event for r in run.records if isinstance(r, EmittedRecord) and r.event.mechanism is Mechanism.MOVE
            and r.event.sender in (a, b)
            and ({a, b} - {r.event.sender}) <= set(r.event.targets) and lo <= r.event.timestamp < hi]


def test_v5_mapping_covers_every_declared_pattern(mapping):
    assert set(mapping.patterns) == {p for p, kind in PATTERNS.items() if kind == "tie"}
    for pattern, (relationship_type, styles, _ending) in mapping.patterns.items():
        assert set(styles) <= FAMDIAG_STYLES[relationship_type], pattern
    assert mapping.block_weeks == RP.window  # a block is one readout window (owner D6)


def test_v5_every_line_traces_back_to_its_reading(run, diagram, mapping):
    """Read from the records, not the readouts: each line's first block holds what its pattern needs."""
    assert_traces_back(run, diagram, mapping)


@pytest.mark.parametrize("seed", [1, 3])
def test_v5_every_line_traces_back_on_other_seeds(seed, mapping):
    """Seeds 1 and 3 have distance lines read from acts and from silence in turn (review 2026-10-10)."""
    other = tool.simulate(seed, mapping)
    assert_traces_back(other, tool.build_diagram(other, mapping, seed), mapping)


def assert_traces_back(run, diagram, mapping):
    family = load_family(REPO / "config" / "bowen" / "family_phase_c.md")
    for line in diagram["emotionalLines"]:
        pattern = line["id"].split("-")[1]
        a, b = (PersonId(line[k][2:]) for k in ("person1_id", "person2_id"))
        tie = TieId.of(a, b)
        block = int(line["startDate"][:4]) - mapping.first_year
        acts = acts_on(run, tie, block, mapping)
        if pattern == "cutoff" and block == 0 and family.ties[tie].tie_state is TieState.CUT_OFF:
            continue  # declared cut off in the family file; from B4 the tie is open, so it may carry acts
        if pattern == "distance" and "first read from silence" in line["notes"]:
            assert acts == [], line["id"]  # silence is distance (owner D2)
            continue
        supporting = [e for e in acts if e.kind in SUPPORTING[pattern]]
        assert len(supporting) >= RP.min_acts, (line["id"], Counter(e.kind for e in acts))


def _blocks_of_conflict(per_block: int, rp, blocks: int = 3):
    """Ravi and Marta exchange ``per_block`` CONFLICT acts in each of ``blocks`` blocks, read as the exporter reads."""
    from src.bowen.engine.events import Channel, Event, EventId, SourcePosition
    from src.bowen.engine.initialise import initialise_run
    from src.bowen.engine.state import new_run_state
    from src.bowen.io.load import load_event_kinds
    from src.bowen.readouts.patterns import current_patterns, tie_readings
    from src.bowen.scenario.params import engine_params

    params = engine_params(load_constants())
    fam = load_family()
    state = initialise_run(new_run_state(dict(fam.people), dict(fam.ties), fam.family, load_event_kinds()), params)
    ravi, marta = PersonId("ravi"), PersonId("marta")
    held, out = {}, []
    for block in range(blocks):
        for i in range(per_block):
            sender, target = (ravi, marta) if i % 2 == 0 else (marta, ravi)
            week = block * rp.window + i
            state.store.record_event(Event(
                id=EventId(week, sender.value, i), kind="CONFLICT", mechanism=Mechanism.MOVE, sender=sender,
                targets=(target,), intensity=50.0, timestamp=week, duration=1, exogenous=False,
                source_position=SourcePosition.NONE, channel=Channel.SCRIPTED))
        state.tick = (block + 1) * rp.window - 1
        held = current_patterns(held, tie_readings(state, TieId.of(ravi, marta), params, rp))
        out.append(dict(held))
    return out


def test_v5_no_line_from_two_acts_a_block(mapping):
    """Adversarial: two CONFLICT acts in every block draw no conflict line; three do. The mutant with a threshold
    of 2 draws one, so the test can fail (review 2026-10-10: the earlier version's precondition never held)."""
    import dataclasses

    def conflict_lines(held):
        run = tool.Run(records=_header_only(), held=held, weeks=len(held) * mapping.block_weeks)
        return [l for l in tool.build_diagram(run, mapping, 0)["emotionalLines"] if l["relationshipType"] == "conflict"]

    assert conflict_lines(_blocks_of_conflict(2, RP)) == []
    assert len(conflict_lines(_blocks_of_conflict(3, RP))) == 1
    assert len(conflict_lines(_blocks_of_conflict(2, dataclasses.replace(RP, min_acts=2)))) == 1


def test_v5_a_line_read_from_acts_and_silence_reports_both(mapping):
    """Review 2026-10-10: weeks of silence were banded and labelled as acts. Each is now on its own scale."""
    tie = TieId.of(PersonId("marta"), PersonId("pia"))
    held = [{tie: Reading("distance", tie, 3, 0)}] + [
        {tie: Reading("distance", tie, 4 * k, 0, silent=True)} for k in range(1, 15)]
    run = tool.Run(records=_header_only(), held=held, weeks=len(held) * mapping.block_weeks)
    [line] = tool.build_diagram(run, mapping, 0)["emotionalLines"]
    assert "first read from acts" in line["notes"]
    assert "peak 3 acts in a block" in line["notes"] and "peak 56 weeks without interaction" in line["notes"]
    assert line["lineStyle"] == "distance-dotted-tight"  # 56 silent weeks: level 4; 3 acts alone would be level 1


def test_v5_simulate_is_a_pure_observer(run, mapping):
    """The export's weekly reading must not change the run (M16.B's rule, applied to the tool)."""
    from tools.phase_c_trace import records

    assert run.records == records(7)
    again = tool.simulate(7, mapping)
    assert again.held == run.held and again.records == run.records


def test_v5_projection_reads_the_sinks_act_set():
    """patterns.PROJECTION_ACTS is the child-projection sink's act set; they must not drift apart."""
    from src.bowen.engine import sinks
    from src.bowen.readouts.patterns import PROJECTION_ACTS

    assert PROJECTION_ACTS == {sinks.OVERFUNCTION, sinks.PURSUE, sinks.TRIANGLE}


def test_v5_silent_severed_tie_draws_distance_not_cutoff(mapping):
    """Owner D3: no interaction is distance, so a silent tie draws a distance line, never a cutoff line."""
    tie = TieId.of(PersonId("marta"), PersonId("ravi"))
    held = [{tie: Reading("distance", tie, 4 * (b + 1), 0, silent=True)} for b in range(3)]
    run = tool.Run(records=_header_only(), held=held, weeks=12)
    lines = tool.build_diagram(run, mapping, 0)["emotionalLines"]
    assert [l["relationshipType"] for l in lines] == ["distance"]
    assert lines[0]["lineStyle"] == "distance-dashed-wide"  # 12 silent weeks: level 2 of silent_weeks


def test_v5_episodes_split_where_the_pattern_changes():
    tie = TieId.of(PersonId("marta"), PersonId("ravi"))
    c, d = Reading("conflict", tie, 3, 0), Reading("distance", tie, 3, 4)
    over_m, over_r = (Reading("over_underfunctioning", tie, 3, 8, toward=PersonId(p)) for p in ("marta", "ravi"))
    eps = tool.episodes([{tie: c}, {tie: c}, {tie: d}, {tie: over_m}, {tie: over_r}])
    assert [(e[1], e[2], e[3][0].pattern) for e in eps] == [
        (0, 1, "conflict"), (2, 2, "distance"), (3, 3, "over_underfunctioning"), (4, 4, "over_underfunctioning")]


def test_v5_a_stopped_pattern_ends_where_it_stopped():
    """A projection can stop with nothing after it (owner, 2026-10-10); its line ends in its last block, not at the
    next pattern read on that tie (found when the hold was added)."""
    tie = TieId.of(PersonId("ravi"), PersonId("sofia"))
    proj, conf = Reading("projection", tie, 3, 0, toward=PersonId("sofia")), Reading("conflict", tie, 3, 20)
    eps = tool.episodes([{tie: proj}, {tie: proj}, {}, {}, {tie: conf}])
    assert [(e[1], e[2], e[3][0].pattern) for e in eps] == [(0, 1, "projection"), (4, 4, "conflict")]


def test_v5_diagram_has_what_famdiag_imports(run, diagram, mapping):
    assert all(isinstance(diagram[k], list) for k in ("people", "partnerships", "emotionalLines"))
    people = {p["id"] for p in diagram["people"]}
    unions = {u["id"] for u in diagram["partnerships"]}
    for u in diagram["partnerships"]:
        assert {u["partner1_id"], u["partner2_id"], *u["children"]} <= people
    for p in diagram["people"]:
        assert set(p["partnerships"]) <= unions and p.get("parentPartnership", next(iter(unions))) in unions
    years = []
    for line in diagram["emotionalLines"]:
        assert {line["person1_id"], line["person2_id"]} <= people
        assert line["lineStyle"] in FAMDIAG_STYLES[line["relationshipType"]] and line["lineEnding"] in FAMDIAG_ENDINGS
        assert line.get("adequatePersonId", line["person1_id"]) in (line["person1_id"], line["person2_id"])
        years += [dt.date.fromisoformat(line[k]).year for k in ("startDate", "endDate") if k in line]
        years += [dt.date.fromisoformat(e["date"]).year for e in line["events"]]
        assert line["hideAfterEnd"] is True  # fam-diag's opt-in: without it an ended line never hides
        if "endDate" in line:  # fam-diag hides a line after its end year: a successor must not share it
            assert line["endDate"].endswith("-12-31")
    for tri in diagram["triangles"]:
        assert {tri["person1_id"], tri["person2_id"], tri["person3_id"]} <= people
        # fam-diag adds undated triangle lines when none are given, and those show at every year
        assert len(tri["tpls"]) == 3
        for tpl in tri["tpls"]:
            assert (tpl.get("startDate"), tpl.get("endDate")) == (tri["startDate"], tri.get("endDate"))
    for p in diagram["people"]:
        years += [dt.date.fromisoformat(e["date"]).year for e in p["events"]]
    blocks = run.weeks // mapping.block_weeks
    assert mapping.first_year <= min(years) and max(years) <= mapping.first_year + blocks


def test_v5_only_acts_sit_on_pattern_lines(diagram):
    """A symptom onset or a stressor is a person's event, never a tie's (found by the trace-back test)."""
    kinds = {e["subtype"] for line in diagram["emotionalLines"] for e in line["events"]}
    assert not kinds & {"SYMPTOM_ONSET", "JOB_LOSS"}


def test_v5_events_carry_what_famdiag_requires(diagram):
    """fam-diag CLAUDE.md: every event sets date and startDate, anchorType and anchorId, eventClass, createdAt, subtype."""
    for key in ("people", "emotionalLines"):
        for owner in diagram[key]:
            for e in owner["events"]:
                assert e["date"] == e["startDate"] and e["anchorId"] == owner["id"] and e["subtype"]
                assert e["anchorType"] and e["eventClass"] and isinstance(e["createdAt"], int)


def test_v5_every_act_is_an_event_once(run, diagram):
    emitted = [r for r in run.records if isinstance(r, EmittedRecord)]
    placed = [e for key in ("people", "emotionalLines") for x in diagram[key] for e in x["events"]]
    assert len(placed) == len(emitted) and len({e["id"] for e in placed}) == len(placed)


def test_v5_says_it_is_not_a_real_family(diagram):
    """M11.F.9: the model is never presented as speaking about a real family."""
    note = diagram["pageNotes"][0]["text"]
    assert "not a real family" in note and "M11.F.9" in note


def test_v5_greyed_people_are_those_the_run_does_not_simulate(diagram):
    run_people = {f"p-{p.value}" for p in load_family(REPO / "config" / "bowen" / "family_phase_c.md").people}
    greyed = {p["id"] for p in diagram["people"] if p.get("foregroundColor") == tool.GREY}
    assert greyed == {p["id"] for p in diagram["people"]} - run_people
    assert greyed == {"p-teodor", "p-viktor", "p-iris", "p-dario", "p-leo"}


def test_v5_export_is_deterministic(run, mapping, diagram):
    assert tool.build_diagram(run, mapping, 7) == diagram


# Opening the export in fam-diag saves it back (autosave). The keys the app changes or adds are the owner's; everything
# else in the file must be what the tool writes now, no more and no less. The sets are written out here, not taken from
# the tool, so a key added to the tool's sets (one the run draws, say) shows here as a failure.
APP_OWNED_KEYS = frozenset({
    "x", "y", "horizontalConnectorY", "displayName", "width", "height", "size", "notesPosition", "notesSize",
    "familyNotesPosition", "familyNotesSize", "familyNameOffsetX", "familyNameOffsetY", "familyNameWidth",
    "familyNameHeight", "connectionAnchorX", "birthOrderOverride", "siblingPositionOverride", "fatherPositionOverride",
    "motherPositionOverride", "partnerPositionOverride",
})
APP_ADDED_KEYS = frozenset({
    "color", "fillColor", "backgroundColor", "backgroundEnabled", "exportedAt", "autoSaveMinutes", "ideasText",
    "eventCategories", "nodalCategories", "predictionSets", "functionalFactCategories",
    "functionalIndicatorDefinitions", "relationshipStatuses", "relationshipTypes",
})


def test_v5_app_key_sets_are_the_declared_ones():
    assert (tool.APP_OWNED_KEYS, tool.APP_ADDED_KEYS) == (APP_OWNED_KEYS, APP_ADDED_KEYS)


def _what_the_tool_drew(saved, rendered):
    """``saved`` cut to the shape of ``rendered``: app-owned keys skipped, app-added keys dropped. Any other key the
    tool does not write (one it stopped writing) is kept, so it makes the comparison fail."""
    if isinstance(rendered, dict) and isinstance(saved, dict):
        drawn = {k: _what_the_tool_drew(saved.get(k), v) for k, v in rendered.items() if k not in APP_OWNED_KEYS}
        extra = {k: saved[k] for k in saved.keys() - rendered.keys() - APP_ADDED_KEYS - APP_OWNED_KEYS}
        return drawn | extra
    if isinstance(rendered, list) and isinstance(saved, list) and len(saved) == len(rendered):
        return [_what_the_tool_drew(s, r) for s, r in zip(saved, rendered)]
    return saved


def test_v5_committed_export_is_the_current_render(diagram):
    """P6: the seed-7 export under docs/review/ still holds what tools/famdiag_export.py renders now. Stage 1's
    regeneration refreshed the trace and chart but not this file, which kept naming spec revision 32 and an old
    config hash. A change to the run, the readouts or the spec revision turns this red until it is regenerated:
    python3 tools/famdiag_export.py, which keeps what the app saved (tool.keep_saved). A save from fam-diag
    (APP_OWNED_KEYS, APP_ADDED_KEYS) does not turn it red."""
    import json

    saved = json.loads((REPO / "docs" / "review" / "phase_c_famdiag_seed7.json").read_text(encoding="utf-8"))
    expected = _what_the_tool_drew(diagram, diagram)
    assert _what_the_tool_drew(saved, diagram) == expected, "stale: rerun tools/famdiag_export.py"


def _saved_from_the_app(diagram):
    """The export as the app might save it: Sofia and the note moved, a couple's connector moved, entries reordered,
    a colour and a settings key added, an entry and a key the tool does not write."""
    import copy

    saved = copy.deepcopy(diagram)
    next(p for p in saved["people"] if p["id"] == "p-sofia").update(x=648, y=86, name="edited in the app")
    saved["partnerships"][0]["horizontalConnectorY"] = 158
    saved["pageNotes"][0].update(x=922, y=15, text="an old note")
    saved["people"].reverse()
    saved["people"].append({"id": "p-nobody", "x": 1, "y": 1})
    saved["emotionalLines"][0]["color"] = "#123456"
    saved["functionalFactCategories"] = [{"id": "ff-1", "name": "Anxiety Level"}]
    saved["fileMeta"]["displayName"] = "phase_c_famdiag_seed7.json"
    next(p for p in saved["people"] if p["id"] == "p-ana")["notesPosition"] = {"x": 5, "y": 6}
    dario = next(p for p in saved["people"] if p["id"] == "p-dario")
    dario["x"] = int(dario["x"])  # the app saves 300 where the tool draws 300.0
    saved["staleKey"] = 1
    return saved


def test_v5_regeneration_keeps_saved_positions(diagram):
    """A rerun keeps each position saved from the app for the same id (a person, a couple's connector, the note),
    whatever order the saved entries are in, and every key the app added; it draws everything else, including the
    text, from the run. A saved id the run does not draw, and a key the tool does not write, are dropped and
    reported."""
    kept, report = tool.keep_saved(diagram, _saved_from_the_app(diagram))

    person = next(p for p in kept["people"] if p["id"] == "p-sofia")
    assert (person["x"], person["y"]) == (648, 86)
    assert person["name"] == next(p for p in diagram["people"] if p["id"] == "p-sofia")["name"]
    assert kept["partnerships"][0]["horizontalConnectorY"] == 158
    assert (kept["pageNotes"][0]["x"], kept["pageNotes"][0]["y"]) == (922, 15)
    assert kept["pageNotes"][0]["text"] == diagram["pageNotes"][0]["text"]
    assert [p["id"] for p in kept["people"]] == [p["id"] for p in diagram["people"]]
    others = [p for p in kept["people"] if p["id"] not in ("p-sofia", "p-ana")]
    assert others == [p for p in diagram["people"] if p["id"] not in ("p-sofia", "p-ana")]  # 300 == 300.0
    assert kept["emotionalLines"][0]["color"] == "#123456"
    assert kept["functionalFactCategories"] == [{"id": "ff-1", "name": "Anxiety Level"}]
    assert kept["fileMeta"]["displayName"] == "phase_c_famdiag_seed7.json"
    assert "staleKey" not in kept
    assert ".people[p-sofia].x = 648" in "\n".join(report)
    assert "dropped .people[p-nobody]: the run does not draw it" in report
    assert "dropped .staleKey: the tool does not write it" in report
    assert next(p for p in kept["people"] if p["id"] == "p-ana")["notesPosition"] == {"x": 5, "y": 6}
    assert "kept the app's own values: color ×1, functionalFactCategories ×1" in report
    assert not any("p-dario" in line for line in report)
    assert type(next(p for p in kept["people"] if p["id"] == "p-dario")["x"]) is int
    assert tool.keep_saved(diagram, {}) == (diagram, [])


def test_v5_write_export_keeps_and_reports(diagram, tmp_path):
    """write_export keeps what the app saved in the file it overwrites and returns the report; --from-table places
    everything by the layout again and still keeps the app's own keys."""
    import json

    out = tmp_path / "export.json"
    assert tool.write_export(diagram, out) == []
    assert json.loads(out.read_text(encoding="utf-8")) == diagram

    out.write_text(json.dumps(_saved_from_the_app(diagram)), encoding="utf-8")
    report = tool.write_export(diagram, out)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert (next(p for p in written["people"] if p["id"] == "p-sofia")["x"]) == 648
    assert any(line.startswith("kept .people[p-sofia].x") for line in report)

    out.write_text(json.dumps(_saved_from_the_app(diagram)), encoding="utf-8")
    report = tool.write_export(diagram, out, from_table=True)
    written = json.loads(out.read_text(encoding="utf-8"))
    sofia = next(p for p in diagram["people"] if p["id"] == "p-sofia")
    assert next(p for p in written["people"] if p["id"] == "p-sofia")["x"] == sofia["x"]
    assert written["fileMeta"]["displayName"] == diagram["fileMeta"]["displayName"]
    assert written["functionalFactCategories"] == [{"id": "ff-1", "name": "Anxiety Level"}]
    assert "notesPosition" not in next(p for p in written["people"] if p["id"] == "p-ana")
    assert not any(line.startswith("kept .") for line in report)
    assert any(line.startswith("discarded .people[p-sofia].x = 648") for line in report)
    assert any(line.startswith("discarded .people[p-ana].notesPosition") for line in report)


def test_v5_keep_saved_reports_malformed_saves(diagram):
    """A saved value of another shape, a second entry with the same id, and an id that is not a string or number are
    dropped and reported, not merged and not raised."""
    import copy

    saved = copy.deepcopy(diagram)
    saved["pageNotes"] = {"not": "a list"}
    saved["people"].append(copy.deepcopy(saved["people"][0]) | {"x": 999})
    saved["triangles"].append({"id": ["a", "list"], "x": 1})
    kept, report = tool.keep_saved(diagram, saved)
    assert kept["pageNotes"] == diagram["pageNotes"]
    assert "dropped .pageNotes: saved as dict, drawn as list" in report
    assert f"dropped .people[{diagram['people'][0]['id']}]: a second saved entry with this id" in report
    assert kept["people"][0]["x"] == diagram["people"][0]["x"]
    assert kept["triangles"] == diagram["triangles"]


def test_v5_main_from_table_redraws(diagram, tmp_path, capsys):
    """The command line: a plain run keeps a position saved in the file it overwrites; --from-table redraws it."""
    import json

    out = tmp_path / "phase_c_famdiag_seed7.json"
    out.write_text(json.dumps(_saved_from_the_app(diagram)), encoding="utf-8")
    sofia_x = next(p for p in diagram["people"] if p["id"] == "p-sofia")["x"]
    assert tool.main([], out_dir=tmp_path) == 0
    assert next(p for p in json.loads(out.read_text())["people"] if p["id"] == "p-sofia")["x"] == 648
    out.write_text(json.dumps(_saved_from_the_app(diagram)), encoding="utf-8")
    assert tool.main(["--from-table"], out_dir=tmp_path) == 0
    assert next(p for p in json.loads(out.read_text())["people"] if p["id"] == "p-sofia")["x"] == sofia_x
    assert "discarded .people[p-sofia].x" in capsys.readouterr().out


def _header_only():
    from src.bowen.engine.log_records import LogHeader

    return [LogHeader(seed=0, config_hash="0" * 64, spec_revision="test", instance_id="phase_c",
                      activation_component="", activation_version="", visibility_component="", visibility_version="",
                      activation_regime="", constants_frozen_at=None, constant_changed_after_freeze=())]
