"""Export one run as a fam-diag diagram — the emotional patterns it produced, block by block.

Purpose: draw a run in the owner's family-diagram app, so it can be reviewed as a picture
         (``docs/PROPOSAL — RUN VISUALIZATION MAPPING.md``, plan V5).
Spec:    docs/bowen_agent_model_spec_v2.md#M5.A.1a (the readouts drawn), #M11.F.9 (not a real family)
Tests:   tests/bowen/test_famdiag_export.py

    python3 tools/famdiag_export.py [--seed 7]

Writes ``docs/review/phase_c_famdiag_seed<seed>.json``. Open it with File > Open in **EPModel Fam-Diag**
(``/Users/davemini2/ProjectsLocal/epmodel-fam-diag``, http://localhost:5174). That app is a copy of fam-diag whose
timeline hides a line after its end year when the line sets ``hideAfterEnd``. The owner's fam-diag is not changed.

What it draws, and from where:
- **People and relationship lines** from the run's family file, plus the reference family
  (``config/bowen/family_phase_d.md``). Anyone the run does not simulate is drawn greyed and says so.
- **Pattern lines** from the declared readouts (``src/bowen/readouts/patterns.py``). They are read at the end of each
  block, and ``current_patterns`` decides which pattern each tie holds. A line starts in the block its pattern
  emerges and ends in the block a new pattern replaces it. This tool defines no pattern of its own.
- **Events** are the run's acts, dated within their block, on the pattern line they fall under, or on the
  sender otherwise.
- **How each is drawn** is ``config/viz/famdiag.md``.

The run is ``M11.C.1``'s baseline arm, the same run as ``docs/review/phase_c_trace_seed7.md``
(``tools/phase_c_trace.py``), so the picture can be checked against the trace.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MAPPING = REPO / "config" / "viz" / "famdiag.md"
REFERENCE_FAMILY = REPO / "config" / "bowen" / "family_phase_d.md"
SECTIONS = {
    "Patterns": ("pattern", "relationship_type", "styles", "ending"),
    "Levels": ("measure", "level_1", "level_2", "level_3", "level_4", "level_5"),
    "Triangle intensity": ("intensity", "min_blocks"),
    "Events": ("kind", "event_type", "category", "event_class"),
    "Symptom channels": ("channel", "category"),
    "Layout": ("id", "x", "y"),
}
GREY = "#9e9e9e"
NOTE = ("Simulated run — not a real family (spec M11.F.9). The people are the model's invented reference family. "
        "Each year on the timeline is one block of {weeks} weeks: {first} is weeks 1–{weeks}, {last} is weeks "
        "{last_start}–{total}. Lines are the model's declared pattern readouts (docs/readouts_phase_c.md); a line "
        "ends when a new pattern emerges on that tie. Greyed people are not simulated in this run. "
        "Run: {arm}, seed {seed}, spec {spec}, config hash {config}.")


@dataclass(frozen=True)
class Mapping:
    block_weeks: int
    first_year: int
    patterns: dict
    levels: dict
    triangle_intensity: list
    events: dict
    channels: dict
    layout: dict


def load_mapping(path: Path = MAPPING) -> Mapping:
    """Purpose: read the drawing rules; a malformed or incomplete file raises.
    Spec:    docs/PROPOSAL — RUN VISUALIZATION MAPPING.md §6
    Tests:   tests/bowen/test_famdiag_export.py::test_v5_mapping_covers_every_declared_pattern
    """
    from src.bowen.scenario.config_parse import parse_sectioned_document

    doc = parse_sectioned_document(path.read_text(encoding="utf-8"), sections=SECTIONS,
                                   metadata_keys=frozenset({"block_weeks", "first_year"}), source=str(path))
    rows = {name: doc.sections[name].rows for name in SECTIONS}
    return Mapping(
        block_weeks=int(doc.metadata["block_weeks"]),
        first_year=int(doc.metadata["first_year"]),
        patterns={r["pattern"]: (r["relationship_type"], [s.strip() for s in r["styles"].split(",")], r["ending"])
                  for r in rows["Patterns"]},
        levels={r["measure"]: [int(r[f"level_{i}"]) for i in range(1, 6)] for r in rows["Levels"]},
        triangle_intensity=sorted(((int(r["min_blocks"]), r["intensity"]) for r in rows["Triangle intensity"]),
                                  reverse=True),
        events={r["kind"]: (r["event_type"], r["category"], r["event_class"]) for r in rows["Events"]},
        channels={r["channel"]: r["category"] for r in rows["Symptom channels"]},
        layout={r["id"]: (float(r["x"]), float(r["y"])) for r in rows["Layout"]},
    )


@dataclass
class Run:
    """What the export reads from one run: its records, and each block's held patterns (``current_patterns``)."""

    records: list
    held: list  # held[b] = {subject: Reading} at the end of block b + 1
    weeks: int


def simulate(seed: int, mapping: Mapping) -> Run:
    """Purpose: run M11.C.1's baseline and read the declared patterns at the end of every block.
    Spec:    docs/bowen_agent_model_spec_v2.md#M5.A.1a
    Tests:   tests/bowen/test_famdiag_export.py::test_v5_every_line_traces_back_to_its_reading
    """
    import src.bowen.ensemble.criteria as criteria
    from src.bowen.engine.objects import TieState
    from src.bowen.io.load import load_constants, load_family
    from src.bowen.readouts.patterns import (
        Reading, ReadoutParams, current_patterns, fixed_triangle, last_projection_act, tie_readings,
    )
    from tools.phase_c_trace import records as trace_records

    rp = ReadoutParams.from_constants(load_constants())
    family = load_family(criteria.FAMILIES["phase_c"])
    if mapping.block_weeks != rp.window:
        raise SystemExit(f"block_weeks {mapping.block_weeks} differs from pattern_window {rp.window}: a block "
                         "must be one readout window, or acts would be counted twice or not at all")
    # A tie declared cut off holds that pattern from the start, until a new one emerges (owner D2).
    held = {t: Reading("cutoff", t, 0, 0) for t, tie in family.ties.items() if tie.tie_state is TieState.CUT_OFF}
    blocks: list = []

    def after_tick(state, records, params):
        week = state.tick - 1  # run_tick advances state.tick past the week it ran
        if (week + 1) % mapping.block_weeks:
            return
        nonlocal held
        state.tick = week  # the readouts read the window ending at state.tick
        try:
            readings = [r for tie in state.ties
                        for r in tie_readings(state, tie, params, rp, family.parent_on_tie.get(tie))]
            readings += [r for tri in state.triangles if (r := fixed_triangle(records, tri, rp)) is not None]
            evidence = {tie: last for tie, parent in family.parent_on_tie.items()
                        if (last := last_projection_act(state, tie, parent)) is not None}
        finally:
            state.tick = week + 1
        held = current_patterns(held, readings, now=week, projection_last=evidence, projection_hold=rp.projection_hold)
        blocks.append(dict(held))

    weeks = criteria.SETTINGS["M11.C.1"]["weeks"]
    if weeks % mapping.block_weeks:
        raise SystemExit(f"{weeks} weeks is not a whole number of {mapping.block_weeks}-week blocks: the last "
                         f"{weeks % mapping.block_weeks} would not be drawn")
    records = trace_records(seed, after_tick)
    return Run(records=records, held=blocks, weeks=weeks)


def _level(bands: list, value: int) -> int:
    return max((i for i, low in enumerate(bands) if value >= low), default=0)


def _year(mapping: Mapping, block: int) -> int:
    return mapping.first_year + block


def _end(mapping: Mapping, block: int) -> str:
    """The last day of a block's year. fam-diag shows a span through the year it ends in and hides it after
    (its ``isSpanVisibleAtCutoff``), so a replaced pattern and its successor never show in the same year."""
    return f"{_year(mapping, block):04d}-12-31"


def _created(date: str) -> int:
    """fam-diag requires ``createdAt`` (ms). The event's own date, so the export is the same every time it runs."""
    return int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


def _date(mapping: Mapping, week: int) -> str:
    """A week's date: its block's year, and a month for its place in the block (weeks 1–4 → Jan, Apr, Jul, Oct)."""
    block, within = divmod(week, mapping.block_weeks)
    month = 1 + within * 12 // mapping.block_weeks
    return f"{_year(mapping, block):04d}-{month:02d}-01"


def episodes(held: list) -> list:
    """Purpose: each subject's runs of one held pattern, as (subject, first block, last block, readings).
    Spec:    docs/PROPOSAL — RUN VISUALIZATION MAPPING.md §0 (D2: a pattern ends when a new one emerges)
    Tests:   tests/bowen/test_famdiag_export.py::test_v5_episodes_split_where_the_pattern_changes
    """
    out, open_ = [], {}
    for block, state in enumerate(held):
        for subject in [s for s in open_ if s not in state]:  # a pattern that stopped with nothing after it
            key, first, _, readings = open_.pop(subject)
            out.append((subject, first, block - 1, readings))
        for subject, reading in state.items():
            key = (reading.pattern, reading.toward)
            current = open_.get(subject)
            if current is not None and current[0] == key:
                current[3].append(reading)
                continue
            if current is not None:
                out.append((subject, current[1], block - 1, current[3]))
            open_[subject] = [key, block, None, [reading]]
    for subject, (key, first, _, readings) in open_.items():
        out.append((subject, first, len(held) - 1, readings))
    return sorted(out, key=lambda e: (e[1], str(e[0])))


def build_diagram(run: Run, mapping: Mapping, seed: int) -> dict:
    """Purpose: the fam-diag diagram JSON for one run.
    Spec:    docs/PROPOSAL — RUN VISUALIZATION MAPPING.md §4, #M11.F.9
    Tests:   tests/bowen/test_famdiag_export.py::test_v5_diagram_has_what_famdiag_imports
    """
    import src.bowen.ensemble.criteria as criteria
    from src.bowen.engine.events import Mechanism
    from src.bowen.engine.identifiers import TieId
    from src.bowen.engine.log_records import EmittedRecord, LogHeader
    from src.bowen.io.load import load_family
    from src.bowen.scenario.family import Relation

    run_family = load_family(criteria.FAMILIES["phase_c"])
    reference = load_family(REFERENCE_FAMILY)
    simulated = set(run_family.people)
    everyone = {**reference.people, **run_family.people}
    names = {**reference.display_names, **run_family.display_names}
    relations = {**reference.relations, **run_family.relations}
    parent_on_tie = {**reference.parent_on_tie, **run_family.parent_on_tie}
    missing = sorted(p.value for p in everyone if p.value not in mapping.layout)
    if missing:
        raise SystemExit(f"config/viz/famdiag.md has no layout for {missing}")

    def pid(p):
        return f"p-{p.value}"

    partnerships = []
    for tie, relation in sorted(relations.items(), key=lambda kv: str(kv[0])):
        if relation is not Relation.SPOUSE:
            continue
        a, b = tie.members()
        children = sorted(pid(c) for c in everyone for t, parent in parent_on_tie.items()
                          if parent in (a, b) and c in t.members() and c != parent)
        y = max(mapping.layout[a.value][1], mapping.layout[b.value][1]) + 80
        partnerships.append({"id": f"u-{a.value}-{b.value}", "partner1_id": pid(a), "partner2_id": pid(b),
                             "horizontalConnectorY": y, "relationshipType": "married", "relationshipStatus": "married",
                             "children": sorted(set(children)), "events": []})
    parent_partnership = {c: u["id"] for u in partnerships for c in u["children"]}

    people = []
    for p, person in sorted(everyone.items(), key=lambda kv: kv[0].value):
        x, y = mapping.layout[p.value]
        notes = []
        if p not in simulated:
            notes.append("Not simulated in this run.")
        if p not in reference.people:
            notes.append("In this run's family only; left the reference family at spec revision 13.")
        entry = {"id": pid(p), "name": names[p], "firstName": names[p], "x": x, "y": y,
                 "gender": person.sex.value, "partnerships": sorted(u["id"] for u in partnerships
                                                                    if pid(p) in (u["partner1_id"], u["partner2_id"])),
                 "events": [], "notes": " ".join(notes), "notesEnabled": False}  # notes open would bury the picture
        if pid(p) in parent_partnership:
            entry["parentPartnership"] = parent_partnership[pid(p)]
        if person.role.value == "external":
            entry["isCoach"] = True
        if p not in simulated:
            entry.update(foregroundColor=GREY, foregroundEnabled=True, borderColor=GREY, borderEnabled=True)
        people.append(entry)
    person_entry = {e["id"]: e for e in people}

    lines, triangles, line_at = [], [], {}
    for subject, first, last, readings in episodes(run.held):
        start, ended = f"{_year(mapping, first):04d}-01-01", last < len(run.held) - 1
        pattern = readings[0].pattern
        if pattern == "fixed_triangle":
            blocks = last - first + 1
            intensity = next(name for low, name in mapping.triangle_intensity if blocks >= low)
            p1, p2, p3 = (pid(m) for m in subject.members)
            outsider = names[readings[0].toward]
            tri_id = f"t-{'-'.join(m.value for m in subject.members)}-{first}"
            span = {"startDate": start, **({"endDate": _end(mapping, last)} if ended else {})}
            # fam-diag draws every triangle with three lines of its own (its default, fusion-dotted-wide) and adds
            # undated ones if none are given, which would show at every year. These carry the triangle's span. They
            # are fam-diag's triangle symbol, not pattern readings, and say so.
            tpls = [{"id": f"{tri_id}-tpl-{i}", "person1_id": x, "person2_id": y, "relationshipType": "fusion",
                     "hideAfterEnd": True,
                     "lineStyle": "fusion-dotted-wide", "lineEnding": "none",
                     "status": "ended" if ended else "ongoing", **span, "events": [], "notesEnabled": False,
                     "notes": "Part of the triangle symbol; not a pattern reading."}
                    for i, (x, y) in enumerate(((p1, p2), (p2, p3), (p3, p1)))]
            triangles.append({"id": tri_id, "person1_id": p1, "person2_id": p2, "person3_id": p3,
                              "intensity": intensity, **span,
                              "notes": f"Fixed triangle, {outsider} outside, blocks {first + 1}–{last + 1}.",
                              "notesEnabled": False, "events": [], "tpls": tpls})
            continue
        relationship_type, styles, ending = mapping.patterns[pattern]
        # One held distance can be read from acts in some blocks and from silence in others (owner D2: the same
        # pattern). Acts and weeks are different units, so each is banded and reported on its own scale.
        peaks = {measure: max((r.count for r in readings if r.silent == (measure == "silent_weeks")), default=None)
                 for measure in ("acts", "silent_weeks")}
        level = max(_level(mapping.levels[m], p) for m, p in peaks.items() if p is not None)
        style = styles[min(level, len(styles) - 1)]
        declared = first == 0 and readings[0].count == 0 and not readings[0].silent
        described = "; ".join(f"peak {p} {'weeks without interaction' if m == 'silent_weeks' else 'acts in a block'}"
                              for m, p in peaks.items() if p is not None and not (declared and m == "acts" and p == 0))
        a, b = subject.members()
        toward = readings[0].toward
        if ending == "arrow_from_toward" and toward is not None:
            a, b = toward, subject.other(toward)
        line = {"id": f"l-{pattern}-{subject.a.value}-{subject.b.value}-{first}", "person1_id": pid(a),
                "person2_id": pid(b), "startDate": start, "status": "ended" if ended else "ongoing",
                # fam-diag hides a line after its end year only when it opts in; the app's own lines never do
                "hideAfterEnd": True,
                "relationshipType": relationship_type, "lineStyle": style,
                "lineEnding": "arrow-p1-to-p2" if ending == "arrow_from_toward" and toward is not None else "none",
                "notes": (f"{pattern.replace('_', '/')}, blocks {first + 1}–{last + 1}; "
                          + ("declared cut off at the start" if declared else
                             f"first read {'from silence' if readings[0].silent else 'from acts'}")
                          + (f"; {described}" if described else "") + "."),
                "notesEnabled": False, "events": []}
        if ended:
            line["endDate"] = _end(mapping, last)
        if ending == "over_side" and toward is not None:
            line["adequatePersonId"] = pid(toward)
        lines.append(line)
        for block in range(first, last + 1):
            line_at[(subject, block)] = line

    header = next(r for r in run.records if isinstance(r, LogHeader))
    for record in run.records:
        if not isinstance(record, EmittedRecord):
            continue
        event = record.event
        kind = event.kind
        if kind not in mapping.events:
            raise SystemExit(f"config/viz/famdiag.md has no event row for {kind!r}")
        event_type, category, event_class = mapping.events[kind]
        week = event.timestamp
        people_named = [p for p in (event.sender, *event.targets) if p is not None]
        if category == "by_channel":
            person = everyone[people_named[0]]
            category = mapping.channels[person.channel_prior.value]
        date = _date(mapping, week)
        entry = {"id": f"e-{week}-{event.id.origin}-{event.id.index}", "date": date, "startDate": date,
                 "createdAt": _created(date),
                 "category": category, "eventType": event_type, "subtype": kind, "status": "discrete",
                 "intensity": 0, "howWell": 0, "eventClass": event_class, "wwwwh": "",
                 "primaryPersonName": names[people_named[0]] if people_named else "",
                 "otherPersonName": ", ".join(names[p] for p in people_named[1:]),
                 "observations": f"Week {week + 1}: {kind}"
                                 + (f" by {names[event.sender]}" if event.sender else "")
                                 + (f" to {', '.join(names[t] for t in event.targets)}" if event.targets else "") + "."}
        # Only an act belongs on a tie's line; a symptom or a stressor belongs to a person. An act to several
        # people counts on each of those ties in the readouts; it is placed once, on the first of them with a line.
        line = None
        if event.mechanism is Mechanism.MOVE and event.sender is not None:
            line = next((found for target in event.targets if target != event.sender
                         if (found := line_at.get((TieId.of(event.sender, target), week // mapping.block_weeks)))),
                        None)
        if line is not None:
            line["events"].append({**entry, "anchorType": "EMOTIONAL_PROCESS_EP", "anchorId": line["id"]})
            continue
        anchor = event.sender if event.sender is not None else (event.targets[0] if event.targets else None)
        if anchor is None:
            raise SystemExit(f"event {event.id} has no sender and no target: nowhere to place it")
        person_entry[pid(anchor)]["events"].append({**entry, "anchorType": "PERSON", "anchorId": pid(anchor)})

    total = len(run.held) * mapping.block_weeks
    note = NOTE.format(weeks=mapping.block_weeks, first=_year(mapping, 0), last=_year(mapping, len(run.held) - 1),
                       last_start=total - mapping.block_weeks + 1, total=total, arm="M11.C.1 baseline", seed=seed,
                       spec=header.spec_revision, config=header.config_hash[:12])
    name = f"EPModel run — seed {seed}"
    return {
        "fileMeta": {"fileName": f"phase_c_famdiag_seed{seed}.json", "displayName": name},
        "people": people,
        "partnerships": partnerships,
        "emotionalLines": lines,
        "pageNotes": [{"id": "n-simulated", "x": 1050, "y": 80, "title": "Simulated run", "text": note,
                       "width": 360, "height": 220}],
        "triangles": triangles,
    }


def summary(diagram: dict) -> str:
    lines = Counter(line["relationshipType"] for line in diagram["emotionalLines"])
    events = sum(len(x["events"]) for key in ("people", "emotionalLines", "triangles") for x in diagram[key])
    return (f"{len(diagram['people'])} people, {sum(lines.values())} pattern lines ({dict(sorted(lines.items()))}), "
            f"{len(diagram['triangles'])} fixed triangles, {events} events")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    mapping = load_mapping()
    diagram = build_diagram(simulate(args.seed, mapping), mapping, args.seed)
    out = REPO / "docs" / "review" / f"phase_c_famdiag_seed{args.seed}.json"
    out.write_text(json.dumps(diagram, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}: {summary(diagram)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
