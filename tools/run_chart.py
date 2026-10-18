"""The week-by-week chart — each person's anxiety and functioning over a run, beside its trace.

Purpose: draw what the trace states as changes — each person's level of anxiety and functioning — as one line per
         week, so a reader can follow a run without adding up the trace's deltas.
Spec:    docs/bowen_agent_model_spec_v2.md#M16.C.1, #M11.F.9, #M11.F.10
Tests:   tests/bowen/test_render.py::test_m16c1_chart_series_reads_state_each_week,
         tests/bowen/test_render.py::test_m16c1_chart_marks_outside_events_on_their_targets,
         tests/bowen/test_render.py::test_m16c1_phase_c_chart_renders

It lives in ``tools/``, not ``src/bowen/``, because it changes nothing a run computes: every ``.py`` file under
``src/bowen/`` is in the records' code hash (``tools/ensemble_record.py:code_hash``), and a renderer there would make
every record stale (learnings P38).

**A rendering, not an interpretation**, as the trace is (M16.C.1): the page is a fixed template over values read from
the run, the same run always gives the same bytes, and nothing is computed here that the run did not hold. Values
are sampled after each week by ``WeeklySeries.observe``, passed to the run loop as its ``after_tick`` hook; it reads
state and changes nothing. The page's framing is the trace's own block (``trace.framing``), so the two cannot drift.

What is drawn, per person and week: ``acute_anxiety`` and ``functional_level``, with ``chronic_anxiety`` (the floor
anxiety decays toward) and ``basic_level`` as dashed lines; events from outside the family (``exogenous``) are marked
on the people they land on. A person absent from the field that week, or not alive, has no value.
"""

from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path
from typing import Iterable

from src.bowen.engine.log_records import EmittedRecord, LogHeader, Record
from src.bowen.engine.objects import Role
from src.bowen.render.trace import framing

TEMPLATE = Path(__file__).with_name("run_chart_template.html")
SERIES = ("anxiety", "functioning", "chronic", "basic")


class WeeklySeries:
    """Purpose: sample each person's anxiety and functioning at the end of every week, observing only.
    Spec:    docs/bowen_agent_model_spec_v2.md#M16.C.1
    Tests:   tests/bowen/test_render.py::test_m16c1_chart_series_reads_state_each_week
    """

    def __init__(self) -> None:
        self.weeks: list[dict[str, dict[str, float]]] = []

    def observe(self, state, records=None, params=None) -> None:
        self.weeks.append({
            str(pid): {
                "anxiety": person.acute_anxiety,
                "functioning": person.functional_level,
                "chronic": person.chronic_anxiety,
                "basic": person.basic_level,
            }
            for pid, person in state.people.items() if person.alive
        })


def outside_events(records: Iterable[Record]) -> dict[str, list[tuple[int, str]]]:
    """Purpose: the events from outside the family, as (week, kind) on each person they land on.
    Spec:    docs/bowen_agent_model_spec_v2.md#M16.C.1
    Tests:   tests/bowen/test_render.py::test_m16c1_chart_marks_outside_events_on_their_targets
    """
    found: dict[str, list[tuple[int, str]]] = {}
    for record in records:
        if isinstance(record, EmittedRecord) and record.event.exogenous:
            for target in record.event.targets:
                found.setdefault(str(target), []).append((record.event.timestamp, record.event.kind))
    return found


def _plain(markdown: str) -> str:
    """The trace's framing block as plain text: no quote markers, emphasis or code ticks."""
    text = " ".join(line.removeprefix(">").strip() for line in markdown.splitlines())
    return re.sub(r"[*`]", "", text).strip()


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


def render_chart(records: Iterable[Record], series: WeeklySeries, family) -> str:
    """Purpose: the chart page for one run, from its records, its weekly samples and its family declaration.
    Spec:    docs/bowen_agent_model_spec_v2.md#M16.C.1, #M11.F.9
    Tests:   tests/bowen/test_render.py::test_m16c1_phase_c_chart_renders
    """
    records = list(records)
    if not records or not isinstance(records[0], LogHeader):
        raise ValueError("a log opens with its header (M16.A.1); this one does not")
    header = records[0]
    if not series.weeks:
        raise ValueError("no weeks were sampled; pass WeeklySeries.observe as the run's after_tick")
    events = outside_events(records)
    seen = {pid for week in series.weeks for pid in week}
    people = []
    for pid, name in family.display_names.items():
        if str(pid) not in seen:
            continue
        person = family.people[pid]
        external = person.role is Role.EXTERNAL
        generation = family.generations.get(pid)
        age = family.ages_years.get(pid)
        info = f"age {age} at the start" if age is not None else ""  # the group heading gives the generation
        people.append({
            "id": str(pid),
            "name": name,
            "info": info,
            "group": "Outside the family" if external else (
                f"Generation {generation}" if generation is not None else "Family"),
            "order": (1, 0) if external else (0, generation if generation is not None else 0),
            **{key: [_round(week[str(pid)][key]) if str(pid) in week else None for week in series.weeks]
               for key in SERIES},
            "events": [[week, kind] for week, kind in events.get(str(pid), [])],
        })
    people.sort(key=lambda p: p["order"])  # stable: declaration order within a group
    for p in people:
        del p["order"]
    peak = max((v for p in people for key in SERIES for v in p[key] if v is not None), default=100.0)
    data = {
        "seed": header.seed,
        "instance": header.instance_id,
        "spec": header.spec_revision,
        "configHash": header.config_hash,
        "weeks": len(series.weeks),
        "ymax": max(100, 25 * math.ceil(peak / 25)),
        "framing": _plain(framing(records)),
        "people": people,
    }
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    title = f"Run chart — {header.instance_id}, seed {header.seed}"
    return (TEMPLATE.read_text(encoding="utf-8")
            .replace("__TITLE__", html.escape(title, quote=False))
            .replace("__FRAMING__", html.escape(data["framing"], quote=False))
            .replace("__DATA__", blob))
