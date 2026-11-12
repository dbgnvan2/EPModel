"""The supply readout: who supplied attention, approval and help to whom, computed after a run.

Purpose: per ordered pair, the count of moves from the kinds ``config/lint/supply_kinds.md`` lists as supplying, and,
         when the social cues were on, the mean perceived attention and approval cue each person received from each
         other, from the readings a ``ReadingTap`` collected while the run went. A role name may label this readout in
         a report, never a person, and nothing in the engine or the policy reads it (`M1.A.14b`).
Spec:    docs/bowen_agent_model_spec_v2.md#M11.G.5, #M1.A.14b
Tests:   tests/bowen/test_roles_readout.py

Which kinds count as supplying is editorial and `[I]`, so the list is a config file, not code (global rule 9). It is
kept outside ``config/bowen/`` because it is read by a readout only and must not move the engine hash.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.bowen.engine.events import Mechanism
from src.bowen.engine.identifiers import PersonId
from src.bowen.engine.log_records import EmittedRecord
from src.bowen.engine.signals import CUES, Cue
from src.bowen.scenario.config_parse import parse_table_document

SUPPLY_KINDS_PATH = Path(__file__).resolve().parents[3] / "config" / "lint" / "supply_kinds.md"


def load_supply_kinds(path: Path = SUPPLY_KINDS_PATH) -> frozenset[str]:
    """The move kinds that count as supplying another person (`M11.G.5`), from config."""
    document = parse_table_document(path.read_text(encoding="utf-8"), columns=("kind", "why"),
                                    metadata_keys=frozenset(), source=str(path))
    return frozenset(row["kind"].strip("`") for row in document.rows)


def supply_counts(records, kinds: frozenset[str] | None = None, *, since: int = 0,
                  until: int | None = None) -> Counter:
    """Purpose: the supplying moves each person sent each other in ``[since, until)``, per ordered (from, to) pair.
    Spec:    docs/bowen_agent_model_spec_v2.md#M11.G.5
    Tests:   tests/bowen/test_roles_readout.py::test_m11g5_supply_counts_per_ordered_pair
    """
    kinds = load_supply_kinds() if kinds is None else kinds
    counts: Counter = Counter()
    for r in records:
        if not isinstance(r, EmittedRecord) or r.event.mechanism is not Mechanism.MOVE or r.event.sender is None:
            continue
        e = r.event
        if e.kind in kinds and e.timestamp >= since and (until is None or e.timestamp < until):
            for target in e.targets:
                counts[e.sender, target] += 1
    return counts


class ReadingTap:
    """A ``watch`` for a run: after each tick it copies the week's cue readings made by targets (it only reads)."""

    def __init__(self) -> None:
        self.sums: dict[tuple[PersonId, PersonId], list[float]] = {}
        self.counts: Counter = Counter()

    def __call__(self, state) -> None:
        for (event_id, recipient), reading in sorted(state.cue_readings.items(), key=lambda kv: (str(kv[0][0]),
                                                                                                kv[0][1].value)):
            event = state.store.event(event_id)
            if event.sender is None or recipient not in event.targets or not state.people[recipient].alive:
                continue  # a witness's reading is not a cue received from the sender; nor is a dead target's
            key = (event.sender, recipient)
            row = self.sums.setdefault(key, [0.0] * len(CUES))
            for i, v in enumerate(reading.total):
                row[i] += v
            self.counts[key] += 1

    def mean_received(self, cue: Cue) -> dict[tuple[PersonId, PersonId], float]:
        """Purpose: the mean perceived ``cue`` each person received from each other, per (sender, receiver).
        Spec:    docs/bowen_agent_model_spec_v2.md#M11.G.5
        Tests:   tests/bowen/test_roles_readout.py::test_m11g5_mean_received_cue_per_pair
        """
        i = CUES.index(cue)
        return {key: self.sums[key][i] / self.counts[key] for key in sorted(self.sums) if self.counts[key]}


def main_supplier(counts: Counter, to: PersonId, among: tuple[PersonId, ...]) -> PersonId | None:
    """The one of ``among`` who sent ``to`` the most supplying moves; None on a tie for the most or when none did."""
    tallies = sorted(((counts.get((p, to), 0), p.value) for p in among), reverse=True)
    if not tallies or tallies[0][0] == 0 or (len(tallies) > 1 and tallies[0][0] == tallies[1][0]):
        return None
    return PersonId(tallies[0][1])

