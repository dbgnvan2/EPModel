"""Run a record at a fraction of each row's seeds (T-299's run mix, Claude's decision 2026-11-26, pending owner review).

Purpose: after an engine batch the two slow records, the Phase D part II ensemble record and the Phase D mutation
         record, are rerun at a fraction F of their seeds (``--seed-fraction F``); every other record runs in full.
         ``docs/review/2026-11-26_short_rerun_test.md`` measured what this catches. At F the ensemble rules'
         ``ensemble_block`` and ``ensemble_cap`` are multiplied by F and ``ensemble_precision`` divided by √F (the
         adaptive stop then needs about F as many seeds, since the half-width falls as 1/√n), and a criterion's fixed
         seed count (``M11.C.10``'s 600, ``M11.C.48``'s 200) by F. ``M17.A.1``'s near-limit rerun keeps its declared
         count: a row near the detection limit is the borderline case, so it gets full precision.
         Nothing here is in the engine hash. A result at F is cached under its own key (``definition``), so a full
         run never reads it; the record states F in its header (``note``), and a rebuild from the cache reads F back
         from the committed record (``declared``).
Spec:    docs/review/2026-11-26_short_rerun_test.md#recommended-mix-of-runs
Tests:   tests/test_seed_fraction.py
"""

from __future__ import annotations

import dataclasses
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# The seed settings a fraction scales: (file, the line's start, how). The value after the start is replaced.
SCALED_CONFIG = (
    ("config/bowen/constants.md", "| `ensemble_block` | ", "times"),
    ("config/bowen/constants.md", "| `ensemble_cap` | ", "times"),
    ("config/bowen/constants.md", "| `ensemble_precision` | ", "over_root"),
    ("config/bowen/criteria.md", "| M11.C.10 | seeds | ", "times"),
    ("config/bowen/criteria.md", "| M11.C.48 | seeds | ", "times"),
)
NOTE = re.compile(r"`--seed-fraction ([0-9.]+)`")


def check(fraction: float) -> float:
    if not 0 < fraction <= 1:
        raise SystemExit(f"--seed-fraction {fraction}: must be above 0 and at most 1")
    return fraction


def scaled_value(value: float, how: str, fraction: float) -> str:
    if how == "times":
        return str(max(1, round(value * fraction)))
    return f"{value / math.sqrt(fraction):.4g}"


def scaled_rules(rules: dict, fraction: float) -> dict:
    """The ensemble rules at ``fraction``; the same dict at 1."""
    if fraction == 1:
        return rules
    return {**rules, "ensemble_block": int(scaled_value(rules["ensemble_block"], "times", fraction)),
            "ensemble_cap": int(scaled_value(rules["ensemble_cap"], "times", fraction)),
            "ensemble_precision": float(scaled_value(rules["ensemble_precision"], "over_root", fraction))}


def scaled_criterion(criterion, fraction: float):
    """A criterion that fixes its own seed count runs ``fraction`` of it."""
    if fraction == 1 or criterion.seeds is None:
        return criterion
    return dataclasses.replace(criterion, seeds=int(scaled_value(criterion.seeds, "times", fraction)))


def definition(base: dict | None, fraction: float) -> dict | None:
    """The cache definition of a result at ``fraction``: the full run's (``base``) at 1, else it plus the fraction."""
    return base if fraction == 1 else {**(base or {}), "seed_fraction": fraction}


def config_edits(fraction: float, root: Path = REPO) -> tuple[tuple[str, str, str], ...]:
    """The (file, old, new) line edits that put ``root``'s seed settings at ``fraction``; each line must occur once."""
    edits = []
    for file, start, how in SCALED_CONFIG:
        lines = [l for l in (root / file).read_text(encoding="utf-8").splitlines() if l.startswith(start)]
        if len(lines) != 1:
            raise SystemExit(f"{file}: {len(lines)} lines start {start!r}, not one")
        rest = lines[0][len(start):]
        value = re.match(r"[0-9.]+", rest).group(0)
        edits.append((file, lines[0], start + scaled_value(float(value), how, fraction) + rest[len(value):]))
    return tuple(edits)


def scaled_mutant(mutant, fraction: float):
    """A mutant whose copy of the repository also carries the scaled seed settings, so its child (hashed, unedited)
    runs at ``fraction``; its definition, and so its cache key, then differ from the full run's."""
    if fraction == 1:
        return mutant
    return dataclasses.replace(mutant, also=(*mutant.also, *config_edits(fraction)))


def note(fraction: float) -> str:
    return (f"**Seeds: {fraction:g} of each row's declared count** (`--seed-fraction {fraction:g}`, the run mix of "
            "T-299, `docs/review/2026-11-26_short_rerun_test.md`). A verdict here is the short run's; the milestone "
            "run reruns this record at full seeds.")


def noted(text: str, fraction: float) -> str:
    """``text`` with the fraction's note after its title line; unchanged at 1."""
    if fraction == 1:
        return text
    title, _, rest = text.partition("\n")
    return f"{title}\n\n{note(fraction)}\n{rest}"


def declared(record: Path) -> float:
    """The fraction a committed record states; 1 when it states none."""
    if not record.exists():
        return 1.0
    found = NOTE.search(record.read_text(encoding="utf-8"))
    return float(found.group(1)) if found else 1.0
