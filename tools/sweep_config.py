"""Read ``config/sweep/sweep.md``: the constant sweep's rules and, per criterion, the ranges it samples.

Purpose: one strict parser for the sweep's own config, so no range or rule is a literal in a tool (rule 9, P4) and a
         malformed row fails at load rather than running a sample at an unintended value (`M10.B.2`'s rule, applied
         to a measurement input). The engine never reads this file; it lives outside ``config/bowen/`` because the
         engine hash covers that directory (docs/implementation_plan_version_2.md §2.1 item 4, D12).
Spec:    docs/bowen_agent_model_spec_v2.md#M17.E.1; docs/implementation_plan_version_2.md#2.2 (D8, D10, D12)
Tests:   tests/test_version_2_sweep.py::test_s2_6_sweep_config_is_loaded_strictly_with_grades
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(REPO))
from src.bowen.scenario.constants import SCHEMA  # noqa: E402  (the constant's kind: int or float)
CONFIG = REPO / "config" / "sweep" / "sweep.md"
CONSTANTS = REPO / "config" / "bowen" / "constants.md"
RULE_NAMES = ("sweep_samples", "sweep_seed", "detection_limit_multiple", "detection_limit_seeds")
SCALES = ("log", "linear", "integer")
RULE_COLUMNS = ("constant", "value", "grade", "unit", "why")
RANGE_COLUMNS = ("criterion", "constant", "low", "high", "scale", "bound", "grade", "why it is read")
GRADE = re.compile(r"^\[[A-Z#]+\]")
_ROW = re.compile(r"^\| `(?P<name>[^`]+)` \| (?P<value>[^|]*?) \|")


class SweepConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Range:
    criterion: str
    constant: str
    low: float
    high: float
    scale: str
    bound: float | None
    grade: str
    why: str


def central_values(path: Path = CONSTANTS) -> dict[str, str]:
    """Each constant's central value as the text of its row in ``constants.md`` (the text a sample's edit replaces)."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        found = _ROW.match(line)
        if found:
            out[found["name"]] = found["value"].strip()
    if not out:
        raise SweepConfigError(f"{path}: no constant rows found")
    return out


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _table(text: str, heading: str, columns: tuple[str, ...]) -> list[list[str]]:
    """The rows of the one table under ``## heading``, whose header row must name exactly ``columns``."""
    if text.count(f"\n## {heading}\n") != 1:
        raise SweepConfigError(f"the section '## {heading}' is missing or repeated")
    body = text.split(f"\n## {heading}\n", 1)[1].split("\n## ", 1)[0]
    lines = [l for l in body.splitlines() if l.startswith("|")]
    header, rule, *body_rows = lines if len(lines) > 1 else ("", "")
    if not header or not set(rule.replace("|", "").strip()) <= {"-"}:
        raise SweepConfigError(f"'## {heading}': a header row and a rule row are required")
    if tuple(_cells(header)) != columns:
        raise SweepConfigError(f"'## {heading}': the header must be {columns}, not {tuple(_cells(header))}")
    rows = []
    for line in body_rows:
        cells = _cells(line)
        if len(cells) != len(columns):
            raise SweepConfigError(f"'## {heading}': a row does not have {len(columns)} cells: {line}")
        rows.append(cells)
    return rows


def _number(raw: str, where: str) -> float:
    try:
        return float(raw)
    except ValueError:
        raise SweepConfigError(f"{where}: {raw!r} is not a number") from None


def _grade(raw: str, where: str) -> str:
    if not GRADE.match(raw):
        raise SweepConfigError(f"{where}: the grade cell must start with a grade such as [I], not {raw!r}")
    return raw


def load_rules(text: str) -> dict[str, int | float]:
    rules: dict[str, int | float] = {}
    for name, value, grade, _unit, _why in _table(text, "Rules", RULE_COLUMNS):
        name = name.strip("`")
        if name not in RULE_NAMES:
            raise SweepConfigError(f"unknown rule {name!r}; the rules are {RULE_NAMES}")
        _grade(grade, name)
        number = _number(value, name)
        rules[name] = int(number) if number == int(number) and "." not in value else number
    missing = [n for n in RULE_NAMES if n not in rules]
    if missing:
        raise SweepConfigError(f"rules missing: {missing}")
    if rules["sweep_samples"] < 1 or rules["detection_limit_seeds"] < 1 or rules["detection_limit_multiple"] <= 0:
        raise SweepConfigError("sweep_samples and detection_limit_seeds must be at least 1, detection_limit_multiple above 0")
    return rules


def load_ranges(text: str, central: dict[str, str]) -> dict[str, list[Range]]:
    """Criterion key -> its ranges, in the file's order (the order the draw reads them in)."""
    out: dict[str, list[Range]] = {}
    for cid, name, low, high, scale, bound, grade, why in _table(text, "Ranges", RANGE_COLUMNS):
        cid, name = cid.strip("`"), name.strip("`")
        where = f"{cid} {name}"
        if name not in central:
            raise SweepConfigError(f"{where}: not a constant in {CONSTANTS.name}")
        if name not in SCHEMA:
            raise SweepConfigError(f"{where}: in {CONSTANTS.name} but not in the constants SCHEMA")
        if scale not in SCALES:
            raise SweepConfigError(f"{where}: scale must be one of {SCALES}, not {scale!r}")
        lo, hi = _number(low, where), _number(high, where)
        if not lo < hi:
            raise SweepConfigError(f"{where}: low {lo} is not below high {hi}")
        if scale == "log" and lo <= 0:
            raise SweepConfigError(f"{where}: a log range must be positive")
        if scale == "integer" and not (lo == int(lo) and hi == int(hi) and lo >= 1):
            raise SweepConfigError(f"{where}: an integer range must be whole and at least 1")
        if (scale == "integer") != (SCHEMA[name].kind is int):
            raise SweepConfigError(f"{where}: the constant is {SCHEMA[name].kind.__name__}, so its scale must "
                                   f"{'be integer' if SCHEMA[name].kind is int else 'be log or linear'}, not {scale!r}")
        mid = _number(central[name], where)
        if not lo <= mid <= hi:
            raise SweepConfigError(f"{where}: the range [{lo}, {hi}] does not bracket the central value {mid}")
        limit = None if bound == "—" else _number(bound, where)
        if limit is not None and hi > limit:
            raise SweepConfigError(f"{where}: high {hi} is above the declared bound {limit}")
        if any(r.constant == name for r in out.get(cid, ())):
            raise SweepConfigError(f"{where}: declared twice")
        if not why:
            raise SweepConfigError(f"{where}: say why the mechanism reads it")
        out.setdefault(cid, []).append(Range(cid, name, lo, hi, scale, limit, _grade(grade, where), why))
    if not out:
        raise SweepConfigError("no ranges declared")
    return out


def load_config(path: Path = CONFIG, constants: Path = CONSTANTS) -> tuple[dict[str, int | float], dict[str, list[Range]]]:
    text = path.read_text(encoding="utf-8")
    return load_rules(text), load_ranges(text, central_values(constants))
