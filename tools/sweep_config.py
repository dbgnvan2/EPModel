"""Read ``config/sweep/sweep.md``: the constant sweep's rules and, per criterion, the ranges it samples.

Purpose: one strict parser for the sweep's own config, so no range or rule is a literal in a tool (rule 9, P4) and a
         malformed row fails at load rather than running a sample at an unintended value (`M10.B.2`'s rule, applied
         to a measurement input). The engine never reads this file; it lives outside ``config/bowen/`` because the
         engine hash covers that directory (docs/implementation_plan_version_2.md §2.1 item 4, D12).
Spec:    docs/bowen_agent_model_spec_v2.md#M17.E.1; docs/implementation_plan_version_2.md#2.2 (D8, D10, D12);
         docs/implementation_plan_version_2_stage_3.md#3.4 (the named ratios, D24, D25)
Tests:   tests/test_version_2_sweep.py::test_s2_6_sweep_config_is_loaded_strictly_with_grades,
         tests/test_version_2_sweep.py::test_s3_19_ratio_row_parsed_strictly
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
NOT_RUN_HEADING = "Declared, not run"
NOT_RUN_COLUMNS = ("criterion", "why")
RATIO_HEADING = "Named ratios"
RATIO_COLUMNS = ("ratio", "numerator", "denominator", "period constant", "low", "high", "scale", "grade", "spec")
RATIO_SCALES = ("log", "linear")
# A share's admissible bound: ``EngineParams`` requires ``tie_wear_rate`` in [0, 1] (src/bowen/engine/params.py), the
# bound the Ranges table's preamble applies to a share or weight. A ratio's derived numerator may not leave it (D25).
SHARE_BOUND = 1.0
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


@dataclass(frozen=True)
class Ratio:
    """A named ratio swept as one draw (`M17.E.1`, D24–D26): ``numerator`` ÷ (``period`` × ``denominator``). A sample
    draws the ratio over [``low``, ``high``] and the denominator over its declared range (``denominator_range``), and
    derives the numerator from the two, so the sample's ratio is exact."""
    name: str
    numerator: str
    denominator: str
    period: str
    low: float
    high: float
    scale: str
    grade: str
    spec: str
    central: float
    denominator_range: Range


class SweepConfig(tuple):
    """``(rules, ranges)``, unpacked as before, with the named ratios beside them as ``ratios``."""
    ratios: list[Ratio]

    def __new__(cls, rules: dict, ranges: dict, ratios: list[Ratio]):
        self = super().__new__(cls, (rules, ranges))
        self.ratios = ratios
        return self


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


def load_not_run(text: str, ranges: dict[str, list[Range]]) -> dict[str, str]:
    """Criterion key -> why it has not run, for the keys listed under ``## Declared, not run``: their ranges are
    declared and no result of theirs is expected yet (empty when the section is absent). A listed key must have
    declared ranges, appear once and say why."""
    if f"\n## {NOT_RUN_HEADING}\n" not in text:
        near = re.findall(r"^## .*\bnot\b.*\brun\b.*$", text, re.MULTILINE | re.IGNORECASE)
        if near:  # a misspelt heading would list nothing, and a full sweep would run the keys it meant to hold back
            raise SweepConfigError(f"{near[0]!r}: the heading must be '## {NOT_RUN_HEADING}'")
        return {}
    out: dict[str, str] = {}
    for cid, why in _table(text, NOT_RUN_HEADING, NOT_RUN_COLUMNS):
        cid = cid.strip("`")
        if cid not in ranges:
            raise SweepConfigError(f"{cid}: listed as not run but no range of it is declared")
        if cid in out:
            raise SweepConfigError(f"{cid}: listed as not run twice")
        if not why:
            raise SweepConfigError(f"{cid}: say why it has not run")
        out[cid] = why
    return out


def _fraction(raw: str, where: str) -> float:
    """A range end written as a decimal or as a fraction ``a/b`` of two numbers."""
    if "/" in raw:
        top, _, bottom = raw.partition("/")
        denominator = _number(bottom.strip(), where)
        if denominator == 0:
            raise SweepConfigError(f"{where}: {raw!r} divides by zero")
        return _number(top.strip(), where) / denominator
    return _number(raw, where)


def load_ratios(text: str, central: dict[str, str]) -> list[Ratio]:
    """The table under ``## Named ratios`` (empty when the section is absent). Each row's three constants must be in
    ``constants.md`` and the SCHEMA, the period an integer; its range must hold the central ratio; its denominator must
    have one declared range in ``## Ranges`` and its numerator none (the ratio derives it); and the derived numerator
    must stay within ``SHARE_BOUND`` at both ends of both ranges (D25), so no sample can leave it."""
    if f"\n## {RATIO_HEADING}\n" not in text:
        return []
    ranges = load_ranges(text, central)
    out: list[Ratio] = []
    for name, numerator, denominator, period, low, high, scale, grade, spec in _table(text, RATIO_HEADING, RATIO_COLUMNS):
        name, numerator, denominator, period = (c.strip("`") for c in (name, numerator, denominator, period))
        where = f"named ratio {name}"
        for constant in (numerator, denominator, period):
            if constant not in central or constant not in SCHEMA:
                raise SweepConfigError(f"{where}: {constant!r} is not a constant in {CONSTANTS.name} and its SCHEMA")
        if SCHEMA[period].kind is not int or SCHEMA[numerator].kind is not float:
            raise SweepConfigError(f"{where}: the period must be an integer constant and the numerator a float one")
        if scale not in RATIO_SCALES:
            raise SweepConfigError(f"{where}: scale must be one of {RATIO_SCALES}, not {scale!r}")
        lo, hi = _fraction(low, where), _fraction(high, where)
        if not 0 < lo < hi:
            raise SweepConfigError(f"{where}: the range must be positive with low {lo} below high {hi}")
        ticks = _number(central[period], where)
        mid = _number(central[numerator], where) / (ticks * _number(central[denominator], where))
        if not lo <= mid <= hi:
            raise SweepConfigError(f"{where}: the range [{lo:g}, {hi:g}] does not hold the central ratio {mid:g}")
        declared = [r for rows in ranges.values() for r in rows if r.constant == denominator]
        if not declared or len({(r.low, r.high, r.scale) for r in declared}) != 1:
            raise SweepConfigError(f"{where}: {denominator} needs one declared range in '## Ranges' "
                                   f"({len(declared)} rows, not agreeing or absent)")
        if any(r.constant == numerator for rows in ranges.values() for r in rows):
            raise SweepConfigError(f"{where}: {numerator} has a row in '## Ranges', but the ratio derives it")
        if any(r.constant == period for rows in ranges.values() for r in rows):
            raise SweepConfigError(f"{where}: {period} has a row in '## Ranges', but the ratio is derived at its "
                                   "central value, so a swept period would record a ratio the run did not have")
        span = declared[0]
        if hi * ticks * span.high > SHARE_BOUND:
            raise SweepConfigError(f"{where}: the derived {numerator} reaches {hi * ticks * span.high:g}, above its "
                                   f"admissible bound {SHARE_BOUND:g}")
        if not spec:
            raise SweepConfigError(f"{where}: name the spec ID that makes it")
        if any(r.name == name for r in out):
            raise SweepConfigError(f"{where}: declared twice")
        out.append(Ratio(name, numerator, denominator, period, lo, hi, scale, _grade(grade, where), spec, mid, span))
    return out


def not_run_keys(path: Path = CONFIG, constants: Path = CONSTANTS) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    return load_not_run(text, load_ranges(text, central_values(constants)))


def load_config(path: Path = CONFIG, constants: Path = CONSTANTS) -> SweepConfig:
    """The rules and the ranges, unpacked as a pair, with the named ratios as ``.ratios``."""
    text = path.read_text(encoding="utf-8")
    central = central_values(constants)
    return SweepConfig(load_rules(text), load_ranges(text, central), load_ratios(text, central))
