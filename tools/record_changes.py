"""Generate the page of what version 2 stage 1 changed in the records, against the tag `version-1`.

Purpose: stage 1 changes no rule, but exact float totals (S1.1) move the last bits of many sums, and a selection draw
         that lands on a boundary can change a seed's trajectory from that week. This tool reads each generated
         record at the tag `version-1` and as committed now, and writes `docs/version_2_stage_1_changes.md`: every
         row whose verdict changed, every effect that moved by more than the precision the tag's record stated (its
         half-width), every row on one side only, and the records it does not compare, with why. A changed verdict
         is a measured change for the owner, not a defect (plan §1.6). A record whose hash line is still the tag's
         was not regenerated, and the tool refuses to write a page that would call it unchanged (P35).
Spec:    docs/implementation_plan_version_2.md#S1.10b
Tests:   tests/test_record_changes.py::test_s1_10b_changes_page_is_current,
         tests/test_record_changes.py::test_s1_10b_a_changed_verdict_and_a_moved_effect_are_listed,
         tests/test_record_changes.py::test_s1_10b_every_field_is_compared,
         tests/test_record_changes.py::test_s1_10b_every_generated_record_is_compared_or_listed_as_not
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.record_json import record_json  # noqa: E402

TAG = "version-1"
OUT = REPO / "docs" / "version_2_stage_1_changes.md"
HASH_LINE = re.compile(r"^(?:code_hash|engine_hash): ([0-9a-f]{64})$", re.M)


@dataclass(frozen=True)
class Record:
    path: str
    unit: str  # what one row is
    fields: tuple[str, ...]  # the row's key
    verdict: str | None  # the verdict field; None: a record of values with no verdict (compared value by value)
    # At the tag the D2 record printed `Criterion.id`, so a variant shared its criterion's key or had none of its own
    # (S1.7b fixed this): its rows are matched by position, each to a current key that names the same criterion.
    positional: bool = False


# A criterion's verdict is its outcome; a mutant's is whether the criterion caught it (`result`); a sweep setting's
# is the criterion's outcome under it.
RECORDS = (
    Record("docs/phase_c_ensemble_record.md", "criterion", ("criterion",), "outcome"),
    Record("docs/phase_d_ensemble_record.md", "criterion", ("criterion",), "outcome"),
    Record("docs/phase_d2_ensemble_record.md", "criterion", ("criterion",), "outcome", positional=True),
    Record("docs/phase_c_mutation_record.md", "mutant", ("mutant", "criterion"), "result"),
    Record("docs/phase_d_mutation_record.md", "mutant", ("mutant", "criterion"), "result"),
    Record("docs/phase_c_sweep_record.md", "setting", ("setting", "criterion"), "outcome"),
    Record("docs/phase_c_level_occupancy.md", "variant", ("variant", "arm"), None),
)
NOT_COMPARED = (
    ("docs/phase_c_diagnostic_record.md", "it has no machine-readable block; "
     "`git diff version-1 -- docs/phase_c_diagnostic_record.md` shows what changed"),
    ("docs/review/phase_c_trace_seed7.md, docs/review/phase_c_chart_seed7.html", "one seed's run, rendered; S1.7a "
     "adds lock-in release rows by design, so `git diff version-1` on them shows that and any change of trajectory"),
    ("docs/review/phase_b_trace_seed7.md, docs/review/phase_b_trace_seed7_nadia.md", "Phase B's seed-7 traces, "
     "rendered at Phase B's close (`3132c69`) and not regenerated since, so they are the tag's"),
    ("docs/phase_c_completion_report.md §11", "generated from the Phase C records compared here (`tools/c16_report.py`)"),
    ("docs/spec_coverage.md", "generated from the tests and the ensemble and mutation records compared here"),
    ("docs/phase_b_mutation_record.md", "the record of Phase B's close at commit `3132c69`, frozen (plan D5) and not "
     "regenerated, so it is the tag's apart from the status line stage 1 added; `tools/mutation_gate.py` refuses to "
     "rewrite it"),
    ("docs/version_1_status.md", "version 1's status page, frozen at the tag: "
     "`tests/test_version_1_status.py` checks that it is the tag's, byte for byte"),
    ("docs/review/phase_c_famdiag_seed7.json", "the seed-7 run as a family diagram (`tools/famdiag_export.py`), not "
     "regenerated in stage 1 because the working copy holds the owner's uncommitted edits to it; built from the "
     "current engine on 2026-10-19 and compared with the committed export, it differs only in its note, which names "
     "revision 32 and the tag's config hash"),
)


def tag_text(path: str) -> str:
    """The file at the tag. A missing tag fails; it is never skipped (CI checks out with fetch-depth 0)."""
    result = subprocess.run(["git", "show", f"{TAG}:{path}"], cwd=REPO, capture_output=True, timeout=120)
    if result.returncode != 0:
        raise SystemExit(f"git show {TAG}:{path} failed: {result.stderr.decode(errors='replace').strip()} "
                         f"(the tag must be present locally: git fetch --tags)")
    return result.stdout.decode("utf-8")


def record_hash(text: str) -> str | None:
    found = HASH_LINE.search(text)
    return found.group(1) if found else None


def _key(row: dict, fields: tuple[str, ...]) -> str:
    return " · ".join(str(row[f]) for f in fields)


def keyed(rows: list[dict], fields: tuple[str, ...]) -> dict[str, dict]:
    out = {}
    for row in rows:
        key = _key(row, fields)
        if key in out:
            raise ValueError(f"key {key!r} twice")
        out[key] = row
    return out


def tag_keyed(then: list[dict], now: list[dict], record: Record) -> dict[str, dict]:
    """The tag's rows by the current key: by position for a positional record, each matched only to a current row
    whose key names the same criterion (``M11.C.10[family of origin]`` is ``M11.C.10``'s variant)."""
    if not record.positional:
        return keyed(then, record.fields)
    if len(then) != len(now):
        raise ValueError(f"{record.path}: {len(then)} rows at the tag and {len(now)} now; stage 1 adds or removes no "
                         "criterion, so the rows cannot be matched by position")
    out = {}
    for i, (old, new) in enumerate(zip(then, now)):
        old_key, new_key = _key(old, record.fields), _key(new, record.fields)
        if new_key.split("[", 1)[0] != old_key:
            raise ValueError(f"{record.path}: row {i} is {old_key!r} at the tag and {new_key!r} now")
        out[new_key] = old
    return out


@dataclass(frozen=True)
class Comparison:
    compared: int
    changed: list[tuple[str, dict, dict]]  # (key, at the tag, now): the verdict changed
    moved: list[tuple[str, str, dict, dict]]  # (key, readout, at the tag, now): moved past the tag's half-width
    values: list[tuple[str, str, object, object]]  # (key, field, at the tag, now): every field not exactly the tag's
    gone: list[tuple[str, dict]]  # rows only at the tag
    new: list[tuple[str, dict]]  # rows only now

    @property
    def differ(self) -> int:
        """Fields not exactly the tag's, on rows on both sides (review 2026-10-19: this counted only mean
        differences, while the page said every value)."""
        return len(self.values)


MISSING = object()  # a field on one side only


def _readouts(row: dict) -> dict[str, dict]:
    """A row's readouts by name. A name twice in one row is refused: the second would hide the first from the
    comparison (fix re-sweep, 2026-10-20)."""
    out = {}
    for r in row.get("readouts", []):
        if r["readout"] in out:
            raise ValueError(f"readout {r['readout']!r} twice in one row, so its fields cannot be paired")
        out[r["readout"]] = r
    return out


def _values(row: dict, fields: tuple[str, ...], prefix: str = "") -> dict[str, object]:
    """A value row's numbers and labels, flattened, its key fields left out."""
    out = {}
    for k, v in row.items():
        if not prefix and k in fields:
            continue
        if isinstance(v, dict):
            out.update(_values(v, fields, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _flat(row: dict, fields: tuple[str, ...]) -> dict[str, object]:
    """Every field of a row but its key, flattened: the verdict, the seed count, the flags, and each readout's mean
    difference, half-width, p and the rest under ``readouts.<name>.``, so readouts pair by name whatever their
    order."""
    out = _values({k: v for k, v in row.items() if k != "readouts"}, fields)
    for name, r in _readouts(row).items():
        out.update(_values(r, (), f"readouts.{name}."))
    return out


def _nan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)


def _same(a, b) -> bool:
    """Equal, or both NaN: a NaN in both records is no difference."""
    return a == b or (_nan(a) and _nan(b))


def differing(then: dict, now: dict, fields: tuple[str, ...]) -> list[tuple[str, object, object]]:
    """Each field of two rows that is not exactly the same, or is on one side only (``MISSING`` on the other)."""
    a, b = _flat(then, fields), _flat(now, fields)
    return [(name, a.get(name, MISSING), b.get(name, MISSING)) for name in sorted(set(a) | set(b))
            if name not in a or name not in b or not _same(a[name], b[name])]


def moved_effects(then: dict, now: dict) -> list[tuple[str, dict, dict]]:
    """Each readout whose effect moved by more than the tag's stated half-width, that became or stopped being NaN,
    or that is only on one side."""
    a, b = _readouts(then), _readouts(now)
    out = []
    for name in sorted(set(a) | set(b)):
        old, new = a.get(name), b.get(name)
        if old is None or new is None:
            out.append((name, old or {}, new or {}))
        elif _nan(old["mean_difference"]) or _nan(new["mean_difference"]):
            if not _same(old["mean_difference"], new["mean_difference"]):
                out.append((name, old, new))
        elif abs(new["mean_difference"] - old["mean_difference"]) > old["half_width"]:
            out.append((name, old, new))
    return out


def compare(then_rows: list[dict], now_rows: list[dict], record: Record) -> Comparison:
    then, now = tag_keyed(then_rows, now_rows, record), keyed(now_rows, record.fields)
    both = [k for k in now if k in then]
    gone = [(k, then[k]) for k in then if k not in now]
    new = [(k, now[k]) for k in now if k not in then]
    values = [(k, name, old, new) for k in both for name, old, new in differing(then[k], now[k], record.fields)]
    if record.verdict is None:
        changed = [(k, then[k], now[k]) for k in both if differing(then[k], now[k], record.fields)]
        return Comparison(len(both), changed, [], values, gone, new)
    changed = [(k, then[k], now[k]) for k in both if then[k][record.verdict] != now[k][record.verdict]]
    moved = [(k, name, old, new) for k in both if then[k][record.verdict] == now[k][record.verdict]
             for name, old, new in moved_effects(then[k], now[k])]
    return Comparison(len(both), changed, moved, values, gone, new)


def _effect(r: dict) -> str:
    if not r:
        return "—"
    p = f"; p {r['p']:.3g}" if r.get("p") is not None else ""
    return f"{r['mean_difference']:+.4g} ± {r['half_width']:.3g}{p}"


def _effects(row: dict) -> str:
    return "<br>".join(f"{name} {_effect(r)}" for name, r in _readouts(row).items()) or "—"


def _value(v) -> str:
    if v is MISSING:
        return "—"
    return repr(v) if isinstance(v, float) else str(v)  # in full: a difference in the last bit must show


def _section(record: Record, c: Comparison) -> list[str]:
    lines = ["", f"## `{record.path}`", ""]
    unit = record.unit.capitalize()
    if not (c.changed or c.moved or c.gone or c.new or c.values):
        return lines + [f"No change: {c.compared} of {c.compared} rows are exactly the tag's, every field."]
    if record.verdict is None and c.changed:
        lines += [f"**Rows whose values changed: {len(c.changed)} of {c.compared}.**", "",
                  f"| {unit} | Value | At the tag | Now |", "|---|---|---|---|"]
        lines += [f"| {key} | {name} | {_value(old)} | {_value(new)} |" for key, name, old, new in c.values]
        lines.append("")
    elif c.changed:
        lines += [f"**Verdicts changed: {len(c.changed)} of {c.compared}.**", "",
                  f"| {unit} | At the tag | Now | Effects at the tag | Effects now |", "|---|---|---|---|---|"]
        lines += [f"| {key} | {old[record.verdict]} | {new[record.verdict]} | {_effects(old)} | {_effects(new)} |"
                  for key, old, new in c.changed]
        lines.append("")
    if c.moved:
        lines += [f"**Effects moved past the tag's half-width, verdict held: {len(c.moved)}.**", "",
                  f"| {unit} | Readout | At the tag | Now |", "|---|---|---|---|"]
        lines += [f"| {k} | {name} | {_effect(old)} | {_effect(new)} |" for k, name, old, new in c.moved]
        lines.append("")
    if record.verdict is not None and c.values:
        lines += [f"**Fields not exactly the tag's, in rows on both sides: {len(c.values)}.**", "",
                  f"| {unit} | Field | At the tag | Now |", "|---|---|---|---|"]
        lines += [f"| {key} | {name} | {_value(old)} | {_value(new)} |" for key, name, old, new in c.values]
        lines.append("")
    for title, rows in (("Rows only at the tag", c.gone), ("Rows only now", c.new)):
        if rows:
            lines += [f"**{title}: {len(rows)}.**", "", f"| {unit} | Verdict | Effects |", "|---|---|---|"]
            lines += [f"| {key} | {row.get(record.verdict, '—') if record.verdict else '—'} | {_effects(row)} |"
                      for key, row in rows]
            lines.append("")
    return lines


def render(sections: list[tuple[Record, str, str, Comparison]]) -> str:
    lines = [
        "# Version 2 stage 1: what changed in the records",
        "",
        f"Generated by `tools/record_changes.py`; do not edit. Each generated record at the tag `{TAG}` against the",
        "record as committed now, after stage 1's one regeneration (`docs/implementation_plan_version_2.md` §1.6,",
        "S1.10b). Stage 1 changed no rule; a changed verdict here is a **measured change**, reported to the owner, not a",
        "defect. A row's verdict is its outcome (a criterion, or a criterion under a sweep setting) or whether the",
        "criterion caught the mutant (a mutation row). An effect **moved** when its mean difference changed by more than",
        "the half-width the tag's record stated; moved effects are listed only for rows whose verdict held. \"Differ at",
        "all\" counts every field of a row on both sides that is not exactly the tag's (the verdict, the seed count, the",
        "flags, each readout's mean difference, half-width and p), including a field on one side only; each is listed",
        "under its record, and \"No change\" means none differs. A record's hash line must differ from the tag's, or the",
        "tool refuses: an unchanged hash means it was not regenerated. A record with no row on both sides is refused too.",
        "",
        "| Record | Hash at the tag | Hash now | Rows compared | Verdicts changed | Effects moved | Differ at all | "
        "Only at the tag | Only now |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for record, h_then, h_now, c in sections:
        lines.append(f"| `{record.path}` | `{h_then[:12]}` | `{h_now[:12]}` | {c.compared} | {len(c.changed)} | "
                     f"{len(c.moved)} | {c.differ} | {len(c.gone)} | {len(c.new)} |")
    lines += ["", "In the level-occupancy record, which has no verdicts, \"Verdicts changed\" counts the rows whose "
              "values changed.", "", "**Not compared here:**", ""]
    lines += [f"- `{path}`: {why}." for path, why in NOT_COMPARED]
    for record, _, _, c in sections:
        lines += _section(record, c)
    return "\n".join(lines).rstrip("\n") + "\n"


def build() -> str:
    sections = []
    for record in RECORDS:
        then_text, now_text = tag_text(record.path), (REPO / record.path).read_text(encoding="utf-8")
        h_then, h_now = record_hash(then_text), record_hash(now_text)
        if h_then is None or h_now is None:
            raise SystemExit(f"{record.path}: no code_hash or engine_hash line, at the tag or now")
        if h_now == h_then:
            raise SystemExit(f"{record.path}: its hash is the tag's, so it was not regenerated on stage 1's engine; "
                             "regenerate it first")
        c = compare(record_json(then_text), record_json(now_text), record)
        if not c.compared:
            raise SystemExit(f"{record.path}: no row is on both sides, so nothing was compared")
        sections.append((record, h_then, h_now, c))
    return render(sections)


def main() -> int:
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
