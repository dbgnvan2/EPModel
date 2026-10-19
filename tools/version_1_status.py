"""Write the version 1 status table: every `M11.C` criterion with its final status under V1, V2 and V3.

Purpose: read the three ensemble records, the two mutation records, the records' not-built lists and the owner's
         dispositions (docs/version_1_dispositions.md), and write docs/version_1_status.md, one row per criterion or
         variant, with whether each of V1, V2 and V3 is met. Frozen since 2026-10-19: this copy no longer writes, and
         no test runs it; the tests run the tag's copy (below).
Spec:    docs/version_1_definition.md#1 (V1, V2, V3); docs/bowen_agent_model_spec_v2.md#M11.5
Tests:   tests/test_version_1_status.py

Frozen (owner 2026-10-19, `docs/version_2_plan.md` stage 0.1). Version 1 ends at the tag `version-1`, and the page
and the dispositions are kept as they stand there. Version 2 regenerates the records this reads, so `main` refuses to
write. `tests/test_version_1_status.py` checks the page and the dispositions against the tag, and runs the tag's own
tests of this tool in a copy of the tagged tree. To render the page, check out the tag and run

    python3 tools/version_1_status.py

There it is generated, never hand-edited, and a default-suite test fails when the committed page is not what this
renders. The tool holds no editorial content: what the owner decided, and why, is in the dispositions file.

The rules, from the version 1 definition (approved 2026-10-15):
* A **pass** is final for a premise or check (§4 question 5: an ad hoc mutation reported in the plan is enough). For a
  composite or mixed criterion it is final only with a red (or reversed) mutant in a mutation record (V3).
* A **failing premise or check** is never final: it must pass, or the owner restates or withdraws it (V2). An
  `accepted finding` disposition on one is refused.
* A **failing composite or mixed** criterion is final when the owner has accepted it as a finding (V1). A
  disposition that is Claude's recommendation pending owner review is shown, and is not final.
* A criterion with no record row is **not built** unless the owner put it out of scope. A criterion registered in
  the code with no record row means the records are stale, and the tool refuses to run.
* A **mixed** criterion's premise limbs are declared in the dispositions file's second table, from `M11.5`. A failing
  mixed criterion whose declared premise limb does not hold is treated as a failing premise (V2).
* For V3 a red on any gating readout counts; the tool does not know which limb a mutant reached.
* A disposition that cannot apply where it stands (a kind the row's state does not admit, a base id whose rows are all
  variants, a row after a break in the table) raises rather than being skipped (P2).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SPEC = REPO / "docs" / "bowen_agent_model_spec_v2.md"
DISPOSITIONS = REPO / "docs" / "version_1_dispositions.md"
PAGE = REPO / "docs" / "version_1_status.md"
ENSEMBLE_RECORDS = (
    ("Phase C", REPO / "docs" / "phase_c_ensemble_record.md"),
    ("Phase D part I", REPO / "docs" / "phase_d_ensemble_record.md"),
    ("Phase D part II", REPO / "docs" / "phase_d2_ensemble_record.md"),
)
MUTATION_RECORDS = (
    ("Phase C mutation record", REPO / "docs" / "phase_c_mutation_record.md"),
    ("Phase D mutation record", REPO / "docs" / "phase_d_mutation_record.md"),
)
DISPOSITION_KINDS = ("accepted finding", "out of scope", "withdrawn", "note")
OWNER = re.compile(r"^owner \d{4}-\d{2}-\d{2}$")
PENDING = re.compile(r"^Claude's recommendation \d{4}-\d{2}-\d{2}, pending owner review$")
DIRECTIONAL = ("composite", "mixed")   # classes whose pass needs a recorded mutation (V3)
PREMISE_LIKE = ("premise", "check")    # classes whose failure is a defect (V2)
RED = ("red", "reversed")


def _json_block(path: Path) -> list[dict]:
    return json.loads(re.search(r"```json\n(.*?)\n```", path.read_text(encoding="utf-8"), re.S).group(1))


def _load(module: str, path: Path):
    spec = importlib.util.spec_from_file_location(module, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def spec_criteria() -> dict[str, str]:
    """Each `M11.C` id in the spec's criteria table, with its `M11.5` class (lower case; '' where it has none)."""
    text = SPEC.read_text(encoding="utf-8")
    ids = sorted(set(re.findall(r"^\| \*\*(M11\.C\.\d+)\*\* \|", text, re.M)), key=lambda c: int(c.split(".")[2]))
    classes = {cid: cls.lower() for cid, cls in
               re.findall(r"^\| (M11\.C\.\d+) \| (Premise|Composite|Check|Mixed)\b", text, re.M)}
    return {cid: classes.get(cid, "") for cid in ids}


def record_rows() -> list[tuple[str, str, dict]]:
    """(key, record name, row) for every verdict row. The part II record prints two variants under one id, so its rows
    are matched to `CRITERIA_D2`'s keys in order, each id checked (as `tools/phase_d_mutation_record.py` does)."""
    from src.bowen.ensemble.criteria_d2 import CRITERIA_D2

    out = []
    for name, path in ENSEMBLE_RECORDS:
        rows = _json_block(path)
        if name == "Phase D part II":
            keys = list(CRITERIA_D2)
            if len(keys) != len(rows) or any(CRITERIA_D2[k].id != r["criterion"] for k, r in zip(keys, rows)):
                raise SystemExit(f"{path.name}'s rows no longer match CRITERIA_D2 in order; regenerate it")
        else:
            keys = [r["criterion"] for r in rows]
        out += [(key, name, row) for key, row in zip(keys, rows)]
    seen = [k for k, _, _ in out]
    duplicates = sorted({k for k in seen if seen.count(k) > 1})
    if duplicates:
        raise SystemExit(f"a criterion has a verdict in more than one record: {duplicates}")
    return out


def red_mutants() -> dict[str, list[str]]:
    """Criterion key -> the red or reversed mutants recorded against it, with the record's name."""
    out: dict[str, list[str]] = {}
    for name, path in MUTATION_RECORDS:
        for row in _json_block(path):
            if row["result"] in RED:
                out.setdefault(row["criterion"], []).append(f"`{row['mutant']}` {row['result']} ({name})")
    return out


def not_built() -> dict[str, str]:
    """Criterion id -> why it has no arm, from the records' own lists. A criterion one list calls not built and
    another record runs (`M11.C.14`: not built in Phase C, built in Phase D) is decided by its record row."""
    from src.bowen.ensemble.criteria import NOT_BUILT as PHASE_C

    phase_d = _load("tools.phase_d_record", REPO / "tools" / "phase_d_record.py")
    out = dict(PHASE_C)
    out.update(phase_d.NOT_BUILT)
    return out


def proved_by_test() -> dict[str, str]:
    """Criterion id -> the test that proves it, for a criterion run as a test rather than an ensemble (`M11.C.37`).
    The named test must exist (P6: the claim is checked against the thing it names)."""
    phase_d = _load("tools.phase_d_record", REPO / "tools" / "phase_d_record.py")
    out = {}
    for cid, why in phase_d.NOT_RUN_2.items():
        found = re.search(r"`(tests/[\w/]+\.py)::(test_\w+)`", why)
        if not found:
            raise SystemExit(f"{cid}: NOT_RUN_2 names no test")
        path, test = found.groups()
        if f"def {test}(" not in (REPO / path).read_text(encoding="utf-8"):
            raise SystemExit(f"{cid}: {path}::{test} does not exist")
        out[cid] = f"`{path}::{test}`"
    return out


PREMISE_LIMBS_HEADER = "| criterion | premise readouts |"


def premise_limbs() -> dict[str, frozenset[str]]:
    """Criterion id or key -> the readouts that are a mixed criterion's premise limbs (the dispositions file's second
    table, from `M11.5`). A base id applies to every variant; a readout that is report-only in a variant is not read."""
    text = DISPOSITIONS.read_text(encoding="utf-8")
    if text.count(PREMISE_LIMBS_HEADER) != 1:
        raise SystemExit(f"{DISPOSITIONS.name}: the premise-limbs table header is missing or repeated")
    out = {}
    for line in text.split(PREMISE_LIMBS_HEADER, 1)[1].splitlines()[2:]:
        if not line.startswith("|"):
            break
        key, names = (c.strip() for c in line.strip().strip("|").split("|"))
        out[key.strip("`")] = frozenset(n.strip().strip("`") for n in names.split(","))
    return out


def registered() -> set[str]:
    """Every criterion key the code registers."""
    from src.bowen.ensemble.criteria import CRITERIA
    from src.bowen.ensemble.criteria_d import CRITERIA_D
    from src.bowen.ensemble.criteria_d2 import CRITERIA_D2

    return set(CRITERIA) | set(CRITERIA_D) | set(CRITERIA_D2)


def dispositions() -> dict[str, dict]:
    text = DISPOSITIONS.read_text(encoding="utf-8")
    header = "| criterion | disposition | decided | reason |"
    if text.count(header) != 1:
        raise SystemExit(f"{DISPOSITIONS.name}: the table header is missing or repeated")
    out = {}
    lines = text.split(header, 1)[1].split(PREMISE_LIMBS_HEADER, 1)[0].splitlines()[2:]
    end = next((i for i, line in enumerate(lines) if not line.startswith("|")), len(lines))
    after = [line for line in lines[end:] if line.startswith("|")]
    if after:
        raise SystemExit(f"{DISPOSITIONS.name}: a row after a break in the table would be skipped: {after[0][:60]}")
    for line in lines[:end]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 4:
            raise SystemExit(f"{DISPOSITIONS.name}: a row does not have four cells: {line[:80]}")
        key, kind, decided, reason = cells
        key = key.strip("`")
        if kind not in DISPOSITION_KINDS:
            raise SystemExit(f"{key}: unknown disposition {kind!r}")
        if not (OWNER.match(decided) or PENDING.match(decided)):
            raise SystemExit(f"{key}: 'decided' must be 'owner DATE' or the pending form, not {decided!r}")
        if key in out:
            raise SystemExit(f"{key}: two disposition rows")
        out[key] = {"kind": kind, "owner": bool(OWNER.match(decided)), "decided": decided, "reason": reason}
    return out


@dataclass
class Status:
    key: str
    cls: str
    source: str
    verdict: str
    evidence: str
    status: str
    final: bool
    blocks: set = field(default_factory=set)
    note: str = ""


def _number(x: float) -> str:
    return f"{x:+.3g}"


def _readouts(row: dict) -> str:
    gating = [r for r in row["readouts"] if not r.get("report_only")]
    return "; ".join(f"`{r['readout']}` {_number(r['mean_difference'])} ± {abs(r['half_width']):.2g}" for r in gating)


def statuses() -> list[Status]:
    spec = spec_criteria()
    rows = record_rows()
    reds = red_mutants()
    unbuilt = not_built()
    by_test = proved_by_test()
    decided = dispositions()
    limbs = premise_limbs()
    out: list[Status] = []
    covered = set()

    def base(key: str) -> str:
        return key.split("[", 1)[0]

    stale = sorted(registered() - {k for k, _, _ in rows})
    if stale:
        raise SystemExit(f"registered in the code with no record row (the records are stale; regenerate): {stale}")
    variant_bases = {base(k) for k, _, _ in rows if "[" in k} - {k for k, _, _ in rows}
    misplaced = sorted(set(decided) & variant_bases)
    if misplaced:
        raise SystemExit(f"{DISPOSITIONS.name}: {misplaced} name a criterion whose rows are all variants; name each variant")

    for key, source, row in rows:
        cid = base(key)
        if cid not in spec:
            raise SystemExit(f"{key}: in {source} but not in the spec's criteria table")
        cls = row["class"]
        if spec[cid] and spec[cid] != cls:
            raise SystemExit(f"{key}: class {cls!r} in {source}, {spec[cid]!r} in the spec's M11.5 table")
        covered.add(cid)
        d = decided.get(key)
        s = Status(key, cls, source, row["outcome"], _readouts(row), "", False)
        if row["outcome"] == "PASS":
            if d and d["kind"] != "note":
                raise SystemExit(f"{key}: passes, so its disposition {d['kind']!r} no longer applies; remove it")
            proof = reds.get(key, [])
            if cls in DIRECTIONAL and not proof:
                s.status, s.blocks = "pass, no recorded mutation", {"V1", "V3"}
            else:
                s.status, s.final = "pass", True
                if proof:
                    s.evidence += "; proved by " + ", ".join(proof)
        else:
            if d and d["kind"] == "out of scope":
                raise SystemExit(f"{key}: is built and has a verdict in {source}, so 'out of scope' does not apply")
            declared = limbs.get(key, limbs.get(cid, frozenset()))
            failed_limbs = sorted(r["readout"] for r in row["readouts"]
                                  if r["readout"] in declared and not r.get("report_only") and not r.get("holds"))
            premise_failed = cls in PREMISE_LIKE or (cls == "mixed" and failed_limbs)
            if d and d["kind"] == "withdrawn":
                s.status, s.final = ("withdrawn by the owner", True) if d["owner"] else \
                    ("withdrawal pending owner review", False)
                if not d["owner"]:
                    s.blocks = {"V1", "V2"} if premise_failed else {"V1"}
            elif premise_failed:
                if d and d["kind"] == "accepted finding":
                    what = cls if cls in PREMISE_LIKE else f"mixed criterion's premise limb ({', '.join(failed_limbs)})"
                    raise SystemExit(f"{key}: a failing {what} cannot be accepted as a finding (V2)")
                s.status = f"fail ({cls})" if cls in PREMISE_LIKE else f"fail (premise limb {', '.join(failed_limbs)})"
                s.blocks = {"V1", "V2"}
            elif d and d["kind"] == "accepted finding":
                s.status, s.final = ("fail accepted as a finding", True) if d["owner"] else \
                    ("fail, acceptance pending owner review", False)
                if not d["owner"]:
                    s.blocks = {"V1"}
            else:
                s.status, s.blocks = "fail, no disposition", {"V1"}
        if d:
            s.note = f"{d['reason']} *({d['decided']})*"
        out.append(s)

    for cid, cls in spec.items():
        if cid in covered:
            continue
        d = decided.get(cid)
        if cid in by_test:
            s = Status(cid, cls, "test", "PASS", f"proved by {by_test[cid]}", "pass, proved by a test", True)
        elif d and d["kind"] == "out of scope":
            s = Status(cid, cls, "—", "not built", unbuilt.get(cid, "no record lists it"),
                       "out of scope" if d["owner"] else "out of scope, pending owner review", d["owner"])
            if not d["owner"]:
                s.blocks = {"V1"}
        elif d and d["kind"] == "withdrawn":
            s = Status(cid, cls, "—", "not built", unbuilt.get(cid, "no record lists it"),
                       "withdrawn by the owner" if d["owner"] else "withdrawal pending owner review", d["owner"])
            if not d["owner"]:
                s.blocks = {"V1", "V2"} if cls in PREMISE_LIKE else {"V1"}
        else:
            if d and d["kind"] not in ("note",):
                raise SystemExit(f"{cid}: has no record row, so its disposition {d['kind']!r} does not apply")
            why = unbuilt.get(cid, "**no record lists it**")
            s = Status(cid, cls or "—", "—", "not built", why, "not built", False,
                       {"V1", "V2"} if cls in PREMISE_LIKE else {"V1"})
        if d:
            s.note = f"{d['reason']} *({d['decided']})*"
        out.append(s)

    known = {s.key for s in out} | set(spec)
    stray = sorted((set(decided) | set(limbs)) - known)
    if stray:
        raise SystemExit(f"{DISPOSITIONS.name} names criteria no record or spec row knows: {stray}")
    return sorted(out, key=lambda s: (int(base(s.key).split(".")[2]), s.key))


def render(rows: list[Status]) -> str:
    def blocked(condition: str) -> list[str]:
        return [f"`{s.key}`" for s in rows if condition in s.blocks]

    hashes = []
    for name, path in (*ENSEMBLE_RECORDS, *MUTATION_RECORDS):
        found = re.search(r"^(code_hash|engine_hash): (\w+)", path.read_text(encoding="utf-8"), re.M)
        hashes.append(f"- {name} (`{path.relative_to(REPO)}`): {found.group(1)} `{found.group(2)[:12]}`")
    lines = [
        "# Version 1 status",
        "",
        "Generated by `tools/version_1_status.py`; do not edit. One row per `M11.C` criterion or variant, read from the",
        "ensemble and mutation records below and from `docs/version_1_dispositions.md`, against the conditions of",
        "`docs/version_1_definition.md` §1. V4–V7 are not read here: V4 is the spec's status line, V5 and V6 are the",
        "suite and CI, V7 is the owner's review of the outputs.",
        "",
        "Records read:",
        *hashes,
        "",
        "## Conditions",
        "",
    ]
    for condition, text in (("V1", "every criterion has a final status"),
                            ("V2", "every premise and check passes"),
                            ("V3", "every passing composite has a recorded mutation")):
        held = blocked(condition)
        state = "**met**" if not held else f"**not met**, {len(held)} rows: " + ", ".join(held)
        lines.append(f"- {condition} ({text}): {state}.")
    counts: dict[str, int] = {}
    for s in rows:
        counts[s.status] = counts.get(s.status, 0) + 1
    lines += ["", "Rows by status: " + "; ".join(f"{k} {v}" for k, v in sorted(counts.items())) + ".", "",
              "## Every criterion", "",
              "| Criterion | Class | Record | Verdict | Evidence | Status | Final | Blocks | Disposition |",
              "|---|---|---|---|---|---|---|---|---|"]
    for s in rows:
        lines.append(f"| `{s.key}` | {s.cls} | {s.source} | {s.verdict} | {s.evidence} | {s.status} | "
                     f"{'yes' if s.final else 'no'} | {', '.join(sorted(s.blocks)) or '—'} | {s.note or '—'} |")
    return "\n".join(lines) + "\n"


def main() -> NoReturn:
    raise SystemExit(f"{PAGE.relative_to(REPO)} is frozen at the tag version-1 (owner 2026-10-19); "
                     "render it from a checkout of the tag")


if __name__ == "__main__":
    main()
