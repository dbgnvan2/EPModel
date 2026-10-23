"""Where version 2 stands: every `M11.C` criterion with its verdict now, its verdict at the tag `version-1`, its
mutation evidence, its sweep fraction and its status under the stage 2 conditions W1–W4.

Purpose: generate ``docs/version_2_status.md`` from the records, so no status on it is hand-written (P6: a status is
         checked against the artifact it claims). On the model of ``tools/version_1_status.py``, which is frozen at
         the tag and whose ``record_rows`` no longer reads ``main``'s part II record; the live readers it still has
         (the spec's criteria table, the mutation records, the not-built lists, the dispositions) are imported from
         it and the rest is here. The tool holds no editorial content: what the owner decided, and why, is in
         ``docs/version_1_dispositions.md``; the sweep's rules are in ``config/sweep/sweep.md``.
Spec:    docs/implementation_plan_version_2.md#2.2 (D11), #2.3 (S2.8–S2.11); docs/bowen_agent_model_spec_v2.md#M11.5
Tests:   tests/test_version_2_status.py

The rules, version 1's (``docs/version_1_definition.md`` §1, approved 2026-10-15) carried into stage 2:
* A **pass** is final for a premise or check. For a composite or mixed criterion it is final only with a red (or
  reversed) mutant in a mutation record (W3).
* A **failing premise or check** is never final (W2). An `accepted finding` disposition on one is refused.
* A **failing composite or mixed** criterion is final when the owner has accepted it as a finding (W1).
* A criterion with no record row is **not built** unless the owner put it out of scope.
* Version 1's dispositions apply only where the verdict is the tag's. A row whose verdict differs from the verdict
  the frozen page records at the tag is **changed since the tag** and is not final until the owner reviews it; a
  criterion the tag's page does not have is new since the tag and is judged by the rules above alone.
* The **sweep column** (W4) shows *n of S held* for a swept criterion, *not swept* for a directional criterion not
  yet swept, and a dash for a premise, a check or a criterion that is not built. A fraction is a measurement and is
  never written as a pass or a fail (`M11.5`); W4 asks only that a passing directional criterion has one.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import tools.version_1_status as v1  # noqa: E402
from tools.record_json import record_json  # noqa: E402

PAGE = REPO / "docs" / "version_2_status.md"
TAG_PAGE = v1.PAGE                       # docs/version_1_status.md, frozen at the tag `version-1`
SWEEP_RECORD = REPO / "docs" / "version_2_sweep_record.md"
CHANGED = "changed since the tag `version-1`, pending the owner's review (T-122)"
CONDITIONS = (("W1", "every criterion has a final status"),
              ("W2", "every premise and check passes"),
              ("W3", "every passing composite has a recorded mutation"),
              ("W4", "every passing directional criterion has a sweep result"))
_TAG_ROW = re.compile(r"^\| `([^`]+)` \| [^|]* \| [^|]* \| ([^|]+) \|", re.M)  # class and record cells may be empty


@dataclass
class Status:
    key: str
    cls: str
    source: str
    verdict: str
    at_tag: str
    evidence: str
    sweep: str
    status: str
    final: bool
    blocks: set = field(default_factory=set)
    note: str = ""


def record_rows() -> list[tuple[str, str, dict]]:
    """(key, record name, row) for every verdict row. Every record prints each row under its own key (S1.7b), so the
    part II variants need no matching against the code's registry."""
    out = []
    for name, path in v1.ENSEMBLE_RECORDS:
        out += [(row["criterion"], name, row) for row in record_json(path)]
    seen = [k for k, _, _ in out]
    duplicates = sorted({k for k in seen if seen.count(k) > 1})
    if duplicates:
        raise SystemExit(f"a criterion has a verdict in more than one record: {duplicates}")
    return out


def tag_verdicts(path: Path = TAG_PAGE) -> dict[str, str]:
    """Criterion key -> its verdict on the frozen page (PASS, FAIL, UNDETERMINED or 'not built')."""
    text = path.read_text(encoding="utf-8")
    table = text.split("## Every criterion", 1)
    if len(table) != 2:
        raise SystemExit(f"{path.name}: no '## Every criterion' table")
    out = {key: verdict.strip() for key, verdict in _TAG_ROW.findall(table[1])}
    rows = sum(1 for line in table[1].splitlines() if line.startswith("| `"))
    if not out or len(out) != rows:
        raise SystemExit(f"{path.name}: {rows} criterion rows in the table, {len(out)} parsed; a row the pattern does "
                         "not read would vanish from the page (P2)")
    return out


def sweep_fractions(path: Path = SWEEP_RECORD) -> dict[str, str]:
    """Criterion key -> its sweep cell, *n of S held*, from the sweep record's machine-readable block; empty when the
    record does not exist yet. An errored count is shown, never folded into S (P2)."""
    if not path.exists():
        return {}
    out = {}
    for result in record_json(path):
        cell = f"{result['held']} of {result['counted']} held"
        if result["errored"]:
            cell += f" ({result['errored']} errored, not counted)"
        out[result["criterion"]] = cell
    return out


def statuses() -> list[Status]:
    spec = v1.spec_criteria()
    rows = record_rows()
    reds = v1.red_mutants()
    unbuilt = v1.not_built()
    by_test = v1.proved_by_test()
    decided = v1.dispositions()
    limbs = v1.premise_limbs()
    tag = tag_verdicts()
    swept = sweep_fractions()
    out: list[Status] = []
    covered = set()

    def base(key: str) -> str:
        return key.split("[", 1)[0]

    stale = sorted(v1.registered() - {k for k, _, _ in rows})
    if stale:
        raise SystemExit(f"registered in the code with no record row (the records are stale; regenerate): {stale}")
    variant_bases = {base(k) for k, _, _ in rows if "[" in k} - {k for k, _, _ in rows}
    misplaced = sorted(set(decided) & variant_bases)
    if misplaced:
        raise SystemExit(f"{v1.DISPOSITIONS.name}: {misplaced} name a criterion whose rows are all variants; name each variant")

    for key, source, row in rows:
        cid = base(key)
        if cid not in spec:
            raise SystemExit(f"{key}: in {source} but not in the spec's criteria table")
        cls = row["class"]
        if spec[cid] and spec[cid] != cls:
            raise SystemExit(f"{key}: class {cls!r} in {source}, {spec[cid]!r} in the spec's M11.5 table")
        covered.add(cid)
        at_tag = tag.get(key, "—")
        same = at_tag == row["outcome"]
        d_tag = decided.get(key)
        d = d_tag if same else None  # version 1's disposition applies only where the verdict is the tag's
        sweep = swept.get(key) or ("not swept" if cls in v1.DIRECTIONAL else "—")
        s = Status(key, cls, source, row["outcome"], at_tag, v1._readouts(row), sweep, "", False)
        proof = reds.get(key, [])
        if proof:
            s.evidence += "; proved by " + ", ".join(proof)
        if row["outcome"] == "PASS":
            if d and d["kind"] != "note":
                raise SystemExit(f"{key}: passes, as at the tag, so its disposition {d['kind']!r} does not apply")
            if cls in v1.DIRECTIONAL and not proof:
                s.status, s.blocks = "pass, no recorded mutation", {"W1", "W3"}
            else:
                s.status, s.final = "pass", True
            if cls in v1.DIRECTIONAL and key not in swept:
                s.blocks.add("W4")
        else:
            declared = limbs.get(key, limbs.get(cid, frozenset()))
            failed_limbs = sorted(r["readout"] for r in row["readouts"]
                                  if r["readout"] in declared and not r.get("report_only") and not r.get("holds"))
            premise_failed = cls in v1.PREMISE_LIKE or (cls == "mixed" and failed_limbs)
            if d and d["kind"] == "out of scope":
                raise SystemExit(f"{key}: is built and has a verdict in {source}, so 'out of scope' does not apply")
            if d and d["kind"] == "withdrawn":
                s.status, s.final = ("withdrawn by the owner", True) if d["owner"] else \
                    ("withdrawal pending the owner's review", False)
                if not d["owner"]:
                    s.blocks = {"W1", "W2"} if premise_failed else {"W1"}
            elif premise_failed:
                if d and d["kind"] == "accepted finding":
                    what = cls if cls in v1.PREMISE_LIKE else f"mixed criterion's premise limb ({', '.join(failed_limbs)})"
                    raise SystemExit(f"{key}: a failing {what} cannot be accepted as a finding (W2)")
                s.status = f"fail ({cls})" if cls in v1.PREMISE_LIKE else f"fail (premise limb {', '.join(failed_limbs)})"
                s.blocks = {"W1", "W2"}
            elif d and d["kind"] == "accepted finding":
                s.status, s.final = ("fail accepted as a finding", True) if d["owner"] else \
                    ("fail, acceptance pending the owner's review", False)
                if not d["owner"]:
                    s.blocks = {"W1"}
            else:
                s.status, s.blocks = "fail, no disposition", {"W1"}
        if not same and at_tag != "—":
            s.status, s.final, s.blocks = f"{s.status}; {CHANGED}", False, s.blocks | {"W1"}
        if d:
            s.note = f"{d['reason']} *({d['decided']})*"
        elif d_tag:  # shown, so the owner sees the decision the new verdict reverses; not applied
            why = "the verdict changed" if at_tag != "—" else "no row at the tag"
            s.note = f"version 1 disposition, not applied ({why}): {d_tag['kind']} — {d_tag['reason']} *({d_tag['decided']})*"
        out.append(s)

    current = {k for k, _, _ in rows}
    gone = sorted(k for k in tag if k not in current and (base(k) not in spec or base(k) in covered))
    if gone:
        raise SystemExit(f"rows at the tag `version-1` with no row now: {gone}; a removed variant or criterion must be "
                         "shown, not dropped (P2)")

    for cid, cls in spec.items():
        if cid in covered:
            continue
        d = decided.get(cid)
        at_tag = tag.get(cid) or "; ".join(f"{k} {v}" for k, v in tag.items() if base(k) == cid) or "—"
        if cid in by_test:
            s = Status(cid, cls, "test", "PASS", at_tag, f"proved by {by_test[cid]}", "—", "pass, proved by a test", True)
        elif d and d["kind"] == "out of scope":
            s = Status(cid, cls, "—", "not built", at_tag, unbuilt.get(cid, "no record lists it"), "—",
                       "out of scope" if d["owner"] else "out of scope, pending the owner's review", d["owner"])
            if not d["owner"]:
                s.blocks = {"W1"}
        elif d and d["kind"] == "withdrawn":
            s = Status(cid, cls, "—", "not built", at_tag, unbuilt.get(cid, "no record lists it"), "—",
                       "withdrawn by the owner" if d["owner"] else "withdrawal pending the owner's review", d["owner"])
            if not d["owner"]:
                s.blocks = {"W1", "W2"} if cls in v1.PREMISE_LIKE else {"W1"}
        else:
            if d and d["kind"] not in ("note",):
                raise SystemExit(f"{cid}: has no record row, so its disposition {d['kind']!r} does not apply")
            why = unbuilt.get(cid, "**no record lists it**")
            s = Status(cid, cls or "—", "—", "not built", at_tag, why, "—", "not built", False,
                       {"W1", "W2"} if cls in v1.PREMISE_LIKE else {"W1"})
        if d:
            s.note = f"{d['reason']} *({d['decided']})*"
        out.append(s)

    known = {s.key for s in out} | set(spec)
    stray = sorted((set(decided) | set(limbs)) - known)
    if stray:
        raise SystemExit(f"{v1.DISPOSITIONS.name} names criteria no record or spec row knows: {stray}")
    return sorted(out, key=lambda s: (int(base(s.key).split(".")[2]), s.key))


def render(rows: list[Status]) -> str:
    def blocked(condition: str) -> list[str]:
        return [f"`{s.key}`" for s in rows if condition in s.blocks]

    hashes = []
    for name, path in (*v1.ENSEMBLE_RECORDS, *v1.MUTATION_RECORDS, ("Stage 2 sweep record", SWEEP_RECORD)):
        if not path.exists():
            hashes.append(f"- {name} (`{path.relative_to(REPO)}`): not yet generated")
            continue
        # every hash the record's rows are keyed by: the sweep record's Phase D rows carry ``phase_d_engine_hash``
        found = re.findall(r"^(code_hash|engine_hash|phase_d_engine_hash): (\w+)", path.read_text(encoding="utf-8"),
                           re.M)
        keyed = ", ".join(f"{kind} `{value[:12]}`" for kind, value in found)
        hashes.append(f"- {name} (`{path.relative_to(REPO)}`): {keyed}")
    hashes.append(f"- the tag's page (`{TAG_PAGE.relative_to(REPO)}`): the verdicts at `version-1`")
    lines = [
        "# Version 2 status",
        "",
        "Generated by `tools/version_2_status.py`; do not edit. One row per `M11.C` criterion or variant, read from the",
        "records below and from `docs/version_1_dispositions.md`, against the stage 2 conditions W1–W4",
        "(`docs/implementation_plan_version_2.md` §2.3). A verdict that is not the one the frozen page records at the",
        "tag `version-1` is **changed since the tag** and is not final until the owner reviews it (T-122); a criterion",
        "the tag's page does not have is new since the tag. The **sweep** column is a measurement, *n of S* samples",
        "over the declared ranges in which the direction held (`docs/version_2_sweep_record.md`), never a verdict",
        "(`M11.5`); a premise or check is not swept.",
        "",
        "Records read:",
        *hashes,
        "",
        "## Conditions",
        "",
    ]
    for condition, text in CONDITIONS:
        held = blocked(condition)
        state = "**met**" if not held else f"**not met**, {len(held)} rows: " + ", ".join(held)
        lines.append(f"- {condition} ({text}): {state}.")
    counts: dict[str, int] = {}
    for s in rows:
        counts[s.status] = counts.get(s.status, 0) + 1
    lines += ["", "Rows by status:", *[f"- {k}: {v}" for k, v in sorted(counts.items())], "",
              "## Every criterion", "",
              "| Criterion | Class | Record | Verdict | At `version-1` | Evidence | Sweep | Status | Final | Blocks | Disposition |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in rows:
        lines.append(f"| `{s.key}` | {s.cls} | {s.source} | {s.verdict} | {s.at_tag} | {s.evidence} | {s.sweep} | "
                     f"{s.status} | {'yes' if s.final else 'no'} | {', '.join(sorted(s.blocks)) or '—'} | {s.note or '—'} |")
    return "\n".join(lines) + "\n"


def main() -> NoReturn:
    PAGE.write_text(render(statuses()), encoding="utf-8")
    print(f"wrote {PAGE.relative_to(REPO)}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
