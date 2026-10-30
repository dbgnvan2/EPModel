"""Where version 2 stands: every `M11.C` criterion with its verdict now, its verdict at the tag `version-1` and at
spec revision 38, its mutation evidence, its sweep fraction and its status under the stage 2 conditions W1–W4.

Purpose: generate ``docs/version_2_status.md`` from the records, so no status on it is hand-written (P6: a status is
         checked against the artifact it claims). On the model of ``tools/version_1_status.py``, which is frozen at
         the tag and whose ``record_rows`` no longer reads ``main``'s part II record; the live readers it still has
         (the spec's criteria table, the mutation records, the not-built lists, the dispositions) are imported from
         it and the rest is here. The tool holds no editorial content: what the owner decided, and why, is in
         ``docs/version_1_dispositions.md`` (frozen at the tag) and ``docs/version_2_dispositions.md``; the sweep's
         rules are in ``config/sweep/sweep.md``.
Spec:    docs/implementation_plan_version_2.md#2.2 (D11), #2.3 (S2.8–S2.11); docs/bowen_agent_model_spec_v2.md#M11.5;
         docs/implementation_plan_version_2_stage_3.md (S3.24: each verdict beside version 1's and revision 38's);
         docs/version_2_definition_plan.md §3 (version 2's dispositions)
Tests:   tests/test_version_2_status.py

The rules, version 1's (``docs/version_1_definition.md`` §1, approved 2026-10-15) carried into stage 2:
* A **pass** is final for a premise or check. For a composite or mixed criterion it is final only with a red (or
  reversed) mutant in a mutation record (W3). Without one it is *pass, named mutant survived* when a named mutant of
  its ran and survived, *pass, mutant survived* when only a deletion or sign-inverted one did (the survivor named in
  the evidence either way), and *pass, no recorded mutation* only when no mutation record holds a mutant of it
  (milestone review MR-28).
* A **failing premise or check** is never final (W2). An `accepted finding` disposition on one is refused.
* A **failing composite or mixed** criterion is final when the owner has accepted it as a finding (W1).
* A criterion with no record row is **not built** unless the owner put it out of scope.
* Version 1's dispositions apply only where the verdict is the tag's. A row whose verdict differs from the verdict
  the frozen page records at the tag is **changed since the tag** and is not final until the owner reviews it; a
  criterion the tag's page does not have is new since the tag and is judged by the rules above alone.
* **Version 2's dispositions** (``docs/version_2_dispositions.md``) each name the verdict they were decided on, and
  apply only while the row's verdict is that one; a row whose verdict has since moved shows the disposition as not
  applied (P6: a decision on one result is not a decision on the next). Where one applies it takes precedence over
  version 1's. Its kinds are version 1's `accepted finding` and `note`, and `verdict reviewed`: the owner's review
  of a verdict that is not the tag's. A version 2 disposition the owner made, of any kind, is the owner's review of
  the verdict it names, so the row is no longer held as changed since the tag; one that is Claude's recommendation
  is shown and leaves the row as it was (``docs/version_2_definition_plan.md`` §3).
* The **revision 38 column** shows the verdict this page gave at commit ``REVISION_38`` (the records as revision 38
  left them, before stage 3's regeneration), read from git; it is shown, never used for a status: a verdict that
  differs from it is a new result under revision 39's rules (`M11.F`), and the rules above still read only the tag.
* The **sweep column** (W4) shows *n of S held* for a swept criterion, *not swept* for a directional criterion not
  yet swept, *refused at the rerun* for one the sweep refused (`M17.A.1`; it has no fraction, so W4 still counts it),
  and a dash for a premise, a check or a criterion that is not built. A fraction is a measurement and is
  never written as a pass or a fail (`M11.5`); W4 asks only that a passing directional criterion has one. A
  fraction from a sweep record made on an engine that is not the current one (its ``engine_hash`` or
  ``phase_d_engine_hash`` line differs) is shown with the hash it was measured on and does not count for W4: it
  measured rules the engine no longer has (P6; the review of 2026-10-26, stage 3 step 10).
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
from tools.record_changes import REVISION_38, ref_text  # noqa: E402
from tools.record_json import record_json  # noqa: E402

PAGE = REPO / "docs" / "version_2_status.md"
TAG_PAGE = v1.PAGE                       # docs/version_1_status.md, frozen at the tag `version-1`
SWEEP_RECORD = REPO / "docs" / "version_2_sweep_record.md"
CHANGED = "changed since the tag `version-1`, pending the owner's review (T-122)"
REVIEWED = "changed since the tag `version-1`, reviewed by the owner"
V2_DISPOSITIONS = REPO / "docs" / "version_2_dispositions.md"
V2_HEADER = "| criterion | disposition | verdict | decided | reason |"
V2_KINDS = ("accepted finding", "verdict reviewed", "note")
STALE_CELL = "earlier engine; not counted for W4"
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
    at_revision_38: str = ""


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


def survived_mutants() -> dict[str, list[tuple[str, str]]]:
    """Criterion key -> (kind, "`mutant` (record name)") for every mutant recorded against it that ran and survived,
    from the mutation records ``version_1_status`` reads (MR-28: a survivor is not "no recorded mutation")."""
    out: dict[str, list[tuple[str, str]]] = {}
    for name, path in v1.MUTATION_RECORDS:
        for row in record_json(path):
            if row["result"] == "survived":
                out.setdefault(row["criterion"], []).append((row.get("kind", ""), f"`{row['mutant']}` ({name})"))
    return out


def tag_verdicts(path: Path = TAG_PAGE) -> dict[str, str]:
    """Criterion key -> its verdict on the frozen page (PASS, FAIL, UNDETERMINED or 'not built')."""
    return page_verdicts(path.read_text(encoding="utf-8"), path.name)


def revision_38_verdicts() -> dict[str, str]:
    """Criterion key -> its verdict on this page as committed at ``REVISION_38``, read from git (S3.24). A missing ref
    fails; it is never skipped (CI checks out with fetch-depth 0)."""
    return page_verdicts(ref_text(REVISION_38, PAGE.relative_to(REPO).as_posix()), f"{PAGE.name} at {REVISION_38}")


def page_verdicts(text: str, name: str) -> dict[str, str]:
    """Criterion key -> the fourth cell of each row of a status page's '## Every criterion' table, its verdict."""
    path = Path(name)
    table = text.split("## Every criterion", 1)
    if len(table) != 2:
        raise SystemExit(f"{path.name}: no '## Every criterion' table")
    out = {key: verdict.strip() for key, verdict in _TAG_ROW.findall(table[1])}
    rows = sum(1 for line in table[1].splitlines() if line.startswith("| `"))
    if not out or len(out) != rows:
        raise SystemExit(f"{path.name}: {rows} criterion rows in the table, {len(out)} parsed; a row the pattern does "
                         "not read would vanish from the page (P2)")
    return out


def sweep_stale(path: Path = SWEEP_RECORD) -> str | None:
    """Why the sweep record's fractions are not of the current engine, or None when they are (or there is no record).
    Each hash the record prints is compared with the one its rows are keyed by now; a record printing neither is
    refused, since its engine cannot be checked."""
    if not path.exists():
        return None
    from tools.phase_d_mutation_record import engine_hash as phase_d_engine_hash
    from tools.record_cache import engine_hash

    printed = dict(re.findall(r"^(engine_hash|phase_d_engine_hash): (\w+)$", path.read_text(encoding="utf-8"), re.M))
    if not printed:
        raise SystemExit(f"{path.name}: no engine_hash line, so its fractions cannot be checked against the engine")
    now = {"engine_hash": engine_hash, "phase_d_engine_hash": phase_d_engine_hash}
    moved = [f"{kind} `{value[:12]}`" for kind, value in printed.items() if value != now[kind]()]
    if not moved:
        return None
    return (f"measured on {', '.join(moved)}, not the current engine; not counted for W4 until the sweep is rerun")


def sweep_fractions(path: Path = SWEEP_RECORD) -> dict[str, str]:
    """Criterion key -> its sweep cell, *n of S held*, from the sweep record's machine-readable block; empty when the
    record does not exist yet. An errored count is shown, never folded into S (P2). A criterion the record shows as
    declared, not run has no fraction and is left out, so its cell says *not swept* and W4 still counts it."""
    if not path.exists():
        return {}
    out = {}
    for result in record_json(path):
        if "not_run" in result or "refused" in result:
            continue
        cell = f"{result['held']} of {result['counted']} held"
        if result["errored"]:
            cell += f" ({result['errored']} errored, not counted)"
        out[result["criterion"]] = cell
    return out


def sweep_refused(path: Path = SWEEP_RECORD) -> dict[str, str]:
    """Criterion key -> its sweep cell, for a criterion the sweep record shows as refused at the rerun (`M17.A.1`;
    T-159, owner 2026-10-29). It has no fraction, so ``sweep_fractions`` leaves it out and W4 still counts it."""
    if not path.exists():
        return {}
    return {r["criterion"]: f"refused at the rerun (`M17.A.1`), sample {r['refused']['sample']}"
            for r in record_json(path) if "refused" in r}


def v2_dispositions(path: Path = V2_DISPOSITIONS) -> dict[str, dict]:
    """Purpose: read version 2's dispositions, strictly: one table, five cells a row, a known kind, a verdict of
    PASS or FAIL, an `accepted finding` only on a FAIL, `decided` in version 1's two forms, one row a key.
    Spec:    docs/version_2_definition_plan.md §3
    Tests:   tests/test_version_2_status.py::test_w1_v2_dispositions_parse_strictly
    """
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if text.count(V2_HEADER) != 1:
        raise SystemExit(f"{path.name}: the table header is missing or repeated")
    lines = text.split(V2_HEADER, 1)[1].splitlines()[2:]
    end = next((i for i, line in enumerate(lines) if not line.startswith("|")), len(lines))
    after = [line for line in lines[end:] if line.startswith("|")]
    if after:
        raise SystemExit(f"{path.name}: a row after a break in the table would be skipped: {after[0][:60]}")
    out = {}
    for line in lines[:end]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 5:
            raise SystemExit(f"{path.name}: a row does not have five cells: {line[:80]}")
        key, kind, verdict, decided, reason = cells
        key = key.strip("`")
        if kind not in V2_KINDS:
            raise SystemExit(f"{key}: unknown disposition {kind!r} in {path.name}")
        if verdict not in ("PASS", "FAIL"):
            raise SystemExit(f"{key}: the verdict a disposition was decided on is PASS or FAIL, not {verdict!r}")
        if kind == "accepted finding" and verdict != "FAIL":
            raise SystemExit(f"{key}: only a failing criterion can be accepted as a finding")
        if not (v1.OWNER.match(decided) or v1.PENDING.match(decided)):
            raise SystemExit(f"{key}: 'decided' must be 'owner DATE' or the pending form, not {decided!r}")
        if key in out:
            raise SystemExit(f"{key}: two disposition rows in {path.name}")
        out[key] = {"kind": kind, "verdict": verdict, "owner": bool(v1.OWNER.match(decided)), "decided": decided,
                    "reason": reason}
    return out


def statuses() -> list[Status]:
    spec = v1.spec_criteria()
    rows = record_rows()
    reds = v1.red_mutants()
    survivors = survived_mutants()
    unbuilt = v1.not_built()
    by_test = v1.proved_by_test()
    decided = v1.dispositions()
    later = v2_dispositions()
    limbs = v1.premise_limbs()
    tag = tag_verdicts()
    earlier = revision_38_verdicts()
    swept = sweep_fractions(SWEEP_RECORD)
    refused = sweep_refused(SWEEP_RECORD)
    sweep_old = sweep_stale(SWEEP_RECORD)
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
        d2 = later.get(key)
        d2_on = d2 is not None and d2["verdict"] == row["outcome"]  # applies only to the verdict it was decided on
        if d2_on and d2["kind"] != "verdict reviewed":
            d = d2
        sweep = swept.get(key) or refused.get(key) or ("not swept" if cls in v1.DIRECTIONAL else "—")
        if sweep_old and (key in swept or key in refused):
            sweep += f" ({STALE_CELL})"
        s = Status(key, cls, source, row["outcome"], at_tag, v1._readouts(row), sweep, "", False)
        proof = reds.get(key, [])
        if proof:
            s.evidence += "; proved by " + ", ".join(proof)
        if row["outcome"] == "PASS":
            if d and d["kind"] != "note":
                raise SystemExit(f"{key}: passes, as at the tag, so its disposition {d['kind']!r} does not apply")
            survived = survivors.get(key, [])
            if cls in v1.DIRECTIONAL and not proof and survived:
                named = any(kind == "named" for kind, _ in survived)
                s.status = "pass, named mutant survived" if named else "pass, mutant survived"
                s.blocks = {"W1", "W3"}
                s.evidence += "; survived: " + ", ".join(text for _, text in survived)
            elif cls in v1.DIRECTIONAL and not proof:
                s.status, s.blocks = "pass, no recorded mutation", {"W1", "W3"}
            else:
                s.status, s.final = "pass", True
            if cls in v1.DIRECTIONAL and (key not in swept or sweep_old):
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
            if d2_on and d2["owner"]:
                s.status = f"{s.status}; {REVIEWED}"
            else:
                s.status, s.final, s.blocks = f"{s.status}; {CHANGED}", False, s.blocks | {"W1"}
        if d:
            s.note = f"{d['reason']} *({d['decided']})*"
        elif d_tag:  # shown, so the owner sees the decision the new verdict reverses; not applied
            why = "the verdict changed" if at_tag != "—" else "no row at the tag"
            s.note = f"version 1 disposition, not applied ({why}): {d_tag['kind']} — {d_tag['reason']} *({d_tag['decided']})*"
        if d2 and d2 is not d:  # a review, or a version 2 disposition on a verdict that has since moved
            head = (f"version 2: verdict reviewed ({d2['verdict']})" if d2_on else
                    f"version 2 disposition, not applied (decided on {d2['verdict']}, now {row['outcome']}): {d2['kind']}")
            text = f"{head} — {d2['reason']} *({d2['decided']})*"
            s.note = f"{text} · {s.note}" if s.note else text
        s.at_revision_38 = earlier.get(s.key, "new since revision 38")
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
        s.at_revision_38 = earlier.get(s.key, "new since revision 38")
        out.append(s)

    gone = sorted(set(earlier) - {s.key for s in out})
    if gone:
        raise SystemExit(f"rows at revision 38 ({REVISION_38}) with no row now: {gone}; shown, not dropped (P2)")
    no_row = sorted(set(later) - {k for k, _, _ in rows})
    if no_row:
        raise SystemExit(f"{V2_DISPOSITIONS.name} names rows no record has: {no_row}; it decides on a verdict, so the "
                         "row must have one")
    known = {s.key for s in out} | set(spec)
    stray = sorted((set(decided) | set(limbs)) - known)
    if stray:
        raise SystemExit(f"{v1.DISPOSITIONS.name} names criteria no record or spec row knows: {stray}")
    return sorted(out, key=lambda s: (int(base(s.key).split(".")[2]), s.key))


def render(rows: list[Status]) -> str:
    def blocked(condition: str) -> list[str]:
        return [f"`{s.key}`" for s in rows if condition in s.blocks]

    hashes = []
    for name, path in (*v1.ENSEMBLE_RECORDS, *v1.MUTATION_RECORDS, ("Sweep record", SWEEP_RECORD)):
        if not path.exists():
            hashes.append(f"- {name} (`{path.relative_to(REPO)}`): not yet generated")
            continue
        # every hash the record's rows are keyed by: the sweep record's Phase D rows carry ``phase_d_engine_hash``
        found = re.findall(r"^(code_hash|engine_hash|phase_d_engine_hash): (\w+)", path.read_text(encoding="utf-8"),
                           re.M)
        keyed = ", ".join(f"{kind} `{value[:12]}`" for kind, value in found)
        hashes.append(f"- {name} (`{path.relative_to(REPO)}`): {keyed}")
    hashes.append(f"- the tag's page (`{TAG_PAGE.relative_to(REPO)}`): the verdicts at `version-1`")
    hashes.append(f"- this page as committed at `{REVISION_38}`: the verdicts at spec revision 38")
    lines = [
        "# Version 2 status",
        "",
        "Generated by `tools/version_2_status.py`; do not edit. One row per `M11.C` criterion or variant, read from the",
        "records below and from `docs/version_1_dispositions.md` and `docs/version_2_dispositions.md`, against the",
        "stage 2 conditions W1–W4 (`docs/implementation_plan_version_2.md` §2.3). A verdict that is not the one the",
        "frozen page records at the tag `version-1` is **changed since the tag** and is not final until the owner",
        "reviews it (T-122; a version 2 disposition the owner made on that verdict is the review); a version 2",
        "disposition applies only while the verdict it names stands; a criterion",
        "the tag's page does not have is new since the tag. The **at revision 38** column is this page's verdict at",
        f"commit `{REVISION_38}`, before stage 3's regeneration: shown beside the verdict, never used for a status, since a",
        "verdict that differs from it is a new result under revision 39's rules (`M11.F`;",
        "`docs/version_2_revision_39_changes.md` lists every changed field). The **sweep** column is a measurement, *n of S* samples",
        "over the declared ranges in which the direction held (`docs/version_2_sweep_record.md`), never a verdict",
        "(`M11.5`); a premise or check is not swept. A criterion the sweep refused at the rerun (`M17.A.1`) is shown",
        "so, with the sample, and has no fraction, so W4 counts it as not swept.",
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
    sweep_old = sweep_stale(SWEEP_RECORD)
    if sweep_old:
        lines += ["", f"The sweep record's fractions were {sweep_old} (each such cell says *{STALE_CELL}*)."]
    counts: dict[str, int] = {}
    for s in rows:
        counts[s.status] = counts.get(s.status, 0) + 1
    moved = [s for s in rows if s.at_revision_38 != s.verdict]
    lines += ["", "Rows by status:", *[f"- {k}: {v}" for k, v in sorted(counts.items())], "",
              f"Verdicts not the one this page gave at revision 38: {len(moved)} of {len(rows)} rows"
              + (": " + ", ".join(f"`{s.key}` ({s.at_revision_38} → {s.verdict})" for s in moved) if moved else "")
              + ".", "",
              "## Every criterion", "",
              "| Criterion | Class | Record | Verdict | At `version-1` | At revision 38 | Evidence | Sweep | Status | Final "
              "| Blocks | Disposition |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in rows:
        lines.append(f"| `{s.key}` | {s.cls} | {s.source} | {s.verdict} | {s.at_tag} | {s.at_revision_38} | "
                     f"{s.evidence} | {s.sweep} | "
                     f"{s.status} | {'yes' if s.final else 'no'} | {', '.join(sorted(s.blocks)) or '—'} | {s.note or '—'} |")
    return "\n".join(lines) + "\n"


def main() -> NoReturn:
    PAGE.write_text(render(statuses()), encoding="utf-8")
    print(f"wrote {PAGE.relative_to(REPO)}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
