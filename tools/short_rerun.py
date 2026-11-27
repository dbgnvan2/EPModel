"""Measure whether a short rerun of the records finds what the full rerun found (T-299, owner question 2026-11-25).

Purpose: after each engine batch every generated record is rerun in full (18 to 30 hours on Mac Mini1). This tool
         tests a shorter rerun on revision 66, the one rerun that found real defects (T-264). It has three steps:

         ``scale ROOT --fraction F`` edits a scratch checkout so the unchanged record tools run on about F of each
         row's seeds: ``ensemble_block`` and ``ensemble_cap`` times F, ``ensemble_precision`` divided by √F (the
         adaptive stop then needs about F as many seeds, since the half-width falls as 1/√n), the fixed seed counts
         of ``M11.C.10`` and ``M11.C.48`` and ``tools/level_occupancy.py``'s ``SEEDS`` times F. It refuses this
         repository's own tree, so a full run, the engine hash and ``docs/records_cache/`` here never change; the
         scratch checkout's own cache is written and thrown away with it.

         ``collect JSON --run LABEL ROOT LOG`` adds one short run to the results JSON: the rows of the seven records
         the changes pages compare (``tools/record_changes.RECORDS``), trimmed to the key, the verdict, the seeds
         and each readout's effect, and the wall time of each queue stage from its log. ``--full LABEL LOG`` adds a
         full queue's stage times.

         ``render JSON`` writes the report from that JSON. It compares each short revision 66 run (a) with revision
         64's full records, as a real short rerun would be read, and (b) with the short revision 64 run at the same
         fraction, by ``record_changes``'s rules (a verdict changed; an effect moved past the earlier side's
         half-width), and (c) checks each of T-264's real problems. The full records are read from git.
Spec:    docs/review/2026-11-26_short_rerun_test.md (the report this tool writes)
Tests:   tests/test_short_rerun.py

    python3 tools/short_rerun.py scale ROOT --fraction 0.2
    python3 tools/short_rerun.py collect docs/review/2026-11-26_short_rerun_test.json --run rev66@0.2 ROOT LOG
    python3 tools/short_rerun.py render docs/review/2026-11-26_short_rerun_test.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.seed_fraction import scaled_value  # noqa: E402  (one scaling rule for both)

# Each seed setting a short run scales: (file, the line's start, the central value). The value after the start is
# replaced; the line must occur once.
SCALED = (
    ("config/bowen/constants.md", "| `ensemble_block` | ", 50, "times"),
    ("config/bowen/constants.md", "| `ensemble_cap` | ", 500, "times"),
    ("config/bowen/constants.md", "| `ensemble_precision` | ", 0.25, "over_root"),
    ("config/bowen/criteria.md", "| M11.C.10 | seeds | ", 600, "times"),
    ("config/bowen/criteria.md", "| M11.C.48 | seeds | ", 200, "times"),
    ("tools/level_occupancy.py", "SEEDS = ", 20, "times"),
)
MARKER = ".short_rerun.json"


def scale(root: Path, fraction: float) -> list[str]:
    """Edit ``root``'s seed settings to ``fraction`` of their central values; return what changed. Refuses this
    repository, a root already scaled, and any setting not at its central value or not found exactly once."""
    root = root.resolve()
    if root == REPO or ((REPO / "src" / "bowen").is_dir() and REPO in root.parents):
        raise SystemExit(f"{root}: refusing to scale this repository's own tree; use a scratch checkout")
    if not 0 < fraction < 1:
        raise SystemExit(f"fraction {fraction}: must lie between 0 and 1")
    if (root / MARKER).exists():
        raise SystemExit(f"{root}: already scaled ({MARKER})")
    texts, done = {}, []
    for file, start, central, how in SCALED:
        text = texts.setdefault(file, (root / file).read_text(encoding="utf-8"))
        lines = [line for line in text.splitlines() if line.startswith(start)]
        if len(lines) != 1:
            raise SystemExit(f"{file}: {len(lines)} lines start {start!r}, not one")
        old = lines[0]
        rest = old[len(start):]
        value = re.match(r"[0-9.]+", rest).group(0)
        if float(value) != float(central):
            raise SystemExit(f"{file}: {start.strip()} is {value}, not its central {central}")
        new = start + scaled_value(central, how, fraction) + rest[len(value):]
        texts[file] = text.replace(old + "\n", new + "\n", 1)
        done.append(f"{file}: {old.strip()} -> {new.strip()}")
    for file, text in texts.items():
        (root / file).write_text(text, encoding="utf-8")
    (root / MARKER).write_text(json.dumps({"fraction": fraction, "edits": done}, indent=1) + "\n", encoding="utf-8")
    return done


# ---- collect ----------------------------------------------------------------------------------------------------

STAGE = re.compile(r"^(START|EXIT) (\S+) (?:(\d+) )?(.+)$")


def stage_seconds(log: str) -> dict[str, dict]:
    """Each stage's wall time and exit code from a queue log's ``START name date`` and ``EXIT name code date``."""
    started, out = {}, {}
    for line in log.splitlines():
        m = STAGE.match(line.strip())
        if not m:
            continue
        kind, name, code, when = m.groups()
        when = re.sub(r" workers \d+$", "", when.strip())  # a stage restarted at fewer workers is summed
        t = datetime.strptime(re.sub(r" [A-Z]{2,5} (\d{4})$", r" \1", when), "%a %b %d %H:%M:%S %Y")
        if kind == "START":
            started[name] = t
        elif name in started:
            before = out.get(name, {}).get("seconds", 0)
            out[name] = {"seconds": before + int((t - started.pop(name)).total_seconds()), "exit": int(code)}
    return out


def _sig(x):
    return float(f"{x:.6g}") if isinstance(x, float) and math.isfinite(x) else x


def trimmed(row: dict, record) -> dict:
    """What the comparison reads of a row: its key, verdict and seeds and each readout's effect (level rows whole)."""
    if record.verdict is None:
        return row
    keep = {f: row[f] for f in record.fields}
    keep[record.verdict] = row[record.verdict]
    keep["seeds"] = row.get("seeds")
    keep["readouts"] = [{k: _sig(r[k]) for k in ("readout", "direction", "mean_difference", "half_width", "p",
                                                  "report_only") if k in r} for r in row.get("readouts", [])]
    return keep


def collect(path: Path, label: str, root: Path | None, log: Path, full: bool = False) -> None:
    sys.path.insert(0, str(REPO))
    from tools.record_changes import RECORDS
    from tools.record_json import record_json

    data = json.loads(path.read_text(encoding="utf-8"))
    stages = stage_seconds(log.read_text(encoding="utf-8"))
    if full:
        data.setdefault("full_stages", {})[label] = stages
    else:
        marker = json.loads((root / MARKER).read_text(encoding="utf-8"))
        records = {r.path: [trimmed(row, r) for row in record_json((root / r.path).read_text(encoding="utf-8"))]
                   for r in RECORDS}
        data.setdefault("runs", {})[label] = {"fraction": marker["fraction"], "edits": marker["edits"],
                                              "stages": stages, "records": records}
    path.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


# ---- render -----------------------------------------------------------------------------------------------------

def _full_rows(ref: str):
    from tools.record_changes import RECORDS, ref_text
    from tools.record_json import record_json
    return {r.path: record_json(ref_text(ref, r.path)) for r in RECORDS}


def partial_share(level_rows: list[dict], variant: str) -> float:
    """The smallest share of member-weeks in the partial band over the variant's arms and spans: the quantity
    ``test_m111a_availability_inversion_acts_over_c16s_run`` holds at 0.25 or more."""
    return min(1.0 - r[s]["deletion_on_every_layer"] - r[s]["layers_above_0_unavailable"]
               for r in level_rows if r["variant"] == variant for s in ("whole_run", "window"))


def row_flags(then_rows: list[dict], now_rows: list[dict], record) -> dict[str, dict]:
    """Each row on both sides: whether its verdict changed and which readouts moved past ``then``'s half-width
    (``record_changes``'s rules). Rows on one side only are listed as such."""
    from tools.record_changes import keyed, moved_effects

    then, now = keyed(then_rows, record.fields), keyed(now_rows, record.fields)
    out = {}
    for k in sorted(set(then) | set(now)):
        if k not in then or k not in now:
            out[k] = {"one_side": "then" if k in then else "now"}
            continue
        if record.verdict is None:
            out[k] = {"changed": then[k] != now[k], "moved": []}
            continue
        out[k] = {"changed": then[k][record.verdict] != now[k][record.verdict],
                  "then": then[k][record.verdict], "now": now[k][record.verdict],
                  "moved": [name for name, _, _ in moved_effects(then[k], now[k])]}
    return out


def summarise(data: dict) -> dict:
    """Every number the report prints."""
    from tools.record_changes import RECORDS

    refs = data["refs"]
    full64, full66 = _full_rows(refs["rev64"]), _full_rows(refs["rev66"])
    by_path = {r.path: r for r in RECORDS}
    full = {p: row_flags(full64[p], full66[p], by_path[p]) for p in by_path if by_path[p].verdict is not None}
    noise = {(c["record"], c["row"]) for c in data["t264"]["noise"]}
    fractions = sorted({run["fraction"] for run in data["runs"].values()}, reverse=True)
    out = {"fractions": fractions, "full_changed": sum(f.get("changed", False) for p in full for f in full[p].values()),
           "comparisons": {}, "problems": [], "hours": {}, "seeds": {}}
    for fraction in fractions:
        s64, s66 = (data["runs"][f"rev{e}@{fraction}"]["records"] for e in ("64", "66"))
        for name, ref in (("a", full64), ("b", s64)):
            counts = {"rows": 0, "flips": 0, "flip_real_moves": 0, "flip_false": 0, "effect_only": 0,
                      "effect_false": 0, "missed_moves": 0, "one_side": 0}
            per = {}
            for path, record in by_path.items():
                if record.verdict is None:
                    continue
                flags = per[path] = row_flags(ref[path], s66[path], record)
                for key, f in flags.items():
                    if "one_side" in f:
                        counts["one_side"] += 1
                        continue
                    counts["rows"] += 1
                    truth = full[path].get(key, {})
                    full_moved = truth.get("changed", False) or bool(truth.get("moved"))
                    false = (path, key) in noise or not full_moved
                    if f["changed"]:
                        counts["flips"] += 1
                        counts["flip_false"] += false
                        counts["flip_real_moves"] += truth.get("changed", False) and (path, key) not in noise
                    elif f["moved"]:
                        counts["effect_only"] += 1
                        counts["effect_false"] += false
                    if truth.get("changed") and not f["changed"]:
                        counts["missed_moves"] += 1
            out["comparisons"][f"{name}@{fraction}"] = {"counts": counts, "flags": per}
        level = {"a": partial_share(full64["docs/phase_c_level_occupancy.md"], "availability-level-inverted"),
                 "b": partial_share(s64["docs/phase_c_level_occupancy.md"], "availability-level-inverted")}
        level_now = partial_share(s66["docs/phase_c_level_occupancy.md"], "availability-level-inverted")
        for problem in data["t264"]["real"]:
            row = {"name": problem["name"], "fraction": fraction}
            for name in ("a", "b"):
                if problem["record"] == "docs/phase_c_level_occupancy.md":
                    row[name] = level_now < 0.25 <= level[name]
                    row[name + "_detail"] = f"partial band {level[name]:.2f} -> {level_now:.2f}"
                    continue
                flags = out["comparisons"][f"{name}@{fraction}"]["flags"]
                f = flags[problem["record"]].get(problem["row"], {})
                if f.get("one_side") == "then" and "criterion_record" in problem:
                    # A mutation record tests only criteria that pass, so a mutant row the short run lacks means its
                    # criterion stopped passing: that criterion's own row is the flag.
                    c = flags[problem["criterion_record"]].get(problem["row"].split(" · ")[1], {})
                    if "one_side" not in c and c and (c["changed"] or problem.get("readout") in c["moved"]):
                        row[name] = True
                        row[name + "_detail"] = f"mutant row gone; criterion {c['then']} -> {c['now']}"
                        continue
                if "one_side" in f or not f:
                    row[name], row[name + "_detail"] = False, "row not on both sides"
                    continue
                row[name] = f["changed"] or problem.get("readout") in f["moved"]
                row[name + "_detail"] = (f"{f['then']} -> {f['now']}" if f["changed"] else
                                         f"{problem.get('readout')} moved" if row[name] else
                                         f"{f['then']} held, {problem.get('readout')} within")
            out["problems"].append(row)
    for label, run in data["runs"].items():
        out["hours"][label] = {k: v["seconds"] / 3600 for k, v in run["stages"].items()}
        out["seeds"][label] = sum(r.get("seeds") or 0 for p, rows in run["records"].items() for r in rows
                                  if by_path[p].verdict is not None)
    for label, stages in data.get("full_stages", {}).items():
        out["hours"][label] = {k: v["seconds"] / 3600 for k, v in stages.items()}
    out["seeds"]["rev64"] = sum(r.get("seeds") or 0 for p, rows in full64.items() for r in rows
                                     if by_path[p].verdict is not None)
    out["seeds"]["rev66"] = sum(r.get("seeds") or 0 for p, rows in full66.items() for r in rows
                                     if by_path[p].verdict is not None)
    return out


STAGES = ("D2", "D1", "DMUT", "C", "CMUT", "SWEEP", "LEVEL")


def render(data: dict) -> str:
    s = summarise(data)
    refs = data["refs"]
    fr = s["fractions"]
    lines = [
        "# Short rerun test on revision 66 (T-299)", "",
        "Generated by `tools/short_rerun.py render` from `" + data["json_name"] + "`; do not edit. Owner question "
        "2026-11-25: does the full rerun of every record after an engine batch find anything a much shorter rerun "
        "would not? Revision 66 is the test case because its rerun found real defects (T-264).", "",
        f"**Method.** Revision 64's engine (`{refs['rev64_engine']}`) and revision 66's (`{refs['rev66_engine']}`) were each "
        "run at a fraction of the seeds, in scratch checkouts edited by `tools/short_rerun.py scale`: "
        "`ensemble_block`, `ensemble_cap` and the fixed seed counts times the fraction, `ensemble_precision` over its "
        "square root, so the adaptive stop needs about that fraction of each row's seeds. The seven records the "
        "changes pages compare were rerun; the emergence audit, diagnostic, sensitivity and trace stages were "
        "skipped (they are not in the 45 verdict moves). A row is **flagged** when its verdict differs from the "
        "earlier side's or, its verdict held, a readout moved past the earlier side's half-width "
        "(`tools/record_changes.py`'s rules). A flag is **false** when the full rerun "
        f"(`{refs['rev64']}` against `{refs['rev66']}`) changed neither that row's verdict nor any of its effects, or "
        "T-264 judged the move noise. Comparison (a) reads short revision 66 against revision 64's full records, as "
        "a real short rerun would be read; (b) against short revision 64 at the same fraction.", "",
        "## T-264's real problems", "",
        "| Problem | " + " | ".join(f"(a) at {f:g} | (b) at {f:g}" for f in fr) + " |",
        "|---|" + "---|---|" * len(fr),
    ]
    names = list(dict.fromkeys(p["name"] for p in s["problems"]))
    caught = {f: {"a": 0, "b": 0} for f in fr}
    for name in names:
        cells = []
        for f in fr:
            p = next(x for x in s["problems"] if x["name"] == name and x["fraction"] == f)
            for side in ("a", "b"):
                caught[f][side] += p[side]
                cells.append(f"{'caught' if p[side] else 'missed'} ({p[side + '_detail']})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("| **Caught** | " + " | ".join(f"{caught[f][side]} of {len(names)}" for f in fr
                                               for side in ("a", "b")) + " |")
    lines += ["", "## Flags and false flags", "",
              f"The full rerun changed {s['full_changed']} verdicts in these six records (the level-occupancy record "
              "has no verdicts).", "",
              "| Comparison | Rows compared | Verdict flips | of them full-rerun moves | false | Effect-only flags | "
              "false | Full-rerun verdict moves missed | Rows on one side only |",
              "|---|---|---|---|---|---|---|---|---|"]
    for f in fr:
        for side in ("a", "b"):
            c = s["comparisons"][f"{side}@{f}"]["counts"]
            lines.append(f"| ({side}) at {f:g} | {c['rows']} | {c['flips']} | {c['flip_real_moves']} | "
                         f"{c['flip_false']} | {c['effect_only']} | {c['effect_false']} | {c['missed_moves']} | "
                         f"{c['one_side']} |")
    lines += ["", "## Hours", "",
              "Wall hours per queue stage at 8 workers. `rev69` is revision 69's full queue, the latest one; its "
              "stages outside these seven are DIAG, AUDIT (the emergence audit), SENS, GG and T176.", "",
              "| Run | Machine | " + " | ".join(STAGES) + " | These seven | All stages | Seeds (six records) |",
              "|---|---|" + "---|" * (len(STAGES) + 3)]
    for label in sorted(s["hours"]):
        h = s["hours"][label]
        seeds = s["seeds"].get(label, "")
        lines.append(f"| {label} | {data['machines'].get(label, 'Mac Mini1')} | " + " | ".join(f"{h[k]:.2f}" if k in h else "—" for k in STAGES)
                     + f" | {sum(h.get(k, 0) for k in STAGES):.2f} | {sum(h.values()):.2f} | {seeds} |")
    lines += ["", "## Recommended mix of runs", "", "Claude's decision 2026-11-25; the owner accepted it on 2026-11-26 (an owner rule in `AGENTS.md`).", ""]
    lines += data["mix"]
    lines += ["", "## Conclusion", "", data["conclusion"], ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scale")
    p.add_argument("root", type=Path)
    p.add_argument("--fraction", type=float, required=True)
    p = sub.add_parser("collect")
    p.add_argument("json", type=Path)
    p.add_argument("--run", nargs=3, metavar=("LABEL", "ROOT", "LOG"))
    p.add_argument("--full", nargs=2, metavar=("LABEL", "LOG"))
    p = sub.add_parser("render")
    p.add_argument("json", type=Path)
    args = parser.parse_args()
    if args.cmd == "scale":
        print("\n".join(scale(args.root, args.fraction)))
    elif args.cmd == "collect":
        if args.run:
            collect(args.json, args.run[0], Path(args.run[1]), Path(args.run[2]))
        if args.full:
            collect(args.json, args.full[0], None, Path(args.full[1]), full=True)
    else:
        sys.path.insert(0, str(REPO))
        data = json.loads(args.json.read_text(encoding="utf-8"))
        out = args.json.with_suffix(".md")
        out.write_text(render(data), encoding="utf-8")
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
