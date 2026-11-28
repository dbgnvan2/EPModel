"""Propose each composite criterion's dominant constant from the sensitivity records (plan D9, `M17.E.2`).

Purpose: `M17.E.2` sweeps a criterion over its *dominant* `[I]` constant, and plan D9 names none (T-077 (c)). The
         owner decided on 2026-11-27 that a tool proposes one per composite criterion and the owner confirms it.
         This tool reads two committed records and runs nothing: the stage-end sweep's joint samples
         (``docs/version_2_sweep_record.md``, its machine-readable block) and the T-236 grid's cache
         (``tools/sensitivity_grid.py``). For each constant a criterion was swept over, the samples that gave a
         verdict are split at the middle of the constant's declared range (the geometric middle on a log scale);
         the constant's *swing* is the share of samples holding (PASS) above the middle less the share below. The
         proposal is the constant with the largest absolute swing. None is proposed where the largest swing is 0,
         where two constants tie for it, or where the criterion was not swept: `M17.E.2` then reports the criterion
         as not covered. A proposal is a measurement for the owner to confirm, never a decision.
Spec:    docs/bowen_agent_model_spec_v2.md#M17.E.2, #M17.E.1
Tests:   tests/test_dominant_constant.py::test_m17e2_t077_dominant_constant_page_is_current

    python3 tools/dominant_constant.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.atomic_write import write_text_atomic  # noqa: E402
from tools.record_cache import ERRORED, Cache  # noqa: E402
from tools.record_json import record_json  # noqa: E402

PAGE = REPO / "docs" / "dominant_constant_proposals.md"
SWEEP_RECORD = REPO / "docs" / "version_2_sweep_record.md"


def middle(low: float, high: float, scale: str) -> float:
    return math.sqrt(low * high) if scale == "log" and low > 0 else (low + high) / 2


def split(samples: list[tuple[float, bool]], mid: float) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """(held, counted) below the middle and at or above it; None when either half is empty."""
    below = [h for v, h in samples if v < mid]
    above = [h for v, h in samples if v >= mid]
    if not below or not above:
        return None
    return (sum(below), len(below)), (sum(above), len(above))


def candidate(source: str, constant: str, low: float, high: float, scale: str, samples: list) -> dict:
    mid = middle(low, high, scale)
    halves = split(samples, mid)
    swing = None if halves is None else halves[1][0] / halves[1][1] - halves[0][0] / halves[0][1]
    return {"source": source, "constant": constant, "low": low, "high": high, "scale": scale, "middle": mid,
            "halves": halves, "swing": swing}


def value(sample: dict, constant: str) -> float:
    """A sample's drawn value; a named ratio (`M17.E.1`) is drawn as itself and kept apart from the row values."""
    return float(sample["values"][constant]) if constant in sample["values"] else float(sample["ratio"])


def sweep_candidates() -> tuple[str, dict[str, list[dict]]]:
    text = SWEEP_RECORD.read_text(encoding="utf-8")
    engine = next(line.split(": ", 1)[1] for line in text.splitlines() if line.startswith("engine_hash: "))
    out: dict[str, list[dict]] = {}
    for row in record_json(text):
        counted = [s for s in row["rows"] if s["outcome"] not in ERRORED]
        out[row["criterion"]] = [
            candidate("sweep", c["constant"], c["low"], c["high"], c["scale"],
                      [(value(s, c["constant"]), s["outcome"] == "PASS") for s in counted])
            for c in row["constants"]]
    return engine, out


def grid_candidates() -> dict[str, list[dict]]:
    from tools import sensitivity_grid as grid

    results = grid.results(Cache(grid.CACHE_NAME), None)  # KeyError when the grid's cache is stale
    out: dict[str, list[dict]] = {}
    for cid in grid.CRITERIA:
        rows = [(p, r[cid]) for p, r in results.items() if r[cid]["outcome"] not in ERRORED]
        out[cid] = [candidate("grid", name, min(grid.VALUES), max(grid.VALUES), "log",
                              [(p[axis], r["outcome"] == "PASS") for p, r in rows])
                    for axis, name in enumerate((grid.ROW, grid.COLUMN))]
    return out


def propose(candidates: list[dict]) -> dict | None:
    measured = sorted((c for c in candidates if c["swing"] is not None), key=lambda c: -abs(c["swing"]))
    if not measured or measured[0]["swing"] == 0:
        return None
    if len(measured) > 1 and abs(measured[1]["swing"]) == abs(measured[0]["swing"]):
        return None
    return measured[0]


def share(half: tuple[int, int]) -> str:
    return f"{half[0]} of {half[1]}"


def render() -> str:
    from src.bowen.ensemble.criteria import CRITERIA

    engine, swept = sweep_candidates()
    gridded = grid_candidates()
    composites = [cid for cid, c in CRITERIA.items() if c.cls == "composite"]
    lines = [
        "# Dominant constant proposals (`M17.E.2`, plan D9)",
        "",
        "Generated by `tools/dominant_constant.py`; do not edit. **Every proposal is pending the owner's "
        "confirmation** (owner decision 2026-11-27, T-077 (c)): the tool proposes and the owner decides.",
        "",
        "For each composite criterion (`M11.5`), each constant it was swept over is scored by its *swing*: the "
        "share of samples whose verdict was PASS at or above the middle of the constant's declared range, less the "
        "share below it (the geometric middle on a log scale). Undetermined samples count as not holding; errored "
        "samples are left out. The proposal is the constant with the largest absolute swing. None is proposed "
        "where every swing is 0, where two constants tie, or where the criterion was not swept: `M17.E.2` reports "
        "that criterion as not covered. The joint sweep draws every constant at once, so a swing is confounded "
        "with the other constants' draws, and at 8 samples a half holds 1 to 7 of them: a swing is a pointer for "
        "the owner, not an estimate.",
        "",
        f"Sources: `docs/version_2_sweep_record.md` (engine `{engine[:12]}`), and for `M11.C.42`, `C.44` and "
        "`C.45` the T-236 grid's two constants (`docs/sensitivity_distance_triangle.md`), scored the same way over "
        "its 25 points.",
        "",
        "| Criterion | Proposed dominant constant | Swing | Below the middle held | At or above held | Source |",
        "|---|---|---|---|---|---|",
    ]
    details = []
    for cid in composites:
        candidates = swept.get(cid, []) + gridded.get(cid, [])
        best = propose(candidates)
        if not candidates:
            lines.append(f"| `{cid}` | not swept: not covered | — | — | — | — |")
            continue
        if best is None:
            lines.append(f"| `{cid}` | none identified: not covered | — | — | — | — |")
        else:
            lines.append(f"| `{cid}` | `{best['constant']}` | {best['swing']:+.2f} | {share(best['halves'][0])} | "
                         f"{share(best['halves'][1])} | {best['source']} |")
        details += [f"### `{cid}`", "", "| Constant | Range | Middle | Below held | At or above held | Swing | Source |",
                    "|---|---|---|---|---|---|---|"]
        for c in sorted(candidates, key=lambda c: (c["swing"] is None, -abs(c["swing"] or 0), c["constant"])):
            halves = c["halves"]
            details.append(
                f"| `{c['constant']}` | {c['low']:g} to {c['high']:g} ({c['scale']}) | {c['middle']:.4g} | "
                + (f"{share(halves[0])} | {share(halves[1])} | {c['swing']:+.2f}" if halves else "— | — | one half empty")
                + f" | {c['source']} |")
        details.append("")
    return "\n".join(lines + ["", "## Every constant scored", ""] + details).rstrip() + "\n"


def main() -> int:
    write_text_atomic(PAGE, render())
    print(f"wrote {PAGE.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
