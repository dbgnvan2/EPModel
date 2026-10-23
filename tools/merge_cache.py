"""Merge another machine's sweep cache into this one: the stage 2 sweep split across two Macs (`M17.E.1`).

Purpose: the union of two cache files, written into the first. A key is a hash of the engine, the sample's edits and
         the criterion, so the same key on two machines is the same run and must hold the same result: a key holding
         two different results stops the merge, and nothing is written; neither is picked. Two files made under
         different NumPy versions (their ``_meta``), or a file that records none, stop it too. A result is never
         dropped: every key of both files is in the merged file. Afterwards ``python3 tools/version_2_sweep.py
         --render-only`` renders the record from the merged cache with no child run.
Spec:    docs/bowen_agent_model_spec_v2.md#M17.E.1; docs/implementation_plan_version_2.md#2.10; docs/two_mac_sweep.md
Tests:   tests/test_sweep_two_macs.py::test_m17e1_merge_is_the_union,
         tests/test_sweep_two_macs.py::test_m17e1_merge_refuses_a_key_with_two_results_and_writes_nothing,
         tests/test_sweep_two_macs.py::test_m17e1_merge_refuses_another_or_an_unknown_numpy,
         tests/test_sweep_two_macs.py::test_m17e1_merged_cache_renders_the_whole_record_without_a_run

    python3 tools/merge_cache.py OTHER [--into docs/records_cache/sweep_v2.json]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.record_cache import CACHE_DIR, read_cache_file, write_cache_file  # noqa: E402

DEFAULT_INTO = CACHE_DIR / "sweep_v2.json"
SHOWN = 10  # conflicting keys named in the refusal; the count is always given in full


def local_sweep_running() -> bool:
    """A sweep running here holds the cache it loaded and rewrites the whole file after each run, so a merge made
    meanwhile would be lost at its next write. ``pgrep`` failing (exit above 1) counts as running: refuse, not guess."""
    found = subprocess.run(["pgrep", "-f", "[v]ersion_2_sweep.py"], capture_output=True, timeout=30)
    return found.returncode != 1


def merge(into: Path, other: Path) -> tuple[int, int]:
    """Write the union of ``other`` and ``into`` into ``into``; return (keys added, keys already there and equal)."""
    if local_sweep_running():
        raise SystemExit("a sweep is running on this machine and would overwrite the merge at its next write; merge "
                         "after it stops. Nothing written.")
    ours, our_meta = read_cache_file(into)
    theirs, their_meta = read_cache_file(other)
    if not our_meta.get("numpy") or not their_meta.get("numpy"):
        missing = ", ".join(p.name for p, m in ((into, our_meta), (other, their_meta)) if not m.get("numpy"))
        raise SystemExit(f"{missing}: no NumPy version recorded, so its results cannot be shown to match; nothing written")
    if our_meta != their_meta:
        raise SystemExit(f"{into} was made under NumPy {our_meta['numpy']} and {other} under NumPy "
                         f"{their_meta['numpy']}; nothing written")
    conflicts = sorted(k for k in ours.keys() & theirs.keys() if ours[k] != theirs[k])
    if conflicts:
        named = [f"{k[:12]} ({ours[k].get('criterion', '?')})" for k in conflicts[:SHOWN]]
        raise SystemExit(f"{len(conflicts)} key(s) hold different results in the two files, so the same run gave two "
                         f"answers; nothing written: {', '.join(named)}")
    added = {k: v for k, v in theirs.items() if k not in ours}
    write_cache_file(into, {**ours, **added}, our_meta)
    return len(added), len(theirs) - len(added)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("other", type=Path, help="the other machine's cache file, copied here")
    parser.add_argument("--into", type=Path, default=DEFAULT_INTO)
    args = parser.parse_args(argv)
    added, equal = merge(args.into, args.other)
    print(f"{args.into}: {added} result(s) added from {args.other}; {equal} already there and equal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
