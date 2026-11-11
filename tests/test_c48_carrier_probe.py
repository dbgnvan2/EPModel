"""T-185's carrier probe reads the baseline `M11.C.48`'s row reads (learning-qa 2026-11-10).

Purpose: `tools/c48_carrier_probe.py` splits the row's comparison, so its `check` column — the row's own baseline
         against the probe's recomputation of it — is the probe's only guard that it still reads what the row reads.
         When T-185 narrowed the row's baseline to the uncut ties with as many sides, `check` still compared it with
         the mean over *every* uncut tie, so it read False on real seeds and `summarise` exited. This test runs the
         probe's own rows on the Phase B family and holds its three baseline columns apart.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.48
Tests:   this file
"""

from __future__ import annotations

import math

import tools.c48_carrier_probe as probe


def test_m11c48_t185_the_probes_check_holds_the_rows_own_baseline():
    """Seeds 0 and 5 of the Phase B family (seed 5 is one the broken check read False on): every row's `check` holds,
    `readout_baseline` is the row's baseline and equals `same_sides_mean`, and `uncut_mean` stays the mean over every
    uncut tie in contact — so the probe's "run − all uncut ties" contrast is not the same quantity as its
    "run − uncut ties with as many sides" one. `summarise` then renders without exiting.
    Mutation: `check` against `uncut_mean` (the pre-fix expression) reads False on seed 5 and `summarise` raises."""
    rows = probe.rows_for_seed(0, "phase_c") + probe.rows_for_seed(5, "phase_c")
    assert rows, "the probe found no matched first cutoff to row"
    assert all(r["check"] for r in rows)
    assert all(math.isclose(r["readout_baseline"], r["same_sides_mean"]) for r in rows)
    assert any(not math.isclose(r["uncut_mean"], r["same_sides_mean"]) for r in rows), (
        "the two baselines are identical on these seeds, so the probe's central contrast reads 0")
    assert probe.summarise(rows)
