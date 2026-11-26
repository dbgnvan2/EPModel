"""The detection limit (`M17.A.1`, D8): a first run whose gating effect is within a declared multiple of its
half-width of zero is run again at a declared fixed seed count, and its verdict is the second run's.

Purpose: one definition of the rule for the constant sweep (``tools/version_2_sweep.py``) and the gating ensemble
         records (``tools/ensemble_record.py``, ``tools/phase_d_record.py``), which adopt it at revision 70 (T-018).
Spec:    docs/bowen_agent_model_spec_v2.md#M17.A.1
Tests:   tests/test_version_2_sweep.py::test_m17a1_s2_1_a_near_limit_sample_is_rerun_at_the_fixed_count,
         tests/bowen/test_ensemble_record.py::test_m17a1_t018_a_near_limit_criterion_is_rerun_at_the_fixed_count

The multiple and the count are ``detection_limit_multiple`` and ``detection_limit_seeds`` in ``config/sweep/sweep.md``,
outside the engine's config: they are inputs to a measurement of the model, not to the model.
"""

from __future__ import annotations

from tools.sweep_config import CONFIG, load_rules


def near_limit(row: dict, multiple: float) -> bool:
    """D8: any gating directional readout whose mean difference is within ``multiple`` half-widths of zero. An
    equivalence limb (direction 0) has no detection limit of this kind and is not read (`M17.A.1`)."""
    gating = [r for r in row.get("readouts", ()) if not r.get("report_only") and r.get("direction")]
    return any(abs(r["mean_difference"]) < multiple * r["half_width"] for r in gating)


def limits() -> tuple[float, int]:
    """The declared multiple and fixed seed count."""
    rules = load_rules(CONFIG.read_text(encoding="utf-8"))
    return float(rules["detection_limit_multiple"]), int(rules["detection_limit_seeds"])


def needs_rerun(first: dict, own_seeds: int | None, multiple: float, fixed: int) -> bool:
    """A criterion's first verdict is run again when it is near the limit on fewer than ``fixed`` seeds. A criterion
    that fixes its own seed count (``own_seeds``) is not: the runner reads that count before the block and cap, and the
    records' fixed counts (600 and 200) are at or above the declared one."""
    return own_seeds is None and first["seeds"] < fixed and near_limit(first, multiple)


def rerun_rules(rules: dict, fixed: int) -> dict:
    """The ensemble rules for the second run: one block of ``fixed`` seeds, capped there."""
    return {**rules, "ensemble_block": fixed, "ensemble_cap": fixed}


def rerun_definition(multiple: float, fixed: int) -> dict:
    """The second run's cache definition, so a change of either declared value reruns it."""
    return {"detection_limit": [multiple, fixed]}


def seeds_cell(seeds: int, first: dict | None) -> str:
    """The record's seed count: both counts, and the first run's verdict, when the criterion was run again."""
    return str(seeds) if first is None else f"{seeds} (first run {first['seeds']} seeds, {first['outcome']})"
