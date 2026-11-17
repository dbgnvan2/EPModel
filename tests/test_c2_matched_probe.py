"""Smoke test for ``tools/c2_matched_probe.py``, T-197's probe (a diagnostic in no record's hash).

Purpose: the probe reads the family's sibling ranks as declared, and its within-seed regression recovers a slope
         planted in rows it is given, without running the engine.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.C.2
Tests:   this file
"""

import re

from tools import c2_matched_probe as probe


def _row(seed, child, rank, acts, load):
    return {"seed": seed, "child": child, "rank": rank, "acts_per_year": acts, "aged_load": load,
            "basic_at_5": 30.0 + rank, "anxiety_at_5": 1.0, "functional_at_5": 30.0, "load_before_5": 0.5 * rank,
            "row_treatment": load, "row_baseline": 0.0}


def test_m11c2_probe_reads_ranks_and_recovers_a_planted_slope():
    """Each child's aged load is 2 per act per year plus a seed offset, so the acts-only within-seed slope is +2.
    Mutation: the seed's mean not removed from the load (the offset then biases the slope)."""
    ranks = probe.declared_ranks()
    assert (ranks["leo"], ranks["nadia"], ranks["pia"]) == (1, 2, 3)
    acts = {"leo": 1.0, "nadia": 3.0, "pia": 2.0}
    rows = [_row(seed, child, ranks[child], acts[child] + seed, 2.0 * (acts[child] + seed) + 10.0 * seed)
            for seed in range(4) for child in acts]
    lines = probe.summarise(rows)
    acts_only = next(line for line in lines if "within seed, acts only" in line)
    slope = float(re.search(r"load per act per year ([+-][0-9.]+)", acts_only).group(1))
    assert slope == 2.0
