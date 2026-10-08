"""The paired statistic — Wilcoxon signed-rank on per-seed differences (plan D8).

Purpose: test a direction of difference between two coupled arms on their per-seed
         differences, defined for degenerate arms, with a declared multiplicity correction.
Spec:    docs/bowen_agent_model_spec_v2.md#M11.4e, #M17.A.3, #M17.A.4
Tests:   tests/bowen/test_ensemble.py::test_m114e_signed_rank_matches_table

NumPy only, no new dependency. Zero differences are dropped (Wilcoxon's treatment); a
difference smaller than the margin counts as zero (`M17.A.4`: it does not count toward the
direction). Ties take average ranks. For n ≤ 25 the one-sided p-value is exact, from the
null distribution of W+ by dynamic programming over integer ranks (exact when there are
no ties; with ties, ranks are doubled to stay integer). Above 25 it is the normal
approximation with the tie-corrected variance and a continuity correction. Holm's step-down
procedure corrects across the readouts of one criterion (`M11.4e`).
"""

from __future__ import annotations

import math

import numpy as np

EXACT_MAX_N = 25


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def signed_rank(differences, margin: float = 0.0) -> tuple[float, int, float]:
    """Purpose: (W+, n, one-sided p for a positive shift) of the paired differences.
    Spec:    docs/bowen_agent_model_spec_v2.md#M11.4e, #M17.A.4
    Tests:   tests/bowen/test_ensemble.py::test_m114e_signed_rank_matches_table
    """
    d = np.asarray(differences, dtype=float)
    d = np.where(np.abs(d) < margin, 0.0, d)
    d = d[d != 0.0]
    n = len(d)
    if n == 0:
        return 0.0, 0, 1.0
    ranks = _ranks(np.abs(d))
    w_plus = float(ranks[d > 0].sum())
    if n <= EXACT_MAX_N:
        doubled = np.rint(ranks * 2).astype(int)
        total = int(doubled.sum())
        counts = np.zeros(total + 1)
        counts[0] = 1.0
        for r in doubled:
            counts[r:] = counts[r:] + counts[: total + 1 - r]
        target = int(round(w_plus * 2))
        p = float(counts[target:].sum() / counts.sum())
        return w_plus, n, min(1.0, p)
    mean = n * (n + 1) / 4.0
    _, tie_counts = np.unique(np.abs(d), return_counts=True)
    variance = n * (n + 1) * (2 * n + 1) / 24.0 - float((tie_counts ** 3 - tie_counts).sum()) / 48.0
    z = (w_plus - mean - 0.5) / math.sqrt(variance)
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return w_plus, n, p


def holm(p_values: list[float], alpha: float) -> list[bool]:
    """Purpose: Holm's step-down rejection across one criterion's readouts.
    Spec:    docs/bowen_agent_model_spec_v2.md#M11.4e
    Tests:   tests/bowen/test_ensemble.py::test_m114e_holm_steps_down
    """
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    reject = [False] * len(p_values)
    for rank, i in enumerate(order):
        if p_values[i] <= alpha / (len(p_values) - rank):
            reject[i] = True
        else:
            break
    return reject
