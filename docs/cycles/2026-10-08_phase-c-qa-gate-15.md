# Phase C QA gate 15 — 2026-10-08

learning-qa failure-pattern sweep over `origin/plan-phase-b..HEAD`. Re-gate after gate 14 was REJECTED on one
stale fallback-rate range in §1 of `docs/phase_c_completion_report.md` (the prose said "0.26–0.41" where the
rerun moved the ceiling to 0.42). Commit 8556a5f replaces the copied range with the claim it supports and a
pointer to the record. This gate verifies that fix, then re-checks every number and verdict in §1–§4 and §10
against the current generated records.

RANGE:       origin/plan-phase-b...HEAD  (caller-supplied range)
COMMITS:     5 commits:
             - d33df13 "docs: decide the ensemble precision rule (M17.A.1), before rerunning"
             - f295e58 "bowen: measure ensemble precision against the pooled sd of both arms, as decided in d33df13"
             - cfd904a "docs: fix the twelfth gate's finding — the report's C.41 summary row after the precision rerun"
             - e63b6fb "docs: fix the thirteenth gate's three stale report facts, and two more found checking the whole report"
             - 8556a5f "docs: fix the fourteenth gate's finding by pointing to the record instead of copying it"
APPLICABLE:  P19 (producer/consumer drift); the caller's explicit checks: the 8556a5f fix matches the current
             records; every remaining number/verdict in §1–§4 and §10 (including prose outside the tables)
             reconciles with the four records (treating §8 and §9's before/after tables as historical); d33df13
             and f295e58 are unchanged and no code, config or record changed since; default suite exit code.
CHECKED:     8556a5f's full diff; the fallback-rate claim against the ensemble record's fallback rates and the
             by-person breakdown; §1's summary table row by row; §1's prose (representation mutants, fallback
             rate); §2's mutant claims against the mutation record; §3 items 1–8; §4's constant-change claims
             against constants_changes.md and its test-name references; §10's results table, precision claims and
             the "Why"/"Rejected" rationale numbers (verified against the pre-rerun record at d33df13 where they
             describe the pre-rerun state); d33df13/f295e58 integrity; files changed in f295e58..HEAD; default
             suite.
NOT COVERED: The 19 ensemble-marked criterion tests (deselected by pytest.ini by design); the mutation/sweep
             records are checked for internal consistency against the report, not re-executed; domain/theory
             fidelity; §5–§9 are outside the caller's §1–§4/§10 scope (their tables are historical).

FINDINGS (ranked)

None.

VERIFIED (not findings)

- The 8556a5f fix is correct. §1's fallback paragraph (line 64–67) no longer copies any rate; it states the
  claim ("above the 0.2 flag rate") and names the criteria, pointing at the record. Every element reconciles:
  · The list (C.1, C.5, C.16, C.38, C.41) is exactly the complete set of criteria whose fallback rate exceeds
    0.2 in the current record — C.1 0.33, C.5 0.24 (0.2375…), C.16 0.37, C.38 0.30/0.36/0.42, C.41
    0.30/0.31/0.26/0.35. No criterion with rate > 0.2 is omitted, and none with rate ≤ 0.2 is included (C.3, C.4,
    C.19, C.25, C.29, C.32, C.35, C.42, C.44, C.45 and all four C.27 cells are all ≤ 0.09).
  · "Most of it is Bruno … he holds every week (rate 1.00)" — the by-person table shows bruno 1.00 in all five
    criteria; every other person is ≤ 0.48.
  · "The flag is shown beside each passing verdict it applies to" — flags appear only on C.1 (0.33), C.16 (0.37)
    and the three C.38 pairs (0.30/0.36/0.42); C.5's 0.24 and C.41's four cells are FAIL verdicts and correctly
    carry no flag. Consistent.
- §1 summary table, row by row, reconciles with the ensemble record's machine-readable entries: C.1 PASS 100
  (level-blind only); C.3 PASS 50 (−2.22, +4.02; triangle-transfer-removed + triangle-roles-swapped); C.4 FAIL
  (−1.02 holds, +0.07 p 0.43); C.5 FAIL (neither readout significant; at least one readout reverses at 5 of 6
  sweep settings — policy_temperature 0.5 is the lone no-reversal setting); C.16 PASS 150 (−0.066 ± 0.023
  p 1.15e-16; learner-disabled and learner-inverted survive; passes all 6); C.19 PASS 50; C.25 PASS 50 (0
  exactly); C.27 1-of-4 ([unstable,remove_one] PASS, [stable,add_third] +0.003 p 0.49); C.29 −0.08 p 0.09, 0
  symptom weeks; C.32 PASS 100; C.35 PASS 100; C.38 all 3 pairs; C.41 no cell passes (+0.002 p 0.32; −0.013);
  C.42 +0.26 p 0.084, passes only at credit_horizon 6, reverses at 3 of 6; C.44 +0.010 p 0.73, passes at
  learning_rate 0.1 and policy_temperature 2.0; C.45 +0.0020 p 0.20, fails all 6, reverses at 4. The intro tally
  (7 pass + 1 C.16 + 1 C.27 cell + 7 fail = 16 built, 3 not built = 19) is internally consistent.
- §1 prose reconciles: the two representation mutants (appraisal-sum-order-reversed, clamp-within-tolerance) ran
  on all 11 passing entries and left every verdict "unchanged" in the mutation record; the other two re-encodings
  are correctly described as not built (matches §5's "two of M11.1c's four re-encodings").
- §2 reconciles: C.1's single-rule mutants (steepness-level-independent, threshold-level-independent,
  standing-load-level-independent, steepness-inverted, threshold-level-inverted, standing-load-level-inverted)
  all survive; only the joint level-blind turns C.1 red. The nine-rule list matches the level-blind mutant's
  description verbatim. C.38: level-blind reds all three pairs; four of the six single-rule mutants red no pair
  and none reds all three. C.3 (triangle-transfer-removed, triangle-roles-swapped), C.27's passing cell
  (one-sided-deviation, deviation-inverted), C.32 (anger-gate-inverted/removed), C.19, C.35, C.25 all match.
- §3 items 1–8 reconcile: C.4 +0.07 p 0.43; C.5 −7.4 ± 19 (pre-§8 +5.9 ± 17 is the recorded historical value);
  C.16's pooled-sd convergence at 150 and surviving learners; C.27 +0.003 and the sign-reversed cells; C.29 0
  symptom weeks both arms; C.41 +0.002 p 0.32 and −0.013; C.42 +0.26 p 0.084 / C.45 +0.0020 p 0.20 with their
  sweep counts.
- §4 reconciles: constants_changes.md has exactly one post-freeze (2026-10-08) row — ensemble_precision
  0.25 → 0.25 "unit changed: pooled seed-to-seed sd" — matching "no constant's value changed; one constant's
  unit did". hold_gain 0.3 and its "86 of 113 at 0.5 / 62 of 110 at 0.3" choice are the pre-freeze row (line 76,
  dated 2026-10-07). All three numerical-fix test names and both M11.D.18 test names exist in the tree.
- §10 reconciles: the results table's "After §10" column matches the current records (C.16 PASS 150 −0.066 ±
  0.023 p ≈ 1e-16; C.41[light: lower level] FAIL 150 anxiety +16.4, reactive share −0.013; C.4 FAIL 100;
  C.35 PASS 100; C.27[unstable,remove_one] PASS 50; C.41's other three cells FAIL 150). "Two verdicts change" is
  exact (C.16 and C.41[light: lower level]; the rest change only seed counts). The "Why" numbers are the
  pre-rerun rationale, verified against the pre-rerun record at d33df13: baseline sd 0.027 (0.0267…), interval
  −0.063 ± 0.012 (−0.0629 ± 0.0117), margin 0.0027 = 0.1 × 0.0267, all exact. "0.196 = 1.96/√100" is exact.
  ensemble_precision = 0.25 unchanged is confirmed by constants_changes.md.
- d33df13 and f295e58 are unchanged (full hashes d33df132df36344747369bd2dd77b1aafd2fe85c and
  f295e58bd66232298c4c87e15443625d5bd6de24, matching gate 14's recorded prefixes). `git diff --name-only
  f295e58..HEAD` shows only four doc files (gate-12, gate-13, gate-14, the report) — no code, config or record
  changed after f295e58. The record-staleness tests (ensemble/mutation/sweep code_hashes) pass, so the records
  still hash-match the current tree.
- Default suite: `python3 -m pytest tests/ -q` → 492 passed, 19 deselected, exit code 0 (matches the report's
  "492 tests pass").

VERDICT: the 8556a5f fix is correct — it removed the stale copied range and now states the claim plus a pointer,
with a complete and accurate criterion list — and every remaining number and verdict in §1–§4 and §10 reconciles
with the current generated records. d33df13 and f295e58 are unchanged, no code, config or record changed after
f295e58, and the default suite is green.

APPROVED
