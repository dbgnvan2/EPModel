# Phase C QA gate 14 — 2026-10-08

learning-qa failure-pattern sweep over `origin/plan-phase-b..HEAD`. Re-gate after gate 13 was REJECTED on three
stale facts (the §1 `M11.C.29` row, §3.3's `M11.C.5` interval, §3.8's C.42/C.45 values). Commit e63b6fb claims to
fix those three plus two more found checking the whole report (§3.2's nodal-anxiety wording, §4's opening about
post-freeze constant changes). This gate verifies all five fixes and then re-checks every number and verdict in
§1–§4 and §10 against the current generated records.

RANGE:       origin/plan-phase-b...HEAD  (caller-supplied range)
COMMITS:     4 commits:
             - d33df13 "docs: decide the ensemble precision rule (M17.A.1), before rerunning"
             - f295e58 "bowen: measure ensemble precision against the pooled sd of both arms, as decided in d33df13"
             - cfd904a "docs: fix the twelfth gate's finding — the report's C.41 summary row after the precision rerun"
             - e63b6fb "docs: fix the thirteenth gate's three stale report facts, and two more found checking the whole report"
APPLICABLE:  P19 (producer/consumer drift); the caller's explicit checks: all five e63b6fb fixes match the current
             records; every remaining number/verdict in §1–§4 and §10 reconciles with the four records (treating §8
             and §9's before/after tables as historical); d33df13 and f295e58 are unchanged and no code, config or
             record changed since; default suite exit code.
CHECKED:     e63b6fb's full diff; §1's summary table row by row against the ensemble record's machine-readable
             entries; §1's prose (representation mutants, fallback rate); §2's mutant claims against the mutation
             record; §3 items 1–8; §4's constant-change claims against constants_changes.md; §10's results table and
             precision claims; the pre-§8 C.5 value and the pre-rerun fallback ceiling via git history; d33df13/
             f295e58 integrity; files changed in f295e58..HEAD; default suite.
NOT COVERED: The 19 ensemble-marked criterion tests (deselected by pytest.ini by design); the mutation/sweep records
             are checked for internal consistency against the report, not re-executed; domain/theory fidelity; §5–§9
             are outside the caller's §1–§4/§10 scope (their tables are historical).

FINDINGS (ranked)

1. P19 · docs/phase_c_completion_report.md:64 · Low · confidence High
   §1's fallback-rate prose says the rate is "above the 0.2 flag rate in the Phase C family runs (0.26–0.41)".
   The current record's highest criterion fallback rate above 0.2 is 0.42, not 0.41:
   `M11.C.38[-10 vs -15]` reads 0.42, flagged "fallback rate 0.42 above 0.2" (ensemble record line 33; JSON
   fallback_rate 0.423016…). The 0.41 was the pre-rerun value: before the precision rerun (f295e58) that same
   cell read 0.41 (dfe3ed9 line 33). The rerun moved the ceiling to 0.42 and the report's range was never
   updated — the same P19 class as the three facts gate 13 rejected, in the same §1 the fix commit claims to
   have checked exhaustively. (The lower bound is also not clean: the record's minimum rate above 0.2 is 0.24
   at `M11.C.5`, and the flagged passing rates are 0.30–0.42; the ceiling is the unambiguous, rerun-introduced
   error and is the basis of this finding.)
   Fix: "0.26–0.41" → "0.26–0.42" (or restate precisely against the record: the flagged rates are 0.30–0.42).

VERIFIED (not findings)

- All five e63b6fb fixes are correct against the current records:
  · §1 C.29 row (line 47) now "budget −0.08 (p 0.09)" — ensemble `budget` −0.08 ± 0.097, p 0.0898. The pre-§8
    "−0.07 (p 0.17)" is gone.
  · §3.2 C.4 (line 110–111) now "not detectably higher … (+0.07, p 0.43)" — ensemble `family_anxiety_at_nodal`
    +0.0664, p 0.432. Correct.
  · §3.3 C.5 (line 113) now "target reaction −7.4 ± 19; it was +5.9 ± 17 before §8" — current record −7.44 ± 19
    (half-width 18.64); pre-§8 value verified in git (cf067f2~1) as +5.91 ± 17, p 0.334. Both halves correct.
  · §3.8 C.42/C.45 (lines 130–135) now post-§9: C.42 +0.26 (p 0.084) passing only at H = 6, reversing at 3 of 6;
    C.45 +0.0020 per person-week (p 0.20), reversing at 4 of 6 — all match the ensemble record (0.26 p 0.0841;
    0.00204 p 0.197) and the sweep record (C.42 reverses at learning_rate 0.1, credit_horizon 2, policy_temperature
    0.5; C.45 reverses at learning_rate 0.1/0.4, credit_horizon 2, policy_temperature 0.5).
  · §4 (lines 140–142) now "no constant's value changed … one constant's unit did: ensemble_precision". Verified:
    constants_changes.md has exactly one post-freeze (2026-10-08) row — `ensemble_precision` 0.25 → 0.25 "unit
    changed: pooled seed-to-seed sd … was the baseline arm's".
- Every other §1 verdict/number reconciles: C.1 PASS 100 (level-blind only); C.3 PASS 50 (−2.22, +4.02); C.4 FAIL
  (−1.02 holds; +0.07 p 0.43); C.5 FAIL, "at least one readout reverses at 5 of 6" (policy_temperature 0.5 is the
  lone setting where neither reverses); C.16 PASS 150 (−0.0657 ± 0.023, p 1.15e-16; learner-disabled and
  learner-inverted both survive; passes all 6); C.19 PASS 50; C.25 PASS 50 (0 exactly); C.27 1-of-4 ([stable,
  add third] +0.003 p 0.49); C.29 −0.08 p 0.09; C.32 PASS 100; C.35 PASS 100; C.38 all 3 pairs; C.41 no cell passes
  (+0.002 p 0.32; −0.013); C.42 +0.26 p 0.084, H = 6, 3-of-6; C.44 +0.010 p 0.73, 2-of-6 (α 0.1, T 2.0); C.45
  +0.0020 p 0.20, all 6 fail, 4 reversed. The intro tally (7 pass + 1 C.16 + 1 C.27 cell + 7 fail = 16 built, 3 not
  built = 19) is internally consistent. The two representation mutants (order-reversed sum, 1e-12 clamp) ran on
  every passing entry and left every verdict unchanged, matching the mutation record.
- §2 reconciles: C.1's single-rule mutants (steepness/steepness-inverted, threshold/…-inverted, standing-load/
  …-inverted, and the level-independent deletions) all survive; only the joint `level-blind` is red; the nine-rule
  list matches the `level-blind` mutant's description verbatim. C.38: four of the six single-rule mutants red no
  pair, none reds all three, `level-blind` reds all three. C.3 (triangle-transfer-removed, triangle-roles-swapped
  both red), C.27's passing cell (deviation-inverted red), and C.32/C.19/C.35/C.25 all match.
- §3 items 1, 4, 5, 6, 7 carry no stale numeric fact (C.16's 150-seed convergence and surviving learners; C.27's
  flat +0.003 and the reversed cells; C.29's 0 symptom weeks in both arms and the right-sign non-significant budget;
  C.41's +0.002/−0.013 and rising anxiety with falling reactive share).
- §4's remaining numbers reconcile: hold_gain 0.3 and its "86 of 113 at 0.5 / 62 of 110 at 0.3" choice match
  constants_changes.md line 76 (dated 2026-10-07, before the freeze). No other post-freeze row exists.
- §10's "After §10" column matches the current records (C.16 PASS 150 −0.066 ± 0.023 p ≈ 1e-16; C.41[light: lower
  level] FAIL 150 anxiety +16.4, reactive share −0.013; C.4 FAIL 100; C.35 PASS 100; C.27[unstable,remove_one] PASS
  50; C.41's other three cells FAIL 150). The "before" numbers are the pre-rule state, consistent with §10's own
  before/after table. ensemble_precision = 0.25 unchanged; the 0.196 = 1.96/√100 claim is exact.
- d33df13 and f295e58 are unchanged (hashes d33df132…, f295e58b…). `git diff --name-only f295e58..HEAD` shows only
  three doc files (gate-12, gate-13, the report) — no code, config, or record changed after f295e58. The
  ensemble/mutation/sweep code_hashes still match the current tree (the staleness tests pass).
- Default suite: `python3 -m pytest tests/ -q` → 492 passed, 19 deselected, exit code 0 (matches the report's
  "492 tests pass").

VERDICT: the five e63b6fb fixes are all correct and no code, config or record changed since f295e58, but the
full §1–§4/§10 check finds one further stale fact of the same P19 class that the fix commit missed: §1's
fallback-rate ceiling "0.41" contradicts the current record's 0.42 (C.38[-10 vs -15]), moved by the precision
rerun. Doc-only and 0.01 in magnitude, but the gate's precedent is to reject on a single stale summary number
contradicting a generated record, and this is the same §1 the fix commit claimed to have checked exhaustively.

REJECTED
