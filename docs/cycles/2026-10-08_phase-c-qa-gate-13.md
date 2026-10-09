# Phase C QA gate 13 — 2026-10-08

learning-qa failure-pattern sweep over `origin/plan-phase-b..HEAD`. Re-gate after gate 12 was REJECTED on one
Low finding (the §1 summary table's `M11.C.41` row carried two pre-§10 facts). Reviews the fix commit plus the
whole §1 summary table and §3 against the current generated records, per the caller's request.

RANGE:       origin/plan-phase-b...HEAD  (caller-supplied range)
COMMITS:     3 commits:
             - d33df13 "docs: decide the ensemble precision rule (M17.A.1), before rerunning"
             - f295e58 "bowen: measure ensemble precision against the pooled sd of both arms, as decided in d33df13"
             - cfd904a "docs: fix the twelfth gate's finding — the report's C.41 summary row after the precision rerun"
APPLICABLE:  P19 (producer/consumer drift); the caller's explicit checks: the C.41 fix matches
             phase_c_ensemble_record.md; the rest of §1 and §3 hold no other stale fact; d33df13 and f295e58 are
             unchanged by the fix commit; default suite exit code.
CHECKED:     cfd904a's diff; the §1 table row by row against the machine-readable entries of
             phase_c_ensemble_record.md and the §5/§8/§9/§10 result tables; every numeric claim in §3 (items 1–8)
             against the ensemble, sweep and mutation records; git history of the report's stale lines to date when
             each drifted; d33df13/f295e58 integrity; the record-hash tests' input; default suite.
NOT COVERED: The 19 ensemble-marked criterion tests (deselected by pytest.ini by design); the mutation/sweep records
             are checked for internal consistency against the report, not re-executed; domain/theory fidelity.

FINDINGS (ranked)

1. P19 · docs/phase_c_completion_report.md:47 · Low · confidence High
   The §1 summary table's `M11.C.29` row is stale. It reads "budget −0.07 (p 0.17)".
   phase_c_ensemble_record.md's current entry (line 21; JSON mean_difference −0.08, p 0.089782…) reads
   `budget` −0.08 ± 0.097, p 0.0898. The report's "−0.07 (p 0.17)" is the pre-TRIANGLE value
   (−0.0667 ± 0.15, p 0.171), changed when the §8 decision (cf067f2) regenerated the record; the §1 row was never
   updated. Same pattern as gate 12's C.41 finding, one row up: a top-of-report summary cell contradicts the
   generated record the report itself is sourced from.
   Fix: "budget −0.07 (p 0.17)" → "budget −0.08 (p 0.09)".

2. P19 · docs/phase_c_completion_report.md:113 · Low · confidence High
   §3.3's explanation of `M11.C.5` carries a pre-§8 value. It reads "wide intervals (±17 on a difference of +6)".
   The current record (line 17) reads `target_reaction` −7.44 ± 19 and `third_person_symptom_load` −8.39 ± 11.
   "±17 on +6" is the pre-TRIANGLE `target_reaction` (+5.91 ± 17, p 0.334); the §8 decision (cf067f2) moved it to
   −7.44 ± 19, flipping the sign. Written at dfe3ed9, never updated.
   Fix: restate against the current readouts, e.g. "wide intervals (±19 on −7.44)" and note the sign changed.

3. P19 · docs/phase_c_completion_report.md:130-131 · Low · confidence High
   §3.8's "why C.42/C.45 fail" prose carries pre-§9 values and a pre-§9 sweep claim. It reads "C.42's reuse is
   −0.17 (p 0.85) and it fails at every sweep setting" and "C.45's … +0.0024 (p 0.17)". The current record reads
   C.42 `triangle_reuse` +0.26 ± 0.33 (p 0.084) and C.45 `triangle_rate` +0.0020 (p 0.197); §9 (lines 311-312) shows
   exactly this before→after transition. §3.8 is framed "after §8", which mitigates but does not cure it: §3's job is
   to explain why each criterion now fails, and the §9 readout change moved C.42's difference to the claimed sign,
   so the operative reason (and the "fails at every sweep setting" claim, contradicted by §9's "passes only at
   H = 6" and the sweep record) is no longer what §3.8 says.
   Fix: update §3.8 to the post-§9 values and point at §9 for the readout correction, or fold it into §9.

VERIFIED (not findings)

- The gate-12 fix is correct. Line 51 now reads "its reactive-share limb is now +0.002 (p 0.32). [light: lower
  level] now fails: its reactive share falls (−0.013), §10", matching phase_c_ensemble_record.md's
  `M11.C.41[lower level: heavier stress]` (reactive_share +0.00196 ± 0.0096, p 0.321) and
  `M11.C.41[light: lower level]` (reactive_share −0.0131 ± 0.0096, p 0.989, "does not hold"). Both pre-§10 facts
  ("+0.004 (p 0.16)" and "is undetermined") are gone. This finding is closed.
- Every other §1 numeric verdict row reconciles with the ensemble record: C.1 PASS 100; C.3 PASS 50 (−2.22,
  +4.02); C.4 FAIL (−1.02 holds; +0.0664 p 0.432 → "+0.07 (p 0.43)"); C.5 FAIL; C.16 PASS 150 (−0.0657 ± 0.023,
  p 1.15e-16); C.19 PASS 50; C.25 PASS 50 (0 exactly); C.27 one-of-four, [stable,add_third] +0.00308 → "+0.003
  (p 0.49)"; C.32 PASS 100; C.35 PASS 100; C.38 PASS all 3 pairs; C.41's four cells; C.42 +0.26 (p 0.084); C.44
  +0.0103 → "+0.010 (p 0.73)"; C.45 +0.00204 → "+0.0020 (p 0.20)". The C.16 mutation claim ("learner-disabled and
  learner-inverted both survive") matches the mutation record (lines 19, 50). The intro's 7-pass/1-cell/7-fail
  tally (16 built + 3 not built = 19) is internally consistent.
- Every §1 sweep claim matches the sweep record: C.5 reverses at 5 of 6; C.16 passes at all 6; C.27's passing cell
  holds at all 6; C.42 passes only at credit_horizon 6 (H = 6) and reverses at 3; C.44 passes at 2 (learning_rate
  0.1, policy_temperature 2.0); C.45 fails at all 6 and reverses at 4.
- §3 items 1, 2, 4, 5, 6, 7 carry no stale numeric fact. Item 7 (C.41) already agrees with the record.
- d33df13 and f295e58 are unchanged. cfd904a's diff touches only the new gate-12 report and the single §1 C.41 line;
  it does not touch src/bowen/ensemble/runner.py, the two toy tests, the three records, or the constants files those
  two commits changed. Both hashes are identical to what gate 12 reviewed (d33df13, f295e58).
- Default suite: `python3 -m pytest tests/ -q` → 492 passed, 19 deselected, exit code 0.

VERDICT: the gate-12 finding is fixed correctly and no code or record changed since, but the sweep requested on
the rest of §1 and §3 finds three further stale facts of the same P19 class (C.29 in §1; C.5 and C.42/C.45 in §3),
each a report cell that contradicts a current generated record. All three are doc-only, but the gate's own
precedent is to reject on a single stale summary row, and the report is the source of truth for the gate.

REJECTED
