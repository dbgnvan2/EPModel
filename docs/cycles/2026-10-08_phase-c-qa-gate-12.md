# Phase C QA gate 12 — 2026-10-08

learning-qa failure-pattern sweep over `origin/plan-phase-b..HEAD`. Reviews the two commits that decide and
implement the ensemble stopping rule's precision (M17.A.1): the decision committed before rerunning, and its
implementation plus all three regenerated Phase C records.

RANGE:       origin/plan-phase-b...HEAD  (caller-supplied range)
COMMITS:     2 commits:
             - d33df13 "docs: decide the ensemble precision rule (M17.A.1), before rerunning"
             - f295e58 "bowen: measure ensemble precision against the pooled sd of both arms, as decided in d33df13"
APPLICABLE:  P19 (producer/consumer drift); plus the caller's explicit checks: decision↔implementation↔spec
             (M17.A.1, M13.4) agreement; margin/equivalence bound on baseline sd (plan D8); new tests fail under
             the old rule; constants change logged honestly; §10 tables match the three records; nothing tuned;
             default suite exit code.
CHECKED:     src/bowen/ensemble/runner.py (_pooled_scale, _converged, _verdict, _scale) against the decision text
             and spec M17.A.1/M13.4 and plan D8; plan D8 text (implementation_plan_phase_c.md §D8); the two new
             tests run under the current rule and under the old rule via a monkeypatch probe; config/bowen/constants.md
             and constants_changes.md; §10's table against the machine-readable entries of phase_c_ensemble_record.md,
             phase_c_mutation_record.md, phase_c_sweep_record.md; the §1 summary table and §3.7/§8 against the same
             records; record-hash tests; default suite.
NOT COVERED: Domain/theory fidelity of the pooled-sd rationale against the corpus; the 19 ensemble-marked criterion
             tests (deselected by pytest.ini by design, not run here); the mutation/sweep records are checked for
             internal consistency and against §10, not re-executed.

FINDINGS (ranked)

1. P19 · docs/phase_c_completion_report.md:51 · Low · confidence High
   The §1 summary verdict table's `M11.C.41` row was not updated for §10. Two facts in it are stale:
   (a) "its reactive-share limb is now +0.004 (p 0.16)" — the §10 rerun moved `[lower level: heavier stress]`'s
       reactive share to +0.002 (p 0.32): §3.7 item 7 (line 124) says "+0.002 (p 0.32)", and
       phase_c_ensemble_record.md's machine-readable entry has mean_difference +0.00196, p 0.321.
   (b) "[light: lower level] is undetermined" — §10 changed that cell to FAIL: §10's own table (line 361) and
       phase_c_ensemble_record.md show reactive_share −0.0131 (p 0.989), "does not hold".
   Risk: the report is the source of truth for the gate. Its top summary now contradicts the report's own §10 and
   §3.7 and the generated record, so a reader trusting the summary table reads two wrong verdict facts about C.41.
   Fix: rewrite line 51 to "+0.002 (p 0.32). [light: lower level] now fails: its reactive share falls (−0.013)",
   replacing "+0.004 (p 0.16)" and "is undetermined".

VERIFIED (not findings)

- Decision ↔ implementation ↔ spec. `_pooled_scale` (runner.py:127) computes √((var_baseline² + var_treatment²)/2),
  falling back to the differences' sd only when the pooled sd is exactly 0 — matching d33df13's "when both arms have
  no spread, the differences' sd is used, as before". `_converged` (runner.py:154) uses it as the precision ruler;
  `M17.A.1`'s half-width comparison (1.96·sd(d)/√n vs ensemble_precision·scale) and `M13.4`'s UNDETERMINED-at-cap
  (runner.py:201-202) are intact. Matches spec M17.A.1 (spec:1898) and M13.4 (spec:1470).
- Margin and equivalence bound stay on the baseline sd, as plan D8 requires. `_verdict` still calls `_scale(base,
  diff)` (runner.py:167) — baseline seed-to-seed sd with a differences'-sd fallback — for both the equivalence
  bound (runner.py:178) and the signed-rank margin (runner.py:182). Only the stopping rule uses `_pooled_scale`.
  plan D8 (implementation_plan_phase_c.md:330-331) governs margins only; the decision's "That precision used the
  same scale was the build's choice" is accurate.
- New tests fail under the old rule (verified empirically, not by inspection). Monkeypatching `_pooled_scale` back to
  the baseline-sd `_scale`: `test_m17a1_precision_is_measured_against_both_arms_spread`'s arm (effect 1.0, noise 0.2,
  extra_noise 2.0) goes UNDETERMINED at 500 (vs PASS at 150 under the new rule); the swap test's two arms stop at 500
  vs 100 (vs 150 == 150 under the new rule). Both assertions flip from True to False under the old rule.
- Constants change logged honestly. config/bowen/constants.md:152 changes only the unit description ("pooled
  seed-to-seed sd of the two arms"), the value stays 0.25. config/bowen/constants_changes.md:107 records frozen 0.25
  → 0.25 with "(unit changed: pooled … was the baseline arm's)", post_hoc = yes, dated 2026-10-08, "decided before
  rerunning". ensemble_margin and equivalence_margin keep their "baseline seed-to-seed sd" unit (constants.md:153,156).
- §10 tables match the three records. Every row of §10's "Results" table reconciles: C.16 UNDETERMINED@500 → PASS@150
  (−0.066 ± 0.023, p ≈ 1e-16; record −0.0657 ± 0.023, p 1.15e-16); C.41[light: lower level] → FAIL@150 (+16.4, −0.013;
  record +16.39, −0.0131); C.4 FAIL@50→100; C.35 PASS@200→100; C.27[unstable,remove_one] PASS@100→50; C.41's other
  three cells FAIL@200-300→150. The mutation record shows learner-disabled and learner-inverted both survived on C.16
  at 150, and both representation mutants unchanged, matching §10's "neither learner mutant turns it red"; the sweep
  record shows C.16 PASS at all six off-central settings, matching "passes at all six settings". Record-hash tests
  (test_ensemble_record.py) pass, so the regenerated hashes are current.
- Nothing tuned. No numeric constant changed value (only the ensemble_precision unit description). The sole code
  change is the stopping-rule scale, decided and committed (d33df13) before any run; margins and the equivalence
  bound are untouched. The toy-arm change (tests/bowen/ensemble_toys.py) shares the extra-noise draw across arms so
  the swap test is exact; it is a test fixture, not a model change.
- Default suite: `python3 -m pytest tests/ -q` → 492 passed, 19 deselected, exit code 0. Matches the updated
  490 → 492 count (two new default-suite tests) in CLAUDE.md, README.md and the report.

VERDICT: one finding (P19 · phase_c_completion_report.md:51 · Low) — the §1 summary table's C.41 row still carries
two pre-§10 facts and contradicts the report's own §10 and §3.7 and the ensemble record. A one-line doc fix. Clean
against P1-P37 with one applicable pattern found.

REJECTED
