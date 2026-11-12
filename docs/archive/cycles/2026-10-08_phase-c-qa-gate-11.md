# Phase C QA gate 11 — 2026-10-08

learning-qa failure-pattern sweep over `origin/plan-phase-b..HEAD`. Re-gate of the fix commit for the
tenth gate's four low findings.

RANGE:       origin/plan-phase-b...HEAD  (caller-supplied range, step-1 rule 1)
COMMITS:     1 commit: 0a50215 — "spec: fix the tenth gate's four findings on the M11.C.1/.38 decision"
APPLICABLE:  P6 (claim vs. artifact), P19 (producer/consumer drift, incl. its source-text corollary); plus the
             caller's explicit consistency checks (M11.5 definition, M11.1d, rule-ID existence, mutant claims,
             report/TODO/CHANGELOG agreement, no code/config/record change, test suite).
CHECKED:     Each of gate-10's four findings re-verified against `docs/phase_c_mutation_record.md`, the spec's
             `M11.5` definition and M11.C.1/.38/.45 rows, `docs/phase_c_completion_report.md` §1/§2, `TODO.md`
             and `CHANGELOG.md`; mutant-claim fidelity re-derived from the six single-rule and joint level-blind
             C.38 rows; diff scope; default suite exit code.
NOT COVERED: This is a failure-pattern + document-consistency sweep only. Not assessed: domain/theory fidelity of
             the "written-in vs. emergent" reasoning against the corpus; whether the nine-rule set is truly minimal
             for onset (M11.1d's "minimal set" clause); the cold-review blind spot (this sweep is the only pass).
             The 19 ensemble-marked tests are deselected by design and were not run.

FINDINGS (ranked)

None. All four of the tenth gate's findings are fixed correctly, and the fix commit introduces no new defect.
One non-blocking observation is recorded under VERIFIED (not findings) below.

VERIFIED (not findings)

- Finding 1 (CHANGELOG.md:28-29, P19). The stale "`.45` a premise" clause is gone. It now reads "`.45` proposed
  as a premise, withdrawn when it stopped passing after the `TRIANGLE` decision (it stays composite)", which
  matches the spec's M11.C.45 row ("Composite … the reclassification is withdrawn until it passes again",
  bowen_agent_model_spec_v2.md:1309), the report §2 ("Neither passes after §8, so neither correction stands",
  phase_c_completion_report.md:94-97) and TODO.md:29-30 ("`M11.C.45`'s reclassification as a premise was
  withdrawn when it stopped passing after the `TRIANGLE` decision").
- Finding 2 (docs/bowen_agent_model_spec_v2.md:1265, P6). The unsupported "and it leaves the arms' difference at
  exactly zero" clause is removed. The fix took the "drop the clause" branch of the two offered, which is the
  correct one: `docs/phase_c_mutation_record.md` persists only outcome/seeds/result, so the magnitude is in no
  committed record. The remaining C.1 claims are supported: six single-rule mutants on C.1 all survived
  (steepness-level-independent, threshold-level-independent, threshold-level-inverted,
  standing-load-level-independent, standing-load-level-inverted, steepness-inverted — record rows 14, 24, 28, 32,
  36, 45), and only the joint `level-blind` mutant turns C.1 red (row 40).
- Finding 3 (docs/bowen_agent_model_spec_v2.md:1302, P19). The note now reads "no single-rule mutant reds all
  three adjacent pairs (four of the six red none); the joint level-blind mutant reds all three". Re-derived from
  the record: of the six single-rule mutants on C.38, steepness-level-independent, threshold-level-independent,
  standing-load-level-independent and standing-load-level-inverted red zero of the three pairs; threshold-level-
  inverted reds one; steepness-inverted reds two; none reds all three (rows 15-17, 25-27, 29-31, 33-35, 37-39,
  46-48). The joint `level-blind` mutant reds all three (rows 41-43). Exact.
- Finding 4 (docs/bowen_agent_model_spec_v2.md:1261, spec-internal consistency). The M11.5 premise definition now
  admits a jointly-stated, non-emergent premise: "A criterion that several rules each state, redundantly, is a
  premise too: the result is written in, not produced by learning or time, even when no one of those rules is
  necessary. Its proof is a mutant that removes the whole set (`M11.C.1`, `M11.C.38`)." The definition and the two
  table rows now agree, and the premise/composite distinction stays clean: a premise is written in (each rule
  states the direction, even redundantly), a composite is produced by mechanisms acting together through learning
  or time. The report §2 carries the same sentence (phase_c_completion_report.md:85-86), and TODO.md:27-30
  records the owner decision it implements.
- CHANGELOG.md:73-75 ("Fixed"): the new bullet accurately summarises the four fixes, including "four of six red
  none" and the removal of the "exactly zero" claim.
- Observation (non-blocking): phase_c_completion_report.md:49's §1 table cell reads "single-rule mutants red on
  some pairs only (§2)". This is a collective summary, not the per-mutant overstatement that finding 3 flagged
  ("each single-rule mutant reds some pairs"), and it points at §2, which is now exact. It is consistent with the
  record and was left unchanged; noted for completeness only.
- No code, config, or record changed: the diff touches only CHANGELOG.md, docs/bowen_agent_model_spec_v2.md and
  docs/phase_c_completion_report.md. The three generated records (phase_c_ensemble_record.md,
  phase_c_mutation_record.md, phase_c_sweep_record.md) and all source/config/tools files are untouched.
- Default suite: `python3 -m pytest tests/ -q` → 490 passed, 19 deselected, exit code 0.

VERDICT: clean against P1-P37, of which 2 were applicable, with no findings (four gate-10 findings verified fixed,
no new defect introduced by the fix commit).

APPROVED
