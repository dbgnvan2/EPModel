# Phase C QA gate 17 — 2026-10-08

learning-qa failure-pattern sweep over `origin/plan-phase-b..HEAD`, a re-gate of the sixteenth gate's
REJECTED verdict. Two commits: `91f8963` (the M11.C.16 composite decision) and `b6a42eb` (the fix
commit claiming to resolve all four of gate 16's findings). Gate 16's four findings were: (1) §1's
"six other" over-count, (2) `mixing-weight-inverted` never run on C.16, (3) TODO's incomplete survivor
enumeration, (4) §10's singular "a rule that reads level" claim.

RANGE:       origin/plan-phase-b...HEAD  (caller-supplied range)
COMMITS:     2 commits:
             - 91f8963 "spec: M11.C.16 stays composite; its rationale and mutation clause are not met"
             - b6a42eb "spec: fix the sixteenth gate's findings on the M11.C.16 decision"
APPLICABLE:  P19 (producer/consumer drift — summary prose vs. generated records); P26 (fix-commit
             regressions — a fix that leaves or introduces a subtler contradiction); M11.1d (each rule in a
             criterion's set MUST run as both sign-inverted and deletion); M11.1a (a mutant must apply
             exactly once); the caller's explicit checks: every number/count/mutant name in the changed
             text of the report (§1, §10, §11), the spec's M11.C.16 rows, TODO.md and CHANGELOG.md
             reconciles with the mutation and ensemble records; the records are current; the decision
             follows from the record; default suite exit code.
CHECKED:     b6a42eb's full diff (8 files); each new/changed mutant's old-string occurrence count in the
             source tree (exact-once); the §11 mutant table and the §1 summary cell against
             `docs/phase_c_mutation_record.md` row by row; the spec's M11.C.16 criterion row and M11.5
             class row against the record; CHANGELOG and TODO enumerations against the record; the three
             record code_hashes recomputed against the current tree; the ensemble and sweep records for
             the C.16 verdict and its six sweep settings; the default suite.
NOT COVERED: The 19 ensemble-marked criterion tests (deselected by pytest.ini by design); the mutation and
             sweep records are checked for internal consistency and hash-currency, not re-executed;
             domain/theory fidelity of the composite decision (consistency with the spec's definitions
             only, not Bowen fidelity).

FINDINGS (ranked)

1.  LOW — P19/P26 — docs/phase_c_completion_report.md:374-375. §10's amended sentence now reads "§11
    tests which rules do. One possible rule, not tested: `M4.D.3a`'s layer availability removes acts from
    the legal set at a lower level …", yet two lines later (376-377) the same paragraph says "that
    candidate rule turned out not to carry it", and §11's own table lists `availability-level-independent`
    and `availability-level-inverted` as run and survived. "Not tested" is contradicted by both the
    paragraph's closing sentence and §11. This is the residue of gate 16's finding 4: the singular claim
    was removed, but the stale "not tested" fragment was left in place, and the fix commit's own added
    sentence ("§11 tests which rules do") makes the contradiction worse. Fix: drop "not tested" — e.g.
    "One possible rule — `M4.D.3a`'s layer availability, which removes acts from the legal set at a lower
    level and lowers the entropy of what is chosen — §11 tests, and it does not carry the direction."

2.  NIT — P19 — CHANGELOG.md:80. "No single rule carries it, and none states it" is flatter than the two
    statements it summarises. The report's §11 (:402) says "only `M4.D.3a`'s availability states a
    narrowing with level", and the spec's M11.5 row says "availability is the one rule that narrows the
    repertoire with level". Availability does state a level-dependent narrowing; it just does not state the
    result on its own (inverting it does not flip C.16). "None states it" overstates unless "it" is read as
    "the result as a premise". Not a false claim in context, but it disagrees on the surface with the
    precise phrasing in §11 and the spec. Recommend aligning with "no single rule states the result on its
    own".

VERIFIED (not findings)

- All four gate-16 findings are fixed. (1) The §1 cell now reads "as do the 6 other single-rule and paired
  mutants run on it": adding `mixing-weight-inverted` to C.16 raised the survivor count from six to seven,
  so "6 other" (seven survivors minus the required learner-disabled) is now correct — the over-count is
  resolved by completing the audit, not by re-wording alone. (2) `mixing-weight-inverted` now targets
  `("M11.C.41", "M11.C.16")` and the record carries `mixing-weight-inverted → M11.C.16 → PASS 150
  survived`. (3) TODO's enumeration now names the learner, layer availability and the mixing weight each
  surviving both deletion and inversion, plus availability-with-learner-removed and the joint level-blind —
  complete. (4) §10's singular "produced by a rule that reads level" is gone; the residual above is a
  different, narrower defect.
- The §11 table matches the mutation record exactly: seven survivors on C.16 (learner-disabled,
  learner-inverted, availability-level-independent, availability-level-inverted,
  availability-and-learner-removed, mixing-weight-level-independent, mixing-weight-inverted) and one red
  (level-blind), at 150 seeds except level-blind (50). The spec's M11.C.16 criterion row ("M4.D.6 …
  MUST turn this red. Not met, 2026-10-08: it survives") and the M11.5 class row (six "disabled or
  inverted … leave it passing" items plus "only the joint level-blind mutant turns it red") reconcile with
  the same record.
- Every new/changed mutant's old string occurs exactly once in the source tree (M11.1a):
  `AVAILABILITY` → `src/bowen/policy/policy.py` (1); the mixing-weight expression → `policy.py` (1); the
  learner delta → `src/bowen/engine/learner.py` (1); the steepness read → `src/bowen/engine/contact.py` (1).
  The joint `availability-and-learner-removed` names both writes via `also=`.
- The decision follows from the record. C.16 passes in the ensemble record (150 seeds, entropy −0.0657 ±
  0.023, p 1.15e-16) and survives every single-rule and paired mutant; only the joint level-blind reds it.
  The three rules each survive both deletion and inversion, so no rule states the result on its own and
  M11.1d's premise test does not fire — composite is correct. The §11 reasoning tracks this precisely.
- Records are current. Recomputing `code_hash`: mutation record `e0a73051…` matches
  `code_hash(mutation_record, ensemble_record)`; sweep record `cab81457…` matches
  `code_hash(sweep_record, mutation_record, ensemble_record)`; ensemble record `f5aa9075…` matches its own
  tool. The sweep record's six C.16 rows are all PASS (learning_rate 0.1/0.4, credit_horizon 2/6,
  policy_temperature 0.5/2.0), so §1's "passes at all 6 settings" holds.
- Coverage override is honest: `docs/spec_coverage.md` and `tools/spec_coverage_phase_b.json` both pin C.16
  to "partial" with a note matching the spec's "clause not met" wording.
- Default suite: `python3 -m pytest tests/ -q` → 492 passed, 19 deselected, exit code 0.

VERDICT: the decision is correct and well-evidenced, all four gate-16 findings are fixed, every mutant
applies exactly once and tests what its description says, the §11 and spec rows reconcile with the record,
the records are current, and the suite is green. But the fix to finding 4 left a stale "not tested" clause
in §10 that now contradicts §11 and its own next sentence — a P26 fix-commit residue of the same class the
gate was re-gated for. Low severity and a one-line fix, but it is a finding, not clean.

REJECTED
