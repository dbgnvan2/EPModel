# Phase C QA gate 16 — 2026-10-08

learning-qa failure-pattern sweep over `origin/plan-phase-b..HEAD`. One commit, `91f8963`: the decision that
`M11.C.16` stays in `M11.5`'s composite class, with three new mutants on C.16 in `tools/mutation_record.py`,
two re-targeted existing mutants, regenerated mutation and sweep records, a coverage override, and spec, report
(§1 table + §11), TODO and CHANGELOG text.

RANGE:       origin/plan-phase-b...HEAD  (caller-supplied range)
COMMITS:     1 commit:
             - 91f8963 "spec: M11.C.16 stays composite; its rationale and mutation clause are not met"
APPLICABLE:  P19 (producer/consumer drift — summary tables vs. generated records); M11.1d (each rule in a
             criterion's set MUST run as both sign-inverted and deletion); M11.1a (a mutant that does not apply
             proves nothing — exact-once check); the caller's explicit checks: every number/verdict in the
             changed text reconciles with the mutation/ensemble/sweep records; the decision is consistent with
             M11.5's premise/composite definitions (including the 2026-10-08 redundant-set premise clause) and
             M11.1d; the coverage override is honest; the records are current; default suite exit code.
CHECKED:     91f8963's full diff (9 files); each new/changed mutant's old-string occurrence count in the source
             tree (exact-once); the §11 mutant table against `docs/phase_c_mutation_record.md` row by row; the
             spec's M11.C.16 criterion row and M11.5 class row against the record; the §1 summary row against
             the record and the sweep record; CHANGELOG and TODO enumerations against the record; the coverage
             override (JSON note vs. generated `spec_coverage.md`) and its honesty against the record; the
             three record code_hashes recomputed against the current tree; the default suite.
NOT COVERED: The 19 ensemble-marked criterion tests (deselected by pytest.ini by design); the mutation/sweep
             records are checked for internal consistency and hash-currency, not re-executed; domain/theory
             fidelity of the composite decision (the classification reasoning is checked for consistency with
             the spec's definitions, not for Bowen fidelity).

FINDINGS (ranked)

1.  LOW — P19 — docs/phase_c_completion_report.md:43. The §1 summary cell says "its required mutant, the
    learner disabled, survives, as do six other single-rule and paired mutants (§11)". The §11 table and the
    mutation record hold **six** survivors on C.16 in total — `learner-disabled`, `learner-inverted`,
    `availability-level-independent`, `availability-level-inverted`, `availability-and-learner-removed`,
    `mixing-weight-level-independent` — so the learner-disabled is survived by **five** others, not six. As
    written the cell implies seven survivors. Fix: "five other" (or "as do the five other …"). The CHANGELOG's
    "Six new and existing mutants on C.16 all survive" and the spec M11.5 row's six-item list are both correct;
    only this §1 cell over-counts.

2.  LOW — M11.1d — tools/mutation_record.py:179 (and the decision it feeds). `mixing-weight-inverted` targets
    `("M11.C.41",)` only, while `mixing-weight-level-independent` was re-targeted to also hit `M11.C.16`
    (line 95). The commit treats `M4.D.1a`'s mixing weight as one of C.16's candidate carrier rules — its
    deletion was run and reported — but M11.1d requires each rule in the set to run as **both** a sign-inverted
    and a deletion mutant. The mixing weight's inversion was never run on C.16, so "no single rule carries it"
    is asserted with the mixing-weight arm audited only half-way. The conclusion is still supported (learner
    and availability are both run inverted, and the joint `level-blind` reds it), but the audit is incomplete.
    Fix: add `"M11.C.16"` to `mixing-weight-inverted`'s criteria and rerun the record.

3.  NIT — TODO.md:42. The decision summary lists "the learner, layer availability and the mixing weight each
    survive deletion, availability survives inversion, and availability with the learner removed still passes"
    — five survivors, omitting that the learner also survives inversion (`learner-inverted`). The CHANGELOG and
    the spec M11.5 row both say "disabled or inverted". No false claim; the enumeration is just incomplete.

4.  NIT — docs/phase_c_completion_report.md:374. §10's unchanged sentence "on this evidence it is produced by
    a rule that reads level" (singular) now stands contradicted by §11's "level acting through several
    mechanisms over time". Only §10's tail was amended; the superseded singular claim was left. Low, but the
    two sections should not disagree within one document.

VERIFIED (not findings)

- Every new and re-targeted mutant's old string occurs exactly once in the source tree, so each applies exactly
  once (M11.1a): `AVAILABILITY` → `src/bowen/policy/policy.py:191` (count 1); the learner `delta` →
  `src/bowen/engine/learner.py:111` (count 1); the mixing-weight expression → `src/bowen/policy/policy.py:180`
  (count 1); the level-blind primary (steepness) → `src/bowen/engine/contact.py:45` (count 1). The
  `availability-and-learner-removed` joint mutant names both writes (policy + learner) via `also=`, per M11.1a.
- Descriptions match what each mutant does: `availability-level-independent` fixes the numerator at 50.0 (the
  value at functional_level 50); `availability-level-inverted` substitutes `100.0 - obs.functional_level` so
  availability rises as level falls; `availability-and-learner-removed` sets both the availability numerator to
  50.0 and the learner delta to 0.0; `mixing-weight-level-independent` fixes the weight at `0.5 ** exponent`
  (= level 50); `level-blind`'s `also=LEVEL_BLIND` covers every level read named in its description, and its
  availability entry is byte-identical to the new `AVAILABILITY` constant.
- The §11 mutant table matches the mutation record exactly, row for row: six survived
  (learner-disabled, learner-inverted, availability-level-independent, availability-level-inverted,
  availability-and-learner-removed, mixing-weight-level-independent) and one red (level-blind), all at 150 seeds
  except level-blind (50). The spec's M11.C.16 criterion row ("it survives") and M11.5 class row (six "leave it
  passing" + "only the joint level-blind mutant turns it red") reconcile with the same record.
- The decision is consistent with M11.5's definitions. Premise (single rule) requires the rule to "state or
  nearly state" the result, tested by M11.1d's inversion-flip: neither the learner nor the availability flips
  C.16 under inversion. Premise (redundant set, added 2026-10-08) requires "several rules each state,
  redundantly": for C.16 only availability even arguably states a narrowing, and its inversion does not flip —
  so no set of rules each states the direction, unlike C.1's nine. Composite ("no single rule states … several
  mechanisms acting together") is therefore the correct class, and the §11 reasoning tracks M11.1d's premise
  test precisely.
- The coverage override is honest. Without it, `tools/spec_coverage.py` would mark C.16 "done" (its generic
  `proved` test only needs *some* non-representation mutant red, and level-blind reds C.16). The override pins
  it to "partial" with a note naming the criterion's own mutation clause (M4.D.6 disabled MUST turn red) and
  the fact that it survives — the stronger, correct reading. The JSON note and the generated
  `docs/spec_coverage.md` row are identical, and `test_m141_spec_coverage_is_current` passes.
- Records are current. Recomputing `code_hash`: mutation record `2e69bc…` matches `code_hash(mutation_record,
  ensemble_record)`; sweep record `28fe9c…` matches `code_hash(sweep_record, mutation_record, ensemble_record)`;
  ensemble record `f5aa90…` matches its own tool. The sweep record's hash moved only because its HASHED_TOOLS
  includes `mutation_record.py` (the mutant list is a declared input); its six C.16 rows still show PASS at all
  six settings, so §1's "passes at all 6 settings" holds. Ensemble record shows C.16 PASS (150,
  −0.0657 ± 0.023, p 1.15e-16).
- Default suite: `python3 -m pytest tests/ -q` → 492 passed, 19 deselected, exit code 0.

VERDICT: the decision itself is correct and well-evidenced, the mutants apply exactly once and test what their
descriptions say, the §11 and spec rows reconcile with the record, the coverage override is honest, and the
records are current and the suite green. But the committed report carries a factual over-count (§1 "six other"
should be "five other"), and the mixing weight — one of the rules the decision lists — was deleted but never
inverted on C.16, leaving M11.1d's both-mutants requirement unmet for that rule. Both are low-severity and
easily fixed, but they are findings, not clean.

REJECTED
