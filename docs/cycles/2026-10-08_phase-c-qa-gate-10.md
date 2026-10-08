# Phase C QA gate 10 — 2026-10-08

learning-qa failure-pattern sweep over `origin/plan-phase-b..HEAD` (one docs-only commit, `58f8520`).

RANGE:       origin/plan-phase-b...HEAD  (caller-supplied range, step-1 rule 1)
COMMITS:     1 commit: 58f8520 — "spec: M11.C.1 and M11.C.38 stay premise; their rule column lists the nine level-reading rules"
APPLICABLE:  P6 (claim vs. artifact), P19 (producer/consumer drift, incl. its source-text corollary); plus the
             caller's explicit consistency checks (M11.5 definition, M11.1d, rule-ID existence, mutant claims,
             report/TODO/CHANGELOG agreement, no code/config/record change, test suite).
CHECKED:     P6, P19; rule-ID existence and level-read mapping (all eight IDs in the corrected rows vs.
             tools/mutation_record.py's LEVEL_BLIND); mutant-claim fidelity vs. docs/phase_c_mutation_record.md;
             M11.5/M11.1d definitional consistency; report/TODO/CHANGELOG agreement; diff scope; default suite.
NOT COVERED: This is a failure-pattern + document-consistency sweep only. Not assessed: domain/theory fidelity of
             the "written-in vs. emergent" reasoning against the corpus; whether the nine-rule set is truly minimal
             for onset (M11.1d's "minimal set" clause); the cold-review blind spot (this sweep is the only pass).
             The 19 ensemble-marked tests are deselected by design and were not run.

FINDINGS (ranked)

1. LOW · P19 · CHANGELOG.md:28-29 · stale ".45 a premise" contradicts the spec, TODO and report
   The "M11.5 corrected" bullet was only half-updated: "`M11.C.1` and `.38` first proposed as joint premises
   (decided later the same day: they stay premise, with the nine rules listed), `.45` a premise." The ".45 a
   premise" clause was left untouched, but the spec's M11.5 table classifies M11.C.45 as **Composite** with its
   premise reclassification "withdrawn until it passes again" (bowen_agent_model_spec_v2.md:1309). TODO.md:29-30
   and the report §2 (phase_c_completion_report.md:93-96) both record the withdrawal. The CHANGELOG still reads as
   if C.45 is currently premise.
   Fix: ".45's premise reclassification withdrawn (stays composite)."

2. LOW · P6 · docs/bowen_agent_model_spec_v2.md:1265 · "leaves the arms' difference at exactly zero" is not in the
   record it cites
   The M11.C.1 note asserts the joint level-blind mutant "leaves the arms' difference at exactly zero", but
   docs/phase_c_mutation_record.md records only the verdict ("red"/FAIL), not the magnitude. mutation_record.py's
   child() prints mean_difference to stdout, but render() persists only outcome/seeds/result, so the magnitude is
   in no committed record. Plausibly true (level-blind arms are behaviourally identical under one seed), but the
   project's own rule is "every number comes from a committed generated record".
   Fix: persist the difference in the record, or drop the "exactly zero" clause.

3. LOW · P19 · docs/bowen_agent_model_spec_v2.md:1302 · "each single-rule mutant reds some adjacent pairs and not
   others" overstates what the record shows
   For M11.C.38, four of the six single-rule mutants red ZERO of the three pairs (steepness-level-independent,
   threshold-level-independent, standing-load-level-independent, standing-load-level-inverted all survive all
   three); one reds one pair (threshold-level-inverted); one reds two (steepness-inverted). Only the joint
   level-blind mutant reds all three. "Each single-rule mutant reds some adjacent pairs and not others" is false
   for the four that red zero pairs. The accurate form is "no single-rule mutant reds all three pairs". This
   sentence is retained, not introduced, by this commit (only "Corrected"→"Rule column corrected" and the class-
   kept clause changed), but it sits in the final corrected note and the sweep's scope includes mutant claims.
   Fix: "no single-rule mutant reds all three pairs; the joint level-blind mutant reds all three."

4. LOW · spec-internal consistency · docs/bowen_agent_model_spec_v2.md:1261 vs. 1265/1302 · M11.5's premise
   definition no longer covers two of its own rows
   M11.5 defines premise as "a criterion that a single rule produces". C.1 and C.38 are now classified Premise but
   are produced by nine rules jointly, none individually necessary — the structure M11.5 reserves for composite
   ("no single rule states … several mechanisms acting together"). The decision's rationale is sound and documented
   (written-in ≠ emergent; M11.1d's single-rule flip is "a sufficient sign of a premise, not a necessary one"), but
   it lives only in the C.1 row note, not in the definition paragraph. The decision states "M11.5 has no 'joint
   premise' class, and none is needed", yet the definition still says "single rule produces", so a reader of the
   definition would not predict C.1/C.38 are premise.
   Fix: amend the M11.5 definition paragraph to admit a jointly-stated, non-emergent premise (written in
   redundantly by several rules, none necessary), so the definition and the table agree.

VERIFIED (not findings)

- All eight rule IDs named in the corrected rows exist and are exactly the rules the level-blind mutant makes
  level-independent: M4.C.1a steepness+band (spec:658), M1.A.6 threshold (220), M4.A.5 self term (633),
  M4.D.1a mixing weight (707), M4.D.3a layer availability (716), M1.A.9 initial outside-ness (230),
  M1.C.3a routing capacity (364), M5.D.3 hold capacity (824). The nine code replacements in LEVEL_BLIND
  (8 tuple entries + the steepness entry in the Mutant) map 1:1 to these. "Hold capacity" is the code's own term
  for the abort/hold capacity (iposition.py:129-133), and hold_gain is tied to M5.D.3 in config/bowen/constants.md:126.
- M11.C.1 note "steepness, threshold and self term each survive deletion and inversion alone" matches the record:
  six single-rule mutants on C.1, all survived.
- "Only the joint level-blind mutant turns it red" matches: level-blind → red, all singles → survived.
- "The joint level-blind mutant reds all three" (C.38) matches: all three pairs red.
- Decision is consistent with M11.1d's "programmed premise rather than derived result" branch; M11.1d's single-rule
  flip trigger does not fire for C.1/.38 (no single rule flips them), which the report states explicitly.
- No code, config, or record changed: the diff touches only CHANGELOG.md, TODO.md,
  docs/bowen_agent_model_spec_v2.md, docs/phase_c_completion_report.md.
- Default suite: `python3 -m pytest tests/ -q` → 490 passed, 19 deselected, exit code 0.

VERDICT: clean against P1-P37, of which 2 were applicable, with 4 low-severity findings and no medium or higher.

APPROVED
