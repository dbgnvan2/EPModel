# Version 1 of the agent model: a definition

**Status: APPROVED 2026-10-15; met 2026-10-18 (§2b).** The owner took the draft's proposals for questions 1, 2, 3 and 8 ("go with your
proposals on the version 1 questions") and for questions 4–7 ("I approve the revision list above", confirmed as both
the question proposals and the review form). Drafted the same day as `version_1_definition_DRAFT.md`.

The owner asked when version 1 will be finished. No document defined version 1, so no date could be given. This
document sets a definition in which every item is checked by a test, a record line or an owner sign-off, so that
progress can be measured and a date estimated from measured run times. The questions in §4 are the owner's to answer;
each answer is recorded with its date.

## 1. The definition

Version 1 is finished when all of the following hold.

| # | Condition | How it is checked |
|---|---|---|
| V1 | Every `M11.C` criterion has a final status: **pass**, proved by a mutation (recorded for a composite; for a premise or check an ad hoc run reported in the plan is enough, §4 question 5); **fail accepted**, with the owner's written reason that it is a finding about the model; or **out of scope**, with the owner's reason (`M11.C.13` and `M11.C.28`, §4 question 4). | a generated status table over the three ensemble records and the two mutation records, and a test that no criterion lacks a status (to build) |
| V2 | Every **premise** and **check** criterion passes. A failing premise means the code does not do what its rule says, which is a defect, not a finding (`M11.5`). | the same table, filtered by class |
| V3 | Every passing **composite** criterion has a mutation in a record (`docs/phase_c_mutation_record.md` or `docs/phase_d_mutation_record.md`), not only an ad hoc run. | the record tests (`test_m111d_*`) |
| V4 | The owner has reviewed spec revisions 24–30, and the spec's status line says so. | the status line; owner sign-off |
| V5 | The documents agree with the code. | the existing tests: `tests/test_doc_checklist.py`, `tests/test_spec_consistency.py`, the record-current tests |
| V6 | The suite and CI are green on every supported Python version. | `.github/workflows/tests.yml` on the last commit |
| V7 | No output presents the model as speaking about a real family (`M11.F.9`). | a review of the outputs, owner sign-off (not code-testable) |

## 2. Where the model stands against it (measured 2026-10-15)

From the committed ensemble records, all current against the code by their tests. A criterion with variants is
counted once per variant.

| Record | Pass | Fail | Not built or not run |
|---|---|---|---|
| `docs/phase_c_ensemble_record.md` | 12 | 12 | `M11.C.7`, `M11.C.13`, `M11.C.14` |
| `docs/phase_d_ensemble_record.md` | 12 | 8 | `M11.C.7`, `M11.C.9`, `M11.C.28` |
| `docs/phase_d2_ensemble_record.md` | 3 | 3 | `M11.C.37` (proved by a test instead) |

- **Failing premises (V2):** `M11.C.4`, `M11.C.29`, `M11.C.32` (Phase C record); `M11.C.14`, `M11.C.21`,
  `M11.C.23`, `M11.C.40` (Phase D record). Seven, which is the largest block of work under this definition.
- **Failing composites and mixed (V1, owner to accept or not):** `M11.C.5`, `M11.C.42`, `M11.C.44`, `M11.C.45`,
  three of four `M11.C.27` variants, two of four `M11.C.41` variants (Phase C); `M11.C.2`, `M11.C.12`, `M11.C.34`,
  `M11.C.43` (Phase D part I); `M11.C.6`, `M11.C.10`, `M11.C.2[twelve-person family]` (part II).
- **Not built:** `M11.C.7`, `M11.C.9`, `M11.C.13`, `M11.C.28`.
- **V4:** revisions 24–30 are pending owner review. *Updated 2026-10-16:* the owner reviewed all 85 entries (`docs/REVIEW_spec_revisions_24-30.md`) and the spec's status line says so; 4 entries were changed (the fourth, `M7.E.1f`'s band, decided the same day), to be applied by revision 31 (plan §4.11). V4 is met once revision 31 is built.
- **V5, V6:** met at commit `2af33db` (854 tests locally; the last CI run measured was 38763412772, at 850).
- **Not counted here:** `M11.C.44` passed at revision 29 and fails in the Phase C record now; `M11.C.2` on the twelve-
  person family passed at revision 29. A revision can turn a passing criterion red, so a count can go down.

## 2a. Where the model stands after revision 31 (measured 2026-10-16)

Every record regenerated once, after revision 31 part B and the owner's decisions on the premises (plan §4.11).

| Record | Pass | Fail | Not built or not run |
|---|---|---|---|
| `docs/phase_c_ensemble_record.md` | 12 | 12 | `M11.C.7`, `M11.C.13`, `M11.C.14` (unchanged) |
| `docs/phase_d_ensemble_record.md` | 13 | 8 | `M11.C.7`, `M11.C.9`, `M11.C.28`; `M11.C.14` is now two variants |
| `docs/phase_d2_ensemble_record.md` | 4 | 2 | `M11.C.37` (proved by a test instead) |

- **Failing premises (V2): five.** `M11.C.4`, `M11.C.29`, `M11.C.32` (Phase C, unchanged); `M11.C.21`, still failing
  after the capacity rule and the lent balance (+0.15 ± 0.19); and `M11.C.22`, **newly failing** (mild limb +0.57 ± 0.87; with the capacity rule removed the same 150 seeds
  pass). *Corrected 2026-10-17:* over 600 seeds the rule shifts the mild limb by −0.15 ± 0.53 and the limb passes
  (+1.00 ± 0.40), so the rule is not shown to cause the failure (plan §4.12). Resolved: `M11.C.14`
  passes as restated (its mutation clause is not met: the gate-removed mutant survives); `M11.C.40` passes (clause 2
  reported, proved by a unit test); `M11.C.23` is a composite now, its failure accepted as a finding (the owner,
  2026-10-16). *Decided 2026-10-17:* the capacity rule is kept and `M11.C.22`'s failure accepted for now; `M11.C.21`
  becomes a composite with its failure accepted as a finding, built 2026-10-17 with every record regenerated. Four
  failing premises remain: `M11.C.4`, `M11.C.29`, `M11.C.32` and `M11.C.22`.
- **Failing composites and mixed (V1):** as in §2, less `M11.C.2[twelve-person family]` (now passes, proved red by
  removing focus on a child's weakness and focus steering), plus `M11.C.17` (turned red by the capacity rule) and
  `M11.C.23` (accepted as a finding).
- **V3:** `M11.C.2[twelve-person family]` has a recorded red mutant (`docs/phase_d_mutation_record.md`).
- **V4: met.** Revision 31 is built, and the spec's status line says so.

## 2b. Where the model stands after revision 32 (2026-10-17)

Revision 32 (plan §4.12) was built under the owner's instruction "proceed using your own recommendations", and the
owner reviewed it one decision at a time the same day: every decision approved, `M4.C.1c`'s nodal-event spike added, and
`M11.C.17` and `M11.C.6` held for a diagnosis. The status against V1–V3 is now generated, from the
records and `docs/version_1_dispositions.md`, as **`docs/version_1_status.md`** (`tools/version_1_status.py`; a test
fails when it is stale), so it is not repeated here. Since 2026-10-19 the page and the dispositions are frozen at the
tag `version-1` (`d62cfdc`, the owner's choice; `docs/version_2_plan.md` stage 0.1): the test checks both against the
tag and runs the tag's own status tests in a copy of the tagged tree, so version 2's regenerations cannot change them.

- **V2:** `M11.C.4` and `M11.C.32` pass after their arms were corrected (`M11.C.32`'s post hoc); `M11.C.29` and
  `M11.C.22` are reclassified mixed; `M11.C.46` is built and passes; `M11.C.7` is built and mixed (its premise limb
  holds; with the nodal-event spike two of its three pairs pass); `M11.C.9` is out of version 1. After the regenerations
  of 2026-10-17 and 2026-10-18 the generated page reads **V2 met**: no failing premise or check is left.
- **V3: met** on the generated page. `M11.C.11`, `M11.C.15` and `M11.C.39` have recorded mutants (`M11.C.39`'s is red by
  construction: its baseline is the mutation its row names).
- **V1: met (2026-10-18).** The owner accepted every proposed disposition on 2026-10-17 except two held for a
  diagnosis. `M11.C.17` passes since the nodal-event spike, so it needs none. `M11.C.6`, traced and restated, fails, and
  the owner accepted it as a finding on 2026-10-18.

**Version 1 is complete (2026-10-18):** V1 to V7 hold (`docs/version_1_status.md` for V1–V3; V4 the spec's status line;
V5 and V6 the suite and CI on the last commit; V7 the owner's sign-off). By §4 question 8 the spec is frozen at version
1: later revisions start version 2.
- **V7: met.** The owner signed off on 2026-10-17 (`docs/REVIEW_version_1_outputs.md`), and "the owner's family" now
  reads "the reference family the owner specified"; the framing each output carries is checked by
  `tests/test_version_1_outputs.py`.

## 3. What version 1 does not include (decided 2026-10-15, question 7)

Retiring the frozen grid engine (`src/engine.py`); the `M15` import of real families; the fam-diag app; calibrating
any `[I]` constant to data. Each is for a later version.

## 4. Questions for the owner

All eight decided 2026-10-15.

1. **What is version 1 for?** A model you can use to explore and that reports honestly what it does and does not
   produce, or a model whose composite criteria pass? **Decided 2026-10-15: the first.** Under the second, the end depends on
   findings, not on work, and cannot be scheduled.
2. **Can a failing composite be accepted as a finding?** **Decided 2026-10-15: yes,** with a written reason per criterion, since
   a composite failing is a result about the model (`M11.5`).
3. **Must every failing premise be fixed before version 1?** **Decided 2026-10-15: yes.** A premise failing says the code does not
   do what its rule says. Or the premise is restated or withdrawn by the owner.
4. **Are the four not-built criteria in or out?** `M11.C.7` (the coach's position), `M11.C.9` (sibling position),
   `M11.C.13` (incidents in a community), `M11.C.28` (the sink allocation feeds no behaviour).
   **Decided 2026-10-15: `M11.C.13` and `M11.C.28` are out** (the first's mechanism does not exist, the second's affects
   no behaviour); **`M11.C.7` and `M11.C.9` are in** (each needs one gate that is partly built).
5. **Must every pass have a recorded mutation (V3), or are ad hoc runs reported in the plan enough?** Moving the ad
   hoc ones into the record costs one run per mutant. **Decided 2026-10-15: a passing composite needs a recorded
   mutation; a passing premise or check may rest on an ad hoc run reported in the plan,** since those passes confirm
   the code rather than report a finding.
6. **Is your review of revisions 24–30 a condition (V4)?** **Decided 2026-10-15: yes;** otherwise version 1 freezes
   rules the owner has not approved. The review form is `docs/REVIEW_spec_revisions_24-30.md`.
7. **Do any of the items in §3 belong in version 1?** **Decided 2026-10-15: no.**
8. **Is a spec revision that reopens a passed criterion allowed after version 1 is declared, or does version 1 freeze
   the spec?** **Decided 2026-10-15: freeze the spec at version 1,** as the grid spec v1.2 was frozen; later revisions start
   version 2.
