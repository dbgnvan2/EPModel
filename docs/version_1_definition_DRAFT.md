# Version 1 of the agent model: a definition (DRAFT)

**Status: DRAFT, 2026-10-15. Questions 1, 2, 3 and 8 decided by the owner on 2026-10-15 ("go with your proposals on
the version 1 questions"): each takes the draft's proposal. Questions 4–7 carried no proposal and are open, so the
definition is not yet approved.**

The owner asked when version 1 will be finished. No document defined version 1, so no date could be given. This
draft proposes a definition in which every item is checked by a test, a record line or an owner sign-off, so that
progress can be measured and a date estimated from measured run times. The questions in §4 are the owner's to answer;
the draft's own answers are marked as proposals.

## 1. Proposed definition

Version 1 is finished when all of the following hold.

| # | Condition | How it is checked |
|---|---|---|
| V1 | Every `M11.C` criterion has a final status: **pass**, proved by a recorded mutation; **fail accepted**, with the owner's written reason that it is a finding about the model; or **out of scope**, with the owner's reason. | a generated status table over the three ensemble records and the two mutation records, and a test that no criterion lacks a status (to build) |
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
- **V4:** revisions 24–30 are pending owner review.
- **V5, V6:** met at commit `2af33db` (854 tests locally; the last CI run measured was 38763412772, at 850).
- **Not counted here:** `M11.C.44` passed at revision 29 and fails in the Phase C record now; `M11.C.2` on the twelve-
  person family passed at revision 29. A revision can turn a passing criterion red, so a count can go down.

## 3. What this does not include, unless the owner adds it

Retiring the frozen grid engine (`src/engine.py`); the `M15` import of real families; the fam-diag app; calibrating
any `[I]` constant to data. Each is a candidate for version 1 or for later.

## 4. Questions for the owner

Open: 4, 5, 6, 7.

1. **What is version 1 for?** A model you can use to explore and that reports honestly what it does and does not
   produce, or a model whose composite criteria pass? **Decided 2026-10-15: the first.** Under the second, the end depends on
   findings, not on work, and cannot be scheduled.
2. **Can a failing composite be accepted as a finding?** **Decided 2026-10-15: yes,** with a written reason per criterion, since
   a composite failing is a result about the model (`M11.5`).
3. **Must every failing premise be fixed before version 1?** **Decided 2026-10-15: yes.** A premise failing says the code does not
   do what its rule says. Or the premise is restated or withdrawn by the owner.
4. **Are the four not-built criteria in or out?** `M11.C.7` (the coach's position), `M11.C.9` (sibling position),
   `M11.C.13` (incidents in a community), `M11.C.28` (the sink allocation feeds no behaviour).
5. **Must every pass have a recorded mutation (V3), or are ad hoc runs reported in the plan enough?** Moving the ad
   hoc ones into the record costs one run per mutant.
6. **Is your review of revisions 24–30 a condition (V4)?**
7. **Do any of the items in §3 belong in version 1?**
8. **Is a spec revision that reopens a passed criterion allowed after version 1 is declared, or does version 1 freeze
   the spec?** **Decided 2026-10-15: freeze the spec at version 1,** as the grid spec v1.2 was frozen; later revisions start
   version 2.
