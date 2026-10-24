# Milestone review: version 2 stage 2 (mini2, 2026-10-24)

The second milestone critical review of EPModel, run as `docs/milestone_review_brief.md` describes. It reviews the
whole model against the brief's four lenses, theory fidelity first, with the changes since the stage 1 review in
focus. The reviewing session edited nothing except this report and fixed nothing.

## Header

- **Commit reviewed:** `d51b2508a4751791e66112d4f79a9835cc410c19` (`origin/main`, "T-125 closed: the owner approved
  all fifty-three overnight decisions"), in a worktree on branch `review/2026-10-24-v2-stage-2`.
- **Baseline:** `9cf85529b4186fa7516996461f29de35601b8a74`, the commit the stage 1 review covered
  (`docs/cycles/2026-10-21_milestone-review_v2-stage-1.md`). 53 commits.
  - The brief's §3 step 5 filter lists 197 files.
  - The extended filter below leaves 151 files (+11,892 / −1,515), which were split into the four lens slices:
    `src/bowen/` 58, `tests/` 34, `tools/` 18, `config/` 9, `docs/` 30.
- **Machine:** Mac mini2 (`mini2.local`, Darwin 24.6.0).
- **Suite:** `1194 passed, 51 deselected, 1 xfailed in 244.90s`, exit 0, on Python 3.12.3. CLAUDE.md says "all 1195
  tests pass"; the 1195th is the strict expected failure `test_m5b3_third_party_cost_is_the_lost_outlet` (see MR-20).
- **CI for the commit:** run `40114400433` (push to `main`, head `d51b250`), `completed success`: `test (3.11)`
  success, `test (3.12)` success, `test (3.13)` success.
- **Tokens used:**
  - Subagents: 426,430, taken from each result's usage total (L1 46,109; L2 49,660; L3 178,760; L4 51,568;
    challenge 100,333).
  - The orchestrating session: not measured as a total. The app reports only the current context window, which held
    220,172 tokens at consolidation after one compaction. That is not the session's cumulative use.
- **Wall time:** 17.5 minutes from the start of §3 step 3 (10:16:53 local) to the challenge pass's return (10:34:23),
  plus the time to write this file.
  - The worktree, the virtualenv and the cost statement came before the clock started.
  - The four lens workers ran in parallel: L1 6.8, L2 9.1, L3 6.0 and L4 6.0 minutes. The challenge pass took 3.1
    minutes.
- **Findings:** 0 blocker, 1 major (confirmed in challenge), 27 minor:
  - 5 confirmed in challenge and downgraded there from major;
  - 22 not challenged, of which the orchestrator re-checked 6 against the files (marked †).

**Deviations from the brief.**

1. **Machine and virtualenv.** The prompt named `~/ProjectsMini1/EPModel/.venv`, which does not exist on this
   machine. The project's own `.venv` is Python 3.11.14 with pytest 9.0.2, which breaks `requirements.txt`'s
   `pytest>=8.4,<9`. The suite therefore ran in a fresh Python 3.12.3 virtualenv built from `requirements.txt` in the
   session's scratchpad (numpy 2.5.4, pytest 8.4.2), outside the repo.
2. **Worktree.** The session started on `main` in the working checkout. Per §3 step 2, it created the review
   worktree from `origin/main` rather than commit there.
3. **Exclusions beyond the brief's filter:**
   - generated pages: `docs/spec_rules/`, `docs/spec_index.jsonl`, `docs/theory/ledger_index.jsonl`,
     `docs/version_2_status.md`, `docs/version_2_revision_36_changes.md`, `docs/TODO_archive.md`;
   - dated logs: `docs/CHANGELOG_archive.md`, `docs/todo/`;
   - `docs/cycles/`, which holds the previous report.

   Workers read `TODO.md`'s index line and then only the record it names, and searched the generated pages for a
   claim being checked.
4. **Spec read.** The orchestrator read the spec end to end (lines 1–2131) before fanning out, finishing while the
   suite ran. `docs/spec_revision_log.md` was read only at its headings and its revision 36–37 entries.

## Coverage matrix

"Full" = read in full; "diff" = every changed hunk read with its enclosing section or function; "sampled" = named
ranges read; "searched" = grepped for the claim being checked; "—" = not in that lens.

| Area | L1 theory fidelity | L2 spec against code | L3 documents | L4 engineering drift |
|---|---|---|---|---|
| Spec `bowen_agent_model_spec_v2.md` (2,131 lines) | M10–M11.E (1165–1314) full, 1361, 1366–1421; M11.F–M11.G headings only | 410–439, 516–523, 822, 850, 983–994, 1060–1071, 1278–1299, 1364–1421, 1730–1737, 2024–2093 | front matter and the rev36/rev37 tags | M16.B (1938–1961) full |
| The orchestrator's read of the spec | lines 1–2131 full | — | — | — |
| `src/bowen/` engine | diff, plus `cutoff_stock.py` in full; literal check of every new hunk | diff of `moves`, `symptoms`, `consolidate`, `rejection`, `tick`, `visibility`, `event_effects`, `cutoff_stock` (full), `event_sizes` (full); `life_course` searched only | searched (I/O calls) | every changed hunk; `cutoff_stock`, `event_sizes` full |
| `src/bowen/` ensemble, readouts, policy, io, render, scenario | `negative_runs.py` full, `criteria_d2.py` 100–239 | `negative_runs.py` full, `criteria_d2.py` 105–174, 460–481, `runner.py` 120–231; policy, render, scenario, io, `readouts/evaluation`, `patterns` not read | searched | diff of each |
| `config/bowen/` | `constants.md`, `event_sizes.md`, `contact_classes.md` full; diff of `criteria.md`, `event_kinds.md`, `constants_changes.md`, `family_phase_d.md`; 13 other files not read | `constants.md` 172–174, 252–259; `event_sizes.md` full | searched | — |
| `config/sweep/sweep.md` | — | searched (named ratio) | searched (row count) | — |
| `docs/model_explainer.md` (2,184 lines) | 1–12, 360–459, 590–649, 690–701, 795–819, 935–1009, 1318–1385, 1512–1523, 1980–2034, 2104–2115 (every changed range) | — | status line and date | — |
| `docs/theory/` | `_SOURCE_QUALITY.md` full; `_LEDGER.md` searched | — | `_STATUS.md` 1–109, 300–324, 520–549 | — |
| Reading list, checklist, README, CLAUDE.md | reading list and README searched | — | full | — |
| `docs/spec_coverage.md` + overrides | — | rows cited in findings; changed override lines (277, 322, 623, 700, 922, 1002, 1006, 1133) | searched | — |
| Generated records (`docs/*_record.md`) | searched (class labels, framing, M11.C.48) | Phase C and D mutation records searched for M11.C.44, M11.C.48; `version_2_sweep_record.md` 1–30, 480–507 | `version_2_sweep_record.md` 32–65 against prose | — |
| `docs/review/` | `group_g_readout.md` full; traces, chart and famdiag JSON not read | — | — | — |
| `tests/` | — | `test_cutoff_stock.py`, `test_tie_repair.py` full; `test_event_sizes.py` 56–209; `test_phase_d2_criteria.py` 225–273; `test_moves.py` M5.B.3 tests | `test_doc_status.py` 1–115; `test_spec_consistency.py` 349–389; `doc_claims.json` full | diff, with `test_cutoff_stock`, `test_event_sizes`, `test_tie_repair`, `test_negative_runs`, `test_engine_purity` read; documentation-tooling tests not read (listed below) |
| `tools/`, CI, `pytest.ini`, `requirements.txt` | — | `version_2_sweep.py` 110–179 | — | `version_2_sweep.py`, `two_mac_sweep.py`, `merge_cache.py`, `group_g_readout.py` full; `record_cache.py` 40–180; `tests.yml`, `pytest.ini`, `requirements.txt` full |
| `TODO.md` and the records it names | — | `TODO.md` full; T-004's record, the revision 36 regeneration and overnight-decisions records full | searched | searched (known items not re-reported) |
| DECISIONS and PROPOSAL files | `DECISIONS FOR APPROVAL.md` A1 read | `PROPOSAL — CUTOFF ANXIETY…` 60–219 | status lines of every changed one | — |

Not reviewed, by any lens:

- these documentation-tooling tests: `test_spec_index`, `test_spec_rules_view`, `test_todo_index`,
  `test_version_2_status`, `test_record_changes`, `test_famdiag_export`, `test_asr_quotations`;
- the matching tools beyond samples: `spec_index.py`, `spec_rules_view.py`, `todo_index.py`, `version_2_status.py`,
  `record_changes.py`, `sweep_config.py`;
- `docs/agent_model_proposal.html`'s hunks;
- the SP-1 proposal's body;
- `docs/version_2_plan.md`.

## Findings

### Major findings

#### MR-1 — `M17.E.1`'s named-ratio clause is not built, is marked *done*, and nothing tracks it

- **Found by:** L2 · **Where:** `docs/spec_coverage.md:601`; spec `:2087` (`M17.E.1`); `docs/implementation_plan_version_2.md:628`, `:701`, `:713`; `config/sweep/sweep.md`
- **Severity:** major · **Confidence:** high · **Verified:** confirmed in challenge. The challenge found the gap wider than reported: no TODO item tracks it.
- **What is wrong:**
  - Stage 2's sweep ran in full, but without the clause that the rate of tie hardening, relative to the rate anxiety
    moves, MUST be swept as a named ratio.
  - The plan deferred the ratio "with the full sweep's ranges". The full sweep then ran without it, and the coverage
    page marks the requirement *done* on 36 tests of sweep mechanics, none of which concerns a ratio.
  - The reading list says stage 2's full sweep is complete.
- **Quoted:**
  - spec:2087 — "The rate at which ties harden or attenuate, relative to the rate at which anxiety moves, **MUST** be swept as a named ratio."
  - plan:701 — "`M17.E.1`'s named ratio | not done in this stage's pilot (D10); declared with the full sweep's ranges"
- **How checked:**
  - L2 read the spec row, the coverage row and the plan.
  - L2 and the challenge pass both searched `config/sweep/sweep.md` and found no ratio row; its only uses of "ratio",
    at 180 and 335, are unrelated.
  - The challenge pass searched `TODO.md` and found no item for it.
- **Fix:** mark `M17.E.1` *partial* in `tools/spec_coverage_overrides.json` with a note naming the ratio clause. Open
  a TODO item to declare and sweep the ratio before the next end-of-stage full sweep.

### Major findings downgraded in challenge

Each was confirmed. The challenge pass rated it minor because the rule is stated correctly elsewhere and nothing a run
computes is wrong. They are listed first among the minor findings because a reader could be misled.

#### MR-2 — Two passages still say `REDUCE_CUTOFF` hands the cutoff's held anxiety to third parties

- **Found by:** L1 · **Where:** `src/bowen/engine/moves.py:63-68` (module docstring), `docs/model_explainer.md:2109-2111` (§19)
- **Severity:** minor (downgraded from major) · **Confidence:** high · **Verified:** confirmed in challenge
- **What is wrong:** revision 36 (D7) removed the transfer. The function, its docstring, the explainer's §11.5 and
  `M5.B.3` all say no stock moves, and the third-party cost is a strict expected failure. The module docstring and
  §19 still describe the old transfer, citing `L22.6`, as a built mechanism.
- **Quoted:**
  - explainer:2109–2110 — "**`REDUCE_CUTOFF`** reopens a severed tie, and the anxiety the cutoff held goes to the third parties around it (`L22.6`)."
  - moves.py:412–414 — "no stock moves to a third party (spec revision 36, D7)"
- **How checked:**
  - L1 read both passages against `moves.py:411-416`, explainer `:1351-1354` and `docs/spec_rules/M05.md:21`.
  - The challenge pass read `moves.py:423-428`: the body only lowers rejection. `release_bound_distance`
    (`moves.py:146`) is called only from reconciliation (`event_effects.py:102`).
- **Fix:** restate both passages in the D7 form: rejection lowered, no stock moves, and the third-party cost (`L22.6`,
  `M5.B.3`) not yet shown. `docs/REVIEW_version_1_guide.md:101` describes the same transfer; it is a historical guide
  (see Questions).

#### MR-3 — The ensemble and sweep records show `M11.C.48` as an unqualified PASS

- **Found by:** L1, L2 (independently) · **Where:** `docs/phase_d2_ensemble_record.md:20-21`; `docs/version_2_sweep_record.md:64-65`, `:483`, `:496`
- **Severity:** minor (downgraded from major) · **Confidence:** high · **Verified:** partly confirmed in challenge
- **What is wrong:** the two generated records give `M11.C.48` PASS ("8 of 8 held" in the sweep) with no note of
  either known problem:
  - T-123: the readout's baseline counts a seed's unmatched cut-off ties as 0. The spec's `M11.C.48` row says the
    recorded pass "is not evidence".
  - T-119: the criterion's named mutant `cutoff-blind-to-learned-value` survived.
- **Quoted:** phase_d2_ensemble_record:20 — "`M11.C.48` | composite | **PASS** | 100 | `negative_run` +0.43 ± 0.15, p 5.23e-07, holds"
- **How checked:**
  - L1 grepped `docs/*_record.md` for "T-123", "not evidence" and "T-119" and found nothing.
  - `docs/readouts_phase_c.md:73-77` gives −0.28 ± 0.15 and −0.21 ± 0.18 with the defect removed.
  - `negative_runs.py:156` still has `if standing else 0.0`.
  - The challenge pass found the caveat in seven documents a reader is likely to open: spec `:1368`, explainer
    `:1377`, `readouts_phase_c.md:75`, `REVIEWER_READING_LIST.md:152`, `_STATUS.md:18`,
    `implementation_plan_version_2.md:551`, and T-119/T-123.
  - `version_2_status.md:100-101` marks the pass not final.
  - The part of the original finding about `docs/review/group_g_readout.md` is refuted (see "Refuted in challenge").
- **Fix:** have the record tools print `M11.C.48`'s verdict with "not evidence (T-123); named mutant survived
  (T-119)" until the readout is fixed and the records regenerated.

#### MR-4 — `M5.B.3` is *done* while its third-party-cost clause is a strict expected failure, and the coverage tool cannot see an xfail

- **Found by:** L2 · **Where:** `docs/spec_coverage.md:268`; `tools/spec_coverage.py:288-289`; `tests/bowen/test_cutoff_stock.py:430`; spec `:850`
- **Severity:** minor (downgraded from major) · **Confidence:** high · **Verified:** partly confirmed in challenge
- **What is wrong:** coverage rule 2 marks an ID *done* when any test named for it passes, and a strict xfail
  "passes". The status hides the clause the spec itself says is "Not yet shown". No override exists for `M5.B.3` (the
  only `M5.B` override is `M5.B.3a`, overrides `:800`).
- **Quoted:**
  - spec_coverage.py:288–289 — `elif named: rows.append({"id": spec_id, "status": "done", ...})`
  - test_cutoff_stock.py:430 — `@pytest.mark.xfail(strict=True, reason="A6 not shown: …`
- **How checked:**
  - L2 grepped `tools/` for "xfail" and found nothing.
  - The challenge pass refuted "the only test named for it": five passing tests cover `M5.B.3`'s other clauses
    (`test_moves.py:250`, `:264`, `test_policy.py:350`, `test_rejection.py:147`, `:300`).
  - This is the only xfail in the suite, so it is the only ID the blind spot affects today.
- **Fix:** add an override marking `M5.B.3` *partial*, naming the third-party-cost clause. Make rule 2 skip tests
  marked xfail, with a test.

#### MR-5 — The reading list's status line and its checklist label stop at 2026-10-21/22

- **Found by:** L3 · **Where:** `docs/REVIEWER_READING_LIST.md:3`, `:9`; `docs/SPEC_REVIEW_CHECKLIST.md:13-15`
- **Severity:** minor (downgraded from major) · **Confidence:** high · **Verified:** confirmed in challenge
- **What is wrong:** the status line lists only the revisions of 2026-10-21 and 2026-10-22 as not yet reviewed and
  says "last checked on 2026-10-21". The body carries text from 2026-10-23 and 2026-10-24 (lines 26, 148–149, 152,
  181) that the status line does not record. The checklist's label says "revised 2026-10-21" only.
- **Quoted:** reading list:3 — "**Status: approved by the owner 2026-10-20 (drafted 2026-10-19); revised 2026-10-21 and 2026-10-22, not yet reviewed by the owner:"
- **How checked:** L3 read the list in full and the diff hunk that added the later rows. The challenge pass confirmed
  that `tests/test_doc_status.py:42-49` classifies only the first status word, and that no test names the reading
  list or "last checked".
- **Fix:** add the 2026-10-23 and 2026-10-24 revisions to the status line, update "last checked", and make the
  checklist label match. For the class fix, see systemic pattern SP-A.

#### MR-6 — The checklist says the model-ergonomics plan's Parts D and E are not started; both were built 2026-10-22

- **Found by:** L3 · **Where:** `docs/SPEC_REVIEW_CHECKLIST.md:239-242`; `docs/plan_model_ergonomics.md:3`
- **Severity:** minor (downgraded from major) · **Confidence:** high · **Verified:** confirmed in challenge
- **What is wrong:** the label was added in this range and not updated when Parts D and E were built the same day.
  The spec's history has been moved out, and the docstrings thinned.
- **Quoted:**
  - checklist:239 — "**Parts A, B and C built 2026-10-22; D and E wait on the owner's answers to its §8:**"
  - checklist:242 — "(not started) thinner docstrings and the spec's history moved out of it."
- **How checked:**
  - L3 read the plan's status line ("Parts A to E built 2026-10-22 … the owner's §8 answers of 2026-10-22 applied").
  - `_STATUS.md:26-35` confirms the work was done.
  - The challenge pass confirmed that spec `:2127` points to `spec_revision_log.md`.
- **Fix:** rewrite the label as "Parts A to E built 2026-10-22, the owner's §8 answers applied", and drop "(not
  started)".

### Minor findings

Not challenged, per §5. Those marked † the orchestrator re-read against the files.

#### MR-7 — † The cutoff stock decays in contact weeks too; `M1.D.2b` says only weeks with no contact

- **Found by:** L2 · **Where:** spec `:418` (`M1.D.2b`), `:1734` (register); `src/bowen/engine/cutoff_stock.py:208-224`; `tick.py:216`
- **Severity:** minor · **Confidence:** high · **Verified:** re-read by the orchestrator
- **What is wrong:** `decay_cutoff_stock` takes no contact batch and decays every `CUT_OFF` tie every week, after
  step 4. A stock renewed to `P` by a negative contact therefore ends that week at `P×f`. The register row and the G1
  proposal (`:92`, "Each week a tie is cut off") agree with the code, not with the requirement. This disagreement is
  not on T-004's list.
- **Quoted:** spec:418 — "The stock **MUST** decay geometrically, in the weeks with no contact across the tie"
- **How checked:** the orchestrator read `cutoff_stock.py:208-224`. The loop's only conditions are `CUT_OFF` and a
  set peak.
- **Fix:** the owner decides which stands. Either the spec says "each week the tie is cut off", or the code skips
  decay in contact weeks. Add the item to T-004 until then.

#### MR-8 — The reference level `R` is reset on entry to `DISTANT`; `M1.D.2c` says it is only ever raised

- **Found by:** L2 · **Where:** spec `:420` (`M1.D.2c`); `cutoff_stock.py:82-83`, `:171-172`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** the code sets `R` when a cut tie reopens as `DISTANT` and when a declared `DISTANT` tie is
  entered, so `R` can fall (`test_m1d2b_a_distant_tie_reopens_distant` takes it from 30 to 6). The G1 proposal §3.6
  says this; the spec does not.
- **Quoted:** spec:420 — "its stock when the run hardened it, raised whenever binding carries the stock above it"
- **How checked:** L2 read both code sites and the test.
- **Fix:** state the reset in `M1.D.2c`, or change the code. The owner decides; add the item to T-004 until then.

#### MR-9 — † Tests named in the spec and in `Tests:` lines that do not exist

- **Found by:** L2, L4 · **Where:** spec `:1368` (the `M11.C.48` row); `src/bowen/engine/rejection.py:62`; `tools/group_g_readout.py:13`
- **Severity:** minor · **Confidence:** high · **Verified:** re-read by the orchestrator
- **What is wrong:** three references point to tests that do not exist:
  - The `M11.C.48` row names `test_m11c48_negative_runs_lengthen_before_a_cutoff` and
    `test_m11c48_phase_d_negative_runs_lengthen_before_a_cutoff`. The real tests are
    `test_m11c48_cutoffs_follow_negative_runs` and `…_in_the_twelve_person_family` (`test_phase_d2_criteria.py:244`,
    `:249`).
  - `rejection.py:62` names `test_rejection.py::test_m1b3c_faded_cutoff_waits_for_a_positive_contact`. The real test
    is `test_cutoff_stock.py::test_m1d2b_faded_cutoff_waits_for_a_positive_contact` (`:212`).
  - `group_g_readout.py:13` names `tests/test_group_g_readout.py`, which does not exist. This is the tool whose
    output `config/bowen/constants.md:254` cites for `cutoff_stock_cap` = 1583, and it has no test at all.
- **Quoted:** group_g_readout.py:13 — "Tests:   tests/test_group_g_readout.py::test_g1_readout_section_is_rendered_from_measurements"
- **How checked:** the orchestrator grepped `src/`, `tests/` and `tools/` for each name. The spec's names occur only
  in the spec, and `tests/test_group_g*` matches nothing.
- **Fix:** correct the three references. Add a test for `group_g_readout.py`'s rendering. Add a gate test that
  resolves every test a spec row or `Tests:` line names (systemic pattern SP-B).

#### MR-10 — T-004 and the `M1.B.3` override still say no code writes `OPEN_CONFLICT`

- **Found by:** L2 · **Where:** `tools/spec_coverage_overrides.json:277`; `docs/todo/2026-08-22_blocking_the_next_phase.md:72`; `src/bowen/engine/consolidate.py:73-74`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** since revision 36, `consolidate.py` sets `TieState.OPEN_CONFLICT` after a run of three
  `CONFLICT`s, and `test_tie_repair.py:75-82` tests it. The override note, and the `M1.B.3` status built on it, are
  stale.
- **Quoted:** override note — "open conflict is declared (`TieState.OPEN_CONFLICT`) but no code assigns it, so the fourth state is never distinguished"
- **How checked:** L2 read `consolidate.py:1-85` and the test.
- **Fix:** rewrite the override note, re-derive `M1.B.3`'s status, and remove the item from T-004.

#### MR-11 — The register's "Move physics" row leaves out `CONFLICT`

- **Found by:** L2 · **Where:** spec `:1731`; `src/bowen/engine/moves.py:472-473`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** `CONFLICT` binds stock (`bind_conflict`, `M1.D.2c`), but the row's trigger list does not
  include it. Revision 36 edited the rows below (1732–1735) and not this one.
- **Quoted:** spec:1731 — "a `DISTANCE`, `TRIANGLE`, `OVERFUNCTION`, `UNDERFUNCTION` or `REDUCE_CUTOFF` delivered"
- **How checked:** L2 read `moves.py:472-473` and the slice's changed spec ranges.
- **Fix:** add `CONFLICT` (`M1.D.2c`) to the row.

#### MR-12 — The sweep's detection-limit refusal fires after the rerun, not before it

- **Found by:** L2 · **Where:** `tools/version_2_sweep.py:165-170`; spec `:2030` (`M17.A.1`); `tests/test_version_2_sweep.py:162-172`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** `M17.A.1` requires the sweep to refuse a criterion whose own seed count is below the declared
  count "rather than run the same seeds twice". The tool runs the rerun and raises `SystemExit` afterwards, and only
  when some sample was near the limit. The test expects only the `SystemExit`.
- **Quoted:** spec:2030 — "the sweep **MUST** refuse it rather than run the same seeds twice"
- **How checked:** L2 read `version_2_sweep.py:110-179`.
- **Fix:** compare the criterion's `seeds` with the declared count before any run. Add a test that asserts no run
  happens.

#### MR-13 — Six passages still say a cutoff ends when rejection reaches zero

- **Found by:** L1 · **Where:** `src/bowen/engine/objects.py:247-248`, `:277`; `rejection.py:10`; `invariants.py:26`; `docs/model_explainer.md:622`, `:628-629`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** `M1.B.3c` as amended at revision 36 keeps the tie `CUT_OFF` after rejection fades, until a
  positive contact, and the code does so. These passages describe the old rule. The explainer contradicts itself at
  `:642-644`.
- **Quoted:** explainer:628–629 — "At zero, the tie is ordinary again"
- **How checked:** L1 read `docs/spec_rules/M01.md:259`, `invariants.py:224` and `rejection.py:16-18`.
- **Fix:** restate all six as "the tie leaves `CUT_OFF` only on a positive contact with no rejection held
  (`M1.B.3c`)".

#### MR-14 — The event sizes are cited to Holmes & Rahe (1967); the config says they came from Wikipedia's copy, unchecked

- **Found by:** L1 · **Where:** `docs/model_explainer.md:1360`; spec `:522` (`M1.F.6a`); `config/bowen/event_sizes.md:7-9`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** the explainer and spec cite the 1967 paper as the source of the life-change-unit values. The
  config file says they were read from Wikipedia and not checked against it. Every value is graded `[I]`, so no grade
  is wrong, but the provenance is overstated.
- **Quoted:**
  - explainer:1360 — "Each kind has a life-change-unit value from Holmes & Rahe (1967)"
  - event_sizes.md — "read from the scale's published table as it appears on Wikipedia, unchecked against Holmes and Rahe (1967)"
- **How checked:** L1 read all three.
- **Fix:** check the values against the paper and say so, or cite them in the explainer and spec as tabulated and
  unchecked.

#### MR-15 — The reading list's body: four statements that no longer hold

- **Found by:** L3 · **Where:** `docs/REVIEWER_READING_LIST.md:26`, `:170`, `:181-182`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:**
  - Line 26 says version 2's "later stages are planned, not built", but stage 2 is measured and group G is built.
  - Line 170 says the spec, explainer and ledger "still quote" the ASR sources, which contradicts the list's own §8
    at `:206`.
  - Lines 181–182 say judgments "wait on the owner's review", but T-125 closed 2026-10-24.
- **Quoted:** reading list:170 — "the spec, the explainer and the ledger still quote them in places (§8)."
- **How checked:** L3 read the T-125 record (`docs/todo/2026-10-23_overnight_decisions.md:9-12`), the sweep record
  and plan `:1090`.
- **Fix:** restate the three lines to the current state.

#### MR-16 — † The revision-log range is quoted as "4 to 35" and "4 to 36"; the log holds 4 to 37 and still calls 36 and 37 pending

- **Found by:** L3, orchestrator · **Where:** spec `:2127`; `docs/SPEC_REVIEW_CHECKLIST.md:31`; `docs/REVIEWER_READING_LIST.md:75`; `docs/spec_revision_log.md:9`, `:16`; `docs/PROPOSAL — EVENT SIZES FROM THE HOLMES–RAHE SCALE.md:8`
- **Severity:** minor · **Confidence:** high · **Verified:** re-read by the orchestrator (spec `:2127` and the log's headings)
- **What is wrong:**
  - The spec's own pointer says the log covers "4 to 35"; the checklist says "4 to 35" and the reading list "4 to
    36". The log's headings run to revision 37.
  - The log's tags for revisions 36 and 37 say "pending the owner's review", while the spec's status line and its
    rev36/rev37 tags say approved 2026-10-24 (T-125).
  - The Holmes–Rahe proposal still says it waits on G2, which is built.
- **Quoted:**
  - spec:2127 — "The record of every revision of this specification, 4 to 35, …"
  - log:9 — "⟦rev37 · decided as recommended under the owner's instruction of 2026-10-22, pending the owner's review ·"
- **How checked:**
  - The orchestrator grepped the log's `### Revision` headings, which include 36 and 37 at lines 16 and 9.
  - L3 searched the spec's `⟦rev3[67]` tags.
- **Fix:** set every restatement to "4 to 37". Change the log's two tags to "approved by the owner 2026-10-24
  (T-125)". Delete the proposal's "waits on G2" clause.

#### MR-17 — `_STATUS.md`'s STATE block says the full sweep did not run, and its "Next" is a review that closed

- **Found by:** L3 · **Where:** `docs/theory/_STATUS.md:3`, `:24`; `docs/implementation_plan_version_2.md:5`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** the block CLAUDE.md says to read first contradicts itself: `:3` says the full sweep "did not
  run", while `:24` says part II "completes stage 2's full sweep". Its "Next" item is T-125, which closed 2026-10-24.
  The plan's status line repeats "did not run".
- **Quoted:** _STATUS:3 — "the full sweep is projected at 44 h and did not run"
- **How checked:** L3 read `_STATUS.md:1-109` and plan `:1090`.
- **Fix:** say the 44-hour single run was replaced by three batches that complete it (34 keys), and set "Next" to
  the stage 2 milestone review's triage.

#### MR-18 — README points only at the frozen version 1 status page

- **Found by:** L3 · **Where:** `README.md:54`, `:76-90`; `docs/REVIEWER_READING_LIST.md:125-126`, `:156`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** README says "the status page says where each criterion stands now" but names only
  `version_1_status.md`, which is frozen at the tag, and never `version_2_status.md`. The reading list says README
  gives the commands that regenerate each record, but none of the version 2 generators is listed.
- **Quoted:** README:54 — "the status page says where each criterion stands now."
- **How checked:** L3 searched README for `version_2_status` and `sweep_record` and found nothing. It globbed the
  tools.
- **Fix:** point both at `version_2_status.md`, and list the version 2 generators or reword "each record".

#### MR-19 — The explainer's front-matter date is 2026-10-21; it was edited 2026-10-22 and 2026-10-24

- **Found by:** L3 · **Where:** `docs/model_explainer.md:5`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** the body has dates of 2026-10-22 to 2026-10-24 (`:455`, `:643`, `:1323`, `:1989`). A stale
  date here was fixed once before (reading list `:213`) and has recurred because no test checks it.
- **Quoted:** explainer:5 — "date: 2026-10-21"
- **How checked:** L3 grepped the explainer for later dates.
- **Fix:** set the date to the last edit, and add a document-claim test comparing it with the newest date in the
  body.

#### MR-20 — CLAUDE.md says "all 1195 tests pass"; one of them is a strict expected failure

- **Found by:** L3 · **Where:** `CLAUDE.md:91`; `tests/test_spec_consistency.py:367`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** the run gives 1194 passed and 1 xfailed. The count test requires the phrase "all N tests
  pass", so the gate forces the wording.
- **Quoted:** CLAUDE.md:91 — "**A red suite is never normal — all 1195 tests pass**"
- **How checked:** against the suite run in the header, and L3 read the regex at `test_spec_consistency.py:367`.
- **Fix:** "1195 collected: 1194 pass and 1 is a strict expected failure (A6)", and widen the regex.

#### MR-21 — `two_mac_sweep.md`'s "Not verified" section predates part II

- **Found by:** L3 · **Where:** `docs/two_mac_sweep.md:153-158`, `:15`
- **Severity:** minor · **Confidence:** medium · **Verified:** not challenged
- **What is wrong:** the how-to says the cross-check runs "before every split". Its "Not verified" section still
  counts 27 swept criteria and says no part II criterion is declared. Part II's four keys ran on Mac Mini1 with no
  recorded part II cross-check.
- **Quoted:** two_mac_sweep:153 — "Two of the 27 swept criteria were compared, … no part II criterion is declared yet."
- **How checked:** L3 searched CHANGELOG and the plan for "cross-check" and found only the 2026-10-23 runs on
  `M11.C.16` and `M11.C.23`.
- **Fix:** update the section to 34 keys and state the part II gap. Run one part II recompute here, or record why
  the 2026-10-23 checks cover it (see Questions).

#### MR-22 — T-011 and T-012 wait on group G's regeneration, which has run; MR-23 of the stage 1 review is still open

- **Found by:** L3 · **Where:** `docs/todo/2026-10-21_stage_1_milestone_review.md:55-56`, `:67-70`; `src/bowen/engine/live_positions.py:1`; `tools/ensemble_record.py:30-33`; `docs/version_2_plan.md:131-133`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** both items were deferred "with group G's regeneration". That ran on 2026-10-23 (`_STATUS.md:18`)
  without either change. The docstring still says "four places", and the Phase C ensemble record is still not
  rendered from a cache.
- **Quoted:** live_positions.py:1 — "The live-position predicate — one rule, used in four places."
- **How checked:** L3 read the code sites and `TODO.md:28-29`.
- **Fix:** re-point T-011, T-012 and the plan sentence at "the next change that moves the engine hash", the trigger
  T-127 already uses.

#### MR-23 — The event-size and contact-class loaders' refusals are untested

- **Found by:** L4 · **Where:** `tests/bowen/test_event_sizes.py:89-96`; `src/bowen/io/load.py:47-106`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** the test named "the table is validated on load" asserts `callable(load_event_sizes)` and that the
  committed table is well formed. None of the `ConfigError` branches of `load_event_sizes` or
  `load_contact_classes` is exercised. The sibling `load_act_classes` has parametrized broken-table tests
  (`test_config.py`, about 360–380).
- **Quoted:**
  - test_event_sizes.py:89 — `def test_m1f6a_the_table_is_validated_on_load():`
  - test_event_sizes.py:96 — `assert callable(load_event_sizes)`
- **How checked:** L4 grepped `tests/` for both loaders and for `pytest.raises(ConfigError` near them.
- **Fix:** add one broken-table case per `ConfigError` message, in `load_act_classes`'s style.

#### MR-24 — The strict xfail for `M5.B.3` would also absorb a broken arm

- **Found by:** L4 · **Where:** `tests/bowen/test_cutoff_stock.py:422-442`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** the arm-construction asserts (act placed, tie `ORDINARY` after bridging) sit inside the xfail
  and raise the same `AssertionError` as the direction check. The test stays green if the arms break.
- **Quoted:** test_cutoff_stock.py:422 — `assert scenario.last_source.status[(bridge_week, RAVI)] == "placed", …`
- **How checked:** L4 read lines 394–442.
- **Fix:** move the arm checks to a test that is not marked xfail, and keep the xfail on the direction comparison
  only, with `raises=AssertionError`.

#### MR-25 — `--recompute` drops cached results from disk before the rerun finishes

- **Found by:** L4 · **Where:** `tools/version_2_sweep.py:366-375`; `tools/record_cache.py:175`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** the dropped entries are deleted in memory and the cache is flushed after every sample. The
  put-back runs only on a normal finish, so an interrupt leaves `docs/records_cache/sweep_v2.json` without them.
  `git checkout` recovers them, but nothing says they were lost.
- **Quoted:**
  - version_2_sweep.py:369 — `del cache.entries[k]`
  - record_cache.py:175 — `cache.flush()  # keep every finished row, whatever happens to the next item`
- **How checked:** L4 read both ranges and `test_sweep_two_macs.py:306`, which covers a failed rerun but not an
  interrupt.
- **Fix:** restore and flush the dropped entries in a `try/finally`.

#### MR-26 — † Pyflakes: four new items since the baseline, one an undefined name

- **Found by:** L4, orchestrator · **Where:** `src/bowen/engine/state.py:118`; `src/bowen/engine/iposition.py:332`; `tests/bowen/test_event_sizes.py:76`; `tools/group_g_readout.py:22`; also `src/bowen/policy/policy.py:232` (older)
- **Severity:** minor · **Confidence:** high · **Verified:** re-run by the orchestrator
- **What is wrong:** `state.py:118` annotates with `Event`, which is never imported. It works only because of
  `from __future__ import annotations`. `policy.py:232` assigns `weights = []` and never uses it. The rest are unused
  imports and locals. Nothing in the suite runs a lint.
- **Quoted:** state.py:118 — `def arising_due(self) -> tuple[Event, ...]:`
- **How checked:**
  - The orchestrator ran `ruff check --isolated --select F` at HEAD (30 items) and at the baseline (25). The items
    new since the baseline are the four sites above.
  - The repo has no ruff config, so only F rules were counted.
- **Fix:** import `Event` under `TYPE_CHECKING`, remove the dead locals, and add `ruff check --select F` to the suite.

#### MR-27 — † Spec text in `M10`: a fragment, a dangling clause, and a withdrawn ceiling

- **Found by:** L1 (first item), orchestrator (all three) · **Where:** spec `:1214` (`M10.C.2`), `:1175` (`M10.A.2`), `:1171` (`M10.A.1`) against `:314` (`M1.A.18b`)
- **Severity:** minor · **Confidence:** high · **Verified:** re-read by the orchestrator, with `git blame`
- **What is wrong:**
  - **`M10.C.2`:** a sentence begins in lowercase with its lead-in missing: "…does not hold still. the durations in…".
    L1 attributed this to the KB11 paraphrase, but the same text stands at the baseline (`git show 9cf8552`, line
    1206), so it predates this range.
  - **`M10.A.2`:** it ends with "→ `kb/kb10.md` · K10.9, because the model has 60–90 free parameters…", a clause
    left over from an earlier wording that repeats the sentence before it. Also at the baseline (`:1167`).
  - **`M10.A.1`:** it lists "the **ceiling** on `systems_perspective` (M1.A.18b)" as derived. `M1.A.18b` was
    corrected on 2026-08-28 to a *rate* coupling and says a ceiling would be `[I]`. The code has neither: fixed gains
    at `external.py:151` and `:224`. `M1.A.18b` is already *partial* in coverage. The contradiction between the two
    requirements is not recorded.
- **Quoted:** spec:314 — "*Corrected 2026-08-28: the requirement made `basic_level` a **ceiling** on the attainable value. The source makes a **rate** claim …*"
- **How checked:**
  - The orchestrator read all three lines.
  - `git blame` of `:1171` gives `44c439f` (2026-08-24); of `:1214`, `31e1146` (2026-10-22), whose fragment was
    present at the baseline.
  - The orchestrator read `spec_index.jsonl`'s `M1.A.18b` line ("its rate is not coupled to `basic_level`").
- **Fix:** restore `M10.C.2`'s lead-in, cut `M10.A.2`'s trailing clause, and change `M10.A.1` to "the rate of change
  of `systems_perspective`", or record the disagreement in T-004.

#### MR-28 — † The status page labels a pass whose named mutant survived as "no recorded mutation"

- **Found by:** challenge pass (side item) · **Where:** `tools/version_2_status.py:155-156`; `docs/version_2_status.md:100-101`
- **Severity:** minor · **Confidence:** high · **Verified:** re-read by the orchestrator
- **What is wrong:** any directional pass with no mutant that went red is labelled "pass, no recorded mutation". That
  includes `M11.C.48`, `M11.C.11` and `M11.C.15`, whose mutants ran and survived (plan `:699`). The label says the
  mutation was not run, when it was run and failed to kill.
- **Quoted:** version_2_status.py:156 — `s.status, s.blocks = "pass, no recorded mutation", {"W1", "W3"}`
- **How checked:** the orchestrator read lines 150–158.
- **Fix:** distinguish "pass, named mutant survived" from "pass, no recorded mutation".

## Systemic patterns

**SP-A — The status-line test compares only the first status word (MR-5, MR-6, MR-15, MR-17).**
- `tests/test_doc_status.py:42-49` classifies a status by its first open/settled word.
- A status that moves within the settled class goes unchecked: a later revision, a later part built, a review that
  closed.
- This is the stage 1 review's SP-2 (T-015, still open), recurring in text written in this range.
- One fix to the test, comparing dates and part lists as well as the first word, would cover all four.

**SP-B — Named tests are not resolved by any check (MR-9, with T-004's existing `M1.F.2` and `M11.D.12` items).**
- Spec rows and `Tests:` lines can name a test that does not exist, and the suite stays green.
- Coverage takes its evidence from `spec_coverage`, not from these names, so a broken name is invisible.
- A gate test that resolves every `path::name` in the spec and in `Tests:` lines would close the class.

**SP-C — A revision 36 rule change left the old rule standing in the surrounding text (MR-2, MR-10, MR-11, MR-13).**
- Revision 36 changed three rules:
  - `REDUCE_CUTOFF`'s transfer;
  - the cutoff's end;
  - `CONFLICT` binding and `OPEN_CONFLICT`.
- Each change landed in the code, the requirement and one explainer passage. Docstrings, a second explainer passage,
  the register row, T-004 and a coverage override kept the old rule.
- This breaks the project's same-commit rule for documents. The status check does not see it, because no status
  word changed.

**SP-D — Coverage *done* overstates when a clause is unbuilt or its test is meant to fail (MR-1, MR-4, MR-28).**
- `tools/spec_coverage.py` marks an ID done when any named test passes. A strict xfail passes, and a
  requirement-level test can pass while one MUST clause has no test.
- The version 2 status page has the matching blind spot for surviving mutants.

## Refuted in challenge

- **MR-4, "the only test named for `M5.B.3` is the xfail":** refuted. `docs/spec_coverage.md:268` cites six tests,
  and five pass on `M5.B.3`'s other clauses (`test_moves.py:250`, `:264`, `test_policy.py:350`,
  `test_rejection.py:147`, `:300`). What stands is that the *done* hides one unmet clause.
- **MR-3, "`docs/review/group_g_readout.md:26-29` presents a PASS":** refuted. That table gives arm means only and no
  verdict.
- **MR-3, "a reader is not warned":** refuted. The caveat is in the spec (`:1368`), the explainer (`:1377`),
  `readouts_phase_c.md:75`, the reading list (`:152`), `_STATUS.md:18`, the plan (`:551`) and T-119/T-123. The status
  page (`version_2_status.md:100-101`) marks the pass not final. What stands is that the two generated records carry
  no caveat.
- **Challenge side item, "the `M11.5` row for `M11.C.48` (spec `:1421`, 'a pass or a fail is informative')
  contradicts its `M11.C` row (`:1368`, 'not evidence')":** not adopted by the orchestrator.
  - The `M11.5` row states the criterion's class: a composite's verdict is informative because no rule writes it.
  - The `M11.C` row says the one recorded pass rests on a defective baseline.
  - These are different claims, and the orchestrator read both rows.
- **L4 unverified item, "`cutoff_stock.bridge_ties` uses `invariant_tolerance` as a model threshold":** refuted by
  the orchestrator.
  - `cutoff_stock.py:188` uses it, at 1e-9 (`constants.md:85`), only to absorb floating-point remainder when the
    third step of `reference / tie_repair_steps` is subtracted.
  - The model's threshold is the step count, not the tolerance.

## Checked and clean

- **L1 · `config/bowen/`:** `constants.md`, `event_sizes.md` and `contact_classes.md` read in full. Every row is
  `[I]`, and the new rows (`constants.md:171-172`, `:253-259`) cite the owner or a measurement, never the corpus.
- **L1 · `src/bowen/` literals:** every changed hunk and all of `cutoff_stock.py`. No new magic numeric literal, only
  structural `+1`, `1.0/len` and `SCALE_MAX`. L4 found the same and confirmed `test_m11d2_no_magic_literals_in_engine`
  passes.
- **L1 · rules that write outcomes:**
  - `CUTOFF` comes only from a chosen act (`cutoff_stock.py:15-17`, `contact.py:71`, `event_effects.py:67`).
  - `OPEN_CONFLICT` (`consolidate.py:64-78`) is a tie label derived from chosen `CONFLICT` acts, like `DISTANT`.
  - The policy does not read `danger`.
- **L1 · explainer citations in the changed ranges:** `:1333` against L10.3, `:1350` against KS04.1, `:1357` against
  KS18.5. `:1519` paraphrases KB01 and cites the interview; `:1515` and `:1521` quote written chapters.
- **L1 · the stage 1 review's KB11 item:**
  - Spec `M10.C.2` (`:1214`) and `_LEDGER.md:1398` (L16.3) now paraphrase.
  - `DECISIONS FOR APPROVAL.md:51` (A1) still quotes verbatim. That is the owner's decision of 2026-10-22 ("A1 stays
    frozen"), so it is an accepted exception, not a defect.
- **L1 · the stage 1 review's MR-2, MR-3, MR-4, MR-7 and MR-8:** confirmed fixed at HEAD (explainer `:970-972`, `:382`,
  `:387`, `:807`, `:811`, `:946`).
- **L1 · premise framing and M11.F.9:**
  - The records carry each criterion's class and the premise warning (`phase_c_ensemble_record.md:3-6`,
    `phase_d2_ensemble_record.md:5`, `version_2_sweep_record.md:8`).
  - `group_g_readout.md:11` says "Not a real family (`M11.F.9`)".
  - No text in the slice presents a run as speaking about a real family.
- **L1 · `docs/readouts_phase_c.md`:** read in full. It discloses T-123 at `:73-77`.
- **L2 · `M1.D.2b` (other than MR-7), `M1.D.2c` repair, `M1.F.6a`, `M6.3g`, `M7.C.1l`, `M7.D.1a`, `M7.D.2a`'s G2
  release:**
  - Each was checked clause by clause against `cutoff_stock.py`, `moves.py`, `consolidate.py`, `symptoms.py`,
    `event_sizes.py`, `life_course.py`'s call sites and their tests.
  - The constants in `config/bowen/constants.md:172-174`, `:252-259` match the register, and the
    `cutoff_threat_weight` sizing gives 504.
  - A cut is entered only at step 4, so no week runs with an uncapped threat.
- **L2 · `M11.C.48` readout and arms:** `negative_runs.py` and `criteria_d2.py:111-166`, `:472-480` match the spec.
  Coverage marks it *partial*, and T-119 and T-123 are recorded.
- **L2 · T-004's list (brief §4 L2, second check):**
  - The sub-list "Spec and code disagree; the owner decides which stands" has not grown since the baseline: lines
    58–75 of the baseline `TODO.md` and lines 42–59 of the record at HEAD are byte-identical (orchestrator, `diff`).
  - New disagreements found here (MR-7, MR-8, MR-9) were not added to it, and one item is stale (MR-10).
- **L2 · absolute thresholds:** no new or changed `M11.C` acceptance test asserts one. The 504 ± 1% check sizes an
  `[I]` constant, not a criterion.
- **L2 · coverage overrides changed in this range** (lines 322, 623, 700, 922, 1002, 1006, 1133): nothing wrong.
  `M11.C.44`'s *done* rests on a mutant that went red (`phase_c_mutation_record.md:37`).
- **L3 · counts in prose:** 581 IDs and 48 `M11.C` criteria match `spec_index.jsonl`, and the spec, ledger and guide
  line counts match their "about" figures.
- **L3 · sweep arithmetic:** the sweep numbers in `_STATUS.md:23-24`, the reading list and the plan `:1090` reconcile
  with `version_2_sweep_record.md:32-65`:
  - 34 keys;
  - 155, 182 and 70 child runs;
  - 431 cached results;
  - 327 `M11` rows in `sweep.md`.
- **L3 · revision 35 dispositions:** 262 rows (35 stay, 51 move, 176 rewritten), matching the reading list and
  checklist.
- **L3 · engine hash:** `66dcbb1e8ba5` agrees across `_STATUS.md` and four record headers.
- **L3 · proposal and plan status lines:** the spec's front matter and its rev36/rev37 tags record T-125. The status
  lines of the G1, G2, Holmes–Rahe and SP-1 proposals, the version 2 plan, the brief, `two_mac_sweep.md`, the
  revision 35 dispositions and the ergonomics plan agree with it.
- **L4 · engine purity (`M16.B`):**
  - File I/O happens only in `io/load.py`, `io/sinks.py`, `run.py` and the config reads in `ensemble/criteria.py`.
  - `test_engine_purity.py` enforces this.
  - `canonical_state` includes the new `arising`, `cutoff_peak` and `reference_level` fields.
- **L4 · skips and vacuous asserts:**
  - Both skips give reasons (`test_python_versions.py:55`, `test_render.py:359`).
  - A grep for `assert True`, `or True`, `assert callable` and bare `except` found only MR-23's assert.
- **L4 · CI and dependencies:**
  - `pytest.ini` deselects `ensemble`, as CLAUDE.md says.
  - `tests.yml` runs the suite on 3.11–3.13.
  - `requirements.txt` and `tests.yml` are unchanged in this range.
- **L4 · P38 caching:**
  - The version 2 sweep record renders from the cache (`--render-only`, `test_s2_11`).
  - `merge_cache.py` refuses conflicting NumPy versions and keys.
  - `negative_run_values` is cached.
- **L4 · external calls:**
  - Every subprocess call in `tools/` has a timeout.
  - `two_mac_sweep.py` retries read-only calls with backoff, never retries `start`, guards a second run with
    `pgrep`, and refuses to overwrite on `fetch`.

## Unverified

- **Short KB phrases:** `config/bowen/constants.md:56` ("grotesque") and `model_explainer.md:364` ("the *shout them
  down* grotesque") use KB wording in italics. Both were borderline in the stage 1 review (its `:489`) and are
  unchanged. Whether a single word or short phrase counts as verbatim quotation needs the owner's ruling (Questions).
- **Other ASR quotations:** whether any verbatim ASR quotation remains that `test_asr_quotations` misses, for example
  with the citation in another paragraph. Neither L1 nor L3 searched the 13 unread `config/bowen` files or the
  unchanged explainer ranges for quoted spans near `KB`/`Tape`.
- **What `negative_runs.py:77-81` counts:** it counts negative acts in either direction, including the cutter's own
  `DISTANCE` and `CONFLICT`. That matches the G1 proposal §4. Whether it ties the readout to the cutter's own state
  would be settled by recomputing `M11.C.48` from acts the cutter received only.
- **`M7.D.1a`'s channels:** it gives every illness onset the `OWN_ILLNESS` size on all three channels. Whether that
  fits the three-channel distinction is a theory question, graded `[I]` (Questions).
- **`M11.C.48`'s comparison:** whether its within-run comparison (a cut tie against matched uncut ties) is a two-arm
  direction under `M0.4`. It is not in `M11.4`'s exceptions table (spec `:1278-1293`).
- **An old Phase D plan status:** `implementation_plan_phase_d.md:1412` (§4.9) says "the proposals below await the
  owner's decisions". §4.10 records them approved; this is probably superseded history outside this range.

## Questions only the owner can answer

1. **`M1.D.2b` decay (MR-7) and `M1.D.2c`'s reset of `R` (MR-8):** in each case, does the spec or the code stand?
2. **`M17.E.1`'s named ratio (MR-1):**
   - Is it still wanted at stage 3's full sweep?
   - Which two rates make the ratio? The spec names the class ("harden or attenuate" against "anxiety moves"), not
     the constants.
3. **Short KB phrases ("grotesque", "shout them down"):** are these verbatim quotation under `_SOURCE_QUALITY.md`, or
   accepted as single words?
4. **The two-Mac cross-check (MR-21):** was "before every split" meant to be met by the 2026-10-23 cross-checks at
   the same engine hash, or should part II get its own?
5. **`M7.D.1a`:** should a social or mental onset carry the "personal injury or illness" size?
6. **`M11.C.48`'s comparison:** does it count as a two-arm direction under `M0.4`, or does it belong in `M11.4`'s
   exceptions table?
7. **Historical review guides:** `docs/REVIEW_version_1_guide.md:101` describes the removed `REDUCE_CUTOFF` transfer
   (MR-2). Are historical guides corrected, or left as records of their time?
