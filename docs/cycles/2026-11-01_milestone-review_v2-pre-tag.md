# Milestone review: version 2 before the tag (mini2, 2026-11-01)

This is the pre-tag milestone critical review of EPModel, the one `docs/milestone_review_brief.md` §2 item 2 requires before the tag
`version-2`. It reviews the model against the brief's four lenses, theory fidelity first, and focuses on the changes since
the stage 3 review. The reviewing session edited nothing except this report and fixed nothing.

## Header

- **Commit reviewed:** `355031f974da8cd1d090f37007f3d26b5a375925` (`origin/main`, the W5–W8 sign-offs and T-151), in
  the worktree `EPModel-review-2026-11-01` on the branch `review/2026-11-01-pre-tag`.
- **Baseline:** `161da4e83d56c0455bc6a1dec7fc8ea8f95a97c0`, the commit the stage 3 review covered
  (`docs/cycles/2026-10-27_milestone-review_v2-stage-3.md`). There are 12 commits since, from `1ca9bb4` (spec revision 40)
  to `355031f`; they carry revisions 40–44, the stage-end sweep's rerun on revision 44, and the W5–W8 sign-offs.
  - The brief's §3 step 5 filter lists 108 files (+6,782 / −723).
  - The extended filter below leaves 74 files (+2,197 / −437). These were split into the four lens slices.
    `.github/workflows/tests.yml` did not change.
- **Machine:** Mac mini2 (`mini2.local`, Darwin 27.0.0).
- **Suite:** `1290 passed, 52 deselected, 2 xfailed in 348.44s`, exit 0, on Python 3.12.3. This matches CLAUDE.md:91
  (1292 tests: 1290 pass, 2 strict expected failures; 52 `ensemble` tests deselected). The two expected failures are
  `test_m5b3_third_party_cost_is_the_lost_outlet` and `test_m5d3_abort_is_the_usual_first_outcome` (T-158).
- **CI for the commit:** run `41327709622` (push to `main`, head `355031f`), `completed success`: `test (3.11)`
  success, `test (3.12)` success, `test (3.13)` success.
- **Tokens used:**
  - Subagents: 923,715, taken from each result's usage total (L1 183,844; L2 228,773; L3 172,892; L4 223,846;
    challenge 114,360).
  - The orchestrating session: not measured as a total. The app reports only the current context window, and the
    session was compacted once before consolidation, so there is no cumulative figure.
  - Expected cost, stated before starting from the stage 3 header: 392,803 subagent tokens and 12.8 minutes. This
    review used about 2.4 times that. Each lens worker used 3.8–5.2 times its stage 3 counterpart (L3 about the
    same). The workers read every changed hunk with its enclosing function, and the brief's diff was smaller.
- **Wall time:** 17.1 minutes. The clock runs from the start of §3 step 3, the suite, at 09:44:26 local, to the
  challenge pass's return at about 10:01:30 (it was launched at about 09:55:10 and ran 371.4 s). Writing this file comes on top.
  - The worktree, the virtualenv and the cost statement came before the clock started.
  - The four lens workers ran in parallel: L1 8.1, L2 7.8, L3 7.1 and L4 7.2 minutes. The challenge pass took
    6.2 minutes.
- **Findings:** 0 blocker, 3 major, 12 minor (15). One major (MR-4) was downgraded to minor in challenge; none was refuted. Before the challenge: 0 blocker, 4 major, 11 minor.

**Deviations from the brief.**

1. **Virtualenv.** The suite ran in a fresh virtualenv, outside the repo, in the session's scratchpad. It was built from
   `requirements.txt` with python.org's Python 3.12.3, a version in CI's matrix, and has numpy 2.5.4, pytest 8.4.2 and
   ruff 0.17.3 (the pin). The system `python3` has no `ruff`.
2. **Exclusions beyond the brief's filter.** This is the stage 3 report's set:
   - generated pages: `docs/spec_rules/`, `docs/spec_index.jsonl`, `docs/theory/ledger_index.jsonl`,
     `docs/version_2_status.md`, `docs/version_2_revision_*_changes.md`, `docs/TODO_archive.md`;
   - dated logs: `docs/CHANGELOG_archive.md`, `docs/todo/`;
   - `docs/cycles/`, which holds the previous reports.

   Workers took `TODO.md`'s index line, then read only the record it names. They searched the generated pages
   (`version_2_status.md` included) only for a claim they were checking.
3. **Spec read (§4 Scope).** The orchestrator did not read the whole spec.
   - **Read in full:** lines 1–1327 (front matter, M0–M10, M11's preamble, M11.1–M11.4).
   - **Not read in full:** lines 1328–2141 (the `M11.C` rows, `M11.5`, `M11.D`–`M17`). The Read tool reported that
     the rest would not fit in the context left. Of these lines, the orchestrator read only the `M11.C.17` (1343) and
     `M11.C.21` (1347) rows.
   - **What the workers read of 1328–2141:** every changed hunk (L2), the `M11.C` and `M11.5` rows named under L1's
     coverage, `M11.F.2` (L1), and `M16.B` 1950–1974 (L4).

   No lens read the unchanged `M11.C` rows, `M11.D`, `M11.E`, `M11.G`, `M12`–`M15` or `M17`. A contradiction between
   one of those and a changed section would not have been seen.
4. **Checks the orchestrator ran.** Workers are read-only, so the orchestrator ran four of their "unverified" items:
   - **The SP-1 check at two words (MR-9).** The rules of `tests/test_asr_quotations.py` were run with `min_words` set
     to 2. They return 15 spans. The four spans read are an owner quotation (spec :765), a written-source quotation
     (`KS04.13`, explainer :2138), a spec phrase (:486) and one coined term beside `KB03` (`_LEDGER.md:594`). The other
     11 were not read.
   - **The golden runs with T-165 removed.** In a throwaway worktree (since removed), `resolve`'s external-mover branch
     was replaced by `if False:` and `tests/bowen/test_engine_golden.py` was run: `3 passed in 3.29s`. The golden runs
     never reach that branch. T-165's own unit test does. The golden docstring credits revision 41 alone with moving
     the fingerprints, which is consistent with this.
   - **CI** for `355031f` (above).
   - **The explainer's seed count for `M11.C.48[phase_d]`** against `criteria_d2.py:478–484` and `criteria.md:228`
     (MR-12).
5. **Known open items not re-reported.** L4 found three items from the stage 3 review still open at HEAD: MR-16
   (T-152), MR-17 (T-153) and MR-21 (T-157). They are tracked in `TODO.md` (:140, :141, :145), so they are listed under
   "Checked and clean" and not counted here. L4 adds one new reason for T-157.

## Coverage matrix

Key: "full" = read in full; "diff" = every changed hunk read with its enclosing section or function; "sampled" = named
ranges read; "searched" = grepped for the claim being checked; "—" = not in that lens.

| Area | L1 theory fidelity | L2 spec against code | L3 documents | L4 engineering drift |
|---|---|---|---|---|
| Spec `bowen_agent_model_spec_v2.md` (2,141 lines) | 1335–1348, 1355, 1359, 1374–1403, 1414, 1428–1429, 1483–1494 | every revision 40–44 hunk | 1–10; counts against `spec_index.jsonl` | `M16.B` (1950–1974) full |
| The orchestrator's read of the spec | 1–1327 full; 1343, 1347 | — | — | — |
| `src/bowen/` engine | every changed hunk | diff, with `iposition.py` 20–359, `act.py` 95–169, `tick.py` 118–263, `stress_response.py` 108–232, `life_course.py` 370–430 and 560–620, `slow_tick.py` 50–139; `observe`, `rejection`, `state`, `objects`, `policy` through the diff only | searched (`displaced_swing`) | every changed hunk; I/O and imports grepped |
| `src/bowen/` ensemble and readouts | every changed hunk; `criteria_d2.py` 470–485 | `negative_runs.py` full; `criteria.py` 298–326, `criteria_d.py` 640–709, `criteria_d2.py` 112–156, 465–485 | — | diff, with `criteria.py` 299–325, `criteria_d.py` 668–697, `criteria_d2.py` 466–485 |
| `config/bowen/`, `config/sweep/sweep.md` | changed hunks full | `criteria.md` 38–77; `sweep.md` and `stress_response.md` searched | `sweep.md` searched | — |
| `docs/model_explainer.md` (2,252 lines) | 250–289, 586–615, 764–776, 1026–1039, 1428–1439, 1562–1577, 2054–2065, 2222–2252 | — | status line and date; rev 44 lines by grep | — |
| `docs/theory/` | `_SOURCE_QUALITY.md` 59–74; `fe06.md` 90–119, 319–337; `fe11.md` 240–276; `ks04.md` 85–94; ledger by index | — | `_STATUS.md` 1–40 | — |
| `version_2_dispositions.md`, `REVIEW_version_2_outputs.md`, `version_2_definition_plan.md` | full; full; 1–34, 106–130 | full; —; — | full; full; 3–9, 104–148 | — |
| Reading list, checklist, README, CLAUDE.md, `version_2_plan.md` | README and reading-list diff | — | README full; checklist 1–50, 145–249; reading list 1–40, 182–316; CLAUDE.md 20–99; plan 1–12, 392–413 | CLAUDE.md suite count |
| Generated records, `phase_c_completion_report.md`, `spec_coverage.md` | report 45–52, 412–481, 528–535; `phase_d2_ensemble_record.md` and `phase_d_mutation_record.md` searched | `phase_d_mutation_record.md` 1–60; `phase_d_ensemble_record.md` 729–753; `spec_coverage.md` 440–444, 456, 471; Phase C mutation record searched; Phase B mutation record not read | `version_2_sweep_record.md` 30–79; `phase_d_ensemble_record.md` 215–236; `spec_coverage.md` searched | — |
| `docs/review/` | framing line grepped on every page | — | — | `group_g_readout.md` searched |
| `tests/` | `test_iposition.py` 422–502; `test_asr_quotations.py` 1–150 | the diff hunks of 12 files under `tests/bowen/`; `test_tick.py` 148–200; `test_phase_d_criteria.py` 380–467 | `test_version_2_outputs.py` full; `test_doc_claims.py`, `test_stale_text.py` searched; `doc_claims.json` full | every changed hunk; `test_stage_3_probes.py` full; ranges of 8 files (L4's list) |
| `tools/`, CI, `pytest.ini`, `requirements.txt` | — | `spec_coverage.py` 140–154, 288–299; diffs of `phase_d_mutation_record.py` and `spec_coverage_overrides.json` | `requirements.txt` full | every changed hunk; `group_g_readout.py` 56–190, `version_2_status.py` 95–174 and 228–382, `version_2_sweep.py` 185–274, `c16_report.py` 340–392; `tests.yml`, `pytest.ini`, `requirements.txt` full |
| `TODO.md` and the records it names | stage 3 triage record 77–94, 160–175 | `TODO.md` full; T-004 record 30–59; triage record 77–90 | `TODO.md` and `docs/todo/` searched | triage record 40–89 |

Not reviewed by any lens:

- spec lines 1328–2141, except the hunks and rows named above (deviation 3);
- `docs/phase_b_mutation_record.md` (no row in it changed in revisions 40–44, as far as the spec diff shows);
- `CHANGELOG.md` beyond lines 1–27 and 80–95; `docs/version_1_status.md` (frozen);
- the reading list's §1–§6 outside the diff hunks; `_STATUS.md` below line 40;
- the bodies of `docs/review/` pages; `version_2_definition_plan.md` 35–103 and 149–178;
- 11 of the 15 two-word spans from deviation 4.

## Findings

### Major findings

#### MR-1 — The explainer says the model meets `M5.D.3`; since revision 41 it does not, and the owner kept that as a finding

- **Found by:** L1 · **Where:** `docs/model_explainer.md:2247–2249`. The same claim is in
  `config/bowen/constants.md:56–57`, and the old measurement is at `config/bowen/constants_changes.md:93` (found in
  challenge).
- **Severity:** major · **Confidence:** high · **Verified:** yes (challenge pass)
- **What is wrong:** the "start here" document says the model meets a spec requirement that the owner has recorded as
  failing.
  - Since revision 41 (T-133), an abort is not the usual first outcome. `test_m5d3_abort_is_the_usual_first_outcome`
    is a strict xfail, and the owner closed T-158 at (b) on 2026-10-31, keeping the divergence as a finding.
  - The paragraph says nothing of this; grep finds no "T-158" in the explainer.
  - It also names the wrong fixture: the test runs on the triad with F as the mover, not on the reference family with
    Ravi. The test's docstring records that in the Phase C family PREPARE failed in 1,907 of 1,941 attempts. So the
    explainer's account of the reference family has no test behind it.
  - `constants.md` says `hold_gain` "was chosen against `M5.D.3`, so that aborts are the usual outcome", and
    `constants_changes.md:93` keeps the revision-30-era measurement with no note that it no longer holds.
  - Explainer :768 and :774 state the theory's sequence, not the model's, and are acceptable.
- **Quoted:**
  - explainer :2247–2249 — "In the reference family, with Ravi given perspective, … most that meet / opposition abort,
    … (`hold_gain`) was set so that aborting is the usual outcome, which the spec requires (`M5.D.3`)."
  - `tests/bowen/test_iposition.py:474` — "calmer: over the 192 seeds, 62 abort there and 66 hold (125 and 106 at
    revision 40), and 118"
  - `docs/todo/2026-10-28_stage_3_milestone_review.md:92–93` — "`M5.D.3` and its strict expected failure are kept as a
    finding: after T-133 an abort is not the usual first outcome in the model."
- **How checked:** L1 read the test (:471–490), its fixture docstring (:422–436) and T-158. The challenge pass re-read
  them, grepped the explainer, and found the `constants.md` and `constants_changes.md` wording. The orchestrator read
  `constants.md:56–57` and the T-158 lines.
- **Fix:**
  - Restate the explainer paragraph: the fixture is the triad; since revision 41, 62 of 192 seeds abort and 66 hold,
    and 118 sequences end said in anger; `M5.D.3`'s ordering no longer holds and is kept as a finding (T-158 (b),
    owner 2026-10-31).
  - Note the same in `constants.md:56–57`, and date `constants_changes.md:93`'s measurement as superseded.
  - Whether spec `M5.D.3` should carry the note is question 6.

#### MR-2 — MR-22's exit-condition change is still pending the owner, and the lists of what remains before the tag leave it out

- **Found by:** L3 · **Where:**
  - `docs/bowen_agent_model_spec_v2.md:4` (status line)
  - `docs/version_2_definition_plan.md:8`
  - `docs/spec_revision_log.md:54`
  - `docs/theory/_STATUS.md:3` and the plan's :145 ("Next: the owner's decision on closing version 2")
- **Severity:** major · **Confidence:** high · **Verified:** in part (challenge pass: "untracked" overstates it)
- **What is wrong:** revision 40 restated `M13`'s Phase C exit. It now reports `M11.C.49`'s verdict as a finding
  instead of requiring a pass, which changes an exit condition, not only its wording. That change is still "pending
  the owner's review". Without it, `M13`'s exit would require `M11.C.49`, which fails, to pass.
  - The plan says plainly that MR-22 is undecided.
  - No `TODO.md` item asks for the decision. The stage 3 triage list counts MR-22 under "Fixed at revision 40" with no
    checkbox. T-168 names MR-22's pending text only as a gap in the stale-text scan.
  - The lists of what remains before the tag (`_STATUS.md:3`, plan :145, README) name only the closing decision.
  - The owner's 2026-10-31 row accepting `M11.C.49`'s failure (`version_2_dispositions.md:28`) agrees with the
    restatement, but it does not mention `M13` or the exit.
- **Quoted:**
  - spec :4 — "`M13`'s Phase C exit reports `M11.C.49`'s verdict as a finding instead of requiring a pass (MR-22, as
    recommended, pending the owner's review)"
  - definition plan :8 — "the owner decided all but MR-22 on 2026-10-31"
  - `CHANGELOG.md:325–327` — "MR-22 restates an exit condition, not only wording … pending the owner's review."
- **How checked:** L3 read the status lines and searched `TODO.md` and `docs/todo/`. The challenge pass re-read each
  line, the triage record (:17–19) and T-168 (:77–78), and found MR-22 in 12 files by grep, none of them README or
  `_STATUS.md`.
- **Fix:** put the decision to the owner before the tag (question 1). Then record it in the spec's status line, the
  revision log and the definition plan, and add it to the lists of what remains until it is decided.

#### MR-3 — `M11.C.34`'s pass is reported as the model's result, but its readout measures the premise share, and no mutant tests the learned limb

- **Found by:** L1 · **Where:**
  - `docs/version_2_dispositions.md:31`, repeated in `docs/version_2_status.md:89`
  - spec `:1414` (`M11.5`'s `M11.C.34` row) and `:1359`
  - `tools/phase_d_mutation_record.py:106–110`
  - `src/bowen/ensemble/criteria_d.py:613`, `:660–663`
- **Severity:** major · **Confidence:** medium · **Verified:** in part (challenge pass; restated here in its narrower
  form)
- **What is wrong:** `M11.5` classes `M11.C.34` as mixed: "the automatic share is premise; which automatic acts appear
  is learned".
  - The readout counts every move whose kind is in `AUTOMATIC_KINDS`, the whole automatic channel. Summing over every
    automatic kind removes the "which acts appear" part. What is left is mostly `M4.D.1a`'s level-dependent channel
    share, which is the premise limb (`policy.py:293–294`), times how much the onset raises acts toward the bearer.
  - The only proving mutant, `mixing-weight-blind-to-level` (revision 43, T-162), fixes that share. It turns the result
    red (−0.04 ± 0.12, p 0.26). That is evidence that the level difference runs through the premise.
  - No mutant switches off the learner or the targeting while keeping `M4.D.1a`, and `M11.1d`'s sign-inverted mutant
    has not run. So the pass is not shown to depend on learned behaviour.
  - The mutant edits `observe.self_channel_weight`, which `M5.F.4`'s demand share also reads (`iposition.py:147`). The
    mutation record discloses this; the disposition does not.
  - `M11.5`'s row carries no correction dated after T-162, although `M11.5` requires each correction to be recorded
    with its date.
  - The pass is weak (−0.113 ± 0.129, p 0.039), and the sweep holds the direction in 1 of 8 samples.
  - CLAUDE.md: a premise passing must never be reported as a finding.
- **Quoted:**
  - dispositions :31 — "so the pass rests on the mechanism the row names. Recommended as the model's result under
    revision 41's rules"
  - spec :1414 — "| M11.C.34 | Mixed | `M4.D.1a` mixing weight | the automatic share is premise; which automatic acts
    appear is learned."
  - `phase_d_mutation_record.py` — `f = 0.5 ** params.self_channel_exponent` in place of `observe.py:105`'s
    level-dependent `f`
- **How checked:** L1 read the class row, the criterion row, the disposition, the status row, the mutation record's
  row 23 and T-162. The challenge pass read the mutant's edit, `self_channel_weight` and both its readers, and the
  readout. Its finding: "shown to rest on a premise" is not proved, but "not shown to depend on learning" is. The
  challenge pass also found T-051 (`docs/todo/2026-10-11_phase_d.md:57`) still open and saying `M11.C.34` fails and its
  mutant cannot bite; see MR-7.
- **Fix:**
  - Add a learner-off or bearer-blind mutant that keeps `M4.D.1a` (Unverified 2), and a mutant that fixes the weight
    only at `policy.py:293` (Unverified 1).
  - Until they run, restate the disposition and the status row: the pass is the premise share's, and the learned limb
    is untested.
  - Record a dated note in `M11.5`'s row.
  - Put the class to the owner (question 4).

### Major findings downgraded in challenge

#### MR-4 — † The checklist calls the sweep record T-159's rerun on revision 41, with `M11.C.48` refused

- **Found by:** L3 · **Where:** `docs/SPEC_REVIEW_CHECKLIST.md:214`
- **Severity:** minor (major before the challenge) · **Confidence:** high · **Verified:** yes (challenge pass)
- **What is wrong:** the checklist's label calls the record "the current record" and says `M11.C.48` is "refused at the
  rerun". The record is the T-170 rerun on `09ca2b46c7e0` (`version_2_sweep_record.md:25`), and `M11.C.48` has a
  fraction, 0 of 8 (:67).
  - The label has been stale since revision 43: `CHANGELOG.md:174–175` says "`M11.C.48` is no longer refused at the
    rerun".
  - T-170's list of documents (`CHANGELOG.md:69–76`) leaves out the checklist.
  - No test reads this label.
- **Why downgraded:** `M11.C.48` fails either way, and its accepted finding is unaffected, so no decision changes. The
  class is MR-10's.
- **Quoted:** checklist :214 — "**Generated, 2026-10-29 (stage 3's stage-end sweep, rerun on revision 41's engine,
  T-159):** … `M11.C.48` is refused at the rerun"
- **How checked:** L3 read the label against the record. The challenge pass also read the CHANGELOG entries and the
  tests.
- **Fix:** relabel it "Generated, 2026-11-01 (T-170, revision 44's engine `09ca2b46c7e0`)" and drop "refused". Add the
  checklist to the documents each sweep rerun updates.

### Minor findings

These were not challenged; the brief challenges blocker and major findings only. The orchestrator re-checked the ones
marked † against the files.

#### MR-5 — † The explainer gives revision 44's coach rule a `[K]` source that does not support it

- **Found by:** L1 · **Where:** `docs/model_explainer.md:771`; the same reasoning in spec `M5.D.5`
  (`docs/spec_rules/M05.md:63`)
- **Severity:** minor · **Confidence:** medium
- **What is wrong:** revision 44 stops the pull-up when the mover is the coach. The explainer says this follows from
  `FE06.10`. That entry says a member can rise above the therapist's level, so the coach's level must not be a ceiling.
  The pull-up as built is not a ceiling: it only raises people below the mover and never lowers anyone above. So
  `FE06.10` gives no reason to remove it. The removal is the owner's D5 (T-165), an `[I]` choice, and it carries no
  `[I]` grade. `FE11.6`, cited beside it, also says the therapist may serve as a model for others.
- **Quoted:**
  - explainer :771 — "a family member can go beyond the coach's level, so the coach does not lift anyone toward its
    own (`[K]` `FE06.10`)"
  - `docs/theory/family_evaluation/fe06.md:328` — "**`M1.E` must not cap a member's `basic_level` at the external
    agent's.**"
  - `iposition.py` `resolve` — "gap = max(0.0, state.people[mover].functional_level - other.functional_level)"
- **How checked:** L1 read `FE06.10` (`fe06.md:319–337`), `FE11.6` (`fe11.md:240–276`) and `resolve`'s diff. The
  orchestrator read explainer :771.
- **Fix:** grade the rule `[I]` (D5, T-165). Cite `FE06.10` and `FE11.6` as context ("no ceiling"; "a leader from
  within"), not as the reason, and drop the "so". Make the same change in spec `M5.D.5`.

#### MR-6 — † `M11.C.48`'s accepted finding says "not longer"; both families show the direction reversed

- **Found by:** L1 · **Where:** `docs/version_2_dispositions.md:29` (row 30's `[phase_d]` reading follows it)
- **Severity:** minor · **Confidence:** high
- **What is wrong:** on both families the interval excludes zero on the negative side. In the model a first cutoff
  follows a *shorter* run of negative contacts than the uncut ties stand on that week. The disposition says only that
  it is not longer, which reads as a null result.
- **Quoted:**
  - dispositions :29 — "So in the model a tie is not cut off after a longer run of negative contacts than its
    family's uncut ties stand on in the same week."
  - `docs/phase_d2_ensemble_record.md:20` — "`negative_run` -0.319 ± 0.1, p 1, does not hold"
  - `docs/phase_d2_ensemble_record.md:21` — "`negative_run` -0.113 ± 0.062, p 1, does not hold"
- **How checked:** L1 read both disposition rows and both record rows. The orchestrator read :29.
- **Fix:** say the direction is reversed, not null, on both families. Add that whether the matched-set selection
  explains the reversal has not been traced (Unverified 3).

#### MR-7 — † `M11.C.34`'s spec row still says the criterion fails and its mutant cannot bite

- **Found by:** L2 · **Where:** `docs/bowen_agent_model_spec_v2.md:1359`
- **Severity:** minor · **Confidence:** high
- **What is wrong:** the row's only measurement is revision 27's. It says the criterion fails and the declared mutant
  "cannot bite in any case". Revision 43 appended the new mutant but left those claims standing. The record shows a
  PASS at revision 41 and the mutant red at revision 43. The owner's `verdict reviewed` of 2026-10-31
  (`version_2_dispositions.md:31`) is not in the spec. The `M11.C.17` and `M11.C.21` rows had their measurements
  restated at revision 44; this row did not. See also MR-3. T-051 (`docs/todo/2026-10-11_phase_d.md:57`, still open)
  makes the same stale claim (found in challenge).
- **Quoted:**
  - spec :1359 — "**fails** — … do not differ by level (−0.04 ± 0.10). No selection reads `M1.A.19`'s detectors, so
    the declared mutant cannot bite in any case ⟦rev27 …⟧"
  - `docs/phase_d_ensemble_record.md:729–742` — `"outcome": "PASS"` (−0.113 ± 0.129, p 0.039)
- **How checked:** L2 read the diff hunk, the record entry and the mutation record's row 23. The orchestrator read
  :1359.
- **Fix:** mark the revision 27 measurement as superseded. Add the current measurement, the mutant's result and the
  owner's 2026-10-31 decision, as the `M11.C.17` and `M11.C.21` rows do.

#### MR-8 — † The coverage page quotes only version 1's dispositions, so three rows contradict the current records

- **Found by:** L2 · **Where:** `tools/spec_coverage.py:293–295`; `docs/spec_coverage.md:440`, `:444`, `:456`
- **Severity:** minor · **Confidence:** high
- **What is wrong:** the tool imports only version 1's dispositions. Three rows are wrong as a result:
  - `M11.C.17` reads as passing;
  - `M11.C.34`'s mutant reads as one that "cannot bite";
  - `M11.C.21`'s arm is described as it was before revision 44.

  The owner's version 2 rows for all three are at `docs/version_2_dispositions.md:31–33`. T-171 is the same cause on
  `docs/version_2_status.md`, but it covers five other rows and not this tool.
- **Quoted:**
  - `spec_coverage.py:293` — "from tools.version_1_status import dispositions"
  - `spec_coverage.md:456` — "…no selection reads `M1.A.19`'s detectors, so its declared mutant cannot bite"
- **How checked:** L2 grepped the tool for `dispositions` and read `version_2_dispositions.md` in full. The orchestrator
  read :293–295.
- **Fix:** have the tool also read `docs/version_2_dispositions.md`, quoting a version 2 row in place of the version 1
  row for the same key, and regenerate the page. One fix could cover T-171 as well.

#### MR-9 — † The SP-1 test flags transcript spans of four or more words; the owner's rule is two

- **Found by:** L1 · **Where:** `tests/asr_quotations.json:3` against `docs/theory/_SOURCE_QUALITY.md:73`
- **Severity:** minor · **Confidence:** high (the mismatch); low (that any ASR quotation slips through)
- **What is wrong:** under the owner's rule of 2026-10-24, a phrase of two or more words taken from a transcript is a
  quotation. The test flags only spans of four or more words. Its docstring lists other blind spots, but not this one.
  At two words the check returns 15 spans. The four the orchestrator read are not ASR quotations (deviation 4). The
  closest is the coined term "Therapeutic-itis" beside `KB03` (`_LEDGER.md:594`), which is one hyphenated word.
- **Quoted:**
  - `asr_quotations.json:3` — `"min_words": 4,`
  - `_SOURCE_QUALITY.md:73` — "a phrase of two or more words taken from a transcript is quotation, and is
    paraphrased."
- **How checked:** L1 read both files. The orchestrator ran the check at two words.
- **Fix:** set `min_words` to 2 and allowlist the written-source spans it exposes. Or name the gap in the docstring as
  an accepted limit. This is the owner's choice (question 8).

#### MR-10 — † Status lines and labels still describe revisions 41–43 as open or unbuilt

- **Found by:** L3 · **Where:**
  - `docs/theory/_STATUS.md:3`
  - `docs/implementation_plan_version_2_stage_3.md:3`
  - `docs/PROPOSAL — DISPLACED COPING (P2).md:3`
  - `docs/two_mac_sweep.md:3`
  - `docs/model_explainer.md:5`
  - `README.md:63–64`
- **Severity:** minor · **Confidence:** high
- **What is wrong:** six status lines or labels were not updated when revisions 41–44 were built or approved. These
  are the same class as the stage 3 review's MR-3, MR-4 and MR-14:
  - `_STATUS.md` says revision 43 is "pending the owner's review". It was approved on 2026-10-31. The stale-text
    test's pattern misses the possessive form, which is T-168.
  - The stage 3 plan lists two items as open: "the owner's answer on `M5.D.3` (T-158)" and the review of revisions 41
    and 42. Both were decided on 2026-10-31.
  - The P2 proposal says T-131 (b) "is not built (the triage batch)". Revision 41 built it (`slow_tick.py:67`).
  - `two_mac_sweep.md`'s status line stops at T-159 and leaves out the T-167 and T-170 runs.
  - The explainer is dated `2026-10-29` but carries revision 44's text of 2026-10-31 (:258, :609, :771, :2242).
  - README says Version 2 changed rules in "spec revisions 33–40". The range is 33–44.
- **Quoted:**
  - `_STATUS.md:3` — "Revision 43 (2026-10-29 to 30 …; Claude's recommendations under the owner's instruction to
    proceed, pending the owner's review)"
  - stage 3 plan :3 — "Open: the owner's answer on `M5.D.3` (T-158) and the owner's review of revisions 41 and 42's
    forms."
  - README :63–64 — "Version 2's stages 1–3 changed rules on top of these (spec revisions 33–40;"
- **How checked:** L3 read each line against the spec's status line and the dispositions. The orchestrator re-read
  each line at HEAD.
- **Fix:** restate each line. See systemic pattern 1 for the structural fix.

#### MR-11 — † The reading list's status line and its two new rows stop before revision 44

- **Found by:** L3 · **Where:** `docs/REVIEWER_READING_LIST.md:3–9`, `:189–190`; `docs/SPEC_REVIEW_CHECKLIST.md:13–15`
- **Severity:** minor · **Confidence:** high
- **What is wrong:** the status line says "last checked 2026-10-28" and omits the rows added from 2026-10-29 to
  2026-11-01. Rows 189–190 are in the future tense ("D4 and D5 go in spec revision 44"; "`M11.C.21` is decided again
  on the verdict revision 44 gives"). The checklist's label for the reading list repeats the stale range.
- **Quoted:** reading list :189 — "§8 holds step 7's six decisions (D1–D6) as Claude's recommendations"
- **How checked:** L3 read both ranges. The orchestrator read :189–190.
- **Fix:** restate both rows in the past tense with revision 44's outcome, and update the status line and the
  checklist label.

#### MR-12 — † The explainer says `M11.C.48[phase_d]` still runs on 100 seeds; both variants run 200

- **Found by:** L1 (out of lens, passed on); checked by the orchestrator · **Where:** `docs/model_explainer.md:2059–2061`
- **Severity:** minor · **Confidence:** high
- **What is wrong:** revision 43 (T-163) raised `M11.C.48`'s seeds to 200, and both variants read
  `SETTINGS["M11.C.48"]["seeds"]`. The explainer's example of a criterion refused by the sweep still names
  `M11.C.48[phase_d]` "on the same 100 seeds".
- **Quoted:**
  - explainer :2060–2061 — "as `M11.C.48` was until revision 41's engine and `M11.C.48[phase_d]`, on the / same 100
    seeds, still is"
  - `config/bowen/criteria.md:228` — "| M11.C.48 | seeds | 200 | [I] |"
  - `criteria_d2.py:484` — "seeds=SETTINGS["M11.C.48"]["seeds"]),"
- **How checked:** the orchestrator read all three.
- **Fix:** drop the example or restate it in the past tense ("as both variants were until revision 43").

#### MR-13 — † W4's "narrow" mark is not applied, and the deferral is not recorded where W4 is reported

- **Found by:** L3 · **Where:** `docs/version_2_plan.md:402`; `docs/version_2_status.md:32`; D14 at
  `docs/implementation_plan_version_2.md:845–849`
- **Severity:** minor · **Confidence:** high
- **What is wrong:** W4's definition marks a pass whose direction holds below a declared fraction of sweep samples as
  **narrow**. The status page reports W4 as "met" with no narrow mark. Three passes held in fewer than half their
  samples:
  - `M11.C.34`: 1 of 8;
  - `M11.C.2[twelve-person]`: 2 of 8;
  - `M11.C.7[joint]`: 3 of 8.

  D14 defers the fraction. Neither the plan's §4 nor the status page records that deferral, so a reader of either
  takes W4 as met in full.
- **Quoted:**
  - plan :402 — "with a pass below a declared fraction marked **narrow** (the fraction is the owner's, Q4)"
  - status :32 — "- W4 (every passing directional criterion has a sweep result): **met**."
- **How checked:** L3 read both and the sweep record (30–79). The orchestrator read :402 and :32.
- **Fix:** record D14's deferral beside W4 on the status page and in the plan's §4, or have the owner set the fraction
  before the tag (question 2).

#### MR-14 — † A record test checks nothing now that `VERDICT_NOTES` is empty

- **Found by:** L4 · **Where:** `tests/bowen/test_phase_d_record.py:186`; `tools/phase_d_record.py:138`
- **Severity:** minor · **Confidence:** high
- **What is wrong:** `test_m17a1_each_verdict_note_is_true_of_the_records_and_the_todo_list` loops over
  `VERDICT_NOTES`, which revision 43 emptied, so it asserts nothing. The test above it asserts that the dict is empty,
  so a note can come back only after that test is edited. This is the same class as the stage 3 review's MR-17
  (T-153).
- **Quoted:**
  - `phase_d_record.py:138` — "VERDICT_NOTES: dict[str, str] = {}"
  - test :186 — "for key, note in _tool.VERDICT_NOTES.items():"
  - test :164 — "assert _tool.VERDICT_NOTES == {}, "a note is back: check it against the records (the next test)""
- **How checked:** L4 grepped `VERDICT_NOTES` across `*.py`; nothing else fills it. The orchestrator read the three
  lines.
- **Fix:** run the check body on a patched dict that holds a known-bad note and assert that it fails, or delete the
  test until a note returns (question 7).

#### MR-15 — `group_g_readout.py` keeps no per-run results, so any change to its page reruns every simulation

- **Found by:** L4 · **Where:** `tools/group_g_readout.py:104–110`, `:180–183`
- **Severity:** minor · **Confidence:** medium (the cost was not measured)
- **What is wrong:** only aggregates are kept, in the page's measurements comment. A change to a renderer, a column or
  a label reruns 20 seeds per family, including Phase D at 2,080 weeks. CLAUDE.md's "Generated records" rule (P38)
  is meant to prevent this. T-166's new raw field did need a rerun; a change of wording will too.
- **Quoted:**
  - "results = measure(args.seeds, args.workers)"
  - "section = render_section(args.label, results, engine, args.seeds)"
- **How checked:** L4 read `one_run`, `measure`, `render_section` and `main`. The tool has no cache import, does not
  read results back, and `main` always calls `measure`.
- **Fix:** cache each `one_run` result in `docs/records_cache/` through `tools/record_cache`, keyed by engine hash and
  `(family, seed)`, and add a `--render-only` path.

## Systemic patterns

1. **Hand-restated status goes stale between steps (MR-2, MR-4, MR-10, MR-11, MR-13).** This is the stage 3 review's
   pattern 1 again. Its own fixes (MR-3, MR-4, MR-14 of that review) were made by editing the sentence, and the next
   revisions made the same sentences stale again. Revisions 41–44 and the W-row sign-offs came within four days.
   - `test_doc_status.py` reads only a status line's first word.
   - `test_stale_text.py` misses the possessive form (T-168).
   - `doc_claims.json` does not cover labels or revision ranges.

   The fix that pays is structural. A status line should point to one generated or test-checked source (the spec's
   status line, the revision log's header, the dispositions) and not restate it.
2. **A revision appends to a row and leaves its older claim standing (MR-7, MR-8, and T-171).** A criterion row or
   disposition text is extended with the new result, but the earlier measurement or "cannot bite" stays unmarked, and
   one tool quotes only the frozen version 1 text. The stage 3 review's T-003 (2) records the same thing for "not met"
   cells.
3. **A criterion's evidence and its class are not re-read together when its mutant changes (MR-3, MR-6, MR-7).** For
   `M11.C.34` and `M11.C.48`, the mutant or the arm changed at revision 43. The verdict was taken up again, but
   neither the spec row, nor `M11.5`'s class, nor the disposition's sentence was re-read against what the new result
   shows.
4. **Tests switched off by state (MR-14, and the stage 3 review's MR-16, MR-17).** Each passes and is counted, and
   none can fail for the reason its name gives.

## Refuted in challenge

None.
- The challenge pass confirmed MR-1 in full.
- It confirmed MR-2 and MR-3 in part:
  - "untracked" overstates MR-2;
  - "shown to rest on a premise" overstates MR-3, which is restated above.
- It confirmed MR-4 and downgraded it to minor.

## Checked and clean

- **L1 · changed `src/bowen/` lines: no rule writes an outcome.** The new rules are:
  - T-133: angry at `DEFINE` gives a demand share of 1.0 and the sequence ends;
  - T-146: a refusing target is not a receiver (`rejection.refuses`);
  - T-131: `displaced_swing` is left out of `M7.E.1f`'s parents' term;
  - T-145: the week's start is the end of the previous week;
  - T-165: no pull-up for an external mover;
  - T-161 (b): `arm_c21`'s scripted acts end at `t0`;
  - T-123: unmatched first cutoffs are left out.

  None sets conflict, over- or underfunctioning, distance, cutoff, projection or a fixed triangle.
- **L1 · numeric literals in the changed engine lines.** Only structural values appear: `1.0` (the all-demand end
  that `M5.D.4a` defines), `0.0` and `t0 < 1`. Decay reads `response_relaxation_rate`. L4 found the same.
- **L1 · `config/bowen/` changes.**
  - `constants_changes.md`: revisions 41, 43 and 44 changed no constant, which matches the diff.
  - `criteria.md`: the `M11.C.21` and `M11.C.32` notes match the code.
  - `FIND_WORK_LOAD` is "household, self reach" and is still `[I]`.
- **L1 · explainer source claims in the changed lines** (:279–286, :1033–1036, :2226–2242). `KS04.7`, `FE01.1`,
  `KS05.5`, `KS06.11`, `FE04.13` and `K12.3` are cited for direction only. Sizes are `[I]`. The interview is cited,
  not quoted.
- **L1 · spec `M11.F.2` (:1487).** `KB02`, `KB10` and `KB13` are paraphrased and each is cited as the interview.
  `K02.4`, `K10.8` and `K13.4` exist with matching titles.
- **L1 · `M11.5` and the changed `M11.C` rows.**
  - Their classes match the dispositions.
  - Every accepted finding is on a composite or mixed criterion.
  - `M11.C.49`'s premise limb is labelled as premise.
  - The failures of `M11.C.17` and `M11.C.21` are framed "in the model".
- **L1 · W8 framing (`M11.F.9`).**
  - Every page `REVIEW_version_2_outputs.md` names carries its framing: 9 files carry "Not a real family", and the
    trace pages carry "It does not concern any real family", which the orchestrator grepped in `phase_c_trace_seed7.md`
    and `phase_b_trace_seed7.md`.
  - `phase_c_completion_report.md:30` and `:522` carry it.
  - No output names a real person.
  - No setting was fitted to a known history.

  The four-page hand list in `test_version_2_outputs.py` is T-172 (c).
- **L1 · explainer :595.** "It passes at revision 41 … see below" is in revision 31's dated paragraph and points ahead
  to the revision 44 result. It is history, not a current claim.
- **L2 · `M5.D.4a`, `M5.F.4`, `M5.F.2c` (rev 41, T-133).**
  - Anger is read again at `DEFINE` (`act.py:132–136`).
  - `step_done` ends with `said_in_anger` and no balance restore (`iposition.py:307–308`).
  - Debit and evidence scale with `demand_share`.
  - The test covers both paths.
- **L2 · `M5.D.5` (rev 44, T-165).** `resolve` sets the gap to 0 for `PersonRole.EXTERNAL` (`iposition.py:273–276`).
  RESOLVE and FOLLOW_UP still run. "Payoff propagates" is not built, and that is known (T-169).
- **L2 · `M1.A.5f` (T-146, T-131).**
  - A refuser is left out of the split through the predicate `refuse()` uses.
  - A held-position receiver is counted but gives nothing, as the spec says. L4 noted that the two cases split the
    amount differently; that is the spec's rule, not a defect.
  - `displaced_swing` decays only in the calm branch, and only `life_course._childhood` reads it.
- **L2 · `M7.E.1f`, `M6.I.4a` (T-145), `M11.C.21`, `M11.C.48`.**
  - `week_start_levels` falls back correctly at tick 0, for a person not stored, and for a stale tick.
  - The watch reads the end of week `t0−1`.
  - `matched = bool(standing)`.
  - Seeds 200 = `detection_limit_seeds`.
  - `count` is imported from `engine.counting`.
- **L2 · mutation evidence for rows changed in revisions 43–44.**
  - The `M11.C.11`, `.15` and `.34` mutants ran and are red.
  - The `M11.C.17`, `.21`, `.48` and `.49` mutants are correctly listed as not run, because their criteria fail.
  - `coach-pull-up-removed` is retired.
  - No coverage is claimed "done" without a red mutant.
- **L2 · absolute thresholds.** None of the changed tests is an acceptance test with an absolute threshold. The
  ensemble criteria remain directional.
- **L2 · T-004's list of spec–code disagreements.** It has not grown. `M1.A.18` is struck through and six items
  remain. New disagreements went to separate items (T-158 is a strict xfail).
- **L2 · stage 3 triaged items in this lens.** MR-6, MR-7, MR-8, MR-9 and MR-22's wording are fixed at HEAD. MR-22
  waits only for the owner's approval; see MR-2.
- **L3 · the engine hash.** `09ca2b46c7e0` is the same everywhere and is enforced by `test_mr13…`.
- **L3 · counts and records.**
  - 585 requirements and 49 `M11.C` criteria.
  - The W1–W4 rows and their count (63) match the status page.
  - The sweep numbers in prose match the record.
  - `M11.C.17`'s 136 of 150 seeds matches.
  - The T-151 claims match `requirements.txt`.
  - `REVIEW_version_2_outputs.md` and its test (four pages, with an adversarial case) are consistent.
- **L4 · `M16.B`: the engine does no I/O.** The only file-related import is `json`, in `state.py` and
  `log_records.py`, for canonical serialisation. The new fields are written after the emitter loop, so logging remains
  a pure observer.
- **L4 · T-147 layering.** The engine never imports the policy. The test catches six import forms.
- **L4 · skips and xfails.** Every one has a reason. The T-158 xfail is strict and limited to `AssertionError`, and
  the two strict xfails match CLAUDE.md.
- **L4 · the new tests can fail.** Each new test L4 read has a negative or positive control. The patches use
  `monkeypatch` or a context manager. Mutant old-texts match exactly once.
- **L4 · requirements and CI.** The stage 3 review's MR-15 is fixed: `ruff==0.17.3` and `numpy>=1.26,<3`. The
  workflow installs only `requirements.txt` and runs no check of its own. CI on `355031f` is green on 3.11, 3.12 and
  3.13.
- **L4 · stage 3 items still open at HEAD, tracked and not re-reported.**
  - MR-16 (T-152): `test_stage_3_probes.py:83–84`.
  - MR-17 (T-153): `test_open_batch.py:61–63`.
  - MR-21 (T-157): `test_stage_3_probes.py:22`, `:69`, still unmarked. The new reason: the coping tool now calls
    itself historical, and its parents' term no longer matches the engine (`stage_3_displaced_coping_probe.py:193–194`).
- **Orchestrator · golden fingerprints and T-165.** The golden runs do not reach revision 44's branch (deviation 4).
  The unit test `test_m5d5_t165_…` does reach it.

## Unverified

1. **`mixing-weight-blind-to-level` edits one function that two rules read** (L2). `observe.self_channel_weight` feeds
   both `M4.D.1a`'s mixing weight and `M5.F.4`'s demand share. Whether `M11.C.34`'s red result comes from the mixing
   weight, as its row declares, was not separated. Check: run a one-off mutant that fixes the weight only at the
   policy's call site, on `M11.C.34` at 150 seeds.
2. **Whether `M11.C.34`'s pass survives a mutant that keeps `M4.D.1a` and removes the learned targeting** (L1, MR-3).
   This needs a new mutant (the policy's learner disabled, or blind to the bearer) in
   `tools/phase_d_mutation_record.py`.
3. **Whether `M11.C.48`'s reversal comes from comparing matched cutoffs with the ties that stayed uncut** (L1, MR-6).
   This needs a trace.
4. **`M11.C.48[phase_d]`'s "284 of 749 against 139 of 700"** (L3). Not checked against a record.
5. **The cost of the new status tests** (L4). About 13 new `tool.statuses()` calls each re-read every record and
   compute both engine hashes. Check: `pytest tests/test_version_2_status.py --durations=15 -q`.
6. **`group_g_readout.py`'s wall time** (MR-15). Run it in a scratch worktree, because it overwrites its page.
7. **The 11 unread two-word spans** from deviation 4 (MR-9).

## Questions only the owner can answer

1. **MR-22 (MR-2).** Revision 40's restatement of `M13`'s Phase C exit has been "pending the owner's review" since
   2026-10-28. Do you approve it before the tag?
2. **W4's narrow mark (MR-13).** Should W4 close under D14's deferral, with that recorded in the plan's §4 and on the
   status page? Or do you set the fraction now, which would mark `M11.C.34` (1 of 8), `M11.C.2[twelve-person]`
   (2 of 8) and `M11.C.7[joint]` (3 of 8) narrow?
3. **W5 and W8's sign-off.** These rows are "signed off by the owner 2026-11-01", and the instruction behind that was
   "go with your recommendations on the sign offs". Every "[x] enough" in `REVIEW_version_2_outputs.md` was filled on
   Claude's recommendation. W8 is defined as checked by the owner's review, not by code. Does that sign-off count as
   W8's review before the `version-2` tag?
4. **`M11.C.34`'s class (MR-3).** The only proving mutant removes `M4.D.1a`'s level-dependent weight, which `M11.5`
   calls the premise limb. Does this change the composite reading of 2026-10-17 or the "verdict reviewed" of
   2026-10-31?
5. **`M11.C.48` (MR-6).**
   - Should the accepted finding say that the model reverses the direction?
   - The code takes each seed's mean over its matched first cutoffs, and a seed with no matched cutoff enters as 0
     against 0 (`negative_runs.py:166–174`). Is that the intended reading of "averaged over the first cutoffs of 200
     seeds" and "contribute nothing"? (L2)
6. **`M5.D.3`.** Under T-158 (b) the MUST stands and the model fails it as a finding. Should spec `M5.D.3` say so, as
   the criterion rows do for their accepted findings? (L2; see MR-1)
7. **`VERDICT_NOTES` and the historical probe tools (MR-14, T-157).** Will verdict notes be used again? If not,
   should the check and its renderer path go? Should probe tools that call themselves historical still be run against
   the current engine in every suite run, or pinned to the engine hash stamped on their pages?
8. **SP-1's word limit (MR-9).** Tighten the test to two words, or accept four as a documented limit?
