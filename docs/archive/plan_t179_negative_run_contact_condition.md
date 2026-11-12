# Plan: `M11.C.48` reads both arms under the same contact condition (T-179)

**Status: built 2026-11-03 as spec revision 47 (engine hash `0b363cb3186c`); the status report is §3.1. Approved by the
owner as written, 2026-11-02, with §5's Q1 as recommended (the status page stays live). Step 7 was changed by the owner
on 2026-11-03: only `M11.C.48`'s two keys were swept, into the cache, and the sweep record waited for the full sweep
(T-181), done 2026-11-04. Written 2026-11-02 on `main` at `033f4fe` (spec revision 46, engine hash `09ca2b46c7e0`).** The owner chose option (a) through `AskUserQuestion` on 2026-11-02: leave out a first
cutoff on a tie with no contact by the cutoff's week, and count it, as unmatched cutoffs are counted. Options (b)
(count uncut ties with no contact at a run of 0) and (c) (keep the readout) were not chosen.

## 1. Why

`first_cutoff_runs` (`src/bowen/readouts/negative_runs.py`) counts an uncut tie in the baseline only if it had a
contact by the cutoff's week, but counts a cut tie in the treatment whether or not it had one, and a cut tie with no
contact enters with a run of 0. T-176's trace (`docs/review/t176_negative_run_trace.md`, "Reading") found that this
produces the criterion's reversal: 329 of 1137 matched cutoffs on the Phase B family and 1419 of 4791 on the
twelve-person family stand on a tie with no contact. With the same condition on both arms the difference is
+0.029 ± 0.136 (Phase B family) and +0.344 ± 0.078 (twelve-person family), against the record's −0.319 ± 0.104 and
−0.113 ± 0.062. The readout is a comparison with unequal conditions on its two arms (learnings P7), the same class as
T-123's unmatched cutoff.

## 2. The change

One condition, applied on both sides by one function: **a tie is in contact at week `w` if it has a contact in any
week `≤ w`** (the uncut side's present test, `any(w <= week for w, _ in runs[t])`). In `first_cutoff_runs`:

- A first cutoff whose tie is not in contact at the cutoff's week is left out of both arms and counted as a
  **no-contact cutoff** (a new field on `FirstCutoff`, read by a new `no_contact(firsts)` beside `unmatched`).
- Each first cutoff is counted in exactly one of: matched, unmatched (in contact, no uncut tie in contact that
  week), no-contact. The no-contact test comes first, so a cutoff that is both no-contact and without an uncut tie is
  counted as no-contact. `unmatched` keeps its meaning for cuts with a contact.
- `arm_c48` (`src/bowen/ensemble/criteria_d2.py`) reports `no_contact_cutoffs` (report only, the treatment's count,
  0 in the baseline), as `unmatched_cutoffs` is reported. `negative_run_values` returns it.

What is not changed: uncut ties with no contact stay out of the baseline and are not counted. Counting them needs the
week from which each tie existed (people are born in the twelve-person family's run), which is option (b) and was not
chosen. So `docs/version_2_plan.md` G1's "reported, not dropped" is met for cut ties (counted) and not for uncut ties;
`docs/readouts_phase_c.md` says so.

**T-177 rides in the same batch** (its record says "with the next batch that moves the hash"): the sentence in
`config/bowen/constants.md:56–57` that since revision 41 aborts are not the usual outcome (T-158 (b)). Both changes move
the engine hash, so one regeneration covers both.

## 3. Acceptance criteria and the test for each

Each new test is written first, run red against the present readout, then the code is changed. A test counts only
once a mutant of the code it covers turns it red.

| ID | Criterion | Test, or the artifact that shows it | Mutation that must turn it red |
|---|---|---|---|
| T179.1 | A first cutoff on a tie with no contact by its week is left out of both arms and counted in `no_contact` | `tests/bowen/test_negative_runs.py::test_m11c48_t179_a_cut_tie_with_no_contact_is_left_out_of_both_arms_and_counted` (a~b cut at week 2 with no contact; c~d cut at week 6 after four conflicts in weeks 3–6; e~f uncut, a conflict in weeks 1 and 4; arms `(4.0, 2.0)`, `no_contact == 1`) | drop the contact condition on the cut tie (the present readout): arms read `(2.0, 1.5)` |
| T179.2 | Every first cutoff is counted once: matched + unmatched + no-contact = all first cutoffs, and a cutoff both without contact and without an uncut tie counts as no-contact | `tests/bowen/test_negative_runs.py::test_m11c48_t179_each_first_cutoff_is_counted_once` | test the uncut set before the contact condition: the cutoff counts as unmatched |
| T179.3 | Both sides use one contact condition: a contact in the cutoff's own week counts, on the cut tie as on the uncut ties | `tests/bowen/test_negative_runs.py::test_m11c48_t179_a_contact_in_the_cutoffs_week_counts_on_both_sides` | `w < week` on the cut tie only |
| T179.4 | `arm_c48` reports `no_contact_cutoffs` on the treatment arm and 0 on the baseline; `negative_run` is unchanged in meaning | extend `tests/bowen/test_phase_d2_criteria.py::test_m11c48_reads_the_run_before_the_first_cutoff` | report the count on both arms |
| T179.5 | The new readout reproduces the trace's same-condition row: on seed 0 of each family the criterion's `run − uncut` is the trace's seed mean of "cut ties with a contact" | `tests/test_t176_negative_run_trace.py::test_m11c48_t176_trace_reproduces_the_criterions_arms`, restated: the criterion's row is now the same-condition row, and the old readout's row is kept in the trace, labelled as revision 46's | restore the old condition in the readout |
| T179.6 | The regenerated record's `negative_run` equals the trace's same-condition row over the 200 seeds (+0.029 ± 0.136, +0.344 ± 0.078, from the T-176 run at `09ca2b46c7e0`), unless the T-177 note changed the run (it cannot: prose) | artifact: `docs/phase_d2_ensemble_record.md`'s two `M11.C.48` rows against `docs/review/t176_negative_run_trace.md` regenerated on the new hash; the commit message quotes both | — (a check on the run, not a test) |
| T179.7 | If `M11.C.48[phase_d]` passes, the mutant `cutoff-blind-to-learned-value` turns it red (the spec row's **MUST**) | artifact: `docs/phase_d_mutation_record.md`'s row for that mutant on `M11.C.48[phase_d]`. If it survives, that is reported as a finding, not hidden (as T-119 was) | — |
| T179.8 | Spec revision 47: `M11.C.48`'s row states the contact condition on both arms and the `no_contact_cutoffs` count; the revision log has the entry | `docs/bowen_agent_model_spec_v2.md`, `docs/spec_revision_log.md`; the generated-index tests and `tests/test_doc_claims.py` (the revision number in each place it is restated) | — |
| T179.9 | `docs/readouts_phase_c.md` "Negative runs" states the condition, the new count, and that uncut ties with no contact are not counted | the section; `tests/test_doc_claims.py` if a claim is added to `tests/doc_claims.json` | — |
| T179.10 | The two disposition rows and the status page follow the new verdicts | `docs/version_2_dispositions.md` (rows restated after the owner decides them, §5), `docs/version_2_status.md` regenerated; `tests/test_version_2_status.py` | — |
| T179.11 | T-177's sentence is in `config/bowen/constants.md` | the line; T-177 ticked | — |
| T179.12 | The suite: the new tests pass, the count in `CLAUDE.md` is updated, the strict expected failures stay two | `python3 -m pytest tests/` exit line; `tests/test_doc_claims.py` (the suite count) | — |

T179.6 and T179.7 are checks on generated records, not code tests: a test cannot run 200 seeds in the suite. The
commit that lands them quotes the numbers, and the reviewer checks them against the records.

### 3.1 Status report (2026-11-03)

| ID | Status | Evidence |
|---|---|---|
| T179.1 | done | `tests/bowen/test_negative_runs.py::test_m11c48_t179_a_cut_tie_with_no_contact_is_left_out_of_both_arms_and_counted`; red under the mutant |
| T179.2 | done | `tests/bowen/test_negative_runs.py::test_m11c48_t179_each_first_cutoff_is_counted_once`; red under the mutant |
| T179.3 | done | `tests/bowen/test_negative_runs.py::test_m11c48_t179_a_contact_in_the_cutoffs_week_counts_on_both_sides`; red under the mutant |
| T179.4 | done | `tests/bowen/test_phase_d2_criteria.py::test_m11c48_reads_the_run_before_the_first_cutoff` (extended) and `::test_m11c48_t179_a_seed_whose_cutoffs_are_all_left_out_reads_0_and_is_counted` (`matched_seed`, added after learning-qa); each red under its mutant |
| T179.5 | done | `tests/test_t176_negative_run_trace.py::test_m11c48_t176_trace_reproduces_the_criterions_arms`; red with revision 46's condition restored |
| T179.6 | done | `docs/phase_d2_ensemble_record.md`: `negative_run` +0.0286 ± 0.14 (Phase B family) and +0.344 ± 0.078 (twelve-person family); `docs/review/t176_negative_run_trace.md` at `0b363cb3186c`: +0.029 ± 0.136 and +0.344 ± 0.078, the same as its same-condition row at `09ca2b46c7e0` |
| T179.7 | done, a finding | `docs/phase_d_mutation_record.md`: `cutoff-blind-to-learned-value` **survived** on `M11.C.48[phase_d]` (+0.229 ± 0.054); reported, not hidden: the disposition row, the spec row's mutation clause and T-180 |
| T179.8 | done | `docs/bowen_agent_model_spec_v2.md` (status line, `M11.C.48`'s row), `docs/spec_revision_log.md` "Revision 47"; `tests/test_doc_claims.py::test_d1_spec_revision_restatements_match_the_front_matter` |
| T179.9 | done | `docs/readouts_phase_c.md`, "Negative runs" (the revision 47 bullet) |
| T179.10 | done | `docs/version_2_dispositions.md` (both rows, owner 2026-11-03); `docs/version_2_status.md` regenerated; `tests/test_version_2_status.py::test_s2_11_status_page_is_current` |
| T179.11 | done | `config/bowen/constants.md`, the sentence after the `hold_gain` note; T-177 ticked |
| T179.12 | done | `python3 -m pytest tests/`: 1300 tests, 1297 pass, 3 strict expected failures (the third is the sweep record, T-181, the owner's choice of 2026-11-03); `CLAUDE.md` restates it |

Also built beyond the table: `tools/group_g_readout.py` counts no-contact cutoffs in their own column
(`tests/test_group_g_readout.py::test_m11c48_t179_readout_counts_no_contact_first_cutoffs_apart_from_unmatched_ones`),
and `docs/version_2_revision_47_changes.md` (`tools/record_changes.py revision-47`).

## 4. Order of work and what it costs

| Step | What | Where | Cost |
|---|---|---|---|
| 1 | Tests T179.1–T179.4 written, run red | here | minutes |
| 2 | The readout and `arm_c48` changed; T-177's sentence; tests green; each test's mutant run by hand | here | under an hour |
| 3 | The trace tool updated (T179.5): its criterion row is the same-condition row, revision 46's readout kept as a labelled row | here | under an hour |
| 4 | Spec revision 47, `docs/readouts_phase_c.md`, `docs/model_explainer.md` (its `M11.C.48` paragraph, about line 1429), `config/bowen/constants_changes.md` | here | under an hour |
| 5 | One regeneration of every record on the new hash | split: this Mac `nice -n 15`, ≤ 8 workers; Mac Mini1 10 workers | about 4 h wall (§4.1) |
| 6 | The trace rerun on the new hash; its "Reading" restated by hand | this Mac, 8 workers | about 20 min |
| 7 | The `M11.C.48` sweep rows rerun: `tools/version_2_sweep.py --criteria M11.C.48 M11.C.48[phase_d] --recompute` | split across both Macs | measured from the cache before it starts, and stated to the owner then |
| 8 | `tools/record_changes.py` (a `docs/version_2_revision_47_changes.md` page), the generated indexes (`todo_index`, `spec_index`, `spec_rules_view`, `spec_coverage` with its two arguments, `version_2_status`), `tools/c16_report.py` | here | minutes |
| 9 | The owner decides the two disposition rows on the new verdicts (`AskUserQuestion`) | here | — |
| 10 | One `learning-qa` pass and one correctness pass over the batch's diff; fixes; commit to `main`, push | here | about 30 min |

### 4.1 The regeneration

As at stage 3 step 10 (`CHANGELOG.md`, 2026-10-26), with times from that run. On this Mac (`nice -n 15`, ≤ 8 workers):
the Phase C ensemble record (5 min), the mutation record (33 min), the sweep record (8 min), the diagnostic record
(8 min), level occupancy (1 min), the group G readout's "after" section, Phase D part I (16 min), then the seed-7 trace
and fam-diag export. On Mac Mini1 (`ssh macmini1`, 10 workers), at the same time: Phase D part II (95 min at stage 3;
`M11.C.48` at 200 seeds is in it and roughly doubles its own share) and the Phase D mutation record (93 min at six
workers), copied back and their `code_hash` checked. About 4 hours of wall time with the two Macs in parallel.

The full constant sweep is not run: it runs at a stage's end (plan §2.10, D17), and between, a change is swept only on
the criteria it targets (step 7).

## 5. What the owner decides, and when

- **Now: this plan.** Approve, change, or reject before step 1.
- **Q1, now: the status page after the tag.** `docs/version_2_status.md` is generated from the current records; it is
  not frozen at the tag `version-2` as `docs/version_1_status.md` is at `version-1`. This batch is the first after the
  tag that moves a verdict. Recommendation: leave the page live, and have its header name the tag and the revisions
  since it (the tag stays at `8c4e0bf`). The other choice is to freeze it at the tag and start a version 3 page, which
  waits on version 3's scope. **Decided 2026-11-02: as recommended** (the owner approved the plan as written).
- **After step 8: the two disposition rows.** On the Phase B family the same-condition difference holds zero inside
  its interval, so `M11.C.48` is expected to stay FAIL with a new reason (no difference, not a reversal); on the
  twelve-person family it is expected to pass. Each row is put to the owner with its record numbers.

## 6. Documents the batch updates

`docs/bowen_agent_model_spec_v2.md` (revision 47, `M11.C.48`'s row and the status line), `docs/spec_revision_log.md`,
`docs/readouts_phase_c.md` ("Negative runs"), `docs/model_explainer.md` (the `M11.C.48` paragraph),
`config/bowen/constants.md` (T-177), `config/bowen/constants_changes.md`, every regenerated record, the trace page and
its "Reading", `docs/version_2_dispositions.md`, `docs/version_2_status.md`, `docs/version_2_revision_47_changes.md`
(new) and its checklist row, `docs/SPEC_REVIEW_CHECKLIST.md`, `docs/REVIEWER_READING_LIST.md`, the status lines that
restate the spec revision or the engine hash (`CLAUDE.md`, `docs/theory/_STATUS.md`, README where the doc-claim tests
check them), T-177's and T-179's record, `TODO.md`, `CHANGELOG.md`, and this plan's status line.

## 7. Adjacent issues found, not fixed

- `docs/version_2_plan.md` G1's "reported, not dropped" for uncut ties with no contact stays unmet (§2); it needs each
  tie's start week (option (b)).
- The trace's "By declared relation" table pools every matched cutoff, the ones with no contact among them, so it
  carries the bias this plan removes; step 6's rerun reads it on the new readout.
