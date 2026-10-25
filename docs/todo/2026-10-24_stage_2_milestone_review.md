# Version 2 stage 2 milestone review — the triage, 2026-10-24

The review is `docs/cycles/2026-10-24_milestone-review_v2-stage-2.md` (mini2, branch `review/2026-10-24-v2-stage-2`,
commit `34530cd`, fast-forwarded into `main` the same day). It found 0 blocker, 1 major and 27 minor findings. The
triage checked the 22 that were not challenged against their files with two agents (about 233k subagent tokens, about
6 minutes): none was dropped. Sixteen hold as stated and six in part:

- MR-8: the spec does not say `R` is "only ever raised"; it never mentions the reset on reopening, and the G1
  proposal §3.6 names entry to `DISTANT` but not the reopening case. The register row (spec `:1661`) omits it too.
- MR-14: spec `M1.F.6a` says "in the ratios of" Holmes and Rahe (1967), which is softer than the explainer's "from".
- MR-15: line 26's "later stages" can be read as stage 3 onward; lines 170 and 181–182 hold.
- MR-22: T-011 and T-012 hold. `live_positions.py:1`'s "four places" is not plainly stale: the spec writes the rule
  for four places, three are called, and the fourth (`avoidance_available`) waits on T-014's avoidance-gate question.
- MR-26: three items are new since the baseline, not four. `iposition.py:332` was there at the baseline; the
  report's "25" was the baseline's fixable count, and its total was 27.
- MR-27: holds; the report's blame commit IDs were not re-checked.

Two counts in the report are off: MR-28 affects four status rows (`M11.C.11`, `M11.C.15`, `M11.C.48`,
`M11.C.48[phase_d]`), not three, and its "plan `:699`" is `docs/implementation_plan_version_2.md:699`.

## The owner's decisions, 2026-10-24

Each was put through `AskUserQuestion`, one decision at a time, and each answer was the recommended option.

- **Q1, MR-7.** The code stands: the stock decays every week the tie is cut off. `M1.D.2b` is amended (revision 38).
- **Q1, MR-8.** The code stands: `R` is reset to the stock when a tie enters `DISTANT` or `OPEN_CONFLICT`, including a
  cut tie reopening into one. `M1.D.2c` and the register row are amended (revision 38).
- **Q2, MR-1.** The named ratio is kept. `M17.E.1` is marked *partial* now; stage 3's plan proposes the two rates
  (named in `M10`) and their range for the owner's decision, before stage 3's full sweep (T-128).
- **Q3, short KB wording.** A single word is not quotation; a phrase of two or more words is. "Shout them down" in
  the explainer is paraphrased, "grotesque" stays, and the rule is recorded in `docs/theory/_SOURCE_QUALITY.md`.
- **Q4, MR-21.** Part II needs no cross-check of its own: the check is of two machines' agreement at one engine hash,
  which the two 2026-10-23 checks showed at the hash part II ran on. `docs/two_mac_sweep.md` says so.
- **Q5, `M7.D.1a`.** Every illness onset keeps the "personal injury or illness" size, `[I]`; the question goes to
  T-014's stage 3 list.
- **Q6, `M11.C.48`.** Its within-run matched comparison is listed in `M11.4`'s exceptions table (revision 38).
- **Q7, historical guides.** Left as records of their time; `docs/REVIEW_version_1_guide.md:101` is not restated.
- **Batch scope.** Every fix that leaves the engine hash alone is made now. Edits to `src/bowen/*.py` or
  `config/bowen/*.md` move the engine hash, which stales every record cache and the stage 2 sweep, so they wait for
  the next change that moves it (T-129, T-127's trigger).
- **SP-A and SP-B.** SP-B's gate test is built now. SP-A's cases are added to T-015, which stays *later*.
- **MR-12** (asked after the tool fix showed the report's fix would drop `M11.C.48` and `M11.C.48[phase_d]`, which
  fix 100 seeds against the sweep's rerun count of 200, from the swept record). The refusal comes when a sample needs
  the rerun, before it runs; such a criterion is swept while no sample needs one. `M17.A.1` says so (revision 38).
- **MR-26.** The items under `tools/` and `tests/` are fixed now; the `src/bowen` items and a ruff test wait (T-129).

## Disposition of each finding

| ID | Disposition | What |
|---|---|---|
| MR-1 | fix now, and T-128 | `M17.E.1` *partial* (override); the ratio itself is T-128 |
| MR-2 | fix now in part; rest T-129 | explainer §19 now; `moves.py`'s module docstring with the next engine-hash change |
| MR-3 | fix now | the two records' renderers print the T-123/T-119 note beside `M11.C.48` |
| MR-4 | fix now | coverage rule 2 no longer counts an expected failure; `M5.B.3` *partial* |
| MR-5 | fix now | reading list status line and checklist label |
| MR-6 | fix now | checklist label for the model-ergonomics plan |
| MR-7 | fix now (owner: code stands) | `M1.D.2b` wording, revision 38 |
| MR-8 | fix now (owner: code stands) | `M1.D.2c` and the `reference_level` register row, revision 38 |
| MR-9 | fix now in part; rest T-129 | spec `M11.C.48` row and `group_g_readout.py`'s test now; `rejection.py:62` with the next engine-hash change |
| MR-10 | fix now | `M1.B.3`'s override note re-derived; `OPEN_CONFLICT` taken off T-004 |
| MR-11 | fix now | register "Move physics" row gains `CONFLICT`, `M1.D.2c` and the `reference_level` write, revision 38 |
| MR-12 | fix now (owner: refuse at the rerun) | the sweep refuses a short-seeded criterion before running a rerun it needs, not after; `M17.A.1`, revision 38 |
| MR-13 | fix now in part; rest T-129 | explainer `:622`, `:628` now; four `src/bowen` docstrings with the next engine-hash change |
| MR-14 | fix now, and T-130 | explainer and spec cite the values as the scale's published table, unchecked; checking them against the paper is T-130 |
| MR-15 | fix now | reading list body |
| MR-16 | fix now | "4 to 38" everywhere; the log's revision 36 and 37 tags; the Holmes–Rahe proposal's "waits on G2" |
| MR-17 | fix now | `_STATUS.md` STATE block and the plan's status line |
| MR-18 | fix now | README points at `version_2_status.md` and lists the version 2 generators |
| MR-19 | fix now; the test is T-015 | the explainer's date; a date test needs a rule for what a front-matter date means |
| MR-20 | fix now | CLAUDE.md's count wording and the regex |
| MR-21 | fix now (owner Q4) | `two_mac_sweep.md`'s "Not verified" |
| MR-22 | fix now | T-011's and T-012's triggers and `version_2_plan.md` 1.8 re-pointed at the next engine-hash change |
| MR-23 | fix now | broken-table tests for both loaders |
| MR-24 | fix now | the arm checks leave the expected failure |
| MR-25 | fix now | `--recompute`'s put-back in `try/finally` |
| MR-26 | fix now in part; rest T-129 | `tools/` and `tests/` now; `src/bowen` and a ruff test with the next engine-hash change |
| MR-27 | fix now | `M10.C.2`, `M10.A.2`, `M10.A.1`, revision 38 |
| MR-28 | fix now | the status page tells a surviving named mutant from no mutation |

None was rejected.

## Open

- [ ] T-128 ⟨later⟩ **`M17.E.1`'s named ratio, before stage 3's full sweep** (owner, Q2, 2026-10-24). Stage 3's plan
  proposes which two `[I]` rates make the ratio — the rate ties harden or attenuate (candidates: `balance_harden_rate`,
  `M1.B.6`; `tie_wear_rate`, `M7.E.4`) against the rate anxiety moves (candidates: `calm_transfer_rate`, `M4.C.10`;
  `usual_level_rate`, `M1.A.5e`) — names it in `M10`, declares its range in `config/sweep/sweep.md`, and asks the
  owner. `M17.E.1` stays *partial* in `tools/spec_coverage_overrides.json` until the ratio is swept (review MR-1).
  *Decided by the owner 2026-10-24, each as recommended, in stage 3's plan (`docs/implementation_plan_version_2_stage_3.md`
  §3.4, D24–D26): `tie_wear_rate` ÷ (52 × `calm_transfer_rate`), a quarter to four times its central value, on every
  directional criterion whose run spans a slow tick. Open until the stage-end sweep runs with it.*
- [ ] T-129 ⟨later⟩ **With the next change that moves the engine hash** (owner, 2026-10-24: these edit
  `src/bowen/` and would stale every record cache and the stage 2 sweep):
  - MR-2: `src/bowen/engine/moves.py`'s module docstring (`:63-68`) still says `REDUCE_CUTOFF` hands the cutoff's
    anxiety to third parties; restate it as rejection lowered, no stock moved, the third-party cost not yet shown.
  - MR-9: `src/bowen/engine/rejection.py:62` names `test_rejection.py::test_m1b3c_faded_cutoff_waits_for_a_positive_contact`;
    the test is `tests/bowen/test_cutoff_stock.py::test_m1d2b_faded_cutoff_waits_for_a_positive_contact`. Empty the
    `src/bowen` allowlist of SP-B's gate test as each name is fixed.
  - MR-13: `objects.py:247-248`, `:277`, `rejection.py:10` and `invariants.py:26` describe the cutoff ending when
    rejection reaches zero; restate as "the tie leaves `CUT_OFF` only on a positive contact with no rejection held
    (`M1.B.3c`)". `invariants.py:26`'s "exactly" is wrong: the invariant checks one direction.
  - MR-26: ruff F items in `src/bowen` (`state.py:118`'s undefined `Event`, to import under `TYPE_CHECKING`;
    `policy.py:232`'s dead `weights`; `iposition.py`'s two `math` imports; the unused imports in `observe.py`,
    `stress_response.py`, `criteria.py`, `criteria_d.py`), and `tools/level_occupancy.py:139`'s unused `code_hash`
    import (that tool is in its own record's hash), then a test that runs `ruff check --select F` over `src/`,
    `tools/` and `tests/`.
- [ ] T-130 ⟨later⟩ **Check the event sizes against Holmes and Rahe (1967)** (review MR-14). `config/bowen/event_sizes.md`
  says the values were read from the scale's table as Wikipedia gives it, unchecked against the paper. Check them
  against the paper's table, then drop "unchecked" from the config, the spec (`M1.F.6a`) and the explainer, or
  correct the values (that moves the engine hash).
  2026-10-24 (stage 3 step 1): not checked. The paper could not be read (ScienceDirect 403, no copy of the article
  found, only reproductions); `docs/review/stage_3_t130_event_sizes.md` records the attempts and the thirteen values to
  check. The owner will supply the PDF.
  2026-10-24, later: checked against the owner's scan of the article (Table 3, p. 216). All thirteen values agree; the
  paper's wording differs for `JOB_LOSS` ("Fired at work") and `FIND_WORK_LOAD` ("Wife begin or stop work"). Dropping
  "unchecked", the wording, and this tick wait for stage 3's step 6 (plan §3.6).
</content>
</invoke>
