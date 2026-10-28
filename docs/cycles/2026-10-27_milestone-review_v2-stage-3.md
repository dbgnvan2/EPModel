# Milestone review: version 2 stage 3 (mini2, 2026-10-27)

The third milestone critical review of EPModel, run as `docs/milestone_review_brief.md` describes. It reviews the
whole model against the brief's four lenses, theory fidelity first, with the changes since the stage 2 review in
focus. The reviewing session edited nothing except this report and fixed nothing.

## Header

- **Commit reviewed:** `161da4e83d56c0455bc6a1dec7fc8ea8f95a97c0` (`origin/main`, "Stage 3 step 12: the reviews of
  stage 3's whole diff"), in a worktree on branch `review/2026-10-28-v2-stage-3`.
- **Baseline:** `d51b2508a4751791e66112d4f79a9835cc410c19`, the commit the stage 2 review covered
  (`docs/cycles/2026-10-24_milestone-review_v2-stage-2.md`). 23 commits.
  - The brief's §3 step 5 filter lists 145 files (+12,285 / −1,308).
  - The extended filter below leaves 113 files (+7,471 / −555), which were split into the four lens slices:
    `src/bowen/` 21, `tests/` 42, `tools/` 21, `config/` 9, `docs/` 15, other 5 (`CLAUDE.md`, `README.md`,
    `TODO.md`, `pytest.ini`, `requirements.txt`).
- **Machine:** Mac mini2 (`mini2.local`, Darwin 27.0.0).
- **Suite:** `1258 passed, 52 deselected, 1 xfailed in 351.91s`, exit 0, on Python 3.12.8. This matches CLAUDE.md
  (1259 tests: 1258 pass, 1 strict expected failure; 52 `ensemble` tests deselected).
- **CI for the commit:** run `40630271972` (push to `main`, head `161da4e`), `completed success` in 12m38s:
  `test (3.11)` success, `test (3.12)` success, `test (3.13)` success.
- **Tokens used:**
  - Subagents: 392,803, taken from each result's usage total (L1 48,782; L2 44,245; L3 185,566; L4 46,396;
    challenge 67,814).
  - The orchestrating session: not measured as a total. The app reports only the current context window, and the
    session was compacted once before consolidation, so no cumulative figure is available.
- **Wall time:** 12.8 minutes from the start of §3 step 3 (22:22:46 local) to the challenge pass's return
  (about 22:35:36: its output file was created at 22:33:12 and it ran 144.2 s), plus the time to write this file.
  - The worktree, the virtualenv and the cost statement came before the clock started.
  - The four lens workers ran in parallel: L1 8.2, L2 7.8, L3 6.2 and L4 4.7 minutes. The challenge pass took
    2.4 minutes.
  - Expected cost, stated before starting from the stage 2 header: 426,430 subagent tokens and 17.5 minutes.
- **Findings:** 0 blocker, 1 major, 21 minor (22). One major (MR-2) was downgraded to minor in challenge; none was refuted. 0 blocker, 2 major and 20 minor before the challenge.

**Deviations from the brief.**

1. **Machine and virtualenv.** The prompt named `~/ProjectsMini1/EPModel/.venv`, which is on Mac Mini1; this review
   ran on mini2. Per the prompt, the suite ran in a fresh virtualenv built from `requirements.txt` with Homebrew's
   Python 3.12.8 (a version in CI's matrix), in the session's scratchpad, outside the repo: pytest 8.4.2, numpy 2.5.4,
   ruff 0.17.3.
2. **Worktree.** The worktree `EPModel-review-2026-10-28` and its branch `review/2026-10-28-v2-stage-3` existed
   before this session, clean, at `161da4e`, with no upstream. The review used it as found. Its name carries
   2026-10-28; the review ran on 2026-10-27, and this report is dated by the day it ran.
3. **Exclusions beyond the brief's filter** (the stage 2 report's set, plus one new generated page):
   - generated pages: `docs/spec_rules/`, `docs/spec_index.jsonl`, `docs/theory/ledger_index.jsonl`,
     `docs/version_2_status.md`, `docs/version_2_revision_36_changes.md`, `docs/version_2_revision_39_changes.md`
     (written by `tools/record_changes.py`), `docs/TODO_archive.md`;
   - dated logs: `docs/CHANGELOG_archive.md`, `docs/todo/`;
   - `docs/cycles/`, which holds the previous reports.

   Workers read `TODO.md`'s index line and then only the record it names, and searched the generated pages for a
   claim being checked.
4. **Spec read.** The orchestrator read the spec end to end (lines 1–2140) while the workers and the suite ran,
   rather than before fanning out; the M11.C table rows were read about 20 at a time. `docs/spec_revision_log.md` was
   read at its headings only.
5. **Checks the orchestrator ran.** Workers are read-only, so two of their "unverified" items were run here:
   - `tools/mutation_gate.py tools/mutation_gate_stage_3.json` at `161da4e`, writing its record to the scratchpad,
     not to `docs/review/`: **25 of 25 gates proved**, exit 0, "working tree unchanged by the run". The committed
     record says it ran at `be35487` with uncommitted changes; it reproduces on the committed tree.
   - `pytest tests/test_stage_3_probes.py --durations=0`: 8 passed in 60.9 s (MR-21).

## Coverage matrix

"Full" = read in full; "diff" = every changed hunk read with its enclosing section or function; "sampled" = named
ranges read; "searched" = grepped for the claim being checked; "—" = not in that lens.

| Area | L1 theory fidelity | L2 spec against code | L3 documents | L4 engineering drift |
|---|---|---|---|---|
| Spec `bowen_agent_model_spec_v2.md` (2,140 lines) | M10 (1169–1260) full; 1353–1375, 1384–1429; 205–207, 716, 774, 885–894, 941–958, 966–974, 1097–1105 | every revision 38–39 hunk; 2136 | front matter; requirement count against `spec_index.jsonl` | M16.B (1949–1968) full |
| The orchestrator's read of the spec | lines 1–2140 full | — | — | — |
| `src/bowen/` engine | diff of every changed file; `iposition.py` 25–114, 185–264; `life_course.py` 292–302, 655–676 | diff of `stress_response`, `moves`, `appraise`, `iposition`, `act`, `live_positions`, `events`, `objects`, `params`, `slow_tick`, `tick`; `invariants`, `observe`, `rejection`, `state` through the diff only | searched (`M1.A.5f`, `avoidance_available`, file I/O) | every changed hunk; I/O calls grepped |
| `src/bowen/` ensemble, io, render, scenario, policy | `criteria.py` 595–639; diff of `criteria`, `criteria_d`, `load`, `policy` | diff of `criteria.py` (613–728, 852–874), `io/load.py`, `render/trace.py`, `scenario/constants.py`; `criteria_d`, `policy` through the diff | searched | diff of each |
| `config/bowen/` | changed hunks full; `event_sizes.md` full | `criteria.md` 160–164; `stress_response.md`; `event_kinds.md` 59–61 | `constants.md`, `stress_response.md` searched | — |
| `config/sweep/sweep.md` | changed lines (named ratio) | — | row count, `## Named ratios` | — |
| `docs/model_explainer.md` (2,241 lines) | every changed range; elsewhere searched | — | status line and date | — |
| `docs/theory/` | `_SOURCE_QUALITY.md` full; ks00, ks04, ks05, ks06, ks15, fe01, fe04, fe05, fe06, kb12 and `_LEDGER.md` at cited lines | — | `_STATUS.md` 1–40 | — |
| Reading list, checklist, README, CLAUDE.md | README diff hunk | — | full | — |
| `docs/spec_coverage.md` + overrides | — | searched by ID; changed overrides | — | — |
| Generated records (`docs/*_record.md`), `phase_c_completion_report.md` | report 1–12, 405–474, 536–543 | Phase C mutation record 1–40, 95–180; Phase D mutation record 1–55; Phase C ensemble record searched | report 410–492; record hash lines; sweep record header and rows counted | — |
| `docs/review/` stage 3 pages | step 7 and 8 full; step 9 1–1406; T-130 page 50–63; probe pages' disclaimers only | `stage_3_unit_mutation_gate.md` full | status lines | — |
| Proposals, plan, revision 39 draft | P2 proposal 1–70, 135–269; draft 231–267, 557–594 | targeted greps | status lines of every changed one; stage 3 plan 1–30 | — |
| `tests/` | — | `test_stress_response` 257–413; `test_iposition` 110–157, 470–521; `test_phase_c_criteria` 1–60, 370–454; `test_phase_d2_criteria` 105–139; diffs of `test_cutoff_stock`, `test_phase_d_record` | `test_doc_claims` 1–75; `test_spec_consistency` 255–394; `doc_claims.json` full | the 10 new files full (incl. `test_open_batch`, `test_stage_3_probes`, `test_named_tests_resolve`, `test_version_2_sweep`); `test_lint` full; changed hunks of the rest |
| `tools/`, CI, `pytest.ini`, `requirements.txt` | `c16_report.py` 90–144 | `sweep_config.py` 200–284 | `c16_report.py` 40–160 | the five new tools full (`stage_3_*`, `sweep_config.py`); changed hunks of the rest; `tests.yml`, `pytest.ini`, `requirements.txt` full |
| `TODO.md` and the records it names | T-131–T-133 record | `TODO.md` full; T-004 record 30–59; stage 3 record searched | full index | searched (known items not re-reported) |

Not reviewed, by any lens:

- the bodies of `tests/bowen/test_moves.py` 574–738 and `test_appraise.py` 152–249 (L2 relied on the unit mutation
  gate, which the orchestrator reran);
- `docs/review/group_g_readout.md`; the bodies of the stage 3 C.2-mutants, projection-trace and i-position probe pages;
- `docs/readouts_phase_c.md` beyond a search (unchanged in this range);
- `CHANGELOG.md`, `docs/version_1_status.md`;
- `docs/agent_model_proposal.html`'s one hunk (a requirement count).

## Owner items for this review's L1: T-131, T-132, T-133

The owner left three items for theory fidelity (`docs/todo/2026-10-25_stage_3_revision_39.md:6–24`). L1's verdicts,
with the source lines it read:

- **T-131 — `KS00.9` against `M7.E.1f`'s parents' term. The finding stands, and the proposal mis-cites `KS00.9`.**
  - `KS00.9` (`docs/theory/kerr_book/ks00.md:92`) says the parents' **basic** levels set the emotional climate. In the
    model that is the anchor `a`, the parents' mean basic level (spec :1103). The parents' term `d` (half the parents'
    functioning above basic) is the owner's own `[I]` rule of 2026-10-13; the spec does not cite `KS00.9` for it.
  - The P2 proposal does: "The parents' levels set the family climate (why `M7.E.1f`'s parents' term stays as built)"
    (`docs/PROPOSAL — DISPLACED COPING (P2).md:63`). That is a misreading; the revision 39 draft (:250) records it
    correctly.
  - The inconsistency inside the model: functioning a parent gains through the exchange (`M6.I.4`'s pseudo-self) is
    netted out of `d`, because "the exchange does not reach development"; functioning gained by displacing onto the
    same child (`M1.A.5f`, booked on `stress_swing`, which the estimator reads at `slow_tick.py:127–129`) is not. So the
    gain a parent takes from the child lifts that child's development (+0.026 ± 0.019 of `M11.C.6`'s +0.018 ± 0.025,
    proposal §6.1). `FE06.3` and `KS04.7` point the other way.
  - Options: (a) leave it and keep `M11.C.6` composite (the owner's current choice); (b) give `M1.A.5f`'s transfer its
    own ledger and net it out of `d` only — a new rule, moves the engine hash, needs a probe; (c) drop `d`, leaving
    `KS00.9`'s climate in the anchor. L1 recommends (b) as closest to the sources, and correcting the proposal's
    citation whichever is chosen.
- **T-132 — `FE04.13`'s second kind of so-called I-position. A gap, confirmed; not a defect in what is built.**
  - `FE04.13` (`docs/theory/family_evaluation/fe04.md:382–386`) names two kinds: attempts to get others to change, and
    attempts to pry oneself loose from intense situations. `M5.F.4` (spec :956) models only the first, as the demand
    share.
  - `M11.C.29`'s "distance in disguise" arm is plain scripted `DISTANCE` (`src/bowen/ensemble/criteria.py:621–622`);
    the disguise is only in the criterion's name. `I-POSITION` is on the self channel only
    (`config/bowen/event_kinds.md:61`).
  - Recommendation: state `FE04.13` in full in `M5.F.4` and record the second kind as not modelled; note in
    `M11.C.29`'s row that its baseline is plain `DISTANCE`. Do not add a third share now (no source gives a size, and
    it needs a motive term the agents lack); a version 3 candidate.
- **T-133 — `M5.D.4a`'s anger check at `DEFINE`. The spec contradicts itself, and the sources favour checking at
  `DEFINE`.**
  - `M5.D.4a` (spec :893): an `I-POSITION` "emitted" while the mover is angry MUST execute as the assertion form.
    `M5.F.2c` (:949) and `M5.F.4` (:956) describe the anger gate as deciding whether a sequence runs, at selection.
  - The code checks anger only at selection (`src/bowen/engine/act.py:123–133`, through `iposition.assertion_form`,
    `iposition.py:124–130`). `advance_sequences` (`iposition.py:199–258`) computes the demand share at `DEFINE` with no
    anger check, and checks anger only in `HOLD` (`M5.D.4`). A `DEFINE` forced by an impinging move during `PREPARE`
    (`iposition.py:220–222`) — the likeliest moment for anger — is said as a mixture even when the mover is angry. The
    only test (`tests/bowen/test_iposition.py:141`) covers the selection path.
  - `KS05.5` (`ks05.md:82`, not taken in anger) and Kerr 1988 as cited in `M5.D.4a` describe the manner at the moment
    the position is said, which is `DEFINE`.
  - `M11.C.49` depends on the current behaviour: its act is said in every seed whatever F's anger, and
    `config/bowen/criteria.md` records F angry at week 4 in 14 of 40 seeds.
  - Options: (a) reword `M5.D.4a` to "selected while angry" — wording only, matches the code; (b) at `DEFINE`, an angry
    mover's position is all demand (share 1) and the sequence ends — moves the engine hash, needs `M11.C.32`, `M11.C.29`
    and `M11.C.49` rerun. L1 recommends (b); (a) removes the contradiction now.

Options T-131(b) and T-133(b) move the engine hash. If chosen, they belong in one batch with T-145, T-146 and T-147,
so the records are regenerated once.

## Findings

### Major findings

#### MR-1 — `M11.C.16`'s evidence record states results that its own regenerated table contradicts

- **Found by:** L3 and L1, independently · **Where:** `docs/phase_c_completion_report.md:455–456`, `:462–465`,
  `:486–487`, `:49`, `:541`; the fixed sentences in `tools/c16_report.py:128–129`, `:135–138`, `:159–160`; spec `:1396`
  (`M11.5`'s `M11.C.16` row)
- **Severity:** major · **Confidence:** high · **Verified:** yes (challenge pass, each sub-claim against the table)
- **What is wrong:** §11 is generated, but several of its sentences are literal text in the generator, and the
  regenerated mutant table at the current engine now contradicts them:
  - "survives … with availability and the learner removed together" (:486–487): the table shows
    `availability-and-learner-removed` **reversed**, +0.002 ± 0.002 (:429), and :414 names it as reversed.
  - "Inverted, it deepens the narrowing (−0.039 ± 0.007)" and "within the half-width of the unmutated difference"
    (:463–465): the unmutated value is −0.039 ± 0.006 (:418), so inversion changes nothing, and learner-disabled's
    −0.014 is four half-widths away.
  - "No other single rule removes the result" (:455): true of the deletions it sits among, but `steepness-inverted` is
    red (−0.001 ± 0.003, :440) and §11 mentions no single-rule inversion that cancels the result. Partly true.
  - Outside §11, the summary rows say `M11.C.16` is "reversed by `availability-level-inverted`" and that
    "`c16-grounds-removed` now survives" (:49, :541). The table shows the first survived (:428) and the second red
    (:439). Under `M11.1d` an inversion flip is the evidence that bears on a premise class, so these two rows misstate
    the fact the owner's class decision rests on.
  - The spec's `M11.5` row (:1396) repeats "No other single rule removes it" and "size, not its sign", and adds that the
    availability inversion "cancels it (both readouts' intervals span zero)" — it now survives at −0.019 ± 0.007 — and
    occupancy figures (40.6%/93.3%, 45.2%/90.5%) that differ from the report's (39.1%/96.4%, 43.9%/95.4%).
  - The generator's docstring (`c16_report.py:11–12`) says a regenerated record "cannot leave a false sentence behind".
  - The sentences that decide the class — INVERSION, FLIP and the cited-grounds clause (`c16_report.py:48–94`) — are
    computed and correct, so §11 as written still supports "composite". What is false is the account of learning's role,
    the "What does not hold" bullet, the two summary rows and the spec row.
- **Quoted:**
  - report :487 — "…survives with … availability and the learner removed together"
  - report :429 — "| `availability-and-learner-removed` | … | **reversed** | +0.002 ± 0.002 …"
  - report :49 — "…reversed by `availability-level-inverted` … `c16-grounds-removed` now survives"
- **How checked:** both lenses read §11 against its table; the challenge pass re-read every sub-claim, the generator
  and the test. `test_m115_report_section_11_is_generated` (`tests/bowen/test_ensemble_record.py:596`) checks only that
  §11 equals the generator's output, which includes the fixed text; no test reads :49, :541 or the spec row (grep of
  `tests/` for `steepness-inverted`, `c16-grounds-removed`, "No other single": nothing). T-143 records the mutant
  changes, not that the documents now state the opposite.
- **Fix:** make the learning bullet, the "No other single rule" sentence and the "What does not hold" bullet conditional
  on the mutant results, as INVERSION is, with a branch for a red or reversed single-rule inversion; generate or test
  rows :49 and :541 and the `M11.5` row from the same results; correct the docstring. Then put the class question to the
  owner (question 5).

### Major findings downgraded in challenge

#### MR-2 — † `M5.F.4` and `M5.F.2a` reuse an interview transcript's wording without quotation or paraphrase

- **Found by:** L1 (as a question); the orchestrator checked it · **Where:** spec `:956` (`M5.F.4`, new at revision 39),
  `:951` (`M5.F.2a`, present at the baseline); mirrored in `docs/spec_rules/M05.md:118`, `:123`
- **Severity:** minor (raised as major; the challenge downgraded it) · **Confidence:** high · **Verified:** yes
- **What is wrong:** the owner's ruling of 2026-10-24 (`docs/theory/_SOURCE_QUALITY.md:73`) is that a phrase of two or
  more words taken from a transcript is quotation, and is paraphrased. `M5.F.4` follows the KB12 transcript's sentence
  closely, sharing runs such as "low-level self can", "an I-position" and "a selfish, dogmatic". `M1.A.9a` (:236) already
  has the project's paraphrase of the same passage ("a self-centred, dogmatic and forceful one").
- **Quoted:**
  - spec :956 — "A low-level self can take an I-position, but a selfish, dogmatic and forceful one"
  - `docs/theory/kb/kb12.md:33` (K12.3) — "the low-level self can have an I-position. That is a selfish, dogmatic, forceful"
- **How checked:** the orchestrator grepped the spec for "dogmatic" and read the transcript line and the ruling; the
  challenge pass re-read both and `tests/test_asr_quotations.py`. That test flags only spans inside quote marks, at a
  four-word threshold (`tests/asr_quotations.json:3`, test :38, :77), so an unquoted lifted phrase cannot fire it.
- **Why downgraded:** nothing a run computes is affected; the interview is cited and nothing is attributed to Bowen. The
  wording was in the revision 39 draft the owner approved on 2026-10-25, a day after the ruling (not an exception, but
  the owner signed off on it). The stage 2 fix for the ruling changed one phrase ("shout them down") and did not sweep
  for others, so this is a class, not one instance.
- **Fix:** reword both in `M1.A.9a`'s form (for example "a poorly differentiated self can take one, but a self-centred
  and overbearing one"), regenerate `docs/spec_rules/M05.md`, and record in `TODO.md` either a lower threshold for the
  ASR test or why it stays at four, and that unquoted reuse cannot be found without reading the transcripts.

### Minor findings

Not challenged (the brief challenges blocker and major findings only). The orchestrator re-checked the ones marked
† against the files.

#### MR-3 — † Stage 3's built work is described as not built in its proposal, its plan and the checklist

- **Found by:** L3 · **Where:** `docs/PROPOSAL — DISPLACED COPING (P2).md:3`; `docs/SPEC_REVIEW_CHECKLIST.md:205`,
  `:231`, `:233`; `docs/implementation_plan_version_2_stage_3.md:4`
- **Severity:** minor · **Confidence:** high
- **What is wrong:** `M1.A.5f` is built (`src/bowen/engine/stress_response.py`, `moves.py`, `events.py`,
  `io/load.py`), and the revision 39 draft says "applied to the spec and built in full" (`spec_revision_39_draft.md:3–6`).
  The P2 proposal's status line says "not built"; the checklist's label for the draft says the spec stays at revision
  38 and "Not built", against its own `:27` and the spec's front matter (revision 39); the stage 3 plan keeps "No engine
  change is built." after a status line recording steps 6–12. Same class as the stage 2 review's MR-6.
- **Quoted:**
  - P2 :3 — "**Status: DECIDED 2026-10-25 (Q1–Q4, each as recommended); not built."
  - checklist :233 — "the spec stays at revision 38 until then. Not built."
  - plan :4 — "No engine change is built."
- **How checked:** L3 grepped `src/bowen` for `M1.A.5f` (4 files) and read the front matter and the draft's status
  line; the orchestrator read P2 :3 and checklist :233. `test_doc_status.py` classifies only a status line's first
  word, so the closing "not built" is never read (known gap, T-015 / SP-A).
- **Fix:** P2: "BUILT 2026-10-26 at stage 3 step 7"; checklist :231 "Built at step 7"; :233 "applied in full at steps
  6–10"; :205 drop "for owner approval" and "no engine change is built"; plan :4 delete or date the sentence.

#### MR-4 — † The reading list's body stops at stage 3 step 10, and still counts 581 requirements

- **Found by:** L3 · **Where:** `docs/REVIEWER_READING_LIST.md:3–9`, `:27`, `:75`, `:149`, `:184`;
  `docs/SPEC_REVIEW_CHECKLIST.md:13–15`
- **Severity:** minor · **Confidence:** high
- **What is wrong:** the list says it was last checked at step 11 and records step 12 in §8, but its body is older:
  `:27` ends "the stage-end sweep follows at step 11"; `:149` calls the status page step 10's; `:184` stops at step 6
  and says the records are stale until step 10 (the open batch closed at step 10, `tests/open_batch.py:42`); `:75` says
  581 requirements (the index has 585, as CLAUDE.md says) and names revisions to 37; the status line and the checklist
  label stop at 2026-10-25 and 2026-10-24. The stage 2 review's MR-5 and MR-15, again.
- **Quoted:**
  - `:27` — "…`M11.C.49` new and failing (`version_2_revision_39_changes.md`); the stage-end sweep follows at step 11."
  - `:75` — "581 numbered requirements: objects (`M1`), …"
- **How checked:** L3 read the list in full against the stage 3 plan, `_STATUS.md:3`, the spec front matter and a
  count of `spec_index.jsonl`; the orchestrator re-read `:75`.
- **Fix:** bring `:27`, `:149`, `:184` to step 12, or cut each to a pointer at the plan's status line; `:75` to 585 and
  revisions 38–39; extend the status line and checklist label to 2026-10-27. See MR-13 for the test that would have
  caught the count.

#### MR-5 — The sweep record is described as stage 2's pilot

- **Found by:** L3 · **Where:** `docs/SPEC_REVIEW_CHECKLIST.md:208`, `:46`; `docs/REVIEWER_READING_LIST.md:150`
- **Severity:** minor · **Confidence:** high
- **What is wrong:** the record is now stage 3's stage-end sweep (`version_2_sweep_record.md:1`), with the named ratio
  in 26 of its 34 rows. The checklist calls it the 2026-10-23 stage 2 pilot, the reading list says "stage 2's full
  sweep is complete", and the checklist's `sweep.md` label does not mention `## Named ratios` (`config/sweep/sweep.md:417`).
- **Quoted:** checklist :208 — "**Generated, 2026-10-23 (stage 2 pilot):** … The pilot: `M11.C.16` and `M11.C.42` at 8 samples."
- **How checked:** L3 read the record's header and counted rows carrying `tie_wear_to_transfer_ratio`.
- **Fix:** describe both as "stage 3's stage-end sweep, 2026-10-27, 34 keys, named ratio in 26"; add "named ratios
  (revision 39)" to the `sweep.md` label.

#### MR-6 — † The register names a writer for `Triangle.bound_anxiety`; `M1.C.1` and the code say none exists

- **Found by:** orchestrator (spec read); L2 confirmed the code independently · **Where:** spec `:1689`; spec `:386`
  (`M1.C.1`); `src/bowen/engine/event_effects.py:145`, `life_course.py:573`, `objects.py:305`
- **Severity:** minor · **Confidence:** high
- **What is wrong:** the state register's row says the field is written by "triangle binding through the sinks
  (`M1.D.1`)". `M1.C.1` says it is not written (the owner's D22), and the code only ever sets it to zero.
  `M11.D.17`'s no-writer check reads the register, so it passes on a false row. T-004's list of "fields that exist and
  are never written" (`docs/todo/2026-08-22_blocking_the_next_phase.md`) does not carry it, though it carries the
  same class for two `M1.A.5b` fields.
- **Quoted:**
  - spec :1689 — "| `bound_anxiety` | Triangle | DV | triangle binding through the sinks (`M1.D.1`); `binder_unavailable` releases it (`M1.F.9`) | C; B |"
  - spec :386 — "`bound_anxiety` is not written: no act binds anxiety into a `Triangle`, so the release paths that read it (`event_effects.py`, `life_course.py`) release nothing"
- **How checked:** `grep -rn bound_anxiety src/bowen` (excluding `distance_bound_anxiety`): a default of 0.0, two
  assignments of 0.0 and one sum. Not in the stage 2 report or `TODO.md` (grep).
- **Fix:** restate the row's writer as "none (D22; `M1.C.1`)", and add the field to T-004's never-written list.

#### MR-7 — † The spec's pointer to its revision log stops at 38

- **Found by:** L2, and the orchestrator's spec read · **Where:** spec `:2136`
- **Severity:** minor · **Confidence:** high
- **What is wrong:** "4 to 38"; the log holds revision 39 (`spec_revision_log.md:3`, `:9`), and the checklist (:31)
  and reading list (:76) say "4 to 39". The line sits in a hunk changed at stage 3. The stage 2 review's MR-16 again:
  a restated range is fixed by hand each revision.
- **Quoted:** spec :2136 — "The record of every revision of this specification, 4 to 38, …"
- **How checked:** grep of all four files for "4 to 3[0-9]".
- **Fix:** "4 to 39", or drop the range from the pointer and let the log's status line carry it.

#### MR-8 — † Two passages of the spec describe a state that no longer exists

- **Found by:** orchestrator (spec read) · **Where:** spec `:1954–1956` (`M16.B.2`); spec `:2140`
- **Severity:** minor · **Confidence:** high
- **What is wrong:** `M16.B.2` says, in the present tense, that the frozen engine writes `sim_audit.csv` and that two
  stray copies of it are in the repo. The grid engine was retired at stage 1 (`M13.2`, :1584), and no `sim_audit` file
  is tracked. The closing line says the spec was written against `model_explainer.md` v1.2; the explainer's front
  matter says version 2.0.
- **Quoted:**
  - :1954–1956 — "The frozen engine opens and writes `sim_audit.csv` from `Simulator.__init__` into the **process working directory** … two stray copies of that file exist in the repo."
  - :2140 — "*Written against `model_explainer.md` v1.2 …"
- **How checked:** `git ls-files | grep -c sim_audit` → 0; `model_explainer.md:4` → `version: 2.0`. Neither is in the
  stage 1 or stage 2 report (grep).
- **Fix:** put `M16.B.2` in the past tense ("the retired grid engine wrote …"), and drop the stray-copies clause;
  update the closing line's version.

#### MR-9 — † T-004's list of spec–code disagreements still holds `M1.A.18`, which revision 39 resolved

- **Found by:** L2 · **Where:** `docs/todo/2026-08-22_blocking_the_next_phase.md:43–44`
- **Severity:** minor · **Confidence:** high
- **What is wrong:** the 0.3 threshold the item objects to was retired at revision 39 (Q8); `M1.A.18` is *done* on the
  coverage page. The list asks the owner to rule on a disagreement that no longer exists (the stage 2 review's MR-10
  class). The list has not grown: stage 3's new disagreements were filed as T-131–T-147, and the record's only change
  in this range is MR-10's strike-through.
- **Quoted:**
  - record :43 — "`M1.A.18`: the 0.3 threshold makes a genuine I-POSITION impossible below it, against "graded, not a hard gate"."
  - `src/bowen/scenario/constants.py:245` — `"assertion_perspective_threshold": KeySpec(float, "[I]", "M5.F.4"),  # spec revision 39 (Q8): retired`
- **How checked:** L2 grepped `src/bowen` (the name appears only in the `RETIRED` table) and read `iposition.py:123–130`;
  the orchestrator ran `git diff --stat d51b250..161da4e` on the record (one 2-line change, MR-10's).
- **Fix:** strike the `M1.A.18` sentence with a closing note pointing at revision 39 Q8; keep the `M1.A.18c`/`18d`
  item, which still holds.

#### MR-10 — † The event-size config says `FIND_WORK_LOAD` goes to the person who finds work; it goes to the household

- **Found by:** L1 · **Where:** `config/bowen/event_sizes.md:44`; `docs/review/stage_3_t130_event_sizes.md:54–57`
- **Severity:** minor · **Confidence:** high
- **What is wrong:** the code (`src/bowen/engine/life_course.py:672–674`), the spec (:1067, "to the household") and
  the explainer (:1407) send the load to the whole household; `who` = self only selects the size's reach. The config
  note, which is the source record for the Holmes–Rahe sizes, says otherwise.
- **Quoted:** `event_sizes.md:44` — "the load to the person who finds work (`who` = self) are the model's"
- **How checked:** L1 read `_addition_load` (`life_course.py:292–302`): the targets are the list passed in. The
  orchestrator re-read :44.
- **Fix:** "the load goes to the household, at the self reach (1)", in the config and the T-130 page.

#### MR-11 — † Three documents state the CHANGELOG archive's boundary three ways

- **Found by:** L3 (checklist, reading list); orchestrator (the brief) · **Where:** `docs/SPEC_REVIEW_CHECKLIST.md:149`;
  `docs/REVIEWER_READING_LIST.md:189`; `docs/milestone_review_brief.md:127–128`
- **Severity:** minor · **Confidence:** high
- **What is wrong:** the archive's header (`docs/CHANGELOG_archive.md:3–5`) and CLAUDE.md say it holds everything
  before 2026-10-22, 2026-10-22's entries and the ten earliest of 2026-10-23. The checklist and the reading list say
  "before 2026-10-22"; the brief says "before 2026-10-19".
- **Quoted:** checklist :149 — "the `CHANGELOG.md` entries dated before 2026-10-22 (version 1, and the first three days of version 2)"
- **How checked:** the orchestrator read the archive header and the three lines.
- **Fix:** state the boundary once, in the archive's header, and point to it from the others.

#### MR-12 — `TODO.md` defines **Now** as blocking stage 2, which closed

- **Found by:** L3 · **Where:** `TODO.md:5`, generated by `tools/todo_index.py:142`; restated at
  `docs/REVIEWER_READING_LIST.md:189`
- **Severity:** minor · **Confidence:** high
- **What is wrong:** "**Now** blocks version 2 stage 2 or the next commit". Stage 2 closed 2026-10-24; the eight Now
  items stayed open through stage 3, so the heading no longer says what the section means.
- **How checked:** L3 read the index and the generator; the orchestrator re-read `TODO.md:5`.
- **Fix:** "blocks the current stage's close or the next commit", in `todo_index.py` and the reading list.

#### MR-13 — The engine hash and the reading list's counts are claims no test checks

- **Found by:** L3 · **Where:** `CLAUDE.md:27`; `README.md:54`; `docs/theory/_STATUS.md:3`;
  `tests/test_spec_consistency.py:40–49` (`CITING_DOCS`)
- **Severity:** minor · **Confidence:** high
- **What is wrong:** CLAUDE.md, README and `_STATUS.md` state the engine hash `2a3b84e2d89a`, and CLAUDE.md makes it a
  hard rule for the next session; no test compares it with the records' `engine_hash` (it agrees today: Phase C
  mutation record :14, Phase C sweep record :12, version 2 sweep record :23). `CITING_DOCS` leaves out the reading
  list and the checklist, which is why `agent_model_proposal.html` was moved from 581 to 585 in this range and the
  reading list's 581 was not (MR-4).
- **How checked:** L3 grepped `tests/` for the hash (one comment, `tests/open_batch.py:40`) and read `CITING_DOCS`.
- **Fix:** add both documents to `CITING_DOCS`; add a test that the last engine hash CLAUDE.md names equals the Phase C
  mutation record's.

#### MR-14 — `_STATUS.md`'s dated block still says "Next: step 7"

- **Found by:** L3 · **Where:** `docs/theory/_STATUS.md:24` against `:3`
- **Severity:** minor · **Confidence:** high
- **What is wrong:** the block dated 2026-10-22 was extended to step 6 and ends "Next: step 7, group C"; the state
  line says "Next: … step 13". The stage 2 review's MR-17, again.
- **Fix:** close the block at step 12 or cut it to a pointer at line 3.

#### MR-15 — CI installs whatever `ruff` and `numpy` are current

- **Found by:** L4 · **Where:** `requirements.txt:2`, `:4`; gated by `tests/test_lint.py:28`
- **Severity:** minor · **Confidence:** high
- **What is wrong:** the new lint gate (`test_s3_22_ruff_f_clean`) runs the ruff CI installs that day; `ruff` is
  unpinned and the repo has no ruff config, so a ruff release that adds an F rule turns CI red with no code change.
  `numpy` is unbounded too, and its draws feed the golden fingerprints. This review's venv resolved ruff 0.17.3 and
  numpy 2.5.4; what CI resolved was not read.
- **Quoted:** `requirements.txt` — "numpy                    # The agent model's draws and the ensemble statistics" / "ruff                     # Linter"
- **How checked:** L4 read `requirements.txt` and looked for `pyproject.toml` / `ruff.toml` (none).
- **Fix:** pin `ruff` to the version that passed run 40630271972 and bound numpy's major version; bump each on
  purpose, in its own commit.

#### MR-16 — A probe test's "as built" assertion cannot fail

- **Found by:** L4 · **Where:** `tests/test_stage_3_probes.py:80–81`
- **Severity:** minor · **Confidence:** high
- **What is wrong:** in the `as_built` variant the probe never installs its displacement wrapper, so the two tallies
  are 0 by construction, whatever the engine does.
- **Quoted:** `assert all(row["as_built"][arm][c] == 0 for arm in coping.ARMS for c in ("displaced_to_child",`
- **How checked:** L4 read `tools/stage_3_displaced_coping_probe.py` in full: the tally is incremented only inside the
  wrapper, patched in only for the P2 variant.
- **Fix:** delete it, or check the engine's own displacement in the `as_built` treatment arm.

#### MR-17 — `test_d31_every_listed_record_is_marked_and_no_other` asserts nothing while the batch is closed

- **Found by:** L4 · **Where:** `tests/test_open_batch.py:61–63`; the `_registry_as_found` fixture in
  `tests/test_version_2_status.py`
- **Severity:** minor · **Confidence:** high
- **What is wrong:** its body runs only when `open_batch.OPEN` is set; it is `None` (`tests/open_batch.py:42`), so the
  test passes and counts among the 1258 while checking nothing. A stale open-batch mark left on a record after the
  close is what it should catch.
- **Quoted:** `    if open_batch.OPEN is not None:` / `        assert sorted(_marked()) == sorted(AT_OPEN)`
- **Fix:** add `else: assert _marked() == []`.

#### MR-18 — The family-of-origin arm's held-weeks check was loosened to "more than none"

- **Found by:** L4 (L2 matched the cause to T-135) · **Where:** `tests/bowen/test_phase_d2_criteria.py:130`
- **Severity:** minor · **Confidence:** medium
- **What is wrong:** the exact `== 4 * 2 * 52` became `0 < … <= 416`, so an arm whose cutoffs lapse in all but one
  week still passes. The docstring records 401 for seed 0. T-135 is open on the cause, not on the bound.
- **Quoted:** `assert 0 < treated.moves[HELD_WEEKS] <= 4 * 2 * 52 and baseline.moves[HELD_WEEKS] == 0`
- **Fix:** until T-135 is decided, a floor that catches a collapse (for example `>= 0.9 * 416`), or `== 401` pinned
  with the golden fingerprints.

#### MR-19 — The projection-trace tool prints the estimator's formula from before revision 39

- **Found by:** L4 · **Where:** `tools/stage_3_projection_trace.py:191–193` against `src/bowen/engine/slow_tick.py:129`
- **Severity:** minor · **Confidence:** high
- **What is wrong:** the page text leaves out `− floor_held`. The committed page is stamped at engine `66dcbb1e8ba5`,
  so it is right as it stands; a regeneration on the current engine would state a formula the code no longer uses.
- **Quoted:** tool — "The estimator's functional level is \`functional_level − pseudo_self − max(0, adjust_swing)\`" ;
  `slow_tick.py:129` — `own = person.functional_level - (person.pseudo_self or 0.0) - max(0.0, person.adjust_swing) - person.floor_held`
- **Fix:** add the term, or have `main()` refuse unless the engine hash is the page's stamped one.

#### MR-20 — The i-position probe copies the criterion's mover setup instead of calling it

- **Found by:** L4 · **Where:** `tools/stage_3_iposition_probe.py:65–70` against `criteria.mover_at` in
  `src/bowen/ensemble/criteria.py`
- **Severity:** minor · **Confidence:** medium
- **What is wrong:** the two copies (including `systems_perspective = 1.0`) can drift apart, and the probe would then
  measure a different mover from `M11.C.29`'s and `M11.C.49`'s. The spell handling differs on purpose.
- **Fix:** call `criteria.mover_at(...)` and keep only the documented spell difference local.

#### MR-21 — † Probe-page tests for frozen pages take 17% of the default suite

- **Found by:** L4 (as a note); timed by the orchestrator · **Where:** `tests/test_stage_3_probes.py`
- **Severity:** minor · **Confidence:** high
- **What is wrong:** the file runs real `M11.C.6` arms, two coping variants and three i-position seeds in the default
  run, for probe pages the tools call frozen: 60.9 s of the suite's 351.9 s here (under `nice -n 15`). The slowest:
  the displaced-coping probe 39.0 s and the trace 18.4 s.
- **How checked:** `pytest tests/test_stage_3_probes.py --durations=0 -q -p no:cacheprovider` in the review venv.
- **Fix:** mark the two slow tests `ensemble` (or a `slow` mark), so the default suite stays the fast check (P38).

#### MR-22 — † `M13`'s Phase C exit says `M11.C.49` "passes … mutation-proved"; it fails and its mutants have not run

- **Found by:** orchestrator (spec read) · **Where:** spec `:1578`; `docs/phase_c_ensemble_record.md:27`;
  `docs/phase_c_mutation_record.md:177–178`
- **Severity:** minor · **Confidence:** high
- **What is wrong:** `M11.C.49` is mixed, its composite limbs a finding either way (`M11.5`, :1429), and it fails at
  the central setting (T-137). Phase C's *Done when* lists it as an exit condition that passes. Phase D's cell treats
  the composite `M11.C.48` the other way: "its verdict reported (a composite …: a pass or a fail is a finding, not an
  exit condition)". The failure itself is known; the exit wording is not recorded.
- **Quoted:**
  - :1578 — "**Added at revision 39:** M11.C.49 passes over the same ensembles, mutation-proved."
  - ensemble record :27 — "| `M11.C.49` | mixed | **FAIL** | 150 | `family_excess` +3.06 ± 2.9, p 0.0317, does not hold; …"
- **Fix:** restate it as `M11.C.48`'s is: run over an ensemble and its verdict reported, a finding, not an exit
  condition.

## Systemic patterns

1. **Hand-restated status goes stale between steps (MR-3, MR-4, MR-5, MR-7, MR-11, MR-12, MR-14).** Seven findings,
   four of them recurrences of stage 2 findings that were fixed by editing the sentence (MR-5, MR-15, MR-16, MR-17 of
   that review). Each fix restated the new state by hand, and the next step made it stale again. The tests read only a
   status line's first word (T-015 / SP-A), `CITING_DOCS` leaves out the reading list and the checklist, and no test
   reads the reading list's per-step narrative. The fix that pays is structural: point at one generated or
   test-checked source (the plan's status line, the log's header, the archive's header) rather than restate it.
2. **Generated pages carry fixed prose that the regenerated numbers can contradict (MR-1, MR-19).** A test that the
   page matches its generator cannot see a sentence that is literal text. Where a sentence states a result, it should
   be computed from the record, as `c16_report.py` already does for two of its sentences.
3. **Tests whose assertion is fixed by construction or switched off by state (MR-16, MR-17, and MR-18's loosened
   bound).** Each passes and is counted, and none can fail for the reason its name gives.
4. **Lists of disagreements are not swept when a revision resolves one (MR-6, MR-9).** T-004's lists are edited only
   when a review names an entry. Stage 3 filed its new disagreements as separate items, so T-004 did not grow, but
   nothing marks an entry resolved or adds a newly found one.

## Refuted in challenge

None. The challenge pass confirmed MR-1 (one sub-claim only partly: "No other single rule removes the result" is
true of the deletions it sits among) and confirmed MR-2 while downgrading it to minor.

## Checked and clean

- **L1 · `src/bowen/` engine and `config/bowen/` — no rule writes an outcome.** `stress_response.displace` names no
  child, role or focus; `returned_contact` names no kind of act; `M6.I.4a` reads capacity, not cause; `TRIANGLE` at
  layer 0 only removes a gate.
- **L1 · `config/bowen/` — grades.** `reciprocal_contact_share` 0.5 is `[I]` with "no source gives a share"
  (`constants_changes.md`); `assertion_perspective_threshold` retired (:203); `TRIANGLE` at layer 0 is `[I]` citing
  KS15.10 (`ks15.md:173`) and FE06.20 (`fe06.md:554–558`), neither `[K-ext]`; the Holmes–Rahe sizes and the C.29/C.49
  criteria rows are `[I]`.
- **L1 · spec `M1.A.5f`, `M4.C.1d`, `M5.F.4` — citations support their rules.** KS04.7 and FE01.1 for `M1.A.5f`'s
  direction; KS03.1 and KS03.2 for `M4.C.1d`; KS05.5, KS06.11 and K12.3 for `M5.F.4`; L08.3 (`_LEDGER.md:728–741`) for
  substitutable channels, the size kept `[I]`.
- **L1 · `config/sweep/sweep.md` — the named ratio.** Central 1/52, range 1/208 to 4/52, `tie_wear_rate` 0.00625–0.4,
  consistent with 0.05, 0.05 and 52.
- **L1 · spec `M5.F.4`, `M11.C.49` — arithmetic.** `w` 0.58/0.68 and 0.16/0.36, demand 0.44 at 35 and 0.32 at 60,
  follow from `self_channel_exponent` 2.0 and `perspective_channel_gain` 0.5 (L2 checked the same against code).
- **L1 · explainer, M11.5, probe pages — premise framing and `M11.F.9`.** C.49's impingement limb is marked never a
  finding (spec :1429, explainer :1312–1320); each stage 3 probe page carries the real-family disclaimer; `M11.C.17`'s
  unproved pass is disclosed (explainer :253–256).
- **L1 · changed text — attribution.** No `[K-ext]` claim attributed to Bowen.
- **L1 · explainer and config — ASR quotation, apart from MR-2.** "shout them down" replaced (explainer :383);
  `config/bowen` has no KB or Tape citation; K12.3 "cited and not quoted" (explainer :2210).
- **L2 · `M1.A.5f`, `M1.A.4k`/`M1.A.5e` `floor_held`, `M7.E.1f`, `M6.I.4a`, `M4.C.1d`, `M5.F.4`/`M5.F.2c`, `M4.D.3a`,
  `M8`/`M8.1`/`M8.5`/`M13.1`, `M7.D.1a`, `M1.D.2b`/`M1.D.2c` — code matches the spec** (the details and lines are in
  L2's checks: `stress_response.py:114–157`, `:201–215`, `slow_tick.py:129`, `life_course.py:402–410`, `moves.py:446–495`,
  `tick.py:219`, `appraise.py:315–350`, `:422–428`, `act.py:123–150`, `moves.py:524–546`, `event_kinds.md:59`,
  `live_positions.py`).
- **L2 · `criteria.py` — the arms of `M11.C.29` and `M11.C.49`** match the spec's arms and directions; C.49 raises if
  the act is not said exactly once; C.29 treats `COMPLETED` as a required move.
- **L2 · changed acceptance tests — no absolute thresholds.** `test_m11c49_…` asserts the runner's two-arm verdict; the
  exact-value checks in `test_iposition.py:481–521` are unit tests of a formula.
- **L2 · coverage page — no coverage claimed without a run mutation.** Every Phase C criterion marked done has a red or
  reversed mutant in the Phase C mutation record; every criterion with mutants under "Not run" (C.42, C.49, C.6, C.21)
  is partial.
- **L2 · `docs/review/stage_3_unit_mutation_gate.md`** — 25 of 25; the orchestrator reran it on the committed tree with
  the same result.
- **L2 · T-004's disagreement list has not grown** (seven items; no revision 39 requirement among them).
- **L3 · CLAUDE.md, `_STATUS.md:3`, README :54, stage 3 plan :3** agree on the step reached (12 done, 13 next), the hash
  sequence `66dcbb1e8ba5` → `f90285984f32` → `b8e9ad1333e2` → `b819660f58b9` → `2a3b84e2d89a`, revision 39 applied in
  full, and T-144 to T-147.
- **L3 · step 10's verdict changes** (6 of 63 rows) and W1/W3/W4 confirmed against `version_2_status.md`; "the ratio
  entered 26 of 34" confirmed by row count; 585 IDs; CLAUDE.md's suite note (`OPEN = None`, one strict xfail at
  `test_cutoff_stock.py:481`) — and the suite count itself confirmed by this review's run.
- **L3 · status lines current:** spec front matter, revision 39 draft, revision log, version 2 plan,
  `two_mac_sweep.md`, the Holmes–Rahe proposal, `model_explainer.md` (dated 2026-10-27; the stage 2 review's MR-19 is
  fixed), the two `_changes.md` headers, `spec_rules/README.md`.
- **L4 · `src/bowen/` — P4 literals.** No new magic literals in the engine hunks (`stress_response`, `moves`,
  `appraise`, `act`, `events`); all are 0, 1 or `SCALE_MAX`; `test_engineering_gates.py:82–95` enforces it.
- **L4 · `src/bowen/engine/` — M16.B I/O and UI.** Nothing new (grep for open, print, os, pathlib, write, read_text).
- **L4 · `tools/`, `tests/` — subprocess timeouts.** Every call has one (`open_batch.py` 60 s, `test_lint.py` 120 s,
  `test_spec_consistency.py` 120 s, ssh through `two_mac_sweep._run`).
- **L4 · patch restoration.** Every new patch is undone in a `finally`, or by monkeypatch or `mock.patch`.
- **L4 · skip and xfail reasons.** The `M5.B.3` strict xfail has `raises=AssertionError`, a full reason, and checks its
  arms through `ArmBroken` (stage 2's MR-24 fixed); `pytest.ini` prints reasons (`-rfEsx`).
- **L4 · P38 caching.** `ensemble_record.py` is cache-backed with a flush after each criterion (stage 2's MR-23 fixed);
  sweep run lengths cached by code hash.
- **Orchestrator · spec read, `M10.C.2`.** The KB11 passage is paraphrased (the stage 1 item is settled).

## Unverified

- **The 52 `ensemble` tests** (CLAUDE.md :91): `mark.ensemble` appears 36 times and the rest come from module marks
  and parametrisation; this review's run deselected 52, which agrees, but `pytest -m ensemble --collect-only -q` was
  not run to list them.
- **README :71 run times** ("about 40 and 20 minutes"): need a timed regeneration.
- **The ruff and numpy versions CI resolved** (MR-15): read the install step of run 40630271972.
- **`M4.C.1d`'s "past the optimum, nothing"** in `deviation_at`: L1 left it to L2, and L2 did not read that path.
  Check: read `appraise.py`'s `deviation_at`, or `pytest tests/bowen/test_moves.py -q -k returned`.
- **`criteria.arm_c49`'s `/ len(per_week)`** raises `ZeroDivisionError` if a swept run is shorter than the measurement
  window (L4). Not reachable at the declared settings; T-144 (no sweep ranges yet) is where it would surface.
- **`criteria.py` counts a seed with no move on the F–M tie as `conflict_share` 0.0, while the i-position probe leaves
  such seeds out (NaN)** (L4), so the probe and the criterion average over different seed sets. Not checked for how
  many seeds it affects.

## Questions only the owner can answer

1. **T-131:** option (a), (b) or (c)? Whichever, should the P2 proposal's `KS00.9` citation be corrected now?
2. **T-133:** option (a) — reword `M5.D.4a` to "selected while angry" — or (b), checking anger at `DEFINE`, with
   `M11.C.32`, `M11.C.29` and `M11.C.49` rerun?
3. **T-132:** record the second kind of so-called I-position as not modelled, or plan a third share for version 3?
4. **MR-2:** was `M5.F.4`'s wording, approved in the revision 39 draft on 2026-10-25, meant as an exception to the
   2026-10-24 ruling on transcript phrases? If not, should the fix sweep the spec for other reused phrases, not only
   these two?
5. **`M11.C.16`'s class (MR-1).** At the current engine, `availability-and-learner-removed` reverses `M11.C.16` and
   `steepness-inverted` turns it red. Does that change the owner's 2026-10-08 decision to keep it composite, or the
   open question in `phase_c_completion_report.md` (:491) about restating it to test a learned contribution?
6. **The reading list's per-step narrative (MR-4).** Keep it, or cut the "Before you start" row and the plan row to
   one-line pointers at the stage 3 plan's status line? It has gone stale at three successive steps.
