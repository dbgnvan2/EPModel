# EPModel — Claude Code instructions

Python + NumPy simulation of emotional process at unit, family, and societal levels.

The model is a behaving-agent model over one three-generation family (`src/bowen/`). It replaced the grid engine
(`src/engine.py`, 10,000 units on a 100×100 lattice), which was retired at version 2 stage 1. Read
`docs/agent_model_proposal.html` before touching architecture — especially §10, which records what reading the source
corpus changed.

## Global standards

Read the relevant file from `~/.claude/standards/` before starting work:

| Standard | When |
|---|---|
| `effort-budget.md` | **Every task**, before planning verification, records or review passes. Rule 1 sets the tier from the paths a change touches (`src/bowen/`, `config/bowen/` and the readouts are Full; tests or config only are Light; a document is None); rule 5 bans tests that assert on prose; rule 9 ends a batch with the ratio |
| `learnings.md` | P4 (hardcoded constants — simulation parameters belong in config), P7 (scoring/ranking proxies), P9 (magic size/count caps in simulation loops), P23 (any stochastic-output regression fixture) |
| `file-maintainability.md` | Any new module or significant refactor |

## Where the theory lives

| File | What it is | Trust |
|---|---|---|
| `docs/theory/_LEDGER.md` | **six corpus sources**, all passes folded in. Per-chapter findings for the 22 papers; consolidated entries for the later sources, each backed by a fuller extraction directory | **the source of truth for what the corpus says** |
| `docs/theory/_CONVERGENCES.md` | ten cross-chapter convergences + independence audit | current |
| `docs/theory/_STATUS.md` | run state, standing decisions, order of work | **read this first** |
| `docs/DECISIONS — FAMILY EVALUATION.md` | how the 4 contradictions and 21 requirement candidates from the sixth source were decided | **closed 2026-08-27 — all applied at spec revision 4** |
| `docs/bowen_agent_model_spec_v2.md` | **the buildable contract** — 602 numbered requirements, the update order, the parameter register, 55 acceptance criteria, the `M11.G` readout schema, the `M15` import contract, the `M16` run log | **Version 1 complete 2026-10-18; version 1 is frozen at v2.0 revision 32 and the tag `version-1`. Version 2 complete 2026-11-01 at revision 45 and the tag `version-2` (`8c4e0bf`). Revision 58 (2026-11-10, wording only) wrote the owner's decisions of 2026-11-10 on the parts of revisions 51–55 that were pending review (`docs/review/2026-11-10_revisions_51_55_review_prep.md` §10): all were approved; R53-1 and R53-2 with a change, so that `M11.C.2`'s row and its version 2 note give the two mutants' numbers and say the record does not show which path carries the twelve-person pass; R55-14 to be decided again with T-209; and `M4.C.1e`'s and `M4.C.14`'s text, which revision 55 left under `M4.C.1f` and `M4.C.15`, is back under its own rules (F-1); the engine hash did not move (`d3779684dd53`). Revision 57 (2026-11-10, the version 2 definition plan's step 2) built the two changes the owner decided 2026-11-09 and held for the next batch that moves the engine hash: `M11.C.55` compares the relief predictor with the supply before the window and with chance, not with birth order, and reads the relief `relief_lead_weeks` (52) before the window rather than at its start (T-206); and `M11.C.48`'s baseline reads only uncut ties acted on by as many of their members as the cut tie, with a first cutoff that has no such tie left out of both arms and counted (`unsided_cutoffs`, T-185, the owner's Q3 (a)). The engine hash moved from `927550dfcb8a` to `d3779684dd53`; every record in scope was regenerated once on it at the plan's step 4 (2026-11-10 to 11, split across both Macs): no criterion's verdict moved, and two mutants of the Phase D mutation record did — `cutoff-blind-to-learned-value` from survived to red on `M11.C.48[phase_d]`, which meets W3, and `supply-by-birth-order` from reversed to red on `M11.C.55` (`docs/version_3_revision_57_changes.md`); the dispositions are the owner's at the plan's step 6. Revision 56 (2026-11-09, wording in `M11` only) wrote T-206's and T-208's probes into `M11.5`'s rows for `M11.C.52` and `M11.C.55` (`docs/review/2026-11-09_sc_pass_probes.md`), as the owner decided 2026-11-09: neither pass is a finding about what its criterion names, `M11.C.55`'s two criterion changes are held for the next batch that moves the engine hash, and `social_cues_enabled` stays off by default; the engine hash did not move (`927550dfcb8a`). Revision 55 (2026-11-08, T-202) added the social cues (`M1.F.1c`, `M4.C.1e`, `M4.C.1f`, `M4.C.11` to `M4.C.15`, `M9.9`, `M11.G.5`, `M11.C.50` to `M11.C.55`), all behind `social_cues_enabled`, off by default, so no existing run changed; the engine hash moved from `0b363cb3186c` to `927550dfcb8a` and every record but the emergence audit and the version 2 sweep was regenerated once on it (`docs/version_3_revision_55_changes.md`); its choices taken as recommended under the owner's overnight instruction were approved by the owner 2026-11-10. Revision 54 (2026-11-06) corrected `M11.5`'s note for `M11.C.42`: `M4.C.1d`'s contact is returned to its sender by a relieved receiver (T-199, Claude's recommendation approved by the owner 2026-11-10; wording in `M11` only, the engine hash did not move). Revision 53 (2026-11-06) restated `M11.C.2`'s mutation cell to the current record, `witness-path-removed` red and `weakness-focus-and-steering-removed` surviving on the twelve-person family (T-171, Claude's recommendation approved by the owner 2026-11-10); wording in `M11` only; the engine hash did not move. Revision 52 (2026-11-06) folded T-196's and T-197's probes into `M11.5`'s rows for `M11.C.48` and `M11.C.2` (T-198, Claude's recommendation approved by the owner 2026-11-10, no class changed): on the Phase B family matching on tie class detects no difference; on the twelve-person family, with which child is picked held, no direction is detected over the three children; wording in `M11` only; the engine hash did not move (`0b363cb3186c`). Revision 51 (2026-11-05) wrote into `M11.5`'s rows for `M11.C.2` and `M11.C.48`, and each criterion's row, that their within-run arms differ in a condition besides the one compared (T-194, T-195, no class changed): `M11.C.48`'s fail on the Phase B family is not a finding about the run before a cutoff; `M11.C.2`'s fail on the Phase C family is not a finding about projection alone, and its pass on the twelve-person family is not separated from the children's difference in birth order and starting state; done ahead of T-196 and T-197, Claude's recommendation approved by the owner 2026-11-10; wording in `M11` only; the engine hash did not move (`0b363cb3186c`). Revision 50 (2026-11-05) wrote what carries `M11.C.48[phase_d]` into `M11.5`'s row, the criterion's row and its `M11.4` exception (T-180, T-185, no class changed): its direction holds with all learning removed and comes in part from what its arms are made of, and, since it passes with learning removed, its named mutant did not turn it red; wording in `M11` only; the engine hash did not move (`0b363cb3186c`). Revision 49 (2026-11-05) wrote the owner's decisions on the emergence audit's survivors into `M11.5`'s rows (T-186 to T-191, no class changed) and `M11.1g`'s scope sentence (T-192), wording in `M11` only; the engine hash did not move (`0b363cb3186c`). Revision 48 (2026-11-03, version 3's first revision) added `M11.1g`, the emergence audit (T-182), wording in `M11` only; the engine hash did not move (`0b363cb3186c`). Revision 46 (2026-11-01, after the tag) changed wording only: `M11.C.34`'s rows restated from T-173's two mutants, approved by the owner 2026-11-02; the engine hash did not move (`09ca2b46c7e0`). Spec revision 47 (2026-11-02–03, T-179, the owner's option (a)) reads both arms of `M11.C.48` under one contact condition: a first cutoff on a tie with no contact by its week is left out and counted. The engine hash moved from `09ca2b46c7e0` to `0b363cb3186c` and every record but the sweep record was regenerated once on it (`docs/version_2_revision_47_changes.md`): `M11.C.48` fails with no difference, and `M11.C.48[phase_d]` passes with its named mutant surviving (T-180); the full sweep reran on it 2026-11-03 to 2026-11-04, split across both Macs (T-181): only `M11.C.48`'s two keys moved against revision 44's sweep, and W4 is met. Version 2 began at revision 33 (stage 1, built 2026-10-19: the grid engine retired, `M8.5` amended; no rule changed). Revision 42, built 2026-10-29 on `main` with T-159's stage-end sweep, changed wording only: `M17.A.1` says how the sweep record shows a criterion refused at the rerun (the owner's choice; the sentence approved by the owner 2026-10-31); the engine hash did not move (`6ce8be104767`). Revision 41, built 2026-10-28 on `main` from the stage 3 milestone review's triage batch (T-131 (b), T-133 (b), T-145 (a), T-146, T-147, T-149, and T-148's generated `M11.C.16` sentences), moved the engine hash once, from `2a3b84e2d89a` to `6ce8be104767`, and the golden fingerprints moved; every record but the stage-end sweep was regenerated once on it (`docs/version_2_revision_41_changes.md`), and the stage-end sweep was rerun on it 2026-10-29, split across both Macs (T-159; `docs/version_2_sweep_record.md`: 33 keys with a fraction, `M11.C.48` refused at the rerun, T-163). Revision 45 (2026-11-01) changed wording only, from the pre-tag milestone review's triage (`docs/todo/2026-11-01_pre_tag_milestone_review.md`); the engine hash did not move. Revision 44 (2026-10-31, the owner's D4 and D5, `docs/version_2_definition_plan.md` §8): `M11.C.21`'s scripted `OVERFUNCTION` ends at `t0` in both arms (T-161 (b)), and `M5.D.5`'s pull-up does not apply when the mover is the external agent (T-165), which retires revision 43's `coach-pull-up-removed`. The engine hash moved from `da5981f6ec0b` to `09ca2b46c7e0` and every record but the stage-end sweep was regenerated once on it on Mac Mini1 (`docs/version_2_revision_44_changes.md`): `M11.C.21` and `M11.C.17` now fail, and no other ensemble verdict changed. Revision 43, built on `main` (2026-10-29 to 30, the version 2 definition batch; Claude's recommendations under the owner's instruction to proceed, approved by the owner 2026-10-31): `M11.C.48` leaves a first cutoff with no uncut tie out of both arms and runs 200 seeds (T-123, T-163), and `M11.C.11`'s, `M11.C.15`'s, `M11.C.17`'s and `M11.C.34`'s rows each name a mutant that turns them red; T-161 was built and withdrawn. The engine hash moved from `6ce8be104767` to `da5981f6ec0b` and every record but the stage-end sweep was regenerated once on it (`docs/version_2_revision_43_changes.md`): `M11.C.48` now fails on both families. Revision 40, built 2026-10-28 on `main` from the stage 3 milestone review's triage, changed wording only (transcript phrases paraphrased, `M5.F.4` states `FE04.13`'s two kinds, `M13`'s `M11.C.49` exit restated); the engine hash did not move (`2a3b84e2d89a`). Revision 39, approved by the owner 2026-10-25 as `docs/spec_revision_39_draft.md`, is applied in full on `main`, in parts, each with its code: at stage 3 step 6 (2026-10-25) T-130's wording (`M1.F.6a`) and T-014's four answers (`M8`, `M8.1`, `M8.5`, `M13.1`, `M11.C.7`, `M7.D.1a`), with `avoidance_available` removed and T-129's lint fixes; the engine hash moved from `66dcbb1e8ba5` to `f90285984f32` and no result changed (the golden fingerprints hold); at step 7 (2026-10-26) group C (`M1.A.5f` displaced coping, `M1.A.5e`'s and `M1.A.4k`'s `floor_held`, `M7.E.1f`, `M6.I.4a`), the hash moved to `b8e9ad1333e2` and the golden fingerprints moved; at step 8 (2026-10-26) group F (`M5.F.4`'s demand share, the assertion threshold retired, `M5.F.2c`, `M11.C.29`'s arm, `M11.C.49` new), the hash moved to `b819660f58b9` and the golden fingerprints moved; at step 9 (2026-10-26) group A (`M4.C.1d` contact returned by a relieved receiver, `reciprocal_contact_share`, `M4.D.3a`'s `TRIANGLE` at layer 0), the hash moved to `2a3b84e2d89a` and the golden fingerprints moved; at step 10 (2026-10-26) T-128 (`M17.E.1`'s named ratio, in the sweep tool and `config/sweep/sweep.md`) and MR-23; the engine hash did not move (`2a3b84e2d89a`), every record was regenerated once on it, and the open batch closed (`docs/version_2_revision_39_changes.md`); at step 11 (2026-10-27) the stage-end sweep ran with the named ratio, split across both Macs (`docs/version_2_sweep_record.md`): T-128 closed, `M17.E.1` partial until `M11.C.49` is swept (T-144), the engine hash unchanged. At step 12 (2026-10-27) the reviews of stage 3's whole diff: no rule changed and the engine hash did not move; three findings whose fixes move it (T-145, T-146, T-147) wait for the milestone review's triage; the triage (2026-10-28) put them in one batch with the owner's T-131 (b) and T-133 (b) and T-149 (`docs/todo/2026-10-28_stage_3_milestone_review.md`). Revision 38, built 2026-10-24 on `main` from the stage 2 milestone review's triage, changed wording only (`M1.D.2b` and `M1.D.2c` restated to the code, the owner's Q1; `M11.C.48` added to `M11.4`'s exceptions, Q6; corrections in `M1.F.6a`, `M10` and the register); nothing a run computes changed and the engine hash did not move. Revision 37, built 2026-10-23 on the branch `stage-2-plan-pilot`, added stage 2's detection-limit rule and sample rule to `M17.A.1` and `M17.E.1`, wording in `M17` only; nothing a run computes changed and the engine hash did not move. Revision 36, built 2026-10-22 on the branch `claude/recursing-bohr-2393a1`, built version 2's group G (G1, G2) and the Holmes–Rahe event sizes (`M1.D.2b`, `M1.D.2c`, `M1.F.6a`, `M7.C.1l`, `M7.D.1a`, `M7.D.2a`, `M11.C.48`), each choice decided as recommended under the owner's instruction of 2026-10-22, approved by the owner 2026-10-24 (T-125). Revision 35, built 2026-10-22, moved the revision history out of the spec into `docs/spec_revision_log.md`, and the inline revision narratives of 129 requirement bodies with it (and, the same day, 44 more sentences a longer phrase list found), each kept, moved or rewritten as the current rule (`docs/spec_revision_35_dispositions.md`, 262 rows, approved by the owner 2026-10-22); no rule changed (model ergonomics, Part E). Revision 34, built 2026-10-22, changed wording only: the quotations of the 1979 tapes and the Kerr–Bowen interviews paraphrased and cited (SP-1), and the stage 1 milestone review's wording fixes. Revision 32 built 2026-10-17 (finishing version 1: Claude's recommendations, reviewed by the owner one at a time the same day, with `M4.C.1c`'s nodal-event spike added; plan §4.12); revision 31 built 2026-10-16 (part A documents; part B code, config and every record regenerated, with the owner's version 1 premise decisions); revision 30 built 2026-10-14 (revision 12 approved 2026-10-07; revisions 24–30 reviewed by the owner 2026-10-16)** |
| `docs/version_1_status.md` | **where version 1 stands**: every `M11.C` criterion under V1–V3, generated from the records and `docs/version_1_dispositions.md` | **frozen at the tag `version-1` (`d62cfdc`, owner 2026-10-19); a test checks it and the dispositions against the tag and runs the tag's status tests** |
| `docs/model_explainer.md` | **every part of the model** — object, field, move, gate, invariant, clock, test — what it is for, what it does, and which chapter it implements | **start here to understand the model** |
| `docs/agent_model_proposal.html` | the architecture proposal, §10 = corpus reconciliation | current |
| `docs/bowen_individual_family_model_spec.md` | **v1.2, FROZEN** — describes the retired grid engine only | historical record; do not extend |

**Read first, to spend fewer tokens.** To look up an ID, grep `docs/spec_index.jsonl`: one JSON line per
requirement with its first sentence, spec lines, grade tags, sources, the ledger entries that cite it, the
code whose `Spec:` line names it, its tests and its coverage status; `docs/theory/ledger_index.jsonl` does the
same for ledger entries, `_LEDGER.md`'s and the per-finding entries of the extraction files (its `file` field says
which). To read a module's current rules, open `docs/spec_rules/M04.md` (one file per
module; the spec with its status line, tags, quotations and revision sections removed, nothing else). Open the full spec
only to change it or to see its sources and history. These files are generated (`tools/spec_index.py`,
`tools/spec_rules_view.py`) and checked current by tests; never edit them. `TODO.md` is a generated index of the open
items, one line each (ID, date, title, record), under **Now** or **Later**; an item's text is in its record under
`docs/todo/`, where it is ticked to close it (`tools/todo_index.py` then moves its line to `docs/TODO_archive.md`).
`CHANGELOG.md` opens with a generated index of its entries and holds the entries from 2026-10-26's stage 3 step 8 on;
the earlier entries are in `docs/CHANGELOG_archive.md`, verbatim. `tests/test_todo_index.py` keeps the generated
parts current and `tests/test_doc_size.py` keeps both live files under a size limit.

**Six corpus sources, and they are not one evidence class.** The 22 papers (1957–76, `L`); the 1979
video lectures (`Tape N`, **ASR — no verbatim quotation**); the Kerr–Bowen interviews (`KB`, **ASR — no verbatim
quotation; cite the interview, not the man**); Kerr & Bowen, *Family Evaluation* (1988) — `FE`, written, and its
**Epilogue is Bowen's own, the latest primary Bowen text in the project**; Kerr, *Bowen Theory's
Secrets* (2019) — `KS`, written, some chapters `[K-ext]`; and the DSI literature. **Every claim carries
an attribution grade as well as an evidence grade** — `[B]` Bowen, `[K]` Kerr, `[K-ext]` Kerr's own
extension, which **must never be attributed to Bowen**. `model_explainer.md`'s citation legend has the
full table.

**Two rules that follow from the corpus and are easy to violate by accident:**

- **Every claim in `model_explainer.md` carries a grade** — `[T]` textual, `[M]` mechanism, `[D]` direction,
  `[#]` a stated quantity, `[I]` invented, `[X]` contested. **If a constant in the code is presented as
  sourced and does not appear there, treat it as `[I]` until proven otherwise.**
- **The corpus supports directions, orderings and mechanisms — almost no magnitudes.** There is no
  instrument, no rater procedure, no comparison group and no number assigned to a person anywhere in the
  22 chapters. Never present an invented constant as sourced. If a parameter needs a citation, check it
  against `_LEDGER.md` first; two of the numbers previously treated as calibration targets turned out to
  be manufactured from illustrations Bowen explicitly bounded.
- **Acceptance tests assert a direction of difference between two arms, not an absolute threshold** — and
  each must be proved failing by mutation before it counts as coverage.
- **Patterns emerge; they are not written** (revision 11, `model_explainer.md` §19). Set: two given drives — toward connection, away from being smothered or controlled — and what each act does. Random: exogenous events, selection noise, initial conditions. Emergent: conflict, over/underfunctioning, distance, cutoff, projection, fixed triangles, learned from felt relief by agents that do not know consequences. A rule that writes one of these outcomes is a defect. Every criterion carries a class in `M11.5` — a **premise** passing confirms the code and must never be reported as a finding.
- **The model must never be presented as speaking about a real family** (`M11.F.9`). In particular: a run
  whose parameters were tuned to reproduce a known history is a **fit**, not a comparison — fitting one arm
  breaks the error cancellation the two-arm design depends on, on one side only and invisibly. Initial
  conditions are not nuisance parameters; they are what the intervention acts on, so their error flips the
  sign at a regime boundary rather than cancelling. Imported families run as **ranges**, and the output is an
  **envelope** (`M15.D`).

## Key rules

- Simulation parameters (thresholds, rates, constants) belong in config or named constants — not magic
  literals scattered through `src/bowen/`.
- Keep the simulation engine pure: **no UI, no file I/O.** `M16.B` makes this a requirement: the engine emits log
  records through a caller-supplied interface and never writes them itself, and logging must be a pure observer —
  same seed with logging on and off reaches the same final state.
- **The grid engine is retired** (version 2 stage 1, S1.6, owner decision D4): `src/engine.py`, `src/main.py`,
  `monte_carlo.py`, `tests/test_simulator.py` and `docs/model_config.md` were removed from the tree and are in git
  history and at the tag `version-1`; its rules (Axiom 1, its `sim_audit.csv` writes, its family-size draw) no longer
  apply, and `tests/test_repo_layout.py` fails if it comes back.
- **Work goes on `main`** (owner decision 2026-10-21). From 2026-10-06 to 2026-10-21 the work was on `plan-phase-b`
  while `main` stood at spec revision 11; on 2026-10-21 `main` was fast-forwarded to it. `plan-phase-b` stays on GitHub
  as history and takes no new commits. A session's own branch merges into `main`.
- **No test asserts on prose** (effort budget rule 5). A test may assert on structure a document restates — the
  spec revision, a repo path, a count, the engine hash, a generated file equal to its generator's output — and on
  the ASR no-quotation rule. It must never judge a wording: D5a and D5b were retired on 2026-11-10 for it
  (`docs/plan_doc_claim_tests.md`), so do not rebuild them.
- Run tests before declaring done: `python3 -m pytest tests/`
- **A red suite is never normal — 1473 tests: 1470 pass and 3 are strict expected failures** (each marked `xfail(strict=True)` with a reason naming why, so it turns the suite red if it starts passing: `M5.B.3`'s third-party cost (G1's A6); `test_m5d3_abort_is_the_usual_first_outcome`, which no longer holds after spec revision 41's T-133 (T-158, kept as a finding by the owner 2026-10-31); and one **record-current test** whose record is not rerun on the current engine by decision — the version 2 sweep record, which reruns at the stage-end sweep (the version 2 definition plan's step 7). The emergence audit record's and the audit's scope test's marks came off at the plan's step 5 (2026-11-11, T-213; the owner's Q7 (a) of 2026-11-09). Spec revision 57's eleven marks came off at the plan's step 4 (T-185, T-206), the one regeneration of every other record on the engine hash `d3779684dd53`, run 2026-11-10 to 11 split across both Macs, the Phase D ensemble part II and mutation records on Mac Mini1 (the owner's Q8 (a) of 2026-11-09; `docs/version_3_revision_57_changes.md`)). Stage 3's open batch closed at step 10: `tests/open_batch.py`'s `OPEN` is None and no test carries its mark. Measured 2026-11-11 on this Mac with `python3 -m pytest tests/`, at spec revision 58 and after the step 5 audit; counts are as collected by the default run; the 58 `ensemble`-marked criterion tests are deselected by `pytest.ini` and run with `python3 -m pytest -m ensemble`). The dated history of counts is in `CHANGELOG.md`, in the entry of 2026-10-22 that moved it out of this file.

## Token budget — how to read this repo (owner rule, 2026-10-22)

Every tool result stays in context for the rest of the session, so how a file is read is part of the work.

- **Read by index, not by file.** `docs/spec_index.jsonl` (one line per requirement), `docs/theory/ledger_index.jsonl`
  (one per ledger entry), the entry index at the top of `CHANGELOG.md`, and `TODO.md`'s one line per open item.
- **Grep to locate, then read a line range.** Never read a file whole to find one thing in it.
- **A module's rules come from `docs/spec_rules/`.** Open the full spec only to change it.
- **Never read whole:** `CHANGELOG.md`, the `docs/todo/` records, the spec, `docs/model_explainer.md`, or
  `docs/theory/_LEDGER.md`. Take the index line, then the one entry, item or line range it points to.
- **Wide reads and reviews go to a subagent** that returns a conclusion, not the files.
- **Summaries are generated by a tool and checked by a test** (`tools/todo_index.py`, `tools/spec_index.py`,
  `tools/c16_report.py`), never hand-written.
- **The size limits in `tests/test_doc_size.py` are the enforcement.** A file that outgrows its limit is split or
  archived, not exempted.
- **One task per session, started from a brief;** a new session once a batch is pushed.
- **A session that finishes its work starts the next session from a brief** (owner rule, 2026-10-24; changed
  2026-11-06 so the owner no longer copies and pastes): the brief goes into a new Claude Code session via
  `start_session`, in its own worktree from `origin/main` — a card the owner clicks, or started at once when the owner has
  already said to go ahead — not into a fenced block for the owner to paste. It names the task and the plan step, the
  files to read (by index and line range), what is already found or decided, what is pending the owner's review, the
  hard rules (the engine hash, the suite count, commit to `main` and push), and the point at which to stop. A brief
  pasted into a session is started the same way unless the owner says to run it there.

## Documentation stays current with the build (owner rule, 2026-10-10)

The documents are the contract the owner reviews runs against, so a document that disagrees with the code is a
defect, not a lag.

- **Same commit.** A change that alters what a run computes (`src/bowen/`, `config/bowen/`, a readout) updates, in
  the same commit, every document that describes it: the spec (`docs/bowen_agent_model_spec_v2.md`),
  `docs/model_explainer.md`, `docs/readouts_phase_c.md`, `config/bowen/constants_changes.md`, and any generated
  record whose numbers it changes. A commit that changes behavior without the matching document change is not done.
- **Status lines are claims (P6).** When a proposal is built or a decision is propagated, update the status line at
  the top of that document in the same commit.
- **The index.** `docs/SPEC_REVIEW_CHECKLIST.md` lists every file under `docs/` and `config/bowen/`. Add, rename or
  remove a document and update the checklist in the same commit; `tests/test_doc_checklist.py` fails otherwise.
- **The reviewer reading list.** `docs/REVIEWER_READING_LIST.md` restates other documents' status: which version to
  review, what is built, which proposals are decided, and its known problems. A change to any of these updates it in
  the same commit.
- **Before reporting work done**, state which documents were updated, or why none needed to be.

## Generated records and reviews — keep the loop cheap (2026-10-09, learnings P38)

- **Never rerun a whole record to change a label, a comment or a renderer.** The mutation and sweep records are
  rendered from `docs/records_cache/` (`tools/record_cache.py`); a result is rerun only when the engine hash or that
  mutant's or setting's own edits change. Only code that changes what a run computes belongs in
  `tools/mutant_runner.py` (it is in the engine hash); everything else goes elsewhere. Add a mutant, run
  `python3 tools/mutation_record.py`, and only that mutant runs.
- **Batch fixes before any rerun that is still slow** (the ensemble record, an engine change): collect every finding
  from a review first, then regenerate once.
- **Numbers in prose come from the records.** A report section that quotes record numbers is generated by a tool
  (`tools/c16_report.py`) and checked by a test, so reviewers do not spend passes on arithmetic.
- **Size reviews to the change.** One full `learning-qa` pass plus one correctness pass per batch, before push. A
  fix-loop re-sweep gets a narrow brief (the fix diff and the claims it changes) on a cheaper model (`model:
  "sonnet"`). If a sweep returns three or fewer minor items, record them in `TODO.md` and move on (owner rule).
  State the cost (wall time, tokens) before proposing another loop.
- **The full constant sweep runs only at a stage's end** (owner, 2026-10-23; plan §2.10, D17), before its milestone
  review or a tag. Between, a change is swept only on the criteria it targets (`tools/version_2_sweep.py --criteria`);
  the central ensemble records still catch a changed verdict.
- **Milestone reviews** (`docs/milestone_review_brief.md`, approved 2026-10-21): at the end of version 2's stages 2
  and 3, before any tag and before an outside review (the brief's §2), a session without the build history
  reviews the whole model against the brief's four lenses, theory fidelity first, and writes one report to
  `docs/cycles/` on its own branch. Triage happens here.
