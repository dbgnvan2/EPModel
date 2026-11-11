# Review: EPModel (whole repo) — 2026-11-11

Read-only review. No code, test, config or document was changed; this file is the only thing written.
Seven independent reviewers each read one slice; an eighth tried to refute the top 16 findings against
the code. Findings marked **verified** survived that pass. Nothing was executed by the reviewers; the
default test suite was run once at the end (see "Test run").

**The working tree was changing during the review.** 7 files were modified at the start and 14 by the
end (`CLAUDE.md`, `CHANGELOG.md`, `tests/test_emergence_audit.py`, the emergence audit record and
others), so another session was editing this checkout. Document findings D1, D2 and P5 were read from
that uncommitted state and may describe a half-finished edit.

## Verdict

The engine is well built: clean layering, no file I/O in the engine, deterministic draws, one mechanism
per file, and tests that assert a direction between two arms. It has one real bug that affects every
life-course run (B1). The test suite is not too large where it tests the model; about a third of it
tests documents, records and tools, and roughly 1,200 to 2,000 lines of that can go. The documentation
is too large and partly wrong: the contract is 4,700 lines in 3 files, about 26,000 lines in 65 files
are history of finished work, the spec's revision history is hand-restated in seven places, and several
status lines say "not built" for things that are built. On purpose: the model is honest about its
results, but it does not yet show Bowen's patterns emerging from learning, and nobody can run the
learning family and read what happened without writing code.

Fix first: B1, then cut the restated history (D3, D4), then decide what to do about P1 to P3.

## Size, measured

| Area | Files | Lines | Note |
|---|---|---|---|
| `src/bowen/` | 79 | 16,205 | the model |
| `tests/` | 109 | 27,244 | 1.7 lines of test per line of model |
| `tools/` | 45 | 14,410 | nearly the size of the model |
| `config/` | 30 | 2,435 | |
| `docs/` | 290 | 293,507 | 147,000 of these are cached JSON records |
| `docs/theory/` | 91 | ~696,000 words | corpus extractions; not judged |

`CLAUDE.md` is about 4,300 words; one table cell (line 28) is 13,011 characters.

## Coverage

| Area | Correctness | Security | Structure | Tests | Docs | Purpose |
|---|---|---|---|---|---|---|
| `engine/` per-tick core + `policy/` (24 files) | covered | n/a | covered | covered | — | covered |
| `engine/` life course, events, invariants (21 files) | covered | n/a | covered | partial | — | partial |
| `scenario/`, `io/`, `run.py` | covered | covered (`io/` only) | covered | partial | — | partial |
| `ensemble/`, `readouts/`, `render/` | covered | n/a | partial | covered | — | partial |
| `config/bowen/` | partial | n/a | partial | — | — | partial |
| `tools/` | not reviewed | not reviewed | partial (2 of 6 largest read in full) | partial | — | — |
| `tests/` (35 of 107 files read in full) | — | — | — | partial | — | — |
| Root docs, `docs/` top level | — | — | — | — | partial (status lines of all, 12 opened) | — |
| `docs/theory/`, `docs/records_cache/` | not reviewed | — | — | — | not reviewed | — |
| `experiments/llm_agents/`, `tests/experiments/` | not reviewed | not reviewed | listing only | not reviewed | — | — |
| `papers/` | not reviewed | — | — | — | not reviewed | — |

Not reviewed means not reviewed: `tools/` line-level correctness, the LLM experiment, the content of the
theory extractions and whether the spec matches the corpus. Four of the eight largest engine test files
were not read in full. Excluded: `.venv`, `.claude`, caches.

## Findings

### Blocker

**B1 · CORRECT · verified (by the lead reviewer and by reading) · `src/bowen/engine/life_course.py:226-233`**
`_introduce` builds one `TieBelief` object and assigns that same object to every belief entry of a
newcomer and to every other person's entry for each new tie. `beliefs.update_beliefs`
(`beliefs.py:121-125`) mutates in place. After any pairing, birth or coach succession, updating one
person's belief about one tie changes other people's beliefs and beliefs about other ties. Founders are
not affected (`initialise_beliefs` builds one object per entry). Fix: construct a new `TieBelief` per
entry, and add a test that updates one belief after a birth and asserts no other entry moved. The fix
changes what a run computes, so it moves the engine hash and every life-course record.

### Major — code

**C1 · CORRECT · verified · `engine/act.py:124-130`, `iposition.py:162, 422`, `policy/policy.py:171-174`**
A person in an I-POSITION sequence toward T who selects I-POSITION toward someone else has the running
sequence overwritten, with no `end` record. If the first sequence was in RESOLVE, T keeps the pull-up
with no follow-up and no revert, which `M5.D.6` forbids. Nothing in legality, the policy or the
redirect prevents it. `M5.D.9` does not say what should happen; that is the owner's call. It may bear on
T-158's expected failure.

**C2 · CORRECT · verified · `iposition.py:212-214, 400-401`**
Two exits skip the cleanup the normal paths do: `mover_died` in RESOLVE leaves the target's pull-up in
place, and a twice-deferred abort ends without `restore_prior_balance`. No test names either exit.

**C3 · CORRECT · verified · `ensemble/runner.py:207-213`**
The "a required move that never occurs fails" gate reads the treatment arm only. A move scripted in the
baseline arm (CUTOFF in `M11.C.18`; same shape in `M11.C.20`, `M11.C.23`) is never gated. No current
verdict is wrong because of it.

**C4 · ERROR · verified · `ensemble/runner.py:84-95`, `criteria_d2.py:393, 404`, `criteria.py:733-735`**
Exclusion counts the arms record "so they are not silent" ("the child died before 20", "children never
seen at the ages read", three `M11.C.49` keys) are stored and never reach a verdict, a tool or a
record. This is the owner's own rule 6 (surface, never silently drop).

**C5 · DATA · verified · `scenario/header.py:27-55`**
The log header's `config_hash` omits event-kind components, the accommodating/displacing sets, contact
classes, event sizes, `family.life` and the policy rules. Two runs with different config can carry the
same header hash. The records are covered by a different hash (`tools/ensemble_record.py:61-68`); a run
log is not.

**C6 · ARCH · high confidence, not in the challenge pass · `tools/ensemble_record.py:46-69`**
The engine hash is one digest over all of `src/bowen` (except `render/`) and every `config/bowen/*.md`.
A change to one criterion's arm invalidates every cached result of every record. Revision 57 changed
two criteria and regenerated everything across both Macs. Keying each result on an engine hash plus
that criterion's own source would remove most of that cost.

### Major — tests

**T1 · verified · `tests/test_review_rev31.py`** (159 lines, 16 tests). Six blocks assert exact wording
of spec, explainer, plan and docstring sentences. The repo bans this (effort budget rule 5), and D5a/D5b
were retired for it on 2026-11-10. Delete the file; its structural tests are covered elsewhere.

**T2 · verified · `tests/open_batch.py`, `tests/test_open_batch.py`, hooks in `conftest.py`,
`test_version_2_status.py:30-38`, `test_spec_consistency.py:404-437`** (about 540 lines, 11 tests).
The stage 3 open-batch mechanism has been closed since 2026-10-26 (`OPEN = None`) and later batches used
plain `xfail(strict=True)`. It now only maintains itself. Delete it.

**T3 · high confidence · `test_spec_consistency.py:265-441`, `bowen/test_engineering_gates.py:121-160`,
`bowen/test_log.py:147-212`, `test_doc_claims.py` (D1)**
Several tests fail on every ordinary commit until a hand-kept number is edited: the suite count in
`CLAUDE.md`, an exact count of `Spec:` references (1097), a set of about 140 constant names, the spec
revision in 4 places and the requirement count in 10 documents. None can catch a wrong result. This is
the main cost of the suite, and it is the test-side half of D3.

**T4 · medium · `test_version_1_status.py:51-64`, `test_record_changes.py:31-70`,
`test_spec_consistency.py:317-320`** Each run re-runs the `version-1` tag's own tests in a subprocess,
rebuilds nine frozen revision pages from git, and runs `pytest --collect-only` inside the suite. They
re-check history that cannot change.

### Major — documentation

**D1 · CONFLICT · verified (working tree) · `docs/proposal_social_cues.md:3-6`,
`docs/spec_revision_55_draft.md:3-4`, `docs/SPEC_REVIEW_CHECKLIST.md:218, 262, 264`,
`docs/REVIEWER_READING_LIST.md:196-199`**
Documents built or applied between 2026-11-08 and 2026-11-11 are still described as "not built" or "not
yet applied". `src/bowen/engine/cues.py` exists and the spec is at revision 58.

**D2 · CONFLICT · verified (working tree) · `docs/REVIEWER_READING_LIST.md:421, 435`,
`CHANGELOG.md:118-120`**
The version 2 sweep record is said to stand on engine hash `927550dfcb8a`. The record itself
(`docs/version_2_sweep_record.md:25`) says `0b363cb3186c`, two hash moves earlier. Also
`docs/model_explainer.md:2332` says 10 of 13 audited rows survive with learning off; the record shows 9.

**D3 · SCOPE · high confidence**
The spec's revision history is hand-restated in at least seven places besides the revision log and the
changelog: `CLAUDE.md:28` and `:96`, `README.md:54` (one paragraph of about 2,300 words),
`docs/REVIEWER_READING_LIST.md:3, 226-436`, the spec's line 4, `docs/theory/_STATUS.md:3`,
`docs/spec_revision_log.md:3`. One hash appears 47 times in 19 files. The copies already disagree (D1,
D2). This is the cause of the stale statuses, not a separate problem. Keep history in the revision log
and the changelog only; everywhere else, one sentence and a pointer.

**D4 · SCOPE · high confidence · `CLAUDE.md:28, 96`**
About half of the always-loaded instruction file is revision history, out of order, in one table cell.
Every session and every subagent pays for it, and the rules that matter sit below it. About 1,900 to
2,200 of its 4,300 words are instruction.

**D5 · SCOPE · high confidence · `docs/` top level**
Of 95 top-level files: 3 are the current contract (4,700 lines), 13 are current generated records, and
about 65 files (about 26,000 lines) are history of finished work: 15 finished plans, 20 applied
proposals, decisions and drafts, 12 past reviews and reports, 8 frozen per-revision comparison pages
(8,150 lines, each renderable from git), 7 documents of the retired grid engine. Git history and the
tags keep all of it.

### Major — purpose fit (for the owner's later review)

**P1 · verified (records) · `docs/emergence_audit_record.md:93-105`, `docs/version_2_status.md`**
Most passing "emergent" criteria do not need learning. Of 13 audited passing rows, 9 still pass with all
learning removed. Five rows need learning (`M11.C.39`, `M11.C.10[family of origin]`, `M11.C.55`,
`M11.C.34`, `M11.C.2[twelve-person]`), and two of those are robust under the sweep. The criteria for
learned triangles (`C.42`, `C.44`, `C.45`), conflict following anxiety (`C.12`), projection lowering a
child's functioning (`C.6`, `C.54`), reciprocity (`C.21`) and coaching raising a level (`C.17`) fail.
The project records all of this; the summary block of the status page does not show the
learning-dependent tally beside the pass count.

**P2 · high confidence · `engine/moves.py:121-127, 245-251`, `learner.py:130-139`**
Which pattern is learned is decided by the relative size of each act's invented immediate relief inside
a 3-week horizon. DISTANCE relieves more than TRIANGLE at once (−4.97 against −1.29 in the Phase C
record), so triangling is not learned. The "set" layer is fixing the "emergent" layer.

**P3 · verified as code fact; consequence unmeasured · `learner.py:120, 130, 215-253`, `tick.py:134, 243`**
The learning signal (a) includes ordinary decay toward the chronic floor with no baseline, so any act
taken while anxious is credited with relief it did not cause, and (b) reads other people's true anxiety
change and true functional level, not what the learner could perceive. Both are stated as intended in
the code. (a) needs a mutant to show whether it carries `C.39`, `C.16` or `C.34`. (b) sits under
`C.10[family of origin]`, one of the two robust learning-dependent passes.

**P4 · verified · `src/bowen/run.py:37-75`**
The only run command executes the scripted Phase B family, in which nobody chooses anything.
`tools/phase_c_trace.py` runs a policy-driven family and renders a trace, but only the Phase C family.
No command runs the twelve-person family with the life course and shows what happened.

**P5 · verified (working tree) · `docs/version_2_status.md:31, 114`**
`M11.C.48[phase_d]` is shown as proved by a learning mutant turning red and counted toward W3, while the
audit shows the row passes with all learning removed, and the same row's disposition text still says
the mutant survives. `CLAUDE.md` says the dispositions are the owner's at the plan's step 6, so this
may be pending, but the generated page contradicts itself today.

**P6 · high confidence · `config/bowen/constants.md`**
216 constants, all graded invented, against at most 5 learning-dependent passing results. The sweep is
8 samples on an older engine, and W4 is unmet for all 16 passing directional rows. The model is not
unfalsifiable (25 rows fail and are recorded), but the passing set does not constrain the constants.

**Concepts, as built:** differentiation is an input number that barely moves; triangles exist as an act
but not as a learned pattern; projection's focus mechanism is written (four steering rules plus a
birth-order and sex multiplier at `life_course.py:807-819`) and its effect on the child is not produced;
multigenerational transmission is written as inheritance; cutoff is a mechanism; sibling position is
stored and never read; societal process is absent; chronic anxiety is a floor. The family is one fixed
cast of 11 people and a coach.

### Minor

- **Shared across slices:** act sets that are editorial content live in Python and are duplicated in
  config (`engine/sinks.py:40-45`, `ensemble/criteria.py:172, 812`, `iposition.py:95`); the two
  definitions of "automatic" have drifted (7 kinds in `criteria.py:172`, 10 in `event_kinds.md`).
- `scenario/constants.py:46`: `fast_tick_weeks` is a required constant no code reads. **verified**
- `engine/invariants.py:203, 235`: `M6.I.2` and the tie half of `M6.I.8` are true by construction and
  cannot fire. **verified**
- `engine/consolidate.py:51-55`: a snap back up to the chronic floor is logged as positive
  `acute_decay`, which the ledger classes as a sink.
- `engine/appraise.py:332-357`: `M5.C.1`'s marital-distance gate reads the reference family only.
- `engine/moves.py:192-194`: `_desc` breaks ties toward the longer id when one id is a prefix of
  another (`x_successor` over `x`), against its comment.
- `engine/iposition.py:327`: logs the full `exchange_gain` when the level was clamped at 100.
- `engine/act.py:113-163`: `params=None` silently degrades I-POSITION; latent, one caller.
- `scenario/constants.py:301-309` and siblings: `nan`, `inf` and out-of-range values load; a NaN
  `invariant_tolerance` would switch the invariants off.
- `scenario/scripted_source.py:127-159`: a bad binder target fails mid-run without file and line.
- `scenario/family.py:140-143`, `io/load.py:253-258`: the `partners:` path is unchecked; matters only
  once families are imported (`M15`).
- `ensemble/criteria_d2.py:391-393`: a child who died before 20 enters `M11.C.6` as development 0.0.
- `ensemble/criteria.py:570-573`: `M11.C.25`'s null can pass with no pole ever forming.
- `ensemble/criteria_d.py:529-533`: partial ties in the "most projected child" pick break by id order.
- `engine/params.py` and `scenario/constants.py`: each constant is declared in four places and the spec
  IDs already disagree (`standing_load_gain`, `functional_level_floor`).
- `src/bowen/`: 416 "revision N" comments in 44 of 77 files carry history where the current rule
  belongs. Comments are outside the hash, so a comment-only pass is free.
- `tools/`: no library layer; record scripts import from each other's entry points, with a cycle; three
  different functions are named `engine_hash`. `tools/c2_matched_probe.py` is named by no test.
- Config is re-read from disk on every arm of every seed (`io/load.py`, no cache).
- `requirements.txt:6`: CI installs `anthropic` for a closed experiment. CI's NumPy is unpinned while
  the sweeps pin 1.26.4. CI never runs `-m ensemble`.
- `tests/bowen/test_determinism.py:58`: cross-process determinism under a changed hash salt is tested
  only on the scripted run, which draws nothing.
- `tests/bowen/test_log.py:31-43`: passes vacuously by its own docstring.
- Five closed probe tools and their tests (about 2,700 lines of tools, 555 lines of tests).
- Root leftovers of the retired grid engine: `gemini.md` (instructs an agent to verify rules
  `CLAUDE.md` says no longer apply), `monte_carlo_results.txt`, `data/`, six documents under `docs/`.
- `TODO.md`: T-111 ("No LEARNINGS.md") and T-168 (names a test file that does not exist) are stale.
- `docs/SPEC_REVIEW_CHECKLIST.md` files the current status page and records under "not for review".
- `README.md`: "Running" gives the test command and the scripted run only.
- 12 document filenames contain spaces and em dashes.

## Systemic patterns

1. **Hand-restated state.** The same revision, hash, count and status is written by hand in many
   documents, and tests then require every copy to be edited on every commit (D3, D4, T3, D1, D2). The
   project already generates its indexes and records; the status lines are the part still done by hand,
   and they are the part that is wrong. One generated status file, one test.
2. **Finished work kept as live documents and live tests.** Plans, proposals, per-revision pages, probe
   tools, the open-batch mechanism and the revision 31 test all outlived what they were for (D5, T1, T2,
   T4, the probes).
3. **Counts recorded and not surfaced** (C4, and the readouts with no "no pole" or "died" count).
4. **Outcomes steered by written rules while classed as emergent** (P1, P2, the focus rules, the
   birth-order multiplier, `settle_functioning`). The project documents most of these; the gap is in
   the headline numbers.
5. **Cleanup done on the normal path and skipped on the exits** (C1, C2).

## Is there too much testing?

| Class | Files | Lines | Tests | Share |
|---|---|---|---|---|
| Engine behaviour | 49 | ~14,250 | ~805 | 53% |
| Acceptance criteria | 7 | 1,887 | 129 | 7% |
| Documents, records, repo process | 24 | 4,915 | 235 | 18% |
| Tests of `tools/` | 15 | 4,170 | 162 | 16% |
| LLM experiment | 10 | 1,541 | 61 | 6% |

The engine and criteria tests that were read are worth their cost. Delete outright: about 1,200 to
1,300 lines and 50 to 55 tests (T1, T2, four closed probe tests, the tag re-run, the suite-count test,
the historical-fixture tests). Optional: about 800 more (SSH driver tests, report-fragment assertions).
Owner's call: `tests/experiments/llm_agents` if that experiment is finished. Add one test:
cross-process determinism of a policy run.

## Is there too much documentation?

Yes, outside the contract and the evidence base. Keep in place: the spec, the explainer,
`readouts_phase_c.md`, the revision log, the 13 current generated records, the generated indexes,
`docs/theory/`, `docs/todo/`. Archive (about 55 files): the 15 finished plans, the 20 applied
proposals, decisions and drafts, the past reviews and phase reports, the eight frozen comparison pages,
the 17 Phase C QA-gate logs in `docs/cycles/`. Delete: `gemini.md`, `monte_carlo_results.txt`, `data/`,
the six grid-engine documents. Before moving anything, check `tests/test_doc_claims.py`,
`tests/test_doc_checklist.py` and `tools/record_changes.py` for path dependencies; they were not read.

## Refuted in the challenge pass

- `tests/test_doc_size.py:98-100` reads today's date, so it was claimed the suite can go red with no
  commit. The call exists, but it only acts on ticked items in `TODO.md`, and the generator moves
  ticked items to the archive, so `TODO.md` holds none. Not checked: that the generator can never emit
  a ticked line.
- Three of the checklist lines cited for D1 (254, 256, 260) do not say "not built".
- A claim that `CLAUDE.md` states 5 strict expected failures against 3 marks in the tree: the committed
  file said 5 and the working-tree file says 3. Both match their own state; the tree is mid-edit.

## Questions only the owner can answer

1. Is a second I-POSITION during a running sequence illegal, redirected, or a restart (C1)?
2. Is `settle_functioning` (`moves.py:370-390`), which drifts any non-zero balance to a fixed pole, act
   physics or a written pattern? It is not listed among the acknowledged ones.
3. Is the learner meant to be exempt from the perception boundary (P3b)? `M4.B.2` does not say.
4. `M11.C.55` reverses with learning off partly by definition (a tie counts as a miss). Does the audit
   treat that as evidence of learning?
5. Should a minimum share of non-zero pairs be required before a within-run criterion can pass?
6. Is the LLM experiment finished, so its code, tests and the `anthropic` dependency can go?
7. `run_tick` emits a tick's records only after the invariants pass, so on a violation the log lacks the
   failing tick. Intended?
8. Do `actions/checkout@v7` and `actions/setup-python@v7` exist, and is CI green on GitHub? Not
   verifiable from here.

## Test run

`python -m pytest tests/ -q`, run once on the working tree as it stood during the review (uncommitted
edits from another session included), at low priority: **1470 passed, 3 expected failures, 58
deselected, exit 0, 10 min 28 s.** This matches the working-tree `CLAUDE.md` (1473 tests, 1470 pass,
3 strict expected failures). The 58 `ensemble` tests were not run. Wall time was measured under
`nice -n 19` with eight reviewers' worth of other load finished, so treat it as an upper bound.

The three expected failures: `M5.B.3`'s third-party cost (a tracked finding), T-158's abort outcome (a
tracked finding), and the version 2 sweep record being stale (a stale-record marker, not a finding).
