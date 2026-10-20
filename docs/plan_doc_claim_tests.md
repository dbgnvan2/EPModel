# Plan: tests that fail when a document restates a fact that has changed

**Status: DRAFT for owner approval, 2026-10-19. Nothing built.** Written after version 2 stage 1 step 7 (`7aa9fc5`).
No test in this plan exists yet. Each one is written, run red, and proved failing by its named mutation before it
counts.

## 1. Why

On 2026-10-19 six documents were found stale: README, the checklist's spec revision, `_STATUS.md`'s state line, the
explainer's front matter, `requirements.txt` and `spec_coverage.md`. The `/csdp` and `/chdp` documentation steps update
the documents a diff touches. These six were copies of a fact kept somewhere else, so no diff touched them when that fact
changed, and no test compared them. Where a test did compare a copy (the suite count in CLAUDE.md) it caught the drift.
README's Layout line "`tests/ 37 tests, all passing`" was not caught, because the pattern requires "tests pass" or "tests green".

The rule this plan applies: a document that restates a fact kept elsewhere gets a test that reads both and fails when
they differ. Every check lives in `tests/`, never only in `.github/workflows/tests.yml`.

## 2. Preconditions and order

- **Stage 1 step 7 is committed** on `plan-phase-b` (`7aa9fc5`, "one regeneration; no verdict changed"). Met.
- **The branch `docs-stale-fixes` must be on `plan-phase-b` before the build** (Q2). It fixes the checklist's spec
  revision, `_STATUS.md`'s state line, `requirements.txt` and README's "pending owner review" sentence. Without it, D1,
  D3 and D5a fail on items that branch already fixes. A trial merge (`git merge-tree`, tree `4c6f664`) has no
  conflicts. The owner decides that merge; this plan does not make it.
- Every result below marked "measured" came from a throwaway prototype of these checks, run on `7aa9fc5` and on the
  trial-merge tree `4c6f664`. The suite at `7aa9fc5` is 917 passed, 49 deselected.

## 3. The tests, and the files each reads

| ID | Test file | Reads | Compares against |
|---|---|---|---|
| D1 | `tests/test_doc_claims.py` (new) | the spec's front matter; `docs/SPEC_REVIEW_CHECKLIST.md`; CLAUDE.md; `docs/theory/_STATUS.md`; README.md | the spec's `version: 2.0, revision N` |
| D2 | `tests/test_doc_claims.py` (new) | README.md, CLAUDE.md; `git ls-files` | the files git tracks |
| D3 | `tests/test_requirements.py` (new) | every git-tracked `.py`; `requirements.txt` | the third-party top-level imports |
| D4 | `tests/test_spec_consistency.py` (changed) | the seven `CITING_DOCS` | pytest's collected count |
| D5a | `tests/test_stale_text.py` (changed) | README.md, CLAUDE.md, the checklist, `_STATUS.md`'s state line, every `docs/` status line | the `PENDING` list |
| D5b | `tests/test_doc_status.py` (new) | the checklist's entry labels; README.md; every `docs/` status line | each named document's own status line |

The word lists these tests use (retirement markers, status vocabularies, history entries, tools installed but not
imported) go in one data file, `tests/doc_claims.json`, not in Python source (global rule 9). The test files hold only
the code that reads them. `test_stale_text.py`'s existing `PHRASE` and `PENDING` stay where they are (§9).

Each check is a function that takes text and file lists and returns its findings, as `stale_text()` already does. The
real-repo test asserts the findings are empty. The self-tests run the same function on fixture text that carries the
mutation, so the mutation proof stays in the suite after the build. Each check also gets one **live mutation**: an edit
to the real file, run red, then reverted. The build record (§8) lists each one with the failure message it produced.

## 4. D1: the spec revision, wherever it is restated

The spec's front matter holds the current revision: `version: 2.0, revision 33`. Four places restate it. Measured at
`7aa9fc5` and on the trial merge:

| Place | What restates the revision | At `7aa9fc5` | After the merge |
|---|---|---|---|
| Checklist, spec entry | "the spec (v2.0, revision N" | 27, wrong | 33 |
| CLAUDE.md, the spec's table row | the largest revision the row names | 33 | 33 |
| `_STATUS.md`, the `▶ STATE` line | the largest revision the line names | none named | 33 |
| README.md, `## Status` section | the largest revision the section names | 33 | 33 |

**Why "the largest revision named".** These places also name past revisions: "revision 28 built 2026-10-13",
"revisions 24–30 reviewed by the owner", "version 1 is frozen at revision 32". No past revision can be larger than the
current one. So the largest number a place names is its claim about the current revision, and it must equal the
spec's. `test_requirement_counts_agree_with_the_spec` already uses this rule. A range counts as its upper end. A place
that names a revision larger than the spec's fails too. So a revision that is only planned is not named by number in
these four places.

Criteria:

- **D1.1** The spec's front matter is read for `version: 2.0, revision N`. If it cannot be read, the test fails.
- **D1.2** The checklist's spec entry states revision N.
- **D1.3–D1.5** The largest revision named in CLAUDE.md's spec row, in `_STATUS.md`'s `▶ STATE` line and in README's
  `## Status` section is N.
- **D1.6** A place that no longer names any revision fails, with a message saying the restatement is gone.
- **D1.7** A smaller (historical) revision does not fail; a larger one does.

Tests:

- `test_d1_spec_revision_restatements_match_the_front_matter`: the real repo. Covers D1.1–D1.5.
- `test_d1_a_moved_spec_revision_is_reported_in_every_place`: copies the five files to `tmp_path` and sets the spec's
  front matter to revision 34. All four places are reported. This is the drift the test exists for: the spec moves and
  the copies do not.
- `test_d1_the_checklist_drift_of_2026_10_19_is_reported`: the checklist's spec entry reads "revision 27", as it did at
  `7aa9fc5`. Reported.
- `test_d1_a_place_that_stops_naming_a_revision_fails`: `_STATUS.md`'s state line as it was at `7aa9fc5`, with no
  revision in it. Reported as gone (D1.6).
- `test_d1_history_does_not_trip_and_a_future_revision_does` (adversarial): "revision 28 built 2026-10-13" added to
  CLAUDE.md's row gives no finding; "revision 34" added gives a finding (D1.7).

**Live mutation:** set the spec's front matter to `revision 34`. The test goes red and names all four places.

## 5. D2: repo paths in README.md and CLAUDE.md exist

This would have caught `python3 src/main.py`. README at `423bbed` still had it in its Running block, and
`src/main.py      orchestration, UI, I/O` in its Layout block, after the grid engine was removed at stage 1 step 2.

- **D2.1 Candidates.** Every backtick span in prose, every line of a fenced block, and every relative link target. A
  token counts as a repo path if it starts with `src/`, `tools/`, `tests/`, `docs/` or `config/`, or ends in `.py`. A
  `-m` module (`python3 -m src.bowen.run`) is checked as `src/bowen/run.py`. A whole span that is an existing path with
  spaces in it (`docs/DECISIONS — FAMILY EVALUATION.md`) counts as one path. Tokens with `*`, `<`, `>`, `{`, `}` or
  `$` are patterns or placeholders and are skipped. A `::test_name` suffix is dropped.
- **D2.2 Exists** means git tracks it, as a file or as a directory holding tracked files. A bare `name.py` exists if
  any tracked file has that name. **Git, not the disk:** a stray untracked copy of a removed file must not hide the
  removal. A new file must be staged (`git add`) before this test sees it. The test fails, rather than skips, when git
  is not available.
- **D2.3 Retired paths.** A missing path passes only if the same sentence (in prose) or the same line (in a fenced
  block) says it is gone: "retired", "removed from the tree", "at the tag `version-1`", or "in git history". A sentence
  is split within its own paragraph, list item or table row, so a marker in a neighbouring row does not count.
- **D2.4 Not vacuous.** The scan must report checking CLAUDE.md's `tests/test_repo_layout.py` and README's
  `tools/record_cache.py`, and must report README's `src/main.py` as missing but marked retired. This shows the scan
  reads the files at all.

Measured: no finding at `7aa9fc5` or after the merge. Every missing path sits in a sentence that marks it retired:
README's `src/main.py`, and CLAUDE.md's `src/engine.py`, `src/main.py`, `monte_carlo.py`, `tests/test_simulator.py` and
`docs/model_config.md`.

Tests:

- `test_d2_repo_paths_in_readme_and_claude_md_exist`: the real repo (D2.1–D2.3).
- `test_d2_the_old_run_command_is_reported`: fixture text is README's two lines at `423bbed`, checked against a file
  list without `src/main.py`. Two findings.
- `test_d2_a_retired_path_passes_only_in_its_own_sentence` (adversarial): "`src/main.py` was retired." passes. "The
  engine was retired. Run `src/main.py`." fails.
- `test_d2_paths_with_spaces_and_module_commands_resolve`: the DECISIONS file and `python3 -m src.bowen.run` pass;
  `python3 -m src.bowen.gone` fails.
- `test_d2_the_scan_is_not_vacuous` (D2.4).

**Live mutation:** in README's Running section, replace "`src/main.py`, was retired at version 2 stage 1 and is at the
tag `version-1`" with "Run `python3 src/main.py`." The test goes red and names `src/main.py`.

## 6. D3: `requirements.txt` is the set of third-party imports

- **D3.1 Imports read.** The test AST-parses every git-tracked `.py` file. It reads `import x`, absolute
  `from x import y`, and `importlib.import_module("x")` or `__import__("x")` with a literal argument. That covers the
  brief's root, `src/`, `tools/` and `tests/`, plus `papers/sweep.py`, so a new directory cannot escape the scan.
- **D3.2 Not third-party:** `sys.stdlib_module_names`; `__future__`; relative imports; the repo's own top-level packages,
  meaning the first path segment of every tracked `.py` in a directory (`src`, `tools`, `tests`, `papers`); and the
  stems of tracked `.py` files under `tests/` and at the root (test helpers such as `act_literals` and
  `ensemble_toys`). Both lists are derived from the tree, so a new helper module needs no edit.
- **D3.3 Import name to distribution.** An installed import is mapped with
  `importlib.metadata.packages_distributions()`. An import that is not installed keeps its own name. Names are
  normalised as PEP 503 does (lower case, `-`, `_` and `.` treated alike).
- **D3.4 `requirements.txt` parsed.** Comments and blank lines are dropped. The name is the part before any version,
  extra or marker. A name listed twice fails.
- **D3.5 Installed, not imported.** `tests/doc_claims.json` lists tools that are installed but never imported, each with
  its reason: today only `ruff` ("the linter, run as a command"). An entry that is not in `requirements.txt` fails, so
  the list cannot hide a missing requirement.
- **D3.6** The requirements minus the allowlist must equal the imported distributions. The failure message lists both
  differences.

Measured: at `7aa9fc5` the imports are numpy and pytest. `requirements.txt` also lists pandas, networkx, pygame-ce,
matplotlib and python-dotenv, and lists numpy twice. After the merge it is numpy, pytest and ruff, with no finding.

Tests:

- `test_d3_requirements_match_the_imports`: the real repo.
- `test_d3_the_requirements_file_of_2026_10_19_is_reported`: fixture text is `requirements.txt` exactly as it was at
  `7aa9fc5`. The five unused packages and the duplicate numpy are all reported.
- `test_d3_an_unlisted_import_is_reported`: a fixture module with `import yaml`. Reported.
- `test_d3_local_stdlib_relative_and_helper_imports_are_not_third_party` (adversarial): `from . import x`,
  `import src.bowen`, `import act_literals`, `import tomllib` and `importlib.import_module("tools.spec_coverage")` give
  no finding.
- `test_d3_an_allowlisted_tool_must_be_listed`: `ruff` dropped from fixture requirements is reported (D3.5).

**Live mutation:** delete the `numpy` line from `requirements.txt`. The test goes red and names numpy.

**Limit.** An optional import inside `try: … except ImportError` counts as required. There is none today. If one is
added, it is either listed or given its own entry in the data file with a reason.

## 7. D4 and D5: suite-size claims and status claims

### D4: widen `test_claimed_test_count_matches_the_suite`

- **D4.1** The pattern finds present-tense claims: "N tests pass", "N tests passing", "N tests green", "N tests, all
  passing", "N tests, all green", "all N tests pass", and one word between, as in "N tests stay green".
- **D4.2** Past-tense run reports are history by their tense and are not read: "850 passed in CI run …", "Suite: 850
  tests."
- **D4.3** Two measured matches are dated history, and the data file lists each one with its document, exact phrase
  and reason: the proposal's closing note "Reviewed against `src/engine.py` at `a6e51b6`, 36 tests, all passing", and
  `_STATUS.md`'s "old engine and its 36 tests stay green", in its section dated 2026-08-28. An entry whose phrase is no longer in its document fails, so the list cannot outlive what it
  excuses.
- **D4.4** CLAUDE.md must still state "all N tests pass" (kept as it is).

Measured: with the wider pattern, the only matches other than CLAUDE.md's 917 are the two in D4.3. The old pattern
misses "`tests/ 37 tests, all passing`"; the new one finds it.

Tests: the existing `test_claimed_test_count_matches_the_suite`, which keeps its name because the brief and the CHANGELOG
cite it, and whose docstring gains `D4`; `test_d4_suite_size_phrasings_are_recognised`, parametrized over D4.1's forms
(found) and D4.2's (not found); `test_d4_history_entries_still_exist`.

**Live mutation:** put README's Layout line back as it was at `423bbed`, "`tests/               37 tests, all
passing`". The test goes red. The same edit leaves the old pattern green.

### D5a: extend `tests/test_stale_text.py`'s scan to the documents that state current status

The test now scans `src/`, `config/` and `tools/` for "pending owner review". The extension adds README.md, CLAUDE.md,
the checklist, `_STATUS.md`'s `▶ STATE` line, and every `docs/` file's own status line. The `PENDING` mechanism is
unchanged: an item truly waiting on the owner is listed with where its decision will be recorded. Dated logs
(CHANGELOG, TODO, the rest of `_STATUS.md`, plans and records) are history and are not scanned.

Measured after the merge: two matches, both stale. Checklist line 156 says the dispositions' Claude rows "are pending
owner review", but the dispositions' own status says the owner reviewed every row on 2026-10-17 and accepted the
remaining notes on 2026-10-19. Line 176 says "spec revisions 24–27 are Claude's
recommendations, pending owner review", but the owner reviewed them on 2026-10-16. No `docs/` status line contains the
phrase.

Tests: `test_d5a_current_state_documents_carry_no_stale_pending_review` (the real repo);
`test_d5a_the_scan_reads_status_lines_and_the_state_line` (fixture files, one match in each new place).

**Live mutation:** put back README's sentence "Spec revisions 24–27 are Claude's recommendations, pending owner
review." The test goes red.

### D5b: a status claim about a document agrees with that document's own status line

- **D5b.1 A document's own status** is its front-matter `status:` or the first `**Status: …**` in its first 12 lines.
  Its first sentence is classed as **open** (draft, for review, for owner review, pending owner review, not approved,
  awaiting), **settled** (approved, decided, reviewed by the owner, signed off, built, applied, complete, frozen, done),
  or neither. Open wins over settled. A settled word that is negated does not count: "not approved", "not an
  approval", "nothing decided", "not built". Each word in the two lists is taken from a real status line.
- **D5b.2 Claims read:** each checklist entry label (`**Label:**` with the entry's first link), each README sentence
  that names a `docs/` path, and each `docs/` status line that names another document.
- **D5b.3** A claim that is open, about a document whose own status is settled, fails. This is the brief's direction.
- **D5b.4 (Q1)** A claim that is settled, about a document whose own status is open, also fails.
- **D5b.5 Not vacuous.** The checklist's entry for `version_2_plan.md` (settled on both sides) must be among the pairs
  compared.

Measured after the merge: five disagreements, all in the checklist.

| Checklist label | Document | Its own status | Direction |
|---|---|---|---|
| For review, 2026-10-15 | `REVIEW_spec_revisions_24-30.md` | REVIEWED BY THE OWNER 2026-10-16 | D5b.3 |
| For review, 2026-10-17 | `version_1_dispositions.md` | complete, 2026-10-18 | D5b.3 |
| For review, 2026-10-17 | `REVIEW_version_1_outputs.md` | signed off by the owner 2026-10-17 | D5b.3 |
| Decided 2026-10-19, not built | `PROPOSAL — CUTOFF ANXIETY, BRIDGING AND REPAIR.md` | DRAFT 2026-10-19, for owner review | D5b.4 |
| Decided 2026-10-19, not built | `PROPOSAL — EVENT SIZES FROM THE HOLMES–RAHE SCALE.md` | DRAFT 2026-10-18, for owner review | D5b.4 |

In the last two rows the checklist is right and the proposal's own status line is stale. Each proposal's next sentence
says the owner decided every question on 2026-10-19 (G1: "Q1–Q11 in §8, all decided by the owner on 2026-10-19";
Holmes–Rahe: "Every question is decided"). So the reverse direction finds real drift too.

Tests: `test_d5b_status_claims_agree_with_each_documents_own_status` (the real repo);
`test_d5b_a_reviewed_document_called_for_review_is_reported` (fixture: the first row's label and status, as they are
today); `test_d5b_a_decided_label_on_a_draft_is_reported` (D5b.4, the fourth row);
`test_d5b_classification_handles_negation` (adversarial, parametrized: "not approved text" is open, "a report, not an
approval" is neither, "Decided 2026-10-19, not built" is settled, "nothing decided, nothing built" is neither).

**Live mutations:** change the checklist's label for `version_2_plan.md` from "Approved 2026-10-19:" to "For review,
2026-10-18:". Red. For D5b.4, change the label for the learning-by-level proposal from "Draft for review" to "Approved
2026-10-19". Red.

## 8. Order of work, and what the build changes besides tests

1. Precondition: `docs-stale-fixes` is on `plan-phase-b` (Q2). Rebase this branch onto it and re-measure.
2. D1 first. The spec revision changes at every revision, so it is the restatement most likely to drift again. Then D4
   (a pattern change to an existing test), D2, D3, D5a, D5b. For each: write the self-tests, run them red, write the
   check, run the real-repo test, apply the live mutation and record it, revert.
3. Fix the documents the new tests find (each read in full first, and every fix taken from the document's own record):
   - The checklist labels in rows 1–3 of §7's table, restated from each document's own status line.
   - Checklist lines 156 and 176 (D5a).
   - If Q1 is yes, the two proposals' own status lines, restated from their own text: decided by the owner
     2026-10-19, not built.
4. The full suite. CLAUDE.md's suite count updated to what pytest collects.
5. Documents updated in the same commit (rule 13): CHANGELOG; CLAUDE.md's count; this plan's status line and a build
   record (one row per live mutation: the edit, the test, the failure message, reverted); the checklist entry for this
   plan; `docs/spec_coverage.md` is not affected (these tests verify no spec ID).
6. `learning-qa` over the diff, fix its findings, commit. Not pushed.

## 9. Owner decisions

- **Q1. D5b in both directions?** Recommended: **yes.** The brief asks for one direction: a claim says open, the
  document says approved. The reverse finds the two stale proposal status lines in §7, which nothing else catches.
- **Q2. Order.** Recommended: **merge `docs-stale-fixes` into `plan-phase-b` first, then build this on top.** The
  alternative is to build on this branch with that branch merged in, which carries its changes in two places.

The word lists go in `tests/doc_claims.json` (rule 9) without a question. Moving `test_stale_text.py`'s existing
`PHRASE` and `PENDING` there is left out, because the brief says to keep that mechanism as it is.

## 10. What these tests cannot catch

- A restatement that is not one of D1's four places, for example a new document that names the current revision. A
  new place is added to D1 by name.
- A copy that is stale in a way that is not a number, path, package or status word. After the merge, `_STATUS.md`'s
  state line says "stage 1's one regeneration run is next", but step 7 ran it on 2026-10-19. D1 passes on that line,
  because its revision is right.
- D2 does not check paths written without backticks, or paths in documents other than README.md and CLAUDE.md.
- D4 does not read suite sizes written without the word "tests" (the proposal's masthead "37 today, all passing"), or
  in past tense, or in documents outside `CITING_DOCS`.
- D5b does not compare status words outside its two lists, or a document named without a link or a backticked path. It
  reads only the first sentence of a status line.

## 11. Adjacent issues found, not fixed

- `tests/bowen/test_log.py` and `tests/bowen/test_render.py` hard-code `"2.0, revision 33"`. They fail at each
  revision, so they are not silent, but they could read the spec's front matter as D1 does.
- `ruff` is in `requirements.txt` and installed in CI. There is no ruff config, and no test, tool or workflow step runs
  it.
- The spec's own status line ends "— no code until the implementation plan is also approved", though version 1 is
  built and complete.
- `_STATUS.md`'s state line on `docs-stale-fixes` says the regeneration run is next. Step 7 ran it.
- CLAUDE.md line 3 says "unit, family, and societal levels". The model is one three-generation family.
- `test_stale_text.py` keeps its `PHRASE` regex and `PENDING` table in Python source, not a data file (rule 9).
