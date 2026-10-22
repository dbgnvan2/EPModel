# Milestone critical review: the brief

**Status: APPROVED 2026-10-21 by the owner, after its first run (the stage 1 review, on branch
`review/2026-10-21-v2-stage-1` until merged into `docs/cycles/`) and that review's triage.** The owner decided its four open
questions the same day (§8). Drafted at the owner's request on 2026-10-21. Revised the same day
after the second Mac reviewed it (`docs/cycles/2026-10-21_brief-review_mini1.md`, branch `review/brief-2026-10-21`),
with the owner's four decisions on that review: the `KB` quotation rule stated where a reviewer reads first, one local
suite run, minor findings checked at triage, and no separate review for Group G. A milestone review is a
critical read of the whole model by a session that has no memory of how it was built. It runs at fixed points and
looks for what the per-batch review and the test suite cannot catch. This file is the brief that session works from,
so a review gives the same kind of result on either Mac. §8 records the owner's decisions on its open questions.

---

## 1. What it is for

Three checks already run on every change:

- **CI** runs the suite on Python 3.11, 3.12 and 3.13 on a fresh machine (`.github/workflows/tests.yml`).
- **The suite** includes the document-claim tests, the tests that fail when a generated record is out of date (for
  example `tests/bowen/test_ensemble_record.py` and `tests/test_record_changes.py`) and the coverage page test.
- **One `learning-qa` pass and one correctness pass per batch** run before the push (CLAUDE.md, "Size reviews to the
  change").

Each of these sees either one diff or a fixed list of claims. None sees slow drift across many commits, and none can
judge the project's core rules about the theory. Two earlier reviews show what a wider read finds:

- **The cold sweep of 2026-10-20** ran in a fresh session after a push. It found a record tool that exited 0 on a
  broken row and two pages left out of a comparison (CHANGELOG, "Cold sweep"; `LEARNINGS.md`, Misses).
- **Compiling `docs/REVIEWER_READING_LIST.md`** (2026-10-19 to 2026-10-20) found six stale documents. It also recorded
  that GitHub's `main` had stopped at spec revision 11. That was fixed on 2026-10-21, and until then no test or
  per-batch review flagged it.

A milestone review does **not** do the following:

- **Fix anything.** It writes one report, and the fixes are made later in the normal way.
- **Rerun the records.** The tests that fail when a record is out of date (above) already check them, and the ensemble and mutation runs are slow (P38).
- **Repeat stage 4's work.** That work checks spec text against `_LEDGER.md` inside each stage 3 proposal
  (`docs/version_2_plan.md` §3, stage 4, and §6).

## 2. When to run it

The milestones (owner, 2026-10-21; §8 Q1):

1. **The end of version 2's stage 2 and stage 3** (`docs/version_2_plan.md` §3). Stage 4 runs inside stage 3, so it
   has no review point of its own. Group G and the event sizes (`docs/version_2_plan.md` §6) are built before stage
   2's pilot and change what every run computes; they get no review of their own, and the end-of-stage-2 review covers
   them (owner, 2026-10-21).
2. **Before any tag.**
3. **Before the model goes to an outside reviewer.**

The first review covers stage 1, which is built and which no milestone review has covered. The owner chose on
2026-10-21 to run it then, not to wait for the end of stage 2 (§8 Q4).

Do not run it on a calendar. If nothing in §3 step 5's list has changed since the last milestone review, skip it and
say so.

## 3. Where and how to set it up

Run it on either Mac, in a **new session that has not seen the build**. The second Mac is the default (owner, §8 Q2): it
leaves this one free, and its clone from GitHub also checks that the shared-repo setup works.

On the machine that runs it:

1. **Prerequisites.**
   - `~/.claude` and `~/.claude/standards` cloned and pulled; global `CLAUDE.md` says how.
   - `gh` signed in.
   - One Python in CI's matrix (`.github/workflows/tests.yml`; 3.11, 3.12 and 3.13 on 2026-10-21), in a virtualenv
     with `requirements.txt` installed. A machine's default `python3` may be outside the matrix (the second Mac's
     was 3.14 on 2026-10-21).

2. **A fresh worktree at the commit under review**, on its own branch, not tracking `main`. Run these from the
   EPModel clone, then record the commit with `git rev-parse HEAD` inside the new worktree:

   ```bash
   git fetch origin --tags
   ```

   ```bash
   git worktree add --no-track -b review/YYYY-MM-DD-MILESTONE ../EPModel-review-YYYY-MM-DD origin/main
   ```

3. **The suite, once, on the interpreter from step 1.** Record its version and the count. CI's per-version results
   (step 4) cover the other versions. `VENV` is the virtualenv's directory.

   ```bash
   VENV/bin/python -m pytest tests/ -q
   ```

   If the suite is also run on an interpreter outside CI's matrix, a failure only there is recorded and does not stop
   the review. Otherwise, tell two failures apart:
   - **The environment is not ready.** A missing package or interpreter means the suite cannot start. Fix it, or
     record it and rely on step 4's CI results; nothing can then be run locally.
   - **The suite is red.** Tests ran and failed. That stops the review: report the failures and nothing else.

4. **CI on the same commit.**

   ```bash
   gh run list --commit "$(git rev-parse HEAD)"
   ```

   That gives one line per run, with the run's ID (`RUN_ID`) and its overall result. Each Python version's result:

   ```bash
   gh run view RUN_ID --json jobs --jq '.jobs[] | [.name, .conclusion] | @tsv'
   ```

   Record the run's ID and the result for each Python version. If there is no run, or it is still in progress,
   record that and go on. If any Python version is red, stop as in step 3.

5. **The baseline and what changed since.** The baseline is the commit in the previous milestone review's header.
   For the first review it is the tag `version-1`. List what changed, leaving out deleted files and generated pages.
   The dated logs (`CHANGELOG.md`, `TODO.md`, `LEARNINGS.md`) are history and are left out too; L2 reads `TODO.md`'s
   one-line index and the records it names under `docs/todo/`.

   ```bash
   git diff --stat --diff-filter=d BASELINE..HEAD -- src/ config/ tools/ tests/ docs/ requirements.txt pytest.ini .github/ README.md CLAUDE.md ':!docs/records_cache' ':!docs/review' ':!docs/*_record.md' ':!docs/spec_coverage.md' ':!docs/version_1_status.md' ':!docs/version_2_stage_1_changes.md' ':!docs/phase_c_level_occupancy.md'
   ```

   Also leave out any other page whose header says a tool generated it.

6. **Read by index, not by file.** To find what a requirement says, where its sources are, and which code and
   tests carry it, grep `docs/spec_index.jsonl` (one JSON line per spec ID) and `docs/theory/ledger_index.jsonl`
   (one per ledger entry, `_LEDGER.md`'s and the extraction files' per-finding entries), both generated and checked current by a test (`docs/plan_model_ergonomics.md`). To
   read a module's rules, open its file in `docs/spec_rules/` (the spec without its tags, quotations and revision
   sections; not normative). Open the spec and the ledger at the lines the index gives. Do not read `CHANGELOG.md` in full; grep it for the
   item in hand, starting from the generated index at its top (and `docs/CHANGELOG_archive.md`, which holds the
   entries dated before 2026-10-19). `TODO.md` is a generated index, one line per open item with a link to its record
   under `docs/todo/`; closed items are listed in `docs/TODO_archive.md`. Read the index, then only the records L2
   needs (step 5); never the records whole.

## 4. What to check

Four lenses. A finding is something the reviewer read or ran, not something reasoned to. Each finding cites a file
and line, quotes the lines it rests on (three at most), and says how it was checked. A suspicion that could not be
checked goes under "Unverified" in the report, never under findings.

**Scope.**

- The orchestrating session reads the spec end to end, as the `reviewer` skill's Phase 0 requires. The read may run
  while the workers do, since they do not depend on it, but it is finished before consolidation (owner, 2026-10-21).
- Each worker reads its lens's slice below. A file marked "searched only" is searched for what a check needs, not read
  whole.
- Workers are `review-worker` subagents and can only read (Read, Grep, Glob): no `git`, and nothing run. Before
  fanning out, the orchestrator writes `git diff -U0 BASELINE..HEAD` with step 5's path filter to a file in its
  scratchpad, and gives each worker that file's path with the changed line ranges in its slice. Anything that has to
  be run (a test, a probe) is the orchestrator's job, and it passes the output to the worker.
- For a file in step 5's list, read the changed parts with their enclosing section or function. For files not in the
  list, sample, and name what was sampled and what was not read.
- With 13,850 lines in `src/bowen/` and a 2,908-line `_LEDGER.md`, reading everything every time costs too much.
- Search `_LEDGER.md` for the source a claim cites; do not read it whole.

### L1. Theory fidelity (the reason this review exists)

**Slice:**
- `config/bowen/`
- `docs/model_explainer.md`
- the changed files in `src/bowen/`
- the spec's sections `M10` and `M11`, through `M11.G`
- `docs/theory/_SOURCE_QUALITY.md` (what each source may be cited for, and how)
- `docs/readouts_phase_c.md`
- `docs/review/`
- the documents that present results, searched only: `README.md`, `docs/REVIEWER_READING_LIST.md`,
  `docs/phase_c_completion_report.md` and the `docs/*_record.md` pages
- `docs/theory/_LEDGER.md` and `docs/version_2_plan.md`, searched only

**Checks:**

- **Invented values presented as sourced.** As of 2026-10-21 every constant in `config/bowen/constants.md` is graded
  `[I]`, so the defects to look for are these:
  - an `[I]` constant whose note, or a document describing it, cites a source as if the source set the value;
  - a numeric literal in `src/bowen/` that is not in `config/bowen/` (spec `M10.B`; CLAUDE.md, "Key rules");
  - a `[T]` or `[#]` claim in `docs/model_explainer.md` whose source in `_LEDGER.md` does not say what the claim says.
    CLAUDE.md says to treat such a claim as `[I]`.
  - The register's rules are spec `M10.C`; `M10.C.1` lists what must be labelled invented.
- **Attribution.**
  - A `[K-ext]` claim must never be attributed to Bowen.
  - The ASR sources must never be quoted verbatim: the 1979 lectures (`Tape N`) and the Kerr–Bowen interviews (`KB`).
    A `KB` claim cites the interview, not the man. The rule is `docs/theory/_SOURCE_QUALITY.md`, "What the Hijack set
    supports", and the explainer's citation legend.
  - Known item to settle on the first review: `M10.C.2`, `_LEDGER.md`'s withdrawn eight-to-ten-generations entry and
    `docs/DECISIONS FOR APPROVAL.md` A1 quote `KB11` verbatim, in the first person. Is that an exception the owner
    accepted, or a defect?
- **A rule that writes an outcome.** Conflict, over- and underfunctioning, distance, cutoff, projection and fixed
  triangles must emerge from the two drives and felt relief. A rule in `src/bowen/` must not set them directly. See
  `docs/model_explainer.md` §19, `M11.1d` (the outcome-directive audit) and `M10.C.5` (the premise column).
- **A premise reported as a finding.** A criterion's class is in `M11.5`. A premise passing confirms the code. A
  report or readout that presents it as a result about families is a defect.
- **The model presented as speaking about a real family** (`M11.F.9`; version 2's W8 in `docs/version_2_plan.md`). This includes a run tuned to a
  known history and called a comparison.

### L2. Spec against code

**Slice:**
- the spec
- the changed files in `src/bowen/` and `tests/bowen/`
- `docs/spec_coverage.md` (read as evidence, not reviewed as a change)
- `tools/spec_coverage_overrides.json`
- the mutation records (`docs/*_mutation_record.md`), read as evidence
- `TODO.md` (the index) and the `docs/todo/` records it names

**Checks:**

- A requirement that the code does not do, or does differently.
- The disagreements already recorded: `docs/spec_coverage.md`'s *partial* notes, and the list beginning
  "Spec and code disagree; the owner decides which stands" (T-004 in `docs/todo/2026-08-22_blocking_the_next_phase.md`;
  for example `M1.A.7`, `M1.A.18`). Say whether that list
  has grown since the baseline. Do not report the listed items again as new.
- An acceptance test that asserts an absolute threshold, not a direction of difference between two arms.
- A test counted as coverage without a mutation that was actually run (`M11.1`). The committed mutation records show
  which mutants ran; workers read them and do not run mutants.

### L3. Documents against code, and against each other

**Slice:**
- `README.md`, `CLAUDE.md`, `docs/SPEC_REVIEW_CHECKLIST.md`, `docs/REVIEWER_READING_LIST.md`
- `docs/theory/_STATUS.md`
- the status line of every changed `docs/` file
- `tests/test_doc_claims.py` and `tests/doc_claims.json` (the claims a test already checks)
- searched only, for a claim being checked: the generated records (`docs/*_record.md`, `docs/spec_coverage.md`,
  `docs/version_1_status.md`), `src/bowen/` and `config/bowen/`

**Checks:**

- **Status lines** that no longer match the state they claim (P6): "approved", "built", "not yet reviewed".
- **Numbers in prose** that no generated record supports.
- **Statements no test checks.** The document-claim tests (`docs/plan_doc_claim_tests.md`) check only the claims
  someone wrote a test for.
- **The reading list.** Its account of which version to review, what is built and its known problems.

### L4. Engineering drift

**Slice:** the changed files in `src/bowen/`, `tools/` and `tests/`, with `pytest.ini`, `requirements.txt` and
`.github/workflows/tests.yml`; the spec's `M16.B`, searched only.

**Checks:**

- Magic constants in `src/bowen/` that belong in `config/bowen/` (P4). This overlaps L1's second check; report each
  item once, under L1.
- File I/O or UI in the engine (`M16.B`).
- A test that passes without checking anything, and a skip without its reason.
- A slow loop that is rerun in full where a cache would do (P38).

## 5. How to run it

Use the `reviewer` skill on the spec against the code:

```text
/reviewer spec docs/bowen_agent_model_spec_v2.md --against src/bowen --quick
```

Give it this file as the brief, with these changes to its defaults:

- **Lenses.** §4's four lenses replace its lens groups, one worker each, with the slices §4 gives (`--quick`; §8 Q3).
- **Changed lines.** Give each worker the diff file and line ranges described in §4's "Scope".
- **Challenge pass.** Keep its challenge pass. It checks blocker and major findings only (the skill's Phase 4) and
  removes their false positives before the report is written. Minor findings reach the report unchallenged; triage
  checks them (§7).
- **Report.** Write the report to the path in §6, not the repo root, in §6's form.
- **Model.** Workers run on the session's default model unless the owner says otherwise.

The reviewing session edits nothing except its report.

## 6. The report

**Path:** `docs/cycles/YYYY-MM-DD_milestone-review_MILESTONE.md`, for example
`docs/cycles/2026-10-21_milestone-review_v2-stage-1.md`. `docs/cycles/` is indexed as a folder in
`docs/SPEC_REVIEW_CHECKLIST.md`, so the report needs no entry of its own.

**Form.** The `reviewer` skill's report, with these additions and limits:

- **Header.** Record:
  - the commit reviewed, the baseline and the machine;
  - the suite count and the Python version it ran on, and the CI run ID with each version's result;
  - tokens used and wall time. The next review is costed from these numbers (§7). Tokens are the sum of the usage
    totals each subagent's result reports, plus the orchestrating session's own figure if the app shows one. Where
    a figure is not available, write "not measured" and why.
- **The skill's coverage matrix.** Lenses across, areas down, each cell saying what was read in full, sampled or not
  read. Keep it.
- **Findings, most severe first.** Each finding has:
  - ID: `MR-n`;
  - found by: the lens or lenses (L1–L4) that found it;
  - file and line;
  - severity: blocker, major or minor;
  - confidence, and verified: survived the challenge pass, or not challenged (every minor finding);
  - what is wrong;
  - the quoted lines;
  - how it was checked;
  - a proposed fix.
- **The skill's systemic patterns** and **"Refuted in challenge"** sections. Keep both: the refuted list stops the
  next review finding the same false positives again.
- **Checked and clean.** Each entry names its lens and the area, and what was checked there. Never a bare "looks good".
- **Unverified.**
- **Questions only the owner can answer.**

**Then:**

1. Commit the report on the review branch.
2. Push the branch: `git push -u origin HEAD`.
3. Do not merge it, and do not push to `main`.

## 7. Triage, on the working machine

1. A session on the working machine reads the report, not a chat summary of it.
2. **Check every minor finding against its file**, since the challenge pass does not cover them. A finding that does
   not hold is dropped, and the triage says so.
3. **Three or fewer minor findings that hold:** record them in `TODO.md` and move on (owner rule).
4. **Everything else:** goes to the owner one decision at a time, through `AskUserQuestion`.
5. **A defect an earlier per-batch pass missed:** record it in `LEARNINGS.md` under Misses, in that section's form.
6. Fixes go through the usual batch: tests first, documents in the same commit, one `learning-qa` pass and one
   correctness pass, push.
7. The review branch is merged into `main` once the owner says so.

**Cost.** State it before starting a review (CLAUDE.md), from the last report's header. The first milestone review
(stage 1, on the second Mac, 2026-10-21) measured 462,483 subagent tokens for four lens workers and the challenge pass,
and 13 to 14 minutes (the report says 13; its clock reads 11:54 to 12:08) from §3 step 3 to the challenge pass's return, plus the time to write the report; the orchestrator's
own tokens were not measured. Its triage checked every finding with three agents, about 534k tokens and 12 minutes.

## 8. The owner's questions, decided 2026-10-21

Each was asked separately, and each answer was the recommended option.

- **Q1. Milestones.** The end of stages 2 and 3, before any tag, and before an outside review (§2). Add or remove any?
  (Group G is decided: covered by the end-of-stage-2 review.) **Decided:** keep all three.
- **Q2. Machine.** The second Mac by default (§3)? **Decided:** yes; either Mac runs it when the second is busy.
- **Q3. Size.** One worker per lens (four, plus the challenge pass) by default, or the `reviewer` skill's full split
  of 4–12 workers? **Decided:** one per lens. If a lens reports slices it could not read, the next review splits that
  lens.
- **Q4. First review.** Run it now over stage 1 (baseline `version-1`), or wait for the end of stage 2? On
  2026-10-21, §3 step 5's list from `version-1` to `bc54217`, leaving out this brief, was 107 files, about 7,400
  lines added and 700 removed. **Decided:** run it now. The estimate given with the question, not a measurement, was
  roughly 400k–700k tokens and 30–60 minutes on the second Mac.

## 9. The prompt to start a review

Paste this into a new session in the review worktree (§3 step 2) on either Mac:

```text
Run a milestone critical review of EPModel following docs/milestone_review_brief.md exactly.
Milestone: MILESTONE. Baseline: BASELINE (the previous milestone review's commit, or the tag version-1 for the first).
Virtualenv: VENV (§3 step 1).
Do §3 steps 3–5, then §5 with §4's lenses and slices, then write the report in §6's form, commit it on this branch and
push the branch. Edit nothing else, fix nothing, do not merge, do not push to main. Never fabricate: every finding
cites a file and line you read. Before starting, state the expected cost from the last report's header, or §7's
reference if there is none.
```
