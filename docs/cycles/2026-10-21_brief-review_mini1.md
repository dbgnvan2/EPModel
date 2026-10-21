# Review of the milestone review brief (mini1, 2026-10-21)

A review of `docs/milestone_review_brief.md` (the draft of 2026-10-21), made at the owner's request through the
EPModel session on davemini2. It reviews the brief itself. It does not run the milestone review the brief describes.

## Header

- **Commit reviewed:** `bc54217954fa0d3244a77422dfe537e36dee7888` (`origin/main`, "docs: a draft brief for milestone
  critical reviews"), in a worktree on branch `review/brief-2026-10-21`.
- **Machine:** Mac mini1 (Darwin 24.6.0), a fresh clone of `https://github.com/dbgnvan2/EPModel.git`.
- **Pythons found:** `python3.11` = 3.11.15 (`~/.local/bin/python3.11`); default `python3` = 3.14.7
  (`/opt/homebrew/bin/python3`). Neither had `numpy` or `pytest` installed. A virtualenv per interpreter was made
  outside the repo, and `requirements.txt` was installed in each (numpy 2.4.6 and 2.5.4, pytest 8.4.2).
- **Suite:** 3.11 venv: `1003 passed, 49 deselected in 185.79s`. 3.14 venv: `1003 passed, 49 deselected in 211.87s`.
- **CI for the commit:** run `39632503495`, `completed success`; jobs `test (3.11)`, `test (3.12)` and
  `test (3.13)` each `success`.
- **Tokens used:** not measured. This session has no counter for its own token use, and no subagents were used.
  See MR-11.
- **Wall time:** about 10 minutes (09:12 to 09:22 local time), from the request to this file, including the two suite runs (about 3 minutes each, run
  in parallel).

## Dry run of §3 steps 1–5

| Step | As written | Result on mini1 |
|---|---|---|
| 1 | `~/.claude` and `~/.claude/standards` pulled | Both at `main...origin/main` after `git pull` (config `deaff85`, standards `9960b01`). |
| 1 | `gh` signed in | `gh auth status`: logged in as `dbgnvan2`. |
| 1 | `python3.11` and `python3` with `requirements.txt` | Both interpreters present; neither had the requirements. Fixed with one virtualenv each, as the brief suggests. |
| 2 | `git fetch origin`; `git worktree add --no-track -b … ../… origin/main` | Worked as written (branch `review/brief-2026-10-21`). `git rev-parse HEAD` = `bc54217…`. |
| 3 | `python3.11 -m pytest tests/ -q` | **Did not work as written:** `/Users/davemini/.local/bin/python3.11: No module named pytest`. Worked from the 3.11 venv (MR-5). |
| 3 | `python3 -m pytest tests/ -q` | **Did not work as written:** `48 errors during collection` (no `numpy`). From the 3.14 venv: `1003 passed, 49 deselected in 211.87s`. |
| 4 | `gh run list --commit "$(git rev-parse HEAD)"` | Worked: `completed success … tests main push 39632503495 5m1s`. It does not show each Python version's result; `gh run view 39632503495 --json jobs` did (MR-6). |
| 5 | `git diff --stat --diff-filter=d BASELINE..HEAD -- …` | With `BASELINE` as the literal word: `fatal: bad revision 'BASELINE..HEAD'` (expected; it is a placeholder). With `version-1`: `103 files changed, 7623 insertions(+), 638 deletions(-)`. |

## Findings, most severe first

### MR-1 · major · `docs/milestone_review_brief.md:141-144`: the brief makes the Kerr–Bowen interviews no-verbatim, which the ledger and CLAUDE.md do not

**What is wrong.** L1's attribution check says both ASR sources must never be quoted verbatim. The project's
source of truth puts the no-verbatim rule on the 1979 tapes only. For the interviews the rule is "cite the
interview, not the man". The "known item" about `M10.C.2` rests on the wider rule. A reviewer following the brief
would report the ledger itself, `DECISIONS FOR APPROVAL.md` and the spec as defects.

**Quoted lines.**
- Brief 141: "The ASR sources must never be quoted verbatim: the 1979 lectures (`Tape N`) and the Kerr–Bowen interviews (`KB`)."
- `docs/theory/_LEDGER.md:79`: "| 1979 Basic Video Series, 6 tapes | Bowen lecturing | `Tape 5` | ASR; substance reliable, **no verbatim quotation** |"
- `docs/theory/_LEDGER.md:80`: "… ASR, no speaker separation; **cite the interview, not the man**; questions are ~99% Kerr |"

**How checked.** Read the ledger's source table (`_LEDGER.md:76-84`) and `CLAUDE.md:33-35`, which says the same:
"`Tape N`, **ASR — no verbatim quotation**" and "`KB`, **ASR — cite the interview, not the man**". Then searched for
`KB11` quotations: the same passage is quoted verbatim at `_LEDGER.md:1402-1405`, `docs/DECISIONS FOR APPROVAL.md:46`
("`KB11`, verbatim — …") and spec `M10.C.2` (`docs/bowen_agent_model_spec_v2.md:1206`).

**Proposed fix.** Split the bullet: "`Tape N` is never quoted verbatim. A `KB` claim cites the interview, not the
man." Replace the `M10.C.2` known item with: "if a `KB` quotation attributes the words to Bowen rather than to the
interview, report it". If the owner does want `KB` to be no-verbatim, that is a rule change: the ledger,
CLAUDE.md, `DECISIONS FOR APPROVAL.md` and the spec all need to change, and it belongs in §8, not in a lens.

### MR-2 · major · `docs/milestone_review_brief.md:107-118`, `:169`: the workers cannot do what the brief asks of them

**What is wrong.** §5 runs the lenses as `reviewer` skill workers. Those are `review-worker` subagents, which can
only read (Read, Grep, Glob). The brief asks each worker to read "the changed parts" of each file and allows
findings that were "ran". L2 also asks whether a mutation "was actually run". A worker has no `git` and cannot
find the changed parts or run anything. The brief does not tell the orchestrator to hand those over.

**Quoted lines.**
- Brief 107: "A finding is something the reviewer read or ran, not something reasoned to."
- Brief 115: "For a file in step 5's list, read the changed parts with their enclosing section or function."
- Brief 169: "A test counted as coverage without a mutation that was actually run (`M11.1`)."

**How checked.** `~/.claude/skills/reviewer/SKILL.md:120` ("Spawn `review-worker` subagents") and `:227`
("Read-only throughout. Workers have no write tools."). The `review-worker` agent type lists its tools as "Read,
Grep, Glob".

**Proposed fix.** In §5, have the orchestrator write `git diff -U0 BASELINE..HEAD` (with step 5's path filter) to a
file in the scratchpad, and give each worker that file plus the changed line ranges for its slice. State that L2's
mutation check is made by reading the committed mutation records, not by running mutants. Anything that has to be
run (a mutant, a probe) is the orchestrator's job, and its output is passed to the worker.

### MR-3 · major · `docs/milestone_review_brief.md:122-128`, `:155-160`, `:173-176`: three lenses' slices leave out the documents their checks rest on

**What is wrong.** The reviewer skill tells a worker to read only its slice. Each slice below lacks files that its
own checks need.
- **L1 has no spec.** Its checks cite `M10.B`, `M10.C`, `M10.C.1`, `M10.C.2`, `M10.C.5`, `M11.1d`, `M11.5` and
  `M11.F.9`, all in `docs/bowen_agent_model_spec_v2.md`. The known item is a sentence in the spec. Its checks on "a
  premise reported as a finding" and "presented as speaking about a real family" need the documents that present
  results. The slice has only `readouts_phase_c.md` and `docs/review/`, not `README.md`, `docs/REVIEWER_READING_LIST.md`,
  `docs/phase_c_completion_report.md` or the `*_record.md` pages.
- **L2 has no mutation records** (`docs/*_mutation_record.md`), so it cannot check `M11.1`.
- **L3 is "documents against code" with no code in its slice.** It also has no generated records for "numbers in
  prose that no generated record supports". For "statements no test checks" it lacks `tests/test_doc_claims.py`
  and `tests/doc_claims.json`.

**Quoted lines.**
- Brief 122-128: L1's slice is `config/bowen/`, `docs/model_explainer.md`, the changed `src/bowen/` files,
  `docs/readouts_phase_c.md`, `docs/review/` and `_LEDGER.md`, searched only.
- Brief 138: "The register's rules are spec `M10.C`; `M10.C.1` lists what must be labelled invented."
- Brief 174-176: L3's slice is `README.md`, `CLAUDE.md`, `docs/SPEC_REVIEW_CHECKLIST.md`,
  `docs/REVIEWER_READING_LIST.md`, `docs/theory/_STATUS.md`, and the status line of every changed `docs/` file.

**How checked.** Each ID was found in the spec (`M10.B` :1169, `M10.C` :1181, `M10.C.2` :1206, `M10.C.5` :1240,
`M11.1` :1252, `M11.1d` :1260, `M11.5` :1361, `M11.F.9` :1485). The slices were compared line by line with each
lens's checks. The record and test files were listed with `ls docs tests`.

**Proposed fix.** Add the spec's §M10–§M11 sections (by line range), the four result documents above and the
`*_record.md` pages to L1, searched only. Add the mutation records to L2. Add the generated records (searched only),
`tests/test_doc_claims.py` and `tests/doc_claims.json` to L3, and allow L3 to grep `src/bowen/` and `config/bowen/`
for a claim it is checking.

### MR-4 · minor · `docs/milestone_review_brief.md:16`, `:256`: the brief says one per-batch pass; CLAUDE.md says two

**What is wrong.** §1 and §7 describe the per-batch review as one `learning-qa` pass. CLAUDE.md, which §1 cites,
also requires a correctness pass. This understates what already runs, and §7 step 5 would drop the correctness
pass from fix batches.

**Quoted lines.**
- Brief 16: "**One `learning-qa` pass per batch** runs before the push (CLAUDE.md, "Size reviews to the change")."
- Brief 256: "Fixes go through the usual batch: tests first, documents in the same commit, one `learning-qa` pass, push."
- `CLAUDE.md:108`: "**Size reviews to the change.** One full `learning-qa` pass plus one correctness pass per batch, before push."

**How checked.** Read both files.

**Proposed fix.** At both places, say "one `learning-qa` pass and one correctness pass".

### MR-5 · minor · `docs/milestone_review_brief.md:58-59`, `:72-80`: the step 3 commands do not use the virtualenvs, and "the default `python3`" is outside CI's range here

**What is wrong.** Step 1 recommends a virtualenv per interpreter, but step 3's commands call the bare interpreters.
On mini1 both commands failed as written. The default `python3` on mini1 is 3.14. CI tests 3.11 to 3.13, so step 3
compares a CI-tested version with one CI has never run. The brief does not say whether a 3.14-only failure counts as
"the suite is red", which stops the review.

**Quoted lines.**
- Brief 58: "`python3.11` and the default `python3`, each with `requirements.txt` installed. A virtualenv per interpreter is"
- Brief 75: "python3.11 -m pytest tests/ -q"
- Brief 79: "python3 -m pytest tests/ -q"

**How checked.** Ran both commands as written (output in the dry-run table above). Ran `python3 --version`
(3.14.7). Read `.github/workflows/tests.yml` (`python-version: ["3.11", "3.12", "3.13"]`) and `README.md:90`.

**Proposed fix.** Write the commands as `<venv-3.11>/bin/python -m pytest tests/ -q` and the same for the second
interpreter. Name the second one as "the newest CI version (3.13), or the default `python3` if it is in CI's
matrix". Say that a failure only on an interpreter outside the matrix is recorded and does not stop the review.

### MR-6 · minor · `docs/milestone_review_brief.md:89-93`: step 4's command does not give each Python version's result

**What is wrong.** Step 4 asks for "the result for each Python version". `gh run list` prints one line per run with
the overall conclusion.

**Quoted lines.**
- Brief 90: `gh run list --commit "$(git rev-parse HEAD)"`
- Brief 93: "Record the run's ID and the result for each Python version."

**How checked.** Ran it: one line, `completed success … 39632503495`. Then
`gh run view 39632503495 --json jobs --jq '.jobs[] | [.name,.conclusion] | @tsv'` gave `test (3.11) success`,
`test (3.12) success` and `test (3.13) success`.

**Proposed fix.** Add the `gh run view <ID> --json jobs …` command as a second command.

### MR-7 · minor · `docs/milestone_review_brief.md:100`: step 5's path filter leaves out root files that changed since `version-1`

**What is wrong.** The filter covers only `src/ config/ tools/ tests/ docs/`. Since `version-1`, `requirements.txt`,
`pytest.ini`, `README.md`, `CLAUDE.md`, `TODO.md`, `LEARNINGS.md` and `CHANGELOG.md` have changed. That is 559
lines added and 112 removed. `pytest.ini` decides which tests run, and L4 looks for "a skip without its reason", but
L4's slice cannot see `pytest.ini`. `.github/` is outside the filter too.

**Quoted line.** Brief 100: `git diff --stat --diff-filter=d BASELINE..HEAD -- src/ config/ tools/ tests/ docs/ …`

**How checked.** `git diff --stat version-1..HEAD -- .github requirements.txt pytest.ini README.md CLAUDE.md TODO.md
LEARNINGS.md CHANGELOG.md` gave `8 files changed, 559 insertions(+), 112 deletions(-)`, including `pytest.ini | 4 +-`
and `requirements.txt | 17 +--`.

**Proposed fix.** Add `requirements.txt pytest.ini .github/` to the filter and to L4's slice. Add the root
documents to the list (L3 reads `README.md` and `CLAUDE.md` already).

### MR-8 · minor · `docs/milestone_review_brief.md:23-25`, `:22`: two details of §1's history do not match the records

**What is wrong.**
1. The reading list was first committed on 2026-10-20, not 2026-10-19. `main` was fast-forwarded on 2026-10-21,
   the next day, not "two days later". (The stale documents it records were fixed on "2026-10-19 and 2026-10-20".)
2. "a page left out of a comparison": the LEARNINGS outcome names two pages.

**Quoted lines.**
- Brief 23-25: "**Compiling `docs/REVIEWER_READING_LIST.md`** on 2026-10-19 found six stale documents. … That was fixed two days later (2026-10-21)"
- `CHANGELOG.md:102`: "### A reading list for an outside reviewer; the fam-diag export brought up to date (2026-10-20)"
- `LEARNINGS.md:57`: "… the two pages missing from `tools/record_changes.py`'s lists (now listed, …"

**How checked.** `git log --diff-filter=A -- docs/REVIEWER_READING_LIST.md` gives `a3627cc 2026-10-20 17:44:53 -0700`.
Read `docs/REVIEWER_READING_LIST.md:189-207` and `LEARNINGS.md:19-58`.

**Proposed fix.** "Compiling … (2026-10-19 to 2026-10-20) found six stale documents. … That was fixed the next day
(2026-10-21)." And "two pages left out of a comparison".

### MR-9 · minor · `docs/milestone_review_brief.md:36-41`: §2's milestones differ from CLAUDE.md's summary of them

**What is wrong.** CLAUDE.md says a milestone review runs at the end of each version 2 stage. The brief names only
the end of stages 2 and 3, and adds "before an outside reviewer". Both texts are drafts pending owner review, but
whichever Q1 settles should be written in both.

**Quoted lines.**
- `CLAUDE.md:112-113`: "at the end of each version 2 stage and before any tag, a session without the build history reviews the whole model"
- Brief 38: "**The end of version 2's stage 2 and stage 3**"

**How checked.** Read both files.

**Proposed fix.** After Q1 is answered, make CLAUDE.md's line match §2.

### MR-10 · minor · `docs/milestone_review_brief.md:209`, `:253`: the challenge pass does not check minor findings, and §7 sends those straight to `TODO.md`

**What is wrong.** The skill's challenge pass verifies blocker and major findings only. The brief says it "removes
false positives" without that limit. Under §7 step 2, up to three unchallenged minor findings go into `TODO.md`
without anyone checking them.

**Quoted lines.**
- Brief 209: "**Challenge pass.** Keep its challenge pass. It removes false positives before the report is written."
- `SKILL.md:163-164`: "here are the top findings (blocker and major only) and their `WHERE` references"

**How checked.** Read `SKILL.md:161-177`.

**Proposed fix.** Either include minor findings in the challenge brief, or say in §5 that minor findings are
unchallenged, and in §7 that the triage session checks each one before recording it.

### MR-11 · minor · `docs/milestone_review_brief.md:226`: the header asks for tokens used but not how to measure them

**What is wrong.** §6 asks the report header for "tokens used", and §7 costs the next review from it. The brief does
not say where the number comes from. While writing this file I had no counter for my own session's token use.

**Quoted line.** Brief 226: "tokens used and wall time. The next review is costed from these numbers (§7)."

**How checked.** This review: no tool in this session reports its own total token count.

**Proposed fix.** Name the source: the sum of the `usage` totals each subagent returns, plus the orchestrating
session's figure from the app's usage view if one is available. Allow "not measured, because …" when neither
exists.

### MR-12 · minor · `docs/milestone_review_brief.md:229-237`: §6's finding fields drop the skill's verification marks

**What is wrong.** The skill gives each finding "confidence", "verified?" and "corroborated-by". §6's list replaces
the skill's fields and leaves these three out. So the report cannot show which findings survived the challenge
pass that §5 keeps.

**Quoted lines.**
- `SKILL.md:207-208`: "<each: TAG · WHERE · WHAT · WHY · FIX · confidence · verified? · corroborated-by if multiple lenses found it>"
- Brief 229-237: the field list (ID, lens, file and line, severity, what is wrong, quoted lines, how it was checked, proposed fix).

**How checked.** Read both.

**Proposed fix.** Add "verified (challenge pass) or not challenged" and "found by: L1–L4" to §6's list.

## Checked and clean

- **§1, CI:** `.github/workflows/tests.yml` runs `python -m pytest tests/ -q` on 3.11, 3.12 and 3.13 from
  `requirements.txt` only. Matches line 14.
- **§1, the suite's named tests:** `tests/test_doc_claims.py`, `tests/bowen/test_spec_coverage.py`, and the record
  tests (`tests/bowen/test_ensemble_record.py`, `test_phase_d_record.py`, `test_phase_d_mutation_record.py`,
  `tests/test_record_changes.py`) exist.
- **§1, the cold sweep:** `LEARNINGS.md:19` is under `## Misses` (`:13`); `CHANGELOG.md:200` has "**Cold sweep**".
  Its cost, "about 14 minutes, about 64k tokens", is `LEARNINGS.md:55`, as §7 says.
- **§1, six stale documents:** `docs/REVIEWER_READING_LIST.md:189-195` lists six: README, the checklist,
  `_STATUS.md`, the explainer's front matter, `requirements.txt` and the fam-diag export.
- **§1, stage 4:** `docs/version_2_plan.md` §3 stage 4 (:364-370) and the §6 table row "Stage 4 checks | inside each
  stage 3 proposal" match lines 31-32 and 38-39.
- **§4, every spec ID cited exists:** `M10.B`, `M10.C`, `M10.C.1`, `M10.C.2`, `M10.C.5`, `M11.1`, `M11.1d`, `M11.5`,
  `M11.F.9`, `M16.B`, `M1.A.7` and `M1.A.18` (the lines are listed under MR-3, with `M16.B` :1931, `M1.A.7` :226 and
  `M1.A.18` :300).
- **§4 L1, every constant graded `[I]`:** all 177 table rows of `config/bowen/constants.md` carry `[I]`; none
  carries another grade.
- **§4 L1, `M10.C.2` quotes `KB11`:** true (spec :1206). The question is what rule applies to it; see MR-1.
- **§4 L1, `model_explainer.md` §19:** exists at :1953, "Set, random, emergent".
- **§4 L2, the `TODO.md` list:** `TODO.md:58` begins "*Spec and code disagree; the owner decides which stands.*" and
  names `M1.A.7` and `M1.A.18`. `tools/spec_coverage_overrides.json` and `docs/spec_coverage.md` exist.
- **§4 Scope, sizes:** the `.py` files in `src/bowen/` total 13,850 lines; `docs/theory/_LEDGER.md` is 2,908 lines.
- **§3 step 2:** the worktree command worked as written. Tag `version-1` resolves to commit `d62cfdc`, as CLAUDE.md
  says.
- **§8 Q4's numbers:** step 5's list at `bc54217`, leaving out the brief itself, is `102 files changed, 7339
  insertions(+), 638 deletions(-)`, which matches "102 files, about 7,300 lines added and 640 removed". With the brief
  it is 103 files and 7,623 lines added.
- **§5, the skill's command:** `SKILL.md:32-35` defines `/reviewer spec <path> --against <code-path>` and `--quick`.
  `--quick` is "one worker per lens group, no area split", which fits one worker per lens. Q3's "4–12 workers" is
  `SKILL.md:78`.
- **§5/§6, the report path:** the skill's default is the repo root (`SKILL.md:183`). The brief overrides it
  explicitly.
- **§6, `docs/cycles/` indexed as a folder:** `docs/SPEC_REVIEW_CHECKLIST.md:209` lists `[cycles/](cycles/)`, and
  `tests/test_doc_checklist.py:56-60` accepts any file under a listed folder.
- **§6/§7 and the CLAUDE.md files:** I found no contradiction beyond MR-4 and MR-9. That covers the branch rules
  ("a session's own branch merges into `main`", `CLAUDE.md:73-75`), the three-or-fewer rule (`CLAUDE.md:110`),
  stating cost before a loop (`CLAUDE.md:111`), the documentation index (global rule 13), and global CLAUDE.md's
  clone instructions for `~/.claude/standards`.

## Unverified

- Whether `git fetch origin` brings the `version-1` tag into an older clone that never had it. My clone was fresh.
- Whether the slices fit a worker's context. Sizes were not measured against a model's window. The spec is 2,585
  lines and `model_explainer.md` is 2,094.
- Which tests "the record-current tests" means. I took the record tests listed above. The brief does not name them.
- Whether any other page in step 5's list has a header saying a tool generated it. A grep of the first five lines
  of each file found only prose that uses the word "generated", not a generated-page header. I did not open each
  file.
- Whether §4's four lenses would find a theory-fidelity defect in practice. That needs a real run, which this
  review did not do.

## Questions for the owner

1. **The `KB` quotation rule (MR-1).** Is `KB` no-verbatim, as the brief says, or "cite the interview, not the man",
   as the ledger and CLAUDE.md say? If the brief is right, four documents need a rule change.
2. **Group G as a milestone?** `docs/version_2_plan.md` §6 puts "Group G and the event sizes", a rule change that
   moves the engine hash, between stage 1 and stage 2. Under §2 its first review would come only at the end of
   stage 2. Should the first rule change of version 2 get its own milestone review?
3. **Minor findings and the challenge pass (MR-10).** Should the challenge pass cover minor findings too, or
   should triage check them?
4. **The second interpreter (MR-5).** Should step 3 use the newest CI version, or the machine's default `python3`
   even when that is outside CI's matrix?
