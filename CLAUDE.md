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
| `learnings.md` | P4 (hardcoded constants — simulation parameters belong in config), P7 (scoring/ranking proxies), P9 (magic size/count caps in simulation loops), P23 (any stochastic-output regression fixture) |
| `file-maintainability.md` | Any new module or significant refactor |

## Where the theory lives

| File | What it is | Trust |
|---|---|---|
| `docs/theory/_LEDGER.md` | **six corpus sources**, all passes folded in. Per-chapter findings for the 22 papers; consolidated entries for the later sources, each backed by a fuller extraction directory | **the source of truth for what the corpus says** |
| `docs/theory/_CONVERGENCES.md` | ten cross-chapter convergences + independence audit | current |
| `docs/theory/_STATUS.md` | run state, standing decisions, order of work | **read this first** |
| `docs/DECISIONS — FAMILY EVALUATION.md` | how the 4 contradictions and 21 requirement candidates from the sixth source were decided | **closed 2026-08-27 — all applied at spec revision 4** |
| `docs/bowen_agent_model_spec_v2.md` | **the buildable contract** — 581 numbered requirements, the update order, the parameter register, 48 acceptance criteria, the `M11.G` readout schema, the `M15` import contract, the `M16` run log | **Version 1 complete 2026-10-18; version 1 is frozen at v2.0 revision 32 and the tag `version-1`. Version 2 began at revision 33 (stage 1, built 2026-10-19: the grid engine retired, `M8.5` amended; no rule changed). Revision 39, approved by the owner 2026-10-25 as `docs/spec_revision_39_draft.md`, is applied in parts on `main`, each with its code: at stage 3 step 6 (2026-10-25) T-130's wording (`M1.F.6a`) and T-014's four answers (`M8`, `M8.1`, `M8.5`, `M13.1`, `M11.C.7`, `M7.D.1a`), with `avoidance_available` removed and T-129's lint fixes; the engine hash moved from `66dcbb1e8ba5` to `f90285984f32` and no result changed (the golden fingerprints hold); groups C, F, A and T-128 land at steps 7–10. Revision 38, built 2026-10-24 on `main` from the stage 2 milestone review's triage, changed wording only (`M1.D.2b` and `M1.D.2c` restated to the code, the owner's Q1; `M11.C.48` added to `M11.4`'s exceptions, Q6; corrections in `M1.F.6a`, `M10` and the register); nothing a run computes changed and the engine hash did not move. Revision 37, built 2026-10-23 on the branch `stage-2-plan-pilot`, added stage 2's detection-limit rule and sample rule to `M17.A.1` and `M17.E.1`, wording in `M17` only; nothing a run computes changed and the engine hash did not move. Revision 36, built 2026-10-22 on the branch `claude/recursing-bohr-2393a1`, built version 2's group G (G1, G2) and the Holmes–Rahe event sizes (`M1.D.2b`, `M1.D.2c`, `M1.F.6a`, `M7.C.1l`, `M7.D.1a`, `M7.D.2a`, `M11.C.48`), each choice decided as recommended under the owner's instruction of 2026-10-22, approved by the owner 2026-10-24 (T-125). Revision 35, built 2026-10-22, moved the revision history out of the spec into `docs/spec_revision_log.md`, and the inline revision narratives of 129 requirement bodies with it (and, the same day, 44 more sentences a longer phrase list found), each kept, moved or rewritten as the current rule (`docs/spec_revision_35_dispositions.md`, 262 rows, approved by the owner 2026-10-22); no rule changed (model ergonomics, Part E). Revision 34, built 2026-10-22, changed wording only: the quotations of the 1979 tapes and the Kerr–Bowen interviews paraphrased and cited (SP-1), and the stage 1 milestone review's wording fixes. Revision 32 built 2026-10-17 (finishing version 1: Claude's recommendations, reviewed by the owner one at a time the same day, with `M4.C.1c`'s nodal-event spike added; plan §4.12); revision 31 built 2026-10-16 (part A documents; part B code, config and every record regenerated, with the owner's version 1 premise decisions); revision 30 built 2026-10-14 (revision 12 approved 2026-10-07; revisions 24–30 reviewed by the owner 2026-10-16)** |
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
`CHANGELOG.md` opens with a generated index of its entries and holds the entries from 2026-10-22 on, except that day's six earliest (the SP-1 proposal's eight questions, SP-1's build, and the model ergonomics Parts A–C, Part D, Part E and their follow-ups);
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
- Run tests before declaring done: `python3 -m pytest tests/`
- **A red suite is never normal — 1236 tests: 1211 pass and 14 are strict expected failures, and 11 are expected failures of stage 3's open batch** (the strict ones: `M5.B.3`'s third-party cost (G1's A6) and the thirteen stage 3 tests written first at plan step 5 that steps 7–10 turn green, each with a reason naming its step; all marked `xfail(strict=True)`, so each turns the suite red if it starts passing. The open batch's: the eleven record-current tests that D31 leaves stale from step 6 until step 10's one regeneration, marked in `tests/open_batch.py`; each is an expected failure only while its record is byte-identical to the file the batch found and the test passes when rerun with the hash computed on the tree the batch found (`9ebf327`), and fails otherwise; step 10 closes the batch. Measured 2026-10-25 on this Mac with `python3 -m pytest tests/`; counts are as collected by the default run; the 52 `ensemble`-marked criterion tests are deselected by `pytest.ini` and run with `python3 -m pytest -m ensemble`). The dated history of counts is in `CHANGELOG.md`, in the entry of 2026-10-22 that moved it out of this file.

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
- **A session that finishes its work ends by writing the next session's brief** (owner rule, 2026-10-24): a prompt
  the owner can paste as it is, in a fenced block in the final reply. It names the task and the plan step, the files to
  read (by index and line range), what is already found or decided, what is pending the owner's review, the hard rules
  (the engine hash, the suite count, commit to `main` and push), and the point at which to stop.

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
