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
| `docs/bowen_agent_model_spec_v2.md` | **the buildable contract** — 575 numbered requirements, the update order, the parameter register, 47 acceptance criteria, the `M11.G` readout schema, the `M15` import contract, the `M16` run log | **Version 1 complete 2026-10-18; version 1 is frozen at v2.0 revision 32 and the tag `version-1`. Version 2 began at revision 33 (stage 1, built 2026-10-19: the grid engine retired, `M8.5` amended; no rule changed). Revision 32 built 2026-10-17 (finishing version 1: Claude's recommendations, reviewed by the owner one at a time the same day, with `M4.C.1c`'s nodal-event spike added; plan §4.12); revision 31 built 2026-10-16 (part A documents; part B code, config and every record regenerated, with the owner's version 1 premise decisions); revision 30 built 2026-10-14 (revision 12 approved 2026-10-07; revisions 24–30 reviewed by the owner 2026-10-16)** |
| `docs/version_1_status.md` | **where version 1 stands**: every `M11.C` criterion under V1–V3, generated from the records and `docs/version_1_dispositions.md` | **frozen at the tag `version-1` (`d62cfdc`, owner 2026-10-19); a test checks it and the dispositions against the tag and runs the tag's status tests** |
| `docs/model_explainer.md` | **every part of the model** — object, field, move, gate, invariant, clock, test — what it is for, what it does, and which chapter it implements | **start here to understand the model** |
| `docs/agent_model_proposal.html` | the architecture proposal, §10 = corpus reconciliation | current |
| `docs/bowen_individual_family_model_spec.md` | **v1.2, FROZEN** — describes the retired grid engine only | historical record; do not extend |

**Six corpus sources, and they are not one evidence class.** The 22 papers (1957–76, `L`); the 1979
video lectures (`Tape N`, **ASR — no verbatim quotation**); the Kerr–Bowen interviews (`KB`, **ASR — cite
the interview, not the man**); Kerr & Bowen, *Family Evaluation* (1988) — `FE`, written, and its
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
- **A red suite is never normal — all 1003 tests pass** (measured 2026-10-20 on this Mac after a test that compiles every tracked file on the oldest Python in CI's matrix was added, `tests/test_python_versions.py`; 1002 earlier that day after the coverage page's Part B added five tests, `docs/plan_spec_coverage_phase_d.md` §7; 997 earlier that day after its Part A and review added nine; 988 earlier that day after the document-claim tests were merged, adding 60, `docs/plan_doc_claim_tests.md`; 928 earlier that day after the fam-diag export's currency test was added; 927 earlier that day after the cold sweep's fixes added three tests; 924 earlier that day after the stage 1 `/csdp` sweep's fixes added seven tests; 917 on 2026-10-19 after version 2 stage 1's build, review fixes and one regeneration; 916 earlier that day after the `/csdp` sweep's fixes added three run-chart tests; 913 earlier that day after version 1's status page was frozen at the tag `version-1`, its eleven status tests now run at the tag by three; 921 on 2026-10-18 after the run chart was added beside the Phase C trace; 915 earlier that day after spec revision 32 with the owner's review and the nodal-event spike; 909 on 2026-10-17 after revision 32's first part; 886 earlier that day after `M11.C.21`'s reclassification and the capacity rule's no-span fix; 884 on 2026-10-16 after revision 31 part B; 868 earlier that day; 854 on 2026-10-15; 850 passed in CI run 38763412772 on Python 3.11, 3.12 and 3.13, before the Phase D mutation record's 4; counts are as collected by the default run; the 49 `ensemble`-marked criterion tests are deselected by `pytest.ini` and run with `python3 -m pytest -m ensemble`).

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
- **Milestone reviews** (draft, `docs/milestone_review_brief.md`, pending owner review): at the end of each version 2
  stage and before any tag, a session without the build history reviews the whole model against the brief's four
  lenses, theory fidelity first, and writes one report to `docs/cycles/` on its own branch. Triage happens here.
