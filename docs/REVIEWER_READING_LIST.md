# Reading list for an outside reviewer

**Status: approved by the owner 2026-10-20 (drafted 2026-10-19); revised 2026-10-21 and 2026-10-22, not yet reviewed by the owner:
"Before you start" (after `main` was brought up to date), §8's 2026-10-21 entry, §9's link to the milestone review
brief, and corrections from the stage 1 milestone review (§2's entry for the two criterion config files, §5's
entry for the stage 1 comparison, §6's entry for the source-quality notes, §7's entry for version 2's plan, and §8's
entries for what that review left open and fixed), and the SP-1 proposal's entries in §7 and §8, updated 2026-10-22 when it was built at spec revision 34; §3's row for the generated indexes and the current-rules view, and §3's row for the spec's revision log (revision 35), added 2026-10-22.** It lists the documents that describe what
EPModel does and how it works, in the order a reviewer with no prior contact with the project should read them. It was
last checked on 2026-10-21, after version 2 stage 1 was pushed (CI green), the document-claim tests were merged, the
coverage page's Parts A and B were built and GitHub's default branch `main` was brought up to date.

This list selects and orders; it does not replace the full index. Every file under `docs/` and `config/bowen/` is in
[SPEC_REVIEW_CHECKLIST.md](SPEC_REVIEW_CHECKLIST.md), and a test keeps that index complete.

---

## Before you start: which version to review

**Review the default branch, `main`.** On GitHub (`dbgnvan2/EPModel`) the default branch `main` stopped on 2026-10-06
at spec revision 11, before any of the agent model was built, and the work after that went on the branch
`plan-phase-b`. On 2026-10-21 `main` was fast-forwarded to `plan-phase-b` (no commit was rewritten), and new work
goes on `main` (owner decision 2026-10-21). `plan-phase-b` stays on GitHub as history and takes no new commits.

| Review this | When |
|---|---|
| the branch **`main`** | to review the current state, and to read this list. Version 2 began at spec revision 33. Its stage 1, an engineering batch that changed no rule, was built 2026-10-19 and changed no criterion's verdict. Its later stages are planned, not built. |
| the tag **`version-1`** (commit `d62cfdc`) | to review the complete, frozen version 1 model. Finished 2026-10-18 and tagged by the owner 2026-10-19; its spec is revision 32. The tag does **not** contain this list or the version 2 documents it links (`version_2_stage_1_changes.md`, `implementation_plan_version_2.md`, `plan_spec_coverage_phase_d.md`, `plan_doc_claim_tests.md`, `milestone_review_brief.md`), and it still contains the retired grid engine. |

Read this list on `main`. To look at version 1 beside it, check the tag out in a second folder:

```bash
git clone https://github.com/dbgnvan2/EPModel.git
```

```bash
git -C EPModel worktree add ../EPModel-version-1 version-1
```

## What the project is

A Python and NumPy simulation of one three-generation family of behaving agents, built on Bowen family systems theory.
The agents act, appraise what happened and learn from felt relief; conflict, distance, over- and underfunctioning,
cutoff, projection and triangles are meant to emerge from that, not be written as rules. The model answers "if
Bowen's account is right, what follows for a family shaped like this?", as a direction of difference between two arms
of runs. It is not a prediction about any real family, and the spec forbids presenting it as one (`M11.F.9`).

Two facts govern how to read every document below:

- **Almost every constant is invented.** The source texts support directions, orderings and mechanisms, and almost no
  magnitudes. Every claim in the explainer carries an evidence grade, so an invented number (`[I]`) can be told apart
  from a sourced one (`[T]`, `[#]`).
- **Acceptance criteria test a direction between two arms, not a threshold,** and each one must be shown to fail
  under a deliberate code mutation before it counts.

---

## 1. Start here

Read these four first. Together they explain what the model is, why it is built this way, and what each part does.

| # | Document | What it tells you | Size, status |
|---|---|---|---|
| 1 | [../CLAUDE.md](../CLAUDE.md) | The most current one-page summary: what the model is, where the theory lives, the rules the code must keep (engine purity, emergence, grading, no claims about real families), and how to run the tests. Written as instructions for the coding assistant, but accurate. | current |
| 2 | [model_explainer.md](model_explainer.md) | **Every part of the model** (object, field, move, gate, invariant, clock, test): what it is for, what it does, which source finding it implements, and its evidence grade. Read "How to read an entry" for the grades. §1 orientation; §17 what the model can be asked; §18 method rationale; §19 what is set, random and emergent. | about 2,100 lines; current |
| 3 | [agent_model_proposal.html](agent_model_proposal.html) | The argument for the design: why a family of agents replaced the earlier grid model, how the agents interact, what the model can and cannot tell you, and (§10) what reading the source texts changed. Its "What is actually in the repo" section describes the repository before the agent model was built. | background; where it differs from the spec, the spec wins |
| 4 | [REVIEW_version_1_guide.md](REVIEW_version_1_guide.md) | A guided walk through the whole model, module by module (`M0`–`M17`), meant to be read beside one run: what each module requires, its explainer section, its config, its criteria and their status, and where to see it in the trace. Written for the owner's review, so it includes the owner's open questions, and ends with a table of every criterion and the entry that reviews it. | about 1,400 lines; for owner review |

## 2. The contract: what the model is required to do

The spec is the authority. A run or a piece of code that disagrees with it is a defect.

| Document | What it tells you |
|---|---|
| [bowen_agent_model_spec_v2.md](bowen_agent_model_spec_v2.md) | **The spec**, v2.0. Version 1 is revision 32 (at the tag); revision 33 began version 2 and changed no rule; revision 34 (2026-10-22) changed wording only, paraphrasing the quotations of the two speech-recognition sources (SP-1); revision 35 (2026-10-22) moved the revision history out of the spec into `spec_revision_log.md`, and the inline revision narratives of 129 bodies with it (`spec_revision_35_dispositions.md`, for the owner's review); no rule changed. 575 numbered requirements: objects (`M1`), the reference family (`M2`), clocks and update order (`M3`), the weekly fast tick step by step (`M4`), moves (`M5`), invariants (`M6`), the slow tick (`M7`), live positions (`M8`), beliefs (`M9`), the parameter register (`M10`), acceptance criteria (`M11`), output framing (`M11.F`), the family-evaluation readout (`M11.G`), what must never be built (`M12`), implementation order (`M13`), family-diagram import (`M15`, not built), the run log and trace (`M16`), the ensemble runner (`M17`, a first draft). §0 says how to read it. About 2,100 lines. |
| [spec_revision_log.md](spec_revision_log.md) | The spec's revision history, 4 to 35, and its front-matter status line as it stood at revision 34, moved out of the spec at revision 35 (2026-10-22). Read it for why a rule is as it is and what the owner decided; the spec holds the rule. |
| [spec_revision_35_dispositions.md](spec_revision_35_dispositions.md) | **For the owner's review.** Revision 35's second step: the 213 inline revision sentences that stood in 129 requirement bodies, as 218 rows — 35 stay (rule text, a recorded result or a cross-reference; the row gives the reason), 32 moved to the log, 151 rewritten as the current rule only. Applied as Claude's dispositions under the owner's decision of 2026-10-22; each row is one the owner may reverse. |
| [readouts_phase_c.md](readouts_phase_c.md) | How each named pattern (conflict, distance, over/underfunctioning, cutoff, …) is read from a run's history of acts. Patterns are readouts, never selectable acts. A test fails if this file and the code disagree. |
| [../config/bowen/constants.md](../config/bowen/constants.md) | Every parameter value the engine uses, with its grade. |
| [../config/bowen/constants_changes.md](../config/bowen/constants_changes.md) | Every change to those values, and why. |
| [../config/bowen/family_phase_d.md](../config/bowen/family_phase_d.md) | The reference family (eleven people in three generations, plus an outside coach) and its life course. |
| [../config/bowen/event_kinds.md](../config/bowen/event_kinds.md), [../config/bowen/policy.md](../config/bowen/policy.md) | The kinds of act, and how an agent selects one. |
| [../config/bowen/criteria.md](../config/bowen/criteria.md), [../config/bowen/criterion_roles.md](../config/bowen/criterion_roles.md) | How each acceptance criterion is set up: `criteria.md` holds the settings its arms declare (horizons, levels, stress, spells), and `criterion_roles.md` names the people who play each role. They do not classify the criteria. The class is in the spec's `M11.5` table and in the class column of each ensemble record: a **premise** (passing confirms the code), a **composite** (several mechanisms acting together; passing or failing says something about the model), a **check** (an instrument, a conservation law or the architecture) or a **mixed** criterion with limbs of two classes. A premise passing must never be reported as a finding. |

The other files in `config/bowen/` (which acts count for each pattern, which acts relieve stress, which acts draw an
outsider onto a side, nodal events, the test families in `fixtures/`, and so on) are model inputs parsed at run time;
Part 1 of [SPEC_REVIEW_CHECKLIST.md](SPEC_REVIEW_CHECKLIST.md) describes each one.

## 3. How it works in code

No document describes the code's package layout; the table below is the summary. The spec's update order (`M3`,
`M4`, `M7`) is the design the code follows. Every function that satisfies a spec requirement has a docstring naming
the requirement and the test that proves it (`Purpose:` / `Spec:` / `Tests:`).

| Where | What is there |
|---|---|
| `src/bowen/engine/` | The engine: objects, the weekly tick, appraisal, moves, the learner, the slow tick, symptoms, the life course, invariants. It does no file I/O and has no UI; it emits log records through an interface the caller supplies (`M16.B`). |
| `src/bowen/policy/`, `src/bowen/scenario/` | How an agent chooses an act; how a run is assembled from the config files. |
| `src/bowen/io/` | The config loader and the log sinks. The engine neither does file I/O nor imports this package: the caller loads the config and supplies the sink. The only other file I/O in `src/bowen/` is `run.py`, which writes the trace, and `ensemble/criteria.py`, which reads the criteria's config tables. |
| `src/bowen/readouts/`, `src/bowen/ensemble/` | Pattern and family-evaluation readouts; acceptance criteria run over many seeds. |
| `src/bowen/render/`, `src/bowen/run.py` | The readable trace; the command-line entry point for one scripted run. |
| `tools/` | The scripts that generate the records in §5 and the review outputs in §4. Each generator's docstring gives its command line. |
| `tests/` | The suite. `CLAUDE.md` records the current count and when it was measured; the slower criterion tests, marked `ensemble`, run separately. `tests/bowen/test_engine_purity.py` and `test_determinism.py` check the engine rules above. |
| [spec_coverage.md](spec_coverage.md) | Every spec ID mapped to the test that proves it, or to what is missing. Generated from every ensemble and mutation record, each named with its hash, and from `tools/spec_coverage_overrides.json`, where each built ID a test does not cover by name carries a judgment the owner reviewed. *Not done* says whether the ID is not built or is checked by review. |
| [spec_index.jsonl](spec_index.jsonl), [theory/ledger_index.jsonl](theory/ledger_index.jsonl) | One JSON line per spec ID (its first sentence, lines, grade tags, sources, the ledger entries citing it, the code whose `Spec:` line names it, its tests, its coverage status) and one per ledger entry (its status as written, chapter, lines, the spec IDs it cites). Generated by `tools/spec_index.py`, checked current by a test. Grep them before opening the spec or the ledger. [spec_rules/](spec_rules/): the spec's current rules, one file per module, with its tags, quotations and revision sections removed (generated by `tools/spec_rules_view.py`; not normative, the spec wins). |
| `.github/workflows/tests.yml`, `requirements.txt` | CI on Python 3.11, 3.12 and 3.13, installing only `requirements.txt`. |

To run it:

```bash
python3 -m pip install -r requirements.txt
```

```bash
python3 -m pytest tests/
```

```bash
python3 -m pytest -m ensemble
```

```bash
python3 tools/phase_c_trace.py --seed 7
```

The last command writes the trace and chart listed in §4. The commands that regenerate each record are in
[../README.md](../README.md) under "Status", with their run times (from a few minutes to about 50 minutes).

## 4. What a run produces

| Document | What it is |
|---|---|
| [review/phase_c_trace_seed7.md](review/phase_c_trace_seed7.md) | One run of 104 weeks, written out week by week in plain language (`M16`). The best way to see what the agents actually do. |
| [review/phase_c_chart_seed7.html](review/phase_c_chart_seed7.html) | The same run as a chart of each person's anxiety and functioning by week. |
| [review/phase_c_famdiag_seed7.json](review/phase_c_famdiag_seed7.json) | The same run as a family diagram. It opens in a separate viewer, `epmodel-fam-diag`, which is a **private repository** an outside reviewer cannot open; the file is readable as JSON, and the trace shows the same run. How a run maps to the diagram is in [<PROPOSAL — RUN VISUALIZATION MAPPING.md>](<PROPOSAL — RUN VISUALIZATION MAPPING.md>). |
| [REVIEW_version_1_outputs.md](REVIEW_version_1_outputs.md) | Every kind of output the model produces and the framing each carries so it cannot be read as a statement about a real family. Signed off by the owner 2026-10-17. |

Seed 7 was used for the first traces and was not chosen for what it shows. A test fails when any of the three files
above is not what its tool renders now.

## 5. Does it do what the spec says: status and evidence

Start with the definition and the status page; open the records when you want to check a number.

| Document | What it tells you |
|---|---|
| [version_1_definition.md](version_1_definition.md) | What "version 1 is finished" means: seven conditions (V1–V7), each checked by a test, a record or an owner sign-off, and the owner's decisions. |
| [version_1_status.md](version_1_status.md) | **Every acceptance criterion's status** at version 1, generated from the records. Frozen at the tag `version-1`. |
| [version_1_dispositions.md](version_1_dispositions.md) | For each criterion that does not pass: accepted as a finding, out of scope, or withdrawn, with the reason and who decided. |
| [version_2_stage_1_changes.md](version_2_stage_1_changes.md) | The records in its table (the ensemble, mutation, sweep and level-occupancy records) at the tag `version-1` compared with the same records after version 2 stage 1; the other records and generated pages are listed under "Not compared here". Generated. No verdict changed. |
| [phase_c_ensemble_record.md](phase_c_ensemble_record.md), [phase_d_ensemble_record.md](phase_d_ensemble_record.md), [phase_d2_ensemble_record.md](phase_d2_ensemble_record.md) | Each criterion's verdict over many seeds, per build phase. |
| [phase_c_mutation_record.md](phase_c_mutation_record.md), [phase_d_mutation_record.md](phase_d_mutation_record.md) | Each criterion shown to fail when the mechanism it tests is removed from the code. |
| [phase_c_sweep_record.md](phase_c_sweep_record.md), [phase_c_diagnostic_record.md](phase_c_diagnostic_record.md), [phase_c_level_occupancy.md](phase_c_level_occupancy.md) | Parameter sweeps, diagnostic variants, and one criterion's level occupancy. Reported, not used as verdicts. |
| [phase_c_completion_report.md](phase_c_completion_report.md), [phase_d_status_2026-10-13.md](phase_d_status_2026-10-13.md), [phase_d2_status_2026-10-13.md](phase_d2_status_2026-10-13.md) | Narrative status at the end of each build phase. Dated snapshots; `version_1_status.md` supersedes their numbers. |

The records are generated by `tools/` and each carries a hash of the code and config it ran against; a test fails when
a record no longer matches the code. Check them against the spec rather than recomputing them.

## 6. Why the model says what it says: the theory basis

Read this part to check that a claim presented as sourced is sourced.

| Document | What it tells you |
|---|---|
| [theory/_LEDGER.md](theory/_LEDGER.md) | **What the six source texts say**, finding by finding, each with an evidence grade and an attribution grade (Bowen, Kerr, or Kerr's own extension). The source of truth for every citation. About 2,900 lines. |
| [theory/_CONVERGENCES.md](theory/_CONVERGENCES.md) | Ten findings that recur across chapters, with an audit of whether the recurrences are independent. |
| [theory/_RESOLUTIONS.md](theory/_RESOLUTIONS.md), [theory/_PASS2_CORRECTIONS.md](theory/_PASS2_CORRECTIONS.md) | How contradictions in the sources were resolved; what a second reading corrected or withdrew from the first, including numbers that had been treated as calibration targets. |
| [theory/_SOURCE_QUALITY.md](theory/_SOURCE_QUALITY.md) | What each source can support. Two are speech-recognition transcripts, which must not be quoted verbatim; the spec, the explainer and the ledger still quote them in places (§8). |
| [<theory/Extractions to be human validated.md>](<theory/Extractions to be human validated.md>) | Extractions the model relies on that no person has checked yet. |
| `theory/ch01.md`–`ch22.md`, [theory/family_evaluation/](theory/family_evaluation/), [theory/kerr_book/](theory/kerr_book/), [theory/kb/](theory/kb/) | The per-source extractions the ledger summarizes. Open one when a ledger entry looks wrong. |
| [<DECISIONS — FAMILY EVALUATION.md>](<DECISIONS — FAMILY EVALUATION.md>), [proposal_rev11_emergence.md](proposal_rev11_emergence.md), [<DECISIONS — PHASE C FAILING.md>](<DECISIONS — PHASE C FAILING.md>) | The main owner decisions behind the spec: the sixth source's contradictions, the rule that patterns emerge rather than being written, and the handling of failing criteria. The other decision and proposal files are listed in the checklist. |
| `../papers/INDEX.md`, `../papers/DIGEST.md` | The method literature on agent-based simulation of human behaviour, separate from the Bowen texts. It supports the explainer's §18. The PDFs are not in the repository. |

## 7. Where the project is going

| Document | What it tells you |
|---|---|
| [version_2_plan.md](version_2_plan.md) | What version 2 is for: version 1's accepted findings grouped by cause, its loose ends, five stages (0 to 4; stage 4 is a check inside each stage 3 proposal), and a definition of "finished". Approved 2026-10-19. |
| [implementation_plan_version_2.md](implementation_plan_version_2.md) | Stage 1, the engineering batch: its decisions, criteria, tests and build record. Built 2026-10-19, pushed 2026-10-20 with CI passing. |
| [<PROPOSAL — CUTOFF ANXIETY, BRIDGING AND REPAIR.md>](<PROPOSAL — CUTOFF ANXIETY, BRIDGING AND REPAIR.md>), [<PROPOSAL — EVENT SIZES FROM THE HOLMES–RAHE SCALE.md>](<PROPOSAL — EVENT SIZES FROM THE HOLMES–RAHE SCALE.md>) | Two version 2 rule changes. Each records the owner's answer to every question (2026-10-19). Neither is built; both wait for a third proposal (G2) to be built with it. |
| [<PROPOSAL — LEARNING BY LEVEL, ASKING WHAT TO DO DIFFERENTLY.md>](<PROPOSAL — LEARNING BY LEVEL, ASKING WHAT TO DO DIFFERENTLY.md>) | A version 2 rule question raised by the owner. A draft; nothing decided. |
| [<PROPOSAL — ASR QUOTATIONS (SP-1).md>](<PROPOSAL — ASR QUOTATIONS (SP-1).md>) | The milestone review's SP-1: a paraphrase and citation for each quotation of the two speech-recognition sources, how frozen records are marked, and a test against new quotations. The owner answered its eight questions 2026-10-22, all as recommended, and approved its §4 texts the same day. Built 2026-10-22 at spec revision 34, cut down by the owner: the paraphrases, the frozen-record notes and the test against new quotations (`tests/test_asr_quotations.py`) are built; the local transcript test, the extraction-note headers and the `kb` heading retitles are deferred (`TODO.md`). |
| [plan_spec_coverage_phase_d.md](plan_spec_coverage_phase_d.md) | The plan for `spec_coverage.md`. Part A (the tool reads every record) and Part B (a judgment for each ID no test is named for, reviewed by the owner) both built 2026-10-20. |
| `../TODO.md`, `../CHANGELOG.md` | Open items and owner questions; the history of every change. Both are long (about 1,000 lines each). |
| [theory/_STATUS.md](theory/_STATUS.md) | The running hand-off log between working sessions. |

## 8. Known problems in these documents

A reviewer should know these before reading.

**Open:** [spec_coverage.md](spec_coverage.md)'s notes record places where the spec and the code disagree, for
example `M1.A.7` (a parents'-mean term against "MUST NOT be computed from a family average"), `M1.A.18` (a hard 0.3
gate against "graded, not a hard gate") and `M7.E.1e` against `M7.E.1f`. They are listed in `TODO.md`, not resolved.

**Open, from the stage 1 milestone review (2026-10-21), what SP-1's build left:** the extraction notes
(`theory/kb/*.md`, `theory/_LECTURES_1979.md`, `theory/_KERR_INTERVIEWS.md`) keep their quotations of the two
speech-recognition sources, which only locate passages, and their headers do not yet say so; four cross-references
in the extraction notes of *Family Evaluation* and Kerr's book also quote them, and those folders are exempt from the
test; `TODO.md` holds two such quotations; and the test against new quotations misses one whose citation sits in
another paragraph, list item or table row, among the limits its docstring lists. `TODO.md` lists each.

**Fixed since this list was first drafted:**

- 2026-10-22, spec revision 34: the spec, the explainer, the ledger and `agent_model_proposal.html` quoted the two
  speech-recognition sources verbatim. Each quotation is now a paraphrase with its citation
  ([<PROPOSAL — ASR QUOTATIONS (SP-1).md>](<PROPOSAL — ASR QUOTATIONS (SP-1).md>)); `DECISIONS FOR APPROVAL.md` and
  `proposal_rev11_emergence.md` keep theirs under a frozen-record note. The spec wording held for revision 34 was fixed
  with it: `M11.4`'s count of exceptions, the "four" callers of the live-positions predicate in M8 and `M13.1` (three
  exist), `M2.A.0i` on Dr Halim, Teodor's "awaiting owner review", and the closing clause of the spec's status line.
- 2026-10-19 and 2026-10-20, on `plan-phase-b`: README described the retired grid engine as running; the checklist
  named an old spec revision; `_STATUS.md` said version 2 was not approved; the explainer's front matter had an old
  date; `requirements.txt` listed packages nothing imports.
- 2026-10-20: the fam-diag export ([review/phase_c_famdiag_seed7.json](review/phase_c_famdiag_seed7.json)) still named
  spec revision 32; it was regenerated, and a test now fails when it is out of date, as the trace and chart tests
  already did.
- 2026-10-20: tests now fail when a document restates a spec revision, a repo path or a review status that has changed
  ([plan_doc_claim_tests.md](plan_doc_claim_tests.md)). They found two proposals whose status lines still read "DRAFT"
  after the owner had decided every question, and three checklist labels; all were fixed.
- 2026-10-20: [spec_coverage.md](spec_coverage.md) read only the Phase C records and was titled "after Phase C". Part A
  of its plan changed the tool to read every ensemble and mutation record, name each with its hash, quote the owner's
  version 1 dispositions, and say "no test named for it" where it gave a bare phase.
- 2026-10-20: an ID marked "not done, no test named for it; its module was built in Phase C" (or D) was untested by
  name, not unbuilt. Part B of [plan_spec_coverage_phase_d.md](plan_spec_coverage_phase_d.md) gave each of those 165
  IDs, and each of the 74 earlier overrides, a judgment — the test that verifies it, *partial* with what is missing, or
  *not done* with the reason — which the owner reviewed group by group. The overrides file is now
  `tools/spec_coverage_overrides.json`.
- 2026-10-21: GitHub's default branch `main` stopped at spec revision 11, and this list and the README told a
  reviewer to avoid it. `main` was fast-forwarded to `plan-phase-b`, and both documents now point to `main`.
- 2026-10-21: the stage 1 milestone review's wording findings outside the spec. This list's entries for the criterion
  config files, the stage 1 comparison page and version 2's stages; the checklist's status for the cutoff decision;
  stale lines in `_STATUS.md` and the version 1 review guide; and the explainer's withdrawn "eight to ten generations"
  target and its grade for the one-year slow tick.

## 9. Not relevant to how the model works now

These describe the retired grid engine, earlier build phases or the build process. Skip them unless you are tracing
history.

- The grid engine (retired; its code is at the tag `version-1`): [spec.md](spec.md), [logic_rules.md](logic_rules.md),
  [constitution.md](constitution.md), [background.md](background.md), [user_stories.md](user_stories.md),
  [implementation_task_list.md](implementation_task_list.md),
  [bowen_individual_family_model_spec.md](bowen_individual_family_model_spec.md).
- Build process: the Phase B, C and D implementation plans, [milestone_review_brief.md](milestone_review_brief.md)
  (approved 2026-10-21: how the project reviews itself at each milestone), [cycles/](cycles/) (QA-gate logs and review reports),
  [records_cache/](records_cache/) (cached results the records are rendered from), and the review forms for earlier
  spec revisions.
- Files at the repository root that predate the agent model: `gemini.md`, `monte_carlo_results.txt`, `data/`.
