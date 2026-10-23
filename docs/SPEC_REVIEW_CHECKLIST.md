# Spec review checklist

This file is the index to every document that defines, justifies or records the agent model. It is the starting point for a review of the spec and, after that, a review of runs against the spec.

**This list is kept current.** Every file under `docs/` and `config/bowen/` must be linked here, either as a document
to review or under "Not for review". `tests/test_doc_checklist.py` fails when a file is added, renamed or removed
without this list being updated, and when a link here points at a file that no longer exists. The rule that keeps the
*content* of these documents in step with the code is in `CLAUDE.md` ("Documentation stays current with the build").

**Where the documents live.** Everything is under `docs/`, except the model's input files in `config/bowen/`. Those
are parsed by `src/bowen/io/load.py` at run time, so they stay with the config; they are linked from here.

**Approved, for an outside reviewer:** [REVIEWER_READING_LIST.md](REVIEWER_READING_LIST.md) (approved by the owner
2026-10-20; revised 2026-10-21, not yet reviewed by the owner: "Before you start", §8's 2026-10-21 entry, §9's link to
the milestone review brief, and corrections from the stage 1 milestone review) selects the documents that describe
what the model does and how it works, in reading order, says which version to review, and lists known problems in the
documents.

Reviewer: ________  Started: ________  Spec revision reviewed: ________

---

## Part 1 — The contract (review every line)

These define what the model does. A run that disagrees with them is a defect.

- [ ] [bowen_agent_model_spec_v2.md](bowen_agent_model_spec_v2.md) — the spec (v2.0, revision 35; version 1 is
      revision 32). Objects (M1), reference family (M2), clocks and ordering (M3), the fast tick (M4), moves (M5),
      invariants (M6), the slow tick (M7), live positions (M8), beliefs (M9), parameters (M10), acceptance criteria
      (M11), framing (M11.F), the readout schema (M11.G), out of scope (M12), the run log and trace (M16).
- [ ] [spec_revision_log.md](spec_revision_log.md) — the spec's revision history, 4 to 35, and its front-matter
      status line as it stood at revision 34, moved out of the spec at revision 35 (2026-10-22). Why a rule is as it
      is; the spec holds the rule.
- [ ] **Approved by the owner 2026-10-22:** [spec_revision_35_dispositions.md](spec_revision_35_dispositions.md) — revision
      35's second step: the 213 inline revision sentences in 129 requirement bodies (218 rows; five sentences held two
      narratives) and the 44 more a longer phrase list found the same day (rows 219–262), each kept, moved to the
      log or rewritten as the current rule, one row per narrative. Applied as Claude's dispositions under the owner's
      decision; all 262 rows approved by the owner the same day.
- [ ] [../config/bowen/constants.md](../config/bowen/constants.md) — the parameter values the engine uses.
- [ ] [../config/bowen/constants_changes.md](../config/bowen/constants_changes.md) — every change to those values,
      and why.
- [ ] [../config/bowen/event_kinds.md](../config/bowen/event_kinds.md) — the kinds of act and their layers.
- [ ] [../config/bowen/policy.md](../config/bowen/policy.md) — the selection policy.
- [ ] [../config/bowen/family_phase_c.md](../config/bowen/family_phase_c.md) — the family run in Phase C.
- [ ] [../config/bowen/family_phase_d.md](../config/bowen/family_phase_d.md) — the reference family the owner specified (spec `M2.A`, revision 13), with its life course from revision 28: starting ages as settings, who earns, the coachee (`M2.A.0j`, `M2.A.0k`).
- [ ] [../config/bowen/partners_phase_d.md](../config/bowen/partners_phase_d.md) — the reference family's partners and their children, drafted for the owner to edit (spec `M7.C.1h`, `M7.C.1i`, revision 28).
- [ ] [../config/bowen/pattern_acts.md](../config/bowen/pattern_acts.md) — which acts each pattern readout counts.
- [ ] [../config/bowen/act_classes.md](../config/bowen/act_classes.md) — which acts draw an external agent onto a side,
  go only toward one, and make up its repertoire (spec `M8.3`, `M5.B.4`, `M5.B.5`; version 2 stage 1, S1.3).
- [ ] [../config/bowen/stress_response.md](../config/bowen/stress_response.md) — which acts avoid or soothe under load
  (spec revisions 18-19).
- [ ] [../config/bowen/attribution.md](../config/bowen/attribution.md) — where each act places the family's difficulty
  (`M9.6`) and which symptom channel an anxious family's attention to the target focuses (`M1.A.11c`); revision 25.
- [ ] [../config/bowen/perspective_steering.md](../config/bowen/perspective_steering.md) — which acts systems perspective
  steers toward or away from (spec revision 19).
- [ ] [../config/bowen/nodal_events.md](../config/bowen/nodal_events.md) — which event kinds are nodal events for the
  basic-level estimator (spec revision 17).
- [ ] [../config/bowen/family_reduced.md](../config/bowen/family_reduced.md) — the reduced family.
- [ ] [../config/bowen/fixtures/](../config/bowen/fixtures/) — the test families (dyad, triad, nuclear four) the
      criteria run on.
- [ ] [../config/bowen/criteria.md](../config/bowen/criteria.md) — the settings each acceptance criterion's arms
      declare (horizons, levels, stress, spells).
- [ ] [../config/bowen/criterion_roles.md](../config/bowen/criterion_roles.md) — who plays each role in a criterion.
      A criterion's class (premise, composite, check or mixed) is in the spec's `M11.5` table and in the class
      column of the ensemble records, not here.
- [ ] [readouts_phase_c.md](readouts_phase_c.md) — how each pattern (conflict, distance, over/underfunctioning, …)
      is read from a run. Needed to judge whether a run shows a pattern.

## Part 2 — Why the contract says what it says

Check the spec against these: that every claim marked as sourced is sourced, and that every approved decision reached the spec.

- [ ] [model_explainer.md](model_explainer.md) — every part of the model: what it is for, what it does, which chapter it implements, and its evidence grade. Read it beside the spec. §19 is set / random / emergent.
- [ ] [theory/_LEDGER.md](theory/_LEDGER.md) — what the six corpus sources say. The source of truth for citations.
- [ ] [theory/_CONVERGENCES.md](theory/_CONVERGENCES.md) — the ten cross-chapter convergences and the independence audit.
- [ ] [theory/_STATUS.md](theory/_STATUS.md) — run state, standing decisions, order of work.
- [ ] [theory/_RESOLUTIONS.md](theory/_RESOLUTIONS.md) — how the live contradictions were resolved.
- [ ] [theory/_PASS2_CORRECTIONS.md](theory/_PASS2_CORRECTIONS.md) — corrections to the first extraction pass.
- [ ] [theory/_SOURCE_QUALITY.md](theory/_SOURCE_QUALITY.md) — what each recorded source can support.
- [ ] [theory/_EXTERNAL_MEASURES.md](theory/_EXTERNAL_MEASURES.md) — external measures of differentiation.
- [ ] [<theory/Extractions to be human validated.md>](<theory/Extractions to be human validated.md>) — **awaiting
      human validation**; extractions the model relies on that no person has checked yet.

Decision records. For each, confirm the decision appears in the spec as approved:

- [ ] [<DECISIONS FOR APPROVAL.md>](<DECISIONS FOR APPROVAL.md>) — revisions 1–2 (lectures, Kerr–Bowen interviews).
- [ ] [<DECISIONS — FAMILY EVALUATION.md>](<DECISIONS — FAMILY EVALUATION.md>) — revision 4 (*Family Evaluation*).
- [ ] [proposal_rev11_emergence.md](proposal_rev11_emergence.md) — revision 11 (set / random / emergent). Where it
      and the spec differ, the spec wins.
- [ ] [<DECISIONS — PHASE C FAILING.md>](<DECISIONS — PHASE C FAILING.md>) — the failing Phase C criteria.
- [ ] [<DECISIONS — CUTOFF SEMANTICS.md>](<DECISIONS — CUTOFF SEMANTICS.md>) — whether the engine's CUTOFF should
      become rejection (built 2026-10-11, option B, `M1.B.3c`).
- [ ] [<PROPOSAL — TRIANGLE RELIEF AND CREDIT.md>](<PROPOSAL — TRIANGLE RELIEF AND CREDIT.md>) — what a triangle
      relieves (`M1.C.1`, `M4.D.6e`, `M11.C.3`). Built.

Per-source extractions. The ledger summarizes these; open one when a ledger entry looks wrong.

- [ ] [theory/](theory/) `ch01.md`–`ch22.md` — the 22 papers (`L`).
- [ ] [theory/_LECTURES_1979.md](theory/_LECTURES_1979.md) — the 1979 lectures (ASR, no verbatim quotation).
- [ ] [theory/_KERR_INTERVIEWS.md](theory/_KERR_INTERVIEWS.md) and [theory/kb/](theory/kb/) — the Kerr–Bowen
      interviews (ASR).
- [ ] [theory/family_evaluation/](theory/family_evaluation/) — *Family Evaluation* (`FE`).
- [ ] [theory/kerr_book/](theory/kerr_book/) — *Bowen Theory's Secrets* (`KS`).

## Part 3 — Reviewing runs against the spec

What is built:

- [ ] [phase_c_completion_report.md](phase_c_completion_report.md) — the status of each Phase C criterion.
- [ ] [spec_coverage.md](spec_coverage.md) — which requirement each test covers (generated by
      `tools/spec_coverage.py`).
- [ ] [plan_phase_c_failing_criteria.md](plan_phase_c_failing_criteria.md) — the plan for the failing criteria.

The runs:

- [ ] [REVIEW_version_1_guide.md](REVIEW_version_1_guide.md) — **for review, 2026-10-18:** the owner's guide for a
      complete review of version 1, module by module (`M0`–`M17`): what each requires, its explainer section, its
      config, its criteria with their status, and where to see it in the trace and the diagram below; Approve / Change /
      Question per module, the open owner questions from `TODO.md`, and four inconsistencies found while drafting.
- [ ] [review/phase_c_trace_seed7.md](review/phase_c_trace_seed7.md) — the readable 104-week trace (M16).
- [ ] [review/phase_c_chart_seed7.html](review/phase_c_chart_seed7.html) — the same run as a line chart: each person's anxiety and functional level week by week (written with the trace by `tools/phase_c_trace.py`; `tools/run_chart.py`). A test fails when either file is not the current render; a declared person alive in no sampled week is named on the page as not drawn, and a person the family file does not declare is drawn under their id.
- [ ] [review/phase_c_famdiag_seed7.json](review/phase_c_famdiag_seed7.json) — the same run as a fam-diag diagram; open it in EPModel Fam-Diag (`../epmodel-fam-diag`, port 5174; `tools/famdiag_export.py`).
- [ ] [phase_c_ensemble_record.md](phase_c_ensemble_record.md) — criterion results over seeds.
- [ ] [phase_c_mutation_record.md](phase_c_mutation_record.md) — each criterion proved failing by mutation.
- [ ] [phase_c_sweep_record.md](phase_c_sweep_record.md) — parameter sweeps.
- [ ] [phase_c_diagnostic_record.md](phase_c_diagnostic_record.md) — diagnostic variants (evidence, never a verdict).
- [ ] [phase_c_level_occupancy.md](phase_c_level_occupancy.md) — `M11.C.16` level occupancy.

The records are generated by `tools/`; check them against the spec, not by recomputing.

## Not for review

Kept for history or as build process. None of these defines current behavior.

**History, moved 2026-10-22:** [TODO_archive.md](TODO_archive.md) — the items ticked in `TODO.md` before 2026-10-22,
verbatim, grouped by the month they were closed, after a generated region listing the items ticked in the records
since; [CHANGELOG_archive.md](CHANGELOG_archive.md) — the `CHANGELOG.md` entries dated before 2026-10-19 (version 1),
verbatim. `tests/test_doc_size.py` keeps the live files short.

**TODO records, 2026-10-22:** [todo/](todo/) — one file per batch of open items, each item with its ID (`T-nnn`), its
tag `⟨now⟩` or `⟨later⟩` and its text as it stood in `TODO.md` (verbatim); `TODO.md` is generated from them by
`tools/todo_index.py`, one line per open item, and `tests/test_todo_index.py` keeps it, the archive's generated region
and `CHANGELOG.md`'s entry index current. Tick an item in its record to close it; never edit the generated files.

**The retired grid engine** (removed from the tree at version 2 stage 1, S1.6; its code and `model_config.md` are at
the tag `version-1`):
[spec.md](spec.md), [logic_rules.md](logic_rules.md),
[constitution.md](constitution.md), [background.md](background.md), [user_stories.md](user_stories.md),
[implementation_task_list.md](implementation_task_list.md),
[bowen_individual_family_model_spec.md](bowen_individual_family_model_spec.md) (v1.2, frozen).

**Superseded or already folded into the spec:**
[agent_model_proposal.html](agent_model_proposal.html) (architecture proposal; §10 is background),
[HANDOFF_spec_revision_10_from_method_literature.md](HANDOFF_spec_revision_10_from_method_literature.md),
[REVIEW_spec_rev10_2026-09-23.md](REVIEW_spec_rev10_2026-09-23.md),
[agent_raising_design_prompt.md](agent_raising_design_prompt.md),
[conversation_agents_maturity_emotion_2026-09.md](conversation_agents_maturity_emotion_2026-09.md).

**Approved 2026-10-11:** [<PROPOSAL — STRESS ON FUNCTIONING, COACHING AND SELF-WORK.md>](<PROPOSAL — STRESS ON FUNCTIONING, COACHING AND SELF-WORK.md>)
— functioning drops under stress (a strain term); how coaching and a person's own practice reach behaviour.

**Approved 2026-10-11:** [<PROPOSAL — NODAL EVENTS, CUTOFF FROM BEHAVIOUR, C.17.md>](<PROPOSAL — NODAL EVENTS, CUTOFF FROM BEHAVIOUR, C.17.md>)
— what counts as a nodal event, the cutoff readout read from behaviour, and `M11.C.17` rewritten (a coach helps and is
not required).

**Approved 2026-10-11:** [<PROPOSAL — PHASE D DEFINITIONS.md>](<PROPOSAL — PHASE D DEFINITIONS.md>) — the definitions of C.14's technique,
C.23's expressed-emotion and suppression terms, C.28's mobility term, and a criterion proving the triangle change.

**Approved 2026-10-10:** [implementation_plan_phase_d.md](implementation_plan_phase_d.md) — the Phase D plan (2026-10-10):
the owner decisions it needs (§1a records them), and the split into D-I and D-II.

**Generated, 2026-10-13:** [phase_d_ensemble_record.md](phase_d_ensemble_record.md) — the Phase D part I criteria's
verdicts over ensembles (`tools/phase_d_record.py`; plan step 12). Mutations are reported in the plan, not here.

**Generated, 2026-10-13:** [phase_d2_ensemble_record.md](phase_d2_ensemble_record.md) — the Phase D part II criteria (the life course, spec revision 28) over adaptive ensembles, from `tools/phase_d_record.py --part 2`; a default-suite test fails when it is stale.

**Generated, 2026-10-15:** [phase_d_mutation_record.md](phase_d_mutation_record.md) — Phase D part II criteria under declared mutants (first: `M4.D.6f` removed on `M11.C.10[family of origin]`), from `tools/phase_d_mutation_record.py`, rendered from `records_cache/phase_d_mutation.json`; a default-suite test fails when it is stale.

**Reviewed by the owner 2026-10-16:** [REVIEW_spec_revisions_24-30.md](REVIEW_spec_revisions_24-30.md) — the owner's review form for spec revisions 24–30 (drafted 2026-10-15): one entry per requirement added or changed, with its source, constants and measured effect, and Approve / Change / Reject; and seven inconsistencies found while drafting it. All 85 entries are decided: 81 approved as built, 4 changed.

**Approved, 2026-10-15:** [version_1_definition.md](version_1_definition.md) — what version 1 of the agent model is: seven conditions, each checked by a test, a record or an owner sign-off, where the model stands against them, and the owner's eight decisions.

**Complete, 2026-10-18:** [version_1_dispositions.md](version_1_dispositions.md) — for each criterion that does not pass, the disposition (accepted as a finding, out of scope, withdrawn, or a note) with its reason and who decided it. The owner reviewed every row on 2026-10-17 and accepted `M11.C.6`'s failure as a finding on 2026-10-18; the notes that were Claude's were accepted by the owner on 2026-10-19 (`version_2_plan.md` §5, Q10).

**Signed off by the owner 2026-10-17:** [REVIEW_version_1_outputs.md](REVIEW_version_1_outputs.md) — version 1's condition V7: every kind of output, the framing it carries (`M11.F.9`, `M11.F.10`), checked by `tests/test_version_1_outputs.py`, and the owner's sign-off.

**Generated, 2026-10-17:** [version_1_status.md](version_1_status.md) — every `M11.C` criterion with its status under V1–V3, read from the ensemble and mutation records and the dispositions, from `tools/version_1_status.py`. Frozen at the tag `version-1` (`d62cfdc`) since 2026-10-19: a default-suite test checks it and the dispositions against the tag and runs the tag's status tests.

**Approved 2026-10-19:** [version_2_plan.md](version_2_plan.md) — what version 2 of the agent model is for: version 1's accepted findings grouped by cause, its loose ends, five stages (0 to 4: freeze version 1, one engineering batch, the constant sweep, the rule questions the owner chooses, and a corpus-fidelity check inside each stage 3 proposal), a definition (W1–W8), and ten questions, all answered by the owner on 2026-10-19. Stage 0 done 2026-10-19; stage 1 built and its records regenerated 2026-10-19 (no verdict changed).

**Approved 2026-10-19:** [implementation_plan_version_2.md](implementation_plan_version_2.md) — version 2's implementation plan, stage 1 (the engineering batch): what the code shows against the version 2 plan's items 1.1–1.8, seven decisions (D1–D7, decided by the owner 2026-10-19, all as recommended), each criterion with its test and mutation, the order of work, and the one run. Code, tests and documents built 2026-10-19, and every record regenerated the same day (its §1.10).

**Generated, 2026-10-19:** [version_2_stage_1_changes.md](version_2_stage_1_changes.md) — the records in its table compared, row by row, with the same rows at the tag `version-1`, from `tools/record_changes.py`: each changed verdict, moved effect and changed value; the other records are listed under "Not compared here" (review MR-20, 2026-10-21). Stage 1 changed none.

**Approved 2026-10-21:** [milestone_review_brief.md](milestone_review_brief.md) — the brief for a
critical review of the whole model at each milestone (end of version 2's stages 2 and 3, before a tag, before an
outside review), run by a session
without the build history on either Mac: what it checks, how it is set up, the report's form and how it is triaged.

**Built 2026-10-20:** [plan_doc_claim_tests.md](plan_doc_claim_tests.md) — tests that fail when a document restates a fact that has since changed: the spec revision in four places, repo paths in README and CLAUDE.md, `requirements.txt` against the imports, the suite-size pattern widened, and status claims checked against each document's own status line, both ways. Each test with the files it reads and the mutation that proves it fails; the build record (§12) and each criterion's status (§13). Approved by the owner 2026-10-20.

**Decided 2026-10-19, not built:** [<PROPOSAL — CUTOFF ANXIETY, BRIDGING AND REPAIR.md>](<PROPOSAL — CUTOFF ANXIETY, BRIDGING AND REPAIR.md>) — version 2's G1: the owner's decisions on a cut-off tie's anxiety (peak, decay to a 10% floor, negative and positive bridging, repair) turned into rules, tests and the new negative-run composite criterion, with eleven questions (Q1–Q11, all decided by the owner) and two stated conflicts with the spec (`M1.D.2a`, `M5.B.3`).

**Draft for review, 2026-10-18:** [<PROPOSAL — LEARNING BY LEVEL, ASKING WHAT TO DO DIFFERENTLY.md>](<PROPOSAL — LEARNING BY LEVEL, ASKING WHAT TO DO DIFFERENTLY.md>) — version 2 rule question raised by the owner: every act teaches a felt lesson and a fact lesson, level sets how much the fact lesson guides choice and anxiety erodes it, a person under high anxiety may ask "what can I do differently?", and coaching raises the chance of asking and of pursuing the answer; what the corpus supports and does not; nine questions for the owner.

**Decided 2026-10-19, not built:** [<PROPOSAL — EVENT SIZES FROM THE HOLMES–RAHE SCALE.md>](<PROPOSAL — EVENT SIZES FROM THE HOLMES–RAHE SCALE.md>) — version 2, at the owner's request: the model's event sizes against the Holmes–Rahe life-event scale (an outside rating, not the corpus), what Kerr's case material says about event size (set by the tie, not the kind; the deciding case is `[K-ext]`), a proposal to take base sizes from the scale and scale them by the tie, and eight questions (Q1–Q8, all decided by the owner).

**Decided and built, 2026-10-22:** [<PROPOSAL — ASR QUOTATIONS (SP-1).md>](<PROPOSAL — ASR QUOTATIONS (SP-1).md>) — the stage 1 milestone review's SP-1 (owner decision D1): each
quotation of the 1979 tapes and the Kerr–Bowen interviews in the spec, the explainer, the ledger and the proposal
html, with a paraphrase checked against its passage and its citation (65 entries); how the frozen records and the
extraction notes are marked; the test against new quotations; and eight questions for the owner (Q1–Q8), answered
2026-10-22, all as recommended. Its §4 texts approved by the owner 2026-10-22 and built the same day at spec revision 34, cut down by the owner: the local transcript test, the extraction-note headers and the `kb` heading retitles are deferred (`TODO.md`).

**Generated, not for review:** [spec_index.jsonl](spec_index.jsonl) — one JSON line per spec ID: its module,
heading, lines, first sentence, grade tags, sources, the ledger entries that cite it, the code whose `Spec:`
line names it, its tests and its coverage status (`tools/spec_index.py`; `grep '"M4.C.3a"'` returns the
record); [theory/ledger_index.jsonl](theory/ledger_index.jsonl) — one line per ledger entry, `_LEDGER.md`'s
and the extraction files' per-finding `FE`/`KS`/`K`/`T` entries: its status as written in the heading, chapter, file,
lines and the spec IDs it cites. Checked current by
`tests/bowen/test_spec_index.py`. [spec_rules/](spec_rules/) — the spec's current rules, one file per module
(`M00.md` to `M17.md`), with the front matter's status line, the `⟦…⟧` tags, the block quotes and the revision
sections removed and nothing else; its `README.md` gives the sizes, and `history_in_body.md` lists every ID
whose body carries a revision narrative, for the owner's Part E decision (`tools/spec_rules_view.py`; checked
current by `tests/bowen/test_spec_rules_view.py`). Not normative: where a view and the spec differ, the spec
wins.

**Parts A, B and C built 2026-10-22; D and E wait on the owner's answers to its §8:**
[plan_model_ergonomics.md](plan_model_ergonomics.md) — making the spec, the ledger and the code cheaper for a
model to read: a machine-readable index of the spec and of the ledger, a current-rules view of the spec, and
(not started) thinner docstrings and the spec's history moved out of it.

**Parts A and B built 2026-10-20 (owner 2026-10-20):**
[plan_spec_coverage_phase_d.md](plan_spec_coverage_phase_d.md) — why `spec_coverage.md` read "after Phase C" and how
it was brought up to date: Part A, the tool reads every ensemble and mutation record (mechanical); Part B, a judgment
for each requirement built in Phases C and D without a test named for it, and for each earlier override, reviewed by
the owner group by group, with the overrides file renamed `tools/spec_coverage_overrides.json`; the owner's decisions.

**For review, 2026-10-13:** [phase_d_status_2026-10-13.md](phase_d_status_2026-10-13.md) — Phase D part I after the
overnight run: every build item and D-I criterion as done, partial or not done, and what the owner should look at
first (spec revisions 24–27 were then Claude's recommendations; the owner reviewed them on 2026-10-16).

**For review, 2026-10-13:** [phase_d2_status_2026-10-13.md](phase_d2_status_2026-10-13.md) — Phase D part II (the life course) after the build: each step and criterion with its evidence, and what the owner decides next.

**Built 2026-10-10:** [<PROPOSAL — RUN VISUALIZATION MAPPING.md>](<PROPOSAL — RUN VISUALIZATION MAPPING.md>) (drawing a
run as a fam-diag diagram; `tools/famdiag_export.py`, `config/viz/famdiag.md`).

**Earlier phases and build process:**
[implementation_plan_phase_b.md](implementation_plan_phase_b.md),
[implementation_plan_phase_c.md](implementation_plan_phase_c.md),
[phase_b_completion_report.md](phase_b_completion_report.md),
[phase_b_mutation_record.md](phase_b_mutation_record.md),
[review/phase_b_trace_seed7.md](review/phase_b_trace_seed7.md),
[review/phase_b_trace_seed7_nadia.md](review/phase_b_trace_seed7_nadia.md),
[../config/bowen/constants_frozen.md](../config/bowen/constants_frozen.md),
[../config/bowen/constants_frozen_phase_b.md](../config/bowen/constants_frozen_phase_b.md),
[../config/bowen/script_phase_b.md](../config/bowen/script_phase_b.md),
[cycles/](cycles/) (QA-gate cycle logs), [records_cache/](records_cache/) (cached results the records render from),
[theory/_TEMPLATE.md](theory/_TEMPLATE.md) (extraction template),
[review/](review/) (the folder of rendered runs; the files to review in it are listed above, and anything else there,
such as a backup saved beside the fam-diag export, is not a record).

