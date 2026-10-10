# Spec review checklist

This file is the index to every document that defines, justifies or records the agent model. It is the starting
point for a review of the spec and, after that, a review of runs against the spec.

**This list is kept current.** Every file under `docs/` and `config/bowen/` must be linked here, either as a document
to review or under "Not for review". `tests/test_doc_checklist.py` fails when a file is added, renamed or removed
without this list being updated, and when a link here points at a file that no longer exists. The rule that keeps the
*content* of these documents in step with the code is in `CLAUDE.md` ("Documentation stays current with the build").

**Where the documents live.** Everything is under `docs/`, except the model's input files in `config/bowen/`. Those
are parsed by `src/bowen/io/load.py` at run time, so they stay with the config; they are linked from here.

Reviewer: ________  Started: ________  Spec revision reviewed: ________

---

## Part 1 — The contract (review every line)

These define what the model does. A run that disagrees with them is a defect.

- [ ] [bowen_agent_model_spec_v2.md](bowen_agent_model_spec_v2.md) — the spec (v2.0, revision 12). Objects (M1),
      reference family (M2), clocks and ordering (M3), the fast tick (M4), moves (M5), invariants (M6), the slow tick
      (M7), live positions (M8), beliefs (M9), parameters (M10), acceptance criteria (M11), framing (M11.F), the
      readout schema (M11.G), out of scope (M12), the run log and trace (M16).
- [ ] [../config/bowen/constants.md](../config/bowen/constants.md) — the parameter values the engine uses.
- [ ] [../config/bowen/constants_changes.md](../config/bowen/constants_changes.md) — every change to those values,
      and why.
- [ ] [../config/bowen/event_kinds.md](../config/bowen/event_kinds.md) — the kinds of act and their layers.
- [ ] [../config/bowen/policy.md](../config/bowen/policy.md) — the selection policy.
- [ ] [../config/bowen/family_phase_c.md](../config/bowen/family_phase_c.md) — the family run in Phase C.
- [ ] [../config/bowen/family_reduced.md](../config/bowen/family_reduced.md) — the reduced family.
- [ ] [../config/bowen/fixtures/](../config/bowen/fixtures/) — the test families (dyad, triad, nuclear four) the
      criteria run on.
- [ ] [../config/bowen/criteria.md](../config/bowen/criteria.md) — how each acceptance criterion is set up.
- [ ] [../config/bowen/criterion_roles.md](../config/bowen/criterion_roles.md) — each criterion's class (premise or
      finding, `M11.5`).
- [ ] [readouts_phase_c.md](readouts_phase_c.md) — how each pattern (conflict, distance, over/underfunctioning, …)
      is read from a run. Needed to judge whether a run shows a pattern.

## Part 2 — Why the contract says what it says

Check the spec against these: that every claim marked as sourced is sourced, and that every approved decision
reached the spec.

- [ ] [model_explainer.md](model_explainer.md) — every part of the model: what it is for, what it does, which
      chapter it implements, and its evidence grade. Read it beside the spec. §19 is set / random / emergent.
- [ ] [theory/_LEDGER.md](theory/_LEDGER.md) — what the six corpus sources say. The source of truth for citations.
- [ ] [theory/_CONVERGENCES.md](theory/_CONVERGENCES.md) — the ten cross-chapter convergences and the independence
      audit.
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

- [ ] [review/phase_c_trace_seed7.md](review/phase_c_trace_seed7.md) — the readable 104-week trace (M16).
- [ ] [phase_c_ensemble_record.md](phase_c_ensemble_record.md) — criterion results over seeds.
- [ ] [phase_c_mutation_record.md](phase_c_mutation_record.md) — each criterion proved failing by mutation.
- [ ] [phase_c_sweep_record.md](phase_c_sweep_record.md) — parameter sweeps.
- [ ] [phase_c_diagnostic_record.md](phase_c_diagnostic_record.md) — diagnostic variants (evidence, never a verdict).
- [ ] [phase_c_level_occupancy.md](phase_c_level_occupancy.md) — `M11.C.16` level occupancy.

The records are generated by `tools/`; check them against the spec, not by recomputing.

## Not for review

Kept for history or as build process. None of these defines current behavior.

**The frozen grid engine** (`src/engine.py`, not the agent model):
[spec.md](spec.md), [logic_rules.md](logic_rules.md), [model_config.md](model_config.md),
[constitution.md](constitution.md), [background.md](background.md), [user_stories.md](user_stories.md),
[implementation_task_list.md](implementation_task_list.md),
[bowen_individual_family_model_spec.md](bowen_individual_family_model_spec.md) (v1.2, frozen).

**Superseded or already folded into the spec:**
[agent_model_proposal.html](agent_model_proposal.html) (architecture proposal; §10 is background),
[HANDOFF_spec_revision_10_from_method_literature.md](HANDOFF_spec_revision_10_from_method_literature.md),
[REVIEW_spec_rev10_2026-09-23.md](REVIEW_spec_rev10_2026-09-23.md),
[agent_raising_design_prompt.md](agent_raising_design_prompt.md),
[conversation_agents_maturity_emotion_2026-09.md](conversation_agents_maturity_emotion_2026-09.md).

**Not built yet:** [<PROPOSAL — RUN VISUALIZATION MAPPING.md>](<PROPOSAL — RUN VISUALIZATION MAPPING.md>) (drawing a
run as a diagram; build plan awaiting approval).

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
[theory/_TEMPLATE.md](theory/_TEMPLATE.md) (extraction template).

**Stray file:** [<model_explainer copy.textClipping>](<model_explainer copy.textClipping>) — a macOS clipboard
fragment, safe to delete.
