# Spec review checklist

This file is the index to every document that defines, justifies or records the agent model. It is the starting point for a review of the spec and, after that, a review of runs against the spec.

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

- [ ] [bowen_agent_model_spec_v2.md](bowen_agent_model_spec_v2.md) — the spec (v2.0, revision 27). Objects (M1),
      reference family (M2), clocks and ordering (M3), the fast tick (M4), moves (M5), invariants (M6), the slow tick
      (M7), live positions (M8), beliefs (M9), parameters (M10), acceptance criteria (M11), framing (M11.F), the
      readout schema (M11.G), out of scope (M12), the run log and trace (M16).
- [ ] [../config/bowen/constants.md](../config/bowen/constants.md) — the parameter values the engine uses.
- [ ] [../config/bowen/constants_changes.md](../config/bowen/constants_changes.md) — every change to those values,
      and why.
- [ ] [../config/bowen/event_kinds.md](../config/bowen/event_kinds.md) — the kinds of act and their layers.
- [ ] [../config/bowen/policy.md](../config/bowen/policy.md) — the selection policy.
- [ ] [../config/bowen/family_phase_c.md](../config/bowen/family_phase_c.md) — the family run in Phase C.
- [ ] [../config/bowen/family_phase_d.md](../config/bowen/family_phase_d.md) — the owner's reference family (spec `M2.A`, revision 13), with its life course from revision 28: starting ages as settings, who earns, the coachee (`M2.A.0j`, `M2.A.0k`).
- [ ] [../config/bowen/partners_phase_d.md](../config/bowen/partners_phase_d.md) — the reference family's partners and their children, drafted for the owner to edit (spec `M7.C.1h`, `M7.C.1i`, revision 28).
- [ ] [../config/bowen/pattern_acts.md](../config/bowen/pattern_acts.md) — which acts each pattern readout counts.
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
- [ ] [../config/bowen/criteria.md](../config/bowen/criteria.md) — how each acceptance criterion is set up.
- [ ] [../config/bowen/criterion_roles.md](../config/bowen/criterion_roles.md) — each criterion's class (premise or
      finding, `M11.5`).
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
- [ ] [<DECISIONS — CUTOFF SEMANTICS.md>](<DECISIONS — CUTOFF SEMANTICS.md>) — whether the engine's CUTOFF should become rejection (open).
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
- [ ] [review/phase_c_famdiag_seed7.json](review/phase_c_famdiag_seed7.json) — the same run as a fam-diag diagram; open it in EPModel Fam-Diag (`../epmodel-fam-diag`, port 5174; `tools/famdiag_export.py`).
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

**For review, 2026-10-15:** [version_1_definition_DRAFT.md](version_1_definition_DRAFT.md) — a DRAFT definition of version 1 of the agent model, its conditions each checked by a test, a record or an owner sign-off, where the model stands against it, and the owner's questions; 1, 2, 3 and 8 decided 2026-10-15, 4–7 open. Not approved.

**For review, 2026-10-13:** [phase_d_status_2026-10-13.md](phase_d_status_2026-10-13.md) — Phase D part I after the
overnight run: every build item and D-I criterion as done, partial or not done, and what the owner should look at
first (spec revisions 24–27 are Claude's recommendations, pending owner review).

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
[theory/_TEMPLATE.md](theory/_TEMPLATE.md) (extraction template).

**Stray file:** [<model_explainer copy.textClipping>](<model_explainer copy.textClipping>) — a macOS clipboard
fragment, safe to delete.
