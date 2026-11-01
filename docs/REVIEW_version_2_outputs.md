# Version 2, condition W8: review of the outputs

**Status: signed off by the owner 2026-11-01** (drafted the same day, at `5957648`, engine hash `09ca2b46c7e0`). W8 of
`docs/version_2_plan.md` §4: "No output presents the model as speaking about a real family (`M11.F.9`), as V7." It is
checked by the owner's review, not by code. This form follows version 1's (`docs/REVIEW_version_1_outputs.md`, signed
off 2026-10-17): the outputs version 1 had are listed again with their current framing, and the outputs version 2
added are listed beside them. Framing lines are checked by `tests/test_version_1_outputs.py` (traces, chart, fam-diag
export, ensemble records, `M11.F.10`'s term) and `tests/test_version_2_outputs.py` (version 2's readout pages), so a
regenerated output that loses its framing fails the suite. Whether the framing is enough is the owner's call.

`M11.F.9` has three clauses: (a) no output presented as concerning a named real family or person, or as advice to one;
(b) initialising from a real family only as ranges, under `M15`; (c) no counterfactual reported from parameters tuned
to reproduce a known history.

## Outputs version 1 had

| Output | Produced by | Framing it carries | Owner |
|---|---|---|---|
| Rendered run traces, `docs/review/phase_b_trace_seed7.md`, `phase_b_trace_seed7_nadia.md`, `phase_c_trace_seed7.md` | `src/bowen/render/trace.py` | the block "It does not concern any real family or person, and nothing here is advice (`M11.F.9`)" | [x] enough |
| The run chart, `docs/review/phase_c_chart_seed7.html` | `tools/run_chart.py` | the trace's framing block, as plain text | [x] enough |
| The fam-diag export, `docs/review/phase_c_famdiag_seed7.json` | `tools/famdiag_export.py` | "Simulated run — not a real family (spec M11.F.9). The people are the model's invented reference family." | [x] enough |
| Ensemble records (Phase C, Phase D parts I and II) | `tools/ensemble_record.py`, `tools/phase_d_record.py` | "A premise passing confirms that the code renders the spec; it is not a finding about families." | [x] enough |
| Mutation, sweep, diagnostic and occupancy records | `tools/mutation_record.py`, `tools/phase_d_mutation_record.py`, `tools/sweep_record.py`, `tools/diagnostic_record.py`, `tools/level_occupancy.py` | none of their own: they report how the code behaves under edits and settings, and name no person | [x] enough |
| `README.md` | by hand | point 3: it answers what follows for a family shaped like this if Bowen's account is right, never what a family should do | [x] enough |
| `docs/model_explainer.md` §1.1 | by hand | the only use of `M11.F.10`'s term, in a section that says why the model is not one | [x] enough |

## Outputs version 2 added

| Output | Produced by | Framing it carries | Owner |
|---|---|---|---|
| Group G's readout and stage 3's probe and trace pages: `docs/review/group_g_readout.md`, `stage_3_displaced_coping_probe.md`, `stage_3_iposition_probe.md`, `stage_3_projection_trace.md` | `tools/group_g_readout.py`, `tools/stage_3_displaced_coping_probe.py`, `tools/stage_3_iposition_probe.py`, `tools/stage_3_projection_trace.py` | "Not a real family (`M11.F.9`)." near the top of each | [x] enough |
| Stage 3's criteria and mutation pages: `docs/review/stage_3_step_7_criteria.md`, `stage_3_step_8_criteria.md`, `stage_3_step_9_criteria.md`, `stage_3_step_7_c2_mutants.md`, `stage_3_unit_mutation_gate.md` | `tools/stage_3_targeted_run.py`; the gate page `tools/mutation_gate.py` | none of their own: verdicts and effects only; none names a person | [x] enough |
| The event-size check, `docs/review/stage_3_t130_event_sizes.md` | by hand, stage 3 step 1 (T-130) | none of its own: it compares the config's event names and sizes with Holmes and Rahe's published scale of life events, which describes no family | [x] enough |
| The version 2 sweep record, `docs/version_2_sweep_record.md` | `tools/version_2_sweep.py` | "Every number here is a measurement of a range, not a verdict"; names no person | [x] enough |
| The version 2 status page, `docs/version_2_status.md` | `tools/version_2_status.py` | none of its own: criteria and verdicts. One criterion's note names Nadia, a child of the invented reference family, to say which child an arm acts on | [x] enough |
| The changes pages, `docs/version_2_stage_1_changes.md` and `docs/version_2_revision_{36,39,41,43,44}_changes.md` | `tools/record_changes.py` | none of their own: they compare records before and after a regeneration | [x] enough |

**Clause (b)** has no output to check: `M15`'s import is not built (`M15.D.1` is *not done* in
`docs/spec_coverage.md`; Phase E). **Clause (c)** has no output to check: no run's parameters are fitted to a known
history. The reference families (`config/bowen/family_phase_c.md`, `family_phase_d.md`) are invented, every value
graded `[I]`; the event sizes T-130 set come from a published scale's means, the same for every agent, and are not
tuned to any family's history.

**Owner sign-off:** [x] W8 met (the owner, 2026-11-01: "go with your recommendations on the sign offs").
