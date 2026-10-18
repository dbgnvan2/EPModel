# Version 1, condition V7: review of the outputs

**Status: for the owner's review, drafted 2026-10-17.** V7 of `docs/version_1_definition.md`: "No output presents the
model as speaking about a real family (`M11.F.9`)". It is checked by the owner's review, not by code. This form lists
every kind of output the model produces, what framing each carries, and where. The framing lines are checked by
`tests/test_version_1_outputs.py`, so a regenerated output that loses its framing fails the suite. Whether the framing
is enough is the owner's call.

`M11.F.9` has three clauses: (a) no output presented as concerning a named real family or person, or as advice to one;
(b) initialising from a real family only as ranges, under `M15` (not built in version 1, §3); (c) no counterfactual
reported from parameters tuned to reproduce a known history. `M11.F.10` adds that the model is never called a
"virtual family" without `M11.F`'s framing beside it.

| Output | Produced by | Framing it carries | Owner |
|---|---|---|---|
| Rendered run traces, `docs/review/phase_b_trace_seed7.md`, `phase_b_trace_seed7_nadia.md`, `phase_c_trace_seed7.md` | `src/bowen/render/trace.py` | the block "It does not concern any real family or person, and nothing here is advice (`M11.F.9`)", and "not a 'virtual family' in the predictive sense (`M11.F.10`)" | [ ] enough |
| The fam-diag export, `docs/review/phase_c_famdiag_seed7.json` | `tools/famdiag_export.py` | "Simulated run — not a real family (spec M11.F.9). The people are the model's invented reference family." | [ ] enough |
| Ensemble records, `docs/phase_c_ensemble_record.md`, `phase_d_ensemble_record.md`, `phase_d2_ensemble_record.md` | `tools/ensemble_record.py`, `tools/phase_d_record.py` | "A premise passing confirms that the code renders the spec; it is not a finding about families." | [ ] enough |
| Mutation, sweep, diagnostic and occupancy records | `tools/mutation_record.py`, `tools/phase_d_mutation_record.py`, `tools/sweep_record.py`, `tools/diagnostic_record.py`, `tools/level_occupancy.py` | none of their own: they report how the code behaves under edits and settings, and name no family | [ ] enough |
| The version 1 status page, `docs/version_1_status.md` | `tools/version_1_status.py` | none of its own: criteria and verdicts only | [ ] enough |
| `README.md` | by hand | "It answers 'if Bowen's account is right, what follows for a family shaped like this?' — never 'what should this family do?'" (point 3) | [ ] enough |
| `docs/model_explainer.md` §1.1 | by hand | the only use of "virtual family", in a section that says why the model is not one | [ ] enough |

**Clause (c)** has no output to check in version 1: no run's parameters are fitted to a known history, and `M15`'s
import is outside version 1 (§3). The reference family (`config/bowen/family_phase_d.md`) is invented, every value
graded `[I]`.

**One wording item for the owner.** `config/bowen/family_phase_d.md` opens "The owner's family, 2026-10-10", and
`docs/theory/_STATUS.md` says "`M2.A` is the owner's family". Both mean the reference family the owner specified (plan
D5); a reader could take them to mean the owner's real family, which `M11.F.9(a)` forbids an output to suggest. A
wording such as "the reference family the owner specified" would remove the reading. Not changed: the family file is
hashed with the engine, so the change waits for the next code change.

**Owner sign-off:** [ ] V7 met.
