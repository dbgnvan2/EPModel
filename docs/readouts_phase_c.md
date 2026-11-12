# Pattern readouts — Phase C

Spec `M5.A.1a`: each core move is a single act. The patterns Bowen named are **readouts** over a
tie's or a triangle's history of reciprocal acts. They are never selectable, and each definition
below is `[I]`. Each is declared here before any criterion reads it. The code is
`src/bowen/readouts/patterns.py`; `read_pattern` refuses a name not listed here, and
`tests/bowen/test_patterns.py` fails if this document and the code disagree.

**W** is `pattern_window` (4 weeks); **N** is `pattern_min_acts` (3). The other parameters are in
`config/bowen/constants.md`.

**The owner's definitions, 2026-10-10** (`docs/archive/PROPOSAL — RUN VISUALIZATION MAPPING.md` §0 and §4; this is the
plan §9 review of this document):
- An emotional pattern is a **repeating** pattern of activity: N or more acts in W. A single act is an event, not a pattern.
- **A pattern ends only when a new pattern emerges** on the same subject (`current_patterns`).
- **No interaction is an emerging level of distance.**
- **Cutoff is rejection, not absence.**
- **Pursuit** is fusion, over/underfunctioning or projection, depending on who pursues whom.

Acts count when they pass between the two members of the tie, in either direction unless stated.

| Pattern | Reads | Definition |
|---|---|---|
| `conflict` | a tie | At least N `CONFLICT` acts on the tie in W, with **both** members contributing. |
| `distance` | a tie | Either at least N `DISTANCE` acts in W, more than the approaches (`PURSUE`, `STAY-IN-CONTACT`, `REDUCE_CUTOFF`), with both members below their contact optimum on the tie; **or no act on the tie for at least W weeks**, at a level equal to the weeks without interaction. |
| `over_underfunctioning` | a tie | At least N acts in W in one direction, more than in the other. An act counts for X as the over side when X takes charge of Y (`OVERFUNCTION`), or Y gives way to X (`UNDERFUNCTION`) or goes along with X (`PURSUE`). A parent's acts that count as projection do not count here. |
| `fusion` | a tie | At least N `PURSUE` acts in W, with **both** members pursuing: each going along with the other. Not read when `over_underfunctioning` is (lopsided pursuit is over/underfunctioning). |
| `projection` | a tie | A parent–child tie: at least N acts in W from the parent focused on the child: `OVERFUNCTION` or `PURSUE` toward the child, or a `TRIANGLE` that recruits the child. The act set is the child-projection sink's (`src/bowen/engine/sinks.py`), on every parent–child tie of the family. |
| `cutoff` | a tie | *Revision 17, read from behaviour* (the owner: a `CUTOFF` act is "just words that need behaviour to back it up"). Member A's cutoff of B: at least N of B's approaches in W refused — refused by the engine because A held rejection (`M1.B.3c`), or answered by A within a week with `CUTOFF` or `DISTANCE` — and no act with contact from A to B in W. A `CUTOFF` act alone reads as nothing. A tie that is only silent reads as `distance`. |
| `fixed_triangle` | a triangle | The same outsider across the last `pattern_fixed_activations` (3) activations of the triangle. |

**Which pattern a subject holds** (`current_patterns`; owner decision D2):
- A subject keeps its pattern until a different pattern's start condition holds, or the same pattern holds pointing
  the other way: a different over side, parent or refuser, or a different outsider.
- When two patterns start in the same read, the one with more acts wins, then the one whose first act was earlier `[I]`.
- One exception, `[I]`: silence does not replace a cutoff, because the silence is the cutoff continuing.

- **Projection holds longer** (the owner, 2026-10-10). Once read, it stays until `pattern_projection_hold` (12)
  weeks pass with no projection act, of any count, from that parent to that child. Until then nothing replaces it.
  After that, the current reading replaces it; with none, the projection has stopped.

**Owner answers, 2026-10-10:**
- Lopsided pursuit is over/underfunctioning: fusion is read only when neither side's going-along outweighs the
  other's.
- A parent's overfunctioning or pursuit toward a child is projection evidence, never over/underfunctioning.

**Not reviewed by the owner:** the tie-break rule and the silence exception.

> **Owner review (plan §9), 2026-10-10:** the definitions above are the owner's. Rendering one trace per pattern is
> done by the run visualization (`tools/famdiag_export.py`). No criterion reads these readouts (checked 2026-10-10).

## Negative runs — what `M11.C.48` reads (version 2 group G1, spec revision 36; T-123 at revision 43; T-179 at revision 47; T-185 at revision 57)

`src/bowen/readouts/negative_runs.py`; `tests/bowen/test_negative_runs.py` checks each definition on hand-built records.
Each definition is `[I]` (the owner's D12 of 2026-10-19, `docs/archive/PROPOSAL — CUTOFF ANXIETY, BRIDGING AND REPAIR.md` §4;
the baseline as built, decided as recommended under the owner's instruction of 2026-10-22, approved by the owner 2026-10-24 in T-125).
The classes of contact are `config/bowen/contact_classes.md`.

- A **negative contact** on a tie is an act in the `negative` or `rebuff` set emitted by one member toward the other,
  or any move across the tie that was refused (a `refusal` record), each counted once.
- A **positive contact** is an act in the `positive` set delivered (emitted and not refused). It ends the run. A week
  holding both a negative and a positive contact counts its negatives and keeps the run (the negative applies, the
  owner's Q6 for bridging).
- A **run** is the number of negative contacts since the last positive contact, in either direction. Weeks with no
  contact do not end it.
- A tie's **first cutoff** is the first `CUTOFF` emitted across it; its run is the run standing in that week, the
  cutoff's own week counted and the cutoff itself not. A tie the family declares cut off has no first cutoff and is
  not an uncut tie: it is left out on both sides.
- A tie's **sides** at a week is how many of its two members had sent across it, by then, a move other than
  `CUTOFF` — two-sided, one-sided or neither. The cutoff itself is not such a move: it is the act the run precedes.
- The **baseline** for a first cutoff is the mean of the runs standing, in the cutoff's week, on the seed's ties with
  no `CUTOFF` by then, a contact in the window, and as many sides as the cut tie: the same point in time, read the
  same way, with the same number of members acting, so neither arm has more weeks to build a run in nor more members
  putting contacts into it. The longest run those ties reached within the window is reported beside it, not compared:
  a maximum over ties and weeks set against one point is biased against the criterion (learnings P7).
- **Defect found at the re-sweep of 2026-10-23 (T-123, learnings P7):** a first cutoff with no such tie (no other tie
  with a contact in the window by that week) was given a baseline of 0 and its own run kept in the treatment arm.
  Measured on the regenerated runs: 139 of 700 first cutoffs on the Phase B family (100 seeds) and 284 of 749 on the twelve-person family (20 seeds); with those left out of both arms the difference is −0.28 ± 0.15 and −0.21 ± 0.18 (2 se) against +0.43 ± 0.16 and +0.44 ± 0.10 as built. **The pass recorded on 2026-10-23 is not evidence.**
- **Since revision 43 (T-123, Claude's recommendation 2026-10-29 under the owner's instruction to proceed, approved by the owner 2026-10-31):** a first cutoff with no uncut tie in its
  week is **unmatched**: it is left out of both arms and counted (`FirstCutoff.matched`, `unmatched`; the
  treatment arm's report-only readout `unmatched_cutoffs`). A seed whose first cutoffs are all unmatched reads 0 in
  both arms, as a seed with no cutoff does. The criterion runs 200 seeds, at least the sweep's rerun count, so
  `M17.A.1` never refuses it (T-163). At revision 43's regeneration (`da5981f6ec0b`, 200 seeds; `docs/phase_d2_ensemble_record.md`) the criterion **fails** on both families: −0.291 ± 0.12 on the Phase B family and −0.11 ± 0.063 on the twelve-person family, with 1.31 and 13.6 unmatched first cutoffs a seed left out.
- **Defect found by T-176's trace (2026-11-02, learnings P7):** the baseline reads an uncut tie only if it had a
  contact by the cutoff's week, but a cut tie with no contact by its week entered the treatment at a run of 0. 329 of
  1137 matched first cutoffs on the Phase B family and 1419 of 4791 on the twelve-person family were such ties; with
  the same condition on both arms the difference was +0.029 ± 0.136 and +0.344 ± 0.078 against the record's −0.319 ±
  0.104 and −0.113 ± 0.062 (`docs/review/t176_negative_run_trace.md`; the runs are those of `09ca2b46c7e0`, and the trace was regenerated at `0b363cb3186c` with the same runs).
- **Since revision 47 (T-179, the owner's option (a) of 2026-11-02; `docs/archive/plan_t179_negative_run_contact_condition.md`):**
  one contact condition on both arms, a contact in the cutoff's week or before (`negative_runs.in_contact`). A first
  cutoff on a tie with no contact by its week is **no-contact**: it is left out of both arms and counted
  (`FirstCutoff.contact`, `no_contact`; the treatment arm's report-only readout `no_contact_cutoffs`). Each first
  cutoff is matched, unmatched or no-contact, and only one of them: a no-contact cutoff with no uncut tie either is
  counted as no-contact, and `unmatched` counts only cutoffs on a tie in contact.
  Uncut ties with no contact are still
  left out and not counted: counting them needs the week from which each tie existed (people are born in the
  twelve-person family's run), the option (b) the owner did not choose. So the version 2 plan's design point for G1,
  that ties with no contact are "reported, not dropped" (`docs/version_2_plan.md`), is met for cut ties and not for
  uncut ones.
- **Since revision 57 (T-185, the owner's Q3 (a) of 2026-11-09):** the baseline reads only the uncut ties in contact
  with as many sides as the cut tie. A first cutoff with none is **unsided**: it is left out of both arms and counted
  (`FirstCutoff.sided`, `unsided`; the treatment arm's report-only readout `unsided_cutoffs`). Each first cutoff is now
  matched, no-contact, unmatched or unsided, and only one of them, in that order of precedence: a no-contact cutoff is
  counted as no-contact whatever its uncut ties, `unmatched` counts a cutoff in contact with no uncut tie in contact at
  all, and `unsided` one with such ties, none of them of its own sides. The reason is T-185's measurement: 51–53% of
  cut ties are two-sided against 32–33% of baseline ties, and a two-sided tie holds a run about 0.8 longer, so an
  unmatched baseline carries part of the direction (learnings P7). What is left after matching is +0.104 ± 0.070 with
  the emergence audit's L0 applied and +0.098 ± 0.082 with learning on.
  A tie's sides are counted over every move but `CUTOFF` sent across it, which is wider than the contact the run itself
  is built from (`M4.C` contacts only): a tie one member only ever pressured or distanced across counts as acted on by
  that member. The sides condition therefore matches on who acts, not on who makes contact.
  Each of the three exits that leaves a first cutoff out of both arms reports the mean run of what it left out, beside
  the arms and report-only: `unmatched_left_out_run` and `unsided_left_out_run` (a no-contact cutoff stands on a run of
  0 by construction, so there is nothing to report for it). A drop that selects on the run itself is then visible and
  not only counted (learnings P2/P9). Measured on the Phase B family over the criterion's own 200 seeds (2026-11-10,
  before the regeneration): of 1400 first cutoffs, 365 are no-contact and 1035 in contact; of those 1035, 696 are
  matched and carry a mean run of 1.91, 227 are unmatched with a mean of 4.18, and 112 are unsided with a mean of 1.35.
  Each drop therefore selects on the run, and in opposite directions: the unmatched exit leaves out longer runs than it
  keeps and lowers the treatment arm, the unsided exit shorter ones and raises it. Neither is a verdict; the arms are
  read at the regeneration.
  On the regenerated record (`0b363cb3186c`, `docs/phase_d2_ensemble_record.md`) `M11.C.48` fails with no difference
  and `M11.C.48[phase_d]` passes; the named mutant survives on the latter (T-180). It passes with all learning removed too: its arms differ in how
  many members act on the tie (51% of cut ties are acted on by both members, against 33% of the baseline's), and such ties
  hold longer runs, and the cutter's other ties run longer than the family's;
  the +0.104 ± 0.070 left after matching on how many members act is not explained (T-185,
  `tools/c48_carrier_probe.py`; spec revision 50). Revision 57 puts that matching into the baseline, so these verdicts
  and counts are reread at the regeneration on its engine. On the Phase B family the fail is not a finding about the run before a cutoff either: its arms are
  unmatched in the same way, and its baseline also mixes tie types, the spouse and parent ties (conductance 1.0) with the
  grandparent ties (0.6), and counts ties cut later in the run (spec revision 51, T-195). T-196's probe (spec revision
  52) found that matched on tie class no difference is detected (−0.004 ± 0.186), and that at every matched cutoff all the uncut
  baseline ties are cut later, so no never-cut baseline exists on this family. The 200-seed trace,
  `docs/review/t176_negative_run_trace.md`, gives the counts: 365 of 1400 first cutoffs no-contact on the Phase B
  family, 2121 of 7509 on the twelve-person family.
- Per seed, the treatment arm is the mean run before its matched first cutoffs and the baseline the mean of their baselines;
  both read one run. A seed with no matched first cutoff (none at all, or all of them unmatched or no-contact) reads 0
  in both; the report-only `matched_seed` is 1 on a seed with a matched first cutoff, so its mean is the share of seeds
  the arms read (learning-qa, 2026-11-02); `first_cutoffs` is reported beside the
  arms. `docs/review/group_g_readout.md` measures the same on 20 seeds before and after the revision.

**Onset loads in two readers.** The loads a symptom onset raises (`M7.D.1a`) travel on the stressor mechanism with
`exogenous=True`, as `M1.F.7` requires of every stressor, on the `ENDOGENOUS` channel. The evaluation readout's stress
term (`M11.G.2`) reads the channel and leaves them out, since they are an output of the run; the run chart
(`tools/run_chart.py`) marks events by the mechanism they travel on and shows them with the exogenous stressors.

## The supply readout — who supplied whom (`M11.G.5`, spec revision 55, T-202)

`src/bowen/readouts/roles.py`; `tests/bowen/test_roles_readout.py`. Computed after a run from its records; nothing in
the engine or the policy reads it (`M1.A.14b`), and no role is stored (`tests/test_repo_layout.py::test_m1a14b_sc11_no_role_field`
scans config keys and `Person` fields for the words in `config/lint/role_labels.md`). Each definition is `[I]`.

- A **supplying move** is a move of a kind listed in `config/lint/supply_kinds.md` (`STAY-IN-CONTACT`, attention
  given; `OVERFUNCTION`, help given). The list is editorial and kept outside `config/bowen/`, so it does not move the
  engine hash.
- **`supply_counts`**: per ordered (sender, target) pair, the supplying moves sent over a declared window, each target
  of a move counted once.
- **`main_supplier`**: of a declared set of people, the one who sent a given person the most supplying moves; no one
  on a tie for the most or when none did.
- **`ReadingTap`** (only when the social cues are on): a `watch` that copies each week's cue readings made by targets,
  never by witnesses, and reports the mean perceived attention and approval cue per (sender, receiver).
- A role name may label this readout in a report ("the one who supplies the mother most"), never a person.
- `M11.C.54` reports the child's supplying moves toward the parent; `M11.C.55` compares, per parent, how often the
  child with the highest learned value on the supplying kinds toward that parent is that parent's main supplier over
  the window against two things: how often the child who supplied that parent most before the window is, and what a
  predictor drawing one of the children at random would name (the parents supplied ÷ the children). Both have to hold.
  Since revision 57 the learned value is read `relief_lead_weeks` (52) weeks before the window, not at its start, so
  it is not the value the policy selects by over the window, and how often the first-born is the main supplier is
  reported beside the arms instead of being the comparison — on the twelve-person family it predicts below chance
  (T-206, the owner's decision of 2026-11-09; `docs/review/2026-11-09_sc_pass_probes.md`).

## What a verdict reports beside its readouts (`M11.1f`, spec revision 59, T-216, T-217, T-239 (c))

- **Each arm's required moves.** A criterion names the moves its treatment arm must make (`requires_moves`) and, since
  revision 59, those its baseline must make (`requires_moves_baseline`); a move that never occurs in its own arm fails
  the verdict and is listed with "(baseline)" when it is the baseline's. `M11.C.18` (baseline `CUTOFF`), `M11.C.20`
  (baseline `UNDERFUNCTION`) and `M11.C.23` (baseline `STAY-IN-CONTACT`) gate both arms.
- **Each arm's counts.** Every count an arm records under a parenthesised key — a child who died before the age read
  (`M11.C.6`), a dyad with no pole formed (`M11.C.25`, new), the drivings booked (`M11.C.26`, which now books all
  three, not the last), the exits `M11.C.48` leaves out — is summed over the seeds per arm and printed in the record's
  "Counts the arms record" section. The scripted-act bookkeeping is reported apart, as before.
- **The share of differing pairs.** Every readout, report-only rows included, carries the share of seeds whose two
  arms differ (`nonzero_share`), printed as "pairs differing". No minimum is set; one is decided after the shares are
  seen at the next regeneration.
