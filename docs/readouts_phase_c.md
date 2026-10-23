# Pattern readouts — Phase C

Spec `M5.A.1a`: each core move is a single act. The patterns Bowen named are **readouts** over a
tie's or a triangle's history of reciprocal acts. They are never selectable, and each definition
below is `[I]`. Each is declared here before any criterion reads it. The code is
`src/bowen/readouts/patterns.py`; `read_pattern` refuses a name not listed here, and
`tests/bowen/test_patterns.py` fails if this document and the code disagree.

**W** is `pattern_window` (4 weeks); **N** is `pattern_min_acts` (3). The other parameters are in
`config/bowen/constants.md`.

**The owner's definitions, 2026-10-10** (`docs/PROPOSAL — RUN VISUALIZATION MAPPING.md` §0 and §4; this is the
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

## Negative runs — what `M11.C.48` reads (version 2 group G1, spec revision 36)

`src/bowen/readouts/negative_runs.py`; `tests/bowen/test_negative_runs.py` checks each definition on hand-built records.
Each definition is `[I]` (the owner's D12 of 2026-10-19, `docs/PROPOSAL — CUTOFF ANXIETY, BRIDGING AND REPAIR.md` §4;
the baseline as built, decided as recommended under the owner's instruction of 2026-10-22, pending the owner's review).
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
- The **baseline** for a first cutoff is the mean of the runs standing, in the cutoff's week, on the seed's ties with
  no `CUTOFF` by then and a contact in the window: the same point in time, read the same way, so neither arm has more
  weeks to build a run in. The longest run those ties reached within the window is reported beside it, not compared:
  a maximum over ties and weeks set against one point is biased against the criterion (learnings P7).
- Per seed, the treatment arm is the mean run before its first cutoffs and the baseline the mean of their baselines;
  both read one run. A seed with no cutoff reads 0 in both and is counted; `first_cutoffs` is reported beside the
  arms. `docs/review/group_g_readout.md` measures the same on 20 seeds before and after the revision.

**Onset loads in two readers.** The loads a symptom onset raises (`M7.D.1a`) travel on the stressor mechanism with
`exogenous=True`, as `M1.F.7` requires of every stressor, on the `ENDOGENOUS` channel. The evaluation readout's stress
term (`M11.G.2`) reads the channel and leaves them out, since they are an output of the run; the run chart
(`tools/run_chart.py`) marks events by the mechanism they travel on and shows them with the exogenous stressors.
