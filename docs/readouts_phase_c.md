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
| `cutoff` | a tie | At least N refusals in W by one member. A refusal is a `CUTOFF` sent to the other in the same or the next week after an approach from them. A tie that is only silent reads as `distance`. From Phase D the engine refuses what a cutter receives (`M1.B.3c`). This readout is unchanged and does not read those engine refusals; whether it should is open (`TODO.md`). |
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
