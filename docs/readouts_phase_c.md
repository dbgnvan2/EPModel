# Pattern readouts — Phase C

Spec `M5.A.1a`: each core move is a single act. The patterns Bowen named are **readouts** over a
tie's or a triangle's history of reciprocal acts. They are never selectable, and each definition
below is `[I]`. Each is declared here before any criterion reads it. The code is
`src/bowen/readouts/patterns.py`; `read_pattern` refuses a name not listed here, and
`tests/bowen/test_patterns.py` fails if this document and the code disagree.

**W** is `pattern_window` (8 weeks). The other parameters are in `config/bowen/constants.md`.

| Pattern | Reads | Definition |
|---|---|---|
| `conflict` | a tie | Both members emit `CONFLICT` toward each other, each at least `pattern_min_acts` (2) times in W. |
| `over_underfunctioning` | a tie | The tie's hardened functioning habit is at or beyond `pattern_pole` (0.5) toward one member. In W, that member emitted `OVERFUNCTION` toward the other, or the other emitted `UNDERFUNCTION`. |
| `distance` | a tie | `DISTANCE` acts on the tie outnumber approaches (`PURSUE`, `STAY-IN-CONTACT`, `REDUCE_CUTOFF`) in W, and both members are below their contact optimum on the tie. |
| `cutoff` | a tie | The tie is non-interactive, and no `CUTOFF` was sent on it within W, so it has been severed for at least W. |
| `fixed_triangle` | a triangle | The same outsider across the last `pattern_fixed_activations` (3) activations of the triangle. |

**Projection** is Phase D: it needs a child-focused readout.

> **Human review before step 14** (plan §9). A definition can be applied correctly and still name the
> wrong thing. These need the owner's reading, and one rendered trace per pattern, before any criterion
> that reads them is believed. *Not yet reviewed.*
