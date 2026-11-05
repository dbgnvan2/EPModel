# EPModel version 3 — the emergence audit's exclusions, controls and stage (`M11.1g`)

> What `tools/emergence_audit.py` reads: which passing rows the audit leaves out and why, the two controls that show
> its ablation removes learning and nothing else, and which stage of the run the record renders. **The engine never
> reads this file.** It is outside `config/bowen/` because the engine hash covers every `*.md` there
> (`tools/ensemble_record.py::code_hash`), as `config/sweep/sweep.md` is: a change here reruns nothing.
> Parsed strictly: a row the records do not hold, an exclusion of a row that is not in scope, a control on a row that
> does not pass, an unknown ablation or expectation, a stage other than `pilot`, `l0`, `carriers` or `full`, or a pilot row that is not
> in scope raises. The ablations and carrier candidates are code edits and stay in the tool, beside the other mutants
> (`docs/plan_v3_emergence_audit.md` §5).

## Exclusions

A passing composite or mixed row that cannot pass with learning removed, by construction. Each is printed in the
record with its reason; none is dropped silently.

| criterion | reason |
|---|---|
| `M11.C.39` | its treatment arm differs from its baseline only in learning (`LEARNING_OFF`), so with learning removed from both arms the two arms are the same model and it cannot pass; it is the positive control below, not a finding |

## Controls

The build refuses to render the record if a control does not give its expected verdict: then the ablation is not
doing what it says. A control is never a finding.

| control | criterion | ablation | expected | reason |
|---|---|---|---|---|
| positive | `M11.C.39` | L0 | red | its arms differ only in learning; if it survives L0, L0 did not remove learning |
| negative | `M11.C.3` | L0 | survived | a premise, `M1.C.1`'s transfer, the physics of an act; if it goes red under L0, L0 changed something beyond learning |

## Stage

`pilot` renders the pilot only (`docs/plan_v3_emergence_audit.md` §7 step 3): L0 on the rows below and on the two
controls, with no carriers. `l0` renders every in-scope row and the two controls under L0 only, with no carriers (step 4; the
owner's choice of 2026-11-03: L0 first, then L1 to L4 and the carriers costed from its survivors). `carriers` renders every
in-scope row and the two controls under L0, and the carriers of each L0 survivor, with no L1 to L4 (steps 4 and 6; the
owner's choice of 2026-11-04, after L0: the carriers of the 10 survivors first, L1 to L4 waiting). `full` renders every
in-scope row under L0 to L4 and the carriers of each L0 survivor (steps 4 to 6, after T-181's full sweep).

| setting | value |
|---|---|
| stage | carriers |

## Pilot rows

`M11.C.41[light: lower level]` was a pilot row (a short Phase C row, 80 ticks, and a mixed one) until the owner
declared its premise limbs on 2026-11-05 (T-192), which took it out of the audit's scope; a pilot row must be in scope.

| criterion | reason |
|---|---|
| `M11.C.48[phase_d]` | T-180: it passes at revision 47 and its named mutant `cutoff-blind-to-learned-value` survives |
| `M11.C.27[unstable,remove_one]` | a short Phase C row (21 ticks) |
