# EPModel v2 — the Phase B script (40 weeks)

> Drives the Phase B family with no policy (spec `M13`, Phase B). Its shape follows
> the worked trace in `docs/agent_model_proposal.html` §4.2 — a job loss, conflict,
> withdrawal, triangling the child, the grandmother recruited — plus the `TRIGGER`
> on the dormant Ana–Bruno tie that Phase B's exit condition tests (`M2.3a`).
>
> **Every value here is invented.** The script asserts nothing about what follows
> from these moves; Phase B tests the plumbing, and the one model claim it tests
> (gate G3) compares two arms. Intensities are scaled so the base appraisal
> (`M4.C.1`: intensity × conductance ÷ functional level) moves anxiety by a few
> points to about ten. Every delivery falls inside the 40 weeks.
>
> In Phase B every move is scripted and none is chosen; an `I-POSITION` here is a
> move record only — its state machine (`M5.D`) is Phase C.
>
> **A stressor acts once, on arrival.** Its duration is recorded (the job loss is a
> 34-week spell, `M1.F.6`) but in Phase B nothing reads it: how a spell weighs on
> people across its weeks is decided with the symptom channels (owner, 2026-10-06).

script_id: phase_b_40_weeks
instance_id: phase_b_reduced
ticks: 40
grade: [I]

## Events

| tick | kind | targets | tie | intensity | duration | binder |
|---|---|---|---|---|---|---|
| 0 | JOB_LOSS | `ravi` | — | 400 | 34 | — |
| 10 | TRIGGER | — | `ana`~`bruno` | 1.0 | 1 | — |

## Moves

| tick | actor | kind | targets | intensity | route | source_position |
|---|---|---|---|---|---|---|
| 1 | `ravi` | CONFLICT | `marta` | 200 | — | none |
| 2 | `marta` | DISTANCE | `ravi` | 100 | — | none |
| 3 | `marta` | DISTANCE | `ravi` | 100 | — | none |
| 4 | `ravi` | TRIANGLE | `nadia` | 150 | — | none |
| 5 | `marta` | DISTANCE | `ravi` | 100 | — | none |
| 6 | `marta` | OVERFUNCTION | `nadia` | 120 | — | none |
| 9 | `nadia` | UNDERFUNCTION | `marta` | 80 | — | none |
| 12 | `nadia` | UNDERFUNCTION | `ravi` | 80 | — | none |
| 15 | `nadia` | UNDERFUNCTION | `marta` | 80 | — | none |
| 18 | `nadia` | UNDERFUNCTION | `ravi` | 80 | — | none |
| 24 | `ana` | TRIANGLE | `marta` | 120 | — | none |
| 31 | `marta` | I-POSITION | `ravi` | 100 | — | none |
| 36 | `ravi` | PURSUE | `sofia` | 60 | — | none |
