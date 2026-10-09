# EPModel v2 — the Phase C criteria's declared settings

> Every number a Phase C criterion (`src/bowen/ensemble/criteria.py`) sets for its arms: horizons, the week of a
> scripted act, the levels and stress compared, the declared spell (`M11.3`) and the state an arm starts from.
> **All `[I]`, declared before any criterion ran** (2026-10-07). They moved here from the module on 2026-10-08,
> unchanged, so that `M11.D.2`'s magic-literal scan covers the criteria (Hermes gate finding 4); the regenerated
> ensemble record's verdicts are identical. Parsed strictly: a row the criteria do not read, or a setting they
> read that is missing, raises at load.
>
> * `spell` — a `JOB_LOSS` of the given intensity to each parent every `every` weeks from week `from`. `M11.C.41`'s
>   *light* stress is the first parent only, every `light_every` weeks from the same week.
> * `run_after` — weeks run after the scripted act's week, so the run is `t0 + run_after` weeks long.
> * `level_*` — how far every member's basic and functional level is lowered in that arm.
>
> **Post-hoc restatement, 2026-10-09 (step S of `docs/DECISIONS — PHASE C FAILING.md`, approved by the owner).**
> The declared weeks stand, but in both arms of every criterion that scripts an act after week 0, the ties its
> scripted acts cross are held open from week 0 until the act (`M11.C.29`: until its last withdrawal): neither
> member may cut them, so `CUTOFF` across them is not in their legal set. Without the hold the act was illegal in
> 25–60% of seeds, and a seed where both arms skipped added a difference of exactly 0. The approved first choice
> was to script each act at the latest week legal in every seed of the ensemble's cap (500). Measured
> (`s-scripted-weeks` in `docs/phase_c_diagnostic_record.md`): that week is 0 for every criterion, and for
> `M11.C.29`'s distance in disguise no week works. Week 0 was not used, for a reason that follows from the code:
> every member starts on their chronic floor, and decay lifts anyone below the floor back to it, so no relief
> can show there. One run with every act at week 0 was made and its verdicts were seen before it was discarded
> (`M11.C.3`'s pair relief came out at +0.09); it is not kept. So the approved fallback, holding the ties open,
> applies to every criterion.
>
> **Longer horizons, 2026-10-09 (owner decision, after step S).** With every act made, `M11.C.4`'s later limb and
> three of `M11.C.27`'s cells still failed at their declared horizons (`M11.C.4`'s nodal event at week 30;
> `M11.C.27` read 3 weeks after its act). The owner allowed a longer horizon. The rule, fixed before running: one
> value, the longest horizon any Phase C criterion already declares (104 weeks, `M11.C.1`, `.16`, `.38`), run once
> and reported whatever it shows. So `M11.C.4`'s nodal event moves to week 104, and `M11.C.27` is read 104 weeks
> after its act. The earlier verdicts stay in `docs/DECISIONS — PHASE C FAILING.md`.

| criterion | setting | value |
|---|---|---|
| spell | kind | JOB_LOSS |
| spell | intensity | 120.0 |
| spell | every | 4 |
| spell | from | 4 |
| spell | light_every | 8 |
| spell | light_parents | 1 |
| scripted_act | intensity | 100.0 |
| M11.C.1 | weeks | 104 |
| M11.C.1 | level_baseline | 0.0 |
| M11.C.1 | level_treatment | 10.0 |
| M11.C.3 | t0 | 12 |
| M11.C.3 | run_after | 2 |
| M11.C.3 | latency | 1 |
| M11.C.4 | t0 | 8 |
| M11.C.4 | nodal | 104 |
| M11.C.4 | run_after_nodal | 2 |
| M11.C.5 | weeks | 60 |
| M11.C.5 | t0 | 4 |
| M11.C.16 | weeks | 104 |
| M11.C.16 | level_baseline | 0.0 |
| M11.C.16 | level_treatment | 15.0 |
| M11.C.16 | window | 52 |
| M11.C.19 | weeks | 1 |
| M11.C.19 | low_axis | 0.2 |
| M11.C.19 | high_axis | 0.8 |
| M11.C.25 | weeks | 52 |
| M11.C.25 | no_pole | 0.5 |
| M11.C.27 | t0 | 8 |
| M11.C.27 | run_after | 104 |
| M11.C.27 | unstable_impingement | 0.8 |
| M11.C.29 | weeks | 80 |
| M11.C.29 | t0 | 4 |
| M11.C.29 | disguise_span | 8 |
| M11.C.29 | disguise_every | 2 |
| M11.C.32 | weeks | 30 |
| M11.C.32 | t0 | 0 |
| M11.C.32 | angry_impingement | 1.0 |
| M11.C.35 | t0 | 6 |
| M11.C.35 | run_after | 2 |
| M11.C.35 | conductance_high | 1.0 |
| M11.C.35 | conductance_low | 0.4 |
| M11.C.38 | weeks | 104 |
| M11.C.38 | level_1 | 0.0 |
| M11.C.38 | level_2 | 5.0 |
| M11.C.38 | level_3 | 10.0 |
| M11.C.38 | level_4 | 15.0 |
| M11.C.41 | weeks | 80 |
| M11.C.41 | level_higher | 0.0 |
| M11.C.41 | level_lower | 10.0 |
| M11.C.42 | t0 | 10 |
| M11.C.42 | weeks | 40 |
| M11.C.44 | weeks | 80 |
| M11.C.45 | weeks | 80 |
