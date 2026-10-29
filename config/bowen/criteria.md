# EPModel v2 — the Phase C criteria's declared settings

> Every number a Phase C criterion (`src/bowen/ensemble/criteria.py`) sets for its arms: horizons, the week of a
> scripted act, the levels and stress compared, the declared spell (`M11.3`) and the state an arm starts from.
> **All `[I]`, declared before any criterion ran** (2026-10-07); each row says so in its `grade` column (revision
> 31, R31.B2: the Phase D rows, `M11.C.2`'s ages, `M11.C.10`'s `renew_every` and `seeds`, `M11.C.47`'s fixture, are
> `[I]` too). They moved here from the module on 2026-10-08,
> unchanged, so that `M11.D.2`'s magic-literal scan covers the criteria (Hermes gate finding 4); the regenerated
> ensemble record's verdicts are identical. Parsed strictly: a row the criteria do not read, or a setting they
> read that is missing, raises at load.
>
> * `spell` — a `JOB_LOSS` of the given intensity to each parent every `every` weeks from week `from`. `M11.C.41`'s
>   *light* stress is the first parent only, every `light_every` weeks from the same week.
> * `run_after` — weeks run after the scripted act's week, so the run is `t0 + run_after` weeks long.
> * `level_*` — how far every member's basic and functional level is lowered in that arm.
>
> **Phase D part II, 2026-10-13** (`src/bowen/ensemble/criteria_d2.py`; the life course, spec revision 28, Claude's
> recommendation, approved by the owner 2026-10-16): `M11.C.15`'s mother carries a symptom held at `severity` (as `M11.C.12`'s)
> until `change_week`, when she dies or recovers, and the others are read for `after` weeks. Grief and lost income are
> off in every arm; `M6.3b`'s transfer stays at 1 (owner decision 2026-10-13); `M11.C.6`, `M11.C.10`
> and the twelve-person `M11.C.2` run the reference family for `years`. *Revision 29 (2026-10-14):*
> `M11.C.10`'s cutoff is renewed every `renew_every` weeks in the treatment and held open in the baseline; `M11.C.2`
> on the twelve-person family counts projection-type acts per year under `projection_until_age` and reads load over
> ages `load_from_age` to `load_to_age`. *Revision 30 (2026-10-14):* `M11.C.10` runs exactly `seeds` seeds (the owner:
> "200 instead"), reads generation 3's rejection toward their partner and own parents, and has a second variant,
> `M11.C.10[family of origin]`, holding each parent's cutoff of their own parents.
>
> **Phase D, 2026-10-11** (`src/bowen/ensemble/criteria_d.py`; the estimator's criteria, spec revision 17):
> * `nodal` — the declared nodal load: one event every `every` weeks to each named person, cycling through the three
>   kinds, so every estimator window holds all three (`estimator_breadth`).
> * `weeks` 208 — four years: the estimator first reads at the end of week 155 and again at week 207. `M11.C.17`
>   runs 520 weeks, ten years (owner decision 2026-10-11: coaching lands slowly).
> * `act_every` — a scripted act every that many weeks; `M11.C.20`'s support (a partner giving way) runs for its first
>   `support_weeks`, every `support_every` weeks, and is then withdrawn.
>
> **Revision 31, the owner's arm designs (2026-10-16):**
> * `M11.C.14` — every `act_every` weeks for `weeks`, both parents use the variant's technique toward C1 in the
>   treatment and hold back (`WITHHOLD`, no target) in the baseline, so both arms lose the same weeks of the parents'
>   own acts. C1 starts with an active symptom at `start_severity` of its threshold.
> * `M11.C.21` — in both arms both parents over-function toward C1 every `act_every` weeks for the whole run
>   (`t0 + after` weeks), so C1 has lent functioning before the collapse; the arms differ only by the treatment's
>   `drop` in both parents' functional level at week `t0`. A composite since 2026-10-17 (owner decision).
>
> **Version 1, the premises' arms (2026-10-17; spec revision 32, Claude's recommendation, approved by the owner 2026-10-17):**
> * `M11.C.4` — the exception to the held-open rule below: the baseline holds `a`–`b` open for the whole run, so it is
>   the matched no-cutoff arm (held only to `t0`, the policy cut the tie by itself in every baseline seed); the
>   treatment holds it to `t0` as before.
> * `M11.C.32` — both arms' mover has basic and functional level `mover_level`, so the calm arm can complete the
>   sequence (post hoc: at the fixture's 40 neither arm reached the peak after spec revision 16). From revision 41
>   (T-133) `assertion_form` also counts a sequence's `DEFINE` said in anger, which `M5.D.4a` makes the assertion
>   form, in either arm.
> * `M11.C.7` — the Phase C family under the `nodal` load for `weeks`; the fall in the spouses' combined deviation is
>   the mean over the first `window` weeks less the mean over the last `window`. The topologies and the scripted
>   `SPLIT` are in `src/bowen/ensemble/criteria_d.py`'s `arm_c7`.
>
> **Version 1, `M11.C.46` (2026-10-17; the arm design is Claude's recommendation, approved by the owner 2026-10-17):** on the
> triad, F turns to C (`TRIANGLE`) at week `t0`, the ties held open as in `M11.C.3`, and the act lands `latency`
> weeks later. The arms differ only in C's bonds from week 0: the baseline's C has bond energy `bond_far` with F and
> `bond_near` with M, the treatment's the reverse. Their sum is the fixture's (50 + 50).
>
> **Spec revision 39, group F (version 2 stage 3 step 8, 2026-10-26; the owner's D20 and D21 of 2026-10-24, the
> draft approved 2026-10-25):**
> * `M11.C.29` — the mover F has basic and functional level `mover_level` in both arms, as in `M11.C.32` (post hoc:
>   at 40, with the spell on the act's window, 58 of 100 seeds failed `PREPARE` and 37 ran as the assertion form), and
>   F's spell events from `t0` to `t0 + prepare_ticks` (inclusive) are removed in both arms; M's spell is kept. The
>   mover's completed exchanges are reported, and the row fails if there is none.
> * `M11.C.49` — four runs per seed on the triad, F at `level_low` and at `level_high`, each once with F's
>   `I-POSITION` toward M said at `t0` and once with nothing scripted. The said act starts its sequence at `DEFINE`,
>   unprepared (`M5.D.2a`; the owner's Q9), so it is said in every seed; from revision 41 (T-133), in a seed where F
>   is angry when it is said it is said as the assertion form, all demand at either level, and the sequence ends
>   (`M5.D.4a`). In all four runs F's systems perspective is
>   1.0, F's spell is suspended over the act's window as in `M11.C.29`, the F–M tie is held open to `t0`, and F's own
>   `I-POSITION` is withheld in every week, so the control has none and the act run one (as in the step 2 probe,
>   `docs/review/stage_3_iposition_probe.md`). Read over the `window` weeks after `t0` (weeks `t0 + 1` to
>   `t0 + window`, end of week): the family's excess anxiety summed over people and weeks, the `CONFLICT` share of
>   the moves between F and M (0 when there is none; such runs are counted), M's mean felt impingement on the tie;
>   over the run, F's completions. `t0` 1: week 0 is every member's chronic floor, and the probe found F angry at
>   week 4 in 14 of 40 seeds. `weeks` 25, so a sequence said at `t0` can finish its windows. All `[I]`.
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
>
> **`M11.C.27` read week by week, 2026-10-09 (owner decision, after the 104-week run).** The owner: "the effect
> is almost immediate and a long time horizon is distorted by other effects. Measure week by week." So the
> criterion gates on the pair's deviation at the end of the week the act's effect lands (`latency`, 1 week after
> t0, as `M11.C.3`), and reports every week from t0 + 1 to t0 + `report_weeks` (12, `[I]`) beside it, never gating.
> `run_after` is retired. `M11.C.4`'s nodal event stays at week 104.

| criterion | setting | value | grade |
|---|---|---|---|
| spell | kind | JOB_LOSS | [I] |
| spell | intensity | 120.0 | [I] |
| spell | every | 4 | [I] |
| spell | from | 4 | [I] |
| spell | light_every | 8 | [I] |
| spell | light_parents | 1 | [I] |
| scripted_act | intensity | 100.0 | [I] |
| M11.C.1 | weeks | 104 | [I] |
| M11.C.1 | level_baseline | 0.0 | [I] |
| M11.C.1 | level_treatment | 10.0 | [I] |
| M11.C.3 | t0 | 12 | [I] |
| M11.C.3 | run_after | 2 | [I] |
| M11.C.3 | latency | 1 | [I] |
| M11.C.4 | t0 | 8 | [I] |
| M11.C.4 | nodal | 104 | [I] |
| M11.C.4 | run_after_nodal | 2 | [I] |
| M11.C.5 | weeks | 60 | [I] |
| M11.C.5 | t0 | 4 | [I] |
| M11.C.16 | weeks | 104 | [I] |
| M11.C.16 | level_baseline | 0.0 | [I] |
| M11.C.16 | level_treatment | 15.0 | [I] |
| M11.C.16 | window | 52 | [I] |
| M11.C.19 | weeks | 1 | [I] |
| M11.C.19 | low_axis | 0.2 | [I] |
| M11.C.19 | high_axis | 0.8 | [I] |
| M11.C.25 | weeks | 52 | [I] |
| M11.C.25 | no_pole | 0.5 | [I] |
| M11.C.27 | t0 | 8 | [I] |
| M11.C.27 | latency | 1 | [I] |
| M11.C.27 | report_weeks | 12 | [I] |
| M11.C.27 | unstable_impingement | 0.8 | [I] |
| M11.C.29 | weeks | 80 | [I] |
| M11.C.29 | t0 | 4 | [I] |
| M11.C.29 | disguise_span | 8 | [I] |
| M11.C.29 | disguise_every | 2 | [I] |
| M11.C.29 | mover_level | 60.0 | [I] |
| M11.C.32 | weeks | 30 | [I] |
| M11.C.32 | t0 | 0 | [I] |
| M11.C.32 | angry_impingement | 1.0 | [I] |
| M11.C.32 | mover_level | 60.0 | [I] |
| M11.C.35 | t0 | 6 | [I] |
| M11.C.35 | run_after | 2 | [I] |
| M11.C.35 | conductance_high | 1.0 | [I] |
| M11.C.35 | conductance_low | 0.4 | [I] |
| M11.C.38 | weeks | 104 | [I] |
| M11.C.38 | level_1 | 0.0 | [I] |
| M11.C.38 | level_2 | 5.0 | [I] |
| M11.C.38 | level_3 | 10.0 | [I] |
| M11.C.38 | level_4 | 15.0 | [I] |
| M11.C.41 | weeks | 80 | [I] |
| M11.C.41 | level_higher | 0.0 | [I] |
| M11.C.41 | level_lower | 10.0 | [I] |
| M11.C.42 | t0 | 10 | [I] |
| M11.C.42 | weeks | 40 | [I] |
| M11.C.44 | weeks | 80 | [I] |
| M11.C.45 | weeks | 80 | [I] |
| M11.C.49 | weeks | 25 | [I] |
| M11.C.49 | t0 | 1 | [I] |
| M11.C.49 | window | 6 | [I] |
| M11.C.49 | level_low | 35.0 | [I] |
| M11.C.49 | level_high | 60.0 | [I] |
| nodal | kind_1 | JOB_CHANGE | [I] |
| nodal | kind_2 | MOVE | [I] |
| nodal | kind_3 | SERIOUS_ILLNESS | [I] |
| nodal | every | 8 | [I] |
| nodal | intensity | 50.0 | [I] |
| M11.C.17 | weeks | 520 | [I] |
| M11.C.18 | weeks | 208 | [I] |
| M11.C.18 | act_every | 6 | [I] |
| M11.C.20 | weeks | 208 | [I] |
| M11.C.20 | support_weeks | 104 | [I] |
| M11.C.20 | support_every | 3 | [I] |
| M11.C.20 | act_every | 6 | [I] |
| M11.C.33 | weeks | 208 | [I] |
| M11.C.33 | act_every | 6 | [I] |
| M11.C.30 | weeks | 52 | [I] |
| M11.C.30 | held_load | 0.9 | [I] |
| M11.C.22 | removal_week | 52 | [I] |
| M11.C.22 | after | 26 | [I] |
| M11.C.22 | mild | 1.5 | [I] |
| M11.C.22 | severe | 4.5 | [I] |
| M11.C.12 | remission_week | 52 | [I] |
| M11.C.12 | after | 26 | [I] |
| M11.C.12 | severity | 1.5 | [I] |
| M11.C.31 | t0 | 10 | [I] |
| M11.C.31 | weeks | 20 | [I] |
| M11.C.31 | held | 20.0 | [I] |
| M11.C.11 | removal_week | 52 | [I] |
| M11.C.11 | away | 52 | [I] |
| M11.C.11 | window | 8 | [I] |
| M11.C.26 | weeks | 52 | [I] |
| M11.C.26 | act_every | 2 | [I] |
| M11.C.36 | weeks | 4 | [I] |
| M11.C.36 | act_every | 1 | [I] |
| M11.C.36 | believed_tension | 0.8 | [I] |
| M11.C.14 | weeks | 52 | [I] |
| M11.C.14 | act_every | 2 | [I] |
| M11.C.14 | start_severity | 1.2 | [I] |
| M11.C.2 | weeks | 520 | [I] |
| M11.C.2 | years | 40 | [I] |
| M11.C.2 | projection_until_age | 20 | [I] |
| M11.C.2 | load_from_age | 20 | [I] |
| M11.C.2 | load_to_age | 30 | [I] |
| M11.C.15 | severity | 1.5 | [I] |
| M11.C.15 | change_week | 52 | [I] |
| M11.C.15 | after | 52 | [I] |
| M11.C.6 | child | nadia | [I] |
| M11.C.10 | years | 40 | [I] |
| M11.C.10 | renew_every | 13 | [I] |
| M11.C.10 | seeds | 200 | [I] |
| M11.C.47 | weeks | 260 | [I] |
| M11.C.47 | level_low | 15 | [I] |
| M11.C.47 | level_high | 50 | [I] |
| M11.C.47 | stress_every | 2 | [I] |
| M11.C.47 | stress_kind | JOB_LOSS | [I] |
| M11.C.47 | stress_intensity | 120.0 | [I] |
| M11.C.48 | weeks | 520 | [I] |
| M11.C.48 | years | 40 | [I] |
| M11.C.48 | seeds | 100 | [I] |
| M11.C.39 | spell_weeks | 26 | [I] |
| M11.C.39 | after | 52 | [I] |
| M11.C.39 | quiet | 8 | [I] |
| M11.C.39 | every | 2 | [I] |
| M11.C.39 | intensity | 100.0 | [I] |
| M11.C.40 | weeks | 208 | [I] |
| M11.C.40 | window | 26 | [I] |
| M11.C.43 | spell_weeks | 26 | [I] |
| M11.C.43 | gap | 13 | [I] |
| M11.C.43 | window | 8 | [I] |
| M11.C.43 | every | 2 | [I] |
| M11.C.43 | intensity | 100.0 | [I] |
| M11.C.34 | t0 | 26 | [I] |
| M11.C.34 | window | 8 | [I] |
| M11.C.34 | level_low | 10.0 | [I] |
| M11.C.34 | level_high | 10.0 | [I] |
| M11.C.34 | severity | 1.2 | [I] |
| M11.C.21 | t0 | 26 | [I] |
| M11.C.21 | after | 52 | [I] |
| M11.C.21 | drop | 10.0 | [I] |
| M11.C.21 | act_every | 2 | [I] |
| M11.C.23 | t0 | 4 | [I] |
| M11.C.23 | weeks | 52 | [I] |
| M11.C.23 | act_every | 2 | [I] |
| M11.C.23 | start_severity | 1.2 | [I] |
| M11.C.23 | suppression | 0.5 | [I] |
| M11.C.24 | diverge_weeks | 52 | [I] |
| M11.C.24 | after | 52 | [I] |
| M11.C.24 | act_every | 2 | [I] |
| M11.C.46 | t0 | 12 | [I] |
| M11.C.46 | latency | 1 | [I] |
| M11.C.46 | bond_near | 70 | [I] |
| M11.C.46 | bond_far | 30 | [I] |
| M11.C.7 | weeks | 520 | [I] |
| M11.C.7 | window | 52 | [I] |
