# Run visualization — how a run becomes a fam-diag diagram

> Read by `tools/famdiag_export.py`. Approved by the owner on 2026-10-10:
> `docs/PROPOSAL — RUN VISUALIZATION MAPPING.md` §0, plan V5.
>
> **Editorial content.** It lives here, not in the tool's code. It sits outside `config/bowen/`, so it is
> outside the engine hash: changing it re-exports a run without making any record stale. **Every value is
> `[I]`.** The patterns themselves are not defined here; they are the declared readouts
> (`docs/readouts_phase_c.md`). This file only says how each one is drawn.
>
> - `block_weeks`: weeks per fam-diag "year" on the timeline slider (owner D6).
> - `first_year`: the year of block 1. It is 2001, not the 1001 the proposal named, because fam-diag's slider
>   always runs to the current year (`timelineYearBounds` in fam-diag's `utils/dateFormatting.ts`). With
>   1001 it would have a thousand steps.
> - Levels: a line's strength is the highest level its count reaches over the episode. Each column is
>   the lowest count for that level.
>   - `acts` counts acts in one block.
>   - `silent_weeks` counts weeks without interaction (owner D2: no interaction is an emerging level of distance).
> - Triangle intensity is set by how many blocks the fixed triangle lasts.
> - Event categories are fam-diag's own (`constants/eventConstants.ts`). `subtype` is always the model's act name.

block_weeks: 4
first_year: 2001

## Patterns

| pattern | relationship_type | styles | ending |
|---|---|---|---|
| conflict | conflict | conflict-dotted-wide, conflict-dotted-tight, conflict-solid-wide, conflict-solid-tight, conflict-double | none |
| distance | distance | distance-dashed-tight, distance-dashed-wide, distance-long, distance-dotted-tight, distance-dotted-wide | none |
| over_underfunctioning | fusion | fusion-dotted-wide, fusion-dotted-tight, fusion-solid-wide, fusion-solid-tight, fusion-triple | over_side |
| fusion | fusion | fusion-dotted-wide, fusion-dotted-tight, fusion-solid-wide, fusion-solid-tight, fusion-triple | none |
| projection | projection | projection-1, projection-2, projection-3, projection-4, projection-5 | arrow_from_toward |
| cutoff | cutoff | cutoff | arrow_from_toward |

## Levels

| measure | level_1 | level_2 | level_3 | level_4 | level_5 |
|---|---|---|---|---|---|
| acts | 3 | 4 | 5 | 7 | 10 |
| silent_weeks | 4 | 8 | 16 | 32 | 64 |

## Triangle intensity

| intensity | min_blocks |
|---|---|
| low | 1 |
| medium | 2 |
| high | 4 |

## Events

| kind | event_type | category | event_class |
|---|---|---|---|
| CONFLICT | EPE | Conflict | emotional-pattern |
| DISTANCE | EPE | Distance | emotional-pattern |
| CUTOFF | EPE | Cutoff | emotional-pattern |
| REDUCE_CUTOFF | EPE | Cutoff | emotional-pattern |
| OVERFUNCTION | EPE | +/- Adequate | emotional-pattern |
| UNDERFUNCTION | EPE | +/- Adequate | emotional-pattern |
| PURSUE | EPE | Emotional Pattern | emotional-pattern |
| STAY-IN-CONTACT | EPE | Emotional Pattern | emotional-pattern |
| SPLIT | EPE | Emotional Pattern | emotional-pattern |
| DISPLACE | EPE | Emotional Pattern | emotional-pattern |
| PROVOKE | EPE | Emotional Pattern | emotional-pattern |
| PREVENT_ALIGNMENT | EPE | Emotional Pattern | emotional-pattern |
| FRAME_AMBIGUITY | EPE | Emotional Pattern | emotional-pattern |
| TRIANGLE | TRIANGLE | Primary | triangle |
| DETRIANGLE | TRIANGLE | Primary | triangle |
| I-POSITION | EA | Emotional Autonomy | individual |
| SYMPTOM_ONSET | SYMPTOM | by_channel | individual |
| JOB_LOSS | NODAL | Individual | individual |

## Symptom channels

| channel | category |
|---|---|
| physical | Physical |
| mental | Emotional |
| social | Social |

## Layout

| id | x | y |
|---|---|---|
| teodor | 100 | 100 |
| ana | 250 | 100 |
| bruno | 400 | 100 |
| sofia | 700 | 100 |
| viktor | 850 | 100 |
| iris | 150 | 300 |
| marta | 300 | 300 |
| ravi | 650 | 300 |
| dario | 800 | 300 |
| halim | 1050 | 300 |
| leo | 375 | 500 |
| nadia | 475 | 500 |
| pia | 575 | 500 |
