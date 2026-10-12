# EPModel v2 — which acts each pattern readout counts

> Read by `src/bowen/readouts/patterns.py` (spec `M5.A.1a`; definitions in `docs/readouts_phase_c.md`). These are
> classification rules, so they live here and not in the code (Phase D plan B2; rule 9). **Every set is `[I]`**, the
> owner's reading of 2026-10-10 where it says so.
>
> - `projection` must equal the child-projection sink's act set (`src/bowen/engine/sinks.py`); a test ties them.
> - `under` is giving way or going along: the owner, 2026-10-10, "pursuit is … a form of over/underfunctioning —
>   going along to stay in agreement with the other".

| set | acts | used by |
|---|---|---|
| approach | PURSUE, STAY-IN-CONTACT, REDUCE_CUTOFF | distance (approaches outnumbered); cutoff (the other's approach, refused) |
| conflict | CONFLICT | conflict |
| withdrawal | DISTANCE | distance |
| over | OVERFUNCTION | over/underfunctioning: the over side taking charge |
| under | UNDERFUNCTION, PURSUE | over/underfunctioning: the under side giving way or going along |
| fusion | PURSUE | fusion: both going along |
| projection | OVERFUNCTION, PURSUE, TRIANGLE | projection: a parent's acts focused on a child |
| refusal | CUTOFF, DISTANCE | cutoff: an act answering the other's approach within a week (revision 17: one of the ways an approach is refused) |
