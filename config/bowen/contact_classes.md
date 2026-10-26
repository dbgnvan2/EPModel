# EPModel v2 — contact classes: which acts across a tie count as negative or positive contact

> Read by `src/bowen/io/load.py::load_contact_classes` and carried on `EventKinds` (spec `M1.D.2b`, `M1.D.2c`,
> `M11.C.48`; proposal `docs/PROPOSAL — CUTOFF ANXIETY, BRIDGING AND REPAIR.md` §3.4 and §4). These are
> classification rules, so they live here and not in the code (rule 9). **Every set is `[I]`**, Claude's reading of
> the owner's decisions of 2026-10-19 (G1's D4–D6, D12), decided as recommended under the owner's instruction of
> 2026-10-22, approved by the owner 2026-10-24 (T-125).
>
> - `negative`: a delivered act that renews a cut-off tie's stock to its peak and adds a third of a distant or
>   conflicted tie's reference level back. A refused move of any kind is also a negative contact; refusals are not
>   acts, so they are not listed. `DISTANCE` is not here: across a cut-off, distant or conflicted tie it binds into
>   the stock instead (`M1.D.2a`, `M1.D.2b`).
> - `positive`: a delivered act that bridges a cut-off tie (80% of its stock dissipates) and repairs a distant or
>   conflicted one (a third of its reference level). It is the only kind of contact that reopens a cut-off tie.
> - `rebuff`: the acts the negative-run readout (`src/bowen/readouts/negative_runs.py`, `M11.C.48`) counts as a
>   negative contact received besides the `negative` set: the owner's "rebuffs" (D12).

| set | acts | used by |
|---|---|---|
| negative | CONFLICT, CUTOFF | bridging and repair (`cutoff_stock.bridge_ties`); the negative-run readout |
| positive | REDUCE_CUTOFF, STAY-IN-CONTACT | bridging and repair; the negative-run readout (a positive ends a run) |
| rebuff | DISTANCE | the negative-run readout only |
