# EPModel v2 — the move kinds the supply readout counts as supplying

> `M11.G.5` (spec revision 55): who supplied attention, approval and help to whom is a readout computed after a run
> (`src/bowen/readouts/roles.py`). Which kinds count as supplying is editorial and `[I]`; the owner may add kinds.
> Kept outside `config/bowen/` because only a readout reads it: editing it does not move the engine hash, and nothing
> in the engine or the policy reads it (`M1.A.14b`).

| kind | why |
|---|---|
| `STAY-IN-CONTACT` | contact offered: attention given (`M11.G.5`, `[I]`) |
| `OVERFUNCTION` | doing for the other: help given (`M11.G.5`, `[I]`) |
