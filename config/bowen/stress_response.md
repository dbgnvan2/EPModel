# EPModel v2 — which acts are avoiding or soothing, under load

> Read by `src/bowen/io/load.py` into `EventKinds.soothing`; `src/bowen/engine/stress_response.py` lowers a person's
> functioning when they meet load with one of these acts (spec revision 18, `M1.A.5e`). The owner, 2026-10-11:
> stress "can lead to avoidance behaviour and 'soothing', e.g. drinking"; the list below was approved the same day.
> Coping — a self-directed act or `WITHHOLD` — raises functioning; a displacing act (the second table, spec revision
> 39, `M1.A.5f`) meets a loaded sender's load through its receiver; any other act leaves it. The second table is read
> into `EventKinds.displacing`. Loading fails if a kind in either table is not a declared move, is listed twice in its
> table, or is both soothing and displacing. Every entry is `[I]`; the displacing list was approved by the owner
> 2026-10-25 (`OVERFUNCTION` alone).

| kind | why it avoids or soothes |
|---|---|
| `DISTANCE` | moving away from the tie that carries the load |
| `CUTOFF` | ending contact, which relieves at once |
| `TRIANGLE` | turning to a third person for relief |
| `UNDERFUNCTION` | giving way, letting another carry it |

## Displacing

| kind | why it meets the sender's load through the receiver |
|---|---|
| `OVERFUNCTION` | taking charge of another's functioning: the caretaking position (`FE01.1`) |
