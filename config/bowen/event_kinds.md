# EPModel v2 — event kinds

> Every event kind the engine accepts, and the mechanism that carries it out
> (spec `M10.B.1`: event kinds are editorial content and live here, not in Python).
> The mechanisms are fixed in `src/bowen/engine/events.py`; this file only names
> kinds and maps each to one. Moves must be exactly the repertoire of `M5.A.1`
> and `M5.B` (`tests/bowen/test_events.py` checks this against the spec).
> Parsed strictly: an unknown mechanism, a duplicate kind or any unrecognised line raises.

| kind | mechanism | spec |
|---|---|---|
| `PURSUE` | move | `M5.A.1` |
| `DISTANCE` | move | `M5.A.1` |
| `CONFLICT` | move | `M5.A.1` |
| `OVERFUNCTION` | move | `M5.A.1` |
| `UNDERFUNCTION` | move | `M5.A.1` |
| `TRIANGLE` | move | `M5.A.1` |
| `CUTOFF` | move | `M5.A.1` |
| `I-POSITION` | move | `M5.A.1` |
| `STAY-IN-CONTACT` | move | `M5.A.1` |
| `DETRIANGLE` | move | `M5.B.1` |
| `PREVENT_ALIGNMENT` | move | `M5.B.2` |
| `REDUCE_CUTOFF` | move | `M5.B.3` |
| `SPLIT` | move | `M5.B.4` |
| `FRAME_AMBIGUITY` | move | `M5.B.4` |
| `DISPLACE` | move | `M5.B.4` |
| `TRIGGER` | trigger | `M4.A.2` |
| `RECONCILIATION` | reconciliation | `M4.A.3` |
| `INSTITUTIONALIZE` | institutionalize | `M4.A.4` |
| `BINDER_UNAVAILABLE` | binder_unavailable | `M1.F.9` |
| `JOB_LOSS` | exogenous_stressor | `M1.F.6` |
