# EPModel v2 — event kinds

> Every event kind the engine accepts, and the mechanism that carries it out
> (spec `M10.B.1`: event kinds are editorial content and live here, not in Python).
> The mechanisms are fixed in `src/bowen/engine/events.py`; this file only names
> kinds and maps each to one. Moves must be exactly the repertoire of `M5.A.1`
> and `M5.B` (`tests/bowen/test_events.py` checks this against the spec).
> Parsed strictly: an unknown mechanism, a duplicate kind or any unrecognised line raises.
>
> `inside_sign` and `outside_sign` multiply an event's appraised effect when its sender
> acts from the inside or the outside position of a triangle (`M1.F.2`: source position
> can change the sign). The corpus case (Ch04 · L04.4) is one class of content — naming
> the projection — which lands oppositely by speaker. Which kinds carry it is a Phase C
> decision, made with `I-POSITION`; until then every sign is +1.
>
> `contact` and `impingement` are each kind's components per unit of scaled intensity
> (spec `M4.C.1`, revision 11; Phase C plan D2): how much a delivered event of the kind moves the
> receiver's felt contact and felt impingement on the tie it arrives on. **Every value is invented,
> `[I]`.** The one source constraint is `CONFLICT`, which carries both — "people maintain contact and
> enforce distance at the same time" (`KS04.13`). `CUTOFF` is 0 and 0 because its effect is structural:
> on delivery it makes the tie non-interactive and removes impingement at once, and contact then decays
> (`M4.C.1c`). Structural and exogenous kinds move neither; an exogenous stressor has no tie.
>
> `accommodates` marks the kinds that give way to the other — the inward axis of `M1.A.9a`, the
> "compliant accommodator" of `FE03.1` (Phase C step 3). Only `UNDERFUNCTION` is marked; which acts count as
> giving way is editorial and `[I]`.

| kind | mechanism | inside_sign | outside_sign | contact | impingement | accommodates | spec |
|---|---|---|---|---|---|---|---|
| `PURSUE` | move | +1 | +1 | 0.5 | 0.3 | no | `M5.A.1` |
| `DISTANCE` | move | +1 | +1 | -0.5 | 0 | no | `M5.A.1` |
| `CONFLICT` | move | +1 | +1 | 0.3 | 0.6 | no | `M5.A.1` |
| `OVERFUNCTION` | move | +1 | +1 | 0.2 | 0.5 | no | `M5.A.1` |
| `UNDERFUNCTION` | move | +1 | +1 | 0.2 | 0 | yes | `M5.A.1` |
| `TRIANGLE` | move | +1 | +1 | 0.3 | 0 | no | `M5.A.1` |
| `CUTOFF` | move | +1 | +1 | 0 | 0 | no | `M5.A.1` |
| `I-POSITION` | move | +1 | +1 | 0.2 | 0.1 | no | `M5.A.1` |
| `STAY-IN-CONTACT` | move | +1 | +1 | 0.5 | 0 | no | `M5.A.1` |
| `DETRIANGLE` | move | +1 | +1 | 0.2 | 0 | no | `M5.B.1` |
| `PREVENT_ALIGNMENT` | move | +1 | +1 | 0.2 | 0 | no | `M5.B.2` |
| `REDUCE_CUTOFF` | move | +1 | +1 | 0.5 | 0.1 | no | `M5.B.3` |
| `PROVOKE` | move | +1 | +1 | 0.2 | 0.4 | no | `M5.B.6` |
| `SPLIT` | move | +1 | +1 | 0.1 | 0.2 | no | `M5.B.4` |
| `FRAME_AMBIGUITY` | move | +1 | +1 | 0.1 | 0.2 | no | `M5.B.4` |
| `DISPLACE` | move | +1 | +1 | 0.1 | 0.2 | no | `M5.B.4` |
| `TRIGGER` | trigger | +1 | +1 | 0 | 0 | no | `M4.A.2` |
| `RECONCILIATION` | reconciliation | +1 | +1 | 0 | 0 | no | `M4.A.3` |
| `INSTITUTIONALIZE` | institutionalize | +1 | +1 | 0 | 0 | no | `M4.A.4` |
| `BINDER_UNAVAILABLE` | binder_unavailable | +1 | +1 | 0 | 0 | no | `M1.F.9` |
| `JOB_LOSS` | exogenous_stressor | +1 | +1 | 0 | 0 | no | `M1.F.6` |
| `SYMPTOM_ONSET` | endogenous_symptom | +1 | +1 | 0 | 0 | no | `M7.D.1` |
