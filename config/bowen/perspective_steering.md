# EPModel v2 — which acts systems perspective steers toward or away from

> Read by `src/bowen/io/load.py` into `EventKinds.steering`; the policy scales each listed act's within-channel weight
> by `1 ± perspective_steer_gain × systems_perspective` (spec revision 19, `M4.D.1a`). The owner, 2026-10-11: a true
> systems perspective would have an agent know "to 'detriangle', not to blame", "don't triangle". Which engine act
> stands for each is `[I]`: blaming is read as `CONFLICT`, the critical push.

| kind | direction | why |
|---|---|---|
| `TRIANGLE` | away | "don't triangle" |
| `CONFLICT` | away | "not to blame" |
| `DETRIANGLE` | toward | "know to detriangle" |
| `PREVENT_ALIGNMENT` | toward | staying out of a side someone offers |
