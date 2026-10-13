# EPModel v2 — which event kinds are nodal events

> Read by `src/bowen/io/load.py` into `EventKinds.nodal`; the basic-level estimator's breadth counts the distinct
> kinds in its window (spec `M1.A.4k`, revision 17). The owner, 2026-10-11: "Death, births, marriages, serious
> illnesses, big moves, big job changes, child moving out or back in, divorces are 'nodal' events. Any major event on
> a person (a node) is a nodal event." Which engine kind stands for which of these is `[I]`.
>
> A nodal event counts for the person it happens to and for everyone it reaches: its targets, its witnesses, the
> sender of a symptom (the bearer) and the members of the tie it names (`reach` = `reached`). A member leaving
> (`reach` = `ties`) also reaches everyone tied to them. A nodal event that reaches no one raises.
> `BINDER_UNAVAILABLE` is not listed: it names a binder (a tie's distance, a triangle position, a symptom channel), not
> a person, so it reaches no one by this rule (review, 2026-10-11). Births, deaths, marriages and a child moving out were added with the life course at spec revision 28 (Phase
> D part II); divorce and a child moving back in are not modelled.

| kind | reach | what it is |
|---|---|---|
| `JOB_LOSS` | reached | a big job change: losing a job |
| `JOB_CHANGE` | reached | a big job change |
| `MOVE` | reached | a big move |
| `SERIOUS_ILLNESS` | reached | a serious illness from outside the symptom model |
| `SYMPTOM_ONSET` | reached | a serious illness: a symptom crossing its threshold (`M7.D.1`) |
| `INSTITUTIONALIZE` | ties | a member moving out of the household (`M4.A.4`) |
| `DEATH` | reached | a death (`M6.3a`), reaching the living family members tied to the dead |
| `BIRTH` | reached | a birth (`M7.C.1i`), reaching the parents and siblings |
| `PARTNERING` | reached | a marriage (`M7.C.1h`), reaching the couple |
| `LAUNCH` | ties | a child moving out (`M7.C.1g`) |
| `FIND_WORK` | reached | a big job change: finding work after a lost income (`M7.C.1k`) |
