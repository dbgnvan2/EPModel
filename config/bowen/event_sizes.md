# EPModel v2 — event sizes from the Holmes–Rahe scale

> Read by `src/bowen/io/load.py::load_event_sizes` and carried on `EventKinds.sizes` (spec `M1.F.6a`, `M7.C.1l`,
> `M7.D.1a`; `docs/PROPOSAL — EVENT SIZES FROM THE HOLMES–RAHE SCALE.md`, decided by the owner 2026-10-19, built at
> spec revision 36). Which Holmes–Rahe item stands for which event kind is editorial, so it lives here (rule 9).
>
> **The scale is used as an ordering of base sizes only** (the owner, Q1). Every value is graded `[I]`: the life
> change units (`lcu`) are read from the scale's published table as it appears on Wikipedia, unchecked against
> Holmes and Rahe (1967); the corpus gives no magnitude for any event (`docs/model_explainer.md` §0). An event's
> total load is `lcu_scale` (6.9, the owner's anchor, Q2) × `lcu` × reach, where reach is 1 for the person the event
> happens to (`who` = self) and bond energy ÷ 100 for a person it reaches through a tie (`who` = spouse, relative,
> tie). It arrives at that total ÷ (1 + `stressor_persistence` × (`weeks` − 1)) and weighs for `weeks` weeks as any
> lasting stressor does (`M1.F.6`), so a long event and a short one of the same `lcu` deliver the same total. `weeks`
> is the model's, not the scale's: 12 weeks for the additions and transitions (the owner, Q4); 12 weeks for an
> illness, Claude's `[I]`; `—` where the life course sets it (grief, `M6.3g`: one to three years by bond energy).
>
> - `who` = spouse: the mourner was the dead person's spouse (the family's marital tie); relative: any other mourner.
> - Lost income (`INCOME_LOSS`, `M6.3h`) is outside the scale (the owner, Q8): its stressor ends with an act, not a
>   duration, so it keeps `income_loss_intensity`. Its row records that.
> - The criteria spell's `JOB_LOSS` (`config/bowen/criteria.md`, 120 for one week) is a declared test input and keeps
>   its declared intensity; the sizes here apply to the life course's events and to a symptom onset. A scripted
>   stressor of a sized kind keeps the intensity its script declares.
>
> Decided as recommended under the owner's instruction of 2026-10-22, pending the owner's review: the weeks of
> `OWN_ILLNESS` and `FAMILY_ILLNESS` (12), and the mapping of `LAUNCH` to both the child and the parents at home.

| kind | who | item | lcu | weeks | source |
|---|---|---|---|---|---|
| `GRIEF` | spouse | Death of spouse | 100 | — | Holmes–Rahe, `[I]`; `M6.3g` sets the weeks |
| `GRIEF` | relative | Death of a close family member | 63 | — | Holmes–Rahe, `[I]`; `M6.3g` sets the weeks |
| `INCOME_LOSS` | self | Change in financial state | — | — | outside the scale (the owner, Q8): `income_loss_intensity` |
| `JOB_LOSS` | self | Dismissal from work | 47 | 1 | Holmes–Rahe, `[I]` |
| `JOB_CHANGE` | self | Change to different line of work | 36 | 1 | Holmes–Rahe, `[I]` |
| `MOVE` | self | Change in residence | 20 | 1 | Holmes–Rahe, `[I]` |
| `SERIOUS_ILLNESS` | self | Personal injury or illness | 53 | 12 | Holmes–Rahe, `[I]` |
| `OWN_ILLNESS` | self | Personal injury or illness | 53 | 12 | Holmes–Rahe, `[I]`; `M7.D.1a` (the owner, Q5) |
| `FAMILY_ILLNESS` | tie | Change in health of family member | 44 | 12 | Holmes–Rahe, `[I]`; `M7.D.1a` |
| `PARTNERING_LOAD` | self | Marriage | 50 | 12 | Holmes–Rahe, `[I]`; `M7.C.1l` (the owner, Q3, Q4) |
| `RETIREMENT_LOAD` | tie | Retirement | 45 | 12 | Holmes–Rahe, `[I]`; `M7.C.1l`: the coachee, through the coach tie |
| `BIRTH_LOAD` | self | Gain of a new family member | 39 | 12 | Holmes–Rahe, `[I]`; `M7.C.1l` |
| `LAUNCH_LOAD` | self | Son or daughter leaving home | 29 | 12 | Holmes–Rahe, `[I]`; `M7.C.1l`: the child and the parents at home |
| `FIND_WORK_LOAD` | self | Spouse begins or stops work | 26 | 12 | Holmes–Rahe, `[I]`; `M7.C.1l` (the owner, Q6: 26) |
