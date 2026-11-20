# Proposal — reactive amount, conflict as consumption, and divorce

**Status:** approved by the owner 2026-11-19 ("write it up - I approve. get it built"); built as revision 65 on the
branch `claude/reactivity-divorce` and merged as spec revision 66, on top of the final engine batch (revision 65).
The §4 predictions were run on that build (revision 64 plus this change) the same day, before the merge, with no constant tuned: P2 met, P1 met for
intensity and not for counts, P3 and P4 not met (§4.1). Drafted by Claude from the owner's corrections of 2026-11-19, made while reinterpreting
van Dijk et al. (2020), a meta-analysis of interparental conflict, parenting and child adjustment after divorce.

## 1. Why

An exploratory run of the model (Phase C family, 750 and then 1,200 simulated families, level and stress varied) found
spousal conflict and child focus uncorrelated across families (r ≈ .02), against .31 in the meta-analysis. A labelled
parameter search over five constants found no setting near .31. Three structural features were the cause, and the owner
corrected the model's account of each.

| Feature as built | The owner's correction, 2026-11-19 |
|---|---|
| Anxiety changes **which** act a person chooses (`M4.D.3`: the mixing weight and learned values), never **how much** they act: one act per person per week. | "anxiety DOES lead to REACTIVE behaviour, where the behavior is meant to reduce Anxiety … Anxiety >>> behaviour >>> relief >>> learning." "By definition, lower levels of differentiation means higher levels of reactivity, which means higher levels of any pattern(s) - conflict, distance, projection, substance use, affairs etc." |
| One act a week makes the patterns compete: a week of conflict is a week without child focus. | "Bowen Theory is not either or about Parental Conflict and child focus. They can go together." |
| `CONFLICT` sits in capacity layer 2 (`M4.D.3a`, the owner's D22), available in full only at functional level 40, so the lowest-level families stop fighting. | "Conflict doesn't 'need' capacity, it consumes it." |
| Divorce is not modelled (`config/bowen/nodal_events.md`). | "divorce should be twice the intensity of a job loss" — checked: Holmes and Rahe (1967) give divorce 73 life change units and dismissal from work 47, a ratio of 1.55; with marital separation (65) the pair is 138, about 2.9. |

The corpus supports the direction of each correction. Bowen (1978, ch. 9, proposition 12, `L09`): the lower a person
on the scale, "the more extreme the mechanisms used to control 'too much closeness' (emotional distance, isolation,
conflict, violence, physical illness)". Bowen (1978, ch. 12): "Most use a combination of all three patterns." Kerr and
Bowen (1988, ch. 7, `FE07.2`): "rarely an all-or-nothing process … The patterns tend to be intermixed." Kerr (2019,
ch. 4, `KS04`): "more than one pattern can be active in a family at the same time." The corpus gives no rate or size
for any of this; every constant below is `[I]`.

## 2. The changes

### C1 — Anxiety raises the amount of reactive behaviour (`M4.D.3`, amended; `M4.D.3c`, new)

`M4.D.3`'s two routes stay. Its prohibition narrows: the policy still carries no term that raises the **weight** of a
reactive act with anxiety, so which act is chosen remains learned from relief (`M4.D.6a`). What anxiety now sets is the
**number** of reactive acts in a week.

`M4.D.3c`: after a person's one selection, a person whose excess anxiety `a` (acute above chronic) is positive makes
extra automatic acts. Their expected number is `reactive_extra_max × a ÷ (a + reactive_half_anxiety)`, the same
saturating form as `M4.D.1`'s force; the whole part is made and the fraction is made with that probability, by a keyed
draw. Each extra act is drawn from the automatic channel only, by the same learned values, toward a person the actor
has not already acted toward this week, so each tie carries at most one act a week. A person owed an `I-POSITION`
sequence step, an external agent and a scripted selection get no extra acts.

Constants (`[I]`): `reactive_extra_max` 3, `reactive_half_anxiety` 100. At the Phase C family's own levels the
parents' median excess anxiety is about 30 (about 0.7 extra acts a week); with every level lowered by 20 it is
80 to 135 (about 1.3 to 1.7).

### C2 — Conflict consumes capacity (`M4.D.3a`, amended; `M4.D.3d`, new)

`CONFLICT` moves from layer 2 to layer 0, beside `DISTANCE`, `CUTOFF` and `TRIANGLE`. Kerr's ordering (cooperation,
conflict, dominant-adaptive, distance; `KS06.1`) is kept as a statement of evolutionary age and of which pattern
dominates in deep regression; it no longer gates availability. This reading is the owner's decision (2026-11-19), and
it replaces D22's placement of `CONFLICT`.

`M4.D.3d`: each `CONFLICT` a person sends lowers their functional level by `conflict_capacity_cost` (`[I]`, 1.0
functional-level point), on the stress ledger (`stress_swing`), so it settles back when calm as any stress-moved
functioning does and is bounded by the stress floor (`stress_floor_below_basic`).

### C3 — Divorce (`M7.C.1m`, new; `config/bowen/event_sizes.md`, `config/bowen/event_kinds.md`, `config/bowen/nodal_events.md`)

A divorce is two events at one tick, scripted by a scenario (not drawn by the life course in this revision):

- `SEPARATION`, structural (mechanism `separation`): the leaving spouse moves to a household of their own. The marital
  tie is kept with its bond energy and stays interactive: coupling does not decay with distance (convergence C7), and a
  separation is an edge rewired with the old edge kept, not removed. The family's `marital_tie` still names it, so the
  sinks (`M1.D.1`) read conflict between former spouses as marital conflict. Nodal reach `ties`.
- `DIVORCE`, an exogenous stressor sized from Holmes and Rahe: 73 life change units to each spouse (`who` = self) and to
  each child through their tie to each parent (`who` = tie, bond energy ÷ 100), for 12 weeks. Nodal reach `reached`.

Which spouse leaves is the scenario's choice. A scenario may size the divorce up for a family under more strain (the
owner, 2026-11-19: "Holmes Rahe is an average. Conflictual Parents would have a higher than average 'score'"), but
by level or chronic anxiety only, never by observed conflict, which would make conflict cause child focus by
construction.

## 3. The owner's calibration estimate

"Conflictual 'divorced' couples would have conflict 25–75% of the time. non 6 over 104 (3%)." This is the owner's
clinical estimate, `[I]`, not a corpus figure. It is recorded as a check on output, never as a parameter.

## 4. Predictions registered before the run

Run after the build, on the probe of 2026-11-19 (Phase C family, 104 weeks, basic level lowered by 0, 10 or 20,
stressor at 0, 1 or 1.55 times the criteria spell, 40 seeds a cell), with **no constant tuned** afterwards:

- P1. Mean spousal conflict (count and intensity) rises as level falls.
- P2. Mean child focus rises as level falls.
- P3. Conflict and child focus correlate positively across families.
- P4. With a divorce at week 20, conflict between the former spouses and child focus are both higher than without.

A failed prediction is reported as failed. If constants are adjusted after the run, the result is labelled calibration.

### 4.1 Results, 2026-11-19 (T-260)

720 runs: 3 levels × 3 stressor sizes × with and without a divorce × 40 seeds. Means are per two-year run; intervals
are 95% bootstrap intervals over runs. Child focus is the parents' `OVERFUNCTION`, `PURSUE` and `TRIANGLE` toward a
child, the sinks' projection acts (`M1.D.1`).

| Prediction | Result | Verdict |
|---|---|---|
| P1. Spousal conflict rises as level falls | Intensity: level 20 minus level 0, +1,285 [+589, +1,983]. Count: +1.2 acts [−1.3, +3.7] | Met for intensity; not met for counts |
| P2. Child focus rises as level falls | Intensity +8,337 [+6,835, +9,841] | Met |
| P3. Conflict and child focus correlate positively across families | Intensity r = .09 [−.02, .20]; counts r = −.01 [−.11, .11]. Divorced families: .10 [−.01, .22] | Not met |
| P4. A divorce raises both | Conflict intensity −340 [−729, +47]; child focus −336 [−1,505, +838] | Not met; dropped (the owner, 2026-11-20, T-263) |

Spousal conflict now falls in 15% of weeks on average (3% at revision 64), and in up to 72% in the most conflictual
run; the owner's estimate for conflictual divorced couples is 25–75%. Conflict now correlates with the children's
symptom load across families, r = .29 [.18, .38]; child focus with it, .52 [.44, .60].

P4 is dropped (the owner, 2026-11-20). The marital tie is cut off before the divorce in almost every run: one
`CUTOFF` cuts it, and it reopens only after its rejection fades and a `self` contact is accepted (revision 36's
rule, kept), so the divorce lands on a tie with no conflict left to raise. The spouses' cutoff stands as the
model's result (T-263).

## 5. What moves

The engine hash moves. Every ensemble, mutation and sweep record keyed by it is stale until regenerated on the Macs;
the regeneration is not part of this build. Criteria that read act counts or conflict (`M11.C.1`, `M11.C.16`,
`M11.C.19`, the triangle and cutoff criteria) may change verdict and are reviewed at regeneration.
