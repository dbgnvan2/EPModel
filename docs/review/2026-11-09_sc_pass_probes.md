# The two social-cue passes probed: `M11.C.52` (T-208) and `M11.C.55` (T-206)

2026-11-09, on spec revision 55's engine (engine hash `927550dfcb8a`; the ensemble records' code hash `a5fc2228eb46`).
A measurement, never a verdict. No rule, constant, criterion or default changed; `social_cues_enabled` stays off by
default. Every condition below is an override passed to the criterion's own `scenario` call for that run only, or the
named mutant applied to a temporary copy of the repository (`tools/sc_pass_probe.py`, in no record's hash; smoke test
`tests/test_sc_pass_probe.py`). The per-seed rows are in [`sc_pass_probes/`](sc_pass_probes/); every number here is
`python3 tools/sc_pass_probe.py summarise <file>` or `paired <a> <b>` on them. Each `±` is a 95% half-width of the
per-seed difference; each p is the runner's one-sided signed-rank test after its margin, as `tools/ensemble_record.py`
reads it. Runs: `M11.C.52` and `M11.C.55` with learning off on Mac Mini1, `M11.C.55` as recorded on davemini2; the
same code on both (`a19a690` plus the probe tool).

These are composites (`M11.5`): what follows says what carries each pass, and the pass or fail stays a finding about
the model, never about any family.

## T-208: `M11.C.52`, self-attribution by channel damps an escalation

The criterion: the dyad, both members held 30 above their chronic anxiety at the start, 26 weeks, the cues on in both
arms; the treatment gives member `a` `systems_perspective` 1.0. Readout `escalation`, the summed acute anxiety of both
over the weeks, lower in the treatment. Its named mutant `cue-attribution-discount-removed` removes `M4.C.6`'s discount
on the nonverbal part of a received move (`appraise.py`, `_appraisal_by_channel`).

| Condition | Seeds | `escalation` (treatment − baseline) | p |
|---|---|---|---|
| as recorded | 0–99 | −23.8 ± 24.4 | 0.046 |
| named mutant | 0–99 | −20.7 ± 25.4 | 0.106 |
| as recorded | 0–499 | −17.0 ± 10.1 | 0.0018 |
| named mutant | 0–499 | −12.5 ± 10.2 | 0.031 |
| nonverbal channels equal (`M4.C.1f`'s reduction) | 0–499 | −9.3 ± 8.6 | 0.053 |
| nonverbal channels equal, named mutant | 0–499 | −9.3 ± 8.6 (identical, seed for seed) | 0.053 |
| cues off in both arms | 0–499 | −22.2 ± 8.3 | 5.9e-7 |

Paired on seeds 0–499, the change in the effect against the recorded condition:

| Against as recorded | Change in the effect |
|---|---|
| named mutant | +4.5 ± 1.5 |
| nonverbal channels equal | +7.8 ± 8.5 |
| cues off | −5.2 ± 8.5 |

The first 100 seeds reproduce the Phase C ensemble record and the Phase C mutation record exactly. "Channels equal"
is the reduction `M4.C.1f` states and `tests/bowen/test_social_cues.py`'s `NEUTRAL` sets: nothing on tone, face and
posture, no meaning error, full detection. The nonverbal part of every move is then zero, so the nonverbal discount has
nothing to act on. The mutant's identical result there confirms that the mutant touches only that discount.

**Reading.**

1. **At 500 seeds the pass holds clear of the runner's test.** −17.0 ± 10.1, p 0.0018; the interval excludes 0. At
   the record's 100 seeds it met the test at the margin (p 0.046); the five blocks of 100 seeds give −23.8, −14.4,
   −20.7, −14.9 and −11.5. The record stops at 100 because the runner's precision rule is met there (`M17.A.1`), not
   because the effect is that size.
2. **The named mutant removes a detected part of the effect, and a detected residual remains.** Paired on the same
   seeds it changes the effect by +4.5 ± 1.5; with it applied the effect is still −12.5 ± 10.2 (p 0.031). Against the
   uncertain −17.0 ± 10.1 the part removed is between about a sixth and about two thirds; the probe does not fix the
   share. With the channels equal, where the discount is zero, the difference is −9.3 ± 8.6, p 0.053: the interval
   barely excludes 0, and the runner's test does not detect it. With the cues off, under `M4.C.6`'s whole-event rule,
   it is −22.2 ± 8.3 (p 5.9e-7); paired against the recorded condition the change is −5.2 ± 8.5, so no difference is
   detected between the cues off and on. The code's other reads of `systems_perspective`, the candidate carriers of
   the residual, include: the whole-event reattribution of returned contact (`M4.C.1d`, `appraise.py`, which still
   reads `own_share` with the cues on); the whole-event reattribution of any delivery that `tie_change_parts` does not
   split by channel (`appraise.py`: a delivery that is not a move, is not to a target, or has no tie); the policy's
   loaded-tie gate and steering (`M4.D.3b`, `policy.py`); the self-channel weight (`iposition.py`, `policy.py`); and
   the learner's look-back (`learner.py`). The probe does not measure which of them carry the residual.
3. **The mutant turned the row red at 100 seeds only for want of seeds.** At 500 seeds the mutant still passes
   (p 0.031), so by the row's own rule it survives. `M11.5`'s row says: "if its named mutant survives the probe, the
   row comes back to the owner before it counts". It now comes back.

So the row's pass is a pass of "a member with `systems_perspective` damps the tie's escalation". The by-channel rule
`M4.C.6` added at revision 55 carries a detected part of it (+4.5 ± 1.5), and a detected part (−12.5 ± 10.2) is carried
by something else; with the cues off no difference from the recorded effect is detected. It is not a finding about
self-attribution by channel alone.

**Wording, applied at spec revision 56 (the owner, 2026-11-09, relayed by another Claude session).** `M11.5`'s row for `M11.C.52` now reads: "needs `M4.C.6` by
channel, `M4.C.13` on both sides and the exchange over weeks; T-208's probe (`docs/review/2026-11-09_sc_pass_probes.md`):
on 500 seeds the pass holds (−17.0 ± 10.1) and its named mutant survives (−12.5 ± 10.2), removing +4.5 ± 1.5 of the
effect; with the nonverbal channels equal it is −9.3 ± 8.6 (p 0.053), and with the cues off −22.2 ± 8.3, no different
from the recorded effect. A detected part of the pass is carried by paths other than the by-channel discount, so it is
not a finding about self-attribution by channel alone." The
criterion's row in `M11` has the matching sentence about its mutant. A mutant that removes all the paths except
the by-channel discount, or a baseline arm that has the other paths and lacks the discount, would test what the
criterion's title names: T-209, opened by the owner 2026-11-09.

## T-206: `M11.C.55`, who supplies a parent is learned, not given by birth order

The criterion: the twelve-person family, 30 years, the cues on; for each of the two parents, the child who was the main
supplier over years 20 to 30 (`M11.G.5`), predicted by the child with the highest learned value on the supplying kinds
toward that parent at year 20 (the treatment) and by the first-born (the baseline). Readout: the parents per seed whose
main supplier the predictor names. The probe adds a third predictor that is not the policy's input: the child who
supplied that parent most over years 0 to 20. Chance is the parents supplied ÷ 3 children.

| Condition | Seeds | Relief − birth order | Hits per seed: relief | birth order | prior supply | chance | Parents supplied | Parents the relief names |
|---|---|---|---|---|---|---|---|---|
| as recorded | 0–149 | +0.760 ± 0.146, p 1.9e-14 | 1.067 | 0.307 | 0.633 | 0.633 | 1.900 | 1.993 |
| learning off (`M10.B.5`) | 0–149 | −0.080 ± 0.047, p 1 | 0.000 | 0.080 | 0.220 | 0.622 | 1.867 | 0.000 |

Learning off is `criteria_d.LEARNING_OFF` (`learning_rate` 0 and `witnessed_cutoff_gain` 0), added to the criterion's
own override for the run. The recorded condition reproduces the Phase D part II ensemble record (+0.76 ± 0.15,
p 1.89e-14, 150 seeds).

**Reading.**

1. **With learning off the two predictors do not become equal. The relief predictor stops existing.** Every learned
   value stays 0, so the relief names no child for any parent and hits 0. Birth order hits 0.080. The criterion fails
   (−0.080 ± 0.047). The fail follows from the predictor's definition (a tie for the highest names no one,
   `criteria_d2.supplier_run`), not from what the run does: with no learning there is nothing for the predictor to
   read. So this probe cannot say whether the pass is more than the policy following its own values.
2. **The relief predictor is the policy's input, and that input predicts the window.** At year 20 the learned values
   predict the next ten years' main supplier (1.067 hits against chance 0.633). The previous twenty years of supplying
   predict it only at chance (0.633). Who supplies most in years 0–20 does not predict who supplies most in years
   20–30; the current learned value does. That is consistent with the policy selecting by learned value (`M4.D.6`),
   and with a value that keeps rising through the window as supplying keeps relieving; the probe does not separate
   the two (a frozen-learning run would; see the end). The run does not show a supplier role that settles in years
   0–20 and persists.
3. **The baseline is below chance with learning on and with it off.** The first-born is the main supplier for 0.307
   parents per seed against chance 0.633 (113 of 150 seeds name no parent's supplier by birth order), and for 0.080
   with learning off. Something in the family as declared, not learning, makes the first-born unlikely to be the main
   supplier in years 20–30. The probe does not identify it; age at the window and leaving the parents' household are
   candidates. A predictor that beats birth order therefore beats a predictor worse than chance, so "more often than
   birth order" is a weak bar.

So the row's pass shows that, with learning on, a child's learned value on the supplying kinds toward a parent at year
20 predicts its supplying over the next ten years, and does so better than a predictor that is below chance. It does
not show that who supplies is *learned rather than given*: the predictor is the policy's selection input, and the
comparison arm is structurally disadvantaged.

**Wording, applied at spec revision 56 (the owner, 2026-11-09, relayed by another Claude session).** `M11.5`'s row for `M11.C.55` now reads: "needs learning from
relief (`M4.D.6`); a within-run predictor. T-206's probe (`docs/review/2026-11-09_sc_pass_probes.md`): with learning off
the relief predictor names no one and the row fails; with it on, the predictor is the policy's own selection input,
and the past twenty years' supply predicts the window only at chance, while birth order predicts it below chance
(0.307 per seed against 0.633). The pass is consistent with selection following learned value. It is not a finding that the
supplier is learned rather than given." Two changes would make the criterion informative; the owner decided both 2026-11-09 and held them for the next batch that moves the engine hash (T-206):
compare the relief predictor with chance and with prior supply rather than with birth order; and read the relief at a
week before the window that is not the window's own start, so the value is not the one the policy selects by over the
window. Either one changes a criterion and moves the records.

## What this record does not cover

- It does not split `M11.C.52`'s residual among `systems_perspective`'s other paths (T-209).
- It does not test a frozen-learning condition for `M11.C.55` (learning on to year 20, off from then), which would show
  whether the window's supply follows the values held at year 20 without further learning (T-210).
- It does not find the cause of birth order's below-chance hit rate on the twelve-person family (T-211).
