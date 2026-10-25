# Proposal — displaced coping (version 2 stage 3, group C, P2), with the floor-held functioning (F1)

> **Status: DRAFT 2026-10-25, for the owner's approval; not built.** The owner chose P2 as group C's mechanism on
> 2026-10-24 (D18, revisited after the projection trace; `docs/implementation_plan_version_2_stage_3.md` §3.2). A
> probe on 2026-10-25 (§6, `docs/review/stage_3_displaced_coping_probe.md`) showed that P2 at coping size does not
> move `M11.C.6`'s direction, and that most of the child's lift is functioning the stress floor holds. The owner then
> chose, the same day, **P2 as drafted plus F1** (the floor-held functioning put on a ledger the estimator leaves
> out), the two to be probed together on Mac Mini1 before revision 39 (D18, third reading). `M11.C.6` is reported as
> it reads and is never tuned. Every form here is Claude's recommendation, graded `[I]` unless a ledger entry is
> cited. Both change what a run computes, so both move the engine hash; under D31 they are built at plan step 7, after
> revision 39 is approved (step 4). Nothing under `src/bowen/` or `config/bowen/` is edited by this page. `M6.I.4a`
> (D19, `M11.C.21`) is unchanged by it.

## 1. What the owner asked

- Group C's direction, the owner's note: "project DOES impact development, that's the problem." `M11.C.6` asks for the
  child's basic level at 20, less the anchor, to be lower with the parents' projection-type acts toward her than with
  them withheld. At revision 38 it is higher: +0.0886 ± 0.0439 on 150 seeds.
- D18, revisited 2026-10-24: P1 (the parents' term net of borrowed pseudo-self) is dropped, and P2, displaced coping
  (`KS04.7`, `FE01.1`), is group C's mechanism, drafted for approval before revision 39.
- 2026-10-25, after the probe, through `AskUserQuestion` (the recommended option): keep P2 as drafted, add F1 as a
  second item for revision 39, and probe both together on Mac Mini1 before revision 39.

## 2. What the code does today

From `docs/review/stage_3_projection_trace.md` and `docs/review/stage_3_displaced_coping_probe.md` (150 seeds each,
engine `66dcbb1e8ba5`).

- **The development step reads functioning on the ledgers that count.** `M7.E.1f`'s `d` is half the child's own
  year-mean less their basic level and half the parents'. Both read the estimator's functional level,
  `functional_level − pseudo_self − max(0, adjust_swing)` (`slow_tick.py::record_week`). So the over/under-functioning
  exchange (`M1.B.5`, `moves.py::functioning_shift`), which moves functional level and pseudo-self together, moves
  neither term: the parents' term carries −0.0072 ± 0.0296 of the difference, the child's own term +0.0951 ± 0.0362.
- **An act on another's behalf moves no functioning the step reads.** Under `M1.A.5e` a loaded person's self-directed
  act or `WITHHOLD` raises their functional level, a listed soothing act lowers it, and "any other act leaves it".
  `OVERFUNCTION` and `PURSUE` are neither, so a parent who meets their own load by taking charge of a child changes the
  child only on the pseudo-self ledger, which the step nets out.
- **Where the child's lift is** (treatment − baseline, each part's contribution to `development`, summed over ages
  6–20). The own term's +0.0951 is the stress swing +0.0470 ± 0.0237, functioning on no ledger +0.0620 ± 0.0308, and
  the step's own pull back toward the anchor −0.0138 ± 0.0099. All of the functioning on no ledger is held by a bound of the stress response: none
  was there at the anchor (age 5) and none comes from anything else, in every variant.
- **How the floor holds it** (`stress_response.respond`). Giving way cannot take a person below `basic_level −
  stress_floor_below_basic` (`M1.A.5e`, the exchange's floor), and a calm week's relaxation is clamped at the same
  floor, while the swings decay by their full share "whether or not a bound let functioning move" (review
  2026-10-11). A child who has lent pseudo-self down to the floor and carries a positive swing therefore keeps the
  functioning the swing would have shed, on no ledger. The estimator adds the lent pseudo-self back (revision 23) and
  subtracts no swing, so it reads that functioning as the child's own.
- **The dose.** About 3.5 more projection-type acts a year toward the child in the treatment arm (+53 over ages 5–20).

## 3. What the corpus says

| Entry | Grade | What it gives P2 |
|---|---|---|
| `KS04.7` | `[K]` `[T]` `[M]` | The projection cycle. The mother worries about the child and relates to the child as if the worry were true; the child goes along "because this rewards the child with a calmer mother"; the mother is calmer because she "sees the anxiety as coming from outside herself"; the child comes to look to the mother to solve problems, which keeps her over-functioning. Both sides are rewarded, so the loop holds without either intending it. |
| `FE01.1` | `[K]` `[T]` `[M]` | The same cycle as positive feedback: the anxious one's preoccupation with the "problem one" makes the problem one exaggerate the problem, which raises the anxious one's anxiety; the anxious one ends in a caretaking position and the problem one in a patient or child position, each more than they wanted. Neither can change their own functioning enough to stop it. |
| `FE06.3` | `[K]` `[T]` | The parents give up self to the child, but less than the child does; the child's functioning is the most impaired. The direction group C has to produce. |
| `FE07.4` | `[K]` `[T]` `[M]` | The same loop in five steps: the mother's anxiety about the child, her meaning, her treating the child as her image of him, the child learning that acting as that image calms her (and him), and the child taking the image in. Neither is malicious; both are held by it. |
| `KS00.9` | `[K]` `[T]` | The parents' levels set the family climate (why `M7.E.1f`'s parents' term stays as built). |

Every entry is Kerr's (`[K]`): Kerr's *Bowen Theory's Secrets* and Kerr's chapters of *Family Evaluation*. None is
`[K-ext]`, and none gives a magnitude: no entry says how much functioning one act moves. Nothing here is Bowen's own
wording, so P2 is never attributed to Bowen (`[B]`). F1 is bookkeeping and has no corpus source: it removes
functioning no rule gives, and adds none.

## 4. The mechanisms

### 4.1 P2 — new `M1.A.5f`, displaced coping

In a week a person carries load above their own usual level (`M1.A.5e`'s "loaded"), an act of a kind listed as
**displacing** in `config/bowen/stress_response.md` meets that load through the act's receiver: the amount coping
would have given the sender, `response_up_gain × above ÷ 100 × basic_level ÷ 100`, moves from the receiver's
functional level to the sender's, on both people's stress swing (`stress_swing`), split evenly over the act's living
family receivers. A receiver who holds a position toward the sender (`M1.B.5`, revision 21) gives nothing. The receiver
is not taken below `basic_level − stress_floor_below_basic`, and the sender not above 100; one amount moves, the
smaller of the three, so the transfer is conserved exactly. Calm, each person's stress swing settles back as `M1.A.5e`
already says. The list is the config's, proposed as `OVERFUNCTION` alone.

| Step | Grade | Attribution | Source |
|---|---|---|---|
| A loaded person who takes charge of another is managing their own anxiety through the other ("sees the anxiety as coming from outside herself") | `[T]` `[M]` | `[K]` | `KS04.7`; `FE01.1` |
| The sender functions up for it, as their own coping would have raised them (the calmer, caretaking parent) | `[D]` | `[K]` | `KS04.7` ("a calmer mother"); `FE01.1` (caretaking position) |
| The receiver functions down for it (the patient or child position; the child's functioning most impaired) | `[D]` | `[K]` | `FE01.1`; `FE06.3` |
| The two amounts are equal: what one gains the other loses | `[I]` | — | a design choice; `FE06.3` says the child loses more than the parents give up, which the exchange (`M6.I.4`) and the child's own coping and soothing can still produce |
| The amount is `M1.A.5e`'s coping amount for the sender | `[I]` | — | the owner's `M1.A.5e` form, reused; no new constant |
| It lands on the stress swing, so the estimator and the development step read it, and it settles back when the parties are calm | `[I]` | — | `M1.A.5e`'s settling; the corpus says the cycle lasts while it is repeated (`FE01.1`: "each person became an emotional prisoner"), and repetition is what learning decides |
| A held position stops it | `[I]` | — | the exchange's rule (revision 21, the owner 2026-10-12), applied to the same act |
| Which kinds displace: `OVERFUNCTION`, taking charge of another's functioning | `[I]` | — | editorial, in config; `FE01.1`'s "caretaking" |

**What P2 is not.** It is not a second pseudo-self exchange: it writes no `pseudo_self`, so `M6.I.4` and `M1.A.5c` are
untouched, and `M6.I.4a` (D19) still returns borrowed pseudo-self on a collapse. It does not read who the receiver is,
the receiver's age, focus, or any readout. It does not touch `M7.E.1f`.

### 4.2 F1 — the floor-held functioning on a ledger (amends `M1.A.5e` and `M1.A.4k`)

Each week the stress response's floor (or the scale's 0–100) stops functional level from moving while the swings
decay, the amount it stopped, `(change in functional_level) − (change in stress_swing + adjust_swing)`, is added to a
new per-person ledger, `floor_held`. The estimator's weekly reading leaves it out:
`functional_level − pseudo_self − max(0, adjust_swing) − floor_held`. In a calm week, once the bound no longer stops
it, the relaxation sheds `floor_held` by the same share as the swings, moving functional level with it; so the
ledger and functioning stay equal, and what the floor held leaves when the floor lets it. Grade `[I]` throughout, no
new constant (the share is `response_relaxation_rate`).

**The trade-off, for the owner.** The review of 2026-10-11 chose to let the swings decay in full under a bound so
that "a clamped relaxation must not leave an amount behind to take back later from an unrelated gain". F1 keeps that
choice for the swings and records what the bound held apart, so the estimator does not credit it; but its last
sentence does take the held amount back later, when the person is off the floor, which may be after an unrelated
gain (a borrowed rise, coping). The alternative is a ledger the estimator leaves out and nothing ever sheds: the
held functioning then stays in `functional_level` until death, read by nothing that develops a level. §9's Q2 asks.

## 5. The §19 check

`model_explainer.md` §19.1 puts "what each act does to the state of the system" in the set layer and "projection onto a
child" in the emergent layer. P2 is the first kind: it states what one act does, for every sender and every receiver
(spouse to spouse, child to parent, a grandparent to anyone). F1 changes no act and no outcome: it records functioning
a bound kept and stops the estimator from reading it as the person's own, for everyone. Neither rule names projection,
a child or an outcome.

What they leave to the run: who sends `OVERFUNCTION` to whom (the policy, focus `M4.D.1g`, the weakness focus
`M1.B.8a`, and what each person has learned relieves them), whether the sender is loaded at the time (the run's
anxiety), whether the receiver holds a position (the receiver's own behaviour), whether a person gives way to the
floor at all (the exchange), and what the receiver does with the functioning they lost (their coping and soothing,
`M1.A.5e`, and their learning).

**One risk, stated plainly.** `M11.C.6`'s two arms differ only in the parents' projection-type acts toward the child,
and P2's direct effect on the child of each `OVERFUNCTION` has a fixed sign. So P2 pushes `M11.C.6` in the asked
direction by construction; what is not written is whether that push outweighs the child's own stress swing, and
whether the parents' gain, which enters `d` too, cancels it. The probe (§6) shows it does cancel at coping size. F1
removes a lift rather than adding a push, so it supplies no sign. The direction is therefore still read, not set, and
`M11.C.6`'s `M11.5` row should say that P2 alone supplies a sign; the new mutants `displaced-coping-removed` and
`floor-held-credited` must each move it back.

## 6. The probe (no engine change)

`tools/stage_3_displaced_coping_probe.py`, in no `HASHED_TOOLS` list; `docs/review/stage_3_displaced_coping_probe.md`;
`M11.C.6`'s own arm, both arms, seeds 0–149, P2 put in by wrapping `tick.respond` in the tool's process. A probe's
result, not a criterion's (`M11.5`). Treatment − baseline, `development`:

| Variant | `development` | own term | parents' term |
|---|---|---|---|
| as built | +0.0886 ± 0.0439 | +0.0951 ± 0.0362 | −0.0072 ± 0.0296 |
| P2, `OVERFUNCTION` | +0.0919 ± 0.0420 | +0.0566 ± 0.0353 | +0.0353 ± 0.0326 |
| P2, `OVERFUNCTION` and `PURSUE` | +0.0853 ± 0.0398 | +0.0158 ± 0.0300 | +0.0692 ± 0.0357 |
| receiver only, `OVERFUNCTION` | +0.1265 ± 0.0388 | +0.0465 ± 0.0329 | +0.0797 ± 0.0274 |
| receiver only, `OVERFUNCTION` and `PURSUE` | +0.0774 ± 0.0464 | +0.0378 ± 0.0329 | +0.0393 ± 0.0356 |

- **P2 does not move the direction at coping size.** It lowers the child's own term (by 0.04 with `OVERFUNCTION`, 0.08
  with `PURSUE` too) and raises the parents' term by about as much, so `development` does not change. The receiver-only
  forms (the sender gains no functioning) do not move it either: against the engine as built they lower `development`
  in both arms, by 0.05 to 0.14, the baseline about as much as the treatment.
- **The amounts are small.** P2 moves 0.17 (`OVERFUNCTION`) to 0.36 (with `PURSUE`) points of functioning to the child
  over ages 5–20 in the treatment arm, and about 0.26 to 0.55 from the child to others in both arms.
- **The other half of the own-term lift** is functioning on no ledger, +0.0620 ± 0.0308, and all
  of it is held by a bound of the stress response. The treatment child is at the stress floor 90.5 weeks a run from age
  5 against 68.5 in the baseline, and a bound holds their functioning 44.4 weeks against 29.7. P2 adds to it: with
  `OVERFUNCTION` the treatment child is at the floor 105.4 weeks, and the held part is +0.0659 ± 0.0316.
- **First-order, not run:** taking the floor-held part out of the as-built own term leaves `development` near +0.03,
  and out of the four P2 variants' +0.03 to +0.06; no combination reverses the sign on this reading. The probe of both together
  (§9) is the measurement; this estimate leaves out every feedback.

## 7. Constants and config

- **No new constant** in the recommended forms: P2's amount is `M1.A.5e`'s coping amount (`response_up_gain`, 0.5,
  `[I]`), the floor is `stress_floor_below_basic` (15, `[I]`), and F1's shedding share is `response_relaxation_rate`
  (0.02, `[I]`). `config/bowen/constants.md` is unchanged.
- **One new config list.** `config/bowen/stress_response.md` gains a second table, *displacing*: the kinds that meet a
  loaded sender's load through the receiver, each with its reason, every entry `[I]`. Proposed: `OVERFUNCTION`.
  `src/bowen/io/load.py` reads it into a new `EventKinds.displacing`. A kind may not be both soothing and displacing:
  load fails if it is. Editorial content stays in config (the global rule 9).
- **One new field.** `Person.floor_held` (F1), beside `stress_swing` and `adjust_swing` (`objects.py`).

## 8. Classes, tests and spec changes at revision 39

**Classes (`M11.5`).**

| Criterion or test | Class | Why |
|---|---|---|
| `M11.C.6` | composite | the decline needs focus, learned projection, the parents' load at the time, the child's own responses and P2 together; the row says that P2 supplies a sign and F1 none (§5) |
| `M11.C.2`, `M11.C.2[twelve-person family]` | composite | unchanged; reported, not targeted |
| `M11.C.21` | composite | unchanged by P2 (D19's `M6.I.4a`); reported, since the collapse arm's parents send `OVERFUNCTION` under load and F1 changes what the estimator reads near the floor |
| S3.1a–S3.1d (below) | premise (unit) or check | one rule produces each, so each confirms the code and is never reported as a finding |

**The tests that replace S3.1** (written red at plan step 5, green at step 7).

| ID | What it checks | Test | Proved failing by |
|---|---|---|---|
| S3.1a (`M1.A.5f`) | premise: a loaded sender's `OVERFUNCTION` moves the coping amount from the receiver's functional level and stress swing to the sender's; the pair's total functional level is unchanged | `tests/bowen/test_stress_response.py::test_m1a5f_s3_1a_displaced_coping_moves_functioning_from_receiver_to_sender` | `displaced-coping-removed` |
| S3.1b (`M1.A.5f`) | premise: nothing moves when the sender is calm (at or below their usual level), when the receiver holds a position toward the sender, or when the receiver is at their floor; the same act from a calm sender leaves both as `M1.A.5e` does | `tests/bowen/test_stress_response.py::test_m1a5f_s3_1b_no_displacement_when_calm_held_or_at_the_floor` | `displaced-coping-unguarded` (the three guards removed) |
| S3.1c (`M1.A.5f`) | check: the displacing list is read from `config/bowen/stress_response.md`, and a kind listed both soothing and displacing fails at load | `tests/bowen/test_config.py::test_m1a5f_s3_1c_displacing_kinds_come_from_config` | the list moved into code |
| S3.1d (`M1.A.5e`, `M1.A.4k`, F1) | premise: a calm week at the floor with a positive swing adds what the floor held to `floor_held`, leaves `functional_level − stress_swing − adjust_swing − floor_held` unchanged, and the estimator's reading leaves it out; off the floor it is shed with the swings | `tests/bowen/test_stress_response.py::test_m1a5e_s3_1d_floor_held_functioning_is_on_a_ledger` | `floor-held-credited` (the estimator reads `floor_held` as own functioning) |

S3.2 (`M11.C.6`) is proved failing by `displaced-coping-removed` and `floor-held-credited` (both new, added to
`tools/phase_d_mutation_record.py`) and `development-fixed-at-anchor` (existing). `parents-term-gross` was P1's and is
not added.

**Spec changes at revision 39.**

- New `M1.A.5f` (§4.1's rule, its grades and its sources).
- `M1.A.5e`: "any other act leaves it" gains "except `M1.A.5f`'s displacing acts"; the floor's paragraph gains F1's
  ledger (§4.2).
- `M1.A.4k` (the estimator's reading): leaves out `floor_held`, beside pseudo-self and the system's adjustment.
- `M7.E.1f`: one sentence, no rule change: the step reads the stress-swing ledger, so `M1.A.5f` reaches it, and it
  nets out pseudo-self and `floor_held`, so the exchange and the floor do not.
- `M11.5`: `M11.C.6`'s row gains §5's note.
- `config/bowen/stress_response.md` (the *displacing* table); `docs/model_explainer.md`'s sections on the stress
  response, the estimator and development; the revision log, the indexes and the rules view.
- **Stage 4 (Q7).** Check `M1.A.5e`, `M1.A.5f`, `M1.A.4k`, `M7.E.1f`, `M6.I.4` and `M11.C.6`'s row against `KS04.7`,
  `FE01.1`, `FE06.3`, `FE07.4` and `KS00.9`, and fix in revision 39 what is found.

## 9. What comes next, and the questions still open

**Next (decided 2026-10-25): probe P2 and F1 together on Mac Mini1, before revision 39.** The probe tool gains a
variant with both (F1 put in by wrapping `stress_response.respond` and `slow_tick.record_week` in the tool's process),
run at 150 seeds on Mac Mini1. Its result goes to the owner with revision 39's draft; `M11.C.6` is reported as it
reads, whatever its sign, and no size is chosen to make it pass.

Open, for the owner with revision 39 (Claude's recommendation first):

- **Q1, the displacing kinds.** `OVERFUNCTION` alone (recommended: taking charge is `FE01.1`'s caretaking; `PURSUE`
  is a closeness move, and the probe shows it doubles what P2 moves without changing the direction), or with `PURSUE`.
- **Q2, F1's shedding.** Shed off the floor with the swings (recommended: what the floor held is the swing's deferred
  relaxation), or kept and never shed (§4.2's trade-off).
- **Q3, the sender's side.** Conserved, the sender gaining what the receiver loses (recommended, as drafted), or the
  receiver's loss only. The probe gives no reason to prefer either for `M11.C.6`.

## 10. Edits this makes to the stage 3 plan

Made in the same commit as this page (`docs/implementation_plan_version_2_stage_3.md`):

- **§3.2:** D18's row records the probe and the owner's choice of 2026-10-25.
- **§3.3.1:** a "P2, the mechanism" paragraph and an "F1" paragraph pointing here, with the probe's result; "Spec
  changes", "§19 check", "Classes" and "Stage 4" restated; P1's paragraph kept as the record of what was proposed.
- **§3.7:** S3.1 replaced by S3.1a–S3.1d; S3.2's mutants `displaced-coping-removed`, `floor-held-credited` and
  `development-fixed-at-anchor`; `M1.A.5f` added to the provisional IDs.
- **§3.8:** step 3 records the probe and the owner's choice, and the joint probe on Mac Mini1 before step 4; step 4's
  list gains `M1.A.5e`, `M1.A.5f`, `M1.A.4k` and `config/bowen/stress_response.md`; step 5's red tests are
  S3.1a–S3.1d; step 7 builds P2, F1 and `M6.I.4a`, with `displaced-coping-removed`, `displaced-coping-unguarded` and
  `floor-held-credited` added to the Phase D mutation record.
- **§3.11 and §3.12:** the documents step 4 updates gain `config/bowen/stress_response.md`; the §19 judgment reads P2
  and F1, not P1.
