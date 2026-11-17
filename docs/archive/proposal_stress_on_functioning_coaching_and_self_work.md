# Proposal — functioning drops under stress; coaching and self-work reach behaviour

> **Status: approved by the owner, 2026-10-11 (§1 revised, §2); applied at spec revision 18, with the follow-ups of
> revision 19 (the stress floor, self-discovered practice, perspective's steering).** It follows the owner's answers on the failing estimator criteria
> (`docs/implementation_plan_phase_d.md`, step 6 result):
> 1. "yes, functioning drops under stress";
> 2. "more self directed behaviour and better managing of reactivity can be outcomes of coaching OR agents can get to
>    this on their own. The idea is that regular coaching would support positive changes."
>
> Every form and number below is `[I]`. Nothing is built until approved. It would be spec revision 18.

## 1. Stress moves functioning, up or down (revised after the owner's answer)

**The owner, 2026-10-11:** "stressor induced anxiety 'forces' one to function up. E.g. job loss leads to responsible
spending and serious job search OR it can lead to avoidance behaviour and 'soothing', e.g. drinking. This would be
proportional to level of basic self and nature of stressor. Some stressors are short lived — partner mad for 2 turns
vs. no job for 15 turns."

The first draft, a strain that only lowered functioning, is withdrawn.

**Today:**
- **Anxiety never moves `functional_level`.** So a gain borrowed from a partner stays after the partner stops giving
  way, and the estimator reads it as basic (`M11.C.20` reversed).
- **A stressor hits once.** An exogenous stressor is appraised once, the week it arrives; its declared `duration`
  does nothing. A job lost for 15 weeks weighs the same as one bad week.

**Proposed:**

**(a) A stressor lasts as long as it is declared to.** An exogenous stressor with duration *d* keeps loading its
targets each week for *d* weeks, as a `TRIGGER` already does on a tie. Each week after the first, the target takes
`stressor_persistence` × what the first week's arrival added.
- A short stressor and a long one differ by the weeks they weigh.
- A partner who is angry for two weeks is not a stressor in the model. It is that partner's acts (`CONFLICT`), week by
  week.

**(b) What the person does under load moves their functioning.** In any week a person carries load (excess acute
anxiety above the chronic floor):
- **Functioning up** (responsible coping: the job search, the careful spending). If they acted in the self-directed
  channel, or held back (`WITHHOLD`), `functional_level` rises by
  `up_gain × excess / 100 × basic_level / 100`.
- **Functioning down** (avoidance and soothing). If they acted with an avoiding or soothing act, `functional_level`
  falls by `down_gain × excess / 100 × (1 − basic_level / 100)`. The avoiding and soothing acts are listed in config:
  `DISTANCE`, `CUTOFF`, `TRIANGLE`, `UNDERFUNCTION`.
- Any other act (`CONFLICT`, `PURSUE`, `OVERFUNCTION`) leaves it as it is. Taking charge already moves functioning by
  borrowing (`M1.B.5`).

**(c) Basic level sets both the direction and the size.** The direction comes from which channel the person chooses,
and that share already rises with level (`M4.D.1a`; with §2, with perspective too). The size scales with level: a
higher basic self rises more when it copes and falls less when it soothes. Nothing writes "a high level copes"; it
follows from what the person does.

**(d) It settles back slowly.** When load is gone, the part of functioning that stress moved relaxes toward zero by
`response_relaxation_rate` a week. That part is kept as a per-person `stress_swing`, apart from borrowed pseudo-self
(`M6.I.4`). The relaxation is slow, so it is not a symmetric restoring force (`M7.A.1`). What lasts is what the person
kept doing.

**What it should produce, with no rule written for it:**
- **A borrowed gain fades.** Someone running on a borrowed gain who soothes under load loses functioning; one who
  copes keeps it.
- **A long stressor tests more.** It weighs for more weeks, so the person's response accumulates.
- **The estimator sees the difference.** It reads functioning under load, so coping under a long stressor is the
  evidence it is looking for.

**The risks, measured before any criterion is read:**
- runaway: soothing lowers functioning, which steepens reactivity, which brings more soothing;
- everyone drifting upward through coping.

**New constants and config:**
- constants `stressor_persistence`, `up_gain`, `down_gain`, `response_relaxation_rate` (all `[I]`);
- the list of avoiding and soothing acts, in `config/bowen/pattern_acts.md` or its own file.

## 2. How coaching, and a person's own work, reach behaviour

**What exists:**
- `systems_perspective` (0 to 1) is a person's capacity to see the process rather than react to it.
- **What raises it:** today, only a coaching contact that lands (`M1.E.7`).
- **What it does:**
  - lets a person engage a loaded tie rather than avoid it (`M4.D.3b`);
  - lets an `I-POSITION` be held rather than asserted (`M5.F.4`);
  - calms appraisal by letting a person see their own part (`M4.C.6`). This is the "better managing of reactivity"
    the owner describes; it exists already.
- **What it does not do:** it does not change how often a person chooses the self-directed channel. That share reads
  `functional_level` only (`M4.D.1a`). So in the runs, coaching barely changes the behaviour the estimator reads
  (`M11.C.17`: no coach effect).

**Proposed, following the owner: coaching supports change, and a person can get there on their own.**
- **(a) Perspective opens the self-directed channel.** The mixing weight becomes
  `w = f + perspective_channel_gain × perspective × (1 − f)`, where `f = (functional_level / 100) ^ exponent` as
  today.
  - A person with no perspective behaves exactly as now.
  - With perspective, they choose self-directed acts more often at the same functional level.
  - This amends `M4.D.1a` ("functional level only").
- **(b) A person can build perspective by their own practice.** Each self-directed act made under load raises
  `systems_perspective` by `self_practice_gain` of its distance from 1.
  - That gain is smaller than a landed coaching contact (`perspective_gain`, 0.1).
  - So regular coaching gives regular boosts, and a person's own practice builds slowly.
  - Both feed the same loop: perspective, then more self-directed behaviour under load, then a rise the estimator
    reads. Coaching makes the loop turn faster.
  - This amends `M1.E.7`'s "only on a landed contact".
- **(c) Managing reactivity** stays as built (`M4.C.6`), now fed by perspective from either source.
- **New constants:** `perspective_channel_gain`, `self_practice_gain`.
- **What `M11.C.17` should then show:**
  - the coached arm rises more, because its perspective grows faster;
  - the coach-free arm can rise, slowly, from practice;
  - "differentiation is rare" is kept as a reported property.
- **The risk:** the loop feeds itself, and over 40 years everyone could drift upward. I would measure it, report it,
  and not tune it.

## Measured after building §2 (2026-10-11; twelve-person family, no stressor schedule, 10 years, seeds 11 and 0)

| `self_practice_gain` | Perspective after 10 years, without a coach (everyone but Ravi and Marta) | Ravi and Marta (coached by Dr Halim) |
|---|---|---|
| 0.01 (first chosen) | 0.52 to 0.99 | 0.90 to 0.99 |
| 0.001 (now set) | 0.03 to 0.18 | 0.52 to 0.77 |
| 0.0003 | 0.00 to 0.08 | 0.54 to 0.71 |

- **At 0.01 almost everyone saturated within ten years.** That contradicts `M1.A.18d`'s population consequence:
  systems perspective is rare. `M11.C.17` then showed no coach effect, because both arms saturate.
- **At 0.001** a person's own practice builds perspective slowly, and coaching clearly adds to it. The constant is set
  to 0.001 `[I]`, pending the owner.
- **No basic level rose in any of these runs.** They have no stressor schedule, so the estimator's breadth is never
  met.

## What the owner decides

1. Approve the revised §1, or correct it.
2. §2 is approved (2026-10-11).

After approval: spec revision 18, build, the four criteria re-run, then one record regeneration.
