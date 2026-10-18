# Proposal — learning by level, and asking "what can I do differently?"

> **Status: DRAFT 2026-10-18, for owner review. Version 2; nothing decided, nothing built.** Written from the owner's
> statement of 2026-10-18 (below). Every form and number is Claude's recommendation and `[I]`. Spec IDs are assigned
> when a revision applies it (`docs/version_2_plan.md` §3); until then the parts are P1–P4, the criteria K1–K6 and
> the questions Q1–Q9. It changes what a run computes, so it moves the engine hash and regenerates every record once
> (about 2 hours on 10 cores, `docs/version_2_plan.md` §2).

**The owner, 2026-10-18:** "All people learn. The higher the level of differentiation the more the learning is fact
based and adhered to. When anxiety gets high enough there is a possibility that a person would ask — what can I do
different. Coaching would definitely help the chances of that question being asked and the answer being pursued."

## 1. What the model does today

| The owner's point | What exists | What is missing |
|---|---|---|
| All people learn | Everyone's automatic channel learns from felt relief: an act is credited with the change in the actor's acute anxiety over 3 weeks, plus the calming of the people they are tied to (`M4.D.6`, `M4.D.6a`, `M4.D.6e`). Witnesses learn about `CUTOFF` from what they see happen to people's functioning (`M4.D.6f`). | Felt relief is the only lesson an act teaches its actor. A person of level 60 and one of level 25 learn from the same signal. |
| Learning is more fact based at higher level | Nothing. Level sets which channel a person acts from (`M4.D.1a`), not what they learn. | A lesson from what an act actually led to. |
| …and more adhered to | Nothing. A learned value is followed with the same randomness at every level, and the next episode of relief can undo it. | Level, and anxiety, setting how firmly a lesson guides choice. |
| Under high anxiety, a person may ask "what can I do differently?" | The soothing review (`M4.D.6`, revision 19): every 12 weeks, whatever the anxiety, a person asks with probability `(basic_level / 100 + perspective) / 2` whether soothing helped; if their load 4 weeks after soothing was no lower, soothing acts lose half their learned value. | A question triggered by high anxiety; a question about every act, not only soothing; an answer that is pursued. |
| Coaching helps the question be asked and the answer pursued | A landed coaching contact raises `systems_perspective` by 0.1 of its gap to 1 (`M1.E.7`); perspective raises the self-directed share (`M4.D.1a`) and the soothing review's chance. Practice that worked raises it too, by 0.001 (`M1.E.7`, revision 19). | A session prompting the question directly; coaching helping the answer be followed through. |

## 2. The proposal

### P1. Every act teaches two lessons: what it felt like, and what it led to

**All people keep both.**
- **The felt lesson** is today's learned value, unchanged (`M4.D.6a`): felt relief over `credit_horizon` (3 weeks).
- **The fact lesson** is new: what the act led to over a longer window, `fact_window` (12 weeks, `[I]`). Its signal is
  - the change in the actor's own load (excess above their usual level, `M1.A.5e`) from the week of the act to
    `fact_window` weeks later, sign reversed so that a fall is positive; plus
  - `fact_others_weight` (0.5, `[I]`) × the bond-weighted mean change in the **functional level** of the act's target
    and witnesses over the same window — the form `M4.D.6f` already uses for a witnessed cutoff ("what the witness
    sees happen").
- It is updated in `M4.D.6`'s form: `fact ← fact + learning_rate × (signal − fact)`, under the same key (anxiety
  band, act, target or triad).
- **Not habituated.** Habituation (`M4.G.3`) is about relief wearing off; a fact does not wear off by repetition.

**Why functioning, not anxiety, for others:** anxiety is what the felt lesson already reads. Functioning is the closest
thing in the model to "what happened" — whether the other person is managing better or worse.

**The limit, stated:** the fact lesson credits an act with everything that changed in the next 12 weeks, so it is
also fooled — by a stressor that ends, by decay toward the floor. It is fooled less by immediate relief, which is
the point. K1 tests that it moves the right way where the long-run effect of an act is known.

### P2. Level sets how much the fact lesson guides choice; anxiety erodes it, faster at lower level

The automatic channel scores an act by a blend of the two lessons instead of the felt lesson alone:

    value = (1 − g) × felt + g × fact
    g = β × (β·K) / (β·K + e)

- `β = basic_level / 100`: the person's differentiation (Q1).
- `e`: the person's excess acute anxiety above their usual level (`M1.A.5e`).
- `K = flood_scale` (`[I]`, 40 anxiety points): at `e = β·K`, half the fact lesson's weight is gone.

**What this gives:**
- **Fact based at higher level.** Calm, a person at level 60 is guided 60% by what acts led to; at level 25, 25%.
- **Adhered to at higher level.** Anxiety floods the fact lesson, and floods it sooner the lower the level: with
  `K = 40`, a level-25 person has lost half at 10 points of excess anxiety, a level-60 person at 24. So under
  pressure the higher-level person keeps acting on what they know; the lower-level person goes back to what relieves.
- **A person with no fact lesson yet** (value 0) is pulled toward neutral, not toward any act.

**Nothing here names a good act.** Which acts the facts favour is whatever the run shows. If the facts favour
`DISTANCE` for someone, a higher-level person learns `DISTANCE`. That is a result to report, not to fix.

**A conflict to decide (Q2):** `M4.D.3` says the policy **MUST NOT** carry a separate term that raises reactive
weights with anxiety. `g` does not raise any act's weight by name; it shifts which lesson guides. But where the felt
lesson favours reactive acts, rising anxiety now moves choice toward them by a third route. Approving P2 amends
`M4.D.3` to admit it.

### P3. When anxiety is high, a person may ask "what can I do differently?" — and may pursue the answer

**When the question can come.** In any week a person's excess anxiety `e` is above `ask_threshold` (15, `[I]`, the
level of `defence_threshold`), they may ask, by one keyed draw, with probability

    p_ask = ask_rate × (β + perspective) / 2 × (1 + coach_ask_gain × coached)

- `ask_rate` (0.1, `[I]`): the chance per week at full level and perspective.
- `coached` is 1 if the person had a coaching session in the last `coach_window` weeks (4, `[I]`), else 0 (P4).
- After asking, a person cannot ask again for `ask_refractory` weeks (12, `[I]`, today's review interval).

**What asking does — the review.** The person looks at the acts they used in the last `ask_refractory` weeks and,
for each one whose fact lesson is below its felt lesson (it felt better than it turned out), moves the felt lesson
`review_step` (0.5, `[I]`) of the way down to the fact lesson. This is the soothing review widened: every automatic
act, judged by the fact lesson rather than by a separate 4-week comparison, and triggered by anxiety rather than by
the calendar. **Recommended: it replaces the soothing review (Q4).**

**Pursuing the answer.** Having asked, the person pursues the answer with probability

    p_pursue = (β + perspective) / 2 × (1 + coach_pursue_gain × landed)

where `landed` is 1 if a coaching contact landed (`M1.E.7`) in the last `coach_window` weeks.

**The pursuit begins when anxiety eases, not at its peak.** The question comes under high anxiety; the steps come
after. Bowen: a continued low level of anxiety is what *permits* motivated members to begin slow steps toward
differentiation (`L22.1` `[B]`); under distress the motivation aims at relieving the symptom (`L06.2` `[B]`). So a
person who will pursue starts doing so in the first week their excess falls back below `ask_threshold`, within
`pursue_wait` weeks (12, `[I]`); if their anxiety has not come down by then, the answer is not pursued (Q9).

Pursuing opens the self-directed channel for `pursue_weeks` (8, `[I]`): the mixing weight (`M4.D.1a`) gains
`pursue_channel_gain × (1 − w)` (0.5, `[I]`), the same form perspective uses. Not pursuing: the review has happened,
nothing more. Abandoning the move at the first opposition, which Bowen names as the usual outcome (`L13.3` `[B]`), is
left to the existing change-back mechanisms (`M5.D`); nothing here makes a pursued answer stick.

**Why this does not write an outcome.** The question shifts which lessons and which channel the person acts from. It
does not choose an act. Whether the self-directed acts made while pursuing hold, provoke a change-back reaction, or
fail, is what the existing mechanisms decide (`M5.D`, `M5.F`). A pursued answer that works builds perspective by the
existing practice rule (`M1.E.7`, revision 19), so the next question is more likely: this is the loop the owner
describes, with coaching making it turn faster.

### P4. Coaching raises both chances

- **A session prompts the question.** Any session in the last `coach_window` weeks multiplies `p_ask` by
  `1 + coach_ask_gain` (1.0, `[I]`, so doubles it). The coach asks; whether the person asks themselves is still a draw.
  *This route rests on the owner's statement; no corpus source says coaching raises the chance of the question (§3,
  point 4). Q5 offers the alternative the corpus fits better.*
- **A landed contact supports the pursuit.** A contact that landed (`M1.E.7d`: coach quality, the person's
  motivation, occasion) in the window multiplies `p_pursue` by `1 + coach_pursue_gain` (1.0, `[I]`), capped at 1.
- **Not monotone in frequency (`M1.E.8`).** `coached` and `landed` are on/off within the window, so a second session in
  the same 4 weeks adds nothing to these chances; `M1.E.8`'s fall in landing above a low contact rate still applies to
  `landed`.
- **Perspective still counts.** A coachee's perspective, raised by landed contacts, enters both chances through
  `(β + perspective) / 2`, so coaching works by both routes; a person with no coach can get there more slowly through
  their own perspective, as the owner said at revision 18.

### What is unchanged

- The self-directed channel is not reinforced by relief or by facts (`M4.D.6d`, `M5.F.5`). Q6 asks whether that should
  change.
- `basic_level` and `systems_perspective` are not reinforcement targets (`M4.D.6d`). Perspective still rises only by
  landed contacts and by practice that worked (`M1.E.7`).
- Felt relief, habituation, the cross-person term and witnessed cutoff (`M4.D.6a`–`M4.D.6f`) are unchanged.

## 3. What the corpus says

Searched in `docs/theory/_LEDGER.md` and the extraction files it points to, 2026-10-18. Attribution: `L` entries and
`FE11` are Bowen `[B]`; `FE01`–`FE10` are Kerr `[K]`; `KS` entries are `[K]` or `[K-ext]`; `KB` and `Tape` entries are
ASR transcripts, cited as the interview or lecture, never quoted. Entries marked † are in the extraction files only,
not yet in the ledger body. **The corpus gives directions for parts of this proposal and no magnitudes**, except one
count of helpful coaching contacts (`K07.4`). Every constant in §6 is `[I]`.

### Point 1 — all people learn

- **Supports:**
  - Childhood learning is the main influence on which kind of symptom a person develops (`FE08.3` `[K]`).
  - Learning passes between people: the child learns that acting as the mother's image predicts calms her, and her
    calm is the reward (`FE07.4` `[K]`). This is already `M4.D.6e`.
  - A family can learn to normalise behaviour by small concessions that are never reversed (`KS-C.7` `[K]`).
  - Patterns learned in the first triangle carry into later relationships (`L22.4` `[B]`).
- **Qualifies:**
  - What is learned in one relationship does not transfer automatically to another (`KS-A.1` `[K]`; `KS17.1`).
    P1 keeps lessons per target, which fits.
  - Basic patterns are set early and usually stay fixed; later learning is mostly functional (`FE04.9` `[K]`,
    `L22.7` `[B]`).
- **Gap:** no source states "all people learn" as a general rule. It is the owner's premise, and the corpus does
  not contradict it.

### Point 2 — higher level, learning more fact based and adhered to

- **Supports:**
  - Solid self is made of convictions formed slowly, changed only from within, and consistent with factual
    knowledge. Pseudo-self is beliefs taken on under emotional pressure, negotiable, and able to conflict with facts
    (`FE04.10` `[K]` †; in the ledger's `FE-B` group). This is the closest statement to the owner's.
  - Above the middle of the scale people are more aware of the difference between feeling and intellectual
    principle (`FE04.3` `[K]`, marked `[X]` against Bowen's own wording at `L16.1`).
  - Below 50, the emotional system leaves the intellect to decide only matters outside the joint life course
    (`L16.1` `[B]`). Decisions from the feeling of the moment against decisions from principle and reality grade
    bosses as they grade parents (`L20.1` `[B]`).
  - Conviction is what lets a change survive later turmoil (`FE10.12` `[K]` †): the "adhered to" half.
  - At the lowest levels people are drawn to comfort; in the middle range some lose motivation once comfortable,
    others keep working (`FE04.2` `[K]`).
- **Qualifies, and changes what P2 may claim:**
  - How firmly a belief is held is produced independently of whether it is true (`FE02.6` `[K]` †), and a stand
    does not have to be right (`FE04.12` `[K]` †). So "adhered to" **MUST NOT** be read as "right". P2's fact lesson can
    be wrong too (§2, P1's stated limit).
  - In the 25–50 range, dogmatic authority, compliance and rebellion all look firm and principled, and intellect can
    serve the relationship system (`L10.12` `[B]`); a self claimed angrily or dogmatically is usually unsure of
    itself (`FE04.4` `[K]`, `FE11.19` `[B]`). **The proposal does not model this pseudo-firmness**: in P2, firmness rises
    with basic level only. A gap, stated.
  - Low level is not lower intellect; the intellect works in the service of the emotional process (`FE02.16` `[K]` †).
    P2's anxiety term is that idea: the fact lesson is there, and anxiety stops it guiding choice.
  - Overt feeling peaks in the middle of the scale, not at the bottom (`L16.4` `[B]`). P2's form is monotone in level
    and does not show this.
- **Gap:** no source says higher-level people learn from longer-term consequences in a reinforcement sense. The
  nearest are reappraisal as self-attribution (`KS-C.9` `[K]`, already `M4.C.6`) and seeing one's own consequences as
  the motive for change (`KS15.13` `[K-ext]` †, Kerr's own extension, not Bowen's). P1's 12-week fact signal is `[I]`.

### Point 3 — high anxiety may prompt "what can I do differently?"

This is where the corpus corrects the owner's statement most.
- **Supports:**
  - Motivation is lost on comfort, and the next period of stress may renew it (`FE04.2`, `FE09.14` `[K]` †).
  - One way a regression ends is a member saying they cannot go on, an I-position taken out of desperation with no
    systems understanding behind it (`KS14.5` `[K]` †).
  - Serious illness and death are named as occasions that make the move possible (`L22.2` `[B]`); the occasion is a
    condition the coach does not control (`KS17.10` `[K]`).
  - Seeing one's own part and taking responsibility for it is what separates real change from moving a symptom
    elsewhere (`FE05.2` `[K]` †); if self changes its part in a conflict, the others change theirs (`FE11.19` `[B]`);
    children's symptoms fell when parents kept the focus on themselves (`L06.1` `[B]`).
- **Qualifies:**
  - Under distress, motivation aims at relieving the symptom, not at changing what lies behind it (`L06.2` `[B]`,
    corrected; its scope is disputed, `C1`).
  - A continued low level of anxiety is what permits slow steps toward differentiation (`L22.1` `[B]`). **This moved
    P3's pursuit to the weeks after anxiety eases.**
  - Most families get relief and only a minority change basic level (`FE10.11` `[K]` †); the usual outcome of getting
    help is lower anxiety with no basic change (`KS14.5` `[K]` †). This supports a low `ask_rate`.
  - Rising anxiety favours disengaging; willingness to engage a hard tie depends on perspective (`KS-C.11` `[K]`).
  - Accepting "I had a part" did nothing until the specifics were found (`KS10.11` `[K]` †).
  - Abandoning the move at the first opposition is the usual outcome (`L13.3` `[B]`).
- **Gap:** no source says high anxiety by itself makes the question more likely; there is no threshold and no rate.
  `ask_threshold` and `ask_rate` are `[I]`, and the trigger rests on the owner's statement.

### Point 4 — coaching raises the chance of asking and of pursuing

- **Supports:**
  - Coaching is an educational process that helps a person tell individual from systems thinking, but a coach cannot
    make anyone differentiate; it takes coach quality, the person's motivation and the occasion together
    (`KS17.10` `[K]`; already `M1.E.7d`).
  - A contact helps by naming a process the person had no name for, or by supplying specifics about their own
    behaviour (`KS13.4`, `KS10.11` `[K]`); a theory makes the hard tie approachable (`KS-C.11` `[K]`).
  - Bowen works with the motivated member (`L06.4`, `L21.10` `[B]`); the coach's task is to move the person's aim off
    symptom relief (`L06.2` `[B]`).
  - The one magnitude: in one account, over two years at six to eight contacts a year, four or five contacts helped
    (`K07.4`, KB interview, ASR). Today's `landing_rate` is 0.15 per contact at full quality.
- **Qualifies:**
  - Progress is limited by the person's own effort between sessions, not by contact (`L22.4` `[B]`); more coaching is
    not faster progress (`L14.3` `[B]`); four sessions a month did worse than one (`FE11.8` `[B]`, already `M1.E.8`).
  - A recipient who is not ready gets no change (`L17.3` `[B]`).
  - A step taken because "the doctor told me" works differently from one the person chose (`L05.6` `[B]`).
- **Gap:** the corpus treats coaching as supplying a way of thinking and specifics; the motivation stays with the
  person. **No source says coaching raises the chance that the question is asked.** P4's session route is the owner's
  statement, `[I]`. The landed-contact route (perspective, and support for the pursuit) fits the corpus better.

## 4. Acceptance criteria (candidates)

K1, K3, K4 and K5 are directions of difference between two arms over an ensemble (`M0.4`, `M11.2`), each with its
class (`M11.5`). A premise passing confirms the code and is never reported as a finding. K2 is a unit test of the
formula, not an `M11.C` criterion. K6 is a reported readout with no direction; it cannot be a criterion, because
`M11.4`'s list of exceptions to `M0.4` is closed.

| # | Class | Arms | Direction | Proved red by |
|---|---|---|---|---|
| K1 | premise | A scripted act whose 12-week consequence is known (its target's functioning falls after it) against the same act with that consequence removed | The fact lesson of the act is lower in the first arm, while its felt lesson is the same | the fact signal reading anxiety instead of functioning; the window set to the credit horizon |
| K2 | unit test | The same person at basic level 60 against 25, same anxiety | `g` is higher at 60; and at the same rise in anxiety, `g` falls less at 60 | `g` reading `functional_level`; `e` dropped from `g` |
| K3 | composite | Fact lessons on (P1–P2) against off (`g = 0`), the Phase C family, 104 weeks | The share of avoiding and soothing acts among the higher-level members' automatic acts is lower with fact lessons on, and the difference is larger for higher-level members than for lower | `g = β` without the anxiety term (the level-by-anxiety interaction should weaken); fact signal set to felt relief |
| K4 | composite | The question on (P3) against off, with a nodal event at week 20 | In the 26 weeks after the event, the asking members' self-directed share and functional level are higher with the question on | `pursue_channel_gain = 0`; the review step removed |
| K5 | composite | A coached person against the same person uncoached (`M11.C.17`'s arms) | The coached person asks more, pursues more, and ends the run with a lower share of soothing acts | `coach_ask_gain = coach_pursue_gain = 0` (the perspective route should still leave a smaller difference) |
| K6 | readout | Everyone, over a 40-year run | Report, without a direction: the fraction of people whose fact weight `g` and self-directed share rise over the run, against `M1.A.18d`'s "differentiation is rare" | — (a reported property, like revision 18's saturation check) |

**K4's first half is a premise in part** (pursuing raises the self-directed share by rule). The finding is the second
half — whether functioning follows — and the report keeps them apart.

**Criteria this touches, rerun with the rest:** `M11.C.16` (repertoire collapse: the fact lesson may slow it),
`M11.C.17` (coaching), `M11.C.29` and `M11.C.32` (`I-POSITION`; P3 raises how often one is tried under load, which is
version 2 plan group F's open question), `M11.C.41` and `M11.C.34` (group E: P2 makes choice read anxiety through
`g`). Each changed result is reported beside version 1's, as a new result under a new rule.

## 5. Risks, measured before any criterion is read

- **Drift upward.** Asking, pursuing and practice feed perspective, which feeds asking. Revision 18 saw everyone
  saturate at `self_practice_gain = 0.01`. Measure perspective and self-directed share over a 40-year run before
  reading any criterion (K6); report, do not tune toward a criterion.
- **The fact lesson teaching reactivity.** In some families, distance may really lower one's load over 12 weeks. Then
  higher-level people learn distance. Report it; it is a question about the window and the signal (Q3), not a defect.
- **Two lessons, one choice.** If the two lessons disagree often, choice may oscillate as anxiety rises and falls.
  Measure how often the act chosen at `g` differs from the act chosen at `g = 0`.
- **Cost.** Two values per key doubles the learner's bookkeeping; 12-week windows hold more pending acts. Time a
  40-year run before and after.

## 6. Constants

All new, all `[I]`, all in `config/bowen/constants.md` and logged in `constants_changes.md`: `fact_window` 12,
`fact_others_weight` 0.5, `flood_scale` 40, `ask_threshold` 15, `ask_rate` 0.1, `ask_refractory` 12, `review_step`
0.5, `pursue_wait` 12, `pursue_weeks` 8, `pursue_channel_gain` 0.5, `coach_window` 4, `coach_ask_gain` 1.0, `coach_pursue_gain` 1.0.
Retired if Q4 is answered as recommended: `soothing_review_every`, `soothing_review_penalty`. Each enters `M17.E.1`'s
sweep with `M4.D.6b`'s learning constants.

## 7. Questions for the owner

1. **Q1. What sets how fact based a person's learning is:** `basic_level` (recommended: your words were "level of
   differentiation"), `functional_level`, or basic level plus perspective? Note that basic level rarely moves within a
   run, so with basic level this is close to a fixed trait per person.
2. **Q2. Does anxiety erode the fact lesson, faster at lower level** (recommended, P2)? If yes, `M4.D.3` is amended to
   admit it as a third route.
3. **Q3. What counts as a fact:** the actor's own load over 12 weeks plus the functioning of the people involved
   (recommended), or the actor's load alone? And is 12 weeks the right order (a season)?
4. **Q4. Does the question replace the soothing review** (recommended), or sit beside it?
5. **Q5. Coaching:** any session prompts the question, and a landed contact supports the pursuit (recommended, as
   your statement)? Or only landed contacts for both, which fits the corpus better: there, coaching supplies a way of
   thinking and specifics, and the motivation stays with the person (§3, point 4)?
6. **Q6. Does the self-directed channel learn?** Recommended: not yet. It already learns in the way the corpus
   describes, through perspective built when a held position works (`M1.E.7`, revision 19). Teaching it per-act
   values would amend `M4.D.6d`.
7. **Q7. What does pursuing the answer do:** open the self-directed channel for a few weeks (recommended, P3), or
   something else you have in mind — for example, choosing one specific person to act differently toward?
8. **Q8. Where in version 2:** a stage 3 rule question after the constant sweep (recommended, as
   `docs/version_2_plan.md` orders rule changes), or before it?
9. **Q9. When is the answer pursued:** in the weeks after anxiety eases (recommended, from `L22.1`), or at once, while
   anxiety is still high?

## 8. Adjacent issues found while drafting, not fixed

- `M4.D.6`'s soothing review asks on a calendar, every 12 weeks, whatever the person's anxiety; the owner's statement
  ties asking to high anxiety. P3 addresses it if adopted.
- `M4.D.1a`'s mixing weight reads `functional_level`, while the owner's statement, and `M4.D.1a`'s own text
  ("a function of differentiation"), speak of level of differentiation. Q1 decides which level P2 reads; whether the
  mixing weight itself should read basic level is a separate question, not raised here.
