# Proposal — social cues and the channels a signal travels on

**Status: DRAFT 2026-11-07, for owner review.** No part of it has been implemented. Q5 was answered by the owner the
same day (§5, §8); the rest is open. No requirement in the spec changed,
no file under `src/bowen/` or `config/bowen/` changed, and the engine hash did not move. The owner asked on
2026-11-07 (1) whether Kerr's four social cues can be a structure in the model's appraisal and (2) whether a
message can be sent as "[words], [tone], [expression]". This document answers both, maps the answer onto the
current rules, proposes requirements `SC-1` to `SC-12` with a criterion for each, and ends with the questions the
owner decides. The IDs `SC-n` are this proposal's; each names the spec place it would go if approved.

Grades follow `model_explainer.md`'s legend: evidence `[T]` textual, `[M]` mechanism, `[D]` direction, `[I]`
invented, `[X]` contested; attribution `[B]` Bowen, `[K]` Kerr, `[K-ext]` Kerr's extension. Kerr's 2019 book and
Kerr and Bowen 1988 are written sources, so short quotation is allowed. Only `ch18` below is Bowen's own text.

---

## 1. What the sources say

### 1.1 The four cues (Kerr, 2019)

Kerr states the same list in five places (`_KS_PASS2.md` line 100 counts them):

| Entry | What it adds | File, lines |
|---|---|---|
| `KS01.7` | "sensitivities to approval, attention, expectations, and distress"; "the less 'self' people have, the more they are governed by reactions to such **real or imagined** cues" | `kerr_book/ks01.md` 59–63 |
| `KS07.6` | reactivity to "real and imagined **verbal and nonverbal** cues perceived in the other related to attention, approval, expectations, and distress", which "varies with basic level"; an external stressor (work) becomes internal when the partner feels "inadequately attended to" | `kerr_book/ks07.md` 95–109 |
| `KS09.4` | the questions Kerr reports Bowen asking him in supervision in 1969 ("What do you think of me? Do you accept me? What do you want me to do?"), which Kerr maps to three of the cues; people with more self have "less intense reactions, **more accurate perceptions** of the answers" | `kerr_book/ks09.md` 69–87 |
| `KS10.1` | each cue is **bipolar**: "approval or disapproval, attention or lack thereof, expectations met or unmet, and upset or equanimity in the other"; inadequate attention triggers "rejection, panic, and emotional isolation"; "uncertainty about if and when such stimuli will occur intensifies the process"; the list is open ("and certainly other cues as well", and through `KS09.1` "It is not a complete list") | `kerr_book/ks10.md` 15–35 |
| `KS23.1` | each member's reactivity to the four "mediates the escalation"; "the strength of the reactivity and ability to self-regulate vary with the basic levels of differentiation" | `kerr_book/ks23.md` 19–54 |

Every row is attribution `[K]`. `KS09.4`'s three questions are Kerr's report of what Bowen said to him, so they
are `[K]` as well and are not cited as Bowen's text.

**What the proposal keeps from these sources:**

1. The cues are **sensitivities to threats**: they are where a person reads a threat, not things a person pursues.
   Nothing in the five entries makes approval or attention a goal.
2. A cue may be **real or imagined** (`KS01.7`, `KS07.6`): it can be perceived without having been sent.
3. Differentiation changes **two** things: how accurately a cue is perceived, and how intense the reaction is
   (`KS09.4`). Both change by degree. "Accurately" here means the meaning given to a signal, not whether a signal
   is noticed: a less differentiated person is *more* sensitive to a signal being sent (the owner, 2026-11-07, Q5;
   §5). Reactivity is never removed: the sensitivities "are part of all people"
   (`KS01.7`).
4. The list is four confirmed and **open**.

### 1.2 The channels (Kerr and Bowen, 1988; Bowen)

- `FE03.3` (`family_evaluation/fe03.md` 114–145), `[K]` `[T]` `[M]`: "The signals monitored are primarily auditory
  and visual ones and include body postures, tone of voice, facial expressions, and actual word content." Signals
  read as too little involvement trigger action to restore attachment; signals read as too much trigger action to
  restore separation. The feelings are "crowded, trapped, controlled, smothered, or absorbed" on one side and
  "isolated, unsupported, unloved, or rejected" on the other, from "the perception (or misperception)". This is the
  two-sided loop `M4.C.1` already implements.
- `ch18.md` line 42, `[B]`: in Bowen's clinical profile the impaired child could "read" the mother's "facial
  expressions, tone of voice, body posture, and footsteps", and the mother could read the daughter's equally well.
  This is Bowen's own text and the only Bowen source in this proposal.
- `KS17.3` (`kerr_book/ks17.md` 60–72), `[K]` `[T]` `[M]`: what he perceived as a tense expression on his wife's face
  "reflected his own facial expression and tone of voice, which in turn were intensified by what he saw in her".
  This is `M4.C.6`'s self-attribution, and `_LEDGER.md`'s `KS-C.9` (around line 2255) records it.
- `KS08.4` (`kerr_book/ks08.md` 58–84), `[K]` `[T]` `[M]`: the mother's phone message. "The anxiety in her voice and
  her message amounted to an expectation that the daughter call her." The changed message had near-identical words,
  a "lighthearted delivery", and was "devoid of any expectation that the daughter respond"; it "erased a
  decades-old emotional cue". Kerr's guard: without emotional neutrality the daughter "could have easily heard her
  comment as sarcastic. People sense the difference between neutrality and same-old same-old."
- `KS08.5` (`kerr_book/ks08.md` 86–88), `[K]` `[T]`: the second part of the objectivity sequence is "recognizing the
  verbal and nonverbal cues in others that trigger the anxiety".

### 1.3 The owner's file, checked

`/Users/davemini2/Downloads/social_cues.md` is a generated summary of six Kerr passages, with its bibliographic
details marked unverified. Each quotation was searched for in the extraction files:

| Quotation in the file | Found in the extraction files | Used here |
|---|---|---|
| "the primary social cues… approval, attention, expectations, and distress"; "real or imagined cues" | yes, `KS01.7` | yes |
| "certainly other cues as well" | yes, `KS10.1` | yes |
| "verbal and nonverbal cues" (internal stressors) | yes, `KS07.6`; also `KS08.5` | yes |
| "facial expressions, tone of voice, and other nonverbal cues. It is not just what they say" | **no** | no |
| "their perceptions tend to be more accurate and reactions less intense. This is a quantitative and not qualitative difference" | **no** | no; `KS09.4`'s verified wording carries the same point |
| "question the accuracy of those perceptions of the emotionally competent stimuli" | **no** | no |
| "a new way of mentally processing potentially anxiety-provoking social cues from Mother" | **no** | no |
| "Many invitations come our way each day to lose 'self' by becoming part of a triangle" | **no** | no |

The unfound quotations may well be in Kerr's book; they are not in the corpus extraction, so nothing here rests on
them. If the owner wants them used, the step is an extraction pass on those chapters, not a citation from the
summary.

---

## 2. What the model has now

| Rule | What it does | Where the cues and channels touch it |
|---|---|---|
| `M4.C.1` | two given drives, the same for everyone and never learned: too little contact and too much impingement; every move kind declares a contact and an impingement component in `config/bowen/event_kinds.md` (`[I]`) | the four cues would decompose these components |
| `M4.C.1a` | level sets the steepness and the width of the tolerated band | the reaction-intensity half of differentiation already lives here |
| `M4.C.1c` | absence builds the too-little side over time | "imagined" inattention when nothing is sent |
| `M4.C.2` | gain above threshold: content defended against | unchanged |
| `M4.C.5` | perception-side readout of inward impingement: reading the other's event as critical is the evidence | the disapproval cue, perception-side |
| `M4.C.6` | an agent with `systems_perspective` attributes part of a received event to its own prior emission | `KS17.3`'s expression and tone, by channel |
| `M4.C.9` | a witness appraises from its own position | witnesses read channels too |
| `M1.F.1`–`M1.F.5` | the Event's fields, including `fidelity`, which degrades per private hop | a new field holds the channels |
| `M1.A.9a` | `outside_ness` has two axes, outward and inward impingement | `KS08.4`'s embedded expectation is the outward axis |
| `M4.D.1a` | an automatic and a self-directed channel of selection | unchanged; these are channels of *choice*, not of *signal* (§6 keeps the two words apart) |
| `M1.A.10`, `M6.I.3` | `life_energy`, zero-sum between relationship-seeking and goal-directed activity | attention is finite (`SC-9`) |
| `M7.C.1k` | `FIND_WORK`, a goal-directed act, not a move | the life goal draws on the same budget |
| `M9.2`, `M9.7` | beliefs may be false, are orthogonal to emotional function, and drive appraisal | expectations are beliefs (`SC-10`) |
| `M7.E.1d` | projection can start with no signal from the child: a self-fulfilling prophecy | projection runs through expectation and distress cues |
| `M1.A.13a`, `M1.A.14b` | no positional attribute is read from a role label or birth order | roles must emerge (§4) |
| `M10.A.1` | reactivity, the `life_energy` ratio and others are derived, not parameterised | accuracy and gain are derived from level |

`M4.C.1` has no cue structure: a move kind carries one contact and one impingement number, and every reading of it
is the same apart from `conductance`, `route`, `source_position` and `fidelity`. `KS01.7`'s model note already
named this as an unbuilt candidate decomposition.

---

## 3. Two axes or four channels — the recommendation

**Recommendation: the four cues decompose `M4.C.1`'s two axes. They are four readings of a received signal, not
four drives. The two drives stay as they are, so the premise of `M4.C.1` does not change.**

The corpus basis:

1. `FE03.3` says every monitored signal, on every channel, is interpreted as too little or too much involvement,
   and that the correction follows the side. The controller has two error signs. The cues are what is read; the
   two sides are what the reading is about.
2. `KS03.2` and `KS06.3` (already `M4.C.1`'s sources) name threats to contact and to distance as "the fundamental
   triggers of anxiety". Kerr does not call the four cues fundamental; he calls them "the most common threats
   people experience" (`KS10.1`).
3. Kerr says the list of cues is open (`KS10.1`, `KS09.1`). An open list cannot be the set of given drives without
   making the set of drives open. As a set of readings it can grow without touching the premise.
4. Making each cue its own drive with its own set point (the reading `KS10.1`'s model note proposed: "`M4.C.1`
   should be rebuilt on this") would add four given drives. That is a premise change, and it would let a rule write
   a wish for approval into every agent. The sources say people are *sensitive to threats* to approval, not that
   they pursue it (§1.1, point 1).

How each cue maps onto the two axes. Each cue is a signed reading in [−1, 1] (threat pole negative, the other pole
positive, following `KS10.1`'s pairs). Each maps onto a change in felt contact and a change in felt impingement
through two weights per cue, held in config and graded `[I]`, in the same way a move kind holds its two components
now. The signs below are directions from the sources; the magnitudes are invented.

| Cue | Threat pole | Moves felt contact | Moves felt impingement | Basis |
|---|---|---|---|---|
| **attention** | lack of attention | down (too little) | — | `KS10.1`: inadequate attention triggers "rejection, panic, and emotional isolation", `FE03.3`'s too-little set; `[K]` `[D]` |
| **approval** | disapproval | down, smaller | up (inward impingement) | `M4.C.5` reads criticism as inward impingement (`KB05`); `FE03.3`'s too-little set includes feeling unloved and rejected; both components, as `CONFLICT` carries both; `[K]` `[D]`, the split `[I]` (Q2) |
| **expectations** | (a) an expectation placed on me; (b) my expectation of the other unmet | (a) —; (b) down | (a) up; (b) — | (a) `KS08.4`: the expectation in the voice is the outward impingement of the sender (`M1.A.9a`); (b) `KS08.4`: the daughter's silence raised the mother's anxiety; `KS10.1`'s "met or unmet"; `[K]` `[D]` |
| **distress** | upset in the other | — | up (a pull to fix it) | `KS10.1`'s "upset or equanimity in the other"; `KS23.1`'s "one stressed member can be sufficient"; the side is `[X]`: the corpus does not say which side another's distress acts on (Q3) |

The owner's sketch (disapproval and inattention as too little; expectation pressure and another's distress as too
much) agrees with this table except for disapproval, which the current spec already reads as inward impingement
(`M4.C.5`). The table keeps `M4.C.5` and gives disapproval a smaller contact component as well.

`KS10.1`'s **uncertainty** term (variance in whether a cue will come intensifies the reaction) is a separate
mechanism. It is left out of this proposal and raised as Q6.

---

## 4. Roles: what may be given and what must emerge

Revision 11's rule (`model_explainer.md` §19): patterns emerge and are not written. An assigned role such as
caretaker or scapegoat writes the outcome, and `M1.A.14b` already forbids reading any positional attribute from a
role label or birth order. The owner's framing — each person wants attention, approval, their expectations met and
help in distress, and others want the same from them, while each pursues their own life goals and keeps tension
down — fits the model if it is split as follows.

**May be given (set):**

- **The sensitivities themselves**, the same for every person (`KS01.7`: "part of all people"). They are readings,
  not wants (§3).
- **Needs and dependency that follow from age.** A young child's capacity to supply attention or help is small and
  its need for a caretaker's attention is large. This is a property of age, not of a role. The corpus source for
  its form has not been checked in this pass, so it is graded `[I]` until a ledger check finds one (Q7).
- **Life goals, through `life_energy`.** Goal-directed activity (`M1.A.10`, `M6.I.3`, `FIND_WORK` in `M7.C.1k`)
  draws on the same budget as relationship-seeking. The model needs no goal objects to express "pursues their own
  life goals": the share of energy that goes to goals is derived from level (`M10.A.1`). `KS07.6` supplies the
  coupling: one partner's preoccupation with work leaves the other "feeling inadequately attended to". So a life
  goal is also an attention cue others read (`SC-9`).
- **The given drive to keep tension down** is `M4.C.1`'s two drives; nothing new.

**Must emerge:**

- **Who comes to supply attention, approval and help to whom.** Supplying is a move (`STAY-IN-CONTACT`,
  `OVERFUNCTION`, `PURSUE`), selected by the existing policy and reinforced by felt relief (`M4.D.6`). The person
  who finds that helping a parent relieves their own tension helps more; that is learned, not assigned.
- **Expectations.** An expectation is a belief one person holds about another (`M9`), may be false (`M9.2`), and
  drives appraisal (`M9.7`). A parent who believes a child will help sends expectation cues in tone; the child reads
  them; helping relieves; the belief is confirmed. That is `M7.E.1d`'s self-fulfilling prophecy in the other
  direction, and it is how a role can **form without being assigned** (`SC-10`).
- **The role as a readout.** "Caretaker" or "the one who worries" is computed after the run from who supplied what
  to whom (`SC-11`), never stored on a person and never read by the policy.

---

## 5. Channels on an Event

**Recommendation: an Event carries four channels — words, tone, face, posture — each holding a cue vector. The
answer to the owner's second question is yes, inside the engine: an event is "[words], [tone], [expression],
[posture]" as four cue vectors, and a run log can render it as text for a human reader. For the LLM agents see §7.**

1. **Words are chosen.** The word channel's cue vector comes from the move kind, as the contact and impingement
   components do now (config, `[I]`).
2. **Tone, face and posture leak the sender's acute anxiety.** Each nonverbal channel carries the word channel's
   cue plus a leak that grows with the sender's acute anxiety (`M1.A.8`), on the threat poles: mostly distress and
   expectation, `KS08.4`'s "anxiety in her voice… amounted to an expectation". The sender does not choose the leak.
   A more differentiated sender leaks less only because their acute anxiety is lower, so no separate level term is
   added (`M10.A.1`). Leak sizes `[I]`.
3. **Receivers weight the nonverbal channels more, and read them more negatively, as their own acute anxiety
   rises.** This is the owner's framing. The corpus supports the general direction — the less self, the more a
   person is "governed by reactions to such real or imagined cues" (`KS01.7`) — but no extraction entry says that
   anxiety shifts weight from words to the nonverbal channels. Graded `[I]` for the direction as well as the size
   (Q4).
4. **A cue can be perceived without having been sent.** The perceived cue is the sent cue plus a perception error.
   With point 5's split, the error is on **meaning**, not on presence: what is imagined is a threatening meaning given
   to a signal that is there (a neutral face read as disapproval, a tired voice read as an expectation), or to an
   absence (`M4.C.1c`). Claude's reading of the owner's Q5 answer, pending the owner's review.
   The error has a bias toward each cue's threat pole that grows with the receiver's acute anxiety and with what the
   receiver believes about the sender (`M9.7`), and a spread. So a neutral event can be read as disapproval, and a
   receiver who believes the sender disapproves reads disapproval more often (`FE03.3` "perception (or
   misperception)"; `KS01.7`, `KS07.6` "real or imagined").
5. **Differentiation changes three things, by degree, and removes none of them** (the owner's answer to Q5,
   2026-11-07):
   - **Sensitivity** — how readily a signal is picked up — **rises as level falls**. It is accurate about one thing:
     that a signal is being sent. A less differentiated receiver notices a smaller leak, sooner.
   - **Meaning accuracy** — whether the meaning given to the signal is right — **falls as level falls**. The error's
     spread and its bias toward the threat pole shrink as `functional_level` rises (`KS09.4`).
   - **Reaction intensity** falls as level rises; the steepness already does (`M4.C.1a`).

   None reaches zero at any level (`KS01.7`; `SC-7`). So at low level the receiver is right that something is being
   sent and often wrong about what it says; at high level the receiver may miss a small leak and is more often right
   about the ones it notices.
6. **Self-attribution by channel.** An agent with `systems_perspective` attributes the nonverbal part of what it
   receives on a tie in proportion to its own recent nonverbal leak on that tie (`KS17.3`). This sharpens `M4.C.6`,
   which now attributes "part of a received event" without saying which part.
7. **Fidelity and witnesses.** A private hop (`M1.F.4`) carries the words and loses the nonverbal channels: a
   reported message has no tone. A witness (`M4.C.9`) reads all four channels from its own position. Both follow
   from `FE03.3`'s list being what is "monitored" in presence; `[I]` in form (Q9).

**A contradiction resolved (Q5, the owner, 2026-11-07).** Bowen's `ch18` says the impaired child "read" the
mother's nonverbal signals and the mother read hers "equally well". Kerr's `KS09.4` says lower self means less
accurate perception. Both hold once **sensitivity** and **meaning accuracy** are two terms, as point 5 states: the
fused pair detects each other's leak strongly — that part of the reading is right — and gives it meaning through
anxiety and belief. The owner's answer: lower differentiation means more sensitivity to a signal, not more accurate
reading of it; the sensitivity is accurate that a signal is being sent, not about what it means. Graded `[D]` as a
direction from the two sources together; every size `[I]`. Point 3 (the anxiety weighting of the nonverbal channels)
is separate and still open as Q4.

---

## 6. The proposed requirements

Every requirement below moves the engine hash when built, because each changes code under `src/bowen/` or
`config/bowen/` (`config/bowen/` prose moves it too). Writing this proposal moves nothing. The classes are `M11.5`'s:
a **premise** is produced by the rule that states it and confirms the code, never a finding; a **composite** needs
learning or more than one rule to produce; a **check** is structural. Each criterion is a direction of difference
between two arms, to be proved failing by a named mutant before it counts.

To keep the word "channel" clear: `M4.D.1a`'s two channels are channels of *selection*. This proposal calls the four
carriers *signal channels* in requirement text.

| ID | Requirement (spec place) | Grade and source | Criterion (two arms, direction) | Class |
|---|---|---|---|---|
| **SC-1** | An event received on a tie is read as a signed **cue vector** over attention, approval, expectations, distress; each cue moves felt contact and felt impingement through two config weights; the two drives of `M4.C.1` are unchanged and remain the only drives. (new `M4.C.1e`) | list `[T]` `[K]` (`KS01.7`, `KS07.6`, `KS10.1`, `KS23.1`); directions `[D]` (`FE03.3`, `KS10.1`); weights `[I]` | Same event, one arm read as disapproving, one as neutral: the receiver's felt inward impingement higher in the first. Mutant: approval's impingement weight zeroed. | premise |
| **SC-2** | **Reduction.** With the four signal channels equal, no leak and no perception error, each move kind's contact and impingement equal the values `event_kinds.md` holds now. The feature is built behind a switch, off by default. (new `M4.C.1f`) | `[I]`, an engineering rule | Switch off against the current engine: the golden fingerprints hold. Mutant: one cue weight perturbed. | check |
| **SC-3** | An Event carries four **signal channels** — words, tone, face, posture — each a cue vector. (amends `M1.F.1`) | `[T]` `[K]` `FE03.3`; `[B]` `ch18` (footsteps folded into posture) | Structural: an Event built without the field is refused. | check |
| **SC-4** | **Words are chosen; tone, face and posture leak** the sender's acute anxiety onto the threat poles (mainly expectation and distress). (new `M4.C.13`, beside `M4.C.10`, under which contact with a calmer sender already lowers the receiver's anxiety; the leak is the channel that difference would travel on) | mechanism `[T]` `[K]` `KS08.4`; sizes `[I]` | `KS08.4`'s two arms: same move kind, same words, the sender's acute anxiety held high or low by script. (i) Receiver's felt impingement from the event higher in the high arm — premise. (ii) Over the following weeks, the receiver's voluntary contact toward the sender higher in the low arm — composite. Mutant: leak set to zero. | (i) premise, (ii) composite |
| **SC-5** | The receiver's **weight on the nonverbal channels**, and its negative reading of them, rise with its own acute anxiety. (new `M4.C.11`) | `[I]`, direction and size (§5 point 3) | Same event, receiver's acute anxiety high or low: the nonverbal share of the felt change higher in the high arm. Mutant: weight held constant. | premise |
| **SC-6** | **Real or imagined.** The perceived cue is the sent cue plus an error biased toward the threat pole by the receiver's acute anxiety and its belief about the sender (`M9.7`); a cue can be perceived that was not sent. (new `M4.C.12`; extends `M4.C.5` to the four cues) | `[T]` `[K]` `KS01.7`, `KS07.6`, `FE03.3`; form `[I]` | Every sender scripted neutral on every cue. (i) Receiver at low level against high level: perceived disapproval higher at low level — premise. (ii) Over the run, the low-level receiver's share of `DISTANCE` and `CONFLICT` on those ties higher — composite. Mutant: bias zeroed. | (i) premise, (ii) composite |
| **SC-7** | **Differentiation by degree, on three terms.** As `functional_level` falls, **sensitivity** to a signal's presence rises, while **meaning error** (spread and threat bias) and **reaction steepness** rise too; all three derived from level (`M10.A.1`), none reaching zero at any level. Sensitivity is about presence only: it never adds meaning. (amends `M4.C.1a`; new `M4.C.14`) | meaning error and steepness `[T]` `[K]` `KS09.4`, `KS23.1`, `KS01.7`; sensitivity `[D]` from `ch18` `[B]` with `KS09.4`, the owner's Q5 answer; sizes `[I]` | Check: at the top of the level range, sensitivity, error and steepness are above zero. Arms at two levels, the same small leak sent: (i) the lower-level receiver detects it more often — premise; (ii) the same leak with a neutral meaning: the lower-level receiver reads a threatening meaning more often — premise. Mutants: sensitivity not scaled by level; error not scaled by level. | check + premise |
| **SC-8** | **Self-attribution by channel.** With `systems_perspective`, the nonverbal part of what is received on a tie is attributed in proportion to the agent's own recent nonverbal leak on that tie. (amends `M4.C.6`) | `[T]` `[M]` `[K]` `KS17.3`; share `[I]` | Two anxious agents on one tie; one arm gives one of them perspective: the tie's escalation (both members' acute anxiety over the exchange) lower in that arm. Mutant: attribution reads the words channel instead. | composite |
| **SC-9** | **Attention is finite.** The attention cue a person can send per week is bounded by the relationship share of its `life_energy`; goal-directed activity and an external stressor reduce what others perceive as attention. (amends `M1.A.10`, `M6.I.3`) | `[T]` `[K]` `KS07.6`; form `[I]` | An external work stressor on one spouse against none: the other spouse's too-little deviation on the tie higher, and their `PURSUE` or `CONFLICT` rate on it higher. Mutant: attention not bounded by the budget. | composite |
| **SC-10** | **Expectations are beliefs.** Each person holds, per tie, a belief about what the other expects of them and what it expects of the other; both are updated from perceived expectation cues, may be false (`M9.2`), and feed appraisal (`M9.7`); an unmet held expectation reads on the too-little side. A parent's belief about a child is sent as expectation and distress leak (`M7.E.1d`). (new `M9.9`) | `[T]` `[K]` `KS08.4`, `KS10.1`; `M7.E.1d`'s sources; form `[I]` | A parent's starting belief about one child set false-negative against neutral, the child's own state equal: that child's later `functional_level` lower and its share of supplying moves toward the parent different. Mutant: the belief not sent as leak. | composite |
| **SC-11** | **No assigned roles.** No config field or person attribute names a role (caretaker, scapegoat, helper). A role readout is computed after the run from who supplied attention, approval and help to whom, and the policy never reads it. (amends `M1.A.14b`; new `M11.G` readout) | the rule of `model_explainer.md` §19; `M1.A.14b` | Check: no such field exists (a lint, as `test_repo_layout.py` does for the grid engine). Composite: across seeds, which child becomes the main supplier to a parent varies and follows learned relief, not birth order. Mutant: supply weighted by birth order. | check + composite |
| **SC-12** | **Dependency by age.** A young child's capacity to supply attention and help is small and its weight on a caretaker's attention cue is large; both follow from age, not from a role. (new `M1.A` requirement; place to be decided) | `[I]` until a source is found (Q7) | Two arms differing only in the child's age at a parent's stressor: the young child's acute anxiety from the parent's leak higher. Mutant: age term removed. | premise |

**What none of these write.** No requirement says who becomes the caretaker, which child the projection falls on,
or that a tie ends in conflict. The premises (`SC-1`, `SC-4`(i), `SC-5`, `SC-6`(i), `SC-7`, `SC-12`) confirm that the
reading works as stated and are never reported as findings. The findings, if any, come from the composites.

**Order if approved.** `SC-2` and `SC-3` first (structure and the reduction check, switch off, golden fingerprints
holding), then `SC-1`, `SC-4`, `SC-6`, `SC-7`, then the rest. Turning the switch on by default is a separate owner
decision, because it moves every record.

---

## 7. The LLM experiment

T-183's main phase (2026-11-07, `docs/llm_agent_experiment_report.md`, "Label-following") found that A3, whose
plain-word act descriptions were shuffled onto other acts' mechanics, followed the words over the mechanics in 5 of
5 seeds. If tone and expression were shown to the agents as words ("she frowns, her voice is tight"), the agents
would answer the words' meaning, which is what A3 showed and what the experiment's A1 arm (opaque tokens) was built
to avoid.

**Recommendation: keep the cues engine-only.** The signal channels change the felt tension the engine computes,
and the agents already see that tension and its change (A1, A2). They then react to the cues the way the learning
agents do: through what they feel, without being told what the cue means. If the owner later wants the agents to
see channels, the form that keeps A1's design is opaque, per-seed permuted channel tokens with numbers ("C2: −0.4"),
never words, and a plain-word rendering only as a contamination arm, as A3 is now.

---

## 8. Questions for the owner

| # | Question | Recommendation |
|---|---|---|
| **Q1** | Do the four cues decompose `M4.C.1`'s two axes, or become four drives of their own? | **Decompose** (§3). Four drives is a premise change, puts a wish for approval into every agent, and turns an open list into the set of drives. |
| **Q2** | Disapproval: inward impingement only (as `M4.C.5` now), or impingement plus a smaller too-little component? | **Both**, impingement larger, as `CONFLICT` carries both. `FE03.3`'s too-little set includes feeling unloved and rejected. |
| **Q3** | Another's distress: which side? | **Impingement** (a pull to fix it), graded `[X]`; the corpus does not say. Alternative: no side of its own, with distress acting only through the sender's leak raising the receiver's anxiety. |
| **Q4** | Keep `SC-5` (nonverbal weighted more as anxiety rises) though no extraction entry states it? | **Keep, graded `[I]`, as a premise.** Or drop it and let accuracy (`SC-6`) carry the effect; the model would then be fully sourced on this point. |
| **Q5** | Sensitivity and accuracy as two terms, to hold `ch18` and `KS09.4` together? | **Answered by the owner, 2026-11-07:** yes — lower differentiation means more sensitivity to a signal, which is accurate that a signal is being sent but not about what it means. Written into §5 point 5 and `SC-7`; that the imagined part of a cue is meaning, not presence (§5 point 4), is Claude's reading of the answer, pending review. |
| **Q6** | `KS10.1`'s uncertainty term (variance in whether a cue comes intensifies the reaction)? | **Later**, as its own proposal; it is a separate mechanism with its own criterion (`KS09.8`'s intermittent withdrawal). |
| **Q7** | `SC-12`'s age dependency: build at `[I]`, or first look for a source? | **Look first.** A ledger search for age and dependency before building; build at `[I]` only if none is found and the owner wants it. |
| **Q8** | Show the cues to the LLM agents? | **No, engine-only** (§7). |
| **Q9** | Does a private hop drop the nonverbal channels, and does a witness read all four? | **Yes to both**, `[I]` in form. |
| **Q10** | Use the owner's file's unfound quotations? | **No** until an extraction pass on Kerr's chapters 2, 8 and 15 finds them (§1.3). |
| **Q11** | Build at all in version 3, and when? | **Yes, behind a switch off by default**, `SC-2` and `SC-3` first, after T-183's cross-check. Every step moves the engine hash, so it goes in one batch with a single regeneration of the records. |

---

## 9. What this document changed

This document only. It is listed in `docs/SPEC_REVIEW_CHECKLIST.md` and `docs/REVIEWER_READING_LIST.md`. No spec
requirement, code, config or record changed; the engine hash did not move.
