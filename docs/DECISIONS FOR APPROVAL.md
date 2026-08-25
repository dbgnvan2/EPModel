---
tags: [model-bt, decisions]
status: partially answered — A, B, C, D, F2 outstanding
date: 2026-08-23
---

# Decisions for approval

The pipeline is complete: chapters → draft spec → **revision 1** (1979 lectures) → **revision 2** (Kerr–Bowen interviews). This is the list of items needing your review before the documents and the spec are approved and code can start.

Each item states **what changed**, **why**, **what I recommend**, and **what it costs if I'm wrong**. Mark each ✅ approve / ✏️ edit / ❌ reject.

**Nothing here is a code decision.** The spec is `docs/bowen_agent_model_spec_v2.md`, now **256 unique requirement IDs, 0 duplicates, 0 unresolved** (+20 this batch, none removed).

---

## Status — 2026-08-24

| Section | State |
|---|---|
| **A** · Withdrawals (2) | ⬜ **awaiting you** |
| **B** · Terminology (1 decision) | ⬜ **awaiting you** |
| **C** · Framing corrections (3) | ⬜ **awaiting you** |
| **D** · Model additions (11) | ⬜ **awaiting you** |
| **E** · Open items (5) | ✅ all answered |
| **F1** · Validation checklist | ✅ accepted |
| **F2** · Non-code-testable criteria (4) | ⬜ **awaiting you** |
| **G** · Resolution of E1 (G1–G9) | ✅ applied to the spec on your instruction |

**Still needed before step 2 can run: A, B, C, D and F2.** B1 is the only one that is a genuine
either/or; A, C and D are approve-or-edit lists with a recommendation on every line.

---

## A · Withdrawals — things we had wrong and have removed

### A1 · "Eight to ten generations" is no longer a calibration target
**Was:** a `[#]` figure in `L16.3`, listed among usable durations in the explainer.
**Why withdrawn:** `KB11`, verbatim — "I introduced the notion of ten generations… **which I put in not to say that it takes ten generations**… but to convey the notion that it's not just three and just ten, but it **is a multi-generational thing**." Both numbers were rhetorical.
**This is the third figure to fail this way**, after the per-generation decline rate and the annual societal-regression rate.
**Recommend:** approve. Keep the direction — multigenerational, more than three, unquantified.
**Cost if wrong:** we lose a calibration anchor the model never had a defensible use for anyway.

☐ ✅ Approved. 
### A2 · The taboo set is no longer monotone
**Was:** `L10.8` — subjects withdrawn from a tie and "not returned".
**Why corrected:** 1979 Tape 6 — "the **purposeful mention of the taboo subject**, if one can control one's own anxious response, **can desensitize the whole mechanism**."
**Recommend:** approve. Growth is the default, not a law; the reversal is the differentiating move applied to one topic and carries the same gates.

☐ ✅ Approved. 

---

## B · Terminology — one decision I need from you

### B1 · "Emotional" means instinct, not feeling — define or rename?
**The finding, two-source and unambiguous.** `KB02`: the emotional process is "**equivalent to instincts** — birds migrating… the salmon… on a cellular level, the force that guides an amoeba toward a morsel of food", and "**feelings are a superficial awareness of the emotional process**." `KB13`: "people have picked up this term I used, emotional, and **made it synonymous with feeling, which I didn't mean at all**. So I'm partially at fault."

**Why it matters:** the model's core quantity is **sub-affective**. It explains why overt emotionality peaks mid-scale and why a withdrawn person is as involved as an effervescent one.

**Options:**
1. **Define the word at first use** and keep the corpus vocabulary. *(implemented as `M1.A.0`)*
2. **Rename** the model's terms to *instinctual* — clearer to a reader with no Bowen background, but breaks every citation to the corpus and diverges from all four sources.

**Recommend:** option 1, already in place. **Cost if wrong:** a reader assumes anxiety is a feeling-state and builds an affective readout, which is exactly the error the corpus documents.

☐ ✅ Approved  - option 1 

---

## C · Framing — three corrections that reach the published proposal

### C1 · The fractal justification is not Bowen's
**Was:** the proposal defers the societal layer on the grounds that "Bowen is explicitly fractal… the family is the unit cell."
**Three independent statements say otherwise:** "an analogy is **not an extension of theory**"; the triangle in society is "an analogy, but **not a reasonable connection — these things don't connect up**"; society is "**similar to a family. Not the same as, but the same principles apply.**"
**Recommend:** keep the architectural decision, replace the reason with his formulation. *(implemented as `M11.F.2`)*
**Cost if wrong:** we would be claiming his authority for a derivation he explicitly refused.

☐ ✅ Approve recommendation

### C2 · The general-systems objection, stated unsoftened
**The finding:** `KB13` — he rejects **general systems** as a foundation in favour of **natural systems**, because general systems "came out of man's head, along with mathematics". `KB09` — but calls the two "**compatible within limits**". **Both are his.**
**A simulation is a general-systems artifact**, so this is a real named objection to the model's foundation.
**Recommend:** record both statements, collapse neither, state the objection plainly rather than softening it. *(implemented as `M11.F.3`)*

☐ ✅  Approve with this addition - The simulation is not trying to be a non-objective explanation of human functioning. this is the distinction - General Systems Theory is an abstract thought up conceptualization,  Bowen Theory is based on observations of how families function. Just like observing the planent moments to derive how they function.    

### C3 · No output presented as Bowen's authority
**The finding:** `KB09` — his goal was "an **open theoretical system**, where the basis for new knowledge is **research and science rather than anything I said**", and he names discipleship as what closes a system off: "the more people that treat it like that, the more **my theory will perish** as being a dogma."
**Recommend:** approve as a standing framing requirement. *(implemented as `M11.F.1`)* This is also the strongest external justification for the grading discipline this project has used.

☐ ✅ Approved. 
---

## D · Model additions — new requirements needing sign-off

 # | Requirement | Source | Recommend |
 |---|---|---|---|
| **D1** | `differentiation_capacity` — a **family-level** variable: does *any* member retain the ability to take a position? Bowen reserved "schizophrenia" for families where none does, and ran "a **test of treatability**". | `KB08` · K08.2 | ✅ — severity becomes a capacity that predicts treatability, not a symptom label |.    Approved. 

| **D2** | The live-issue gate carries a **family-type** term, and it is a **safety** property. Peace-agree families need an issue raised; reactive families need the opposite. `KB08` records three outcomes of getting it wrong: acute psychosis, a self-blinding, a suicide. | `KB05`/`KB06`/`KB08` | ✅ — the spec previously stated the gate unconditionally |.  Approved.

| **D3** | Spouses paired at matched **`basic_level`**, never functional. | `KB14`, 1979 Tape 3 | ✅ — a construction rule, testable | Approved - but it doesn't have to be EXACTLY matached. + or - one point.

| **D4** | `structural_importance` **derived** from functional-doer position + suddenness, not assigned as a static tier. Role labels excluded — a matriarch outranks a patriarch where she held the position. | Tape 6 · T6.1 | ✅ — better sourced and computable from existing state |  Approved. 

| **D5** | Asserting a differentiated state is **negative evidence** for it. Five independent forms across five interviews. | `_KB_PASS2.md` | ✅ — the readout `outside_ness` never had |   Approved. 

| **D6** | The coach's objective is **to understand, not to help.** Four statements, an experiment (best resident lasted ten hours), and an independent replication by Kerr. | `KB03` · K03.1 | ✅ — best-evidenced coach claim in the project |  Approved. 

| **D7** | Coach detectors are **two-sided** — praise is as much a loss as blame. | `KB07` · K07.1 | ✅ — ours was negative-valence only |.  Approved.

| **D8** | `REDUCE_CUTOFF` has a **floor**; cutoff is "your lifeline". Target is **one** open relationship, not universality. | `KB05`, `KB08`, Tape 6 | ✅ — four sources agree contact is not monotonically good |. Approved

| **D9** | Distance **absorbs anxiety without symptomising** — the half the spec was missing. | `KB04` · K04.1 | ✅ — explains why a family can discharge into distance and read as untroubled |. approved. (the Distance IS the symptom, and since it indicates high intensity, that intensity can lead to other symptoms.)

| **D10** | Fusion is **life-stage dependent**; the infant–caretaker symbiosis is normal, not pathological. | `KB14` · K14.5 | ✅ |  approved. 

| **D11** | On return from removal, re-escalation **exceeds** the original removal disturbance. | Tape 6 · T6.8 | ✅ — changes what `M11.C.11`'s release arm asserts |. approved. 

---

## E · Open items — **all five now answered** (2026-08-24)

Your replies are recorded verbatim. Where an answer added model content, it is queued in
`theory/_STATUS.md` under the step-2 propagation list.

| # | Item | Your answer | Consequence |
|---|---|---|---|
| **E1** | The dials versus Bowen's three societal variables — *mode of thinking*, *differentiation*, *anxiety intensity*. Our three dials drive only the third. | *"Reply in thread."* Worked through in conversation and **§G below is the result**; you then instructed *"do option 1"*, so G4, G6 and G9 were applied to the spec. | **Applied** — commit `44c439f`. `systems_perspective` (`M1.A.18`) is the individual-level representation of the first variable; `M1.D.7a1` still governs the societal layer. |
| **E2** | The marriage-ceremony break — Bowen states it as a fact and says he never worked out why. | *"Once marriage happens individuals' fusion gets stronger. And the couple is ONE SELF so then who gets to decide for the self. The lower the level of differentiation the more this fusion can create more anxiety and reactivity."* | **Resolved — a mechanism where there was none.** Encodes against `M1.A.3`, whose transition at 50 is already *a licence over joint decisions*: the ceremony changes neither person, it reclassifies a large class of decisions into the shared-life-course domain where the transition bites. Queued **S2.5**. |
| **E3** | A ninth concept was proposed in 1980 and never consolidated. | *"Yes implement 8 concepts."* | **Confirmed** — already what the model does. No change. |
| **E4** | Tape 3's post-removal sibling symptom versus Ch04's retraction of the durable target queue. | *"Go with Bowen."* | **Confirmed** — `L01.7` stays withdrawn. Bowen retracted it himself; a later symptom intensifying is not durable position uptake. No change. |
| **E5** | A7, C4 and B4 are single-sourced to the book. | *"Answered in thread."* **All three confirmed**, each with added mechanism. | Verdicts recorded in `theory/Extractions to be human validated.md`. Queued **S2.6** (change-back is caused by the move demanding others function more responsibly; settles over months, `[I]`), **S2.7** (an ally is a **pseudo-self transaction**, not only a peripheral triangle), **S2.8** (societal damping is **two-sided** — positive or negative). |

---

## F · Still needing you, unchanged

### F1 · The validation checklist
`docs/theory/Extractions to be human validated.md` — 23 items, the extraction-versus-source check. **The lectures and interviews have since settled or corroborated 15 of them**, so the list is smaller than when I handed it over. A7, C4 and B4 (E5 above) were the ones that most needed a human eye. **All three are now confirmed by you** (2026-08-24), with verdicts written into that file.

**Your answer: “that’s fine.”** Recorded as acceptance of the checklist's current state. **Stating the residual honestly:** 21 of the 24 verdict boxes in that file remain unticked. Fifteen items now carry *source corroboration* from the lectures or interviews, which is not the same thing as a hand-check of our reading against the chapter. That gap — ~19,000 lines of extraction never re-read against the source — remains the load-bearing unverified claim in this work, and is recorded as such in `theory/_STATUS.md`.
REPLY - that's fine.  

### F2 · Four acceptance criteria that cannot be made code-testable in Phases B–D
`M11.E`, each with a human-review proposal: `M11.C.8` (incidence versus published rates — needs Phase E), `M11.C.9` (sibling position — the effect size is invented, so the test is close to unfalsifiable), `M11.C.11`'s shape claim, and `M5.F.2`'s counterfeit threshold — **the model's most consequential invented constant.**

---

## G · Resolution proposed for E1 — *mode of thinking* as a graded gate

Raised by you on 2026-08-24 and worked through against the corpus. This is the one section here that **adds** to the model rather than correcting it, so it needs approval on its own.

### G1 · The coupling, and which way it runs

Differentiation and systems thinking are not independent, and they are not one derived from the other. They are coupled, and the coupling that matters for the model runs through **whether the differentiating move is comprehensible to the agent at all.**

Both directions are in the corpus:

- **Differentiation → thinking.** `KB07` · K07.3 names undifferentiation as "the **cement, the  hardener**, that fixes" a way of thinking — reduce it and the thinking can change faster.
  `KB07` · K07.2: his own capacity to think systems tracked "**step by step**" the change in the relationship with his mother.
- **Thinking → differentiation.** Never stated as a proposition, but assembled from four places.
  K07.3's first obstacle — "you can't change a way of thinking until you have some notion of a  **new way of thinking**" — means the frame must arrive before anything moves. `KB07` · K07.4  says what a productive coaching contact actually delivered: the awareness that "there was an
  **alternative way to respond**, and also that **I was emotionally caught in it**." `KB07` · K07.1
  makes blame the gauge — systems thinking is impossible until the person can be "**emotionally
  neutral. That is without blaming or praising.**" And `KB02` has him renaming *mental illness* to
  *emotional illness* because "mental illness belongs to a **way of thinking** that sees the
  dysfunction as a product of the brain" — the individual-is-broken frame, identified by him as a
  mode of thinking rather than a mistaken belief.

**Why this matters more than a missing variable.** Without the systems frame, an agent told to differentiate has no model of how it would help, so it executes the move in the only frame it has — a stand taken *against* people. That is precisely what Bowen calls grotesque in interview #1: they
think differentiation is "something you do in an hour a weekend… to somehow separate a self from the family, **to shout them down and let them know you are different**."

So pseudo-differentiation is not a separate error to be modelled. **It is what the differentiating move degrades into when systems perspective is low** — which also joins up with `M5.F.2a` (asserting a differentiated state is negative evidence for it).

**Graded, not a hard gate.** Nothing in the corpus says a person with no systems perspective is incapable of a genuine differentiating move — only that they will not see why it would help and will reach for blame. A hard gate would make the first differentiating move in any family impossible and leave no way in.
G1 is approved. 

### G2 · How and when an agent's systems perspective rises — *not* by learning

The natural architecture is to give agents a discomfort-reduction objective and let them discover that differentiation pays off over a long horizon. 

**That architecture would make the model wrong**,
in a way that would be easy to miss because the runs would look encouraging.

Three reasons from the corpus:

1. **The binders work.** `KB10` · K10.7 is explicit that anxiety-relieving action *relieves the   anxiety* — "cause and effect laws designed to **relieve the anxiety of the moment**, and the   more we do that, **the more we promote the thing we're trying to fix**." The relief is real.
   That is why it is chosen. A learner is not making an error when it picks distance.
2. **The payoff of differentiation is invisible without the frame.** That is G1. An agent cannot   optimise toward a value it cannot perceive, so the long-horizon return is not available to the   learner in the first place.
3. **The moves are instinct-level, not deliberative.** `KB02` · K02.1 and `KB13` · K13.1 define   *emotional* as **instinct**, and `M1.A.0` already forbids modelling the anxieties as   feeling-states. An agent that *selects* a binder by evaluating its expected discomfort reduction
   has imported the individual-deliberative frame the theory rejects — before any parameter is set.

**The consequence is a negative prediction, and it is the strongest validation test in the model:basic level MUST NOT drift upward on its own.** If a population of agents optimising for comfort learns its way to differentiation, the model has contradicted multigenerational transmission, inmwhich levels are roughly conserved and differentiation is rare and hard.

**What does raise it, per the corpus:**

- **Frame supply from an external agent who already has it** — `KB07` · K07.4, and it is rare and low-yield: "**four times at most, five maybe, in two years**" out of six to eight contacts a year.
  What landed was not instruction; it was being shown an alternative *and* being shown one is  caught.
- **Capacity exhaustion — the binders stop working.** `KB10` · K10.6 at societal scale: "**societal  attitudes change when society no longer has an option.**" Pain is necessary and not sufficient:
  without frame supply, exhaustion produces symptom escalation, not insight.
- **One at a time.** `KB12` · K12.1 — only one person in a family can differentiate at a time.

**So the model needs two mechanisms, not one learner over nine moves.** Reinforcement (`M4.D.6`) operates on the **reactive** repertoire, which is what gives a family its characteristic style. The differentiating move sits **outside** the learner, gated by `systems_perspective`, which is raised
only by an exogenous input conditioned on the binders failing.

Mapping your four examples to the current repertoire, for the record: separation and cutoff are `DISTANCE` / `CUTOFF`; an affair is a `TRIANGLE`; substance use is not a move at all but the dysfunction-in-a-spouse symptom channel. All four are on the automatic side.

G2 Commment - refer to discussion on Functional level of Diff. of Self. This can happen more quickly. A frustrated human, by defining themselves in an immediate situation can get relief.  this is "functioning better".  Doing this repeatly (short term) increases "functional level of DOS".   

### G3 · A one-sided test found while working this through

`M4.D.6a` forbids the short-horizon relief proxy and `M4.D.6b` requires a longer declared horizon. 
`M11.C.16` tests **one direction only** — that the short-horizon arm collapses onto `CUTOFF` and `DISTANCE`. It does not test the opposite failure that G2 identifies: a long enough horizon producing **spontaneous universal differentiation**. Lengthening the horizon fixes the collapse and
buys the wrong model if nothing bounds the other end.

G3 - approved, but with the context of Functional Level of DOS which is shorter time horizon.

### G4 · Spec changes this implies — ✅ **APPLIED 2026-08-24** (commit `44c439f`)

| # | Change | Where |
|---|---|---|
| **G4.1** | New per-agent state `systems_perspective`, graded, `[M]` — mechanism sourced, magnitudes invented | `M1.A` |
| **G4.2** | Rises **only** on a landed external-agent contact, conditioned on binder failure; **MUST NOT** be a reinforcement target | `M4.D`, `M1.E` |
| **G4.3** | Falls with acute anxiety — polarity capture is a **loss** in someone who had it (`KB07` · K07.6, `KB12` · K12.7) | `M4.C` |
| **G4.4** | Ceiling coupled to `basic_level` — K07.3's cement | `M10.A.1` derived-not-parameterised list |
| **G4.5** | `I-POSITION` selected at low `systems_perspective` executes as the **assertion form**: raises tie reactivity, counts as negative evidence per `M5.F.2a` | `M5.F` |
| **G4.6** | Blame as the observable readout — the two-sided neutrality gauge (`KB07` · K07.1) pointed at the agent, not only the coach | `M1.E.2` sibling |
| **G4.7** | New acceptance criterion: **no upward drift in `basic_level` in a coach-free arm.** Two arms, identical seeds, differing only in external-agent presence | `M11.C` |
| **G4.8** | `M11.C.16` gains its second direction — the long-horizon arm **MUST NOT** exceed a declared `I-POSITION` ceiling | `M11.C.16` |

### G5 · `basic_level` is not written — it is **inferred** from functional history

Your correction, and it replaces a mechanism the spec currently has. A person works on **functional** level. If that improvement holds up over years **and across dozens of situations**, that is what indicates a rise in **basic** level.

**What the spec has now, and why it is weaker.** `M1.A.4` + `M5.D.7` + `M7.A.1` give `basic_level` a **ratchet** that advances on a *completed differentiating exchange*. That makes basic level a counter of successful moves — one exchange, one increment. It is directly writable, it is
gameable, and it puts the weekend-differentiation grotesque **inside the mechanism**: perform the exchange, collect the ratchet.

**The proposed replacement.** `basic_level` **MUST NOT** be directly writable by any move. It is an **estimator over `functional_level` history**: an elevated mean, sustained over years, with low
variance, across many distinct situations.

**This is already in the spec, stated forwards.** `M1.A.5` requires `functional_level` **variance** to be a *decreasing function of* `basic_level`. High basic level ⇒ functioning that does not swing.
The proposal is to run that same relationship **backwards as the update rule**: sustained low variance at an elevated mean, across varied situations, *is* the evidence of basic level. One
relationship, not two mechanisms.

**It also matches the measurement position.** There is no instrument for basic level (`_EXTERNAL_MEASURES.md`) — what is observable is functioning. An estimator over observed functioning is the honest form, and it puts the model's own epistemic position where the corpus puts it.

**The corpus has the anecdote, and it is the breadth test exactly.** `KB05`, on years of working both parental relationships individually: "I did **pretty good**, but I **still would get caught up in it when I would go home**." Years of sustained work, genuine functional improvement — and it
failed in one situation. The gain had not generalised, so it was not yet basic. `KB07` · K07.2 is the same shape from the other side: the change "parallels **step by step**" work in one primary tie, over years.

**The requirement that makes or breaks it: the situations must be loaded.** A benign decade produces a stable, elevated `functional_level` and demonstrates nothing. If the estimator does not condition on anxiety exposure, **an easy life reads as differentiation.** Two consequences:

- The estimator **MUST** weight situations by the load they imposed, not count them.
- The sample **MUST** include **nodal events**. This is what discriminates the binders. `CUTOFF`   drops acute anxiety immediately with its cost deferred to the next nodal event (`M11.C.4`), so a   cut-off agent shows a durable-looking calm that holds right up until a nodal event arrives.
  Narrow gain, broad-looking until tested.

**What this buys.** The estimator separates binder-relief from differentiation **without a special rule**. Binder gains are situation-specific and collapse under load; differentiating gains hold
across situations. `M5.F.2a` (asserting a differentiated state is negative evidence for it) stops being a bolted-on counterfeit check and becomes an *output*: an assertion produces a functional spike that does not survive the window.

**Honest grading.** "Over years" and "dozens of situations" are **your judgement, not the corpus** — no source states a window or a count. Both **MUST** be declared in config and graded `[I]`, and the model's conclusions **MUST** be shown to be insensitive to their exact values, or the sensitivity reported.

G5 Approved. .

### G6 · Spec changes G5 implies — ✅ **APPLIED 2026-08-24** (commit `44c439f`)

| # | Change | Where |
|---|---|---|
| **G6.1** | `basic_level` **MUST NOT** be directly writable by any move; the ratchet is removed | `M1.A.4`, `M7.A.1` |
| **G6.2** | `basic_level` is derived as an estimator over `functional_level` history — elevated mean, low variance, sustained window, breadth of situations | new, under `M1.A` / `M10.A` |
| **G6.3** | The estimator **MUST** weight situations by imposed load and **MUST** include nodal events; an unloaded window is **not** evidence | new |
| **G6.4** | A completed differentiating exchange (`M5.D.7`) raises `functional_level` and remains the triangle-decrement trigger; it no longer touches `basic_level` | `M5.D.7` |
| **G6.5** | `M7.A.2` (peripheral-system gain transfers to the nuclear family) is **strengthened and now explained**: breadth across systems is precisely what the estimator measures, so transfer is not a special rule | `M7.A.2` |
| **G6.6** | Estimator window and breadth count declared in config, graded `[I]`, with a reported sensitivity analysis | `M10.B` |
| **G6.7** | New acceptance criterion: a `CUTOFF`-heavy arm and an `I-POSITION`-heavy arm reaching the **same** `functional_level` **MUST** diverge in estimated `basic_level` once nodal events enter the window | `M11.C` |

### G7 · Agency is graded by differentiation — and it is already in the model as **solid self**

A correction to how G2 was put. "The moves are instinct, not deliberation" is too strong. Agency is not absent; **it is a function of differentiation.** The higher the level, the less the individual
is guided and governed by what the system wants — which is close to the definition of the concept rather than a consequence of it.

**The corrected architecture is two channels with different objectives, not one channel:**

| | Automatic channel | Self-directed channel |
|---|---|---|
| Driven by | the relationship system | the person |
| Objective | discharge anxiety **now** | hold a position, **accepting** discomfort |
| Repertoire | the seven reactive moves | `I-POSITION`, `STAY-IN-CONTACT` |
| Learns? | yes — `M4.D.6`, this is family style | **no** |

The mixing weight between them **MUST** be a function of differentiation. At low level the person is nearly all automatic; as level rises a real self-directed channel opens.

**This preserves G2's conclusion and fixes its reasoning.** Differentiation is not learnable from a discomfort-reduction objective — not because agents lack agency, but because **the two channels optimise different things.** Discomfort reduction is the automatic channel's objective. The
self-directed channel's objective is a position held *through* discomfort. No amount of horizon lengthening moves an agent from one to the other, because the target is not in the first channel's objective at all.

IMPORTANT ADDITION _ the whole mechanism of Functional level of DOS in the above process. 

**No new variable is needed for agency.** The model already carries it: **pseudo-self is the portion of the person the relationship system can move, and solid self is the portion it cannot.**
`M6.I.4` already makes pseudo-self the conserved, negotiable quantity that transfers between people in a fused relationship, and 1979's definition of pseudo-self as "**negotiable**" is exactly "governed by what the system wants". So the agency fraction is the **solid-self fraction**, which is
present, sourced, and derived rather than added. This satisfies K10.9's parsimony constraint instead of costing against it.

IMPORTANT _ increase in in Functional level of DOS means pseudo self is lower. 

Three distinct things, then, and only one of them is new:

| Capacity | What it is | Status in the model |
|---|---|---|
| **Seeing** — systems perspective | whether the differentiating move is comprehensible | **new** (G4.1) |
| **Acting** — agency | whether the person can act against the system's pull | **already there** as solid-self fraction |
| **Aiming** — the position itself | what a differentiated move is *for* | **G8 below** |

`KB07` · K07.4 has the first two as the two halves of what a productive contact delivered: the awareness that "there was an **alternative way to respond**" (seeing) "**and also that I was emotionally caught in it**" (the limit on acting).

G7  Approved with the IMportant additions.


### G8 · Kerr's two-axis definition — and the counterfeit detector should be two-dimensional

> **Be for self without being selfish** (non-impingement — do not impinge on the other).
> **Be for other without being selfless** (do not let the other impinge on you).
> — Kerr, supplied by you 2026-08-24

This defines the differentiated position as a **conjunction of two negatives**, on two independent axes. It is the objective function for the self-directed channel in G7.

**The corpus states the same structure twice, in the model's own terms.** `KB12` · K12.3: "the
**low-level self can have an I-position**. That is a **selfish, dogmatic, forceful** kind of an I-position… a high-level differentiated self is **neither offensive nor defensive** to the other. So you can tell pretty much the **level of functioning of an I-position from the way they do it**."
And `KB07` · K07.1's neutrality gauge is the same two-sidedness on the coach: **blaming or praising are both losses.**

**Why this matters for the spec.** `M5.F.1` makes a move's effect turn on a **single hidden scalar** (`outside_ness`). One scalar **conflates the two counterfeits**, and they are different failures that need different corrections:

| Axis | Failure | How it looks | Corpus |
|---|---|---|---|
| Impinges outward | **selfish / offensive** | assertive, dogmatic, forceful — the "shout them down and let them know you are different" grotesque | `KB12` · K12.3, `_KERR_INTERVIEWS.md` Corr. 2 |
| Impinged inward | **selfless / defensive** | compliant, accommodating, reads the other as critical | `KB12` · K12.3, `KB05` — "if this person is **hearing** mother as being critical, then they probably are being critical, **defensive**" |

A move scoring low on one axis and high on the other is a counterfeit **either way**, and a single
scalar cannot say which. The compliant peace-keeper and the forceful declarer currently land at the
same `outside_ness` and would receive the same correction, which is wrong in opposite directions.

**Note the readout in the `KB05` line: the defensive pole is detectable in *perception*, before any move is emitted** — hearing the other as critical is itself the evidence. That is a free observable the model can compute in the appraisal step rather than the act step.

G8. Approved. 

### G9 · Spec changes G7–G8 imply — ✅ **APPLIED 2026-08-24** (commit `44c439f`)

| # | Change | Where |
|---|---|---|
| **G9.1** | Two move-selection channels — automatic and self-directed — with the mixing weight a function of differentiation; only the automatic channel is subject to reinforcement | `M4.D.1`, `M4.D.6` |
| **G9.2** | Agency **MUST** be derived as the solid-self fraction; **MUST NOT** be an independent parameter | `M10.A.1` derived list |
| **G9.3** | `outside_ness` splits into **two axes** — outward impingement and inward impingement — and the differentiated position is **both low** | `M1.A.9`, `M5.F.1` |
| **G9.4** | The counterfeit detector **MUST** report *which* axis failed; a single scalar score is a failing implementation | `M5.F.2`, `M5.F.2a` |
| **G9.5** | Inward impingement gets a **perception-side** readout (hearing the other as critical), computed at appraisal, not only at act | `M4.C` |
| **G9.6** | New acceptance criterion: a forceful-declarer arm and a compliant-accommodator arm **MUST** be distinguishable by axis, at equal overall counterfeit magnitude | `M11.C` |

☑ **applied on your instruction (“do option 1”).** Retrospective edits still welcome — say so and I will revise the spec.

☑ **applied on your instruction (“do option 1”).** Retrospective edits still welcome — say so and I will revise the spec.

☑ **applied on your instruction (“do option 1”).** Note: G1–G4 introduces `systems_perspective`, the only genuinely **new** state variable in this batch rather than a repair of an existing one — the one most worth a second look.

---

## What happens after approval

1. Approved documents and spec are frozen at v2.0.
2. Implementation **plan** — criteria mapped to files and order, per the project's planning rule.
3. Plan approved.
4. Code, starting at Phase B.

**No code before step 3.**
