---
tags: [model-bt, decisions]
status: awaiting review
date: 2026-08-23
---

# Decisions for approval

The pipeline is complete: chapters → draft spec → **revision 1** (1979 lectures) → **revision 2** (Kerr–Bowen interviews). This is the list of items needing your review before the documents and the spec are approved and code can start.

Each item states **what changed**, **why**, **what I recommend**, and **what it costs if I'm wrong**. Mark each ✅ approve / ✏️ edit / ❌ reject.

**Nothing here is a code decision.** The spec is `docs/bowen_agent_model_spec_v2.md`, now **276 requirement IDs, 0 unresolved**.

---

## A · Withdrawals — things we had wrong and have removed

### A1 · "Eight to ten generations" is no longer a calibration target
**Was:** a `[#]` figure in `L16.3`, listed among usable durations in the explainer.
**Why withdrawn:** `KB11`, verbatim — "I introduced the notion of ten generations… **which I put in not to say that it takes ten generations**… but to convey the notion that it's not just three and just ten, but it **is a multi-generational thing**." Both numbers were rhetorical.
**This is the third figure to fail this way**, after the per-generation decline rate and the annual societal-regression rate.
**Recommend:** approve. Keep the direction — multigenerational, more than three, unquantified.
**Cost if wrong:** we lose a calibration anchor the model never had a defensible use for anyway.

☐ ✅ ☐ ✏️ ☐ ❌

### A2 · The taboo set is no longer monotone
**Was:** `L10.8` — subjects withdrawn from a tie and "not returned".
**Why corrected:** 1979 Tape 6 — "the **purposeful mention of the taboo subject**, if one can control one's own anxious response, **can desensitize the whole mechanism**."
**Recommend:** approve. Growth is the default, not a law; the reversal is the differentiating move applied to one topic and carries the same gates.

☐ ✅ ☐ ✏️ ☐ ❌

---

## B · Terminology — one decision I need from you

### B1 · "Emotional" means instinct, not feeling — define or rename?
**The finding, two-source and unambiguous.** `KB02`: the emotional process is "**equivalent to instincts** — birds migrating… the salmon… on a cellular level, the force that guides an amoeba toward a morsel of food", and "**feelings are a superficial awareness of the emotional process**." `KB13`: "people have picked up this term I used, emotional, and **made it synonymous with feeling, which I didn't mean at all**. So I'm partially at fault."

**Why it matters:** the model's core quantity is **sub-affective**. It explains why overt emotionality peaks mid-scale and why a withdrawn person is as involved as an effervescent one.

**Options:**
1. **Define the word at first use** and keep the corpus vocabulary. *(implemented as `M1.A.0`)*
2. **Rename** the model's terms to *instinctual* — clearer to a reader with no Bowen background, but breaks every citation to the corpus and diverges from all four sources.

**Recommend:** option 1, already in place. **Cost if wrong:** a reader assumes anxiety is a feeling-state and builds an affective readout, which is exactly the error the corpus documents.

☐ ✅ option 1 ☐ ✏️ ☐ ❌ prefer option 2

---

## C · Framing — three corrections that reach the published proposal

### C1 · The fractal justification is not Bowen's
**Was:** the proposal defers the societal layer on the grounds that "Bowen is explicitly fractal… the family is the unit cell."
**Three independent statements say otherwise:** "an analogy is **not an extension of theory**"; the triangle in society is "an analogy, but **not a reasonable connection — these things don't connect up**"; society is "**similar to a family. Not the same as, but the same principles apply.**"
**Recommend:** keep the architectural decision, replace the reason with his formulation. *(implemented as `M11.F.2`)*
**Cost if wrong:** we would be claiming his authority for a derivation he explicitly refused.

☐ ✅ ☐ ✏️ ☐ ❌

### C2 · The general-systems objection, stated unsoftened
**The finding:** `KB13` — he rejects **general systems** as a foundation in favour of **natural systems**, because general systems "came out of man's head, along with mathematics". `KB09` — but calls the two "**compatible within limits**". **Both are his.**
**A simulation is a general-systems artifact**, so this is a real named objection to the model's foundation.
**Recommend:** record both statements, collapse neither, state the objection plainly rather than softening it. *(implemented as `M11.F.3`)*

☐ ✅ ☐ ✏️ ☐ ❌

### C3 · No output presented as Bowen's authority
**The finding:** `KB09` — his goal was "an **open theoretical system**, where the basis for new knowledge is **research and science rather than anything I said**", and he names discipleship as what closes a system off: "the more people that treat it like that, the more **my theory will perish** as being a dogma."
**Recommend:** approve as a standing framing requirement. *(implemented as `M11.F.1`)* This is also the strongest external justification for the grading discipline this project has used.

☐ ✅ ☐ ✏️ ☐ ❌

---

## D · Model additions — new requirements needing sign-off

| # | Requirement | Source | Recommend |
|---|---|---|---|
| **D1** | `differentiation_capacity` — a **family-level** variable: does *any* member retain the ability to take a position? Bowen reserved "schizophrenia" for families where none does, and ran "a **test of treatability**". | `KB08` · K08.2 | ✅ — severity becomes a capacity that predicts treatability, not a symptom label |
| **D2** | The live-issue gate carries a **family-type** term, and it is a **safety** property. Peace-agree families need an issue raised; reactive families need the opposite. `KB08` records three outcomes of getting it wrong: acute psychosis, a self-blinding, a suicide. | `KB05`/`KB06`/`KB08` | ✅ — the spec previously stated the gate unconditionally |
| **D3** | Spouses paired at matched **`basic_level`**, never functional. | `KB14`, 1979 Tape 3 | ✅ — a construction rule, testable |
| **D4** | `structural_importance` **derived** from functional-doer position + suddenness, not assigned as a static tier. Role labels excluded — a matriarch outranks a patriarch where she held the position. | Tape 6 · T6.1 | ✅ — better sourced and computable from existing state |
| **D5** | Asserting a differentiated state is **negative evidence** for it. Five independent forms across five interviews. | `_KB_PASS2.md` | ✅ — the readout `outside_ness` never had |
| **D6** | The coach's objective is **to understand, not to help.** Four statements, an experiment (best resident lasted ten hours), and an independent replication by Kerr. | `KB03` · K03.1 | ✅ — best-evidenced coach claim in the project |
| **D7** | Coach detectors are **two-sided** — praise is as much a loss as blame. | `KB07` · K07.1 | ✅ — ours was negative-valence only |
| **D8** | `REDUCE_CUTOFF` has a **floor**; cutoff is "your lifeline". Target is **one** open relationship, not universality. | `KB05`, `KB08`, Tape 6 | ✅ — four sources agree contact is not monotonically good |
| **D9** | Distance **absorbs anxiety without symptomising** — the half the spec was missing. | `KB04` · K04.1 | ✅ — explains why a family can discharge into distance and read as untroubled |
| **D10** | Fusion is **life-stage dependent**; the infant–caretaker symbiosis is normal, not pathological. | `KB14` · K14.5 | ✅ |
| **D11** | On return from removal, re-escalation **exceeds** the original removal disturbance. | Tape 6 · T6.8 | ✅ — changes what `M11.C.11`'s release arm asserts |

☐ approve all ☐ approve except: ______

---

## E · Open — items I did not resolve, and would rather you saw than had smoothed

| # | Item | Status |
|---|---|---|
| **E1** | **The dials versus Bowen's three societal variables.** He names *mode of thinking*, *differentiation*, *anxiety intensity*. Our three dials are drivers of his third — one level below, not a replacement. | **Resolved in §G below** — mode of thinking is representable, as a graded gate coupled to differentiation. `M1.D.7a1` stands for the societal layer; §G is the individual-level mechanism. **Approval needed on §G.** |
| **E2** | **The marriage-ceremony break.** "Pretty good friendship relationships before marriage and then the whole thing gets messed up as of the time of the marriage ceremony. **I've often wondered the why of that**, but there it exists as a fact." No mechanism in any source. | Open. Possibly the same phenomenon as `L05.3`'s immediate pole assignment. |
| **E3** | **A ninth concept** (supernatural phenomena) was proposed in 1980 and never consolidated. The concept list was never closed — six in 1972, six in 1975, eight in 1976, nine proposed in 1980. | Recorded. The model implements the eight. |
| **E4** | **Tape 3's post-removal sibling symptom.** Later than Ch04's retraction of the durable target queue, but a symptom intensifying is not durable position uptake — and the withdrawn claim was withdrawn by Bowen himself. | **`L01.7` not reinstated.** Flagged for your judgement. |
| **E5** | **A7 (measurement bias), C4 (decoupling guard) and B4 (the ally rule) are single-sourced to the book** — silent across all six lectures and all fifteen interviews. | Worth knowing before you validate those three by hand. |

---

## F · Still needing you, unchanged

### F1 · The validation checklist
`docs/theory/Extractions to be human validated.md` — 23 items, the extraction-versus-source check. **The lectures and interviews have since settled or corroborated 15 of them**, so the list is smaller than when I handed it over. A7, C4 and B4 (E5 above) are the ones that most need a human eye, because nothing corroborates them.

### F2 · Four acceptance criteria that cannot be made code-testable in Phases B–D
`M11.E`, each with a human-review proposal: `M11.C.8` (incidence versus published rates — needs Phase E), `M11.C.9` (sibling position — the effect size is invented, so the test is close to unfalsifiable), `M11.C.11`'s shape claim, and `M5.F.2`'s counterfeit threshold — **the model's most consequential invented constant.**

---

## G · Resolution proposed for E1 — *mode of thinking* as a graded gate

Raised by you on 2026-08-24 and worked through against the corpus. This is the one section here
that **adds** to the model rather than correcting it, so it needs approval on its own.

### G1 · The coupling, and which way it runs

Differentiation and systems thinking are not independent, and they are not one derived from the
other. They are coupled, and the coupling that matters for the model runs through **whether the
differentiating move is comprehensible to the agent at all.**

Both directions are in the corpus:

- **Differentiation → thinking.** `KB07` · K07.3 names undifferentiation as "the **cement, the
  hardener**, that fixes" a way of thinking — reduce it and the thinking can change faster.
  `KB07` · K07.2: his own capacity to think systems tracked "**step by step**" the change in the
  relationship with his mother.
- **Thinking → differentiation.** Never stated as a proposition, but assembled from four places.
  K07.3's first obstacle — "you can't change a way of thinking until you have some notion of a
  **new way of thinking**" — means the frame must arrive before anything moves. `KB07` · K07.4
  says what a productive coaching contact actually delivered: the awareness that "there was an
  **alternative way to respond**, and also that **I was emotionally caught in it**." `KB07` · K07.1
  makes blame the gauge — systems thinking is impossible until the person can be "**emotionally
  neutral. That is without blaming or praising.**" And `KB02` has him renaming *mental illness* to
  *emotional illness* because "mental illness belongs to a **way of thinking** that sees the
  dysfunction as a product of the brain" — the individual-is-broken frame, identified by him as a
  mode of thinking rather than a mistaken belief.

**Why this matters more than a missing variable.** Without the systems frame, an agent told to
differentiate has no model of how it would help, so it executes the move in the only frame it has —
a stand taken *against* people. That is precisely what Bowen calls grotesque in interview #1: they
think differentiation is "something you do in an hour a weekend… to somehow separate a self from
the family, **to shout them down and let them know you are different**."

So pseudo-differentiation is not a separate error to be modelled. **It is what the differentiating
move degrades into when systems perspective is low** — which also joins up with `M5.F.2a`
(asserting a differentiated state is negative evidence for it).

**Graded, not a hard gate.** Nothing in the corpus says a person with no systems perspective is
incapable of a genuine differentiating move — only that they will not see why it would help and
will reach for blame. A hard gate would make the first differentiating move in any family
impossible and leave no way in.

### G2 · How and when an agent's systems perspective rises — *not* by learning

The natural architecture is to give agents a discomfort-reduction objective and let them discover
that differentiation pays off over a long horizon. **That architecture would make the model wrong**,
in a way that would be easy to miss because the runs would look encouraging.

Three reasons from the corpus:

1. **The binders work.** `KB10` · K10.7 is explicit that anxiety-relieving action *relieves the
   anxiety* — "cause and effect laws designed to **relieve the anxiety of the moment**, and the
   more we do that, **the more we promote the thing we're trying to fix**." The relief is real.
   That is why it is chosen. A learner is not making an error when it picks distance.
2. **The payoff of differentiation is invisible without the frame.** That is G1. An agent cannot
   optimise toward a value it cannot perceive, so the long-horizon return is not available to the
   learner in the first place.
3. **The moves are instinct-level, not deliberative.** `KB02` · K02.1 and `KB13` · K13.1 define
   *emotional* as **instinct**, and `M1.A.0` already forbids modelling the anxieties as
   feeling-states. An agent that *selects* a binder by evaluating its expected discomfort reduction
   has imported the individual-deliberative frame the theory rejects — before any parameter is set.

**The consequence is a negative prediction, and it is the strongest validation test in the model:
basic level MUST NOT drift upward on its own.** If a population of agents optimising for comfort
learns its way to differentiation, the model has contradicted multigenerational transmission, in
which levels are roughly conserved and differentiation is rare and hard.

**What does raise it, per the corpus:**

- **Frame supply from an external agent who already has it** — `KB07` · K07.4, and it is rare and
  low-yield: "**four times at most, five maybe, in two years**" out of six to eight contacts a year.
  What landed was not instruction; it was being shown an alternative *and* being shown one is
  caught.
- **Capacity exhaustion — the binders stop working.** `KB10` · K10.6 at societal scale: "**societal
  attitudes change when society no longer has an option.**" Pain is necessary and not sufficient:
  without frame supply, exhaustion produces symptom escalation, not insight.
- **One at a time.** `KB12` · K12.1 — only one person in a family can differentiate at a time.

**So the model needs two mechanisms, not one learner over nine moves.** Reinforcement (`M4.D.6`)
operates on the **reactive** repertoire, which is what gives a family its characteristic style. The
differentiating move sits **outside** the learner, gated by `systems_perspective`, which is raised
only by an exogenous input conditioned on the binders failing.

Mapping your four examples to the current repertoire, for the record: separation and cutoff are
`DISTANCE` / `CUTOFF`; an affair is a `TRIANGLE`; substance use is not a move at all but the
dysfunction-in-a-spouse symptom channel. All four are on the automatic side.

### G3 · A one-sided test found while working this through

`M4.D.6a` forbids the short-horizon relief proxy and `M4.D.6b` requires a longer declared horizon.
`M11.C.16` tests **one direction only** — that the short-horizon arm collapses onto `CUTOFF` and
`DISTANCE`. It does not test the opposite failure that G2 identifies: a long enough horizon
producing **spontaneous universal differentiation**. Lengthening the horizon fixes the collapse and
buys the wrong model if nothing bounds the other end.

### G4 · Spec changes this implies — **listed, not applied**

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

☐ approve §G as the resolution of E1 ☐ approve with edits: ______ ☐ leave E1 open

---

## What happens after approval

1. Approved documents and spec are frozen at v2.0.
2. Implementation **plan** — criteria mapped to files and order, per the project's planning rule.
3. Plan approved.
4. Code, starting at Phase B.

**No code before step 3.**
