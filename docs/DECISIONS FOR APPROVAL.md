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
| **E1** | **The dials versus Bowen's three societal variables.** He names *mode of thinking*, *differentiation*, *anxiety intensity*. Our three dials are drivers of his third — one level below, not a replacement. **Mode of thinking has no representation in the model and is not currently implementable.** | Recorded in `M1.D.7a1`; the gap must be stated wherever societal results are reported. **Your call whether that is acceptable.** |
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

## What happens after approval

1. Approved documents and spec are frozen at v2.0.
2. Implementation **plan** — criteria mapped to files and order, per the project's planning rule.
3. Plan approved.
4. Code, starting at Phase B.

**No code before step 3.**
