---
tags: [model-bt, validation]
status: awaiting human validation
date: 2026-08-22
---

# Extractions to be human validated

## What this is

`docs/theory/ch01.md`–`ch22.md` are summaries **this project wrote** from Dr Bowen's chapters. The chapters are the source and are not in question. What has never been checked is whether our summaries say what the chapters say — and everything downstream (the ledger, the explainer, the v2 spec) is built on those summaries rather than on the papers.

This is not a hypothetical risk. Pass 2 withdrew nineteen pass-1 findings: two rates manufactured from illustrations Bowen explicitly bounded, invented rankings, stripped hedges, topology the text does not have. In every case Bowen was right and our extraction was wrong. It happened again on 2026-08-22, when `resource_pressure` was graded "invented" although it is Ch18's own thesis.

So this list is not a re-read of the corpus. It is the **twenty-three extractions the model leans on hardest** — the ones where being wrong changes the architecture, not just a footnote. Checking these covers most of the load-bearing surface in an afternoon.

## How to use it

Each entry gives the claim as we recorded it, the words we rely on, and where to find them. Source chapters are in `/Volumes/CrucialX9/Downloads Duplicate SERP/fticp_chapters  TXT files/`; line numbers are approximate because the text layer wraps mid-word in places (you will see `corre ct`, `psychopath ological`), so search the phrase rather than trusting the number.

The question to hold for each one is not "is this true?" but **"does the chapter say this, at this strength, with these limits?"** Most of pass 1's errors were not fabrications — they were real claims reported without their hedges, or narrowed claims reported as general ones.

Three verdicts:

- **confirmed** — the chapter says this, at this strength.
- **narrower** — the chapter says something like it but hedged, scoped, or weaker. *This is the most common failure and the most valuable to catch.*
- **wrong** — the chapter does not say this.

---

# Tier A — structural

If one of these is wrong, the architecture changes, not a parameter.

### A1 · Three sinks, with emotional distance outside the budget

**We claim:** the family's undifferentiation is absorbed by exactly three mechanisms — marital conflict, dysfunction in one spouse, projection onto a child — and emotional distance is named alongside them every time but excluded from the count, making it an always-on baseline rather than a fourth competitor.

**Words we rely on:** "Other than the emotional distance, there are three major areas…"

**Where:** `Chapter16…txt:1702`. Corroborating counts: `Chapter09…txt:833` ("a certain amount of…"), `Chapter12…txt` (four patterns listed, three counted), `Chapter18…txt` (enumerates twice and disagrees with itself).

**What depends on it:** spec M6.I.1 and M6.I.2; proposal finding F4; the entire conservation architecture. Pass 1 read four sinks and inserted the word *absorbing* to make distance one of them.

**1979 lectures — CONFIRMED and sharpened.** Tape 2 names distance first and separately ("keeping distance from each other is one of the better ways"), then lists "the three ways… **because these are the three things that end up with symptoms**." The book establishes that distance sits outside the budget; the lecture supplies the criterion. Lower priority for validation than it was.

**Verdict:** ☐ confirmed ☐ narrower ☐ wrong — notes:

### A2 · A per-person life-energy budget, and the debit on the tie

**We claim:** a zero-sum allocation between relationship-seeking and goal-directed activity, its ratio set by scale level, and a differentiating step *debits the important other*. We treat this as the mechanism behind the change-back reaction — the ladder is the surface, the withdrawal of energy is the cause. We record it as stated four times in the chapter.

**Words we rely on:** "The life energy that goes into defining a principle for self goes in a self-determined direction, which detracts from the former energy devoted to the system, especially to the important other."

**Where:** `Chapter10…txt:1452`; the relationship-seeking / goal-directed split around `:776`.

**What depends on it:** spec M1.A.10, M6.I.3, M5.E.7; acceptance criterion M11.C.5. **Please check the "four times" claim specifically** — if it is once, this is a much weaker foundation than we have given it.

**1979 lectures — SILENT. This raises its priority.** No lecture statement of the relationship-seeking / goal-directed allocation was found, so it remains **Ch10 only**. The model uses it as the *cause* of the change-back reaction — and that reaction is now two-source while its proposed mechanism is not.

**Verdict:** ☐ confirmed ☐ narrower ☐ wrong — notes:

### A3 · Two variables only; reactivity is derived, not stored

**We claim:** Bowen names level of self and level of anxiety as the only two major variables, and derives reactivity from them rather than treating it as a third state.

**Words we rely on:** "the lower the level of self, the more reactive"

**Where:** `Chapter17…txt:756`.

**What depends on it:** spec M1.A.1 — this removes a whole state column that the grid engine carries (`TX`).

**Verdict:** ☐ confirmed ☐ narrower ☐ wrong — notes:

### A4 · The scale: 0–100, one transition at 50, top quartile withdrawn

**We claim:** 0–100 with four profiles; a single behavioural transition at 50 which is a *licence* over joint decisions rather than a general suppression of intellect; 75–100 explicitly withdrawn as hypothetical; roughly 90% of people in the lower half.

**Words we rely on:** "more hypothetical than real"; and for the licence, that the emotional system permits the intellect its own corner "as long as it does not interfere in joint decisions that affect the total life course."

**Where:** `Chapter16…txt:1403` and the band descriptions around it; population skew at `Chapter17…txt:686`.

**What depends on it:** spec M1.A.2 and M1.A.3 — the only source in the corpus for the scale's numbers. Pass 1 truncated the 50 threshold and read it as suppression.

**1979 lectures — NARROWED.** Bowen: the 0–100 range "**is arbitrary**", he could have used Fahrenheit, and "any kind of a scale is alright as long as one can get the notion of a continuum". Also "**I'm not liking this scale as I go through the years with it**". Please check whether the *book* also bounds the range this way, or whether we have been treating an arbitrary choice as a finding.

**Verdict:** ☐ confirmed ☐ narrower ☐ wrong — notes:

### A5 · Pseudo-self is the conserved quantity; solid self is exempt

**We claim:** dyadic exchange is zero-sum in pseudo-self only, and solid self does not participate in fusion at all.

**Words we rely on:** "In any exchange, one gives up a little self to the other, who gains an equal amount."

**Where:** `Chapter16…txt:1212`.

**What depends on it:** spec M6.I.4 and M6.I.5. The exemption is what makes this sharper than the grid engine's undifferentiated transfer — please check the exemption is actually stated and not our inference.

**Verdict:** ☐ confirmed ☐ narrower ☐ wrong — notes:

### A6 · No instrument, no rater procedure, no number assigned to a person — *in the corpus*

**We claim:** across all 22 chapters there is no measurement instrument, no rater procedure, no comparison group, and no number ever assigned to an individual — so the corpus supports directions, orderings and mechanisms but almost no magnitudes.

**Where:** a negative finding, so it cannot be pointed at a line. The nearest positive evidence is Ch16 on the clinical scale being slowed, and Ch17 on that research being stopped to prevent misuse.

**What depends on it:** the whole parameter register, spec M0.1–M0.4, and the grading system in the explainer. **This is the single highest-leverage item on the list.** If there is measurement anywhere in the chapters we missed, a large part of the model becomes calibratable rather than invented.

**Scope corrected 2026-08-22.** This claim is about **Bowen's papers**, not about the field. External instruments exist — the Differentiation of Self Inventory (Skowron & Friedlander 1998; Skowron & Schmitt 2003) is the closest attempt, and it is still not a good instrument. The project had been using A6 loosely to mean "nothing exists to calibrate against anywhere", which was wrong. See `_EXTERNAL_MEASURES.md`; one calibration target is now admitted from it.

**Kerr interview — the framing was wrong, though the finding stands.** Bowen anticipates measurement explicitly ("biochemical markers or indicators that could be measured"; "we wouldn't have to observe people, the chemistry would tell us") and insists theory be grounded "in the force that causes grass to grow". A6 records what the papers contain. It must not be read as Bowen rejecting measurement — he wanted it.

**Verdict:** ☐ confirmed ☐ narrower ☐ wrong — notes:

### A7 · Estimates of differentiation are biased by construction

**We claim:** because pseudo-self is lent, borrowed and traded between people, any attempt to estimate someone's level of differentiation produces false readings, recoverable only over a life course.

**Words we rely on:** "results in false readings when one attempts to estimate levels of differentiation"

**Where:** `Chapter17…txt:745`.

**What depends on it:** explainer §10.1 — the model must treat every readout-side estimate as biased and noisy rather than as a clean read. It now also governs how the model may be compared against the DSI at all.

**This one has external corroboration, which nothing else on this list does.** The DSI's Fusion With Others subscale — measuring the construct most directly about self borrowed from others — failed for five years (α .57–.74, no relationship to psychological adjustment, problem-solving or relationship satisfaction) while its other three subscales worked. That is consistent with Bowen's prediction, and it is suggestive rather than proof. Worth checking the original claim carefully given how much it now carries.

**Verdict:** ☐ confirmed ☐ narrower ☐ wrong — notes:

---

# Tier B — mechanism

If one of these is wrong, a named mechanism in the model changes shape.

### B1 · The reaction ladder, verbatim, and its two diagnostic rules

**We claim:** the system's response to a differentiating move runs "You are wrong" → "Change back" → "If you do not, these are the consequences"; that it decays unless fed by the mover defending or counterattacking; and that *absence* of a reaction means the move did not land.

**Where:** `Chapter21…txt:1192`.

**What depends on it:** spec M5.E.1–M5.E.3, acceptance criterion M11.C.5. Pass 2 recorded this as verbatim-correct; it is worth confirming, because the model treats the third rule as a hard diagnostic.

**1979 lectures — CONFIRMED independently.** Tape 4 gives the same three steps in the same order, with a worked case, thirteen years after Ch21's events. This is now the best-evidenced mechanism in the model and needs the least validation.

**Verdict:** ☐ confirmed ☐ narrower ☐ wrong — notes:

### B2 · Anger is the gate to the peak, not a fourth abort branch

**We claim:** the abort branches (defend, counterattack, go silent) sit at the *first* opposition and are the usual outcome; and anger is what admits the sequence to the final intense attack, rather than being an abort in itself.

**Words we rely on:** "when he is finally able to maintain his course without getting angry… the opposition does a final intense emotional attack"

**Where:** `Chapter13…txt:390`.

**What depends on it:** spec M5.D.3 and M5.D.4. Pass 1 had anger as a fourth branch, which inverts the state machine.

**Verdict:** ☐ confirmed ☐ narrower ☐ wrong — notes:

### B3 · The outside-ness gate — an emotional system knows the difference

**We claim:** until the mover is partly outside the system, an identical technique lands as either empty words or an attack — so a move's effect is multiplied by a hidden state of the actor that receivers can read and the actor may not.

**Words we rely on:** "either hollow meaningless words or a hostile assault on the system, and an emotional system knows the difference"

**Where:** `Chapter21…txt:2219`.

**What depends on it:** spec M1.A.9 and all of M5.F; convergence C10. This is what makes move resolution non-trivial — without it, effect is a function of move type and tie alone.

**Verdict:** ☐ confirmed ☐ narrower ☐ wrong — notes:

### B4 · The ally rule is displacement, not cancellation

**We claim:** an ally opens a *new peripheral triangle* — which Bowen calls his undoing — and the countermeasure is to detriangle them. We claim he never says support from a family member *cancels* the move, and that the gated quantity is taking a side, not knowing.

**Words we rely on:** "worked out a plan that permitted no 'allies'"; "to keep the entire family in one big emotional clump, and to detriangle any ally who tried to come over to my side"; "in the past I had been 'undone' by partners."

**Where:** `Chapter21…txt:1741`–`1744` (the passage spans a page break).

**What depends on it:** spec M8.6; resolution R2. Pass 1 reported cancellation as the corpus's most counter-intuitive finding, and pass 2 withdrew it. **Please confirm the withdrawal was right** — this is a case where we reversed ourselves.

**Verdict:** ☐ confirmed ☐ narrower ☐ wrong — notes:

### B5 · Coupling does not decay; separation is edge-rewiring

**We claim:** reunion restores full coupling immediately, and the mechanism is substitution on both sides rather than a slow-decaying edge on an isolated node — nobody is ever uncoupled. We also record this as hedged ("it appears impossible"), with no separation duration given, and scoped to severe cases.

**Words we rely on:** "the old emotional fusion of the triad is immediately operative again"; "any long-term separation from parents is accomplished only by finding a new family ego mass to which they can append themselves."

**Where:** `Chapter08…txt:989` and `:313`. Earliest and strongest source is `Chapter02…txt:78` — "The surface distance controls a deeper interdependence on each other."

**What depends on it:** spec M1.B.2–M1.B.4; the entire bond-energy term and the cut-off scenario. Pass 1 read the reunion sentence backwards.

**Verdict:** ☐ confirmed ☐ narrower ☐ wrong — notes:

### B6 · Removal fires off the remaining members' tolerance

**We claim:** hospitalisation is triggered by what the family will tolerate, explicitly decoupled from the patient's severity.

**Words we rely on:** "when you are well" meaning "when you no longer disturb the family"

**Where:** `Chapter08…txt:806`.

**What depends on it:** spec M7.D.4 and acceptance criterion M11.C.11 — the whole institutionalisation scenario.

**Verdict:** ☐ confirmed ☐ narrower ☐ wrong — notes:

### B7 · Investment is valence-blind — conflict counts as high investment

**We claim:** investment is share of thought occupied by the target regardless of sign, so conflict-laden preoccupation registers as *high* investment. We treat any implementation deriving it from warmth as inverting the sign.

**Words we rely on:** "the thoughts of both, whether positive or negative, are largely invested in each other"

**Where:** `Chapter04…txt:519`.

**What depends on it:** spec M1.B.8. A sign error here would invert the model's behaviour on exactly the families it is about.

**Verdict:** ☐ confirmed ☐ narrower ☐ wrong — notes:

### B8 · The seesaw is a conjunction over both parents, not a difference

**We claim:** the patient improves only while *both* parents are more invested in each other than in the patient, and *either* parent crossing drives the regression — a minimum over both, not a difference.

**Words we rely on:** "immediately and automatically"; and the "either… than either" phrasing around it.

**Where:** `Chapter04…txt:962`.

**What depends on it:** explainer §4.5. Pass 1 implemented a difference, which lets a strong marriage mask a defecting parent.

**Verdict:** ☐ confirmed ☐ narrower ☐ wrong — notes:

### B9 · Symptoms threshold on functional level, not basic — with the corpus's only arithmetic

**We claim:** symptoms are a threshold on the functional level rather than the basic one, and transfer magnitude scales inversely with the basic level. We use the one worked example: a husband functioning at 55 on strength drawn from a wife at 15, both with a basic level of about 35.

**Words we rely on:** "almost no functional shifts" high on the scale

**Where:** `Chapter09…txt:765` and the arithmetic nearby.

**What depends on it:** spec M1.A.5, M1.A.6. This is the only place in the corpus that puts numbers to the two-level distinction, so it carries more weight than a single passage normally should.

**Verdict:** ☐ confirmed ☐ narrower ☐ wrong — notes:

### B10 · Three live positions — and the therapist can be the third

**We claim:** any two can avoid the anxious issue and three cannot, that Ch03's count is of *family* members because its therapist is deliberately outside the working group, and that Ch14's "two people" method is a triangle in which the therapist occupies the third position.

**Words we rely on:** "Any two members of the father-mother-patient threesome can successfully avoid anxiety issues"; "the family work on its own problem in the hour while the therapist observes from the sideline"; and "the triangle of the two most important family members and the therapist."

**Where:** `Chapter03…txt:595` and `:444`; `Chapter14…txt:1074`.

**What depends on it:** spec M8.1; resolution R1. This is the reconciliation of a contradiction we had carried as unresolved — **please check we have not reconciled it too neatly.**

**Verdict:** ☐ confirmed ☐ narrower ☐ wrong — notes:

---

# Tier C — gates, calibration and negatives

### C1 · Financial dependence is a hard gate on the differentiating move

**We claim:** this is the chapter's only absolute — a differentiating move by a financially dependent person has never succeeded. It is the sole justification for keeping any material variable in the model at all.

**Words we rely on:** "has never been successful"

**Where:** `Chapter10…txt:2223`.

**What depends on it:** spec M1.A.15. If this is narrower than we have it, the model has no material variable at all.

**Verdict:** ☐ confirmed ☐ narrower ☐ wrong — notes:

### C2 · Symptom onset depends on chronicity, not instantaneous level

**We claim:** any unit recovers from periodic overload, and collapse follows when the panic becomes chronic — which requires an integrator rather than a threshold test.

**Where:** `Chapter18…txt:352`.

**What depends on it:** spec M4.C.3. Also please check the **three anxiety regimes** we withdrew: pass 1 read invisible / legible / chaotic as a claim about the system, and pass 2 concluded it describes the *observer's* ability to see. That reversal is worth a second opinion.

**1979 lectures — CONFIRMED, plus an addition.** Tape 5 defines anxiety as "the emotional responsiveness of a person to situational stress" and says its intensity "can be judged by the intensity **and fixedness** of these symptoms, the relationship patterns". *Fixedness* is a second observable the model does not currently read out.

**Verdict:** ☐ confirmed ☐ narrower ☐ wrong — notes:

### C3 · Resource pressure and the disappearance of land frontiers

**We claim:** the depletion of resources and the loss of land frontiers is Bowen's own proposed driver of societal regression, and that it acts through *awareness of availability* rather than consumption.

**Words we rely on:** "a spectrum of problems associated with population explosion play a major role in man's deeper anxieties"; "society appears to be functioning on a less differentiated emotional level than twenty-five years ago, that this may be related to the disappearance of land frontiers"; "It was important for him to know there was new land for him, even if he never went to it."

**Where:** `Chapter18…txt:1173` and `:1189`.

**What depends on it:** spec M1.D.7b–M1.D.7c. **We had this graded as invented until 2026-08-22**, which is exactly the failure this list exists to catch.

**Verdict:** ☐ confirmed ☐ narrower ☐ wrong — notes:

### C4 · The decoupling guard — differentiated families outrun the societal level

**We claim:** well-differentiated families come to function far better than the society around them, so a societal dial that moves every family proportionally is implemented wrongly.

**Where:** `Chapter18…txt:1438`.

**What depends on it:** spec M1.D.7 and M1.D.7d — the modulation of `media_amplification` by family differentiation.

**Verdict:** ☐ confirmed ☐ narrower ☐ wrong — notes:

### C5 · Overt emotionality peaks in the middle of the scale

**We claim:** people in the moderate range show the most intense overt feeling — so visible emotionality is not a monotonic proxy for differentiation, and is most misleading in the band where most agents sit.

**Where:** `Chapter16…txt:1320`.

**What depends on it:** explainer §10.1. This is a readout warning; if it is wrong, a natural readout becomes usable again.

**Verdict:** ☐ confirmed ☐ narrower ☐ wrong — notes:

### C6 · Two negatives — please confirm these are genuinely absent

These are claims that something is *not* in the corpus. They are the hardest kind to check and the easiest to get wrong, and the model relies on both.

**No durable projection target queue.** We claim Bowen retracts it himself in Ch04 and dates the error: six months of observation confirmed a replacement sibling taking the position, and two and a half years reversed it. We keep the fast redirection of anxiety and drop the durable uptake. *Depends on it:* the institutionalisation scenario cannot lean on a target queue.

**No anniversary effect.** We claim Ch15 reports delayed and prolonged effects — two-year chains, symptoms at five years, mourning six years late — but names no date-anchored recurrence anywhere. *Depends on it:* explainer §12.3, recorded specifically so a later pass does not invent one.

**Verdict, queue:** ☐ confirmed absent ☐ present after all — notes:

**Verdict, anniversary:** ☐ confirmed absent ☐ present after all — notes:

---

# If you find something

A **narrower** verdict is the most likely and the most useful. Note the hedge or the scope you found, and the entry gets restated rather than withdrawn — that is what pass 2 did to most of the nineteen.

A **wrong** verdict on anything in Tier A means the architecture moves, and the v2 spec should not be approved until it does. Tier B means one mechanism is respecified. Tier C usually means a parameter or a readout changes.

Either way the finding goes into `_LEDGER.md` against the entry's ID, and `_RESOLUTIONS.md` if it settles or reopens a contradiction.**1979 lectures — the anniversary negative is CONFIRMED.** Tape 6 is an entire lecture on family reaction to death and the word "anniversary" appears nowhere in any of the six transcripts.


