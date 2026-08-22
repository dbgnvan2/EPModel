---
tags: [model-bt, alignment]
status: consolidated
pass: 2 (corrections folded in)
---

# Alignment ledger — Bowen source chapters vs. the agent model

**This file is now self-contained.** Every pass-1 entry has been re-read against
`_PASS2_CORRECTIONS.md` and rewritten so it states what is actually supported. You do **not** need to
cross-reference the corrections file to act on an entry here. `_PASS2_CORRECTIONS.md` is retained as
the audit trail — what changed and why — not as a required companion.

## How to read an entry

Each entry carries a **verdict** (its relation to the model) and a **status** (what pass 2 did to it).

Verdicts:
- **ALIGNS** — the model already represents this.
- **REFINES** — the model represents it but too coarsely, or with the wrong shape.
- **CONTRADICTS** — the model asserts something the source denies.
- **UNMODELLED** — no place in the model to put this. The valuable ones.

Status markers:
- `[stands]` — pass 2 checked it and it survives as written.
- `[narrowed]` — survives, but scoped down. The narrowing is stated in the entry.
- `[corrected]` — pass 1 got the mechanism or the sign wrong. The entry now states the corrected one.
- `[withdrawn]` — not supported by the text. **The ID is kept as a tombstone**, with the reason, so
  that nothing downstream re-derives it. Do not cite a withdrawn entry as evidence for anything.
- `[new at p2]` — found on the second read; no pass-1 counterpart.

Tombstones rather than deletions is a deliberate departure from "delete what was withdrawn". Eleven
withdrawn IDs are cited in `docs/agent_model_proposal.html` and `_CONVERGENCES.md`; deleting them
silently would leave dangling references and invite exactly the re-invention that pass 2 spent its
effort undoing.

Targets: `Person` · `Relationship` · `Triangle` · `Family` · `move:<NAME>` · `coach` ·
`test` · `event-kind` · `policy` · `readout`

---

# ⚠ Read these four things before using any entry

## 1. The bias in pass 1, in one line
**Pass 1 systematically over-read, and every error ran the same direction** — it made the source look
more quantitative and more decided than it is. It invented rankings, manufactured rates from
illustrations Bowen explicitly bounded, added topology the text does not have, stripped hedges, read
later vocabulary backwards into earlier chapters, turned the study's analytic emphasis into therapy
rules, and reported inferences as measurements. When an entry below looks unusually crisp, check
whether its status marker says `[stands]`.

**The corpus supports directions, orderings and mechanisms. It almost never supports magnitudes.**

## 2. Chapter counts are not study counts
Bowen reports the same clinical material across many papers. Agreement between two chapters drawing on
the same project is one study reported twice.

| Setting | Chapters | Note |
|---|---|---|
| NIMH live-in project, 1954–59 | 01, 02, 03, 04, 05, 07, 08, 16 (in part), 19 | **Nine chapters, one project** |
| Menninger / Topeka, 1949–54 | 08 | |
| Georgetown private practice | 09 onward | |
| Multiple-family pilot, 1965 | 11 | |
| Bowen's own family | 21 | n = 1 |
| No clinical series at all | 13, 17, 18, 20 | |

Known double-counts inside the NIMH cluster: **Ch02 (n=5) and Ch04 (n=6) are the same father case
series.** **Ch07 is a near-verbatim rewrite of sixteen of Ch06's propositions**, same order, often the
same words — it is not an independent witness. **Ch01's father sequence is n = 1, not 2** ("the second
father is still in this stage"), so every timing in it belongs to one family.

## 3. Two corpus-wide negatives, checked in all 22 chapters
- **Q-VALIDATION is NO everywhere.** No instrument, no rater procedure, no comparison group, no number
  ever assigned to a person. The only ordinal grading found is Ch04's 1/9/1 ordering of eleven
  families on emotional-divorce *style*, and Ch19's "token concurrence" scalar (L19.5).
- **Q-MATERIAL is NO as a per-person stock.** The engine's `M` metabolic column — a resource whose
  depletion kills — has no basis anywhere. What the corpus *does* support is resources as
  **preconditions and capacities**: Ch10's financial dependence as a hard gate on one move (L10.11a),
  Ch13's family-level "ability to provide material demands", Ch18's four hits, Ch14's professional-time
  and motivation budgets, Ch15's poverty gating the *form* of contact. Never a stock that keeps a
  person alive. Ch20 is the strongest negative: it is the corpus's only workplace chapter and tie
  weight there is set by emotional importance **explicitly not by economic relation**.

## 4. Two live contradictions, unresolved, carried into pass 3
- **Ch10 vs Ch21 — does a witness help or displace?** Ch10's flagship differentiating step is stated
  aloud and its spouse witnesses the whole effort and it succeeds. Ch21 permits no allies and
  detriangles any that appear. The object-level resolution (a specific withdrawal vs. the overall
  programme) holds and is settled — see L21.3 — but this residue is genuine.
- **Ch03 vs Ch14 — is the threesome or the twosome productive?** Ch03 says progress happens with all
  three present and any two can avoid the issue. Ch14 abandons the threesome for the twosome. This
  directly opposes L02.4 and L03.4, the group-size finding pass 1 called three-times corroborated.

---

## Ch01 — Treatment of Family Groups With a Schizophrenic Member  *(1957)*

**Provenance:** NIMH live-in project. Father sequence is **n = 1**.
**Chronological baseline.** No triangle concept, no differentiation scale, no multigenerational
transmission. The causal vocabulary is symbiosis + transfer of anxiety + projection; "family ego mass"
and "undifferentiated" each appear exactly once and are never joined. Everything the model treats as
core Bowen is absent at the origin — this is the zero point for the evolution track.

### L01.1 — Anxiety transfers as a quantity, with a location and a per-edge latency  `[stands]`
**REFINES → `Relationship`, `Family`, invariant**
Reported as "almost a quantitative transfer": the source's anxiety measurably falls as the recipient's
symptom rises. Stated latencies differ by edge — mother→patient "very soon", mother→younger son
"within hours", patient's gains→mother's physical illness over months.
*Model impact:* support for the conservation invariant. Adds something the model lacks: **transfer
latency is a property of the edge**, where the model currently delivers every event within one tick.
Improvement propagates as a *perturbation*, not only as a good.
*Note:* the **fast** transfer is what survives here. Durable position uptake by the replacement is
withdrawn — see L01.7.

### L01.2 — The peripheral member moves first, and his move is a precondition  `[narrowed]`
**UNMODELLED → `policy`, `move:I-POSITION`, `Triangle`**
The father changed first; mother and patient remained stuck until he established himself as a person,
and only then could the mother make use of anything.
*Narrowing:* pass 1 read this as two families and treated it as independent corroboration of L02.5.
It is **one family**, and Ch02/Ch04 — the other legs of that convergence — are the same case series.
**Ch05 states the opposite:** the family *leaders* began differentiation first, usually the
overadequate mother. Ch05's own reconciliation is what to keep: **the leader drives the process, the
peripheral member performs the visible position change. Two roles, not one contested slot.**
*Model impact, as narrowed:* still a **gate** — certain moves are unavailable until a structural
precondition holds elsewhere — but the gate is on the *visible position change*, and the model must
not encode "peripheral member moves first" as a single ordering rule.

### L01.3 — Position is not involvement  `[stands]`
**UNMODELLED → `Relationship`, `Person`**
A father who entered the family by siding "became mother's agent, using her words", and the patient
related to him exactly as to the mother.
*Model impact:* a single engagement or contact scalar cannot represent this. The model needs
**positional identity** — whose side a member is functionally occupying — as distinct from how much
contact the tie carries. Two members can be maximally involved and occupy one position.

### L01.4 — The coach is capturable, and topology beats insight  `[stands]`
**CONTRADICTS → `coach`, `test` 7**
The effective variable was relationship topology, not therapist skill: multi-channel contact produced
distortion relayed "within minutes" and compounding across staff; a single-privileged-channel rule
also failed; the rule that nobody discusses an issue until all parties are in the same room worked.
Meanwhile staff had excellent intellectual understanding "but the process continued on an unconscious
level."
*Corroborated from a different setting:* Ch11's parallel individual therapists made the family problem
"diffuse and compartmentalized, and difficult to define" (L11.6).
*Model impact:* the coach's ability to stay detriangled must **not** be a high-FD parameter or an
insight score — capture is a property of proximity and channel structure. Rewrite test 7 around
channel topology rather than around a coach who chooses to stay outside.

### L01.5 — Message fidelity degrades per private hop  `[stands]`
**UNMODELLED → `Relationship`, `event`**
Distortion compounded across each private relay between staff members.
*Model impact:* events currently arrive intact. A per-hop fidelity term makes "who talks to whom in
whose presence" mechanically consequential, which is what L01.4's working rule turns on.

### L01.6 — Projection is stress-gated  `[stands]`
**REFINES → `move:TRIANGLE`, `policy`**
Denial and reaction formation under calm; projection appears only "in periods of stress".
*Model impact:* the projection-type move should be gated by system anxiety rather than continuously
available at a low propensity.

### L01.7 — Projection has a durable target queue  `[withdrawn]`
**⛔ RETRACTED BY BOWEN HIMSELF, IN Ch04.**
Pass 1 called this "the strongest quantitative anchor found so far": remove the symptom-bearer and the
projection relocates durably, with a well sibling taking the position at 7 months. **Ch04 retracts it
and dates the error** — six months of observation confirmed the durable uptake; two and a half years
reversed it. Ch01's evidence sits at seven months, inside the window Bowen later says was too short.
**What survives:** fast redirection (hours to months) of anxiety after a member is removed — that is
L01.1 and it is untouched. **What is withdrawn:** *durable* position uptake by the replacement.
*Consequence:* the institutionalization scenario cannot lean on a target queue, and test 11 cannot be
calibrated against one.

### L01.8 — The change-back ladder is already here in 1957, and the reaction decays unfed  `[new at p2]`
**REFINES → `move:I-POSITION`, C9**
Two-rung escalation: "You are weak, go slow" → mover "showed little concern" → mother to bed within
hours on each further gain → she "maintained this command for several months and then relinquished it."
**The success condition is an inequality** — "his strength seemed greater than the mother's attack" —
and the reaction **decays if unfed**.
*Model impact:* pushes C9's state machine back twenty years from Ch13/21/22, and supplies the
comparison form of the calm-check gate: not a threshold on the mover, a *comparison* between the
mover's hold and the attack.

### L01.9 — The basic/functional distinction predates the vocabulary  `[new at p2]`
**→ evolution, `Person`**
"Mother was *really more impaired* than the daughter" while holding the adequate role. The terms are
absent; the distinction is drawn. Relevant to any claim that two-level functioning arrives with Ch09.

---

## Ch02 — The Role of the Father in Families With a Schizophrenic Patient  *(1959)*

**Provenance:** NIMH; the five-father series. **Same case series as Ch04's six.** Ten families for the
three-edge pattern.
**Q-VALIDATION:** none. **Q-MATERIAL:** none — no money, income, employment or cost anywhere; the only
scarce goods are attention/investment and the patient's time, both relational.

### L02.1 — Closeness is permission-gated by the excluded third  `[withdrawn]`
**⛔ WITHDRAWN. There is no permission scalar.**
Pass 1 proposed a consent/permission quantity held by the excluded third. **The five-father series
refutes it:** "The mothers **approved** the father-son efforts, but they did not give up their prior
intense attachments, and the fathers' efforts all failed." Approval was granted and the transfer still
failed. The gating quantity is the incumbent's **unreleased investment in the competing tie** — which
the model already has. This *simplifies* the model: it collapses into L02.2 rather than adding a
mechanism.
**The sexed-asymmetry claim is also withdrawn.** It is stated in sexed terms; pass 1's "explicitly not
to sex" is not in the text. Bowen's one position-not-sex clause attaches to *who changes first*, a
different claim, and the incumbent/challenger reading actually **inverts** the sentence, since the
mother is the usual incumbent. Attaching asymmetry to occupied position is the modeller's decision and
must be labelled as such.

### L02.2 — Investment across ties is zero-sum  `[stands, with the definition corrected]`
**UNMODELLED → `Person`, `Relationship`, `test`**
Parents more invested in each other than in the patient → patient gains; either more invested in the
patient than the spouse → the process intensifies.
**Definition, from Ch04, and pass 1 inverted its sign:** "the thoughts of both, **whether positive or
negative**, are largely invested in each other." Investment is **share of thought occupied by the
target, valence-blind**, traded across targets. **Conflict-laden preoccupation is HIGH investment.**
Any implementation deriving investment from warmth or agreement inverts the sign on exactly the
families this is about. It is an **inference by the observers** — not self-report, not time spent, no
instrument.
*Model impact:* a second conserved quantity, allocated across a person's ties, with the marital tie
load-bearing rather than one tie among equals.
*Ready-made regression test:* a father subdued his son, symptoms remitted within a week, held about a
month, then collapsed entirely — both parent–child ties improved, the marital tie did not, and the gain
decayed. Two arms, a stated time course, a stated outcome.

### L02.3 — Symptom moves opposite to system health in the short run  `[stands]`
**REFINES → `test`, `Person`, `readout`**
"When the parents change their functioning, the patient becomes more disturbed." The course is a damped
oscillation — reversion, then repetition at greater frequency and less turmoil — with a hysteretic
threshold once the stand-taker holds.
*Model impact:* the push-back is not merely tension on the acting tie, it is **symptom in a third
person**; and the predicted shape is a **damped oscillation with hysteresis**, not a spike-and-settle.
The model accumulates symptom load monotonically and cannot express this trajectory.
**This entry and L03.3 are the correct sources for "the right move gets worse before it gets better".**
Ch14's version of that claim is withdrawn (see L14.2) — it was built from a risk clause. Ch02 and Ch03
state it as an observation.

### L02.4 — Avoidance capacity is a step function of group size  `[stands — but see the live contradiction]`
**UNMODELLED → `Triangle`, `policy`**
Any two members can successfully avoid the anxious issue; with three present and the time unstructured,
the conflict cannot be avoided.
*Model impact:* supplies the *reason* triangles form, which the model currently asserts rather than
derives. Avoidance availability is a property of the active group size, and it is a step, not a
gradient.
**✅ RESOLVED — see `_RESOLUTIONS.md` R1.** The apparent Ch14 conflict was with a section *heading*
("Family Systems Therapy with Two People"); the section itself describes **"the triangle of the two most
important family members and the therapist."** The finding is not about family members — it is about
**live positions**, and Ch03 counts only the family because its therapist is structured out ("observes
from the sideline"), while Ch14's therapist *is* the third position. **Restate as: avoidance is available
when fewer than three live positions are in the working group.** A position counts only if its occupant is
not fused into another.

### L02.5 — The under-functioning member moves first  `[narrowed — see L01.2]`
**UNMODELLED → `policy`, `move:I-POSITION`**
The first to take a stand is the parent in the inadequate position.
*Narrowing:* Ch02 and Ch04 are one case series, and **Ch05 says the leaders move first**. Restate as
the two-role split in L01.2.
*What still stands regardless of which reading wins:* propensity for the differentiating move is **not
monotonic in functioning**. The model raises I-POSITION propensity with FD, which predicts the
*over*-functioning member moves first — and no reading of the corpus supports that.

### L02.6 — One distinguished primary triad, on exclusivity grounds  `[new at p2 — replaces L02.1's conclusion]`
**REFINES → `Triangle`**
The triad-as-primary conclusion survives L02.1's withdrawal, on different and weaker grounds: a joint
exclusivity/allocation constraint across three edges — closeness to the child is a **rivalrous good**
("win the patient *from*") — plus the deviant case. The three-edge pattern is asserted as **one constant
pattern across all ten families**; pass 1 split it across three bullets and lost that.
**Scope limit: fixed membership, no recruitable third.** This licenses **one distinguished primary
triad**, not a general triangle primitive. Do not over-generalise it in the spec.

### L02.7 — Distance is a regulator over an undiminished coupling  `[new at p2]`
**REFINES → `Relationship`, C7** — *the earliest source in the corpus for C7*
"The surface distance controls a deeper interdependence on each other."
*Model impact:* distance does not attenuate coupling; it **regulates** it. Together with Ch08 this is
the strong evidence for C7 — Ch06's college-student case is weaker than pass 1 reported (see L06.3).

---

## Ch03 — Family Relationships in Schizophrenia  *(1959a)*

**Provenance:** NIMH.
**Chronological marker.** Vocabulary is **interdependent triad** — with *fixed* membership (father,
mother, patient) — plus **emotional divorce** and **overadequate–inadequate reciprocity**. Not
*triangle*. Differentiation of self, projection process, undifferentiated family ego mass and
multigenerational transmission are all absent.
**⛔ Pass 1's header claim "positions do not yet shift" is FALSE** and it propagated into the timeline.
Membership is fixed and nobody is swapped in or out — **but positions shift on two axes**, and Bowen
states it as his general finding: "the parent in the overadequate or strong position would **shift** to
the inadequate position." What is genuinely later is the *generalisation to any threesome*,
*recruitment of a third under tension*, and *interlocking triangles*.

### L03.1 — Over/under-functioning: one anti-symmetric coupling, held per area of joint activity  `[corrected — absorbs old L03.2]`
**REFINES → `Relationship`, `move:DISTANCE`**
Both parties "equally immature"; whoever decides becomes overadequate and the other helpless; neither
can find mid-ground; **either can occupy either pole**.
**The switching rule, missed by pass 1:** the pole flips when the inadequate one's self-assertion is
**"greater than"** the other's aggression and domination — a **relative** comparison — it flips
**immediately**, and it holds only if the mover sustains it through the counter-reaction.
**Pass 1 split one variable into two.** The pole is held **per area of joint activity**, and "reducing
the areas of joint activity" (father to business, mother to home) operates on the **same variable** —
it lowers anxiety while leaving the reciprocity untouched. Old L03.2 was not a second mechanism.
*Model impact:* `functioning_balance` is **bistable with no stable centre**, indexed **per area**, with
an immediate relative-comparison flip and a sustain requirement. Not a linear scalar, and not a
personal trait. The model has contact *rate* but not contact *scope*; scope is the axis this mechanism
actually moves along.

### L03.2 — *(merged into L03.1)*  `[withdrawn as a separate finding]`
Reciprocity-resolves-by-scope-reduction is the same variable as the pole assignment, not a second
mechanism. Retained as an ID so downstream citations resolve.

### L03.3 — The full change trajectory, with time constants  `[stands]`
**REFINES → `test` 5, scenario**
Inadequate parent asserts → overadequate parent destabilizes **within days** → settles calm and firm
**within two weeks** → emotional divorce resolves → investment shifts from patient to spouse → patient
improves → the whole configuration reverts **after a month** → but thereafter it is easier for the
father to pull up and less threatening for the mother to let go.
*Model impact:* a completely specified acceptance test with stated durations. Corroborates L02.2 and
L02.3. **Reversion is part of the course, not failure** — a model that treats relapse as a failed run
is wrong.
*Also:* when the patient improves "it is usually the mother who becomes symptomatic". This corroborates
**fast redirection**, not the withdrawn durable queue of L01.7.

### L03.4 — Presence determines anxiety, and the avoidance ladder has TWO rungs  `[corrected]`
**UNMODELLED → `Triangle`, `policy`**
With all three present and the structure held: conflict, high anxiety, progress. Any two of the three
can successfully avoid the issue, making the work "more intellectual, more sterile."
**⛔ The five-rung avoidance ladder was invented below rung 2.** Bowen ranks exactly two: (1) "the most
frequent is for the decision-making family member to engage the therapist in conversation"; (2) "the
next most frequent is for the parents to talk to the psychotic one about the psychosis", which
"frequently changes rapidly to criticism". **Criticism is a drift inside rung 2, not a rung.** Small
talk and silence are an **unranked residual**. **Rung 2 unlocks only when the coach closes rung 1.**
*Model impact:* a two-rung fallback with an unlock condition, plus an unordered residual. Not a
five-step ordered sequence and not an unordered softmax over moves.
**✅ RESOLVED — see `_RESOLUTIONS.md` R1.** Ch14 does invert the sign of the act, and the reason is
mechanical rather than a change of mind: in Ch03 the therapist is outside the working group, so engaging
him **removes** the third position and leaves a twosome that can avoid; in Ch14 he **is** the third
position, so engaging him keeps three live. Same act, opposite meaning, because the third position moved —
**C10 applied to technique.** The two-rung avoidance ladder and the presence finding both stand.

### L03.5 — Physical illness as a stochastic exit from an attempted position change  `[stands]`
**UNMODELLED → `Person`, endogenous event**
"Not unusual for a parent in an inadequate position to develop a physical illness when he attempts to
pull up."
*Model impact:* an endogenous symptom generated by a *specific move*, not by accumulated load — and it
attaches to the differentiating move, the one the model currently treats as unambiguously good.

### L03.6 — The leadership office must be occupied  `[stands]`
**REFINES → `Family`, `Person`**
The family stalled for two months when the mother did not resume the lead after a vacation, and resumed
"immediately" when the therapist recognized her position.
*Model impact:* the mechanism is *vacancy* — an unoccupied office stalls the system, and the stall is
released by external recognition rather than by the occupant. Generalised to work systems by L20.1.

---

## Ch04 — A Family Concept of Schizophrenia  *(1960; started 1957, finished 1958)*

**Provenance:** NIMH; the six-father series — **the same series as Ch02's five**.
**Chronological marker.** "Maturity/immaturity" is used **under protest** ("I do not like the terms"),
with no scale, no zero point, no measurement procedure.
**Q-VALIDATION:** one ordinal 1/9/1 ordering of eleven families on emotional-divorce *style*. That is
the whole of it, and it grades a style, not a person. It corrects pass 1's "no scale of any kind" but
supports nothing quantitative.

### L04.1 — The parental-closeness seesaw is a CONJUNCTION over both parents  `[corrected]`
**REFINES → `Relationship`, `Family`, `test`**
"more invested in each other than **either** was in the patient" / "when **either** parent became more
invested…". This is a **min over both parents**, and **one parent alone drives the regression**.
**Pass 1's `investment(parent→spouse) − investment(parent→child)` is wrong** — a difference form lets a
strong marriage mask a defecting parent, which is the case the chapter is about.
Correct form: patient improves only while *both* parents are more invested in each other than in the
patient; the patient "immediately and automatically" regresses when *either* crosses. Fast time
constant.

### L04.2 — Management technique has zero independent effect  `[stands]`
**CONTRADICTS → `coach`, `test`**
When parents are close "they could do no wrong" — firmness, permissiveness, punishment, talking it out
all work. When emotionally divorced, "any and all management approaches" fail equally.
*Adversarial test:* any intervention aimed at the symptom-bearer while marital distance is high must
produce **no improvement**. A model in which coaching helps regardless of marital state is wrong.

### L04.3 — The overadequate pole is assigned by ACTS, not held as a trait  `[stands]`
**REFINES → `Relationship`, `policy`**
Explicitly "functioning states and not fixed states." Whoever makes a decision *becomes* the overadequate
one — which is why these families avoid all decisions.
*Model impact:* the pole is an outcome of move selection, and decision-avoidance is a derived behaviour
rather than a separate move.

### L04.4 — Source position determines an intervention's effect  `[stands, strengthened]`
**UNMODELLED → `event`, `coach`, `policy`**
The identical content lands oppositely by speaker: a therapist or outside figure naming the mother's
projection → mother, father *and* patient attack or withdraw. The same thing said by the patient or the
father → "a significant beneficial emotional reaction."
**Two additions from pass 2.** The verbs are asymmetric — an outsider "suggests", an insider "confronts"
— so **the milder outside act is worse than the stronger inside act**. And Ch05 corroborates with
provenance: self-initiated versus "the doctor told me to" (L05.6).
*Model impact:* appraisal currently depends on intensity, tie conductance and receiver FD — not on **who
sent it**. This bears directly on the coach: an outsider saying the true thing makes it worse.

### L04.5 — The father–child edge has two gates at two levels  `[corrected]`
**UNMODELLED → `Triangle`**
Pass 1 recorded one hard precondition: the father cannot have a primary relationship with the patient
until he has first changed his emotional divorce with the mother; six families, six attempts, every one
failed.
**Corrected:** there are **two gates at two levels**. *Contact-level* closeness is granted freely by the
mother — and doing so assigns the father the "representative of the mother" position (this is L01.3's
mechanism). Only the **primary relationship position** is gated. And the six failures are attributed
**in-text to the mother's non-relinquishment**, not to the emotional divorce; the emotional-divorce
sentence is a separate claim **with no case evidence attached at all**.
*Model impact:* keep the gate, but gate the *position*, not the contact — and source the gate to
unreleased incumbent investment (L02.1's replacement), which is what the cases actually show.

### L04.6 — Ch04 retracts Ch01's durable target queue, and dates the error  `[new at p2]`
**→ discipline, `test` 11**
Six months of observation confirmed durable uptake of the projection by a replacement sibling; two and
a half years reversed it. See L01.7. Recorded here so the retraction is visible from the chapter that
made it.

### L04.7 — "Investment" is defined, and it is valence-blind  `[new at p2]`
**REFINES → `Relationship`**
"The thoughts of both, **whether positive or negative**, are largely invested in each other." See L02.2
for the full statement and the sign warning.

---

## Ch05 — Family Psychotherapy  *(1961)*

**Provenance:** NIMH.
**Chronological marker.** "Undifferentiated family ego mass" is described as discarded "because it has
certain inaccuracies" — **and then used anyway two sentences later**. Retired-but-in-service, no
successor; working substitutes are "family oneness" and "the family as the unit of illness".
"Interdependent triad" does the job "triangle" does later. **Differentiation of self is a process with
ordinal grading, not yet a scale.**
*Note the conflict:* **Ch08 (~1962) actively retains and defends the ego-mass term.** The arc is messier
than "used → discarded → revived → abandoned" and must be re-derived at pass 3.

### L05.1 — Symptomatic change and basic change are separate variables  `[stands]`
**REFINES → `Person`, `Relationship`, `readout`**
A passive father took one unilateral stand → psychotic symptoms gone within **three days** → one month
later the stand collapsed and the psychosis returned, because the father–mother relationship had not
changed. Bowen states the general rule that change in a fixed rigidity of the *parental relationship* is
followed by change in the patient "irrespective of the immediate level of psychotic symptoms" — and says
the project revised its outcome criterion accordingly.
*Model impact:* fast reversible symptom state driven by one dyad, distinct from slow durable structural
state requiring marital change. Corroborates L03.3 with tighter constants.

### L05.2 — Function is absorbed by whoever will carry it, including the professional  `[stands]`
**UNMODELLED → `coach`, `Family`, access vector**
The leader-mother became "a helpless complaining person" the moment the therapist tried to help her deal
with her family. Seven in-residence families "with hospital staff nearby, were never able to deal with
their helplessness"; outpatient families at equivalent chronic impairment did much better.
*Model impact:* a burden-transfer term on the coach, and an environmental function-absorption variable.
Proximity of help is not neutral — it is a load-bearing input with a negative sign.

### L05.3 — Reciprocity is a paired allocation that STICKS with dyad age  `[stands]`
**REFINES → `Relationship`**
One *immediately* takes the overstrong position and the other the helpless one; neither can occupy the
middle; either can occupy either; **over years the assignment becomes fixed**. Decision paralysis
follows: deciding reads as dominating, yielding as submitting.
*Model impact:* a stickiness term growing with time-in-configuration.

### L05.4 — Calibration set  `[narrowed — most of it withdrawn]`
**→ `test`**
**⛔ Withdrawn:** the outcome tally, which **has no stated denominator** and is therefore not a usable
calibration target. And **the "leaderless family" was invented by pass 1** — the three counted
leadership types sum exactly to fifteen (8 + 4 + 3), a complete partition with no fourth category.
**What survives:** 15 families with fathers; three leadership types with rank-ordered outcomes; and the
case timeline — 4 months / 8 hours to remission, relapse at 1 month, 16 months / 73 hours, 94 hours over
3 years. Treat these as *orderings and durations*, not as rates.

### L05.5 — The leaders move first  `[new at p2]`
**CONTRADICTS → C5, L01.2, L02.5**
"The family leaders were the first to begin working on differentiation of self" — usually the
**overadequate** mother.
*Resolution the chapter itself supplies:* the leader drives the process, the peripheral member performs
the visible position change. **Two roles, not one contested slot.** This is the form to implement.

### L05.6 — Provenance changes an act's effect  `[new at p2]`
**UNMODELLED → `event`, `coach`** — *corroborates L04.4*
Self-initiated versus "the doctor told me to" produce different effects from the same act.

---

## Ch06 — Out-Patient Family Psychotherapy  *(1961a)*

**Provenance:** NIMH outpatient. **Ch07 rewrites sixteen of this chapter's propositions near-verbatim**
— when both are cited, that is one witness.

### L06.1 — The child's symptom is a continuously-driven output, not an autonomous state  `[stands]`
**REFINES → `Person`, `Relationship`** — *corroborates L04.1*
In three families Bowen never saw the symptomatic child at all; symptoms fell "within a few weeks" of the
parents holding focus on themselves, and recurred briefly whenever the parents hit intense anxiety in
therapy. Decay ~weeks, re-drive ~immediate.
*Critical qualifier:* the load variable is **directed parental attention**, not parenting content — the
problem is created "just as surely by a project that was psychologically correct as by one that was
psychopathological."

### L06.2 — Symptom relief opposes progress — and the mechanism is RETARGETING, not fuel  `[corrected]`
**CONTRADICTS → `coach`, objective function**
Pass 1 glossed C1 as "distress is the fuel of differentiation." **The chapter says something different
and more useful:** motivation is **aimed at** symptom level — "The motivation is toward alleviation of
symptoms rather than toward changing the familiar and comfortable situation behind the symptoms."
**Relief satisfies the goal rather than depleting an energy source.** Stated three times, in three
populations.
*Model impact:* the coach's task is **retargeting the objective**, not preserving distress — a materially
different intervention, and a materially different implementation. The coach's objective cannot be
"reduce family anxiety."
*Also:* couples who displaced part of the problem onto a child are *easier* to work with than undisplaced
couples, whose defences are more rigid.
**⚠ C1 has a direct counterexample in Ch22** — reducing cut-off reduces symptoms *and* makes therapy more
productive. C1 likely survives only as a claim about *symptom relief obtained without structural change*.
See L22.1.

### L06.3 — Physical distance does not attenuate coupling  `[narrowed]`
**CONTRADICTS → `Relationship`**
*Narrowing:* pass 1 leaned on the college-student case. That case is **n = 1**, the magnitude is "almost
as dramatic", the student had "brief vacation contacts" (**low contact, not zero**), and the next sentence
is "More experience is needed."
**The better evidence in this chapter is the two general statements:** triad members "can be physically
separated from each other" without any of them differentiating a self, and physical distance is one of
two equivalent routes to pseudo-separation.
**The strong sources for C7 are Ch02 (L02.7) and Ch08 (L08.2).**
*Model impact, unchanged:* kills any distance- or contact-frequency-based conductance. **Access gates
which moves are available; it does not attenuate coupling strength.**

### L06.4 — The unit can be acted on through any single member  `[stands]`
**ALIGNS → `coach`**
Bowen routinely works with one person and insists it not be called individual therapy; change in the
motivated one propagates until the other expresses a wish to join. Supports coaching one member as the
headline experiment.
*See L21.10:* in Ch21 this is explicitly framed as an **on-ramp** — worked "until the unmotivated spouse
is willing to join" — as often as a substitute.

### L06.5 — Two stated gaps, recorded not filled  `[stands]`
**UNMODELLED**
No rule for what determines whether a problem stays in the spouse dyad or is transmitted to a child ("For
some reason, their children remain relatively uninvolved"), and no rule for what selects which spouse
takes the dominant pole at identical levels. Bowen does not know.

### L06.6 — A precursor of the sink taxonomy, with no budget asserted  `[new at p2]`
**→ evolution, conservation invariant**
All four of Ch18's mechanisms are present here, **never enumerated**, and with **no budget asserted across
them**. The conservation claim is later; the mechanisms are not.

---

## Ch07 — Intrafamily Dynamics in Emotional Illness  *(1965)*

**Provenance:** NIMH. **⚠ Not an independent witness for reciprocal functioning** — sixteen of its
propositions rewrite Ch06's, same order, often the same words. C2 is 6 chapters, not 7.
**Chronological marker.** No *triangle*, *family projection process*, *multigenerational transmission*,
*emotional cutoff* or *sibling position* — though several are described in plain language. **No numbers
at all** beyond dates and ordinal positions.
*Terminology datum:* **"emotional divorce" occurs in all of Ch02–Ch06 and zero times in Ch07.** A
withdrawal, not a not-yet.

### L07.1 — Ego-mass membership is a continuous stress-dependent involvement weight  `[corrected]`
**UNMODELLED → `Family`, architecture**
Calm → the mass includes only a few most-involved members. Stress → the fusion extends to multiple
extended-family members "and even nonrelatives." And: "Live-in servants can be more emotionally fused into
the family emotional system than certain blood relatives."
**Corrected shape:** pass 1 made this a *set* that grows and shrinks. Ch07 has three distinct claims
including **per-ego temporal modulation by stress** — so the primitive is a **continuous per-agent
involvement weight recomputed each tick, with membership a thresholded readout**, not a set operation.
*Model impact:* architectural, not a parameter. A fixed twelve-person roster keyed on kinship gets the
wrong set, and gets it wrong exactly when it matters — under stress.

### L07.2 — Self is conserved and traded; the symptom absorbs the deficit  `[stands]`
**REFINES → `Person`, `Relationship`**
Spouses enter marriage with equal levels of self; thereafter one functions with more than an equal share;
the one who gives in loses self to the one who gains it; a habitual giver-in reaches "no-self" and is
incapacitated by physical illness, emotional illness, or social dysfunction. The chronic illness "seems to
absorb the ego deficit between them," and the marriage stays harmonious **as long as the disabled spouse
does not recover**.
*Model impact:* basic level and current functional level as separate attributes; a transfer term with
approximate conservation; and **negative feedback from symptom onto family tension** — so curing a symptom
without changing the deficit must *raise* tension.

### L07.3 — Stabiliser efficacy is a partly independent third variable  `[narrowed]`
**UNMODELLED → `Family`, `Person`**
"The clinical course can depend more on the effectiveness of mechanisms to stabilize the problem than on
the intensity of the problem" — high-identity spouses whose stabiliser fails experience almost as much
anxiety as a family with low identity.
**Narrowing — pass 1 said "independent of level". It is not.** High-identity spouses "generally are able
to find **a wider range** of effective stabilizing mechanisms": **level gates the repertoire**; only the
*realised course* is decoupled.
**And pass 1 recorded the benefit without the price:** both of Bowen's low-identity exemplars are
**another member's impairment**.
**Second termination condition, missed:** the stabilisation holds "as long as the incapacitated one
**lives**" — so **death destabilises exactly as recovery does** (L12.2's spike, by the other route).
*Model impact:* differentiation sets the **baseline, the repertoire and the recovery**; the stabiliser
sets the **peak**.

### L07.4 — Explicit sensitivity ranking  `[stands]`
**→ `test`, `readout`**
Course determined first by the dynamics between the two spouses, second by maintenance of relationships
outside the family ego mass, third by the intensity of the fusion. A sensitivity analysis that does not
reproduce this ordering is mis-parameterised.

### L07.5 — The fused default is the term-for-term inverse of the differentiating move  `[new at p2]`
**UNMODELLED → `policy`, default state**
Each alters self to manage the other's functioning while demanding the other change — "neither responsible
for self".
*Model impact:* this belongs in the policy as the **default state**, not as an absence of a move. The
differentiating move is then a departure from a specified baseline rather than an event on an empty one.

---

## Ch08 — Family Psychotherapy with Schizophrenia in Hospital and Private Practice  *(1965a; written 1963–64)*

**Provenance:** Menninger/Topeka 1949–54 **and** the NIMH live-in ward 1954–59 — **the same ward reported
in Ch01–Ch06**. Agreement between Ch08 and any of those is not independent replication.
*Terminology datum:* Ch08 **actively retains and defends "undifferentiated family ego mass"**, against the
timeline's claim that it was discarded at Ch05.

### L08.1 — Removal fires off the REMAINING members' tolerance, and is gated by decision ownership  `[stands, refined]`
**REFINES → scenario, `event-kind`**
Families keep severe illness at home as long as the person behaves; the real trigger is that the family
wants the disturbing behaviour removed. "Hospitalization is more for the family than the patient"; "when
you are well" means "when you no longer disturb the family." After removal, parental anxiety subsides once
the diagnosis is confirmed and the member becomes "a patient."
**Refined at pass 2:** the trigger is a **per-agent tolerance comparison** against disturbing behaviour —
explicitly decoupled from severity — with **three tolerance-holders**. And it is **gated by who owns the
decision**: in the one case where tolerance was reached, the event did **not** fire, because Bowen returned
the decision to its owner.
*Model impact:* `INSTITUTIONALIZE` fires off the remaining members' tolerance, not the symptom-bearer's
severity, and the coach can suppress it by relocating decision ownership.

### L08.2 — Separation is edge-rewiring with the old edge dormant  `[corrected]`
**REFINES → `Relationship`, bond energy**
"If the threesome is reunited, the old emotional fusion of the triad is immediately operative again." The
triad survives intact with the member institutionalised as a permanent ward of the state.
**⛔ Pass 1 read this backwards.** It treated the sentence as coupling persisting through an
*interaction-free interval*. Bowen's stated mechanism is **substitution on both sides**: the triadic one
lives away "in dependent attachments that do not involve the parents", the parents "borrow self from
outside themselves and project their inadequacies to others", and the rule is that "any long-term
separation from parents is accomplished only by finding a new family ego mass to which they can append
themselves." **Nobody is uncoupled.**
*What the passage licenses:* a **zero re-activation latency on reunion** — a different parameter from a
decay rate. It licenses **no** claim about an isolated node carrying a slow-decaying edge, which materially
changes the bond-energy mechanism proposed in §4.3.
*Hedges pass 1 dropped:* "it *appears* impossible", **no separation duration given**, and the claim is
scoped to *severe* schizophrenia — the next paragraph says differentiation "does occur" in less severe
families.

### L08.3 — Anxiety has substitutable discharge channels; blocking one reroutes it  `[corrected]`
**UNMODELLED → `Family`, `test`**
**⛔ "Three measured instances" was wrong on both counts — nothing is measured, and it is two.** The
chapter contains **no counts at all**. Phase one measures *incidents* ("the frequency of incidents remained
the same"); phase two measures *complaints* ("complaints … stopped") — and complaints are what the
intervention explicitly named as its goal, so **the outcome variable switches mid-sequence**. Pass 1's
"frequency dropped to zero" **is not in the text**. The third "instance" is a typology (the no-self spouse
presenting interchangeably as physical illness, emotional illness or social dysfunction), not an observed
rerouting.
**And relocation carries a sign:** acting out moving from the community back into the family is "a hopeful
sign".
*Adversarial test, corrected:* a coach that supplies help should produce **the same incident count in
different locations**, not fewer — but **the test cannot score incident count alone**, because location
changes the sign. Score count *and* location.

### L08.4 — The separation crisis: collapse from the best-functioning state, by involuntary reflex  `[stands]`
**UNMODELLED → `policy`, `test`**
Fully timed: six prior years institutionalised → working within six months → at one year functioning at the
highest level Bowen had seen, parents calm, parental pattern receded → on the day she announced definite
plans to move out, parents became "anxious, pleading, attacking, and helpless" → projection returned at
full intensity → she gave up her own goals "to help her parents" → lost her job within six months →
re-institutionalised ten months later.
Bowen names the return move an "automatic emotional reflex to save the parents" — **involuntary and fast,
not deliberate**. The model has only deliberated moves.

### L08.5 — Parental conflict and child projection are explicit alternatives  `[stands]`
**ALIGNS → conservation invariant**
In the seventeen-year-old case, parental conflict subsided precisely as the parents rejoined forces and
resumed projecting. Direct support for the anxiety-routing budget.

### L08.6 — The professional system is part of the family's emotional system  `[stands]`
**UNMODELLED → `coach`, external agents**
Psychiatrist, hospital, school, courts and merchants run the same three-step projection process the mother
runs — the school "had long followed the usual steps of thinking about, diagnosing, and treating the son as
sick." Responsibility is a conserved transferable quantity the family cannot hold while staff holds it.
Strongest form: "if the medical structure did not exist, the families could find other means to make the
environment responsible."
*Model impact:* external agents need the ability to **absorb responsibility** and to **certify a defect** —
neither exists in §5.1's repertoire.

### L08.7 — The change-back ladder at ~1962, and its first rung is a SYMPTOM  `[new at p2]`
**REFINES → C9, `move:I-POSITION`**
Five rungs, recorded by pass 1 as one bullet: helplessness → self-labelling → demand for caretaking →
attack on the tie → a week of withdrawal → capitulation. **The opening move is a symptom, not a verbal
challenge** — a symptom-mediated variant of C9's state machine, predating Ch13/21/22.
The severe-end version is the **"petition for sickness"**, which **succeeds** instead of failing.
*Model impact:* the reaction ladder needs a symptom rung at its foot and a success branch at its head.

---

## Ch09 — The Use of Family Theory in Clinical Practice  *(1966)*

**Provenance:** Georgetown private practice.

### L09.1 — Two-level functioning, with the first arithmetic in the corpus  `[stands]`
**REFINES → `Person`**
A husband functioning at **55** on strength drawn from a wife at **15**, both with a basic level of about
**35**. Transfer magnitude scales **inversely** with basic level ("almost no functional shifts" high on the
scale). **Symptoms are a threshold on functional level, not basic.**
*Model impact:* confirms FD-vs-C as designed, and adds that the *spread* between the pair is itself set by
their common basic level.
*Note:* **Ch09 also states the full scale** — 0–100 endpoints **and** all four quarters. On the scale
question it belongs with Ch16, not with Ch21. See L16.1.

### L09.2 — Conserved "immaturity" allocated across exactly three channels  `[stands]`
**ALIGNS → conservation invariant**
"The system operates as if there is a certain amount of 'immaturity' to be absorbed" — spread across
marital conflict, dysfunction in one spouse, and transmission to children, with overflow possible as "still
free-floating immaturity". **The overflow destination is "conflict with families of origin", not distance.**
Emotional distance appears as "the most common mechanism" a paragraph earlier and is **not counted** — see
L16.2 for the settled sink count.
*Two falsifiers Bowen supplies himself:* conflict alone does **not** impair children; children **are**
impaired in calm marriages. A chronically ill parent binding load **protects** the children.

### L09.3 — Triangle: threshold recruitment, position-value inversion, and hysteresis  `[stands]`
**REFINES → `Triangle`**
Tension above threshold recruits a third. The outside position is **unfavoured when calm and favoured under
tension** — the value of a position inverts with load. Tension reroutes onto "old preestablished circuits",
so triangles need persistent identity and activation memory. An "I" position held "for even a few days"
produces a **permanent** decrease in that triangle's intensity — the state does not fully revert.
*Model impact:* inside/outside preference cannot be fixed. The permanent decrement is the counterweight to
the change-back reaction of L02.3.

### L09.4 — The family's own account of its history is a separate, actively distorted layer  `[stands, load extended]`
**UNMODELLED → `Family`, `readout`**
"The family emotional system operates always to obscure and misremember and to treat such events as
coincidental."
*Model impact:* a belief layer distinct from ground truth. **Ch15's "hidden dependence network" folds in
here** — that was pass 1 inventing a second edge type; what Bowen actually says is that the dependence is
*denied*, i.e. invisible to the family, and the propagation is multi-hop on the ordinary graph (see L15.2).
**Ch19 adds a constraint on this layer:** truth value and emotional function are orthogonal — "the estimate
might be accurate or not and yet be largely in the service of a denial" (L19.6). A belief layer must not be
implemented as "the emotionally driven claim is the false one".

### L09.5 — Evolution note: the triangle is excluded from basic theory, and Bowen never reverses that in words  `[corrected]`
**→ evolution**
Ch09 names **five** concepts and calls **three** "major": differentiation of self (the Scale) → the
relationship system in the nuclear family ego mass (**family projection process is a sub-part, not yet its
own concept**) → multigenerational transmission. Sibling position and triangles are named but held **out**
of the three, the triangle because it has "more to do with therapy than the basic theory".
**⛔ Pass 1 read a later "promotion" of the triangle that is not asserted.** The "basic building block"
phrasing in Ch17 is **Berenson's, in the question**; Bowen's own origin story — "a precision therapeutic
technique" — **agrees** with Ch09. The real evolution is **positional**: triangles are **sixth of six** in
Ch21 (1972) and **second of six** in Ch14 (~1975) and Ch18.

### L09.6 — Sideline coaching is ranked WORST of three avenues, in 1966  `[new at p2]`
**→ evolution, `coach`** — *the biggest evolution datum, hidden by a pass-1 flattening*
Pass 1 wrote "three avenues, ranked". Bowen's actual text: "two main avenues… A **third avenue is less
effective**: the entire process under the guidance of a supervisor who **coaches from the sidelines**" —
because "[d]irect use of the 'triangle' is lost."
**Sideline coaching is the method Bowen champions in Ch14 and Ch22.** He reverses this completely and, so
far as the corpus shows, never says he is doing so.
*Model impact:* the coaching agent's design is being drawn from a method its own author ranked worst nine
years earlier. That is a framing obligation, not a parameter.

### L09.7 — "≈15 points lost per generation"  `[withdrawn]`
**⛔ MANUFACTURED.** No rate is stated anywhere in the chapter. The schematic's steps are −15, −15, then
−10 and −5, and the third-generation parent pair is never assigned a level. Pass 1 also dropped Bowen's two
cap sentences — "in the average situation the immaturity would progress at a much slower rate". A
calibration target was fabricated from an illustration he explicitly bounded.

---

## Ch10 — Family Therapy and Family Group Therapy  *(1971)*

**Provenance:** Georgetown private practice.
**⚠ Ch14 reports different numbers for this same series** — see L14.5.

### L10.1 — Symptom level and differentiation are independent and can move oppositely  `[stands]`
**CONTRADICTS → `readout`, `test`**
Family group therapy reliably relieves symptoms, spreads the problem evenly across members, plateaus at
**12–20 sessions**, and "does not provide the structure for a higher level of differentiation of self."
Prolonged past its short-term goal it recruits the **more adequate** children into carrying the problem — a
fresh load-shift onto a previously uninvolved child.
*Model impact:* a single wellbeing scalar cannot represent this chapter. **Two outcome axes, minimum.**

### L10.2 — Distribution governs the response to therapy; the total still governs the trouble  `[narrowed]`
**UNMODELLED → `Family`, `readout`**
Families with symptoms in all three areas do best; families concentrated in one area resist everything but
symptom relief. Chronic illness in a spouse is a sink with a **stability bonus** — that marriage is
"enduring". Symptom is not pure cost.
**Narrowing:** pass 1 conflated two quantities. **Total undifferentiation governs the quantity of trouble;
distribution governs the response to therapy.**

### L10.3 — Cutting off raises system tension; enlarging the system lowers it  `[stands]`
**ALIGNS → bond energy, `move:CUTOFF`**
Anxiety dilutes over nodes.
*Model impact:* cutoff carries a *system-level* cost on top of the actor's relief, and L07.1's
stress-dependent involvement weight becomes functional rather than descriptive — recruiting more members is
how a family lowers its own load.

### L10.4 — The neutral third position IS the intervention  `[stands]`
**REFINES → `coach`, `test` 7** — *generalises L01.4*
Stated three times and explicitly generalised: "would probably proceed with any third person… no matter
what the subject matter." Its converse is the chain-reaction interrupt — an emotional system responds to
emotional stimuli, so one member controlling his response breaks the chain. **The therapist's neutrality
and a family member's self-control are the same operation from different positions.**
*Model impact:* the coach needs no special mechanism. It is a Person in the outside position, and test 7
should compare positions rather than professional skill.

### L10.5 — Silence, withdrawal and cutoff are moves INSIDE the system, never exits  `[stands]`
**REFINES → `move:CUTOFF`, `move:DISTANCE`**
The runaway fuses into a new family and reproduces the pattern. There is no exit from the field, only a
change of address.

### L10.6 — Differentiation transfers between systems; intensity does not  `[stands, strengthened]`
**ALIGNS → architecture**
Differentiation gained in a peripheral system transfers automatically to the nuclear family. **Self is a
per-person scalar; intensity is per-relationship.**
*Strengthened at pass 2 by Ch21's before/after on a system he never worked on* — see L21.7, the cleanest
cross-system result in the corpus.

### L10.7 — An announced withdrawal does not escalate  `[narrowed]`
**UNMODELLED → `event`, `policy`**
Announcing an intention before withdrawing prevents escalation; others read *unannounced* silence as a move
and escalate to force the habitual response.
**Narrowing — pass 1 overstated it four ways.** The announcement goes to **the son**, a party in the
conflictual dyad, not to the system. It is **one of three conditions**, not the operative one. The first
thing communicated is **confidence in the son's competence**. And vignette 2 is a **control case** where an
unannounced withdrawal *failed* and **Bowen blames residual reactivity, not the missing announcement**.
*Model impact, as narrowed:* receivers still need a model of what they expected, and appraisal should be
driven by **deviation from expectation** rather than by the act alone. But announcement is not established
as the causal variable.

### L10.8 — Monotonically growing taboo-subject set per dyad  `[stands]`
**UNMODELLED → `Relationship`** — *the mechanism behind L03.1's scope reduction*
Subjects are withdrawn from a tie and not returned.

### L10.9 — Institutional acts write to a belief with hysteresis  `[stands]`
**UNMODELLED → external agents, `event-kind`**
Hospitalization, a prescription, an insurance form write to a family-level who-is-sick belief that verbal
denial cannot reverse.

### L10.10 — The coach's own fusion state, with three stated readouts  `[stands]`
**→ `coach`, `test`**
Atypical patterns, slowed progress, family passivity. Directly usable as test assertions on the coach
agent. Corroborated by Ch19 (L19.1) and Ch20 (L20.2) — **three chapters, three different signal sources**.

### L10.11 — A per-person LIFE-ENERGY budget  `[new at p2]` — **the biggest miss of pass 1**
**UNMODELLED → `Person`, `move:I-POSITION`, conservation invariant**
Stated **four times** in Ch10 as a zero-sum allocation between **relationship-seeking** and
**goal-directed** activity, with the ratio set by scale level, an empty-budget state at the bottom of the
scale, and an explicit **debit on the tie** when a differentiating step is taken — the step "detracts from
the former energy devoted to the system, especially to the important other."
*This is a second conservation law, per-person,* alongside the per-family undifferentiation budget. And it
supplies something better than pass 1 had: **the actual mechanism of the change-back reaction.** C9 treats
it as a scripted sequence of responses to a message; Ch10 makes it **a withdrawal of energy the other was
receiving**. The script is the surface; the debit is the cause.
*Model impact:* the differentiating move gets a **cost**, not only a **reaction** — which also explains C1
without needing a separate mechanism, and replaces the withdrawn "motivation budget" of L22.5.

### L10.11a — Financial dependence is a hard gate on the differentiating move  `[new at p2]`
**UNMODELLED → `move:I-POSITION`, Q-MATERIAL**
The chapter's **only absolute precondition**: a differentiating move by a financially dependent person "has
never been successful". Plus financial security as an input to the case couple's equilibrium, work
performance as a 35–40 band readout, and a better job as a downstream gain.
*Model impact:* **this does not rescue the engine's `M` column**, which is a metabolic account whose
depletion causes death. It establishes a different and narrower role for a material variable: **a gate on
one move.** See the corpus-wide Q-MATERIAL note at the top of this file.

### L10.12 — The fifth scale band: agents that look differentiated and are not  `[new at p2]`
**UNMODELLED → `Person`, `readout`, C10**
"The upper part of the 25-to-50 segment": dogmatic authoritativeness, the compliance of a disciple, or the
opposition of a rebel, with "**intellect in the service of the relationship system**."
*Model impact:* **any readout keyed to position-taking behaviour scores these agents wrong.** This is C10
from a fifth chapter — an act's identity depends on the actor's hidden state.

### L10.13 — Family-of-origin distance as field attenuation  `[withdrawn]`
**⛔ OVER-READ, and it would have contradicted C7.** Pass 1 read Ch10's family-of-origin distance passage as
intensity attenuating with distance. The chapter says nothing of the kind: Bowen's claims are about **the
work** ("the process goes better", "may be slowed"), and the thousand-mile case explicitly **retains
contact** via visits, letters and calls. As written this would have licensed a distance-decayed conductance
term — the exact thing C7 forbids.

### L10.14 — The communication reversal is complete here, not in Ch14  `[new at p2]`
**→ evolution**
Ch10 already states Ch14's position **in full** — each spouse to the therapist, never to each other, with
the technique "**in use more than five years**" (so ~1965) — names spouse-to-spouse talk as a superseded
format, **and** credits family group therapy's fast symptom relief to open communication among members.
Ch11 dates the change further back: "after about 1962, I stopped suggesting that they talk directly to each
other." **Ch14 presents as new something Ch10 published four years earlier.**

### L10.15 — Ch10 has no secrecy requirement, and its witness succeeds  `[new at p2]` — **live contradiction**
**⚠ UNRESOLVED → `move:I-POSITION`**
Ch10's differentiating step is **by construction stated aloud**, and its flagship case has **the spouse
witnessing the whole effort and succeeding** — which sits badly beside Ch21's detriangle-every-ally rule
and its secrecy requirement.
**✅ RESOLVED — see `_RESOLUTIONS.md` R2. The gate is alignment, not knowledge.**
Ch21's rule is stated in the language of *sides* throughout — "come over to **my side**", "getting on my
'side'", "align themselves" — and its purpose is named: peripheral triangles, "this pattern had been my
undoing". Its one worked ally case is the younger sister's "**I am back of you if I can be of help**",
which Bowen calls "a red flag" and handles until "she **retreated from taking sides** with me". She kept
her knowledge; only her position changed.
And Ch21 contains a witness who knows nothing, **takes no position**, and the effort succeeds — his wife,
whose silence he singles out as unprecedented.
Ch10 bars alliance from the helper's side in the same terms: the therapist may help "**without being
perceived as against the family**." Its spouse is the **target** of the move and the next mover, never a
supporter.
**Secrecy is a risk-reduction heuristic over alignment**, because inside the system knowledge produces
alignment — "messages run back and forth in such a family system as if by telepathy". That is why the rule
is scoped in the same sentence to "another person **who is part of the system**".

---

## Ch11 — Principles and Techniques of Multiple Family Therapy  *(1971a)*

**Provenance:** the 1965 multiple-family pilot. Distinct setting from the NIMH cluster.

### L11.1 — Closing a channel displaces the flow onto another  `[narrowed]`
**REFINES → C3**
The ward sequence is verified and Bowen writes "externalize intrafamily anxiety" himself at steps 2 and 3:
staff participated → families externalised into conflict with staff. Staff silenced → anxiety moved to
differences between the four co-therapists. Therapists cut to one → anxious members of *other* families
interrupted at the exact moment an issue was being defined.
**⛔ Ch11 asserts NO conservation — pass 1 imported it from Ch08/Ch09.** Three constructs are withdrawn:
the extended family as a "lower-intensity sink", family-of-origin work as a "destination for togetherness
pressure", and a "shared conserved togetherness budget". **None is in this chapter.** The "unintentional
controlled experiment" framing is also pass 1's, and the four co-therapists were present throughout, so the
structural changes **close** channels rather than creating them.
**C3 drops Ch11 as a conservation witness.** What Ch11 supplies is the **displacement sequence**, which is
a weaker and different claim.

### L11.2 — Cross-family observational learning, on separability not design intent  `[corrected]`
**UNMODELLED → architecture**
"If one family made a breakthrough in an area, within a week or two other spouses would be trying some
version of that in their own families" — progress ~**50% faster**. Bowen rejects the families' own
explanation (reassurance) for a different mechanism: "it is easier to really see and know your own problem
when you watch it in other people."
**⛔ "Engineered to maximise observation" is half false.** Bowen did **not** design the setting for
cross-family observation. The stated motive was **teaching-time economy**, and he reports it as finding (1)
of "two major findings … not accurately predicted" — i.e. **it failed**. Observational learning is finding
(2), "the surprise", explained post hoc under "Apparently". Only the *suppression* half is verbatim (no
social contact, no mention of other families to mutual friends); the seating is described but never listed
as a precaution, and "no emotional tie" is an aspiration, not an achieved state.
*What survives:* **two inter-family edge types, one wanted and one destructive, independently switchable** —
standing on **separability** (his rules govern outside-session contact while observation happens inside the
session), **not on design intent**. Also: auditing beat participating, because preparing your next comment
blocks hearing — attention is finite and the two modes compete.
*And note:* the speed-up occurred **despite** the coach being more easily triangled in that setting.

### L11.3 — More contact does not produce more change  `[stands, with three provenance corrections]`
**CONTRADICTS → `coach`, scheduling**
Frequency went 2–3/week → weekly → monthly → settled fortnightly. Monthly families made as much progress
"and possibly even more" than weekly. "It takes a certain amount of time on the calendar… not decreased by
increasing the frequency of appointments." Symptom relief **and increased togetherness** are both listed as
reasons families quit — the reward the system delivers early is the very quantity the work exists to reduce.
*Corrections:* the calendar-time claim is introduced as **"my conviction"** that the result "fits with", not
as a finding. The **four-year average is scoped to "upper middle class families"**. And **the one-year
honeymoon decay is mis-filed** — it belongs to the ~1955 ward open-system conversion, not to session
frequency.

### L11.4 — Design constants  `[stands]`
**→ `test`**
Three families at start, four optimum, five "too rushed and pressured"; 1.5-hour sessions plus 30-minute
research summary; two hours as the attention ceiling; no family skipped in any session.

### L11.5 — An open channel works without being used  `[new at p2]`
**UNMODELLED → access vector** — *no analogue anywhere else in the corpus except Ch18*
"What was important was that the system was open and they could attend if they wished" — stated **after
attendance had collapsed**. **Availability itself carries the effect, independent of use.**
*Model impact:* access is **not** a gate on moves; it is a **standing input** whether or not any move is
taken. Converges with L18.3 — "it was important for him to know there was new land for him, **even if he
never went to it**." **Two chapters, one mechanism: a belief about what is available, not a stock and not a
gate.**

### L11.6 — Topology beats insight, second instance  `[new at p2]`
**→ `coach`** — *corroborates L01.4 from a different setting*
Parallel individual therapists made the family problem "diffuse and compartmentalized, and difficult to
define".

---

## Ch12 — Alcoholism and the Family  *(1974)*

**Provenance:** the alcoholism series — an independent setting.
**Chronological anomaly, and pass 2 made it sharper.** Alcoholism is **1974** — two years after Ch21's
six-concept enumeration and one year after Ch18 — and it still contains **no triangle anywhere**, proven by
exhaustive string search. Every mechanism is dyadic plus a generational cutoff link. This is an *omission of
a construct he already had*, not a not-yet.
*Sinks:* lists four fusion patterns and **counts three** — emotional distance is "almost universal" and
**excluded from the count**. See L16.2.

### L12.1 — Signed, equal-magnitude transfer that hardens with duration, and is asymmetric to reverse  `[narrowed]`
**REFINES → `Relationship`**
The dominant spouse "gained functional strength at her expense"; the adaptive one goes into "an equal degree
of dysfunction." Organ analogy: an organ that has functioned for another for long periods "does not return
to normal so easily."
**⛔ The organ analogy does NOT ground the direction asymmetry.** Duration-hysteresis and direction-asymmetry
are **two separate claims, pages apart**, and Bowen explicitly withholds the reasons for the second. Pass 1's
derivation is withdrawn; **both findings survive independently**.
*And the asymmetry is scoped:* it is far easier to tone down a **"marked"** overfunctioner than to raise a
**"marked"** dysfunctioner — not a general claim about every delta.

### L12.2 — Recovery of the de-selfed spouse produces a CONFLICT SPIKE  `[stands]`
**REFINES → `test`** — *corroborates L06.2 and L07.2*
The wife's drinking remitted "fairly promptly"; her regaining self was followed by "a period of fairly intense
marital conflict"; she discovered their "thinking alike" had been her failure to think for herself. The prior
harmony was purchased with her self, so restoring it necessarily destroys the peacekeeping mechanism.
*The general rule behind it is in Ch21:* marital conflict has a second trigger — **a chronic adapter who
stops** (L21.9). And **L07.3's second termination condition reaches the same spike by the other route**: the
stabilisation holds only as long as the incapacitated one *lives*, so death destabilises exactly as recovery
does.
*The chapter's sharpest falsifiable prediction — belongs in the test suite verbatim.*

### L12.3 — A single non-symptomatic node can control the system  `[stands]`
**ALIGNS → `coach`, `policy`**
The most-dependent members are *more* overtly anxious than the drinker. The spiral (threat → anxiety →
criticism → isolation → drinking → anxiety) terminates in functional collapse or a chronic plateau, and "any
one significant family member who can cool the anxious response" de-escalates it. Complete remissions with
the husband never attending; two favourable outcomes worked entirely through an oldest daughter.
*Model impact:* outcome predicted by **basic differentiation, explicitly not by consumption volume** — two
variables with different time constants: a near-fixed basic level capping the range, and a fast
relationship-driven functional level carrying the symptom.

### L12.4 — The closeness CEILING: a two-sided band  `[new at p2]`
**UNMODELLED → `Relationship`, `move:*`**
"A narrow margin between too much closeness and too much emotional isolation."
*Model impact:* a **two-sided band**, not a one-sided threshold — so a move that reduces isolation must
target a band, and **over-correction is a named failure state**. Nothing in the model penalises overshoot.

---

## Ch13 — Societal Regression as Viewed Through Family Systems Theory  *(1974a)*

**Provenance:** no clinical series at all. Origin in a 1972–73 EPA invitation.

### L13.1 — Society→family coupling is a SYMPTOM-THRESHOLD SHIFT, not a new mechanism  `[stands]`
**ALIGNS → societal dials** — *vindicates the three-scalar decision*
"Changing societal attitudes creates an environment that encourages behavior problems that would not have
previously been symptomatic… a regression increases the incidence of human problems." The same family at
unchanged differentiation produces symptoms in a regressed era it would not otherwise.
*Verbatim magnitude:* **ten scale points of societal regression in twenty-five years** against a population
spread of about **fifty points**.
**⛔ "~0.4 scale points per year" is manufactured and withdrawn.** No annual rate is stated and **none can be
derived** — Bowen's own curve is non-monotone with an *upward* final segment, so dividing one figure by the
other contradicts his chart.
*Model impact:* society belongs as a dial on the **symptom threshold**, not as a separately simulated layer.

### L13.2 — Togetherness ratchets, but only while anxiety continues  `[narrowed]`
**CONTRADICTS → `Person`, `Family`**
Each won togetherness contest establishes a persistent new "norm", so regression is stepwise and
path-dependent. And: "There is never a threat of too much individuality. The human need for togetherness
prevents going beyond a critical point."
**Narrowing — pass 1 read the ratchet as irreversible by default, which would license a model with no
restoring force at all.** Bowen qualifies it one sentence later: "In calmer periods the shift can go back and
forth, with neither overriding for long periods." **Persistence is conditional on continuing anxiety.**
*What still stands:* the **hard cap on the individuality side** is unconditional, and regression is triggered
specifically by **decisions taken to allay the anxiety of the moment** — not by anxiety itself — and only by
*chronic* anxiety. Acute anxiety produces regression that reverses on its own.
*Model impact:* a **conditional** ratchet with an asymmetric cap. The repo's symmetric `update_c` recovery is
still wrong, but a model with no restoring force would be wrong in the other direction.

### L13.3 — The differentiation move is a state machine with a scripted escalation peak  `[corrected]`
**REFINES → `move:I-POSITION`, `coach`, `test` 5**
Define self → immediate opposition ("selfish and mean and does not love the others") → hold course → a
**final intense emotional attack** → if the mover stays calm, "the opposition becomes calm and pulls up to
**his** level of individuality," and then "another, and another will do the same."
**Three corrections.** (1) The abort branches — defend, counterattack, go silent — sit at the **first**
opposition and are the **usual** response, not an exception. (2) **Anger is not a fourth branch; it is the
gate** admitting the sequence to the peak: "when he is finally able to maintain his course without getting
angry… the opposition does a final intense emotional attack." (3) The contagion claim is real, and the payoff
is that the opposition **pulls up to the mover's level** — stronger than a group-mean increment.
*Model impact:* I-POSITION is a multi-step sequence with a calm gate at the peak, a usual-case abort at the
first opposition, a whole-family payoff, and contagion to subsequent movers. Success usually follows several
failures.

### L13.4 — Gaps and disclaimed figures  `[stands]`
**UNMODELLED**
The only stated *reversal* mechanism at societal scale requires a single principled leader who can assemble a
team — which a scalar-dial society cannot represent. Sibling position is entirely absent. **The 55/60
togetherness figures are explicitly disclaimed as illustrative and are not usable for calibration.**

### L13.5 — The counterfeit move  `[new at p2]`
**UNMODELLED → `move:I-POSITION`, C10**
"At this level of differentiation, to 'stand up to' means to attack and shock the other with language and
behavior, and to get away with breaking rules." **The same act is a differentiating move at one level and an
assault at another.**
*Model impact:* with Ch20's blame-versus-responsibility pair and Ch21's outside-ness gate, this is **three
chapters, three settings, one mechanism** (five with Ch10's fifth band and Ch19's token concurrence): **an
act's identity depends on the actor's state, not on the act.** That is C10.

### L13.6 — Ch13's "Stream B" therapy rules  `[withdrawn]`
**⛔ ALMOST ENTIRELY PASS-1 INFERENCE.** The chapter contains **no therapist action, session, case, or
prescription**. "Primary emphasis on how the parents think and act" is the *study's* analytic emphasis;
"distinguish symptoms of progress from regression" is a *measurement* problem about society. Both were
converted into therapy rules Bowen never states.

### L13.7 — Q-MATERIAL, first hit  `[new at p2]`
The parents' **"ability to provide material demands"** terminates the escalation loop. It is a **per-family
capacity**, not a per-person financial stock. Societal economic security appears separately as an anxiety
*source*.

---

## Ch14 — Family Therapy After Twenty Years  *(1975)*

**Provenance:** retrospective; no new series.
**⚠ Ch14 is the chapter the project would naturally trust most, and it is the least reliable in the corpus.**
See L14.5. ~40% of it is field history and meta-commentary.
*Concept count:* **exactly six**, same membership as Ch21, with the extended-family and social-system
concepts **explicitly deferred** — they "will be added to the theory at a later time." **Ch14 advocates the
extended-family method its own concept list does not cover.**

### L14.1 — The same message carries different gain depending on its route  `[stands — but the "reversal" is misdated]`
**UNMODELLED → `event`, routing**
Each spouse talks **to the therapist, never to each other**: "Even when the emotional climate is calm, direct
communication can increase the emotional tension."
*Model impact:* **direct dyad amplifies, routed through a neutral third damps.** With L04.4 (source position)
and L01.5 (per-hop fidelity), routing is the third property of an event the model does not represent.
**Correction to the history, not the mechanism:** Ch14 flags this as "a major change from earlier methods".
**It is not new here.** Ch10 (1971) states it in full with the technique "in use more than five years"
(~1965), and Ch11 dates the change to "after about 1962" (L10.14). Ch14 gives no year, no first person, and
no admission the "earlier methods" were his.
*Also unreconciled inside Ch14:* he credits family group therapy's fast results to getting members talking to
each other — and so does Ch10.

### L14.2 — Satisfaction and symptom relief can diverge from structural change  `[narrowed]`
**CONTRADICTS → `coach` objective, `readout`**
25–40 appointments over about a year yielded "the aggressive mother becoming less aggressive, the passive
father less passive, and the child's symptoms much improved… with high praise for family therapy but with no
basic change in the family problem." He now removes the child entirely.
**⛔ "Gets worse before it gets better" is withdrawn from this chapter.** Pass 1 built it from "a temporary
increase in the child's symptoms". The full text is "**at the risk of** a temporary increase" — a risk clause,
not an observation, with no frequency, duration or case, and the phrase occurs **exactly once in all 22
chapters**.
**The requirement survives and must be re-sourced to L02.3 and L03.3**, which state it as an observation.
*What Ch14 does support:* an outcome metric where satisfaction and symptom relief **diverge from** structural
change. That much is stated plainly here.

### L14.3 — The extended family displaces the nuclear family as the site of work  `[narrowed]`
**REFINES → architecture, `coach`**
Residents doing family-of-origin work while in **no formal psychotherapy** "had made as much progress with
spouses and children as similar residents in formal weekly family therapy". He calls it "the beginning of a
new era in my own professional orientation"; the extended-family approach "bypasses the nuclear family… It
appears to produce better results."
**Narrowing — the comparison group was not uncoached.** Ch22 supplies the dose: **15–30 minutes of coaching
every month or two**, plus a weekly conference. So the finding is **dose reduction, not intervention
removal**. And **"bypasses the nuclear family" names the site of *work*, not of outcome** — the gains still
land on spouse and children.
*What stands:* gains transfer automatically downstream; monthly coaching works as well as or better than
weekly; "it appears to take a certain amount of time, on the calendar." A model where more coaching means
faster progress contradicts him. Notes against Phase B's four-person nuclear-only spike.

### L14.4 — A failed prediction of his own, and a concession  `[stands]`
**→ discipline**
The 1957–58 "healthy unstructured state of chaos" was called healthy on the premise that clinical experience
would force theoretical clarification: "This has not evolved to the degree it was predicted." And in summary:
"The type of approach is not a positive index of success in therapy"; family therapy "is still more of an art
than a science" — therapist skill may dominate method choice, which cuts against any model where method fully
determines outcome.
**No numeric anchors for the differentiation scale are given here.**

### L14.5 — Ch14's retrospective contradicts his own earlier text, in seven places  `[new at p2]` — **the single most valuable output of pass 2**
**→ discipline, evolution**
1. **Ch03 inverts Ch14's diagnostic signs.** Ch03's basic rule: the family works on its own problem "while the
   therapist observes from the sideline", and the *most frequent avoidance move* is the decision-maker
   engaging the therapist. Ch14 makes **exactly that the prescribed structure**, and makes spouse-to-spouse
   talk "evidence of building emotional tension." **It is a reversal of what the same act means, and he never
   says his earlier text assigned the opposite sign.**
2. **Ch03 says the threesome is where progress happens** — "any two members can successfully avoid anxiety
   issues and the therapy becomes more intellectual, more sterile." Ch14 abandons the threesome for the
   twosome. **This directly opposes L02.4 and L03.4.** Unresolved; carried into pass 3.
3. **Ch11 dates the change Ch14 leaves undated** — "after about 1962, I stopped suggesting that they talk
   directly to each other."
4. **Ch11's "about 50 percent faster" becomes "a little faster"** in Ch14, unremarked — and the proposed
   mechanism changes too.
5. **Ch10's numbers differ for the same series:** 35–45 sessions → 25–40; "the therapist **may** see no basic
   change" → "**with no** basic change"; plateau 12–20 → 10–20.
6. **Ch22 loses its hedge and its dosage:** "Observations suggested… as much **or more** progress" becomes
   "In 1968 I **discovered**… as much progress", with the weekly conference and the 15–30-minutes-every-month-
   or-two dropped.
7. **The 1968 comparison group was not uncoached** — see L14.3.
*Model impact:* **prefer the contemporaneous chapter over Ch14 wherever they disagree.**

### L14.6 — Q-MATERIAL: two budgeted resources  `[new at p2]`
**Professional time** (coach side) and **motivation** (person side, falling with symptom relief). Neither is a
per-person financial stock. See L22.5 for why the motivation one must not be over-formalised.

---

## Ch15 — Family Reaction to Death  *(1976)*

**Provenance:** clinical vignettes, no series, no counts.

### L15.1 — Integration inverts the OVERT response; the delayed channel is the better-evidenced half  `[narrowed]`
**CONTRADICTS → `Person`, `readout`**
"A well integrated family may show more overt reactiveness at the moment of change but adapt to it rather
quickly. A less integrated family may show little reaction at the time and respond later with symptoms of
physical illness, emotional illness, or social misbehavior."
**Narrowing, three ways.** (1) **"Low integration ⇒ larger total disturbance" is not in the chapter** — the
sentence pass 1 quoted governs **recovery time only**. (2) The two-governor sentence uses **"or"** and gives
**no direction**; pass 1's model section turned it into a product. (3) Hedges were stripped throughout ("may
show", "can be impossible", "may have been a burden"). The inverted quantity is **overt** reaction, not
total, and the inversion rests on **one hedged sentence with no case, count or rate**.
*Model impact:* amplitude and recovery time are separate functions of integration, and **the quiet immediate
response is the dangerous case**. A test should assert **the better-evidenced half — the delayed-symptom
channel** — not the inversion.

### L15.2 — The shock wave travels multi-hop through asymptomatic carriers  `[corrected]`
**REFINES → `Family`, belief layer**
Symptoms appear "anywhere in the extended family system in the months or years following"; "the grandchild is
often one who had little direct emotional attachment to the grandparents." Documented two-hop case:
grandmother dies → daughter shows only ordinary grief but is deeply affected → son who was never close to the
grandmother becomes delinquent. A *threatened* death (mastectomy) produced four affected branches within two
years, symptoms persisting at five, with no member removed at all.
**⛔ Pass 1 invented a second edge type.** Bowen does **not** say the dependence runs on a different graph; he
says it is **denied — invisible to the family**. The unattached grandson is explained by the **two-hop path he
supplies** (grandmother → daughter → son).
*Model impact, corrected:* **a belief layer plus multi-hop propagation on the ordinary graph** — folds into
L09.4. **No dependence-vs-attachment edge weight.**

### L15.3 — Loss impact is scaled by structural importance — three tiers, not a ladder  `[narrowed]`
**UNMODELLED → `Person`, `event-kind`**
**⛔ The seven-rung severity ladder was not Bowen's.** He gives **three tiers** — shock-wave-likely / neutral /
relief — with an **unranked** list inside the top tier. Pass 1 presented an ordinal ladder as "Bowen's own
ordering" **and silently swapped the grandmother pair so it would descend** (the text puts shadow before
central). It was being used as a calibration ladder. Withdrawn as a ladder.
**What survives:** the three tiers, and two pairwise relations — **central > shadow grandmother**, and
**dysfunction that was load-bearing produces a shock wave where ordinary dysfunction does not**. A burdensome,
non-critical member's death is followed by **improved** family functioning. Suicide produces prolonged grief
but a **minor** shock wave.
*Model impact:* grief magnitude and system-disturbance magnitude are **separate output channels**, and a
person needs a structural-importance attribute distinct from their level of functioning. Three tiers is what
can be calibrated; anything finer is invention.

### L15.4 — Negative finding, recorded so a later pass does not invent it  `[stands]`
**→ discipline**
There is **no anniversary effect** in this chapter. Delayed and prolonged effects are reported (2-year chains,
symptoms at 5 years, mourning 6 years late) but **no date-anchored recurrence is named**.

### L15.5 — Q-MATERIAL: poverty gates the FORM of contact  `[new at p2]`
No per-person resource variable. Poverty gates the funeral form — "we are poor people. We can't afford a
mausoleum" — **the only causal material input found in this chapter, and it acts on contact-directness**, not
on functioning. Economic failure appears three times as a shock-wave *output*.

---

## Ch16 — Theory in the Practice of Psychotherapy  *(1976)*

**Provenance:** NIMH in part. Cites the 1975 renaming to "the Bowen theory"; cutoff and societal regression
added in 1975 → **eight** concepts.
**📌 Ch16 is the best source for STRUCTURE and the worst for DYNAMICS.** It drops every rate, latency and
dosage in the corpus — no time constants, no reaction ladder, none of Ch21's efficacy gates, none of
Ch11/Ch22's frequency data, and C1 only in weak form. **Its silence must not be read as a constraint.**

### L16.1 — THE SCALE — and three corrections to how pass 1 read it  `[corrected]`
**CONTRADICTS → `Person`**
0–100. 0 = lowest possible human functioning; 100 = a hypothetical perfection. Profiles at **0–25, 25–50,
50–75, 75–100**. **75–100 is explicitly withdrawn** as "more hypothetical than real."
**Correction 1 — the 50 threshold was truncated, and the mechanism is different.** Pass 1 read "below 50 the
emotional system tells the intellect what to think" as suppression. The full text continues: "The intellect is
a pretend intellect. The emotional system permits the intellect to go off into a corner… **as long as it does
not interfere in joint decisions that affect the total life course**." That is a **licence, not suppression**
— materially different to implement.
**Correction 2 — the quartile dating was wrong in both directions.** Ch16 says the profiles were done in an
earlier paper and are "still amazingly accurate **ten years later**" (≈1966) — matching **Ch09, which has
them**. The real history is **non-monotonic**: quartiles present 1966 (Ch09) → **absent** 1972 (Ch21) →
present again 1976 (Ch16). Pass 1's "arrives between 1970 and 1976" is wrong.
**Correction 3 — Ch16 is the only source for the numbers.** Ch17 has no bands at all — only "0 to 100", "four
detailed profiles", "lower half", "third segment". **The scale gets *less* specified over time.**
*Model impact:* the repo's `C ∈ [10, 80]` with a linear clip has no threshold at all. The real structure is a
**0–100 scale with one behavioural transition at 50 — a licence over joint decisions — and an unusable top
quartile**. And note **Ch21 has no regime change at 50 at all** (L21.1): the threshold is a later import, not a
constant of the theory.
*The honest caveat, also corrected:* Ch16 says the letters "**slowed down** my effort to develop a more
definite scale that could be used clinically" — a **delay**, not abandonment. Ch17 reports the **stopping**,
and for **misuse, not invalidity** ("Thus far I do not see enough disadvantage to try to modify the 'scale'").
**No external validation is offered anywhere in the book.**

### L16.2 — The sink count is THREE, with distance an always-on baseline OUTSIDE the budget  `[corrected]`
**REFINES → conservation invariant**
(a) **Dyadic:** "In any exchange, one gives up a little self to the other, who gains an equal amount" —
zero-sum, and operating on **pseudo-self only**, since solid self does not participate in fusion.
(b) **Family-level:** "a quantitative amount of undifferentiation to be absorbed in the nuclear family",
across **marital conflict / spouse dysfunction / child projection**.
**⛔ There was never a three→four sink transition. Pass 1 manufactured it.** Ch16 is explicit: "**Other than
the emotional distance**, there are three major areas…" — pass 1 inserted the word *absorbing* to make
distance a fourth sink. **Four chapters agree: three sinks plus a universal leaky baseline.** Ch12 lists four
patterns and counts three, calling distance "almost universal"; Ch09 has distance a paragraph earlier and
uncounted; Ch18 enumerates twice and disagrees with itself (see L18.1).
*Severity depends on **concentration**, not total* — "the more the problem shifts from one area to another the
less chance the process will be crippling in any single area" — but see L10.2's narrowing: **total governs the
quantity of trouble, distribution governs the response to therapy.**
*And a falsifier restated:* marital conflict does **not** by itself harm children — only the projection
channel does. Corroborates L09.2.
*Model impact:* the conserved quantity is **pseudo-self**, not self. Solid self is exempt. Distance is
**always-on and outside the budget** — it must not be implemented as a competing sink.

### L16.3 — The generational update rule, and the cutoff positive-feedback loop  `[stands]`
**REFINES → `Person`, multigenerational**
Primary projection object emerges **lower** than the parents; minimally involved siblings **about the same**;
those relatively outside the emotional process **better**. Rate is **stochastic** — fast for a few
generations, static for one or two, then fast again — and reversible at both extremes.
*The clearest positive feedback loop in the corpus:* the more intense the cutoff, the more exaggerated the
parental problem in one's own marriage, and the more intense the next generation's cutoff.
*Calibration:* "**eight to ten generations**" to schizophrenia — revised **up** from "at least three"; a
five-or-six-generation stress produces a social failure less impaired than a schizophrenic. **No per-generation
rate is stated here or anywhere** — see L09.7.
*A clean controlled comparison, ready to implement as a test:* two families at identical levels — one keeping
contact with the parental family (symptom-free for life, level preserved next generation), one cutting off
(symptoms, dysfunction, lower level next generation).

### L16.4 — Overt emotionality PEAKS IN THE MIDDLE of the scale  `[new at p2]`
**CONTRADICTS → `readout`**
"People in the moderate range of differentiation have the most intense versions of overt feeling."
*Model impact:* pass 1 read visible emotionality as monotonically decreasing with level. **Any readout using
overt emotionality as a proxy for level is wrong in exactly the band where most agents sit.** With L10.12 and
C10, this is the second reason a behavioural readout of differentiation is unreliable.

### L16.5 — Q-MOVEMENT: the fixing point, and "only a functional shift" withdrawn  `[new at p2]`
**REFINES → `Person`**
The determination point moves **earlier** than Ch21's: differentiation is set when **the young adult
establishes self separately from his family of origin**, not on consolidation in a marriage — Ch16 and Ch17
agree, against Ch21. And **"only a functional shift" is withdrawn**: "it is possible to make slow changes."
Lifetime range: "not possible ever to make more than minor changes" — unquantified.
*See L22.7 for the full cross-corpus table; Ch21's freeze-at-marriage is outvoted three to one.*

---

## Ch17 — An Interview With Murray Bowen  *(1976)*

**Provenance:** no clinical series.
**⚠ C2 and C3 must not cite Ch17** — zero representation of the conservation machinery (*marital*, *spouse*,
*distance*, *reciproc-*, *absorb* all zero occurrences).

### L17.1 — Anxiety-driven expansion and contraction of the system boundary  `[stands]`
**UNMODELLED → architecture** — *corroborates L07.1*
Complete reversible sequence: calm twosome → triangulation at a threshold → anxiety dilution across three
edges → overflow into interlocking triangles, with the displaced member becoming **emotionally inactive** →
spread outside the family to neighbours, schools, agencies, courts → **reversion to the original triangle on
subsidence**.
*Model impact:* the agent population and the system boundary are themselves functions of anxiety. And
**triangles are latent when calm** — "the system is calm and the triangles inoperative" — so persistent
topology must be stored separately from currently-active triangles.

### L17.2 — Two variables only; reactivity is DERIVED; and the measurement is biased  `[stands, one clause narrowed]`
**CONTRADICTS → `Person`, `readout`**
Level of self and level of anxiety are the only two major variables — **reactivity is derived**, not a third
state ("the lower the level of self, the more reactive"). Functional level is superimposed, fluctuates widely,
and **its variance is itself a function of the basic level**.
*The measurement model:* pseudo-self is traded between people — "lending, borrowing, trading… at the expense of
another" — which "**results in false readings when one attempts to estimate levels of differentiation**." Any
coach-side or readout-side estimate of differentiation must be **biased and noisy**, recoverable only over a
life course.
*Population skew:* ~90% in the lower half, ≤10% in the third segment; the top profile is explicitly
**extrapolation, not observation**. **Ch17 gives no bands** — see L16.1 correction 3.
*Narrowed clause:* basic level is fixed once "the young adult establishes self separately from his family of
origin" — but **that fixing point is disputed across the corpus** and Ch16 says slow changes remain possible.
See L16.5 and L22.7.
*Model impact:* the repo carries C and TX as independent state columns. Reactivity should be **derived** from
self and anxiety, not stored.

### L17.3 — The coach is a node inside the system, and efficacy is CONJOINTLY gated  `[stands]`
**REFINES → `coach`, `test` 7**
"The groupings were different when the therapist was not a part of the emotional responsiveness" — the
observer changes the configuration, so **a costless external coach models something Bowen says does not
exist**. Efficacy requires three conditions **together**: accurate observation, really knowing the system, and
control of one's own emotional inputs.
*And a hard zero:* a well-formed intervention delivered to an **unready** recipient produces **zero** change,
not a small one — stated three times ("I do my explanation and the questioner stops asking questions but it
does not change their thinking").

### L17.4 — His own retractions  `[corrected — one misquote]`
**→ discipline**
Abandons "undifferentiated family ego mass" as conceptually inaccurate. Calls **"triangle" his "most
unfortunate term"**, with no replacement he can name. **Stopped his own scale-level research** — for **misuse,
not invalidity** (L16.1).
**⛔ Misquote corrected.** Pass 1 recorded multigenerational transmission as "**the** concept on which I have
done the least detailed work". He says "**one of the concepts** on which I have done the least detailed work,
and one that needs the most attention" — **and he is recruiting a researcher for it.** Still a flag on the
model's generational spine, but not the superlative pass 1 reported.
*Ego-mass term arc:* used → discarded on *conceptual* grounds (Ch05) → **retained and defended** (Ch08) →
revived on purely *pragmatic* grounds ("because of its usefulness") → abandoned again on conceptual grounds
(Ch17). **The objection is overridden once and never answered.** The arc is messier than pass 1's four-step
story and must be re-derived at pass 3.

### L17.5 — Bowen's methodological preference against inanimate-science models  `[narrowed — pass 1 and the proposal both overstated it]`
**→ framing**
Pass 1 recorded, and the proposal repeats, that Bowen "deliberately excluded models from the sciences of
inanimate things" as an objection to this whole class of model. **Ch17 gives the full scope, and it is much
narrower:**
- "**For my research**"
- "This decision governed nothing except the **background thinking of the research staff**"
- "**without saying this should be done**"
- grounded on "no more than an educated guess"
- and **in the same interview he concedes the charge against his own *triangle*** — "It sounds almost
  mathematical." / "You are correct."
*How to record it:* **a named methodological preference, scoped to his own research staff's background
thinking.** **Do not quote it as "Bowen ruled out this class of model."** It still belongs in the proposal's
framing section — accurately.

---

## Ch18 — Society, Crisis, and Systems Theory  *(1973)*

**Provenance:** no clinical series. EPA commission 1972.

### L18.1 — A three-input intensity function, discharging into THREE sinks  `[corrected]`
**REFINES → conservation invariant, architecture**
Intensity governed by (i) degree of undifferentiation, (ii) **degree of emotional cutoff with families of
origin**, (iii) degree of stress. Conservation stated directly: what fuses with the child is set by her total
undifferentiation "**and by the amount absorbed elsewhere**."
**⛔ The four-sink reading is withdrawn.** Ch18 **enumerates twice and disagrees with itself**: the concept
list has four lettered items with distance as (a); forty lines later the clinical exposition calls distance
"one way that almost all use… hard to maintain over time" and then gives "**three important patterns**".
**Distance is never promoted to a peer mechanism** — here or in Ch09, Ch12 or Ch16. See L16.2.
*What stands, and it is the important part:* **cutoff is an INPUT, not an outcome** — so the model **cannot be
closed at the household**. Corroborates L14.3 and argues against Phase B's four-person nuclear spike.

### L18.2 — Anxiety is the master gain, and symptom onset depends on CHRONICITY  `[corrected]`
**CONTRADICTS → `Person`, `policy`, appraisal**
All patterns intensify with anxiety and **vanish when calm** — "the functioning of the triangle patterns is not
observable in a completely calm system." Symptom onset depends on **chronicity, not instantaneous level**: "Any
unit can recover from periodic panic or overloads, but when the panic becomes chronic one or more of the
individual units can collapse."
**⛔ The "three anxiety regimes" are a claim about the OBSERVER, not the system.** Ch18's invisible / legible /
chaotic triple describes "**the ability to observe and 'see' emotional reflexes**". **The chapter's dynamics
claim is monotone.** Implemented as pass 1 wrote it, this would give the model a spurious high-anxiety regime
Bowen does not assert.
*What stands:* an **integrator, not a threshold test**. And anxiety **gates information transfer** — above
threshold, content is not heard, only defended against. *(Note: pass 1 paired this with Ch20's "content is
noise", which is withdrawn — see L20.3. The gating claim stands on its own.)*

### L18.3 — Distance relief comes from AVAILABILITY, not from a depletable stock  `[corrected]`
**UNMODELLED → `Family`, societal dials, access vector**
Pass 1: frontiers → colonies → mobile jobs → nothing; a society-level reservoir the distancing move draws down.
**⛔ Withdrawn. There is no per-move draw-down anywhere in the text.** The operative sentence is: "It was
important for him to know there was **new land for him, even if he never went to it**." **The relief comes
from availability knowledge, not consumption.**
*And this converges with L11.5* — "what was important was that the system was open and they could attend if
they wished", stated after attendance had collapsed. **Two chapters, one mechanism: availability itself carries
the effect, independent of use.** Implement as a **belief about what is available**, not a stock and not a
gate on moves.

### L18.4 — Named downward channels from society to family  `[stands]`
**ALIGNS → societal dials**
Labelling/diagnosis propensity (fixes the problem in the patient, absolves the family), overleniency of
officials and laws, helping-programme intensity (**impairs recipients** — C1 at societal scale), school
structure at junior high, population density, era-dependent symptom form.
*Timescale:* decades — twenty-five years of decline since WWII, possibly cyclical. He asks explicitly that
"crisis" be replaced by "a term implying a long-term process."

### L18.5 — The decoupling guard: the dial must not be an unavoidable multiplier  `[stands]`
**→ `test`**
Well-differentiated families "come to function far better than the societal level." If turning the societal
dial moves every family proportionally, the implementation is wrong.

### L18.6 — Q-MATERIAL: four hits, none of them a per-person stock  `[new at p2]`
Family reserves "**exhausted**" by the projection process with spending gated on non-improvement; parental
money and privileges as an **appeasement transfer**; **work as a destabiliser of fusion** ("as long as neither
works"); job mobility as a per-spouse attribute. Real, and none of them supports the `M` column.

---

## Ch19 — Problems of Medical Practice Presented by Families With a Schizophrenic Member  *(1959b, with R. H. Dysinger)*

**Provenance:** **the same NIMH residential project as Ch01, Ch05 and Ch08.** C6 counts Ch01 and Ch19 as two
independent chapters; corrected — **C6 is 5 independent, not 6.**
**Chronology:** 1959b. This is one of the **earliest** papers in the volume despite its late position in the
book's ordering.
**⚠ Three of Ch19's B5 "implied mechanisms" were inventions written in the chapter's voice** — marked
`⚠ NOT SOURCED` in `ch19.md`. One had propagated into a proposed feedback loop and a test; both withdrawn.
**⚠ The hedge "seemed" is house style, not signal** — pass 1's own X2 lists **ten identical hedges in 2,600
words**, including on claims the authors plainly hold.

### L19.1 — The professional's own dysfunction is the readout  `[stands]`
**UNMODELLED → external agents**
Difficulties in the medical situation are studied "in much the same way that difficulties in psychotherapy are
studied as derivatives of emotional processes." The pressure is an **induction**: the family works to get the
doctor to assume responsibility "by virtue of a diagnosis" — of illness in the acting-out-helplessness mode, of
health in the denial mode.
*Model impact:* the external agent needs its **own drifting state**, and the *direction* of its error is itself
an observable. With L10.10 and L20.2, three chapters and three different signal sources.

### L19.2 — The critical point — three triggers, and the "third branch" is not in the chapter  `[corrected]`
**UNMODELLED → `coach`, `event-kind`, `test`**
When medical finding diverges from the family's emotional position, the family creates "an unmistakable
impression that he found the doctor unsatisfactory," and the doctor seemed to be faced with the choice of
losing working contact or compromising his judgment.
**⛔ Correction 1 — there is no stated third branch.** Pass 1 recorded "if the doctor holds his judgment
through the critical point, then…" as the chapter's technique and its only stated good outcome. The chapter
supplies **only the consequent** — "This anxious encounter between the emotional process and medical judgment
could then resolve toward a more adequate recognition of the actual problems" — whose subject is **the
encounter**, not the doctor. **No verb of holding appears anywhere.** Pass 1 also read the hedge "*seemed*" as
marking a deliberate false dilemma; it is house style.
**⛔ Correction 2 — a definite diagnosis is the REWARD branch, not a critical point.** Pass 1 listed four
triggers; the chapter names **three**. A definite diagnosis "tended to be seen as the source of all problems",
and any physical finding is "taken as at least a token concurrence". **Pass 1 put a rupture where the chapter
puts a payoff.**
*Keep the design decision — **holding must not be implemented as a variant of conceding** — but re-source it to
Ch10, Ch13, Ch20 and Ch21, and stop citing "seemed".*

### L19.3 — Three concrete moves by which a family manipulates the professional field  `[stands, one claim narrowed]`
**UNMODELLED → `move:*` for family→external ties**
(a) **Splitting** — the mother consulted outside gynecologists with no mention on the ward, withheld her
history so the opinion was uncheckable, reported nothing back; it surfaced only because that doctor telephoned
unprompted.
(b) **Frame ambiguity** — health talk as social greeting, symptoms spoken to others in the doctor's presence,
ailments arriving as rumour, after which "his response **or lack of it**" carried the weight of a professional
opinion. **Silence is captured too — abstention is not an exit.**
(c) **Displacement** — a second member taking over another's contact with the doctor; the mild form is to
"simply invite himself to be also present."
*Narrowing:* the intensity ranking is scoped "of all the situations **involving more than one member**" — there
is **no multi-versus-single comparison** in the chapter. And severity-independence is a **separate claim about a
different quantity** (difficulty of the medical work), not about intensity.

### L19.4 — Mode is role-BIASED, not role-determined  `[narrowed]`
**REFINES → `policy`, position**
Mothers and the symptomatic child amplify; fathers deny.
**Narrowing:** mode 1 is "also common for the fathers"; mode 2 "occurred also in mothers". **There is exactly
one hard constraint: the child never denies.** Siblings are "**rarely**" affected, **not exempt** — **and they
were not resident members of the treatment unit**, an exposure confound pass 1 missed.
*The position-over-person inference is not this chapter's* — re-source to Ch16 and Ch10.

### L19.5 — "Token concurrence" — the only continuous grading in the corpus  `[new at p2]`
**UNMODELLED → `event-kind`, `readout`**
Concession is a **scalar the family scores continuously** against ordinary competent acts, not a binary. Any
physical finding is "taken as at least a token concurrence".
*Model impact:* this is the one place Bowen grades anything continuously, and it grades **an act's concessive
value as read by the receiver** — which is C10's discriminator from the receiving side.

### L19.6 — Truth value and emotional function are ORTHOGONAL  `[new at p2]`
**UNMODELLED → belief layer, `readout`**
"The estimate might be accurate or not and yet be largely in the service of a denial."
*Model impact:* **forbids the shortcut of representing an emotionally driven claim as a false one.** A claim
can be correct and still be doing emotional work. Constrains the belief layer of L09.4 directly.

### L19.7 — Q-MATERIAL, and an affordance invisible to the author  `[new at p2]`
None. No fee, cost, insurance or means anywhere. **Free-at-point-of-use access is what makes the family's
splitting move cheap — and the chapter does not notice.** The affordance is invisible to Bowen precisely
because the setting removed it. Relevant to any access-vector design: cost is a real gate on the splitting
move that the corpus cannot calibrate.

---

## Ch20 — Toward the Differentiation of Self in Administrative Systems  *(1972)*

**Provenance:** **⚠ Ch20 contains ZERO cases.** The faculty-overcriticism claim was recorded by pass 1 as
having an observational history. It has none: the case is announced ("A clear-cut issue… will illustrate") and
**never delivered**, and every verb in the section is habitual ("would become overcritical", "I often
noticed").
**Q-MATERIAL: the strongest negative in the corpus.** None, **and this is the only workplace chapter**. Tie
weight is set by emotional importance, **explicitly not by economic relation**. If a material resource variable
were going to appear anywhere in this book, it would be here.

### L20.1 — The leadership office is general; sphere-of-responsibility is a PERMISSION  `[corrected]`
**REFINES → `Family`, `Person`, `policy`** — *generalises L03.6*
Differentiation, triangles, anxiety transmission, over/under-functioning and process-vs-content all carry
across to work systems unchanged; the only imported mechanism is "clearly defined contracts." Bosses are graded
on differentiation exactly as parents — feeling-of-the-moment vs. principle-and-reality decisions — and the
mapping is stated **bidirectionally**.
**⛔ Correction 1 — the transfer direction was reversed by pass 1.** The import runs **family → work**, with
good-administration principles as the pre-existing base — not work → family. Everything transfers unchanged;
nothing is modified or excluded; the only qualifications are lower work intensity and "it is not a family."
**⛔ Correction 2 — sphere-of-responsibility is a permission, not a propagation boundary.** "The person who
works toward the differentiation of self does not have to be the boss… His effort can be effective in the area
in which he has administrative responsibility" **relaxes an authority precondition** and says nothing about
effects stopping at a boundary — while the headline claim asserts resolution of "the problem in **the
organization**". **L20.1's "propagation bounded by it" is withdrawn.** Keep sphere as a **move permission
scope** only.

### L20.2 — The coach's own reactive state is a primary detector  `[narrowed]`
**UNMODELLED → `coach`**
Bowen's earliest reliable indicator of rising system anxiety is **his own urge to become critical** of the
faculty for being overcritical of trainees. Prescribed sequence: restrain the urge → assume he is playing a
part → work on himself → identify which of three named lapses applies (**lost contact / failed to state
position / failed to detriangle**) → modify it.
*Narrowing:* "I **often** noticed this first" — not always. The triangles readout is "**another** way of
detecting", so self-state is **one of two routes, not the only one**. And the three lapses are **recalled**,
not diagnosed in the moment.
*Model impact:* a stateless external coach loses one of the chapter's two detection channels.

### L20.3 — A fixed observable set whose RATE reads out system anxiety  `[corrected]`
**→ `readout`, architecture**
Emotional issues between people, withdrawal, silence, cliques, alliances, gossip about an absent member — "the
language of the triangles".
**⛔ "Content is explicitly designated noise" is NOT in the chapter, and is withdrawn.** Bowen gives two
**attention** prescriptions — "listen to the incidence… rather than focusing on the content of what is said"
and "avoid an unwitting focus on the content of issues". He **never says content lacks information**, and his
own blame-versus-responsibility hazard *requires* reading an "inner orientation of self" that no rate can
supply.
***This matters beyond the ledger.*** "Content is noise" was used in `agent_model_proposal.html` as an argument
for the deterministic core. **The argument survives only in its weaker, correct form: the anxiety readout is
computable from event rates alone.** It does **not** establish that content carries no signal. Fix the proposal.

### L20.4 — Two adversarial cases  `[stands]`
**→ `test`**
(a) Offering solutions "worked very well" in the short run yet built dependence and blocked staff from
developing responsibility — positive immediate, negative long-term. *Corroborates C1.*
(b) The biggest named hazard of the whole principle: accepting **"blame"/"fault"** instead of responsibility for
the part self plays — "a fine line". Two moves that are near-identical on the wire and opposite in effect; the
discriminator Bowen names is **"the inner orientation of self"**. This is C10 (L13.5, L21.2).
*Note:* "looks identical" and "produces none of the resolution effect" are **derived**, not textual.

### L20.5 — He weakens his own headline claim within the chapter  `[stands, sharpened]`
**→ discipline**
"Any time one key member of an organization can be responsibly responsible for self, the problem in the
organization will resolve" — stated **three** times, not two, and weakened **four** ways at the close: an added
antecedent, "some progress", sphere-scoping, and "will resolve" → "will work toward automatic resolution".
Needs an explicit propagation rule and a test that it actually fires.

### L20.6 — Four further findings  `[new at p2]`
- An undifferentiation sink **outside the family**: work relationships.
- An **organisation-level policy variable**.
- The leadership-office construct stated **bidirectionally** (family ↔ work).
- A projection readout located at the **sending** node, not the receiving one.

---

## Ch21 — On the Differentiation of Self  *(1972; presented 1967, written 1970)*

**Provenance:** **Bowen's own family, n = 1.** The richest technique chapter in the corpus and the thinnest
evidentially.

### L21.1 — The scale here has NO quartiles and NO regime change at 50  `[corrected]`
**REFINES → `Person`** — *evolution marker against Ch16*
Named divisions: 0, lower half, upper half, 50–75, above 60, 75, mid-90, 100. **75 is "a very high-level
person"; "those above 60 constitute a small percentage of society"**. The mid-90s originally expected for
historical figures was **revised away**. **The quartile scheme of Ch16 is absent here** — and it was *present*
in Ch09 (1966), so the history is non-monotonic (L16.1).
**⛔ Correction — pass 1 wrote "50–75 is the only band with a behavioural description."** It is the only
**numerically bounded** one. The **lower half**, the **upper half** and **"high-scale people"** all get
descriptions, **and the lower/upper transition is explicitly gradual**. **There is no regime change at 50 in
Ch21. Ch16's threshold is a later import, not a restatement.**
*The unreconciled contradiction, confirmed as unreconciled:* basic level is "consolidated in a marriage,
following which the only shift is a functional shift" — yet a completed differentiating exchange produces "a
basic increase in bilateral differentiation **which can never return to the former level**." The two statements
sit **~5,000 words apart in different sections and neither refers to the other**. The only hooks are two hedges
inside the immutability claim ("few", "unless there is some unusual circumstance") which Bowen never connects to
the move. **Ch21's freeze-at-marriage is outvoted by Ch16, Ch17 and Ch22** — see L22.7.
*Model impact:* a **ratcheted** `basic_level` moved only by a *completed* reaction cycle, plus a fast
bidirectional `functional_level`, plus conservation across a fused dyad — one spouse's functional gain equals
the other's loss, stated explicitly.
*Note the second ratchet:* Ch13 has togetherness ratcheting **up** under regression (conditionally — L13.2);
Ch21 has differentiation ratcheting **up** on completion. Two ratchets, opposite directions, different triggers.

### L21.2 — Efficacy is gated by hidden state — and the ally mechanism is DISPLACEMENT  `[corrected]`
**UNMODELLED → `move:I-POSITION`, `Person`**
**Gate 1 — secrecy.** A differentiating effort "routinely fails if anyone else knows anything about it."
**⛔ Pass 1's reason is wrong.** It explained secrecy by the family being a near-perfect channel for intent.
Bowen's own reason, in the same sentence: "each action and move must come from within the person who makes the
effort. **These decisions and actions often have to be made instantaneously and, for better or worse, the
individual has the responsibility.**" **Two mechanisms: sub-deliberative latency during the reaction, and
undivided ownership.** The absolute is also narrowed in the very next sentence to "another person **who is part
of the system**", and restated a third time hedged ("it is doubtful that any differentiation will result").
**Gate 2 — the ally. ⛔ "Ally cancellation" is OVER-READ. The named mechanism is DISPLACEMENT.**
Pass 1 reported this as the most counter-intuitive finding in the corpus: that support from a family member
*cancels* a differentiating move. **Bowen never says that.** What he says: he "worked out a plan that permitted
no 'allies'"; the purpose was to handle "all the potential peripheral triangles that could align themselves with
issues"; the countermeasure was "to keep the entire family in one big emotional clump, and to detriangle any
ally who tried to come over to my side"; and "in the past I had been 'undone' by partners."
**An ally opens a new peripheral triangle, and that is what he calls "my undoing."** Active detriangling of an
ally is verbatim and goes into the model as stated. **The cancellation term does not, and must never be cited as
a quotation.**
**Restated at step 6 (`_RESOLUTIONS.md` R2): the gated quantity is *alignment*, not knowledge or support.**
A third party who knows and takes no position carries no penalty. A third party who takes the mover's side
opens the triangle. The countermeasure returns them to neutral and leaves their knowledge intact.
**And there is a preemptive move the repertoire lacks:** the two deliberately contradictory letters were
"designed to prevent any one segment of the family from getting on my 'side'" — action on *potential*
alignment before it forms.
**Gate 3 — outside-ness.** Until the mover is partly outside the system, an identical technique is "either
hollow meaningless words or a hostile assault on the system, **and an emotional system knows the difference**."
**⛔ Outside-ness has THREE inputs, not one.** Pass 1 named private rehearsal as "the only stated mechanism" —
it is only the fast, move-scoped one. The other two: the **observation/control ratchet** ("Observation is not
possible until one can control one's reactions sufficiently to be able to observe… which in turn, in a series of
slow steps, allows for better observation") and **practice on a peripheral emotional system**.
*Model impact:* technique efficacy is multiplied by an `outside_ness` state **distinct from basic
differentiation**, with three inputs at three time scales. This is C10's gate.

### L21.3 — The announcement question, object-level: CLOSED  `[stands — confirmed by Ch20]`
Ch10's announcement rule concerns telling the *other party in a dyad* about a specific withdrawal. Ch21 and Ch22
concern *anyone knowing about the differentiation programme*. Different objects, so the apparent contradiction
(old L22.3) dissolves. **Ch20 independently confirms this** — its secrecy rule concerns the plan, matching Ch21
and Ch22. **Settled.**
**⚠ What is NOT settled:** Ch10 vs Ch21 on whether a **witness** helps or displaces. See L10.15.

### L21.4 — A timed, reproducible trace of one move and the system's re-stabilisation  `[stands verbatim, two framings corrected]`
**REFINES → `move:I-POSITION`, `test`, scenario** — *the most implementable passage in the book*
*Anxiety wave, with intervals:* brother-in-law's sudden death → **~2 weeks** → the projection-recipient sister
becomes symptomatic → **~2 weeks** → overt conflict over business stock → **~3 weeks** → second brother
immobilised with a herniated disc.
*The move:* trip scheduled ~2 months out; 8 weeks of planning; letter to the target brother mailed exactly
**T−14**; reading call to parents **T−7**; two **deliberately contradictory** letters to sister and mother
within the hour, sent specifically to prevent any segment allying with him; arrival Saturday midnight; a 2-hour
meeting Sunday in which "about two-thirds through… the family system had lost its emotional punch"; **mandatory
follow-up the next day** — "this is the point where the feeling system dictates withdrawal, which will result in
the system 'tightening up' again"; then re-stabilisation, the stock conflict "completely faded", and almost
three years of the family's best adaptation.
*The reaction ladder:* **"You are wrong" → "Change back" → "If you do not, these are the consequences"** — with
the diagnostic rule that **absence of reaction means the move did not land**.
**✅ Every interval above is verbatim-correct and must not be re-litigated.** So is the reaction ladder, the
diagnostic rule, and the six-concept enumeration of L21.5.
**Two framings corrected.** (1) **The wave enters from OUTSIDE the family** — the deceased was the *second
brother's wife's brother*, so the wave crosses in through a **marital link, from a family of origin that is not
the one being modelled**. And Bowen states the mechanism in advance: the anxiety "**amplifies minor problems
into major ones at vulnerable points**" — **the wave does not create pathology; it applies gain at nodes the
family's absorption profile has already named.** (2) The mandatory follow-up is next-day relative to the
**meeting**, not to arrival.
*Model impact:* `move:I-POSITION` becomes a state machine with a reaction ladder, a **follow-up window with
revert-on-timeout**, and a reaction that **decays unless fed** by a defend or counterattack from the mover
(L01.8 states the decay independently, in 1957). Converges with L13.3 at finer resolution.

### L21.5 — Concept enumeration is SIX  `[stands]`
**→ evolution**
Stated twice, in order: scale → nuclear family emotional system → family projection process → multigenerational
transmission process → sibling position profiles → **triangles, sixth**. **Emotional cutoff and societal process
are absent as named concepts** — though cutoff is fully described as an unnamed four-type taxonomy of fusion
styles. Six concepts in 1967/70 → **still six in Ch14 (1975), with the two additions explicitly deferred** →
eight by Ch16 (1976).

### L21.6 — Further unmodelled moves  `[stands]`
The deliberate low-stakes provocation ("tempest in a teapot"); the "stories" and "reversals" moves; the **clump**
communication topology; and the post-success role of "the differentiating one".
*`ch21.md`'s B5 carries 61 entries converting technique back into mechanism — the richest in the corpus.*

### L21.7 — The corpus's cleanest cross-system result  `[new at p2]`
**ALIGNS → architecture, L10.6, L22.4**
Pass 1 recorded that the old work system re-fused him in thirty minutes after two or three years away. **It
omitted what followed:** "**after I had mastered the experience with my own family… I returned to the old work
system for a long visit and was able to relate intimately to those important to the system without a single
episode of 'fusion.'**"
*A before/after on a system he never worked on* — **the strongest evidence in the corpus that differentiation is
a per-person scalar that transfers.**

### L21.8 — Sibling conflict is NOT dyadic  `[new at p2]`
**UNMODELLED → `Triangle`, `event-kind`** — a hard constraint
"Conflict between siblings consists **almost universally** of a triangle between mother and two children in
which mother has a positive relationship to each child and the conflict is fought out between the children."
*Model impact:* **a peer-conflict event must instantiate the parent-level triangle, and intervening on the peer
pair alone must fail.** Same displacement rule as L21.2's ally mechanism and the administrative-hierarchy
statement, one level down.

### L21.9 — Four further mechanisms, previously absent from the corpus  `[new at p2]`
- **Gossip volume rises as direct contact falls** — C3's rerouting at the *communication* layer, the only
  instance in the corpus.
- **Marital conflict has a second trigger: a chronic adapter who stops.** This is the general rule behind Ch12's
  conflict spike (L12.2).
- **Message intensity is a dial separate from content** — "terrible, awful, pressing, horrible… designed to stir
  up the 'tempest in a teapot'".
- **Two separable success criteria with different completion times and failure tolerances:** interrupting the
  wave **survives partial failure**; personal non-fusion is **all-or-nothing**.
- And: **deliberate perturbation is hard** — automatic disturbances are easy, purposeful ones are not.

### L21.10 — There are THREE therapeutic configurations, and the middle one reverses pass 1's reading  `[new at p2]`
**→ `coach`, L06.4**
Each has its own heading. The missing middle one is **"Family psychotherapy with one spouse in preparation for
family psychotherapy with both spouses"** — for families "in which one spouse is negative and unwilling", worked
"**until the unmotivated spouse is willing to join.**"
**The single-member method is an on-ramp as often as a substitute.** Pass 1 also truncated the quotation; in
full: "if I can get the couple to cooperate, then I do it with them, **and if I can't, then I work with the
motivated one**" — the reverse of pass 1's implication.

### L21.11 — Surprise is a THIRD secrecy mechanism  `[new at step 6]`
**REFINES → `move:I-POSITION`**
Pass 2 recorded two reasons for the secrecy rule — sub-deliberative latency and undivided ownership. There
is a third, stated earlier in the same chapter and generalised beyond the occasion for it:
"The second goal was **the element of surprise that is essential if a differentiating step is to be
successful.**"
*Model impact:* surprise attaches to **the target**, where latency and ownership attach to **the mover**.
This is why it does not conflict with Ch10's announced act — announcing *what you are about to do* inside
one dyad does not forfeit surprise about the programme. Three mechanisms, two objects.

---

## Ch22 — Toward the Differentiation of Self in One's Family of Origin  *(1974; presented Oct 1971, rewritten Oct 1974; origin March 1967)*

**Provenance:** trainee cohort. **The primary source document for the coaching agent.** Technique is stated as
instructions to a **person**, not as therapist manoeuvres.
**Q-VALIDATION: none. Q-MATERIAL: none.** Short-horizon scale use is explicitly ruled out. Every dosage quantity
verifies verbatim but there is **no sample size, no instrument, no denominator anywhere**; the only percentages
(25%, 50%) have unstated N and the second measures participation, not outcome.

### L22.1 — Contact reduces anxiety; it does NOT raise differentiation  `[stands — and it carries C1's counterexample]`
**CONTRADICTS → `coach`, `Person`**
"Relative openness does not increase the level of differentiation in a family, but it reduces anxiety, and a
continued low level of anxiety **permits** motivated family members to begin slow steps toward better
differentiation."
*Model impact:* contact breadth is an **anxiety-diffusion parameter over the kin graph**; differentiation is a
**slow integrator gated by low anxiety**. A model where contact raises differentiation directly is wrong. Also:
symptom state is dissociable from level — low-scale people can be "symptom free."
**🚨 C1 has a direct counterexample here.** **Reducing cut-off reduces symptoms *and* makes therapy more
productive** — relief and progress move together. C1 (relief opposes progress) was pass 1's headline finding at
eleven chapters; it now needs restating as a claim about **symptom relief obtained without structural change**,
not about relief as such. **Pass 3 must re-derive C1's scope from the chapter texts, not from the pass-1
summaries.**

### L22.2 — The move needs a live issue, and the non-monotonicity is on the MOVER'S axis  `[corrected]` — *a design error, caught*
**UNMODELLED → `coach`, `policy`, `test`**
A calm family gives nothing to relate to, **and** "the family system works hard to prevent issues from
surfacing." Natural issues — illness, death, homecomings, holidays — supply the working material.
**⛔ Pass 1 put the inverted U on the wrong axis.** It read the working window as a band of **family anxiety**:
calm too low, confrontation too high. **It is not.** Bowen names **serious illness and death — the
highest-anxiety events in the chapter — as *enabling*.** What backfires is the **mover's own manner**:
"emotional confrontation", set explicitly against "introduce small emotional issues from the past, **without
getting into emotional confrontation**." Confrontation carries a stated recovery cost of "months, or even a year
or two, to get beyond the family's cut-off with the confronter."
**A model gating I-POSITION on mid-band family anxiety would suppress the move at exactly the events Bowen sends
the trainee toward.** The non-monotonicity is real but it lives on the **mover's engagement axis**, and **calm is
a delay, not a hard gate** — "it is necessary to introduce small emotional issues from the past."
**⛔ "There is no null move" is withdrawn.** It was a general law built from one clause about silence inside the
detriangling paragraph. **Going silent is not a named error in Ch22** (one clause, no stated consequence). The
failing-move set is **not closed**. The failing moves Bowen does name: **announcing the plan** ("It is impossible
to tell a family what one is trying to do and still make it work") and **bringing spouse and children**
(switches the family into group-relating mode). **The one he actually ranks is confrontation — "probably the one
biggest error".**

### L22.3 — *(resolved — see L21.3)*  `[closed]`
The apparent Ch10/Ch22 contradiction on announcement dissolves at the object level and Ch20 confirms it. The
residual Ch10-vs-Ch21 witness question is L10.15.

### L22.4 — Change is unilateral, propagates without consent, and transfers because patterns are person-level  `[stands]`
**REFINES → architecture, `coach`**
Trainees in **no formal therapy**, getting **15–30 minutes of coaching every month or two** plus a weekly
conference, produced marital and child improvement equal to matched residents in weekly formal family therapy —
because the primary triangle sets "the triangle relationship patterns that remain relatively fixed in **all**
relationships," and marital fusion magnitude is *identical* to each spouse's family-of-origin attachment.
*Model impact:* patterns live on `Person` and **instantiate onto edges** — confirms L10.6's split and L21.7's
before/after. **The rate limit is the person's between-session effort, not coach contact.** Dosage band: monthly
to ~6 one-hour appointments a year; **1–2 a year fails**.
*Note:* **Ch14 retells this and loses both the hedge and the dose** (L14.5 item 6). Cite Ch22, not Ch14.

### L22.5 — Motivation as a third conserved budget  `[withdrawn]`
**⛔ INVENTED, and it had propagated into C1.** Pass 1 recorded that offering a present-generation solution
"**measurably starves**" family-of-origin work, and made motivation a third conserved quantity after anxiety and
investment. **The words *finite*, *budget*, *competing* and *measurably* do not occur.** Bowen offers a
**double-hedged directional force** ("I believe… may be") plus habituation ("addicted") — unquantified and
partial.
**The real second conservation law is Ch10's per-person life-energy budget (L10.11)**, which is stated four
times and does supply C1's mechanism. Use that.

### L22.6 — No optimum contact rate; and the anxiety cost belongs to a third party  `[corrected]`
**CONTRADICTS → `coach`, scheduling**
Bowen urges visits "**as frequently as possible**". **Pass 1's "optimum contact rate" is contradicted by the
chapter.**
And the transient anxiety cost pass 1 attached to `REDUCE_CUTOFF` belongs to **a third party whose distancing
mechanism is being stripped**, not to the person increasing their own contact. **Cut-off intensity is not a
fourth synonym for undifferentiation.**
*Note also:* Ch22's six-appointments-a-year comparison **varies frequency and focus together** and is confounded
— **C8 loses Ch22 as a witness** and is Ch11 and Ch14 only, with Ch14's version being the weakened retelling of
Ch22's.

### L22.7 — Q-MOVEMENT: the fixing point moves EARLIER across the corpus  `[new at p2]`
**REFINES → `Person`**

| Chapter | When basic level is fixed |
|---|---|
| Ch21 (1972) | consolidated **in a marriage** |
| Ch22 (1974) | **early childhood**, with movement stated in both directions in childhood *and* adulthood |
| Ch16, Ch17 (1976) | when the young adult **leaves the parental family** |

Ch22 lands on the **movable** side and **contradicts freeze-at-marriage**. It never uses basic/functional for
those movements, so it does not settle *which* level moves — but **Ch21's freeze claim is outvoted by three
chapters, two of them later**, and Ch16 explicitly restores slow movement ("it is possible to make slow
changes").
*Model impact:* do not implement `basic_level` as frozen at marriage. Implement it as **slow-moving with a
ratchet on completed differentiating exchanges** (L21.1) and a **much larger** fast `functional_level`.

---

# Appendix — withdrawn IDs, at a glance

Do not cite any of these as evidence. Each is kept only so downstream references resolve.

| ID | What was claimed | Why it went |
|---|---|---|
| L01.7 | projection has a durable target queue | **Bowen retracts it in Ch04** — 6 months confirmed, 2.5 years reversed. Fast redirection survives as L01.1 |
| L02.1 | closeness is permission-gated by the excluded third | mothers **approved** and the efforts still failed; the gate is unreleased investment (L02.2). Sexed-asymmetry claim also withdrawn |
| L03.2 | reciprocity resolves by scope reduction | same variable as L03.1, not a second mechanism |
| L05.4 (part) | Ch05 outcome tally as a calibration target | **no stated denominator**; the "leaderless family" was invented — 8+4+3 = 15 is a complete partition |
| L09.7 | ≈15 scale points lost per generation | **no rate stated**; manufactured from an illustration Bowen explicitly bounded |
| L10.13 | family-of-origin distance attenuates field intensity | not in the chapter; contact is retained in the case; would have contradicted C7 |
| L13.6 | Ch13 "Stream B" therapy rules | the chapter contains no therapist action, session, case or prescription |
| L14.2 (part) | "gets worse before it gets better", from Ch14 | built from "**at the risk of**" — a risk clause. Re-sourced to L02.3 / L03.3 |
| L15.2 (part) | a second "dependence" edge type | Bowen says the dependence is *denied*, not on a different graph. Belief layer + multi-hop instead |
| L15.3 (part) | seven-rung loss-severity ladder | not Bowen's; he gives three tiers, and pass 1 silently reordered the grandmother pair |
| L18.3 (part) | physical distance as a depletable stock | relief comes from **availability knowledge** — "even if he never went to it" |
| L20.3 (part) | "content is explicitly designated noise" | not in the chapter; two *attention* prescriptions only. **Used in the proposal as an argument for the deterministic core — fix it there** |
| L22.5 | motivation as a third conserved budget | *finite / budget / competing / measurably* do not occur. Use L10.11 instead |
| — | ally **cancellation** (inside L21.2) | over-read; the named mechanism is **displacement** — an ally opens a new peripheral triangle |
| — | a three→four sink transition (L16.2, L18.1) | never happened; three sinks throughout, distance an always-on baseline outside the budget |
| — | three anxiety **regimes** (L18.2) | a claim about the **observer**, not the system; the dynamics claim is monotone |
| — | the "third branch" at Ch19's critical point (L19.2) | no verb of holding appears; the chapter supplies only the consequent, whose subject is the encounter |
| — | Ch11 as a conservation witness (L11.1) | Ch11 asserts no conservation; pass 1 imported it from Ch08/Ch09 |
| — | Bowen "ruled out this class of model" (L17.5) | scoped to his own research staff's background thinking; he concedes the same charge against *triangle* |
