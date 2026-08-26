---
tags: [model-bt, explainer]
status: current
version: 1.0
date: 2026-08-22
---

# The model, part by part

**What this document is for.** Every object, field, move, gate, invariant, clock and test in the agent model, with three things stated for each: **what it is for**, **what it does**, and **which finding in the theory it implements**, citing the chapter.

**The goal is that you can understand the whole model without reading the corpus.** Where a part has no basis in the corpus, this document says so in the same breath — that is the point of it.

Companions: [`agent_model_proposal.html`](agent_model_proposal.html) is the architecture and the argument for it; [`theory/_LEDGER.md`](theory/_LEDGER.md) is the evidence, 149 findings across 22 chapters. This file is the bridge. It does not argue for the design and it does not quote the source at length; it tells you what each piece is and where it came from.

---

## How to read an entry

Every part is written the same way:

> **`field_name`** — *one-line purpose* **Does:** the mechanics. **Source:** `[grade]` chapter · ledger ID — what the chapter actually says.

### The grade is the most important thing on the page

The corpus contains **no instrument, no rater procedure, no comparison group, and no number ever assigned to a person**, in any of its 22 chapters. It supports directions, orderings and mechanisms. It supports almost no magnitudes. Every numeric constant in this model is therefore invented, and the grade tells you which parts of a claim are which.

**That is a statement about Bowen's papers, not about the field.** Instruments were built later by other researchers — the Differentiation of Self Inventory is the closest attempt, and it is still not a good instrument. It supplies exactly one figure this model may check itself against, and its own failures bear on §10.1's readout warnings. See [`theory/_EXTERNAL_MEASURES.md`](theory/_EXTERNAL_MEASURES.md).

| Grade | Meaning | What you may do with it |
|---|---|---|
| **`[T]`** | **Textual.** Bowen states this, verbatim or nearly. | Implement as stated. Cite it. |
| **`[M]`** | **Mechanism.** The corpus gives the shape of the process, not its rate. | Implement the shape. Invent the rate and label it. |
| **`[D]`** | **Direction or ordering.** The corpus gives the sign, or which is larger, not by how much. | Test as a comparison between two arms. Never as a threshold. |
| **`[#]`** | **A stated quantity.** Verbatim in the source — a duration, a count, a scale point. | Usable as a calibration *target*. Check the ledger entry first: two figures previously treated this way were manufactured. |
| **`[I]`** | **Invented.** No source. A modelling decision. | Free to tune. **Must never be presented as sourced.** |
| **`[X]`** | **Contested.** The corpus disagrees with itself. | Do not implement as an absolute. See §12. |

A part carrying **`[I]`** is not a defect — the model cannot run without invented constants. A part carrying `[I]` while *claiming* a source is a defect, and that is the failure this grading exists to prevent.

### Reading the citations

`Ch16 · L16.2` means chapter 16 of *Family Therapy in Clinical Practice*, ledger entry L16.2. Chapter numbers are the book's, and the book is **not** in chronological order — Ch19 is one of the earliest papers, Ch12 one of the latest. Where the date matters the entry says so.

**One caution that applies throughout.** Nine of the 22 chapters report the same 1954–59 residential project. When two chapters agree, that is often one study reported twice, not replication. Where an entry leans on agreement across chapters, it says how many *independent settings* are behind it.

---

## 1. Orientation — what the model is

Twelve people in one three-generation family. Each is an agent that, on every fast tick, reads what reached it, appraises it, picks one move, and emits it as an event to named recipients and witnesses. Anxiety flows along **ties**, not along a grid, and is divided on arrival by the receiver's functional level. Ties, triangles and the family each hold their own state. Nothing about a person is computed from a lattice position.

**What it is not.** It is not a measurement instrument and cannot make dated claims about real people. It is a consistency engine for one theory: *if Bowen's account is right, what follows for a family shaped like this?* Its trustworthy output is the **difference between two arms of a counterfactual**, run over ensembles, because most of what is uncertain is shared between the arms and cancels.

---

## 2. Objects

### 2.1 `Person`

**For:** the agent. One human being, or one external professional.

**Does:** holds a slow-moving basic level and a fast-moving functional level, two anxieties, a symptom load in three channels, a move repertoire with propensities, an inbox, and a set of beliefs that may differ from the truth. Selects exactly one move per fast tick.

**Source:** `[T]` Ch17 · L17.2 — Bowen names **two** major variables, level of self and level of anxiety, and derives everything else from them. The object is built to that constraint: two state variables, and as much as possible computed rather than stored.

### 2.2 `Relationship` — the tie

**For:** the unit of analysis. Bowen's theory is about relationships, not people, and the old engine had no place to put one.

**Does:** connects two people and carries the state of *that specific* connection — how much traffic it takes, how much unresolved intensity is bound in it, who over-functions for whom, and in which areas of joint life. Conductance is undirected; functioning balance and investment are directed.

**Source:** `[M]` Ch02, Ch03, Ch07, Ch12 — the reciprocity findings only make sense as properties of a pair. `[T]` Ch16 · L16.2 — "In any exchange, one gives up a little self to the other, who gains an equal amount." A per-person array cannot express an exchange.

### 2.3 `Triangle`

**For:** the smallest stable relationship system. Three ties with an inside pair and an outside member, and the positions shift under load.

**Does:** binds anxiety that the three ties cannot hold on their own. Holds which pair is currently inside, who is outside, how much is bound, and whether it is currently active at all.

**Source:** `[M]` Ch09 · L09.3 — tension above a threshold recruits a third; the **value of a position inverts with load** (outside is unfavoured when calm, favoured under tension); tension reroutes onto "old preestablished circuits", so a triangle needs persistent identity and activation memory. `[T]` Ch17 · L17.1 — **triangles are latent when calm**: "the system is calm and the triangles inoperative". Persistent topology is therefore stored separately from the currently-active set.

> **Scope warning.** The earliest chapters license **one distinguished primary triad** with *fixed* membership and no recruitable third, on exclusivity grounds — closeness to the child is a rivalrous good (`[M]` Ch02 · L02.6). The general, mobile, recruitable triangle arrives later. Do not read the general primitive back into the early evidence. And note Bowen calls "triangle" his "most unfortunate term" and offers no replacement (`[T]` Ch17 · L17.4).

### 2.4 `Family`

**For:** the system. Holds what belongs to no single person or tie.

**Does:** carries the undifferentiation budget and its sink allocations, the ambient societal anxiety, the access vector, the nodal-event calendar, the leadership office, the family's shared beliefs, and the multigenerational ledger.

**Source:** `[T]` Ch16 · L16.2 — "a quantitative amount of undifferentiation to be absorbed in the nuclear family". `[T]` Ch09 · L09.2 — "The system operates as if there is a certain amount of 'immaturity' to be absorbed."

> **The family is not the boundary of the system.** See §6.2 — membership is a function of anxiety, and cutoff with the families of origin is an *input*. A model closed at the household cannot compute its own driving term.

### 2.5 External agents

**For:** the counsellor, doctor, teacher, employer, close friend.

**Does:** `Person` with `role = external`, a restricted repertoire, and real ties. Occupies a real position in whatever triangle it is pulled into. It is not a service the family consumes.

**Source:** `[T]` Ch08 · L08.6 — psychiatrist, hospital, school, courts and merchants run the same projection process the mother runs; the school "had long followed the usual steps of thinking about, diagnosing, and treating the son as sick". `[T]` Ch17 · L17.3 — the observer **changes the configuration**: "the groupings were different when the therapist was not a part of the emotional responsiveness". A costless external coach models something Bowen says does not exist.

### 2.6 `Event` — the interaction record

**For:** everything that happens. Agents do not talk; they exchange typed records.

**Does:** carries sender, target(s), witnesses, move type, intensity, timestamp, **route**, **fidelity**, duration, and an `exogenous` flag. Popped in time order into inboxes; simultaneous events batch so that order within a tick cannot silently decide the outcome.

**Source:** see §8. Route, source position and per-hop fidelity are three separate findings that the original event record did not carry.

---

## 3. `Person` fields

### 3.1 `basic_level` — basic level

**For:** the slow, stable level of self. The primary modulator of everything.

**Does:** divides incoming anxiety on arrival, sets the ratio in the life-energy budget, gates which stabilising mechanisms are available, and caps the achievable range of functioning.

**Source:**
- `[#]` Ch16 · L16.1 — **0 to 100.** 0 is the lowest possible human functioning; 100 a hypothetical perfection. **The top quartile (75–100) is explicitly withdrawn as "more hypothetical than real."**
- `[#]` Ch17 · L17.2 — **~90% of people sit in the lower half**; ≤10% reach the third segment. The top profile is stated to be extrapolation, not observation.
- `[T]` Ch16 · L16.1 — **one behavioural transition, at 50**, and it is a **licence, not suppression**: the emotional system *permits* the intellect its corner "as long as it does not interfere in joint decisions that affect the total life course."
- `[X]` The band scheme is **present in 1966, absent in 1972, present again in 1976**, and Ch21 has no regime change at 50 at all — its lower/upper transition is explicitly gradual (L21.1). **The bands are not a constant of the theory.** Ch16 is the only source for the numbers, and the scale gets *less* specified over time.
- `[I]` The engine's old `C ∈ [10, 80]` with a linear clip has **no source**. It has no threshold, and 10 and 80 appear nowhere.

**Does not:** ⚠ **`basic_level` is not written by anything.** There is no ratchet and no per-exchange increment. It is **derived** — an estimator over `functional_level` history (§3.2a) — which is why no move, however successful, moves it directly. `M1.A.4`, `M1.A.4a`

**Two more constraints on the number, from the 2019 book:**
- `[#]` `KS05.3` — the scale **was never an instrument**: "*Many people interpreted the word scale to mean a psychological instrument… this is not the case… It is not a measurement tool. **Bowen subsequently dropped the term scale**.*" And Bowen revised the top **twice** — 90–95 in 1966, then "no higher than 60", then ~75 in 1976. A constructed family with members above 75 is outside anything the corpus describes.
- `[#]` `KS05.4` — the distribution is a **left-shifted bell**: the 50–75 band is ~**10%** of the population, ~**90%** sit below 50, and the ≤25 group is ~**20%**. Hedged in the source ("it appears as if", "estimated"); usable for construction, **never** as a calibration target.
- `[T]` `KS05.2` — a **second threshold, at 25**, of a different kind from the one at 50. Below it `basic_level` **cannot rise at all** — "*they lack the flexibility to make basic change*" — while `functional_level` still moves freely. A capacity bound on change, not a licence over decisions. `M1.A.4d`

> **Two caveats that must travel with any use of this number.** Bowen *slowed* development of a clinical scale because readers wanted the scale without the concept (Ch16), then *stopped* that research entirely to prevent misuse (Ch17) — for misuse, not for invalidity. No external validation is offered anywhere in the book, and `KS11.3` confirms from 2019 that "*a computer program or questionnaire does not yet exist that can estimate a person's basic level with sufficient accuracy.*"

### 3.2 `functional_level` — functional differentiation

**For:** what the person is actually running at right now, as opposed to their basic level.

**Does:** superimposed on `basic_level`, fluctuates widely, and is **the divisor in every appraisal**. Its *variance* is itself a function of the basic level — high-`basic_level` people barely move.

**Source:**
- `[#]` Ch09 · L09.1 — the corpus's only real arithmetic: a husband functioning at **55** on strength drawn from a wife at **15**, both with a basic level of about **35**. Transfer magnitude scales **inversely** with basic level.
- `[T]` Ch09 · L09.1 — **symptoms are a threshold on the functional level, not the basic one.**
- `[T]` Ch17 · L17.2 — functional level "fluctuates widely" and its variance is set by the basic level.
- `[M]` Ch01 · L01.9 — the distinction predates the vocabulary: "mother was *really more impaired* than the daughter" while holding the adequate role.
- `[T]` `KS05.11` — "*The higher their basic level… the more consistent are people's emotional functioning… **As basic levels drop, functional levels of differentiation can rise and fall, sometimes dramatically**.*"

### 3.2a The operational decomposition — and how `basic_level` is actually obtained

**`functional_level` *is* the operational level of differentiation**, and it decomposes:

> **operational (`functional_level`) = `basic_level` (slow floor) + swing (fast)**

The **self-directed channel** (§5.2) writes the **swing** directly — a person can decide to stop or start doing something and functioning shifts at once. `basic_level` does not move. `M1.A.5a`

**`basic_level` is then an estimator over that swing's history**: an elevated mean, held at low variance, **sustained over years**, and demonstrated **across many distinct situations**. `M1.A.4a`

- This is `M1.A.5`'s stated relationship **run backwards**. High basic level ⇒ low functional variance; therefore sustained low variance at an elevated mean **is** the evidence. One relationship, not two mechanisms.
- `[T]` **The situations must be loaded, and must include nodal events.** A benign decade produces a stable elevated functional level and demonstrates nothing — without load-conditioning, **an easy life reads as differentiation**. `M1.A.4b`
- This is what separates a binder from the real thing without a special rule: `CUTOFF` produces a durable-looking calm that holds until the next nodal event, so it fails on breadth.
- `[#]` `KS05.10`, `KS11.7` — the observables and their **asymmetric** reading rules: occupation is uninformative when high but informative when **low despite opportunity**; health is read through the **adaptation**, not the diagnosis; longevity is **not** a proxy, but a long-lived generation above a dysfunctional one is *evidence of projection*; **both** tails of courtship length are informative. `M1.A.4e`
- `[T]` `KS06.8` — contexts are **not** comparable: "*In terms of solid self, one's personal life is where the rubber meets the road*", and occupational success "*is not a reliable measure*."

### 3.2b Which channel raised it — and why the sign of pseudo-self flips

The **same** rise in `functional_level` means opposite things depending on what produced it. `M1.A.5c`

| Rise produced by | `functional_level` | pseudo-self | `basic_level` |
|---|---|---|---|
| **Automatic channel** — borrowing from a partner, a group, a tribe | ↑ | **↑** (borrower) / ↓ (lender) | unchanged |
| **Self-directed channel** — differentiating effort | ↑ | **↓**, converting toward solid | rises if sustained and broad |

- `[T]` `KS05.1` — "*Pseudo-self can be the basis of increasing a person's functional level of differentiation*": the aimless young man who finishes college after falling in love "*has borrowed pseudo-self from his romantic partner*"; cult members "*experience similar improvements*."
- `[T]` `KS05.1` — and it is **not pathological**: "*Everyone borrows and lends self to some degree… This is not a bad thing.*" What distinguishes it is only that **it does not survive withdrawal of the source**.
- Because the two channels leave opposite signatures, the event record has to carry **which channel selected the move** (§2.6) — otherwise neither this nor "going toward a goal versus running away from a problem" is recoverable. `M1.F.1a`

### 3.3 Chronic anxiety — **two quantities, not one**

⚠ Corrected. What was one field is now a **disposition** and a **state**.

**`programmed_reactivity`** — the disposition. Fixed in childhood from the child's own **witnessed event history**, never from a family average. `M1.A.7`

**`chronic_anxiety`** — the state, **derived** each slow tick from three terms: that reactivity, **the field** (the system's differentiation and current load), and **the person's functioning position**, which changes. It acts as the floor acute anxiety decays toward but never below. `M1.A.7a`

**Why the split matters:** `[T]` `KS10.2` — "*chronic anxiety is a consequence of various types of social interaction and, consequently, is **most usefully conceptualized as a property of the emotional field***", set by two processes "*not under individual control*" — the system's differentiation, and "*the person's functioning position in the system*." A model with only the fixed childhood term cannot produce a removal that helps some members and harms others, nor a reciprocity inversion, because **both act through position**.

**Source:** `[M]` Ch06 · L06.1 — the load variable is **directed parental attention**, and pointedly not parenting *content*: the problem is created "just as surely by a project that was psychologically correct as by one that was psychopathological." `[M]` Ch18 · L18.2 — symptom onset depends on **chronicity, not instantaneous level**: "Any unit can recover from periodic panic or overloads, but when the panic becomes chronic one or more of the individual units can collapse." That requires an **integrator, not a threshold test**. `[I]` Fixing it at age 10 specifically is a modelling decision; the corpus says childhood.

### 3.4 `A` — acute anxiety

**For:** the fast state. What moves on the weekly clock.

**Does:** raised by each incoming event by `intensity × conductance / `functional_level``; decays toward the chronic floor; also takes a **standing load** from every tie each tick whether or not anything happened (§7.2).

**Source:** `[T]` Ch18 · L18.2 — all patterns intensify with anxiety and **vanish when calm**. `[M]` Ch01 · L01.1 — transfer is "almost quantitative": the source's anxiety measurably falls as the recipient's symptom rises.

> **Anxiety gates information.** Above threshold, content is not heard, only defended against (`[M]` Ch18 · L18.2). Appraisal needs a gain function, not just a summation.

### 3.5 Reactivity — **derived, not stored**

**For:** how sharply this person responds. It is *not* a field.

**Does:** computed from level of self and level of anxiety at the point of use.

**Source:** `[T]` Ch17 · L17.2 — Bowen names **two** major variables and derives reactivity from them: "the lower the level of self, the more reactive." The old engine's `TX` column is a third stored state the theory does not have.

### 3.6 `outside_ness` — the efficacy gate

**For:** the hidden state that decides whether a differentiating move lands or backfires. Distinct from basic level.

**Does:** multiplies the effect of `I-POSITION` and `STAY-IN-CONTACT`. Below threshold, an identical technique is received as either empty words or an attack.

**Does not:** it is not raised by doing the move. It has three inputs at three time scales:
1. **Private rehearsal** of one's own charged material until anger is impossible — fast, move-scoped.
2. **The observation/control ratchet** — "Observation is not possible until one can control one's reactions sufficiently to be able to observe… which in turn, in a series of slow steps, allows for better observation." Slow, self-reinforcing.
3. **Practice on a peripheral emotional system** — a work system, an extended family. Medium.

**Source:** `[T]` Ch21 · L21.2 — "either hollow meaningless words or a hostile assault on the system, **and an emotional system knows the difference**."

**Does not:** ⚠ **it is not a scalar.** `outside_ness` is **two-dimensional** — **outward impingement** (acting on the other) and **inward impingement** (being acted on by the other) — and the differentiated position is **both low**. `M1.A.9a`

The definition is a **conjunction of two negatives**:

> *Be for self without being **selfish**. Be for other without being **selfless**.* — Kerr & Bowen, *Family Evaluation* (1988); `KS24.4`

| Axis | Kerr's term | How it looks |
|---|---|---|
| Outward impingement | "**selfish**" | forceful, dogmatic, encroaching — the *shout them down* grotesque |
| Inward impingement | "**unselfish**" | pleasing, placating, accommodating; hears the other as critical |

- `[T]` `KS12.3` — "*a high-level differentiated self is **neither offensive nor defensive** to the other. So you can tell pretty much the level of functioning of an I-position from the way they do it.*"
- `[T]` `KS07.3` — **both are the same failure**, and Kerr warns why a single score is dangerous rather than merely imprecise: "*If the selfish and unselfish terms are thought of as character traits rather than as polar opposite positions produced by a reciprocal process, **it is almost impossible not to view the 'unselfish' partner as a victim and the 'selfish' partner as the culprit**.*"
- The **inward** axis has a **perception-side** readout available *before any move is emitted*: hearing the other as critical is itself the evidence. Cheaper and harder to game than anything read off the emitted act. `M4.C.5`
- The counterfeit detector **must report which axis failed** — the forceful declarer and the compliant accommodator need **opposite** corrections. `M5.F.2b`

> This field is the model's implementation of **C10**, the corpus's cleanest cross-chapter convergence: *an act's identity depends on the actor's hidden state, not on the act.* Five chapters, four independent settings. See §5.6.

### 3.6a `systems_perspective` — the mode of thinking

**For:** whether the differentiating move is **comprehensible to the agent at all**. Graded, `[M]` — mechanism sourced, magnitudes invented. `M1.A.18`

**Does:** gates the **differentiating path only** — move quality (§5.6), engagement with a loaded tie, and therefore the estimator's capacity to rise.

**Does not:** ⚠ **it does not gate `functional_level`.** An agent with none can function well across a whole life, and **most agents in any population have essentially none**. Systems thinking is *rare*; functioning is *common*; the model reproduces both. `M1.A.18c`

**Why it exists.** Bowen names **three** variables for societal process — "*one, a different way of thinking… another is differentiation of self… and a third is the intensity of anxiety*" (`KB10` · K10.1). The three societal dials are drivers of the **third** only. This is the individual-level representation of the **first**.

**How it moves:**
- **Rises** only on a *landed* external contact, conditioned on the agent's binders failing. Never spontaneously, never as a reinforcement target. Four forms of landed contact — specifics, category, **non-participation**, and **delayed self-observation**. `M1.E.7`, `M1.E.7c`
- **Falls** with acute anxiety. Polarity capture is a **loss** in someone who had it. `M4.C.4`
- **Ceiling** coupled to `basic_level` — undifferentiation is "*the cement, the hardener, that fixes*" a way of thinking. `M1.A.18b`
- **Per-person, attenuated per tie.** Not a per-tie variable: one capacity that **fails at the highest load**. Kerr taught the reciprocity idea for twenty-five years before seeing it in his own marriage. `M1.A.18d`

**Readout:** blame — and it is two-sided **twice over**: blame *or* praise, and blame of **others** *or* of **self**. "*Blaming others and blaming oneself are the enemies of gaining a systems perspective.*" `M1.A.18a`, `KS00.2`

**Not additive with cause-and-effect thinking.** Partial acquisition is a liability, not a partial benefit — "*the people who make the most progress are those who **stop mixing theories**… flip-flopping in highly anxious situations.*" `KS18.6`

### 3.7 `life_energy` — the per-person budget

**For:** the second conservation law. The cost side of the differentiating move.

**Does:** a zero-sum allocation between **relationship-seeking** and **goal-directed** activity. The ratio is set by scale level. At the bottom of the scale the budget is empty. Taking a differentiating step **debits the tie** — the step "detracts from the former energy devoted to the system, especially to the important other."

**Source:** `[T]` Ch10 · L10.11 — stated four times in one chapter.

> **This is the mechanism of the change-back reaction.** The reaction ladder (§5.5) is the surface; the withdrawal of energy the other was receiving is the cause. It also explains why symptom relief opposes progress without needing a separate mechanism for it. It was the largest single omission of the first reading of the corpus.

### 3.8 `involvement_weight` — recomputed every tick

**For:** deciding who is currently *in* the emotional system. Not a roster.

**Does:** a continuous per-agent weight, recomputed each tick from load and involvement. Membership is a **thresholded readout** of it, never a set operation.

**Source:** `[T]` Ch07 · L07.1 — calm, the mass includes only a few most-involved members; under stress the fusion extends to multiple extended-family members "and even nonrelatives", and "**live-in servants can be more emotionally fused into the family emotional system than certain blood relatives**." `[M]` Ch17 · L17.1 — the full cascade: twosome → triangle → interlocking triangles, with the displaced member becoming *emotionally inactive* → neighbours, schools, agencies, courts → **reversion to the original triangle on subsidence.**

### 3.9 `symptom_load[3]` — physical, **mental**, social

**For:** the model's dependent variable. The interesting prediction is not *whether* but **in whom**.

**Does:** accumulates from the load routed to this person; crossing threshold emits an endogenous event. The three channels are **substitutable** — the same deficit presents interchangeably as physical illness, mental illness, or social dysfunction.

**On the name.** The third channel was *emotional*; it is **mental** on the author's own proposal — "*the symptoms manifest in aberrant cerebral cortical processes, but the core of the symptom-generating force is the subcortical emotional system*" (`KS23.11`). Under `M1.A.0` **everything** in this model is emotional, so *emotional* as a channel name was a collision.

**A second axis crosses the three:** physical and mental **internalise**; social **externalises**. That is what makes "curing the symptom raises conflict" and its inverse — depression lifting as a couple starts fighting — the same mechanism rather than two. `M1.A.11a`, `KS10.5`

**Which channel is exogenous.** Constitution sets the **channel**; level sets the **amplitude and sign**. The model never derives symptom *type* from relational position — "*genes would be seen as perhaps having a role in **whether** the chronic anxiety plays out as schizophrenia rather than some other clinical dysfunction*", and "*certain predispositions can become **assets or liabilities**, depending on the degree of family anxiety*." `M1.A.11b`

**Source:** `[T]` Ch07 · L07.2 — the habitual giver-in reaches "no-self" and is incapacitated by one of the three; the chronic illness "seems to absorb the ego deficit between them". `[T]` Ch08 · L08.3 — the three present interchangeably.

> **The symptom feeds back negatively onto family tension.** The marriage stays harmonious "as long as the disabled spouse does not recover" (`[T]` Ch07 · L07.2). So curing a symptom without changing the deficit must **raise** tension. See test 12 (§11).

> ⚠ **And the system locks in around it.** Once a member carries the load, **family anxiety falls**, producing a stable configuration that **resists the symptom's removal**: "*a family can stabilize somewhat around the presence of a symptom, **which fosters it becoming chronic**.*" `M7.D.2a`, `KS23.1`
>
> **That one requirement produces three observed behaviours as consequences** — chronicity with no pathology anywhere; relapse when a symptom is cured without changing the deficit; and family **relief** when the symptomatic member is removed. Removal therefore has **opposite-signed** effects on different members, keyed to position: when Gary Gilmore went to reform school family tension dropped and **only his mother** wanted him back. `M7.D.2b`

### 3.10 `structural_importance`

**For:** how much the system is disturbed if this person is lost. Separate from how well they are functioning, and separate from how much they are grieved.

**Does:** scales the shock wave on a death or removal. Three tiers only.

**Source:** `[D]` Ch15 · L15.3 — Bowen gives **three tiers** (shock-wave-likely / neutral / relief) with an **unranked** list inside the top tier, plus two pairwise relations: central > shadow grandmother, and dysfunction that was *load-bearing* produces a wave where ordinary dysfunction does not. A burdensome, non-critical member's death is followed by **improved** family functioning; suicide produces prolonged grief but a **minor** wave.

> **The seven-rung severity ladder is not Bowen's.** It was assembled on the first reading, and the grandmother pair was silently reordered so it would descend. Three tiers is what can be calibrated; anything finer is invention. **Grief magnitude and system-disturbance magnitude are separate output channels.**

### 3.11 `sibling_position` — static data, **derived position**

**For:** static **birth-order** data. One of Bowen's named concepts.

**Does:** ⚠ **nothing directly.** The propensity vector reads a **derived `functional_sibling_position`**, computed from observed functioning — because the projection process can make a younger son "*a **functional oldest**.*" `M1.A.14`, `M1.A.14a`

> **The general rule, of which this is the second instance** (`structural_importance` was the first): **no positional attribute the model acts on is read from a role label or from birth order. Every one is derived from observed function.** Labels are static data; positions are computed. `M1.A.14b`

**Does not:** it is **not an additive offset**. Each position carries **both** an adaptive and a maladaptive expression, and `basic_level` selects which — "*An **immature** older brother of brothers is likely to be overly controlling and dogmatic… A **mature** older brother of brothers can be a very effective and responsible leader.*" A **gating** relationship, not additive. `M1.A.14c`

**Source:** `[T]` Ch21 · L21.5 — fifth of the six named concepts. `[T]` `KS12.1` — ⚠ **the effect is not constant across the scale.** Bowen "*qualified his thoughts about Toman's profiles by saying that they accurately describe people **at the midrange***", and "*a poorly differentiated oldest brother may exhibit **very few** characteristics of an oldest profile.*" It peaks mid-scale and attenuates at both ends. **This repaired the acceptance test that had been flagged as near-unfalsifiable** — it is now a three-arm mid-peak comparison needing no invented effect size. `[#]` `KS12.10` — Toman estimated 10–25% of personality; **Bowen theory explicitly declines the figure**, so it must not enter the model.

**Suppressed, not erased, at the low end:** six patients on a locked ward who responded to a fire "*were the oldest children in their families*". Position effects are suppressed by anxiety in the **relational** domain and re-emerge on an unambiguous **task** demand. `[D]`, n≈6, no control. `M1.A.14d`

### 3.12 `financially_dependent` — a gate, not a stock

**For:** the one place a material variable earns its keep.

**Does:** a hard precondition on the differentiating move. When set, the move fails.

**Source:** `[T]` Ch10 · L10.11a — the chapter's **only absolute**: a differentiating move by a financially dependent person "has never been successful".

> **There is no `M` column.** No per-person material stock exists anywhere in 22 chapters, and the one workplace chapter sets tie weight by emotional importance **explicitly not by economic relation** (Ch20 · L20). What the corpus supports is resources as **preconditions and capacities** — this gate, Ch13's family-level "ability to provide material demands", Ch15's poverty gating the *form* of contact, Ch18's four hits. Never a stock whose depletion causes death.

### 3.13 `beliefs`

**For:** what this person holds to be true, which the model must allow to differ from what is true.

**Does:** see §9.

---

## 4. `Relationship` fields

### 4.1 `conductance` (`g`)

**For:** how much of a sender's anxiety arrives. The `friction` term, made a property of the specific tie.

**Does:** multiplies intensity on delivery. Undirected.

**Source:** `[M]` Ch01 · L01.1 — transfer is a quantity with a location. `[I]` The values are invented.

> **Conductance must not decay with distance or contact frequency.** See §4.3.

### 4.2 `bond_energy`

**For:** the unresolved intensity still bound in the tie. **Not** the same as contact — this is the field that tells a rupture from a resolved distance, which is the most consequential discrimination in this part of the theory.

**Does:** four states the model must keep distinguishable:

| Tie state | Events | Bond energy |
|---|---|---|
| Cut off | none | high |
| Distant | few | moderate |
| Genuinely resolved, low contact | few | **low** |
| Open conflict | many | high |

Feeds the standing load (§7.2) every tick whether or not anything happened.

**Source:** `[T]` Ch02 · L02.7 — the earliest and most direct: "**The surface distance controls a deeper interdependence on each other.**" Distance is a *regulator over an undiminished coupling*, not an attenuation of it. `[T]` Ch10 · L10.5 — silence, withdrawal and cutoff are moves **inside** the system, never exits; the runaway fuses into a new family and reproduces the pattern.

### 4.3 Why coupling does not decay — and the correction to how it persists

**Source:** `[T]` Ch08 · L08.2 — "If the threesome is reunited, the old emotional fusion of the triad is immediately operative again", with the member institutionalised as a permanent ward of the state.

> **The mechanism is not a slow-decaying edge on an isolated node.** It is **substitution on both sides**: the separated one lives away "in dependent attachments that do not involve the parents", the parents "borrow self from outside themselves and project their inadequacies to others", and "any long-term separation from parents is accomplished only by finding a new family ego mass to which they can append themselves." **Nobody is uncoupled.** Separation is **edge-rewiring with the old edge dormant**.
>
> What this licenses is a **zero re-activation latency on reunion** — a different parameter from a decay rate. It is also hedged ("it *appears* impossible"), gives no separation duration, and is scoped to *severe* cases; the next paragraph says differentiation "does occur" in less severe families.

**What is forbidden:** any distance- or contact-frequency-based conductance term. **Access gates which moves are available; it does not attenuate coupling strength.**

### 4.4 `functioning_balance` — directed, bistable, held per area

**For:** who is over-functioning for whom. One anti-symmetric coupling with no stable midpoint.

**Does:**
- Both parties are equally immature; whoever **decides** becomes the overadequate one and the other helpless. `[T]` Ch03, Ch04 · L03.1, L04.3 — explicitly "functioning states and not fixed states".
- **Either party can occupy either pole.** The same woman held opposite poles in two marriages.
- **Neither can find the middle.** Deciding reads as dominating, yielding as submitting — which is why these families avoid all decisions. `[T]` Ch05 · L05.3.
- **The flip is relative and immediate.** The pole flips when the inadequate one's self-assertion is "**greater than**" the other's aggression and domination — a comparison, not a threshold — and it holds only if the mover sustains it through the counter-reaction. `[T]` Ch03 · L03.1.
- **It is held per area of joint activity**, and "reducing the areas of joint activity" (father to business, mother to home) operates on the **same variable** — lowering anxiety while leaving the reciprocity untouched.
- **The assignment hardens with time in configuration.** `[M]` Ch05, Ch12 · L05.3, L12.1.
- **Reversal is direction-asymmetric:** far easier to tone down a "marked" overfunctioner than to raise a "marked" dysfunctioner. `[D]` Ch12 · L12.1 — and note Bowen explicitly withholds the reason. The organ analogy grounds the *duration* hysteresis, not this.

> Implement as a **switch with hysteresis, indexed per area** — not a linear scalar and not a personal trait. The old engine's `functioning_balance` is close in name and wrong in shape.
>
> **Six chapters carry this, across about four independent settings** — the corpus's second-strongest convergence. Ch07 is *not* one of them: sixteen of its propositions rewrite Ch06's near-verbatim.

### 4.5 `investment` — directed, and **valence-blind**

**For:** share of thought occupied by the target. The quantity the parental seesaw runs on.

**Does:** traded across targets, zero-sum across a person's ties, with the marital tie load-bearing rather than one tie among equals.

**Source:** `[T]` Ch04 · L04.7 — defined in text: "the thoughts of both, **whether positive or negative**, are largely invested in each other."

> **Conflict-laden preoccupation is HIGH investment.** Any implementation deriving investment from warmth or agreement **inverts the sign on exactly the families this model is about.** It is also an inference by observers — not self-report, not time spent, no instrument.

**The seesaw is a conjunction, not a difference.** `[T]` Ch04 · L04.1 — "more invested in each other than **either** was in the patient" / "when **either** parent became more invested…". It is a **min over both parents**, and **one parent alone drives the regression**. A difference form lets a strong marriage mask a defecting parent.

### 4.6 `taboo_set` — growing **by default**, but reversible

**For:** the mechanism behind scope reduction.

**Does:** subjects are withdrawn from a tie, and growth is the **default** — each party learns what makes the other anxious, "so begins the communication cutoff between spouses."

**Does not:** ⚠ **it is not monotone.** Growth is a default, not a law. **Purposeful mention of a taboo subject, if one can control one's own anxious response, can desensitise the whole mechanism** — the differentiating move applied to a single topic, carrying the same gates as any other. `M1.B.10a`

**The opposite pole is defined too.** An "open relationship" is one with an **empty** taboo set — "*both parties are able to communicate their innermost thoughts and feelings without fear of hurting the other person*" — and it is **health-promoting for both**, not merely neutral. Bowen names only four contexts where it naturally occurs: the early mother–infant relationship, courtship, a psychoanalytic relationship, and a fantasied relationship. Three of the four are transient or artificial, which is why growth is the default. `KS04.10`

**Source:** `[T]` Ch10 · L10.8. `[T]` 1979 Tape 6 — the reversal, and the self-control condition on it.

### 4.7 `latency` — transfer delay, per edge

**For:** the corpus's clearest statement that events do not all arrive at once.

**Does:** delays delivery on this specific tie.

**Source:** `[#]` Ch01 · L01.1 — stated latencies differ by edge: mother→patient "very soon", mother→younger son "**within hours**", patient's gains→mother's physical illness over **months**.

> Neither the old engine nor the first draft of the proposal had this — every event arrived within one tick.

### 4.8 `cutoff_flag`

**For:** marking the tie as severed.

**Does:** stops interaction; leaves bond energy high; raises system tension.

**Source:** `[T]` Ch10 · L10.3 — cutting off raises system tension, enlarging the system lowers it, because anxiety dilutes over nodes. `[M]` Ch16 · L16.3 — the corpus's clearest positive feedback loop: the more intense the cutoff, the more exaggerated the parental problem in one's own marriage, and the more intense the next generation's cutoff.

### 4.9 `dyad_age`

**For:** the stickiness term.

**Does:** raises the cost of flipping the functioning balance as time in configuration grows.

**Source:** `[T]` Ch05 · L05.3 — "over years the assignment becomes fixed." `[I]` The curve is invented.

---

## 5. Moves and the policy

Every fast tick, each person selects **exactly one** move. The repertoire is deliberately small and is Bowen's own vocabulary.

### 5.1 The nine

| Move | What it does | Source |
|---|---|---|
| `PURSUE` | Seek contact, close distance, ask for reassurance | `[M]` Ch12 · L12.4 — the pursuit/isolation band |
| `DISTANCE` | Withdraw, go quiet, reduce contact without ending it | `[T]` Ch03 · L03.1 — reducing areas of joint activity |
| `CONFLICT` | Criticise, blame, press a position | `[T]` Ch16 · L16.2 — one of the three sinks |
| `OVERFUNCTION` | Do for, advise, take over the other's responsibility | `[T]` Ch03, Ch05 · L03.1, L05.3 |
| `UNDERFUNCTION` | Defer, collapse, hand responsibility over | `[T]` Ch03, Ch05 · same |
| `TRIANGLE` | Route it through a third — recruit an ally, talk about, focus on a child | `[T]` Ch09 · L09.3 |
| `CUTOFF` | End contact | `[T]` Ch10 · L10.5 |
| `I-POSITION` | State a self without attacking or accommodating | `[T]` Ch13, Ch21 · L13.3, L21.4 |
| `STAY-IN-CONTACT` | Remain present under tension without acting | `[T]` Ch22 · L22.1 |

**And a tenth outcome that is not a move.** `WITHHOLD` — the automatic move **computed, detected, and not emitted**. It is distinct from every move above and from the fused default, and ⚠ **a withheld move still changes tie state**; an implementation in which not acting is a no-op cannot represent the two canonical instances in the corpus. `M4.D.1b`

> Kerr, on the staircase: "*I even started to move slightly, but **I caught myself and stopped**… **I did not take any obvious I-position with Mother; I just did not anxiously hover over her.**" (`KS08.1`) And on the couch, leaving his distressed wife to go upstairs: "*My legs felt like they weighed a hundred pounds each.*" (`KS16.2`)

**But `WITHHOLD` alone is insufficient**, and this is easy to get wrong. It is **not** a weak `I-POSITION`. Kerr's three-year trajectory ran **counter-argument** ("accomplished nothing") → **non-reaction** ("*an insufficient response to her*") → **position** (lands, draws the reaction, resolves). Phase 2 is a necessary stage that must be **passed through and exceeded**. `M4.D.1c`, `KS16.1`

### 5.2 The policy — **two channels, not one**

**Does:** selection runs over **two channels with different objectives**, and the mixing weight between them is a function of differentiation. `M4.D.1a`

| | **Automatic** | **Self-directed** |
|---|---|---|
| Driven by | the relationship system | the person |
| Objective | discharge anxiety **now** | hold a position **through** discomfort |
| Carries | the seven reactive moves | `I-POSITION`, `STAY-IN-CONTACT` |
| Learns? | **yes** — this is family style | **no** |

At low level a person is nearly all automatic; as level rises a real self-directed channel opens. **Agency is not absent — it is graded**, and it is already in the model as the **solid-self fraction**: pseudo-self is what the system can move, solid self is what it cannot. No new variable. `M10.A.1a`

- `[T]` `KS00.3` — "*Cortical components of differentiation **guide** actions; subcortical components **motivate** actions.*" The self-directed channel does not supply its own motive force; it **redirects** what the emotional system supplies.
- `[T]` `KS02.1` — "*better differentiated people have more control than do less differentiated people*" — but never full autonomy at any level.
- **Consequence:** differentiation is **not reachable by lengthening a reinforcement horizon**, because the two channels optimise different things. Anxiety-relieving action genuinely relieves anxiety, so an agent choosing a binder is not making an error a longer horizon would correct.

**The ordering within the automatic channel.** Rising anxiety does not simply "raise the reactive weights" — it **slides selection down a complexity ordering**: **cooperation** (requires both systems as a working team) → **conflict** → **dominant-adaptive** ("older evolutionarily, more primitive") → **distance**, the oldest of all. `M4.D.3a`, `KS06.1`

**Engagement is gated by perspective, not by anxiety.** Nothing else in the model makes an agent approach its hardest tie — every reactive move reduces engagement. "*Of course it is difficult if you lack a theory to guide you.*" Without mindware the loaded tie is **correctly** avoided; with it, the same tie becomes approachable. `M4.D.3b`, `KS15.11`

**Ambivalence is itself anxiogenic.** Both parties hold **simultaneous** approach and withdraw urges, and "*the conflicting urges raise each person's anxiety*" — independently of whichever move resolves. An entropy term over the propensity distribution produces the vicious circle for free. `M4.D.1d`, `KS06.4`

**Source:** `[D]` Ch18 · L18.2 — all patterns intensify with anxiety and vanish when calm. `[I]` The functional form and every coefficient are invented.

> **The default state is not "no move".** `[T]` Ch07 · L07.5 — the fused default is each altering self to manage the other's functioning while demanding the other change, "neither responsible for self". That is the **term-for-term inverse of the differentiating move**, and it belongs in the policy as a baseline rather than as an absence.

> **Propensity for the differentiating move is NOT monotonic in functioning.** `[D]` Ch02 · L02.5. The old engine raises `I-POSITION` propensity with `basic_level`, which predicts the *over*-functioning member moves first. No reading of the corpus supports that. See §12.1 for what the corpus does say.

### 5.3 Move gates — preconditions that must hold

| Gate | Effect | Source |
|---|---|---|
| **`outside_ness` below threshold** | `I-POSITION` is received as empty words or an assault | `[T]` Ch21 · L21.2 |
| **`financially_dependent`** | `I-POSITION` fails outright | `[T]` Ch10 · L10.11a |
| **No live issue** | `I-POSITION` is **delayed**, not blocked | `[T]` Ch22 · L22.2 |
| **Mover's own engagement too high** | `I-POSITION` backfires, with a recovery cost of "months, or even a year or two" | `[T]` Ch22 · L22.2 |
| **System calm** | Triangles are inoperative; projection does not fire | `[T]` Ch17, Ch01 · L17.1, L01.6 |
| **Marital distance high** | *Any* intervention aimed at the symptom-bearer produces no improvement | `[T]` Ch04 · L04.2 |
| **Decision ownership held by staff** | Removal does not fire even at tolerance | `[T]` Ch08 · L08.1 |

> **The non-monotonicity is on the mover's axis, not the family's.** This was a design error caught on the second reading and it matters. Bowen names **serious illness and death — the highest-anxiety events in the chapter — as *enabling*.** What backfires is the mover's own "emotional confrontation", set explicitly against "introduce small emotional issues from the past, without getting into emotional confrontation." **A model gating `I-POSITION` on mid-band family anxiety would suppress the move at exactly the events Bowen sends the trainee toward.**

### 5.4 `I-POSITION` is a state machine, not a move

**Does:**

1. **Define self.**
2. **Immediate opposition** — "selfish and mean and does not love the others."
3. **Abort branches — defend, counterattack, go silent.** These sit at the *first* opposition and are the **usual** response, not the exception. Each returns the mover to the prior balance.
4. **Hold course.** **Anger is the gate**, not a fourth branch: "when he is finally able to maintain his course without getting angry… the opposition does a final intense emotional attack."
5. **The peak** — a final intense emotional attack.
6. **If the mover stays calm:** "the opposition becomes calm and **pulls up to his level** of individuality," and then "another, and another will do the same."
7. **Mandatory follow-up the next day** — "this is the point where the feeling system dictates withdrawal, which will result in the system 'tightening up' again." **Skipping it reverts the gain.**

**Source:** `[M]` Ch13 · L13.3 for the sequence; `[T]` Ch21 · L21.4 for the follow-up and the timings. **Success usually follows several failures.**

### 5.5 The reaction ladder

**Does:** the system's response runs a fixed escalation, and it **decays unless fed** by the mover defending or counterattacking.

> **"You are wrong" → "Change back" → "If you do not, these are the consequences"**

**The diagnostic rule: absence of reaction means the move did not land.**

**Source:** `[T]` Ch21 · L21.4 — verbatim, and confirmed verbatim on the second reading.

**Two older variants the ladder must accommodate:**
- `[T]` Ch01 · L01.8 — present in **1957**, two rungs, and the success condition is an **inequality**: "his strength seemed greater than the mother's attack."
- `[T]` Ch08 · L08.7 — five rungs at ~1962, and **the opening move is a symptom, not a verbal challenge**: helplessness → self-labelling → demand for caretaking → attack on the tie → a week of withdrawal → capitulation. The severe-end version is the "**petition for sickness**", which **succeeds** instead of failing.

**And the cause underneath it:** the life-energy debit of §3.7. The ladder is the surface.

### 5.6 Counterfeit moves — the same act, opposite effect

**For:** the model's hardest constraint on move resolution.

**Does:** a move's effect **cannot** be a function of the move type and the tie alone. It is multiplied by a hidden state of the actor that the *receivers* can read and the actor may not.

**Source:** `[T]`, five chapters, ~four independent settings — the corpus's convergence **C10**:
- Ch13 · L13.5 — at a low level, "to 'stand up to' means to attack and shock the other with language and behavior, and to get away with breaking rules."
- Ch20 · L20.4 — accepting **blame** versus taking **responsibility for the part self plays**: "a fine line", discriminated by "the inner orientation of self".
- Ch21 · L21.2 — the outside-ness gate.
- Ch10 · L10.12 — **the fifth scale band.** "The upper part of the 25-to-50 segment": dogmatic authoritativeness, the compliance of a disciple, or the opposition of a rebel, with "**intellect in the service of the relationship system**." These agents *look* differentiated and are not.
- Ch19 · L19.5 — "**token concurrence**": the family scores concession on a **continuous scale** against ordinary competent acts. The one place in the corpus Bowen grades anything continuously, and it grades an act's value **as read by the receiver**.

### 5.7 Moves the nine do not cover

| Move | What it does | Source |
|---|---|---|
| `DETRIANGLE` | Actively push an ally back out of the position they took | `[T]` Ch21 · L21.2 — "detriangle any ally who tried to come over to my side" |
| `REDUCE_CUTOFF` | Increase contact on a severed tie | `[T]` Ch22 · L22.1, L22.6 — visits "as frequently as possible"; **no optimum rate** |
| `SPLIT` | Consult an external agent without telling the others; withhold history so the opinion is uncheckable | `[T]` Ch19 · L19.3 |
| `FRAME_AMBIGUITY` | Speak symptoms as social talk, in the professional's presence, so "his response **or lack of it**" carries professional weight | `[T]` Ch19 · L19.3 — **silence is captured; abstention is not an exit** |
| `DISPLACE` | Take over another member's contact with an external agent; mildest form is "simply invite himself to be also present" | `[T]` Ch19 · L19.3 |
| `PROVOKE` | The deliberate low-stakes provocation — "tempest in a teapot" | `[T]` Ch21 · L21.6, L21.9 — **message intensity is a dial separate from content** |

> **The ally mechanism is displacement, not cancellation.** An ally **opens a new peripheral triangle**, and that is what Bowen calls "my undoing." He never says support from a family member *cancels* the move. The first reading reported cancellation as the corpus's most counter-intuitive finding; it is not in the text and must never be cited as a quotation.

> **Deliberate perturbation is hard.** Automatic disturbances are easy; purposeful ones are not (`[T]` Ch21 · L21.9). A policy in which any agent can provoke at will is too permissive.

### 5.8 The external agent's repertoire

**Does:** a subset. A counsellor can `STAY-IN-CONTACT` and hold an `I-POSITION`; can `OVERFUNCTION` — the doctor who takes over what the family should carry; and can `TRIANGLE` — the counsellor who sides with one partner. A discharge or dropped referral is a `CUTOFF`.

**Two capabilities §5.1 of the proposal did not give them, and the corpus requires:**
- **Absorb responsibility.** `[T]` Ch08 · L08.6 — responsibility is a conserved transferable quantity the family **cannot hold while staff holds it**. Strongest form: "if the medical structure did not exist, the families could find other means to make the environment responsible."
- **Certify a defect.** `[T]` Ch10 · L10.9 — hospitalization, a prescription, an insurance form write to a family-level who-is-sick belief that **verbal denial cannot reverse**.

**And a burden-transfer term.** `[T]` Ch05 · L05.2 — the leader-mother became "a helpless complaining person" the moment the therapist tried to help her deal with her family. Seven in-residence families "with hospital staff nearby, were never able to deal with their helplessness"; outpatient families at equivalent impairment did much better. **Proximity of help is a load-bearing input with a negative sign.**

---

## 6. `Triangle` and `Family` fields

### 6.1 Triangle

| Field | For | Source |
|---|---|---|
| `members[3]` | the three people | `[T]` Ch09 · L09.3 |
| `inside_pair`, `outside_member` | who is currently where | `[T]` Ch09 · L09.3 — **position value inverts with load**: outside is unfavoured when calm and **favoured under tension** |
| `bound_anxiety` | how much the structure is currently holding | `[M]` Ch17 · L17.1 — anxiety dilutes across three edges |
| `active` | whether it is operative at all | `[T]` Ch17 · L17.1 — "the system is calm and the triangles inoperative". **Persistent topology is stored separately from the active set.** |
| `routing_capacity` | how much anxiety it can carry | `[T]` 1979 Tape 5 — intensity "is determined by level of differentiation and principally anxiety". **A function of the members' `functional_level`, not of the triangle.** Well differentiated: patterns are mild, appear under anxiety, and go away. Poorly differentiated: more intense and fixed. |
| `activation_memory` | which circuits it has used before | `[T]` Ch09 · L09.3 — tension reroutes onto "old preestablished circuits" |
| `intensity_floor` | a permanent decrement | `[T]` Ch09 · L09.3 — an "I" position held "for even a few days" produces a **permanent** decrease in that triangle's intensity. **The state does not fully revert.** This is the counterweight to the change-back reaction. |

> **The coach never works the triangles.** Triangle activity is a *consequence* of anxiety and differentiation, so the levers are those two — get the anxiety down, then work toward differentiation — and the triangle activity follows. This is what Bowen means late in life by "I don't do much with triangles anymore… no difference how many triangles are out there": a claim about **coaching technique**, not about whether triangles operate. The object stays, because it is the structure anxiety routes *through*; what differentiation changes is how much it carries.
>
> And *detriangle* names two different operations. **Detriangling self** — withdrawing one's own emotional participation while staying in contact — is the mechanism of change. **Detriangling another** — returning an aligned third party to neutral — is a countermeasure during a move. The model must keep them apart.

> **Sibling conflict is not dyadic.** `[T]` Ch21 · L21.8 — "Conflict between siblings consists **almost universally** of a triangle between mother and two children in which mother has a positive relationship to each child and the conflict is fought out between the children." **A peer-conflict event must instantiate the parent-level triangle, and intervening on the peer pair alone must fail.**

### 6.2 Family

| Field | For | Source |
|---|---|---|
| `undifferentiation_budget` | the quantity to be absorbed | `[T]` Ch16, Ch09 · L16.2, L09.2 |
| `sinks[3]` | marital conflict / spouse dysfunction / child projection | `[T]` — see §7.1 |
| `distance_baseline` | always-on, **outside** the budget | `[T]` — see §7.1 |
| `overflow` | "still free-floating immaturity"; destination is **conflict with families of origin**, not distance | `[T]` Ch09 · L09.2 |
| `ambient_anxiety` | L, X, E — the societal dials | `[T]` Ch13 · L13.1 |
| `access_vector` | which external ties are reachable | see §6.3 |
| `nodal_calendar` | deaths, births, launches, marriages, illnesses, job loss | `[M]` Ch22 · L22.2 — natural issues supply the working material |
| `leadership_office` | occupant + **sphere of responsibility** | see below |
| `beliefs` | the family's shared account of itself | §9 |
| `tolerance[person]` | per-agent tolerance for disturbing behaviour | `[T]` Ch08 · L08.1 |

**`leadership_office`.** `[T]` Ch03 · L03.6 — the family stalled for two months when the mother did not resume the lead after a vacation, and resumed "immediately" when the therapist recognised her position. **The mechanism is vacancy**, and the stall is released by *external recognition* rather than by the occupant. `[T]` Ch20 · L20.1 — the office generalises to work systems, stated bidirectionally, and carries a **sphere**: "the person who works toward the differentiation of self does not have to be the boss… His effort can be effective in the area in which he has administrative responsibility."

> **Sphere is a move *permission scope*, not a propagation boundary.** The sentence relaxes an authority precondition and says nothing about effects stopping at a boundary — while the headline claim asserts resolution of "the problem in **the organization**". The old engine's global `family_leader_mask` is wrong in the other direction. And note the headline claim is stated three times and **weakened four ways at the chapter's close** — it needs an explicit propagation rule and a test that it actually fires.

### 6.3 `access_vector` — a standing input, not a gate

**For:** what the societal dials concretely mean. Which external ties are available at all — healthcare, counselling, stable employment, a school that notices.

**Does:** contributes its effect **whether or not any move uses it.**

**Source:** `[T]` Ch11 · L11.5 — "What was important was that the system was open and they could attend if they wished", stated **after attendance had collapsed.** `[T]` Ch18 · L18.3 — "It was important for him to know there was **new land for him, even if he never went to it.**"

> **Two chapters, two settings, one mechanism: availability itself carries the effect, independent of use.** This is a belief about what is available — **not a stock**, and **not a gate on moves**. The first reading modelled physical distance as a depletable relief reservoir (frontiers → colonies → mobile jobs → nothing). There is no per-move draw-down anywhere in the text.

### 6.4 The societal dials

**Does:** three scalar dials shift the family's ambient anxiety and stressor schedule. Society is a **dial on the symptom threshold**, not a separately simulated layer.

> **The three dials, and where they actually come from.** They arrive from the grid engine as `L`, `X` and `E`, and until this revision their meanings appeared nowhere in any design document — only as Pygame log strings in `src/main.py`: Leadership, Social Media, Climate. Renamed `societal_leadership`, `media_amplification` and `resource_pressure`.
>
> **`resource_pressure` is `[T]`, and it is Ch18's own thesis** — not an inheritance. "A spectrum of problems associated with **population explosion play a major role in man's deeper anxieties**", with "the rapid depletion of world's natural resources" and "certain natural resources are nearing exhaustion". The chapter's central claim is that society "appears to be functioning on a less differentiated emotional level than twenty-five years ago, that this **may be related to the disappearance of land frontiers**." It acts through **awareness of diminishing availability**, not consumption — "It was important for him to know there was new land for him, even if he never went to it" — which makes it the societal-scale instance of §6.3's availability mechanism, and **not** the grid engine's metabolic drain.
>
> **`media_amplification` is a societal input the family modulates.** Its effect is scaled by the receiving family's differentiation rather than applied uniformly — §6.4's decoupling guard at the level of one dial — and it is the clearest case of a stressor that **converts from exogenous to endogenous**: it arrives from outside and then sustains itself through the chronicity integrator (§3.3) once internalised. Ch18's population-density channel is the corpus's nearest named analogue.
>
> **`societal_leadership` stays `[I]` in its functional form.** Ch18's six named downward channels do not include leadership quality, though Ch13's only stated reversal mechanism at societal scale does require a single principled leader.


**Source:** `[T]` Ch13 · L13.1 — "Changing societal attitudes creates an environment that encourages behavior problems that would not have previously been symptomatic." The same family at unchanged differentiation produces symptoms in a regressed era it would not otherwise.

**Calibration:** `[#]` Ch13 — **ten scale points of societal regression in twenty-five years**, against a population spread of about **fifty points**.

> **No annual rate is stated and none can be derived.** Bowen's own curve is non-monotone with an *upward* final segment, so dividing the two figures contradicts his chart. A "~0.4 points per year" constant was manufactured on the first reading and is withdrawn.

**Named downward channels** `[T]` Ch18 · L18.4: labelling/diagnosis propensity (fixes the problem in the patient, absolves the family), overleniency of officials and laws, helping-programme intensity (**impairs recipients**), school structure at junior high, population density, era-dependent symptom form.

**The decoupling guard** `[T]` Ch18 · L18.5: well-differentiated families "come to function far better than the societal level." **If turning the dial moves every family proportionally, the implementation is wrong.**

**The togetherness ratchet, conditionally** `[T]` Ch13 · L13.2: each won togetherness contest establishes a persistent new norm, so regression is stepwise and path-dependent — **but only while anxiety continues.** "In calmer periods the shift can go back and forth, with neither overriding for long periods." Acute anxiety produces regression that reverses on its own; only *chronic* anxiety ratchets, and the trigger is specifically **decisions taken to allay the anxiety of the moment**, not anxiety itself. The **individuality side is hard-capped and that cap is unconditional**: "There is never a threat of too much individuality."

---

## 7. Invariants

These are asserted every tick. They are what makes the engine a model of *this* theory rather than a plausible-looking simulation.

### I1 — The family undifferentiation budget has **three** sinks

**Does:** marital conflict, dysfunction in one spouse, and projection onto a child draw on one quantity. What fuses with a child is set by the family's total undifferentiation "**and by the amount absorbed elsewhere**."

**Source:** `[T]` Ch16 · L16.2, Ch09 · L09.2, Ch18 · L18.1.

**Two falsifiers Bowen supplies himself** `[T]` Ch09 · L09.2: marital conflict alone does **not** impair children; children **are** impaired in calm marriages. A chronically ill parent binding load **protects** the children.

### I2 — Emotional distance runs **outside** the budget

**Does:** always-on, universal, and not a competitor for the budget.

**Source:** `[T]` Ch16 — "**Other than the emotional distance**, there are three major areas…". Ch12 calls it "almost universal" and excludes it from its own count of four patterns. Ch09 has it a paragraph earlier and uncounted. Ch18 enumerates twice and disagrees with itself, listing distance as (a) and then giving "three important patterns" forty lines later.

> **Four chapters agree. There was never a three→four sink transition** — that was manufactured by inserting the word *absorbing* into Ch16's sentence. **Implemented as a fourth sink, a family discharges its whole load into distance and reads as untroubled**, which is the opposite of the claim.

### I3 — The per-person life-energy budget

**Does:** relationship-seeking and goal-directed activity are zero-sum per person. §3.7.

**Source:** `[T]` Ch10 · L10.11.

### I4 — Dyadic exchange conserves **pseudo-self**

**Does:** "In any exchange, one gives up a little self to the other, who gains an equal amount." One spouse's functional gain equals the other's loss.

**Source:** `[T]` Ch16 · L16.2, Ch21 · L21.1, Ch07 · L07.2, Ch12 · L12.1.

### I5 — Solid self is **exempt** from fusion

**Does:** the conserved quantity in I4 is pseudo-self only. Solid self does not participate.

**Source:** `[T]` Ch16 · L16.2.

> This is sharper than the old engine's undifferentiated `functioning_balance`, which conserves everything.

### I6 — Anxiety is conserved and redirected, never destroyed

**Does:** when a symptom-bearer is removed, a tie is cut, or a triangle discharges, the bound anxiety must be accounted for — relief in one member, re-focus onto another, or a standing residual. **Blocking one channel raises flow on the rest.**

**Source:** `[M]` Ch08 · L08.3, Ch11 · L11.1, Ch09 · L09.2. Five chapters, about three independent settings.

> **Two things the first reading got wrong here, both now corrected.** **Nothing in Ch08 was measured.** The chapter contains no counts at all. Phase one measures *incidents* ("the frequency of incidents remained the same"); phase two measures *complaints* — and complaints were the intervention's stated goal, so **the outcome variable switches mid-sequence**. "Frequency dropped to zero" is not in the text. **Relocation carries a sign.** Acting out moving from the community back into the family is "**a hopeful sign**." So the adversarial test cannot score incident count alone — it must score **count and location**. **Ch11 asserts no conservation of its own**; it supplies the displacement *sequence*, which is weaker.

### I7 — There is no exit from the field

**Does:** silence, withdrawal and cutoff are moves *inside* the system. The runaway fuses into a new family and reproduces the pattern.

**Source:** `[T]` Ch10 · L10.5. Corroborated from the professional side: `[T]` Ch19 · L19.3 — the doctor's silence is captured too.

### 7.2 The standing load

**For:** the term that stops an event-driven loop from getting cut-off backwards.

**Does:** every tick, before any event is delivered, each person takes a load from **every** tie as a function of its bond energy, divided by their `functional_level`. Runs whether or not anything happened.

**Why:** a pure event loop reads *fewer events* as *less arriving anxiety*, so severing a tie looks like pure relief. In the quiet years between nodal events that shows a cut-off family as calm — when the whole claim is that the intensity has gone underground rather than away.

**Source:** `[T]` Ch02 · L02.7, Ch08 · L08.2, Ch10 · L10.3.

**Three consequences, each testable:** a `TRIGGER` event (a holiday, a mention, someone else's news) spikes the standing term with **no contact at all**; a `RECONCILIATION` event converts standing load back into interaction-driven load; and the same primitive covers institutionalization — a removed member becomes non-interactive while keeping a worry edge to everyone.

---

## 8. Clocks, latencies and windows

| Clock | Period | Carries | Source |
|---|---|---|---|
| **Fast tick** | 1 week | events, appraisal, acute anxiety, tie tension, triangle position shifts, symptom accumulation | `[I]` the week is a modelling choice; `[M]` Ch18, Ch01 require a fast scale |
| **Slow tick** | 1 year | differentiation drift, chronic anxiety, life stage, mortality, nodal calendar | `[#]` Ch16 · L16.3 — generational change; `[#]` Ch11 · L11.3 — "a certain amount of time **on the calendar**" |
| **Edge latency** | hours to months | per-tie delivery delay | `[#]` Ch01 · L01.1 |
| **Follow-up window** | next day, relative to **the meeting** | revert-on-timeout after `I-POSITION` | `[T]` Ch21 · L21.4 |
| **Reaction decay** | unfed | the change-back reaction fades unless the mover feeds it | `[T]` Ch01 · L01.8, Ch21 · L21.4 |

**Durations the corpus actually states** — usable as calibration targets, each verified verbatim:

| Quantity | Value | Source |
|---|---|---|
| One unilateral stand → psychotic symptoms gone | **3 days** | `[#]` Ch05 · L05.1 |
| …then collapsed, marital tie unchanged | **1 month** | `[#]` Ch05 · L05.1 |
| Overadequate parent destabilises after the inadequate one asserts | **within days** | `[#]` Ch03 · L03.3 |
| …settles calm and firm | **within two weeks** | `[#]` Ch03 · L03.3 |
| …whole configuration reverts | **after a month** | `[#]` Ch03 · L03.3 |
| Child symptoms fall once parents hold focus on themselves | **a few weeks** | `[#]` Ch06 · L06.1 |
| The anxiety wave: death → sister symptomatic → stock conflict → brother immobilised | **~2 / ~2 / ~3 weeks** | `[#]` Ch21 · L21.4 |
| Planning before a differentiating trip | **8 weeks** | `[#]` Ch21 · L21.4 |
| Letter mailed / reading call | **T−14 / T−7** | `[#]` Ch21 · L21.4 |
| The meeting; "lost its emotional punch" | **2 hours; about two-thirds through** | `[#]` Ch21 · L21.4 |
| Best adaptation after | **~3 years** | `[#]` Ch21 · L21.4 |
| Family group therapy plateau | **12–20 sessions** | `[#]` Ch10 · L10.1 |
| Average motivated family | **about 4 years** (scoped to "upper middle class families") | `[#]` Ch11 · L11.3 |
| Coaching dose that works | **15–30 min every month or two**; **1–2 a year fails** | `[#]` Ch22 · L22.4 |
| Generations to schizophrenia | **8 to 10** (revised *up* from "at least three") | `[#]` Ch16 · L16.3 |

> **Reversion is part of the course, not failure.** `[T]` Ch03 · L03.3 — after reversion "it is easier for the father to pull up and less threatening for the mother to let go." **A model that treats relapse as a failed run is wrong.**

> **More contact does not produce more change.** `[T]` Ch11 · L11.3 — monthly families made as much progress "and possibly even more" than weekly. "It takes a certain amount of time on the calendar… not decreased by increasing the frequency of appointments." Note this is introduced as **"my conviction"** that the result "fits with", not as a finding.

> **No per-generation decline rate exists.** `[T]` Ch16 · L16.3 — the rate is **stochastic**: fast for a few generations, static for one or two, then fast again, and reversible at both extremes. A "≈15 points per generation" constant was manufactured from a schematic whose steps are −15, −15, −10, −5, dropping Bowen's own cap sentence ("in the average situation the immaturity would progress at a much slower rate").

---

## 9. Events, routing and the belief layer

### 9.1 The record

`year · week · target · kind · magnitude · duration · exogenous · route · fidelity · witnesses`

**`duration`** — `[M]` Ch18, and the old engine's counter-example: statistical input must generate **spells with a start and a duration**, never per-tick coin flips. Duration is the entire reason job loss stresses a family.

**`exogenous`** — keeps endogenous cases countable at readout. Without the flag the two are indistinguishable and the incidence calibration is impossible.

### 9.2 Illness is an output before it is an input

**For:** the trap that would make the model unfalsifiable.

**Does:** the dependent variable is the symptom and **which member carries it**. If illness is drawn from a national incidence table, the most interesting thing the model has to say becomes one of its inputs.

| Origin | Examples | How it enters |
|---|---|---|
| **Exogenous** | plant closure, recession, road accident, pandemic, out-of-family death | sampled or scheduled; arrives **undamped** — differentiation governs the response, not the event |
| **Endogenous** | stress-linked illness, emotional symptoms, job loss through deteriorating functioning | generated by accumulated load; **never sampled** |
| **Mixed** | most real illness, much real job loss | both fire; the flag keeps them countable separately |

> **The wave applies gain; it does not create pathology.** `[T]` Ch21 · L21.4 — Bowen states the mechanism in advance: the anxiety "**amplifies minor problems into major ones at vulnerable points**." The nodes are already named by the family's absorption profile. And in that case the wave **entered from outside the family** — through a marital link, from a family of origin that is not the one being modelled.

### 9.3 Routing, source and fidelity

Three separate findings that all constrain how an event resolves:

- **Source position.** `[T]` Ch04 · L04.4 — identical content lands oppositely by speaker. A therapist or outside figure naming the mother's projection → mother, father *and* patient attack or withdraw. The same thing said by the patient or the father → "a significant beneficial emotional reaction." The verbs are asymmetric too: an outsider "suggests", an insider "confronts" — **the milder outside act is worse than the stronger inside act.** Corroborated by provenance: self-initiated versus "the doctor told me to" (`[T]` Ch05 · L05.6).
- **Route.** `[T]` Ch10 · L10.14, Ch14 · L14.1 — **direct dyad amplifies, routed through a neutral third damps**: "Even when the emotional climate is calm, direct communication can increase the emotional tension."
- **Per-hop fidelity.** `[T]` Ch01 · L01.5 — distortion compounded across each private relay between staff.

**Together these make "who talks to whom in whose presence" mechanically consequential** — which is what Ch01's working rule turns on: nobody discusses an issue until all parties are in the same room.

### 9.4 The witness path

**Does:** a person who overhears an event not addressed to them still appraises it. Accumulated witnessing is what sets chronic anxiety in childhood and makes someone the projection target.

**Source:** `[M]` Ch06 · L06.1. The projection target stops being assigned at birth by a threshold and becomes an **outcome**: the child who received the most projection-type events.

### 9.5 The belief layer

**For:** what the family holds to be true, separate from what is true.

**Does:** a family-level and per-person account of its own history, actively distorted, that the agents act on. Institutional acts (hospitalization, a prescription, an insurance form) write to it **with hysteresis** — verbal denial cannot reverse them.

**Source:** `[T]` Ch09 · L09.4 — "The family emotional system operates always to obscure and misremember and to treat such events as coincidental." `[T]` Ch10 · L10.9 — the hysteresis.

**Two hard constraints on how it is built:**
- `[T]` Ch19 · L19.6 — **truth value and emotional function are orthogonal.** "The estimate might be accurate or not and yet be largely in the service of a denial." **A belief layer must not be implemented as "the emotionally driven claim is the false one."**
- `[T]` Ch15 · L15.2 — the shock wave reaching an unattached grandchild does **not** need a second edge type. The dependence is **denied — invisible to the family** — and the path is the ordinary two-hop one Bowen supplies: grandmother → daughter → son. **Belief layer plus multi-hop propagation, not new topology.**

> A research instrument that reads out what *happened* while the family acts on what it *believes* is missing the mechanism.

---

## 10. Readouts — including the three that lie

**Two outcome axes, minimum.** `[T]` Ch10 · L10.1 — family group therapy reliably relieves symptoms and "does not provide the structure for a higher level of differentiation of self." **A single wellbeing scalar cannot represent this chapter at all.**

**What to report:**

| Readout | Source |
|---|---|
| Which generation carries the symptom, as a probability across three | `[M]` Ch16 · L16.3 |
| Which specific person is most often the symptom-bearer, and how concentrated | `[T]` Ch09 · L09.2 |
| Which tie ruptures first, and when | `[M]` Ch10 · L10.5 |
| Mean differentiation trajectory over 40 years, with a band | `[M]` Ch16 · L16.3 |
| Share of anxiety bound by each of the **three** mechanisms, with the distance baseline reported **separately** | `[T]` §7.1–7.2 |
| Rate of "the language of the triangles" — emotional issues between people, withdrawal, silence, cliques, alliances, gossip about an absent member | `[T]` Ch20 · L20.3 |
| The coach's own reactive state | `[T]` Ch20 · L20.2, Ch10 · L10.10, Ch19 · L19.1 |

**Never a single run.** With twelve agents there is no law of large numbers to lean on. A single trajectory is a sample, not a result.

### 10.1 Three readouts that will lie to you

**1. Overt emotionality is not monotonic in level.** `[T]` Ch16 · L16.4 — "People in the moderate range of differentiation have the most intense versions of overt feeling." **It peaks in the middle.** Any readout using visible emotionality as a proxy for differentiation is wrong **in exactly the band where most agents sit.**

**2. Position-taking behaviour does not mean differentiation.** `[T]` Ch10 · L10.12 — the fifth band, upper 25–50: dogmatic, compliant or rebellious, "intellect in the service of the relationship system." These agents score high for the wrong reason.

**3. Any estimate of differentiation is biased by construction — and the measurement literature bears this out.** `[T]` Ch17 · L17.2 — pseudo-self is lent, borrowed and traded between people, which "**results in false readings when one attempts to estimate levels of differentiation.**" Recoverable only over a life course. **A coach-side or readout-side estimate must be modelled as biased and noisy**, not as a clean read of the true value.

> This is the one place the model has external corroboration. The DSI's Fusion With Others subscale — the construct most directly about self borrowed from others — produced reliabilities of .57–.74 and **no significant relationship to psychological adjustment, problem-solving or relationship satisfaction**, while the instrument's other three subscales worked at α ≥ .80. It took a five-year rebuild. Suggestive, not proof; but it is what Bowen's claim predicts would happen, and it means agreement between this model and a DSI score is weak evidence for both.

> **Gossip is a readout of the communication layer, and it moves inversely.** `[T]` Ch21 · L21.9 — gossip volume **rises as direct contact falls.** The only instance in the corpus of rerouting at the communication layer.

### 10.2 The coach's own state is a detector

**Does:** Bowen's earliest reliable indicator of rising system anxiety is not the group's behaviour but **his own urge to become critical** of the faculty for being overcritical of trainees. Prescribed sequence: restrain the urge → assume he is playing a part → work on himself → identify which of three named lapses applies (**lost contact / failed to state position / failed to detriangle**) → modify it.

**Source:** `[T]` Ch20 · L20.2 — with three qualifications: "I **often** noticed this first" (not always); the triangles readout is "**another** way of detecting", so self-state is one of two routes; and the three lapses are *recalled*, not diagnosed in the moment. `[T]` Ch10 · L10.10 gives three more readouts of coach fusion: atypical patterns, slowed progress, family passivity.

> **A stateless external coach loses one of the two detection channels entirely.**

---

## 11. The acceptance tests

Eleven property tests over ensembles, written before tuning. Two properties matter more than their content: each **asserts a direction of difference between two arms**, not an absolute threshold, so it survives recalibration; and each must be **proved failing by mutation** — delete the mechanism it names, confirm red, restore — because a test over a stochastic ensemble is exactly the kind that passes for the wrong reason.

| # | Test | Implements | Grade |
|---|---|---|---|
| 1 | **Differentiation protects.** Two families identical but for `basic_level`, same stressor schedule: the lower-`basic_level` family reaches symptom threshold sooner in a significant majority of seeds. | Ch09 · L09.1 — symptoms are a threshold on the functional level, and transfer scales inversely with the basic level | `[D]` |
| 2 | **Symptoms concentrate.** Load lands disproportionately on the member who received the most projection-type events, not uniformly across children. | Ch09 · L09.2 — the two falsifiers Bowen supplies; Ch16 · L16.3 — the primary projection object emerges *lower*, minimally involved siblings about the same, those outside the process *better* | `[D]` |
| 3 | **Triangling relieves the pair and costs the third.** After a `TRIANGLE` move the dyad's tension falls and the third person's anxiety rises within the same tick. | Ch09 · L09.3; Ch08 · L08.5 — parental conflict subsided precisely as the parents rejoined forces and resumed projecting | `[D]` |
| 4 | **Cut-off trades now against later.** `CUTOFF` drops the actor's acute anxiety immediately and raises the family's total anxiety at the next nodal event, against a matched no-cutoff arm. | Ch10 · L10.3; Ch08 · L08.2 — zero re-activation latency on reunion | `[D]` |
| 5 | **The system pushes back — and it is a debit, not a script.** After `I-POSITION`: tension rises for a bounded window, the push-back appears as **symptom in a named third person**, the trajectory is a **damped oscillation with hysteresis**, the reaction **decays unless fed**, and skipping the next-day follow-up **reverts** the gain. Absence of any reaction means the move did not land. | Ch02 · L02.3 and Ch03 · L03.3 for the observation; Ch21 · L21.4 for the ladder, the follow-up and the diagnostic rule; Ch10 · L10.11 for the mechanism | `[M]` `[T]` |
| 6 | **Transmission is multigenerational.** Over three generations with no external stressor change, mean `basic_level` in the projection line declines while the non-target line does not. | Ch16 · L16.3 — and the clean controlled comparison it supplies: two families at identical levels, one keeping contact (symptom-free for life, level preserved next generation), one cutting off (symptoms, dysfunction, lower level next generation) | `[D]` |
| 7 | **Position, not skill, makes a professional useful.** Hold the coach's parameters fixed and vary **who talks to whom in whose presence**. The topology arms must differ. | Ch01 · L01.4 — multi-channel contact distorted within minutes while staff held excellent intellectual understanding; Ch10 · L10.4 — "any third person… no matter what the subject matter"; Ch19 · L19.3 — silence is captured; Ch17 · L17.3 — the observer changes the configuration | `[M]` |
| 8 | **Endogenous incidence lands near the national rate.** Aggregated over a large ensemble, model-generated illness and job-loss incidence by age falls in a stated band of published rates — **counting endogenous events only**. | §9.2 — statistics as a calibration target, never an event generator | `[I]` bands |
| 9 | **Sibling position shapes functioning.** Holding differentiation constant, position in the sibling order produces a detectable difference in propensity and in symptom incidence. | Ch21 · L21.5 | `[D]` |
| 10 | **Cut-off begets cut-off.** A generation containing a cut-off produces more in the next than a matched arm without one. | Ch16 · L16.3 — the positive feedback loop; Ch18 · L18.1 — cutoff is an *input* to intensity | `[M]` |
| 11 | **Removal produces three phases, not a step down.** The institutionalization arm reproduces rise → partial relief with redirected focus → sustained residual; hospitalize-and-release reproduces a temporary dip and re-escalation on return. Total system anxiety is conserved across the removal to a stated tolerance. | Ch08 · L08.1 — removal fires off the **remaining members' tolerance**, not the patient's severity; §7 I6 | `[M]` |

### 11.1 Four tests the corpus asks for that the eleven do not cover

| # | Test | Source |
|---|---|---|
| 12 | **Curing a symptom without changing the deficit raises tension.** Remit a spouse's dysfunction while leaving the functioning balance intact: marital conflict must **rise**. | `[T]` Ch12 · L12.2 — the wife's drinking remitted and her regaining self was followed by "a period of fairly intense marital conflict"; `[T]` Ch07 · L07.2 — harmony holds "as long as the disabled spouse does not recover". The general rule is Ch21 · L21.9: **a chronic adapter who stops.** *The chapter's sharpest falsifiable prediction.* |
| 13 | **Help relocates incidents; it does not reduce them.** A coach that supplies help and instruction produces **the same incident count in different locations**. Score count **and location** — relocation from the community back into the family is "a hopeful sign", so count alone gets the sign wrong. | `[M]` Ch08 · L08.3 |
| 14 | **Management technique has zero independent effect while marital distance is high.** Any intervention aimed at the symptom-bearer must produce **no improvement** in that condition. When the parents are close, "they could do no wrong" — firmness, permissiveness, punishment and talking it out all work. | `[T]` Ch04 · L04.2 |
| 15 | **Death destabilises exactly as recovery does.** A stabilising arrangement built on one member's impairment must break on that member's **death** as well as on their recovery — the stabilisation holds "as long as the incapacitated one **lives**". | `[T]` Ch07 · L07.3 |

---

## 12. What is deliberately not in the model

### 12.1 Five mechanisms that must never be built

Each was proposed on the first reading of the corpus and withdrawn on the second. They are recorded rather than deleted because a mechanism that is merely absent gets re-invented.

| Not this | Why | Ledger |
|---|---|---|
| **A permission scalar** — the excluded third gates a dyad's closeness | The mothers **approved** the father–son efforts and the efforts still failed. The gate is the incumbent's **unreleased investment in the competing tie**, which the model already has. | L02.1 |
| **Ally cancellation** — support from a family member cancels a differentiating move | Bowen never says it. The mechanism is **displacement**: an ally opens a new peripheral triangle. Active detriangling of an ally is verbatim and goes in; the cancellation term does not. | L21.2 |
| **A durable projection target queue** — remove the symptom-bearer and a replacement permanently takes the position | **Bowen retracts this himself and dates the error**: six months of observation confirmed the uptake, two and a half years reversed it. Fast redirection survives; durable uptake does not. | L01.7, L04.6 |
| **Distance as a depletable relief stock** | No per-move draw-down exists. The relief is **availability knowledge** — "even if he never went to it." | L18.3 |
| **"Content is noise"** as an argument for the deterministic core | Not in the text. Bowen gives two *attention* prescriptions; his own blame-versus-responsibility hazard **requires** reading an inner orientation no rate can supply. What survives: the anxiety readout is computable from event rates alone. | L20.3 |

### 12.2 Two things Bowen says he does not know

Recorded so the model does not quietly invent an answer and present it as theory.

- **What decides whether a problem stays in the spouse dyad or is transmitted to a child.** "For some reason, their children remain relatively uninvolved." `[T]` Ch06 · L06.5.
- **What selects which spouse takes the dominant pole at identical levels.** `[T]` Ch06 · L06.5.

Both are `[I]` in the model. Whatever rule is implemented is the modeller's, and must be labelled as such.

### 12.3 One negative finding, recorded so it is not invented later

**There is no anniversary effect.** `[T]` Ch15 · L15.4 — delayed and prolonged effects are reported (2-year chains, symptoms at 5 years, mourning 6 years late) but **no date-anchored recurrence is named anywhere.**

---

## 13. Two contradictions — **resolved at step 6**

Both were carried as genuinely unresolved through pass 2. **Both are now settled**, and both settled the same way: against the primary source rather than the summaries. Full argument in [`theory/_RESOLUTIONS.md`](theory/_RESOLUTIONS.md). What follows is what the model implements.

> **Why they survived pass 2.** In each case the resolving sentence is in the source and in neither extraction. One of them was a contradiction with a section *heading*. Where a contradiction blocks a decision, go to the text.

### 13.1 Threesome or twosome? — **neither. Three live positions.**

**The finding was never about family members.** Ch03 counts only the family because its therapist is deliberately structured out — "the family work on its own problem in the hour while the therapist **observes from the sideline**". Ch14's section is headed "Family Systems Therapy with Two People" and describes, in its own words, "**the triangle of the two most important family members and the therapist**", with "clearer definition of the therapist's functioning **in the triangle**."

**Both configurations satisfy one rule.** What changed between 1959 and 1975 is who occupies the third position, not how many are needed.

```
positions_live(group) = |{ p in group : present AND not fused into another member }|
avoidance_available   = positions_live < 3          # a step, not a gradient
```

- An external agent in the outside position **counts**. `[T]` Ch10 · L10.4 — "any third person… no matter what the subject matter."
- An external agent who has taken a side **does not** — he has fused into one of the two.
- A member present but **emotionally inactive** does not. `[T]` Ch17 · L17.1 names that state directly.

**This also dissolves the "reversal" in Ch14.** In Ch03 a family member talking to the therapist *removes* the third position, leaving a twosome that can avoid; in Ch14 the therapist *is* the third position, so talking to him keeps three live. **Same act, opposite meaning, because the third position moved — C10 applied to technique.**

**What Ch14 actually changed**, each with its own stated motive and neither about group size: the child was removed because "in the **physical presence of the child**, it was difficult to get the parents to focus on themselves"; and communication was routed through the therapist — which is not new in Ch14 at all (Ch10 states it in 1971 as "in use more than five years"; Ch11 dates it to "after about 1962").

### 13.2 Does a witness help or displace? — **alignment is the gate, not knowledge.**

**Nothing in the corpus penalises a witness.** What is penalised is a second person **taking a position on the mover's behalf**, because that opens a new peripheral triangle and the tension slips into it — which Bowen names as his own undoing: "I had done fairly well in detriangling myself from one triangle, only to have the tension slip into another triangle; **this pattern had been my undoing.**"

Every statement of the rule is in the language of sides: "come over to **my side**", "getting on my 'side'", "align themselves", "taking sides with me".

**The chapter's one worked ally case settles it.** His younger sister wrote "**I am back of you if I can be of help**" — she already knew everything. Bowen calls it "a red flag" and handles it until "**she retreated from taking sides with me**." Her knowledge was untouched; only her position changed.

**And the same chapter contains a witness who succeeds.** His wife "had no direct knowledge", was present throughout, and "**did not ask a single question nor make positive or negative comments about my family at any time during the trip. This had never happened before.**" He reports her non-participation approvingly, in his most successful effort.

**Ch10 bars alliance from the helper's side in the same terms:** the therapist may help "**without being perceived as against the family**." Its spouse is the *target* of the move and the next mover — the "important other" from whom energy is withdrawn — never a supporter.

**Three objects, three rules.** Conflating them is what produced the contradiction:

| Object | Rule |
|---|---|
| **The programme** | Tell no one **inside the system** — not because knowledge leaks, but because inside the system knowledge produces alignment |
| **The act** | **May be announced**, to the party in that dyad. Ch10 does so, leading with confidence in the other's competence |
| **A third party's position** | **Must be neutral.** A knowing non-aligner is free; a supporter is detriangled; a helper helps only from a position not perceived as against the family |

**Secrecy is therefore a hazard rate over alignment, not an independent gate** — which is why the rule is scoped in the same sentence to "another person **who is part of the system**", and restated a third time hedged ("it is doubtful that any differentiation will result").

```
if third.position == NEUTRAL:            effect = none
if third.position == ALIGNED_WITH_MOVER: open_peripheral_triangle(third, mover)
if third.role == EXTERNAL and perceived_against_family(third):
                                         open_peripheral_triangle(third, mover)

P(align | knows, inside_system)  = high     # "as if by telepathy"
P(align | knows, outside_system) = low      # the wife
```

**One move this adds to §5.7.** `PREVENT_ALIGNMENT` — preemptive, and previously unmodelled. Ch21's two deliberately contradictory letters, sent in the same mail within the hour, exist for exactly this: "**The conflicting messages were designed to prevent any one segment of the family from getting on my 'side.'**" The repertoire has `DETRIANGLE`, which is reactive. It has nothing that acts on *potential* alignment.

**And a third secrecy mechanism**, in the source and in neither extraction — `[T]` Ch21 · L21.11: "**the element of surprise that is essential if a differentiating step is to be successful.**" Surprise attaches to **the target**; latency and ownership attach to **the mover**. Three mechanisms, two objects — which is why an announced *act* costs nothing.

### 13.3 A third, smaller one: when is the basic level fixed?  `[X]` — **still open, but decided for the model**

| Chapter | When |
|---|---|
| Ch21 (1972) | consolidated **in a marriage** |
| Ch22 (1974) | **early childhood**, with movement in both directions in childhood *and* adulthood |
| Ch16, Ch17 (1976) | when the young adult **leaves the parental family** |

Ch21 contradicts *itself* here — the freeze claim and a "basic increase… which can never return to the former level" sit ~5,000 words apart in different sections and neither refers to the other.

**Decided for the model:** Ch21's freeze-at-marriage is **outvoted by three chapters, two of them later**, and Ch16 explicitly restores slow movement ("it is possible to make slow changes"). So `basic_level` moves — but ⚠ **not by a ratchet.** It is **derived**, an estimator over sustained, broad, load-tested `functional_level` history (§3.2a), with a **capacity floor at 25** below which it cannot rise at all. The fast quantity is the **swing** term inside `functional_level`. Do not implement it as frozen, and do not implement it as incremented. This is a modelling decision over a genuine disagreement, not a resolution of it.

### 13.4 What both resolutions have in common

The same predicate sits under both, and it is already the model's own:

> **A position counts as live only if its occupant is not fused into another.**

In 13.1 it decides whether three people can avoid their issue. In 13.2 it decides whether a third party is a witness or a peripheral triangle. **The coach's neutrality, a family member's self-control, an ally's detriangling and the three-position floor are one rule seen from four angles** — which is what Ch10 says outright: the therapist's neutrality and a family member's self-control are "the same operation from different positions."

**Implement it once.**

---

## 14. A note on framing

Bowen states that he deliberately excluded "models from the sciences of inanimate things" — the class this simulation belongs to.

**The scope is much narrower than it first appears.** It is "for my research"; it "governed nothing except the **background thinking of the research staff**"; it is offered "**without saying this should be done**"; it rests on "no more than an educated guess"; and in the same interview **he concedes the identical charge against his own *triangle***: "It sounds almost mathematical." / "You are correct."

So: **a named methodological preference, scoped to his own research staff.** It belongs in the record and it is worth taking seriously. It is not a prohibition and should not be quoted as one. `[T]` Ch17 · L17.5.

**And the wider framing was wrong, which matters more than the narrowing.** In a late interview with Michael Kerr, Bowen insists theory must be grounded in nature — "in the force that causes grass to grow, that causes the world to turn" — and repeatedly anticipates measurement: "there are **biochemical markers or indicators that could be measured** if we were smart enough to pick them out"; "we wouldn't have to observe people, the chemistry would tell us." **`Q-VALIDATION = NO` records a limit of what he had, not a principle he held.** He wanted measurement and expected it to arrive. Objecting to models of *inanimate* things while demanding that theory be grounded in *nature* are compatible positions, and this project had been collapsing them into one. See [`theory/_KERR_INTERVIEWS.md`](theory/_KERR_INTERVIEWS.md).

**And one more of his own, worth keeping in view.** `[T]` Ch14 · L14.4 — "The type of approach is not a positive index of success in therapy"; family therapy "is still more of an art than a science." Therapist skill may dominate method choice, which cuts against any model where method fully determines outcome.

---

## 15. Source index — what each chapter supplies

| Ch | Year | What the model takes from it |
|---|---|---|
| 01 | 1957 | anxiety transfer as a quantity with **per-edge latency**; position ≠ involvement; the coach is capturable by topology; per-hop fidelity; projection is stress-gated; the change-back ladder already present, with an **inequality** success condition |
| 02 | 1959 | investment is zero-sum across ties; symptom moves opposite to system health short-run; avoidance is a **step function of group size**; **"the surface distance controls a deeper interdependence"** — the earliest source for no-decay |
| 03 | 1959a | over/under-functioning as one bistable coupling **held per area**, with a relative, immediate flip; the full change trajectory with time constants; the **two-rung** avoidance ladder; illness on an attempted pull-up; leadership **vacancy** |
| 04 | 1960 | the seesaw as a **conjunction over both parents**; management technique has zero independent effect; the pole is assigned **by acts**; **source position** determines effect; investment **defined and valence-blind**; the retraction of the target queue |
| 05 | 1961 | symptomatic vs basic change, with tight constants; function absorbed by whoever will carry it, **including the professional**; reciprocity sticks with dyad age; **leaders move first** |
| 06 | 1961a | the child's symptom as a continuously-driven output of **directed parental attention**; relief satisfies the goal — the coach's task is **retargeting**; the unit is actionable through any single member; two stated unknowns |
| 07 | 1965 | ego-mass membership as a **continuous stress-dependent weight**; self conserved and traded, symptom absorbs the deficit; stabiliser efficacy, with level gating the repertoire; the sensitivity ranking; the **fused default** |
| 08 | 1965a | removal fires off the **remaining members' tolerance**; separation as **edge-rewiring**; substitutable discharge channels; the separation crisis as an **involuntary reflex**; the professional system is part of the family system; the symptom-first change-back ladder |
| 09 | 1966 | two-level functioning with the corpus's **only arithmetic**; the conserved "immaturity" across three channels with two falsifiers; triangle threshold, **position-value inversion**, permanent decrement; the **belief layer**; sideline coaching ranked worst in 1966 |
| 10 | 1971 | **two outcome axes**; the **per-person life-energy budget**; the neutral third *is* the intervention; no exit from the field; differentiation transfers between systems; the taboo set; institutional acts write beliefs with hysteresis; the **fifth scale band**; **financial dependence as a hard gate** |
| 11 | 1971a | the displacement sequence; **an open channel works without being used**; more contact ≠ more change; design constants; topology beats insight, second instance |
| 12 | 1974 | signed equal-magnitude transfer, hardening with duration, asymmetric to reverse; the **conflict spike** on recovery; a single non-symptomatic node can control the system; the **closeness ceiling** as a two-sided band |
| 13 | 1974a | society as a **symptom-threshold shift**; the conditional togetherness ratchet with a hard individuality cap; the **differentiation state machine** with anger as the gate; the **counterfeit move** |
| 14 | 1975 | routing gain; satisfaction diverging from structural change; the extended family as the site of work — **and seven contradictions with his own earlier text, which is why it is cited last** |
| 15 | 1976 | integration inverts the **overt** response; multi-hop propagation through asymptomatic carriers; **three tiers** of structural importance; **no anniversary effect** |
| 16 | 1976 | **the scale** — the only source for the numbers; three sinks with distance outside the budget; **pseudo-self** as the conserved quantity, solid self exempt; the generational update rule and the cutoff feedback loop; overt emotionality **peaks in the middle** |
| 17 | 1976 | boundary expansion and contraction; **two variables only, reactivity derived**; the **measurement bias**; the coach as a node with conjointly-gated efficacy; his own retractions; the framing note |
| 18 | 1973 | three inputs to intensity — including **cutoff as an input**; anxiety as master gain with **chronicity, not level**; **availability, not a stock**; the downward channels; the decoupling guard |
| 19 | 1959b | the professional's own dysfunction as readout; the critical point (three triggers); three concrete manipulation moves, **including capture-by-silence**; **token concurrence**; **truth and emotional function are orthogonal** |
| 20 | 1972 | the leadership office generalised, with **sphere as a permission**; the coach's reactive state as detector; the rate readout; blame vs responsibility; a sink **outside the family** |
| 21 | 1972 | the timed trace of one move — the most implementable passage in the book; the **verbatim reaction ladder** and diagnostic rule; three efficacy gates; **sibling conflict is a triangle**; the cross-system before/after; six concepts |
| 22 | 1974 | contact reduces anxiety but does **not** raise differentiation; the live-issue precondition and the **mover's-axis** non-monotonicity; unilateral change propagating without consent; the dosage band |

---

## 16. Provenance and how much to trust a convergence

Nine of the 22 chapters report the **same 1954–59 residential project** — Ch01, Ch02, Ch03, Ch04, Ch05, Ch07, Ch08, Ch16 (in part) and Ch19. Agreement among them is one study reported repeatedly.

Known double-counts: **Ch02 (n=5) and Ch04 (n=6) are the same father case series.** **Ch07 rewrites sixteen of Ch06's propositions**, same order, often the same words. **Ch01's father sequence is n = 1, not 2** — "the second father is still in this stage" — so every timing in it belongs to one family. **Ch21 is n = 1**, Bowen's own family. **Ch13, Ch17, Ch18 and Ch20 carry no clinical series at all** — and Ch20 contains **zero cases**, its announced illustrative case never delivered.

| Convergence | Chapters | Independent settings | Weight |
|---|---|---|---|
| Reciprocal functioning | 6 | ~4 | strong |
| The outside position | 6 | ~5 | strong |
| An act's identity depends on hidden state | 5 | ~4 | strong, and missed entirely on the first reading |
| Anxiety conserved and rerouted | 5 | ~3 | good |
| No decay with distance | 4 | ~3 | good, mechanism revised |
| The investment seesaw | 5 | ~2 | weaker than it looks |
| The differentiating state machine | 4 (+2) | ~3 | strong on shape, weak on constants |
| Relief opposes progress | disputed | — | **has a direct counterexample in Ch22** — reducing cut-off reduces symptoms *and* makes therapy more productive. Restate as *relief obtained without structural change*. |
| Calendar time, not frequency | 2 | ~2 | weak; one retells the other |
| The peripheral member moves first | 4 | **1 setting, 2 series** | **weak, and contradicted by Ch05** |

**On that last one.** Ch05 says the family *leaders* began differentiation first, usually the overadequate mother. Its own reconciliation is what the model implements: **the leader drives the process, the peripheral member performs the visible position change. Two roles, not one contested slot.**

---

*Written against `docs/theory/_LEDGER.md` at 149 findings, all 22 chapters, two passes. Every `[T]` and `[#]` claim traces to a named ledger entry; every `[I]` is a modelling decision with no source. If a constant in the code is presented as sourced and does not appear here, treat it as `[I]` until proven otherwise.*
