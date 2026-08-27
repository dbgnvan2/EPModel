---
tags: [model-bt, spec]
version: 2.0-draft, revision 4
status: FOR APPROVAL — no code until approved
date: 2026-08-27
supersedes: bowen_individual_family_model_spec.md (v1.2, frozen)
---

# EPModel v2 — agent model specification

## 0. Status, scope, and how to read

### 0.1 Scope

This specification covers **Phases B, C and D** of the sequence in `agent_model_proposal.html` §9: the objects, the event loop, the behaviour policy, the slow clock, and the twelve-person reference family.

**Out of scope, deliberately:** Phase E (the ensemble runner, seeded batches, counterfactual arms, distribution readouts, `HazardSource`, `DialSource`) and Phase F (the language layer). Phase E is specified after the core is real, because most of its content is decisions about what to measure and those are cheaper to make against a running model. Where a Phase B–D requirement exists only to make Phase E possible, it is marked **`→E`** and the reason is stated.

### 0.2 Relationship to the other documents

| Document | Role | This spec's relationship to it |
|---|---|---|
| `model_explainer.md` | what each part is for, what it does, which chapter it implements, graded `[T]`/`[M]`/`[D]`/`[#]`/`[I]`/`[X]` | **Authoritative for rationale and sourcing.** This spec states requirements and cites the explainer section; it does not restate the evidence. |
| `theory/_LEDGER.md` | 149 findings, all 22 chapters | The evidence behind the explainer. Not cited directly here. |
| `theory/_RESOLUTIONS.md` | the two settled contradictions | Cited where a requirement depends on a resolution. |
| `agent_model_proposal.html` | the architecture and the argument for the pivot | Context. Where it and this spec disagree, **this spec wins**. |
| `bowen_individual_family_model_spec.md` | v1.2, **frozen** — the grid engine | Historical. Not a source for v2. |

**One rationale lives in one place.** If a requirement here needs justification, that justification is in the explainer and this spec links to it. This rule exists because the project has already been bitten by two documents drifting apart.

### 0.3 Normative language

- **MUST** / **MUST NOT** — required. A violation is a defect, and each has a test in M11 or an invariant assertion in M6.
- **SHOULD** — required unless there is a stated reason not to; the reason goes in the code comment.
- **MAY** — genuinely optional.

Anything not expressed in these terms is explanation, not requirement.

### 0.4 ID scheme

`M<module>.<part>.<item>` — e.g. `M1.A.3`, `M11.C.5`, `M6.I.2`. **Every ID matches the single pattern `M\d+(\.[A-Za-z0-9]+)*` with each level separated by a dot, and a lowercase suffix permitted for an inserted requirement (`M4.D.6a`)**, so one regular expression extracts all of them (M11.D.10). Two-level IDs (`M0.3`, `M6.1`, `M13.1`) are IDs in their own right, not truncations.

IDs are stable and **MUST NOT** be renumbered; a withdrawn requirement keeps its ID and is marked `WITHDRAWN`. **Because renumbering is forbidden, any ID-scheme defect must be corrected before this document is approved, not after.**

Per the project convention, every function or module satisfying a requirement carries a docstring naming it:

```
Purpose: <one sentence>
Spec:    docs/bowen_agent_model_spec_v2.md#M4.C.2
Tests:   tests/bowen/test_appraisal.py::test_m4c2_gain_function_gates_above_threshold
```

Acceptance test names **MUST** embed the lowercased ID (`test_m4c2_…`).

### 0.5 The parameter rule — the most important rule in this document

**The corpus supports directions, orderings and mechanisms. It supports almost no magnitudes.** There is no instrument, no rater procedure, no comparison group and no number ever assigned to a person **in any of the 22 chapters** (external instruments exist and are recorded in `theory/_EXTERNAL_MEASURES.md`; they supply exactly one usable target, M10.C.2a) (`model_explainer.md`, "How to read an entry", and §3.1).

Therefore:

- **M0.1** Every numeric constant **MUST** carry a grade in the parameter register (M10), and the grade **MUST** match the explainer's grade for the same claim.
- **M0.2** A constant graded `[I]` **MUST NOT** be described in code, comment, docstring or output as sourced, derived from Bowen, or theoretically grounded.
- **M0.3** Every constant **MUST** be either derived from `basic_level` (M10.A) or declared in the markdown config (M10.B). **No magic literals in engine source.**
- **M0.4** Acceptance tests **MUST** assert a *direction of difference between two arms*, never an absolute threshold, so that they survive recalibration. The four exceptions are listed in M11 and each is justified there.

---

## M1 — Core objects

### M1.A `Person`

**M1.A.0** — *the word* **emotional** *in this document means INSTINCT, not feeling.* Every use of *emotional* — emotional system, emotional process, emotional reactivity, emotional cutoff — denotes the instinctual substrate, of which felt emotion is one surface expression. Bowen, twice: the emotional process is "**equivalent to instincts** — to the forces that result in birds migrating… the salmon finding his way back… on a cellular level, the force that guides an amoeba toward a morsel of food"; and "people have picked up this term I used, emotional, and **made it synonymous with feeling, which I didn't mean at all**." → `kb/kb02.md` · K02.1, `kb/kb13.md` · K13.1
**Consequence:** `acute_anxiety` and `chronic_anxiety` **MUST NOT** be modelled as feeling-states, and no readout **MUST** treat visible emotion as their measure. This is why overt emotionality peaks mid-scale (M1.A.2 note) and why a withdrawn person is as involved as an effervescent one.

**M1.A.1** A `Person` **MUST** hold exactly two state variables from which reactive behaviour is derived: a basic level and an anxiety level. Reactivity **MUST NOT** be stored. → `model_explainer.md` §3.5

**M1.A.2** `basic_level` **MUST** be on a 0–100 scale. The implementation **MUST NOT** clip to `[10, 80]`, and **MUST NOT** apply a linear transform in place of the one behavioural transition. → §3.1

**M1.A.3** There **MUST** be exactly one behavioural transition on `basic_level`, at 50, and it **MUST** be implemented as a *licence over joint decisions* — below 50 the emotional system permits the intellect its own domain **except** where a decision affects the shared life course — not as a general suppression of intellect. → §3.1

**M1.A.3b** — *the transition's **readout** is awareness; its **implementation** stays the licence.* Kerr 1988 describes the same transition differently: "**Above 50, the intellectual system is sufficiently developed to make a few decisions of its own**"; "**A criterion for distinguishing people who are above rather than below 50 is that above 50 there is more awareness of the difference between feelings and intellectual principle.**" That is a **band discriminator**, and it **MUST** be implemented as one — a readout over M4.D.1a's mixing weight — and **MUST NOT** be implemented as a second behavioural transition. M1.A.3's licence remains the behavioural form, because a Kerr formulation **MUST NOT** silently overwrite a Bowen one.
**Neither MUST be implemented as reduced cognitive capacity.** What differs across the transition is the **strength of the emotional circuits relative to the cognitive ones**, not the quality of the cognitive ones: "**the intellect operates in the service of the feeling and emotional process**". A low-`basic_level` agent **MUST** argue as fluently, and hold positions as confidently, as a high one. → *decision A2, 2026-08-27*; `theory/family_evaluation/fe04.md` · FE04.3, `fe02.md` · FE02.16

**M1.A.3a** — *the marriage-ceremony break, resolved.* Bowen states it as a fact and says he never worked out why: "pretty good friendship relationships before marriage and then the whole thing gets messed up as of the time of the marriage ceremony. **I've often wondered the why of that**, but there it exists as a fact." → user mechanism, E2, 2026-08-24; formerly open
**The mechanism:** marriage strengthens fusion, and **the couple is treated as ONE SELF — so the live question becomes who gets to decide for that self.** The lower the level, the more anxiety and reactivity this produces.
**Why it needs no new machinery.** The ceremony **changes neither person**. It **reclassifies a large class of decisions into the shared-life-course domain** — precisely the domain in which M1.A.3's transition bites. So a pair whose decisions were previously individual, and therefore outside the licence, finds most of them inside it overnight.
**Requirement:** `M2.A` **MUST** treat the marriage event as a **reclassification of decision scope**, not as a stressor and not as a change to either person's state. Below 50 the effect **MUST** scale inversely with `basic_level`. `M11.C` candidate: two arms differing only in whether the pair marries, matched on everything else, with the married arm showing higher anxiety **and no change in either party's basic level**.

**M1.A.4** `basic_level` **MUST NOT** be frozen at marriage, and it **MUST NOT** be directly writable by any move. It is **derived** (M1.A.4a), and therefore moves slowly as a consequence of the estimator's window rather than by a rate constant. → §13.3

**M1.A.4a** — *`basic_level` is an **estimator over `functional_level` history**, not a stored quantity a move increments.* It **MUST** be computed from an **elevated mean** in `functional_level`, held at **low variance**, **sustained** over a declared window, and demonstrated **across many distinct situations**. A person works on functional level; sustained, broad improvement in functioning is what constitutes evidence of a rise in basic level. **No move, and no completed differentiating exchange, MUST write `basic_level` directly.**
**Why this form:** M1.A.5 already requires `functional_level` **variance** to be a *decreasing function of* `basic_level`. M1.A.4a is that same relationship used **backwards as the update rule**, so the model carries one relationship rather than two mechanisms (M10.A.2). It also matches the model's own epistemic position: there is no instrument for basic level and what is observable is functioning. → `theory/_EXTERNAL_MEASURES.md`
**Corroboration:** years of sustained work on both parental relationships — "I did **pretty good**, but I **still would get caught up in it when I would go home**" — real functional improvement that failed in one situation, and so was not yet basic. → `kb/kb05.md`; `kb/kb07.md` · K07.2

**M1.A.4b** — *the situations in the window **MUST** be weighted by the load they imposed, and the sample **MUST** include nodal events.* An unweighted count is a failing implementation, because a benign window produces a stable, elevated `functional_level` and demonstrates nothing: **an easy life would read as differentiation.** Nodal events are what discriminate the binders — M11.C.4 requires `CUTOFF` to drop acute anxiety immediately with its cost deferred to the next nodal event, so a cut-off agent shows a durable-looking calm that holds until one arrives.
**Consequence:** the estimator separates binder-relief from differentiation **without a special rule**, because binder gains are situation-specific and collapse under load while differentiating gains hold across situations.

**M1.A.4c** The estimator's **window length** and **breadth count** **MUST** be declared in config, graded `[I]`, and the run **MUST** report a sensitivity analysis over both. No source states a window or a count; they are judgement. → M10.B.3

**M1.A.4d** — *a **capacity floor at 25**.* Below `basic_level` 25 the estimator **MUST NOT** raise `basic_level` regardless of functional history; above it, the rate **MUST** be monotone increasing in current `basic_level`. `functional_level` is **not** so constrained and moves freely at every level. A different kind of transition from M1.A.3's behavioural licence at 50 — a **capacity bound on change**, not a licence over decisions. → `kerr_book/ks05.md` · KS05.2
> "**they lack the flexibility to make basic change**… **People above 25 on the continuum can make basic changes in differentiation. The higher a person's basic level of differentiation, the more potential that person has to increase the basic level.**" And: "**Their functional levels of differentiation can change** as the level of chronic anxiety fluctuates."

**M1.A.4e** — *the estimator's observables, with per-observable reading rules.* Four domains — **work performance, education relative to opportunity, health history, relationship stability** — with "**No one piece of data is sufficient**". Each is read **asymmetrically**:

| Observable | Reading |
|---|---|
| Occupation | high → **uninformative**; **low despite opportunity** → informative |
| Health | the **adaptation to illness**, not the diagnosis |
| Longevity | **not** a proxy; a long-lived, well-functioning generation **above a dysfunctional one is evidence of projection** |
| Courtship length | **both** tails informative, middle uninformative |

Contexts **MUST** be weighted by emotional intensity, not treated as comparable: "**In terms of solid self, one's personal life is where the rubber meets the road**", and occupational success "**is not a reliable measure of their basic level**". → `kerr_book/ks05.md` · KS05.10; `ks11.md` · KS11.7; `ks06.md` · KS06.8

**M1.A.4f** The family-level estimate **MUST** be taken over **all** members, never from the best-functioning one, because "**the patterns of emotional functioning in a family system can result in one person functioning well at the expense of another person doing poorly.**" Sibling divergence **MUST NOT** be read as basic-level divergence — the higher-functioning sibling may be running on pseudo-self. → `ks05.md` · KS05.10; `ks11.md` · KS11.4

**M1.A.5** A `Person` **MUST** hold a `functional_level` distinct from `basic_level`. → §3.2

**M1.A.5a** — *the operational decomposition, stated.* `functional_level` **is** the **operational level of differentiation**, and **MUST** decompose as `basic_level` (the slow floor) **+ a swing term** (fast). `M1.A.5`'s variance requirement is a statement about the **swing**: its variance **MUST** be a decreasing function of `basic_level`. → user, 2026-08-24; `kerr_book/ks06.md` · KS06.5
**The self-directed channel (M4.D.1a) writes the swing directly.** An agent can decide to stop or start doing something — a shift in functioning that is "more mature, more responsible for self" — and the swing moves **at once**, while `basic_level` does not move at all. That is why `M1.A.4a`'s estimator exists: sustained, broad, load-tested swing is what constitutes evidence of a basic-level rise.
**And the conversion is one event, not two:** a `basic_level` rise **is** a pseudo-self→solid-self conversion, so agency (M10.A.1a) rises as a consequence with nothing extra modelled.

**M1.A.5b** — *relationships carry the same two-level structure.* A `Relationship` **MUST** hold a **basic** individuality–togetherness balance and a **functional** one. Anxiety moves **only** the functional one — "**If anxiety goes up, fusion increases. This represents a functional shift in individuality-togetherness balance, not a basic change**"; and "**anxiety does not affect basic relationship balance.**" → `ks06.md` · KS06.5

**M1.A.5c** — *the sign of the pseudo-self change depends on **which channel** produced the functional rise.* This is a **hard requirement**, and getting it backwards inverts the model's central discriminator. → user question S2.12, resolved by `kerr_book/ks05.md` · KS05.1

| Rise produced by | `functional_level` | pseudo-self | `basic_level` |
|---|---|---|---|
| **Automatic channel** — borrowing from a partner, a group, a tribe | ↑ | **↑** (borrower) / ↓ (lender) | unchanged |
| **Self-directed channel** — differentiating effort | ↑ | **↓**, converting toward solid | rises if sustained and broad |

> "**Pseudo-self can be the basis of increasing a person's functional level of differentiation and reduce chronic anxiety.**" The aimless young man who finishes college after falling in love "**has borrowed pseudo-self from his romantic partner**"; people who join cults "**experience similar improvements in their functional level of differentiation.**"

**Consequence:** the two channels leave **opposite signatures in pseudo-self**, which is a computable discriminator the model would otherwise lack. It requires that the event record carry the channel (M1.F.1a).

**M1.A.5d** Borrowing **MUST NOT** be modelled as pathological. "**Everyone borrows and lends self to some degree**… **This is not a bad thing**" — it is the ordinary mechanism by which a stable culture supports functioning (`ks14.md` · KS14.2). What distinguishes it is only that it **does not survive withdrawal of the source**.

**M1.A.6** Symptom thresholds **MUST** be evaluated against `functional_level`, never against `basic_level`. → §3.2

**M1.A.7** A `Person` **MUST** hold `programmed_reactivity`, fixed in childhood from that person's own **witnessed** event history. It **MUST NOT** be computed from a family average. → §3.3, §9.4

**M1.A.7a** — *manifest chronic anxiety is **not** a per-person constant.* `chronic_anxiety` **MUST** be **derived** each slow tick as a function of three terms: the person's `programmed_reactivity` (M1.A.7), the **field** — the system's differentiation and current load — and the person's **functioning position** in that system, which changes. It **MUST** act as the floor below which acute anxiety cannot decay. → `kerr_book/ks10.md` · KS10.2
**Why the split:** "chronic anxiety is a consequence of various types of social interaction and, consequently, is **most usefully conceptualized as a property of the emotional field**", determined by two processes "**not under individual control**" — the system's differentiation, and "**the person's functioning position in the system**". An implementation with only the fixed childhood term cannot produce `M11.C.11`'s opposite-signed removal effects, nor `KS25.2`'s reciprocity inversion, because both act through **position**.
**Consequence:** `M1.A.7`'s value is a *disposition*; `M1.A.7a`'s is a *state*. `M4.C`, `M7.D` and the `M1.A.4a` estimator all read the derived one.

**M1.A.8** A `Person` **MUST** hold `acute_anxiety`, updated per event and decaying toward the chronic floor.

**M1.A.9** A `Person` **MUST** hold `outside_ness`, distinct from `basic_level`, with three inputs at three time scales (M5.F). → §3.6

**M1.A.9a** — *`outside_ness` **MUST** be two-dimensional, not a scalar.* The two axes are **outward impingement** (acting on the other) and **inward impingement** (being acted on by the other), and the differentiated position is **both low**. The definition is a conjunction of two negatives: *be for self without being selfish; be for other without being selfless.* The corpus states the same structure in the model's terms — "the **low-level self can have an I-position**. That is a **selfish, dogmatic, forceful** kind of an I-position… a high-level differentiated self is **neither offensive nor defensive** to the other. So you can tell pretty much the **level of functioning of an I-position from the way they do it**." → `kb/kb12.md` · K12.3; the same two-sidedness on the external agent is `kb/kb07.md` · K07.1.
**Why a scalar fails:** the two counterfeits need **opposite** corrections. The forceful declarer fails outward; the compliant accommodator fails inward. A single score lands them at the same value and would correct them the same way. → M5.F.1, M5.F.2a

**M1.A.10** A `Person` **MUST** hold a `life_energy` allocation, zero-sum between relationship-seeking and goal-directed activity, whose ratio is a function of `basic_level`. → §3.7

**M1.A.11** A `Person` **MUST** hold `symptom_load` over exactly three channels — **physical, mental, social** — which **MUST** be substitutable (M7.D.1–M7.D.4). → §3.9
**On the name:** the third channel was *emotional*; it is renamed **mental** on the author's own proposal — "a better term… would be 'mind' or 'mental' — the symptoms manifest in aberrant cerebral cortical processes, but **the core of the symptom-generating force is the subcortical emotional system**." Under M1.A.0 *everything* in this model is emotional, so *emotional* as a channel name is a collision. → `kerr_book/ks23.md` · KS23.11

**M1.A.11a** The three channels **MUST** carry `KS10.5`'s second axis: **physical and mental internalise; social externalises.** This is what makes `M11.C.12`'s symptom-relief-raises-conflict result and `KS04.13`'s inverse — depression lifting as conflict rises — the same mechanism rather than two.

**M1.A.11b** Which channel a person expresses **MUST** be exogenous constitutional data, and the model **MUST NOT** derive symptom *type* from relational position. "genes would be seen as perhaps having a role in **whether the chronic anxiety plays out as schizophrenia rather than some other clinical dysfunction**" (`kerr_book/ks22.md` · KS22.14); "**Certain predispositions in a person can become assets or liabilities, depending on the degree of family anxiety**" (`ks21.md` · KS21.5). Level sets amplitude and **sign**; constitution sets the channel **prior** (M1.A.11c).

**M1.A.11c** — *the channel prior **MUST** be movable by a relational term, and the relational term **MUST** be overridable.* Kerr 1988 states the reverse emphasis of M1.A.11b: "**Genes are an important influence on the type of symptom that develops, but learning based on childhood experience appears to be the most important influence on the category of clinical dysfunction (physical, emotional, social) that develops.**" He names the relational determinant — the category tracks "**what others in the system focus on in that individual when they get anxious**" — and he supplies the override himself: "**Genetic predisposition to a disease… can be strong enough to override relationship programming.**"
Therefore three terms, in this order: (1) M1.A.11b's constitutional assignment is the **prior**; (2) a **family-focus term MUST** be able to shift the expressed channel away from that prior, driven by what the family attends to in that member under anxiety (M9); (3) a **constitutional-strength term MUST** be able to override the shift. A model in which the channel is purely constitutional, or purely relational, is a failing implementation.
**The reconciliation, and its limit.** `FE09.1` decomposes it — **constellation → ripeness; learning → category; constitution → specific symptom**. The *specific symptom within a channel* therefore stays constitutional and is **not** moved by the relational term; only the **category** moves. → *decision A1, option (c), 2026-08-27*; `theory/family_evaluation/fe08.md` · FE08.3, `fe09.md` · FE09.1, `fe07.md` · FE07

**M1.A.12** A `Person` **MUST** hold `involvement_weight`, recomputed every fast tick. Family membership **MUST** be derived as a threshold over it and **MUST NOT** be a stored set. → §3.8

**M1.A.13** A `Person` **MUST** hold `structural_importance` on exactly **three** tiers — shock-wave-likely, neutral, relief. A finer ranking **MUST NOT** be implemented.

**M1.A.13a** The tier **MUST** be **derived, not assigned**, from two inputs the model already carries: whether the person holds a **functional-doer position** (the head of the clan, "nominated by the family and the responsibility accepted by the person"; or a child carrying emotional endowment — an only or oldest child, or a gifted one), and **how suddenly** they go from full functioning to loss. 1979 Tape 6: "the most important cue… has to do with the **functional position in the family** of the one who dies. Potential reactiveness is **greatest when that person goes from full functioning to death in a brief period**."
**Role labels MUST NOT determine the tier.** Tape 6 is explicit that a matriarch outranks a patriarch where she held the position. A **disabling injury or life-threatening illness** in an endowed child **MUST** rank with a death. The relief tier is someone who has not contributed and has become a burden. → `theory/_LECTURES_1979.md` · T6.1

**M1.A.14** A `Person` **MUST** hold `sibling_position` as static **birth-order data**. It **MUST NOT** be read directly by the propensity vector. → §3.11

**M1.A.14a** — *a derived `functional_sibling_position` **MUST** be computed from observed functioning, and the propensity vector **MUST** read the derived one.* "the family projection process can so impair the functioning of a firstborn son that **his younger brother may function more like an older brother than the firstborn does. In such instances, the younger son becomes a 'functional oldest.'**" → `kerr_book/ks12.md` · KS12.2

**M1.A.14b** — *the general rule, of which M1.A.13a and M1.A.14a are the two instances.* **No positional attribute the model acts on may be read from a role label or from birth order.** Every one **MUST** be derived from observed function. Labels are static data; positions are computed.

**M1.A.14c** Sibling position **MUST NOT** be an additive offset on the propensity vector. Each position carries **both** an adaptive and a maladaptive expression, and `basic_level` selects which — "An **immature** older brother of brothers is likely to be overly controlling and dogmatic… A **mature** older brother of brothers can be a very effective and responsible leader." So the relationship is **gating**, not additive, which is why the effect peaks mid-scale (M11.C.9). → `ks12.md` · KS12.5, KS12.7

**M1.A.14d** Sibling-position effects **MUST** be suppressed by anxiety in the **relational** domain while remaining available on an unambiguous **task** demand. Six patients on a locked ward who responded to a fire "**were the oldest children in their families**" — suppression, not erasure. `[D]`, n≈6, no control. → `ks12.md` · KS12.4

**M1.A.15** A `Person` **MUST** hold `financially_dependent: bool`, a hard gate on one move (M5.C). **The mechanism is now explicit:** the change-back reaction's third rung — "if you do not, these are the consequences" — is **materially enforceable** against a dependent person. 1979 Tape 4: "you couldn't work with a child who's still **financially and emotionally dependent**… it has to be somebody who can **risk the chance**, because the family system is going to say, '**if you do that, we are going to disinherit you**.'" The gate is therefore not arbitrary; it is the condition under which the threat has teeth. There **MUST NOT** be a per-person material stock, resource pool, or any quantity whose depletion causes death. → §3.12

**M1.A.16** A `Person` **MUST** hold `beliefs` (M9), which **MUST** be permitted to differ from ground truth.

**M1.A.17** A `Person` **MUST** hold `role ∈ {MEMBER, EXTERNAL}`. `EXTERNAL` is not a separate type. → §2.5

**M1.A.18** A `Person` **MUST** hold `systems_perspective`, **graded**, distinct from `basic_level` and from `functional_level`. `[M]` — mechanism sourced, magnitudes invented.
**What it is.** Bowen names **three** variables for societal process: "**one, a different way of thinking** — the world is different depending on the way the head is that observes it… another is **differentiation of self**… and a third is the **intensity of anxiety**." `systems_perspective` is the first. The three dials of M1.D.7 are drivers of the third only (M1.D.7a1); this is the individual-level representation of the first. → `kb/kb10.md` · K10.1
**Why it is needed.** Without the systems frame, an agent has no model of how differentiating would help, so it executes the differentiating move in the only frame it has — a stand taken *against* people. That is the misconception Bowen names as "**grotesque** … to somehow separate a self from the family, **to shout them down and let them know you are different**." Pseudo-differentiation is therefore **not a separate error to be modelled**; it is what the move degrades into at low `systems_perspective` (M5.F.4). → `theory/_KERR_INTERVIEWS.md`
**Graded, not a hard gate.** No source says a person with no systems perspective is incapable of a genuine differentiating move, only that they will not see why it would help and will reach for blame. A hard gate would make the first differentiating move in any family impossible.

**M1.A.18a** The **observable readout** for `systems_perspective` **MUST** be **two-sided**: emitting blame **or** praise is a loss, per the neutrality gauge — systems thinking is not possible until the person can be "**emotionally neutral. That is without blaming or praising.**" A negative-valence-only readout misses half the cases. This is M1.E.2's third detector pointed at a `MEMBER`, not only at an `EXTERNAL`. → `kb/kb07.md` · K07.1

**M1.A.18c** — *`systems_perspective` is **orthogonal to `functional_level`** and **MUST NOT** gate it.* An agent with `systems_perspective` at or near zero **MUST** be able to function well — in work, in health, in stable relationships, across a whole life. **Most agents in any population have essentially none.** → user decision, 2026-08-25
**What it gates, and only this:** the **differentiating path** — `M5.F.4`'s move quality, `M4.D.3b`'s engagement with a loaded tie, and therefore the `M1.A.4a` estimator's capacity to rise. It does **not** gate ordinary functioning, symptom resistance, competence, or adaptation to unremarkable stressors.
**Why this must be stated rather than assumed:** the natural implementation makes an agent's perspective raise everything, which would predict that the ~90% of a population below `basic_level` 50 (`kerr_book/ks05.md` · KS05.4) are broadly dysfunctional. They are not. Corpus support is direct — Bowen theory "**describes what is, not what should be**", differences are "**quantitative, not qualitative**" (`ks05.md` · KS05.17), and `KS05.11`'s three-band life pattern has the *intermediate* range doing well in work while running into trouble in personal life. Systems thinking is **rare**; functioning is **common**; the model must reproduce both.
**Population consequence:** at construction, `systems_perspective` **MUST** be initialised at or near zero for the large majority of agents, and `M11.C.17`'s coach-free arm **MUST** show a population that is *functional and undifferentiating*, not a population that is failing.

**M1.A.18d** — *resolution of the M7.A.2a question: **per-person state, per-tie attenuation**.* `systems_perspective` **MUST** be a **single per-person state**, not a per-tie state variable. Its **effective value on a given tie** **MUST** be attenuated by that tie's emotional intensity, so that an agent can hold the perspective across most of its relationships and lose it in the most loaded one. → user decision, 2026-08-25; `kerr_book/ks16.md` · KS16.6, `ks17.md` · KS17.1
**This is what the evidence actually shows.** Kerr had been "**teaching this cocreation idea… for over twenty-five years**" and had not observed it in his own marriage. Mr. S. applied it to his sister's triangle, his sons' triangle and his family history while stating "**I acknowledge Bowen theory, but I don't feel it**" of his marriage. Neither is a person holding *different* perspectives; each is one capacity **failing at the highest load** — which is `M4.C.4`'s anxiety-driven decay evaluated per tie rather than globally.
**Consequences:** `M1.E.7`'s rise applies to the **person**; `M4.D.3b`'s gate and `M5.F.4`'s degradation are evaluated **per tie** against the attenuated value; and `M7.A.2`'s "application re-earned per tie" follows without a second state variable — the re-earning is the tie's load falling far enough for the existing capacity to reach it.

**M1.A.18b** `systems_perspective` **MUST NOT** be a free parameter. Its **ceiling MUST** be coupled to `basic_level` — undifferentiation is "the **cement, the hardener**, that fixes" a way of thinking, and reducing it lets the thinking change faster. → `kb/kb07.md` · K07.3; M10.A.1

### M1.B `Relationship`

**M1.B.1** A `Relationship` **MUST** connect exactly two `Person`s and **MUST** be the unit on which coupling state is stored. Per-family scalars **MUST NOT** stand in for a specific tie.

**M1.B.2** `conductance` **MUST** be undirected. It **MUST NOT** be a function of physical distance, contact frequency, or elapsed time since last interaction. → §4.1, §4.3

**M1.B.3** `bond_energy` **MUST** distinguish four tie states that differ in events and energy independently: cut off (no events, high), distant (few, moderate), resolved low-contact (few, **low**), open conflict (many, high). Failing this discrimination is a failing implementation. → §4.2

**M1.B.4** `bond_energy` decay **MUST** be at or near zero. Reunion after separation **MUST** restore full coupling with **zero re-activation latency**. → §4.3

**M1.B.5** `functioning_balance` **MUST** be directed, bistable with no stable midpoint, and indexed **per area of joint activity**. It **MUST NOT** be a single linear scalar and **MUST NOT** be a fixed personal trait. → §4.4

**M1.B.6** The pole **MUST** flip on a *relative* comparison — the under-functioning party's self-assertion exceeding the other's domination — not on an absolute threshold; the flip **MUST** be immediate; and it **MUST** revert unless sustained through the counter-reaction. → §4.4

**M1.B.7** Reversal cost **MUST** be asymmetric: reducing a marked over-functioner **MUST** cost less than raising a marked under-functioner. → §4.4

**M1.B.8** `investment` **MUST** be directed and **valence-blind**. 1979 Tape 1 states the systems rationale: "a **negative feeling is just as successful in maintaining a family system as a positive feeling**, so I make no difference. That is psychic investment in the other… the only difference is that people **react** differently." Valence changes the reaction, not the investment. — share of thought occupied by the target. It **MUST NOT** be derived from warmth, agreement, or tie quality. Conflict-laden preoccupation **MUST** register as *high* investment. → §4.5

**M1.B.9** A `Relationship` **MUST** hold `areas_of_joint_activity`; narrowing them **MUST** operate on the same variable as the functioning balance, not on a separate distance scalar. → §4.4

**M1.B.10** A `Relationship` **MUST** hold `taboo_set`, which grows by default as each party learns what makes the other anxious — "so begins the communication cutoff between spouses." → §4.6

**M1.B.10a** The set **MUST NOT** be implemented as monotone. A subject **MUST** be returnable by **purposeful mention under self-control**: "the purposeful mention of the taboo subject, **if one can control one's own anxious response** to the other, **can desensitize the whole mechanism**." This is the differentiating move applied to a single topic, and it **MUST** carry the same gates (`M5.C`). → `theory/_LECTURES_1979.md` · T6.4

**M1.B.11** A `Relationship` **MUST** hold `latency`, a per-edge delivery delay. Events **MUST NOT** all arrive within one tick. → §4.7

**M1.B.12** A `Relationship` **MUST** hold `dyad_age`, raising the cost of a pole flip with time in configuration. → §4.9

### M1.C `Triangle`

**M1.C.1** A `Triangle` **MUST** hold three members, the current inside pair, the outside member, and `bound_anxiety`.

**M1.C.2** Position value **MUST** invert with load: the outside position **MUST** be unfavoured when calm and favoured under tension. A fixed preference is a failing implementation. → §6.1

**M1.C.3** Persistent triangle topology **MUST** be stored separately from the currently-active set. Triangles **MUST** be inoperative when the system is calm. → §6.1

**M1.C.3a** — *the routing capacity of a triangle MUST be a function of its members' `functional_level`, not of the triangle alone.* A triangle among well-differentiated members **MUST** route little anxiety, activate only under real load, and **resolve when the load passes**; the same topology among poorly-differentiated members **MUST** route more and stay fixed. Verbatim: *the intensity of these patterns "is determined by level of differentiation and principally anxiety"* — "**well differentiated, these patterns will be mild, they'll be there in periods of anxiety and they'll go away**, and as we go down into lower levels of differentiation, **these patterns are more intense**."

**M1.C.3b** It follows that **the coach MUST NOT act on triangles as a mechanism of change.** Triangle activity is a *consequence* of anxiety and differentiation, so the two levers are the two named in `KB04` — get the anxiety down, then work toward differentiation — and triangle activity falls out. A move that manipulates the family's triangles directly **MUST NOT** exist in the repertoire. *This is why Bowen says late in life that he does "not do much with triangles anymore" and that it makes "no difference how many triangles are out there": a statement about **coaching technique**, not about whether triangles operate.* `Triangle` remains a first-class object (M1.C) because it is the structure anxiety routes **through**; what changes with differentiation is how much it carries.

**M1.C.3c** Two distinct operations share the name *detriangle* in the sources and **MUST NOT** be conflated in the implementation:
- **detriangle self** — the actor withdraws their own emotional participation while staying in contact (`KB06`: "to calmly de-triangle self from the primary triangle with the parents"). This is the outside position and it is what `M8` counts.
- **detriangle another** — returning an aligned third party to neutral (Ch21, `M5.B.1`).
The first is the mechanism of change; the second is a countermeasure during a move.

**M1.C.4** A `Triangle` **MUST** hold `activation_memory`; tension **MUST** preferentially reroute onto previously-used circuits. → §6.1

**M1.C.5** A `Triangle` **MUST** hold `intensity_floor`, a permanent decrement applied when an I-position is held. The triangle's intensity **MUST NOT** fully revert. → §6.1

**M1.C.6** A sibling-conflict event **MUST** instantiate the parent-level triangle. An intervention on the sibling pair alone **MUST** fail. → §6.1

### M1.D `Family`

**M1.D.1** A `Family` **MUST** hold one `undifferentiation_budget` and exactly **three** sink allocations: marital conflict, spouse dysfunction, child projection. → M6.I.1

**M1.D.2** Emotional distance **MUST** run outside the **symptom** budget as an always-on baseline and **MUST NOT** be implemented as a fourth sink. → M6.I.2

**M1.D.2a** — *distance **binds**; it neither destroys anxiety nor is costless.* `DISTANCE` **MUST** move anxiety **out of `Person` and into `Relationship` state**, where it (a) **persists**, (b) is **readable as the distance itself**, and (c) **returns to the person if distancing is prevented**. An implementation in which `DISTANCE` reduces total system anxiety is a **failing implementation**. → `kerr_book/ks04.md` · KS04.1; `kb/kb04.md` · K04.1
**The mechanism, from the source:** "Both people experience **less internal anxiety**… but **the anxiety is now evident or bound in distancing behaviors**… **If people are unable to distance from each other for whatever reason, internalized anxiety reappears**"; "the anxiety is **integrated into the structure of a relationship**."
**This settles the three-versus-four-sinks question permanently.** Bowen's fourth mechanism "**inevitably accompan[ies] each of the other patterns**" — distance is **not** a fourth sink beside three, it is the **substrate under all of them**. `M6.I.1`/`M6.I.2` keep distance outside the symptom budget; this states where the anxiety goes instead.
**Consequence for readouts:** a family discharging into distance reads as untroubled on every person-level measure while carrying its load on the ties. `M11.C.13`'s relocation test must score **location** as well as count, and the tie-borne quantity is an observable in its own right.

**M1.D.3** A `Family` **MUST** hold an `overflow` term whose destination is conflict with families of origin — **not** distance. → §6.2

**M1.D.4** A `Family` **MUST** hold `leadership_office` with an occupant and a **sphere of responsibility**. Vacancy **MUST** stall the system, and the stall **MUST** be releasable by external recognition rather than by the occupant. Sphere **MUST** be implemented as a move permission scope, **MUST NOT** be a propagation boundary, and **MUST NOT** be a global flag. → §6.2

**M1.D.4a** A `Family` **MUST** expose `differentiation_capacity` — whether **any** member retains the ability to take a position against the objections of the others. Bowen reserved *schizophrenia* for families where no member has it, calling the rest functional psychoses, and ran "a **test of treatability, a test of differentiation**… and if one person had that ability, I'd stick with them" expecting a good result. → `kb/kb08.md` · K08.2
**Consequence:** severity is a **family-level capacity**, not a property of the symptom-bearer, and it predicts treatability rather than describing symptoms.

**M1.D.5** A `Family` **MUST** hold a per-agent `tolerance` for disturbing behaviour (M7.D.4).

**M1.D.6** A `Family` **MUST** hold `access_vector`, contributing its effect whether or not any move uses it. It **MUST NOT** be a gate on moves and **MUST NOT** be a depletable stock. → §6.3

**M1.D.7** A `Family` **MUST** hold `ambient_anxiety` driven by the societal dials, acting on the **symptom threshold**. Turning a dial **MUST NOT** move every family proportionally. → §6.4

**M1.D.7a** The dials **MUST** be named `societal_leadership`, `media_amplification` and `resource_pressure`. The single letters `L`, `X` and `E` **MUST NOT** be used. *Their meanings were previously recorded nowhere except Pygame log strings in `src/main.py`, where the third was labelled "Climate"; `resource_pressure` is the accurate name for what Ch18 actually argues.*

**M1.D.7a1** The three dials are **drivers of anxiety**, which is one of three variables Bowen names for societal process — "the way the person thinks, another is differentiation of self, and a third is the **intensity of anxiety**." The dials therefore sit **one level below** his decomposition and **MUST NOT** be presented as the societal model. His first variable, *mode of thinking*, has no representation here and is not currently implementable; that gap **MUST** be stated wherever societal results are reported. → `kb/kb10.md` · K10.1

**M1.D.7b** `resource_pressure` **MUST** be implemented, and it is **`[T]`, not `[I]`** — it is Ch18's own proposed cause of societal regression, not an inheritance from the grid engine. Bowen: "a spectrum of problems associated with **population explosion play a major role in man's deeper anxieties**", with "the rapid depletion of world's natural resources" and "certain natural resources are nearing exhaustion"; and the chapter's thesis, that society "appears to be functioning on a less differentiated emotional level than twenty-five years ago, that this **may be related to the disappearance of land frontiers**."

**M1.D.7c** `resource_pressure` **MUST** act through **awareness of diminishing availability**, not through consumption or a per-capita stock. "It was important for him to know there was **new land for him, even if he never went to it.**" It is therefore the societal-scale instance of the availability mechanism in M1.D.6, and **MUST NOT** be implemented as the grid engine's metabolic drain — that column is deleted (M1.A.15).

**M1.D.7d** `media_amplification` **MUST** be a societal input whose effect on a given family is **modulated by that family's differentiation**, not applied uniformly. This is M1.D.7's decoupling guard at the level of a single dial: a well-differentiated family **MUST** be able to damp it, and a poorly-differentiated one to amplify it.

**M1.D.7i** — *the damping is **two-sided**.* Differentiation **MUST** modulate the **magnitude** of societal influence, `|effect|`, in **both** directions: a well-differentiated family is less moved by **favourable** societal conditions as well as by unfavourable ones. An implementation that damps only harm is wrong. → user, C4 verdict 2026-08-24
> "**societal forces will have less of an impact (positive or negative) on a well differentiated family — they will be less reactive to outside forces.**"
**This is the fourth instance of one pattern**, and it is now established enough to state generally (M11.F.6): blame **and** praise are both losses; selfish **and** selfless are both counterfeits; too much distance **and** too much closeness both trigger the stress response; favourable **and** unfavourable societal input are both damped. **Every naive one-sided detector in this model is wrong.**

**M1.D.7j** Decoupling protects the family **without** conferring societal influence. A well-differentiated minority "**can float above the regression that surrounds them**" but "**their contributions are commonly ignored or overruled by the regressed majority**" — so a high-differentiation family **MUST NOT** damp the societal dial *for anyone else*. → `kerr_book/ks14.md` · KS14.8

**M1.D.7k** `resource_pressure` (M1.D.7c) **MUST NOT** be monotone. **Both scarcity and superabundance degrade social structure**, so the dial requires an **optimum**, not a direction. The Galápagos ground finches under a 1983 El Niño: a copulating frenzy, monogamous mothers becoming polygamous, "**Females commonly abandoned their begging young offspring**", then mass die-off when the rain stopped — "**the orderly social structure of the frenzied birds regressed. Stable families disappeared.**" → `ks14.md` · KS14.7 (Grant & Grant 1985)

**M1.D.7l** The societal individuality–togetherness balance **MUST** have its optimum at **parity**, not at maximum individuality, and a regression **MUST** be bounded at a **5–10 point** shift toward togetherness. "**Bowen theory holds that optimal functioning for a society is a 50-50 balance**… a new balance can occur, such as **55 or 60 on the togetherness side**." → `ks14.md` · KS14.3 `[#]`

**M1.D.7e** A dial-driven stressor **MUST** be able to convert from exogenous to endogenous within an individual — arriving from outside, then sustaining itself through the chronicity integrator (M4.C.3) once internalised. The `exogenous` flag (M1.F.7) **MUST** record which it was at emission, so the two remain countable separately at readout.

**M1.D.7f** `societal_leadership` remains `[I]` in its functional form. Ch18 names six downward channels — labelling and diagnosis propensity, overleniency of officials and laws, helping-programme intensity, school structure at junior high, population density, and era-dependent symptom form — and leadership quality is not among them, though Ch13's only stated reversal mechanism does require a single principled leader.

**M1.D.8** The family **MUST NOT** be the boundary of the simulated system. At least one family-of-origin tie per adult **MUST** exist, because degree of cutoff with the families of origin is an *input* to intensity. → §2.4, §6.2

### M1.E External agents

**M1.E.1** An external agent **MUST** be a `Person` with `role = EXTERNAL`, a restricted repertoire (M5.B.4), and real ties.

**M1.E.2** An external agent **MUST** hold its own drifting reactive state, observable as a readout, with **three detectors** and the third **two-sided**: the urge to become critical (`L20.2`); the agent's attention inside the other's problem, with a factual relationship question as the recovery move and the return of humour as the signal (`kb/kb04.md` · K04.11); and catching itself **diagnosing, criticising or praising** — **praise is as much a loss as blame**, and a negative-valence-only detector misses half the cases (`kb/kb07.md` · K07.1). A stateless coach is a failing implementation.

**M1.E.2a** An external agent's **objective MUST be to understand, not to help.** Four statements, an experiment and an independent replication: families seen in research with no therapeutic goal did better than families seen in therapy; residents instructed to learn rather than cure lasted at most **ten hours** before giving in to the demand for an answer; and Kerr, interviewing cancer families with no intent to fix anything, found they produced more ideas per session and "were **doing better, no question about it**." → `kb/kb03.md` · K03.1, observable as a readout. A stateless coach is a failing implementation. → §10.2

**M1.E.3** An external agent **MUST** be able to **absorb responsibility** — a conserved transferable quantity the family cannot hold while the agent holds it. → §5.8

**M1.E.4** An external agent **MUST** be able to **certify a defect**, writing to a family-level belief that verbal denial cannot reverse. → §5.8, M9

**M1.E.5** Proximity of an external agent **MUST** carry a burden-transfer term with a negative sign. → §5.8

**M1.E.6** An external agent's presence **MUST** change the configuration. A costless observer is a failing implementation. → §2.5

**M1.E.7** `systems_perspective` (M1.A.18) **MUST** rise **only** on a *landed* external-agent contact, and **MUST** be conditioned on the person's binders failing. It **MUST NOT** rise spontaneously and **MUST NOT** be a reinforcement target (M4.D.6d).
**What a landed contact delivers** — not instruction, but two things together: the awareness that "there was an **alternative way to respond**, and also that **I was emotionally caught in it**". → `kb/kb07.md` · K07.4
**The rate MUST be low.** Measured from the recipient's side over two years at six to eight contacts a year: "there were probably **four times at most, five maybe, in two years**, that something helpful came out of the contact." → `kb/kb07.md` · K07.4 `[#]`
**The binder-failure condition** is capacity exhaustion, stated at societal scale: "**societal attitudes change when society no longer has an option.**" Pain is necessary and **not sufficient** — without frame supply, exhaustion **MUST** produce symptom escalation, not insight. → `kb/kb10.md` · K10.6
**One at a time.** → `kb/kb12.md` · K12.1

**M1.E.7c** — *a landed contact has **four** forms, and they are not interchangeable.*
1. **Instance supply** — specific functional facts from the recipient's own history. Expensive; requires the log. "**saying someone has an equal part in a process is not especially helpful until that person can comprehend the specifics of it all.**" → `kerr_book/ks10.md` · KS10.11
2. **Category supply** — naming a process the agent is engaged in and has no name for. Cheap, one-shot, works from an impersonal source: Kerr recognising his own cutoff during a lecture. → `kerr_book/ks13.md` · KS13.4
3. **Non-participation** — the agent's *failure to be recruited*. Transmits no information at all. Bowen's slight smile after a half-hour anxious monologue: "**he simply did not get caught in my anxiety, and that was enough to get me out of it.**" Implementable as M4.D.5's fused default **not** being played — and the reason a stateless coach transmits nothing. → `kerr_book/ks16.md` · KS16.4
4. **Delayed self-observation** — presenting the agent with **its own event log, delayed**. Bowen required session tapes to be "**at least six months old**", explicitly against instant replay, because self-observation accuracy rises as the episode's acute anxiety decays toward the floor. Kerr on his own tape a year later: "**Whom does that guy think he is kidding?**" → `kerr_book/ks18.md` · KS18.4

**M1.E.7d** Landing **MUST** require **coincidence with the recipient's current state**, not merely correct content — which is why the same content can fail and later succeed, and why the rate is low. Three conditions jointly: coach quality, recipient motivation, and **occasion**, the last of which the coach does not control. → `kerr_book/ks17.md` · KS17.10

**M1.E.7e** The coach tie **MUST** remain thin and **MUST NOT** accumulate `investment` (M1.B.8). A coach perceived as taking sides becomes a **lender of pseudo-self**, raising the recipient's functional level while doing nothing for basic level — precisely the artefact `M11.C.17`'s no-drift arm must not be able to exploit. → `kerr_book/ks18.md` · KS18.5

**M1.E.7f** `M1.E`'s entry threshold **MUST** be **family-relative**, not absolute (M4.D.5b), and **MUST** carry a **rejection hazard** proportional to how far the account displaces the family's own attribution. Nancy Lanza sought help repeatedly across years and multiple professionals, and no contact landed, because what she wanted was a diagnosis located in the child. "**some parents do not want to hear it and seek another therapist no matter how nonjudgmental the therapist is.**" **The highest-need families are the hardest to retain** — a real dynamic for `M11.C.17`'s coach arm. → `kerr_book/ks21.md` · KS21.10

**M1.E.8** — *external-agent contact frequency **MUST** be non-monotone, with a **low** optimum.* Four sessions a month were observed **worse** than one, and the named mechanism is transference: frequent contact thickens the coach tie, which is exactly what `M1.E.7e` forbids. Therefore the landing probability of M1.E.7 **MUST** fall above a low contact rate, and an implementation in which more coaching is monotonically better is a failing implementation. **This is distinct from `M5.B.3`**, which governs family ties and correctly carries no optimum — the coach tie is the one tie in the model where more contact is a hazard. → `theory/family_evaluation/fe11.md` · FE11.8

### M1.F `Event`

**M1.F.1** An `Event` **MUST** carry: sender, targets, witnesses, move type, intensity, timestamp, `duration`, `exogenous`, `source_position`, `route`, `fidelity`.

**M1.F.2** `source_position` **MUST** be able to change the *sign* of an event's effect, not only its magnitude. → §9.3

**M1.F.3** `route` **MUST** modulate gain: a direct dyad amplifies; routing through a neutral third damps. → §9.3

**M1.F.4** `fidelity` **MUST** degrade per private hop. → §9.3

**M1.F.4a** The **register** of a message **MUST** constrain the register of the reply, independently of content: "when I introduce the subject with **tangential words, the other will respond with tangential words**", against the deliberate use of plain ones — *death, die, bury* rather than *passed, gone, deceased*. → `theory/_LECTURES_1979.md` · T6.6

**M1.F.5** Witnesses **MUST** appraise events not addressed to them, and that accumulated witnessing **MUST** be the source of `chronic_anxiety` (M1.A.7). → §9.4

**M1.F.6** Statistical inputs **MUST** generate spells with a start and a duration. Per-tick probability draws **MUST NOT** be used for any stressor. → §9.1

**M1.F.7** Endogenous events **MUST NOT** be sampled from incidence data. The `exogenous` flag **MUST** keep the two countable separately. → §9.2

**M1.F.8** Simultaneous events **MUST** be batched so that order within a tick cannot decide the outcome.

**M1.F.9** — *a second event type: **a binder becomes unavailable**.* Every stressor in M1.F to date **adds load**. This one adds none: it **removes a mechanism that was binding anxiety**, and the anxiety it was holding **MUST** surface. The worked case is a retirement — physical proximity removed the emotional distance that had been doing the binding, and the anxiety "**spilled over**" into symptoms with no new stressor anywhere. The event **MUST** therefore name the binder it removes (a tie's distance, a triangle position, a symptom channel, an external agent's absorbed responsibility) and **MUST** return that binder's held anxiety to the family budget rather than discarding it (M6.I.1). Without this kind the model can only make a family worse by hitting it. → `theory/family_evaluation/fe09.md` · FE09.2, `fe08.md` · FE08.6

---

## M2 — The reference family

The instance everything is tested against. It is **invented and tunable** — validation is against the theory's regularities, not against one family's history — so every value here is `[I]` and the whole of M11 carries the falsification burden.

**M2.1** The reference family **MUST** be declared in markdown config, not in source. → M0.3

**M2.2** It **MUST** span three generations and **MUST** include at least one cut-off tie and at least one family-of-origin tie per adult (M1.D.8).

**M2.3** Phase B **MUST** run on a reduced instance — the nuclear four of M2.A.0 **plus** their family-of-origin ties, even if those agents only hold a bond energy. A closed nuclear four **MUST NOT** be used, because it cannot compute its own driving term.

### M2.A Membership `[I]`

| # | Name | Gen | Age | `basic_level` | `chronic_anxiety` | `sibling_position` | `financially_dependent` | `structural_importance` | Note |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Teodor | 1 | 81 | 34 | 42 | eldest of 2 | no | shadow | Ana's husband |
| 2 | Ana | 1 | 78 | 37 | 38 | eldest of 2 | no | **central** | Marta's mother |
| 3 | Bruno | 1 | 74 | 29 | 55 | youngest of 2 | no | peripheral | Ana's brother; **cut off since t0 − 17y** |
| 4 | Sofia | 1 | 76 | 41 | 35 | only | no | shadow | Ravi's mother; lives 200 miles away |
| 5 | Ravi | 2 | 52 | 38 | 44 | eldest of 3 | no | **head of household** | married to Marta |
| 6 | Marta | 2 | 50 | 40 | 46 | eldest of 2 | no | **head of household** | Ana's daughter |
| 7 | Iris | 2 | 47 | 43 | 33 | youngest of 2 | no | peripheral | Marta's sister; low contact, **resolved** |
| 8 | Leo | 3 | 22 | 36 | 48 | eldest of 3 | no | peripheral | launched |
| 9 | Nadia | 3 | 17 | 41 | 52 | middle of 3 | **yes** | peripheral | at home; becomes the projection target |
| 10 | Pia | 3 | 14 | 44 | 37 | youngest of 3 | **yes** | peripheral | minimally involved |
| 11 | Toma | 3 | 20 | 39 | 40 | only | **yes** | peripheral | Iris's son |
| 12 | Dr Halim | — | 55 | 55 | 30 | — | no | — | `role = EXTERNAL`; counsellor, **not a family member** |

**M2.A.0** The instance declares **eleven family members plus one external agent**. "Twelve-person reference family" is loose shorthand: Dr Halim is `role = EXTERNAL` and **MUST NOT** be counted in family membership, in the tie matrix of M2.B.1, or in `positions_live` unless the live-position test of M8 admits him. The nuclear family for M2.3 is **Ravi, Marta, Nadia and Pia** — Leo is launched and is not part of the Phase B spike.

**M2.A.0a** `chronic_anxiety` for generations 1 and 2 **MUST** be supplied as an initial condition, because M1.A.7 and M7.B.1 fix it in childhood from witnessed history and those childhoods are never simulated. The test is **life stage, not generation**: an initial value **MUST** be supplied for every agent already past the fixation age at `t0`, and **MUST** be an error for any agent who is not. Every agent in M2.A is past it, so all twelve carry a supplied value; the *derived* path applies only to agents born during the run.

**M2.A.0b** There is **no calendar epoch**. All dates **MUST** be expressed as a duration relative to `t0`, the first slow tick. M3 defines only relative clocks, so a literal year has nothing to resolve against.

**M2.A.0c** Spouses **MUST** be paired at comparable **`basic_level`**, never `functional_level`. 1979 Tape 3 specifies the field: "when I say spouses marry others with the same basic level of differentiation, **I mean this base level and not functional levels**" — functioning "goes up and down real easy." "People choose spouses [at] **almost identical levels of differentiation of self**", offered as an observed constant and "par for the course". → `kb/kb14.md` · K14.2

**M2.A.0e** — *the match is close, not exact.* Pairing **MUST** use a declared **tolerance**, not equality: **±1 point** on the 0–100 scale, declared in config. → user amendment to D3, 2026-08-24
**Why a tolerance is the right shape.** The corpus's own claim is "**almost identical**", not identical, and Kerr adds that pairing is **strict for spouses and looser for friendships** — basic level "**also influence[s] the development of close friendships but somewhat less precisely**" (`kerr_book/ks07.md` · KS07.7). A wider tolerance therefore applies to non-family ties.
**And it is what makes `M11.C.24` observable**: the initial gap must be small enough that the *developed* divergence dominates it.

**M2.A.0f** Pairing **MUST** carry a second, orthogonal dimension: **rank and sex complementarity** of sibling position (M1.A.14). Complementary pairs form more readily and carry lower baseline conflict, and incompatible ones **MUST** be able to fail to form — "**if the female in this case was less deferential and the male is bothered by it, the relationship might not take.**" This affects **conflict propensity, not differentiation**: "**This does not mean that one marriage is more mature than the other.**" → `kerr_book/ks12.md` · KS12.3; `ks07.md` · KS07.9

**M2.A.0d** Fusion **MUST** be life-stage dependent, not uniformly costly. The infant–caretaker symbiosis is stated as a **normal state, not a pathology**. → `kb/kb14.md` · K14.5

**M2.A.0g** — *pole assignment **MUST** be independent of sex.* "**males and females assume the dominant position with equal frequency.**" `M1.B.5`'s dominant pole **MUST NOT** correlate with sex across an ensemble. This is a hard constraint a natural implementation could easily violate — any asymmetry in the assignment rule surfaces as a sex effect at readout — and it is assertable: see `M11.C.25`. → *decision A4-tail, 2026-08-27*; `theory/family_evaluation/fe07.md` · FE07.3

**M2.A.1** `Iris` and `Bruno` **MUST** have comparable contact frequency and **MUST** differ in bond energy. This pair is the fixture for M11.C.4 and the direct test of M1.B.3.

**M2.A.2** `Nadia` **MUST** become the projection target as an **outcome** of accumulated witnessed events, not by assignment at initialisation. → §9.4

### M2.B Ties `[I]`

**M2.B.1** Eleven family members give 55 possible pairs (Dr Halim is external and is not in this matrix — M2.A.0); the instance **MUST** declare only the ties that carry traffic, and **MUST NOT** instantiate the full matrix.

**M2.B.2** The declared set **MUST** include: the Ravi–Marta marriage; both parental ties to each of Leo, Nadia and Pia; Marta–Ana; Ravi–Sofia; Ana–Bruno (cut off); Marta–Iris (resolved, low contact); Ana–Teodor; Iris–Toma; and Dr Halim's tie to whichever member is being coached.

---

## M3 — Clocks, ordering and determinism

**M3.A.1** The fast tick **MUST** be one week and **MUST** carry: the standing load, event delivery, appraisal, move selection, tie updates, triangle position shifts, and symptom accumulation.

**M3.B.1** The slow tick **MUST** be one year and **MUST** carry: differentiation drift, chronic anxiety drift, life stage, the nodal calendar, mortality, and the multigenerational update. It **MUST** fire every 52 fast ticks.

**M3.C.1** Delivery latency **MUST** be a property of the tie (M1.B.11), not of the tick. An event scheduled on the fast clock **MUST** arrive when its edge's latency says it arrives. → §8

**M3.D.1** — *the update order.* Each fast tick **MUST** execute in exactly this order. This replaces the frozen spec's §5.4.

1. **Standing load** — every person takes a load from every tie as a function of bond energy ÷ `functional_level`, *before any event is delivered*. → M6.I.8
2. **Deliver** — pop events whose latency has elapsed, in timestamp order, batching simultaneous ones (M1.F.8).
3. **Perceive** — each person reads its inbox, its ties and its own state.
4. **Appraise** — M4.C.
5. **Recompute `involvement_weight`** and derive membership (M1.A.12).
6. **Recompute active triangles** from current tension (M1.C.3).
7. **Select** — each person draws exactly one move (M4.D).
8. **Act** — moves become events, queued with their edge latency (M4.E).
9. **Consolidate** — anxiety decays toward the chronic floor; repeated moves harden ties; triangles re-tally; invariants assert (M6).

**M3.D.2** Step 1 **MUST** precede step 2. A pure event loop reads fewer events as less arriving anxiety, which makes severing a tie register as relief — the opposite of the claim being modelled.

**M3.D.3** Steps 5 and 6 **MUST** precede selection, because both the move gates and the propensity vector read them.

**M3.D.4** — *determinism.* The engine **MUST** be a deterministic function of `(seed, config, scenario)`. Every stochastic draw **MUST** come from a single seeded generator threaded explicitly; module-level or global RNG state **MUST NOT** be used.

**M3.D.5** Two runs with the same seed **MUST** produce byte-identical event logs. This is a test, not an aspiration (M11.D.5).

**M3.D.6** No LLM **MUST** appear in the decision path. → proposal §9, Decisions settled

---

## M4 — The fast tick, step by step

### M4.A Standing load

**M4.A.1** Every person **MUST** take, each tick, a load from **every** tie, whether or not any event occurred on it, as a function of that tie's bond energy divided by the receiver's `functional_level`. → §7.2

**M4.A.2** A `TRIGGER` event **MUST** be able to spike the standing term with no contact at all with the cut-off person.

**M4.A.3** A `RECONCILIATION` event **MUST** convert standing load back into interaction-driven load.

**M4.A.4** `INSTITUTIONALIZE` **MUST** be implemented as converting a member's ties to worry edges — non-interactive, bond energy retained — and **MUST NOT** require new machinery beyond M4.A.1.

**M4.A.5** — *a **self-generated** stress term **MUST** exist, derived from `basic_level`.* Two sources of chronic stress are named and both scale with level; the model has only the inter-person one (M4.A.1) and nothing intra-person, so a person alone in a calm system currently has no load at all. Each person **MUST** therefore take, each tick, a second standing load that is a function of `basic_level` only. It **MUST** be derived, never independently parameterised (M10.A.1), and it **MUST NOT** be large enough to swamp M4.A.1 — otherwise `M11.C.1` passes for the wrong reason, on a term that never touches a tie. → `theory/family_evaluation/fe08.md` · FE08.5

### M4.B Perceive

**M4.B.1** A person **MUST** read events addressed to it *and* events it witnessed (M1.F.5).

### M4.C Appraise

**M4.C.1** Each incoming event **MUST** raise acute anxiety by `intensity × conductance / functional_level`, modulated by `route` (M1.F.3) and `source_position` (M1.F.2), and attenuated by `fidelity` (M1.F.4).

**M4.C.2** Appraisal **MUST** apply a gain function on anxiety: above threshold, event content **MUST** be defended against rather than absorbed. A plain product is a failing implementation. → §3.4

**M4.C.3** Symptom onset **MUST** be driven by an **integrator over chronicity**, not by a test on instantaneous anxiety. → §3.3

**M4.C.3a** — *the integrand is **time above the floor**, not peak.* The integrator **MUST** accumulate `acute_anxiety − chronic_anxiety` where positive, so that **many resolved excursions and one unresolved excursion are distinguishable**. Citing McEwen's allostatic load: the cost is "**the wear and tear inflicted on the body due to repeated cycles of allostasis, such as occurs from repeated stresses or when the stress response system does not turn off when it should**" — **the damage is in the failure to return to baseline.** → `kerr_book/ks23.md` · KS23.3
Duration is independently supported: enduring problems "**one month or longer**" were the most frequent threats, and "**the longer the stressful life event or events lasted, the greater the risk**" (`ks23.md` · KS23.5).

**M4.C.8** — *attending to a channel **amplifies** it.* Directing attention at the feeling channel **MUST** increase its activity, not discharge it; directing it at the intellectual channel **MUST** order it. This is where feelings legitimately enter the model — **not as state** (forbidden by M1.A.0) but as a **target of attention with an amplifying effect**. → `kb/kb12.md` · K12.6

**M4.C.8a** — *the amplification **MUST** be gated by objectivity.* As M4.C.8 stands, **every** act of reflection is an escalation, which would make `M1.E.7c`'s category supply and its delayed self-observation impossible — the person could never look at their own process without inflaming it. The gate is the **stance** of the attention, not its target: attention paid **from inside** the reaction amplifies; attention paid with the observing distance M1.A.9 measures does not. `outside_ness` **MUST** therefore scale M4.C.8's amplification toward zero, and a scalar implementation that amplifies unconditionally is a failing implementation. → `theory/family_evaluation/fe03.md` · FE03.6
> focusing on feelings is "almost as if you are **uncovering a volcano**, and the more you uncover it, the more of this coming and coming… the only time you get a kind of resolution is when the **feelings run down**, when the volcano gets tired." Work the intellect and "the feeling world is no longer a volcano. It is an **orderly little fountain**."
**This generalises `C1` beyond symptom relief**: attending to a channel is itself an intervention on it, so a coach or family member who focuses on the feeling channel **MUST** raise it. It is also the mechanism behind `M4.C.2`'s gain function seen from the other side.

**M4.C.4** `systems_perspective` (M1.A.18) **MUST** fall with acute anxiety. Polarity capture is a **loss** in someone who had the perspective, not merely a correlate of never having had it: "there is **no debate in systems**… any time anybody is stuck in a polarity, they have either not arrived at systems thinking, or **they have lost it** if they did arrive", and "the **higher the level of functioning, the more an individual can get beyond polarities**". → `kb/kb07.md` · K07.6, `kb/kb12.md` · K12.7

**M4.C.6** — *reappraisal **MUST** be self-attribution, not re-interpretation of the other.* An agent with `systems_perspective` **MUST** attribute part of a received event to **its own prior emission on that tie**. The shift is not "she is not hostile" but *part of what I am receiving is my own output returning*. This is why the effect is **immediate** — no new information arrives, only a re-attribution of existing information. → `kerr_book/ks17.md` · KS17.3
> "what he perceived as a tense expression on her face **reflected his own facial expression and tone of voice, which in turn were intensified by what he saw in her**… his thinking shifted radically from 'you don't understand me' to '**we just have differences in viewpoint**.'"

**M4.C.7** — *`route` (M1.F.3) **MUST** modulate appraisal on **both** sides.* A statement **addressed to a third party and witnessed** by the other **MUST** produce lower reactivity in the **listener** *and* in the **speaker** than the same content addressed directly. `M1.F.5` already distinguishes addressed from witnessed; this makes the witnessed form systematically less reactive and adds the **emitting-side** effect, which the spec did not carry. → `kerr_book/ks18.md` · KS18.2
> "**The fact that the husband does not direct his comments to the wife helps her to listen better and react less. The husband has less reactivity as well** in discussing an emotionally charged subject with the therapist, a neutral party."
This is the corpus's cleanest demonstration because **the content is identical by construction** — only the addressing changes.

**M4.C.5** The **inward-impingement** axis of M1.A.9a **MUST** have a **perception-side** readout computed at appraisal, before any move is emitted: reading the other's event as critical is itself the evidence. "Finally, I can be with my mother **without hearing her as being critical**. Well, if this person is **hearing** mother as being critical, then they probably are being critical, **defensive**." → `kb/kb05.md`. This readout is cheaper and harder to game than one taken from the emitted move.

### M4.D Select

**M4.D.1** Each person **MUST** resolve exactly one **outcome** per fast tick, by softmax over propensity scores. An outcome is either an emitted move (M5.A.1) or a **withheld** move (M4.D.1b).

**M4.D.1b** — *`WITHHOLD` **MUST** exist as an outcome distinct from every move in M5.A.1 and from M4.D.5's fused default.* The automatic channel's move **MUST** be computed, detected, and **not emitted**; and **a withheld move MUST still change tie state.** An implementation in which not acting is a no-op cannot represent the two canonical instances in the corpus. → `kerr_book/ks08.md` · KS08.1, `ks16.md` · KS16.2
> "I even started to move slightly, but **I caught myself and stopped**… **I did not take any obvious I-position with Mother; I just did not anxiously hover over her.**"

**M4.D.1c** — *`WITHHOLD` is **not** a weak `I-POSITION`, and on its own it is **insufficient**.* Kerr's three-year trajectory runs **counter-argument** ("accomplished nothing") → **non-reaction** ("**an insufficient response to her**") → **position** (lands, reaction, resolution). `M11.C` **MUST** assert that a `WITHHOLD`-only arm does not produce the `M5.E` resolution an `I-POSITION` arm produces. → `ks16.md` · KS16.1

**M4.D.1d** — *unresolved competition between candidate moves **MUST** itself raise `acute_anxiety`*, independently of which outcome resolves. Both parties in a loaded tie hold **simultaneous** approach and withdraw urges — responsibility for the other's distress pulling toward, fear of entanglement pulling away — and "**The conflicting urges raise each person's anxiety, which further infects their interactions.**" Implementable as an entropy or margin term over the propensity distribution fed back into M1.A.8; it produces the observed vicious circle without a separate rule. → `ks06.md` · KS06.4

**M4.D.1a** — *selection **MUST** run over **two channels with different objectives**, not one.* The **automatic** channel is driven by the relationship system, its objective is to discharge anxiety **now**, and it carries the seven reactive moves. The **self-directed** channel is driven by the person, its objective is to hold a position **through** discomfort (M5.F.5), and it carries `I-POSITION` and `STAY-IN-CONTACT`. The **mixing weight between them MUST be a function of differentiation**: at low level the person is nearly all automatic, and as level rises a real self-directed channel opens. Agency is **not absent** — it is graded, and the higher the level the less the individual is governed by what the system wants.
**Consequence:** differentiation is **not reachable by lengthening a reinforcement horizon**, because the two channels optimise different things and the target is not in the automatic channel's objective at all. This is what M4.D.6d and M11.C.17 assert. Anxiety-relieving action genuinely relieves anxiety — "cause and effect laws designed to **relieve the anxiety of the moment**, and the more we do that, **the more we promote the thing we're trying to fix**" — so an agent selecting a binder is not making an error a longer horizon would correct. → `kb/kb10.md` · K10.7; M1.A.0 (the moves are instinct-level, not feeling-states)

**M4.D.2** The propensity score **MUST** be a function of acute anxiety, `functional_level`, the state of the tie in question, the person's position in the active triangle, and their learned repertoire.

**M4.D.3** Rising anxiety **MUST** raise the weight on the seven reactive moves.

**M4.D.3a** — *the repertoire **MUST** carry a **complexity ordering**, and rising anxiety **MUST** slide selection down it.* A binary reactive/non-reactive split is too coarse. The ordering, from the source: **cooperation** ("a complex behavior… that requires the intellectual and emotional systems to function as a working team") → **conflict** → **dominant-adaptive** ("older evolutionarily, more primitive") → **distance**, the oldest ("single-celled organisms could not survive without a distancing mechanism"). Regression is "**dominated by less thoughtful and more reactive ways of interacting that are older in an evolutionary sense**". → `kerr_book/ks06.md` · KS06.1, `ks02.md` · KS02.3
**Consequence:** the propensity vector acquires a principled shape rather than nine free weights — a derivation in M10.A.2's preferred sense — and regression becomes **loss of the newer regulatory layer exposing an intact older one**, not damage, which is why it is reversible (`ks23.md` · KS23.2).

**M4.D.3b** — *engagement with a loaded tie **MUST** be gated by `systems_perspective` (M1.A.18), not by anxiety alone.* Without mindware the loaded tie is **correctly** avoided; with it, the same tie becomes approachable. This is the fourth of the six named ingredients of the differentiating process and nothing else in the spec supplies it — M4.D.3 alone drives every move toward *less* engagement. → `ks15.md` · KS15.1, KS15.11
> "'It's too difficult.' … **of course it is difficult if you lack a theory to guide you.**"

**M4.D.4** Propensity for `I-POSITION` **MUST NOT** be monotonically increasing in `basic_level`. An implementation in which the highest-`basic_level` member reliably moves first is a failing implementation. → §5.2

**M4.D.5** The default state **MUST** be the fused default — each altering self to manage the other's functioning while demanding the other change — and **MUST NOT** be modelled as an absence of a move. → §5.2

**M4.D.5a** — *a per-tie **accommodation stock** **MUST** grow monotonically under the fused default and **MUST NOT** decay.* Each concession is individually small and defensible; none is reversed; the cumulative effect is the family's behaviour progressively constrained by one member's reactivity. → `kerr_book/ks19.md` · KS19.3
> The father gave up hunting after his son's reaction to a killed rabbit; later he gave up watching television news when the son was home. "**Peace at any price is part of the problem, not part of the solution.**"

**M4.D.5b** — *the accommodation stock **MUST** shift the family's own reporting baseline.* "**the family learned to normalize Ted's aberrant behaviors, to weave them into the family fabric. This is part of what keeps highly influential emotional processes hidden.**" Two mothers in two unrelated families gave the same account of the years before catastrophe — "**there was nothing gross**" — and both were **sincere**. → `ks19.md` · KS19.3, KS19.8; `ks25.md` · KS25.7
**Consequence:** `M1.E`'s entry threshold is **family-relative**, not absolute, so a family whose baseline has drifted may never trigger it. That is a real dynamic for `M11.C.17`'s coach-free arm, not an artefact.

**M4.D.5c** The decision rule behind M4.D.5a **MUST** be short-horizon by construction: "**All parenting involves choosing between the day… and the years.** Nancy's error seems to have been that **she always focused on the day**… her willingness to indulge his isolation **may well have exacerbated the problems it was intended to ameliorate**." This is M4.D.6a's forbidden proxy in its human form, and `M11.C.16` is the test. → `ks21.md` · KS21.1

**M4.D.5d** — *`emotional_reserve` **MUST** be derived, and **MUST** gate symptom onset.* M4.D.5a's accommodation stock grows monotonically and currently has **no consequence** — nothing reads it. The source names the missing quantity repeatedly across two chapters: reserve is **capacity minus what is already committed to accommodation**. Therefore `emotional_reserve = f(basic_level) − accommodation_stock`, derived and never independently parameterised (M10.A.1), and symptom onset in M7.D **MUST** be gated on it rather than on absolute anxiety alone. This is what lets two agents at the same anxiety differ in whether they break, and it gives M4.D.5a the consequence it lacks. → `theory/family_evaluation/fe03.md` · FE03.10

**M4.D.6** Moves that worked before **MUST** be reinforced, so that a family develops a characteristic style.

**M4.D.6a** — *the reinforcement signal MUST be named, and it MUST NOT be short-horizon anxiety relief.* The obvious proxy — the post-move change in the actor's own acute anxiety — is **forbidden**, because this specification pins the timing that makes it self-defeating: M11.C.4 requires `CUTOFF` to drop the actor's acute anxiety *immediately* with its cost deferred to the next nodal event, and M11.C.5 requires `I-POSITION` to *raise* tension for a bounded window before it settles. A short-horizon anxiety proxy therefore rewards the seven reactive moves and punishes the two differentiating ones **by construction**, and every agent converges on `CUTOFF`.

**M4.D.6b** The signal **MUST** be evaluated over a horizon longer than the reaction window of M5.D, and **MUST** be declared in config with its horizon. `[I]`
**The horizon belongs to `functional_level`, which is the *shorter* timescale** (M1.A.5a) — **not** to the `M1.A.4a` estimator's window. Reinforcement operates on the swing term, over months; the estimator operates on `basic_level`, over years. Scaling the reinforcement horizon to the estimator window is a failing implementation and would reproduce the very over-correction `M11.C.16`'s second direction exists to catch. → user, G3 approval 2026-08-24

**M4.D.6c** This failure would be invisible to M11.C.1–M11.C.15, because all of them hold the policy fixed across both arms (M0.4) and a uniformly degenerate policy shifts both arms together. M11.C.16 exists to catch it.

**M4.D.6d** Reinforcement **MUST** operate on the **automatic channel only** (M4.D.1a) — that is what gives a family its characteristic style. `systems_perspective` (M1.A.18) and `basic_level` (M1.A.4) **MUST NOT** be reinforcement targets, and the self-directed channel **MUST NOT** be subject to M4.D.6 at all.

**M4.D.6e** — *reinforcement **MUST** be able to arrive from another person's state.* M4.D.6 reinforces on the actor's **own** signal, so nothing in the model currently drives the projection loop's step 5, internalisation. The named mechanism is cross-person: the child learns that acting as the parental image predicts **calms the mother**, and **her calming is the reinforcement**. Therefore a move's reinforcement signal **MUST** be able to include the change in a **witness's or target's** anxiety, not only the actor's, and the projection target's repertoire **MUST** shape itself to that signal. This is what makes `M11.C.2`'s concentration a learned outcome rather than an assigned one. It remains inside M4.D.6d: the channel reinforced is still the automatic one. → `theory/family_evaluation/fe07.md` · FE07.4

### M4.E Act

**M4.E.1** A selected move **MUST** become an event with sender, targets, witnesses, intensity, timestamp, route, fidelity and the tie's latency.

**M1.F.1a** — *the event record **MUST** carry the **channel** that selected the move* — `AUTOMATIC` or `SELF_DIRECTED`, or a mixture weight, since "**a given decision may contain elements of both.**" → `kerr_book/ks13.md` · KS13.5
**Without it the model cannot distinguish two cases the theory says are opposite.** "Going toward a goal" and "running away from a problem" can be the **same emitted act** — a geographic move, or staying put, is uninformative in **both** directions. Only the driving channel separates them, and no readout, and not the `M1.A.4a` estimator, can recover it after the fact.
It is also what makes `M1.A.5c`'s pseudo-self sign computable.

### M4.G Consolidate

**M4.G.1** Repeated moves **MUST** harden tie state: three withdrawals in a row **MUST** register as a distant relationship, not as three independent events.

**M4.G.2** All invariants in M6 **MUST** be asserted at the end of every fast tick.

---

## M5 — The move repertoire

### M5.A The nine core moves

**M5.A.1** The core repertoire **MUST** be exactly: `PURSUE`, `DISTANCE`, `CONFLICT`, `OVERFUNCTION`, `UNDERFUNCTION`, `TRIANGLE`, `CUTOFF`, `I-POSITION`, `STAY-IN-CONTACT`. → §5.1

**M5.A.2** No move outside M5.A.1 and M5.B **MAY** be added without a citation in the explainer.

### M5.B The six the corpus adds

**M5.B.1** `DETRIANGLE` — return an aligned third party to neutral. **MUST** leave the third party's knowledge intact and change only their position. → `_RESOLUTIONS.md` R2

**M5.B.2** `PREVENT_ALIGNMENT` — act on *potential* alignment before it forms. **MUST** be available preemptively, not only in response to an alignment that has already occurred. → R2

**M5.B.3a** `REDUCE_CUTOFF` **MUST** have a **floor as well as no ceiling**, and the target is **one open relationship, not universality** — "one can't have them with everybody, but if a person can have an open relationship with… **one important other**, there's indication that this is a more healthy way of life" (1979 Tape 6 · T6.7). Cutoff is load-bearing for the person maintaining it — "cutting off from people is your **lifeline** which enables you to live and adjust" — so it may be reduced only as fast as that person can absorb, and the far side of a cutoff has its own state and its own willingness. Total openness is also not the target: "to maintain a self, there has to be **some self that self does not communicate** to the other", and complete disclosure is "a **de-selfing** kind of thing". With `L12.4`'s two-sided closeness band, four sources agree that **contact is not monotonically good**. → `kb/kb05.md` · K05.2, `kb/kb08.md` · K08.4

**M5.B.3** `REDUCE_CUTOFF` — increase contact on a severed tie. **MUST** lower anxiety and **MUST NOT** directly raise `basic_level`. There **MUST NOT** be an optimum contact rate; more contact **MUST NOT** be penalised. → §5.7, §13.2

**M5.B.4** Family→external moves `SPLIT`, `FRAME_AMBIGUITY`, `DISPLACE` **MUST** exist. `FRAME_AMBIGUITY` **MUST** capture the external agent's *silence* as carrying professional weight — abstention **MUST NOT** be an exit. → §5.7

**M5.B.5** The external agent's repertoire **MUST** be a strict subset: `STAY-IN-CONTACT`, `I-POSITION`, `OVERFUNCTION`, `TRIANGLE`, `CUTOFF`, plus M1.E.3 and M1.E.4. → §5.8

### M5.C Gates

**M5.C.1** Each gate below **MUST** be evaluated before selection and **MUST** be able to remove a move from the draw.

| Gate | Effect | Explainer |
|---|---|---|
| `outside_ness` below threshold | `I-POSITION` lands as empty words or an assault — **not** as a weakened success | §5.3 |
| `financially_dependent` | `I-POSITION` **MUST** fail outright | §3.12 |
| no live issue | `I-POSITION` **delayed**, **MUST NOT** be blocked | §5.3 |
| mover's own engagement too high | `I-POSITION` backfires, with a long recovery cost on that tie | §5.3 |
| system calm | triangles inoperative; projection **MUST NOT** fire | §5.3 |
| marital distance high | any move targeting the symptom-bearer **MUST** produce no improvement | §5.3 |
| decision ownership held externally | removal **MUST NOT** fire even at tolerance | §5.3 |

**M5.C.1a** — *the live-issue gate MUST carry a family-type term, and this is a SAFETY property.* For a **peace-agree** family, whose differences are obliterated quickly, an old issue is deliberately raised to produce workable reactivity. For a **reactive, explosive** family the technique **inverts**: calm self first, and the success criterion is **relative** — "just be **less reactive than the others**, and you've made a step toward calming a family."
Applying the peace-agree technique to a reactive family is not merely ineffective. `KB08` records three outcomes of forcing conjoint contact on families already at capacity: acute psychosis, a patient who "**took scissors and put out her eyes**", and one who "**got a pass to go out of the hospital and went and killed herself**" two weeks into a protocol Bowen had warned against. → `kb/kb05.md` · K05.1, `kb/kb06.md` · K06.4, `kb/kb08.md` · K08.1

**M5.C.2** — *the non-monotonicity is on the mover's axis.* `I-POSITION` **MUST NOT** be gated on mid-band *family* anxiety. High-anxiety family events — serious illness, death — **MUST** be enabling. What backfires is the mover's own engagement level. An implementation that suppresses the move at high family anxiety is a failing implementation. → §5.3

### M5.D `I-POSITION` is a state machine

**M5.D.1** `I-POSITION` **MUST NOT** be a single-tick move with a propensity.

**M5.D.2** States, in order: `PREPARE` → `DEFINE` → `OPPOSITION` → (`ABORT` | `HOLD`) → `PEAK` → `RESOLVE` → `FOLLOW_UP`.

**M5.D.2a** — *`PREPARE` **MUST** exist and **MUST** be able to fail.* A differentiating move is not emitted on the tick it is decided. Bowen's own February 1967 effort ran **months of planning** and **one private letter per important triangle**, timed "**to cause the triangles to come to me.**" `PREPARE` therefore **MUST** occupy multiple fast ticks, **MUST** be able to act on each triangle separately (M5.B's `PREVENT_ALIGNMENT`), and a sequence entering `DEFINE` without it **MUST** carry a lower probability of reaching `PEAK`. This is also where M5.D.8's "several failures" partly lives: an unprepared move is one of the ways an attempt fails. → `theory/family_evaluation/fe11.md` · FE11.9

**M5.D.3** The abort branches — defend, counterattack, go silent — **MUST** sit at the *first* opposition and **MUST** be the usual outcome, not an exception. Each **MUST** return the mover to the prior balance.

**M5.D.4** — *corrected at revision 4; the earlier form was inverted.* The **mover's freedom from anger MUST** be the gate admitting the sequence to `PEAK`. Ch13 mentions anger once and only once, and it is the mover's: "When he is finally able to maintain his course **without getting angry at the opposition**, the opposition does a final intense emotional attack. If he remains calm with this, the opposition becomes calm and pulls up to his level of individuality." An angry mover **MUST** therefore **stall** the sequence — the final attack does not come at all — rather than abort it loudly. Anger **MUST NOT** be implemented as a fourth abort branch **and MUST NOT** be implemented as the condition that admits the peak. → `theory/ch13.md` · verified against the primary chapter, 2026-08-27

**M5.D.4a** — *the mover's anger MUST degrade the move, not merely delay it.* `I-POSITION` emitted while the mover's own anger is above threshold **MUST** execute as `M5.F.4`'s assertion form and **MUST** count as negative evidence under `M5.F.2a`. Three statements, two authors, two decades, every one of them about the **mover's** anger and not the system's: Ch13 above; Kerr 1988 — "**it is not fueled by anger. Anger can sometimes be a stimulus to clarify one's thinking, but it is not a reliable guide for action. When someone angrily and dogmatically claims to be a 'self,' he is usually unsure of his position and is blaming others for his plight in life**"; and Bowen 1988, written independently in the same volume — "**A dogmatic person is rarely sure of self.**"
**What anger is, in this model.** Not a gate and not a state (M1.A.0 still binds): an **intensity level on the negative side of the appraisal**, whose *handling* is the differentiation readout — noticed and managed, or discharged reactively as defence. The abort branches of M5.D.3 are what discharging it looks like. → *decision A3, 2026-08-27*; `theory/family_evaluation/fe04.md` · FE04.4, `fe11.md` · FE11.19

**M5.D.5** On a held `PEAK`, the opposition **MUST** pull up **to the mover's level**, not to a group mean, and the payoff **MUST** propagate to subsequent movers.

**M5.D.6** `FOLLOW_UP` **MUST** be mandatory on the next day *relative to the resolving encounter*. Skipping it **MUST** revert the gain.

**M5.D.7** Only a sequence reaching `FOLLOW_UP` counts as a *completed* exchange. A completed exchange **MUST** raise `functional_level` and **MUST** trigger the triangle decrement (M1.C.5). It **MUST NOT** write `basic_level`, which is derived (M1.A.4a) and reachable only through sustained, broad functional improvement.

**M5.D.7a** The `functional_level` increment **MUST** be small enough, and M1.A.4a's window and breadth requirements strict enough, that no realistic number of completed exchanges moves an agent materially up the `basic_level` scale within a run. The config **MUST** carry a comment saying why. Bowen names the contrary reading as the field's characteristic misconception — of someone who "went home to see their parents over the weekend and differentiated" he says it is "**grotesque** … they think of differentiation is something **you do in an hour a weekend**." Ch21's trip is **one step in a decades-long effort**, not the unit of differentiation.

**M5.D.7b** No readout **MUST** describe an agent that has completed an `I-POSITION` sequence as *differentiated*, or report completed exchanges as a differentiation score. → `theory/_KERR_INTERVIEWS.md`

**M5.D.8** Success **MUST** usually follow several failures. An implementation in which the first attempt typically succeeds is a failing implementation.

### M5.E The reaction ladder

**M5.E.1** The system response **MUST** run: *"You are wrong"* → *"Change back"* → *"If you do not, these are the consequences"*.

**M5.E.2** The reaction **MUST** decay unless fed by the mover defending or counterattacking.

**M5.E.3** Absence of any reaction **MUST** be interpreted as *the move did not land*, and **MUST NOT** be interpreted as success.

**M5.E.4** The ladder **MUST** admit a symptom rung at its foot — a symptom, not a verbal challenge, as the opening response — and a success branch at its head (the petition for sickness, which succeeds).

**M5.E.5** The push-back **MUST** be able to surface as symptom in a **third person**, not only as tension on the acting tie.

**M5.E.6** The trajectory **MUST** be a damped oscillation with hysteresis, not a spike-and-settle. Reversion **MUST NOT** be treated as run failure. → §8

**M5.E.7** The cause of the reaction **MUST** be the life-energy debit (M1.A.10, M6.I.3) — the move withdraws energy the other was receiving. The ladder is the surface expression. → §3.7

**M5.E.7a** — *what the debit **means** to the party who bears it.* The reaction occurs because the move **pushes the other to function more responsibly**, which they initially resist. The debit and the demand are one event seen from two ends, and stating it this way connects the change-back ladder to `M12`'s counterfeit of responsibility, which it otherwise touches nowhere. → user, A7 verdict 2026-08-24

**M5.E.8** — *the success state.* `M5.E` **MUST** have a terminal state that is **not** a return to baseline. A completed differentiating exchange **MUST** leave the tie **more solid** than before, with increased mutual respect — "**A hallmark of a successful effort to define 'self' is that it does not disrupt a relationship but, rather, solidifies it.**" `M11.C.5` **MUST** assert it. → `kerr_book/ks00.md` · KS00.6
Observed form: the pursue/distance polarity can **invert** — after a successful sequence, the previously pursuing party distances and the previously distancing party pursues (`ks17.md` · KS17.11). That inversion is assertable.

**M5.E.9** — *three timescales, and they are not one.* The config **MUST** declare all three separately, each `[I]` where the corpus gives no figure:

| Scale | What settles | Source |
|---|---|---|
| **~1 week** | the acute reaction at peak, before it breaks | `ks07.md` · KS07.1 |
| **months** | the system's re-equilibration after a held position | user, A7 verdict |
| **years to a decade** | resolution of a primary tie; **ten years** on one relationship, **twelve** of preparation for one visit | `ks00.md` · KS00.6; `ks15.md` · KS15.6 |

Together these bound `M5.D.7a`: **the unit of differentiation is not the exchange**, and no realistic number of exchanges moves an agent materially within a run.

### M5.F Act identity — the counterfeit problem

**M5.F.1** A move's effect **MUST NOT** be a function of move type and tie state alone. It **MUST** be multiplied by a hidden actor state (`outside_ness`) that receivers can read and the actor may not. That state is **two-dimensional** per M1.A.9a — outward impingement and inward impingement — and a scalar implementation is a failing implementation. → §5.6

**M5.F.2** The same move type at low `outside_ness` **MUST** be able to produce the *opposite* sign of effect, not merely a smaller one.

**M5.F.2a** — *asserting a differentiated state MUST be negative evidence for it.* Five independent forms across five interviews: declaring non-involvement "**in itself is an indicator of involvement**"; "I'm out of it, you handle it" is "**mostly denial, because they are in it**"; "the more the individual has to say **I've worked it out**, is evidence of an attachment"; a low-level self holds a **selfish, dogmatic, forceful** I-position; and catching oneself **diagnosing, criticising or praising** is a loss of perspective. → `kb/_KB_PASS2.md`
**Consequence:** a cheap readout computable from the event log, and the discriminator M5.F.3 asks for. An agent that announces its own neutrality **MUST** score lower on `outside_ness`, not higher.

**M5.F.2b** The counterfeit detector **MUST** report **which axis failed** (M1.A.9a). A single scalar score is a failing implementation, because the forceful declarer and the compliant accommodator are counterfeits in **opposite** directions and require opposite corrections. → M11.C.19

**M5.F.4** `I-POSITION` selected at low `systems_perspective` (M1.A.18) **MUST** execute as the **assertion form**: it **MUST** raise reactivity on the tie rather than lower it, and **MUST** count as negative evidence under M5.F.2a. This makes pseudo-differentiation an **output** of the model rather than a case the model must be told about separately.

**M5.F.5** The self-directed channel's objective (M4.D.1a) **MUST** be the two-axis position of M1.A.9a — both impingement axes low — and **MUST NOT** be discomfort reduction. A differentiated move is held **through** discomfort; scoring it by the actor's own relief reproduces M4.D.6a's forbidden proxy at the objective level.

**M5.F.3** Concession **MUST** be gradable continuously by the receiver, not as a binary. → §5.6

---

## M6 — Invariants

Asserted at the end of every fast tick (M4.G.2). A violation **MUST** raise, not warn.

| ID | Invariant | Explainer |
|---|---|---|
| **M6.I.1** | The family undifferentiation budget has exactly **three** sinks and is conserved across them | §7 I1 |
| **M6.I.2** | Emotional distance runs **outside** the budget as an always-on baseline; it **MUST NOT** be a fourth sink | §7 I2 |
| **M6.I.3** | `life_energy` is zero-sum per person between relationship-seeking and goal-directed activity | §7 I3 |
| **M6.I.4** | Dyadic exchange conserves **pseudo-self** — defined in 1979 Tape 2 as what is *negotiable*: basic self is "what they believe in, what they stand for… where they will stand **no matter what**", pseudo-self is acquired belief and "**the pseudo-self is negotiable**". The negotiability test is the exemption test: one spouse's functional gain equals the other's loss | §7 I4 |
| **M6.I.5** | Solid self is **exempt** from fusion and **MUST NOT** participate in M6.I.4 | §7 I5 |
| **M6.I.6** | Anxiety is conserved and redirected, never destroyed; blocking one channel raises flow on the rest | §7 I6 |
| **M6.I.7** | There is no exit from the field; `CUTOFF`, `DISTANCE` and silence are moves *inside* the system | §7 I7 |
| **M6.I.8** | The standing load runs before delivery, every tick, on every tie | §7.2 — *this spec elevates it to an invariant; the explainer describes it as a mechanism* |

**M6.1** M6.I.1 conservation **MUST** be asserted to a stated tolerance, declared in config, and the tolerance **MUST** be `[I]`.

**M6.2** M6.I.6 **MUST** be assertable across a removal event specifically (M11.C.11).

---

## M7 — The slow tick

**M7.A.1** `basic_level` **MUST NOT** be advanced by any move or by any per-exchange increment. It **MUST** be recomputed on the slow tick as the estimator of M1.A.4a–M1.A.4c over `functional_level` history. Recovery of `functional_level` toward baseline when calm **MUST NOT** be a symmetric restoring force.

**M7.A.1a** — *the dependence gate **MUST** run on the slow clock too.* `M1.A.15`'s `financially_dependent` flag currently gates one **move** (M5.C). `FE04.9` puts the same condition on **basic-level change itself**: change requires that the person be **self-sustainingly independent of the family of origin**. Therefore `basic_level` **MUST NOT** rise on the slow tick while the agent is financially dependent, however many exchanges completed. Without this gate an agent accumulates `functional_level` gains and ratchets `basic_level` upward while still inside the exact condition under which the change-back reaction's third rung has teeth (M1.A.15) — which would make M5.C's fast-clock gate cosmetic. → *decision B1-tail, 2026-08-27*; `theory/family_evaluation/fe04.md` · FE04.9

**M7.A.2** — *amended.* Differentiation gained in a peripheral system **MUST NOT** transfer automatically to the nuclear family. **The capacity transfers; the application is re-earned per tie.** `basic_level` is per-person and rises on the M1.A.4a estimator; **applying** it in a given relationship requires observing the reciprocity **in that relationship**. → §10, §15 Ch10/Ch21; `kerr_book/ks16.md` · KS16.9
> "**Progress in the family of origin does not transfer automatically to nuclear family relationships, but it helps considerably.**" Two reasons: the spouse "**bring[s] aspects of their own unresolved attachment**", and one "**lives day in and day out with them, which makes the emotional process more intense and more difficult to observe objectively.**"
**The evidence is unambiguous.** Kerr had been "**teaching this cocreation idea to students of Bowen theory for over twenty-five years**" before observing it in his own marriage (`ks16.md` · KS16.6); and Mr. S. held systems thinking for his sister's triangle, his sons' triangle and his family history while lacking it in his marriage — "**I acknowledge Bowen theory, but I don't feel it**" (`ks17.md` · KS17.1).

**M7.A.2a** — *resolved, 2026-08-25.* `systems_perspective` is a **per-person state with per-tie attenuation by load** (M1.A.18d), not a per-tie state variable. The "application re-earned per tie" in M7.A.2 is therefore **not** a second acquisition — it is the same capacity reaching a tie whose intensity had previously exceeded it. No new state; the attenuation term does the work.

**M7.A.2b** The **family of origin is the privileged peripheral system**, not any peripheral system — it is the highest-load context (`ks06.md` · KS06.8) so gains demonstrated there carry the most evidential weight, and it is where the assumptions being tested were formed. Where the parents are dead, ties to **anyone who carried the same triangle** substitute — "much can be gained by working on relationships with people who were close to the parents, such as aunts and uncles." → `ks13.md` · KS13.11

**M7.B.1** `chronic_anxiety` **MUST** be fixed once in childhood from witnessed history (M1.A.7). The age is `[I]`.

**M7.C.1** Life stage **MUST** govern childhood, launch, partnering and mortality.

**M7.C.1a** — *the addition of a tie **MUST** be modelled as a **redistribution of a conserved investment budget**, not as a stressor.* A new claimant reduces every existing tie's share **by construction**; nothing bad need happen. This is M1.A.10's zero-sum allocation operating **between ties**. → `kerr_book/ks22.md` · KS22.2
> "**Her reduced investment is a fact insofar as part of her emotional investment in John is constrained by her investment in the unborn child.**"
Two instances with different claimant types — a child (KS22.2) and a fiancée (`ks25.md` · KS25.5) — so the mechanism is the **redistribution**, not the kind of tie.

**M7.C.1b** The redistribution **MUST** be able to move **`investment` and `expectations` in opposite directions on one tie simultaneously** — her investment fell while her expectations of him rose. A scalar tie state cannot express this; `M4.C.1`'s four signed channels (M1.A.9a, KS10.1) can. → `ks22.md` · KS22.2

**M7.C.1c** The **effect** of the redistribution **MUST** scale with the affected tie's **share of the agent's total investment**, not with the event. Same nodal event type, same person, six years apart, opposite outcomes: a low-investment partner's pregnancy produced no episode; a high-investment partner's preceded the breakdown. "**John not committing fully to either relationship distanced him from their dependence on him and also his dependence on them.**" → `ks22.md` · KS22.3

**M7.C.1d** — *withdrawal at extreme fusion is a **removal**, not a differentiation.* Where a person's fusion is concentrated in a **single** tie with no alternatives, a reduction in the supporting party's investment **MUST** be resolved as a removal event (`M11.C.11`), **not** as a differentiating move — regardless of the withdrawing party's intent. The two are distinguishable **only** by the other's remaining alternative ties. → `ks25.md` · KS25.3, KS25.4; `ks21.md` · KS21.9
**Two independent cases, forty-three years apart.** Bowen, the morning after: "**Mike, I think your mother pulled up and Billy suicided.**" And Nancy Lanza's withdrawal had the *form* of the move — reducing overfunctioning, redirecting energy to her own life, explicitly hoping it would produce independence — with **none** of the preconditions in `kerr_book/ks15.md` · KS15.1. **This is the model's most consequential asymmetry and MUST NOT be softened.**

**M7.D.1** Symptom load **MUST** accumulate from routed load and, on crossing threshold, **MUST** emit an **endogenous** event (M1.F.7).

**M7.D.2** The three channels **MUST** be substitutable — the same deficit **MUST** be able to present as any of the three.

**M7.D.2a** — *symptom **lock-in**.* Once a member is carrying symptom load, the **family's subsequent anxiety MUST fall**, producing a stable configuration that **resists the symptom's removal**. → `kerr_book/ks23.md` · KS23.1
> "**a family can stabilize somewhat around the presence of a symptom, which fosters it becoming chronic**, or the regression can continue to get worse."
**This one requirement yields three observed behaviours as consequences**, none of which the spec previously produced: **chronicity** without any pathology; **relapse on cure** (`M11.C.12`); and **relief on removal** of the symptomatic member (`ks20.md` · KS20.8 — "the family tensions dropped significantly during Gary's time in reform school"). It is also why the Nash marriage stabilised around the deteriorating son (`ks22.md` · KS22.13).

**M7.D.2b** Removal of a member **MUST** produce **opposite-signed** effects on different members, keyed to their position relative to the removed one — not a uniform family-level shift. When Gary Gilmore was removed the family relaxed and **only his mother** wanted him back; when both of Nash's son's parents died, **the projection target improved**. → `ks20.md` · KS20.8; `ks22.md` · KS22.8

**M7.D.2c** — *the three channels are **mutually protective**, not merely substitutable.* M7.D.2 makes them alternatives for the same deficit. The source goes further: **occupancy of one channel lowers the hazard on the others** — reciprocal functioning **within** a single person, the same shape M1.B.6 gives a dyad. Therefore symptom hazard in each channel **MUST** be reduced by load already carried in another, and a model that treats the three as independent draws is a failing implementation. Consequence: **a person carrying one symptom is protected against a second**, which is what makes channel-switching visible as an event rather than as accumulation. → `theory/family_evaluation/fe01.md` · FE01.6, `fe05.md` · FE05.8

**M7.D.2d** — *lock-in **MUST** be non-monotone in severity, and this **amends** M7.D.2a.* M7.D.2a as written makes the stabilising effect grow without limit, so a severe enough symptom becomes maximally stabilising. Three independent statements say the curve turns over: past a severity threshold the symptom **destabilises** the configuration it was stabilising, and removal then brings **relief** rather than the resistance M7.D.2a predicts. The stabilising term **MUST** therefore rise with symptom load to a peak and fall beyond it, and `M11.C.22` **MUST** test both limbs. This is what reconciles M7.D.2a's resistance-to-removal with M7.D.2b's relief-on-removal: they are the same curve either side of the turn. → `theory/family_evaluation/fe03.md` · FE03.9, `fe07.md` · FE07.15

**M7.D.3** Curing a symptom without changing the underlying deficit **MUST** raise family tension. → M11.C.12

**M7.E.4** — *ties **MUST** deteriorate by default, at a rate inverse to level.* A relationship left alone does not hold its state: without active work it degrades, and it degrades **faster the lower the pair's `basic_level`**. Each tie **MUST** therefore carry a slow-clock decay on conductance and a slow-clock accrual on bond energy, scaled inversely to level, which any `STAY-IN-CONTACT` or completed exchange offsets. **This is what makes `M11.C.17`'s coach-free arm a real test rather than a trivial one**: a flat arm produced by *nothing happening* proves nothing, whereas a flat arm produced by decay and effort cancelling is a result. → `theory/family_evaluation/fe03.md` · FE03.13

**M7.D.4** Removal (`INSTITUTIONALIZE`) **MUST** fire off the **remaining members'** tolerance for disturbing behaviour, explicitly decoupled from the symptom-bearer's severity, and **MUST** be suppressible by relocating decision ownership (M5.C.1). → §6.2

**M7.E.1** The generational update **MUST** produce: the primary projection object **lower** than the parents, minimally-involved siblings about the same, those outside the process **better**.

**M7.E.1a** — *the transmission rule, stated.* Parents are at the **same** level (M2.A.0c), and "**they can produce children with basic levels a little higher, a little lower, or the same as their levels**" — a **three-outcome distribution centred on the parental level**, with the child's own parental triangle (M7.E.1b) deciding which offset. It **MUST NOT** average dissimilar parents (they are not dissimilar) and **MUST NOT** impose a guaranteed decline: the projection **line** declines, the family does not. → `kerr_book/ks11.md` · KS11.2

**M7.E.1b** — *siblings differ because they occupy **different parental triangles**, not because they received different event counts.* "**siblings grow up in the same family but in different triangles.**" The four differentiating properties are of the *triangle*: less/more anxious investment, more/less mature interactions, more goal-directed / more relationship-oriented, more/less "self". Strongest evidence in the corpus: a mother of **identical twins** describing one relationship as "**almost addicted**" to mutual reactivity and the other as markedly less so. → `ks09.md` · KS09.1

**M7.E.1c** — *the projection target **MUST** be selected from a closed list of five situations, evaluated against the family's history at each birth:* **the firstborn; the firstborn of a given sex; a child born with a reality defect; a child born at a time of high stress in the family, nuclear or extended; the last-born.** More than one child may be a focus. Because the fourth is a **timing** condition, target selection is history-dependent and therefore **emergent** — which is the correct shape. → `ks09.md` · KS09.2

**M7.E.1d** — *the projection **MUST** be able to initiate with **no signal from the child**.* It is a self-fulfilling prophecy in Merton's sense: "the process begins with **a false definition of a situation that evokes a new behavior, which makes the false conception come true**", and "**The defect may also be a total product of the mother's imagination.**" Checked against an independent record in two families — home movies in both. → `ks09.md` · KS09.3; `ks25.md` · KS25.7

**M7.E.1e** Transmission **MUST** read the parents' functional level **on the marital tie**, not a global average: "**how people manage themselves in the intimate relationship with their spouses, and not how they manage themselves in their work lives, is the primary determinant of the basic levels of differentiation of their offspring.**" → `ks06.md` · KS06.8

**M7.E.2** The generational rate **MUST** be stochastic — fast for a few generations, static for one or two, then fast again — and reversible at both extremes. A fixed per-generation decrement **MUST NOT** be implemented; no such rate exists in the corpus. → §8

**M7.E.3** Cutoff **MUST** form a positive feedback loop: more intense cutoff → more exaggerated parental problem in the next marriage → more intense cutoff in the generation after.


---

## M8 — The live-position predicate

This is one rule used in four places. It **MUST** be implemented once and called from all of them. → `_RESOLUTIONS.md`

**M8.1** `positions_live(group)` **MUST** count only occupants **not fused into another member of the group**.

```
positions_live(g) = |{ p in g : present(p) AND NOT fused_into(p, other ∈ g) }|
avoidance_available(g) = positions_live(g) < 3        # a step, not a gradient
```

**M8.2** An external agent in the outside position **MUST** count as a live position.

**M8.3** An external agent who has taken a side **MUST NOT** count — they have fused into one of the two.

**M8.4** A member present but **emotionally inactive** — the displaced member of an interlocking triangle — **MUST NOT** count.

**M8.5** The same predicate **MUST** decide: whether a group can avoid its issue (M5.C), whether a third party is a witness or a new peripheral triangle (M8.6), whether the coach is in the outside position (M11.C.7), and whether a triangle is active (M1.C.3).

**M8.6** — *alignment, not knowledge.* A third party's effect on a differentiating move **MUST** be determined by their *position*, never by what they know.

**M8.6a** — *an ally is a **pseudo-self transaction**, not only a peripheral triangle.* Alignment **MUST** transfer pseudo-self on `M6.I.4`'s conserved quantity — borrowing from the ally, or lending to them. → user, B4 verdict 2026-08-24
**Why this makes the ally penalty computable rather than declared.** A differentiating move is meant to come from **solid** self, and agency is the solid-self fraction (`M10.A.1a`). A move fuelled by borrowed self is **not drawing on that reserve at all**, so it is counterfeit by construction under `M5.F.2a` — the model works it out instead of being told. It also explains why the effect is *undoing* rather than mere complication, which the triangle account alone did not.
**Corroborated at scale.** Fifty to sixty observers at the Medical College of Virginia, and the harm mechanism was **alignment, not information** — "**a colleague… touched me softly on the arm in a way that convinced me that she had taken sides**"; "**It is too easy to be emboldened to think you are right when someone else reinforces that view.**" → `kerr_book/ks18.md` · KS18.3

**M8.6b** The countermeasure **MUST** be available to an external agent: **routing**. All traffic addressed to the neutral party, who decides what passes — a gate on the event graph, which `M1.E` did not previously carry. → `ks18.md` · KS18.3

- Position `NEUTRAL` → **no effect**, regardless of knowledge.
- Position `ALIGNED_WITH_MOVER` → **MUST** open a new peripheral triangle; the move's gain leaks until the ally is detriangled.
- `role = EXTERNAL` and perceived as against the family → **MUST** open a peripheral triangle.

**M8.7** Knowledge **MUST** be modelled as a *hazard rate on alignment*, high inside the system and low outside it — **MUST NOT** be an independent gate.

**M8.8** Announcing a specific **act** to the party in that dyad **MUST** be permitted and **MUST NOT** carry a penalty. Announcing the **programme** to anyone inside the system **MUST** carry one. Three distinct objects; conflating them is what produced the contradiction this resolves.

---

## M9 — The belief layer

**M9.1** A family-level and per-person belief store **MUST** exist and **MUST** be permitted to differ from ground truth. → §9.5

**M9.2** Beliefs **MUST NOT** be implemented such that emotionally-driven claims are the false ones. Truth value and emotional function are orthogonal — a claim **MUST** be able to be accurate and still be serving a denial. → §9.5

**M9.3** Institutional acts (M1.E.4) **MUST** write to the who-is-sick belief with hysteresis that verbal denial cannot reverse.

**M9.4** Shock-wave propagation to an unattached member **MUST** use multi-hop propagation on the ordinary tie graph plus the belief layer. A second "dependence" edge type **MUST NOT** be introduced. → §9.5

**M9.5** Readouts **MUST** be able to report ground truth and belief separately. `→E`

**M9.6** — *sink allocation **MUST** be identifiable from the belief configuration.* The three sinks of M1.D.1 carry three distinct attribution patterns: **marital conflict — each says the other**; **spouse dysfunction — both say the same one**; **child projection — both say the children**. The belief store **MUST** therefore hold, per person, an attribution of where the family's difficulty lies, and the readout **MUST** be able to recover the active sink from that configuration and the configuration from the active sink. This is the first connection between `M1.D.1` and `M9`, and it is assertable in **both** directions — `M11.C.26`. → `theory/family_evaluation/fe07.md` · FE07.2

**M9.7** — *belief **MUST** be able to drive appraisal, not only record it.* M9.1 as written makes the belief layer a parallel store. Three independent arguments make it a **channel**: the three systems influence one another **in both directions** (`FE02.7`); chronic anxiety runs on **what might be** rather than on what is, so its input is belief and not event (`FE05.10`); and Bowen corrects his own term — the transfer is not "projection" but runs **through descriptions** (`FE11.3`), which makes `M9` the medium the family projection process operates **on** rather than a readout beside it.
Therefore M4.C's appraisal **MUST** read the receiver's belief about the sender and the situation, not the ground-truth event alone. **M9.2 still binds**: a belief that drives appraisal **MUST NOT** be assumed false because it is emotionally loaded, and the model **MUST NOT** implement "the emotionally driven claim is the wrong one". → `theory/family_evaluation/fe02.md` · FE02.7, `fe05.md` · FE05.10, `fe11.md` · FE11.3

---

## M10 — Parameters

**M10.1** Every constant **MUST** appear in the register with a grade (M0.1) and **MUST** resolve either by derivation from `basic_level` or from markdown config (M0.3).

### M10.A Derived from `basic_level` — preferred

**M10.A.1** The following **MUST** be derived, not independently parameterised: reactivity (M1.A.1), `functional_level` variance (M1.A.5), the `life_energy` ratio (M1.A.10), the stabiliser repertoire available to a person, transfer magnitude in M6.I.4 (which scales *inversely* with basic level), the **ceiling** on `systems_perspective` (M1.A.18b), and **agency**.

**M10.A.1a** **Agency MUST be derived as the solid-self fraction and MUST NOT be an independent parameter.** The model already carries it: **pseudo-self is the portion of the person the relationship system can move, and solid self is the portion it cannot.** M6.I.4 already makes pseudo-self the conserved, negotiable quantity that transfers between people in a fused relationship, and the 1979 definition of pseudo-self is precisely "**negotiable**" — which is what "governed by what the system wants" means. Adding a separate agency parameter would cost against M10.A.2 for nothing.

**M10.A.2** Deriving from `basic_level` is preferred wherever it is defensible, and the source states the principle directly: "a theory is made up of the **least number of pieces that will hang together into a story**, rather than trying to put in all of the others." A model with 60–90 free parameters is already in tension with that, so every added parameter **MUST** be justified against it rather than added by default. → `kb/kb10.md` · K10.9, because the model has 60–90 free parameters against a single invented family and the risk is overfitting to the modeller's intuitions.

### M10.B Config

**M10.B.1** All remaining constants **MUST** live in markdown config. Editorial content — event kinds, the reference family, the access vector, tie declarations, dial ranges — **MUST NOT** live in Python.

**M10.B.2** Config parsing **MUST** fail loudly on an unrecognised key or a malformed line. Silently skipping and falling back to a default **MUST NOT** occur. *(This is a defect in the frozen engine's `_apply_config`; v2 must not inherit it.)*

**M10.B.3** The `basic_level` estimator's **window length** and **breadth count** (M1.A.4c) **MUST** be declared here, graded `[I]`, and every run reporting `basic_level` **MUST** carry a sensitivity analysis over both. A conclusion that depends on choosing 30 situations rather than 12 **MUST** be visible rather than buried.

### M10.C The `[I]` register

**M10.C.1** The following are **invented** and **MUST** be labelled as such wherever they surface: the fast tick length; the softmax temperature and every propensity coefficient; all conductance values; all bond-energy values and its decay rate; the standing-load function; every gate threshold including the `outside_ness` threshold; `societal_leadership`'s functional form; the appraisal gain function's shape; the chronic-anxiety fixation age; sibling-position effect sizes; the M6.I.1 conservation tolerance; every value in M2.

**M10.C.2** The following are **stated in the corpus** and are usable as calibration targets: the durations in `model_explainer.md` §8, the 0–100 scale and the transition at 50, the ~90%-in-the-lower-half population skew, the three-sink count, and — **withdrawn at pass 3** — the eight-to-ten-generation figure, which `KB11` shows was rhetorical: "I put [it] in **not to say that it takes ten generations**."

**M10.C.2a** One **external** calibration target is admitted: the DSI–trait-anxiety association, **r = .64** (Skowron & Friedlander 1998, N = 609). It **MAY** be used to check an ensemble's differentiation–anxiety coupling and **MUST NOT** be used to set a parameter. Two limits **MUST** travel with it in any report: it relates two self-reports, so shared method variance inflates it; and it is a population association in a sample that is 82.7% White and largely northeastern US, not a norm. → `theory/_EXTERNAL_MEASURES.md`

**M10.C.2b** No output **MUST** be described as reproducing *measured* differentiation. The best available instrument reports α ≈ .88 overall, and its Fusion With Others subscale — the construct most central to the theory — required a five-year rebuild after failing to relate to psychological adjustment at all. The model **MUST NOT** claim precision the measurement literature does not have.

**M10.C.4** — *bounds and shapes stated in the corpus, admitted as **checks**, never as parameters.* *Family Evaluation* is the one source that gives more than a handful of these. Every entry below is a **bound or a shape**, not a rate; each **MUST** be usable to check an ensemble output and **MUST NOT** be settable in config (M10.C.3a). All carry `[#]` and the attribution grade of their segment — `[K]` unless marked `[B]`.

| Quantity | Value | Source |
|---|---|---|
| Species median `basic_level` | **≈ 40** — below M1.A.3's transition | `FE03.5` |
| Band edges | 0–10 very poor · >60 well · >70 very well · **100 unreachable** | `FE03.5` |
| Pseudo-self transfer, worked | **35 + 35 → 55 + 15** — conserved exactly; the corpus's only arithmetic instance of M6.I.4 | `FE04.1` |
| Per-generation step, typical | **< 5 points** | `FE08.7` |
| Per-generation step, consequential | **5–10 points** | `FE07.7` |
| Parent-to-child bound | **never 30 points** | `FE07.7` |
| Quantum jump | **~10 points per generation over two generations**, under three conjunctive conditions | `FE08.1` |
| Full traversal | 3 generations (quantum jump) to 5–10 (typical), **symmetric in both directions** | `FE08.2` |
| Generation length | ~25 years | `FE08.2` |
| Sibling-position decay | a gap of **≥ 5 years** reduces predictability | `FE10.7` |
| Treatability floor | **upper two-thirds** of the schizophrenic range | `FE11.10` |
| Borrowed-gain gap, worked | apparent **40–45**, true **25–30**, revealed by a matriarch's death | `FE04.8` |

**M10.C.4a** — *the quantum-jump conditions are a three-way ablation, not a rate.* The most specific mechanism-plus-magnitude statement in the whole corpus, and the only one that yields a factorial test: the jump requires, **held over two successive generations**, (1) anxiety bound **primarily in focus on one child**, (2) the family **poorly connected to the extended family**, and (3) **at least average stress**. Each condition **MUST** be independently ablatable, and removing any one **MUST** remove the jump. Condition (2) gives `M1.D.8` and `M2.3` a quantified consequence for the first time — this is the test that makes the family-of-origin ties load-bearing rather than decorative. → `theory/family_evaluation/fe08.md` · FE08.1


**M10.C.3** The following **MUST NOT** be implemented as rates, because they were manufactured from illustrations Bowen explicitly bounded: any per-generation `basic_level` decrement, and any annual societal-regression rate. → §8

**M10.C.3a** — *amended: the prohibition is on the **parameter**, not on the quantity.* M10.C.3 reads as forbidding a per-generation change altogether. It does not. A per-generation `basic_level` change **MUST** be an **output** of the three mechanisms in M1.D.1 and M7, observed at readout, and **MUST NOT** appear anywhere as a configurable rate. The bounds in M10.C.4 are then usable as **checks on that output**, which is the only role a stated magnitude can play in this model.

---

## M11 — Acceptance criteria

Each criterion **MUST** have a named test, **MUST** assert a direction of difference between two arms (M0.4), and **MUST** be proved failing by mutation before it counts as coverage: delete or invert the mechanism it names, confirm red with the **verbatim** original restored afterwards.

**M11.1** A test **MUST NOT** be counted as coverage until its mutation has actually been run, not reasoned about.

**M11.1a** A mutation **MUST** be a real behaviour change. Where the value under test is written more than once — by a redundant assignment, or through two names bound to the same object — the mutation **MUST** name every write, or it proves nothing while looking exactly like a test that cannot fail. *Measured in the frozen engine: `nuclear_family_id` is bound to the same array as `family_ids`, so mutating one line of the triangle attach is overwritten by the next line and the test stays green.* This is why M1.A gives each field exactly one name.

**M11.2** These are ensemble property tests. A single-run assertion **MUST NOT** be used for any of M11.C.1–M11.C.16.

**M11.3** — *every discriminating criterion **MUST** run under a stressor.* Cohesion produced by togetherness and cohesion produced by individuality are **indistinguishable when the system is calm**; so are most of the differentiation readouts in M11.C. A criterion whose two arms are compared in a quiet stretch is measuring nothing, and will pass. Each M11.C criterion **MUST** therefore name the stressor its arms run under, and a criterion that cannot name one **MUST** be moved to M11.E. → `theory/family_evaluation/fe04.md` · FE04.18

| ID | Criterion | Test | Phase | Mutation target |
|---|---|---|---|---|
| **M11.C.1** | Differentiation protects: identical families differing only in basic `basic_level`, same stressor schedule → the lower-`basic_level` family reaches symptom threshold sooner in a significant majority of seeds | `test_m11c1_lower_c_reaches_threshold_sooner` | C | the `/ functional_level` divisor in M4.C.1 |
| **M11.C.2** | Symptoms concentrate on the member who received the most projection-type events, not uniformly across children | `test_m11c2_symptom_concentrates_on_projection_target` | D | the witness path, M1.F.5 |
| **M11.C.3** | Triangling relieves the pair and costs the third, within the same tick. **One of four cells** — see `M11.C.27`, which the corpus requires and this criterion alone does not cover | `test_m11c3_triangle_relieves_pair_costs_third` | C | `bound_anxiety` transfer, M1.C.1 |
| **M11.C.4** | Cut-off trades now against later: `CUTOFF` drops the actor's acute anxiety immediately and raises family total anxiety at the next nodal event, against a matched no-cutoff arm | `test_m11c4_cutoff_trades_now_against_later` | C | the standing load, M4.A.1 |
| **M11.C.5** | The system pushes back — as **symptom in a named third person**, as a damped oscillation with hysteresis, decaying unless fed, reverting if `FOLLOW_UP` is skipped; absence of reaction means the move did not land | `test_m11c5_change_back_reaction_shape` | C | the life-energy debit, M5.E.7 |
| **M11.C.6** | Transmission is multigenerational: over three generations with no external stressor change, mean `basic_level` in the projection line declines while the non-target line does not. **Two shape constraints.** (a) **Traversal:** a decline from the top of the scale to the bottom **MUST NOT** complete in one or two generations and **MUST** be achievable within roughly ten — "**It could take eight to ten generations or as few as three or four**" (`kerr_book/ks11.md` · KS11.1 `[#]`; recorded as a **shape** constraint only — `M10.C.3`'s prohibition on a per-generation *rate* stands, per approved decision `A1`). (b) **Variance:** the **spread** of `basic_level` within the family **MUST** widen across generations — "**The essence of the multigenerational transmission process concept is that it describes a natural process that generates variation**" (`ks11.md` · KS11.9), bounded below by line extinction (`ks20.md` · KS20.1) and above by the species ceiling (`ks05.md` · KS05.3) | `test_m11c6_multigenerational_decline_in_projection_line` | D | M7.E.1 |
| **M11.C.7** | **Position, not skill.** Hold the coach's parameters fixed and vary who talks to whom in whose presence; the topology arms **MUST** differ | `test_m11c7_topology_not_coach_skill` | C | M8.2/M8.3 |
| **M11.C.8** | Endogenous incidence lands near published rates | — | **`→E`** | — |
| **M11.C.9** | **Sibling-position effect peaks mid-scale.** Three arms — low / mid / high `basic_level` — same sibling position, same stressor schedule. The **mid arm MUST show the largest position-typical behaviour** and the low arm the least. The former criterion held `basic_level` constant, which is the variable that *modulates* the effect, and needed an invented effect size; this needs none. Bowen "**qualified his thoughts about Toman's profiles by saying that they accurately describe people at the midrange**", and "**a poorly differentiated oldest brother may exhibit very few characteristics of an oldest profile**" → `kerr_book/ks12.md` · KS12.1 | `test_m11c9_position_effect_peaks_midscale` | D | M1.A.14c's gating — making the effect additive **MUST** turn this red |
| **M11.C.10** | Cut-off begets cut-off: a generation containing a cut-off produces more in the next than a matched arm | `test_m11c10_cutoff_begets_cutoff` | D | M7.E.3 |
| **M11.C.11** | **Removal produces three phases, not a step down — and the return is the larger disturbance.** Rise, partial relief with redirected focus, sustained residual; total anxiety conserved across the removal to a stated tolerance; and in the hospitalize-and-release arm the **re-escalation on return MUST exceed the original removal disturbance**, because the family has re-equilibrated in the member's absence — 1979 Tape 6 names the return of a long-removed member as "probably the **most intense**" addition a family system faces | `test_m11c11_removal_three_phase_shape` | D | M6.I.6 |
| **M11.C.12** | **Curing a symptom without changing the deficit raises tension.** Remit a spouse's dysfunction leaving the functioning balance intact → marital conflict rises | `test_m11c12_symptom_relief_raises_conflict` | D | M7.D.3 |
| **M11.C.13** | Help relocates incidents; it does not reduce them. Score **count and location** — relocation from community into family carries a positive sign | `test_m11c13_help_relocates_not_reduces` | C | M6.I.6 |
| **M11.C.14** | Management technique has zero independent effect while marital distance is high | `test_m11c14_technique_null_under_marital_distance` | C | the marital-distance gate, M5.C.1 |
| **M11.C.16** | **The learned repertoire does not collapse onto relief-seeking.** Two arms, identical seeds, differing only in the reinforcement horizon: the short-horizon arm (M4.D.6a's forbidden proxy) against the declared-horizon arm (M4.D.6b). The short-horizon arm **MUST** show a strictly higher share of `CUTOFF` and `DISTANCE`, and a strictly lower `I-POSITION` selection rate, than the declared-horizon arm. A test that only asks whether more than one move type is used passes on the very failure it is meant to catch, because the failure spreads mass across seven reactive moves. **The criterion is two-sided:** the declared-horizon arm **MUST NOT** exceed a declared `I-POSITION` ceiling either, because lengthening the horizon until agents optimise their way into differentiation is the opposite failure and M4.D.1a forbids it. | `test_m11c16_repertoire_does_not_collapse` | C | M4.D.6b's horizon — setting it to one tick **MUST** turn this red, and setting it to the run length **MUST** turn the ceiling half red |
| **M11.C.15** | Death destabilises exactly as recovery does — a stabilising arrangement built on one member's impairment breaks on their death as well as their recovery | `test_m11c15_death_destabilises_like_recovery` | D | M7.D.3 |
| **M11.C.17** | **Differentiation is not self-generating.** Two arms, identical seeds, differing only in the presence of an external agent. The coach-free arm **MUST NOT** show upward drift in mean `basic_level` over the run. A population of agents that optimises its way to differentiation has contradicted multigenerational transmission, in which levels are roughly conserved and differentiation is rare. This is the model's strongest negative test. **Two amendments:** the flat arm **MUST** be flat against `M7.E.4`'s default deterioration, not against nothing; and the coached arm **MUST** show a **relay**, not a single mover — "**leadership shifts back and forth**", the second mover following after "**several weeks**" (`theory/family_evaluation/fe11.md` · FE11.6) | `test_m11c17_no_basic_level_drift_without_coach` | C | M1.E.7's gate — letting `systems_perspective` rise spontaneously **MUST** turn this red |
| **M11.C.18** | **The estimator discriminates the binder from the real thing.** A `CUTOFF`-heavy arm and an `I-POSITION`-heavy arm brought to the **same** `functional_level` **MUST** diverge in estimated `basic_level` once nodal events enter the window. If they do not, M1.A.4a is not discriminating and the estimator has failed. | `test_m11c18_estimator_separates_cutoff_from_iposition` | C | M1.A.4b's load weighting — flattening it to an unweighted count **MUST** turn this red |
| **M11.C.24** | **The whisper of nature: dissolve a discrepant pair and the discrepancy reverses.** Construct a dominant-adaptive pair at matched `basic_level` (M2.A.0e), run until functional levels diverge, then dissolve the tie. The underfunctioner's `functional_level` **MUST** rise and the overfunctioner's **MUST** fall, converging toward their common basic level. If the model cannot produce this, `M6.I.4`'s pseudo-self transfer is not doing its work. The initial gap must be inside M2.A.0e's tolerance so the **developed** divergence dominates it → `kerr_book/ks07.md` · KS07.2 | `test_m11c24_whisper_of_nature_reverses_on_dissolution` | D | M6.I.4's transfer — zeroing it **MUST** turn this red |
| **M11.C.20** | **The estimator separates a borrowed gain from a basic one.** Two arms brought to the **same** `functional_level`, one by a supporting external condition and one by a completed differentiating sequence; then **withdraw the supporting condition**. Estimated `basic_level` **MUST** diverge. The corpus contains **eight** instances of this discriminator (`kerr_book/_KS_PASS2.md` §KS-D), including a within-subject control on one nodal event type, and one already-worked fixture in which the source labels which change was functional and which was basic | `test_m11c20_estimator_separates_borrowed_from_basic` | C | M1.A.4b's load weighting — flattening it to an unweighted count **MUST** turn this red |
| **M11.C.21** | **Reciprocity inverts when the overfunctioners' capacity falls.** An arm in which both projecting parents' functional level drops **MUST** show the projection target's functional level **rise**, with no change to the target. Two instances: a schizophrenic brother becoming the family's functional member when both parents collapsed, and a projection target's symptoms fading after both parents died | `test_m11c21_reciprocity_inverts_on_projector_collapse` | D | M6.I.4's transfer direction — making it one-way **MUST** turn this red |
| **M11.C.22** | **Symptom lock-in, and its turn.** After a symptom establishes, family anxiety **MUST** fall and the configuration **MUST** resist the symptom's removal — so that a cure arm shows rising conflict (`M11.C.12`) and a removal arm shows family relief, from the same mechanism. **Both limbs MUST be asserted** (M7.D.2d): a mild-symptom arm resists removal, and a severe-symptom arm past the turn **destabilises** and shows relief on removal. Testing only the rising limb passes a monotone implementation | `test_m11c22_symptom_lock_in_is_non_monotone` | D | M7.D.2a's stabilising term, **and** M7.D.2d's turn — flattening the curve **MUST** turn this red |
| **M11.C.23** | **Expressed emotion dominates medication.** Two crossed arms over hostility / over-involvement / critical comments and a symptom-suppression term: **high-EE-with-suppression MUST relapse more than low-EE-without-suppression.** All three EE variables are computable from the event log — over-involvement from `M1.B.8`, critical comments from `M1.A.18a`'s blame readout. → `kerr_book/ks25.md` · KS25.9 `[#]` | `test_m11c23_expressed_emotion_dominates_suppression` | D | the EE term — zeroing it **MUST** turn this red |
| **M11.C.19** | **The two counterfeits are distinguishable by axis.** A forceful-declarer arm and a compliant-accommodator arm at **equal overall counterfeit magnitude MUST** be separated by which axis of M1.A.9a failed. A scalar detector cannot pass this. | `test_m11c19_counterfeit_axis_is_identified` | C | M5.F.2b — collapsing the two axes to their mean **MUST** turn this red |
| **M11.C.25** | **Pole assignment is independent of sex.** Over an ensemble, the frequency with which each sex occupies `M1.B.5`'s dominant pole **MUST NOT** differ beyond sampling error. "**males and females assume the dominant position with equal frequency**" → `theory/family_evaluation/fe07.md` · FE07.3 | `test_m11c25_dominant_pole_independent_of_sex` | C | M2.A.0g — introducing any sex term into the assignment rule **MUST** turn this red |
| **M11.C.26** | **The active sink is recoverable from the belief configuration, and the configuration from the sink.** Three arms driven to conflict, spouse dysfunction and child projection respectively **MUST** produce the three distinct attribution patterns of `M9.6` — *each says the other* / *both say the same one* / *both say the children* — and a classifier reading only the belief store **MUST** recover the sink. Assertable in both directions | `test_m11c26_sink_recoverable_from_belief_configuration` | D | M9.6's attribution write — randomising it **MUST** turn this red |
| **M11.C.27** | **The twosome 2×2, all four cells.** A **stable** twosome **MUST** be destabilised by adding a third *and* by removing one; an **unstable** twosome **MUST** be stabilised by adding a third *and* by removing one. `M11.C.3` tests one cell, and the sign depends on the twosome's prior state, so a model tuned to that cell alone gets the other three wrong → `theory/family_evaluation/fe06.md` · FE06.1 | `test_m11c27_twosome_two_by_two_all_cells` | C | the prior-state term — making the effect sign-invariant **MUST** turn this red |
| **M11.C.28** | **Sink mobility is protective, so symptom count is not a differentiation readout.** Two families at equal adaptiveness and equal stress, differing only in whether sink allocation is **fixed** or **rotates**: the fixed arm **MUST** produce fewer, more severe outcomes and the rotating arm more numerous, milder ones. A readout that ranks the rotating family as worse because it counted symptoms has failed → `theory/family_evaluation/fe07.md` · FE07.6, `fe10.md` · FE10.3 | `test_m11c28_sink_mobility_is_protective` | D | the allocation-mobility term — pinning it **MUST** turn this red |
| **M11.C.29** | **Relief and differentiation are distinguishable by the third person's time course.** Both a genuine `I-POSITION` and distance-in-disguise produce a symptom in a third party; nothing in M11.C currently separates them. The discriminator is duration: **transient** after a genuine move, **persistent** when the move was distance wearing its clothes. Relief **reallocates** the budget; differentiation **reduces** it → `theory/family_evaluation/fe05.md` · FE05.2 | `test_m11c29_relief_and_differentiation_differ_in_time_course` | C | M6.I.1's budget reduction — making the differentiating move merely reallocate **MUST** turn this red |
| **M11.C.30** | **Carrying one symptom protects against a second.** An arm with established load in one channel **MUST** show lower onset hazard in the other two than a matched arm with no load, at equal anxiety → `theory/family_evaluation/fe01.md` · FE01.6, `fe05.md` · FE05.8 | `test_m11c30_channels_are_mutually_protective` | D | M7.D.2c's cross-channel term — zeroing it **MUST** turn this red |
| **M11.C.31** | **Removing a binder raises anxiety with no stressor at all.** A `binder_unavailable` event (M1.F.9) delivered into a calm family **MUST** raise family anxiety while adding **zero** exogenous load, and the rise **MUST** equal the anxiety the binder was holding, to M6.I.1's tolerance | `test_m11c31_binder_removal_releases_bound_anxiety` | D | M1.F.9's return-to-budget step — discarding the held anxiety instead **MUST** turn this red |
| **M11.C.32** | **The mover's anger degrades the move.** Two arms, identical seeds, differing only in the mover's own anger at emission: the angry arm **MUST NOT** reach `PEAK` — it stalls, it does not abort loudly — and its `I-POSITION` **MUST** execute as `M5.F.4`'s assertion form, raising reactivity on the tie. An implementation in which anger *admits* the peak has the corrected `M5.D.4` inverted | `test_m11c32_mover_anger_stalls_and_degrades` | C | M5.D.4's gate — inverting it back **MUST** turn this red |
| **M11.C.33** | **The dependence gate binds the slow clock.** A financially dependent agent completing an unbounded number of exchanges **MUST NOT** show any rise in `basic_level`, while an otherwise identical independent agent does → `theory/family_evaluation/fe04.md` · FE04.9 | `test_m11c33_dependence_gate_blocks_basic_level_rise` | D | M7.A.1a — removing the slow-clock gate **MUST** turn this red |

### M11.D — Engineering criteria

| ID | Criterion | Test |
|---|---|---|
| **M11.D.1** | Engine purity: no file I/O and no UI in the engine package | `test_m11d1_engine_has_no_io` |
| **M11.D.2** | No magic literals: every numeric constant resolves to config or a `basic_level` derivation | `test_m11d2_no_magic_literals_in_engine` |
| **M11.D.3** | Config strictness: an unknown key or malformed line raises | `test_m11d3_config_rejects_unknown_key` |
| **M11.D.4** | Every `[I]` constant is labelled and none is described as sourced | `test_m11d4_invented_constants_labelled` |
| **M11.D.5** | Determinism: two runs at the same seed produce byte-identical event logs | `test_m11d5_same_seed_same_log` |
| **M11.D.6** | Dirty state: a second run in the same process sees no state from the first | `test_m11d6_second_run_is_clean` |
| **M11.D.7** | Every test constructs objects with explicit temporary paths; a conftest guard fails the run if a real artifact path is resolved | `test_m11d7_no_production_paths_in_tests` |
| **M11.D.8** | Spec traceability: every `Spec:` reference in a docstring resolves to an ID in this document, asserted by **exact count** | `test_m11d8_spec_references_resolve` |

**M11.D.9** M11.D.8 **MUST** assert an exact count, not a floor. A floor assertion would not notice the scan silently narrowing.

**M11.D.10** Every requirement ID in this document **MUST** be machine-extractable by a single pattern — a bolded bare token, with any explanatory phrase placed *outside* the bold. IDs **MUST** be fully qualified at their point of definition: an invariant is `M6.I.1`, not `I1`; a criterion is `M11.C.4`, not `C4`. *This requirement exists because the first draft of this document defined invariants and criteria in short form while referring to them in long form, which would have made M11.D.8's scan under-count silently — the exact failure M11.D.9 guards against, in the guard's own source.*

### M11.E — Criteria that cannot be made code-testable in Phases B–D

Flagged explicitly, with a human-review proposal, per the project's planning rule.

| ID | Why not | Proposal |
|---|---|---|
| M11.C.8 | Requires the Phase E ensemble runner and published incidence data; also requires deciding which sources are usable, which is editorial | Defer to Phase E. **Human review** at Phase E entry: confirm the chosen sources are genuinely exogenous, per proposal §5.3 |
| M11.C.9 | The effect size is `[I]`, so the test can only assert *a detectable difference*, which is close to unfalsifiable | Keep as a smoke test. **Human review:** decide at Phase D whether an unfalsifiable criterion is worth keeping in the suite at all, or whether it should be demoted to a readout |
| M11.C.11 shape | "Three phases, not a step down" is a claim about curve *shape*; any automated shape test embeds an invented tolerance | Automate the direction (residual > pre-event baseline; a local relief between) and **human-review the curve** at Phase D against `_LEDGER.md` L08.4's timings |
| M5.F.2 | That a counterfeit move produces the *opposite sign* is testable, but the threshold separating counterfeit from genuine is `[I]` and sets the result | Test the sign flip across the threshold; **human review** the threshold's plausibility. Record it as the model's most consequential invented constant |

---

## M11.F — Framing that must travel with any output

**M11.F.1** No output **MUST** be presented as an implementation of Bowen's authority. His stated goal was "an **open theoretical system**, where the basis for new knowledge is **research and science rather than anything I said**", and he named discipleship as the failure that closes a system off from science — "the more people that treat it like that, the more **my theory will perish** as being a dogma." The model is a thing to be tested. → `kb/kb09.md` · K09.1

**M11.F.2** Any claim that the model's family level informs a societal level **MUST** be stated as **analogy, not derivation**, in this form: *the same principles apply at different levels; the levels are not the same system; reasoning from one to another is an analogy.* Three independent statements — "an analogy is **not an extension of theory**"; the triangle in society is "an analogy, but **not a reasonable connection — these things don't connect up**"; society is "**similar to a family. Not the same as, but the same principles apply.**" → `kb/_KB_PASS2.md`

**M11.F.3** The objection to the model's foundation **MUST** be stated accurately and not softened. Bowen rejected **general systems theory** as a base in favour of **natural systems**, on the ground that general systems "came out of man's head, along with mathematics" — and separately called the two "**compatible within limits**". A simulation is a general-systems artifact, so this is a real named objection. Both of his statements **MUST** be recorded; neither collapses into the other. → `kb/kb13.md` · K13.2, `kb/kb09.md` · K09.3

**M11.F.3a** — *the distinction that answers the objection, and it **MUST** accompany it.* **General Systems Theory is an abstract conceptualisation thought up by man; Bowen theory is built from observation of how families actually function** — as one derives planetary motion by observing the planets. **The simulation is not offering a non-objective explanation of human functioning**; it is a computation over observed regularities. → user addition, C2 approval 2026-08-24
Kerr's framing supports it: Bowen's stated validity criterion was that a theory be "**synonymous with the universe, the earth, the tides, the seasons, the predictable cycles of life**" (`kerr_book/ks24.md` · KS24.8), and the claim is continuity with the Baconian programme rather than with cybernetics (`ks24.md` · KS24.9).

**M11.F.6** — *the general two-sidedness rule.* **No detector, readout or modulation in this model may be one-sided.** Established across four independent instances: blame **and** praise are both losses (`kb/kb07.md` · K07.1); selfish **and** selfless are both counterfeits (M1.A.9a); too much distance **and** too much closeness both trigger the stress response (`kerr_book/ks03.md` · KS03.2); favourable **and** unfavourable societal input are both damped (M1.D.7i). Further instances: over- **and** under-reactivity are both faults (`ks11.md` · KS11.11); blame of **other** and of **self** both count (`ks00.md` · KS00.2); **both** tails of courtship length are informative (M1.A.4e).
**Every naive one-sided proxy for differentiation in this model would be wrong**, and the corpus supplies at least nine readout traps of exactly that shape — overt emotionality, absence of adolescent rebellion, an apparently fine marriage, the idealised child, type A and type B, the high-functioning sibling, conflict rising during recovery, geographic distance, and reported closeness.

**M11.F.4** No output **MUST** claim to reproduce *measured* differentiation (M10.C.2b), and no emotional-state-to-physical-disease mechanism **MUST** be implemented on the basis of the cancer material in `kb/kb11.md` — single anecdotal cases, one with the diagnosis contested by the treating oncologists.

**M11.F.7** — *the model has no opinion on trauma.* No output **MUST** be presented as adjudicating **discrete traumatic events against ongoing relational process**. The source states a position — "the child's life course is more influenced by the lack of emotional separation… **than by the abuse itself**"; "events are not the process" — offered in 1988 with no series, no comparison group and no measurement, on a question where the wider evidence base has moved considerably. **Nothing in this model requires it and nothing implements it.** The model does not distinguish the two, so it **MUST NOT** be read, quoted or reported as having weighed them. → `theory/family_evaluation/fe07.md` · FE07.21 `[K]` `[X]`

**M11.F.8** — *societal readouts **MUST NOT** inherit an editorial symptom list.* `M1.D.7`'s readouts **MUST NOT** score any of the 1988 societal symptom list as evidence of regression, and specifically **MUST NOT** score claims of **rights**. "an incessant clamor for '**rights**'" is an editorial judgement of its period, not an observation, and a readout that treats rights claims as regression is one-sided by construction and fails `M11.F.6`. → `theory/family_evaluation/fe10.md` · FE10.17 `[K]` `[X]`

---

## M11.G — The family-evaluation readout

**M11.G.1** — *the model **MUST** emit a structured family evaluation, and its schema is given.* The project has had acceptance criteria but no **readout schema**; `FE10.1` supplies a ten-component family diagnosis, and **components 1–8 MUST** be emitted. Every one is computable from state the model already carries.

| # | Component | Computed from |
|---|---|---|
| 1 | Symptom — who carries it, in which channel, at what severity | M7.D, M1.A.11 |
| 2 | The nuclear family's binding pattern — which of the three sinks is active, in what proportion | M1.D.1, M9.6 |
| 3 | Intensity of the family emotional process | M1.D's budget occupancy |
| 4 | The multigenerational picture — level by generation, and the projection line | M7.A, M11.C.6 |
| 5 | Degree of emotional cutoff with the families of origin | M1.B.3 bond energy on family-of-origin ties |
| 6 | Stress currently on the system, exogenous and endogenous separately | M1.F `exogenous` |
| 7 | Adaptiveness — **derived, not measured** (M11.G.2) | reactivity against stress |
| 8 | The extended family and social network available to the system | M8's live positions, the access vector |

**M11.G.2** — *adaptiveness **MUST** be derived as reactivity **compared against** stress, never reported as a level.* A family showing little reactivity under heavy stress and a family showing little reactivity under none are not the same family, and a readout that reports reactivity alone cannot tell them apart. This makes `M1.A.4b`'s load weighting **definitional** rather than a correction applied afterwards. → `theory/family_evaluation/fe10.md` · FE10.2

**M11.G.3** — *components 9 and 10 **MUST NOT** be emitted.* The source's ninth and tenth components are **therapeutic focus** and **prognosis**. Both are clinical judgements about a real family in treatment. This model has no patient, no clinician and no outcome data, and emitting either would be the single most misreadable thing it could produce. → `theory/family_evaluation/_FE_PASS2.md` §3.4

**M11.G.4** The evaluation **MUST** carry `M11.F`'s framing block wherever it is emitted, and **MUST** report belief and ground truth separately (M9.5). `→E` for the rendering; the schema itself is Phase D.

---

## M12 — Out of scope, and must never be built

**M12.1** The following **MUST NOT** be implemented. Each was proposed and withdrawn on the second reading of the corpus; they are named rather than omitted because a mechanism that is merely absent gets re-invented. → `model_explainer.md` §12.1

- A **permission scalar** gating a dyad's closeness by the excluded third
- **Ally cancellation** — support from a family member cancelling a differentiating move
- A **durable projection target queue** — Bowen retracts this himself and dates the error
- **Distance as a depletable relief stock**
- **"Content is noise"** as an argument for anything

**M12.2** Two things Bowen states he does **not** know **MUST** be labelled `[I]` wherever the model decides them: what determines whether a problem stays in the spouse dyad or transmits to a child, and what selects which spouse takes the dominant pole at identical levels. → §12.2

**M12.2a** — *reviewed at revision 4; **M12.2 stands in full**.* Kerr 1988 was read as supplying a mechanism for the first of the two: the parents' **emotional complementarity**, formed in their families of origin and exaggerated by anxiety — "**The nature of these exaggerated elements determines whether the problem emerges as marital conflict, spouse dysfunction, or child dysfunction.**" On review this **names the locus, not the rule**: it says *where* the answer lives without saying what it is, and its own hedge is "determined **largely by**". The sink allocation therefore remains `[I]`. `M9.6` is the consequence that does follow — the allocation is **identifiable** from the belief configuration, which is a readout, not a determinant. For the second unknown the source gives only "by mutual agreement, the product of the emotional fit", which is the same shape. → *decision A4, option (b), 2026-08-27*; `theory/family_evaluation/fe07.md` · FE07.3

**M12.3** There **MUST NOT** be an anniversary effect. No date-anchored recurrence is named anywhere in the corpus. → §12.3

**M12.4** The `M` metabolic column **MUST NOT** be reintroduced in any form. → M1.A.15

**M12.5** — *the only prohibition in this section stated **by the author** rather than inferred by the project.* `power` and `punishment` **MUST NOT** exist as mechanisms, and **no move MUST be represented as one agent acting *against* another**. Bowen, in his own written prose:
> "**there is no such thing as one person taking action against another. The issue of 'power' or 'punishing' another person does not apply with the concept of differentiation of self.**"

This binds three places that could each be implemented adversarially and where nothing currently forbids it: `M1.B.5`'s dominant pole is a **reciprocal** position, not a victory; `M5.E.1`'s consequence rung is the system's **automatic** reaction, not a punishment chosen by anyone; and `M8.6`'s alignment penalty is a **structural** cost of a peripheral triangle, not a sanction. A readout describing any of the three in adversarial terms is a failing implementation. → *decision C, 2026-08-27*; `theory/family_evaluation/fe11.md` · FE11.7 `[B]`

---

## M13 — Implementation order

| Phase | Builds | Done when |
|---|---|---|
| **B** | M1 objects; M3 clocks and update order; M4.A standing load **including M4.A.5**; M4.B, M4.E and M4.G; M1.F event record **including M1.F.9**; M8 the live-position predicate; `ScriptedSource`. **No policy** — a fixed script drives it. Reduced instance per M2.3. | A scripted 40-week trace runs, the event log reads correctly, and a `TRIGGER` on a dormant family-of-origin tie moves anxiety with no contact. M11.D.1, M11.D.3, M11.D.5, M11.D.6 and M11.D.7 pass. |
| **C** | M4.C appraisal **including M4.C.8a**; M4.D policy; M5 the full repertoire, gates and the `I-POSITION` state machine **including `PREPARE` (M5.D.2a) and the corrected anger gate (M5.D.4)**; M6 invariants M6.I.1–M6.I.8. | M11.C.1, M11.C.3, M11.C.4, M11.C.5, M11.C.7, M11.C.13, M11.C.14, M11.C.16, **M11.C.25, M11.C.27, M11.C.29 and M11.C.32** pass over 1,000-seed ensembles, each mutation-proved. M11.D.2, M11.D.4 and M11.D.8 pass. |
| **D** | M7 the slow clock **including M7.A.1a, M7.D.2c/2d and M7.E.4**; M9 beliefs **as a channel (M9.6, M9.7)**; the twelve-person reference family; the three symptom channels and endogenous events; **M11.G's readout schema**. | M11.C.2, M11.C.6, M11.C.9, M11.C.10, M11.C.11, M11.C.12, M11.C.15, **M11.C.22 (both limbs), M11.C.26, M11.C.28, M11.C.30, M11.C.31 and M11.C.33** pass. A 40-year three-generation run completes. |

**M13.1** M8 **MUST** land in Phase B, before the policy, because four separate Phase C mechanisms call it.

**M13.2** The frozen grid engine and its tests **MUST** stay green throughout. v2 lives in a new package `src/bowen/`.

---

## M14 — Spec coverage

**M14.1** `docs/spec_coverage.md` **MUST** be generated as part of the completion report, listing every ID in this document with `done` / `partial` / `not done`, each `done` carrying the file path and test name that proves it.

**M14.2** "Implementation complete" without that report **MUST NOT** be accepted.

---

### Revision 4 — *Family Evaluation* (Kerr & Bowen, 1988), applied 2026-08-27

The sixth corpus source, folded in against the decisions recorded in `docs/DECISIONS — FAMILY EVALUATION.md`. **One correction, three amendments, and about twenty new requirements.**

| | What changed | Where |
|---|---|---|
| **Corrected** | `M5.D.4` was **inverted**. Ch13 says the mover's *freedom from* anger admits the peak; the spec said anger did. Settled by a re-read of the primary chapter, which mentions anger exactly once | M5.D.4, M5.D.4a, M11.C.32 |
| **Amended** | The channel prior is movable by a relational term and overridable by a constitutional one (A1, option c) | M1.A.11b, M1.A.11c |
| **Amended** | The transition at 50 keeps its licence implementation and gains an awareness **readout**; neither is reduced cognitive capacity (A2) | M1.A.3b |
| **Amended** | The dependence gate now binds the **slow** clock, not only the `I-POSITION` move (B1 tail) | M7.A.1a, M11.C.33 |
| **Reviewed, unchanged** | `M12.2` stands in full — Kerr 1988 names the locus of the first unknown, not the rule (A4, option b) | M12.2a |
| **New mechanisms** | Belief becomes a channel; cross-person reinforcement; a self-generated stress term; mutually protective channels; non-monotone lock-in; an objectivity gate on amplification; a `PREPARE` phase; a binder-unavailable event; emotional reserve; default tie deterioration | M9.6–M9.7, M4.D.6e, M4.A.5, M7.D.2c–2d, M4.C.8a, M5.D.2a, M1.F.9, M4.D.5d, M7.E.4 |
| **New constraints** | Pole assignment independent of sex; non-monotone coach contact; every criterion needs a stressor | M2.A.0g, M1.E.8, M11.3 |
| **New readout** | The eight-component family evaluation, with components 9 and 10 explicitly barred | M11.G |
| **New magnitudes** | Twelve stated bounds and shapes, admitted as **checks** and never as parameters | M10.C.4, M10.C.4a, M10.C.3a |
| **New prohibitions** | Power and punishment as mechanisms — the only `M12` entry stated by the author; no adjudication of trauma against process; no rights-as-regression readout | M12.5, M11.F.7, M11.F.8 |

**One item remains open and is not applied:** nothing. All nine decision items are resolved.

---

*Written against `model_explainer.md` v1.2 and `theory/_LEDGER.md` across six corpus sources. Every requirement traces to an explainer section or a ledger entry; the ledger traces to a chapter, a tape, an interview or a page. Nothing in this document is sourced to a summary.*
