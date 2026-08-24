---
tags: [model-bt, spec]
version: 2.0-draft
status: FOR APPROVAL — no code until approved
date: 2026-08-22
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

**M1.A.4** `basic_level` **MUST NOT** be frozen at marriage. It **MUST** be slow-moving, with a ratchet that advances only on a *completed* differentiating exchange (M5.D.7). → §13.3

**M1.A.5** A `Person` **MUST** hold a `functional_level` distinct from `basic_level`, and its variance **MUST** be a decreasing function of `basic_level`. → §3.2

**M1.A.6** Symptom thresholds **MUST** be evaluated against `functional_level`, never against `basic_level`. → §3.2

**M1.A.7** A `Person` **MUST** hold `chronic_anxiety`, fixed in childhood from that person's own **witnessed** event history, and it **MUST** act as a floor below which acute anxiety cannot decay. It **MUST NOT** be computed from a family average. → §3.3, §9.4

**M1.A.8** A `Person` **MUST** hold `acute_anxiety`, updated per event and decaying toward the chronic floor.

**M1.A.9** A `Person` **MUST** hold `outside_ness`, distinct from `basic_level`, with three inputs at three time scales (M5.F). → §3.6

**M1.A.10** A `Person` **MUST** hold a `life_energy` allocation, zero-sum between relationship-seeking and goal-directed activity, whose ratio is a function of `basic_level`. → §3.7

**M1.A.11** A `Person` **MUST** hold `symptom_load` over exactly three channels — physical, emotional, social — which **MUST** be substitutable (M7.D.1–M7.D.4). → §3.9

**M1.A.12** A `Person` **MUST** hold `involvement_weight`, recomputed every fast tick. Family membership **MUST** be derived as a threshold over it and **MUST NOT** be a stored set. → §3.8

**M1.A.13** A `Person` **MUST** hold `structural_importance` on exactly **three** tiers. A finer ranking **MUST NOT** be implemented. → §3.10

**M1.A.14** A `Person` **MUST** hold `sibling_position` as static profile data affecting the propensity vector and nothing else. → §3.11

**M1.A.15** A `Person` **MUST** hold `financially_dependent: bool`, a hard gate on one move (M5.C). There **MUST NOT** be a per-person material stock, resource pool, or any quantity whose depletion causes death. → §3.12

**M1.A.16** A `Person` **MUST** hold `beliefs` (M9), which **MUST** be permitted to differ from ground truth.

**M1.A.17** A `Person` **MUST** hold `role ∈ {MEMBER, EXTERNAL}`. `EXTERNAL` is not a separate type. → §2.5

### M1.B `Relationship`

**M1.B.1** A `Relationship` **MUST** connect exactly two `Person`s and **MUST** be the unit on which coupling state is stored. Per-family scalars **MUST NOT** stand in for a specific tie.

**M1.B.2** `conductance` **MUST** be undirected. It **MUST NOT** be a function of physical distance, contact frequency, or elapsed time since last interaction. → §4.1, §4.3

**M1.B.3** `bond_energy` **MUST** distinguish four tie states that differ in events and energy independently: cut off (no events, high), distant (few, moderate), resolved low-contact (few, **low**), open conflict (many, high). Failing this discrimination is a failing implementation. → §4.2

**M1.B.4** `bond_energy` decay **MUST** be at or near zero. Reunion after separation **MUST** restore full coupling with **zero re-activation latency**. → §4.3

**M1.B.5** `functioning_balance` **MUST** be directed, bistable with no stable midpoint, and indexed **per area of joint activity**. It **MUST NOT** be a single linear scalar and **MUST NOT** be a fixed personal trait. → §4.4

**M1.B.6** The pole **MUST** flip on a *relative* comparison — the under-functioning party's self-assertion exceeding the other's domination — not on an absolute threshold; the flip **MUST** be immediate; and it **MUST** revert unless sustained through the counter-reaction. → §4.4

**M1.B.7** Reversal cost **MUST** be asymmetric: reducing a marked over-functioner **MUST** cost less than raising a marked under-functioner. → §4.4

**M1.B.8** `investment` **MUST** be directed and **valence-blind** — share of thought occupied by the target. It **MUST NOT** be derived from warmth, agreement, or tie quality. Conflict-laden preoccupation **MUST** register as *high* investment. → §4.5

**M1.B.9** A `Relationship` **MUST** hold `areas_of_joint_activity`; narrowing them **MUST** operate on the same variable as the functioning balance, not on a separate distance scalar. → §4.4

**M1.B.10** A `Relationship` **MUST** hold `taboo_set`, which **MUST** grow monotonically. Subjects **MUST NOT** be returned to a tie. → §4.6

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

**M1.D.2a** Distance **MUST** nevertheless **absorb anxiety**. Bowen enumerates four ways of *absorbing* anxiety with distance first among them, and three that *end in symptoms* with distance excluded — two questions, two answers, both true. **Distance absorbs without symptomising**, which is why a family can discharge into it indefinitely and read as untroubled, and why `M11.C.13`'s relocation test must score location as well as count. → `kb/kb04.md` · K04.1

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

### M1.F `Event`

**M1.F.1** An `Event` **MUST** carry: sender, targets, witnesses, move type, intensity, timestamp, `duration`, `exogenous`, `source_position`, `route`, `fidelity`.

**M1.F.2** `source_position` **MUST** be able to change the *sign* of an event's effect, not only its magnitude. → §9.3

**M1.F.3** `route` **MUST** modulate gain: a direct dyad amplifies; routing through a neutral third damps. → §9.3

**M1.F.4** `fidelity` **MUST** degrade per private hop. → §9.3

**M1.F.5** Witnesses **MUST** appraise events not addressed to them, and that accumulated witnessing **MUST** be the source of `chronic_anxiety` (M1.A.7). → §9.4

**M1.F.6** Statistical inputs **MUST** generate spells with a start and a duration. Per-tick probability draws **MUST NOT** be used for any stressor. → §9.1

**M1.F.7** Endogenous events **MUST NOT** be sampled from incidence data. The `exogenous` flag **MUST** keep the two countable separately. → §9.2

**M1.F.8** Simultaneous events **MUST** be batched so that order within a tick cannot decide the outcome.

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

**M2.A.0c** Spouses **MUST** be paired at comparable `basic_level`. "People choose spouses [at] **almost identical levels of differentiation of self**", offered as an observed constant and "par for the course". → `kb/kb14.md` · K14.2

**M2.A.0d** Fusion **MUST** be life-stage dependent, not uniformly costly. The infant–caretaker symbiosis is stated as a **normal state, not a pathology**. → `kb/kb14.md` · K14.5

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

### M4.B Perceive

**M4.B.1** A person **MUST** read events addressed to it *and* events it witnessed (M1.F.5).

### M4.C Appraise

**M4.C.1** Each incoming event **MUST** raise acute anxiety by `intensity × conductance / functional_level`, modulated by `route` (M1.F.3) and `source_position` (M1.F.2), and attenuated by `fidelity` (M1.F.4).

**M4.C.2** Appraisal **MUST** apply a gain function on anxiety: above threshold, event content **MUST** be defended against rather than absorbed. A plain product is a failing implementation. → §3.4

**M4.C.3** Symptom onset **MUST** be driven by an **integrator over chronicity**, not by a test on instantaneous anxiety. → §3.3

### M4.D Select

**M4.D.1** Each person **MUST** select exactly one move per fast tick, by softmax over propensity scores.

**M4.D.2** The propensity score **MUST** be a function of acute anxiety, `functional_level`, the state of the tie in question, the person's position in the active triangle, and their learned repertoire.

**M4.D.3** Rising anxiety **MUST** raise the weight on the seven reactive moves.

**M4.D.4** Propensity for `I-POSITION` **MUST NOT** be monotonically increasing in `basic_level`. An implementation in which the highest-`basic_level` member reliably moves first is a failing implementation. → §5.2

**M4.D.5** The default state **MUST** be the fused default — each altering self to manage the other's functioning while demanding the other change — and **MUST NOT** be modelled as an absence of a move. → §5.2

**M4.D.6** Moves that worked before **MUST** be reinforced, so that a family develops a characteristic style.

**M4.D.6a** — *the reinforcement signal MUST be named, and it MUST NOT be short-horizon anxiety relief.* The obvious proxy — the post-move change in the actor's own acute anxiety — is **forbidden**, because this specification pins the timing that makes it self-defeating: M11.C.4 requires `CUTOFF` to drop the actor's acute anxiety *immediately* with its cost deferred to the next nodal event, and M11.C.5 requires `I-POSITION` to *raise* tension for a bounded window before it settles. A short-horizon anxiety proxy therefore rewards the seven reactive moves and punishes the two differentiating ones **by construction**, and every agent converges on `CUTOFF`.

**M4.D.6b** The signal **MUST** be evaluated over a horizon longer than the reaction window of M5.D, and **MUST** be declared in config with its horizon. `[I]`

**M4.D.6c** This failure would be invisible to M11.C.1–M11.C.15, because all of them hold the policy fixed across both arms (M0.4) and a uniformly degenerate policy shifts both arms together. M11.C.16 exists to catch it.

### M4.E Act

**M4.E.1** A selected move **MUST** become an event with sender, targets, witnesses, intensity, timestamp, route, fidelity and the tie's latency.

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

**M5.B.3a** `REDUCE_CUTOFF` **MUST** have a **floor as well as no ceiling**. Cutoff is load-bearing for the person maintaining it — "cutting off from people is your **lifeline** which enables you to live and adjust" — so it may be reduced only as fast as that person can absorb, and the far side of a cutoff has its own state and its own willingness. Total openness is also not the target: "to maintain a self, there has to be **some self that self does not communicate** to the other", and complete disclosure is "a **de-selfing** kind of thing". With `L12.4`'s two-sided closeness band, four sources agree that **contact is not monotonically good**. → `kb/kb05.md` · K05.2, `kb/kb08.md` · K08.4

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

**M5.D.2** States, in order: `DEFINE` → `OPPOSITION` → (`ABORT` | `HOLD`) → `PEAK` → `RESOLVE` → `FOLLOW_UP`.

**M5.D.3** The abort branches — defend, counterattack, go silent — **MUST** sit at the *first* opposition and **MUST** be the usual outcome, not an exception. Each **MUST** return the mover to the prior balance.

**M5.D.4** Anger **MUST** be the *gate* admitting the sequence to `PEAK`, not a fourth abort branch.

**M5.D.5** On a held `PEAK`, the opposition **MUST** pull up **to the mover's level**, not to a group mean, and the payoff **MUST** propagate to subsequent movers.

**M5.D.6** `FOLLOW_UP` **MUST** be mandatory on the next day *relative to the resolving encounter*. Skipping it **MUST** revert the gain.

**M5.D.7** Only a sequence reaching `FOLLOW_UP` counts as a *completed* exchange for the `basic_level` ratchet (M1.A.4) and the triangle decrement (M1.C.5).

**M5.D.7a** The ratchet increment **MUST** be small enough that no realistic number of completed exchanges moves an agent materially up the scale within a run, and the config **MUST** carry a comment saying why. Bowen names the contrary reading as the field's characteristic misconception — of someone who "went home to see their parents over the weekend and differentiated" he says it is "**grotesque** … they think of differentiation is something **you do in an hour a weekend**." Ch21's trip is **one step in a decades-long effort**, not the unit of differentiation.

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

### M5.F Act identity — the counterfeit problem

**M5.F.1** A move's effect **MUST NOT** be a function of move type and tie state alone. It **MUST** be multiplied by a hidden actor state (`outside_ness`) that receivers can read and the actor may not. → §5.6

**M5.F.2** The same move type at low `outside_ness` **MUST** be able to produce the *opposite* sign of effect, not merely a smaller one.

**M5.F.2a** — *asserting a differentiated state MUST be negative evidence for it.* Five independent forms across five interviews: declaring non-involvement "**in itself is an indicator of involvement**"; "I'm out of it, you handle it" is "**mostly denial, because they are in it**"; "the more the individual has to say **I've worked it out**, is evidence of an attachment"; a low-level self holds a **selfish, dogmatic, forceful** I-position; and catching oneself **diagnosing, criticising or praising** is a loss of perspective. → `kb/_KB_PASS2.md`
**Consequence:** a cheap readout computable from the event log, and the discriminator M5.F.3 asks for. An agent that announces its own neutrality **MUST** score lower on `outside_ness`, not higher.

**M5.F.3** Concession **MUST** be gradable continuously by the receiver, not as a binary. → §5.6

---

## M6 — Invariants

Asserted at the end of every fast tick (M4.G.2). A violation **MUST** raise, not warn.

| ID | Invariant | Explainer |
|---|---|---|
| **M6.I.1** | The family undifferentiation budget has exactly **three** sinks and is conserved across them | §7 I1 |
| **M6.I.2** | Emotional distance runs **outside** the budget as an always-on baseline; it **MUST NOT** be a fourth sink | §7 I2 |
| **M6.I.3** | `life_energy` is zero-sum per person between relationship-seeking and goal-directed activity | §7 I3 |
| **M6.I.4** | Dyadic exchange conserves **pseudo-self**: one spouse's functional gain equals the other's loss | §7 I4 |
| **M6.I.5** | Solid self is **exempt** from fusion and **MUST NOT** participate in M6.I.4 | §7 I5 |
| **M6.I.6** | Anxiety is conserved and redirected, never destroyed; blocking one channel raises flow on the rest | §7 I6 |
| **M6.I.7** | There is no exit from the field; `CUTOFF`, `DISTANCE` and silence are moves *inside* the system | §7 I7 |
| **M6.I.8** | The standing load runs before delivery, every tick, on every tie | §7.2 — *this spec elevates it to an invariant; the explainer describes it as a mechanism* |

**M6.1** M6.I.1 conservation **MUST** be asserted to a stated tolerance, declared in config, and the tolerance **MUST** be `[I]`.

**M6.2** M6.I.6 **MUST** be assertable across a removal event specifically (M11.C.11).

---

## M7 — The slow tick

**M7.A.1** `basic_level` drift **MUST** be slow and **MUST** advance only via the ratchet on a completed differentiating exchange (M5.D.7). Recovery toward baseline when calm **MUST NOT** be a symmetric restoring force.

**M7.A.2** Differentiation gained in a peripheral system **MUST** transfer automatically to the nuclear family. `basic_level` is per-person; intensity is per-relationship. → §10, §15 Ch10/Ch21

**M7.B.1** `chronic_anxiety` **MUST** be fixed once in childhood from witnessed history (M1.A.7). The age is `[I]`.

**M7.C.1** Life stage **MUST** govern childhood, launch, partnering and mortality.

**M7.D.1** Symptom load **MUST** accumulate from routed load and, on crossing threshold, **MUST** emit an **endogenous** event (M1.F.7).

**M7.D.2** The three channels **MUST** be substitutable — the same deficit **MUST** be able to present as any of the three.

**M7.D.3** Curing a symptom without changing the underlying deficit **MUST** raise family tension. → M11.C.12

**M7.D.4** Removal (`INSTITUTIONALIZE`) **MUST** fire off the **remaining members'** tolerance for disturbing behaviour, explicitly decoupled from the symptom-bearer's severity, and **MUST** be suppressible by relocating decision ownership (M5.C.1). → §6.2

**M7.E.1** The generational update **MUST** produce: the primary projection object **lower** than the parents, minimally-involved siblings about the same, those outside the process **better**.

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

---

## M10 — Parameters

**M10.1** Every constant **MUST** appear in the register with a grade (M0.1) and **MUST** resolve either by derivation from `basic_level` or from markdown config (M0.3).

### M10.A Derived from `basic_level` — preferred

**M10.A.1** The following **MUST** be derived, not independently parameterised: reactivity (M1.A.1), `functional_level` variance (M1.A.5), the `life_energy` ratio (M1.A.10), the stabiliser repertoire available to a person, and transfer magnitude in M6.I.4 (which scales *inversely* with basic level).

**M10.A.2** Deriving from `basic_level` is preferred wherever it is defensible, and the source states the principle directly: "a theory is made up of the **least number of pieces that will hang together into a story**, rather than trying to put in all of the others." A model with 60–90 free parameters is already in tension with that, so every added parameter **MUST** be justified against it rather than added by default. → `kb/kb10.md` · K10.9, because the model has 60–90 free parameters against a single invented family and the risk is overfitting to the modeller's intuitions.

### M10.B Config

**M10.B.1** All remaining constants **MUST** live in markdown config. Editorial content — event kinds, the reference family, the access vector, tie declarations, dial ranges — **MUST NOT** live in Python.

**M10.B.2** Config parsing **MUST** fail loudly on an unrecognised key or a malformed line. Silently skipping and falling back to a default **MUST NOT** occur. *(This is a defect in the frozen engine's `_apply_config`; v2 must not inherit it.)*

### M10.C The `[I]` register

**M10.C.1** The following are **invented** and **MUST** be labelled as such wherever they surface: the fast tick length; the softmax temperature and every propensity coefficient; all conductance values; all bond-energy values and its decay rate; the standing-load function; every gate threshold including the `outside_ness` threshold; `societal_leadership`'s functional form; the appraisal gain function's shape; the chronic-anxiety fixation age; sibling-position effect sizes; the M6.I.1 conservation tolerance; every value in M2.

**M10.C.2** The following are **stated in the corpus** and are usable as calibration targets: the durations in `model_explainer.md` §8, the 0–100 scale and the transition at 50, the ~90%-in-the-lower-half population skew, the three-sink count, and — **withdrawn at pass 3** — the eight-to-ten-generation figure, which `KB11` shows was rhetorical: "I put [it] in **not to say that it takes ten generations**."

**M10.C.2a** One **external** calibration target is admitted: the DSI–trait-anxiety association, **r = .64** (Skowron & Friedlander 1998, N = 609). It **MAY** be used to check an ensemble's differentiation–anxiety coupling and **MUST NOT** be used to set a parameter. Two limits **MUST** travel with it in any report: it relates two self-reports, so shared method variance inflates it; and it is a population association in a sample that is 82.7% White and largely northeastern US, not a norm. → `theory/_EXTERNAL_MEASURES.md`

**M10.C.2b** No output **MUST** be described as reproducing *measured* differentiation. The best available instrument reports α ≈ .88 overall, and its Fusion With Others subscale — the construct most central to the theory — required a five-year rebuild after failing to relate to psychological adjustment at all. The model **MUST NOT** claim precision the measurement literature does not have.


**M10.C.3** The following **MUST NOT** be implemented as rates, because they were manufactured from illustrations Bowen explicitly bounded: any per-generation `basic_level` decrement, and any annual societal-regression rate. → §8

---

## M11 — Acceptance criteria

Each criterion **MUST** have a named test, **MUST** assert a direction of difference between two arms (M0.4), and **MUST** be proved failing by mutation before it counts as coverage: delete or invert the mechanism it names, confirm red with the **verbatim** original restored afterwards.

**M11.1** A test **MUST NOT** be counted as coverage until its mutation has actually been run, not reasoned about.

**M11.1a** A mutation **MUST** be a real behaviour change. Where the value under test is written more than once — by a redundant assignment, or through two names bound to the same object — the mutation **MUST** name every write, or it proves nothing while looking exactly like a test that cannot fail. *Measured in the frozen engine: `nuclear_family_id` is bound to the same array as `family_ids`, so mutating one line of the triangle attach is overwritten by the next line and the test stays green.* This is why M1.A gives each field exactly one name.

**M11.2** These are ensemble property tests. A single-run assertion **MUST NOT** be used for any of M11.C.1–M11.C.16.

| ID | Criterion | Test | Phase | Mutation target |
|---|---|---|---|---|
| **M11.C.1** | Differentiation protects: identical families differing only in basic `basic_level`, same stressor schedule → the lower-`basic_level` family reaches symptom threshold sooner in a significant majority of seeds | `test_m11c1_lower_c_reaches_threshold_sooner` | C | the `/ functional_level` divisor in M4.C.1 |
| **M11.C.2** | Symptoms concentrate on the member who received the most projection-type events, not uniformly across children | `test_m11c2_symptom_concentrates_on_projection_target` | D | the witness path, M1.F.5 |
| **M11.C.3** | Triangling relieves the pair and costs the third, within the same tick | `test_m11c3_triangle_relieves_pair_costs_third` | C | `bound_anxiety` transfer, M1.C.1 |
| **M11.C.4** | Cut-off trades now against later: `CUTOFF` drops the actor's acute anxiety immediately and raises family total anxiety at the next nodal event, against a matched no-cutoff arm | `test_m11c4_cutoff_trades_now_against_later` | C | the standing load, M4.A.1 |
| **M11.C.5** | The system pushes back — as **symptom in a named third person**, as a damped oscillation with hysteresis, decaying unless fed, reverting if `FOLLOW_UP` is skipped; absence of reaction means the move did not land | `test_m11c5_change_back_reaction_shape` | C | the life-energy debit, M5.E.7 |
| **M11.C.6** | Transmission is multigenerational: over three generations with no external stressor change, mean `basic_level` in the projection line declines while the non-target line does not | `test_m11c6_multigenerational_decline_in_projection_line` | D | M7.E.1 |
| **M11.C.7** | **Position, not skill.** Hold the coach's parameters fixed and vary who talks to whom in whose presence; the topology arms **MUST** differ | `test_m11c7_topology_not_coach_skill` | C | M8.2/M8.3 |
| **M11.C.8** | Endogenous incidence lands near published rates | — | **`→E`** | — |
| **M11.C.9** | Sibling position shapes functioning at constant `basic_level` | `test_m11c9_sibling_position_shifts_propensity` | D | M1.A.14 |
| **M11.C.10** | Cut-off begets cut-off: a generation containing a cut-off produces more in the next than a matched arm | `test_m11c10_cutoff_begets_cutoff` | D | M7.E.3 |
| **M11.C.11** | Removal produces three phases, not a step down — rise, partial relief with redirected focus, sustained residual; and total anxiety is conserved across the removal to a stated tolerance | `test_m11c11_removal_three_phase_shape` | D | M6.I.6 |
| **M11.C.12** | **Curing a symptom without changing the deficit raises tension.** Remit a spouse's dysfunction leaving the functioning balance intact → marital conflict rises | `test_m11c12_symptom_relief_raises_conflict` | D | M7.D.3 |
| **M11.C.13** | Help relocates incidents; it does not reduce them. Score **count and location** — relocation from community into family carries a positive sign | `test_m11c13_help_relocates_not_reduces` | C | M6.I.6 |
| **M11.C.14** | Management technique has zero independent effect while marital distance is high | `test_m11c14_technique_null_under_marital_distance` | C | the marital-distance gate, M5.C.1 |
| **M11.C.16** | **The learned repertoire does not collapse onto relief-seeking.** Two arms, identical seeds, differing only in the reinforcement horizon: the short-horizon arm (M4.D.6a's forbidden proxy) against the declared-horizon arm (M4.D.6b). The short-horizon arm **MUST** show a strictly higher share of `CUTOFF` and `DISTANCE`, and a strictly lower `I-POSITION` selection rate, than the declared-horizon arm. A test that only asks whether more than one move type is used passes on the very failure it is meant to catch, because the failure spreads mass across seven reactive moves. | `test_m11c16_repertoire_does_not_collapse` | C | M4.D.6b's horizon — setting it to one tick **MUST** turn this red |
| **M11.C.15** | Death destabilises exactly as recovery does — a stabilising arrangement built on one member's impairment breaks on their death as well as their recovery | `test_m11c15_death_destabilises_like_recovery` | D | M7.D.3 |

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

**M11.F.4** No output **MUST** claim to reproduce *measured* differentiation (M10.C.2b), and no emotional-state-to-physical-disease mechanism **MUST** be implemented on the basis of the cancer material in `kb/kb11.md` — single anecdotal cases, one with the diagnosis contested by the treating oncologists.

---

## M12 — Out of scope, and must never be built

**M12.1** The following **MUST NOT** be implemented. Each was proposed and withdrawn on the second reading of the corpus; they are named rather than omitted because a mechanism that is merely absent gets re-invented. → `model_explainer.md` §12.1

- A **permission scalar** gating a dyad's closeness by the excluded third
- **Ally cancellation** — support from a family member cancelling a differentiating move
- A **durable projection target queue** — Bowen retracts this himself and dates the error
- **Distance as a depletable relief stock**
- **"Content is noise"** as an argument for anything

**M12.2** Two things Bowen states he does **not** know **MUST** be labelled `[I]` wherever the model decides them: what determines whether a problem stays in the spouse dyad or transmits to a child, and what selects which spouse takes the dominant pole at identical levels. → §12.2

**M12.3** There **MUST NOT** be an anniversary effect. No date-anchored recurrence is named anywhere in the corpus. → §12.3

**M12.4** The `M` metabolic column **MUST NOT** be reintroduced in any form. → M1.A.15

---

## M13 — Implementation order

| Phase | Builds | Done when |
|---|---|---|
| **B** | M1 objects; M3 clocks and update order; M4.A standing load; M4.B, M4.E and M4.G; M1.F event record; M8 the live-position predicate; `ScriptedSource`. **No policy** — a fixed script drives it. Reduced instance per M2.3. | A scripted 40-week trace runs, the event log reads correctly, and a `TRIGGER` on a dormant family-of-origin tie moves anxiety with no contact. M11.D.1, M11.D.3, M11.D.5, M11.D.6 and M11.D.7 pass. |
| **C** | M4.C appraisal; M4.D policy; M5 the full repertoire, gates and the `I-POSITION` state machine; M6 invariants M6.I.1–M6.I.8. | M11.C.1, M11.C.3, M11.C.4, M11.C.5, M11.C.7, M11.C.13, M11.C.14 and M11.C.16 pass over 1,000-seed ensembles, each mutation-proved. M11.D.2, M11.D.4 and M11.D.8 pass. |
| **D** | M7 the slow clock; M9 beliefs; the twelve-person reference family; the three symptom channels and endogenous events. | M11.C.2, M11.C.6, M11.C.9, M11.C.10, M11.C.11, M11.C.12, M11.C.15 pass. A 40-year three-generation run completes. |

**M13.1** M8 **MUST** land in Phase B, before the policy, because four separate Phase C mechanisms call it.

**M13.2** The frozen grid engine and its tests **MUST** stay green throughout. v2 lives in a new package `src/bowen/`.

---

## M14 — Spec coverage

**M14.1** `docs/spec_coverage.md` **MUST** be generated as part of the completion report, listing every ID in this document with `done` / `partial` / `not done`, each `done` carrying the file path and test name that proves it.

**M14.2** "Implementation complete" without that report **MUST NOT** be accepted.

---

*Written against `model_explainer.md` v1.0 and `theory/_LEDGER.md` at 149 findings. Every requirement traces to an explainer section; the explainer traces to a ledger entry; the ledger traces to a chapter. Nothing in this document is sourced to a summary.*
