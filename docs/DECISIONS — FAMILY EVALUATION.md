---
tags: [model-bt, decisions, family-evaluation]
status: DECIDED 2026-08-27 — all nine items resolved and applied at spec revision 4
date: 2026-08-26 · decided and applied 2026-08-27
---

# Decisions arising from *Family Evaluation* (Kerr & Bowen, 1988)

**A separate file deliberately.** `DECISIONS FOR APPROVAL.md` is yours and you are editing it; this
one carries only what the sixth corpus source raised, so the two do not collide.

> ## ✅ Closed — decided 2026-08-27, applied at spec revision 4
>
> All nine items are resolved. `docs/bowen_agent_model_spec_v2.md` carries the changes and its own
> revision-4 log lists them. **This file is now the record of how each was decided, not an open question.**
>
> **One decision changed the answer.** A3 authorised the Ch13 re-read. The reconciliation proposed
> below — *they are about different people's anger* — **did not survive it**. Ch13's anger is the
> **mover's**, and the chapter says the *absence* of it admits the peak. `M5.D.4` was therefore
> **inverted** and has been corrected. See §A3 below.

**Nothing here has been applied.** *(Superseded — see the box above.)* The spec was approved on 2026-08-25. Three of the items below
change requirements the model is built on, and one closes an open contradiction.

**What the run produced:** 12 segments, 219 findings, pass 2 complete, folded into `_LEDGER.md` as
Source 7. Fifteen load-bearing quotations were verified character-for-character against the source
files; six pass-1 readings were corrected, one of them a term confusion that would have put a wrong
invariant on `M1.B.8`.

| | |
|---|---|
| Extraction | `docs/theory/family_evaluation/fe00.md` – `fe11.md` |
| Comparative pass | `docs/theory/family_evaluation/_FE_PASS2.md` |
| Ledger | `docs/theory/_LEDGER.md`, **Source 7** |

---

## A. Contradictions with the approved spec — **4 items, decision needed**

### A1 ⚠ `M1.A.11b` — does **learning** or **constitution** select the symptom category?

**The spec says:** the symptom **channel** (physical / mental / social) is exogenous constitutional
data. *Level sets amplitude and sign; constitution sets channel.* Cites `KS22.14`, `KS21.5` (2019).

**Kerr 1988 says the opposite** (`FE08.3`, verified verbatim):

> "**Genes are an important influence on the type of symptom that develops, but learning based on
> childhood experience appears to be the most important influence on the category of clinical
> dysfunction (physical, emotional, social) that develops.** … **Genes can influence the specific symptom
> that develops, but they seem to have less influence on the category of dysfunction.**"

and names the relational determinant (`FE07`): the category tracks "**what others in the system focus
on in that individual when they get anxious.**" A third passage (`FE09.1`) gives a decomposition that
makes the two 1988 statements consistent — **constellation → ripeness; learning → category;
constitution → specific symptom** — and consistent with the spec only if `M1.A.11b` is about the
specific symptom rather than the category.

**Why I cannot settle it.** `KS22.14`'s wording — "whether the chronic anxiety plays out as
schizophrenia **rather than some other clinical dysfunction**" — is genuinely ambiguous between
*symptom* and *category*, and `KS21.5` is about amplitude, not channel. Resolving it means re-reading
those two 2019 passages in context, which is the unchecked-extraction risk `_STATUS.md` names.

| Option | Consequence |
|---|---|
| **(a)** The 2019 citations are about the **specific symptom**. `M1.A.11b` has the level wrong; amend it. | `M1.A.11a`'s internalise/externalise axis becomes an **output** of the family's anxious focus rather than a constitutional input. `M11.C.2` gains a second dimension: symptoms concentrate on the projection target **and in the channel the family attends to**. |
| **(b)** 2019 revised 1988. Keep `M1.A.11b`; record `FE08.3` as superseded. | Nothing changes. But see §D — the measured direction of Kerr's drift over 31 years is *toward* wider constitutional claims, which makes a quiet revision here less likely than a scope difference. |
| **(c)** Both hold, via `FE08.3`'s own override: "**Genetic predisposition to a disease… can be strong enough to override relationship programming.**" | `M1.A.11b` keeps constitution as the channel default and gains a relational term that can move it. Most machinery, least contradiction. |

**My reading: (a) or (c).** (b) requires believing Kerr reversed himself on a mechanism he stated twice in one book and never flagged as revised.

**Until this is decided, `model_explainer.md` §3.9 is marked provisional.**

> ### ✅ DECISION — **(c)**. A genetic disposition can override relationship programming.
> **Applied** as `M1.A.11c`: three terms in order — constitution sets the channel **prior**, a
> family-focus term can **shift** it, a constitutional-strength term can **override** the shift.
> `FE09.1`'s decomposition bounds it: only the **category** moves; the *specific symptom within a
> channel* stays constitutional. `model_explainer.md` §3.9 is no longer provisional.
---

### A2 ⚠ `M1.A.3` — the transition at 50: **licence** or **awareness**?

**The spec says:** exactly one behavioural transition at 50, implemented as *a licence over joint
decisions* — below 50 the emotional system permits the intellect its own domain **except** where a decision affects the shared life course. Derived from Ch16, **Bowen's own**.

**Kerr 1988 describes the same transition differently** (`FE04.3`):

> "**Above 50, the intellectual system is sufficiently developed to make a few decisions of its own.**"
> "**A criterion for distinguishing people who are above rather than below 50 is that above 50 there is > more awareness of the difference between feelings and intellectual principle.**"

That maps onto `M4.D.1a`'s mixing weight — the self-directed channel opening — rather than onto a decision-scope licence.

**Recommendation: keep both, neither overwritten.** `M1.A.3`'s licence as the behavioural 
implementation; `FE04.3`'s awareness criterion as the **readout** that distinguishes the bands. The index rule forbids a Kerr formulation silently overwriting a Bowen one, and `M1.A.3` was chosen deliberately to avoid modelling low differentiation as reduced cognitive capacity — which `FE02.16`
independently confirms is wrong ("**the intellect operates in the service of the feeling and emotional process**"; low-level agents argue fluently).

**Decision needed:** accept the recommendation, or treat `FE04.3` as superseding.

> ### ✅ DECISION — accept the recommendation. Keep both.
> **Note from the project owner, and it is now normative:** *it is not about cognitive ability — it is
> about the strength of the emotional circuits overriding the cognitive ones.*
> **Applied** as `M1.A.3b`: `M1.A.3`'s licence stays the behavioural implementation; `FE04.3`'s
> awareness criterion becomes the **band discriminator at readout**, over `M4.D.1a`'s mixing weight.
> The requirement states explicitly that a low-`basic_level` agent **MUST** argue as fluently and hold
> positions as confidently as a high one — the difference is which circuits are driving, not how good
> the cognitive ones are.

---

### A3 ⚠ `M5.D.4` — is anger the gate, or negative evidence?

**The spec says:** *"Anger **MUST** be the **gate** admitting the sequence to `PEAK`, not a fourth abort branch."* From Ch13.

**Kerr 1988** (`FE04.4`): "**it is not fueled by anger. Anger can sometimes be a stimulus to clarify one's thinking, but it is not a reliable guide for action. When someone angrily and dogmatically claims to be a 'self,' he is usually unsure of his position and is blaming others for his plight in
life.**"

**Bowen agrees, independently, in the same volume** (`FE11.19`): "**A dogmatic person is rarely sure of self.**"

**Probable resolution: they are about different people's anger.** `M5.D.4`'s gate sits in the *system's* reaction ladder — the opposition escalating to anger is the signal the move landed
(`M5.E.3`). `FE04.4` is about the *mover's* state. Read that way both stand, and the model gains a cheap gate it lacks: **an `I-POSITION` emitted while the mover's own anger is above threshold MUST
execute as `M5.F.4`'s assertion form.**

⚠ **But this needs `L13` checked against the primary chapter before anything is encoded.** If Ch13's
anger is the *mover's*, the two genuinely conflict and `M5.D.4` is in trouble. That check is a half-hour of reading Ch13 and I have not done it.

**Decision needed:** authorise the Ch13 re-read, or accept the reconciliation on its face.

> ### ✅ DECISION — re-read Ch13. **Done 2026-08-27, and it changed the answer.**
>
> **The reconciliation above does not hold.** It supposed that `M5.D.4`'s anger was the *system's* and
> `FE04.4`'s the *mover's*. Ch13 mentions anger **exactly once**, and it is the mover's:
>
> > "When he is finally able to maintain his course **without getting angry at the opposition**, the
> > opposition does a final intense emotional attack. If he remains calm with this, the opposition
> > becomes calm and pulls up to his level of individuality."
>
> Verified character-for-character against the primary chapter. (Five apparent further hits on
> `anger|angry|rage` are substring matches inside *average*, *encourages* and *courageous*.)
>
> **So `M5.D.4` was inverted.** The gate is the mover's **freedom from** anger, not anger. The spec
> said the opposite. `docs/theory/ch13.md` had it right in its body — "anger-free holding gates entry
> to the peak" — but carried a lossy heading, "Anger gates the escalation", and that heading is what
> reached the spec. The heading is now corrected in place with a note saying why.
>
> **The outcome is better than the reconciliation would have been.** Ch13 (Bowen, 1970s), `FE04.4`
> (Kerr, 1988) and `FE11.19` (Bowen, 1988) are all about the **mover's** anger and all say the same
> thing. Three statements, two authors, two decades, one rule — and `FE11.19` is an independent
> witness under the Epilogue rule.
>
> **Applied** as the corrected `M5.D.4`, the new `M5.D.4a`, and the acceptance criterion `M11.C.32`.
>
> **Note from the project owner, folded into `M5.D.4a`:** *anger represents a level of intensity of
> negative energy; what I do with it reflects level of differentiation — do I notice it, manage it,
> learn from it, or do I reactively get defensive and do something unproductive.* The requirement now
> states that anger is **not a state** (M1.A.0 still binds) but an intensity on the negative side of
> the appraisal, whose **handling** is the readout — and that M5.D.3's abort branches are what
> discharging it looks like.
---

### A4 `M12.2` — one of the two stated unknowns now has a mechanism

**The spec says:** two things Bowen states he does not know **MUST** be labelled `[I]` wherever the model decides them — (i) what determines whether a problem stays in the spouse dyad or transmits to a child, and (ii) what selects which spouse takes the dominant pole at identical levels.

**For (i), Kerr 1988 gives a mechanism** (`FE07.3`): the parents' **emotional complementarity**, formed in their families of origin and exaggerated by anxiety. "**The nature of these exaggerated elements determines whether the problem emerges as marital conflict, spouse dysfunction, or child
dysfunction.**"

**For (ii) he gives only a locus, not a rule** — "by mutual agreement, the product of the emotional fit." *(Pass 1 claimed he answered both; corrected at `_FE_PASS2.md` §3.1.)*

| Option | |
|---|---|
| **(a)** Accept the mechanism for (i). `M12.2`'s first clause is superseded; the model **derives** the sink allocation from `M2.A`'s membership rather than declaring it. | Removes an `[I]`. Requires the complementarity to be represented, which `M2.A.0f` partly is. |
| **(b)** Treat it as naming the locus only. `M12.2` stands in full. | Safer. `FE07.3`'s own hedge — "determined **largely by**" — supports it. |

**(ii) stands either way.**

**And one part applies regardless and is not contested.** `FE07.3`: "**males and females assume the dominant position with equal frequency.**" That is a hard constraint the spec does not state and a
natural implementation could easily violate — any asymmetry in the pole-assignment rule would show up as a sex effect across an ensemble.

> **Proposed, uncontested:** `M2.A` gains a requirement that pole assignment be **independent of sex**, assertable over an ensemble.

> ### ✅ DECISION — (i): **(b)**, locus only. `M12.2` stands in full. Tail: **agreed**.
> `FE07.3` names *where* the answer lives without saying what it is, and its own hedge is "determined
> **largely by**", so the sink allocation stays `[I]`. Recorded as `M12.2a`. What does follow is
> `M9.6`: the allocation is **identifiable** from the belief configuration — a readout, not a
> determinant — with `M11.C.26` asserting it in both directions.
> The sex-independence requirement is applied as `M2.A.0g`, with `M11.C.25` as its ensemble test.
---

## B. What settles without a decision — **for information**

Applied where it costs nothing; recorded where it does not.

1. **Explainer §13.3 / `_RESOLUTIONS.md` R3 — when `basic_level` is fixed — is closed.** `FE04.9` states
   the exact shape the project had provisionally chosen: established by adolescence, usually fixed,
   changeable by structured effort. **Applied** to the explainer. It adds one thing the spec has too
   narrowly: *self-sustaining independence from the family of origin* gates **basic-level change
   itself**, not only the `I-POSITION` move — a slow-clock condition on `M7.A.1`, not just a fast-clock
   gate at `M5.C.1`. That part is a spec change and is **not applied**.
2. **Three sinks, and distance is not a fourth — retired.** `FE07.1` gives both halves in one passage.
3. **`KS11.1`'s traversal timing — no longer in tension with `FE07.7`.** Three generations is the
   quantum-jump case; five to ten is typical; at ~10 points/generation that is 30 over three, which is
   `FE07.7`'s stated maximum. Same claim, two statements.
4. **Twelve requirements move to written primary text**, five onto Bowen's own hand. Nothing in the
   model changes. `_LEDGER.md` Source 7 `FE-B` has the table. `M1.A.9a` — the two-axis definition the
   model is built on — has until now reached the project *secondhand*; the primary is stronger than
   the paraphrase and supplies the fixture `M11.C.19` needs.
5. **Explainer §16's weakest convergence is better resolved.** "Leaders move first" and "the peripheral
   member moves first" are one alternating process — Bowen: "**leadership shifts back and forth**".
   **Applied** to the explainer.

---

## C. Requirement candidates — **21, none applied**

Full list at `_LEDGER.md` Source 7, `FE-C` and `FE-D`. The ten that change a mechanism:

| # | Candidate | Source |
|---|---|---|
| 1 | **Sink allocation is identified by the belief configuration** — three sinks, three distinct attribution patterns. Connects `M1.D.1` to `M9` for the first time; assertable both ways. | `FE07.2` |
| 2 | **Belief MUST be able to drive appraisal**, not only record it. Three independent arguments, one of them Bowen's own correction of "projection" — the transfer runs **through descriptions**, which makes `M9` the channel the projection process operates on. | `FE02.7`, `FE05.10`, `FE11.3` |
| 3 | **Reinforcement across persons** — the projection loop's step 4: the child's compliance is reinforced by the **mother's** calming. Nothing currently drives internalisation. | `FE07.4` |
| 4 | **A self-generated stress term**, derived from `basic_level`. The model has inter-person load and nothing intra-person. | `FE08.5` |
| 5 | **Symptom channels are mutually protective**, not merely substitutable. | `FE01.6`, `FE05.8` |
| 6 | **Lock-in is non-monotone** — past a severity threshold the symptom destabilises and removal brings relief. Three independent statements. Amend `M7.D.2a`, `M11.C.22`. | `FE03.9`, `FE07.15` |
| 7 | **`M4.C.8`'s amplification MUST be gated by objectivity** — as written, every act of reflection is an escalation, which makes `M1.E.7c`'s category supply impossible. | `FE03.6` |
| 8 | **A `PREPARE` phase before `M5.D`** — Bowen's 1967 trip: months of planning, one private letter per triangle, "**to cause the triangles to come to me.**" | `FE11.9` |
| 9 | **A second event type: a binder becomes unavailable** — raises anxiety while adding no load. | `FE09.2` |
| 10 | **Emotional reserve**, derived as *capacity − accommodation stock*, gating symptom onset. | `FE03.10` |

**And one prohibition, from Bowen directly** (`FE11.7`):

> "**there is no such thing as one person taking action against another. The issue of 'power' or
> 'punishing' another person does not apply with the concept of differentiation of self.**"

> **Proposed `M12` entry:** `power` and `punishment` **MUST NOT** exist as mechanisms; no move may be represented as one agent acting *against* another. This matters — `M1.B.5`'s dominant pole,
> `M5.E.1`'s consequence rung and `M8.6`'s alignment penalty could each be implemented adversarially,
> and nothing currently forbids it. It is the only `M12` entry stated as a prohibition **by the
> author** rather than inferred by the project.

> ### ✅ DECISION — **apply all 21**, and adopt the `M12` prohibition.
> Read as covering both tiers of §C: the ten mechanism candidates in `_LEDGER.md` `FE-C` and the
> readout and criterion candidates in `FE-D`. Applied as:
> `M9.6`, `M9.7`, `M4.D.6e`, `M4.A.5`, `M7.D.2c`, `M7.D.2d`, `M4.C.8a`, `M5.D.2a`, `M1.F.9`,
> `M4.D.5d`, `M7.E.4`, `M1.E.8`, `M11.3`, `M11.G.1`–`M11.G.4`, and amendments to `M7.D.2a`,
> `M11.C.3`, `M11.C.17` and `M11.C.22`. New acceptance criteria `M11.C.25`–`M11.C.33`.
> **The prohibition is `M12.5`** — the only `M12` entry stated by the author rather than inferred by
> the project. It binds `M1.B.5`, `M5.E.1` and `M8.6`, each of which could have been implemented
> adversarially and none of which was previously forbidden from it.
> `FE-D`'s twelve stated magnitudes are admitted as `M10.C.4` — **checks on an output, never
> parameters** — and `M10.C.3` gains `M10.C.3a` to say that its prohibition is on the *parameter*,
> not on the quantity.
---

## D. Two things worth knowing about the source

**Kerr 1988 → Kerr 2019, measured.** The project now holds two texts by the same author 31 years
apart — its only longitudinal control on a single voice. Two divergences were found and **both are
about disease, and both run the same way: the claim widened and the hedge weakened.** In 1988 it was
"**not to say that what is occurring in the family causes the cancer**… mutual influence… **cannot be
regarded as a proven fact**"; by 2019 it is the *unidisease* chapter. This **strengthens** the
`[K-ext]` grading of the 2019 disease material — the extension is now visible as one, because both
endpoints are in the corpus. On everything that is not disease the two texts agree closely, several
times in near-identical language.

**The one place this corpus has independent witnesses.** Bowen wrote the Epilogue without sight of
Kerr's chapters — "**without my knowledge of its content. This has been purposeful.**" Six claims are
stated separately by both men. Against the standing problem that nine chapters of the papers report
one NIMH project, that is the project's first clean two-witness convergence. The limit: independence
of *drafting*, not of *training*.

---

## E. Not to be implemented — **for the record**

- **`FE07.21` — the trauma claim.** "the child's life course is more influenced by the lack of emotional separation… **than by the abuse itself**"; "events are not the process". A 1988 clinical position with
  no series, no comparison group and no measurement, on a question where the wider evidence base has moved considerably. Nothing in the model requires it. 
  **Proposed `M11.F` entry: no output may be presented as adjudicating discrete traumatic events against ongoing relational process** — the model does not distinguish them and must not be read as having an opinion.
- **`FE09.16`** — LeShan's reported **50% cure rate** in "medically hopeless" cancer patients (1977), at face value, uncontrolled. `[X]`, excluded. Also evidence that the widening in §D was underway in 1988.
- **`FE08.17` / `FE02.19`** — the unidisease, and the claim that the emotional system sits "at a level probably more basic than genes". `[K-ext]`, `[X]`. `M11.F.4` unaffected and better supported.
- **`FE10.17`'s societal symptom list** — "an incessant clamor for '**rights**'" is a 1988 editorial judgement, not an observation. `M1.D.7`'s readouts **MUST NOT** inherit it; a readout that scores rights claims as regression is one-sided by construction and fails `M11.F.6`.

> ### ✅ DECISION — adopt both.
> `M11.F.7`: no output may be presented as adjudicating discrete traumatic events against ongoing
> relational process. `M11.F.8`: `M1.D.7`'s societal readouts must not inherit the 1988 symptom list,
> and specifically must not score claims of **rights** as regression. `FE09.16`, `FE08.17` and
> `FE02.19` remain `[X]` and excluded.

---

## What I need from you

| # | Item | Kind |
|---|---|---|
| 1 | **A1** — learning or constitution for the symptom category? | contradiction |
| 2 | **A2** — `M1.A.3`: keep both, or supersede? | divergence |
| 3 | **A3** — authorise the Ch13 re-read on anger, or accept the reconciliation? | contradiction |
| 4 | **A4** — does `M12.2`'s first unknown now have a mechanism? | narrowing |
| 5 | **A4 tail** — apply the uncontested sex-independence requirement? | new requirement |
| 6 | **C1–C10 + the `M12` prohibition** — which to apply in spec revision 4? | 11 candidates |
| 7 | **C, tier 2** — eleven readout and criterion candidates, `_LEDGER.md` `FE-D` | 11 candidates |
| 8 | **B1 tail** — extend `M1.A.15`'s dependence gate to basic-level change? | spec change |
| 9 | **E** — adopt the two proposed `M11.F` entries? | framing |

**Nothing is applied until you say so.** *(Superseded 2026-08-27 — all nine are decided and applied.)*

## Outcome — all nine, 2026-08-27

| # | Item | Decision | Applied as |
|---|---|---|---|
| 1 | **A1** — learning or constitution? | **(c)** — both, with an override | `M1.A.11c` |
| 2 | **A2** — `M1.A.3` keep both or supersede? | **keep both** | `M1.A.3b` |
| 3 | **A3** — Ch13 re-read or accept? | **re-read** → the reconciliation failed; `M5.D.4` was **inverted** | `M5.D.4`, `M5.D.4a`, `M11.C.32` |
| 4 | **A4** — does `M12.2`'s first unknown have a mechanism? | **(b)** — locus only; `M12.2` stands | `M12.2a`, `M9.6` |
| 5 | **A4 tail** — sex independence | **agreed** | `M2.A.0g`, `M11.C.25` |
| 6 | **C1–C10 + `M12`** | **apply all** | 10 requirements + `M12.5` |
| 7 | **C tier 2** — readouts and criteria | **apply all** | `M11.3`, `M11.G`, `M1.E.8`, `M7.E.4`, 4 amendments |
| 8 | **B1 tail** — extend the dependence gate | **yes** | `M7.A.1a`, `M11.C.33` |
| 9 | **E** — the two `M11.F` entries | **adopt both** | `M11.F.7`, `M11.F.8` |

**Two IDs I had to move.** `M1.A.3a` and `M7.E.1` were already taken; the new requirements are
`M1.A.3b` and `M7.E.4`. IDs are stable and are not renumbered, so the new arrival moves, not the
incumbent.
