---
tags: [model-bt, alignment, pass2, family-evaluation]
status: complete
date: 2026-08-26
---

# Pass 2 — *Family Evaluation* read against the spec and against the other five sources

Pass 1 extracted each of the twelve segments on its own. This pass re-reads the 219 findings against
each other, against the approved spec, and against the five existing sources, and separates what
**changes the spec** from what **confirms it**.

## Method note, and what the verification actually showed

Pass 2 of the Bowen book found that pass 1 had **systematically over-read in one direction** — it
made the source look more quantitative and more decided than it is — and withdrew nineteen findings.
The same risk applies here and was tested two ways.

**Verbatim verification.** Fifteen of the highest-stakes quotations were checked character-for-character
against the whitespace-normalised source files: `FE03.1` (two sides of the same coin), `FE03.5`
(median 40), `FE04.1` (55/15), `FE04.2` (addicted to comfort), `FE06.1` (propositions 1 and 4),
`FE06.5` (push together), `FE07.2` (the three convictions), `FE07.7` (not 30 points), `FE08.1` (the
20-point figure), `FE08.3` (genes/category), `FE10.2` (adaptiveness by comparison), `FE10.16` (the
ninth concept), `FE11.1` (the prediction rule), `FE11.2` (the analogy breaks down), `FE11.3` (each
describing the other). **All fifteen match.** No quotation in this run is paraphrase presented as
quotation.

**Reading verification.** Six pass-1 findings were nonetheless over-read — not in the quotations but
in the *inferences drawn from them*. They are corrected in §3, and they are the same shape as the
2019 book's pass 2: **scope corrections, not withdrawals.** One of them (`FE03.2`) is a term
confusion serious enough that acting on it would have introduced a wrong invariant.

---

## §1 — What this source contradicts in the **approved** spec

Four items. **None has been applied.** The spec was approved on 2026-08-25 and three of these change
requirements the model is built on, so they belong in a decision document, not in an edit.

### 1.1 ⚠ `M1.A.11b` — genes set the symptom, learning sets the **category** `FE08.3`

`M1.A.11b` requires that **which channel** a person expresses (physical / mental / social) be
**exogenous constitutional data**, and forbids deriving symptom type from relational position:
*level sets amplitude and sign; constitution sets channel.* It cites `KS22.14` and `KS21.5` (2019).

Kerr 1988, verified verbatim:

> "**Genes are an important influence on the type of symptom that develops, but learning based on
> childhood experience appears to be the most important influence on the category of clinical
> dysfunction (physical, emotional, social) that develops.** … **Genes can influence the specific symptom
> that develops, but they seem to have less influence on the category of dysfunction.**"

`FE07` says the same from the relational side — the category is "connected both to the particular way
an individual manages anxiety and to **what others in the system focus on in that individual when they
get anxious.**"

**The assignment is inverted.** `FE09.1` then supplies a three-level decomposition that makes the two
texts consistent with each other but not with the spec: **constellation → ripeness; learning →
category; constitution → specific symptom.**

**Possible resolutions, in order of likelihood:**
1. The 2019 citations are about the **specific symptom**, not the category — `KS22.14` reads
   "whether the chronic anxiety plays out as schizophrenia **rather than some other clinical
   dysfunction**", which is ambiguous, and `KS21.5` is about amplitude. If so the two texts agree and
   `M1.A.11b` simply has the level wrong.
2. The 2019 book revised the 1988 position. Precedence would favour 2019 — but see §5: the
   measured direction of Kerr's change over thirty-one years is **toward wider claims about
   constitution and disease**, which makes a quiet revision here less likely than a scope difference.
3. Both hold, with `FE08.3`'s override clause doing the work: "**Genetic predisposition to a disease…
   can be strong enough to override relationship programming.**"

**Consequence if 1 or 3 is right:** `M1.A.11a`'s internalise/externalise axis becomes an **output** of
the family's anxious focus rather than a constitutional input, and `M11.C.2` gains a second dimension
— symptoms concentrate on the projection target **and in the channel the family attends to**.

### 1.2 ⚠ `M1.A.3` — the transition at 50 is described differently `FE04.3`

`M1.A.3` implements the single behavioural transition at 50 as **a licence over joint decisions**,
derived from Ch16 (Bowen's own). Kerr 1988 describes the same transition as an **awareness and
capacity** boundary: "**above 50 there is more awareness of the difference between feelings and
intellectual principle**"; "the intellectual system is sufficiently developed to **make a few decisions
of its own.**"

The second form maps onto `M4.D.1a`'s mixing weight — the self-directed channel opening — rather than
onto a decision-scope licence. **They are probably complementary** (awareness is the precondition,
the joint-decision domain is where it first bites) but the index rule forbids a Kerr formulation
silently overwriting a Bowen one, and `M1.A.3` was chosen deliberately to avoid modelling low
differentiation as reduced cognitive capacity — which `FE02.16` independently confirms is wrong.

**Recommended landing:** keep `M1.A.3` as the behavioural implementation; add `FE04.3`'s awareness
criterion as the **readout** that distinguishes the bands. Both stated, neither overwritten.

### 1.3 ⚠ `M5.D.4` — anger is called an unreliable guide `FE04.4`

`M5.D.4`: *"Anger **MUST** be the **gate** admitting the sequence to `PEAK`, not a fourth abort
branch."* From Ch13.

Kerr 1988: "**it is not fueled by anger. Anger can sometimes be a stimulus to clarify one's thinking,
but it is not a reliable guide for action. When someone angrily and dogmatically claims to be a
'self,' he is usually unsure of his position and is blaming others.**"

**Bowen's epilogue supports Kerr here**, independently: "**A dogmatic person is rarely sure of self**"
(`FE11.19`).

**The probable reconciliation is that the two passages are about different people's anger** —
`M5.D.4`'s gate sits in the *system's* reaction ladder, `FE04.4` describes the *mover's* state. Read
that way both stand and the model gains a cheap gate: **an `I-POSITION` emitted while the mover's own
anger is above threshold MUST execute as `M5.F.4`'s assertion form.**

**This must be checked against `L13`'s wording before anything is encoded.** If Ch13's anger is the
mover's, `M5.D.4` is in genuine trouble. Pass 2 cannot resolve it without re-reading Ch13 against
the primary, which is outside this run's scope and is exactly the unchecked-extraction risk
`_STATUS.md` flags.

### 1.4 ⚠ `M12.2` — a mechanism is offered for **both** stated unknowns `FE07.3`

`M12.2` records two things Bowen says he does not know, to be labelled `[I]` wherever the model
decides them. Kerr 1988 offers a mechanism for both: the parents' **emotional complementarity**,
formed in their families of origin and exaggerated by anxiety, determines *which sink* carries the
load and *who takes the dominant pole*.

**See §3.1 — pass 1 over-read this.** The first is a real mechanism; the second is a *locus*, not a
rule, and `M12.2`'s second clause is untouched.

**One part is immediately applicable regardless and is not contested:** "**males and females assume
the dominant position with equal frequency.**" That is a hard constraint the spec does not state and
a natural implementation could violate. **`M2.A` requirement candidate: pole assignment MUST be
independent of sex, assertable over an ensemble.**

---

## §2 — Citation upgrades: what moves to written primary text

Eleven spec requirements currently rest on ASR sources under a no-verbatim constraint, or on Kerr
2019 reporting Bowen. All are now quotable from a published, author-edited 1988 text — and five of
them from **Bowen's own hand**.

| Requirement | Currently cites | Now available | Author |
|---|---|---|---|
| `M1.A.0` *emotional = instinct* | `kb02`·K02.1, `kb13`·K13.1 (ASR) | `FE01.9`, `FE02.2`, `FE11.11` | Kerr + **Bowen** |
| `M1.A.9a` two-axis identity | `kb12`·K12.3 (ASR); `KS24.4` (2019, secondhand) | **`FE03.1`** — both counterfeits named and paired | Kerr |
| `M1.A.4d` capacity floor at 25 | `KS05.2` (2019) | `FE04.2` — with its reason, *addicted to comfort* | Kerr |
| `M1.A.14a` functional sibling position | `KS12.2` (2019) | `FE02.13`, `FE07.13`, `FE10.7` — three instances | Kerr |
| `M1.D.2a` distance binds | `KS04.1` (2019), `kb04`·K04.1 (ASR) | `FE03.15`, `FE08.6` | Kerr |
| `M6.I.4`/`M6.I.5` pseudo/solid self | 1979 Tape 2 (ASR) | `FE04.10` — both defined | Kerr |
| `M5.E.1`/`M5.E.3` the ladder | Ch21, `L01.8` | `FE04.11` — both in one sentence | Kerr |
| `M7.E.1a` three-outcome transmission | `KS11.2` (2019) | `FE04.19`, **`FE11.4`** | Kerr + **Bowen** |
| `M1.D.4a` differentiation capacity | `kb08`·K08.2 (ASR) | **`FE11.10`** — with the two-thirds threshold | **Bowen** |
| `M11.F.3` general-systems objection | `kb13`·K13.2, `kb09`·K09.3 (ASR) | **`FE11.2`** — three passages, verbatim | **Bowen** |
| `M11.F.3a` validity criterion | `KS24.8` (2019) | **`FE11.11`** — *sun and earth, tides and seasons* | **Bowen** |
| `M7.A.2b` substitution on a dead parent | `KS13.11` (2019) | `FE06.4` — with a worked case and the mechanism | Kerr |

**This is the largest single improvement in evidential quality the project has had.** Nothing in the
model changes; what changes is that a dozen load-bearing requirements stop depending on transcripts
the project may not quote, or on one author reporting another.

---

## §3 — Corrections to pass 1 of **this** source

Six. All are **scope corrections**, none is a withdrawal, and one would have introduced a wrong
invariant.

### 3.1 `FE07.3` — the second `M12.2` unknown is **not** answered

Pass 1 said `FE07` "offers a mechanism for both" of `M12.2`'s stated unknowns. Re-read: for the
first — whether the problem stays in the spouse dyad or transmits to a child — the text gives a
mechanism ("**the nature of these exaggerated elements determines whether the problem emerges as
marital conflict, spouse dysfunction, or child dysfunction**"). For the second — which spouse takes
the dominant pole at identical levels — it gives only "**by mutual agreement — the product of the
emotional fit of a relationship**", which names *where* the answer lives and not *what it is*.

**Corrected:** one unknown gets a mechanism, one gets a locus. `M12.2`'s second clause stands.

### 3.2 `FE03.2` — ⚠ a **term confusion**, and the proposed invariant was wrong

Pass 1 proposed a new invariant: `investment(A→B) == investment(B→A)` on any emotionally significant
tie, from "**each person invests an equal amount of 'life energy' in the relationship.**"

**That conflates two quantities the spec keeps separate.** The source says *life energy* — which is
`M1.A.10`'s per-person zero-sum allocation between relationship-seeking and goal-directed activity.
`M1.B.8`'s `investment` is a different field: *share of thought occupied by the target*, directed and
valence-blind.

**Corrected:** the claim is that **the two parties commit equal shares of `life_energy` to the tie**,
which is a statement about `M1.A.10`, not about `M1.B.8`. It is still a real and useful constraint —
and `FE05.13`'s bossy/helpless and schizoid/hysterical pairs still support it — but the invariant
must be written on the right field. Had it been applied as drafted it would have forced a symmetric
`investment` and broken `M1.B.8`'s directedness, which `L04.7` and 1979 Tape 1 both require.

The **readout trap** survives intact and is unaffected: "**the person who seems more indifferent is
just as dependent on and influenced by the relationship as the person who seems preoccupied with
it.**"

### 3.3 `FE11.1` — "this project is its mechanisation" is too strong

Pass 1 wrote that the master theory *is* what the spec is, and called the model "a computational
instance of a device the theory's author designed". The continuity is real and important — impersonal,
exhaustive, prediction-generating, revised when a prediction fails, insulated from the operator's
feelings. But Bowen's device was **a written rule book applied by a human staff and communicated to
the families in advance**. A simulation is not that artefact.

**Corrected:** the claim is one of **continuity of method**, not identity of artefact. What it earns
is narrower and still worth having: the *enterprise* of building an impersonal predictive apparatus
over this theory is the author's own, so `M11.F.3`'s objection applies to the **implementation
choices** (mathematics, technology) and not to the project's existence. That is exactly what
`FE11.2`'s "**Anyone is welcome to that field if they wish**" leaves open, and no more.

This is the finding most at risk of being over-read later, because it is the most flattering to the
project. It is recorded here so it cannot be quoted without the qualification.

### 3.4 `FE10.1` — only **eight** of the ten components are computable

Pass 1 called the ten-component family diagnosis "a readout schema" and tabled all ten against model
state. Components **9 (therapeutic focus)** and **10 (prognosis)** are clinical judgements about what
to do next and what will happen, made by a person with an agenda. The model can compute the *inputs*
to a prognosis (`FE10.6`'s four rules) but not the prognosis, and it has no therapeutic agenda at all.

**Corrected:** components 1–8 are a readout schema; 9–10 are not, and should not be emitted.

### 3.5 `FE02.3` — Calhoun's rats are illustrative, not the warrant

Pass 1 proposed a new acceptance criterion (`M11.C.25` candidate — reciprocity re-forms in a group
selected to have none) and headed it with Calhoun's inbred-rat experiment. The rats are a 1963 study
read by Kerr as consistent; they did not measure differentiation and no transfer to humans is
established.

**Corrected:** the criterion's warrant is `FE01.3`'s mirror-opposite rule and `FE04.1`'s transfer
arithmetic, both of which are claims about human families. The rats belong in the rationale as an
illustration of the *shape*, and the criterion should not cite them as support.

### 3.6 `FE11.4` — the defining feature is **role-absence**, not the unplanned pregnancy

Pass 1's table said the upward child is identified by being "an **'extra'** — unplanned". The text
says "**commonly** an 'extra', **perhaps** the product of an unplanned pregnancy" — two hedges — and
gives the actual criterion in the same sentence: "**grows up outside the emotional process between
parents and children**", "**whose basic emotional needs are met without having to live out a special
role assigned by parental immaturity.**"

**Corrected:** the mechanism is **no role assigned**; an unplanned pregnancy is a frequent occasion
for it, not the criterion. This matters for implementation — the model would select on the wrong
variable, and `M7.E.1c`'s five downward situations are also *role assignments*, so the complement is
exactly right and needs no separate rule.

---

## §4 — New requirement candidates, consolidated

Twenty-one across the twelve segments, ranked by how much they change. **None applied.**

### Tier 1 — change a mechanism

| # | Candidate | Source |
|---|---|---|
| 1 | **Sink allocation is identified by the belief configuration** — three sinks, three distinct attribution patterns | `FE07.2` |
| 2 | **Belief MUST be able to drive appraisal**, not only record it — a held belief raises appraised intensity on conflicting events | `FE02.7`, `FE05.10`, `FE11.3` |
| 3 | **Reinforcement across persons** — the projection loop's step 4: the child's compliance is reinforced by the *mother's* calming | `FE07.4` |
| 4 | **A self-generated stress term**, derived from `basic_level` — the model has no intra-person load | `FE08.5` |
| 5 | **Symptom channels are mutually protective**, not merely substitutable — occupancy lowers the hazard on the others | `FE01.6`, `FE05.8` |
| 6 | **Lock-in is non-monotone** — past a severity threshold the symptom destabilises and removal brings relief. Amend `M7.D.2a`, `M11.C.22` | `FE03.9`, `FE07.15`, `FE05.1` |
| 7 | **`M4.C.8`'s amplification MUST be gated by objectivity** — as written, all reflection is escalation | `FE03.6` |
| 8 | **A `PREPARE` phase before `M5.D`**, or `M5.B.2` usable pre-emptively over a triangle set | `FE11.9` |
| 9 | **A second event type: a binder becomes unavailable** — raises anxiety while adding no load | `FE09.2`, `FE08.6` |
| 10 | **Emotional reserve**, derived as capacity − accommodation stock, gating symptom onset | `FE03.10`, `FE08.6` |

### Tier 2 — change a readout or a test

| # | Candidate | Source |
|---|---|---|
| 11 | **Adopt `FE10.1`'s components 1–8 as the readout schema** (see §3.4) | `FE10.1` |
| 12 | **The four-proposition 2×2** as an acceptance criterion — `M11.C.3` tests one of four cells | `FE06.1` |
| 13 | **Sink mobility is protective** — fixed vs rotating allocation at equal budget | `FE07.6`, `FE10.3` |
| 14 | **Relief reallocates; differentiation reduces the budget** — no criterion distinguishes them | `FE05.2` |
| 15 | **Ties deteriorate by default** at a rate inverse to level — makes `M11.C.17` non-trivial | `FE03.13` |
| 16 | **Coach contact frequency is non-monotone with a low optimum**, family-dependent | `FE11.8` |
| 17 | **`M11.C.1` strengthened to three directions**: onset sooner, remission slower or absent, more members symptomatic | `FE03.14`, `FE07.12` |
| 18 | **Every discriminating criterion needs a stressor** — cohesion types are identical when calm | `FE04.18` |
| 19 | **Pole assignment MUST be sex-independent** — assertable over an ensemble | `FE07.3` |
| 20 | **`M11.C.17` asserts a relay**, not a single mover; `[#]` ~several weeks to the second mover | `FE11.6` |
| 21 | **Stress weighting MUST include inter-event spacing**, not only magnitude and count | `FE10.5` |

### And one prohibition, from Bowen directly

> **`power` and `punishment` MUST NOT exist as mechanisms.** No move may be represented as one agent
> acting *against* another. `M12` candidate — a sixth entry beside the five withdrawn mechanisms.
> `FE11.7`

---

## §5 — Kerr 1988 → Kerr 2019: two measured divergences, one direction

The project now holds **two texts by the same author thirty-one years apart** — the only longitudinal
control it has on a single voice. Two divergences were measured, and both run the same way.

**Both are about disease claims, and in both the claim widened and the hedge weakened.**

| | 1988 | 2019 |
|---|---|---|
| Family process and disease | "**not to say that what is occurring in the family causes the cancer**… mutual influence… **cannot be regarded as a proven fact**" (`FE00.3`) | `KS23`'s *unidisease* chapter, stated as a proposal in its own right |
| Unidisease | "**may be** anchored in the emotional processes" — conditional on a premise (`FE08.17`) | a named concept with a chapter |

**Three consequences.**

1. **The `[K-ext]` grading of the 2019 disease material is strengthened, not weakened.** The extension
   is now visible *as* an extension, because both endpoints are in the corpus. `M11.F.4`'s
   prohibition is better supported than before.
2. **Stability elsewhere is notable.** On everything that is not disease — the two-axis identity, the
   capacity floor at 25, functional sibling position, the transmission distribution, distance-binds —
   the two texts agree closely, several times in near-identical language. A single author holding a
   position for thirty-one years is meaningful evidence about the theory's stability, and it is the
   one place this project can measure it.
3. **It bears on §1.1.** If Kerr's drift over thirty-one years is *toward* wider constitutional and
   disease claims, then a quiet 2019 revision *away* from 1988's "learning sets the category" is less
   likely than a scope difference between the two passages.

**One 1988 claim exceeds anything in 2019 and must be flagged:** `FE09.16`'s report of a **50% cure
rate** in "medically hopeless" cancer patients (LeShan 1977), stated without qualification. `[X]`,
excluded, and evidence the widening was already underway.

---

## §6 — The independence finding, and what it does to the convergence audit

Bowen opens the epilogue:

> "**Dr. Kerr has written a major portion of the book, without my knowledge of its content. This has
> been purposeful.**"

**The two halves of this book are independent witnesses.** That matters because the project's
standing evidential problem, recorded at the top of `_LEDGER.md` and in `_CONVERGENCES.md`, is that
**nine chapters of the papers report one NIMH project** — agreement among them is one study reported
repeatedly.

Six points are stated independently by both authors in this volume:

| Claim | Kerr | Bowen |
|---|---|---|
| Projection is **mutual**, not one-way | `FE07.4` fn 22 | `FE11.3` — "each describing the other" |
| The general-systems objection | `FE00.1`, `FE01.14` | `FE11.2` |
| Three-outcome transmission | `FE04.19` | `FE11.4` |
| Dogmatism is negative evidence for self | `FE04.4`, `FE06.13` | `FE11.19` |
| The family is not the system's boundary | `FE10.9` | `FE11.14` |
| Basic and functional levels are separable by intervention | `FE04.6` | `FE11.5` |

That is not a large number, but it is **genuinely independent** in a corpus where almost nothing
else is. It should be recorded in `_CONVERGENCES.md`'s independence audit as the project's first
clean two-witness convergence.

**One caution.** Kerr had worked in Bowen's programme for twenty years. Independence of *drafting* is
not independence of *training*, and Bowen says so plainly: "**He probably knows more about my
theoretical, therapeutic, and organizational orientation than any other person.**" So this is two
observers from one school writing without sight of each other — better than co-authorship, weaker
than two schools.

---

## §7 — Must not be implemented

Four items, recorded so they are not re-invented.

- **`FE07.21` — the trauma claim.** "the child's life course is more influenced by the lack of
  emotional separation… **than by the abuse itself**"; "events are not the process". A 1988 clinical
  position with no series, no comparison group and no measurement, on a question where the wider
  evidence base has moved. **`M11.F` candidate: no output may be presented as adjudicating the
  relative weight of discrete traumatic events against ongoing relational process.** The model does
  not distinguish them.
- **`FE09.16` — LeShan's 50% cure rate.** Excluded; see §5.
- **`FE08.17` / `FE02.19` — the unidisease and the sub-genic emotional system.** `[K-ext]` and `[X]`.
  `M11.F.4` unaffected and better supported.
- **`FE10.17`'s societal symptom list.** "an incessant clamor for '**rights**'" is a 1988 editorial
  judgement, not an observation. `M1.D.7`'s readouts **MUST NOT** inherit it; a readout that scores
  rights claims as regression is one-sided by construction and fails `M11.F.6`.

---

## §8 — What this source settles

Three things that were open.

1. **`_RESOLUTIONS.md` R3 / explainer §13.3 — when `basic_level` is fixed.** Recorded as `[X]`, still
   open, with the provisional decision to implement it as slow-moving with a ratchet. `FE04.9`:
   "**fairly well established by the time a child reaches adolescence and usually remains fixed for
   life, although unusual life experiences or a structured effort… can lead to some change in it.**"
   **The provisional decision was right and now has a written primary source.** The contradiction can
   be closed.

2. **The three-versus-four-sinks question.** Reopened repeatedly across the corpus. `FE07.1` states
   the resolution and both halves of it: three categories, and distance is not a fourth **because it
   is a feature of all relationships and intertwined with all the patterns** — while being no less
   important. Retired.

3. **`FE07.7`'s apparent tension with `KS11.1`'s traversal timing**, raised in pass 1 and dissolved by
   `FE08.2` in the same run: three generations is the **quantum-jump** case requiring `FE08.1`'s three
   conditions; five to ten is typical. At ~10 points per generation under the conjunction, three
   generations is 30 points — exactly `FE07.7`'s stated maximum. **Same claim, two statements.**

## And what it leaves open

- **§1.3's anger question** cannot be closed without re-reading Ch13 against the primary text.
- **§1.1's category question** cannot be closed without re-reading `KS22.14` and `KS21.5` in context.

Both are instances of the risk `_STATUS.md` names as the load-bearing claim in this work: ~19,000
lines of extraction have never been checked back against their chapters. This run adds ~4,000 more,
and the fifteen verbatim checks in the method note above cover its highest-stakes quotations only.
