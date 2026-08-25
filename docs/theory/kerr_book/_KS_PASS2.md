---
tags: [model-bt, source, kerr-secrets, pass2]
status: pass 2 complete
date: 2026-08-25
---

# KS Pass 2 — *Bowen Theory's Secrets* read against the other four sources

**Method.** Every pass-1 finding checked against `_LEDGER.md` (book), `_LECTURES_1979.md`, `kb/` (Kerr interviews) and `_EXTERNAL_MEASURES.md`. Reported in four classes: **contradictions**, **corrections to spec requirements**, **convergences at strength**, and **new mechanisms**. Nothing here is applied to the spec — this document is the argument; `_STATUS.md`'s S3 queue is the work.

**Headline.** Across 262 findings, **no finding of this book contradicts a finding of the Bowen book, the 1979 lectures, or the interviews.** What it does is (a) contradict **six spec requirements** the project wrote, (b) reopen **one approved decision**, and (c) supply mechanism where the earlier sources gave only direction.

---

## 1 · The one item that touches an approved decision

### 1.1 `A1` — "eight to ten generations" appears in print

| | |
|---|---|
| **Approved** | `A1` withdrew the figure as a calibration target, on `KB11` showing it was rhetorical. User approved. |
| **Now** | `KS11.1`: "**It could take eight to ten generations or as few as three or four**" — for traversal from a very good to a very poor level. |
| **Assessment** | The withdrawal **stands**. This is a **range with a 3× spread**, describing **full-scale traversal**, not a per-generation decrement. `M10.C.3` forbids a *rate*; untouched. No method given. |
| **Recommend** | Record as a `[#]` **shape constraint** on `M11.C.6`: a top-to-bottom decline **MUST NOT** complete in one or two generations and **MUST** be achievable within roughly ten. |
| **Status** | ⬜ **user's call** — already queued in `_STATUS.md`. |

---

## 2 · Contradictions with spec requirements the project wrote

Six. Each is the book correcting *us*, not correcting an earlier source.

### 2.1 `M7.A.2` — "MUST transfer automatically" is wrong

**Spec:** differentiation gained in a peripheral system **MUST** transfer automatically to the nuclear family.

**`KS16.9`:** "**Progress in the family of origin does not transfer automatically to nuclear family relationships, but it helps considerably.**" Two reasons given: the spouse brings their own unresolved attachment, and the nuclear tie is higher-load and harder to observe.

**Evidence:** `KS16.6` — Kerr had been **teaching the co-creation idea for over twenty-five years** before observing it in his own marriage.

**Reconciliation with `KS13.11`** ("as people made progress in their family of origin they also started making progress in their current family systems"): the **capacity** transfers; the **application** is re-earned per tie.

**Consequence:** amend `M7.A.2`. And it raises a real design question — **should `systems_perspective` be per-tie rather than a global scalar?** Two independent cases say yes (`KS16.6`, `KS17.1`).

### 2.2 `M1.A.14` — `sibling_position` cannot be "static profile data… and nothing else"

**`KS12.2`:** "**the family projection process can so impair the functioning of a firstborn son that his younger brother may function more like an older brother than the firstborn does. In such instances, the younger son becomes a 'functional oldest.'**" Kerr's own case likewise: "**some functional oldest traits in me, despite my position as last-born.**" And `KS19.6`: Wanda Kaczynski, third of five, functioning as oldest.

**Consequence:** birth-order stays static data; a **derived** `functional_sibling_position` is computed from actual functioning, and the propensity vector reads the derived one.

**This is the second instance of the same class.** Tape 6 already forced it on `structural_importance` (`M1.A.13a`). Worth stating once, generically: **no positional attribute is read from a role label; every one is derived from function.**

### 2.3 `M1.A.7` — chronic anxiety is not only a fixed per-person scalar

**Spec:** `chronic_anxiety` fixed in childhood from witnessed history, acting as a floor.

**`KS10.2`:** "**chronic anxiety is a consequence of various types of social interaction and, consequently, is most usefully conceptualized as a property of the emotional field**", determined by (1) the system's differentiation and (2) "**the person's functioning position in the system**" — which changes.

**Reconciliation:** the spec's fixed value is the **programmed reactivity** (`KS09.10`'s first deficit). Manifest chronic anxiety is *that reactivity × field × position*. The model has one and needs both.

**Consequence:** `M1.A.7` splits. **High priority** — it is upstream of `M4.C`, `M7.D` and the `M1.A.4a` estimator.

### 2.4 `M11.C.9` — tests the effect at constant `basic_level`, which is what modulates it

**`KS12.1`:** Bowen "**qualified his thoughts about Toman's profiles by saying that they accurately describe people at the midrange**"; "**a poorly differentiated oldest brother may exhibit very few characteristics of an oldest profile.**"

**Consequence:** rewrite as a **three-arm** test — low / mid / high `basic_level`, same position, same stressors, with the **mid arm showing the largest position-typical behaviour**. Direction-of-difference form, no invented effect size. **This resolves an `F2` item**, taking it from four to three.

### 2.5 `M1.D.2a` — "absorbs without symptomising" is half the story

Already resolved as `S2.15` from the user's D9 note; the book supplies the mechanism.

**`KS04.1`:** distance **binds** — "**the anxiety is now evident or bound in distancing behaviors**… **If people are unable to distance from each other for whatever reason, internalized anxiety reappears**"; "**the anxiety is integrated into the structure of a relationship.**" And it "**inevitably accompan[ies] each of the other patterns**" rather than competing with them.

**Consequence:** `DISTANCE` moves anxiety from person to tie, where it persists, is readable, and **returns if distancing is prevented**. A model in which `DISTANCE` reduces total system anxiety is failing. This also dissolves the old 3-vs-4-sinks question for good.

### 2.6 `M4.D.1` — exactly one move per tick cannot express the book's canonical cases

Two of the book's central instances are moves **generated, detected and not emitted**:

- `KS08.1` (the staircase): "**I even started to move slightly, but I caught myself and stopped**… **I did not take any obvious I-position with Mother; I just did not anxiously hover over her.**"
- `KS16.2` (the couch): "**I have something I need to do upstairs**… **My legs felt like they weighed a hundred pounds each.**"

And `KS16.1` establishes that **non-reaction is a distinct and insufficient move**, not a weak I-position — phase 2 of a three-phase trajectory that "**was an insufficient response.**"

**Consequence:** `M5` needs a `WITHHOLD` move distinct from both the reactive set and `I-POSITION`, and a suppressed move **MUST** still change tie state.

---

## 3 · Convergences at strength — claims now carried by four or five sources

These are settled. Recorded so they are not re-litigated.

| Claim | Sources |
|---|---|
| **The theory was built on actions, not on what people said** | `KB02`·K02.2, `KB08`, `KS00.12`, `KS01.2`, `KS08.7` — **five** |
| **The differentiating move carries no explanatory content** | `KB12` (secrecy), `KS15.2`, `KS16.3`, `KS16.4`, `KS17.4` — **five**; "**trying to explain it to the family usually reflects one's own anxiety**" |
| **The coach's objective is to understand, not to help** | `KB03`·K03.1, `KS15.5`, `KS17.10`, `KS18.9`, `KS19.9` — plus the experiment and Kerr's replication |
| **Coach capacity is capped by their own family-of-origin work** | `KB07`·K07.2, `KS13.10`, `KS15.7`, `KS18.10` — **four** |
| **The four social cues** — approval, attention, expectations, distress | `KS01.7`, `KS07.6`, `KS09.4`, `KS10.1`, `KS23.1` — **five**, identical list, list open |
| **Asserting a differentiated state is negative evidence for it** | five interview forms + `KS05.5`, `KS15.7` |
| **Blame is the readout for systems perspective** | `KB07`·K07.1 + `KS00.2`, `KS01.3`, `KS08.5`, `KS16.7` — and now **two-sided twice over**: blame/praise *and* other/self |
| **Maximal devotion accompanies the worst outcomes** | `KS19.1` Wanda, `KS20.7` Bessie, `KS21.11` Nancy, `KS25.8` Kerr's mother — **four families** |
| **Nothing in the model is pathological** | triangles `KS03.3`, distance `KS04.11`, anxiety `KS10.12`, cutoff `KS13.6` — **four**, same shape: real mechanism, real benefit, displaced cost |

### 3.1 The two-axis definition, sourced and dated

`KS24.4` locates the formulation the user supplied: Kerr responding to Bowen's "complete selflessness" —

> "**I expressed this idea in a previous book as a high level of differentiation being reflected in the ability to act for self without being selfish and the ability to act for others without being selfless (Kerr & Bowen, 1988).**"

`KS07.3` has him naming the two failures as "**selfish**" and "**unselfish**" and warning that treating them as traits produces "victim and culprit". `KS05.16` states it in full prose. `M1.A.9a` is well founded, and its primary citation is **Family Evaluation (1988)** — the next source queued.

### 3.2 The natural-experiment set — now **eight** instances of one discriminator

`M11.C.18` asks whether the estimator can separate a borrowed functional gain from a basic one. The corpus contains eight instances of the same shape: remove or restore a relational condition, watch functioning move, observe whether it holds.

| # | Case | Removed / restored | Direction |
|---|---|---|---|
| 1 | `KS07.1` | husband's cousin visits a week | ↑ then reverts |
| 2 | `KS10.8` | wife's business trip | ↑ then reverts |
| 3 | `KS11.4` | dominant twin dies at 21 | ↑ permanent |
| 4 | `KS10.4` | both grandparents die within six months | ↓ across four households |
| 5 | `KS19.5` | five years' isolation in Montana | ↑ then reverts on encroachment |
| 6 | `KS22.3` | **Eleanor vs Alicia pregnancy** — same event, different tie investment | no episode vs breakdown |
| 7 | `KS22.8` | both parents die | ↑ in the projection target |
| 8 | `KS25.2` | **parents' capacity collapses** | ↑ in the projection target |

Number 6 is a **within-subject control on the same nodal event type**. Numbers 7 and 8 are the projection target improving when the projectors are removed or decline. Number 4 is the reverse. **`M11.C.18` is not an artificial test — it is the corpus's most repeated observation.**

---

## 4 · New mechanisms the earlier sources did not supply

Ranked by how much they change the model.

1. **`KS23.1` — symptom lock-in.** "*A family can stabilize somewhat around the presence of a symptom, which fosters it becoming chronic.*" One line — symptom presence reduces subsequent family anxiety — yields chronicity, relapse-on-cure and relief-on-removal as consequences. **Nothing in the spec.**
2. **`KS22.2` — arrival as budget redistribution.** A new claimant reduces an existing tie's share by construction. Nothing bad need happen. Two instances (`KS25.5`: a fiancée, same effect).
3. **`KS06.4` — simultaneous conflicting urges raise anxiety independently of the emitted move.** An entropy/margin term over the propensity distribution feeding back into `acute_anxiety`. Produces the vicious circle for free.
4. **`KS06.1` — the repertoire has a complexity ordering** (cooperation → conflict → dominant-adaptive → distance), and anxiety slides selection down it. Replaces `M4.D.3`'s binary split with a derivation.
5. **`KS18.2` — route changes appraisal on *both* sides.** Same content, addressed to a third party, lowers reactivity in speaker and listener. `M4.C` currently has no emitting-side route effect.
6. **`KS18.4` — self-observation accuracy rises with elapsed time.** Six-month threshold; delayed replay beats instant replay. Implementable as the episode's acute anxiety decaying toward the floor.
7. **`KS19.3` — the accommodation ratchet.** Each concession small, reasonable, never reversed; the family's own baseline shifts, so "there was nothing gross" is sincere. Per-tie accommodation stock that grows and does not decay.
8. **`KS09.2` — five situations that draw the family focus**, a closed list, all computable at construction. `M7.E` currently has no target-selection rule.
9. **`KS17.3` — the reappraisal is self-attribution.** "*What he perceived as a tense expression on her face reflected his own facial expression and tone of voice.*" `M4.C` appraises incoming events with no attribution to the agent's own prior emission.
10. **`KS23.3` — allostatic load.** The cost is **time above floor**, not peak. Better-specified `M4.C.3`.
11. **`KS05.2` — a second threshold at 25**, below which basic level cannot rise while functional level still moves.
12. **`KS15.1`'s ingredient 4 — engaging the difficult**, gated by `systems_perspective` rather than by anxiety (`KS15.11`). Nothing in the spec makes an agent approach its hardest tie.

---

## 5 · Regrades and scope corrections

| Item | Was | Now |
|---|---|---|
| `resource_pressure` | `[T]` via Ch18 + `KB10` | **Confirmed and sharpened** — `KS14.1` gives Bowen's three conditions as a numbered list, *and* records that Bowen was **certain of the regression, uncertain of its driver**. Carry the uncertainty. |
| `resource_pressure` shape | monotone | **Not monotone.** `KS14.7`'s Galápagos finches regressed under **superabundance**. The dial needs an optimum. |
| Societal I-T balance | unquantified | **Optimum at parity (50-50)**, regression = a 5–10 point shift (`KS14.3`). Parity is the optimum, so more individuality is *not* better. |
| `M1.A.11` channel names | physical / **emotional** / social | **physical / mental / social** (`KS23.11`) — "emotional" collides with `M1.A.0`, since everything in the theory is emotional. Kerr proposes the rename himself. |
| Ninth concept | recorded as unconsolidated | **`E3` documented at source** (`KS24.1`) — proposed 1980, renamed once, presented twice, never completed, and its origin was two colleagues' cancer remissions. |
| Part III / IV clinical claims | — | **Blanket `[K-ext]`** per `KS15.12`: Kerr states he cannot prove his conclusions on cancer, autism, sociopathy, schizophrenia, depression or the addictions. |

---

## 6 · What the model must *not* take from this book

- **No disease biology.** `M11.F.4` (from `KB11`) stands. Ch 23 is Kerr's own unproven proposal and he says so; his cancer study (`KS23.7`) is a single-rater retrospective quartile comparison whose *method* is recorded as a caution, not a result.
- **No cancer, autism, sociopathy or addiction mechanisms** — the author's own exclusion (`KS15.12`).
- **No diagnostic categories.** `KS20.5` and `KS22.5` make psychotic and psychopathic one continuum differing in discharge direction; `KS11.10` makes level orthogonal to symptom type.
- **No belief adjudication.** `KS24.2`: assess the **function** of a belief, never its validity.
- **No claim that a systems account displaces treatment.** `KS23.10` — Kerr treated his own ulcer pharmacologically and says so.
- **No retrospective-fit claims.** `KS20.12`'s mold metaphor is a methodological caution: a framework that fits a case perfectly in hindsight is not thereby predictive. Part III citations carry it.

---

## 7 · Two items to put to the user

1. **`A1`** — the eight-to-ten-generations range in print (§1.1). Recommendation: withdrawal of the *rate* stands; record the range as a shape constraint on `M11.C.6`.
2. **Expressed emotion** (`KS25.9`) — hostility, over-involvement and critical comments predicting relapse **even on medication**, and low EE predicting low relapse **off** medication. All three are computable from the model's event log and it is the strongest `M11.C.8` candidate in the corpus. **But this is Kerr's second-hand summary** and the effect sizes matter. Recommend checking the primary EE literature before any criterion depends on it.
