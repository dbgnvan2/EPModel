---
tags: [model-bt, review, method-literature]
status: external review — not a requirement, not approved text
date: 2026-09-23
reviews: the material that produced the revision-10 spec candidates (branch spec-rev10-draft)
author: Claude (claude.ai chat session), for Dave
---

# Review of the material behind the spec candidates (C1–C42)

## Scope

**Read in full:**

- the readers' brief and task (`sweep_readers_brief_2026-09-19/EPMODEL_BRIEF.md`, `READER_TASK.md`);
- `SPEC_CANDIDATES_from_preprints_2026-09-20.md`;
- `SPEC_CANDIDATES_plain_language_2026-09-20.md`.

**Read in part:**

- `SWEEP_READING_REPORTS_2026-09-20.md` — its structure, the PIMMUR report in full, and targeted passages for the TRAILS, Holland, Kurz and Kalluri numbers used below;
- `PREPRINT_SWEEP_2025-09_to_2026-09.md` — §0 method, plus the note on how the fourteen were chosen;
- `DESIGN_LESSONS_model_design_papers_2026-09-17.md` — the outline and §8;
- `sweep.py` — the header.

**Checked against the papers themselves:**

- the arXiv listing and abstract of Buffalo, Pearson & Klein (2603.11084);
- the arXiv listing and earlier-version abstracts of PIMMUR (2509.18052).

**Not checked:** no other paper was opened. Where this review says what a paper found, it relies on the project's own reading reports, and says so.

## Bottom line

**The pipeline is well built.** Its strengths:

- There is a written brief.
- Each paper was read in full.
- Every finding is tagged `[PAPER]` or `[INFERENCE]` and cited to a section.
- Candidates carry an evidence grade (SHOWN / ARGUED / INFERENCE) and a cost/risk line.
- The consolidation step says what it did and did not do.
- The LLM-line conclusion — nothing here argues for relaxing `M3.D.6` — follows from the reports.
- The strongest candidate, C1 (keyed random draws), checks out against the paper's abstract. It also has earlier independent support, which makes it standard practice rather than a 2026 novelty: Stout & Goldie 2008, and Klein et al. 2024, are both cited by Buffalo et al.

**Three weaknesses matter for what reached the spec:**

1. **C9's headline evidence does not support C9, and the project's own reading report says so.** The number travelled unchanged into the candidates file, the explainer (§18.3) and the plain-language file. `M4.B.2` rests on it.
2. **Coverage was checked by keyword search against the spec only**, not against the explainer or the ledger. That is how C9 came to be recorded as "No conflict", and how C19 concluded the corpus has no mechanism ranking.
3. **The method literature got one reading pass.** The project's own experience with the corpus is that a single pass over-reads in a consistent direction; pass 2 withdrew nineteen findings. Nothing equivalent was done here.

**The plain-language file has errors of its own.** It misstates two spec objects: it calls `WITHHOLD` "do nothing", and it lists a `FIGHT` move that does not exist. It drops every evidence grade and every LLM-versus-rule-based caveat. And it restates two statistical results incorrectly.

## 1. C9: the headline number points the other way

**What the candidate says.** C9 (the no-god-view rule, now spec `M4.B.2`) cites PIMMUR §2.3.2: when LLM agents had to infer other dyads' relations instead of being handed the relationship graph, balanced states fell from 60.7% to 34.4%. The candidate labels this "SHOWN for LLM agents". Explainer §18.3 repeats it ("from about 61% to about 34%"), and the plain-language file presents it as showing "how much this matters".

**What the reading report says.** In the PIMMUR report (Part 5, item 5, "Social balance"), the reader records the paper's ablation (§2.4, Fig. 4e):

- **Interaction alone** — the god-view removal — **had a negligible effect**;
- adding Profile moved balanced states by −14 points;
- adding Unawareness moved them by −26 points.

**Why that matters.** The drop is carried by agent heterogeneity and by the model recognising the experiment, not by denying the god-view. The candidate text even says "with the Unawareness and Profile principles carrying the effect" — and still cites the number as support for a rule about information access. The paper's own ablation is weak evidence *against* the claim that the god-view matters, in LLM agents.

**Two further cautions from the same report**, neither carried into the candidates file:

- The reader writes that **PIMMUR explicitly places theory-driven agent-based models outside its scope**; "every analogue above is mine, not the paper's". C9, C22, C23, C35 and C42 all draw on PIMMUR.
- The Minimal-Control coding rests on LLM judges with a mean κ of 0.49.

**Consequence.** C9 may still be a good rule. The design argument — the belief layer is pointless if the policy can bypass it — stands on its own, and VISA r14 gives an `[ARGUED]` source. But its `[SHOWN]` support should be struck, and the rule should be argued on design grounds. See also `docs/REVIEW_spec_rev10_2026-09-23.md` §A4, on the approved requirements it conflicts with.

## 2. Coverage checked against the spec by keyword only

The consolidation step re-checked each reader's "already covered" judgement by "text search" of the spec (§0, §7). It did not check against the explainer or the ledger, and keyword search cannot find a conflict that uses other words. Four consequences:

| Candidate | What keyword search missed |
|---|---|
| **C9** ("No conflict") | Approved requirements that read other people's true state, under words none of the searches used: `M5.F.1` (receivers read the actor's hidden `outside_ness`), the `M5.C` gates ("system calm", "marital distance high"), `M8.1`'s `fused_into`, and `M4.D.2`'s triangle position |
| **C19** (ordinal criteria) | The candidate says it applies "where the corpus supplies an ordering of effect strengths across three or more mechanisms", and revision 10 concluded none exists. The ledger has one: **L07.4** — course determined first by the spouses' dynamics, second by relationships outside the ego mass, third by fusion intensity. `ch07.md` quotes it twice |
| **C7** (witnesses computed) | Treats `M8.5`/`M8.6` as a visibility rule. It is an **alignment** rule: whether a third party is a neutral witness or has taken a side. Who overhears and whose side they are on are different questions |
| **C16** (four-level sweeps) | Names `chronic_anxiety` as a graded parameter to sweep. The spec derives it every slow tick (`M1.A.7a`); the swept quantity would have to be `programmed_reactivity` |

The readers flagged this limit themselves: they had the brief, not the spec.

**The brief itself is thin and partly stale.** It describes the invariants as "anxiety conserved and redirected, never destroyed" — the global wording the other review finds unsatisfiable (§A3). It says nothing of:

- the `basic_level` estimator;
- `systems_perspective`;
- the two-dimensional `outside_ness`;
- the `M8` predicate.

## 3. One reading pass, unverified

The corpus work taught the project that a single reading pass over-reads, "consistently in one direction", and it built a second pass plus a primary-source sweep to catch it. The method literature got neither.

- **Fourteen papers were read once, by five sub-agents working from `pdftotext`.** The consolidator says nothing was added from memory and that every candidate traces to a reader report. It does not say that any reader's number was re-checked against a paper.
- **The revision-10 "eight-word quotation" check was a copying check, not an accuracy check.** It compared text against the PDFs to catch reproduced wording, not to confirm what the papers say.

The C9 case in §1 is the kind of error a second pass exists to catch. A reported effect was attached to the wrong component — and the reader had the right attribution sitting in the same paragraph.

**Selection.**

- 1,115 papers were harvested and screened by sub-agents on abstracts.
- 128 were selected, and fourteen of those were read in full.
- The window was one year of preprints (2025-09 to 2026-09).

The established methodology behind several candidates predates the window and is peer-reviewed:

- common random numbers and separate random streams (C1–C4);
- the ODD protocol (C10);
- global sensitivity analysis (C31, C37, C38).

Citing it would give those requirements firmer grounds than a single 2026 preprint.

**Source class.** Most of the evidence concerns LLM agents: TRAILS, PIMMUR, Sachdeva, Buitrago López, Wang, Li. The candidates grade the transfer as INFERENCE, which is correct. But the markers in the spec name a paper and first author (⟦proposed rev10 · C35 · Zhou 2026⟧), which reads as citation to authority.

## 4. The plain-language file

The file was written as an explanation for the owner, and in places it is clearer than the source. It also contains errors of its own.

**Misstatements of the spec:**

- **The glossary's move list includes `FIGHT`.** There is no such move; the spec's move is `CONFLICT` (`M5.A.1`).
- **The glossary defines `WITHHOLD` as "do nothing this week".** The spec says the opposite: `WITHHOLD` is the automatic move "computed, detected, and not emitted", and "**a withheld move MUST still change tie state**; an implementation in which not acting is a no-op cannot represent the two canonical instances" (`M4.D.1b`).
- **It glosses chronic anxiety as "accumulated anxiety".** The spec derives it each slow tick from programmed reactivity, the field and the person's functioning position (`M1.A.7a`).
- **Its module map is wrong in two places.** "M8 triangles": `M8` is the live-position predicate, and triangles are `M1.C`. "M7 life-stage": `M7` is the slow tick, of which life stage is one part.

**Evidence presented without its caveat:**

- **C9** presents the 60.7% → 34.4% drop as showing "how much this matters". It omits that the agents were LLMs, and the report's attribution of the effect to other principles (§1 above).
- **C37** says "the source saw 76 percentage points in one model and about 1 in another", in an example about low- versus high-differentiation families. In TRAILS the 76-point versus 1-point contrast is between **LLMs** (gpt-5.2 against deepseek-v3). The candidates file marks the transfer to configurations as INFERENCE; the plain file drops that.
- **C34 and C35** give LLM results — framing shifting a verdict; 1.77× more balanced outcomes — with no statement that they are about LLMs.

**Statistical misstatements:**

- **C27** says "a non-rejection at power 1.0 = genuinely no difference". Power is computed for an assumed effect size. High power at a non-rejection says an effect of that size would probably have been detected, not that there is no difference. The reading report records Blando et al. reading it as saturation.
- **C13** says geometric decay "produces the graded, in-between curve **the theory expects**". Bowen's corpus says nothing about habituation of relief; the graded curve is Prasad's reinforcement-learning result.

**An example that crosses a spec boundary.** C18's worked example uses "a spouse's illness" as the exogenous spell. The spec treats illness primarily as an **output** of accumulated load, and only mixed illness as partly exogenous (explainer §9.2). The example therefore picks the case the spec warns about.

**What the file omits.** It drops all evidence grades (SHOWN / ARGUED / INFERENCE) and all coverage statuses. Its closing note says this was deliberate, to keep it readable. But the owner's decisions of 2026-09-22 were taken on this material, and the grade is the information that says how much weight a candidate can bear. A one-word grade per item would cost little.

## 5. Smaller points

- **C29** cites Holland's N = 10 versus N = 100 regime maps as "direct evidence" that EPModel's regime boundaries flip under marginal constant changes. That is a transfer from a random-mixing opinion model with no network. It is suggestive, not direct.
- **The C18 test is well chosen.** "No persistence rule is written; persistence is emergent" is the form of test the other review (§A1) finds scarce in `M11.C`.
- **C1–C4 would stand on standard simulation methodology alone**, and the spec would be better served citing that methodology than one preprint.
- **The candidates file's C9 cost line says "No conflict".** This should be corrected wherever it is quoted.

## Recommendations

1. **Correct C9.** Strike its `[SHOWN]` support, in the candidates file, explainer §18.3 and the plain-language file, and restate it on design grounds.
2. **Carry PIMMUR's scope exclusion** into every candidate that draws on it (C9, C22, C23, C35, C42).
3. **Re-run the coverage check for C7, C9, C16 and C19** against the explainer and the ledger, not the spec's text alone.
4. **Commission a second pass over the fourteen reports.** An independent reader should check each number that reached a candidate against the paper — the same discipline the corpus got.
5. **Where a candidate rests on standard methodology, cite that literature** alongside or instead of the 2026 preprint.
6. **Fix the plain-language errors in §4, and restore a one-word evidence grade per candidate.**
