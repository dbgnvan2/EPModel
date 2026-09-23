---
tags: [model-bt, review, method-literature]
status: external review — not a requirement, not approved text
date: 2026-09-23
reviews: SWEEP_READING_REPORTS_2026-09-20.md, SPEC_CANDIDATES_from_preprints_2026-09-20.md and its plain-language companion, and model_explainer.md §18 (branch spec-rev10-draft), against the papers themselves
author: Claude (claude.ai chat session), for Dave
---

# The papers, read against what was written about them

## What was read

The PDFs are not in the repository (they are untracked). This review fetched them from arXiv. Everything here is
limited to the papers I could open, and each row of the table says how much of each paper I read.

| Paper | Read | Candidates it supports |
|---|---|---|
| Buffalo, Pearson & Klein 2026 (2603.11084) | **full text** (HTML v1) | C1–C4, C28 |
| Zhou et al., PIMMUR (2509.18052) | **full text of v3** (6 April 2026). The reader read v4, which I could not open; a version with the 576-study audit exists (it appears in a search-engine copy of the arXiv PDF) | C9, C22, C23, C35, C42 |
| Holland, Saynor, Svingen & Luo 2026 (2607.29546) | **full text** (HTML) | C8, C29, H2 |
| Prasad 2026 (2607.07753) | **full text including supplement** (HTML) | C13, C16, C17, C18, C24, C38 |
| Kalluri 2026 (2603.01189) | **full text** (PDF) | C19, C20, C23, C41 |
| Ye et al., TRAILS (2605.18890) | abstract only | C12, C21, C25, C33, C36, C37 |
| Li & Tao 2026 (2603.00113) | abstract and introduction | C5, C6, C7, C32 |
| He 2026, VISA (2607.28027) | abstract only | C9, C10, C15 |
| Sachdeva & van Nuenen (2510.10002) | abstract only (v1/v2) | C5, C30, C34, C40 |
| Buitrago López et al. 2026 (2606.12369) | title and date only | C11, C39 |
| Kurz 2025 (2512.18016) | model definition only | C14, C31 |
| Liu 2026, SEAA (2609.17331) | abstract only | the M17 drafts E-DR, E-SH, E-RM, E-RE |
| Blando et al. 2026 (2604.04543); Wang et al. 2026 (2608.06485); Li et al. 2026 (2608.24912) | **not checked** | C26, C27; X1–X4 |

**Five papers were read in full.** Between them they carry the two revision-10 items that change the model itself
(C8 → `M4.C.9`, C9 → `M4.B.2`), and the evidence behind C1–C4, C16–C24 and C28.

## Summary

**The readers' reports are accurate.** For all five papers read in full, section A of the reading report (the
summary of the paper) is correct in detail: equations, table values, seed counts, caveats. In two cases the reader
caught the paper overclaiming:

- **Prasad.** The paper says every dose–response is "graded, monotone"; its own Table 2 is not monotone.
- **Kalluri.** The paper reports two different sizes for the meta-analysis it validates against.

**The errors are downstream of those summaries, and they all point one way.** They sit in the readers' own
candidate lines (section B), and in the candidates file, explainer §18 and the plain-language file. In every case
the paper is made to support more than it does:

| Where it went wrong | Paper says | Written as |
|---|---|---|
| **C8 / `M4.C.9`** (Holland) | a *necessary* condition for oscillation (§4.2), and simulation examples | "this channel alone suffices for oscillation **and polarisation** (**SHOWN formally**)" |
| **C8 / `M4.C.9`** (Holland) | a witness weighs its opinion *distances* to the two parties; agents mix at random and have **no ties** | a witness weighs its **ties** to both parties |
| **C9 / `M4.B.2`** (PIMMUR) | an LLM effect the authors attribute to the model recognising balance theory | support for a rule-based information-access rule |
| **C19 / `M11.4c`** (Kalluri) | Spearman ρ = 0.833 against benchmarks that contain ties | "ranked the eight correctly" |
| **C23** (Kalluri as "worked example") | a parameter calibrated to a target that the output then **missed** | a target "then reported as a validation result" |
| **Plain-language C37** (TRAILS) | 76 points versus 1 point between two **LLMs** | presented in a low- versus high-differentiation family example |

This is the direction the project already documented for the first corpus pass: sources made to look "more
quantitative and more decided" than they are. It recurred here in the step from a correct summary to a candidate.
None of these candidates had a second pass.

## Paper by paper

### Buffalo, Pearson & Klein — accurate; the idea is established rather than new

The report and C1–C4 and C28 match the paper:

- Proposition 1 and Corollary 3.1 (§3.3);
- the placebo test, with "identical outcomes do not confirm" (§2.3);
- slot- versus dyad-keying as an exchangeability assumption (§4.2);
- the three key-design guidelines, including stable offspring identifiers (§4.3);
- principal strata (Table 1);
- the consequences for variance, sensitivity analysis and mediation (Appendix A).

Two points of calibration:

- **Proposition 1 has a "proof sketch".** It is a formalisation, not a deep result.
- **The paper calls these "well-known issues" that it frames in a new way.** It cites Stout & Goldie 2008,
  Kaminsky et al. 2019, Klein et al. 2024 and L'Ecuyer et al. 2002. The candidates file's "strongest single finding
  of the sweep" overstates the novelty, not the correctness. C1 is sound and would be better cited to that
  literature as well.

### PIMMUR — numbers accurate; the paper labels its own manipulations inconsistently

The social-balance result (60.7% → 34.4%, n = 640), the ablation and the 1.77× figure all match v3.

**The inconsistency.** §2.3.2 files the god-view (agents told the global relationship graph) under *Interaction*.
§4.3.2, which describes what was actually changed, files "fact-feeding" of dyad relations under *Unawareness*, and
uses *Interaction* for replacing binary labels with natural-language stances. The ablation's "Interaction alone had
negligible impact" therefore does **not** mean removing the god-view was negligible. On the §4.3.2 labels, removing
it is part of the −26-point Unawareness arm.

**The authors' explanation of that arm is LLM-specific:** agents that were told the relations invoked Heider's
theory in their reasoning and performed it. The result therefore does not bear on a rule-based agent in either
direction. *This corrects the earlier review* (`REVIEW_spec_candidates_material_2026-09-23.md` §1, now amended),
which read the reading report's summary of the ablation as evidence against C9.

**Scope, from v3 itself.** PIMMUR is "not … a rigid, universal checklist for all computational modeling"; it is a
standard for simulations that make claims about real human collectives. Its Realism principle classes validation
against a theory, rather than against data, as circular. EPModel validates against Bowen's theory by design, so
citing PIMMUR as an authority imports that objection.

### Holland et al. — report accurate; C8's evidence line overstates it twice

**The report is accurate.** Its account of equations (2)–(5), the thresholds, Corollary 4.2 (±1 are sinks), the
necessary condition of §4.2, and Fig. 6's N = 10 / N = 100 grids all match the paper. The report itself says the
oscillation condition is "necessary not sufficient".

**Overstatement 1 — what was shown.** The reader's candidate H1 then says §4.2 and §5 show the witness channel
"alone suffices for oscillation and polarisation. SHOWN formally."

- **Oscillation:** §4.2 proves only a necessary condition. With reciprocity set to zero, the paper shows by
  simulation example (Fig. 4) that most propensities still oscillate.
- **Polarisation:** it is not shown for the witness channel alone. The paper attributes it to a near-neutral shared
  perception of the environment combined with high reciprocity **and** high retribution (§5.1, §5.2, §6).

**Overstatement 2 — what the witness depends on.** Holland's agents mix at random with no persistent ties (the
report says so). The witness's weight depends on the **opinion distances** between the witness and each party
(equations 4–5). "The witness's ties to both the sender and each target" is the reader's substitution, taken from
the Bowen side.

**Consequence.** `M4.C.9` may well be right, but its support is triangle theory in the corpus, not Holland. Explainer
§18.3's "`[SHOWN]` Holland … the witness channel alone was enough to produce oscillation and polarisation" should be
corrected.

**The C29 transfer is also weaker than written.** The candidates call the N = 10 / N = 100 result "direct evidence"
for EPModel's regime boundaries. It comes from a random-mixing, single-scalar opinion model with valence drawn
independently of the agents. It is suggestive, not direct.

### Prasad — report accurate, and it caught the paper's overclaim

These all match:

- the bistable checkpoint bonus without habituation, and the λ^k fix (C13);
- the matched-magnitude corruptions collapsing avoidance from 1.00 to 0.00–0.20 over 10 seeds (C17);
- remission versus resistance, on 5 seeds with ±0.39 intervals for the resistant cells (C18);
- the saturated-assay nulls on LavaGap and MiniWorld (C24);
- the interaction residuals of 0.82 and 0.50 (C38).

The reader was right that "every disorder shows a graded, monotone dose-response" is not what Table 2 shows. Impulsivity
runs 0.61 → 0.56 → 1.00 → 0.94; addiction 0.02 → 0.00 → 0.50 → 0.76; depression is a step.

**One point to record.** C18's learning-on versus learning-off design is the reader's own. The paper's contrast is
knob removed versus knob kept, with learning on in both. The transfer is reasonable, but it is `[INF]`.

### Kalluri — C19 and C23 misstate the paper

**C19 (ordinal criteria).** The candidates file says the model "ranked the eight correctly (Spearman ρ = 0.833)";
explainer §18.6 and the plain-language file repeat it. Two problems:

- **ρ = 0.833 is not a correct ranking.**
- **The benchmark cannot be ranked cleanly.** Table 4 assigns eight factors benchmark values taken from six
  meta-analytic categories. Three factors share r = .45, and two share .22.

Compare the two orders:

| | 1st | 2nd | 3rd | … | 7th | 8th |
|---|---|---|---|---|---|---|
| **Model** | reliability (.59) | communication (.41) | collaboration (.39) | … | expertise | tenure |
| **Benchmark** | reliability (.60) | transparency, warmth and communication, tied (.45) | | … | | |

In the benchmark, collaboration ties at .27 for roughly 5th–6th place. The paper overclaims ("correctly reproduced
the relative ranking"), and the project repeated the overclaim.

**C23 ("a worked example of the failure").** Table 3 says the loss rate was "calibrated to produce asymmetry ratio
1.3–1.7". Table 5 then reports 0.13–0.53: the target was **missed**. The paper presents the miss as a "boundary
condition that extends theory", which is spin, but it is not the circularity C23 describes.

The defensible criticism is different: the gain rate was "calibrated to Hancock et al. (2021) trust trajectories"
and the model was then "validated" against Hancock et al. (2021) correlations. The report also noted the two
different sizes the paper gives for that meta-analysis (69 studies, N = 7,769, and 142 studies, N = 7,458); both
appear in the paper.

**C20 and C41 are accurate.** The asymmetry ratios run from 0.53 at 30% reliability to 0.13 at 90%, 0.31 overall,
and 0.069–0.552 across scenarios. Calibration error runs 8.93–52.05.

**The paper overall is weak evidence for a methods standard.** It is titled "Empirically-Validated" on the strength
of interval validity for four of eight predictors.

### Abstract-level checks

- **TRAILS.** The abstract confirms that the 76-point versus 1-point contrast is between model families. This
  settles the plain-language misframing of C37.
- **Li & Tao.** A position paper (one arXiv version is titled "Position: AI Agents Are Not (Yet) a Panacea for Social
  Simulation"). Scheduling and exposure as model content match the candidates' `[ARGUED]` grading.
- **VISA.** The abstract matches: eight tables, nineteen consistency rules, and the AnyLogic reproduction barrier.
  ODD (Grimm et al. 2006, 2010, 2020) is the established standard C10 belongs beside.
- **Sachdeva & van Nuenen.** arXiv lists v1 (October 2025) and v2 (March 2026) as "Deliberative Dynamics and Value
  Alignment in LLM Debates". The report cites "v4, COLM 2026" under a different title. I could not confirm v4 or its
  numbers.
- **Kurz.** The model is synchronous Hegselmann–Krause averaging. C14 and C31 are transfers from a deterministic
  one-dimensional model.
- **SEAA.** Single author, independent researcher, not peer reviewed. Four `M17` drafts come from it. The consolidator
  read it himself (`DESIGN_LESSONS` §7), which is better provenance than a sub-agent report. The drafts themselves —
  dose–response, shock and recover, a rival-mechanism arm — are standard methodology and need no special authority.

## What this changes

1. **`M4.C.9` (witness appraisal).** Re-source it to the corpus's triangle findings. Remove "SHOWN … Holland" and
   "alone … polarisation" from explainer §18.3 and the candidates file.
2. **`M4.B.2` (information access).** Remove the PIMMUR number as support. The rule rests on the belief-layer design
   argument and VISA r14 (`[ARGUED]`). This is independent of its conflicts with approved requirements
   (`REVIEW_spec_rev10` §A4).
3. **`M11.4c` (ordinal criteria).** Correct the Kalluri description. The form stands or falls on the corpus: ledger
   L07.4 supplies a mechanism ranking (`REVIEW_spec_rev10` §G1).
4. **C23 / `M10.B.4`.** The requirement is sound. Replace the Kalluri example with the accurate one (calibrated and
   validated against the same meta-analysis), or drop it.
5. **A second pass.** Give the candidate lines the second pass the corpus got. Five of five full reads confirmed the
   summaries and found the overstatements in the step after them. The nine papers read only at abstract level, or
   not checked, have had no pass of this kind; their numbers rest on one reader each.
6. **Cite established methodology beside the preprints** where it exists: common random numbers and streams (C1–C4),
   ODD (C10), global sensitivity analysis (C31, C37, C38).
