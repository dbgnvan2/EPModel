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
| Ye et al., TRAILS (2605.18890) | **full text** through Appendix D (PDF); Appendix E not reached | C12, C21, C25, C33, C36, C37 |
| Li & Tao 2026 (2603.00113) | **full text** (HTML v2) | C5, C6, C7, C32 |
| He 2026, VISA (2607.28027) | abstract and search extracts only; the PDF has no machine-readable text and no HTML version was found | C9, C10, C15 |
| Sachdeva & van Nuenen (2510.10002) | **full text of v4** (COLM 2026, 27 Aug 2026, retitled "Interaction Protocol Shapes Moral Judgment in Multi-Agent Debate") | C5, C30, C34, C40 |
| Buitrago López et al. 2026 (2606.12369) | **full text** (HTML v1); figure cell values not extractable | C11, C39; LLM-line paragraph |
| Kurz 2025 (2512.18016) | **full text** (PDF) | C14, C31, C32 |
| Blando et al. 2026 (2604.04543) | **full text** (PDF) | C25, C26, C27 |
| Wang et al. 2026 (2608.06485) | **full text** (PDF) | LLM-line paragraph; X1, X2 |
| Li et al. 2026 (2608.24912) | abstract and two search extracts only | LLM-line paragraph; X3, X4 |
| Liu 2026, SEAA (2609.17331) | **full text** (HTML v1) | `DESIGN_LESSONS` §7; the drafts E-DR, E-SH, E-RM, E-RE |

**Thirteen of the fifteen papers were read in full.** The first five carry the two revision-10 items that change the
model itself (C8 → `M4.C.9`, C9 → `M4.B.2`). The second batch (addendum, same day) covers the remaining papers. VISA
and Li et al. are the two I could not read in full.

## Summary

**The readers' reports are accurate.** For all thirteen papers read in full (five in the first batch, eight in the addendum), section A of the reading report (the
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
| **C21** (TRAILS) | memory representation had *small* effects (most shifts < 0.2 payoff, < 0.1 cooperation); persona format was the large one | memory representation cited as the case that "moved outcomes" |
| **C40** (Sachdeva) | the interaction terms lower AIC by 2,094 (77,734 → 75,640) | "interact positively (AIC drops from 92,233 to 75,640)", the whole nested sequence |
| **LLM-line paragraph** (Wang) | median \|Δ\| 0.02 against the paper's own band 0.035–0.14; 10–32% of responses overshoot | "about ten times smaller than human bands" |
| **LLM-line paragraph** (Li) | the malicious persona pushes two categories below the models' *own normal values* | pushes them below the *human baseline* |
| **`DESIGN_LESSONS` §7.5** (SEAA) | prototype agents do not interact; LLM narrators are *given* their labels and gaps | differentiation, self-narrative and social topology all emerge from "reinforcement plus observation" |

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

### Second batch (addendum): the other ten papers

**TRAILS — report accurate; C21 cites the wrong perturbation.** App. C.1 (Mann–Whitney U because some conditions have
zero run-level variance; Holm within each metric), C.2.1 and C.4 (parse-validity logged; unparseable output falls
back to DEFECT after one retry), C.3.3 (degree-preserving swaps; hubs reassigned with degree sequence and homophily
band fixed) and §4 (76 pp in gpt-5.2, ~77 pp in claude-haiku, ~36 pp in gemini, ~1 pp in deepseek) all match C12,
C25, C33, C36 and C37. C12 also cites App. E.1.7, which I did not reach. **C21** cites §4.1 P3 as the case where
representation "moved outcomes while memory content was identical". The paper reports that memory representation had
only small effects and did not shift the equilibrium. The reading report states this correctly (its line on P3); the
candidate dropped it. The representation-level result that did move outcomes is persona format (P1: the same content
as prose, bullets or a table; 76 pp in the two-agent game). Re-cite C21 to P1. The paper is explicit that its
perturbation surface is textual and LLM-specific, so the transfer to rounding and summation order stays `[INFERENCE]`.

**Li & Tao — accurate.** Definition 4.1 (Sch, Vis, D₀ as named parts of the simulator), Action 1 (versioned,
inspectable, logged), Action 2(3) and Action 3, and §3.2.3 (independent attribute sampling breaks joint structure;
identical aggregate statistics can encode different mechanisms) are all as stated in C5, C6, C7 and C32, and all
are correctly graded `[ARGUED]`. The paper's "environment" is institutions and platforms; what transfers to a
twelve-person family is the scheduler and visibility objects, which is what the candidates take.

**VISA — still not read.** Search extracts confirm rule r4 (a composite execution step is allowed if declared) and
rule r8 (a variable-count agent set owns a creation and a removal function). I could not verify r13 or r14. C9, C10
and C15 are graded `[ARGUED]` and none needs VISA's authority: C15 (disposition at death) follows from the spec's own
conservation invariants.

**Sachdeva & van Nuenen — v4 exists; numbers accurate; one misattributed figure.** This corrects my earlier row:
v4 is on arXiv (COLM 2026, 27 August 2026) under a new title, as the report says. Change-of-verdict rates, first-round
consensus of about 90% versus 40% by speaking order, the >70% NTA steering in one three-way order, Table 1
(γ_within 2.22 for GPT-4.1, 0.03 for Claude) and App. D (Gemini's verdict distribution shifted by the debate framing
alone) all match C5, C30 and C34. **C40** says the two quantities "interact positively (nested-model AIC drops
from 92,233 to 75,640)". 92,233 is the fixed-effects-only model. The interaction terms account for 77,734 → 75,640;
the report has this right. Two caveats the candidates omit: the authors ran each experiment once, and adding the
interaction or a debate random effect changes the magnitudes considerably (Claude's inertia 1.51 → 0.87; Claude's
within-round conformity 0.05 → −0.20). Only the ordering between models holds. That limits C40's
"statistically separable".

**Buitrago López — text accurate; figure values unverified.** Every number in the text matches: mean JSD 0.212 over
nine configurations; prompt means 0.148 / 0.172 / 0.317; best prompt differs by model; 135.1× to 1,337.1× slower,
mean 563.3×; no repetitions and no variance reported. The report's cell counts (for example LLaMA v1 Passive read
5,124 → 1,071; follow driven to 0 in most LLM cells) come from heatmap images I could not extract. The text says
only that rare actions were "reduced". The candidates' "drove the rare actions to zero in most cells" and "inverted
the rare/common actions" rest on those figures, and I have not checked them. The contextual mask is described only as
an input to the v3 prompt ("final normalized probabilities after masking"). The paper implies that the FSM masks and
renormalises but does not specify the procedure.

**Kurz — accurate; the idea is Hegselmann's.** Lemma 3 (finitely many ε-intervals with identical trajectories; switch
points are pairwise distances), Lemma 7 (realisability as an LP), Table 1 (51,505 unit interval graphs at n = 12) and
Example 5 all match C14, C31 and C32. As with Buffalo, the core result is not new: ε-switches and their algorithm are
from Hegselmann (2023, *JASSS* 26(4)). Kurz proves Hegselmann's conjecture and gives a breadth-first version. Cite
Hegselmann with C31. On C32: Example 5 is n = 4, and the paper's next sentence says the outcome is "different for
larger numbers of agents or tighter constraints". The input bands are ±0.1 on opinions of 2–5 (about ±2–5%), and ±10%
only on ε. The report's "n ≈ 4–12" and the candidate's "±10% bands on inputs" both extend it.

**Blando — report accurate; the paper's inference is not.** C25, C26 and C27 describe the practice correctly (Welch
test per step, batches of 30 until every interval is below δ, non-converging configurations reported as such). The
paper reads ρ = 3.0 versus 5.0, which "does not reject equality at any time step (power = 1.0)", as a saturation effect.
A non-rejection does not establish equality, and a power figure with no stated effect size does not change that. The
candidate quotes this as "SHOWN as practice" without flagging it. EPModel's `M11.4a` (a null must carry an
equivalence bound) is the correct rule and already excludes this reading. The paper also applies a separate test at
each of 21 time points with no multiplicity correction. C25's Holm requirement covers that.

**Wang et al. — report accurate; two of the paper's own claims are weaker than stated.** 0.19 versus ~0.7, 14 of 27
matched and 13 reversed, retirement reversed by every model, the agreeableness pull, ρ ≤ 0.105 against scenario
decisions and the three-turn horizon all match. Two problems carry into the LLM-line paragraph and X2:
- *"About ten times smaller."* This is the annotation in the paper's Figure 1. The body gives median |Δ| = 0.02
  against a human reference band of 0.035–0.14, which is 1.75 to 7 times smaller. 9.9–31.6% of responses overshoot the
  band. Cite the in-band rate (11–16%), not a ratio.
- *Heterogeneity collapse.* The paper compares the SD of *change scores* across personas (σ_LLM = 0.19) with the human
  SD of trait *levels* (0.5–0.8, from Bühler 2024 and Roberts 2006). These are different quantities, so the
  "three- to four-fold compression" is not established by that comparison. X2 ("collapse relative to baseline SD")
  inherits the same mismatch. I do not know the human SD of individual change after these events; the comparison
  needs it.

**Li et al. — only partly read.** Confirmed: 18 models; prompt-level interventions change the size of the bias but not
its direction; the calibration divides the persona-conditioned next-token distribution by the neutral-persona one to
the power α. Two wording issues in the LLM-line paragraph. The paper says the malicious persona pushes fairness
optimism and emotional softening below the models' *normal values*, not below the human baseline. And the method
needs the token probability vector, which the paper says black-box APIs supply; "needs logits" suggests model access.
I did not confirm the report's "needs a human reference".

**SEAA — the design-lessons numbers are accurate; §7.5 attributes too much to the loop.** The consolidator's §7
reports the paper's numbers correctly. Three corrections:
1. **The prototype agents do not interact.** In Listing 2 the group mean enters only an EMA "others-model", which
   nothing reads. Each agent's reward depends on its own state and its own random walk. Differentiation therefore comes
   from positive feedback on independent noise. §7.5 says "reinforcement plus observation alone". Observation plays no
   causal part. The null-model argument (E-RM) is stronger for this: differentiation needs no interaction at all.
2. **The narratives and the deliberation topology are supplied, not emergent.** Appendix A hands each LLM agent its
   dominant-state label, its trait numbers and its signed gaps to the group, and tells it to "stay true to this
   disposition". Control agents are all labelled "none (no lock-in)". An agent labelled *impulsive* among agents
   labelled *pessimistic* and *calm* becoming the outlier follows from the prompt. §7.5 (b) and (c) should not be
   counted as outputs of the mechanism.
3. **§7.7(3):** the control is not "no dynamics at all". It keeps the HMM transitions under P_base and the experience
   walk; it lacks only the preference update. The manipulation-check objection stands.
The paper also describes control occupancy 0.40 → 0.25 as "unchanged". E-DR, E-SH and E-RM rest on design reasoning
and are unaffected.

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
5. **A second pass.** Give the candidate lines the second pass the corpus got. Across thirteen full reads the reading
   reports were accurate every time; the errors are in the candidate lines, the LLM-line paragraph and design lessons
   §7.5. Corrections from the second batch: re-cite C21 to TRAILS P1; correct C40's AIC figure and add Sachdeva's
   single-run and magnitude caveats; add Kurz Example 5's own scope sentence to C32 and cite Hegselmann 2023 with C31;
   flag Blando's saturation reading as unsound; in the LLM-line paragraph replace "ten times smaller" with the in-band
   rate and correct the Li wording; redraft X2 or drop it; rewrite `DESIGN_LESSONS` §7.5 per the SEAA note above.
   VISA and Li et al. remain unread in full.
6. **Cite established methodology beside the preprints** where it exists: common random numbers and streams (C1–C4),
   ODD (C10), global sensitivity analysis (C31, C37, C38).
