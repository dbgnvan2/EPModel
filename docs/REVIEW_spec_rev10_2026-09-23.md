---
tags: [model-bt, review]
status: external review — not a requirement, not approved text
date: 2026-09-23
reviews: docs/bowen_agent_model_spec_v2.md at 2.0-draft revision 10 (branch spec-rev10-draft, commit cff02c9)
author: Claude (claude.ai chat session), for Dave
---

# Critical review of the v2 spec, revision 10

## Scope — what was read and what was not

**Read in full:**

- the spec at revision 9 (`main`), and the revision-10 diff (`spec-rev10-draft`) — 518 IDs, about 46,000 words;
- `model_explainer.md` at revision 10, §1–§18;
- `docs/theory/_LEDGER.md` for the 22 chapters (Ch01–Ch22, all 149 findings, lines 1–1998), including its header sections.

**Read in part:**

- `agent_model_proposal.html` §0–§4 and §9;
- `docs/theory/_STATUS.md` (the handoff block);
- `docs/theory/_CONVERGENCES.md` C3;
- the ledger's `KS-A` and `U8` entries;
- candidate C9 in `papers/SPEC_CANDIDATES_from_preprints_2026-09-20.md`;
- keyword searches of `DECISIONS FOR APPROVAL.md` and `_RESOLUTIONS.md`.

**The chapter extractions** (`docs/theory/ch01.md`–`ch22.md`, about 444,000 words) were **not read end to end**. They were searched, passage by passage, for every item section G relies on, and each quoted phrase in section G was located in the extraction named.

**Not read:**

- the corpus itself;
- the 2019 and 1988 book extractions, beyond the ledger's summaries;
- the preprints behind revision 10.

**What this means for section G.** It checks the spec against the project's own ledger and extractions, **not against the primaries**. Revision 9's sweep read the primaries for quotation existence. Section G asks a different question: whether the spec carries what the chapters, as extracted, actually claim — at the scope and modality they claim it. The 2026 preprints were not verified.

## What the spec is trying to build, as I understand it

The spec describes a deterministic, seeded agent model of one invented three-generation family. Twelve agents — eleven members and one external professional — interact through typed events on a weekly clock, with a yearly slow clock above it. The model is used as a **consistency engine** for Bowen theory: *if the account is right, what follows for a family shaped like this?*

Every magnitude in the model is invented. The trusted output is therefore the **direction of a paired two-arm difference** over ensembles; the invented constants are expected to cancel between arms (explainer §17.1).

There is no external validation. The proposal therefore says the acceptance tests are "the whole of the falsification story" (§9). It describes the model's job as composing "ordinary moves into a non-obvious outcome" (§4.2).

The design's strengths are real:

- every claim is graded, and every invented constant is labelled `[I]`;
- first-pass over-reading of the corpus has been withdrawn;
- guards against misuse are built in (`M11.F.9`, `M15.D`);
- the anti-lever framing is carried throughout.

The findings below are measured against the project's own two aims: falsification through the acceptance tests, and non-obvious composition.

## Bottom line

As a build contract, the spec has three problems that further consistency sweeps will not fix.

1. **Many acceptance criteria restate a mechanism rather than test a consequence.** Since the tests are the whole falsification story, this is the central risk to the project's aim.
2. **The two-arm cancellation argument rests on a premise the project's own turning-point table contradicts.** The premise is that invented constants do not interact with the intervention.
3. **Revision 10's information rule (`M4.B.2`) conflicts with a dozen approved requirements.** The candidates file judged it "No conflict" after a keyword search.

There is also a set of concrete contradictions, data defects and explainer–spec drift. Several of these block specific phases.

Checked against the ledger and the chapter extractions (section G), the spec is faithful on most of the 22 chapters' mechanisms. But:

- it **omits several chapter findings the ledger marks as model requirements**, including Bowen's own two falsifiers of the budget, which are exactly the emergent tests A1 finds missing;
- **six approved requirements are stronger than their chapter source**;
- **the ledger and one extraction still carry five readings the spec has corrected**, including the inverted anger gate.

## A. Structural problems

### A1. Many criteria verify rather than test

For a large share of `M11.C`, the mutation target is the criterion restated as a rule:

| Criterion | Mutation target | What the target says |
|---|---|---|
| `M11.C.10` cut-off begets cut-off | `M7.E.3` | cutoff **MUST** form a positive feedback loop |
| `M11.C.12` curing a symptom raises tension | `M7.D.3` | curing a symptom **MUST** raise family tension |
| `M11.C.15` death destabilises like recovery | `M7.D.3` | the same rule |
| `M11.C.14` technique null under marital distance | `M5.C.1` | the gate: **MUST** produce no improvement |
| `M11.C.33` dependence blocks basic-level rise | `M7.A.1a` | `basic_level` **MUST NOT** rise while dependent |
| `M11.C.32` anger stalls the move | `M5.D.4` | anger **MUST** stall the sequence |
| `M11.C.41` level and stress exacerbate | `M4.C.1` + `M4.D.3` | the divisor, and "rising anxiety **MUST** raise the weight on reactive moves" |

These tests catch implementation errors. They cannot show that Bowen's account **produces** the regularity, because the regularity is written in as a rule. They verify the spec; they do not test the theory.

The proposal's own standard is the right one: ordinary moves composing into a non-obvious outcome. `M11.C.39` meets it — "no persistence rule is written; persistence is emergent".

Revision 10's `M11.1d` (the outcome-directive audit) is the right tool. What the spec does not anticipate is where it leads: applied honestly, `M11.1d` will likely reclassify a substantial fraction of the suite as programmed premises (`M10.C.5`).

**Recommendation.** Run the `M11.1d` audit on paper, for all 41 criteria, before approval. Then split `M11.C` into two tables:

- **mechanism verification** — the criterion is a rule, and the test checks the code implements it;
- **emergent consequence** — no single rule names the outcome.

Only the second table bears on the theory.

### A2. The cancellation premise is contradicted by the project's own table

The premise appears in three places:

- explainer §17.3: "**An invented constant is a nuisance — it does not interact with the intervention**";
- `_STATUS.md`, revision 5, as "the finding worth keeping": "Invented constants cancel between arms because they do not interact with the intervention";
- `M15.D.4`, which builds on it.

§17.1 is more careful: cancellation is clean only for error that does not interact with the intervention.

The turning-point table in §17.3 then lists this entry: "the same move below the **outside-ness threshold** produces the **opposite** sign". That threshold is `[I]` (`M10.C.1`), and `M11.E` calls it the model's most consequential invented constant. So at least one listed turning point is located on an invented constant.

A direction measured at one setting of the constants therefore carries the same sign-reversal exposure that §17.3 attributes to initial conditions.

Revision 10 answers this with `M17.E.1` (the constant sweep, reported as fraction of range), but only in Phase E. Phases C and D still gate on single-point directions.

**Recommendation.**

- Correct the premise in §17.3 and `_STATUS.md`.
- Bring a dominant-constant sweep (`M17.E.2`) into Phase C gating, on the same argument `M13.4` already used to pull adaptive stopping into Phase C. At minimum, cover the `outside_ness` threshold and the appraisal gain.

### A3. `M6.I.6` is unsatisfiable as worded; its intent is narrower

The source claim is about **rerouting**. Convergence C3 reads: "Blocking one channel reroutes the flow; it does not reduce it." Explainer I6 describes it as accounting for **bound** anxiety when a member is removed, a tie is cut or a triangle discharges.

The spec states something broader: "Anxiety is conserved and redirected, never destroyed". `M4.G.2` asserts that every fast tick from Phase C onward. Other requirements then break it on every tick:

- `M4.C.1` **creates** acute anxiety for each event.
- `M1.A.8` and `M3.D.1` step 9 **decay** it toward the chronic floor, and no sink is named for the decay.
- `M11.C.29` says differentiation **reduces** the budget, while `M6.I.1` says the budget is conserved.
- `undifferentiation_budget` and anxiety are two separate quantities, and no stated relation connects them.

**Recommendation.** Restate `M6.I.6` at the scope of C3 and I6: conservation of bound quantities across rerouting and transfer events (removal, cut, discharge, `binder_unavailable`, death). Add one stock-and-flow table stating:

- each conserved quantity;
- its sources and sinks — event appraisal, decay;
- which transfers the invariant covers.

`M11.C.11`, `M11.C.31` and `M11.C.37` cannot be written until this table exists.

### A4. `M4.B.2` (revision 10) conflicts with approved requirements

`M4.B.2` limits what the policy may read to three things: the person's own state, their beliefs, and delivered events. Approved text reads true state in at least these places:

- `M5.F.1`: receivers read the actor's hidden `outside_ness`. Explainer §5.6 repeats this: a hidden state "the *receivers* can read".
- `M5.C` gates on "system calm", "marital distance high" and "decision ownership held externally".
- `M8.1`'s `positions_live` reads `fused_into` across the group.
- `M4.D.2` reads the person's position in the active triangle, which `M1.C.3` computes from true tension.

Candidate C9 records "Cost and risk … **No conflict**". Its coverage status came from searching the spec for "god-view", "true state" and "believed". Those strings do not appear in the gates listed above.

C9's evidence also comes from outside this kind of model:

- the `[SHOWN]` result is for LLM agents — balanced configurations fell from 60.7% to 34.4% when agents had to infer relations;
- the transfer to a rule-based model is graded `[INF]` in explainer §18.3.

**Recommendation.** Before approval, either scope `M4.B.2` row by row (which gates read truth, which read belief), or defer it.

### A5. Phase ordering

Phase C gates on criteria that need Phase D machinery:

- `M11.C.17` needs `M7.E.4`'s default deterioration, by its own amendment.
- `M11.C.17`, `M11.C.18` and `M11.C.20` need the `basic_level` estimator, which `M7.A.1` recomputes on the slow tick.
- `M11.C.18` needs nodal events in the estimator window.

Likely also affected, though the text is less clear:

- `M11.C.4` — its cost lands "at the next nodal event", and the nodal calendar is slow-tick content.
- `M11.C.1` and `M11.C.38` — time to symptom threshold, where the threshold is emitted in `M7.D.1`.

### A6. Phase D is blocked by open decisions on death and birth

**Death.** Generation 1 is aged 74–81 at `t0`, and Phase D requires a 40-year run, so deaths are certain. `M6.3`'s disposition at death is deferred, yet `M11.C.37` gates Phase D.

**Birth.** `M11.C.6` needs births. The intent is clear from `FE04.9` and explainer §13.3 — basic level is "fairly well established by the time a child reaches adolescence". So transmission (`M7.E.1a`) should set a child's level, and the estimator (`M1.A.4a`) should govern later change. The spec never states that handover, and as written the two conflict:

- `M1.A.4` and `M7.A.1` say `basic_level` is written by nothing but the estimator;
- `M7.E.1a` assigns a child's level from the parents.

**Recommendation.** State when the handover from transmission to estimator happens. Decide `M6.3` before Phase D is planned.

### A7. Sub-week items on a weekly clock

The weekly tick suits the worked trace in proposal §4.2, which runs in weeks. But the spec and explainer also carry items shorter than one tick:

- `M5.D.6`, explainer §5.4 and §8: the follow-up must happen "the next day".
- Explainer §4.7 and §8 give edge latency as "within hours" to months. `M3.D.1` delivers events once per tick.
- Explainer §8 lists "3 days", "within days" and a "2 hour" meeting among its calibration durations.

**Recommendation.** Either add intra-tick scheduling for latency and the follow-up, or restate these items in ticks and drop the sub-week durations as calibration targets.

A related gap: one outcome per person per week (`M4.D.1`) does not say how a multi-tick `I-POSITION` sequence coexists with selection on the person's other ties.

## B. Internal contradictions and stale residue

| Where | Problem |
|---|---|
| `M7.B.1` vs `M1.A.7a` | `M7.B.1` says `chronic_anxiety` is fixed once in childhood. `M1.A.7a` derives it every slow tick and moves the fixed part to `programmed_reactivity`. `M2.A.0a` still supplies `chronic_anxiety`, and `M10.C.1` still lists a "chronic-anxiety fixation age". Revision 10's `M11.C.38` sweeps `chronic_anxiety` as if it were a parameter. |
| `M10.A.1` | Lists "the ceiling on `systems_perspective` (M1.A.18b)". Revision 9 changed `M1.A.18b` from a ceiling to a rate. |
| `M5.B.3` vs `M5.B.3a` | `M5.B.3`: no optimum contact rate, and more contact **MUST NOT** be penalised (Ch22). `M5.B.3a`, directly above it: contact is "not monotonically good" (1979 Tape 6, KB interviews). This is a source-level tension, and the spec carries both sides unreconciled. |
| `M5.C` table | Gates on "`outside_ness` below threshold". `M1.A.9a` makes `outside_ness` two-dimensional, so a single threshold is undefined. |
| `M1.A.13` vs `M2.A` | `M1.A.13` allows three tiers, derived rather than assigned (`M1.A.13a`). "Central" and "shadow" are Bowen's pairwise terms *within* a tier (explainer §3.10). The `M2.A` table assigns them by hand, alongside role labels — "head of household", "peripheral" — that are not tiers. |
| `M11.C.23` | Takes critical comments from `M1.A.18a`'s blame readout, which is two-sided by design (blame or praise). Using only the blame half violates `M11.F.6`. Using both mis-operationalises expressed emotion, where praise is not a critical comment. |
| `M15.A.4`, `M15.B.5` | Refer to a threshold in `M1.A.3` and to "M1.A.4d's floor". Revision 7 removed both. |
| `M11.E` | Still describes `M11.C.9` as a near-unfalsifiable smoke test. It was rewritten as a three-arm peak and gates Phase D. |
| `M11.4` | `M11.C.9` (three arms), `M11.C.22` (two limbs), `M11.C.27` (2×2) and the absolute clauses of `M11.C.6` are not two-arm directions, and are absent from the "exhaustive" exceptions list. |
| `M11.C.38` | A SHOULD, placed in Phase C's Done-when cell. |
| `M6.I.4` vs `M5.D.5` | `M6.I.4` makes dyadic functional exchange zero-sum. `M5.D.5` has the opposition "pull up" to the mover's level, so both rise. The spec does not say whether the gain comes from solid self, which `M6.I.5` exempts. |
| Undefined fields | **Sex** is used by `M2.A.0f`, `M2.A.0g`, `M7.E.1c` and `M11.C.25`, but is neither a field in `M1.A` nor a column in `M2.A`. **Household / co-residence** is read by `M1.F.1b` (revision 10) but not defined in `M1`. **`expectations`** is needed by `M7.C.1b`, which also cites "M4.C.1's four signed channels"; no such channels exist. |
| Target selection | Two unreconciled mechanisms. `M7.E.1c` selects from a closed list at birth; `M2.A.2` and `M4.D.6e` make the target an outcome of witnessing and learning (explainer §9.4). Nadia, a middle child born before `t0`, matches none of `M7.E.1c`'s five situations unless a birth-time stress is declared. |

**Withdrawn from the first draft of this review:**

- `M4.D.4` vs `M1.A.3c`. `M4.D.4` has its own source (Ch02 · L02.5, explainer §5.2). It is not the corollary `M1.A.3c` withdrew.
- "`M1.A.1` says exactly two state variables". This is Ch17's rule that reactivity is derived from two variables, not a limit on the number of fields.

## C. Reference family (`M2`) data defects

- **Spouse tolerance.** `M2.A.0e` requires spouses within ±1 point. Ravi and Marta are 38 and 40; Teodor and Ana are 34 and 37. `M11.C.24` depends on the tolerance.
- **Family-of-origin ties.** `M1.D.8` and `M2.2` require at least one family-of-origin tie per adult. Teodor and Sofia have none in `M2.B.2`. The table also does not say whether Teodor is Marta's and Iris's father.
- **Fixation age.** `M2.A.0a` states that every agent is past the fixation age, including 14-year-old Pia. That silently constrains an `[I]` value to 14 or below. Explainer §3.3 mentions age 10 as the modelling choice.

## D. Explainer–spec drift

The spec names `model_explainer.md` authoritative for rationale (§0.2), and the proposal calls it "the one to build from". Yet the explainer carries text the spec has superseded:

| Explainer | Says | Spec now says |
|---|---|---|
| §3.1, §13.3, §15.1 | a capacity floor at 25 below which `basic_level` "cannot rise at all" | a continuous falloff (`M1.A.4d`, revision 7) |
| §15.3 | median ≈ 40 is "below the transition at 50" | no transition (`M1.A.3`, revision 7) |
| §3.6a | a "**Ceiling** coupled to `basic_level`" on `systems_perspective` | a rate (`M1.A.18b`, revision 9) |
| §3.9, §15.2 | only the symptom **category** moves; the specific symptom "stays constitutional" | exclusivity withdrawn; learning reaches both levels (`M1.A.11c`, revision 9) |
| §11, test 9 | "holding differentiation constant" | three arms across level (`M11.C.9`); §3.11 has the new form, §11's table does not |
| §17.2 | "The bias in the difference is **unbounded**" | unidentified, not unbounded (`M11.F.9(c)`, revision 9) |

**The "one rationale lives in one place" rule is not holding.** A builder who follows the proposal's advice gets superseded content in six places. `tests/test_spec_consistency.py` checks IDs and counts, not content, so no guard catches this.

## E. Revision 10 specifically

- **The method items are sound, and do not need the preprints.** Counter-based, event-keyed draws, fixed-count uniforms and a placebo test (`M3.D.4a`–`M3.D.4c`) are standard practice: common random numbers, and Random123's Philox and Threefry (Salmon et al., 2011).
- **The model-content items rest on inference.** Explainer §18 is candid about the evidence. The two items that change the model itself are C8 (witness appraisal, "the one place the method literature adds a mechanism") and C9 (the information rule). Both rest on results in other systems — C9's on LLM agents — with the transfer graded `[INF]`. The ⟦proposed rev10 · C8 · Holland 2026⟧ markers name a paper and read like citations of authority. The transfer grade should travel in the marker.
- **`M11.D.16`, as I read it, conflicts with keyed draws.** Its equivariance clause asks that permuting person identifiers permutes the outcome. Under `M3.D.4a`, draws are keyed by identifier, so a permutation reassigns draws to roles. Equivariance then holds in distribution, not byte-for-byte at a fixed seed. The requirement should say which is meant.
- **`M17.A.4`'s minimum effect margin is an absolute magnitude in invented units** — the form `M0.4` exists to avoid. A margin is statistically necessary. Expressing it relative to seed-to-seed spread would at least make it unit-free.
- **Stopping on interval width (`M17.A.1`) is the benign form of adaptive stopping**, unlike stopping on significance. Acceptable as drafted.
- **`M17.E.1`'s "fraction of range" is conditional on invented ranges.** It has no probability interpretation, and should be reported as conditional on the declared box.
- **No compute budget is stated.** The cost multiplies across arms, seed caps, constant samples, reference configurations, activation regimes, orderings and triad enumerations, on runs of about 2,080 ticks.
- **The document grows about 36% (33.6k → 46.1k words), mostly Phase E methodology,** while no v2 code exists.

## F. Document form

§0.2 says the spec cites and does not argue. In practice it carries extensive quotation, argument and 14 inline "Corrected 2026-…" notes. Three genres are mixed in one file — evidence, requirements and change log. That mix is why stale residue survives each revision (sections B and D).

The project's own record points the same way. Every correction at revisions 4, 7 and 9 came from reading a primary, not from a guard (`_STATUS.md`). Sweeps that read extractions keep missing things of this kind.

## G. Fidelity to the 22 chapters, against the ledger and extractions

Every item below was checked in `_LEDGER.md` and, where a quotation is given, located in the named `docs/theory/chNN.md` extraction. None was checked against a primary.

### G1. Chapter findings the ledger marks as model requirements, absent from the spec

| Chapter finding | What the ledger says the model needs | Spec |
|---|---|---|
| **Bowen's two falsifiers of the budget** — L09.2, restated L16.2 | Marital conflict alone does **not** impair children; children **are** impaired in calm marriages; a chronically ill parent binding load **protects** the children | No criterion. These are the corpus's own falsification tests, and they are **emergent**: no single rule names the outcome. They are the kind of test A1 finds missing. |
| **Explicit sensitivity ranking** — L07.4, `[stands]` | Course determined first by the spouses' dynamics, second by relationships outside the ego mass, third by fusion intensity: "a sensitivity analysis that does not reproduce this ordering is mis-parameterised" | Absent. Revision 10's `M11.4c` says no ranking of mechanisms has been found, and TODO proposes striking it "unless a ranking turns up". **One is in the ledger.** Single source, and Ch07 is a near-rewrite of Ch06. |
| **The seesaw is a conjunction** — L04.1, `[corrected]` | The patient improves only while **both** parents are more invested in each other than in the patient: a min over both parents, so one defecting parent drives regression. A difference form lets a strong marriage mask the defector | In explainer §4.5; **no requirement in the spec**. `M1.B.8` defines investment, and nothing specifies the seesaw's form. |
| **Two-rung avoidance ladder** — L03.4, `[corrected]` | Rung 1 is engaging the therapist; rung 2 is talking to the patient about the psychosis, drifting to criticism. Rung 2 unlocks only when rung 1 is closed; an unordered residual follows | `M8.1` has only `avoidance_available` as a step. The ladder and its unlock condition are absent. |
| **Illness in the mover on a pull-up** — L03.5, `[stands]` | "Not unusual for a parent in an inadequate position to develop a physical illness when he attempts to pull up": an endogenous symptom generated by the differentiating move itself, in the mover | Absent. `M5.E.4` puts a symptom at the foot of the **system's** reaction ladder, which is a different person. |
| **`PROVOKE`**, the deliberate low-stakes provocation — L21.6, L21.9; `ch21.md` "tempest in a teapot" | Message intensity as a dial separate from content, plus "deliberate perturbation is hard" | Listed in explainer §5.7 as one of the six added moves; **absent from the spec**, whose `M5.B` swaps in `PREVENT_ALIGNMENT`. Yet `M5.C.1a` (raise an old issue in a peace-agree family) and the no-live-issue gate (`M5.C`) both need an agent to raise an issue deliberately, and no move does it. |
| **Delayed-symptom channel after a loss** — L15.1, `[narrowed]` | A less integrated family "may show little reaction at the time and respond later with symptoms". The ledger names this the better-evidenced half, and the one a test should assert | No criterion. |
| **Where `REDUCE_CUTOFF`'s anxiety cost lands** — L22.6, `[corrected]` | The transient cost belongs to **a third party whose distancing mechanism is being stripped**, not to the person increasing contact | `M5.B.3` says the move **MUST** lower anxiety and does not say whose. `M5.B.3a` places the constraint on "the person maintaining" the cutoff. |

Lower priority, and possibly out of scope by design:

- L13.2's **conditional** togetherness ratchet, with an unconditional cap on the individuality side (societal scale);
- L04.5's gating of the father's primary **position** but not his contact;
- L10.7's appraisal by **deviation from expectation** (narrowed);
- L19.4's one hard constraint, "the child never denies".

### G2. Approved requirements stronger than their chapter source

| Requirement | Spec says | Chapter says |
|---|---|---|
| `M1.B.4` | Reunion **MUST** restore full coupling with zero re-activation latency — for every tie | Ch08 (L08.2): "it *appears* impossible" for any one to differentiate; "immediately operative again" is said of the **severe schizophrenic triad**, and the next paragraph says differentiation "does occur" in less severe families. No separation duration is given. **The decay claim has four chapters behind it (C7); the zero-latency claim has one hedged, scoped passage.** |
| `M1.C.6` | A sibling-conflict event **MUST** instantiate the parent triangle; intervening on the sibling pair alone **MUST** fail | Ch21 (L21.8, `ch21.md`): sibling conflict "consists **almost universally** of a triangle…". The hedge became a MUST. "Intervening on the pair alone must fail" is the ledger's model-impact **inference**, not the chapter's claim. |
| `M5.B.3a` | "four sources agree that **contact** is not monotonically good" | This conflates three quantities. **Contact frequency:** Ch22 (L22.6) contradicts an optimum — visits "as frequently as possible". **Closeness:** Ch12 (L12.4) gives a two-sided band on closeness versus isolation. **Disclosure:** the KB interviews say not all self is communicated. Only the second and third bound anything, and neither is about contact rate. `M5.B.3` directly below says the opposite. |
| `M7.A.2` | Differentiation gained in a peripheral system **MUST NOT** transfer automatically | Ch10's proposition 48 (`ch10.md`): gains "**may** be automatically manifested in the nuclear family"; Ch21 (L21.7) reports a return to an unworked system without "a single episode of fusion". The sentence is Kerr's (`KS16.9`), and his own qualifier, "but it helps considerably", is kept only as a quotation. The **mechanism** in `M7.A.2a` (per-tie attenuation by load) is consistent with both authors. The **MUST NOT sentence** is not consistent with Bowen, and `M1.A.3b`'s rule — a Kerr formulation **MUST NOT** silently overwrite a Bowen one — applies. |
| `M1.D.7l` | Regression **SHOULD** be checked against a 5–10 point shift toward togetherness, `[#]`, from Kerr's "55 or 60" | Ch13 (L13.4): "**The 55/60 togetherness figures are explicitly disclaimed as illustrative and are not usable for calibration.**" Revision 9 fixed the modal (MUST to SHOULD) but not the source question. Bowen's disclaimer on the original figure does not travel with Kerr's restatement. |
| `M1.A.4`, `M5.D.7`, `M5.D.7a` | A completed exchange **MUST NOT** write `basic_level`, and no realistic number of exchanges moves it | Ch21 (L21.1, `ch21.md`): a held position produces "a basic increase in bilateral differentiation **which can never return to the former level**". The spec departs from this **deliberately**, by the expert decision recorded as U8. That departure is defensible — Ch21 is n = 1 and contradicts itself ~5,000 words away. But U8 does not mention the sentence it overrides, and the spec grades none of this. The departure from a `[T]` chapter statement should be recorded as such. |

### G3. The ledger and one extraction still carry superseded readings

`CLAUDE.md` calls `_LEDGER.md` "**the source of truth for what the corpus says**". Five entries contradict the corrected spec:

| Where | Still says | Corrected in the spec at |
|---|---|---|
| **L13.3**, correction (2); **`ch13.md` pass-2 note 4** | "Anger is not a fourth branch; **it is the gate** admitting the sequence to the peak." The body text beneath each is correct (the angry mover stalls). The bolded sentence is the same **lossy-heading** defect that inverted `M5.D.4`; it was fixed at the section heading (`ch13.md`, the "[p3]" heading) and not here | `M5.D.4`, revision 4 |
| **L06.2**, the KB03 paragraph | "an experiment… and an **independent replication by Kerr**" | `M1.E.2a`, revision 9: "Neither is" |
| **L16.1**, model impact | "a 0–100 scale with **one behavioural transition at 50**" | `M1.A.3`, revision 7 |
| **L21.1**, **L22.7**, model impact | "a **ratcheted** `basic_level` moved only by a completed reaction cycle" | `M1.A.4a`, per U8 |
| **L10.6** | differentiation "transfers **automatically**" — stronger than Ch10's own "may"; and `KS-A.1` contradicts it, although ledger §2 says the 2019 book "contradicts **no** finding of the papers" | `M7.A.2` (see G2) |

The ledger's header warns that an entry without a correction note was not re-verified. That warning does not cover these five: they are entries whose **own** readings the project later corrected elsewhere. A reader who takes the ledger as the source of truth inherits the uncorrected versions, and nothing guards against it.

## Recommendations, in priority order

1. **Account for anxiety.** Restate `M6.I.6` at C3's scope and add the stock-and-flow table (A3). Decide death disposition and the child-to-adult handover of `basic_level` (A6). These block Phases C and D.
2. **Audit the criteria.** Run `M11.1d` on paper for all 41 criteria, and split verification from emergent consequence (A1).
3. **Sweep in Phase C.** Correct the cancellation premise in explainer §17.3 and `_STATUS.md`, and bring a dominant-constant sweep into Phase C gating (A2).
4. **Scope or defer `M4.B.2`** (A4).
5. **Fix phases and time base.** Correct the phase placement of `M11.C.17`, `M11.C.18` and `M11.C.20` (A5), and resolve the sub-week items (A7).
6. **Correct the residue, data and drift.** Fix the contradictions in B and the `M2` data in C. Bring the explainer into line with the spec (D), and add a content-level guard if one is feasible.
7. **Extract a normative kernel.** Keep requirement sentences only in the spec; move quotations and correction notes to the explainer.
8. **Correct against the chapters.** Add criteria for Bowen's two falsifiers and for L15.1's delayed channel. Decide whether L07.4's ranking instantiates `M11.4c`. Add or explicitly exclude L04.1, L03.4, L03.5 and `PROVOKE`. Rescope the six requirements in G2. Correct the five ledger and extraction entries in G3.
9. **Approve and build Phase B only** (scripted, no policy). Many of the defects above sit in objects and invariants that code will expose faster than another revision will.
