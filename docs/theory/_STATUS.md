# Run status — HANDOFF

> ## ▶ STATE — awaiting approval
>
> **The pipeline is complete.** Chapters → draft spec → revision 1 (1979 lectures) → revision 2 (Kerr–Bowen interviews). Every source has been read.
>
> | Stage | State |
> |---|---|
> | Book, 22 chapters, 2 passes + resolutions | complete |
> | 1979 Basic Video Series, 6 tapes — all 23 validation items probed, Tape 6 read end to end | complete |
> | Kerr–Bowen interviews, 15 — pass 1, pass 2 comparative, pass 3 into the ledger | complete |
> | External measures (the DSI) | recorded |
> | Spec v2.0 — 276 IDs, 0 unresolved | **revision 2 applied; awaiting approval** |
> | **`docs/DECISIONS FOR APPROVAL.md`** | **the open item — needs your review** |
> | Code | **none. Nothing until the spec and then the plan are approved.** |
>
> **Next after approval:** implementation plan (criteria mapped to files and order), plan approved, then Phase B.
>
> ### Still needing a human
> - The **decision list** — 2 withdrawals, 1 terminology choice, 3 framing corrections, 11 new requirements, 5 open items.
> - The **validation checklist** (`Extractions to be human validated.md`) — 23 items, of which the lectures and interviews settled or corroborated 15. **A7, C4 and B4 are single-sourced to the book and corroborated by nothing**; those most need a human eye.
> - **Four acceptance criteria** that cannot be made code-testable in Phases B–D (`M11.E`).
>

> **Starting a new session? Read this file first, then `_LEDGER.md`.** As of step 3 the ledger is
> **self-contained** — every pass-1 entry has been rewritten against the pass-2 corrections, and you no
> longer need to cross-reference two files to act on one. `_PASS2_CORRECTIONS.md` is now the audit
> trail (what changed and why), not a required companion. `_CONVERGENCES.md` is current.

## File map
| File | What it is | Trust |
|---|---|---|
| `_STATUS.md` | this file — state, decisions, order of work | current |
| `_PASS2_CORRECTIONS.md` | 10 batches of corrections to pass 1 | audit trail — **folded into the ledger at step 3**, no longer needs reading first |
| `_CONVERGENCES.md` | 10 cross-chapter convergences, independence audit, chronology, terminology timeline | current, corrected |
| `_LEDGER.md` | **149** per-chapter findings + a withdrawn-ID appendix | **current — corrections folded in, self-contained.** The pass-1 version is in git history, not kept as a file. |
| `ch01.md`–`ch22.md` | per-chapter extraction, both passes, `[p2]` marks additions | current |
| `Extractions to be human validated.md` | the 22 extractions the model leans on hardest, with source line numbers and verdict boxes | **awaiting human validation — the open item** |
| `_KERR_INTERVIEWS.md` | Kerr–Bowen interview #1, late period — the misconceptions interview | current; rest of the series not located |
| `_LECTURES_1979.md` | the Basic Video Series (1979) — the latest Bowen in the project — and a first adjudication pass | current; 12 of 23 items probed |
| `_EXTERNAL_MEASURES.md` | the DSI and what it does and does not give the model | current |
| `_TEMPLATE.md` | extraction schema | reference |
| `../agent_model_proposal.html` | **the model proposal — repo copy, source of truth** | **STALE — see "Known stale" below** |

Source texts: `/Volumes/CrucialX9/Downloads Duplicate SERP/fticp_chapters  TXT files/`
Published proposal: https://claude.ai/code/artifact/86573afe-ecb3-4083-9c82-2e6094a04f43
**To update it:** edit `docs/agent_model_proposal.html`, then call the Artifact tool with that
file_path AND `url` set to the link above — publishing without `url` from a new conversation creates a
*separate* artifact instead of updating this one. Favicon 🔺, title "The Twelve-Agent Family".

## ~~Known stale in the proposal~~ — RESOLVED at step 4 (2026-08-22)

All eleven items are reconciled and the artifact is republished to the same URL. The reconciliation lives
in the proposal itself as **§10, "What reading Bowen changed"** — nine design changes, five things that must
not be built, four unhomed mechanisms, two open contradictions, and the framing note. Nine inline `rev`
markers flag affected passages, and a banner at the top points at §10.

Of the eleven, **five were never actually asserted in the proposal** — the permission scalar, ally
cancellation, the target queue, the depletable distance stock, and "content is noise". They are recorded in
§10's *must not be built* table rather than removed, because a mechanism that is merely absent gets
re-invented. The other six were live and are fixed: the four-sink budget, reactivity as stored state, the
C range and its threshold, acceptance tests 5 and 7, and the missing life-energy budget.

## Pass 1 — COMPLETE
All 22 chapters extracted. 268k words. Every template section present in every file.
Outputs: `ch01.md`–`ch22.md`, `_LEDGER.md` (109 findings), `_CONVERGENCES.md` (9 convergences + timeline).

## Pass 2 — COMPLETE (all 22 files carry `pass: 2`)
Comparative re-read: each chapter re-read against the full corpus + targeted verification questions.
Agents write `# PASS 2 — COMPARATIVE` into their own `chNN.md` and mark in-place additions `[p2]`.

- All 22 complete. Corrections folded in from every chapter.
- **Corrections live in `_PASS2_CORRECTIONS.md` — that file OVERRIDES `_LEDGER.md` and
  `_CONVERGENCES.md`.** Do not act on a pass-1 finding without checking it there first.
- 8 correction batches recorded. Ch21 and Ch22 summaries arrived last and their per-chapter files are
  written; fold any remaining detail from them at the start of pass 3.

### The one-line finding of pass 2
**Pass 1 systematically over-read, and every error ran the same direction** — it made the source look
more quantitative and more decided than it is. Invented rankings, rates manufactured from illustrations
Bowen explicitly bounded, topology added that the text does not have, hedges stripped, later vocabulary
read backwards into earlier chapters, and inferences reported as measurements.

### Two hard conclusions for the spec
1. **Q-VALIDATION is NO in every chapter checked.** No instrument, no rater procedure, no comparison
   group, no number assigned to a person, anywhere in the book. The corpus supports **directions,
   orderings and mechanisms — almost no magnitudes.**
2. **Q-MATERIAL is NO in every chapter checked**, including Ch20, the only workplace chapter, where
   tie weight is set by emotional importance *explicitly not by economic relation*. The `M` column has
   no basis in the source.

### Outstanding task before pass 3
**Audit every convergence for shared data source.** Ch01, Ch05, Ch08 and Ch19 all report the same NIMH
residential project (1954–59); agreement between them is not independent replication. C3 and C6 are
already corrected; C2 and C5 still need it. Report each convergence as "N chapters / M independent
studies".

## Pass 3 — NOT STARTED
Order of work:
1. ~~Fold in remaining Ch21/Ch22 detail.~~ **DONE** — batches 9 and 10.
2. ~~Audit every convergence for shared data source.~~ **DONE** — see the independence audit at the
   top of `_CONVERGENCES.md`. Headline: the NIMH live-in project covers nine chapters. **C5 collapses
   to one setting / two case series and is contradicted by Ch05.** C4 is weaker than it looked.
3. ~~**Rewrite `_LEDGER.md`** so every entry reads correctly on its own.~~ **DONE.** All 109 pass-1 IDs
   preserved and re-verdicted, 40 new pass-2 entries added, every correction from all ten batches folded
   in. Withdrawn findings are kept as **tombstones** (ID + reason) rather than deleted, because eleven of
   them are cited in the proposal and `_CONVERGENCES.md` — deleting them silently would leave dangling
   references and invite re-invention. Each entry carries a status marker: `[stands]` / `[narrowed]` /
   `[corrected]` / `[withdrawn]` / `[new at p2]`. Six residual contradictions in `_CONVERGENCES.md` were
   fixed at the same time (C2's count, C5's body, the Ch16/Ch17/Ch18 timeline rows, the scale summary).
4. ~~**Bring all documentation up to date.**~~ **DONE 2026-08-22.**
   - `docs/agent_model_proposal.html` — new **§10** (the corpus reconciliation), nine inline `rev` markers,
     a read-this-first banner, F4 corrected to three sinks, tests 5 and 7 rewritten, the roster/cutoff
     problem raised against the twelve-person premise, the quantification weakness strengthened with
     Q-VALIDATION, and Use case closed as *research instrument*. **Republished to the same URL.**
     The pre-corpus version is in git history at `36dfa02^`; no `.bak` file is kept.
   - `docs/bowen_individual_family_model_spec.md` — **frozen as a v1.2 historical record** with a header
     table of the nine things the corpus contradicts in it. It is *not* the base for v2; v2 is written
     fresh. Also fixed a false sub-header ("not yet implemented" above a fully-ticked list).
   - `CLAUDE.md` — rewritten: where the theory lives and which file to trust, the two corpus rules that
     are easy to violate by accident, the `sim_audit.csv` purity violation recorded as a known defect,
     and **Axiom 1 scoped to the frozen grid engine only** — at N≈12 it pushes the design the wrong way
     and cannot express per-tie or per-triangle state without the exact flattening the pivot undoes.
   - **Found in passing, and since fixed:** `test_spouse_dysfunction_asymmetric_penalty` was flaky at 6
     failures in 12 runs — unseeded `Simulator`, asserting on an arbitrary pair, both deltas 0.0 on a
     failing draw. A second test had the same defect. Both repaired in `3dfa301`; suite now 36/36 across
     12 consecutive full runs.
5. ~~**Write the model explainer.**~~ **DONE 2026-08-22** — `docs/model_explainer.md`, ~13,000 words,
   16 sections. Every object, field, move, gate, invariant, clock, event property, readout and test, each
   with *what it is for* / *what it does* / *which finding it implements + chapter*.
   **The device that carries it is a grade on every claim** — `[T]` textual, `[M]` mechanism, `[D]`
   direction/ordering, `[#]` a stated quantity, `[I]` invented, `[X]` contested — so an invented constant
   can never be mistaken for a sourced one. That is the only honest response to Q-VALIDATION being NO in
   all 22 chapters.
   Also carries: four acceptance tests (12–15) the corpus asks for that the original eleven do not cover;
   three readouts that lie (overt emotionality peaks mid-scale, the fifth band, the measurement bias);
   the five must-not-build mechanisms; Bowen's two stated unknowns; the three open contradictions; a
   per-chapter source index; and the provenance table.
   **Verified mechanically at the step-6 fix commit:** **101** distinct ledger IDs cited, all resolving;
   every quote attributed to Bowen traces to a ledger entry; **215** graded claims. The ledger's own
   figure reconciles as 151 `### L…` headings − `L03.2` (merged into `L03.1`) − `L10.11a` (a lettered
   sub-entry) = **149 findings**.
6. ~~Resolve the two live contradictions.~~ **DONE 2026-08-22** — `_RESOLUTIONS.md`. **Both resolved,
   and both against the primary source rather than the extractions** — in each case the resolving sentence
   is in the text and in neither summary.
   - **R1 — threesome vs twosome: neither. Three *live positions*.** Ch14's "two people" section describes,
     in its own words, "the triangle of the two most important family members and the therapist". Ch03
     counts only family members because its therapist is structured out ("observes from the sideline").
     **L02.4/L03.4 stand, restated in terms of positions.** `L14.5` item 2 is **withdrawn — it was a
     contradiction with a section heading.**
   - **R2 — witness vs ally: alignment is the gate, not knowledge.** Ch21's one worked ally case is a
     sister who already knew everything and offered to stand with him; the countermeasure returned her to
     neutral and left her knowledge intact. The same chapter has a witness present throughout who takes no
     position — his wife — in the most successful effort in the corpus. Ch10 bars alliance from the
     helper's side in identical terms. **Secrecy is a hazard rate over alignment, not an independent gate.**
   - **Both reduce to one predicate the model already needs:** *a position counts as live only if its
     occupant is not fused into another.*
   - **Two additions:** a `PREVENT_ALIGNMENT` move (acts on *potential* alignment; Ch21's contradictory
     letters), and `L21.11` — **surprise as a third secrecy mechanism**, in the source and in neither pass.
   - **Still genuinely open, but decided for the model:** when the basic level is fixed. Ch21 contradicts
     itself and is outvoted by three chapters; implement `C` as slow-moving with a ratchet, not frozen.
7. **Write the v2 spec.** **DRAFTED 2026-08-22, sweep-corrected; awaiting approval. No code until approved.**
   Scope: Phases B–D (objects, loop, policy, slow clock, reference family). Phases E and F excluded.
   Normative only — rationale and sourcing stay in `model_explainer.md`, cited by section.

   **Counts are measured, not asserted.** Reproduce the first with exactly this command; the rest are in
   the same shape:
   ```
   grep -oE '\*\*M[0-9]+(\.[A-Za-z0-9]+)*\*\*' docs/bowen_agent_model_spec_v2.md | sort -u | wc -l
   ```
   At the sweep-fix commit: **212 requirement definitions** (plus 39 section headings; 251 distinct tokens
   in all), **0 unresolved references, 0 duplicate definitions**, all matching the single pattern
   `M\d+(\.[A-Za-z0-9]+)*`. **226 MUST / 68 MUST NOT.** **16 acceptance criteria + 10 engineering
   criteria.** **50** explainer cross-references, all resolving. Explainer: **216 graded claims**, **101**
   distinct ledger IDs cited, all resolving. Ledger: 151 `### L…` headings − `L03.2` (merged) −
   `L10.11a` (a lettered sub-entry) = **149 findings**.

   **Four criteria are flagged not code-testable in B–D**, each with a human-review proposal (M11.E).

   **Two pre-push sweeps ran** over this batch: the first found 9 findings (5 high), the second — over the
   fix commits only — found 15 (5 high), which is the expected shape: fixes are the least-reviewed code in
   a batch. Both are resolved. The loop was bounded there deliberately rather than swept a third time.

   **What neither sweep assessed — and the distinction matters.** Bowen's papers are the source and are not
   in question. What is unverified is **our reading of them**: `ch01.md`–`ch22.md` are summaries this
   project wrote, and nobody has checked a summary back against its chapter. Both sweeps verified that
   citations *resolve*, never that the entry they resolve to reports the chapter correctly.

   This is not a hypothetical risk. **Pass 2 withdrew nineteen pass-1 findings** — two manufactured rates,
   invented rankings, stripped hedges, topology the text does not have. In every case Bowen was right and
   the extraction was wrong. It happened again on 2026-08-22: `resource_pressure` was graded `[I]`
   (invented) when it is Ch18's own thesis, and it took the user to catch it.

   ~19,000 lines of extraction remain unchecked against the chapters. That is the load-bearing claim in
   this work.

   *Separately, and not a judgement on the source:* **Q-VALIDATION is NO** is scoped to the papers — external instruments exist and are recorded in `_EXTERNAL_MEASURES.md`. It means the papers contain no
   instrument, no rater procedure, no comparison group and no number assigned to a person. That is a
   statement about what is *in* them — clinical observation across decades, reported as such. It
   constrains what the model can calibrate against (directions, orderings, mechanisms; not magnitudes).
   It says nothing about whether Bowen is right.

---

## Where the work is now — 2026-08-24

**All four sources are read.** Book (22 chapters, two passes), 1979 video lectures (6 tapes), Kerr
interviews (15, three passes), and the DSI measurement literature (`_EXTERNAL_MEASURES.md`).

**The spec has moved ahead of the decision document, deliberately.** On the user's instruction, the
G4, G6 and G9 change sets were applied to `bowen_agent_model_spec_v2.md` (commit `44c439f`): +20 IDs,
none removed, 256 unique, 0 duplicates, 0 unresolved. Consequence: the G4/G6/G9 headings inside
`DECISIONS FOR APPROVAL.md` still read *"listed, not applied"* and are **stale**. The user is editing
that file; do not edit it without asking.

### The blocking item
`docs/DECISIONS FOR APPROVAL.md`, sections A, B, C, D, E1, E4, E5. **E2 and E3 are answered.**

### Step 2 — propagation queue, to run once the decision document comes back

Applying G4/G6/G9 to the spec made three previously-consistent documents stale. They **MUST** be
brought back into agreement before the user reviews the set.

| # | Target | What is stale / what to add |
|---|---|---|
| **S2.1** | `model_explainer.md` | still describes the `basic_level` **ratchet** (removed) and a **scalar** `outside_ness` (now two-axis, M1.A.9a); has no `systems_perspective` (M1.A.18) and no two-channel selection (M4.D.1a) |
| **S2.2** | `agent_model_proposal.html` | same three gaps; published artifact, needs `revmark` spans |
| **S2.3** | `_LEDGER.md` | the findings behind G1-G9 exist only in the decision document — record them as ledger entries with their `KB`/lecture citations |
| **S2.4** | `bowen_agent_model_spec_v2.md` | whatever A/B/C/D/E1/E4/E5 decide |
| **S2.5** | `M2.A` + `_LEDGER.md` | **E2 — the marriage-ceremony break.** User's mechanism, approved for folding in: *marriage strengthens fusion; the couple is treated as **one self**, so the live question becomes who gets to decide for that self; the lower the level, the more anxiety and reactivity this produces.* Encode against `M1.A.3`, whose behavioural transition at 50 is already **a licence over joint decisions** — the ceremony does not change either person, it reclassifies a large class of decisions into the shared-life-course domain where the transition bites. Record as the resolution of the long-open E2. |

| **S2.6** | `M5.E.6`, `M5.E.7` | **A7 confirmed, with a mechanism.** The change-back reaction happens because the move **pushes others to function more responsibly**, which they resist; **hold the position and the system shifts over months.** `M5.E.7` currently names the life-energy debit as the cause — compatible, but the responsibility framing connects it to `M12`. `M5.E.6`'s damped oscillation needs a **settling timescale of months**, graded `[I]` (user's judgement, not a corpus quantity). |
| **S2.7** | `M8.6`, `M6.I.4`, `M5.F` | **B4 confirmed — the pass-2 withdrawal was right**, and the mechanism is deeper than a triangle. An ally creates an "us against them" triangle **and** implies **borrowing self from the ally, or lending self to the ally**. So the ally penalty is a **pseudo-self transaction** on `M6.I.4`'s conserved quantity, not only a structural one. A move fuelled by borrowed self does not draw on the solid-self reserve (`M10.A.1a`) and is therefore counterfeit by construction (`M5.F.2a`) — computable rather than declared. |
| **S2.8** | `M1.D.7d` | **C4 confirmed, and two-sided.** Societal forces have less impact **positive or negative** on a well-differentiated family. `M1.D.7d` currently reads as being about harm; the modulation applies to **`|effect|`**, both directions. Third instance of the same two-sided pattern (blame/praise, selfish/selfless, favourable/unfavourable) — treat as a general property. |

| **S2.9** | `M1.A.5`, `M1.A.4a` | **The operational decomposition, stated out loud.** `functional_level` **is** the operational level of differentiation: `functional_level = basic_level` (slow floor) `+ swing` (fast). The self-directed channel (`M4.D.1a`) writes the swing directly — an agent can decide to stop or start doing something and shift functioning at once. `M1.A.5`'s variance requirement is really a statement about the **swing**, and should say so. `M1.A.4a`'s estimator converts sustained, broad swing into basic level — **and that conversion is the same event as pseudo-self becoming solid self**, so agency (`M10.A.1a`) rises as a consequence with nothing extra modelled. |
| **S2.10** | `M4.C`, `M5` | **Attending to a channel amplifies it.** Focusing on feelings is "almost as if you are **uncovering a volcano**… the only time you get a kind of resolution is when the **feelings run down**"; work the intellect and "the feeling world is… an **orderly little fountain**." → `kb/kb12.md` · K12.6. This is where feelings legitimately enter the model — **not as state** (forbidden by `M1.A.0`) but as a **target of attention with an amplifying effect**. Generalises C1 beyond symptom relief. Currently unrepresented. |
| **S2.11** | `M4.D.1a` | **G7 IMPORTANT (user):** *"the whole mechanism of Functional level of DOS in the above process."* The two-channel account must be written in terms of functional level of DOS — the self-directed channel's output **is** a functional-level shift, which is why it can be fast while `basic_level` stays slow. |
| **S2.12** | `M6.I.4`, `M4.D.1a` | **G7 IMPORTANT (user):** *"increase in Functional level of DOS means pseudo self is lower."* ⚠ **Needs confirmation before encoding** — see the open question below. Proposed reading: the **sign depends on which channel produced the rise**. A rise via the **self-directed** channel converts pseudo-self toward solid (pseudo-self ↓). A rise via the **automatic** channel is borrowed from another in the reciprocity (pseudo-self ↑) and is a gain in *apparent functioning*, not in functional DOS. That makes the two channels leave **opposite signatures in pseudo-self** — a computable discriminator the model does not currently have. |
| **S2.13** | `M2.A.0c` | **D3 approved with an amendment (user):** spouse pairing on `basic_level` *"doesn't have to be EXACTLY matched. + or - one point."* Change the construction rule from equality to a ±1 tolerance, declared in config. |
| **S2.14** | `M11.F.3` | **C2 approved with an addition (user):** the distinction to state is that **General Systems Theory is an abstract conceptualisation thought up by man, whereas Bowen Theory is built from observation of how families actually function** — "just like observing the planet movements to derive how they function." The simulation is not offering a non-objective explanation of human functioning. |
| **S2.15** | `M1.D.2a` | ⚠ **D9 approved with a note that pulls against the requirement (user):** *"the Distance IS the symptom, and since it indicates high intensity, that intensity can lead to other symptoms."* `M1.D.2a` currently says distance **absorbs without symptomising**. Reconciliation to encode: distance does not produce the three **named** symptom channels, but it is **itself an observable of high intensity and a predictor of later symptoms** — not a neutral sink. Do not leave it graded as costless. |
| **S2.16** | `M4.D.6b` | **G3 approved with context (user):** the reinforcement horizon belongs to **functional level of DOS**, which is the shorter time horizon. The declared horizon should be scaled to the swing term, not to the `basic_level` estimator window. |

### Open question to put back to the user
**S2.12.** The sentence *"increase in Functional level of DOS means pseudo self is lower"* and Bowen's
standard reciprocity account can diverge: in the classic overfunctioner/underfunctioner exchange, one
party's **functional level rises on borrowed self**, which is pseudo-self going *up*. The reading above
resolves it by making the channel decide the sign. **Confirm before encoding** — if the user means it
holds unconditionally, `M6.I.4`'s reciprocity needs re-examining instead.

### Then
User reviews the finished set -> implementation plan -> plan approved -> **only then** code.

## Standing decisions
- Engine: deterministic core, no LLM in the decision path.
- Subject: synthetic reference family.
- Societal: three scalar dials (L, X, E) + access vector.
- Use case: **research** — ensembles, seeds, falsification.
- Repo: new package `src/bowen/` alongside `engine.py`; old engine and its 36 tests stay green until
  the new one passes the acceptance tests. Freeze `bowen_individual_family_model_spec.md` as a v1.2
  record; write v2 fresh.
