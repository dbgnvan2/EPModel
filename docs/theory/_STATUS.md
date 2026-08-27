# Run status — HANDOFF

> ## ▶ STATE — awaiting approval on the *Family Evaluation* decisions
>
> **Sixth corpus source complete, 2026-08-26.** Kerr & Bowen, *Family Evaluation* (1988) — 12 segments,
> 219 findings, pass 2, folded into `_LEDGER.md` as **Source 7**. See `family_evaluation/_INDEX.md`.
>
> **The open item is `docs/DECISIONS — FAMILY EVALUATION.md`** — 4 contradictions with the approved
> spec, 21 requirement candidates, 1 prohibition from Bowen. **Nothing applied.** That file is separate
> from `DECISIONS FOR APPROVAL.md` on purpose, because you are editing that one.
>
> **The pipeline is complete.** Chapters → draft spec → revision 1 (1979 lectures) → revision 2 (Kerr–Bowen interviews) → revision 3 (Kerr 2019) → **the 1988 book, pending revision 4.** Every source has been read.
>
> | Stage | State |
> |---|---|
> | Book, 22 chapters, 2 passes + resolutions | complete |
> | 1979 Basic Video Series, 6 tapes — all 23 validation items probed, Tape 6 read end to end | complete |
> | Kerr–Bowen interviews, 15 — pass 1, pass 2 comparative, pass 3 into the ledger | complete |
> | Kerr, *Bowen Theory's Secrets* (2019), 26 segments — 262 findings | complete |
> | **Kerr & Bowen, *Family Evaluation* (1988), 12 segments — 219 findings** | **complete 2026-08-26** |
> | External measures (the DSI) | recorded |
> | Spec v2.0 — 321 unique IDs, 0 unresolved | **approved 2026-08-25; revision 3 applied. Revision 4 blocked on the decisions below.** |
> | **`docs/DECISIONS — FAMILY EVALUATION.md`** | **the open item — 4 contradictions, 21 candidates, nothing applied** |
> | `docs/DECISIONS FOR APPROVAL.md` | approved 2026-08-25; propagation complete |
> | Code | **none. Nothing until the spec and then the plan are approved.** |
>
> **Next after approval:** implementation plan (criteria mapped to files and order), plan approved, then Phase B.
>
> ### Still needing a human
> - **`DECISIONS — FAMILY EVALUATION.md`** — 4 contradictions with the approved spec, 21 requirement candidates, 1 prohibition from Bowen.
> - The **Ch13 re-read** on anger (`A3` in that file) — half an hour of reading, not done.
> - ~~The **decision list**~~ *(approved 2026-08-25)* — 2 withdrawals, 1 terminology choice, 3 framing corrections, 11 new requirements, 5 open items.
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
| `_LEDGER.md` | **six corpus sources.** The structure differs by source: **149 findings** from the 22 papers (151 `### L` headings − `L03.2` merged − `L10.11a` a lettered sub-entry); **24 consolidated `KS` entries** standing in for 262 findings in `kerr_book/`; **Source 7's 6 `FE` sections** standing in for 219 findings in `family_evaluation/`; plus Source 6 (domain expert, `U1`–`U14`, **not corpus findings**) and a withdrawn-ID appendix. **Do not quote a single total** — the units are not the same. | **current — corrections folded in, self-contained.** The pass-1 version is in git history, not kept as a file. |
| `ch01.md`–`ch22.md` | per-chapter extraction, both passes, `[p2]` marks additions | current |
| `Extractions to be human validated.md` | the 22 extractions the model leans on hardest, with source line numbers and verdict boxes | **awaiting human validation — the open item** |
| `family_evaluation/` | **Kerr & Bowen, *Family Evaluation* (1988)** — 12 files, 219 findings, `_FE_PASS2.md`, `_INDEX.md` | **current.** ⚠ Two authors writing **independently**: Chs 1–10 are `[K]`, the Epilogue is **`[B]`, the latest primary Bowen text in the project** |
| `kerr_book/` | Kerr, *Bowen Theory's Secrets* (2019) — 26 files, 262 findings, `_KS_PASS2.md` | current |
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

### Step 2 — propagation **COMPLETE**

**Decision document APPROVED 2026-08-25. All sixteen propagation items are applied.** Spec at
**321 unique IDs**, 0 duplicates, 0 unresolved, 355 MUST / 100 MUST NOT. Explainer, published proposal and
ledger all agree with it; every spec cross-reference in both documents resolves; 37 tests green.

**The document set is consistent. The next step is the implementation plan.**

| # | Item | State |
|---|---|---|
| S2.4 | A/B/C/D decisions into the spec | ✅ |
| S2.5 | E2 marriage-ceremony mechanism → `M1.A.3a` | ✅ |
| S2.6 | A7 change-back cause + `M5.E.8` success state + `M5.E.9` three timescales | ✅ |
| S2.7 | B4 ally as pseudo-self transaction → `M8.6a/b` | ✅ |
| S2.8 | C4 two-sided damping → `M1.D.7i–l` | ✅ |
| S2.9 | operational decomposition → `M1.A.5a/b` | ✅ |
| S2.10 | attention amplifies the channel → `M4.C.8` | ✅ |
| S2.11 | functional DOS in the two-channel account → `M1.A.5a` | ✅ |
| S2.12 | pseudo-self sign by channel → `M1.A.5c/d` + `M1.F.1a` | ✅ |
| S2.13 | spouse pairing ±1 + complementarity → `M2.A.0e/f` | ✅ |
| S2.14 | C2 observational-vs-abstract addition → `M11.F.3a`; general two-sidedness → `M11.F.6` | ✅ |
| S2.15 | distance binds → `M1.D.2a` rewritten | ✅ |
| S2.16 | reinforcement horizon belongs to functional level → `M4.D.6b` | ✅ |
| S2.1 | `model_explainer.md` | ✅ |
| S2.2 | `agent_model_proposal.html` — corrected in place, `revmark` spans, HTML validated | ✅ |
| S2.3 | `_LEDGER.md` — **Source 6** added: 14 expert-supplied and cross-source resolutions, graded `[user]` / `[user→T]` / `[resolved]` so they are never mistaken for corpus findings | ✅ |

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

---

## Family Evaluation — the sixth corpus source, 2026-08-26

**Run to the same steps as the Bowen book and the 2019 book:** pass 1 per segment → pass 2 comparative
→ pass 3 into `_LEDGER.md` → (spec revision 4, **blocked on approval**).

**Pass 2 verified fifteen load-bearing quotations character-for-character** against the source files
and **corrected six pass-1 readings** — the same shape as the 2019 book's pass 2, scope corrections
rather than withdrawals. One was a term confusion (`FE03.2`) that would have put a symmetric invariant
on `M1.B.8` and broken its directedness. That is what the pass is for.

### What it changes

**Four contradictions with the approved spec, none applied** — `_LEDGER.md` Source 7 `FE-A`:

- **`M1.A.11b`** — Kerr 1988 says genes set the **specific symptom** and **learning** sets the
  **category** (physical / emotional / social). The spec has that assignment inverted, citing 2019
  material whose wording is ambiguous on exactly this point. **The most consequential item.**
- **`M1.A.3`** — the transition at 50 described as an *awareness* boundary rather than a decision-scope
  licence.
- **`M5.D.4`** — anger called an unreliable guide, and dogmatic self-assertion negative evidence.
  Bowen agrees independently in the same volume. Probably a mover/system distinction; **needs `L13`
  checked against Ch13 before encoding.**
- **`M12.2`** — a mechanism for the first of the two stated unknowns. The second gets a locus, not a
  rule, and stands.

**Twelve requirements move to written primary text**, five of them onto **Bowen's own hand** — `FE-B`.
Nothing in the model changes; what changes is that a dozen load-bearing requirements stop depending on
ASR transcripts the project may not quote. `M1.A.9a`, the two-axis identity definition the model is
built on, had until now reached the project *secondhand* through Kerr in 2019 quoting himself citing
this book. **That was the reason for the run, and the primary is stronger than the paraphrase.**

**Three things settled:** explainer §13.3 (when `basic_level` is fixed — the provisional decision was
right); the three-versus-four-sinks question, for good; and the apparent tension between `FE07.7`'s
per-generation bound and `KS11.1`'s traversal timing, which dissolved inside the run.

### Still needing a human

- **`docs/DECISIONS — FAMILY EVALUATION.md`** — the 4 contradictions, 21 requirement candidates, the
  `M12` prohibition from Bowen, and 2 proposed `M11.F` entries.
- **The Ch13 re-read** on anger (`A3`), which is half an hour of reading and which I have not done.

### Documents brought up to date

| File | State |
|---|---|
| `_LEDGER.md` | ✅ Source 7 added; header §2 now six corpus sources |
| `model_explainer.md` | ✅ **reworked** — six-source index, full citation legend, `[B]`/`[K]`/`[K-ext]` grades, magnitudes table, §13.3 closed, §14 now quotes Bowen's own general-systems passage |
| `kerr_book/_INDEX.md` | ✅ queue line closed; the "source #6" numbering corrected |
| `_SOURCE_QUALITY.md` | ✅ scope note — it analyses the **recorded** corpora only |
| `_CONVERGENCES.md` | ✅ independence audit updated |
| `CLAUDE.md` | ✅ where-the-theory-lives table updated |
| `agent_model_proposal.html` | ✅ §10 note added; **needs republishing to the same URL** — see above |
| `bowen_agent_model_spec_v2.md` | ⏸ **untouched — revision 4 is blocked on the decisions above** |
