# Run status — HANDOFF

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
     Pre-corpus copy kept at `docs/agent_model_proposal.pre-corpus.html.bak`.
   - `docs/bowen_individual_family_model_spec.md` — **frozen as a v1.2 historical record** with a header
     table of the nine things the corpus contradicts in it. It is *not* the base for v2; v2 is written
     fresh. Also fixed a false sub-header ("not yet implemented" above a fully-ticked list).
   - `CLAUDE.md` — rewritten: where the theory lives and which file to trust, the two corpus rules that
     are easy to violate by accident, the `sim_audit.csv` purity violation recorded as a known defect,
     and **Axiom 1 scoped to the frozen grid engine only** — at N≈12 it pushes the design the wrong way
     and cannot express per-tie or per-triangle state without the exact flattening the pivot undoes.
   - **Found in passing:** `test_spouse_dysfunction_asymmetric_penalty` is flaky, measured 6 failures in
     12 runs — unseeded `Simulator`, asserts on an arbitrary pair, both deltas 0.0 when it fails. The
     "36 tests passing" claim in the proposal was false and is corrected. Not fixed (out of scope for a
     documentation step); a task chip is open for it.
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
   **Verified:** all 103 cited ledger IDs resolve, and every quote attributed to Bowen traces to a ledger
   entry.
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
7. **Write the v2 spec.** **DRAFTED 2026-08-22** — `docs/bowen_agent_model_spec_v2.md`, v2.0-draft,
   **awaiting approval; no code until approved.** Scope: Phases B–D (objects, loop, policy, slow clock,
   reference family). Phase E and F deliberately excluded. Normative-only — rationale and sourcing stay in
   `model_explainer.md`, cited by section, so one rationale lives in one place.
   158 numbered requirements across 14 modules; 272 MUSTs, 65 MUST NOTs. 15 acceptance criteria + 8
   engineering criteria, each with a named test and a mutation target. **Four criteria flagged as not
   code-testable in B–D, each with a human-review proposal**, per the planning rule.
   Verified: all 51 explainer cross-references resolve; all 158 IDs unique.
   **Next: approval, then the implementation plan, then code.**

## Standing decisions
- Engine: deterministic core, no LLM in the decision path.
- Subject: synthetic reference family.
- Societal: three scalar dials (L, X, E) + access vector.
- Use case: **research** — ensembles, seeds, falsification.
- Repo: new package `src/bowen/` alongside `engine.py`; old engine and its 36 tests stay green until
  the new one passes the acceptance tests. Freeze `bowen_individual_family_model_spec.md` as a v1.2
  record; write v2 fresh.
