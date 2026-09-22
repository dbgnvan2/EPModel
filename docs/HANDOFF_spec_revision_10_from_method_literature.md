---
tags: [model-bt, handoff]
status: READY TO RUN — Claude Code, on the Mac, in the EPModel repo
date: 2026-09-22
author: Cowork session (Claude), for Dave
---

# Handoff: fold the method-literature findings into the master spec as revision 10

## What the owner asked for

One master spec that he can review, containing everything the paper research (the 30-paper design-lessons reading, the SEAA addendum, the fourteen full-read sweep papers and their 42 candidates) proposes for the model. Not a separate decisions file. The spec is an approved contract, so nothing from the research becomes an unmarked requirement: every addition and amendment carries a visible proposal marker until the owner strikes it or accepts it. The result is `docs/bowen_agent_model_spec_v2.md` at **version 2.0-draft, revision 10, status FOR APPROVAL**, with a revision-10 section at the end in the same form as revisions 4–9.

Decisions the owner has already made (2026-09-22): Claude Code does the revision; Phase E material goes into the same document as a new module; the untracked deliverables are committed and pushed before the work starts.

## Step 0 — repo housekeeping (do this first, in this order)

1. `.git/index.lock` is stale (0 bytes, created 2026-09-22 22:02 by a status check from a sandbox that cannot delete files). Confirm no git process is running, then remove it.
2. Commit, with a message in the existing `papers: …` style, the untracked deliverables: `papers/sweep_readers_brief_2026-09-19/` (EPMODEL_BRIEF.md and READER_TASK.md — the readers' brief that commits `0153c81` and `0980998` recorded as not preserved; it was in the Cowork session's cloud workspace) and `papers/2026_Light_Society_2506.12078_model_design_refs.md`. Leave the PDFs (root-level and `papers/Model Design papers/`) untracked as they are now unless the owner says otherwise.
3. In `SWEEP_READING_REPORTS_2026-09-20.md` and `SPEC_CANDIDATES_from_preprints_2026-09-20.md`, drop the "brief not preserved" notes added by `0153c81`/`0980998` and point the citations at `papers/sweep_readers_brief_2026-09-19/`. Commit.
4. Push `main`.
5. Create a branch `spec-rev10-draft` and do all spec work on it. Commit at the end of each work-order step below, so the owner can read the diff step by step. Push the branch; do not merge.

## Inputs, and what each is allowed to contribute

| File | Role in this revision |
|---|---|
| `docs/bowen_agent_model_spec_v2.md` (revision 9) | The document being revised. Read §0.1–§0.5 in full before editing anything; they state the rules below. |
| `papers/SPEC_CANDIDATES_from_preprints_2026-09-20.md` | **Primary source of the additions**: candidates C1–C42 with requirement text, module, evidence grade, coverage status against revision 9 (by keyword search — re-verify each against the actual requirements before placing it), and priority. X1–X4 are for the exploratory LLM line and do **not** enter the spec. |
| `papers/DESIGN_LESSONS_model_design_papers_2026-09-17.md` | Second source. §7.8 drafts four Phase E requirement families (`E-DR`, `E-RE`, `E-RM`, `E-SH`) and §7.10 six tests; §8 (and the §8.9 reconciliation) says which of these are superseded by or merged with a C-number — follow §8.9's mapping table where the two overlap, and do not enter a thing twice. §2 lessons that have no C-number and no §7.8 requirement (for example §2.1's alternative-activation-regime test, §2.3's constant sweep with fraction-of-range reporting, §2.7's functional-form checks) enter as requirements only where the design-lessons text already states a testable form; otherwise they are recorded in the revision-10 section as "noted, not specified". |
| `papers/SWEEP_READING_REPORTS_2026-09-20.md` | Evidence behind the candidates. Cite it by part and candidate; do not restate it in the spec. |
| `papers/SPEC_CANDIDATES_plain_language_2026-09-20.md` | Explanation only. Not a source of requirement wording. |
| `papers/2026_Light_Society_2506.12078_model_design_refs.md`, `papers/DIGEST.md`, `papers/PREPRINT_SWEEP_2025-09_to_2026-09.md`, `papers/INDEX.md` | Catalogues and abstract-level notes. **Nothing enters the spec from these**: the 2026-09-21 DIGEST entries (8 papers) were not read in full and no candidate was derived from them. List them in the revision-10 section under "read at abstract level only; no requirement derived". |
| `papers/sweep_readers_brief_2026-09-19/` | Provenance of the reports. Not a source. |

## Rules the revision must obey (from spec §0, plus three new ones)

1. **ID scheme (§0.4).** Every ID matches `M\d+(\.[A-Za-z0-9]+)*`. Existing IDs are never renumbered or re-used. A new requirement inserted next to an existing one takes the next free suffix letter (`M3.D.4` → `M3.D.4a`; if `4a` exists, `4b`). A new requirement at the end of a part takes the next integer. Withdrawn IDs stay and are marked `WITHDRAWN`; this revision withdraws nothing.
2. **Amendments never overwrite.** Where a candidate changes what an approved requirement says (C1 replaces the second sentence of `M3.D.4`), leave the approved text as it is and add a suffixed requirement in the form the spec already uses: `**M3.D.4a** — *amended: …* …`, stating what it replaces and why in one sentence, with the marker. Revisions 7–9 contain this pattern (`M7.D.2d — … this amends M7.D.2a`).
3. **Normative vocabulary (§0.3).** MUST / MUST NOT / SHOULD / MAY, bolded. No subject-negated prohibitions (`M11.D.13`): write "X **MUST NOT** …", never "No X **MUST** …".
4. **One rationale lives in one place (§0.2).** The spec cites; it does not argue. For corpus-derived requirements that place is `model_explainer.md`. For method-derived requirements it is the candidates file. Add one row to the §0.2 table: `papers/SPEC_CANDIDATES_from_preprints_2026-09-20.md` and `papers/DESIGN_LESSONS_model_design_papers_2026-09-17.md` — method-literature rationale for requirements marked `[proposed rev10]`; the spec cites `→ papers/SPEC_CANDIDATES… C8` the way it cites `→ model_explainer §9.4`. **Do not edit `model_explainer.md`** in this revision; whether method rationale should later be mirrored there is a question for the owner (put it in the open-questions list of the revision-10 section).
5. **Parameter rule (§0.5, M0.1, M10.C.1).** Every constant a candidate introduces (the witness weighting constant of C8, the habituation decay of C13, the precision δ and seed cap of C26, the fallback-rate threshold of C12, the minimum effect margins) is `[I]`, is added to the `M10.C.1` list with the marker, and is never given a value in the spec.
6. **Every new `M11.C` criterion** gets a row in the criteria table (ID | Criterion | Test | Phase | Mutation target), a test name embedding its lowercased ID (`test_m11c35_…`), a mutation target, and an entry in the `M13` *Done when* cell of its phase (`M13.2a`, `M11.D.11` assert exact set equality both ways). Direction criteria follow `M0.4`; the three candidates that extend the criterion form (C16 four-level monotonicity, C19 ordinal ranking, C26's UNDETERMINED outcome) are admitted the way `M11.C.16`'s declared ceiling was: state in the row that the levels or ordering are inputs, not calibrated magnitudes, and add each to the `M11.4` exceptions table.
7. **New — the proposal marker.** Every requirement, table row, register entry and amendment added by this revision ends with `⟦proposed rev10 · C8 · Holland 2026⟧` (candidate number and first author; for §7.8 items `· E-RM · SEAA`). Markers are plain text inside the requirement paragraph so the ID regex and the document scans are unaffected. They are removed only by the owner at approval. A requirement without a marker is approved text and is not touched.
8. **New — no paper quotations.** Paraphrase. The candidates and reports files already paraphrase; do not go back to the PDFs for wording.
9. **New — no magnitudes, no invented shape.** If a candidate's requirement text implies a number (four levels, two reference configurations, a majority of seeds), keep it only where the candidates file already states it as a declared input; otherwise write "a declared number".

## Work order

Commit after each step. Each step's commit message starts `spec rev10 (n/7):`.

**Step 1 — Front matter and §0.** Version `2.0-draft, revision 10`; date 2026-09-22; status unchanged (FOR APPROVAL). §0.1: add a marked paragraph stating that Phase E is now drafted in `M17` as a first draft, superseding the sentence that says Phase E is specified after the core is real, and that the owner's 2026-09-22 decision is the reason. §0.2: the new table row (rule 4). §0.3: the stated counts of MUST tokens and named tests will change; `M11.D.14` requires them to be asserted, so recompute and restate them at the end of step 7, not here.

**Step 2 — Design-level items (P1: C1–C15).** Placement, from the candidates file's own module column; verify each against the current text of the module before inserting.

| Candidate | Placement | Form |
|---|---|---|
| C1 | next free suffix on `M3.D.4` (`M3.D.4a` is free as of revision 9) | amendment of M3.D.4's second sentence: counter-based, event-keyed draws; no mutable generator; fixed-count uniforms |
| C2 | `M1.A` (person identifier), `M1.B` (tie id), `M1.C` (triangle id); cross-reference in `M2.A` and `M15.A` | new requirements |
| C3 | the following suffix on `M3.D.4` plus a table of draw classes and keys immediately after it | new requirement + table; the slot/dyad assignments are the reader's proposal and are marked as such in the table |
| C4 | `M11.D` (placebo arm test, next to M11.D.5) and a further `M3.D.4` suffix (single query, cached, debug assert) | new criterion row + new requirement |
| C5 | `M11.D` (order-permutation test) | new criterion row; note it operationalises `M1.F.8` and DESIGN_LESSONS §7.10(b) |
| C6 | `M3` new part (`M3.E` is free) Activation and visibility as named objects; `M16.A` header fields | new requirements |
| C7 | next free suffix on `M1.F.1` (`M1.F.1a` is taken) and next free suffix on `M4.E.1` | new requirements |
| C8 | next free integer in `M4.C` (`M4.C.9` is free) witness appraisal; `M10.C.1` new `[I]` constant; `M11.C` new criterion with the scaled-copy mutant | new requirement + register entry + criterion row |
| C9 | `M4.B.2` (free; inputs the policy may read), `M9.8` (free; per-person belief about ties it is not party to), `M16.A` (log the belief value used), `M11.C` criterion with the true-state mutant | new requirements + criterion row |
| C10 | `M14` (spec coverage) or a new `M14.A` register; the static check as an `M11.D` row | new requirement + criterion row; the register itself can be a stub table the owner fills, marked |
| C11 | next free suffix on `M4.D.1` (`M4.D.1a`–`M4.D.1d` are taken) legality mask; `M16.A` record field | new requirements |
| C12 | the following `M4.D.1` suffix, tie-break and fallback rule stated; `M16.A` flag; `M11.D` guard | new requirements + criterion row |
| C13 | next free integer in `M4.G` (`M4.G.2` is taken) habituation, conditional wording as in the candidates file | new SHOULD requirement + `[I]` constant |
| C14 | `M11.D` reachability check (SHOULD) | new criterion row |
| C15 | `M6.3` (free) disposition at death (and at birth if births occur); next free suffix on `M7.C.1` (`M7.C.1d` is taken); `M11.C` invariant-across-death criterion | new requirements + criterion row; the owner decision it needs (is death an exit?) goes in the open-questions list |

**Step 3 — Test-design items (P2: C16–C25).** All into `M11`: C16, C19 as criterion-form extensions (rule 6); C17, C21, C22 as additions to the mutation protocol paragraph at the head of M11 (suffixed `M11.1b`, `M11.1c`, …); C18 as a new `M11.C` row (Phase D, learning switch in `M10.B`); C20 as an `M16.A` logging rule plus an `M11` test-design rule; C23 as `M10.B` and `M16.A` rules; C24 and C25 next to `M11.4a`.

**Step 4 — Phase E module (P3: C26–C42, plus DESIGN_LESSONS §7.8 `E-DR`, `E-RE`, `E-RM`, `E-SH`).** New module `## M17 — Phase E: ensemble runner, arms and readouts  ⟦first draft, rev10⟧`, with parts: `M17.A` ensemble size and stopping (C26, C27, C25's paired statistic), `M17.B` per-seed readouts (C28, C29, C41, C39, C40), `M17.C` variance components and initial conditions (C30, C32, C33, `E-RE`), `M17.D` control and structural arms (C34, C35, `E-RM`), `M17.E` sweeps and sensitivity (C31, C37, C38, `E-DR`, DESIGN_LESSONS §2.3 fraction-of-range), `M17.F` shocks and settling (`E-SH`, §7.10(e)), `M17.G` reporting and audit (C36, C42). Follow §8.9's mapping where a §7.8 item and a C-number coincide, and enter each once with both references in the marker. Every `M17` requirement carries the marker; the module header states that `M17` is Phase E scope and that `M13.3`'s rule (Phase E not built during B–D) applies to all of it.

**Step 5 — Registers.** `M10.C.1`: every new `[I]` constant. `M13`: new criteria in their phases' *Done when* cells; `M17` gets a Phase E row. `M11.4` exceptions table: the criterion-form extensions.

**Step 6 — Revision-10 section.** Append `### Revision 10 — the method literature, 2026-09-22` after Revision 9, in the same style: what was read (the four bodies of material, with file paths), a table of every marked item (spec ID | candidate or E-ID | source paper | form: new / amendment / criterion / register / Phase E), the list of candidates judged already covered and not entered (candidates file §9), the abstract-only papers not entered (DIGEST 2026-09-21, Light Society refs), and an **open questions for the owner** list: death as exit or ledger (C15); whether method rationale should be mirrored in the explainer (rule 4); whether `M17` stays in this document after approval or is split out; the slot/dyad key assignments in C3's table; whether C13's relief term exists.

**Step 7 — Consistency checks, then restate the counts.** Run, and record the results in the revision-10 section:
- new `M11.C` criteria start at `M11.C.35` (the table ends at `M11.C.34` in revision 9);
- every ID in the document matches the §0.4 regex; no duplicates; the set of IDs present in revision 9 is unchanged (extract both with the same regex and diff);
- every `M11.C` and `M16.F` criterion with a phase appears in that phase's *Done when* cell and vice versa (`M11.D.11`);
- no subject-negated MUST (`M11.D.13`);
- every marker resolves to a C-number in the candidates file or an E-ID in DESIGN_LESSONS §7.8;
- no line contains a quotation of eight or more words from a paper;
- the §0.3 counts (MUST/MUST NOT tokens, named tests) restated to the recomputed values, with the date;
- `M14`'s coverage figures restated likewise.
If the project has scripts for the `M11.D.8`–`M11.D.14` scans, run them; if not (no code exists yet), do the checks with `grep`/Python one-offs and say so.

## What not to do

- Do not merge to `main`, do not remove a marker, do not alter approved requirement text except by suffixed amendment.
- Do not enter X1–X4, the DIGEST papers, the Light Society reference list, or anything from the plain-language file.
- Do not invent a magnitude, a functional form or a default value anywhere; do not choose between slot and dyad keying (C3) — present the reader's proposal as a proposal.
- Do not edit `model_explainer.md`, `CLAUDE.md`, or `docs/theory/`. (`CLAUDE.md`'s table still says revision 6; note it in the open questions rather than fixing it here.)
- Do not run this concurrently with a Cowork session editing the same files; the design-lessons file already diverged once that way.

## Deliverable

Branch `spec-rev10-draft` pushed to origin, seven commits, and a one-paragraph note to the owner saying how many requirements were added (by module), how many amendments, how many new criteria, how many `[I]` constants, and which checks in step 7 passed, with the open-questions list copied out so he can answer it before reading the document.
