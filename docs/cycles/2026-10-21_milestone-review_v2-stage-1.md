# Milestone review: version 2 stage 1 (mini1, 2026-10-21)

The first milestone critical review of EPModel, run as `docs/milestone_review_brief.md` (DRAFT 2026-10-21) describes,
at the owner's request through the EPModel session on davemini2. It reviews the whole model against the brief's four
lenses, theory fidelity first, with the changes since the baseline in focus. The reviewing session edited nothing
except this report and fixed nothing.

## Header

- **Commit reviewed:** `9cf85529b4186fa7516996461f29de35601b8a74` (`origin/main`), in a worktree on branch
  `review/2026-10-21-v2-stage-1`.
- **Baseline:** the tag `version-1` (`d62cfdc`). This is the first milestone review.
  `git diff --stat version-1..HEAD` lists 144 changed files; the brief's §3 step 5 filter (no deleted files, no
  generated pages, no dated logs) leaves 110 (+8,036 / −708), which were split into the four lens slices.
- **Machine:** Mac mini1 (Darwin 24.6.0).
- **Suite:** `1003 passed, 49 deselected in 180.93s` on Python 3.11.15 (a virtualenv with `requirements.txt`, outside
  the repo), matching CLAUDE.md's "all 1003 tests pass".
- **CI for the commit:** run `39656896009` (push to `main`, head `9cf8552`), `completed success`: `test (3.11)`
  success, `test (3.12)` success, `test (3.13)` success.
- **Tokens used:** 462,483 for the subagents (L1 46,465; L2 45,176; L3 209,140; L4 55,086; challenge 106,616, from
  each result's usage total). The orchestrating session's own use: not measured, because the app shows no counter for
  it.
- **Wall time:** 13 minutes from the start of §3 step 3 (11:54 local) to the challenge pass's return (12:08), plus
  the time to write this file. The worktree, the virtualenvs and the cost statement came before the clock started.
  The four lens workers ran in parallel (5.6–6.6 minutes each); the challenge pass took 4.1 minutes.
- **Findings:** 0 blocker, 4 major (all confirmed in challenge), 22 minor (2 of them downgraded from major in
  challenge, 20 not challenged).

**Deviation from the brief.** §4 asks the orchestrating session to read the spec end to end (the `reviewer` skill's
Phase 0) before fanning out. The four workers were launched first and the read was done while they ran; the slices
came from the spec's section headings and §4's slice list. The read was completed before consolidation, and the
orchestrator's own spec findings (MR-12, MR-13) were checked against the code.

## Coverage matrix

"Full" = read in full; "diff" = every changed hunk read; "sampled" = named ranges read; "searched" = grepped for the
claim being checked; "—" = not in that lens.

| Area | L1 theory fidelity | L2 spec against code | L3 documents | L4 engineering drift |
|---|---|---|---|---|
| Spec `bowen_agent_model_spec_v2.md` (2,585 lines) | M10–M11.G (1157–1529) full | changed lines (3–4, 1116, 1566) + 631–633, 751, 848–852, 1252–1282 | front matter and status | M16.B (1931–1956) full |
| The orchestrator's read of the spec | lines 1–2121 full (M1–M17); revision history (2122–2585) headings only | — | — | — |
| `src/bowen/` | diff (every hunk) + literal grep of `engine/` | diff, with enclosing functions in symptoms, recompute, external, act, load, criteria | searched (I/O calls, docstrings) | diff (engine, counting, ensemble, io, policy, readouts, render, scenario) |
| `config/bowen/` | `constants.md`, `criterion_roles.md`, `act_classes.md` full; diff of `constants_changes.md`, `criteria.md`; 13 other files not read | `act_classes.md` against M5.B.4/5 | — | — |
| `docs/model_explainer.md` (2,094 lines) | 1–116, 374–423, 791–866, 935–1084, 1441–1540, 1953–2094 + diff; rest grepped for KB/Tape | — | status line | — |
| `docs/theory/` | `_SOURCE_QUALITY.md` full; `_LEDGER.md`, `_KERR_INTERVIEWS.md`, `kb/kb04.md` searched | — | `_STATUS.md` full | — |
| Reading list, checklist, README, CLAUDE.md | reading list searched | — | full | — |
| `docs/spec_coverage.md` + overrides | — | header, M11.C rows and named rows sampled; overrides 1–120 + every "done" note grepped | searched | 5 override entries sampled |
| Generated records (`docs/*_record.md`) | searched (class labels, framing) | Phase C mutation 31–106, Phase D mutation 1–42; Phase B not read (frozen) | `version_2_stage_1_changes.md` 1–63 | `ensemble_record.py` and its test full |
| `tests/` | — | diff (except `test_mutant_runner.py`, skipped as infrastructure); `test_phase_c_gate.py` sampled | `test_doc_claims.py`, `doc_claims.json` full; `test_doc_status.py` 20–119 | diff + `test_engine_purity.py` full |
| `tools/`, CI, `pytest.ini`, `requirements.txt` | — | — | — | diff + `tests.yml`, `pytest.ini`, `requirements.txt` full |
| `TODO.md` | — | full (1,045 lines) | searched | searched (known items not re-reported) |
| DECISIONS files | `DECISIONS FOR APPROVAL.md` searched | — | CUTOFF SEMANTICS status read | — |

Left out by the brief's §3 step 5 filter, and searched only for a claim being checked: the deleted grid-engine files,
`docs/records_cache/`, `docs/review/`, the generated records, `docs/spec_coverage.md`, `docs/version_1_status.md`,
the dated logs (`CHANGELOG.md`, `TODO.md`, `LEARNINGS.md`), `papers/` and `tools/spec_coverage_phase_b.json`.

## Findings

### Major findings

#### MR-1 — The spec's framing and calibration rules quote ASR interviews verbatim, two of them attributed to Bowen in person

- **Found by:** L1 · **Where:** `docs/bowen_agent_model_spec_v2.md:1467` (M11.F.1), `:1471` (M11.F.3); also `:1206` (M10.C.2) and `:1167` (M10.A.2), which quote verbatim but cite the interview
- **Severity:** major · **Confidence:** high · **Verified:** confirmed in challenge (major for M11.F.1 and M11.F.3; M10.A.2 and M10.C.2 alone would be minor)
- **What is wrong:** the clauses that govern how the model may be presented quote Hijack-set interviews (KB09, KB13)
  word for word and attribute the words to Bowen. `_SOURCE_QUALITY.md:71` requires these to cite the interview and
  forbids verbatim quotation. No exception is recorded, and the brief itself (:169–171) still asks whether the KB11
  quote is one.
- **Quoted:**
  - spec:1467 — "His stated goal was "an **open theoretical system**, where the basis for new knowledge is **research and science rather than anything I said**" … → `kb/kb09.md` · K09.1"
  - spec:1471 — "Bowen rejected **general systems theory** as a base in favour of **natural systems**, on the ground that general systems "came out of man's head, along with mathematics""
  - `_SOURCE_QUALITY.md:71` — "Any finding drawn from it **MUST** cite the interview rather than a person, and **MUST NOT** be quoted verbatim."
- **How checked:** L1 read spec 1157–1529 and `_SOURCE_QUALITY.md` in full. The challenge pass found the passages
  verbatim at `kb09.md:14,18,32` and `kb13.md:37` (`kb13.md:11`: "ASR source — no verbatim quotation"), searched
  `docs/` for a recorded exception and found none, and found that spec :2455 already names this defect class. The
  challenge pass also found that these clauses are "checked by review only" (`docs/plan_spec_coverage_phase_d.md:279`, G6), so no test will catch them.
- **Fix:** paraphrase M11.F.1, M10.A.2 and M10.C.2 and cite the interview. Restate M11.F.3 from Bowen's written
  Epilogue (FE11.2, already quoted at `model_explainer.md:1457`), which may be quoted and attributed. See SP-1 for the class fix.

#### MR-2 — The explainer still lists the withdrawn "8 to 10 generations" figure as a usable calibration target

- **Found by:** L1 · **Where:** `docs/model_explainer.md:963` (under the heading at :945); spec `:1206` (M10.C.2)
- **Severity:** major · **Confidence:** high · **Verified:** confirmed in challenge
- **What is wrong:** ledger L16.3 and DECISIONS A1 (approved) withdrew the figure as rhetorical. The explainer's §8
  table, headed "usable as calibration targets", still lists it as `[#]` with no mark. M10.C.2 records the withdrawal
  but also points to "the durations in `model_explainer.md` §8", so a reader of the table is told the opposite of a
  reader of the spec. `kb/kb11.md:17` names this exact table as where the withdrawal had to land.
- **Quoted:**
  - explainer:945 — "**Durations the corpus actually states** — usable as calibration targets, each verified verbatim."
  - explainer:963 — "| Generations to schizophrenia | **8 to 10** (revised *up* from "at least three") | `[#]` Ch16 · L16.3 |"
  - `_LEDGER.md:1402` — "**⛔ WITHDRAWN at pass 3.**"
- **How checked:** L1 read explainer 935–1084, ledger 1395–1413 and spec 1206. The challenge pass found no
  "withdrawn", "A1" or "KB11" mark at §8; found that DECISIONS A1 is approved (:51) with "propagation complete" (:3);
  and found that the 2019 reinstatement (`_LEDGER.md:2088–2093`, KS-B.1) brings the range back only as a shape
  constraint on `M11.C.6`, cited to KS11.1, not as a Ch16 duration "revised up".
- **Fix:** remove the row from §8 or mark it "withdrawn, see L16.3 / DECISIONS A1; the range survives only as
  `M11.C.6`'s shape constraint (KS-B.1)".

#### MR-3 — KB10 quoted verbatim and attributed to Bowen in the explainer

- **Found by:** L1 · **Where:** `docs/model_explainer.md:382`
- **Severity:** major · **Confidence:** high · **Verified:** confirmed in challenge
- **What is wrong:** KB10 is Hijack-only, so it can be neither quoted nor attributed to one speaker; the explainer's
  own legend (:57) says so. The defect starts in the extraction file: `kb/kb10.md:11` states the rule and :15–17 head
  the passage "Bowen names THREE variables" and quote it.
- **Quoted:**
  - explainer:382 — "Bowen names **three** variables for societal process — "*one, a different way of thinking… another is differentiation of self… and a third is the intensity of anxiety*" (`KB10` · K10.1)."
  - `kb/kb10.md:11` — "**Citation rule:** cite `KB10`. ASR source — no verbatim quotation."
- **How checked:** L1 read explainer 374–423 and `_SOURCE_QUALITY.md`; the challenge pass read `kb10.md:1–25` and `_SOURCE_QUALITY.md:24–25, 41` (only interview #1 has an Otter version).
- **Fix:** "The KB10 interview names three variables … (K10.1)", paraphrased. Fix `kb10.md`'s heading the same way.

#### MR-4 — KB04 quoted verbatim and read as "what Bowen means", in the explainer and the spec

- **Found by:** L1; spec instance found in challenge · **Where:** `docs/model_explainer.md:805`; `docs/bowen_agent_model_spec_v2.md:394`
- **Severity:** major · **Confidence:** high · **Verified:** confirmed in challenge (the verbatim quotation and the attribution); the claim that part of the passage is Kerr's turn could not be shown
- **What is wrong:** an unlabelled Hijack-set passage is quoted and its meaning attributed to Bowen. The explainer's
  ellipsis spans a second-person line ("You don't talk about them as much as you used to") that reads like an
  interjection, though without speaker labels this cannot be proved either way.
- **Quoted:**
  - explainer:805 — "Bowen means late in life by "I don't do much with triangles anymore… no difference how many triangles are out there""
  - spec:394 — "Bowen says late in life that he does "not do much with triangles anymore" and that it makes "no difference how many triangles are out there""
  - `kb/kb04.md:11` — "findings cite the interview (`KB04`), not a speaker … ASR source — **no verbatim quotation**."
- **How checked:** L1 read explainer 791–866 and `kb04.md:117`; the challenge pass read `kb04.md:1–12, 110–129` and found the spec instance; the orchestrator confirmed spec :394.
- **Fix:** paraphrase both as "the KB04 interview lands on …" and drop "Bowen means" / "Bowen says".

### Findings downgraded from major in challenge

#### MR-5 — KB01 quoted verbatim in the explainer (attribution is allowed; quotation is not)

- **Found by:** L1 · **Where:** `docs/model_explainer.md:1449`
- **Severity:** minor (downgraded from major) · **Confidence:** high · **Verified:** confirmed in challenge, downgraded
- **What is wrong:** the passage quotes the Otter transcript of KB01 word for word. KB01 has clean speaker labels, so
  attributing the substance to Bowen is allowed (`_SOURCE_QUALITY.md:24`), but it is ASR, so quotation is not. The
  explainer contradicts its own table at :1497 and its note at :1455.
- **Quoted:**
  - explainer:1449 — "grounded in nature — "in the force that causes grass to grow, that causes the world to turn" — and repeatedly anticipates measurement: "there are **biochemical markers or indicators that could be measured**"
  - `_KERR_INTERVIEWS.md:17` — "ASR, so substance is reliable and verbatim quotation is not."
- **How checked:** L1 read explainer 1441–1540; the challenge pass read `_KERR_INTERVIEWS.md:1–40`.
- **Why downgraded:** the attribution and the substance are both permitted; only word-level exactness misleads, and
  the next subsection (:1455) warns the reader that the text rests on transcripts that may not be quoted.
- **Fix:** paraphrase; keep the attribution.

#### MR-6 — The coverage page's headline counts 291 IDs done without saying most carry no run mutation

- **Found by:** L2 · **Where:** `docs/spec_coverage.md:23` (and :4–8, :396); spec `:1252` (M11.1)
- **Severity:** minor (downgraded from major) · **Confidence:** high · **Verified:** confirmed in challenge, downgraded
- **What is wrong:** M11.1 says a test must not count as coverage until its mutation has been run. The page requires
  that only of `M11.C` criteria, so about 277 "done" rows rest on tests with no recorded mutation. The page's own
  M11.1 row says so; the headline does not.
- **Quoted:**
  - spec:1252 — "**M11.1** A test **MUST NOT** be counted as coverage until its mutation has actually been run, not reasoned about."
  - coverage:23 — "**575 IDs: 291 done, 136 partial, 148 not done.**"
  - coverage:396 — "every other ID on this page counts as done on a test with no recorded mutation"
- **How checked:** L2 read spec 1252 and coverage 1–23 and 396, and compared TODO.md 56–91 (not listed). The
  challenge pass found that M14.1 (spec :1578) defines *done* without a mutation condition, that the M11 preamble
  (:1250) speaks of two-arm criteria, and that the owner approved the M11.1 row as partial in G4 on 2026-10-20
  (`docs/plan_spec_coverage_phase_d.md:277`).
- **Why downgraded:** the page follows M14.1's definition and discloses the M11.1 gap in an owner-approved row; what
  remains is a headline with no caveat.
- **Fix:** add one sentence after :23 saying which "done" rows are mutation-proved; settle in the spec whether M11.1
  is scoped to the criteria (owner question Q4).

### Minor findings

Every minor finding below is **not challenged** (brief §5: the challenge pass checks blocker and major findings
only); triage checks them (§7).

#### MR-7 — The one-year slow tick is graded `[#]` in the explainer and `[I]` in the config

- **Found by:** L1 · **Where:** `docs/model_explainer.md:940`; `config/bowen/constants.md:84`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** the explainer presents the one-year period as a stated quantity. Neither ledger entry it cites
  (L16.3, L11.3) gives a period of one year, and the config grades the same value `[I]`.
- **Quoted:**
  - explainer:940 — "| **Slow tick** | 1 year | … | `[#]` Ch16 · L16.3 — generational change; `[#]` Ch11 · L11.3 — "a certain amount of time **on the calendar**" |"
  - constants.md:84 — "`slow_tick_fast_ticks` | 52 | [I]"
- **How checked:** L1 read `_LEDGER.md` L16.3 (1395–1413) and L11.3 (1036–1045).
- **Fix:** grade the one-year period `[I]`; keep the `[#]` citations only for the claim that a slow clock is needed.

#### MR-8 — 1979 Tape text quoted verbatim in the explainer and the spec

- **Found by:** L1 · **Where:** `docs/model_explainer.md:801`; `docs/bowen_agent_model_spec_v2.md:1323` (M11.C.11)
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** Tape transcripts are ASR and cleared for substance only; both lines quote them.
- **Quoted:**
  - explainer:801 — "`[T]` 1979 Tape 5 — intensity "is determined by level of differentiation and principally anxiety"."
  - `_SOURCE_QUALITY.md:23` — "**Not** verbatim quotation"
- **How checked:** L1 read explainer 791–866 and `_SOURCE_QUALITY.md`; grepped the spec for the Tape 6 quote.
- **Fix:** paraphrase both Tape citations. Same class as the systemic pattern SP-1.

#### MR-9 — The reading list names criterion classes the spec does not have, in files that do not hold them

- **Found by:** L1 (overlaps L3) · **Where:** `docs/REVIEWER_READING_LIST.md:77`
- **Severity:** minor · **Confidence:** medium · **Verified:** not challenged
- **What is wrong:** the entry says `criteria.md` and `criterion_roles.md` classify each criterion as "premise" or
  "finding". Neither file does, and "finding" is not an `M11.5` class (premise / composite / check / mixed).
- **Quoted:** :77 — "whether it is a **premise** (passing confirms the code) or a **finding** (passing or failing says something about the model)."
- **How checked:** L1 read `criterion_roles.md` in full and grepped `criteria.md` (hits only at its status lines 42, 44); read spec `M11.5`.
- **Fix:** point the entry at spec `M11.5` and the class columns in the phase records, using the `M11.5` names.

#### MR-10 — M11.4 says "Three criteria" in an "exhaustive" list whose table has six active rows

- **Found by:** L2 · **Where:** `docs/bowen_agent_model_spec_v2.md:1270–1281`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** the table's active rows are M11.C.25, .31, .26, .11, .37 and .14 (plus two withdrawn rows and one
  listed for completeness). A reader applying "a fourth exception is a defect" would flag four of them.
- **Quoted:** :1270 — "Three criteria are **not** a direction of difference between two arms, each for a stated reason. **The list is exhaustive**: any other criterion that is not a two-arm direction is a defect, not a fourth exception."
- **How checked:** L2 read 1270–1282 and counted rows. Not in the stage 1 diff; not on TODO's disagreement list.
- **Fix:** replace "Three" with the count, or drop the number and keep "exhaustive" pointing at the table.

#### MR-11 — M4.D.1f is marked done although its exhausted-`life_energy` fallback cannot exist

- **Found by:** L2 · **Where:** `docs/spec_coverage.md:227` (from `tools/spec_coverage_overrides.json`); spec :751
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** the spec requires a fallback for exhausted `life_energy`; the coverage note says `life_energy`
  is not built. The same page marks M1.A.10 (:63) and M6.I.3 (:302) "not done" for the same absence.
- **Quoted:**
  - spec:751 — "a fallback rule for a non-finite score, exhausted `life_energy` and an empty legal set."
  - coverage:227 — "`life_energy` is not built, so it cannot be exhausted"
- **How checked:** L2 read the three coverage rows; grep for `life_energy` in `src/bowen` finds only `life_energy_ratio` (`objects.py:142`).
- **Fix:** mark M4.D.1f partial in the overrides, naming the unbuilt clause and M1.A.10.

#### MR-12 — The spec says the live-positions predicate is called from four places; the code calls it from three

- **Found by:** orchestrator (L2) · **Where:** `docs/bowen_agent_model_spec_v2.md:1101`, `:1116` (first sentence of M8.5), `:1564` (M13.1); `docs/spec_coverage.md:503`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** the M8 preamble and M13.1 say four mechanisms call the predicate. `positions_live` is called from
  `recompute.py:133`, `visibility.py:109–110` and `external.py:104`; `avoidance_available` (`live_positions.py:82`,
  the M5.C site) has no caller in `src/bowen`. `spec_coverage.md:362` already marks M8.5 partial for this and the
  explainer (:1369) says so, but `spec_coverage.md:503` marks M13.1 done. Present at the tag too, so not introduced by stage 1.
- **Quoted:**
  - spec:1101 — "This is one rule used in four places. It **MUST** be implemented once and called from all of them."
  - spec:1564 — "**M13.1** M8 **MUST** land in Phase B, before the policy, because four separate Phase C mechanisms call it."
  - explainer:1369 — "`avoidance_available` is defined and tested, but no mechanism calls it."
- **How checked:** `grep -rn avoidance_available src tests` (hits only in `live_positions.py` and its test); read
  `recompute.py:121–150`; `git diff version-1 HEAD` shows neither sentence changed.
- **Fix:** an owner question (Q5) decides whether the M5.C gate is to be built; until then, say "three of four" in
  the M8 preamble and M13.1, and mark M13.1 partial.

#### MR-13 — Two stale status statements in the reference-family table

- **Found by:** orchestrator (L3) · **Where:** `docs/bowen_agent_model_spec_v2.md:544`, `:581`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** Teodor's row still says "awaiting owner review", though the front matter (:4) says revision 13
  was "decided 2026-10-10". `M2.A.0i` still says Dr Halim's sex is "not stated" and **MUST** be declared, though the
  table (:555) and `M2.A.0k` (:589) declare her female (revision 28, owner 2026-10-13). Both predate the tag.
- **Quoted:**
  - :544 — "36 proposed at revision 13, as Ravi's was corrected — awaiting owner review* ⟦rev13⟧"
  - :581 — "**Leo and Dr Halim are not stated** and **MUST** be declared before Phase D"
  - :589 — "Dr Halim is female and 35 at `t0`"
- **How checked:** read the lines; `config/bowen/family_phase_d.md:18` confirms Teodor's level of 36 is in use.
- **Fix:** drop "awaiting owner review" (cite the 2026-10-10 decision); add "Dr Halim declared female, owner 2026-10-13" to M2.A.0i.

#### MR-14 — `_STATUS.md` keeps "the grid engine stays green" as a standing decision, and a `doc_claims.json` entry hides it

- **Found by:** L3 · **Where:** `docs/theory/_STATUS.md:469–470`; `tests/doc_claims.json:37–39`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** the file CLAUDE.md calls "read this first" lists, under an undated "Standing decisions" heading,
  a rule revision 33 withdrew (M13.2). The suite-size test strips the phrase as "dated" history, but it is not in a
  dated section. It also says "36 tests"; the retired file held 37 (`git show version-1:tests/test_simulator.py | grep -c "def test_"` = 37, matching spec M13.2).
- **Quoted:**
  - `_STATUS.md:469` — "old engine and its 36 tests stay green until"
  - `doc_claims.json:39` — "the run log's section dated 2026-08-28, describing the grid engine's suite then"
- **How checked:** L3 read `_STATUS.md` in full (nearest dated section :419 is 2026-08-24) and `test_spec_consistency.py:341–346`; the orchestrator counted the tag's tests.
- **Fix:** mark the decision withdrawn at revision 33 (S1.6, D1, M13.2); correct the `doc_claims.json` entry's `why`.

#### MR-15 — The spec's status line still ends "no code until the implementation plan is also approved"

- **Found by:** L3 · **Where:** `docs/bowen_agent_model_spec_v2.md:4`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** the same line records four phases and revision 33 as built. `test_doc_status.classify` reads only the first status word, so the clause is never checked. Carried from the tag.
- **Quoted:** :4 — "built 2026-10-19) — no code until the implementation plan is also approved"
- **Fix:** delete the clause, or scope it to "version 2 stage 2 onward".

#### MR-16 — The checklist calls DECISIONS — CUTOFF SEMANTICS "(open)"; the document says BUILT

- **Found by:** L3 · **Where:** `docs/SPEC_REVIEW_CHECKLIST.md:79`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **Quoted:**
  - checklist:79 — "whether the engine's CUTOFF should become rejection (open)."
  - `DECISIONS — CUTOFF SEMANTICS.md:3` — "status: BUILT 2026-10-11 — option B"
- **How checked:** L3 read both; `test_doc_status.py:37` `_ENTRY` matches only bold `**Label:**` entries, so `- [ ]` items are never compared.
- **Fix:** change "(open)" to "built 2026-10-11 (option B, `M1.B.3c`)"; extend D5b to list items or record them as out of its scope.

#### MR-17 — The reading list's status line covers less than was changed after the owner's approval

- **Found by:** L3 · **Where:** `docs/REVIEWER_READING_LIST.md:3–4` (repeated at `SPEC_REVIEW_CHECKLIST.md:13–14`)
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** it says only "Before you start" changed after the 2026-10-20 approval; §8's 2026-10-21 entry (:206) and §9's link to the brief drafted 2026-10-21 (:218) were also added after it.
- **Fix:** name all three revised parts as "revised 2026-10-21, not yet reviewed by the owner".

#### MR-18 — The reading list's list of what the tag lacks leaves out two documents it links

- **Found by:** L3 · **Where:** `docs/REVIEWER_READING_LIST.md:24`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** `plan_doc_claim_tests.md` and `milestone_review_brief.md` are new since the tag and linked at :196 and :218, but not named.
- **Fix:** add both to the parenthesis, or write "for example".

#### MR-19 — "Four stages" for a version 2 plan that has five

- **Found by:** L3 · **Where:** `docs/REVIEWER_READING_LIST.md:171`; `docs/SPEC_REVIEW_CHECKLIST.md:166`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** `docs/version_2_plan.md` has stages 0–4 (:77, :106, :134, :150, :364); both pages omit stage 4 (corpus fidelity).
- **Fix:** "five stages (0–4)" in both places.

#### MR-20 — "Compares every record" overstates `version_2_stage_1_changes.md`

- **Found by:** L3 · **Where:** `docs/theory/_STATUS.md:9–10`; `docs/REVIEWER_READING_LIST.md:143`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** the page compares seven machine-readable records and lists others under "Not compared here" (:26–35), including the regenerated `phase_c_diagnostic_record.md`.
- **Quoted:** `_STATUS.md:9–10` — "compares every / record with the tag `version-1`: no verdict changed and no value differs."
- **Fix:** "compares the seven machine-readable records … the diagnostic record and traces are listed as not compared".

#### MR-21 — The version 1 review guide still says version 2 work waits on questions answered on 2026-10-19

- **Found by:** L3 · **Where:** `docs/REVIEW_version_1_guide.md:18–19` (also :6–7)
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** `version_2_plan.md:3–4` records §5 answered and the plan approved on 2026-10-19; revision 33 is
  built; TODO.md no longer holds the condition the guide cites. :6–7 says "the spec is frozen at v2.0 revision 32"; only version 1 is.
- **Fix:** route Change items to version 2 stage 3 proposals; reword :6–7 to "version 1's spec is frozen at revision 32 (the tag `version-1`)".

#### MR-22 — `_STATUS.md`'s file map contradicts its own sections on two trust labels

- **Found by:** L3 · **Where:** `docs/theory/_STATUS.md:254`, `:257`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **Quoted:**
  - :257 — "| `../agent_model_proposal.html` | … | **STALE — see "Known stale" below** |"
  - :265 — "## ~~Known stale in the proposal~~ — RESOLVED at step 4 (2026-08-22)"
  - :254 — "current; 12 of 23 items probed" (against :137 "all 23 validation items probed")
- **Fix:** set the proposal's trust to "background; §10 current" and the lectures row to "all 23 items probed".

#### MR-23 — The Phase C ensemble record still reruns in full for a renderer change, and its currency test checks only the hash

- **Found by:** L4 · **Where:** `tools/ensemble_record.py:31–33, 211`; `tests/bowen/test_ensemble_record.py:33`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** stage 1 (S1.8c) moved the Phase D records onto `docs/records_cache` and made their test compare a
  full render; this record was left on the old path. `docs/version_2_plan.md:129–131` reads as if Phase C's records were already cached. A hand edit to its verdict rows still passes its test.
- **Quoted:**
  - ensemble_record.py:31–33 — "the record is not re-rendered from a cache" … "HASHED_TOOLS = (Path(__file__).resolve(),)"
  - test_ensemble_record.py:33 — "assert match.group(1) == _tool.code_hash(*_tool.HASHED_TOOLS)"
- **How checked:** L4 read the tool in full and compared with `tests/bowen/test_phase_d_record.py` (`_rendered(1) == record_text()`).
- **Fix:** cache each verdict under `result_key(code_hash(), None, cid)`, render from the cache, drop the tool from its own hash, and compare the full render in the test (P38).

#### MR-24 — The "frozen" `tools/version_1_status.py` is still imported by live code and a live test, and stage 1 edited it

- **Found by:** L4 · **Where:** `tools/version_1_status.py:5–6`; `tools/spec_coverage.py:235`; `tests/bowen/test_spec_coverage.py:95`
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **Quoted:**
  - version_1_status.py:5–6 — "Frozen since 2026-10-19: this copy no longer writes, and" / "no test runs it; the tests run the tag's copy (below)."
  - spec_coverage.py:235 — "from tools.version_1_status import dispositions"
- **Fix:** move `dispositions()` to a live module, or say in the docstring that `spec_coverage` depends on this file.

#### MR-25 — A config test is named for a failure it never triggers

- **Found by:** L4 · **Where:** `tests/bowen/test_config.py:374–382` (`test_s1_3a_kinds_without_act_classes_fail_where_a_class_is_read`)
- **Severity:** minor · **Confidence:** high · **Verified:** not challenged
- **What is wrong:** it asserts only that the default is `None`; it never calls the policy or `external.coach_side` with unclassed kinds, and has no `pytest.raises`. A later `or ()` default would pass it.
- **Fix:** build a state with `EventKinds(kinds.mechanisms)` and a non-empty window, call `coach_side` and `legal_outcomes`, assert `TypeError`.

#### MR-26 — `test_render` reruns the 104-week seed-7 Phase C run the tool already made

- **Found by:** L4 · **Where:** `tests/bowen/test_render.py:267`; `tools/phase_c_trace.py:81`
- **Severity:** minor · **Confidence:** high (repeat); cost not measured · **Verified:** not challenged
- **Fix:** a `run_with_series(seed)` helper returning `(run, series)`, built on by `outputs()` and cached in the test (P38).

## Systemic patterns

- **SP-1 — The ASR quotation rule is half applied (MR-1, MR-3, MR-4, MR-5, MR-8).** The rule has two parts: cite the
  interview, and do not quote verbatim. The project applies the first part unevenly and the second hardly at all.
  - **Where it starts.** The `kb/*.md` extraction files state "no verbatim quotation" in their headers and then quote
    in their bodies (`kb10.md:11` against :17; `kb04.md:11` against :117). The explainer, the spec, the ledger and
    DECISIONS copy from there.
  - **Why no check caught it.** The 2026-08-28 corpus-fidelity sweep (`_STATUS.md:185–188`, `TODO.md:975`) checked
    that quotations were "verbatim in the corpus", which treats an exact ASR match as a pass instead of the breach.
    The fix at spec :896 on the same day corrected the attribution and kept the verbatim quote.
  - **Dates.** No quote was found to predate the rule (`_SOURCE_QUALITY.md:4`, 2026-08-22); the challenge pass could
    not order events within 2026-08-22.
  - **Fix as a class:** grep the spec, the explainer, the ledger, DECISIONS and `kb/*.md` for quotation marks next to
    a `KB` or `Tape` citation, paraphrase them in one batch, and add a document-claim test that fails on one.
- **SP-2 — Prose no test reads goes stale (MR-9, MR-10, MR-12 to MR-22).** The document-claim tests (D5b) check
  bold checklist labels, README sentences and each document's first status word. Trailing status clauses, the
  checklist's `- [ ]` items, the reading list's body, standing-decision lists, counts in prose ("three criteria",
  "four places", "four stages", "36 tests") and the reference-family table's status notes are outside them, and every
  stale statement this review found sits in one of those places. CLAUDE.md :92–94 asks for the reading list to be
  updated in the same commit, but no test enforces it.
- **SP-3 — "Done" claims more than is proved (MR-6, MR-11, MR-12).** The coverage page counts as done an ID whose
  required clause cannot exist (M4.D.1f), an ID whose spec text names a caller that is not built (M13.1), and, in its
  headline, IDs with no run mutation.
- **SP-4 — A cached path is added for some outputs and a sibling still reruns (MR-23, MR-26).** Stage 1 moved the
  Phase D records onto `docs/records_cache`; the Phase C ensemble record and `test_render`'s seed-7 run still
  recompute from scratch (P38).

## Refuted in challenge

No major finding was refuted. Three parts of findings were changed in challenge:

- **MR-5 (L1-05), downgraded to minor.** Attribution to Bowen is allowed for KB01, whose Otter transcript has clean
  speaker labels (`_SOURCE_QUALITY.md:24`); only the verbatim quotation breaches the rule.
- **MR-6 (L2-01), downgraded to minor.** Spec M14.1 (:1578) defines *done* without a mutation condition, and the
  coverage page's M11.1 row, which states the gap, was approved by the owner as partial on 2026-10-20
  (`docs/plan_spec_coverage_phase_d.md:277`).
- **MR-4 (L1-04), one claim not shown.** That part of the KB04 passage is Kerr's turn cannot be shown from an
  unlabelled transcript; the finding stands on the quotation and the attribution alone.

## Checked and clean

- **L1 · `config/bowen/constants.md`:** every value is graded `[I]`; no note presents the corpus as setting a value;
  `nodal_spike_intensity` is cited to the owner, and the mortality and budget values make no source claim
  (explainer :1001 agrees).
- **L1 · `config/bowen/act_classes.md` (new):** SPLIT is cited to L19.3 only for what the act is; side-taking is
  framed as revision 32's owner-approved choice (checked against `_LEDGER.md:1619–1631`).
- **L1 · `constants_changes.md`, `criteria.md` diffs:** stage 1 changed no constant; `criteria.md` changed status lines only.
- **L1 · changed `src/bowen` code, rules that write outcomes:** every hunk read; the changes are `math.fsum` and
  `count()` swaps, act classes moved to config, `canonical_state` fields, the trace lock-in row and docstrings. None
  writes conflict, over/underfunctioning, distance, cutoff, projection or fixed triangles.
- **L1 · premise passes:** `phase_c_ensemble_record.md` classes every verdict and warns (:5) that a premise passing
  confirms the code; the Phase D records carry class columns.
- **L1 · real-family framing (M11.F.9/10):** all five `docs/review/` files carry the framing; the trace header says it
  "does not concern any real family or person"; README :92 says nothing produced is a finding about families; real-
  family import is deferred to version 3 (`version_2_plan.md:374–381`).
- **L1 · `[K-ext]` attribution:** the explainer's only K-ext citation (:418, KS23.11) is labelled and not attributed
  to Bowen; spec M11.C.6(b) and M11.C.21 are not attributed to Bowen.
- **L1 · `docs/readouts_phase_c.md`:** read in full; patterns are graded `[I]` readouts; W=4, N=3, fixed=3, hold=12
  match `constants.md:240–243`.
- **L1 · M11.5 and M11.G against code:** classes present; components 9 and 10 left unbuilt as M11.G.3 requires.
- **L1 · settled item M10.C.2 / L16.3 / DECISIONS A1:** the withdrawal is consistent between the spec, the ledger and
  DECISIONS. The two problems left are MR-2 (explainer §8) and SP-1 (the quotation form).
- **L2 · M8.5 revision 33:** presence decides activity (`recompute.py:133`, `Occupant(m, present=…alive)`); no fusion
  argument at `act.py:151, 171`; the coach's side is read only at `external.py:103–104`. Spec matches code.
- **L2 · M13.2 revision 33:** `tests/test_repo_layout.py::test_s1_6_the_grid_engine_is_retired` exists (:39); the
  tag's `tests/test_simulator.py` held 37 tests, as M13.2 says.
- **L2 · S1.3 act classes:** `config/bowen/act_classes.md` matches M5.B.4/M5.B.5 apart from the unbuilt M1.E.3/M1.E.4
  acts (already recorded as M5.B.5 partial); `load_act_classes` (`io/load.py:46–72`) validates the lists.
- **L2 · S1.7a lock-in trace rows:** `render/trace.py`'s rule (show change ≤ 0) agrees with `symptoms.py:155–206`
  (hold positive and hidden, release negative, spill 0.0).
- **L2 · S1.2 `canonical_state`:** covers every `RunState` field except `NOT_CANONICAL = {"kinds", "draws"}`.
- **L2 · `draws.categorical` top edge:** still one keyed uniform by inverse transform, consistent with M3.D.4a (:633).
- **L2 · two-arm form:** no new or changed acceptance test asserts an absolute threshold for an M11.C criterion.
- **L2 · M11.1 for the M11.C criteria:** every M11.C row marked done (C.1, 3, 4, 11, 15, 17, 19, 22, 25, 32, 35, 37,
  38, 39) has a mutant that went red or reversed in a committed record; criteria with surviving or unrun mutants are
  not marked done.
- **L2 (orchestrator) · M14.A register (spec :1667, :1697, :1698):** consistent with the code; `recompute_triangles`
  calls `positions_live` on presence and `visibility` calls it. Line 1698 omits the third caller, `external.coach_live`
  (folded into MR-12).
- **L2 (orchestrator) · §0.3 MUST/test counts (spec :38):** guarded by
  `tests/test_spec_consistency.py::test_m11d14_section_0_3_counts_are_current`, which passes.
- **L2 (orchestrator) · M11.C.6 (spec :1318):** "Claude's recommendation, pending owner review" is followed by the
  owner's acceptance on 2026-10-18; consistent.
- **L3 · spec revision restatements:** CLAUDE.md, README, `_STATUS.md`'s STATE line and the checklist all say 33.
- **L3 · document sizes in the reading list:** the explainer (2,094), review guide (1,371), spec (2,585), ledger (2,908)
  and TODO (1,044) match their "about" figures.
- **L3 · "575 requirements":** `spec_coverage.md:23` agrees.
- **L3 · file I/O statement (reading list §3):** I/O calls appear only in `io/load.py`, `io/sinks.py`, `run.py:72–73`
  and `ensemble/criteria.py:56, 142`; `src/bowen/engine` imports nothing from `bowen.io`.
- **L3 · paths named as existing:** all exist, including `test_engine_purity.py`, `test_determinism.py`,
  `test_repo_layout.py`, `test_doc_checklist.py`, `papers/INDEX.md`, `papers/DIGEST.md` and `data/`.
- **L3 · CI claims:** `tests.yml` runs 3.11, 3.12 and 3.13, installs only `requirements.txt`, uses `fetch-depth: 0`.
- **L3 · staleness tests for trace, chart and diagram:** `test_render.py:248` and `test_famdiag_export.py:304` exist.
- **L3 · proposal and plan statuses restated in the reading list and checklist:** each matches its document's own
  status line (G1, Holmes–Rahe, learning by level, the version 2 plans, `version_1_*`, `plan_doc_claim_tests`,
  `plan_spec_coverage_phase_d`, the brief).
- **L3 · the reference family:** "eleven people plus an outside coach" matches `family_phase_d.md:4–8`.
- **L3 · "no verdict changed" at stage 1:** all seven compared records show zero changed verdicts.
- **L3 · CLAUDE.md :77 suite-count history:** each step adds up (928+60=988, +9=997, +5=1002, +1=1003; 921−11+3=913),
  and the orchestrator's run measured 1003.
- **L3 · retired paths:** README and CLAUDE.md mark each, matching `doc_claims.json`'s `retired_paths`.
- **L4 · engine purity (M16.B.1, M11.D.1):** no changed engine line adds an I/O import or call; the only entries
  removed from `FORBIDDEN_IMPORT_PREFIXES` are the retired grid-engine prefixes, and the scanner's self-test still
  plants a bad module.
- **L4 · `render/` and `io/` against M16.B:** only `src/bowen/run.py` imports `render`
  (`test_s1_8b_only_run_imports_render`), so leaving `render/` out of the code hash cannot hide a verdict change.
- **L4 · no builtin `sum` in `src/bowen`:** `test_float_sums.py` scans for it and has a planted self-test.
- **L4 · the hash tests can fail:** `test_s1_8a_render_is_not_hashed` and
  `test_m134_code_hash_ignores_prose_but_not_code` each make an edit that must change the hash.
- **L4 · Phase D record test:** compares the full render; `test_s1_8c_a_stale_entry_is_rerun` covers a stale entry;
  cache writes are atomic and `.gitignore:24` ignores the `.partial` files.
- **L4 · skips:** both carry reasons (`tests/test_python_versions.py:55`, `test_render.py:350`); `pytest.ini` prints them.
- **L4 · CI workflow:** runs the suite only (`python -m pytest tests/ -q`), so no check lives only in YAML.
- **L4 · new numeric literals:** none in the changed `src/bowen` lines. Pre-existing act-name sets in code
  (`iposition.py:89`, `rejection.py:37`, `criteria.py:165, 712`, `policy.py` `NEEDS_TRIAD`, `sinks.py`, `moves.py`) are
  already recorded in `docs/implementation_plan_version_2.md:450–451` and `TODO.md:227`.

## Unverified

- **L1:** the 13 unread `config/bowen/` files (`stress_response.md`, `policy.md`, `family_*.md` and others) were not
  checked for notes that claim a source sets a value; the unread explainer ranges were grepped for `KB`/`Tape` only, so
  a quote without a prefix would be missed; `constants.md:56` quotes one KB word ("grotesque"), attributed to the
  interviews, which L1 judged borderline and did not raise.
- **L2:** whether each "done" override beyond lines 1–120 of `tools/spec_coverage_overrides.json` names a test that
  exists; whether a config entry declares M4.D.1f's tie-break rule.
- **L3:** the reading list's :87 claim that every function satisfying a requirement has a docstring (276 `Spec:` lines
  in 64 files; not compared function by function); whether `spec_coverage.md`'s M1.A.7 note records the conflict
  TODO.md names.
- **L4:** the wall time of `tests/test_version_1_status.py:56–70` (a `git archive` of the tag and a pytest subprocess on
  every default run), of `test_spec_coverage.py`'s `--collect-only` subprocess, and of MR-26's extra run (settle with
  `pytest --durations=20`); `render/trace.py:249` would raise `ValueError` rather than `UnrenderableRecord` if a
  lock-in record ever carried two people entries (all three producers emit one); `tar.extractall(..., filter="data")`
  needs Python 3.11.4 or later locally; `requirements.txt` leaves numpy and ruff unpinned.
- **Challenge pass:** it had no git access, so it could not order events within 2026-08-22 (SP-1 dates).
- **Revision history:** the orchestrator read the spec's revision history (2122–2585) by heading only.

## Questions only the owner can answer

1. **KB11 in M10.C.2, `_LEDGER.md` L16.3 and DECISIONS A1:** is the verbatim quote an accepted exception (the brief
   asks this at :169–171), or part of SP-1? Did approving A1 approve the quotation's form, or only the withdrawal?
2. **Closed records:** should the approved DECISIONS records and the ledger's quotations be rewritten as paraphrase,
   or kept as frozen history with a note?
3. **M11.F.3:** should it be restated from the written FE11.2 Epilogue instead of KB13 (MR-1)?
4. **M11.1's scope:** does it apply to every requirement, or only to the `M11.C` criteria (MR-6)?
5. **M5.C:** is the avoidance gate (`avoidance_available`) to be built, or should M8 and M13.1 drop it (MR-12)?
6. **M4.D.1f:** done or partial while `life_energy` (M1.A.10) is unbuilt (MR-11)?
7. **Fusion in runs:** revision 33 records that no caller passes a fusion map, so the fused-occupant branches of
   `positions_live` never run. The spec leaves "whether a fused occupant should make its triangle inactive" as a
   theory question with a case each way. Is that to be decided at stage 3?
8. **The coach and SPLIT:** is the coach always going along with a SPLIT (`external.coach_side`; M11.C.7 is classed
   Mixed) the intended choice?
9. **The Phase C ensemble record:** should it move onto `docs/records_cache` as Phase D's did (MR-23)?
   `docs/version_2_plan.md:129–131` reads as if it already had.
10. **The tag-copy status test:** run on every default run, or only when the tag commit changes?
11. **`tools/version_1_status.py`:** is `spec_coverage` meant to depend on this frozen file for good (MR-24)?
12. **`REVIEW_version_1_guide.md`:** still awaiting your review, or replaced by version 2 (MR-21)?
13. **This brief:** `milestone_review_brief.md` is still DRAFT; CLAUDE.md treats it as the procedure. Approve it after
    this first run?
14. **Holmes–Rahe Q3:** it was decided on a misread case and later corrected, with a note that the owner "may revisit
    it". Ask again before group G is built?
15. **`_STATUS.md`:** its run log ends at 2026-10-19 with no "Next" for version 2. Should it record 2026-10-20/21 and
    the next stage?
