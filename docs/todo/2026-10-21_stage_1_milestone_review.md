# Version 2 stage 1 milestone review — open after the triage, 2026-10-21

The review is `docs/cycles/2026-10-21_milestone-review_v2-stage-1.md` (the second Mac, branch
`review/2026-10-21-v2-stage-1`). The triage refuted none of its 26 findings: 22 hold as stated and 4 in part. The
owner's ten decisions and what this batch fixed are in `CHANGELOG.md`. Open:

- [ ] T-008 ⟨later⟩ **SP-1, the parts the owner deferred on 2026-10-22** (the build was spending more on the transcripts than on
  the model; the proposal's §7 and §9):
  - SP1.3: the local test, marked `transcripts` and deselected by default, that matches quoted spans against the
    transcript folder named by `EPMODEL_TRANSCRIPTS`, with its line in `docs/milestone_review_brief.md`.
  - SP1.6: the header note on each extraction note in the proposal's §3 (their quotations only locate passages), and
    the six headings in `kb02.md`–`kb15.md` that name Bowen as the speaker of a Hijack-set interview.
  - This file's two ASR quotations: the backlog item on hedges trimmed off the front of quotations (2026-08-28)
    quotes spec phrases from the tapes and interviews, which revision 34 paraphrased, so that item can close or
    narrow; the other is in the item on locating the rest of the interview series, and is on
    `tests/asr_quotations.json`'s allowlist until it is paraphrased. The default-suite test does not catch the first.
  - The test's known limits (its docstring; the proposal's §9): it misses a quotation whose citation sits in another
    paragraph, list item or table row, or that has none; one set as a block quote without quote marks; one whose
    marks do not pair (a curly opening closed by a straight one); one after a stray quote mark in its own block; one
    longer than 800 characters; one in straight single quotes or guillemets; one beside a citation form not in the
    rules (`KB-04`, `T6.8`, "tape five"); an HTML entity in a Markdown file is not decoded; only tracked `.md` and
    `.html` files are read. The local test (SP1.3) is the check for most of those.
- [ ] T-009 ⟨later⟩ **SP-1 Q7: six requirements whose passage supports less than they say** (owner, 2026-10-22: each its own
  decision; the paraphrases keep to the passage and the requirement's MUST is unchanged at revision 34; the
  proposal's Q7 table):
  - `M11.C.11` (SP1-S40): Tape 6 ranks a long absentee's return as probably the strongest of the additions, and says a
    family generally reacts more to a loss; it does not by itself support the return exceeding the removal.
    `_LECTURES_1979.md` T6.8 makes the same step. Revision 34's paraphrase now says so inside the criterion's own row.
  - `M1.C.3a` (SP1-S16): Tape 5 speaks of the intensity of symptoms and relationship patterns, not of triangles.
    Decide with the next item's `SP1-E06`, which cites the same passage for triangles.
  - `M1.E.7` (SP1-S24): the KB10 passage is about a society accepting what it can no longer deal with; its use as the
    condition for a rise in `systems_perspective` is the spec's analogy.
  - `M5.B.3a` (SP1-S32): KB08's statement on de-selfing is limited to a family where everyone does it, and the same
    interview recommends near-full disclosure between spouses as an exercise.
  - `M5.C.1a` (SP1-S34): KB08's warning was given to one family before it entered the research, not as a general
    warning against the protocol.
  - `M1.B.10a` (SP1-S15): Tape 6 states a closed subject's reopening as a wish, and says couples usually cannot
    reopen it themselves by the time they seek help; it supports the MUST as a possibility, not a common outcome.
- [ ] T-010 ⟨later⟩ **Found while building SP-1 (2026-10-22), not fixed:**
  - `SP1-E06`, the explainer's §6.1 triangle table (`routing_capacity`): the proposal's check note says the text now
    says that applying Tape 5 to triangles is the project's inference (K04.15); the approved text does not say it.
    Pair with `M1.C.3a` above.
  - `SP1-S38` (`M10.A.2`) cites K10.9 twice, once inline and once in its source arrow.
  - `SP1-S35` (`M5.D.7a`) dropped the old note's explanation of why the attribution changed.
  - `FE03.21`: the block quotation in `M4.D` ends a quoted run in the chapter whose speaker is not known; it is cited
    as Kerr's chapter until it is checked against the printed book (`family_evaluation/fe03.md`).
- [ ] T-011 ⟨later⟩ **From the SP-1 build's review (2026-10-22), not fixed:**
  - Six exempt paths in `tests/asr_quotations.json` hide no finding today (`CHANGELOG.md`, `LEARNINGS.md`, `papers/`,
    `_KERR_INTERVIEWS.md`, `_PASS2_CORRECTIONS.md`, `_SOURCE_QUALITY.md`). Kept: each is exempt by the proposal's §3
    for what it is (a dated record, the owner's files, extraction notes), so a quotation added there later should
    not fail. `test_sp1_4` checks that each path exists, not that it hides anything.
  - `test_sp1_4` does not catch two allowlist entries for the same file and span.
  - The test scans tracked files only, so a new document fails it only once it is added to git. Scanning untracked
    files would scan the owner's untracked files too.
  - `src/bowen/engine/live_positions.py:1` says the predicate is used in four places; revision 34 says three callers
    (M8's preamble). The file is in the engine hash, so the fix waits for the next change that moves the engine hash
    (re-pointed 2026-10-24, stage 2 review MR-22: group G's regeneration ran without it). The spec writes the rule
    for four places and the fourth, `avoidance_available`, is not called, so the wording follows T-014's
    avoidance-gate answer.
  - **Owner decision:** `docs/theory/family_evaluation/` and `docs/theory/kerr_book/` are exempt from the test as
    written-source notes, but four of their cross-references quote an ASR source: `ks07.md:159` (KB07 · K07.4),
    `ks19.md:148` (KB04), `fe10.md:303` (KB06) and `fe07.md:38` (Tape 2). Paraphrase them, or hold these notes like
    the `kb` notes under D1 (their quotations only locate passages), or list their spans and drop the folder
    exemptions so a fifth is caught. The exemption reasons now say so.
  - `tools/famdiag_export.py` keeps the app keys it names (positions and sizes, colours, settings). Any other key the
    app writes when the owner edits a field the tool does not draw (a birth date, say) is dropped, with a report
    line. The tool writes in the app's format, so a key name cannot tell an owner's edit from a value an earlier
    version of the tool wrote; keeping every unknown key would keep stale run output. Owner edits to fields the tool
    draws (a name, a line's notes) are redrawn by design. The file is written in place, not atomically; it is in git.
- [ ] T-012 ⟨later⟩ **With the next change that moves the engine hash.** Each of these edits moves a record hash, so it waits for a change that
  reruns the records anyway (owner, D6 and D7, 2026-10-21; until 2026-10-24 the trigger was group G's regeneration, which ran on
  2026-10-23 without it, stage 2 review MR-22):
  - MR-23: move the Phase C ensemble record onto `docs/records_cache`, as Phase D's records are: render it from the
    cache, take the tool out of its own hash, and compare the full render in `tests/bowen/test_ensemble_record.py`.
  - ~~`config/bowen/family_phase_d.md:19`: drop "awaiting owner review" from Teodor's 36.~~ *Done at revision 36
    (2026-10-22), with group G's regeneration. MR-23 remains.*
- [x] T-013 ⟨now⟩ **Before group G's build starts:** ask the owner the Holmes–Rahe Q3 again (D10, the review's Q14; see
  "Blocking the next phase"). *Closed 2026-10-22: group G was built overnight under the owner's instruction of
  2026-10-22 ("proceed based on your recommendations"); Q3 stands as decided, cites no source, and is listed in the
  revision 36 report's decisions, with the correction quoted, for the owner's review.*
- [ ] T-014 ⟨later⟩ **For the stage 3 question list** (owner, D4 and D10, 2026-10-21):
  - Build the `M5.C` avoidance gate (`avoidance_available`), or drop it from M8 and `M13.1`.
  - Q7, fusion: no caller passes a fusion map, so `positions_live`'s fused-occupant branches never run. The spec
    leaves open whether a fused occupant makes its triangle inactive.
  - Q8, the coach and SPLIT: `external.coach_side` has the coach always go along with a SPLIT (`M11.C.7`, classed
    Mixed). Is that intended?
  - `M7.D.1a` (stage 2 review Q5, owner 2026-10-24): every illness onset takes the "personal injury or illness"
    size on all three channels. Should a social or mental onset carry it?
  - *Decided by the owner 2026-10-24, each as recommended, in stage 3's plan (`docs/implementation_plan_version_2_stage_3.md`
    §3.5, D27–D30): the avoidance gate dropped; fusion left unbuilt, `M8.1` partial; the coach's going along with a
    SPLIT declared a premise of `M11.C.7`'s arm; the illness size kept on all three channels. Open until built.*
- [ ] T-015 ⟨later⟩ **SP-2: extend the document-claim tests** (2026-10-21) to the places where this review found stale prose: closing clauses
  of status lines, the checklist's `- [ ]` items, the reading list's body, lists of standing decisions, counts in
  prose, and the status notes in the reference-family table. Stage 2's review (SP-A, 2026-10-24) adds:
  `tests/test_doc_status.py` classifies a status line by its first word only, so a status that moves within the
  settled class goes unchecked — a later revision (MR-5, MR-16), a later part built (MR-6), a review that closed
  (MR-15, MR-17). Compare the dates and part lists as well. A front-matter `date:` older than the newest date in the
  body (MR-19) needs a rule first: the field means creation in some documents and last edit in others.
- [ ] T-016 ⟨later⟩ **Adjacent, not fixed** (found by the batch's learning-qa pass): `docs/theory/_LECTURES_1979.md:46` still says
  twelve of twenty-three items were probed and the rest are outstanding. Its own "Revision 1" section and
  `_STATUS.md`'s file map say all 23 were probed.
