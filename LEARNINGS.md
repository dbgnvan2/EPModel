# EPModel — learnings

This file records how EPModel's reviews have failed: what got past a review and was found later. The failure-pattern
catalogue (P1–P39) and its review checklist are global, in `~/.claude/standards/learnings.md`. They are not copied
here, so the two lists cannot drift apart. Repo rules are in `CLAUDE.md`. Review findings that are not fixed go in
`TODO.md`.

## Open risks

Findings from a review that have caused no failure yet are tracked in `TODO.md`. The 2026-10-20 cold sweep's items
under Misses are fixed or in `TODO.md`; the entry's outcome says which.

## Misses

Each entry gives the batch, how the miss was found, what the earlier passes missed (with the pattern), and why they
missed it: the pattern was absent from the catalogue, the pattern was present but did not fire, or the code was
outside the reviewed range.

### 2026-10-21 · version 2 stage 1 (`version-1...9cf8552`) · first milestone review

- **Why it ran.** The first milestone review (`docs/milestone_review_brief.md`), at the end of stage 1, from the tag
  `version-1`. The second Mac ran it with none of the build history: four lens workers and a challenge pass. The
  report is `docs/cycles/2026-10-21_milestone-review_v2-stage-1.md`, on branch `review/2026-10-21-v2-stage-1`
  until it is merged.
- **What it found.** 26 findings: 0 blocker, 4 major, 22 minor, in four themes. The triage refuted none: 22 hold as
  stated and 4 in part. Most were already present at the tag, so the passes that missed them are version 1's, not
  stage 1's.
- **The main miss (SP-1: three of the four majors, MR-1, MR-3 and MR-4, and two minors).** The 1979 tapes and the
  Kerr–Bowen interviews are speech recognition, and the rule for them has two parts: cite the source, and do not quote
  it. The triage's scan found 103 verbatim quotations in the spec, the explainer, `_LEDGER.md` and `DECISIONS FOR
  APPROVAL.md`, 34 of them naming Bowen or Kerr, or "he", as the speaker. None was written before the rule: it was
  committed for the tapes on 2026-08-23 at 11:50 (`528bb50`) and for the interviews at 13:09 the same day (`e14d3d7`).
- **Why it was missed.** For the 2026-08-28 sweep, the pattern was absent from the catalogue: P33, an audit exhaustive
  along one axis, was added on 2026-08-30. Quotations went on being written until version 1 ended, so for every pass
  after that date the pattern was present and did not fire.
  - The corpus-fidelity sweep of 2026-08-28 checked every quotation in the spec and counted 160 as "verbatim in the
    corpus". For a written source an exact match is the pass. For an ASR source it is the breach. The sweep checked
    the first part of the rule and reported the whole as checked.
  - The `kb/*.md` extraction notes say "no verbatim quotation" in their headers and quote in their bodies. Every later
    document copied from them, so the breach spread from the notes.
  - No test reads the rule. It lives in `_SOURCE_QUALITY.md` and in the notes' headers.
- **The smaller misses.**
  - SP-2: stale prose in places the document-claim tests do not read, such as closing clauses of status lines,
    counts in prose and the reading list's body (P6).
  - SP-3: "done" claims more than is proved. The coverage headline did not say which rows a mutation proves, and the
    spec says four mechanisms call the live-positions predicate where three do: the fourth site, `avoidance_available`,
    is defined and tested but never called (P21).
- **Rule.** When a source rule has more than one part, write a check for each part. A fidelity check for an ASR
  source must flag an exact match, not pass it.
- **Cost.** The review: 462,483 subagent tokens and 13 to 14 minutes (the report says 13; its clock reads 11:54 to 12:08), plus writing; the
  orchestrator's own tokens were not measured. The triage: three agents checked every finding, about 534k tokens and
  12 minutes. This batch's learning-qa and correctness passes: about 320k tokens; its narrow re-sweep, about 124k.
- **Outcome (2026-10-21).** The owner decided ten questions. This batch fixed the wording findings outside the spec,
  the explainer's two items (MR-2, MR-7), the coverage headline (MR-6) and two engineering items (MR-24, MR-25, the
  test shown failing under an `or ()` default at each of its three reads). SP-1 goes to a proposal before any spec edit. The
  spec's own wording fixes wait for revision 34 so the spec changes once. `TODO.md` lists the rest.

### 2026-10-20 · version 2 stage 1 (`579170c...3d70d3e`) · cold sweep

- **Why it ran.** `/csdp` calibration: the batch needed four fix rounds. These were the step 6 review, its fix
  re-sweep, step 8's three narrow passes (learning-qa, correctness, security), and a final re-sweep.
- **The cold pass.** One `learning-qa` agent in a fresh context reviewed the whole range. It did not see the
  generated records or the three files that hold the earlier findings (`docs/implementation_plan_version_2.md`,
  `TODO.md`, `CHANGELOG.md`). It found 9 items: 0 high, 2 medium, 7 low or PLAUSIBLE. After the pass they were
  compared with the plan's §1.10. The full report is in the session notes as `cold_sweep_stage_1_report.md`.
- **Already listed by the warm passes (4).** The diagnostic probe stops on a failed child (§1.10, adjacent issues).
  The changes page is not frozen (`TODO.md`). `tools/probes.py` uses the builtin `sum` (adjacent issues). The
  act-literal scan cannot see name constants (a documented limit).
- **Missed by the warm passes (5, and part of a sixth):**
  - *Medium, P24/P6:* `tools/record_cache.py` `exit_status` counts only FAILED rows. A RAISED or BROKEN row is
    written into the record but never cached, and the tool exits 0. The record-current test then fails with
    "rerun the tool", and a RAISED row that comes back on every run can never clear it. `exit_status` is new in this
    range (`423bbed`). Never caching RAISED rows is older.
  - *Low, P3/P33:* `tools/record_changes.py` puts `docs/phase_b_mutation_record.md` and `docs/version_1_status.md` in
    neither its compared list nor its not-compared list. New in this range.
  - *PLAUSIBLE, low, P38:* `tools/ensemble_record.py` hashes `config/bowen/*.md` as raw bytes, so rewording the prose
    in the new `act_classes.md` reruns every record. The raw hashing is older than this range.
  - *Low, P1:* `tools/record_cache.py` matches machine errors by exact class name, so an OSError subclass in the
    child is a red (RAISED), not FAILED. Older than this range.
  - *PLAUSIBLE, low, P5:* the in-process pools of `tools/phase_d_record.py` and `tools/ensemble_record.py` get no
    process-group kill on a stop signal. §1.10 lists only their missing timeout. Older than this range.
  - *Part of a sixth, P2/P19:* `tools/level_occupancy.py:130` never checks that every expected row came back.
    §1.10 lists only that it stops when its child fails.
- **Why they were missed.** In every case the pattern was in the catalogue and did not fire. The medium item is in
  code this batch wrote, and the full pass and the later narrow passes all read it. Those passes followed the new
  FAILED path and its test. They did not ask what the function reports for the other two outcomes in `ERRORED`.
  Three of the low items are older code that the warm passes read but treated as outside "what this batch changed".
- **Rule.** When a status or exit function is written for one member of a set of outcomes, review and test it
  against every member of the set.
- **What the cold pass confirmed.** It recomputed every record against the tag `version-1`: 0 verdicts changed, 0
  effects moved, 0 fields differ. This agrees with the warm passes.
- **What the cold pass did not cover.** It ran only the learning-qa lens; the correctness and security lenses were
  not rerun. It judged tests by reading them, not by reverting each fix. It ran 79 targeted tests, not the full suite.
- **Cost.** One agent, about 14 minutes, about 64k tokens.
- **Outcome (2026-10-20, the session that owns the batch).** Fixed, each test shown failing under its mutation: the
  medium item (a RAISED or BROKEN row now makes the tool exit 1); the two pages missing from `tools/record_changes.py`'s
  lists (now listed, and a test fails when a generated page is in neither list); the level-occupancy row check (the
  record was rerun, and only its hash line changed). Recorded in `TODO.md`: the raw-byte config hash, the pools not
  killed as a group, and the exact-name match of machine errors (that one had been in `TODO.md` since 2026-10-09),
  plus an owner question: a row that raises on every run is never cached. One narrow `learning-qa` re-sweep on
  sonnet over the fixes found two high items, the coverage page and the test count left stale (the suite was red),
  and four low ones, among them two Phase B traces also missing from the lists. All were fixed. Plan §1.10 has the
  detail.

## Fix log

Entries are newest first, in the form: Issue → Root cause (Pn) → What would have caught it → Fix → Rule.
There are no entries yet.
