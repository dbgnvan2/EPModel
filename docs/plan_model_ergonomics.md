# Plan — making the spec, ledger and code cheaper for a model to read

**Status: Parts A to E built 2026-10-22 on the branch `model-ergonomics` (the CHANGELOG entries of that date), the
owner's §8 answers of 2026-10-22 applied (Q1 thin them; Q2 yes, as revision 35; Q3 delete all four; Q4 fix now). Part E's
second step, Claude's dispositions in `spec_revision_35_dispositions.md`, is for the owner's review. §11's sonnet
trial has not been run.** Written in a Cowork session from a read-only scan of the repo at `fc9939e`, then moved
here from the repo root (A1).

## 1. Why

A session that changes one function today has to search or load a 504 KB spec (about 125k tokens), whose
requirements carry their history inline, and then read docstrings that restate the same rules. Measured at `fc9939e`:

| Item | Size | Note |
|---|---|---|
| `docs/bowen_agent_model_spec_v2.md` | 504 KB, 2,585 lines, 575 IDs | 520 inline `⟦rev…⟧` tags; front-matter `status:` is one 5 KB line; revision history from `### Revision 12` to the end is 45 KB |
| `docs/theory/_LEDGER.md` | 211 KB | 8 entries `[withdrawn]`, others merged, kept in place |
| `docs/model_explainer.md` | 217 KB | |
| `src/` | 743 KB | **214 KB (29%) is docstrings**; 321 of 698 defs cite IDs |
| `docs/spec_coverage.md` | 208 KB | ID → test → status, as markdown prose |
| ledger ↔ spec links | 166 spec IDs cited in the ledger; 256 source IDs cited in the spec | in prose only |

The goal is to cut what a session must read to answer three questions: *what does ID X require now, what implements
and tests it, what does the source say behind it.* The rule throughout: **the spec, the ledger and the explainer stay
the source of truth; everything new here is generated from them and checked by a currency test**, in the same pattern as
`docs/spec_coverage.md` and `tests/bowen/test_spec_coverage.py`. No ID is renumbered (spec §0.4).

## 2. What must not change

- What a run computes. `tools/ensemble_record.py::code_hash` drops docstrings and comments, so Parts A–D must leave
  `engine_hash()` (`tools/record_cache.py`) **identical**. Record it before Part A and check it after every part; a
  change means a code edit slipped in. No record is rerun in this plan.
- The suite: all tests pass after every commit (`python3 -m pytest tests/`).
- CLAUDE.md's same-commit rule, the checklist rule (`tests/test_doc_checklist.py`) and the reading-list rule.

## 3. Part A — housekeeping (no owner decision needed except A4)

- **A1.** Move this file to `docs/plan_model_ergonomics.md` and link it in `docs/SPEC_REVIEW_CHECKLIST.md` in the same
  commit.
- **A2.** Condense CLAUDE.md's "A red suite is never normal" bullet to the current count, the date and the command. Move
  the dated history of counts to CHANGELOG.md under today's entry. CLAUDE.md is loaded by every session.
- **A3.** Leftover worktrees: `git worktree list` shows three marked prunable under `.claude/worktrees/` (44 MB of
  duplicate `src/`, `docs/`, `tests/`). For each branch (`claude/cranky-snyder-764f33`, `claude/quizzical-burnell-818aa7`,
  `docs-stale-fixes`), run `git log main..<branch>`. If empty, remove the worktree. If not empty, **stop and list the
  unmerged commits for the owner**; do not delete them.
- **A4.** `docs/model_explainer copy.textClipping` looks like a stray macOS clipping. Ask the owner before deleting;
  update the checklist if it goes.

## 4. Part B — a machine-readable index (highest payoff)

**B1.** New tool `tools/spec_index.py` writing `docs/spec_index.jsonl`: **one JSON object per line, one line per spec
ID**, in document order, so a single `grep '"M4.C.3a"'` returns the whole record. Reuse `tools/spec_coverage.py`'s
functions (`spec_ids`, `named_tests`, `tests_for`, `coverage`) rather than reparsing. Fields:

| Field | Source | Rule |
|---|---|---|
| `id` | spec | as defined in §0.4 |
| `module`, `heading` | spec | the enclosing `##`/`###` heading |
| `lines` | spec | `[start, end]` of the requirement's block (from its bold ID to the next ID or heading) |
| `first_sentence` | spec | the requirement's first sentence, **generated**, truncated at 300 chars. No hand-written summaries — they would drift |
| `withdrawn` | spec | true if the block says `WITHDRAWN` |
| `grade_tags` | spec | every grade tag (`[T]`, `[M]`, `[D]`, `[#]`, `[I]`, `[X]`) appearing in the block, as a list. Do **not** infer a single grade |
| `sources` | spec | source IDs cited in the block (`L01.1`, `KS23.3`, `K12.6`, `FE03.6`, …; take the patterns from the spec's citation legend) |
| `ledger_mentions` | ledger | ledger entry IDs whose text cites this spec ID |
| `code` | `src/` | `path::qualname` of every module, class or function whose docstring `Spec:` line names this ID (parse with `ast`) |
| `tests` | tests | from `spec_coverage.py` |
| `status`, `status_note` | coverage | exactly what `spec_coverage.py`'s `coverage()` returns |

**B2.** Test `tests/bowen/test_spec_index.py`: (a) the file is current (regenerate in memory and compare, as
`test_m141_spec_coverage_is_current` does); (b) every spec ID appears once; (c) every ID named in a `Spec:` line in
`src/` exists in the spec.

**B3.** Run (c) once before writing the test. If it finds IDs cited in code that the spec does not define, **do not fix
them in this batch**: list them in `TODO.md` and report them, and mark (c) `xfail` with the list until the owner decides.

**B4.** Ledger reverse view: also write `docs/theory/ledger_index.jsonl`, one line per ledger entry: `id`, `title`,
`status` (`stands`, `narrowed`, `withdrawn`, `merged`, `new at p2`, as written in the heading), `chapter`, `lines`,
`spec_ids` cited. A test checks it is current. Withdrawn and merged entries stay in the index with their status, so a
search can filter them.

## 5. Part C — a "current rules" view of the spec

**C1.** New tool `tools/spec_rules_view.py` writing `docs/spec_rules/M01.md` … `M17.md` (one per `## M` module, plus §0
as `M00.md`) and `docs/spec_rules/README.md`. Each file opens with: *Generated from the spec at revision N by
`tools/spec_rules_view.py`. Not normative: where this and the spec differ, the spec wins. Do not edit.*

**C2.** What the view removes — **mechanical removals only**:
- the front matter's `status:` line (keep `version:`);
- every `⟦…⟧` tag;
- block-quote lines (`>`), i.e. the source quotations; the `→ file · ID` pointers stay;
- the revision-history sections from `### Revision 12` to the end of the file.

**C3.** What it must **not** remove: any sentence in a requirement body, including those that start
`*Revision N:*` or "re-derived at revision N". Some of those carry the current rule (for example `M4.C.3a`'s
"*Revision 24:* the floor is the person's usual level"). Do not try to strip them automatically.

**C4.** Instead, the tool writes `docs/spec_rules/history_in_body.md`: every ID whose body contains an inline revision
narrative (`Revision \d+`, `re-derived at`, `Earlier revisions`, `previously`, `was changed`), with the matching
sentence. That list is the input to Part E's owner decision.

**C5.** Tests: the view is current; every non-withdrawn spec ID appears in exactly one view file; the view's word count is
reported (not asserted). In the completion note, state the measured size of `docs/spec_rules/` against the spec.

## 6. Part D — thin the code docstrings (owner decision first, §8 Q1)

**D1.** Keep in every docstring: the `Purpose:` / `Spec:` / `Tests:` block (spec §0.4), the formula the code implements,
and every grade statement M0.2 needs (an `[I]` form must never read as sourced).

**D2.** Remove: revision and approval narration ("*Revision 25 (…approved 2026-10-16)*", "*Changed at step 3*"),
and rationale or quotations that restate the spec. Before removing a passage, confirm the spec or CHANGELOG already
holds it; if neither does, move it to CHANGELOG rather than lose it.

**D3.** One package per commit (`engine/`, `policy/`, `scenario/`, `ensemble/`, `readouts/`, `io/`). After each:
suite green, `engine_hash()` unchanged, docstring bytes before/after recorded in the commit message.

**D4.** If a long `Spec:` line (for example `appraise.py`'s ~20 IDs) suggests a module does too much, **record it in
TODO.md**; do not split modules in this plan.

## 7. Part E — move history out of the spec (owner decision first, §8 Q2)

**E1.** New `docs/spec_revision_log.md`. Move into it: the front-matter `status:` text (leave a short `status:` such as
`APPROVED; revision history in spec_revision_log.md`) and the sections `### Revision 12` to the end. Keep the
front-matter `version:` field exactly as `tests/test_doc_claims.py` parses it (`spec_revision`).

**E2.** Inline narratives from C4's list are edited **one at a time with the owner**, as with revisions 24–32: each one
either stays, moves to the revision log, or is rewritten as the current rule only.

**E3.** Ask whether this needs a spec revision number. It changes no rule, but it edits the approved contract.

## 8. Owner decisions needed

- **Q1 (Part D).** May docstrings be thinned to D1's content, with history moved to CHANGELOG?
- **Q2 (Part E).** May the front-matter status and the revision sections move out of the spec into
  `docs/spec_revision_log.md`? Does that take a revision number?
- **Q3 (A3/A4).** Delete the leftover worktrees whose branches are fully merged, and the `.textClipping` file?
- **Q4 (B3).** If code cites IDs the spec lacks, fix in a later batch or now?

## 9. Documents to update (same commit as the change they describe)

`docs/SPEC_REVIEW_CHECKLIST.md` (every new file under `docs/`); `CLAUDE.md` — add a "Read first" line to "Where the
theory lives": *to look up an ID, grep `docs/spec_index.jsonl`; to read current rules, open `docs/spec_rules/M<n>.md`;
open the full spec only to change it or to see sources and history*; `README.md` "Where to start";
`docs/REVIEWER_READING_LIST.md` if it lists how to read the spec; `docs/milestone_review_brief.md` §3 (tell a review
session to start from the index and the views, and not to read TODO.md or CHANGELOG.md in full); `CHANGELOG.md`.

## 10. Order, review and cost

A → B → C, each its own commit or short series, on a session branch merged into `main`. Then stop for §8. Then D, then E.
Review per CLAUDE.md: one `learning-qa` pass and one correctness pass per batch before push; a re-sweep of a fix on
`sonnet`. No ensemble, mutation or sweep record is rerun; if any record test goes red, stop — something computed changed.

## 11. Done when

- `docs/spec_index.jsonl`, `docs/theory/ledger_index.jsonl` and `docs/spec_rules/` exist, are generated, and are tested
  current.
- `engine_hash()` equals the value recorded before Part A.
- The completion note states: the size of each new file; the spec-to-view size ratio; docstring bytes before and after
  (if D ran); the IDs found by B3; and which documents were updated.
- **Check that it helps:** in a fresh session on `sonnet`, ask three lookups (for example: what `M4.C.3a` requires now
  and which function implements it; which tests prove `M5.D.1`; which ledger entries stand behind `M4.C.8a`) using only
  the index and the views, then check the answers against the full spec. Report the result and the tokens used.
