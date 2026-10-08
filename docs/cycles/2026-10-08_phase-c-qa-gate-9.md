# Phase C QA gate 9 — learning-qa failure-pattern sweep of the eighth gate's fixes

Date: 2026-10-08
Branch: plan-phase-b (ahead of origin/plan-phase-b by 1 commit)
Reviewer: learning-qa failure-pattern sweep (inline review of commit `b80db51`)

RANGE: origin/plan-phase-b..HEAD
COMMITS: 1.
1. `b80db51` — "bowen: fix the eighth gate's four test-strength findings (tests only)"
6 files, +50/−20. All four findings from `docs/cycles/2026-10-08_phase-c-qa-gate-8.md`
addressed in `tests/bowen/test_phase_c_criteria.py`; the rest are CHANGELOG/CLAUDE/TODO/
completion-report/coverage edits recording the fix and the 488→490 count.
APPLICABLE: P19 (producer/consumer drift — the C.45 test's recompute mirrors the producer's
numerator), P27 (a test must be provable-failing, proved by mutation), P29 (floor vs exact /
non-vacuous guards), the CLAUDE.md rules (acceptance tests assert a direction and are
mutation-proved; a red suite is never normal), and learning-qa Pitfall 2 (derive checks from
the producer's real filter, not a parallel hand-copy).
CHECKED: the full one-commit diff materialised to `/tmp/sweep-phase-c-9.diff`; the whole of
`tests/bowen/test_phase_c_criteria.py`; `criteria.py` `arm_c42`, `arm_spell_triad`, `Forced._without`,
`Forced.selections`, `act`, `scenario`, `result`; `log_records.py` (every record type's fields —
`EmittedRecord.event.timestamp`, `DeliveredRecord.delivery.delivered_tick`, `SelectionRecord.legal_set`,
`.withheld_toward`); `events.py` (`Mechanism`, `Event` fields); `act.py` (`_record`, `withheld_toward`
derivation); `runner.py` (`ArmResult.readouts`); `tools/ensemble_record.py` `code_hash` (its inputs are
`src/bowen/**/*.py` + `config/bowen/**/*.md` + the tool files, never tests); and the default suite with
its exit code captured without a pipe. Each fix was re-run through a mutation that reproduces the
breakage it names; every mutation was restored via `git checkout` and the tree verified clean.
NOT COVERED: the 19 `ensemble`-marked criterion runs (deselected by design; their committed records are
checked by the default suite's staleness tests), and plan §9's pending human reviews — as before.

---

## Test result

`python3 -m pytest tests/ -q` — **490 passed, 19 deselected** (29.96 s), **exit code 0. GREEN.**

Exit code captured without a pipe and is 0. The 19 `ensemble`-marked criterion tests are deselected
by `pytest.ini` by design. `tests/bowen/test_phase_c_criteria.py` runs 5 passed; the record/engineering
staleness tests in `tests/bowen/test_ensemble_record.py` pass (13 passed) inside the 490. The count
change 488→490 is exactly the two new tests (the no-triangle test and the pair-wide readout test).

---

## Each fix, verified

**Fix 1 — the C.45 rate check applies the readout's `MOVE` filter** (`tests/bowen/test_phase_c_criteria.py:95-96`).
The recompute now reads `isinstance(r, EmittedRecord) and r.event.mechanism is Mechanism.MOVE and
r.event.kind == "TRIANGLE" and r.event.sender in {*PAIR, THIRD}`, matching the producer's numerator
(`criteria.py:539-540,545`: `mechanism is MOVE` + `sender in triad`, then `kind == "TRIANGLE"`).
`{*PAIR, THIRD}` is `{f, m, c}` = the producer's `triad`, so the two predicates are now equivalent.
Verified by mutation that *removing* the filter from the test still passes (1 passed): at the frozen
constants no non-MOVE `TRIANGLE` event is emitted by C.45's arms, so this fix is defensive alignment,
not a mutation-provable guard — exactly the "low that it matters at the frozen constants" the eighth
gate assigned it. Correct as far as it goes.

**Fix 2 — the before-t0 check compares every record by its tick** (`tests/bowen/test_phase_c_criteria.py:50-63`).
`tick_of` normalises `EmittedRecord` via `event.timestamp` and `DeliveredRecord` via
`delivery.delivered_tick`, and the test now asserts the type set before `t0` includes
`{EmittedRecord, DeliveredRecord, SelectionRecord}` before comparing `[open] == [closed]`. Both
field names verified against `log_records.py`/`events.py` (`Delivery.delivered_tick` exists). The
guard is non-vacuous: all three types actually appear before `t0` (the suite is green). Mutation that
reverted the guard's filter to the old `getattr(r, "tick", None) is not None and r.tick < T0` failed
the test, the failure naming `EmittedRecord` and `DeliveredRecord` as present-but-uncompared — so the
strengthened test rejects the exact weakness the finding named. Fix correct and provable-failing.

**Fix 3 — the C.42 pair-wide readout now has a default-suite test**
(`tests/bowen/test_phase_c_criteria.py:75-86`). It reads `arm_c42("treatment", seed, settings)
.readouts["triangle_reuse"]`, re-runs the identical treatment-arm scenario, and asserts
`reuse == sum(1 for e in later if e.sender in PAIR)`, guarded by `assert by_partner` so the pair-wide
count is provably different from a sender-only count. The re-run matches the producer's arm exactly
(`forced=act("f", "TRIANGLE", "c", T0)`, no `unavailable`, no `setup`). Mutation of the producer to
`sender == pair[0]` (sender-only) failed the test — `reuse` 0.0 vs the pair count 2 (both from the
partner `m`; in that seed the scripted sender `f` did not re-triangle at all). Fix correct and
provable-failing.

**Fix 4 — the absence's `triangle_for` clause now has a direct test**
(`tests/bowen/test_phase_c_criteria.py:66-72`). It asserts that at `t0` neither pair member's
`SelectionRecord.legal_set` contains any `TRIANGLE>` label — i.e. the pair cannot `TRIANGLE` at all,
because every triad they could form holds the absent member. Mutation that deleted
`absent not in tri.members` from `_without` (`criteria.py:193`) failed the test, surfacing
`['TRIANGLE>m', 'TRIANGLE>f']` as newly legal. Fix correct and provable-failing; the one untested
clause named in the eighth gate is now covered.

---

## No src or config changed; records still current

The diff touches `CHANGELOG.md`, `CLAUDE.md`, `TODO.md`, `docs/phase_c_completion_report.md`,
`docs/spec_coverage.md`, `tests/bowen/test_phase_c_criteria.py` — and nothing under `src/` or
`config/`. The three records' `code_hash` inputs (`src/bowen/**/*.py` + `config/bowen/**/*.md` +
the generating tools, `tools/ensemble_record.py:34-50`) are untouched by this commit, so the hashes
recorded in `docs/phase_c_ensemble_record.md`, `docs/phase_c_mutation_record.md` and
`docs/phase_c_sweep_record.md` remain valid; the three staleness tests in
`tests/bowen/test_ensemble_record.py` re-assert exactly this and pass. No record was regenerated, and
none needed to be: the eighth gate's decision stands (C.42 and C.45 both FAIL), untouched by a
test-only commit.

---

## Findings (fix-commit re-sweep)

1. **Informational · P19-adjacent / latent robustness · `tests/bowen/test_phase_c_criteria.py:56`** —
   `tick_of` falls through to `record.tick` for any record that is neither `EmittedRecord` nor
   `DeliveredRecord`. `LogHeader` (the eighth member of the `Record` union, `log_records.py:155-164`)
   also carries no `.tick`, so `tick_of(LogHeader)` would raise `AttributeError` rather than compare.
   Not a live defect — `run_tick` never emits a `LogHeader` (it is constructed only in
   `scenario/header.py:74`, which is off the `scenario` path) — but `tick_of` is not total over the
   `Record` union, the same latent class the eighth gate flagged. Risk: low; fix optional (return a
   sentinel or assert the record type is one of the seven `run_tick` emits). Confidence: high that the
   gap exists, low that it fires.

2. **Informational · learning-qa Pitfall 2 (residual) · `tests/bowen/test_phase_c_criteria.py:95-96`** —
   the C.45 recompute is still a parallel hand-copy of the producer's numerator, not a shared
   predicate. The eighth gate offered "import `Mechanism` and add the same filter, **or better**, have
   the test read the producer's numerator through one shared predicate"; the fix took the first option,
   so a future change to the producer's filter would drift the test again. The current predicates are
   equivalent (verified), so this is accepted rather than blocking; noted so the drift is not re-swept
   as a surprise next time.

3. **Trivial · documentation · `CLAUDE.md:79`** — the red-suite bullet now reads "all 490 tests pass
   (measured 2026-10-07 …)" but 490 was first measured 2026-10-08, when the two new tests landed. The
   count is right; the measured date is one day stale. Cosmetic only.

No finding is medium or higher.

---

## Verdict

VERDICT: APPROVED
