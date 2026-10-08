# Phase C QA gate 7 — learning-qa failure-pattern sweep of the `TRIANGLE` decision

Date: 2026-10-08
Branch: plan-phase-b (ahead of origin/plan-phase-b by 1 commit)
Reviewer: learning-qa failure-pattern sweep (inline review of commit `cf067f2`)

RANGE: origin/plan-phase-b..HEAD
COMMITS: 1. `cf067f2` — "bowen: decide TRIANGLE's roles — the target is the recruited third
(M1.C.1, M11.C.3)". `origin/plan-phase-b` is at `b39be15`. 18 files, +659/−734. The only
behavioural changes are in `src/bowen/engine/moves.py`, `src/bowen/engine/observe.py`,
`src/bowen/engine/recompute.py`, and `tools/mutation_record.py`; the rest are the regenerated
records, the spec/explainer/report/TODO/CHANGELOG/coverage edits, and the three role-encoding
unit tests.
APPLICABLE: P4 (hardcoded constants / after-the-freeze changes), P6 (a regenerated record whose
hash does not cover what it ran against), P19 (producer/consumer drift — a reader of a module's
role assignment falling out of step with the producer), P26 (a rule change after the freeze, the
least-reviewed class of change), plus the CLAUDE.md rules: acceptance tests assert a direction and
are mutation-proved, no invented constant presented as sourced, and a red suite is never normal.
CHECKED: the full `cf067f2` diff; `moves.py`, `recompute.py`, `observe.py`, `objects.py`,
`patterns.py`, `trace.py`, and `policy.py` in context (every reader of the triangle's positions);
the three role-encoding tests, empirically reverted to the old reading and shown to fail; the
regenerated ensemble/mutation/sweep records against their staleness tests; the §8 before/after
table against both the parent and current ensemble records (including a full verdict-by-verdict
diff of all 26 entries); the constants freeze log for post-freeze changes; and the default suite
with its exit code checked without a pipe.
NOT COVERED: the 19 `ensemble`-marked criterion runs (deselected by design; their committed
records are checked by the default suite), and plan §9's pending human reviews — as before.

---

## Test result

`python3 -m pytest tests/ -q` — **485 passed, 19 deselected** (25.29 s), **exit code 0. GREEN.**

The 19 `ensemble`-marked criterion tests are deselected by `pytest.ini` by design. The exit code
was captured without piping to `tail` and is 0. The 13 record/engineering staleness tests in
`tests/bowen/test_ensemble_record.py` pass inside the 485, so the committed `code_hash` values for
the ensemble, mutation and sweep records still match what their tools compute today.

---

## The decision, and whether the code matches it

The decision (`docs/phase_c_completion_report.md:239-248`, spec `M1.C.1` at
`docs/bowen_agent_model_spec_v2.md:358`, `moves.py:25-34`): the `TRIANGLE` target is the third
party the act recruits and stands **outside**; the **inside pair** is the sender and the triad
member it is most strained with. The code matches this exactly:

- `moves.py:180-181` — `outsider = target`; `insiders = (state.people[event.sender],
  state.people[partner])`, where `partner` is the `triangle_partner` result (`moves.py:142-155`),
  the third member furthest from the sender's optimum on the sender's tie to them.
- `recompute.py:130-131` — `outside = move.targets[0]`; `inside = TieId.of(*(m for m in members if
  m != outside))`. Both assignments agree with the transfer: the target is outside, the sender and
  the remaining member are inside.

The rename `triangle_outsider` → `triangle_partner` (`moves.py:142`) is propagated to its only
consumer, `observe.py:37` (import) and `observe.py:106` (call), which reads only `found[0]` (the
triad id), so the observation is behaviourally unchanged. A whole-tree grep for the old symbol
returns nothing.

## Every reader of the triangle's positions

- `objects.py:234-240` (`Triangle.__post_init__`) — the invariant
  `inside_pair.members() | {outside} == trio` still holds under the new assignment (inside =
  {sender, partner}, outside = target).
- `patterns.py:94-109` (`fixed_triangle`) — derives the outsider from the `inside_pair` log field
  (`:105-107`), so it follows the new semantics automatically: the "same outsider" is now the
  recruited third. It is a derived readout, not a hand-maintained parallel copy (P19 clean).
- `trace.py:146` and `:196` — display only; reads the `"outsider"` field of the transfer effect and
  the `"inside_pair"` field of the recompute effect, both of which now carry the target as
  outsider. Consistent.
- `policy.py:137,164` — `value_key` uses `obs.triangle_for` only for the triad's members
  (topology), never the roles; `triangle_position` reads beliefs over the other pair's contact,
  not the transfer roles. Unaffected by the flip.
- `invariants.py:48` — the named transfer "insiders to outsider" is comment-consistent with the
  new roles.

No reader was left on the old reading. The `DETRIANGLE`/`PREVENT_ALIGNMENT` path
(`recompute.py:74-100`, `_voided`) still keys on the act's two parties `{move.sender,
move.targets[0]}` — unchanged code, and the commit and report §8 state this is deliberate. The
docstring reword (`recompute.py:26-33`) now accurately describes the unchanged behaviour ("sent or
were the target of") instead of the now-wrong "inside pair". Consistent, not a defect.

## The tests encode the new roles and fail under the old ones

Three tests encode the roles: `test_moves.py:104`
(`test_m1c1_triangle_relieves_the_insiders_and_loads_the_outsider`), `test_moves.py:117`
(`test_m1c1_the_target_is_recruited_into_the_senders_most_strained_twosome`, assertion `:131`),
and `test_mechanisms.py:380` (`test_m1c3_triangle_move_sets_the_inside_pair`, assertion `:386`).

Verified empirically: I re-applied the old reading to `moves.py` (`outsider = partner`,
`insiders = (sender, target)`) and `recompute.py` (inside = `TieId.of(sender, target)`, outside =
the remaining member) and ran the three tests. All three failed — `test_moves.py:131` (the
strained child no longer relieved), `test_moves.py:114` (the recruited third no longer absorbs),
and `test_mechanisms.py:386` (inside_pair/outside inverted). Files were restored to HEAD
(`git status` empty) and the affected test files pass again (72 passed). The commit's "three unit
tests encode the new roles; each fails when the old reading is restored" is accurate.

The sign-inverted mutant `triangle-roles-swapped` (`tools/mutation_record.py:155-158`) is the same
reversal and is recorded red for `M11.C.3` (`docs/phase_c_mutation_record.md:44`).

## The records and the §8 before/after table

The three records pass their staleness tests, so they are current. The §8 table
(`docs/phase_c_completion_report.md:260-268`) was reconciled two ways:

- The "after" column matches the current `docs/phase_c_ensemble_record.md` exactly: C.3 PASS
  (pair −2.22, third +4.02), C.42 FAIL (−0.17), C.45 FAIL (+0.0024, p 0.17), C.27[stable,add_third]
  FAIL (+0.003), C.41[lower level: heavier stress] FAIL (reactive share +0.004, p 0.16),
  C.41[light: lower level] UNDETERMINED.
- The "before" column matches the parent record (`origin/plan-phase-b`) exactly: C.3 FAIL
  (pair +2.59, third −1.21), C.42 PASS (+0.22), C.45 PASS (+0.0051), C.27[stable,add_third] PASS
  (+0.044), C.41[lower level: heavier stress] PASS, C.41[light: lower level] FAIL.
- A full verdict-by-verdict diff of all 26 entries between parent and HEAD shows the verdict
  changes are exactly the six rows in the §8 table and nothing else; "all other entries — verdict
  unchanged" holds.

## Nothing was tuned to recover the stopped-passing criteria

The constants register is frozen at 2026-10-07 (`config/bowen/constants_frozen.md:64`) and
`config/bowen/constants_changes.md` has no row dated after the freeze — every row is 2026-10-07.
The commit touches no constant, and `src/bowen/ensemble/criteria.py` (the criterion arms and
readouts) is not in the diff, so no readout, arm, seed or threshold was changed to lift C.42, C.45,
C.27 or C.41. The four stopped-passing entries are reported (§8, §3.5, §3.7, §3.8), not recovered;
C.3's change is a direct consequence of the role flip, not of any criterion-side edit.

---

## Findings

None. No failure-pattern defect (P4/P6/P19/P26) was found in the range. Two observations, both
non-blocking and both already recorded in the tree, are noted for completeness rather than as
findings:

- **Informational — `triangle-roles-swapped` is a transfer-only reversal.** The mutant
  (`tools/mutation_record.py:155-158`) swaps the roles in `moves.py` only, leaving
  `recompute.py`'s readout on the new reading. This is correct for its purpose: C.3 reads the
  transfer's acute-anxiety `EffectRecord`, not the readout, so flipping the transfer alone flips
  the criterion's sign. The description "roles swapped back to step 5's alliance reading" refers
  to the transfer's roles. Not a defect.
- **Informational — `DETRIANGLE`/`PREVENT_ALIGNMENT` key on the act's parties, not the new inside
  pair.** Under the new reading the "aligned third party" of `M5.B.1` is the target, and `_voided`
  (`recompute.py:90-99`) keys on `{sender, target}`, so it voids acts where the counter's target
  was the recruited third — consistent with `M5.B.1`. The commit and §8 flag this as deliberate
  and the code is unchanged; a future reader should not assume it should now key on
  `{sender, partner}`. Not a defect in this commit.

---

## Verdict

VERDICT: APPROVED
