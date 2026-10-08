# Phase C QA gate 8 — learning-qa failure-pattern sweep of M11.C.42 / M11.C.45 re-test

Date: 2026-10-08
Branch: plan-phase-b (ahead of origin/plan-phase-b by 2 commits)
Reviewer: learning-qa failure-pattern sweep (inline review of commits `0670ee5` and `5a610a0`)

RANGE: origin/plan-phase-b..HEAD
COMMITS: 2.
1. `0670ee5` — "docs: decide what M11.C.42 and M11.C.45 test, before rerunning them"
   (decision only: TODO.md + `docs/phase_c_completion_report.md` §9).
2. `5a610a0` — "bowen: bring M11.C.42's arms and M11.C.45's readout to the spec, as decided in 0670ee5"
   (implementation + the three Phase C records regenerated).
10 files, +231/−80. Behavioural change is confined to `src/bowen/ensemble/criteria.py`
(`Forced.unavailable`, `scenario(unavailable=…)`, `arm_c42`, `arm_spell_triad`) and the new
`tests/bowen/test_phase_c_criteria.py`; the rest are the regenerated records, the report/TODO/
CHANGELOG/CLAUDE/coverage edits.
APPLICABLE: P19 (producer/consumer drift — a derived check that must mirror the producer's filter,
and must not be a parallel hand-copy), P27 (a test that cannot fail for the defect it names), the
CLAUDE.md rules (acceptance tests assert a direction and are mutation-proved; no invented constant
presented as sourced; nothing tuned against a result; a red suite is never normal), and the
learning-qa Pitfall 2 (derive checks from the producer's real code, not a hand-copy).
CHECKED: the full two-commit diff materialised to `/tmp/sweep-phase-c-8.diff`; `criteria.py` in full
context; the spec rows `M11.C.42` (`bowen_agent_model_spec_v2.md:1256`), `M11.C.45` (`:1259`),
`M17.D.3` (`:1947`) and `M4.D.1`; `policy.py` `decide`/`legal_outcomes` and `policy_source.py`
`selections` (the draw path the absence redraws); `draws.py` (keyed, cached, no generator state);
`observe.py` (what `ties` and `triangle_for` mean); `log_records.py`/`act.py` (record shapes and
`withheld_toward` derivation); the §9 numbers against `docs/phase_c_ensemble_record.md` and
`docs/phase_c_sweep_record.md`; the three record `code_hash` values recomputed against HEAD source
(all three match); and the default suite with its exit code checked without a pipe.
NOT COVERED: the 19 `ensemble`-marked criterion runs (deselected by design; their committed records
are checked by the default suite's staleness tests), and plan §9's pending human reviews — as before.

---

## Test result

`python3 -m pytest tests/ -q` — **488 passed, 19 deselected** (29.32 s), **exit code 0. GREEN.**

The 19 `ensemble`-marked criterion tests are deselected by `pytest.ini` by design. The exit code was
captured without piping to `tail` and is 0. The three new default-suite tests in
`tests/bowen/test_phase_c_criteria.py` run and pass (3 passed in 2.31 s). The record/engineering
staleness tests in `tests/bowen/test_ensemble_record.py` pass inside the 488.

---

## The decision, and whether the code matches it

The decision (`0670ee5`, `docs/phase_c_completion_report.md` §9) makes three changes, each read from
the spec's text and committed before either criterion was rerun:

**C.42 baseline arm** — the spec reads "that member is unavailable at that tick … so no triangle
forms"; the build had instead scripted the sender to `STAY-IN-CONTACT`. Now, at `t0`, the third
member (`c`) is unavailable as the target of any act by either member of the pair, under the same
keyed draw. Treatment arm is unchanged (the sender's `TRIANGLE` toward `c`).

`criteria.py:518-521` matches: `arm == "treatment"` sets `forced = act("f", "TRIANGLE", "c", t0)`,
`unavailable = None`; the baseline sets `forced = {}`, `unavailable = {t0: (pair, third)}` with
`pair = (f, m)`, `third = c`. The old `STAY-IN-CONTACT` script is gone.

**C.42 readout** — the spec reads "the **pair** MUST select `TRIANGLE` toward that member more
often"; the build counted only the sender. `criteria.py:524-525` now reads
`r.event.sender in pair … r.event.targets == (third,) … r.event.timestamp > t0`, counting both
members of the pair, from the tick after the act to the horizon.

**C.45 readout** — the spec reads "the rate of `TRIANGLE` selection"; every person selects exactly
one outcome per tick (`M4.D.1`), so the rate is `TRIANGLE` selections per person-week. The build
had divided by emitted moves. `criteria.py:543-545` now divides by `len(triad) * weeks`. The
`outside_inside_ratio` readout (C.44) is untouched.

The scenario settings (horizons, the week of the act, the spell) are unchanged in both arms.

---

## `Forced.unavailable` does only what it says, and cannot leak

`Forced.selections` (`criteria.py:159-179`) consults `self.unavailable` once, at
`if tick in self.unavailable`, and only that tick's key is present (`{t0: (pair, third)}`), so no
earlier or later tick is affected. It iterates `sorted(set(actors) & set(active))` and re-decides
only those actors (the pair), so the third person and every other person is untouched.

`_without` (`criteria.py:182-196`) rebuilds the actor's observation with `ties` filtered to
`v.other != absent` and `triangle_for` filtered to `t != absent and absent not in tri.members`, then
calls `decide` and returns the selection with its urge. Two properties verified against the source:

- **No random-stream shift.** `decide` draws `move_selection(tick, actor, purpose, index)` under
  `Keying.SLOT` (`policy.py:269-270`), whose key does not include the target, and `DrawService`
  caches by key with no generator state (`draws.py:165-208`). The redraw is the same key, so it
  returns the same uniform and inverts it through the *filtered* distribution — the documented
  "same keyed draw", not a fresh draw. Nothing else shifts.
- **The absent person cannot be a target, nor complete a triad.** `legal_outcomes` builds every
  target from `obs.ties` (`view.other`) and gates `TRIANGLE` on `view.other in obs.triangle_for`
  (`policy.py:150-171`). Both are filtered, so no emitted act and no withheld act (`withheld_toward`
  is `selection.targets[0]`, `act.py:68`) can name `c`. The `absent not in tri.members` clause also
  removes `triangle_for[m]` for actor `f` (its triangle is `(f,m,c)`, which contains `c`), so a
  pair member cannot `TRIANGLE` toward the *other* pair member either — correct, because in a triad
  any such act still forms a triangle involving the absent third, and the spec says "so no triangle
  forms". This is a superset of the decision's "unavailable as a target", and it is the right one.

---

## The new tests fail under a wrong implementation

- `test_m11c42_absence_removes_the_third_from_the_pairs_choices_that_tick` — asserts, over 12 seeds,
  that with the absence there is **no** emitted act toward `c` and **no** withheld act toward `c` at
  `t0` (`closed == []`), guarded by `assert open_` so the check is non-vacuous. It checks emitted
  (`EmittedRecord`, `THIRD in targets`) and withheld (`SelectionRecord.withheld_toward == THIRD`)
  separately. Removing the ties filter, or the triangle filter, or both, would leave an act toward
  `c` and fail this test.
- `test_m11c42_absence_changes_nothing_before_its_tick` — asserts the records before `t0` are
  identical between the open and absent runs. The absence is keyed to `t0`, so this holds; it would
  fail if the check were broadened to an earlier tick.
- `test_m11c45_triangle_rate_is_per_person_week` — recomputes the `TRIANGLE` count from the raw
  records and asserts `rate * 3 * weeks == triangles`, i.e. the denominator is `3 * weeks`
  (per person-week), not the old `len(acts_)`. Under the pre-decision implementation the assertion
  fails (it only holds when the denominator equals the person-week count).

---

## §9 numbers match the records

| Source | C.42 | C.45 |
|---|---|---|
| Report §9 table | FAIL, reuse +0.26 ± 0.33 (p 0.084); passes only at H = 6; reverses at α 0.1, H 2, T 0.5 | FAIL, +0.0020 per person-week (p 0.20); fails at all 6; reverses at 4 |
| `phase_c_ensemble_record.md` | `triangle_reuse` +0.26 ± 0.33, p 0.0841 | `triangle_rate` +0.00204 ± 0.0041, p 0.197 |
| `phase_c_sweep_record.md` | lr 0.1 −0.2 rev; lr 0.4 +0.15; ch 2 −0.15 rev; ch 6 +0.37 **PASS**; pt 0.5 −0.13 rev; pt 2.0 +0.14 | lr 0.1 −0.00329 rev; lr 0.4 −0.00169 rev; ch 2 −0.00263 rev; ch 6 +0.000889; pt 0.5 −0.00229 rev; pt 2.0 +0.00106 |

C.42 passes at exactly one of six settings (credit_horizon 6 = "H = 6") and reverses at three
(learning_rate 0.1, credit_horizon 2, policy_temperature 0.5), matching "passes only at H = 6;
reverses at α 0.1, H 2 and T 0.5" and "reverses at 3 of 6". C.45 fails at all six and reverses at
four, matching "fails at all 6; reverses at 4". Both central verdicts are FAIL, and neither
criterion passes.

---

## Nothing was tuned

The diff touches no file under `config/bowen/` — no `constants.md`, no `criteria.md`, no freeze
register, no `constants_changes.md`. The only source change is `src/bowen/ensemble/criteria.py`
(the arms/readouts the decision names). I recomputed the three records' `code_hash` values against
HEAD using the tools' own `code_hash()` (which hashes all of `src/bowen/**/*.py` +
`config/bowen/**/*.md` + the tool files):

- ensemble: computed `6aac4041…fe46399` == recorded `6aac4041…fe46399`
- mutation: computed `ce8a1c52…626f5c` == recorded `ce8a1c52…626f5c`
- sweep: computed `650e8ad8…8c32bf` == recorded `650e8ad8…8c32bf`

So the three records were regenerated from exactly the current source, including the current
config. The mutation record's mutant table is byte-identical to the parent (only its hash changed),
which is correct: its mutants exercise only the *passing* criteria, and no passing criterion's code
changed (`arm_spell_triad`'s `outside_inside_ratio` limb, which C.44 reads, is untouched). C.42 and
C.45 remain FAIL — no rule or constant was moved to make either pass.

---

## Findings

1. **LOW · P19 / learning-qa Pitfall 2 · `tests/bowen/test_phase_c_criteria.py:62-63`** — the C.45
   test recomputes the `TRIANGLE` count with
   `isinstance(r, EmittedRecord) and r.event.kind == "TRIANGLE"`, omitting the
   `r.event.mechanism is Mechanism.MOVE` filter the producer applies
   (`criteria.py:539-540`). The two predicates agree today because C.45's arms emit no non-MOVE
   `TRIANGLE` event, so the assertion is a parallel hand-copy that can drift rather than a shared
   single source of truth. Risk: if a `TRIANGLE`-kind event ever arrives with a non-MOVE mechanism,
   the test fails spuriously while the production readout stays correct. Fix: import `Mechanism`
   and add the same filter, or better, have the test read the producer's numerator through one
   shared predicate. Confidence: high that the divergence exists; low that it matters at the frozen
   constants.

2. **LOW · test-strength · `tests/bowen/test_phase_c_criteria.py:51-52`** —
   `test_m11c42_absence_changes_nothing_before_its_tick` filters with
   `getattr(r, "tick", None) is not None`, which silently drops `EmittedRecord` and
   `DeliveredRecord` (they carry `event.timestamp` / `delivery`, no top-level `.tick`). The
   assertion's name claims "changes nothing before its tick" but only compares Selection/Effect/
   Tick/Invariant/BeliefWrite records. Not a defect — the absence is keyed to `t0` and provably
   cannot affect earlier ticks — but the test verifies less than it claims. Fix: normalise each
   record to a timestamp and compare everything before `t0`, or narrow the name/comment.
   Confidence: high.

3. **LOW (informational) · coverage · `src/bowen/ensemble/criteria.py:524-525`** — the C.42 readout
   change ("the pair" instead of the sender only) has no default-suite test, unlike the C.45 readout
   change. The three new tests cover the absence mechanism and the C.45 denominator; the pair-wide
   `triangle_reuse` numerator is exercised only by the ensemble-marked criterion (deselected by
   design). The report §9 is honest about this — it claims only the absence and the C.45 readout are
   tested in the default suite. Fix (optional): a small unit test on `arm_c42`'s readout, or leave
   as documented. Confidence: high that it is untested; low severity.

4. **Informational · `src/bowen/ensemble/criteria.py:192-193`** — the `triangle_for` clause
   `absent not in tri.members` (which also removes the ability to `TRIANGLE` toward the *other* pair
   member) is load-bearing for "so no triangle forms" but is not directly covered by the
   target-removal test, which checks only `targets == (third,)`. A regression deleting this clause
   would pass the default suite while letting a pair member form a triangle involving the absent
   third. Not a defect in this commit; noted as the one untested clause of the guarantee.

---

## Verdict

VERDICT: APPROVED
