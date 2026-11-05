# Version 3, part 1: the emergence audit — 2026-11-03

The owner decided that version 3 adds two utilities: this emergence audit and a separate LLM-agent experiment
(planned in its own session). The audit reruns every passing composite and mixed criterion with learning from felt
relief removed, reports which results survive, and traces each survivor to the non-learning term that carries it.
The plan is `docs/plan_v3_emergence_audit.md`.

- [ ] T-182 ⟨later⟩ **The emergence audit: build the approved plan (2026-11-03).** `docs/plan_v3_emergence_audit.md`,
  written on spec revision 47 (engine hash `0b363cb3186c`), approved by the owner 2026-11-03. It sets
  five ablations (L0 all learning off; L1–L4 one learning path each), the scope (passing composite and mixed rows of
  the three ensemble records; premises and checks out; `M11.C.39` the positive control, `M11.C.3` the negative), the
  verdicts (survives, red, reversed), the carrier trace (L0 plus each policy term C1–C6 and the row's own mutants C7),
  the record (`docs/emergence_audit_record.md`, `tools/emergence_audit.py`, outside the engine hash), acceptance
  criteria EA-1 to EA-16, and a cost of about 6–17 h of machine time split across both Macs, run after T-181. Open
  until it is built; T-180 is the first row it answers.
  **Decided 2026-11-03:** the owner answered Q1–Q7 through `AskUserQuestion`, each as recommended, and approved the
  plan as written: `M11.1f` in the spec as version 3's first revision; L0 and L1–L4; joint mutants C1–C7; the
  records' adaptive seed rule; a survivor is triaged as its own T-item and the audit's batch changes no rule; not a
  stage condition yet; the pilot once the tool is built, the full run after T-181.
  **Built 2026-11-03 (§7 steps 1–3):** the tool, the config file and EA-1 to EA-16, each proved red by its §6
  mutation; the requirement is `M11.1g` at spec revision 48 (`M11.1f` was taken). The pilot ran in 28 min on this Mac:
  both controls hold; `M11.C.27[unstable,remove_one]`, `M11.C.41[light: lower level]` and `M11.C.48[phase_d]` all
  survive L0. Open until steps 4–6 run, after T-181, with the owner's approval of the restated cost; each survivor
  gets its T-item once its carriers are known.
  **Owner 2026-11-03, on the restated cost** (17,700 seed-ticks a minute measured; carriers 6–30 joint mutants a
  row, 134 for all 13 rows against the plan's 21): after T-181, run step 4 only, L0 on all 13 rows split across
  both Macs (about 1.3–3.8 h); then cost L1–L4 and the carriers from the real survivor count and put them to
  the owner before they start.
  **Step 4 run 2026-11-04** (`docs/emergence_audit_record.md`, stage `l0`, added to the tool for it): L0 on the 13
  in-scope rows, the two 2080-tick rows not cached by the pilot on Mac Mini1 (43 min) and eight here (6 min). Both
  controls hold; 10 of 13 survive; `M11.C.34`, `M11.C.10[family of origin]` and `M11.C.2[twelve-person family]` go red. Open until L1–L4 and
  the carriers (113 joint mutants for the 10 survivors) run, or the owner closes it.
  **Owner 2026-11-04, on the restated cost** (carriers on the 10 survivors about 2.5–6 h of wall time on both
  Macs, `M11.C.48[phase_d]` 61% of it; L1–L4 on all 13 rows about 2.5–6.5 h): fix T-184 first, then run the
  carriers of the 10 survivors only; L1–L4 waits. The tool needs a stage for that (L0 and the carriers, no
  L1–L4), with a test and its mutation, as `l0` was added.
  **Step 6 run 2026-11-04** (stage `carriers`, added with T-184's fixes): the 113 joint mutants of the 10 survivors
  on `0b363cb3186c` (Phase D `020160abb6db`), split across both Macs in 2 h of wall time, 15:19 to 17:19: Mac Mini1
  ran `M11.C.48[phase_d]`'s C1–C4 and `M11.C.15` at 10 workers, this Mac `M11.C.48[phase_d]`'s C5, C6 and its own
  mutant and the other eight rows at 8 under `nice -n 15`. Both shares exited 0; the merge added Mac Mini1's 12
  results and found 15 equal (128 in the cache). The cross-check of the two Macs at this engine hash is T-181's
  (510 equal), under the owner's ruling MR-21. Carriers against each row's `M11.5` note
  (`docs/emergence_audit_record.md`, *Carriers*):
  - `M11.C.48[phase_d]`: **no carrier** among its 7 candidates, and its own mutant `cutoff-blind-to-learned-value`
    survives again (T-180). `M11.5` says it rests on `CUTOFF`'s learned value and comes from learning; with all
    learning off it still passes, so the note is contradicted and what carries it is not among C1–C7.
  - `M11.C.15`: carried by `death-all-bound-anxiety-discarded` only. `M11.5` names no rule (`M7.D.3` withdrawn).
  - `M11.C.7[joint over one spouse]`: carried by C2 (the mixing weight blind to level, policy only), C4
    (perspective steering), C6 (competing urges) and `landed-contact-changes-nothing`. `M11.5` names only `M8`'s
    predicate, for the landed-contacts limb, and calls the deviation ordering composite.
  - `M11.C.7[joint over sided]`: carried by `landed-contact-changes-nothing` and `coach-position-blind`; no policy
    term. Matches `M11.5` (`M8`'s predicate; the coach's going along with a `SPLIT` a declared premise, D29).
  - `M11.C.16`: carried by C1 (every capacity layer available, `M4.D.3a`), `level-blind`, the three
    layer-availability mutants (`availability-level-inverted` reverses it) and `c16-grounds-removed`. `M11.5`'s note
    says inverting `M4.D.3a`'s availability leaves it passing with learning on; with learning off it reverses.
  - `M11.C.27[unstable,remove_one]`: carried by C2 and `M4.C.1`'s deviation mutants (both reverse it). `M11.5` says
    it follows from `M4.C.1`'s two-sided shape; C2 is not in the note.
  - `M11.C.41[light: lower level]` and `[heavy: lower level]`: each carried by C2, the two mixing-weight mutants,
    `level-blind` and `steepness-inverted`. Matches `M11.5` (premise through `M4.D.1a`'s mixing weight); its premise
    limbs are not declared (pending the owner, `docs/version_1_dispositions.md`), so judged on every limb.
  - `M11.C.22`: carried by `lock-in-removed`, under which both limbs go to zero. `M11.5` rests the severe limb on
    `M7.D.2a`/`M7.D.2d` and calls the mild limb composite; the mild limb is carried by lock-in too.
  - `M11.C.11`: carried by `removed-member-holds-not-released-on-return` (its `disturbance` limb to zero). `M11.5`
    rests the conservation limb on `M6.I.6` and calls the return limb composite.
  **Proposed for triage** (one T-item per survivor, owner to order; none made yet): (1) `M11.C.48[phase_d]`, no
  carrier found, note contradicted (with T-180); (2) `M11.C.15`, carried by a death rule `M11.5` does not name;
  (3) `M11.C.7[joint over one spouse]`, carried by three policy terms; (4) `M11.C.22`, the mild limb carried by
  lock-in; (5) `M11.C.16`, `M4.D.3a`'s availability reverses it with learning off; (6) `M11.C.27`, C2 beside
  `M4.C.1`; (7) `M11.C.11`, the return limb carried by `M6.I.6`'s hold; (8) `M11.C.41`, both cells, as `M11.5`
  says, with the pending premise-limb question; (9) `M11.C.7[joint over sided]`, as `M11.5` says. Each is decided
  by `M11.1d`'s test: whether a rule that carries it names its outcome.
  **Owner 2026-11-04, through `AskUserQuestion`:** the nine T-items are made in the order proposed above, the four
  rows whose carriers match `M11.5` last; L1–L4 do not run now (the survivors are triaged first). Open until the
  nine T-items are made and triaged, or the owner closes it.
- [x] T-184 ⟨later⟩ **The emergence audit's cache records no NumPy version, so `tools/merge_cache.py` refuses it
  (2026-11-04).** `tools/emergence_audit.py` never sets `Cache.meta["numpy"]`, as `tools/version_2_sweep.py`'s
  `check_numpy` does, so a two-Mac split of the audit cannot be merged by the merge tool as it stands. Step 4's
  merge stamped NumPy 1.26.4 on scratch copies of both files (both interpreters measured at 1.26.4 by
  `tools/two_mac_sweep.py check` and again after the run), ran `merge_cache.merge` on them for its conflict check,
  and wrote the union back without the stamp. Also: `merge_cache.local_sweep_running` looks only for
  `version_2_sweep.py`, not for `emergence_audit.py`. And: the tool has no way to run a subset of rows, and
  `Cache.save` keeps only the keys read since loading, so a subset run (step 4 used `_results` on each Mac's
  rows) silently drops every other cached result; it dropped the pilot's five, recovered from git. Add a row
  subset to the tool that flushes without pruning, with a test that a subset run keeps the other keys. Fix all
  three, with tests, before step 5 is split (learning-qa, 2026-11-04).
  **Done 2026-11-04:** `tools/emergence_audit.py` runs `tools/version_2_sweep.check_numpy` before any run (its
  `check_numpy`), so its cache records NumPy and a mismatch or an unstamped cache holding results is refused;
  `merge_cache.local_sweep_running` looks for both tools (`RUNNING_TOOLS`); `--criteria` runs a share
  (`share`: the stage's ablations on the named rows and the carriers of each L0 survivor; `--candidates` splits one
  row's carriers) and flushes without pruning; `--render-only` runs nothing. The 15 unstamped committed results were
  adopted by stamping NumPy 1.26.4 once by hand, as the sweep's cache was on 2026-10-23: each was made on one of the
  two Macs, both measured at 1.26.4 before and after step 4 (`docs/two_mac_sweep.md`, *The emergence audit*). Tests
  `test_t184_a_share_runs_only_its_rows_and_keeps_the_others`,
  `test_t184_a_share_refuses_an_unknown_row_or_label_before_running`,
  `test_t184_a_run_records_numpy_and_refuses_another`, `test_t184_merge_sees_a_running_audit`, each proved red by
  its mutation (`save` for `flush`; the NumPy check dropped; the sweep-only pattern; the label filter and the two
  refusals dropped); after the reviews, `test_t184_a_share_runs_no_carriers_for_a_row_red_under_l0`,
  `test_t184_render_only_runs_nothing_and_refuses_a_missing_result` and `test_t184_argument_refusals`, and the
  running check's pattern made to skip the test files' names (six more mutations).
