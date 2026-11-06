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
  **Made 2026-11-04:** T-185 to T-193, in the owner's order (T-192 holds both `M11.C.41` cells) (below), each with its carriers,
  its `M11.5` note, `M11.1d`'s question and Claude's recommendation; T-185 is put to the owner first. Open until
  the nine are decided.
  **2026-11-05:** T-186 to T-193 decided by the owner (each below); T-185 is still open. T-186 to T-191 built the same day at spec revision 49, with `M11.1g`'s scope sentence from T-192.
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

## The ten survivors, nine T-items (T-182's triage, 2026-11-04)

Made in the owner's order of 2026-11-04; T-192 holds `M11.C.41`'s two cells, so ten survivors make nine items. Carriers and results are from `docs/emergence_audit_record.md` (stage
`carriers`, engine `0b363cb3186c`, Phase D `020160abb6db`); each row's class and note are `M11.5`'s table. Each item
asks `M11.1d`'s question of its carriers: does a rule that carries the row name its outcome? A survivor is not a
defect by itself; the owner decides each. No rule changes in this batch: an item decided as a wording change waits
for one spec revision that takes all nine.

- [x] T-185 ⟨later⟩ **`M11.C.48[phase_d]` survives with all learning off and no carrier is found (2026-11-04).**
  With T-180. Under L0 the row passes at 200 seeds (`negative_run` +0.245 ± 0.076, p 5.3e-13), and it passes under
  every one of its 7 candidates: C1–C6 (+0.187 to +0.306) and its own mutant `cutoff-blind-to-learned-value`, whose
  numbers equal L0's exactly, since with learning off there is no learned value for the mutant to remove.
  `M11.5` says it rests on `CUTOFF`'s learned value (`M4.D.6`) and the cut-off tie's stock (`M1.D.2b`), and that it
  "comes from learning over time"; the first is contradicted by L0, and the second holds only on a tie already cut
  off, so it is not an obvious source of the run before a tie's first cutoff. The legal set's structural
  preconditions were not mutated (plan §4).
  **`M11.1d`'s question** cannot be asked yet: the minimal set of rules that produces the direction is not known.
  **Claude's recommendation:** trace it before deciding: read the policy path, without learning, from a run of
  negative contacts on a tie to `CUTOFF`'s weight on that tie, then run a per-act probe under L0
  (`tools/probes.py`; the plan's Q3 names it as useful if C1–C7 find no carrier) to find which term carries it;
  then ask `M11.1d`'s question of what it finds and decide T-180 with it (whether the mutant, the rule or the row's
  mutation clause changes). `M11.5`'s note stays as it is until then. Alternatives: amend `M11.5`'s note now to say
  the row passes without learning and its carrier is not found (wording, at the next revision); replace the row's
  mutation clause with a mutant chosen by reading and run it; or record the survivor as reviewed and change nothing.
  **Owner 2026-11-05, through `AskUserQuestion`:** as recommended: trace, then probe. A read-only trace of the
  policy path from a run of negative contacts to `CUTOFF`'s weight with learning off, then a per-act probe under L0
  (`tools/probes.py`); then `M11.1d`'s question of the carrier found, and T-180 decided with it. `M11.5` unchanged
  until then. Open until the carrier is found and the owner decides.
  **The trace (2026-11-05, read only, learning off).** Before a tie's first cutoff nothing in the policy is tie-specific
  for `CUTOFF`: its within-channel weight is `exp(0) × availability(CUTOFF)` (`policy.channel_weights`), the same on
  every live tie of the person; focus steering (`M4.D.1g`) bears on `OVERFUNCTION`, `PURSUE` and `TRIANGLE` only;
  perspective's steering on `TRIANGLE`, `CONFLICT`, `DETRIANGLE` and `PREVENT_ALIGNMENT` only; the loaded-tie gate on
  the self channel only. `CONFLICT` and `DISTANCE` are as tie-blind. On a tie not declared cut off, no refusal can
  fall before its first cutoff: rejection is raised by a delivered `CUTOFF` (`event_effects`), by a refusal, which needs
  rejection held (`rejection.refuses`), and at the start only on a declared cut-off tie
  (`rejection.seed_declared_rejection`), which the row leaves out. A cutoff cannot make a refusal count in its own week
  either: refusals are taken at the tick's step 2, cutoffs applied at step 4. So with learning off the run before a
  cutoff is `CONFLICT` and `DISTANCE`
  against `STAY-IN-CONTACT`, and the policy does not pick the tie with the longer run to cut.
  **The probe (2026-11-05)** (`tools/c48_carrier_probe.py`; not `tools/probes.py`, whose every edit reruns the
  diagnostic record's probes): the row's own scenario, 200 seeds, each matched first cutoff's run read against subsets
  of its baseline, under L0 (in a temporary copy) and on the unmodified engine. It reproduces both records' differences.
  A tie's **sides** are how many of its members had sent a move other than `CUTOFF` across it by the cutoff's week
  (the run's window); a member under the acting age (`acting_age_years`, 10) sends none, but the probe does not read
  age, so a one-sided tie is any tie one member has not acted on. A subset line reads only the cutoffs whose subset is
  not empty, so the row's own difference on those same cutoffs is given beside it.

  | Difference in the run (cutoffs read: L0 / learning on) | L0: subset · the row on those cutoffs | learning on: subset · the row on those cutoffs |
  |---|---|---|
  | the row: against every uncut tie in contact (3432 / 3372) | +0.245 ± 0.076 | +0.344 ± 0.078 |
  | against uncut ties with as many sides (2876 / 2800) | +0.104 ± 0.070 · +0.193 ± 0.070 | +0.098 ± 0.082 · +0.242 ± 0.080 |
  | two-sided cut ties, against two-sided uncut ties (1325 / 1344) | +0.066 ± 0.127 · +0.412 ± 0.106 | +0.038 ± 0.120 · +0.470 ± 0.097 |
  | one-sided cut ties, against one-sided uncut ties (1551 / 1456) | +0.123 ± 0.093 · −0.032 ± 0.102 | +0.147 ± 0.123 · +0.017 ± 0.124 |
  | against the cutter's other uncut ties (1522 / 1478) | −0.014 ± 0.105 · +0.081 ± 0.092 | −0.009 ± 0.125 · +0.082 ± 0.115 |
  | against the other member's other uncut ties (1835 / 1854) | +0.293 ± 0.093 · +0.230 ± 0.085 | +0.276 ± 0.102 · +0.318 ± 0.090 |
  | the cutter's other uncut ties, against every uncut tie (1522 / 1478) | +0.094 ± 0.053 | +0.091 ± 0.054 |
  | two-sided against one-sided uncut ties, baseline only (1775 / 1774) | +0.785 ± 0.129 | +0.842 ± 0.116 |
  | share of ties two-sided, cut / baseline (pooled over cutoffs) | 0.509 / 0.327 | 0.531 / 0.324 |

  (Per-seed means over 200 seeds, ± 1.96 standard errors over seeds, except the pooled shares; the counts are
  cutoffs.) **What it shows.** Within a person the policy is tie-blind for `CUTOFF` before a first cutoff (the trace),
  and the row's direction comes from what its two arms are made of, in two parts. (1) *Sides:* a cut tie is more often
  one both members act on (51% against 33% under L0; 53% against 32% with learning on), and those ties hold runs longer
  by about 0.8 (two members can add to the run and either can cut; the probe does not test this). Matching on sides removes about 0.09 of the row's
  difference under L0 and 0.14 with learning on, on the same cutoffs, and among two-sided cut ties it removes nearly all
  of it (+0.41 to +0.07; +0.47 to +0.04). (2) *Who cuts:* the cutter's other uncut ties hold runs longer than the
  family's by about +0.09 in both, a person-level term the probe does not trace (what makes a person cut more, such as
  the automatic channel's share or anxiety, is not tested). About +0.10 is left after matching on sides, the same with
  learning and without, and it sits in the one-sided cut ties (+0.12, +0.15); it may be part (2) and is not
  explained. The line against the cutter's own other ties cannot decide between the two: the row on those cutoffs is
  itself only +0.08 (± 0.09 under L0, ± 0.12 with learning on). Learning's addition to the row (+0.245 to +0.344) is not left after matching on sides
  (+0.104 against +0.098), within intervals of about ± 0.08, which fits `cutoff-blind-to-learned-value` leaving the
  row passing.
  **`M11.1d`'s question:** no rule in the set found names the run or the cutoff — the acting age, the draw of `CUTOFF`
  from a person's live ties with equal weight, and the readout's baseline (every uncut tie in contact) — so it is not a
  programmed premise. It is a comparison whose arms differ in a condition (how many members act on the tie, and whose
  ties they are), the same class as T-123 and T-179 (learnings P7). Not shown: the person-level part and the +0.10
  left after matching. T-180 is put to the owner with this.
  **Owner 2026-11-05, through `AskUserQuestion` (with T-180):** wording only, as recommended. At spec revision 50,
  `M11.5`'s row for `M11.C.48` and the criterion's own row say that on the twelve-person family the direction holds
  with all learning off and comes in part from what its arms are made of (how many members act on the tie, and who cuts), not
  from `CUTOFF`'s learned value; the mutation clause says `cutoff-blind-to-learned-value` did not turn
  `M11.C.48[phase_d]` red, since the row passes with learning removed, and W3 stays unmet on that row with this reason. The pass
  is recorded as reviewed and is not reported as a finding about learning. No rule, code or config changes; the engine
  hash stays `0b363cb3186c`. Closes when revision 50 is built.
  **Built 2026-11-05:** spec revision 50 (`docs/spec_revision_log.md`), wording in `M11` only; the engine hash stays `0b363cb3186c`.
- [x] T-186 ⟨later⟩ **`M11.C.15` is carried by the release of bound anxiety at a death, a rule `M11.5` does not name
  (2026-11-04).** Under L0 it passes (`death_rise` +3.83 ± 0.65, `recovery_rise` +7.45 ± 0.9, 50 seeds). Its one
  carrier is `death-all-bound-anxiety-discarded` (`M6.3c`'s release at a death, under `M6.I.1`, replaced by
  discarding): `death_rise` goes to -0.872 ± 0.65 and `recovery_rise` does not move. Discarding the symptom hold
  alone (`death-bound-anxiety-discarded`) leaves it passing (`death_rise` +1.4 ± 0.64). `M11.5` classes it
  composite, "rests on" blank, note "`M7.D.3` withdrawn"; the criterion row's own mutation clause already names
  `M6.I.1` and `M6.3c`.
  **`M11.1d`'s question:** does `M6.3c` or `M6.I.1` name the row's outcome, that death destabilises "exactly as
  recovery does"? `M6.3c` has three clauses: tie-bound anxiety on a tie to the dead goes to the tie's living member;
  a triangle with the dead discharges equally to its living members; and the dead person's `symptom_bound` goes back
  to the partners, "as when a symptom ends". The third states the equivalence with recovery, but for the symptom
  hold only, and that hold alone does not carry the row; the first two do not name destabilisation. `M6.I.1` says
  the budget is conserved.
  **Claude's recommendation:** keep it composite and fill `M11.5`'s "rests on" cell with `M6.3c`'s release (under
  `M6.I.1`), the carrier found (wording, at the next revision). Alternatives: class it premise, reading `M6.3c`'s
  "as when a symptom ends" as writing the equivalence in (the text supports this for the symptom hold only); or
  change nothing.
  **Decided 2026-11-05:** the owner chose through `AskUserQuestion`, as recommended: keep it composite and fill `M11.5`'s "rests on" cell with `M6.3c`'s release at a death (under `M6.I.1`), the carrier found. Wording, waiting for the one spec revision that takes all nine.
  **Built 2026-11-05:** spec revision 49 (`docs/spec_revision_log.md`), wording in `M11.5` only; the engine hash stays `0b363cb3186c`.
- [x] T-187 ⟨later⟩ **`M11.C.7[joint over one spouse]` is carried by three policy terms and the landed contact
  (2026-11-04).** Under L0 the composite limb passes thinly (`deviation_fall` +0.0503 ± 0.057, p 0.024, 100 seeds).
  Carriers: C2 (`M4.D.1a`'s mixing weight blind to level, in the policy only), C4 (`M4.D.1a`'s perspective steering),
  C6 (`M4.D.1d`'s competing urges) and `landed-contact-changes-nothing` (`M1.E.7`'s `perspective_gain` 0); under each
  the limb goes red (p 0.12 to 0.43). `M11.5`: mixed, `M8`'s predicate for the landed-contacts limb; the deviation
  ordering composite, "no rule writes it".
  **`M11.1d`'s question:** does any of the four name the ordering joint > one spouse in the couple's
  deviation fall? None does: each is a general term of selection or of a coach's contact, and the row needs
  all four. The thin L0 margin means part of what the joint mutants show is how little the pass has to lose.
  **Claude's recommendation:** no defect; keep it mixed and add the four carriers to `M11.5`'s note (wording, at the
  next revision). Alternatives: change nothing; or rerun L0 on the row at more seeds first, to see whether the thin
  pass holds.
  **Decided 2026-11-05:** the owner chose through `AskUserQuestion`, as recommended: no defect; keep it mixed and add the four carriers (C2, C4, C6 and `landed-contact-changes-nothing`) to `M11.5`'s note. Wording, at the next revision.
  **Built 2026-11-05:** spec revision 49 (`docs/spec_revision_log.md`), wording in `M11.5` only; the engine hash stays `0b363cb3186c`.
- [x] T-188 ⟨later⟩ **`M11.C.22`'s mild limb is carried by lock-in alone (2026-11-04).** Under L0 both limbs pass
  (`mild_removal_effect` +1.51 ± 0.73, `severe_removal_effect` -1.73 ± 0.88, 100 seeds). The one carrier,
  `lock-in-removed` (`M7.D.2a`'s stabilising term flattened), takes both to the same -0.0142 ± 0.058 (50 seeds), so
  it removes the whole difference between the arms rather than the mild limb alone. `M11.5`: mixed since revision 32
  (it was premise), the severe limb on `M7.D.2a` and `M7.D.2d`, the mild limb composite as "the net of lock-in and
  other mechanisms"; the `M11.C.22` row gives the net as `M7.D.2a`'s lock-in (+7.3) against other mechanisms (about
  −6.8). Those other mechanisms work against the limb, so the audit could not find them as carriers.
  **`M11.1d`'s question:** does `M7.D.2a` name the mild limb's outcome? It names the direction: the configuration
  "resists the symptom's removal". It does not name the margin the limb asserts, that the resistance outweighs the
  relief removing any member brings (the row: "Unrestated, the mild limb fails").
  **Claude's recommendation:** keep it mixed and amend `M11.5`'s note to say lock-in alone carries the mild limb's
  direction and the other mechanisms only reduce it (wording, at the next revision). Alternatives, about as strong:
  class the mild limb premise as well, on `M7.D.2a`'s "resists the symptom's removal", which reverses revision 32's
  reasoning and takes the row out of the audit's scope; or change nothing.
  **Decided 2026-11-05:** the owner chose through `AskUserQuestion`, as recommended: keep it mixed and amend `M11.5`'s note to say lock-in alone carries the mild limb's direction and the other mechanisms only reduce it. Wording, at the next revision.
  **Built 2026-11-05:** spec revision 49 (`docs/spec_revision_log.md`), wording in `M11.5` only; the engine hash stays `0b363cb3186c`.
- [x] T-189 ⟨later⟩ **`M11.C.16` is reversed by inverting `M4.D.3a`'s availability once learning is off
  (2026-11-04).** Under L0 it passes (`repertoire_entropy` -0.0151 ± 0.0019, 100 seeds), as `M11.5`'s note already
  records ("With the learner disabled it still passes"). Carriers: C1 (every capacity layer available),
  `level-blind`, `availability-level-independent`, `availability-and-learner-removed`, `c16-grounds-removed`, and
  `availability-level-inverted`, which reverses it (+0.0117 ± 0.0018). What is new is that reversal: with learning
  on, the same inversion leaves it passing. `M11.5`: composite by owner decision. Its note says that inversion
  equals the deletion over much of the lowered arm's member-weeks, so "a stronger inversion might reverse it; none
  was tried"; that composite "rests on a reading: of the nine rules only availability names a narrower repertoire";
  and that a same-day reclassification to premise was withdrawn. (The note gives `availability-and-learner-removed`
  as reversed, +0.002 ± 0.002; the audit has it red, +0.00143 ± 0.0015.)
  **`M11.1d`'s question:** `M11.1d` says a criterion that flips under inversion of a single rule is restated one
  level up or recorded under `M10.C.5` as a programmed premise. Here it flips under the inversion together with all
  learning removed, so it is not a clean single-rule flip; the rule inverted (`M4.D.3a`: rising anxiety slides
  selection down the complexity ordering) is the one the note reads as naming a narrower repertoire.
  **Claude's recommendation:** keep it composite and add the L0 reversal to `M11.5`'s note, with the corrected
  `availability-and-learner-removed` result (wording, at the next revision); the note's own caveat (the inversion is
  weak) is answered in one direction by L0, and the premise reading was already put and withdrawn. Alternatives:
  class it premise on `M4.D.3a` and mark the rule under `M10.C.5`; or restate it one level up (for instance, require
  the learned narrowing to exceed what availability alone gives under L0).
  **Decided 2026-11-05:** the owner chose through `AskUserQuestion`, as recommended: keep it composite and add to `M11.5`'s note the L0 reversal under `availability-level-inverted` (+0.0117 ± 0.0018) and the corrected `availability-and-learner-removed` result (+0.00143 ± 0.0015, red, not reversed). Wording, at the next revision. (At the plan's review the same day the owner chose to keep the mutation record's reversal beside it, each run named, rather than call the audit's result a correction; see the Built line.)
  **Built 2026-11-05:** spec revision 49 (`docs/spec_revision_log.md`): `tools/c16_report.py` reads the two results from the audit record's JSON into `M11.C.16`'s row, beside the mutation record's reversal, each run named (the owner, 2026-11-05: keep both rather than call the audit's a correction); test `tests/bowen/test_ensemble_record.py::test_t189_c16_row_l0_sentence_follows_the_audit_record`, proved failing by mutation. The engine hash stays `0b363cb3186c`.
- [x] T-190 ⟨later⟩ **`M11.C.27[unstable,remove_one]` is carried by `M11.5`'s `M4.C.1` and also by C2
  (2026-11-04).** Under L0 it passes (`pair_deviation` -0.0696 ± 0.049, 100 seeds). Carriers: C2 (`M4.D.1a`'s mixing
  weight blind to level, in the policy only; red), `one-sided-deviation` and `deviation-inverted` (both reverse it).
  `M11.5`: composite, "follows from `M4.C.1`'s two-sided shape"; checked 2026-10-08, it flips only under inversion of
  `M4.C.1`'s deviation, and was kept composite. C2 is not in the note.
  **`M11.1d`'s question:** does `M4.C.1` (anxiety rises with deviation from the contact optimum, either side) or the
  mixing weight name that removing one stabilises an unstable twosome? Neither does; but the row flips under one rule's
  inversion, which `M11.1d` answers by restating or by `M10.C.5`, and that was decided 2026-10-08 with learning on.
  **Claude's recommendation:** keep it composite and add C2 to `M11.5`'s note (wording, at the next revision); the
  single-rule flip is the one already decided, and L0 adds no new one. Alternatives: record `M4.C.1` under `M10.C.5`
  for this cell; or change nothing.
  **Decided 2026-11-05:** the owner chose through `AskUserQuestion`, as recommended: keep it composite and add C2 to `M11.5`'s note. Wording, at the next revision.
  **Built 2026-11-05:** spec revision 49 (`docs/spec_revision_log.md`), wording in `M11.5` only; the engine hash stays `0b363cb3186c`.
- [x] T-191 ⟨later⟩ **`M11.C.11`'s return limb is carried by the removed member's two holds coming back on return
  (2026-11-04).** Under L0 it passes (`disturbance` +6.33 ± 1.6, 150 seeds). The one carrier,
  `removed-member-holds-not-released-on-return` (`M7.D.2a`'s symptom hold released while the member is away, and
  `M1.D.2a` (c)'s tie stock not returned by the reunion), takes it to -0.0171 ± 1.3. Releasing the symptom hold alone
  while away (`removed-member-hold-released-while-away`) leaves it passing (+1.51 ± 1.3, p 0.0055), so neither hold
  alone carries it. (The proposal above called the carrier "`M6.I.6`'s hold"; it is these two.) `M11.5`: mixed, the
  conservation limb premise on `M6.I.6`, the return limb composite, "no rule says the return is larger". Its premise
  limbs are not declared in `tools/version_1_status.premise_limbs`, so the audit judged it on its one readout.
  **`M11.1d`'s question:** do `M7.D.2a`'s hold or `M1.D.2a` (c)'s tie stock name that the return is the larger
  disturbance? Each says what a stock holds and when it is released or returned, not that the return is larger;
  the row needs both.
  **Claude's recommendation:** keep the return limb composite and name both holds in `M11.5`'s note (wording, at
  the next revision). Alternatives: class the return limb premise, reading the holds' return at once as what makes
  the return large; or change nothing.
  **Decided 2026-11-05:** the owner chose through `AskUserQuestion`, as recommended: keep the return limb composite and name both holds (`M7.D.2a`'s symptom hold and `M1.D.2a` (c)'s tie stock) in `M11.5`'s note. Wording, at the next revision.
  **Built 2026-11-05:** spec revision 49 (`docs/spec_revision_log.md`), wording in `M11.5` only; the engine hash stays `0b363cb3186c`.
- [x] T-192 ⟨later⟩ **`M11.C.41`'s two level cells are carried by the terms `M11.5` names (2026-11-04).** Under L0
  both pass (`[light: lower level]` `mean_acute` +128 ± 12, `reactive_over_chance` +0.0653; `[heavy: lower level]`
  +124 ± 14, +0.0677). Each is carried by C2, `mixing-weight-level-independent`, `level-blind`,
  `steepness-inverted` and `mixing-weight-inverted`. `M11.5`: mixed, resting on "`M4.C.1a`; `M4.D.1a` mixing
  weight", both limbs premise in the level cells. The audit judged every limb because the premise limbs are not
  declared (pending the owner, `docs/version_1_dispositions.md`, with `M11.C.34`).
  **`M11.1d`'s question:** the mixing weight reads functional level, so it names a lower level's higher reactive
  share; `steepness-inverted` (`M4.C.1a`) reverses the anxiety limb (`mean_acute` -17.8 and -25.2) and leaves the
  reactive share up. Both are rules `M11.5` already names for these cells.
  **Claude's recommendation:** no rule change; declare these premise limbs when the owner answers the pending
  question, after which the two cells have no composite limb and leave the audit's scope. Alternative: change
  nothing until that question is answered.
  **Decided 2026-11-05:** the owner chose through `AskUserQuestion`, as recommended: declare the premise limbs. In `M11.C.41`'s two level cells (`[light: lower level]`, `[heavy: lower level]`) `mean_acute` and `reactive_over_chance` are premise limbs, to be added to `docs/version_1_dispositions.md`'s premise-limb table; the heavier-stress cells' reactive share stays composite (the owner's reading of 2026-10-17). The two level cells then have no composite limb and leave the audit's scope. In the same call, on `M11.C.34`: no premise limb is declared; its one readout is the learned part (the owner's reading of 2026-10-17, borne out by T-173), the class stays mixed, and the question pending since version 1 closes with no table row. Needs a build: the table rows and the sentence above the table that calls the question the owner's, with whatever checks the dispositions file against the tag `version-1`.
  **Built 2026-11-05:** `docs/version_1_dispositions.md` is byte-identical to the tag (`tests/test_version_1_status.py`), so the owner chose (`AskUserQuestion`, as recommended) to put the rows in a second table of `docs/version_2_dispositions.md`, with the sentence closing the question; version 1's file and the tag test are unchanged. `tools/version_2_status.py`'s `premise_limbs` merges both tables (a criterion in both raises) and is read by the status page and `tools/emergence_audit.py`. A mixed row each of whose gating readouts is a declared premise limb now leaves the audit's scope, out by class; the two level cells left it (13 rows to 11), and `M11.C.41[light: lower level]` left the audit config's pilot rows. The owner chose to re-render `docs/emergence_audit_record.md` from the cache (`--render-only`, nothing run); the engine hash stays `0b363cb3186c` (Phase D `020160abb6db`). Tests: `tests/test_version_2_status.py::test_t192_v2_premise_limbs_parse_strictly`, `::test_t192_premise_limbs_merge_both_tables`, `tests/test_emergence_audit.py::test_t192_a_mixed_row_with_only_premise_limbs_leaves_scope`, each proved failing by mutation. Not changed: `M11.1g`'s wording says every passing mixed criterion is rerun and does not say that one with no composite limb leaves the scope; that sentence is for the spec revision that takes T-186 to T-191's wording. Written at spec revision 49 (2026-11-05).
- [x] T-193 ⟨later⟩ **`M11.C.7[joint over sided]` is carried by the landed contact and the coach's position, as
  `M11.5` says (2026-11-04).** Under L0 it passes (`deviation_fall` +0.0806 ± 0.045, p 0.00015, 150 seeds). Carriers:
  `landed-contact-changes-nothing` (`M1.E.7`) and `coach-position-blind` (`M8.3` made position-blind: a coach drawn
  onto one spouse's side still counts as a live position); no policy term. Under `coach-position-blind` the
  landed-contacts premise limb collapses too (+0.34 ± 0.84, p 0.19), so the deviation limb going red follows from
  the arms no longer differing. `M11.5`: mixed, `M8`'s predicate for the landed-contacts limb; the coach's going
  along with a `SPLIT` a declared premise of the sided arm (D29).
  **`M11.1d`'s question:** neither carrier names the ordering joint > sided; both remove what distinguishes the
  sided arm, which `M11.5` already declares premise.
  **Claude's recommendation:** no change; record the survivor as reviewed. Alternative: add the two carriers to
  `M11.5`'s note at the next revision.
  **Decided 2026-11-05:** the owner chose through `AskUserQuestion`, as recommended: no change; the survivor is recorded as reviewed, and the item closes.
