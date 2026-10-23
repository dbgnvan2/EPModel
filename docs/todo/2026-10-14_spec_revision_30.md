# Spec revision 30 (origin cutoff, danger, focus lock-in) — open after the build, 2026-10-14

Plan §4.10 holds the evidence.
- [ ] T-022 ⟨later⟩ **`M11.C.10`'s input barely differs between its arms** (correctness review, 3 seeds, 12 years, ad hoc). The
  origin cutoff of the first gen-3 child was 0.79 (open) against 0.87 (held), and 0.80 against 0.92 in the
  family-of-origin variant: parents already hold some rejection on some tie most weeks, and the held cutoff adds 0.08
  to 0.12. This limits what 200 seeds can detect.
- [ ] T-023 ⟨later⟩ **`M11.C.17` passes by a fragile margin, not by one mechanism** (ad hoc, 2026-10-14, the revision 29 worktree):
  switching off any one of the four revision 29 gains leaves the coached rise at about 0; switching off all four
  reproduces revision 28's +0.000881 ± 0.000828.
- [ ] T-024 ⟨later⟩ **Minor, from the fix re-sweep (2026-10-14):** `update_danger`'s docstring puts the dead-person sentence in the
  middle of the paragraph on cost (move it to the end; left during the regeneration so the engine hash would not move);
  `arm_c47` reads only people with a tie to the person, so a removed tie would drop someone from both readouts
  silently (not possible on the fixture today).
- [ ] T-025 ⟨later⟩ **Adjacent, not fixed (2026-10-19):** the ad hoc probes in `tools/probes.py` (lines 139, 141, 268, 458) still
  take float means with the builtin `sum`. No record reads them; move them to `math.fsum` when the file is next
  touched.
- [ ] T-026 ⟨later⟩ **Adjacent, not fixed (stage 1 review, 2026-10-19):** the Phase D ensembles (`tools/phase_d_record.py`) run
  with no timeout, and `tools/diagnostic_record.py:136` and `tools/level_occupancy.py:145` stop when their child
  fails, where the Phase C records keep the rows and go on (S1.5a). Both run once per regeneration; give them the
  Phase C handling when either is next touched. *The cold sweep (2026-10-20) added to `tools/level_occupancy.py` a
  check that the child printed one row for each of C.16's arms, so a short record is no longer written; a failed
  child still stops it.*
- [ ] T-027 ⟨later⟩ **Adjacent, not fixed (stage 1 review, 2026-10-19):** the act-literal scan, widened to lists, tuples and one-act
  comparisons, finds act names in `src/bowen/ensemble/criteria.py`, `criteria_d.py` and `criteria_d2.py` (the acts
  each criterion reads), as well as `NEVER_REFUSED` (`rejection.py`) and `ABORT_ACTS` (`iposition.py`). Rule 9 asks
  whether each belongs in config; the criteria are definitions, not engine rules, and are for a later stage.
- [x] T-028 ⟨later⟩ **Owner: freeze `docs/version_2_stage_1_changes.md` at stage 1** (`/csdp` sweep, step 8, 2026-10-20). The page
  compares the tag `version-1` with the records as they are now, so a later regeneration fails
  `test_s1_10b_changes_page_is_current`, and rerunning `tools/record_changes.py` would rewrite the page under stage
  1's title with the later stage's changes. Freezing it needs a ref to read stage 1's records from, such as a tag at
  the stage 1 commit (the owner chose the tag `version-1`); then the tool reads both sides from git and the test
  stays true. Until then: do not rerun the tool after stage 1. *Done 2026-10-23 with revision 36's regeneration,
  decided as recommended under the owner's instruction of 2026-10-22, pending the owner's review: the tool now reads
  both sides from git refs — the tag `version-1` and stage 1's regeneration commit `7aa9fc5` (a commit, since tags
  are the owner's) — so the page is frozen without a new tag, and a second page,
  `docs/version_2_revision_36_changes.md`, compares `7aa9fc5` with revision 36's records.*
- [ ] T-029 ⟨later⟩ **Minor, from the `/csdp` sweep (step 8, 2026-10-20), not fixed:** (1) a stop signal while
  `tools/mutant_runner.py:run_in_copy` copies the repository, or while its temporary directory is removed, leaves a
  `bowen-mutant-*` directory in `$TMPDIR` (no process is left; the handlers are set only around the child), as does
  a late stop passed on to the default handler, which ends the process at once (`_stop_signals_raise`'s docstring);
  (2) two runs of one record tool at once overwrite each other's cache file (`tools/record_cache.py`), the last save
  winning; (3) the timed-out path's second `communicate()` in `tools/child_process.py` has no timeout, which would hang
  only if a worker left the child's process group, and none calls `setsid`.
- [ ] T-030 ⟨later⟩ **Adjacent, not fixed (`/csdp` fix re-sweep, 2026-10-20):** the generated records are written in place with
  `write_text` (`tools/ensemble_record.py:211`, `tools/level_occupancy.py:148` and the other record tools), and
  later tools read them (`mutation_record.passing`, `sweep_record.central`, `phase_d_mutation_record.passing`). A
  kill mid-write fails the next tool loudly and costs a rerun of minutes, so this is lower priority than the cache
  (now written whole and moved into place); give them the same write when any is next touched.
- [ ] T-031 ⟨later⟩ **Adjacent, not fixed (cold sweep, 2026-10-20, C3):** `code_hash` (`tools/ensemble_record.py:54-60`) hashes
  each `config/bowen/*.md` file as raw bytes; only `constants_changes.md` is skipped. Rewording a config file's prose,
  such as the text under the table in `config/bowen/act_classes.md`, changes the engine hash, so every record reruns
  (P38). The Python files are hashed without comments and docstrings; the config files could be hashed by what the
  loader reads from them. Older than stage 1 (PLAUSIBLE: no such edit has been made yet).
- [ ] T-032 ⟨later⟩ **Adjacent, not fixed (cold sweep, 2026-10-20, C4):** the worker pools that `tools/phase_d_record.py` and
  `tools/ensemble_record.py` start in their own process (`src/bowen/ensemble/runner.py:116`, a
  `ProcessPoolExecutor`) are not killed as a group when the parent gets a stop signal, as `tools/child_process.py`
  does for a mutant child (S1.5a), so after a SIGTERM to the parent, workers can run on until their current task
  ends. Older than stage 1 (PLAUSIBLE: not reproduced).
- [ ] T-033 ⟨later⟩ **Owner: a row that raises on every run is never cached** (cold sweep, 2026-10-20, after the C1 fix). A RAISED
  row is a red under `tools/mutation_record.py`'s `judge` (the criterion can no longer pass), and the sweep and
  diagnostic records print its error as its result, but `tools/record_cache.py` caches no errored row. So a mutant,
  sweep setting or diagnostic variant that makes the engine raise every time leaves its record stale for good: the
  tool now exits 1 and says so, and the record's test fails. No committed record holds a
  RAISED or BROKEN row (checked 2026-10-20). Caching a RAISED row whose cause is the model, not the machine, would
  let such a record be committed; what counts as evidence is the owner's choice.
- [ ] T-034 ⟨later⟩ **Minor, from the document-claim tests' fix re-sweep (2026-10-20), not fixed** (`docs/plan_doc_claim_tests.md`;
  recorded under the owner's three-or-fewer rule). (1) D5b's `classify` (`tests/test_doc_status.py`) reads "Yet to be
  approved", "To be decided by the owner", "Not yet fully approved", "Not formally approved" and "Isn't approved" as
  settled: a negator is matched only when followed by fillers and then a settled word. A document whose own status says
  "To be decided" would agree with a checklist label "Decided" and nothing would be reported. No status in the repo
  reads that way today. Fix: add "to be" and "yet to be" to the open list, and adverbs and "isn't" to the negation lists
  in `tests/doc_claims.json`, with classification cases. (2) D2's roots (`tests/test_doc_claims.py` `path_scan`) are
  the five project folders and the top-level folders git tracks, so a path under a whole top-level folder that was
  deleted (`papers/x.pdf` after `papers/` is gone) is not read and not reported. Fix: take root candidates from the
  folders the documents name as well.
- [ ] T-035 ⟨later⟩ **`M11.C.47`: the dangerous person is cut off on fewer weeks** (−16.6 ± 4.9 `held_weeks`, 150 seeds), though felt
  danger and the principled part are higher. Trace why before reading the premise as a finding about cutoff.
- [ ] T-036 ⟨later⟩ **`M11.C.2` on the twelve-person family fails again at revision 30** (+0.70 ± 1.5, p 0.16; +1.31 ± 0.95 at
  revision 29). *Traced 2026-10-15, each revision 30 change removed alone:* unmodified +0.70 ± 1.5, p 0.16 (reproduces the record); `M4.D.1h`'s force removed +0.70 ± 0.97, p 0.13; `M1.B.8b`'s compounding removed +1.12 ± 1.7, p 0.09; `M1.B.8b`'s reinforcement removed +1.11 ± 1.6, p 0.093; `M1.D.8a` removed +0.96 ± 1.5, p 0.14; `M1.B.3e`'s danger removed +0.08 ± 1.8, p 0.46 (`symptom_load`, seeds 0–149, one-sided signed-rank with the runner's margin; ad hoc, 2026-10-15). No single removal brings
  the pass back. Removing the force gives revision 29's spread with revision 30's mean (the two means agree to three
  figures with different spreads; measured, not explained); removing either focus term gives back most of the mean
  with the wider spread. Next, if wanted: force and both focus terms removed together, about 10 minutes. Under the
  version 1 definition this composite can also be accepted as a finding.
- [ ] T-037 ⟨later⟩ **Phase D mutation record, three minor items** (re-review of 7c8a0fb, 2026-10-15): (1) ~~`engine_hash` reads
  `inspect.getsource` of `apply` and `run_in_copy`, comments included, so a docstring edit there reruns the record~~
  *(fixed at version 2 stage 1, rider (iv): read through `source_text`)*;
  (2) `build()` does not flag an `AD_HOC` entry that now has a mutant or no longer passes (only the test checks the
  first); (3) the spec's `M11.C.10` row still opens with the revision 11 and revision 29 mutation clauses, superseded by
  the later amendments but not struck (item 3 is in revision 31's plan, approved in the review 2026-10-16).
- [ ] T-038 ⟨later⟩ **Seven inconsistencies in the spec found while drafting the review form** (listed in
  `docs/REVIEW_spec_revisions_24-30.md`, "Inconsistencies found while drafting"): `channel_focus` "from belief" (line
  1630) against `M1.A.11c`; `tolerance` Family field (line 1679) against per person (`M7.D.4`); `TRIANGLE`'s channel in
  the B8 plan against `attribution.md`; `M7.E.1c` "open owner decision" (line 1084) against plan line 1355; `M11.C.2`
  settings "to be added" (line 1314) though present; criteria settings ungraded; revision 28 tags with no source. Fix
  with the review's revision, so the spec moves once. *2026-10-16:* all seven are in revision 31's plan (§4.11); the
  `TRIANGLE` one needs no spec change (the owner approved the config as built).
- [ ] T-039 ⟨later⟩ **rca: record findings and commit review fixes separately** (rca audit, 2026-10-14): with fixes folded into the
  feature commit, the audit cannot run a reproducer at the parent. Have csdp write `.rca/findings.md`.
- Dismissed with reason (correctness review, 2026-10-14): intrusion is read by `tie_change` after the week's move
  effects and landed contacts, not on the state appraisal saw. Both read the same function; the spec now says which
  state.
