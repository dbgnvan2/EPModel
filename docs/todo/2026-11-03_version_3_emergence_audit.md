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
- [ ] T-184 ⟨later⟩ **The emergence audit's cache records no NumPy version, so `tools/merge_cache.py` refuses it
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
