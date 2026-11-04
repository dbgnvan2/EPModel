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
