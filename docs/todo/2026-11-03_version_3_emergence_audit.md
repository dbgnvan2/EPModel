# Version 3, part 1: the emergence audit — 2026-11-03

The owner decided that version 3 adds two utilities: this emergence audit and a separate LLM-agent experiment
(planned in its own session). The audit reruns every passing composite and mixed criterion with learning from felt
relief removed, reports which results survive, and traces each survivor to the non-learning term that carries it.
The plan is `docs/plan_v3_emergence_audit.md`.

- [ ] T-182 ⟨later⟩ **The emergence audit: plan for approval (2026-11-03).** `docs/plan_v3_emergence_audit.md`,
  written on spec revision 47 (engine hash `0b363cb3186c`), status "draft for approval". Nothing is built. It sets
  five ablations (L0 all learning off; L1–L4 one learning path each), the scope (passing composite and mixed rows of
  the three ensemble records; premises and checks out; `M11.C.39` the positive control, `M11.C.3` the negative), the
  verdicts (survives, red, reversed), the carrier trace (L0 plus each policy term C1–C6 and the row's own mutants C7),
  the record (`docs/emergence_audit_record.md`, `tools/emergence_audit.py`, outside the engine hash), acceptance
  criteria EA-1 to EA-16, and a cost of about 6–17 h of machine time split across both Macs, run after T-181. Open
  until the owner answers the plan's §9 (Q1–Q7) and approves it; T-180 is the first row it answers.
