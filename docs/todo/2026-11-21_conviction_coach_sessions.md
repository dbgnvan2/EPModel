# Spec revision 67 — conviction, and the coach acting only in sessions (2026-11-21)

Built from `docs/proposal_conviction_and_coach_sessions.md`, approved by the owner 2026-11-21. Open:

- [ ] T-266 ⟨now⟩ **Regenerate every record on revision 67's engine (2026-11-21).** The batch moved the engine hash
  from `4aaa9029f63a`, so every engine-keyed record (ensemble, mutation, sweep, level occupancy, diagnostic, Phase D
  parts I and II, Phase D mutation, emergence audit, sensitivity, group G readout, T-176 trace) is stale. Mac Mini1's
  queue `~/regen67.sh` was started on 2026-11-21 at the revision 67 commit; when it ends, merge its records here,
  remove each strict-xfail mark naming T-266 as its record is regenerated, and report each verdict move to the owner
  with T-254's seed-block rule. The LLM experiment is frozen at revision 64 (T-265) and is not in the queue.
