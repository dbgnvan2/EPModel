# Spec revision 69 — a divorce is built in its week (T-269) (2026-11-24)

Built from T-269 (Claude's recommendation 2026-11-24, pending owner review). Open:

- [ ] T-294 ⟨now⟩ **Regenerate every record on revision 69's engine (2026-11-24).** The batch moved the engine hash
  from `1ecd38073e84` to `c9b45e586f6d` (every `src/bowen/` file is hashed, the scenario layer too), so every
  engine-keyed record is stale. No run schedules a divorce, so each record is expected to match revision 68's; a
  verdict that moves is a defect to trace, not a finding. Mac Mini1's queue `~/regen69.sh` runs at the revision 69
  commit in `~/ProjectsMini1/EPModel-regen69`, logging to `~/regen69.log`; when it ends, merge its records, remove each
  strict-xfail mark naming T-294, and report any move with T-254's seed-block rule.
- [ ] T-295 ⟨later⟩ **Revision 69's minor review items (2026-11-24).** From the batch's learning-qa and correctness
  passes, none medium in today's code since no runner uses `DivorceSource`: (1) a dropped divorce reaches the run log
  only through `DivorceSource.dropped` and `summary()`, which the first runner that schedules a divorce must report
  ("N of M"), or the source needs a log record; (2) the wrapper has no `.rules` or `.ticks`, so a tool reading those
  from its source (`tools/phase_c_trace.py:51`) fails on it; (3) calling `scheduled` twice for one tick records a drop
  twice.
