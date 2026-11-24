# Spec revision 68 — revision 66's minor review items (T-261) (2026-11-23)

Built from T-261's plan of 2026-11-23 (Claude's recommendation, pending owner review; item 4, the cost charged at choice, approved by the owner 2026-11-24). Open:

- [x] T-268 ⟨now⟩ **Regenerate every record on revision 68's engine (2026-11-23).** The batch moved the engine hash
  from `9321b8523500` to `1ecd38073e84`, so every engine-keyed record is stale. No run's state changed, so each record
  is expected to match revision 67's; a verdict that moves is a defect to trace, not a finding. Mac Mini1's queue
  `~/regen68.sh` was started on 2026-11-23 at 08:07 (Mini1 time) at the revision 68 commit `3168392`, engine
  `1ecd38073e84`, in `~/ProjectsMini1/EPModel-regen68`, logging to `~/regen68.log`; when it ends, merge its records, remove each strict-xfail mark naming
  T-268, and report any move with T-254's seed-block rule.
  **Closed 2026-11-24:** all 12 steps exited 0 (queue done 02:57 Mini1 time); the records are merged and the
  eleven marks removed. `docs/version_3_revision_68_changes.md`: no verdict, effect or field changed in any compared
  record, and the other caches equal revision 67's entry for entry apart from their hashes.
- [ ] T-269 ⟨later⟩ **A divorce built before the run does not see a spouse who dies before its week (2026-11-23).**
  From revision 68's learning-qa pass. `divorce_events` checks that both spouses live in the state it is given; a
  scenario that builds the events at the start and a spouse who dies before `week` gives a `SEPARATION` that moves no
  one and loads addressed to the dead. For the owner: whether the engine should drop a divorce
  whose spouse has died, or a scenario should build the events at their week.
