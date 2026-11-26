# Short rerun test (2026-11-25)

The owner's question of 2026-11-25: is the full rerun of every record after an engine batch (18 to 30 hours on Mac
Mini1) worth its cost, or would a short rerun catch the same real problems?

- [x] T-299 ⟨now⟩ **Measure a short rerun on revision 66 (2026-11-25).** Revision 64's engine (`429afd4`, the commit
  its queue and T-264 ran at; `0359119` holds its records but revision 66's engine) and revision 66's (`6abe75a`) ran
  at 1/5 of the seeds with `tools/short_rerun.py` in scratch checkouts (seven records, no emergence audit): revision 64
  on Mac Mini1, revision 66 on this Mac, 2026-11-26. The result is `docs/review/2026-11-26_short_rerun_test.md`: read
  against revision 64's full records, the short revision 66 run caught 9 of T-264's 10 real problems in 3.7 hours
  instead of 13; it missed `M11.C.12`'s borderline effect (p 0.04). 1/10 was not run: the 1/5 result was clear.
  The report's "Recommended mix of runs" is Claude's decision, pending owner review: after an engine batch the D2
  and Phase D mutation records run at 1/5 (`--seed-fraction 0.2`, `tools/seed_fraction.py`) and every other record
  at full seeds, without the emergence audit; full seeds and the audit at a milestone or before a report quotes a
  number. Revision 70 (T-296) ran under it. **Pending the owner:** the relayed brief asked for the mix to be written
  into `AGENTS.md` as an owner rule; the parent session wrote it there on 2026-11-26, and the owner accepted it the same day.
  Proposed line, for "Generated records and reviews": "After an engine batch, D2 and the Phase D mutation record
  rerun at `--seed-fraction 0.2` and every other record in full, without the emergence audit, on Mac Mini1 first,
  the MacBook Pro second (6 workers), this Mac only as overflow (at most 4 workers 00:00-07:30, 2 otherwise); full seeds and the audit at a milestone or before a report quotes a number (`docs/review/2026-11-26_short_rerun_test.md`)."
