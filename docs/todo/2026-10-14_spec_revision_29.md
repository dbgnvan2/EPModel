# Spec revision 29 (cutoff and focus) — open after the build, 2026-10-14

Plan §4.8 holds the evidence.
- [x] T-040 ⟨later⟩ **Focus steering has no measurable effect in Phase C** after the review made it a redistribution (share 0.578
  against 0.580 off, ad hoc, 20 seeds). Whether `focus_steer_gain` 1.0 is too small to matter is for the owner.
  ⟦batch: owner decisions (a rule, a criterion's meaning or a design choice)⟧ *(triage 2026-11-17, Claude's recommendation, pending owner review)*
  *Decision 2026-11-17 (Claude's recommendation 2026-11-17, pending owner review):* Closed, gain kept: the corpus gives no size for focus steering, so raising `focus_steer_gain` until it shows an effect would tune to an outcome. A mechanism with no measurable effect at its `[I]` value is a result to report, which this record does.
- Dismissed with reason (correctness review, 2026-10-14): `weakness_focus` reads the usual level after this week's
  `update_usual`. The symptom channels and `find_work` read it the same way at step 8 and 9, so it is consistent.
