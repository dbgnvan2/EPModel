# Spec revision 29 (cutoff and focus) — open after the build, 2026-10-14

Plan §4.8 holds the evidence.
- [ ] T-040 ⟨later⟩ **Focus steering has no measurable effect in Phase C** after the review made it a redistribution (share 0.578
  against 0.580 off, ad hoc, 20 seeds). Whether `focus_steer_gain` 1.0 is too small to matter is for the owner.
- Dismissed with reason (correctness review, 2026-10-14): `weakness_focus` reads the usual level after this week's
  `update_usual`. The symptom channels and `find_work` read it the same way at step 8 and 9, so it is consistent.
