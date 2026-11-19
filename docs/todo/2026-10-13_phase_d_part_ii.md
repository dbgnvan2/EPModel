# Phase D part II — open after the build, 2026-10-13

Plan §4.7 and `docs/archive/phase_d2_status_2026-10-13.md` hold the evidence.
- [x] T-041 ⟨later⟩ **Owner: the ±2% band against `M11.C.6`'s traversal limit (a)**, and development as `M7.E.1f`'s rule against
  the estimator bounded to the band. `M11.C.6` measured +0.018 ± 0.028 under a band of about ±0.8 points.
  *Closed 2026-11-17 (triage 2026-11-17, Claude's recommendation, pending owner review): superseded: revision 31 replaced the flat band (`M7.E.1f`); `M11.C.6`'s FAIL accepted as a finding 2026-10-18.*
- [x] T-042 ⟨later⟩ **Owner: `M11.C.10`.** The declared cutoff ends within about 2 years, and the uncut baseline is cut within weeks,
  so the arms run the same process. Witnessed-cutoff learning and origin cutoff as an input to intensity (`M1.D.8`)
  have no mechanism. The act count is dominated by generation 3 cutting their own parents (about 775 per run). Hold
  the cutoff? Read tie states? Build link two or three? (plan §4.7)
  *Closed 2026-11-17 (triage 2026-11-17, Claude's recommendation, pending owner review): superseded: the revision-29 amendment holds the cutoff and reads tie states; `M1.D.8` and `M4.D.6f` are built; `M11.C.10` passes.*
- [x] T-043 ⟨later⟩ **Owner: `M11.C.2` on the twelve-person family.** Projection is spread evenly (about 80 acts per child). Chronic
  anxiety takes only the year at 11, and the band fixes the level. Late load follows age (Leo 24.9, Pia 15.4). Read
  load over matched ages? Make chronic anxiety integrate childhood? Why does focus not settle on one child?
  (plan §4.7)
  *Closed 2026-11-17 (triage 2026-11-17, Claude's recommendation, pending owner review): superseded: the by-age reading is adopted for `M11.C.2` and its twelve-person row passes; choosing the child is T-195 and T-197.*
- [x] T-044 ⟨later⟩ **Chronic anxiety ratchets across generations.** A child's chronic anxiety is derived from the living parents'
  values and its own witnessed history and then fixed at 12 (`M7.B.1a`); nothing in the run lowers it again for later
  generations. Check over several generations whether it only rises; the 40-year run is too short to see it.
  *adopt. Confirmed by construction: `M7.B.1a` adds `witnessed_gain` × a load that is never negative to the parents' mean, and `M7.B.1` lets the value fall only when `basic_level` rises, so a line's chronic anxiety can only climb. That contradicts `M7.E.1`'s direction that a child outside the process comes out better. **Build note:** read the childhood load term signed, as the child's yearly mean load minus the parents' mean load over the same years, so a child who carries less than its parents ends below their mean. No new constant. Guard: a two-arm test (focused vs unfocused child) with the unfocused child below the parents' mean.* *(Claude's recommendation 2026-11-17, pending owner review)* ⟦batch: final engine batch⟧
  *Done 2026-11-19 in spec revision 65 (the final engine batch), Claude's recommendation, pending owner review: `M7.B.1a`'s childhood load term is signed (the child's yearly mean load minus the living parents'), so a line's chronic anxiety can fall. Guard: `test_m7b1a_t044_a_child_who_carries_less_than_its_parents_ends_below_their_mean`, mutation-proved.*
- [x] T-045 ⟨later⟩ **Starting ages as settings in the fam-diag app.** The owner asked for them as settings in the app; they are
  config settings (`M2.A.0j`) and the app was not touched (standing instruction). Deferred.
  ⟦batch: owner decisions (a rule, a criterion's meaning or a design choice)⟧ *(triage 2026-11-17, Claude's recommendation, pending owner review)*
  *Decision 2026-11-17 (Claude's recommendation 2026-11-17, pending owner review):* Closed: the starting ages are config settings (`M2.A.0j`) and the app is outside this repo by standing instruction; the request belongs in the app's own backlog.
