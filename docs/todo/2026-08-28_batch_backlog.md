# Arising from the 2026-08-28 batch

- [x] T-090 ⟨later⟩ **Audit the spec for requirements sourced from an extraction *heading* rather than a quoted sentence.** `M5.D.4` was inverted for exactly this reason: `docs/theory/ch13.md` carried the correct quotation in its body under a heading that said the opposite, and only the heading travelled into the spec. The heading is fixed and the requirement corrected, but **nothing has checked whether there are others of the same shape.** The check is mechanical: for each requirement citing a chapter, confirm the claim appears in a *quoted* passage in the extraction, not only in a section title or a bolded gloss. Highest risk in the earliest requirements, written when the extractions were newest.
  *Closed 2026-11-17 (Claude's recommendation 2026-11-17, pending owner review): audited 2026-11-17 over the 23 requirements that cite one of the 22 papers (M1.A.3, M1.A.3c, M1.A.4d, M1.A.4h, M1.A.19, M1.C.1, M1.C.3c, M1.D.7a, M1.D.7b, M1.D.7f, M1.E.2, M5.B.3a, M5.B.3, M5.B.6, M5.D.4, M5.D.4a, M5.D.7a, M5.F.4, M7.A.2, M7.D.1a, M8.3, M14.A.4, M11.C.14). None takes its claim from a heading or a bolded gloss. The hedge findings are under T-095, T-098 and T-099.*
- [x] T-091 ⟨later⟩ **Two changes to the family-diagram application, both needing information the diagram does not currently hold.** Neither can be recovered afterwards, which is why `M13.3` names them as the ones to make first.
  - `M15.A.4` — export the **interval** a rater would defend, with the scale point defined, not a bare `3` on a 1–5 scale. The model divides by, differences and thresholds these values; a rank supports none of that.
  - `M15.C.2` — split the diagram's single **"distant"** line into *rupture* and *resolved low-contact*. `M1.B.3` calls telling those apart the most consequential discrimination in this part of the theory: same contact frequency, opposite bond energy. Until the app emits it, bond energy on any distant tie imports as a free range.
  ⟦batch: owner decisions (a rule, a criterion's meaning or a design choice)⟧ *(triage 2026-11-17, Claude's recommendation, pending owner review)*
  *Decision 2026-11-17 (Claude's recommendation 2026-11-17, pending owner review):* Closed here: both changes are to the family-diagram application, outside this repo. `M15.A.4` and `M15.C.2` already state the import contract, and until the app emits them those values import as free ranges (`M15.D`), which is the honest default.
## Deferred from the 2026-08-28 re-sweep — graded below medium, backlogged rather than fixed

The re-sweep of the fix commit returned fifteen findings. Four high and seven medium were fixed; these four
were graded low and are recorded instead, because each fix is new unreviewed surface and the project's own
rule is to bound the loop rather than trade one defect class for another.

- [x] T-092 ⟨now⟩ **`docs/theory/` is outside the reference guard, and has a dangling reference.**
  `family_evaluation/fe03.md` cites an `M0` requirement numbered **one past the last one §0 defines** (§0 runs
  `M0.1`–`M0.4`). `fe03.md` and `fe08.md` also cite `M1.E.7c`'s numbered sub-forms as though they were IDs —
  the spec's own uses of that notation were corrected at revision 5, the extractions were not. Adding the
  directory to `CITING_DOCS` turns the suite red until all three are fixed, so do both together.
  *(The literal tokens are deliberately not written out here: `TODO.md` is inside `CITING_DOCS`, so quoting a
  dangling ID as an example makes the guard fail on the note describing it. The guard is right to be strict —
  an exclusion mechanism would be the loophole that later hides a real one.)*
  *Closed 2026-11-17 (triage batch 2026-11-17, Claude's recommendation, pending owner review): the three citations repointed; `docs/theory/**/*.md` is in `CITING_DOCS`.*
- [x] T-093 ⟨now⟩ **`_REF` truncates compound references.** `M7.D.2c/2d` yields only `2c`; `M1.D.7i–l` only `7i`. Neither
  is currently dangling, so nothing is wrong today — but the scan is narrower than it reads, and
  `_STATUS.md` describes it without that qualifier.
  *Closed 2026-11-17 (triage batch 2026-11-17, Claude's recommendation, pending owner review): the scan expands `/b`, `/2d` and `–l` forms (`test_t093_compound_references_are_read_in_full`).*
- [x] T-094 ⟨later⟩ **Name the stressor per criterion when the tests are written.** `M11.3` now places the obligation on the
  test rather than the criterion's prose, which is honest but defers it: 27 of 33 criteria are unclassified
  as discriminating or not. Classify them when each test is written, and consider a marking column then —
  `M11.3` originally required one and the table has no such column.
  ⟦batch: owner decisions (a rule, a criterion's meaning or a design choice)⟧ *(triage 2026-11-17, Claude's recommendation, pending owner review)*
  *Decision 2026-11-17 (Claude's recommendation 2026-11-17, pending owner review):* Closed as superseded: `M11.5` now gives every criterion a class, and each criterion row declares its stressor; no separate marking column is needed.

- [x] T-095 ⟨later⟩ **Sweep the spec for other descriptions firmed up into mechanisms.** Two are now known — `M5.D.4`
  (anger, revision 4) and `M1.A.3` (the transition at 50, revision 7) — and both were caught by reading the
  primary, not the extraction. **Neither was catchable by a regex**, which is why this is a reading task and
  not a guard. The shape to look for: a requirement whose source is a *descriptive* passage, whose hedges
  ("begins", "a few", "tends to", "usually") the requirement dropped, and which the spec then states as a
  threshold, a switch, or an exact count. Highest risk where a single chapter is the only source.
  *re-read 2026-11-17 over the same 23 requirements as T-090. One remaining defect: `M5.B.3`'s third-party cost is a MUST, but Ch22 (`ch22.md:405`, L22.6) says only that removing a symptom without regard for the family "can" stir anxiety, and that a person's own contact increase carries no cost. Final revision: grade the cost `[D]` and say Ch22 states a possibility. The MUST's other source, external review G1, and the strict xfail stay.* *(Claude's recommendation 2026-11-17, pending owner review)* ⟦batch: final engine batch⟧
  *Done 2026-11-19 in spec revision 65 (the final engine batch), Claude's recommendation, pending owner review: `M5.B.3`'s cost cites G1 first and grades `L22.6` `[D]`; the strict xfail stays.*
- [x] T-096 ⟨later⟩ **Re-read `KS05.2` on the capacity floor at 25.** `M1.A.4d` was regraded to a continuous falloff at
  revision 7 **on the principle rather than on a reading** — its source says "they lack the flexibility to
  make basic change", which is a description of a group and the same grammatical shape as Ch16's band. The
  falloff is applied; the *location* stands pending the primary. It is the one number `M1.A.3d` admits as
  structural, and it is admitted provisionally.
  *re-read 2026-11-17 (`ks05.md:34-48` and the book). The 25 is Kerr's own number `[K]`, a group boundary ("25 or below on the continuum", about 20 percent, "based on observations to date"); it is not attributed to Bowen, and `FE04.2` repeats it. The ledger grade is right. Final revision: replace `M1.A.4d`'s "has not been re-read" and "pending" sentences with this reading, and drop its "same shape as Ch16" remark, since Kerr states the number. T-106 (in E1's revision) covers the gate wording.* *(Claude's recommendation 2026-11-17, pending owner review)* ⟦batch: final engine batch⟧
  *Done 2026-11-19 in spec revision 65 (the final engine batch), Claude's recommendation, pending owner review: `M1.A.4d` states the 25 as Kerr's own group boundary (`KS05.4`, `[K]`); the pending sentences are gone. The quoted phrases in this record's note are not in `ks05.md`; the spec quotes `KS05.4`'s own words.*
- [x] T-097 ⟨later⟩ **Re-read the source behind `M5.F.2`'s `outside_ness` threshold.** "Below threshold the same move
  produces the opposite sign" is the same shape as the transition at 50 — a threshold on a continuous
  quantity, doing load-bearing work in the sign-flip argument at `M15.D.4`. It may be sound; it has not been
  checked with this question in mind.
  *Closed 2026-11-17 (Claude's recommendation 2026-11-17, pending owner review): re-read 2026-11-17. `KS05.5` and `FE07.5` support the sign flip (a reactive position raises anxiety) and give no threshold. `M5.F.2c` (revision 39) already makes the flip continuous, crossing zero at a declared 0.5 `[I]`. The one leftover, `M15.D.4`'s "below threshold", is under T-101.*

## Backlogged from the corpus-fidelity sweep, 2026-08-28 — graded LOW, recorded not fixed

The sweep checked **every** quoted passage in the spec: 214 total, 160 verbatim in the corpus, 52 legitimate
self-quotes of project documents, 0 unexplained. Integrity and modal findings are fixed; these are the
hedge-level remainder, held back because each fix is new unreviewed surface.

- [x] T-098 ⟨later⟩ **Ellipses that remove a qualifier.** `M1.A.19` drops "**would probably**" and a trailing condition
  ("as long as discussion touched emotional issues") from Ch10's neutral-third sentence. `M1.A.14c` drops
  "**provided his spouse functions in a reciprocal opposite way**", which is a condition on the whole claim.
  `M1.A.5d` splices two passages ~5,700 characters apart into one quotation and drops a "per se, **but…**"
  that states the harmful case. `M11.F.3a` truncates a Bowen epigraph with no ellipsis shown.
  *re-read 2026-11-17; all four stand, one corrected. `M1.A.19`: restore Ch10's "would probably" and "as long as discussion touched emotional issues" (`ch10.md:459, 519`). `M1.A.14c`: restore "provided his spouse functions in a reciprocal opposite way" (`ks12.md:67`). `M1.A.5d`: the two spliced sentences are close together and the ellipsis shows the gap; the defects are the citation (`KS05.1`, not `KS14.2`) and the dropped ", but having more understanding…" clause. `M11.F.3a`: show the ellipsis and keep "somehow" and the tail of `KS24.8`'s sentence.* *(Claude's recommendation 2026-11-17, pending owner review)* ⟦batch: final engine batch⟧
  *Done 2026-11-19 in spec revision 65 (the final engine batch), Claude's recommendation, pending owner review: All four quotations restored or corrected in the spec.*
- [x] T-099 ⟨later⟩ **Hedges trimmed off the front of quotations.** `M1.A.13a` (the leading *I believe* hedge of Tape 6 T6.1 dropped; ASR, paraphrased 2026-11-17),
  `M1.D.7b` (*I believe a spectrum of problems*), `M1.A.18d` (*not observed it fully*),
  `M8.6a` (*as many as fifty or sixty* flattened to *fifty to sixty*).
  *re-read 2026-11-17. `M1.A.13a` is fixed (revision 34 paraphrases Tape 6 and keeps the hedge). Three stand: `M1.A.18d` should read *had not fully observed it in his own marriage until that day* (`KS16.6`); `M8.6a` turns `KS18.3`'s *as many as fifty or sixty* in a general conference layout into a headcount at the Medical College of Virginia and calls it *corroborated at scale* (drop the number and *at scale*); `M1.D.7b`'s two quotations are not in `ch18.md`; cite `KS14.1` or check them against the printed chapter.* *(Claude's recommendation 2026-11-17, pending owner review)* ⟦batch: final engine batch⟧
  *Done 2026-11-19 in spec revision 65 (the final engine batch), Claude's recommendation, pending owner review: `M1.A.18d` and `M8.6a` corrected; `M1.D.7b`'s first two quotations are replaced by a paraphrase of `ch18.md` and `KS14.1`'s own words. Its third quotation is not found in `ch18.md` either; that is a new item.*
- [x] T-100 ⟨later⟩ **Third parties quoted by an author, unmarked.** `M7.C.1c` rests on **Sylvia Nasar's** thought, quoted
  by Kerr. (`M4.D.5c`'s Andrew Solomon attribution is fixed; this one is the same shape.)
  Same revision: drop `M11.D.12`'s "Not built: it needs `M10.C.4b`'s checks module (T-244)", built 2026-11-17.
  ⟦batch: the next spec revision (wording only; ride with engine batch E1's revision)⟧ *(triage 2026-11-17, Claude's recommendation, pending owner review)*
  *Built 2026-11-17 in spec revision 64 (engine batch E1), Claude's recommendation, pending owner review; see CHANGELOG.md's revision 64 entry. `M7.C.1c` names the reading as Sylvia Nasar's, reported by Kerr; the quotation is unchanged.*
- [x] T-101 ⟨later⟩ **A scope narrower than the requirement.** `M5.B.3a`'s band is stated of *the alcoholic person*;
  `M1.B.10` quotes a taboo-set claim made of **spouses** and applies it to every tie; `M15.D.4`'s *they could
  do no wrong* comes from the inpatient schizophrenia project, the most severe families studied.
  *re-read 2026-11-17; all three stand, with two additions. `M5.B.3a`: say `L12.4`'s band is stated of the alcoholic and generalised here, and that KB08 concerns disclosure of self, which the spec extends to contact (from T-009). `M1.B.10`: the growth claim is Ch10's, of spouses (L10.8); the extraction of Tape 6 T6.4 does not hold *the start of the communication cutoff in a marriage*, so cite L10.8 and scope it to the marital tie or say the step is the spec's. `M15.D.4`: scope *they could do no wrong* to the inpatient schizophrenia families (`ch04.md:50`), and read *below the declared zero-crossing (`M5.F.2c`)* for *below threshold* (from T-097).* *(Claude's recommendation 2026-11-17, pending owner review)* ⟦batch: final engine batch⟧
  *Done 2026-11-19 in spec revision 65 (the final engine batch), Claude's recommendation, pending owner review: All three scoped as decided.*
- [x] T-102 ⟨later⟩ **`M4.D.3a`'s four-element evolutionary ordering has three sourced slots and one unsourced** —
  conflict's position between cooperation and dominant-adaptive is nowhere given in the source.
  *re-read 2026-11-17 (`KS06.1`). Kerr's example runs cooperation, then control and deference, which he calls dominant-adaptive and older; conflict is a second regression from cooperation, not ranked against dominant-adaptive. `ks06.md` is corrected. Final revision: grade conflict's slot in `M4.D.3a` `[I]`, as the spec already does for triangling.* *(Claude's recommendation 2026-11-17, pending owner review)* ⟦batch: final engine batch⟧
  *Done 2026-11-19 in spec revision 65 (the final engine batch), Claude's recommendation, pending owner review: Conflict's slot in `M4.D.3a` is graded `[I]`.*
- [x] T-103 ⟨later⟩ **`M1.A.5a` quotes a phrase absent from the book it cites** ("more mature, more responsible for self",
  cited to Kerr 2019 ch6; nearest analogue is in ch24 and says something different). Its decomposition is
  tagged as user-derived, so the model claim survives — the citation does not.
  *re-read 2026-11-17. The phrase is in neither `ks06.md` nor the book; the nearest, in the last chapter, is "being a more mature and responsible family member", about family-of-origin work. Final revision: drop the quotation marks in `M1.A.5a` and state the claim as the project's decomposition.* *(Claude's recommendation 2026-11-17, pending owner review)* ⟦batch: final engine batch⟧
  *Done 2026-11-19 in spec revision 65 (the final engine batch), Claude's recommendation, pending owner review: `M1.A.5a` states the claim as the project's decomposition, no quotation.*
- [x] T-104 ⟨later⟩ **`M2.A.0e`'s *almost identical* reads as Kerr-2019-grounded** and is in fact from KB Interview #14.
  Kerr 2019 says spouses marry at the "**same**" basic levels — stricter than the ±1 tolerance built on it.
  *re-read 2026-11-17. The near-match is the `KB14` interview's (K14.2, ASR); Kerr 2019 says *the same basic level* (`KS07.7`). Ledger U6 is corrected. Final revision: `M2.A.0e` cites KB14 for the near-match and `KS07.7` for the stricter claim, and grades the ±1 tolerance `[I]`, looser than Kerr.* *(Claude's recommendation 2026-11-17, pending owner review)* ⟦batch: final engine batch⟧
  *Done 2026-11-19 in spec revision 65 (the final engine batch), Claude's recommendation, pending owner review: `M2.A.0e` cites KB14 for the near-match and `KS07.7` for the same basic level, and grades ±1 `[I]`.*
- [x] T-105 ⟨later⟩ **`M5.C.1a` borrows severity from an adjacent intervention class.** The three catastrophes are real and
  correctly sourced to *forcing conjoint contact*; the recorded consequence of the peace-agree/reactive
  mismatch itself is relational rupture. Careful drafting, suggestive juxtaposition.
  *re-read 2026-11-17. `KB08`'s three outcomes are recorded for forcing conjoint contact on families at capacity; the peace-agree mismatch's own recorded result is rejection and ineffectiveness (`KB05` · K05.1, `KB06` · K06.4). The explainer row now grades the safety framing `[M]`. Final revision: `M5.C.1a` says the link from KB08's cases to this technique is the spec's inference.* *(Claude's recommendation 2026-11-17, pending owner review)* ⟦batch: final engine batch⟧
  *Done 2026-11-19 in spec revision 65 (the final engine batch), Claude's recommendation, pending owner review: `M5.C.1a` says the link from KB08's cases is the spec's inference.*
- [x] T-106 ⟨later⟩ **`M1.A.4d`'s source sentence is gate-shaped.** Kerr 2019: "**People above 25** on the continuum can
  make basic changes." Revision 7 replaced the gate with a continuous falloff on principle; the requirement
  should say plainly that the source's own wording is the gate it forbids.
  ⟦batch: the next spec revision (wording only; ride with engine batch E1's revision)⟧ *(triage 2026-11-17, Claude's recommendation, pending owner review)*
  *Built 2026-11-17 in spec revision 64 (engine batch E1), Claude's recommendation, pending owner review; see CHANGELOG.md's revision 64 entry.*

- [x] T-107 ⟨later⟩ **Run a cold sweep over this batch.** The 2026-08-28 sweep was warm — the same session that wrote the documents commissioned it. It found fifteen defects, which says the sweep works, not that the set is clean. The project's own rule is that a falling finding count from self-review is not a stopping condition, and that a pass which does not know the change's history changes the *distribution* of findings. `/csdp --cold-sweep` over `4e79697..HEAD` from a fresh session.
  *Closed 2026-11-17 (triage 2026-11-17, Claude's recommendation, pending owner review): superseded: the cold reviews since (stage 1–3 and pre-tag milestone reviews in `docs/cycles/`, the whole-repo review `bda211e`) cover the range.*
- [x] T-108 ⟨later⟩ **The sweep covered one failure family only.** It reports itself as not covering: theoretical fidelity (it did not open `_LEDGER.md`, the `family_evaluation/` extractions, or Ch13's primary text), whether the ~88 new requirements are individually implementable, whether the specified model is coherent, and the reasoning in the DECISIONS document. **The `M5.D.4` correction rests on one re-read of one chapter by the context that made the correction** — a second reader on the sources is the highest-value follow-up.
  *Closed 2026-11-17 (triage 2026-11-17, Claude's recommendation, pending owner review): superseded: the milestone reviews read the model theory-fidelity first without the build history; the stage 3 review re-read `M5.D.4a` (T-133).*
- [x] T-109 ⟨later⟩ **Decide whether `M16`'s renderer output format is worth fixing in the spec.** `M16.C.2` says what a rendered line must carry and points at the proposal's §4.2 table as the shape, but does not fix a format. That is deliberate for now — the format is cheaper to settle against a running Phase B than in advance — but it should be settled *before* a second consumer exists, or the two will drift.
  ⟦batch: owner decisions (a rule, a criterion's meaning or a design choice)⟧ *(triage 2026-11-17, Claude's recommendation, pending owner review)*
  *Decision 2026-11-17 (Claude's recommendation 2026-11-17, pending owner review):* Closed: there is one consumer of `M16`'s rendered lines and none planned; fix the format when a second one is planned, not before.
- [x] T-110 ⟨later⟩ **`M11.F.9(c)` deserves a worked demonstration before anyone relies on it.** The claim is that fitting one arm of a counterfactual to a known history breaks the error cancellation `M0.4` depends on. It is argued, not demonstrated. Phase E could show it directly: take a synthetic family, fit one arm to its own history, and measure how far the counterfactual moves against an unfitted control. If the effect is small the requirement is over-strong; if it is large it is the most important sentence in `M11.F`.
  *Closed 2026-11-17 (Claude's recommendation 2026-11-17, pending owner review): no tool or requirement fits an arm: `M15.D` runs an imported family as ranges, and `M11.F.9(c)` forbids fitting. The demonstration matters only if fitting is proposed, and it should be run then.*
- [x] T-255 ⟨later⟩ **`M1.D.7b`'s third quotation is not in its source (2026-11-19).** "appears to be functioning on a less differentiated emotional level than twenty-five years ago" is not in `ch18.md`; it appears only in `docs/theory/Extractions to be human validated.md`. Its second half ("may be related to the disappearance of land frontiers") is at `ch18.md:498`. Check it against the printed chapter, or paraphrase and cite `ch18.md:498`. Found by revision 65's T-099 fix. *Closed 2026-11-23 (Claude's recommendation, pending owner review): the quotation is in the printed chapter, FTCP Ch. 18 ¶63 (p. 487); the exact-phrase search missed it because the PDF text splits words. The same paragraph holds the two quotations T-099 removed, so `M1.D.7b` has them back, cited to ¶63, and `model_explainer.md` §6 cites ¶63.*
- [x] T-256 ⟨later⟩ **Lock-in's slowed leak has no bound (2026-11-19).** `M7.D.2e` multiplies an active channel's leak by `1 − min(1, s)`, and `s` peaks at `lock_in_gain × lock_in_turn ÷ 4` (0.15 now). At `lock_in_gain ≥ 4 ÷ lock_in_turn` the leak stops and a symptom lasts as long as lock-in holds, and `test_m7d2e_lock_in_lengthens_a_symptom_without_feeding_it`'s loop would not end. Validate the product in `EngineParams` or bound the test's loop before a sweep moves `lock_in_gain`. From revision 65's learning-qa pass. *Closed 2026-11-23 (Claude's recommendation, pending owner review): the test's loop is capped at 520 weeks and fails with a message (both arms end within 10 weeks at revision 67). Measured on the test's bearer: the symptom still ends at `lock_in_gain` 1.35, and never ends at 2.0 or above. `EngineParams` is not validated: that would move the engine hash, so it waits for an engine batch that moves `lock_in_gain`.*
- [x] T-257 ⟨later⟩ **`M5.D.3`'s abort-before-hold claim passes again at revision 65 (2026-11-19).** `test_m5d3_abort_is_the_usual_first_outcome` was a strict xfail from revision 41 (T-158, kept as a finding by the owner 2026-10-31); at revision 65 it reads 59 abort and 54 hold over 192 seeds (55 and 64 at revision 59), so its marker was removed. The margin is within seed noise: run fresh, disjoint seed blocks (as `tools/t253_seed_blocks.py` does for criteria) before the pass is read as the claim restored.
  *Closed 2026-11-23 (Claude's recommendation 2026-11-23, pending owner review): the claim is restored at revision 67; the test stays unmarked.* Five fresh, disjoint blocks of the test's 192 seeds (first seeds 50000 to 54000), run on Mac Mini1 with `tools/t257_seed_blocks.py` on engine `9321b8523500`, read 67/7, 64/8, 50/8, 58/8 and 59/1 abort/hold (`docs/review/2026-11-23_t257_seed_blocks.md`). Pooled: 298 abort, 32 hold, abort share 0.90; abort exceeds hold in 5 of 5 blocks; exact binomial p 1.6e-55, one-sided.
  The test's own seeds 0-191 read 54 abort and 9 hold at revision 67, against 59 and 54 at revision 65. Aborts barely moved; holds fell by five sixths, so the claim now passes because holding at the first opposition has become rare (about one opposition in ten), not because aborting rose. Measured on the same seeds on Mac Mini1 (clean checkouts `1987e49`, `6abe75a`, `0be453c`): 59/54 at revision 65, 49/14 at revision 66 and 54/9 at revision 67, so the fall came with revision 66's batch (reactive amount, conflict as consumption, the extra reactive acts of `M4.D.3c`); revision 67's change from 14 holds to 9 is within seed noise.
  For the owner: conviction (`M5.D.3b`, revision 67), meant to help a mover hold, did not bring holding back from revision 66's fall. The test's `held > 0` bound is now thin: one fresh block had a single hold, so a later engine change that removes holding would fail the test on that bound, not on the abort claim. The test's seed loop was split into `attempt_marks` and `first_outcomes` so the tool runs the test's own code; the test itself is unchanged.
