# Arising from the 2026-08-28 batch

- [ ] T-090 ⟨later⟩ **Audit the spec for requirements sourced from an extraction *heading* rather than a quoted sentence.** `M5.D.4` was inverted for exactly this reason: `docs/theory/ch13.md` carried the correct quotation in its body under a heading that said the opposite, and only the heading travelled into the spec. The heading is fixed and the requirement corrected, but **nothing has checked whether there are others of the same shape.** The check is mechanical: for each requirement citing a chapter, confirm the claim appears in a *quoted* passage in the extraction, not only in a section title or a bolded gloss. Highest risk in the earliest requirements, written when the extractions were newest.
- [ ] T-091 ⟨later⟩ **Two changes to the family-diagram application, both needing information the diagram does not currently hold.** Neither can be recovered afterwards, which is why `M13.3` names them as the ones to make first.
  - `M15.A.4` — export the **interval** a rater would defend, with the scale point defined, not a bare `3` on a 1–5 scale. The model divides by, differences and thresholds these values; a rank supports none of that.
  - `M15.C.2` — split the diagram's single **"distant"** line into *rupture* and *resolved low-contact*. `M1.B.3` calls telling those apart the most consequential discrimination in this part of the theory: same contact frequency, opposite bond energy. Until the app emits it, bond energy on any distant tie imports as a free range.
## Deferred from the 2026-08-28 re-sweep — graded below medium, backlogged rather than fixed

The re-sweep of the fix commit returned fifteen findings. Four high and seven medium were fixed; these four
were graded low and are recorded instead, because each fix is new unreviewed surface and the project's own
rule is to bound the loop rather than trade one defect class for another.

- [ ] T-092 ⟨later⟩ **`docs/theory/` is outside the reference guard, and has a dangling reference.**
  `family_evaluation/fe03.md` cites an `M0` requirement numbered **one past the last one §0 defines** (§0 runs
  `M0.1`–`M0.4`). `fe03.md` and `fe08.md` also cite `M1.E.7c`'s numbered sub-forms as though they were IDs —
  the spec's own uses of that notation were corrected at revision 5, the extractions were not. Adding the
  directory to `CITING_DOCS` turns the suite red until all three are fixed, so do both together.
  *(The literal tokens are deliberately not written out here: `TODO.md` is inside `CITING_DOCS`, so quoting a
  dangling ID as an example makes the guard fail on the note describing it. The guard is right to be strict —
  an exclusion mechanism would be the loophole that later hides a real one.)*
- [ ] T-093 ⟨later⟩ **`_REF` truncates compound references.** `M7.D.2c/2d` yields only `2c`; `M1.D.7i–l` only `7i`. Neither
  is currently dangling, so nothing is wrong today — but the scan is narrower than it reads, and
  `_STATUS.md` describes it without that qualifier.
- [ ] T-094 ⟨later⟩ **Name the stressor per criterion when the tests are written.** `M11.3` now places the obligation on the
  test rather than the criterion's prose, which is honest but defers it: 27 of 33 criteria are unclassified
  as discriminating or not. Classify them when each test is written, and consider a marking column then —
  `M11.3` originally required one and the table has no such column.

- [ ] T-095 ⟨later⟩ **Sweep the spec for other descriptions firmed up into mechanisms.** Two are now known — `M5.D.4`
  (anger, revision 4) and `M1.A.3` (the transition at 50, revision 7) — and both were caught by reading the
  primary, not the extraction. **Neither was catchable by a regex**, which is why this is a reading task and
  not a guard. The shape to look for: a requirement whose source is a *descriptive* passage, whose hedges
  ("begins", "a few", "tends to", "usually") the requirement dropped, and which the spec then states as a
  threshold, a switch, or an exact count. Highest risk where a single chapter is the only source.
- [ ] T-096 ⟨later⟩ **Re-read `KS05.2` on the capacity floor at 25.** `M1.A.4d` was regraded to a continuous falloff at
  revision 7 **on the principle rather than on a reading** — its source says "they lack the flexibility to
  make basic change", which is a description of a group and the same grammatical shape as Ch16's band. The
  falloff is applied; the *location* stands pending the primary. It is the one number `M1.A.3d` admits as
  structural, and it is admitted provisionally.
- [ ] T-097 ⟨later⟩ **Re-read the source behind `M5.F.2`'s `outside_ness` threshold.** "Below threshold the same move
  produces the opposite sign" is the same shape as the transition at 50 — a threshold on a continuous
  quantity, doing load-bearing work in the sign-flip argument at `M15.D.4`. It may be sound; it has not been
  checked with this question in mind.

## Backlogged from the corpus-fidelity sweep, 2026-08-28 — graded LOW, recorded not fixed

The sweep checked **every** quoted passage in the spec: 214 total, 160 verbatim in the corpus, 52 legitimate
self-quotes of project documents, 0 unexplained. Integrity and modal findings are fixed; these are the
hedge-level remainder, held back because each fix is new unreviewed surface.

- [ ] T-098 ⟨later⟩ **Ellipses that remove a qualifier.** `M1.A.19` drops "**would probably**" and a trailing condition
  ("as long as discussion touched emotional issues") from Ch10's neutral-third sentence. `M1.A.14c` drops
  "**provided his spouse functions in a reciprocal opposite way**", which is a condition on the whole claim.
  `M1.A.5d` splices two passages ~5,700 characters apart into one quotation and drops a "per se, **but…**"
  that states the harmful case. `M11.F.3a` truncates a Bowen epigraph with no ellipsis shown.
- [ ] T-099 ⟨later⟩ **Hedges trimmed off the front of quotations.** `M1.A.13a` ("**I believe** the most important cue"),
  `M1.D.7b` ("**I believe** a spectrum of problems"), `M1.A.18d` ("not observed it **fully**"),
  `M8.6a` ("**as many as** fifty or sixty" flattened to "fifty to sixty").
- [ ] T-100 ⟨later⟩ **Third parties quoted by an author, unmarked.** `M7.C.1c` rests on **Sylvia Nasar's** thought, quoted
  by Kerr. (`M4.D.5c`'s Andrew Solomon attribution is fixed; this one is the same shape.)
- [ ] T-101 ⟨later⟩ **A scope narrower than the requirement.** `M5.B.3a`'s band is stated of "**the alcoholic person**";
  `M1.B.10` quotes a taboo-set claim made of **spouses** and applies it to every tie; `M15.D.4`'s "they could
  do no wrong" comes from the inpatient schizophrenia project, the most severe families studied.
- [ ] T-102 ⟨later⟩ **`M4.D.3a`'s four-element evolutionary ordering has three sourced slots and one unsourced** —
  conflict's position between cooperation and dominant-adaptive is nowhere given in the source.
- [ ] T-103 ⟨later⟩ **`M1.A.5a` quotes a phrase absent from the book it cites** ("more mature, more responsible for self",
  cited to Kerr 2019 ch6; nearest analogue is in ch24 and says something different). Its decomposition is
  tagged as user-derived, so the model claim survives — the citation does not.
- [ ] T-104 ⟨later⟩ **`M2.A.0e`'s "almost identical" reads as Kerr-2019-grounded** and is in fact from KB Interview #14.
  Kerr 2019 says spouses marry at the "**same**" basic levels — stricter than the ±1 tolerance built on it.
- [ ] T-105 ⟨later⟩ **`M5.C.1a` borrows severity from an adjacent intervention class.** The three catastrophes are real and
  correctly sourced to *forcing conjoint contact*; the recorded consequence of the peace-agree/reactive
  mismatch itself is relational rupture. Careful drafting, suggestive juxtaposition.
- [ ] T-106 ⟨later⟩ **`M1.A.4d`'s source sentence is gate-shaped.** Kerr 2019: "**People above 25** on the continuum can
  make basic changes." Revision 7 replaced the gate with a continuous falloff on principle; the requirement
  should say plainly that the source's own wording is the gate it forbids.

- [ ] T-107 ⟨later⟩ **Run a cold sweep over this batch.** The 2026-08-28 sweep was warm — the same session that wrote the documents commissioned it. It found fifteen defects, which says the sweep works, not that the set is clean. The project's own rule is that a falling finding count from self-review is not a stopping condition, and that a pass which does not know the change's history changes the *distribution* of findings. `/csdp --cold-sweep` over `4e79697..HEAD` from a fresh session.
- [ ] T-108 ⟨later⟩ **The sweep covered one failure family only.** It reports itself as not covering: theoretical fidelity (it did not open `_LEDGER.md`, the `family_evaluation/` extractions, or Ch13's primary text), whether the ~88 new requirements are individually implementable, whether the specified model is coherent, and the reasoning in the DECISIONS document. **The `M5.D.4` correction rests on one re-read of one chapter by the context that made the correction** — a second reader on the sources is the highest-value follow-up.
- [ ] T-109 ⟨later⟩ **Decide whether `M16`'s renderer output format is worth fixing in the spec.** `M16.C.2` says what a rendered line must carry and points at the proposal's §4.2 table as the shape, but does not fix a format. That is deliberate for now — the format is cheaper to settle against a running Phase B than in advance — but it should be settled *before* a second consumer exists, or the two will drift.
- [ ] T-110 ⟨later⟩ **`M11.F.9(c)` deserves a worked demonstration before anyone relies on it.** The claim is that fitting one arm of a counterfactual to a known history breaks the error cancellation `M0.4` depends on. It is argued, not demonstrated. Phase E could show it directly: take a synthetic family, fit one arm to its own history, and measure how far the counterfactual moves against an unfitted control. If the effect is small the requirement is over-strong; if it is large it is the most important sentence in `M11.F`.
