# Spec revision log

**Status: the record of every revision of `bowen_agent_model_spec_v2.md`, 4 to 35. Moved out of the spec at revision 35, 2026-10-22; nothing a run computes changed.**

The spec holds the current rules; this file holds how they got there. Until revision 34 the revision sections stood at the end of the spec, under `M17`, and the front matter's `status:` line listed every revision since 10. `docs/plan_model_ergonomics.md` Part E moved both here (the owner's answer to its §8 Q2, 2026-10-22, "as revision 35"), so a session reading the spec reads 480 fewer lines of history. The sections below are the spec's own text as it stood, tags and all, with the newest first; revisions 13 to 33 were written inline in the requirement bodies and have no section of their own, which Part E's second step records under revision 35.

### Revision 35 — the revision history moved out of the spec, 2026-10-22 ⟦rev35 · owner decision 2026-10-22, plan_model_ergonomics.md §8 Q2⟧

**Decided 2026-10-22** by the owner: the front matter's status text and the revision sections `### Revision 34` to `### Revision 5` leave the spec for this file, and the spec takes revision 35 for it, wording only. No rule changed; `engine_hash()` is unchanged and no record was rerun. The front matter's `status:` line now reads `APPROVED` with a pointer here; its text as it stood is the last section of this file.

### Revision 34 — the ASR quotations paraphrased, and the stage 1 review's wording fixes, 2026-10-22 ⟦rev34 · owner decisions 2026-10-21 and 2026-10-22⟧

**Approved 2026-10-22** by the owner: the §4 texts of `docs/PROPOSAL — ASR QUOTATIONS (SP-1).md`, as written, and its eight questions, each as recommended; the stage 1 milestone review's wording fixes were decided at its triage on 2026-10-21 (D2–D6, `CHANGELOG.md`). Nothing a run computes changed, so no record was rerun. Revisions 13 to 33 are recorded in the front matter's status line and in `CHANGELOG.md`, not here.

| Item | Spec change |
|---|---|
| SP-1 | The 1979 tapes and the Kerr–Bowen interviews are ASR transcripts, cited and not quoted. The spec's quotations of them, in the proposal's 47 spec entries (SP1-S01 to SP1-S47), are replaced by the approved paraphrases, each with its citation. The KB01 interview is credited to Bowen by name; the Hijack-set interviews (KB02–KB15), including KB03 and KB04, are cited as the interview (Q5). Written-source quotations stay |
| Q3 | `M11.F.3` restated from *Family Evaluation*'s Epilogue (FE11.2, Bowen's own writing) in place of the KB13 interview; it now records three statements, the third being that Bowen declined the debate (alternative B). A requirement change, approved |
| Q8 | `M4.D`'s *Family Evaluation* quotation is kept and traced to a new finding, `family_evaluation/fe03.md` · FE03.21; the revision 9 entry below cites it |
| MR-10 | `M11.4` drops its count ("Three criteria"); the table is the list (`M11.4b`) |
| MR-12 (D4) | `M8`'s preamble, `M8.5` and `M13.1`: three mechanisms call the predicate; `M5.C`'s `avoidance_available` is defined and tested but not called. `M13.1` stays met |
| MR-13 (D6) | `M2.A.0i` records Dr Halim's declaration (revision 28); Teodor's row drops "awaiting owner review" |
| MR-15 | The front matter's closing clause, "no code until the implementation plan is also approved", is deleted: it was written before the first phase was planned, and Phases B, C and D and version 2's stage 1 have been built since |
| D3 | `M11.1`'s scope is the `M11.C` criteria |

### Revision 12 — the Phase C prerequisites, 2026-10-07 ⟦rev12 · owner decision 2026-10-07⟧

**Approved 2026-10-07** by the owner: every recommendation of `docs/implementation_plan_phase_c.md` §1 and §2, as written. This section records the spec changes from §1's P1–P10 and P12. P11 (closing Phase B with the owner's read of the rendered traces) is an action, not text. §2's build decisions D1–D10 live in the plan, not here; they choose forms the spec leaves open.

| Item | Spec change |
|---|---|
| P1 | `M6.I.6` restated: transfers balance, every change is logged against a named source, sink or transfer. New `M6.4`, the stock-and-flow table. `M6.I.1` amended: the budget is reduced only by a completed differentiating exchange. `M4.G.2a`'s hold-back ends with Phase C |
| P2 | New `M4.B.2a`, `M4.B.2` row by row. `M5.F.1` amended: `outside_ness` acts through the event. `M5.C`'s system-calm row annotated |
| P2a | `M9.8`'s belief store built in Phase C |
| P3 | `M11.C.17`, `M11.C.18` and `M11.C.20` moved to Phase D. `M7.D.1` built in Phase C, channels fixed at their constitutional prior until Phase D. `M11.C.4`'s nodal event scripted in Phase C |
| P4 | `M5.D.6`'s next day is the next fast tick. New `M5.D.9`: a multi-tick sequence and one outcome per tick. Sub-week explainer durations are not calibration targets |
| P5 | Explainer §17.3 and `_STATUS.md` corrected to §17.1's form: cancellation holds only for error that does not interact with the intervention |
| P6 | Six explainer passages brought into line (capacity floor, the transition at 50, the `systems_perspective` ceiling, symptom exclusivity, §11 test 9, "unbounded") |
| P7 | `M11.D.16`'s symmetric-family test holds in distribution, not byte for byte |
| P8 | `M1.F.2`: the sign comes from the event's components relative to the receiver's optimum; no hard-coded flip |
| P9 | `M5.C`'s `outside_ness` gate has one threshold per axis. `M5.D.5`'s mutual rise is solid self. `M5.B.3` reconciled with `M5.B.3a`, and `REDUCE_CUTOFF`'s cost lands on the third party (`L22.6`). `M11.C.27` listed at `M11.4`. `M11.C.38` sweeps `chronic_anxiety`'s initial condition and is built although a SHOULD |
| P10 | New `M5.B.6`, `PROVOKE`. Bowen's two falsifiers (`L09.2`, `L16.2`), `L15.1`'s delayed channel, `L03.4`'s two-rung ladder and `L03.5`'s illness in the mover are recorded for the Phase D plan, not entered here |
| P12 | The paper material held in `TODO.md` under the stale "revision 11" label is relabelled "held for a later revision" |

**Phase C now gates 19 acceptance criteria.** Of them, by `M11.5`: 6 composite, 10 premise, 2 check, 1 mixed. Phase D gains `M11.C.17`, `M11.C.18` and `M11.C.20`.

---

### Revision 11 — what is set, what is random, what emerges, 2026-10-06 ⟦rev11 · owner decision 2026-10-06⟧

**Approved 2026-10-06** by the owner: A1 (decided first), A2 with A2.1, A3 option (a), A4, A5 and A6 of `docs/proposal_rev11_emergence.md`. Every requirement this revision adds or amends carries a ⟦rev11 · …⟧ marker naming the item. The rationale is in `model_explainer.md` §19; the proposal is the working record.

**Why.** The owner's concern: a model told how families behave will behave that way and can tell us nothing. Read against `M11.1d`, 2 of the 41 criteria at revision 10 could produce a result no single rule states. The principle adopted has three layers. **Set:** two given drives — toward connection, and away from being smothered, controlled or losing autonomy — and what each act does to the state of the system. **Random:** exogenous events, selection noise, initial conditions, visibility. **Emergent:** the patterns Bowen recorded — conflict, over- and underfunctioning, distance, cutoff, projection, fixed triangles — produced by learning from felt relief, by agents that do not know the consequences of what they do.

| Item | What changed |
|---|---|
| A1 | `M4.D.6a` reversed: the automatic channel learns from felt relief over a short horizon. `M4.D.6b` (horizon swept with learning rate and temperature; the long-horizon requirement withdrawn), `M4.D.6c`, `M4.D.1a`'s note, `M4.G.3` (its condition is now met, so it binds) and `M11.C.16` (restated) follow from it |
| A2, A2.1 | `M4.C.1` re-derived as two-sided deviation from a felt contact optimum, with `M4.C.1a`–`M4.C.1c`; the standing load (`M4.A.1`) becomes the "too little" side of the same function; `M3.D.2` annotated; `M11.C.27`'s mutation target moved off the prior-state term |
| A3 | `M5.A.1a`: each core move is one act; the pattern of the same name is a readout |
| A4 | `M1.C.2`, `M1.C.3`, `M1.C.4`, `M4.D.3`, `M4.G.1`, `M7.E.1`, `M7.E.1c` amended to make their result emergent; `M5.E.0` makes the change-back ladder a learned response; `M7.D.3` and `M7.E.3` **withdrawn as mechanisms**, kept as assertions. Consequential: `M7.E.1a` becomes a shape check on the output and `M7.E.1e` is met through the mechanism, because a child's level now comes from the estimator. Criteria `M11.C.6`, `M11.C.10`, `M11.C.12` and `M11.C.15` have new mutation targets; `M11.C.42`, `M11.C.44` and `M11.C.45` are new |
| A5 | `M4.C.10`: contact with a calmer person lowers anxiety, on every tie; seeking it is learned, not a move. `M1.E.5` annotated. New `M11.C.43`. The term *social buffering* is outside the corpus and is not cited until sourced |
| A6 | `M4.B.3`: the policy holds no representation of consequences beyond the learning horizon |
| Classification | `M11.5` classifies every criterion as premise, composite, mixed or check, by reading; `M10.C.5` fills the register's premise column from it. `M13` gates the four new criteria |

**Corrections to the proposal made while applying it.** `M11.C.41`'s reactive-share limb stays a premise, not a composite as the proposal said, because amended `M4.D.3` keeps anxiety's effect on the reactive share through `M4.D.1a`'s mixing weight. `M7.E.1a` and `M7.E.1e` were not named in the proposal and had to change with `M7.E.1`. The proposal quoted `K10.7` in support of A1; the spec already flags that quotation as societal in scope, so `M4.D.6a` cites `KS04.13` instead.

**Phase B code that no longer conforms.** Phase B was built against revision 10, and these parts of it now differ from the spec. None was changed at this revision. Each is a Phase C plan item:

| Code | Revision-10 rule it implements | Revision-11 rule |
|---|---|---|
| `src/bowen/engine/appraise_base.py` | `M4.C.1`'s base product | `M4.C.1`–`M4.C.1b` |
| `src/bowen/engine/standing_load.py` | `M4.A.1` as bond energy over functional level | `M4.C.1c` |
| `src/bowen/engine/recompute.py`, the `tension_activation_threshold` test of triangle activity, and `tests/bowen/test_mechanisms.py::test_m1c3_triangles_are_inoperative_when_calm` | `M1.C.3`'s calm-system rule | `M1.C.3` amended: activity is a readout of recent `TRIANGLE` acts |
| `config/bowen/constants.md` and `config/bowen/constants_frozen.md`, the `tension_activation_threshold` row | cites `M1.C.3` as its source | the constant goes with the threshold |
| the run-log header (`src/bowen/io/load.py::load_spec_revision`, `M16.A.1`) | stamps the spec's version line at run time | a run made from this code now records revision 11 although the three rows above implement revision 10. **Until the Phase C rework, a trace or log produced from this code MUST be read as revision-10 behaviour for `M4.C.1`, `M4.A.1` and `M1.C.3`.** The committed review traces under `docs/review/` say revision 10 and are correct. ⟦rev11 · learning-qa finding 5, 2026-10-06⟧ |

**The ledger's case against the old signal, and why it does not stand against the new one** ⟦rev11 · learning-qa finding 9, 2026-10-06⟧. Several ledger entries argue against a short-horizon relief signal: `FE05.4` ("actions are taken to relieve the anxiety of the moment rather than being determined by a long-term view" — regression **defined** as that signal), `FE03.19`, `KS21.1` and `K03.2`. Each argues that **differentiated** action, and a helper's objective, are not driven by relief. Revision 11 agrees and keeps all of it: relief drives only the automatic channel, which `FE05.4` identifies as the regressed mode; the self-directed channel is never scored by relief (`M5.F.5`) and never reinforced (`M4.D.6d`); and the coach's objective stays understanding, not relief (`M1.E.2a`). `FE05.4`'s note that such a signal models "a permanently regressed family" holds only for a family running wholly on the automatic channel. `M4.D.1a`'s mixing weight by level prevents that, and `M11.C.16` measures how far each family goes toward it.

**Not decided at this revision.** The proposal marked five rule groups for review without proposing a change: `M1.A.14c`/`M1.A.14d`, `M1.B.10`, `M4.D.5a`, `M7.D.2a`/`M7.D.2d` and `M9.6`. They stand as written. **Adjacent issue, recorded for revision 12:** `M1.D.1`'s three-sink budget writes where anxiety goes; under this revision's principle it would be accounting over learned patterns.

**The material held in `TODO.md` under "Spec revision 11"** is not part of this revision. Its label predates it and is now stale; it waits for the owner to assign it to a later revision.

---

### Revision 10 — the method literature, 2026-09-22 ⟦rev10 · owner decision 2026-09-22⟧

**Approved 2026-10-06.** See *Approval* at the end of this section. The paragraph below is kept as written at revision 10, except that the markers no longer say *proposed*.

**Everything this revision adds was a proposal until approval.** Each requirement, table row, register entry and amendment
ends with a marker of the form ⟦rev10 · C8 · Holland 2026⟧, naming the candidate or design-lessons
item and the first author of the paper it rests on. A requirement without a marker is approved text and was
not touched, except where an approved statement had to be extended — those edits are listed below and are
marked too. At approval the owner kept the markers as provenance and removed only the word *proposed*. The rationale for every marked item lives in
`model_explainer.md` §18, on the owner's decision of 2026-09-22; the candidates and design-lessons files are
the evidence behind it (§0.2).

**What was read.** Four bodies of material, all under `papers/`:

| | File | Contributed |
|---|---|---|
| 29 model-design papers | `DESIGN_LESSONS_model_design_papers_2026-09-17.md` §1–§6 | §2 lessons, entered only where the text states a testable form |
| The SEAA addendum (Liu 2026, 2609.17331) | the same file, §7 | the four drafts `E-DR`, `E-SH`, `E-RM`, `E-RE` and the six test designs of §7.10 |
| Fourteen sweep preprints, read in full | `SWEEP_READING_REPORTS_2026-09-20.md`, from the brief in `sweep_readers_brief_2026-09-19/` | the evidence behind the candidates, cited by part and candidate, not restated |
| Their 42 candidates | `SPEC_CANDIDATES_from_preprints_2026-09-20.md` | the primary source of requirement text; each coverage status was re-checked against revision 9 before placing it |

Where a §7.8 draft and a C-number overlap, `DESIGN_LESSONS` §8.9's mapping table was followed and the item
was entered once, with both references in its marker: `E-DR` with C16, C24 and C31 (`M17.E.2`); `E-SH` with
C29 (`M17.F.1`); `E-RM` as arm (d) of C34's list (`M17.D.1`, `M17.D.2`); `E-RE` as a column on C39's readout
(`M17.B.5`). No wording was taken from the PDFs; the paraphrases are the candidates file's.

**Two placements differ from the handoff that drove this revision, both on the owner's decision of
2026-09-22.** `E-RE` is at `M16.A.9` and `M11.C.40`, gated in Phase D, not in `M17.C` — `DESIGN_LESSONS` §7.9
says it is a Phase B–D readout and the only observable `M4.D.6d` has. And the three §7.10 tests with no
C-number are entered: (c) at `M17.D.4`, (d) at `M11.D.20`, (f) at `M17.B.4`.

**Parts of three Phase E candidates were placed outside `M17`**, because they must exist during B–D: C29's
non-absorbing-bound test (`M11.D.21`), C31's margin log (`M16.A.10`) and C41's discrepancy log (`M16.A.5a`).

| | Count |
|---|---|
| New IDs | **91** — M1 4, M2 1, M3 5, M4 6, M6 1, M7 1, M9 1, M10 4, M11 25, M13 1, M14 3, M15 1, M16 9, M17 29 |
| Amendments of approved requirements, by suffixed ID | **3** — `M3.D.4a` (C1, replaces `M3.D.4`'s second sentence), `M11.4b` (the exceptions count) and `M13.4` (adaptive ensembles in Phases C and D) |
| New acceptance criteria | **7** in `M11.C` (`M11.C.35`–`M11.C.41`) and **8** in `M11.D` (`M11.D.15`–`M11.D.22`) |
| New `[I]` register entries | **18** in `M10.C.1a`, naming **25** constants; none is given a value |
| Candidates entered | **all 42** (C1–C42), the four §7.8 drafts, and all six §7.10 tests; C19 enters as a criterion form with no instance; `M11.C.41` is the owner's answer to it |
| New part headings | `M3.E`, `M14.A`, and the module `M17` with parts A–G |

**Every marked item with an ID.** Form is *new*, *amendment*, *criterion*, *register* or *Phase E*.

| Spec ID | Candidate or E-ID | Source paper | Form |
|---|---|---|---|
| `M1.A.20` | C2 | Buffalo, Pearson & Klein 2026 (2603.11084) | new |
| `M1.B.13` | C2 | Buffalo, Pearson & Klein 2026 (2603.11084) | new |
| `M1.C.7` | C2 | Buffalo, Pearson & Klein 2026 (2603.11084) | new |
| `M1.F.1b` | C7 | Li & Tao 2026 (2603.00113) | new |
| `M2.A.0h` | C2 | Buffalo, Pearson & Klein 2026 (2603.11084) | new |
| `M3.D.4a` | C1 | Buffalo, Pearson & Klein 2026 (2603.11084) | amendment |
| `M3.D.4b` | C3 | Buffalo, Pearson & Klein 2026 (2603.11084) | new |
| `M3.D.4c` | C4 | Buffalo, Pearson & Klein 2026 (2603.11084) | new |
| `M3.E.1` | C6 | Li & Tao 2026 (2603.00113) | new |
| `M3.E.2` | C6, DL §2.1 | Li & Tao 2026 (2603.00113); Axtell & Farmer 2022 | new |
| `M4.B.2` | C9 | He 2026, VISA (2607.28027) | new |
| `M4.C.9` | C8 | Holland et al. 2026 (2607.29546) | new |
| `M4.D.1e` | C11 | Buitrago López et al. 2026 (2606.12369) | new |
| `M4.D.1f` | C12 | Ye et al. 2026, TRAILS (2605.18890) | new |
| `M4.E.1a` | C7 | Li & Tao 2026 (2603.00113) | new |
| `M4.G.3` | C13 | Prasad 2026 (2607.07753) | new |
| `M6.3` | C15 | He 2026, VISA (2607.28027) | new |
| `M7.C.1e` | C15 | He 2026, VISA (2607.28027) | new |
| `M9.8` | C9 | He 2026, VISA (2607.28027) | new |
| `M10.B.4` | C23 | Zhou et al. 2026, PIMMUR (2509.18052) | new |
| `M10.B.5` | C18 | Prasad 2026 (2607.07753) | new |
| `M10.C.1a` | one per entry — C6, C8, C12, C13, C16, C19, C21, C26, C30, C32, C37, E-RE, E-SH, E-DR, DL §2.3, §2.6, §7.10(c), §7.10(e), and the owner's decision for `M11.C.41` | as each entry's marker | register |
| `M10.C.5` | C22 | Zhou et al. 2026, PIMMUR (2509.18052) | register |
| `M11.1b` | C17 | Prasad 2026 (2607.07753) | new |
| `M11.1c` | C21 | Ye et al. 2026, TRAILS (2605.18890) | new |
| `M11.1d` | C22 | Zhou et al. 2026, PIMMUR (2509.18052) | new |
| `M11.1e` | C20 | Kalluri 2026 (2603.01189) | new |
| `M11.1f` | DL §2.6 | Mou, Wei & Huang 2024 | new |
| `M11.4b` | C15, C16, C19, C26 | He 2026, VISA (2607.28027); Prasad 2026 (2607.07753); Kalluri 2026 (2603.01189); Blando et al. 2026 (2604.04543) | amendment |
| `M11.4c` | C19 | Kalluri 2026 (2603.01189) | new |
| `M11.4d` | C24 | Prasad 2026 (2607.07753) | new |
| `M11.4e` | C25 | Ye et al. 2026, TRAILS (2605.18890) | new |
| `M11.4f` | owner decision 2026-09-22 (`DESIGN_LESSONS` Q10), E-RM | owner; Liu 2026, SEAA (2609.17331) | new |
| `M11.C.35` | C8 | Holland et al. 2026 (2607.29546) | criterion |
| `M11.C.36` | C9 | He 2026, VISA (2607.28027) | criterion |
| `M11.C.37` | C15 | He 2026, VISA (2607.28027) | criterion |
| `M11.C.38` | C16 | Prasad 2026 (2607.07753) | criterion |
| `M11.C.39` | C18 | Prasad 2026 (2607.07753) | criterion |
| `M11.C.40` | E-RE | Liu 2026, SEAA (2609.17331) | criterion |
| `M11.C.41` | owner decision 2026-09-22 (answer to C19) | owner, as domain expert; `M1.C.3a` | criterion |
| `M11.D.15` | C4, DL §7.10(a) | Buffalo, Pearson & Klein 2026 (2603.11084); Liu 2026, SEAA (2609.17331) | criterion |
| `M11.D.16` | C5, DL §7.10(b) | Li & Tao 2026 (2603.00113); Liu 2026, SEAA (2609.17331) | criterion |
| `M11.D.17` | C10 | He 2026, VISA (2607.28027) | criterion |
| `M11.D.18` | C12 | Ye et al. 2026, TRAILS (2605.18890) | criterion |
| `M11.D.19` | C14 | Kurz 2025 (2512.18016) | criterion |
| `M11.D.20` | DL §7.10(d) | Liu 2026, SEAA (2609.17331) | criterion |
| `M11.D.21` | C29 | Holland et al. 2026 (2607.29546) | criterion |
| `M11.D.22` | C35 | Zhou et al. 2026, PIMMUR (2509.18052) | criterion |
| `M13.4` | C26 | Blando et al. 2026 (2604.04543) | amendment |
| `M14.A.1` | C10 | He 2026, VISA (2607.28027) | new |
| `M14.A.2` | C10 | He 2026, VISA (2607.28027) | new |
| `M14.A.3` | C10 | He 2026, VISA (2607.28027) | new |
| `M15.A.1a` | C2 | Buffalo, Pearson & Klein 2026 (2603.11084) | new |
| `M16.A.1a` | C6 | Li & Tao 2026 (2603.00113) | new |
| `M16.A.3a` | C9 | He 2026, VISA (2607.28027) | new |
| `M16.A.3b` | C11 | Buitrago López et al. 2026 (2606.12369) | new |
| `M16.A.3c` | C12 | Ye et al. 2026, TRAILS (2605.18890) | new |
| `M16.A.5a` | C41 | Kalluri 2026 (2603.01189) | new |
| `M16.A.7` | C23 | Zhou et al. 2026, PIMMUR (2509.18052) | new |
| `M16.A.8` | C20 | Kalluri 2026 (2603.01189) | new |
| `M16.A.9` | E-RE | Liu 2026, SEAA (2609.17331) | new |
| `M16.A.10` | C31 | Kurz 2025 (2512.18016) | new |
| `M17.A.1` | C26 | Blando et al. 2026 (2604.04543) | Phase E |
| `M17.A.2` | C27 | Blando et al. 2026 (2604.04543) | Phase E |
| `M17.A.3` | C25 | Ye et al. 2026, TRAILS (2605.18890) | Phase E |
| `M17.A.4` | DL §2.6 | Röchert et al. 2022 | Phase E |
| `M17.B.1` | C28 | Buffalo, Pearson & Klein 2026 (2603.11084) | Phase E |
| `M17.B.2` | C29 | Holland et al. 2026 (2607.29546) | Phase E |
| `M17.B.3` | C41 | Kalluri 2026 (2603.01189) | Phase E |
| `M17.B.4` | DL §7.10(f) | Liu 2026, SEAA (2609.17331) | Phase E |
| `M17.B.5` | C39, E-RE | Buitrago López et al. 2026 (2606.12369); Liu 2026, SEAA (2609.17331) | Phase E |
| `M17.B.6` | C40 | Sachdeva & van Nuenen 2025 (2510.10002) | Phase E |
| `M17.C.1` | C32, DL §2.6 | Li & Tao 2026 (2603.00113); Guan et al. 2026, Light Society | Phase E |
| `M17.C.2` | C30 | Sachdeva & van Nuenen 2025 (2510.10002) | Phase E |
| `M17.C.3` | C33 | Ye et al. 2026, TRAILS (2605.18890) | Phase E |
| `M17.D.1` | C34, E-RM | Sachdeva & van Nuenen 2025 (2510.10002); Liu 2026, SEAA (2609.17331) | Phase E |
| `M17.D.2` | E-RM, C34 | Liu 2026, SEAA (2609.17331); Sachdeva & van Nuenen 2025 (2510.10002) | Phase E |
| `M17.D.3` | C35 | Zhou et al. 2026, PIMMUR (2509.18052) | Phase E |
| `M17.D.4` | DL §7.10(c) | Liu 2026, SEAA (2609.17331) | Phase E |
| `M17.D.5` | DL §2.6 | Yang et al. 2024, OASIS | Phase E |
| `M17.E.1` | DL §2.3, DL §2.4 | Castellano, Fortunato & Loreto 2009 | Phase E |
| `M17.E.2` | E-DR, C16, C24, C31 | Liu 2026, SEAA (2609.17331); Prasad 2026 (2607.07753); Kurz 2025 (2512.18016) | Phase E |
| `M17.E.3` | C31 | Kurz 2025 (2512.18016) | Phase E |
| `M17.E.4` | C37 | Ye et al. 2026, TRAILS (2605.18890) | Phase E |
| `M17.E.5` | C38 | Prasad 2026 (2607.07753) | Phase E |
| `M17.E.6` | DL §2.1 | Axtell & Farmer 2022 | Phase E |
| `M17.F.1` | E-SH, C29 | Liu 2026, SEAA (2609.17331); Holland et al. 2026 (2607.29546) | Phase E |
| `M17.F.2` | DL §7.10(e), DL §2.5 | Liu 2026, SEAA (2609.17331); Axtell & Farmer 2022 | Phase E |
| `M17.G.1` | C36 | Ye et al. 2026, TRAILS (2605.18890) | Phase E |
| `M17.G.2` | C42 | Zhou et al. 2026, PIMMUR (2509.18052) | Phase E |
| `M17.G.3` | DL §2.6 | Axtell 2016 | Phase E |

**Marked edits without an ID of their own.** Front matter (revision, date); §0.1's Phase E paragraph; §0.2's
method-literature row; §0.3's counts (restated at step 7, below); `M3.D.4b`'s ten draw-class rows, every one
the sweep reader's proposal; four rows added to `M11.4`'s exceptions table (`M11.C.37`, `M11.C.38`, the
`M11.4c` form, `M17.A.1`'s `UNDETERMINED`); the Done-when cells of Phases B, C and D in `M13`, and a new
Phase E row carrying `M17` and `M11.C.8`; the `M14.A` stub table.

**Candidates judged already covered, and not entered** (candidates file §9): the engine driver surface —
reset, next, eval (`M16.B`); refactor-equivalence by draw order (superseded by C1); interval-valued initial
conditions (`M15.D.2`–`M15.D.4`); exact rational arithmetic (not adopted; byte identity plus C31 suffices); a
solo no-interaction baseline (extended by C34); information availability against awareness (the belief
layer and exogenous-flagged events — a one-line check that spell onsets enter as events, `M1.F.6`, is still
worth doing); one-component-at-a-time mechanism ablation (`M11`'s mutation protocol); readout frequency per
output (`DESIGN_LESSONS` §2.8); a single seeded generator (`M3.D.4`, now amended by C1); execution mode as an
assumption to test (§2.1, now `M17.E.6`); one-factor and factorial sweeps (§2.3, now `M17.E.1`); witness
radius with a visibility weight (per-hop fidelity); self-describing run files (`M16`); memory as digested
persistent state and event-driven interaction (`M1`, `M16.B.3`); turn order and initial event as
perturbations, and freezing a slow variable (§2.1, §2.6). The exploratory LLM-line items X1–X4 are not spec
content and were not entered.

**Design lessons noted, not specified** — no testable form is stated in the text, or the item is a cost
decision: §2.2's point that statistical-physics critical points and exponents do not carry to twelve agents;
§2.7's functional-form artefacts (a product with a zero term, equality created by rounding, accumulation with
no decay, division by a `functional_level` near zero); §2.8's snapshot-and-fork (a cost decision; under
`M3.D.4a` a fork is a plain state copy), naming of three intervention channels (`M17.D.3` declares the
channels but has no mid-run perturbation), appraisal and belief records citing event IDs, a `WITHHOLD` count
in readouts, and windowed trajectories for slow variables outside a shock; §2.9's ODD format and an
"ever varied" register column (partly met by `M10.B.4`, `M16.A.7` and `M17.E.3`); §2.10's candidate
functional shapes; §2.11's Latin-hypercube design and the history-matching question already on the
design-lessons question list; §2.12's supporting statements.

**Read at abstract level only; no requirement derived.** The eight papers of `DIGEST.md`'s 2026-09-21 entry
(2609.16395, 2609.15849, 2609.16344, 2609.21857, 2609.21636, 2609.21349, 2609.19913, 2609.21940), which were
not read in full and from which no candidate was drawn; the Light Society reference list
(`2026_Light_Society_2506.12078_model_design_refs.md`); and the catalogues `PREPRINT_SWEEP_2025-09_to_2026-09.md`
and `INDEX.md`. `SPEC_CANDIDATES_plain_language_2026-09-20.md` is explanation only and supplied no wording.

**The owner's answers of 2026-09-22, and what was done with each.** The fourteen questions this section
first listed, in their original order.

| # | Question | Answer | Done |
|---|---|---|---|
| 1 | Death: an exit from the field, or absorbed by the family? (C15, `M6.3`) | **Deferred for later review.** Guidance: relationship energy behaves like chemical bond energy and does not disappear | Guidance recorded in `M6.3` as not yet a requirement; `TODO.md` |
| 2 | Where method rationale lives | In `model_explainer.md` | Mirrored in explainer §18; every method citation in this spec now points there; §0.2 recasts the `papers/` files as evidence |
| 3 | `M17` in this document or split out | One master document | Recorded in §0.1 |
| 4 | C3's keys | Do the recommendation | The reader's six keys adopted as the recommendation; the four uncovered classes given recommended keys (`M3.D.4b`, marked *recommended rev10*) |
| 5 | C13's relief term | **Review later** | `M4.G.3` stays marked; `TODO.md` |
| 6 | C19's corpus ordering | Most mechanisms scale with differentiation of self and the stress being experienced: lower differentiation and higher stress mean more anxiety and more exaggerated patterns of functioning, such as more reactivity | New criterion `M11.C.41`, gated in Phase C: four two-arm directions over a 2×2 of level and stress. `M11.4c`'s mechanism ranking stays uninstantiated; strike it at approval unless a ranking turns up |
| 7 | The directions of `M11.C.35` and `M11.C.36` | Both confirmed | No change |
| 8 | `E-RM` within `M0.4` (`DESIGN_LESSONS` Q10) | Switching off a mechanism is a change; the owner then adopted the recommendation: such comparisons are allowed, as tests of the model only | New rule `M11.4f`: such comparisons are between two models, are reported as tests of structure, never as effects on a family, and are swept in Phase E. It reaches `M17.D.1`, `M17.D.2`, `M11.C.39` and `M11.C.40`, and — by the owner's decision — the approved ablation of `M10.C.4a`, which it leaves unchanged |
| 9 | `E-RM` and C34, one requirement or two (Q16) | **Needs review** | Left as `M17.D.1`(d) plus `M17.D.2`; `TODO.md` |
| 10 | Floors (Q15) | Keep as SHOULD | Recorded in `M17.D.4` |
| 11 | `E-DR` against the constant sweep (Q8) | Asked for a recommendation, then **approved** it | Kept separate, because they answer different questions; `M17.E.2` sweeps the same range `M17.E.1` samples |
| 12 | Ensemble size in Phases C and D | Adaptive | `M13.4` amends the fixed 1,000-seed ensembles; `M17.A.1`, `M17.A.3` and `M17.A.4` are built in Phase C, and `UNDETERMINED` does not pass a phase |
| 13 | The arm-blindness check from Phase B | Do the recommendation | `M11.D.22`, gated in Phase B |
| 14 | `CLAUDE.md` still says revision 6 and 427 | Yes, update after approval | `TODO.md`, together with the four other documents in the count check below |

**Consistency checks, run 2026-09-22 at the end of this revision.** No `M11.D` scan code exists for the v2
engine yet, but `tests/test_spec_consistency.py` implements the `M11.D.10`, `M11.D.11`, `M11.D.13` and
`M11.D.14` scans against this document and was run; the other checks were one-off Python and `grep`
scripts, not committed.

| Check | Result |
|---|---|
| New criteria numbered from `M11.C.35` (revision 9's table ended at `M11.C.34`) | pass — `M11.C.35`–`M11.C.41`; `M11.D` continues at `M11.D.15`–`M11.D.22` |
| Every ID matches `M\d+(\.[A-Za-z0-9]+)*`; no duplicates | pass — 518 defined IDs, 0 failures, 0 duplicates |
| Revision 9's IDs unchanged | pass — all 427 still defined, 0 removed, 91 added; extracted from both versions with the same pattern and diffed |
| `M11.D.11`, both directions | pass — every `M11.C` criterion's phase matches its `M13` Done-when cell (B 0, C 19, D 21, E 1) and no cell names an orphan; the project's three `M11.D.11` tests pass |
| `M11.D.13`, no subject-negated `MUST` | pass — the project's test, and a separate scan of the added text |
| Every marker resolves | pass — 136 markers; every part names a C-number heading in the candidates file, an `E-DR`/`E-SH`/`E-RM`/`E-RE` draft in `DESIGN_LESSONS` §7.8–§7.9, or a `DESIGN_LESSONS` section or §7.10 item, except three administrative forms (*owner decision 2026-09-22*, *first draft, rev10*, *register extension*) and the *reader's proposal* and *recommended rev10* tags on `M3.D.4b`'s rows |
| No quotation of eight or more words from a paper | pass — every added line compared as eight-word sequences against the `pdftotext` output of all 70 PDFs under `papers/`, including the fourteen sweep papers and SEAA, and the same comparison over the new `model_explainer.md` §18: 0 matches |
| §0.3's counts | restated above: 835 tokens, 79 named tests (revision 9: 663, 51). The project's `M11.D.14` test passes against them |
| `M14`'s coverage figures | nothing to restate — `M14` states no figures; `docs/spec_coverage.md` does not exist until implementation |
| Requirement count cited by other documents | **fails, and is left failing on this branch.** The project's requirement-count test finds `CLAUDE.md`, `README.md`, `CHANGELOG.md`, `docs/theory/_STATUS.md` and `agent_model_proposal.html` claiming 427 against the spec's 518. This revision may not edit `CLAUDE.md` or `docs/theory/`, and restating the others would present unapproved proposals as the count. Update all five at approval, with the count as approved |

**Adjacent issues found, not fixed.** `M11.2`'s note says the criteria table "now ends at 33" and `M11.3`
counts "4 of 33 criteria" and "27 unclassified"; at revision 9 the table already held 34 rows, and it now
holds 40. `M11.3`'s stressor classification has not been applied to the six new criteria. `M1.F.1a` sits
physically inside `M4.E`. Revision sections 4 and 5 are out of order below. These are approved text and
were left for the owner.

---

#### Approval — 2026-10-06

Revision 10 was approved by the owner on 2026-10-06, on these terms:

| Item | Decision |
|---|---|
| Scope of approval | All revision-10 text, plus `M11.F.10` (entered 2026-10-06 as a revision-11 proposal, approved with revision 10 because it is the owner's own decision and changes no mechanism) |
| The external review, `docs/REVIEW_spec_rev10_2026-09-23.md` | Approved without first revising for its findings, following the review's recommendation 9: *approve and build Phase B only*. Each finding is carried into the implementation plan as a blocker on the phase it affects. The spec is not changed for them at this revision |
| `M11.4c` | **Kept.** The revision-10 condition was to strike it unless a ranking of mechanisms turned up. The review found one in the ledger — `L07.4`: spouses' dynamics first, relationships outside the ego mass second, fusion intensity third — so the condition for striking is not met. Whether `L07.4` instantiates it is left to the phase that builds those mechanisms |
| `M4.G.3` | **Kept as written.** It is conditional and a SHOULD; it binds only if a term of the kind it names exists, so approving it decides nothing early |
| `M11.E` | The four human-review proposals are accepted as written |
| Still open | `M6.3` (disposition at death; blocks `M11.C.37` and Phase D) and the `E-RM`/C34 question (Phase E). Neither blocks Phase B |
| Revision-11 material in `TODO.md` | Held. Not part of this approval |
| Phase B plan, approved the same day | All nine recommendations (D1–D9) of `docs/implementation_plan_phase_b.md` adopted as written. Spec changes: `M4.C.1` pulled into Phase B (`M13`); `M1.A.21`, `M1.A.22`; `M2.3a` and Ravi's `basic_level`; `M3.C.2`; `M4.G.2a`; `M14.A.3`, `M14.A.4` and the filled register; `M11.D.17` reads "no writer in any phase" |

### Revision 9 — the corpus-fidelity sweep, 2026-08-28

**Every quoted passage in this document was checked against the primary sources.** Five cold agents on a
different model, reading the source `.txt` files and forbidden the extractions, plus a deterministic string
match by the author over all 214 quotations.

| | |
|---|---|
| Quoted passages in requirement text | **214** |
| Located verbatim in the corpus (punctuation- and OCR-proof match) | **160** |
| Legitimate self-quotes of project documents | **52** |
| Unexplained after inspection | **0** |
| Requirements resting on a verified corpus quote | 94, **all judged** |

**One fabricated quotation existed and is fixed.** `M1.A.4e` read *"`KS11.7` says so in its opening line:
**The family diagram records**, per person: birth date…"*. That phrase appears nowhere in the book. It was a
paraphrase placed inside quotation marks and attributed to the source, and it was the evidence for revision
8's demotion of the four assessment domains. Corrected, and the demotion restated to what the source carries.

**The dominant class was modal inflation** — a hedged, permissive or illustrative source becoming a mandate:

| | Source | Had become |
|---|---|---|
| `M7.D.2a` | "**can** stabilize somewhat… **or** the regression can continue to get worse" | "**MUST** fall" — one branch mandated, the other erased |
| `M16.D.2` | "Bowen **suggested**… the family **could perhaps**" | "Bowen **required**" |
| `M1.A.11b` | genes "**perhaps** having a role"; theory "**does not rule out**" | channel **MUST** be constitutional |
| `M1.D.7l` | "a new balance, **such as** 55 or 60" | a regression **MUST** be **bounded** |
| `M1.A.18b` | KB07, **hedged**: undifferentiation is roughly what sets thinking hard, and loosening it would let the thinking **change faster** (K07.3) | a **ceiling** on the attainable value |

**Scope shifts, three of them introduced at revisions 7 and 8.** A phrase restricted to "**Some of those in
this group**" applied to a whole band (`M1.A.3c`); a **societal** trend used to ground an **individual**
mechanism (`M1.A.4h`, `M4.D.1a`); a consequence presented as the quotation's when the corpus arguably cuts
the other way — the actions that can make a system more flexible "are based on a **broad and long-range
assessment**" (`M4.D.1a`; `family_evaluation/fe03.md` · FE03.21).

**Evidence framing.** "An experiment and an independent replication" were two clinical accounts, one of them
the author confirming what he had been told (`M1.E.2a`). "Two families — home movies in both" was one family
(`M7.E.1d`). "Three statements, every one about the mover's anger" was two (`M5.D.4a`).

**Attribution.** ASR transcripts attributed to Bowen personally against the project's own rule (`M5.D.7a`);
words belonging to **Andrew Solomon**, quoted by Kerr, grounding a requirement (`M4.D.5c`); a line whose
speaker is genuinely ambiguous between Bowen and Kerr (`M1.A.18b`).

**What this says about the method.** Two warm failure-pattern sweeps and eight revisions had passed over all
of it. Every one of these was found by reading the **primary** rather than the extraction — and the
mechanical existence check, which is the part a language model does worst, was done with `grep` rather than
by an agent, after two false "not found"s taught that lesson. Hedge-level findings are backlogged in
`TODO.md` rather than fixed, per the bounded-loop rule.

---

### Revision 8 — what the estimator may read, 2026-08-28

Raised by the project owner across three points, each checked and each upheld. Together they replace the
estimator's observable set.

| | What changed | Where |
|---|---|---|
| **Demoted** | `M1.A.4e` presented **genogram intake fields** as "the estimator's observables". `KS11.7`'s own opening says they are what *"the family diagram records"*. They are now a weak prior at import and **MUST NOT** set an estimate | `M1.A.4e` |
| **New** | A **three-tier** reading: a single adverse event is mostly **load**; a chronic self-maintained pattern reads **functional** level under load correction; only a position held under pressure reads **basic** level | `M1.A.4g` |
| **New** | The discriminator is **which channel was selected under discomfort, at comparable load** — `M4.D.1a`'s mixing weight read backwards out of behaviour | `M1.A.4h` |
| **New** | The **system's reaction to a symptom** is the best-controlled reading in the model: one event, N observers, one moment, so load is constant **by construction** | `M1.A.4i`, `M11.C.34` |
| **New** | The circularity boundary stated precisely: symptom **magnitude** may not feed the estimator; the **channel selected** must, because it reads policy rather than accumulated load | `M1.A.4j` |
| **Generalised** | `M1.E.2`'s three reaction detectors were scoped to the coach. They are now on **every `Person`** — without them the family's own reaction is unreadable | `M1.A.19`, `M1.E.2` |

**Three things the corpus already said, which the spec had not carried through.**

1. `KS06.8` **already corrects** `KS05.10`: "**In terms of solid self, one's personal life is where the rubber meets the road**", and it is the intimate relationship "**and not how they manage themselves in their work lives**" that determines offspring level. `M1.A.4e` cited this correction **underneath the table it invalidates**, so the table still read as the instrument.
2. The outcomes in that table are **base-rate common**. A divorce or a job loss is an event a large share of the population has; it carries almost no information about who had it, and it confounds the load on a person with their functioning.
3. Ch10 generalises the neutral third to "**any third person… no matter what the subject matter**", and `agent_model_proposal.html` §10 records the conclusion outright — "**one rule seen from four angles… Implement it once**". `M1.E.2` implemented one of the four.

**The unification, which is the reason to believe it.** "Adhering to one's convictions under relationship
pressure" and "carrying discomfort rather than discharging it" are **one** observable, not two — `M5.F.5`
says so in a sentence, and `M4.D.1a` already carries the quantity as the mixing weight. Solid self is the position a person keeps **whatever the pressure**; pseudo-self is the **negotiable** part, which pressure from others can change (1979 Tape 2). Everything needed is already state the
model holds: the accommodation stock, the taboo set, gradable concession, the solid-self fraction. **No new
machinery.**

**And the load correction applies twice over.** Channel mix is *set by* level, so reading level back out of an
uncorrected run recovers its own input. `M1.A.4b` was already the answer and had not been extended to these
readings.

---

### Revision 7 — the scale is a continuum, 2026-08-28

Raised by the project owner: *Bowen's quadrants are bands across a continuum, not shift points.* Checked
against the primary and upheld.

| | What changed | Where |
|---|---|---|
| **Corrected** | `M1.A.3` required *"exactly one behavioural transition on `basic_level`, at 50"*. The licence is now a **slope** — continuous in level, with the joint-decision domain lagging every other domain at every level, and no discontinuity anywhere | `M1.A.3`, `M1.A.2` |
| **Guarded** | Removing the step **MUST NOT** flatten the model: the corpus asserts real, **non-monotone** structure over the range — emotionality peaks in the middle, and a stratum looks differentiated and is not | `M1.A.3c` |
| **Generalised** | **A band edge MUST NOT be a branch point anywhere** — 25, 50, 60, 70, 75. The scheme moved three times inside the corpus and the two authors give different edges; branching on one hard-codes a coordinate the source does not hold still | `M1.A.3d` |
| **Regraded, provisionally** | The capacity floor at 25 becomes a continuous falloff, applied on the principle rather than on a fresh reading — its source has **not** been re-read, so its location stands pending that | `M1.A.4d` |
| **Withdrawn as a target** | "the transition at 50" had been listed as a corpus-stated calibration target | `M10.C.2` |
| **Rewritten** | The import rule that no rating point may land on 50 — its reason went with the step, and the problem underneath is larger without it | `M15.B.3` |
| **Word corrected** | "regime boundary" → **turning point** in the sign-flip argument. Conclusion unchanged; the picture is a roll-off, not a cliff | `M15.D.4`, `M11.F.9(c)` |

**The evidence, since this overturns a MUST cited to Bowen's own hand.** Ch16 states it as a description with
hedges — the intellect "**begins** making **a few** decisions of its own". Ch21 has **no regime change at 50
at all** and calls the lower/upper transition **explicitly gradual** (`L21.1`, already marked `[corrected]`
in the ledger). The band scheme is **present 1966, absent 1972, present again 1976**, graded `[X]` in the
explainer as "not a constant of the theory". Kerr 1988 gives different edges entirely — 0–10, >60, >70
(`FE03.5`). Bowen revised the top twice and "**subsequently dropped the term scale**" (`KS05.3`). Ch16 is the
latest of the three and is **not** outvoted — but it describes a band, and this requirement had implemented
a cliff.

**This is the same failure as `M5.D.4` at revision 4**: a source *description* firmed up into a mechanism the
source does not assert. Two instances now, both caught by reading the primary rather than the extraction, and
neither catchable by a regex. A sweep for others of this shape is logged in `TODO.md`.

---

### Revision 6 — the run log, 2026-08-28

| | What changed | Where |
|---|---|---|
| **New module** | The run log and the readable trace. Six existing requirements already depended on a log; none of them said what one is | `M16` |
| **Exit condition made checkable** | Phase B's "the event log reads correctly" named no requirement and no test. It now names `M16.C.2` and `M16.T.1` | `M13` |
| **Engine purity** | The engine emits records; the caller persists them. The frozen engine's `sim_audit.csv` is named as the anti-pattern | `M16.B` |
| **Ordering** | `M14` was sitting after `M15`; modules are now in numeric order |  |

**Two things this settles.** A log records **effects beside causes** and the **selection rationale**, not
just the moves — without those an event log is a list, not a causal trace, and the causal trace is most of
the argument for the agent model over the grid. And the deterministic renderer is **not** the language
layer: rendering a finished log by template is reproducible and carries no interpretation, where narration
by a language model is Phase F, off by default, and never in the ensemble path (`M16.E`).

---

### Revision 4 — *Family Evaluation* (Kerr & Bowen, 1988), applied 2026-08-27

The sixth corpus source, folded in against the decisions recorded in `docs/DECISIONS — FAMILY EVALUATION.md`. **One correction, three amendments, and about twenty new requirements.**

| | What changed | Where |
|---|---|---|
| **Corrected** | `M5.D.4` was **inverted**. Ch13 says the mover's *freedom from* anger admits the peak; the spec said anger did. Settled by a re-read of the primary chapter, which mentions anger exactly once | M5.D.4, M5.D.4a, M11.C.32 |
| **Amended** | The channel prior is movable by a relational term and overridable by a constitutional one (A1, option c) | M1.A.11b, M1.A.11c |
| **Amended** | The transition at 50 keeps its licence implementation and gains an awareness **readout**; neither is reduced cognitive capacity (A2) | M1.A.3b |
| **Amended** | The dependence gate now binds the **slow** clock, not only the `I-POSITION` move (B1 tail) | M7.A.1a, M11.C.33 |
| **Reviewed, unchanged** | `M12.2` stands in full — Kerr 1988 names the locus of the first unknown, not the rule (A4, option b) | M12.2a |
| **New mechanisms** | Belief becomes a channel; cross-person reinforcement; a self-generated stress term; mutually protective channels; non-monotone lock-in; an objectivity gate on amplification; a `PREPARE` phase; a binder-unavailable event; emotional reserve; default tie deterioration | M9.6–M9.7, M4.D.6e, M4.A.5, M7.D.2c–2d, M4.C.8a, M5.D.2a, M1.F.9, M4.D.5d, M7.E.4 |
| **New constraints** | Pole assignment independent of sex; non-monotone coach contact; every criterion needs a stressor | M2.A.0g, M1.E.8, M11.3 |
| **New readout** | The eight-component family evaluation, with components 9 and 10 explicitly barred | M11.G |
| **New magnitudes** | Twelve stated bounds and shapes, admitted as **checks** and never as parameters | M10.C.4, M10.C.4a, M10.C.3a |
| **New prohibitions** | Power and punishment as mechanisms — the only `M12` entry stated by the author; no adjudication of trauma against process; no rights-as-regression readout | M12.5, M11.F.7, M11.F.8 |

**One item remains open and is not applied:** nothing. All nine decision items are resolved.

### Revision 5 — the import question and what it exposed, 2026-08-27

Arising from a working session on whether the model can answer *"if you changed this one thing, what would happen to this family?"*

| | What changed | Where |
|---|---|---|
| **New guard** | Three clauses on speaking about a real family. (a) no output about a named family or person; (b) initialisation must carry its uncertainty; **(c) no counterfactual from parameters tuned to reproduce a known history** — a *correctness* clause, because fitting one arm breaks the cancellation `M0.4` depends on, invisibly and in one direction only | `M11.F.9` |
| **New module** | The family-diagram import contract — export format, the value/range split, four adapter traps, the envelope requirement, and fixture mode | `M15` |
| **Scope** | `M15` is Phase E but specified early, because the source application is under the project owner's control and the format is cheaper to fix before it exists than after | §0.1, `M13.3` |

**The finding worth keeping.** Initial conditions are not nuisance parameters. Invented constants cancel
between two arms because they do not interact with the intervention; **initial conditions are precisely what
the intervention acts on**, so their error crosses a regime boundary and flips the sign rather than shifting
both arms together. The model has at least five such boundaries and one of them (`M5.C.1a`) is a SAFETY
property with recorded harms. **Direction is the least robust output under mis-specified inputs, not the
most** — which is why `M15.D.2` requires an envelope over ranges and forbids a point direction.

---

*Written against `model_explainer.md` v1.2 and `theory/_LEDGER.md` across six corpus sources. Every requirement traces to an explainer section or a ledger entry; the ledger traces to a chapter, a tape, an interview or a page. Nothing in this document is sourced to a summary.*

## The front matter's status line, as it stood at revision 34

APPROVED 2026-10-07 (revision 10 and 11 on 2026-10-06; revision 12, the Phase C prerequisites, on 2026-10-07; revision 13, the owner's reference family and pattern readouts, decided 2026-10-10; revision 14, the Phase D decisions, 2026-10-10; revision 15, the Phase D definitions and `M11.C.46`, 2026-10-11; revision 16, graded rejection in `M1.B.3c`, 2026-10-11; revision 17, nodal events, cutoff read from behaviour, `M11.C.17` and `M11.C.7`'s direction, 2026-10-11; revision 18, stress moves functioning and coaching or own practice opens the self-directed channel, 2026-10-11; revision 19, the stress floor, self-discovered perspective and perspective's steering, 2026-10-11; revision 20, the system's adjustment is a level that settles back and is not read by the estimator, and `M11.C.18` and `M11.C.20`'s mutation clauses name what carries them, 2026-10-11 and 2026-10-12; revision 21, holding a position stops the functioning exchange, 2026-10-12; revision 22, calm is relative to the person, 2026-10-12; revision 23, the estimator reads pseudo-self with its sign, 2026-10-12; revision 24, the symptom channels (Phase D B7) and `M11.C.22`'s control arm; revision 25, beliefs as a channel (B8); revision 26, the cross-person term weighted by the tie (B9); revision 27, borrowed pseudo-self settles back when the exchange stops — revisions 24 to 27 are Claude's recommendations of 2026-10-12, reviewed by the owner 2026-10-16: the owner asked for the work to proceed overnight without waiting for input; revision 28, the life course — death, income, a child's development, starting ages, acting age and the coach — applies the owner's decisions of 2026-10-13 (P13–P15), and the forms it gives them are Claude's recommendation, reviewed by the owner 2026-10-16, and the owner approved the part II plan that builds it on 2026-10-13, and it was built that day; revision 29, cutoff and focus — a held cutoff and tie-state readout for `M11.C.10`, learning from a witnessed cutoff (`M4.D.6f`), reactive and principled cutoff (`M1.B.3d`), chronic anxiety built over childhood (`M7.B.1a`), focus on a child's weakness (`M1.B.8a`) and focus steering acts (`M4.D.1g`), and `M11.C.2` read by age — applies the owner's decisions of 2026-10-13, and its forms are Claude's recommendation, reviewed by the owner 2026-10-16; its plan (§4.8) approved by the owner ("i approve proceed") and built 2026-10-14, with the review fixes in §4.8; revision 30, origin cutoff as an input (`M1.D.8a`), danger and the self-chosen principled cutoff (`M1.B.3e`), focus that locks in (`M1.B.8b`), `M11.C.10` read on ties to partners and parents at 200 seeds with a family-of-origin variant, and the danger premise `M11.C.47` — applies the owner's decisions of 2026-10-14, its forms Claude's recommendation reviewed by the owner 2026-10-16, its plan (§4.10) approved and built 2026-10-14; **revisions 24 to 30 reviewed by the owner 2026-10-16** (`docs/REVIEW_spec_revisions_24-30.md`): 81 of 85 entries approved as built, 4 changed (`M14.A`'s `channel_focus` and `tolerance` rows, `M11.C.14` restated, `M11.C.6` limit (a) restated, `M7.E.1f`'s band opened by the parents' acts); revision 31, the owner's review of revisions 24 to 30 applied — its plan (§4.11) approved by the owner 2026-10-16; part A, the documents, built 2026-10-16; part B, the code, config and records (`M11.C.14` as restated, `M7.E.1f`'s band, the criteria settings' grades), built 2026-10-16 with the owner's decisions on the version 1 premises (`M11.C.14`'s hold-back baseline, `M6.I.4`'s capacity rule for `M11.C.21`, `M11.C.23` reclassified, `M11.C.40`'s clause 2 reported) and every record regenerated once; `M11.C.21`'s reclassification and keeping the capacity rule decided by the owner 2026-10-17, the reclassification built 2026-10-17 with every record regenerated; revision 32, finishing version 1 — the version 1 status table, `M11.C.4`, `M11.C.29`, `M11.C.32` and `M11.C.22` resolved, `M11.C.7` and `M11.C.46` built, `M11.C.9` out of version 1 (for version 2), `M4.C.1c`'s nodal-event spike built, recorded mutations for `M11.C.11`, `M11.C.15` and `M11.C.39` — built 2026-10-17 under the owner's instruction "proceed using your own recommendations" and reviewed by the owner one decision at a time the same day, the nodal-event spike of `M4.C.1c` added at that review (plan §4.12); `M11.C.17` and `M11.C.6` held for a diagnosis, `M11.C.17` passing since the spike and `M11.C.6` restated, failing, accepted as a finding by the owner 2026-10-18; every record regenerated 2026-10-18; **version 1 complete**, `docs/version_1_status.md`, frozen at revision 32 and the tag `version-1`; **version 2 begins at revision 33** (version 2 plan Q9): revision 33, version 2 stage 1 — `M13.2` amended for the grid engine's retirement and `M8.5` amended so that presence decides whether a triangle is active — decided by the owner 2026-10-19 (`docs/implementation_plan_version_2.md` D1, D3) and built 2026-10-19; revision 34 — every quotation of the 1979 tapes and the Kerr–Bowen interviews paraphrased and cited, as those are ASR transcripts (`docs/PROPOSAL — ASR QUOTATIONS (SP-1).md`, its §4 texts approved and its eight questions answered by the owner 2026-10-22: `M11.F.3` restated from *Family Evaluation*'s Epilogue, `M4.D`'s *Family Evaluation* quotation traced to `FE03.21`), and the stage 1 milestone review's wording fixes to `M2.A.0i`, Teodor's row, `M8`, `M11.1`, `M11.4` and `M13.1` (owner decisions D3, D4 and D6, 2026-10-21) — built 2026-10-22; nothing a run computes changed)
