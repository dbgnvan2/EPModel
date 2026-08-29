# Changelog

All notable changes to this project are recorded here. Dates are ISO.

## [Unreleased] — 2026-08-28

The sixth corpus source decided and applied, plus two modules that came out of working through what the model may and may not be asked. Documentation only; no code. Spec revisions 4, 5 and 6.

### Fixed

**From a failure-pattern sweep of the batch, 2026-08-28 — fifteen findings, all verified against the files and all fixed.** Four were high:

- **`M16.B.3` required logging to be a pure observer, contradicting `M16`'s own preamble twelve lines above**, which makes an agent's delayed log an *input* to a mechanism. Worse, its test gated **only Phase B** — the phase with no policy and no external agent, where it passes vacuously — and was not re-run in Phase C where `M16.D` makes it false. Split into an **event store** (always present, causally load-bearing) and a **persistence sink** (optional, pure); `M16.T.3` re-listed in Phases C and D, and `M16.T.6` added as its converse.
- **`M11.2` bounded ensemble testing at `M11.C.16`** while the table grew to 33, leaving seventeen criteria — including three distributional claims a single run cannot establish — outside a bound nobody had moved. Restated over the whole set.
- **`M11.3` shipped with eight counterexamples in the same commit.** It required every criterion to name a stressor; 8 of the 9 criteria added alongside it name none, and `M11.C.31` *requires a calm family by construction*. Scoped to discriminating criteria, with the conservation, identifiability and null classes named as outside it.
- **Seven `M11.C` criteria and `M16.T.4` carried a phase and gated nothing**, so Phases C and D could each be declared done with them unwritten — among them `M11.C.17`, which the spec calls "the model's strongest negative test", both estimator discriminators, and the only guard stopping an agent reading another agent's private hops. All eight listed; `M13.2a` and `M11.D.11` make the correspondence mechanical.

And eleven more: `M10.C.4`'s twelve magnitudes had **no permitted home** — forbidden in config, underivable, and `M11.D.2` requires one or the other, so the violating route was also the compliant-looking one (`M10.C.4b` names a checks module; `M11.D.12` guards it). **Nine prohibitions were written `No X MUST Y`**, which under §0.3 states a permission — including `M11.F.9(c)`, the clause the document marks as correctness rather than framing (`M11.D.13` now guards the form). §0.3 claimed every MUST carries a test: 616 MUSTs against 44 tests, restated honestly. `M0.4` pointed at an exception list that did not exist (`M11.4` now is one). `M11.F.5` never existed and the gap read as a withdrawal. `M11.C.25` and `M11.C.17` are nulls that pass when underpowered (`M11.4a`). `M16.C.3`'s rationale was stale inside its own commit. `M1.F.9` never named `binder_unavailable`. And the published proposal carried revision 4's requirement count through revisions 5 and 6 while `_STATUS.md` asserted the set was consistent.

- **`M11.F.9(c)`'s stated mechanism was wrong, though its conclusion stands.** It said fitting absorbs the model's error "on one side only"; under `M0.4` both arms share one parameter set, so nothing is absorbed asymmetrically. The real failure is **loss of independence, non-exchangeability across a regime boundary, and non-identifiability** — 60–90 free constants against a single realisation, so many parameter sets reproduce the history and disagree about the counterfactual, making the bias in the difference *unbounded* rather than merely unknown. A stronger objection than the one it replaced. The line "the better the fit, the less the counterfactual can be trusted" is now scoped to the overfitting regime, where it holds, rather than stated as a general law, where it does not.
- **`M5.D.4` was inverted, and had been since it was written.** The requirement said *anger* is the gate admitting the `I-POSITION` sequence to its peak. Ch13 says the opposite — "when he is finally able to maintain his course **without getting angry** at the opposition, the opposition does a final intense emotional attack" — and mentions anger exactly once in the whole chapter. `docs/theory/ch13.md` had it right in the body and wrong in the **heading** above it ("Anger gates the escalation"), and the heading is what reached the spec. Corrected in all three places, with `M11.C.32` added to hold it. Kerr 1988 (`FE04.4`) and Bowen 1988 (`FE11.19`) both agree with the chapter, about the same person's anger.
- **A phase exit condition that could not be checked.** Phase B was done when "the event log reads correctly" — an adjective naming no requirement, no test and no artifact, in a table where every other condition names one. It now names `M16.C.2` and `M16.T.1`.
- **Module ordering.** `M14` had ended up after `M15`.

### Added
- **`tests/test_spec_consistency.py`** — ten guards over the document set, each mutation-proved by reverting the fix it protects and confirming red. They assert what a failure-pattern sweep had to find by hand: IDs unique and machine-extractable, every cross-reference resolving in seven documents, no prohibition written in the inverted form §0.3 reads as a permission, every criterion gating the phase its own table row names, `M11.2` stated over the whole criteria set rather than a numeric range, and the requirement and suite counts in `README`/`CLAUDE.md`/`_STATUS.md`/the proposal reconciling to the artifacts they describe. They read documents only — nothing imports `src/`.
- **`M15` — the family-diagram import contract.** A Phase E capability specified early, because the diagram application that would feed it is under the project owner's control and its export format is cheaper to fix before it exists than after. Structure imports as **values**; ratings import as **ranges**; the readout is an **envelope**, never a point direction. Four adapter traps named, including a warmth rating wired to a valence-blind `investment` (which would invert the sign on exactly the conflicted families the model is for) and a genogram's single "distant" line, which covers both a rupture and a resolved low-contact tie — the discrimination `M1.B.3` calls the most consequential in this part of the theory. Fixture mode included: anonymised real topologies beat hand-built ones wherever a criterion depends on asymmetry.
- **`M16` — the run log and the readable trace.** Six requirements already depended on a log and none of them said what one is. Effects and the selection rationale are recorded beside the events, so the log is a causal trace rather than a list; belief writes are tagged apart from ground truth; the engine emits and the caller persists; logging is a pure observer. A **deterministic renderer** — template, no language model — is built in Phase B, and a per-agent **delayed view** serves the fourth form of a landed coaching contact, defaulting to the corpus's six months.
- **`M11.F.9`** — three clauses on speaking about a real family. The third is a *correctness* requirement, not a framing one: no counterfactual may be reported from parameters tuned to reproduce a known history, because fitting one arm breaks the error cancellation the two-arm design depends on, on one side only and invisibly.
- **~20 requirements from *Family Evaluation***, the largest being that the belief layer becomes a **channel** (`M9.6`, `M9.7`) rather than a parallel store — chronic anxiety runs on what *might be*, and Bowen's own correction of "projection" has the transfer running through descriptions.
- **`M10.C.4`** — twelve magnitudes stated in the corpus, admitted as **checks on an output** and never as parameters, with `M10.C.3a` clarifying that the existing prohibition is on the parameter and not on the quantity.
- **`M11.G`** — the eight-component family evaluation readout, with the source's ninth and tenth components (therapeutic focus, prognosis) explicitly barred.
- **Nine acceptance criteria**, `M11.C.25`–`M11.C.33`, and five for the log, `M16.T.1`–`M16.T.5`.
- **`docs/model_explainer.md` §17** — what the model can be asked, and the two ways the answer goes wrong. Marked as method rather than corpus.

### Changed
- **`docs/DECISIONS — FAMILY EVALUATION.md` closed.** All nine items decided 2026-08-27; the file is now the record of how each was decided.
- **`M1.A.11c`** — constitution sets the symptom channel as a *prior*; a family-focus term can shift it; a constitutional-strength term can override the shift. `model_explainer.md` §3.9 is no longer provisional.
- **`M1.A.3b`** — the transition at 50 keeps its licence implementation and gains an awareness *readout*. Neither is reduced cognitive capacity: what differs is the strength of the emotional circuits over the cognitive ones, and a low-level agent argues just as fluently.
- **`M7.A.1a`** — the financial-dependence gate now binds the slow clock, not only the `I-POSITION` move.
- **`M12.2` reviewed and left standing in full.** Kerr 1988 names the *locus* of the first unknown, not the rule; its own hedge is "determined largely by".
- **`M12.5`** — power and punishment barred as mechanisms. The only entry in that module stated as a prohibition by the author rather than inferred by the project.
- Spec is **405 requirements over 16 modules**, 33 acceptance criteria, 0 duplicate IDs, 0 dangling cross-references; every `M…` reference in the spec, explainer, proposal, `CLAUDE.md` and `_STATUS.md` resolves. 37 tests green throughout.

### Learned
- **A summary heading is a lossy artifact, and a requirement sourced from one can invert without anything looking wrong.** The `M5.D.4` inversion above is the worked instance: the quoted sentence was right, the heading over it was not, and only the heading travelled.
- **Initial conditions are not nuisance parameters.** Invented constants cancel between two arms of a counterfactual because they do not interact with the intervention. Initial conditions are *what the intervention acts on*, so their error crosses a regime boundary and flips the sign instead of shifting both arms together. Five such boundaries are named at `M15.D.4`; one of them, `M5.C.1a`, is a SAFETY property with recorded harms. **Direction is the least robust output under mis-specified inputs, not the most.**

## 2026-08-22 — corpus alignment

The corpus-alignment batch. All 22 chapters of *Family Therapy in Clinical Practice* have now been read twice and the project's documents have been brought into line with what the source actually says.

### Added
- **`docs/model_explainer.md`** — every object, field, move, gate, invariant, clock and test in the agent model: what it is for, what it does, and which chapter it implements. 216 claims, each graded `[T]` textual / `[M]` mechanism / `[D]` direction / `[#]` stated quantity / `[I]` invented / `[X]` contested.
- **`docs/bowen_agent_model_spec_v2.md`** — the v2 specification, v2.0-draft, **awaiting approval**. 212 requirement definitions over 14 modules; 16 acceptance criteria and 10 engineering criteria, each with a named test and a mutation target; four criteria flagged as not code-testable in Phases B–D with a human-review proposal each.
- **`docs/theory/_RESOLUTIONS.md`** — the two contradictions carried out of pass 2, both resolved against the primary source.
- **`docs/theory/`** brought under version control: 22 chapter extractions, the ledger, the convergences, the corrections and the status file.
- Ledger entry **L21.11**, surprise as a third secrecy mechanism — present in the source and in neither pass.
- Spec requirement **M11.D.10**, requiring every ID to be machine-extractable.

### Changed
- **`docs/theory/_LEDGER.md`** rewritten and made self-contained: all 109 pass-1 IDs preserved and re-verdicted, 40 new pass-2 entries, all ten correction batches folded in. Withdrawn findings kept as tombstones with their reason.
- **`docs/agent_model_proposal.html`** corrected throughout rather than annotated. The undifferentiation budget has three sinks, not four. Reactivity is derived, not stored. The move repertoire is no longer claimed closed. The event record carries source position, route, fidelity and latency. Acceptance tests: eleven became fifteen. **Phase B's four-person nuclear spike is withdrawn** — cutoff with the families of origin is an *input* to intensity, so a model closed at the household cannot compute its own driving term.
- **`docs/bowen_individual_family_model_spec.md`** frozen as a v1.2 historical record of the grid engine, with a header table of the nine things the corpus contradicts in it. It is not the base for v2.
- **`CLAUDE.md`** rewritten for the pivot. Axiom 1's vectorisation mandate is scoped to the frozen grid engine — at N ≈ 12 it pushes the design the wrong way.
- **`README.md`** expanded from two lines.

### Fixed
- Two flaky tests, `test_spouse_dysfunction_asymmetric_penalty` and `test_triangle_mechanism_attaches_and_releases_circle`. Each constructed an unseeded `Simulator` and asserted on a family whose randomly drawn size did not satisfy the code path's precondition; the first was measured failing 6 times in 12 runs, with the asymmetry it names never exercised on a failing draw. Verified 36 passed across 12 consecutive full-suite runs; the suite is 37 after the working-directory guard below.
- `.gitignore`, whose last line read `.DS_Store (for Mac users)` — a literal filename, so `.DS_Store` was never ignored.
- Spec-internal ID drift: invariants were defined as `I1` and referenced as `M6.I1`, criteria as `C4` and referenced as `M11.C4`. A traceability scan would have resolved 19 of 216 references to nothing and reported a lower count rather than an error.

### Withdrawn
Five mechanisms proposed on the first reading and removed on the second, recorded rather than deleted because a merely-absent mechanism gets re-invented: the permission scalar, ally cancellation, the durable projection target queue, distance as a depletable relief stock, and "content is noise" as an argument for anything.
