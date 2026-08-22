# Changelog

All notable changes to this project are recorded here. Dates are ISO.

## [Unreleased] — 2026-08-22

The corpus-alignment batch. All 22 chapters of *Family Therapy in Clinical Practice* have now been read twice and the project's documents have been brought into line with what the source actually says.

### Added
- **`docs/model_explainer.md`** — every object, field, move, gate, invariant, clock and test in the agent model: what it is for, what it does, and which chapter it implements. 202 claims, each graded `[T]` textual / `[M]` mechanism / `[D]` direction / `[#]` stated quantity / `[I]` invented / `[X]` contested.
- **`docs/bowen_agent_model_spec_v2.md`** — the v2 specification, v2.0-draft, **awaiting approval**. 158 numbered requirements over 14 modules; 15 acceptance criteria and 8 engineering criteria, each with a named test and a mutation target; four criteria flagged as not code-testable in Phases B–D with a human-review proposal each.
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
- Two flaky tests, `test_spouse_dysfunction_asymmetric_penalty` and `test_triangle_mechanism_attaches_and_releases_circle`. Each constructed an unseeded `Simulator` and asserted on a family whose randomly drawn size did not satisfy the code path's precondition; the first was measured failing 6 times in 12 runs, with the asymmetry it names never exercised on a failing draw. Verified 36 passed across 12 consecutive full-suite runs.
- `.gitignore`, whose last line read `.DS_Store (for Mac users)` — a literal filename, so `.DS_Store` was never ignored.
- Spec-internal ID drift: invariants were defined as `I1` and referenced as `M6.I1`, criteria as `C4` and referenced as `M11.C4`. A traceability scan would have resolved 19 of 216 references to nothing and reported a lower count rather than an error.

### Withdrawn
Five mechanisms proposed on the first reading and removed on the second, recorded rather than deleted because a merely-absent mechanism gets re-invented: the permission scalar, ally cancellation, the durable projection target queue, distance as a depletable relief stock, and "content is noise" as an argument for anything.
