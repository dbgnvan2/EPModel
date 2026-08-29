# TODO

Deferred and adjacent items. Each carries enough context to act on later without the conversation that produced it.

## Blocking the next phase

- [x] ~~**Approve or revise `docs/bowen_agent_model_spec_v2.md`.**~~ *Approved 2026-08-25; revisions 1–6 applied, the last on 2026-08-28.*
- [ ] **Write the implementation plan.** The next gate, and the only thing between here and code. Every acceptance criterion mapped to the file and module that satisfies it, in dependency order, with the mutation that proves each one. Spec convention is spec → plan → build, each approved before the next. Inputs: the 33 criteria in `M11.C`, the 5 in `M16.F`, the engineering criteria in `M11.D`, and the phase table at `M13`.
- [ ] **Decide four criteria that cannot be made code-testable in Phases B–D** (spec §M11.E). Each has a human-review proposal:
  - `M11.C.8` (endogenous incidence vs published rates) — needs Phase E and an editorial call on which sources are genuinely exogenous.
  - `M11.C.9` (sibling position) — the effect size is invented, so the test can only assert "a detectable difference", which is close to unfalsifiable. Decide whether it belongs in the suite or should be demoted to a readout.
  - `M11.C.11` shape — "three phases, not a step down" is a claim about curve shape; any automated version embeds an invented tolerance.
  - `M5.F.2` — the threshold separating a counterfeit move from a genuine one is invented and sets the result. **This is the model's most consequential invented constant.**

## Arising from the 2026-08-28 batch

- [ ] **Audit the spec for requirements sourced from an extraction *heading* rather than a quoted sentence.** `M5.D.4` was inverted for exactly this reason: `docs/theory/ch13.md` carried the correct quotation in its body under a heading that said the opposite, and only the heading travelled into the spec. The heading is fixed and the requirement corrected, but **nothing has checked whether there are others of the same shape.** The check is mechanical: for each requirement citing a chapter, confirm the claim appears in a *quoted* passage in the extraction, not only in a section title or a bolded gloss. Highest risk in the earliest requirements, written when the extractions were newest.
- [ ] **Two changes to the family-diagram application, both needing information the diagram does not currently hold.** Neither can be recovered afterwards, which is why `M13.3` names them as the ones to make first.
  - `M15.A.4` — export the **interval** a rater would defend, with the scale point defined, not a bare `3` on a 1–5 scale. The model divides by, differences and thresholds these values; a rank supports none of that.
  - `M15.C.2` — split the diagram's single **"distant"** line into *rupture* and *resolved low-contact*. `M1.B.3` calls telling those apart the most consequential discrimination in this part of the theory: same contact frequency, opposite bond energy. Until the app emits it, bond energy on any distant tie imports as a free range.
### Deferred from the 2026-08-28 re-sweep — graded below medium, backlogged rather than fixed

The re-sweep of the fix commit returned fifteen findings. Four high and seven medium were fixed; these four
were graded low and are recorded instead, because each fix is new unreviewed surface and the project's own
rule is to bound the loop rather than trade one defect class for another.

- [ ] **`test_claimed_test_count_matches_the_suite` counts definitions, not collection.** It walks the AST for
  `def test_*`. `@pytest.mark.parametrize` multiplies collection, `*_test.py` files fall outside the glob, and
  a `skip`/`xfail` keeps the guard green while `CLAUDE.md`'s claim — "all N tests **pass**" — is false. Today
  the three numbers agree exactly (49 defined, 49 collected, 49 passed). Fix by shelling to
  `pytest --collect-only -q`, or state the assumption in the docstring.
- [ ] **`docs/theory/` is outside the reference guard, and has a dangling reference.**
  `family_evaluation/fe03.md` cites an `M0` requirement numbered **one past the last one §0 defines** (§0 runs
  `M0.1`–`M0.4`). `fe03.md` and `fe08.md` also cite `M1.E.7c`'s numbered sub-forms as though they were IDs —
  the spec's own uses of that notation were corrected at revision 5, the extractions were not. Adding the
  directory to `CITING_DOCS` turns the suite red until all three are fixed, so do both together.
  *(The literal tokens are deliberately not written out here: `TODO.md` is inside `CITING_DOCS`, so quoting a
  dangling ID as an example makes the guard fail on the note describing it. The guard is right to be strict —
  an exclusion mechanism would be the loophole that later hides a real one.)*
- [ ] **`_REF` truncates compound references.** `M7.D.2c/2d` yields only `2c`; `M1.D.7i–l` only `7i`. Neither
  is currently dangling, so nothing is wrong today — but the scan is narrower than it reads, and
  `_STATUS.md` describes it without that qualifier.
- [ ] **Name the stressor per criterion when the tests are written.** `M11.3` now places the obligation on the
  test rather than the criterion's prose, which is honest but defers it: 27 of 33 criteria are unclassified
  as discriminating or not. Classify them when each test is written, and consider a marking column then —
  `M11.3` originally required one and the table has no such column.

- [ ] **Run a cold sweep over this batch.** The 2026-08-28 sweep was warm — the same session that wrote the documents commissioned it. It found fifteen defects, which says the sweep works, not that the set is clean. The project's own rule is that a falling finding count from self-review is not a stopping condition, and that a pass which does not know the change's history changes the *distribution* of findings. `/csdp --cold-sweep` over `4e79697..HEAD` from a fresh session.
- [ ] **The sweep covered one failure family only.** It reports itself as not covering: theoretical fidelity (it did not open `_LEDGER.md`, the `family_evaluation/` extractions, or Ch13's primary text), whether the ~88 new requirements are individually implementable, whether the specified model is coherent, and the reasoning in the DECISIONS document. **The `M5.D.4` correction rests on one re-read of one chapter by the context that made the correction** — a second reader on the sources is the highest-value follow-up.
- [ ] **Decide whether `M16`'s renderer output format is worth fixing in the spec.** `M16.C.2` says what a rendered line must carry and points at the proposal's §4.2 table as the shape, but does not fix a format. That is deliberate for now — the format is cheaper to settle against a running Phase B than in advance — but it should be settled *before* a second consumer exists, or the two will drift.
- [ ] **`M11.F.9(c)` deserves a worked demonstration before anyone relies on it.** The claim is that fitting one arm of a counterfactual to a known history breaks the error cancellation `M0.4` depends on. It is argued, not demonstrated. Phase E could show it directly: take a synthetic family, fit one arm to its own history, and measure how far the counterfactual moves against an unfitted control. If the effect is small the requirement is over-strong; if it is large it is the most important sentence in `M11.F`.

## Known defects in the frozen grid engine

Recorded, not fixed. The engine is frozen in behaviour; these matter because the v2 spec forbids inheriting them.

- [ ] **Engine purity violated.** *(Partially contained 2026-08-22: the test suite now runs from a temp directory, so it no longer writes into the repo root. The engine still writes.)*  `Simulator.__init__` opens and writes `sim_audit.csv` in the process working directory, and `log_telemetry` appends every 10 cycles. `CLAUDE.md` says the engine stays pure — no UI, no file I/O. Now gitignored so the stray copies stop appearing as untracked, but the write itself remains.
- [ ] **`_apply_config` fails silently.** It skips any line failing its regex and any key not already in `defaults`, with no warning, so a typo in `docs/model_config.md` vanishes and the default is used. Spec `M10.B.2` forbids v2 inheriting this; the v1 engine still has it.
- [ ] **Magic literals.** Spec §14 of the frozen document claims no model parameters are hard-coded in engine source. In practice 12 keys are in markdown config, 27 are class constants, and roughly 52 distinct float literals remain inline in executable code.
- [ ] **`family_ids` and `nuclear_family_id` are the same array object**, not a copy — `sim.family_ids is sim.nuclear_family_id` is `True`. The frozen spec documents `family_ids` as a "compatibility alias", so this is intentional, but two consequences are not obviously intended and are worth checking before any further engine work: in `update_triangles` the pair `self.family_ids[circle_idx] = fid` / `self.nuclear_family_id[circle_idx] = fid` writes the same element twice, so the first line is dead; and the triangle release path at lines 324/327 restores `family_ids` and thereby silently rewrites `nuclear_family_id` too. Any future change that gives the two names different values will silently lose one. **The v2 spec must not carry an alias pair like this** — `M1.A` gives each field one name.
- [ ] **`update_m` windfall can target a dead unit.** It picks `np.random.randint(0, num_units)`, so it can multiply the resources of a dead unit or an unused nursery slot.

## Repository hygiene

- [ ] **Untracked files predating this batch need a decision.** Not staged, because they were not touched in this work: `requirements.txt`, `monte_carlo.py`, `monte_carlo_results.txt`, `data/`, `gemini.md`, `docs/background.md`, `docs/constitution.md`, `docs/logic_rules.md`, `docs/spec.md`, `docs/user_stories.md`. **`requirements.txt` is the urgent one** — the README tells people to install dependencies and the file is not in the repo.
- [ ] **No CI workflow.** The project standard is that a repo with a test suite pushed to GitHub gets `.github/workflows/tests.yml` — checkout, install from the real dependency file, run the suite on each supported Python version. The value is the blank machine: it catches what "works on this Mac" hides, which matters more here because the repo is shared between two Macs. Blocked on `requirements.txt` being tracked.
- [ ] **No `LEARNINGS.md`.** The project convention is a repo-local fix log for repo-specific lessons, with generic patterns going to `~/.claude/standards/learnings.md`. Candidates: the ID-drift incident recorded in `M11.D.10`; the two unseeded flaky tests; and from 2026-08-28, **the `M5.D.4` heading inversion** — a requirement sourced from a summary heading rather than the sentence beneath it, which inverted silently and survived four spec revisions. The last of those may be general enough for the global catalogue rather than the repo-local log.
- [ ] Stray file `docs/model_explainer copy.textClipping` in the working tree — not mine to delete, but it is clutter.

## Documentation hygiene, deferred

- [ ] **`§n` cross-references are ambiguous across documents.** All 50 in the v2 spec resolve against `model_explainer.md`, but several sentences address other documents ("proposal §9", "the frozen spec's §5.4") and `§5.4` exists in both. A traceability scan over `§` cannot tell the targets apart. Prefix them with the document.

## Sources

- [ ] **Locate the rest of the Kerr–Bowen interview series.** The folder holds #1 of a series; Kerr closes by saying later tapes "will concentrate on some of the more specific areas of the theory". Bowen names two he wanted covered: the difference between *distance* and *differentiation*, and how much of behaviour is intellectually directed versus emotionally reactive. An interview format with Kerr probing specific concepts is the highest-value form this material could take.
- [ ] **Probe the remaining 11 validation items against the 1979 lectures.** Tape 6 especially — it is the direct counterpart to Ch15, where this project withdrew both the seven-rung severity ladder and the "hidden dependence network", and a whole lecture on family reaction to death is the natural place to test both withdrawals.

## Carried from the theory work

- [ ] **`M11.C.9`'s and sibling position's status generally.** Ch13 omits sibling position entirely, and its effect size has no source. Consider whether it earns a place in the model at all.
- [ ] **Convergence C1 needs its scope re-derived from the chapter texts**, not from the pass-1 summaries. It was the headline finding at eleven chapters; Ch22 supplies a direct counterexample, so it likely survives only as a claim about *symptom relief obtained without structural change*.
- [ ] **The ego-mass terminology arc must be re-derived.** Ch08 actively retains and defends "undifferentiated family ego mass" against the timeline's claim that it was discarded at Ch05. The arc is messier than used → discarded → revived → abandoned.
- [ ] **When the basic level is fixed is genuinely unresolved** (`model_explainer.md` §13.3). Decided for the model — slow-moving with a ratchet, not frozen — but that is a modelling decision over a real disagreement, not a resolution of it.
