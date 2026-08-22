# TODO

Deferred and adjacent items. Each carries enough context to act on later without the conversation that produced it.

## Blocking the next phase

- [ ] **Approve or revise `docs/bowen_agent_model_spec_v2.md`.** It is v2.0-draft. The project's convention is spec → plan → build, each approved before the next, so no code moves until this is signed off.
- [ ] **Decide four criteria that cannot be made code-testable in Phases B–D** (spec §M11.E). Each has a human-review proposal:
  - `M11.C8` (endogenous incidence vs published rates) — needs Phase E and an editorial call on which sources are genuinely exogenous.
  - `M11.C9` (sibling position) — the effect size is invented, so the test can only assert "a detectable difference", which is close to unfalsifiable. Decide whether it belongs in the suite or should be demoted to a readout.
  - `M11.C11` shape — "three phases, not a step down" is a claim about curve shape; any automated version embeds an invented tolerance.
  - `M5.F.2` — the threshold separating a counterfeit move from a genuine one is invented and sets the result. **This is the model's most consequential invented constant.**

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
- [ ] **No `LEARNINGS.md`.** The project convention is a repo-local fix log for repo-specific lessons, with generic patterns going to `~/.claude/standards/learnings.md`. Two candidates from this batch: the ID-drift incident recorded in `M11.D.10`, and the two unseeded flaky tests.
- [ ] Stray file `docs/model_explainer copy.textClipping` in the working tree — not mine to delete, but it is clutter.

## Documentation hygiene, deferred

- [ ] **`§n` cross-references are ambiguous across documents.** All 50 in the v2 spec resolve against `model_explainer.md`, but several sentences address other documents ("proposal §9", "the frozen spec's §5.4") and `§5.4` exists in both. A traceability scan over `§` cannot tell the targets apart. Prefix them with the document.

## Carried from the theory work

- [ ] **`M11.C9`'s and sibling position's status generally.** Ch13 omits sibling position entirely, and its effect size has no source. Consider whether it earns a place in the model at all.
- [ ] **Convergence C1 needs its scope re-derived from the chapter texts**, not from the pass-1 summaries. It was the headline finding at eleven chapters; Ch22 supplies a direct counterexample, so it likely survives only as a claim about *symptom relief obtained without structural change*.
- [ ] **The ego-mass terminology arc must be re-derived.** Ch08 actively retains and defends "undifferentiated family ego mass" against the timeline's claim that it was discarded at Ch05. The arc is messier than used → discarded → revived → abandoned.
- [ ] **When the basic level is fixed is genuinely unresolved** (`model_explainer.md` §13.3). Decided for the model — slow-moving with a ratchet, not frozen — but that is a modelling decision over a real disagreement, not a resolution of it.
