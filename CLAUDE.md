# EPModel — Claude Code instructions

Python + NumPy simulation of emotional process at unit, family, and societal levels.

**The project is mid-pivot.** The grid engine (`src/engine.py`, 10,000 units on a 100×100 lattice) is being
replaced by a behaving-agent model over one three-generation family. Read `docs/agent_model_proposal.html`
before touching architecture — especially §10, which records what reading the source corpus changed.

## Global standards

Read the relevant file from `~/.claude/standards/` before starting work:

| Standard | When |
|---|---|
| `learnings.md` | P4 (hardcoded constants — simulation parameters belong in config), P7 (scoring/ranking proxies), P9 (magic size/count caps in simulation loops), P23 (any stochastic-output regression fixture) |
| `file-maintainability.md` | Any new module or significant refactor |

## Where the theory lives

| File | What it is | Trust |
|---|---|---|
| `docs/theory/_LEDGER.md` | **six corpus sources**, all passes folded in. Per-chapter findings for the 22 papers; consolidated entries for the later sources, each backed by a fuller extraction directory | **the source of truth for what the corpus says** |
| `docs/theory/_CONVERGENCES.md` | ten cross-chapter convergences + independence audit | current |
| `docs/theory/_STATUS.md` | run state, standing decisions, order of work | **read this first** |
| `docs/DECISIONS — FAMILY EVALUATION.md` | how the 4 contradictions and 21 requirement candidates from the sixth source were decided | **closed 2026-08-27 — all applied at spec revision 4** |
| `docs/bowen_agent_model_spec_v2.md` | **the buildable contract** — 381 numbered requirements, the update order, the parameter register, 33 acceptance criteria, the `M11.G` readout schema, the `M15` import contract | **v2.0-draft revision 5, approved 2026-08-25 — no code until the implementation plan is also approved** |
| `docs/model_explainer.md` | **every part of the model** — object, field, move, gate, invariant, clock, test — what it is for, what it does, and which chapter it implements | **start here to understand the model** |
| `docs/agent_model_proposal.html` | the architecture proposal, §10 = corpus reconciliation | current |
| `docs/bowen_individual_family_model_spec.md` | **v1.2, FROZEN** — describes the grid engine only | historical record; do not extend |

**Six corpus sources, and they are not one evidence class.** The 22 papers (1957–76, `L`); the 1979
video lectures (`Tape N`, **ASR — no verbatim quotation**); the Kerr–Bowen interviews (`KB`, **ASR — cite
the interview, not the man**); Kerr & Bowen, *Family Evaluation* (1988) — `FE`, written, and its
**Epilogue is Bowen's own, the latest primary Bowen text in the project**; Kerr, *Bowen Theory's
Secrets* (2019) — `KS`, written, some chapters `[K-ext]`; and the DSI literature. **Every claim carries
an attribution grade as well as an evidence grade** — `[B]` Bowen, `[K]` Kerr, `[K-ext]` Kerr's own
extension, which **must never be attributed to Bowen**. `model_explainer.md`'s citation legend has the
full table.

**Two rules that follow from the corpus and are easy to violate by accident:**

- **Every claim in `model_explainer.md` carries a grade** — `[T]` textual, `[M]` mechanism, `[D]` direction,
  `[#]` a stated quantity, `[I]` invented, `[X]` contested. **If a constant in the code is presented as
  sourced and does not appear there, treat it as `[I]` until proven otherwise.**
- **The corpus supports directions, orderings and mechanisms — almost no magnitudes.** There is no
  instrument, no rater procedure, no comparison group and no number assigned to a person anywhere in the
  22 chapters. Never present an invented constant as sourced. If a parameter needs a citation, check it
  against `_LEDGER.md` first; two of the numbers previously treated as calibration targets turned out to
  be manufactured from illustrations Bowen explicitly bounded.
- **Acceptance tests assert a direction of difference between two arms, not an absolute threshold** — and
  each must be proved failing by mutation before it counts as coverage.
- **The model must never be presented as speaking about a real family** (`M11.F.9`). In particular: a run
  whose parameters were tuned to reproduce a known history is a **fit**, not a comparison — fitting one arm
  breaks the error cancellation the two-arm design depends on, on one side only and invisibly. Initial
  conditions are not nuisance parameters; they are what the intervention acts on, so their error flips the
  sign at a regime boundary rather than cancelling. Imported families run as **ranges**, and the output is an
  **envelope** (`M15.D`).

## Key rules

- Simulation parameters (thresholds, rates, constants) belong in config or named constants — not magic
  literals scattered through `engine.py`.
- Keep the simulation engine pure: **no UI, no file I/O.** This is currently violated —
  `Simulator.__init__` opens and writes `sim_audit.csv` in the process working directory, and
  `log_telemetry` appends to it every 10 cycles. Two stray copies of the file exist (repo root and `src/`).
  Fix this when the engine is next touched; do not add new I/O to it in the meantime.
- **Axiom 1's no-loops / vectorisation mandate is obsolete and now pushes the design the wrong way.** It
  was written for N = 10,000. The agent model runs N ≈ 12, where a 40-year history is roughly 2,080 fast
  ticks — clarity beats vectorisation at that scale, and per-agent, per-tie and per-triangle state cannot
  be expressed as a dense array without exactly the flattening the pivot exists to undo. Treat Axiom 1 as
  applying to the frozen grid engine only. It does not govern `src/bowen/`.
- New work goes in a new package `src/bowen/` alongside `engine.py`. The old engine and its 36 tests stay
  green until the new one passes the acceptance tests.
- Run tests before declaring done: `python3 -m pytest tests/`
- **A red suite is never normal — all 36 tests pass.** The previously documented flake in
  `test_spouse_dysfunction_asymmetric_penalty` was fixed on 2026-08-22 (measured 12/12, plus 20 clean
  full-suite runs); `test_triangle_mechanism_attaches_and_releases_circle` had the same latent fault at
  a ~2.7% rate and was fixed with it.
- **Family sizes are drawn at construction, so any test with a family-size precondition must fix the
  draw.** `Simulator.__init__` tiles the grid with size-2 and size-4 blocks at random: the family
  containing unit 0 is size 2 or size 4 with roughly equal probability, and ~2.7% of draws contain no
  size-2 family at all (measured over 2,000 draws). `apply_divorce` and `_update_family_distance_flags`
  act only on families with **exactly two active members**, so a test that picks a family arbitrarily and
  expects those code paths to fire silently no-ops on half its runs. Use the `_seed_rng` helper in
  `tests/test_simulator.py`, which seeds the global NumPy RNG and restores the prior state on cleanup so
  later tests keep varying inputs, and select the family by `family_member_counts == 2` rather than by
  `family_ids[0]`.
