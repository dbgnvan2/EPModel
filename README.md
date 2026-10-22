# EPModel

A simulation of emotional process in one three-generation family, built on Bowen family systems theory.

The model is a set of **behaving agents in one three-generation family** (`src/bowen/`). It replaced the project's first engine, a stress field over 10,000 units on a 100×100 grid, because the grid could not express the part of the theory the project wants — there is no dyad in it, and therefore no triangle. The grid engine was retired at version 2 stage 1 (spec revision 33, 2026-10-19); its code and tests are kept at the tag `version-1`.

> **Which branch to read.** GitHub's default branch `main` carries the current work. Until 2026-10-21 it stopped at 2026-10-06 (spec revision 11) while the work went on the branch `plan-phase-b`; that day `main` was fast-forwarded to it, and new work goes on `main`. The complete, frozen version 1 is at the tag `version-1`.

## Where to start

| If you want to | Read |
|---|---|
| understand **what the model is and why** | [`docs/agent_model_proposal.html`](docs/agent_model_proposal.html) |
| understand **every part of the model**, without reading Bowen | [`docs/model_explainer.md`](docs/model_explainer.md) |
| **build it** | [`docs/bowen_agent_model_spec_v2.md`](docs/bowen_agent_model_spec_v2.md) |
| check **what the source actually says** | [`docs/theory/_LEDGER.md`](docs/theory/_LEDGER.md) |
| know **where the work is up to** | [`docs/theory/_STATUS.md`](docs/theory/_STATUS.md) |

## The theory work

All 22 chapters of Bowen's *Family Therapy in Clinical Practice* have been extracted twice — a cold pass chapter by chapter, then a comparative pass re-reading each chapter against the whole book. The output is in `docs/theory/`: 149 per-chapter findings, ten cross-chapter convergences with an independence audit, and a resolution document for the contradictions.

**Three results from that work govern everything else in this repo:**

1. **The corpus contains no validation of any kind.** No instrument, no rater procedure, no comparison group, and no number ever assigned to a person, in any of the six sources. It supports *directions, orderings and mechanisms*, and almost no *magnitudes*. Every constant in the model is invented, and the source cannot narrow the range even in principle. *(The 1988 book supplies twelve stated bounds and shapes — recorded at `M10.C.4` as checks on an output, never as parameters.)*

2. **The first pass over the corpus over-read it, consistently in one direction** — making the source look more quantitative and more decided than it is. Nineteen findings were withdrawn on the second pass, including two numeric "calibration targets" that turned out to be manufactured from illustrations Bowen explicitly bounded. Every claim in the explainer is therefore graded, so an invented constant can never be mistaken for a sourced one.

3. **The model is not a fortune teller, and the theory itself is what forbids it.** It answers *"if Bowen's account is right, what follows for a family shaped like this?"* — never *"what should this family do?"* Nearly every place you would want a lever, the corpus says the lever does not work: management technique has zero independent effect while marital distance is high; help relocates incidents without reducing them; curing a symptom raises conflict. **A model faithful to this corpus is anti-lever by construction.** The guards are `M11.F.9` and `M15.D`; the reasoning is `model_explainer.md` §17.

## Layout

```
src/bowen/           the agent model: engine, policy, readouts, ensemble runner, renderer
tools/               the record generators (ensemble, mutation, sweep, Phase D, trace)
tests/               the suite (see CLAUDE.md for the current count)
docs/theory/         the corpus extraction, ledger and convergences
docs/                proposal, explainer, v2 spec, frozen v1.2 spec
```

## Running

```bash
python3 -m pytest tests/
```

Requires Python 3.11 or later and the packages in `requirements.txt`. The grid engine's entry point,
`src/main.py`, was retired at version 2 stage 1 and is at the tag `version-1`.

## Status

**Version 1 is complete and frozen** at spec revision 32 and the tag `version-1` (2026-10-19). Where each criterion stands is in `docs/version_1_status.md`. **Version 2** (`docs/version_2_plan.md`) began at revision 33: stage 1, one engineering batch with no new rule (the grid engine retired, exact float totals, the act classes in config, the Phase D records cached), was built and every record regenerated 2026-10-19 (`docs/implementation_plan_version_2.md` §1.10); no verdict changed (`docs/version_2_stage_1_changes.md`). Revision 34 (2026-10-22) changed wording only: the quotations of the 1979 tapes and the Kerr–Bowen interviews, which are ASR transcripts, are paraphrased and cited (SP-1), and the stage 1 milestone review's wording fixes are made; nothing a run computes changed. The paragraphs below record each phase as it was built; the status page says where each criterion stands now.

**Phase B is built and closed** (`src/bowen/`): the objects, the weekly loop, the standing load, the base appraisal, the event record, the scripted source, the run log and the renderer. See `docs/phase_b_completion_report.md`.

**Phase C is built, and its acceptance gate does not pass** (2026-10-08). Agents now select acts through a policy and learn from felt relief. Of 16 criteria built, 7 pass and are mutation-proved, 1 passes without a mutant proof, 1 passes in one of its four cells, and 7 fail. The status of each criterion and the owner decisions it needs are in `docs/phase_c_completion_report.md`; coverage in `docs/spec_coverage.md`.

**Phase D, part I, was built** (`docs/implementation_plan_phase_d.md`): engine speed, channel stamping,
cutoff as rejection, the slow tick and the basic-level estimator (B1–B6); the three symptom channels with lock-in (B7);
beliefs as a channel (B8); the tie-weighted cross-person term (B9); the `M11.G` family-evaluation readout (B10); and
borrowed pseudo-self settling back. Spec revisions 24–27 were Claude's recommendations; the owner reviewed them on
2026-10-16. Criteria, as recorded on 2026-10-13,
from `docs/phase_d_ensemble_record.md`: `M11.C.17` (weakly), `.18`, `.20`, `.33`, `.30`, `.22`, `.31`,
`.11`, `.26`, `.36`, `.39` and `.24` pass; `.12`, `.14`, `.2`, `.40`, `.43`, `.34`, `.21` and `.23` fail. `M11.C.7`,
`.9` and `.28` need mechanisms first; who acts (B11) moves to part II.

**Phase D, part II (the life course, spec revision 28) is built**: ages, childhood, launch, partners and births,
death and its disposition, lost income and finding work, and the coach's successor, over a 40-year run. From
`docs/phase_d2_ensemble_record.md`: `M11.C.15` passes, and `M11.C.37` passes as a test; `M11.C.6`, `.10` and `.2` on the
twelve-person family fail. Status: `docs/phase_d2_status_2026-10-13.md`.

The criteria run outside the default suite, over ensembles. On 10 cores the ensemble record takes about 5 minutes. The mutation and sweep records cache each result by the engine hash and the mutant's or setting's own edits (`tools/record_cache.py`, `docs/records_cache/`): a regeneration reruns only what changed, so re-rendering takes under a second, one new mutant a few minutes, and only an engine change reruns everything (about 40 and 20 minutes):

```
python3 tools/ensemble_record.py     # docs/phase_c_ensemble_record.md
python3 tools/mutation_record.py     # docs/phase_c_mutation_record.md (cached)
python3 tools/sweep_record.py        # docs/phase_c_sweep_record.md (cached)
python3 tools/level_occupancy.py     # docs/phase_c_level_occupancy.md (about 3 minutes)
python3 tools/c16_report.py          # report §11, generated from the records
python3 tools/phase_d_record.py      # docs/phase_d_ensemble_record.md (cached; a full rerun about 10 minutes)
python3 tools/phase_d_record.py --part 2   # docs/phase_d2_ensemble_record.md (cached; a full rerun about 50 minutes)
python3 tools/phase_d_mutation_record.py   # docs/phase_d_mutation_record.md (cached; one new mutant about 30 minutes)
```

```
python3 -m src.bowen.run --seed 7 --log runs/phase_b.jsonl --trace runs/phase_b_trace.md
python3 -m pytest tests/
```

The suite runs in CI on Python 3.11, 3.12 and 3.13 (`.github/workflows/tests.yml`), installing only `requirements.txt`.

Phase B runs a fixed script over an invented seven-person family. It tests one model claim, as a direction between two runs; nothing it produces is a finding about families. The family-diagram importer (`M15`) is not in version 1 or version 2 (`docs/version_2_plan.md` §2).
