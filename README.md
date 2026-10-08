# EPModel

A simulation of emotional process at unit, family and societal level, built on Bowen family systems theory.

> **The project is mid-pivot.** The original engine is a stress field over 10,000 units on a 100×100 grid. It is being replaced by a model of **behaving agents in one three-generation family**, because the grid substrate cannot express the part of the theory the project now wants — there is no dyad in it, and therefore no triangle. The grid engine still runs and its tests stay green until the replacement passes its acceptance criteria.

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

**Two results from that work govern everything else in this repo:**

1. **The corpus contains no validation of any kind.** No instrument, no rater procedure, no comparison group, and no number ever assigned to a person, in any of the six sources. It supports *directions, orderings and mechanisms*, and almost no *magnitudes*. Every constant in the model is invented, and the source cannot narrow the range even in principle. *(The 1988 book supplies twelve stated bounds and shapes — recorded at `M10.C.4` as checks on an output, never as parameters.)*

2. **The first pass over the corpus over-read it, consistently in one direction** — making the source look more quantitative and more decided than it is. Nineteen findings were withdrawn on the second pass, including two numeric "calibration targets" that turned out to be manufactured from illustrations Bowen explicitly bounded. Every claim in the explainer is therefore graded, so an invented constant can never be mistaken for a sourced one.

3. **The model is not a fortune teller, and the theory itself is what forbids it.** It answers *"if Bowen's account is right, what follows for a family shaped like this?"* — never *"what should this family do?"* Nearly every place you would want a lever, the corpus says the lever does not work: management technique has zero independent effect while marital distance is high; help relocates incidents without reducing them; curing a symptom raises conflict. **A model faithful to this corpus is anti-lever by construction.** The guards are `M11.F.9` and `M15.D`; the reasoning is `model_explainer.md` §17.

## Layout

```
src/engine.py        the grid engine — frozen in behaviour, still under test
src/main.py          orchestration, UI, I/O
src/bowen/           the agent model (not yet started)
tests/               37 tests, all passing
docs/theory/         the corpus extraction, ledger and convergences
docs/                proposal, explainer, v2 spec, frozen v1.2 spec
```

## Running

```bash
python3 -m pytest tests/
```

```bash
python3 src/main.py
```

Requires Python 3, NumPy and Pygame. `requirements.txt` is present but not yet tracked — see `TODO.md`.

## Status

The v2 specification is **approved** — 542 numbered requirements over 17 modules, 45 acceptance criteria, at revision 12 (approved 2026-10-07).

**Phase B is built and closed** (`src/bowen/`): the objects, the weekly loop, the standing load, the base appraisal, the event record, the scripted source, the run log and the renderer. See `docs/phase_b_completion_report.md`.

**Phase C is built, and its acceptance gate does not pass** (2026-10-08). Agents now select acts through a policy and learn from felt relief. Of 16 criteria built, 7 pass and are mutation-proved, 7 fail, 1 passes in one of its four cells and 1 is undetermined. The status of each criterion and the owner decisions it needs are in `docs/phase_c_completion_report.md`; coverage in `docs/spec_coverage.md`.

The criteria run outside the default suite, over ensembles. On 10 cores the ensemble record takes about 5 minutes, the mutation record about 15 and the sweep about 20:

```
python3 tools/ensemble_record.py     # docs/phase_c_ensemble_record.md
python3 tools/mutation_record.py     # docs/phase_c_mutation_record.md
python3 tools/sweep_record.py        # docs/phase_c_sweep_record.md
```

```
python3 -m src.bowen.run --seed 7 --log runs/phase_b.jsonl --trace runs/phase_b_trace.md
python3 -m pytest tests/
```

The suite runs in CI on Python 3.11, 3.12 and 3.13 (`.github/workflows/tests.yml`), installing only `requirements.txt`.

Phase B runs a fixed script over an invented seven-person family. It tests one model claim, as a direction between two runs; nothing it produces is a finding about families. The family-diagram importer (`M15`) is Phase E.
