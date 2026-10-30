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
| look up **one requirement** — its first sentence, sources, ledger entries, code, tests and status | [`docs/spec_index.jsonl`](docs/spec_index.jsonl), one JSON line per ID (`grep '"M4.C.3a"'`); [`docs/theory/ledger_index.jsonl`](docs/theory/ledger_index.jsonl) for ledger entries, `_LEDGER.md`'s and the extraction files' per-finding entries (`grep '"FE03.6"'`) |
| read **one module's current rules** without its history | [`docs/spec_rules/`](docs/spec_rules/), one file per module, generated from the spec; where they differ, the spec wins |
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

**Version 1 is complete and frozen** at spec revision 32 and the tag `version-1` (2026-10-19). Where each criterion stood at the tag is in `docs/version_1_status.md`. Where each stands now is in `docs/version_2_status.md`. **Version 2** (`docs/version_2_plan.md`) began at revision 33: stage 1, one engineering batch with no new rule (the grid engine retired, exact float totals, the act classes in config, the Phase D records cached), was built and every record regenerated 2026-10-19 (`docs/implementation_plan_version_2.md` §1.10); no verdict changed (`docs/version_2_stage_1_changes.md`). Revision 34 (2026-10-22) changed wording only: the quotations of the 1979 tapes and the Kerr–Bowen interviews, which are ASR transcripts, are paraphrased and cited (SP-1), and the stage 1 milestone review's wording fixes are made; nothing a run computes changed. The paragraphs below record each phase as it was built. The status page, `docs/version_2_status.md`, says where each criterion stands now. Revision 35 (2026-10-22) moved the spec's revision history into `docs/spec_revision_log.md`, and the inline revision narratives of 129 requirement bodies with it; no rule changed. The owner approved its dispositions (`docs/spec_revision_35_dispositions.md`) on 2026-10-22. Revision 36 (2026-10-22, on the branch `claude/recursing-bohr-2393a1`, since merged into `main`) built version 2's group G and the Holmes–Rahe event sizes, each choice made as recommended under the owner's instruction of 2026-10-22 and approved by the owner 2026-10-24 (T-125). Every record was regenerated once on 2026-10-23 (`docs/version_2_revision_36_changes.md`). Revision 37 (2026-10-23, on the branch `stage-2-plan-pilot`) added stage 2's detection-limit rule and sample rule to `M17.A.1` and `M17.E.1`; nothing a run computes changed. Stage 2's constant sweep ran 2026-10-23 and 2026-10-24 over 34 directional criteria, split across two Macs (`docs/two_mac_sweep.md`). Revision 38 (2026-10-24) changed wording only, from the stage 2 milestone review's triage (`docs/todo/2026-10-24_stage_2_milestone_review.md`); nothing a run computes changed. Revision 39 (version 2 stage 3, approved by the owner 2026-10-25) is applied in full, in parts, each with its code: at step 6 (2026-10-25) T-130's wording and T-014's four answers, the engine hash moved to `f90285984f32` and no result changed; at step 7 (2026-10-26) group C, displaced coping, the floor-held ledger and lent functioning returned with lost capacity, which moved the hash to `b8e9ad1333e2` and leaves `M11.C.6` failing in its targeted run; at step 8 (2026-10-26) group F, an I-position as a mixture of self-statement and demand by the mover's level and the new `M11.C.49`, which moved the hash to `b819660f58b9` and leaves `M11.C.49` and `M11.C.29` failing in its targeted run; at step 9 (2026-10-26) group A, contact returned to the sender by a relieved receiver and `TRIANGLE` placed with distance, which moved the hash to `2a3b84e2d89a` and leaves `M11.C.42` and `M11.C.45` failing and `M11.C.44` no longer passing in its targeted run; at step 10 (2026-10-26) the sweep's named ratio (T-128), with the hash unchanged, and every record regenerated once: `M11.C.2[twelve-person family]`, `M11.C.7[one spouse over sided]`, `M11.C.10[family of origin]` and `M11.C.17` now pass, `M11.C.44` and the new `M11.C.49` fail (`docs/version_2_revision_39_changes.md`). At step 11 (2026-10-27) the stage-end sweep ran all 34 declared directional keys with `M17.E.1`'s named ratio, split across both Macs; the ratio entered the 26 whose run is longer than 52 fast ticks, and nothing a run computes changed (`docs/version_2_sweep_record.md`, `docs/two_mac_sweep.md`). The stage 3 milestone review (`docs/cycles/2026-10-27_milestone-review_v2-stage-3.md`) was triaged 2026-10-28 (`docs/todo/2026-10-28_stage_3_milestone_review.md`): its wording fixes are spec revision 40, with the engine hash unchanged, and its hash-moving fixes are spec revision 41. Revision 41 was built the same day: a target who would refuse a displaced act is not its receiver, and what displaced coping moved is kept on its own ledger and left out of the parents' term of a child's development; anger is read again when an I-position's `DEFINE` is said; lent functioning returns in the week a scripted drop falls into; the engine no longer imports the policy. The engine hash moved once, from `2a3b84e2d89a` to `6ce8be104767`, and every record but the stage-end sweep was regenerated once on it (`docs/version_2_revision_41_changes.md`), and the stage-end sweep was rerun on it 2026-10-29, split across both Macs (T-159): 33 of its 34 keys have a fraction, and `M11.C.48` is refused at the rerun, because it fixes 100 seeds of its own where a sample needed 200 (`M17.A.1`; spec revision 42, wording only; T-163). Four Phase D verdicts changed: `M11.C.21`, `M11.C.34` and `M11.C.7[joint over one spouse]` now pass, and `M11.C.7[one spouse over sided]` fails; `M11.C.44` fails at `learning_rate` 0.1 in the Phase C sweep record; `M11.C.34` has no mutant that turns it red (T-162).

**What is built** (`src/bowen/`), by phase. **Phase B**: the objects, the weekly loop, the standing load, the base
appraisal, the event record, the scripted source, the run log and the renderer (`docs/phase_b_completion_report.md`).
**Phase C**: agents select acts through a policy and learn from felt relief (`docs/phase_c_completion_report.md`).
**Phase D, part I** (`docs/implementation_plan_phase_d.md`): engine speed, channel stamping, cutoff as rejection, the
slow tick and the basic-level estimator; the three symptom channels with lock-in; beliefs as a channel; the
tie-weighted cross-person term; the `M11.G` family-evaluation readout; and borrowed pseudo-self settling back.
**Phase D, part II** (spec revision 28): ages, childhood, launch, partners and births, death and its disposition, lost
income and finding work, and the coach's successor, over a 40-year run. Version 2's stages 1–3 changed rules on top of
these (spec revisions 33–40; `docs/spec_revision_log.md`).

**Where each criterion stands now** is `docs/version_2_status.md`, generated from the records: every `M11.C`
criterion's verdict, its mutation evidence, its sweep fraction, and its verdict at the tag and at revision 38 beside
it. The verdicts in the phase reports and `docs/phase_d_status_2026-10-13.md` / `docs/phase_d2_status_2026-10-13.md`
are as each phase closed; they are history, not the current state.

The criteria run outside the default suite, over ensembles. The ensemble, mutation and sweep records cache each result by the engine hash and the mutant's or setting's own edits (`tools/record_cache.py`, `docs/records_cache/`): a regeneration reruns only what changed, so re-rendering takes under a second, one new mutant a few minutes, and only an engine change reruns everything (about 40 and 20 minutes):

```
python3 tools/ensemble_record.py     # docs/phase_c_ensemble_record.md
python3 tools/mutation_record.py     # docs/phase_c_mutation_record.md (cached)
python3 tools/sweep_record.py        # docs/phase_c_sweep_record.md (cached)
python3 tools/level_occupancy.py     # docs/phase_c_level_occupancy.md (about 3 minutes)
python3 tools/c16_report.py          # report §11, generated from the records
python3 tools/phase_d_record.py      # docs/phase_d_ensemble_record.md (cached; a full rerun about 10 minutes)
python3 tools/phase_d_record.py --part 2   # docs/phase_d2_ensemble_record.md (cached; a full rerun about 50 minutes)
python3 tools/phase_d_mutation_record.py   # docs/phase_d_mutation_record.md (cached; one new mutant about 30 minutes)
python3 tools/version_2_sweep.py --render-only   # docs/version_2_sweep_record.md from its cache (a run: docs/two_mac_sweep.md)
python3 tools/version_2_status.py   # docs/version_2_status.md, every M11.C criterion now
```

```
python3 -m src.bowen.run --seed 7 --log runs/phase_b.jsonl --trace runs/phase_b_trace.md
python3 -m pytest tests/
```

The suite runs in CI on Python 3.11, 3.12 and 3.13 (`.github/workflows/tests.yml`), installing only `requirements.txt`.

Phase B runs a fixed script over an invented seven-person family. It tests one model claim, as a direction between two runs; nothing it produces is a finding about families. The family-diagram importer (`M15`) is not in version 1 or version 2 (`docs/version_2_plan.md` §2).
