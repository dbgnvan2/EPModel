# Plan — the failing and unbuilt Phase C criteria

> Proposed 2026-10-09. **Approved by the owner 2026-10-09**, with `M11.C.7`, `.13` and `.14` moved to Phase D (§4).
> Status of each criterion: `docs/phase_c_completion_report.md` §1 and §3. Coverage: `docs/spec_coverage.md`.

## 0. What this plan does and does not do

**The goal is a correct verdict for each criterion, not a pass.** The constants are frozen (`M10.B.4`), and patterns
must emerge rather than be written (`M11.5`, revision 11). So no rule or constant changes to make a criterion pass.
Each failing criterion ends in exactly one of three outcomes, decided by evidence:

| Outcome | Meaning | What happens |
|---|---|---|
| **A. Test defect** | The arms or readout cannot test the claim (a readout that cannot move, an arm that does not isolate the cause) | The owner approves a restated arm or readout, declared and committed **before** the rerun, logged as post hoc |
| **B. Underpowered or too short** | The direction is right, but the run is too short or noisy to decide it | The owner approves a declared change of horizon, also before the rerun, logged as post hoc. One change per criterion, no searching |
| **C. Finding** | The test is valid and the model does not produce the claimed direction | Reported as a finding about the model, with the mechanism traced. The owner decides whether the spec's claim or the model's rule is wrong. Never tuned |

**Ground rules:**
- **Diagnosis before decision.** Every hypothesis in report §3 marked "not verified" is tested before anything is restated.
- **Decisions before reruns.** Each change to an arm, readout or horizon is approved, committed and logged (`config/bowen/criteria.md`, with a post-hoc note) before its rerun, as with the precision rule (report §10).
- **Diagnostics are reported, never gating.**

## 1. Tooling first (one step, small)

**D0 — diagnostic runs on failing criteria.** The mutation tool runs only on passing criteria, because a mutant cannot prove a failing one. Diagnosis needs the same machinery pointed at failing criteria.
- **What:** a `DIAGNOSTIC` mutant kind, rendered in its own record (`docs/phase_c_diagnostic_record.md`), never counted by coverage, and cached through `tools/record_cache.py`. Each diagnostic runs in minutes.
- **Test:** `test_d0_diagnostics_never_count_as_proof`, which checks that `spec_coverage.mutation_reds()` ignores the diagnostic record; and `test_d0_diagnostic_record_is_current`, which re-renders the record from its cache.
- **Cost:** about an hour, plus minutes per diagnostic.

## 2. Diagnosis, ordered by cost and by how much each result unblocks

| # | Criterion | Report's account | Diagnostic (evidence, not a fix) | Artifact that settles it | Likely outcome |
|---|---|---|---|---|---|
| D1 | `M11.C.29` | The third person has **0 symptom weeks in both arms**, so the readout cannot move | Trace why: for the third person, compare `M1.A.6`'s symptom threshold with the load reached in this scenario over 80 weeks. Does any arm reach threshold at any seed? | A per-seed table of the third person's peak load against threshold, in the diagnostic record | **A**: the scenario never stresses the third person enough. Restating the arm (whose load, what spell) is an owner decision. Also blocked in part by plan P1 (`M6.I.6`'s stock-and-flow restatement) |
| D2 | `M11.C.41` (reactive-share limb) | At a lower level anxiety rises, but the share of reactive acts **falls**. Unverified cause: `M4.D.3a`'s availability removes automatic acts from the legal set | Run the four cells under `availability-level-independent` (availability fixed). If the share then rises at a lower level, availability is the cause | The diagnostic record's C.41 rows, both arms' reactive share with and without availability | If confirmed, **A or C**: the readout mixes two level effects (more pressure toward reactive acts, fewer of them available). The owner decides whether the share should be taken over available acts, or whether this is a finding |
| D3 | `M11.C.42`, `M11.C.45` | Both lost significance when `TRIANGLE` started loading the recruited third (§8). Unverified cause: the act's relief, and so what is learned, now depends on the parents' tie | Decompose relief per `TRIANGLE` by which tie was strained, under the old and new roles (`triangle-roles-swapped` already exists). Also check power: C.42 is p 0.084 at its seed count | A table of relief per act by strained tie, and C.42's seed count against the cap | **B** for C.42 if power is short. **C** for both if the new roles stop relief from rewarding triangling, which would be a finding about §8's decision |
| D4 | `M11.C.27` (2 cells reversed, 1 flat) | Stable plus remove one *lowers* deviation; unstable plus add third *raises* it, at every setting | Trace 5 seeds per cell. Which term moves the pair's deviation when the third arrives or leaves: contact optimum, band, or the transfer? | A per-term decomposition of the pair's deviation change, per cell | Probably **C**. The cells reverse at every setting, so the model's twosome dynamics may differ from FE06.1. The owner decides spec or model |
| D5 | `M11.C.44` | Not yet diagnosed: +0.010, p 0.73 | Count the inside-ward and outside-ward acts per arm. Is the ratio's denominator large enough to measure? Does a spell change the triangle count at all (shared with C.45)? | Act counts per arm, and the ratio's standard error | **A** if the counts are too small, otherwise **C** |
| D6 | `M11.C.4` (later limb), `M11.C.5` | C.4's cutoff accrual does not build in 22 weeks; C.5's ladder may not form in 60 | Measure, without changing any setting: C.4's accrual on the severed tie week by week, and when, if ever, C.5's ladder first appears in a long run (e.g. 260 weeks) | Accrual and ladder onset curves | **B** if both clearly develop later: the owner approves one declared horizon each. **C** if they never develop |

**Order:** D0, then D1 and D2 (cheap, likely test defects), then D3 and D5 (they share triangle counts), then D4 and D6 (trace-heavy).

**Batching:** each diagnosis adds rows to one diagnostic record, and all findings go to the owner in one decision memo (§3), so there is one round of decisions, not one per criterion.

## 3. Owner decisions, collected in one memo

After D1–D6, one memo (`docs/DECISIONS — PHASE C FAILING.md`) gives each criterion its outcome (A, B or C), the evidence, and a recommendation. The owner decides once. Approved restatements are then committed and rerun together; with the cache, only the changed criteria run.

**Acceptance for this phase:** every failing criterion in report §1 carries one of these:
- a verdict under an approved, pre-declared test (done);
- or a finding, with its mechanism traced (done, reported as a finding).

None may be left "fails, unexplained". Each change is checked by the existing record tests, plus one test per restated arm or readout (`test_m11cN_...`), written before the rerun.

## 4. The three unbuilt criteria: a scope decision, not part of this plan

Each needs engine work that does not exist yet:

| Criterion | Needs | Size |
|---|---|---|
| `M11.C.7` topology, not coach skill | `M8.2`/`M8.3`'s position predicates, plus the Dr Halim instance; its arms declare no direction yet | Large: a new mechanism |
| `M11.C.13` help relocates, not reduces | Incidents located in a community; blocked by plan P1 (`M6.I.6` restated) | Large |
| `M11.C.14` technique null under marital distance | `M5.C.1`'s marital-distance gate | Medium |

**Decided 2026-10-09: moved to Phase D.** The spec's criterion rows now carry phase D.

## 5. Cost and process

- **Diagnostics:** D0 about an hour. D1–D6 run in minutes each with the cache; the work is in the traces.
- **Reviews:** one full `learning-qa` pass and one correctness pass at the end of each batch (D0; D1–D6; the restatements). Fix loops get narrow passes on a cheaper model. Three or fewer minor items go to TODO (`CLAUDE.md`, "keep the loop cheap").
- **Stops:** after the diagnostic record and memo (for the owner's decisions), and after the restated reruns.
