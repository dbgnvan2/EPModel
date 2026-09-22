# Reader task (round 2, full-text)

Read first, in full: EPMODEL_BRIEF.md and DESIGN_LESSONS.md in this folder. The owner's goal for this round: find things that should be ADDED to the EPModel specification (v2, 427 requirements, 16 modules M1-M16) to make the model as good and as realistic as possible. "Realistic" here cannot mean fitted to data, because the project has none; it means (a) the mechanisms are faithful to the theory and to how human interaction is structured (timing, ordering, witnesses, ties, emotion dynamics), and (b) the inferences drawn from runs are sound (seeding, ensembles, sensitivity, counterfactual validity, documentation), so that a directional result is a property of the mechanism and not an artefact.

Then read each assigned paper IN FULL (pdftotext output; use Read with offset/limit in chunks; pages separated by form feeds; figures survive only as captions; equations may be garbled).

For EACH paper produce:

## A. Report
Numbered items following the brief's "What I need from each paper" (1-8), but compact: only items with content; each finding cites a section/figure/table; tag [PAPER] or [INFERENCE]; quotes under 15 words. Include concrete numbers. End with "Not transferable / cautions".

## B. Candidate additions to the EPModel spec
For each candidate (0-6 per paper; zero is an acceptable answer):
- **Proposed requirement** written in the spec's own style: one or two sentences, MUST/SHOULD, naming the object it constrains.
- **Where it would live**: module guess (M1 objects/ties/events, M3 clocks/ordering/determinism, M4 policy, M6 consolidation/invariants, M10 parameters/config, M11 acceptance tests, M15 import, M16 run log; or "Phase E: ensemble runner / counterfactual arms" which is not yet specified).
- **Evidence**: what in the paper supports it (section), and its strength: SHOWN (experimental or formal result in the paper), ARGUED (position or reasoning), or INFERENCE (yours).
- **What it changes**: does it correct something the current design gets wrong, add a mechanism, add a test, or add a reporting rule? Does it improve theory fidelity, inference validity, or both?
- **Cost / risk**: implementation cost, and whether it conflicts with any stated EPModel rule (determinism M3.D.4/M3.D.5; no LLM in the decision path M3.D.6; no fitting to known histories M11.F.9; constants invented and labelled; engine purity M16.B).

Do not pad. A candidate that merely restates something the spec already has (per the brief or design lessons) should be flagged as "already covered" in one line, not written up. Do not rely on memory of the papers; only the text. Your final message is the report itself, plain markdown, nothing else.
