# LLM-agent experiment — the report's fixed text

> The words of `docs/llm_agent_experiment_report.md` that are not numbers or table headings
> (`docs/plan_v3_llm_agent_experiment.md` §6). Read by `experiments/llm_agents/report.py`, which only fills the
> `{fields}`; the report is generated and checked current by a test (LX.16). `how` completes the model's own framing
> block (`src/bowen/render/trace.py::FRAMING`, `M11.F`), which heads the report (`M11.G.4`, LX.20).

| key | text |
|---|---|
| `how` | In arm A0 each person's outcome is selected by the model's policy; in arms A1–A4 each family member's outcome is chosen by a language model shown only its own felt tension, its ties by token and its options, and the external agent stays on the policy. A language-model agent is not a person, and no arm describes a real family. |
| `differences` | The arms differ in more than who chooses. In A1–A4 an agent gets no credit for what its act did to others, no mix of two channels, no competing urges, and its last rounds in place of learned values; the external agent is on the policy in every arm. A0 is a model too, not ground truth that the other arms should match, and no result here confirms or refutes a mechanism. |
| `method` | Every number below is read from the runs' summaries by the model's own readout functions, the same for every arm. A comparison is a direction against A0 with the number of seeds in which it holds; no claim uses an absolute threshold. A run that did not complete (stopped at the budget cap, out of retries, not replayable or invalid) is not a result and is left out. A run is kept on a later engine only when its replay from the cached responses reaches the same behavior: the same choices, acts, readouts and pattern weeks. Each summary carries the engine it was last replayed on, and each run record the engine its live run ran on. |
| `no_runs` | No complete run yet. |
| `no_pair` | Not computed: it needs complete runs of both arms on the same seeds. |
| `label_following` | Seeds in which A3's act frequencies follow A2's more closely by the label shown than by the mechanics run: {more} of {n}. Where this holds in most seeds, the agent follows the words rather than the consequences, and A2's patterns are counted as driven by what the words mean. |
| `feedback_removed` | Seeds in which a pattern's direction against A0 is the same in A4 (no felt change shown) as in A2, per pattern. Where removing felt change changes nothing, the pattern did not come from felt relief. |
| `opaque_plain` | Patterns that appear in A2 and not in A1 (counted as needing the plain words): {patterns}. |
| `cross_check` | The same arms on the second model, on the same seeds: each pattern's and readout's direction against A0, and the seeds in which that direction is the same as the main phase's on the same seed. A direction that holds on both models is not one model's habit; one that differs depends on the model. |
| `position` | Whether the language model chooses by where an option is listed rather than by what it is (§13.4), per complete run of an LLM arm, the pilots included. Position 1 is the share of choices of the first option listed, against the mean of 1/N, the share a choice blind to position would give it. Same id and same option count the pairs of one person's consecutive weeks that repeat the option number, and that repeat the option itself (by its mechanics). A position effect is same-id repeats above same-option repeats and a position-1 share above the mean of 1/N, a direction with no threshold. The first pilot (`pilot-A1-s1`) ran in the fixed order, where the same option stays at nearly the same place every week, so its row cannot tell position from content and is shown for comparison only; every later run is in the shuffled order (§13.1). |
| `position_effect` | {arm}: a position effect in {k} of {n} main-phase seeds. |
| `position_none` | No complete run of an LLM arm yet. |
