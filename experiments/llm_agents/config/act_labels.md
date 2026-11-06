# LLM-agent experiment — the plain act labels (arms A2, A3, A4)

> **Status: draft, for the owner's review before any run** (`docs/plan_v3_llm_agent_experiment.md` §3.2, Q2). No run
> has used these words.
>
> One row per act kind the policy can select (`config/bowen/event_kinds.md`, channel `automatic` or `self`). Read by
> `experiments/llm_agents/config.py`; a kind the engine has and this table lacks, or the reverse, is refused at load.
>
> * `group` — the legality group `A3`'s shuffle stays inside (§3.3): `tie` (an act on one tie), `triad` (needs a
>   closed triad holding the target), `outside` (only toward an outside person), `reopen` (only across a broken-off
>   tie). `A3` gives each kind another kind's labels from its own group, by a derangement fixed per seed.
> * `option` — what the agent sees in its list of options. `{target}` is the other person's token (`P3`); `{third}`
>   is the third member of the triad the act involves.
> * `seen` — how the act reads in another person's round when it reached them, in the third person: `{target}` is
>   `you` for the person it was aimed at, or the target's token for a witness. The triad's third is not known to the
>   receiver, so the `triad` rows do not use `{third}`.
> * No word here may appear in `banned_terms.md`; `tests/experiments/llm_agents/test_prompts.py` checks every
>   rendered prompt.

| kind | group | option | seen |
|---|---|---|---|
| `PURSUE` | tie | seek more closeness with {target} | sought more closeness with {target} |
| `DISTANCE` | tie | pull back from {target} | pulled back from {target} |
| `CONFLICT` | tie | argue with {target} | argued with {target} |
| `OVERFUNCTION` | tie | take over a task for {target} | took over a task for {target} |
| `UNDERFUNCTION` | tie | let {target} take over a task for you | let {target} take over a task for them |
| `CUTOFF` | tie | stop all contact with {target} | stopped all contact with {target} |
| `I-POSITION` | tie | say plainly where you stand to {target} | said plainly where they stand to {target} |
| `STAY-IN-CONTACT` | tie | keep in touch with {target}, calmly | kept in touch with {target}, calmly |
| `PROVOKE` | tie | stir things up with {target} | stirred things up with {target} |
| `TRIANGLE` | triad | turn to {target} about {third} | turned to {target} about someone else |
| `DETRIANGLE` | triad | step out of the middle between {target} and {third} | stepped out of the middle between {target} and someone else |
| `PREVENT_ALIGNMENT` | triad | decline to take sides with {target} against {third} | declined to take sides with {target} against someone else |
| `REDUCE_CUTOFF` | reopen | reopen contact with {target} | reopened contact with {target} |
| `SPLIT` | outside | talk to {target} privately, without telling the others | talked to {target} privately, without telling the others |
| `FRAME_AMBIGUITY` | outside | take {target}'s silence as being on your side | took {target}'s silence as being on their side |
| `DISPLACE` | outside | take out your frustration with the others on {target} | took out their frustration with the others on {target} |
