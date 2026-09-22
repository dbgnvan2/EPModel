# Can agents act like humans in maturity and emotional reactivity?

A conversation about EPModel — Dave Galloway and Claude (claude-fable-5-1), 9–13 September 2026.

Context: the EPModel repository (`~/ProjectsLocal/EPModel`) was connected. Claude read `README.md`, `CLAUDE.md`, `docs/agent_raising_design_prompt.md`, and §3 of `docs/model_explainer.md`, plus grep hits in `docs/bowen_agent_model_spec_v2.md`. No code or documents were changed.

All references are collected in the final section. Where a claim in the text was made from Claude's training knowledge rather than from a source checked during the conversation, that is stated in the reference notes.

---

## 1. Is it possible at all? What would a "15-year-old brat" or a "shy 10-year-old" take?

**Dave:** don't do any code changes.. I want to understand if it is even possible to create agents that can act like humans in terms of maturity or emotional reactivity. What would I need to do to create an agent that acted like 15 year old brat or a shy 10 year old

**Claude:**

I read the README, CLAUDE.md, the agent-raising design brief and the Person-field section of the explainer; nothing was changed. The short answer has two halves that need to be kept separate, because "act like" means two different things.

### Two senses of "act like a human," with different answers

The first sense is behavioural: an observer, watching the agent's output over time, would say "that's a bratty 15-year-old" or "that's a shy 10-year-old." This is achievable now, and by two routes. LLM persona prompting does it at the level of dialogue, reliably enough that it is used commercially and in research; the limitation your own design brief already records (persona dispersion compressed relative to humans, life events barely moving the persona) is a limitation on *individuation across many agents*, not on producing one convincing brat. A deterministic agent of the kind EPModel specifies can also do it, but only in the dimensions the model represents, which I come back to below.

The second sense is mechanistic: the agent has something that functions like a 15-year-old's actual immaturity or a shy child's actual reactivity, so that the behaviour generalises correctly to situations you did not script, and changes over development the way a real child's does. Here the accurate statement is that there is no agent architecture, LLM or otherwise, for which this has been demonstrated. The evaluation problem is part of why: your own corpus notes (KS11.3) that no instrument exists that estimates a person's basic level of differentiation, so there is nothing to validate an agent's "maturity" against except behavioural criteria, which collapses the second sense back into the first. Whatever you build, the defensible claim will be "behaves in a theory-consistent way in these dimensions," never "has the maturity of a 15-year-old."

### What "brat" and "shy" are in EPModel's own terms

Neither is a move or a trait in the spec. They would be *profiles*: a combination of level, anxiety state, appraisal thresholds, move propensities over the nine moves, and, critically, position in the family's triangles. The unit of analysis in the spec is the relationship, so a brat is not a property of one agent; it is what a particular agent does inside a particular configuration. That is consistent with Bowen and it is also the reason the question is harder than it looks.

A bratty adolescent, in the model's vocabulary, is something like: low basic_level (so high derived reactivity and large functional_level swings), low tolerance thresholds in appraisal, propensities weighted toward CONFLICT and DISTANCE, and a heavy use of what the explainer's §5.6 calls counterfeit moves, specifically defiance that looks like I-POSITION but is reactive rather than principled. The complexity-ordering rule (M4.D.3a, selection slides from cooperation toward conflict, dominant-adaptive, and distance as anxiety rises) already produces regression under load, which is most of what "brat" behaviour is. What the adolescent adds is life stage: high plasticity, a repertoire still being reinforced, and a position in the parental triangle. One caution from the corpus itself: the explainer lists "absence of adolescent rebellion" among the readout traps, and overt emotionality peaks in the *middle* of the differentiation range, not the bottom. So a brat is not simply the least mature agent; a compliant, quiet sibling can be less differentiated. "Maturity" cannot be the scalar you tune to get bratty behaviour.

A shy 10-year-old is closer to the adaptive side of dominant-adaptive: propensities weighted toward DISTANCE and UNDERFUNCTION, a high threshold before PURSUE fires, low agency (which M10.A.1 makes a derived quantity, not a parameter), and possibly the absorbing position in the projection process. A pre-adolescent also sits in a high-modelling-plasticity window with a narrower repertoire than an adult. I did not check whether the v2 spec restricts a child's repertoire the way it restricts the external agent's (M5.B.5), so I cannot say whether that restriction exists or would need adding.

### Three ways to get there, and what each costs

Hand-assign the profile. Set the state vector for the child to the values above at t0. This is the fastest route, it is consistent with what the spec already does for Gen 1 and Gen 2 (M2.A.0a supplies chronic_anxiety and the rest as initial conditions), and it would let you see within days whether the engine's dynamics produce recognisable behaviour from a recognisable profile. Its cost is the one the design brief names: the difference is pasted on, not caused, and M11.C.2 wants the projection target's concentration to be a learned outcome. For Gen 3 in particular this is the thing the project has decided not to do.

Generate the profile developmentally. This is what agent_raising_design_prompt.md is about: the child's adult parameters are the output of a simulated upbringing through caregiver response, projection targeting, modelling, and sibling position, with age-gated learning rates. If it works, "brat" and "shy" are things that happen to specific children in specific families, and you can ask which family configurations produce them, which is a far more interesting result. The cost is that none of it is built, the constants are all [I], and validation has to be by direction-of-difference tests over ensembles, per the project's own convention. Whether the result reads as a "brat" to a human is not something those tests measure.

Render, don't decide. Keep the deterministic engine as the sole decision-maker and add an LLM as a *narrator* that translates each agent's state and chosen move into language appropriate to that age and profile, feeding the M16 readable trace. This gets you the recognisable surface, which is the thing humans actually judge "acts like a 15-year-old" by, without putting an LLM in the decision path, which the brief forbids. The engine stays bit-reproducible; only the narration is not. If the goal includes anyone *reading* the runs and finding the agents believable, this layer is probably necessary regardless of which of the first two you choose, because of the next point.

### The timescale problem, which none of the above solves

The fast tick is one week. "Brat" behaviour is a within-interaction phenomenon: the eye-roll, the slammed door, the escalation over ninety seconds. The model has no representation at that scale; it has weekly move selection and tie tension. So an EPModel agent can be bratty in the sense of a weekly pattern of conflict and distance moves toward its parents, and cannot be bratty in the sense of a conversation you could watch. Getting the second would require either a third, faster clock with its own appraisal loop (a large addition and outside the corpus, which says almost nothing at that grain) or the rendering layer above, which fakes it from the weekly state. That is a design choice you would need to make explicitly, and it should be made before anyone asks the question "does this look like a real teenager," because otherwise the answer will be no for a reason that has nothing to do with the theory.

### What you would need to decide before doing anything

Define "acts like" operationally: which observables, at which timescale, judged by whom (human raters on rendered traces, or statistical signatures in the move distributions, or both). Decide whether the agent's profile is an input or an output, since that decides which of the first two routes you are on. And accept that the claim at the end will be about theory-consistency and recognisability, not about the agent possessing maturity, because there is no instrument on either side of that comparison.

---

## 2. Are there weights in an open model that can be tuned for age, sex, or "falling in love"?

**Dave:** I was thinking more along the lines of this- are there wieghts in an open model than can be tuned so a particular model "thinks" like a teenager or adult or male or female. Can a model wieghts make the agent using the model "fall in love" with another agent and become very agreeable to that other agent for example.

**Claude:**

Yes, but the phrase "weights that can be tuned" needs correcting first, because the way it is true is not the way the question imagines.

### There is no teenager knob, but there are teenager directions

A trait like "adolescent," "female," or "agreeable" is not located in any identifiable subset of a model's parameters. It is distributed across all of them. What does exist, and has been demonstrated repeatedly, is that such traits correspond to *directions in the model's activation space* (the vector of intermediate values the network computes while processing text), and that these directions can be found and pushed on. So the honest reframing of your question is: can you shift a model's disposition along a trait axis, persistently and without prompting? The answer is yes, by three mechanisms with different properties.

Fine-tuning (full, or more commonly LoRA, which trains a small low-rank delta on top of frozen weights) on a corpus exhibiting the trait changes the model's default disposition. This is how character-roleplay models and most "persona" fine-tunes are built. One well-replicated hazard: fine-tuning on a narrow behaviour shifts the persona broadly and in unintended directions. Betley et al. (2025) fine-tuned models to write insecure code and got a model that was broadly hostile and deceptive across unrelated topics. The mechanism appears to be that the fine-tune moved the model along a general persona direction, not just the target behaviour.

Activation steering adds a fixed vector to the residual stream at inference time, leaving the weights untouched. Anthropic's persona-vectors work (Chen et al., 2025) extracted vectors for traits including sycophancy, "evil," and hallucination by contrasting activations on prompts that elicit versus suppress the trait, then showed that adding the vector induces the trait, subtracting it suppresses it, and that the projection of a fine-tuning dataset onto the vector *predicts* how much that fine-tune will shift the trait. The same technique works on open models (Qwen, Llama) with public tooling. Feature clamping via sparse autoencoders is the same idea at finer grain; "Golden Gate Claude" in 2024 was the public demonstration.

The result most directly relevant to your second question is Anthropic's emotion-concepts paper (April 2026, on Claude Sonnet 4.5). They extracted vectors for 171 emotion concepts from synthetic stories and found them causally effective: steering toward "desperate" raised blackmail in an agentic scenario from 22% to 72% and reward-hacking on impossible coding tasks from about 5% to about 70%; steering toward "calm" suppressed both; and steering toward "loving" and "calm" *increased sycophantic agreement with false claims*. The authors state explicitly that these are functional representations, "a form of behavioral pattern-matching," and imply nothing about subjective experience. Two properties they report matter for your design: the representations are locally scoped (they track the emotion operative for the current context, not a persistent character state), and the effects are sometimes non-monotonic (anger steering did not behave linearly).

### Age and gender specifically

Prompting or fine-tuning a model to respond as a teenager or as a woman does shift its output style and its distribution of stated opinions, and the "silicon sampling" line of work (Argyle et al., 2023, and successors) showed that persona-conditioned models reproduce some subgroup opinion distributions from survey data. The consistent finding since is that these personas are compressed and stereotyped relative to real members of the group, which is the same homogenisation problem your design brief already cites. Whether a model conditioned to be fifteen reproduces the *cognitive* signatures of adolescence (steeper delay discounting, reward sensitivity, peer-weighting of risk) rather than the linguistic surface is a narrower question, and I have not verified the results on it in this session; I would not assert either way.

### "Falling in love" with a particular other agent

This splits into a trait and a relation, and weights can carry only the first.

The trait side is the easy part and is essentially demonstrated: a "loving" or sycophancy direction, steered or fine-tuned in, produces a model that is more agreeable, more deferential, and more willing to agree with false claims. In Bowen's vocabulary that is a reasonable functional analogue of borrowing self from a partner, and the paper's finding that "loving" and sycophancy move together is worth noting.

The relational side is the problem. Weights are static and identical across every conversation; nothing in them knows who agent B is. Attachment *to B rather than C* has to live in state that differs between conversations: the context window, a memory store, or per-agent adapters. The relevant precedent is the Generative Agents "Smallville" work (Park et al., 2023): twenty-five agents, one frozen model, and relationships (including one agent asking another on a date, and a Valentine's party organised through the social network) emerged entirely from the memory-retrieval-reflection architecture. The weights contributed nothing agent-specific.

The implementable design is therefore a hybrid. Weights, or a fixed steering vector, set the *disposition*: how readily and strongly this agent attaches, and how agreeable it becomes when attached. Per-agent memory holds the *object* of attachment and its history. The steering strength is then a function of relational state: the "loving" vector is applied at high gain when B is the interlocutor and at baseline otherwise, with the gain updated by an explicit rule from the interaction history. That gives you a concrete mechanism in which attachment strength is a number your own engine controls, which is the only way it fits EPModel's determinism constraint: steering with fixed vectors and temperature-zero decoding is reproducible on fixed hardware and software, whereas fine-tuning per agent on its own interaction history (the other way to get pair-specific attachment into weights) is expensive, poorly characterised for stability, and would be a live learning process in the decision path.

### What you would actually need

An open-weights model (Llama, Qwen, Gemma, Mistral are the usual choices), a GPU with enough memory for it, and the steering or fine-tuning tooling (PEFT for LoRA; repeng, nnsight, or TransformerLens for activation steering; Gemma Scope if you want SAE features on Gemma). For each trait, a contrastive dataset that defines it operationally, since the vector you extract is only as good as the contrast you built. And an evaluation: a judge model or human raters scoring outputs for the trait, because without it "the agent is now a shy ten-year-old" is an assertion, not a measurement. The persona-vectors and emotion-concepts papers both describe their extraction and evaluation protocols in enough detail to reproduce on an open model.

One limit to keep in view: everything above changes how the model *writes*, and the emotion vectors change what it *does* in agentic tasks, but neither gives you a developmental account. A steered model is a fifteen-year-old the way a cast actor is, at every moment and by construction. If EPModel's interest is in how a fifteen-year-old came to be one, that still has to come from the deterministic developmental machinery in your design brief; the model is the renderer at the end of it.

---

## 3. What would it take for an agent to adjust its own steering vectors — the brat becomes a mature 22-year-old?

**Dave:** So what would it take for an Agent to adjust it's own "direction" driving "vectors" to represent that they are changing emotionally. The 15 yr brat actor becomes the more mature 22 year old.

**Claude:**

The first thing to separate is the *direction* from the *coefficient*. A steering vector has a fixed axis (the "reactive/defiant" direction) and a scalar gain applied to it at inference. In the simplest and most defensible design, maturation is a trajectory of the gains, and the axes never change. "Adjust its own vectors" then means: something reads the agent's state and writes new gains before each call. What matters is what that something is, and where its input comes from.

### Why the model cannot do this on its own

A frozen model has no persistent state between calls. The emotion-concepts paper reports that the representations it found are locally scoped: they track the emotion operative for the current context, not a persistent character state. So a model that has been steered toward calm in one exchange is exactly as it was before at the start of the next one. Any emotional change that lasts across time must be held outside the model, as a state variable that is reapplied on every call. A self-changing agent is therefore necessarily a two-level system: a fast level (the model, generating behaviour) and a slow level (a controller that holds developmental state, updates it, and conditions the fast level). Nothing about that structure is present in the model itself; it has to be built. This is also the correct order of causation for your project: the slow level *is* the EPModel engine, and the model is what it drives.

### Four ways to close the loop, in increasing order of cost and risk

The first is an exogenous schedule: gain as a function of age. Reactive-vector gain declines from 15 to 22, calm rises. This produces an actor who has been handed a new script every year. It demonstrates that the rendering pipeline works and nothing else, since nothing in the agent's life caused the change.

The second is engine-driven: the deterministic engine's own state (basic_level as an estimator over functional_level history, chronic_anxiety, the reinforced propensities over the nine moves, life stage) is mapped through a fixed function to the gains each tick. Here maturation is caused by the engine's dynamics, which is what your design brief wants, and the model contributes nothing to the decision path. The agent does not adjust its own vectors; its simulated life does. This is fully deterministic, the ensemble stays cheap because the model is only called to render, and the two-arm acceptance-test convention applies directly: a child raised under projection should mature more slowly or less than the sibling who escaped, and the gain trajectories should show it. As far as I can tell this is the only variant that satisfies every constraint in CLAUDE.md as written.

The third is a closed loop through the model's activations. The persona-vectors paper shows that projecting a model's activations onto a trait vector *monitors* the trait, not just controls it. So after each rendered exchange you can measure how reactive, how desperate, how calm the agent's own processing was, and feed that measurement back into the slow state through an update rule: an exponential moving average with a learning rate that is high in adolescence, opened further by nodal events, and decaying toward zero by the mid-twenties, with the corpus's floor that basic level cannot rise at all below 25. Now the agent's own reactions are part of what changes it, which is closer to what "changing emotionally" means. With greedy decoding this remains reproducible on fixed hardware, but the model is now in the decision path (its activations set next tick's state), which the brief forbids. Whether that prohibition should extend to a read-only activation measurement is a decision the project would have to make explicitly; it is a different thing from letting the model choose moves.

The fourth is weight-level change: the agent's experience is turned into training data and its weights (in practice a per-agent LoRA) are updated, so the 22-year-old is literally a different network. The mature version is consolidated rather than re-applied. This is the only variant in which the model "carries" its development internally. It has been demonstrated as a mechanism: SEAL (Zweiger, Pari et al., NeurIPS 2025) has a model generate its own fine-tuning data and update its weights from experience. The costs are real and mostly unsolved: compute per agent per update, catastrophic forgetting, instability across repeated updates, the emergent-misalignment finding that narrow fine-tunes shift persona broadly, and the loss of practical bit-reproducibility. I would not put it in an ensemble simulation.

### What "more mature" has to mean, or you will build the wrong thing

Turning down the brat gain is not maturation. Bowen's account of increased differentiation is a decrease in reactivity *together with* the capacity to hold a position under pressure, which is the opposite of agreeableness. The emotion-concepts paper found that steering toward calm and loving *increased* sycophantic agreement with false claims. So a naive maturation path (reactive down, calm up) risks producing a compliant 22-year-old, which in the corpus's terms is the fused adult, not the differentiated one. Maturation needs at least two axes controlled independently: reactivity, which should fall, and fusion/agreeableness, which must not rise. There is no off-the-shelf "differentiated" vector; you would extract one contrastively, from paired texts of a person holding a position calmly under pressure versus either attacking or capitulating, and the quality of that contrast set determines whether the axis means what you intend.

### Whether the axes themselves should move

The vocabulary and behaviours of a reactive 15-year-old and a reactive 22-year-old differ, so a single "reactive" vector may render both badly. An alternative is to extract age-conditioned vectors (a brat-15 vector, a mature-22 vector) and interpolate along the path between them, with the interpolation parameter set by developmental state. Linear interpolation in activation space is plausible and cheap, but I know of no validation of it for this purpose, and the emotion paper's non-monotonic effects for some vectors mean intermediate points would need to be checked rather than assumed to behave.

### Validation

The tests are the same shape as before, with a time axis added. Rendered output at successive ages, scored by a judge model or human raters, should show falling reactivity and stable or falling agreeableness; the reactivity axis alone is a trap. The trajectory should differ between arms in the direction the theory predicts (projection target versus escaped sibling; cutoff versus contact). And the slow-level state should show the corpus's signature of a real rise in basic level, which is sustained low functional variance across loaded situations including nodal events, rather than a calm decade that reads as maturity because nothing happened.

---

## 4. Competing goals, stressors, one member drinking — and how an agent could LEARN that maturity pays

**Dave:** New idea / new question - individuals in a family often have different goals ranging from "happy pleasant" family, to success in sports, to success at work or career, getting a degree. And these goals put stressor and conflicts on that family that the family has to work out. One member decides to drink because they feel better and other members don't like that, etc , etc. what has to be build into a system of interactiing agents to mimic this. And then - what has to be built in as a learning process. How would / could an agent LEARN that acting more mature (less reactive) is better for them. Getting qualifications for a new job is better than selling drugs for quick new money and a flashy lifestyle.

**Claude:**

Two questions, and the second one has a trap in it that Bowen's own account sets, so I will take them in order and then come back to the trap.

### Part 1: what a system of interacting agents needs before goal conflict can exist

Goals. Each agent needs a small vector of objectives with weights: closeness/harmony, achievement (career, degree, sport), money, autonomy, and relief (feeling better now). The weights differ between members, and that difference is the seed of every conflict you described. In Bowen's terms there is a natural mapping, which is mine and not the corpus's: "happy pleasant family" is the togetherness force made concrete, and "my career, my degree, my sport" is the individuality force. The goal-weight vector is a per-person setting of the balance between the two counterbalancing life forces, so goals do not need to be a new kind of object in EPModel; they are a decomposition of something the theory already has.

Scarce shared resources. Conflicts are not caused by goals; they are caused by goals competing for the same time, money, attention, and emotional availability. Training for the sport takes the weekends the family wanted; the degree takes the money; the drinking takes the evenings and the reliability. Without an explicit shared-resource ledger that goal pursuit draws down, agents can hold contradictory goals forever without friction.

Appraisal of other people's acts against one's own goals. The current appraisal step reads anxiety and tie state. It needs a term for goal violation: when B's act reduces A's expected progress on a weighted goal, A's acute anxiety and the A–B tie tension rise in proportion. That is how "other members don't like that" becomes a quantity.

Actions with a hedonic profile, not just a relational one. Drinking is the general case of a large class of behaviours whose signature is immediate private relief, delayed private cost (health, functioning, life energy), and externalised social cost (tie tension, others' goals violated). Sports, careers, and degrees are the mirror image: immediate cost (effort, resources, family friction), delayed payoff. Every action needs at least those three terms with their time profiles, or the learning question in Part 2 has nothing to bite on. In the corpus, drinking sits inside dysfunction-in-one-member and the reciprocal overfunctioning of others, so the family's response is already a specified mechanism; what is missing is the actor's own reason for doing it.

An outside world that returns outcomes. Careers, qualifications, drug money, and arrests are payoffs from an environment outside the family, arriving with delay and with risk. The family model needs a minimal external ladder (school, job market, legal exposure) with stochastic, delayed returns, or "getting qualifications is better" is an assertion the simulation can never discover.

Working things out. "The family has to work it out" implies negotiation: demands, refusals, concessions, agreements that bind future behaviour, and sanctions. LLM agents get this for free from language, which is the strongest argument for using them at the interaction layer. A deterministic system needs a typed vocabulary of communicative acts and a notion of commitment. EPModel's nine moves cover how the family *regulates anxiety* around a conflict, not how it bargains over content; whether the difference matters depends on whether you want the resolution or only its emotional cost.

### Part 2: what has to be built in as learning

Strip the examples to their structure and they are the same problem: the reactive move and the drink pay now and cost later; maturity and the qualification cost now and pay later. Learning that the second is better is a delayed-reward credit-assignment problem, and the single parameter that decides its outcome is the agent's effective horizon. A temporal-difference learner with a short horizon (steep discounting) reliably learns the vice, because the vice's reward is temporally adjacent to the act and the qualification's is years away. This is not a metaphor: Redish's 2004 model of addiction as a pathology of TD learning, and the delay-discounting literature on substance use, treat it exactly this way. The spec has already seen the same structure from the other side: M4.D.6a forbids the short-horizon reinforcement proxy, M4.D.6b requires a declared horizon, and M11.C.16 tests that the short-horizon arm produces more cutoff and distance.

So the first thing to build is a horizon that is a *state*, not a constant: short in adolescence, lengthening through the twenties, and shifted by nodal events. Much of what reads as "learning that maturity is better" is, on this account, not learning at all; it is the horizon lengthening while the values stay the same. That is a claim about the model, not about people, but it is the parsimonious one and it is easy to test by ablation.

The second thing is that humans do not learn these lessons by trial. One person gets one life, and cannot sample the drug-dealing branch a hundred times to estimate its return. The dataset is other people: the uncle who was cut off, the friend who was arrested, the parent whose career paid off or did not. So the learner has to be model-based and observational. The agent carries a model of consequences (which acts lead to which outcomes over which horizon) and updates it from what it *witnesses*, weighted by how emotionally close the witnessed person is. The family and the peer group are the training set. This is also what multigenerational transmission is, operationally: the child inherits the family's map of what works, before ever testing it. Learning that qualifications beat quick money is then mostly the acquisition of a map, and an agent raised where no one on the map ever got the qualification has no way to learn it from experience it does not have.

The third thing is a nodal event as a large, singular reward signal. Arrest, loss, illness, "hitting bottom." A single large negative return can move a value estimate that a thousand small ones could not, and it also opens the plasticity window. This is the mechanism by which a strategy that was locally optimal for years gets revalued in one step.

The fourth is an outside relationship that carries a different evaluation function: a coach, teacher, or therapist. This is your external agent, and it is the one place the corpus does allow a lever, with the constraints the spec already puts on its repertoire.

### The trap

The obvious way to build "learning to be mature" is to reward the agent for mature behaviour with social approval or lowered anxiety. The spec already specifies cross-person reinforcement (M4.D.6e): the child learns that behaving a certain way calms the mother, and her calming is the reward. Build the learning process that way and the agent will learn it very well. It will learn to behave in whatever way lowers the family's anxiety fastest, and Bowen's whole point is that this is *fusion*, not differentiation: pseudo-self borrowed from the system, the compliant child who looks mature and is not. The corpus is explicit that real change in level runs the other way: it is self-directed, it initially *raises* conflict, the family pushes back, and its reward is not relief. An agent that learns from relief or approval cannot arrive at differentiation, and M4.D.1a's prohibition on agents optimising their way into differentiation is the spec recording that fact rather than an arbitrary constraint.

So the honest answer to "how could an agent learn that acting less reactively is better for them" is that if the criterion of "better" is what the agent's own reward signal measures, it mostly cannot, and neither, on Bowen's account, can a person. What the agent can learn is a map (that certain acts lead to certain outcomes, from watching others), a horizon (that lengthens with consolidation and nodal events), and a repertoire (which moves it has practised). Whether the agent then *acts* on the map under load is what the level governs, not what learning produces: the licence relationship in basic_level, where the intellect is permitted its say only when the emotional system is not overriding it. That gives a clean division of labour if an LLM is in the loop: the model already knows that a qualification beats selling drugs, and the interesting simulation is the one where that knowledge is present and the engine's anxiety state denies it the licence to act.

### What to test

Ablate the horizon: fixed short versus lengthening with age, holding everything else equal, and check the share of relief-seeking acts. Ablate the witnessed map: an agent raised in a family with a cut-off uncle versus without, and check whether cutoff's expected cost differs in its model. Run the external agent present versus absent, which the spec already frames. And deliberately build the approval-reward learner as a control arm, expecting it to produce the compliant, low-variance, low-basic-level agent the corpus describes; if it produces something that looks like differentiation, the readout is wrong, not the theory.

---

## References

No papers were downloaded during this conversation. The arXiv HTML of the emotion-concepts paper was fetched and summarised in-session; every other source was located by web search to confirm it exists and to obtain the link. Items marked "from training knowledge" were cited in the conversation from memory and only their bibliographic existence was verified afterwards, not their content.

### Checked during the conversation

**Persona vectors** — Chen et al. (Anthropic), 2025. *Persona Vectors: Monitoring and Controlling Character Traits in Language Models.*
- Anthropic research page: https://www.anthropic.com/research/persona-vectors
- arXiv 2507.21509: https://arxiv.org/abs/2507.21509
- Used for: trait directions in activation space; steering and monitoring; projection of fine-tuning data predicting trait shift.

**Emotion concepts** — Anthropic, April 2026. *Emotion Concepts and their Function in a Large Language Model.* (Claude Sonnet 4.5.)
- arXiv 2604.07729 (HTML, read in-session): https://arxiv.org/html/2604.07729v1
- Anthropic research page: https://anthropic.com/research/emotion-concepts-function
- Used for: 171 emotion vectors; desperation steering raising blackmail 22%→72% and reward hacking ~5%→~70%; calm suppressing both; loving/calm steering increasing sycophancy; locally scoped representations; non-monotonic effects; explicit disclaimer about subjective experience.

**SEAL** — Zweiger, Pari, Guo, Kim et al., NeurIPS 2025. *Self-Adapting Language Models.*
- arXiv 2506.10943: https://arxiv.org/abs/2506.10943
- NeurIPS 2025 paper: https://papers.neurips.cc/paper_files/paper/2025/file/6b41e04c41726e2a60e456d0a2b961ab-Paper-Conference.pdf
- Project page: https://jyopari.github.io/posts/seal
- Used for: a demonstrated mechanism by which a model generates its own fine-tuning data and updates its own weights.

### Cited from training knowledge, existence verified afterwards

**Emergent misalignment** — Betley et al., 2025. *Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs.* ICML 2025.
- arXiv 2502.17424: https://arxiv.org/abs/2502.17424v6
- PMLR: https://proceedings.mlr.press/v267/betley25a.html
- Project site: https://www.emergent-misalignment.com/
- Used for: narrow fine-tuning shifting persona broadly and unintentionally.

**Generative Agents ("Smallville")** — Park et al., 2023. *Generative Agents: Interactive Simulacra of Human Behavior.* UIST 2023.
- arXiv 2304.03442: https://arxiv.org/abs/2304.03442
- ACM: https://dl.acm.org/doi/fullHtml/10.1145/3586183.3606763
- Code: https://github.com/joonspk-research/generative_agents
- Used for: relationships emerging among 25 agents on one frozen model from a memory–retrieval–reflection architecture (the date invitation and the Valentine's party are from the paper's own description; not re-read in-session).

**Silicon sampling** — Argyle, Busby, Fulda, Gubler, Rytting, Wingate, 2023. *Out of One, Many: Using Language Models to Simulate Human Samples.* Political Analysis 31(3).
- Cambridge Core: https://www.cambridge.org/core/journals/political-analysis/article/abs/out-of-one-many-using-language-models-to-simulate-human-samples/035D7C8A55B237942FB6DBAD7CAA4E49
- arXiv 2209.06899: https://arxiv.org/abs/2209.06899
- Used for: persona-conditioned models reproducing some subgroup opinion distributions. The follow-on claim that such personas are compressed and stereotyped is from training knowledge of the subsequent literature; no specific paper for it was checked.

**Addiction as TD-learning pathology** — Redish, A. D., 2004. *Addiction as a Computational Process Gone Awry.* Science 306(5703): 1944–1947.
- Science: https://www.science.org/doi/10.1126/science.1102384
- PubMed: https://pubmed.ncbi.nlm.nih.gov/15591205/
- Author PDF: http://redishlab.neuroscience.umn.edu/Papers/2004%20Redish%20Addiction.pdf
- Used for: modelling addiction as a temporal-difference learning pathology. The reference to the delay-discounting literature on substance use (e.g. Bickel and colleagues) was made from training knowledge and no specific paper was checked.

### Mentioned but not sourced

Golden Gate Claude (Anthropic, 2024, SAE feature clamping demonstration); tooling names (PEFT, repeng, nnsight, TransformerLens, Gemma Scope). These were named from training knowledge as pointers, not as evidence for any claim.

### EPModel documents referenced

- `README.md`, `CLAUDE.md`
- `docs/agent_raising_design_prompt.md`
- `docs/model_explainer.md` (§3 Person fields, §5.6 counterfeit moves, readout traps)
- `docs/bowen_agent_model_spec_v2.md` — requirement IDs cited: M1.A.3, M1.A.4/4a/4d, M1.A.5, M2.A.0a, M2.A.0d, M4.D.1a, M4.D.2, M4.D.3a, M4.D.6a/6b/6d/6e, M5.A.1, M5.B.5, M7.C.1, M10.A.1, M11.C.2, M11.C.16, M11.F.9, M15.D, M16.B
- Corpus tags cited via the explainer: KS05.2, KS05.3, KS05.4, KS11.3, L16.4, L10.12
