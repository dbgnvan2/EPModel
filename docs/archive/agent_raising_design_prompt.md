# Design brief: can agents "raise" other agents the way parents raise children?

You are an AI systems architect. I want a well-reasoned design proposal, not a
summary of the problem. Read the context, then answer THE QUESTION below with
concrete architecture, data structures, and algorithms. Think in terms of
deterministic, reproducible systems (seeded, no randomness from a live LLM in
the decision path), because that is the environment this must live in. Where
you make an assumption, state it explicitly. End with: recommended design,
alternative designs you rejected and why, top 5 risks, and open questions.

## What the system is trying to do

EPModel is a simulation of one three-generation family (12 agents: grandparents
in Gen 1, parents in Gen 2, young adults/adolescents in Gen 3, plus one
external counsellor), built on Bowen family systems theory. Bowen theory says
emotional process — how much anxiety a person absorbs, how they react under
stress, how they manage closeness/distance — is transmitted across generations
inside families. The model's objects are Person, Relationship, Triangle, and
Family; the unit of analysis is the relationship, not the individual. It runs
two clocks: a fast weekly tick (events, appraisal, acute anxiety, tie tension,
symptom accumulation) and a slow yearly tick (chronic anxiety, differentiation
drift, life stage, mortality). Each agent loops perceive → appraise → select →
act from a fixed repertoire of nine moves (distance, conflict, projection,
over/under-functioning, etc.), chosen by a seeded softmax over propensities.
There is NO language model in the decision path: everything is deterministic so
thousands of seeded ensemble runs are cheap and bit-reproducible, and outputs
are reported as ranges/envelopes over seeds, never as predictions about a real
family. Constants are flagged [I] = invented magnitude where the theory only
gives direction.

The end goal: the simulator should produce believable, heterogeneous family
members whose emotional fingerprints differ in theory-consistent ways — the
way real siblings and generations differ — and whose differences are *caused*
by the family's own dynamics, not hand-assigned.

## Established findings the design must respect

1. Human emotional process/reactivity/regulation is formed primarily in the
   family of origin across roughly ages 0–20, with two high-plasticity
   windows (early childhood; adolescence, which recalibrates and extends into
   the mid-20s). By the start of emerging adulthood the level of
   "differentiation of self" is largely stable; later life changes it only
   with sustained effort (e.g., therapy).
2. Transmission is multi-channel, not one mechanism:
   - genetic/temperament: personality is ~50% heritable; a child arrives with
     innorn reactivity ("basic building blocks of self");
   - relational/emotional (the strong channel): attachment — the child builds
     internal working models of self and others from how caregivers respond
     to their distress; and emotion socialization — parents who coach vs.
     dismiss emotions raise children with different regulation abilities;
     parents' own anxiety and unresolved history shapes children directly;
   - nonshared experience: children in the SAME family end up markedly
     different (siblings correlate only ~0.1–0.16 on personality;
     parent–child ~0.1–0.14 on traits). Differential parental focus matters
     more than uniform parenting;
   - bidirectional/evocative: the child's temperament evokes different
     parenting, and children change their parents.
3. Bowen-specific mechanics the literature supports: the multigenerational
   transmission process (patterns move down generations); the family
   projection process (the child most emotionally involved in parental
   anxiety becomes the least differentiated — the "identified patient"
   dynamic); triangulation (two people recruit a third to absorb anxiety);
   emotional cutoff (reduced contact = less acute exchange but unresolved
   chronic anxiety); sibling position shapes roles.
4. Generative-agent evidence: LLM agents with no individual history collapse
   to a homogeneous average ("clones"); persona dispersion is compressed
   3—4 versus humans; even scripted life events barely move their
   parameters. What individuates real-agent replicas is rich life history
   (2-hour interviews beat demographics at reproducing real people). Human
   uniqueness therefore looks like it must be *generated* by a developmental
   process, not pasted on as a trait vector.

## THE QUESTION

How should EPModel be designed so that agents "raise" other agents — i.e., a
child agent's adult parameters (differentiation, chronic-anxiety baseline,
appraisal thresholds, move propensities, working models of self/others) are
the OUTPUT of a simulated upbringing by its parent agents, in the way real
parents emotionally program their children?

Design for the specific cast: Gen 3 contains Nadia (17, the projection
target), Pia (14, minimally involved), and Leo (22, launched), being raised
by Gen 2 (Ravi & Marta); Gen 2 was itself raised by Gen 1 (Teodor & Ana,
plus siblings Bruno — cut off — and Sofia — 200 miles away). Most childhoods
precede the simulation's start, so part of the design is *offline*(generating
adult priors from a simulated childhood) and part may be *in-window* (the run
includes ~ages 14–20 for Pia/Nadia).

Answer these explicitly:

A. WHAT is transmitted vs. INBORN? Split each child agent's makeup into a
   heritable/temperament prior (draw correlated with parents, ~weak: 0.1–0.2
   trait correlation) and an environmentally calibrated layer (appraisal
   thresholds, regulation strategy, working models, chronic-anxiety setpoint,
   move-propensity offsets). Which specific parameters exist in each bucket,
   and how do they combine at adulthood?

B. WHAT ARE THE "RAISING" MECHANISMS? Design 3–6 concrete, deterministic
   transmission channels running through the Relationship layer during
   childhood years, e.g.: (1) caregiver response to child distress (coaching
   vs. dismissing → child's regulation competence and model of others);
   (2) exposure to parental chronic anxiety and triangulation (child absorbs
   tension when parents recruit it); (3) projection targeting (parental
   anxiety focused on one child lowers that child's differentiation while
   siblings escape — Bowen's projection process); (4) modeling (child's
   propensities drift toward observed parental moves); (5) information
   given/withheld (secrets, cutoff explanations → models of the world);
   (6) sibling-position effects (oldest/middle/youngest get different
   parenting because parents have changed). For each: what state it reads,
   what it writes on the child, and at what rate.

C. TIMING & PLASTICITY. How does the child's learning rate change across
   developmental stages (high in early childhood, dip, reopen at
   adolescence, consolidate ~20)? How is this implemented — age-gated update
   rates on the slow tick, windows where specific channels are active?
   What happens after consolidation (why a 40-year-old agent's parameters
   resist change except through rare nodal events)?

D. BIDIRECTIONALITY. How do children change their parents (raising Nadia
   raises Marta's anxiety and shifts Ravi & Marta's marriage; Leo launching
   lowers household tension)? How does differential involvement (the
   continuous involvement-weight machinery) decide WHICH child absorbs the
   projection, and how do siblings diverge from identical inputs?

E. OFFLINE vs. IN-WINDOW. Two regimes must be designed: (i) an offline
   "childhood simulator" that produces each adult's priors from its parents'
   own parameters + a temperament seed + stochastic nodal events, run once
   per ensemble seed so developmental randomness is part of the ensemble;
   (ii) the full-fidelity in-window regime for Pia (14→) and Nadia (17→)
   where raising happens live under the real clocks. How do the two regimes
   share mechanisms so an offline-raised adult is indistinguishable in kind
   from an in-window-raised one?

F. VALIDATION. Propose falsifiable signatures that would show the raising
   machinery works, e.g.: projection target ends up the least differentiated
   sibling; within-family variance exceeds what hand-assigned priors
   produced; parent–child correlations land in the empirical ballpark
   (weak for traits, stronger for relational style); Gen 3's spread is
   comparable to real sibling divergence, not clones and not random.
   How would you test each in a seeded ensemble?

G. PARENTHOOD AS A LIFESTAGE. Gen 2 and Gen 3 agents are themselves being
   changed by raising children — does the parent agent need an explicit
   "parenting mode" (new moves? new appraisal triggers?) or do existing
   nine moves suffice with new targets/contexts?

Constraints: deterministic and seeded throughout; no LLM calls in the
decision path; ensembles must stay cheap (thousands of runs); every constant
flagged [I] with a suggested sweep range; must stay faithful to Bowen theory's
mechanisms (see glossary) while using empirical numbers only where the
literature gives them. Do NOT propose making the agents LLM-driven as the
solution — the question is how to do development deterministically.

## Mini glossary (Bowen theory)

- Differentiation of self: capacity to separate thinking from feeling under
  stress; low = fused/absorbing others' anxiety, high = calm, principled.
- Multigenerational transmission: emotional patterns pass down generations,
  each generation's least-differentiated child carrying the most forward.
- Family projection process: parents' anxiety focuses on one child, lowering
  that child's differentiation.
- Triangle/triangulation: a dyad under stress pulls in a third member to
  absorb/distribute anxiety.
- Emotional cutoff: managing unresolved intensity by reducing contact.
- Chronic vs. acute anxiety: chronic = ongoing system-level tension;
  acute = triggered spikes. Acute anxiety exposes the family's limits.
- Sibling position: birth order shapes roles and expectations.
