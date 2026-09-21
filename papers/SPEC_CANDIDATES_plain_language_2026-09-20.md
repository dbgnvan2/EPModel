# The EPModel spec candidates, in plain language (C1–C42)

Date: 2026-09-20. A plain-English translation of `SPEC_CANDIDATES_from_preprints_2026-09-20.md`, which holds 42 proposed additions to the EPModel specification drawn from fourteen papers read in full. This file restates each candidate in everyday terms with a worked example. It is a companion to the source file, not a replacement: every item ends with a trace-back line (spec module + source paper) so you can go back to the technical version.

The source file assigns each candidate a priority (P1–P4). The P1 items (C1–C15) are design decisions that are cheap to make before Phase B and expensive to retrofit, so they get the full treatment below (plain restatement, a worked Person-1…N example, what breaks if skipped, cost/risk). The P2 items (C16–C25) are test-design rules and the P3 items (C26–C42) are Phase E "how to run and read out the experiment" rules; those get the compact treatment.

---

## A short glossary, once

These are the words the spec uses as shorthand. Each item below leans on them.

- **The model / engine / simulation** — the EPModel computer program: a simulated family (up to ~12 people) whose emotional interactions play out week by week over years and decades.
- **Person** — one simulated family member, labelled Person 1, Person 2, … Person N.
- **Tick** — one step of the clock. A **fast tick** is roughly a week (people choose actions, events happen). A **slow tick** is longer (aging, mortality, slow drift of chronic anxiety).
- **Move** — one action a person chooses in a tick: one of nine named moves (PURSUE, DISTANCE, CUTOFF, OVERFUNCTION, UNDERFUNCTION, TRIANGLE, FIGHT, and the like), plus **WITHHOLD** ("do nothing this week").
- **Event** — one move actually happening between people. It has a **sender** (who did it), a **target** (who it was aimed at), and **witnesses** (who else observed it).
- **Appraisal** — what a person's inner state does when an event reaches them: roughly, how much their anxiety goes up.
- **Fidelity** — how much of the original event survives the trip to a listener. A remark overheard from another room arrives weaker than one said to your face.
- **Tie / relationship / conductance** — a relationship between two people. "Conductance" is how open or close the connection is — how much anxiety or influence flows through it.
- **Triangle** — a three-person pattern in which a third person gets pulled into the tension between two others.
- **Anxiety, basic_level, chronic anxiety, functioning level** — the model's state variables. `basic_level` is each person's baseline level of differentiation (emotional maturity). `chronic anxiety` is accumulated anxiety. `functional level` is how well the person is functioning day to day.
- **[I]** — a number the model invents because Bowen's writing gives no value for it. The brackets mark "this is an invented constant, labelled as such."
- **Arm** — one version of the simulation. To test a mechanism you run the same family twice, changing only one thing, and compare: each version is an "arm." "Two arms" = the same family under two different rules.
- **Seed** — a random starting point. Same seed + same inputs should produce the same run. "Per seed" means you line up the two arms using the same seed and compare the same chance events.
- **Ensemble** — many runs with different seeds, combined to see whether a result holds on average rather than by luck.
- **Mutant** — a deliberately broken copy of the model (one rule changed or deleted) used to check that a test is actually testing something. **"Go red"** means the test fails — which is exactly what you want when you've broken the thing the test is supposed to catch.
- **Criterion / acceptance test** — one of the checks the model must pass (the spec's M11 section), e.g. "does the model form triangles the way Bowen says."
- **Readout** — the number you measure at the end of a run (who ends up triangled, the anxiety distribution, etc.).
- **Module (M1 … M16)** — the spec is split into numbered modules: M1 objects, M3 clocks/randomness/ordering, M4 move-selection policy, M6 conservation invariants, M7 life-stage, M8 triangles, M9 beliefs, M10 constants, M11 acceptance tests, M15 import/initial conditions, M16 logging/trace.
- **Phase B–E** — the build stages. Phase B is building the model; later phases are testing and running the full experiment. **Phase E** is the "run the whole experiment" stage (ensembles, arms, readouts), deliberately left unspecified in the current spec.
- **Corpus / design lessons** — the body of Bowen-theory reading (and an earlier set of 29 papers) the model is built against.

---

## Part 1 — Design decisions (P1)

### C1 — Random numbers keyed to *what they're for*, not drawn in sequence

**Plain version.** Every time the model needs a random number, it must compute that number from the run's seed plus a fixed label saying what the number is for — which tick, which person, what purpose — using a special kind of generator (Philox or Threefry) that doesn't carry a running state forward. The engine must *not* keep one big random-number generator that gets consumed one draw at a time in sequence. Also, each random decision must use a fixed number of random values — never "rejection sampling," which uses however many it happens to need.

**Worked example.** Suppose the model flips a coin in week 5 to decide whether Person 1 pursues Person 2. With a sequential generator, that coin is "whatever happens to be the 37th draw." Now run a second arm in which Person 3 died earlier and skipped four draws: the "Person 1 pursues in week 5" coin is now the 33rd draw — a *different* random number. The two arms aren't comparing the same coin; they're comparing different chance events, so the difference you measure is partly re-aligned noise rather than the mechanism you changed. With keyed draws, "Person 1, week 5, pursuit decision" always gets the same number in every arm, so the comparison really is the same chance under two mechanisms.

**Why it matters / what breaks if you skip it.** The project's trusted output is the per-seed paired difference between arms. A sequential generator quietly turns that difference partly into a measurement of noise re-alignment whenever an arm changes control flow — and any arm that deletes a mechanism does. This is the strongest single finding of the sweep, and it *corrects* the current spec (M3.D.4, which specifies a single seeded generator). Within one arm the current design is a valid model; the defect is only in across-arm comparison, which is exactly where EPModel's trusted output lives.

**Cost/risk.** Low to moderate. NumPy ships Philox; a `draw(seed, key)` wrapper is the whole interface. The real risk: the hash that turns a key into a counter must be fixed and documented — Python's built-in `hash()` on tuples is salted per process and would break byte-identical logs. Also, keys that are too fine-grained approach independent seeds and lose the pairing.

**Trace-back:** M3.D.4 (correction) · Buffalo, Pearson & Klein 2026 (2603.11084); Holland et al. 2026 (2607.29546); Li & Tao 2026 (2603.00113).

---

### C2 — People keep the same identity across arms

**Plain version.** Each person (and each tie, each triangle) must have an identifier that does not depend on what happened in the run. Founders get fixed identifiers from the family definition; a person born during the run gets an identifier derived from the parents' identifiers plus birth order. Identifiers must *never* be handed out from a running counter.

**Worked example.** Suppose identifiers are assigned 1, 2, 3… in order of appearance. In arm A, Person 3 dies at 25 before having children, so Person 4's first child becomes #7. In arm B, Person 3 lives and has a child first, so Person 4's same child becomes #8. Same person, different number in each arm — so you can't line the two arms up person by person, and C1's keyed draws break, because "Person #8" points at a different person in each arm.

**Why it matters / what breaks if you skip it.** It's a prerequisite for C1 (the keys carry stable identifiers), and it lets the trace viewer align two arms person by person. Trivial to do now, painful to retrofit.

**Cost/risk.** Trivial at design time; no conflicts.

**Trace-back:** M1, M2, M15 · Buffalo et al. §4.3.

---

### C3 — Every kind of random draw declares what its key is

**Plain version.** The spec must list every kind of random draw in the engine (move selection, fidelity per hop, latency, witness overhearing, spell onset and duration, mortality, symptom onset, tie-breaks) and, for each, say exactly what its key is — and whether it is **slot-keyed** (tick + actor + purpose, so *which partner* enters only through state) or **dyad-keyed** (tick + actor + partner + purpose, so a different partner is a different chance event). Deliberately coarse keys must be marked as such.

**Worked example.** When Person 1 decides whether to pursue this week, is the coin "Person 1's pursuit decision in week 5" (slot-keyed: one coin regardless of whom she pursues), or "Person 1 pursuing Person 2 in week 5" (dyad-keyed: a different coin for each possible target)? Slot-keying assumes the choice of target is explained by the modelled state; dyad-keying allows partner-specific residual chance. A sequential generator silently sets "event identity" to "draw index," hiding this decision. The reader's concrete proposal: move selection slot-keyed on (tick, actor); fidelity and overhearing dyad-keyed; exogenous spells keyed by (family, spell class, occurrence index); symptom onset keyed by (tick, person, channel).

**Why it matters / what breaks if you skip it.** Defining the key *is* the decision of what counts as "the same event" across two worlds. It's a documentation rule that makes an assumption explicit, so results can't be silently contaminated by an accidental choice.

**Cost/risk.** Documentation plus wrapper discipline.

**Trace-back:** M3 (a table) · Buffalo et al. §4.2.

---

### C4 — A "placebo arm" must reproduce the baseline byte-for-byte

**Plain version.** Add a placebo test: an arm that turns on an extra mechanism at zero strength (extra random draws, no change to state) must reproduce the baseline trajectory byte-for-byte on every seed. The engine must also query each event key at most once per run and cache the answer; a debug build must fail loudly if the same key is queried twice.

**Worked example.** Take the baseline family. Build a "placebo arm" identical to it except that it also performs, say, a witness-appraisal step that multiplies everything by zero — it draws random numbers but changes nothing. If those extra draws (in a sequential generator) shift every later draw, the placebo arm's trajectory diverges from baseline and the test fails — exposing the coupling violation. With keyed, cached draws (C1), the extra zero-magnitude step perturbs nothing and the trajectory stays byte-identical.

**Why it matters / what breaks if you skip it.** A cheap test that fails loudly under any regression to a stateful generator, and it makes the existing "delete the mechanism, confirm the test goes red" discipline measure the *mechanism* rather than noise re-alignment. Note: passing does *not* prove invariance — the paper says so explicitly.

**Cost/risk.** Trivial.

**Trace-back:** M11.D, M3 · Buffalo et al. §2.3, §4.3.

---

### C5 — Shuffling the order of people must change nothing

**Plain version.** Add a test that, for a fixed seed, permutes the order in which persons are processed in the selection step and the order in which events are delivered within each same-tick batch, and asserts a byte-identical final state and an order-normalised identical log. The test must be *proven* to fail when you replace the commutative batch reduction with a sequential one.

**Worked example.** In one week, Persons 1, 2 and 3 all act "simultaneously" (a batch). If the engine processes them in order 1-2-3 versus 3-2-1, the outcome must be the same, because the model says they act at once, not in sequence. The test runs both orders and checks the results match. To prove the test is real, you build a broken version that applies each person's effect one after another (so order matters) and confirm the test now fails.

**Why it matters / what breaks if you skip it.** The source shows real systems where order alone flips results (an LLM debate: first-round consensus ~40% under one speaking order, ~90% under the other). This guards against order secretly deciding EPModel's outcomes. Watch the trap: a commutative reduction can still be defeated if a stateful generator is consumed in loop order — C1 removes that trap.

**Cost/risk.** Low.

**Trace-back:** M11.D · Li & Tao Action 2(3); Sachdeva & van Nuenen 2025 (2510.10002).

---

### C6 — "Who acts" and "who sees" become named, versioned parts

**Plain version.** The engine must implement two things as separate, named, versioned components: **activation** (which persons choose a move in a given fast tick) and **visibility** (which persons are targets or witnesses of an event, with what latency and fidelity). The run-log header must record the identity and version of each. The default activation — "every person acts every fast tick" — must be recorded as such, not left implicit.

**Worked example.** Right now the spec fixes "everyone acts every fast tick," and visibility is scattered across several modules (the witnesses field, the route/fidelity rules, per-edge latency, the witness-vs-triangle rule). But neither is a named object with a version in the log header. This matters when you want to run the "what if people act in turn instead of simultaneously?" experiment: it becomes a component swap instead of a rewrite, and an auditor can reconstruct who could act and who could see from the log alone.

**Why it matters / what breaks if you skip it.** Software structure and reporting; inference validity. A social simulation is only well-defined once it names a scheduler and a visibility object.

**Cost/risk.** Low at design time; none of the existing logging rules are touched.

**Trace-back:** M3, M1.F, M16.A · Li & Tao Def 4.1, Action 1; Sachdeva; Buitrago López et al. 2026 (2606.12369).

---

### C7 — Witnesses are computed by the visibility rule, not picked by the sender

**Plain version.** Who witnesses an event must be computed by the visibility component (C6) from the family's ties and household state (co-residence, how open the relationship is, the route). The sender's policy must not be allowed to freely set the witness list. Where the theory needs a sender to choose an audience, that choice must be expressed as a move (a TRIANGLE addressed to a third party), not as witness selection.

**Worked example.** Person 1 criticises Person 2 at the dinner table. Who overhears — Person 3 in the next room, Person 4 on the phone — must be decided by the visibility rule (who lives here, how close the ties are), not by Person 1's policy saying "let's make Person 3 witness this." If Person 1 wants to deliberately draw Person 3 in, that's a TRIANGLE move, which is a different thing with its own rules.

**Why it matters / what breaks if you skip it.** Theory fidelity: who overhears is a property of the household, not of the speaker's intent. This is probably already the intended implementation; the requirement costs one sentence and prevents a shortcut.

**Cost/risk.** Low.

**Trace-back:** M1.F, M4.E · Li & Tao Def 4.1.

---

### C8 — A witness's reaction is computed from *both* sides of the conflict

**Plain version.** When Person 3 watches Person 1 attack Person 2, how upset Person 3 gets must be worked out from three things: Person 3's own state, Person 3's relationships with *both* Person 1 and Person 2, and how heated the exchange was — with a witness-specific weighting constant labelled [I]. It must *not* be a fidelity-scaled copy of the target's reaction.

**Worked example.** Mother (Person 1) snaps at father (Person 2) over dinner. Two children witness it, and the remark reaches both at the same fidelity. Under the shortcut ("witness = a fraction of what father felt"), both children are shaken by the same fraction, so they respond identically. But Person 3 is close to mother and distant from father, while Person 4 is close to father and wary of mother. In Bowen's account those two children are not in the same position at all: Person 3 is pulled toward mother's side and feels the conflict as a threat to that alliance; Person 4 is pulled toward father. Which child gets drawn into the parents' conflict, and on which side, depends on each child's ties to *both* parents — that is what makes a triangle rather than three separate reactions. The shortcut throws away exactly the information that decides who gets triangled in.

**The [I] constant.** A dial setting how strongly a bystander reacts compared with a direct target. Invented and labelled as such, because the corpus gives no number.

**Why it matters / what breaks if you skip it.** This is the one place the sweep adds a *mechanism* rather than just a test. The current spec (M1.F.5, M4.C.1) says witnesses appraise events and gives one appraisal formula, but never says which tie's strength applies when the person is a witness rather than a target. In Bowen's account the third party's involvement depends on its ties to *both* members of the anxious dyad — that's the gap this fills.

**Cost/risk.** Moderate — adds one appraisal path and one [I] constant. Caution from the reader: the source paper's "pull toward ±1 as absorbing extremes" is a property of its functional form and must *not* be imported along with the witness rule.

**Trace-back:** M4.C, M1.F.5, M11 · Holland et al. 2026 (2607.29546).

---

### C9 — A person may only act on their own state and beliefs (no god-view)

**Plain version.** A person's chosen move must be computed only from (a) their own state, (b) their beliefs — including their belief about any tie they are not party to — and (c) events delivered to them as target or witness after per-hop fidelity. It must *not* read another person's true state, the true state of a tie it is not party to, or an undelivered event. The log must record the belief value actually used, so the trace can show "believed" vs "true." Add a mutant that reads true state instead of belief, which must fail a belief-layer test.

**Worked example.** Person 3 is deciding whether to distance from Person 1. She may use her *belief* about how strong the Person 1–Person 2 tie is, but not the model's actual stored value for that tie — she isn't part of it, so she shouldn't know it. If the policy is allowed to read the true value, misperceived alliances become impossible, and every triangle would be based on accurate information, which contradicts the theory. The source shows how much this matters: when agents had to *infer* relationships from interaction instead of being handed the relationship graph, the balanced-state rate fell from 60.7% to 34.4%.

**Why it matters / what breaks if you skip it.** Converts an intent into a mutation-provable constraint. The spec has a belief layer (M9.1–M9.7) but nowhere forbids reading true state — and no test would catch it. This is both theory fidelity (misperceived alliances become possible) and inference validity.

**Cost/risk.** Low for the rule and the mutant; moderate if a per-person, per-tie belief has to be added to M1.

**Trace-back:** M4, M1, M9, M11, M16 · He 2026 (VISA, 2607.28027); Zhou et al. 2026 (PIMMUR, 2509.18052).

---

### C10 — A register of every variable and mechanism, with automatic checks

**Plain version.** The object model must come with a register that classifies every state variable (exogenous vs endogenous, decision vs derived) and lists which mechanisms read and write it. Every mechanism must list what it reads, what it writes on its owner, and what it writes on other objects. The phase order must list each mechanism with its execution mode and trigger. A check must fail on a variable with no writer, a mechanism not placed in the phase order, or a mechanism that reads a quantity its owner cannot observe.

**Worked example.** If "Person 3's chronic anxiety" is a state variable, the register says who writes it (the slow-tick anxiety mechanism) and who reads it (the appraisal mechanism). If some mechanism reads Person 3's functioning level but Person 3's policy is not supposed to observe it (that's C9), the static check flags it. A variable nothing writes — something the model can never change — gets caught automatically.

**Why it matters / what breaks if you skip it.** Documentation plus a static check. The third check (reading what the owner can't observe) is the static, compile-time form of C9.

**Cost/risk.** Low — a table plus a scan.

**Trace-back:** M1, M3, M10, M16 · VISA r1, r3, r10, r11, r14, r16.

---

### C11 — The list of legal moves is computed and logged *before* choosing

**Plain version.** Each week, before a person chooses, the policy must first compute the *legal* set of moves (CUTOFF requires a live tie; TRIANGLE requires a reachable third party; PURSUE requires a non-cutoff target; plus the financial-dependence and I-POSITION gates), select only from that set with renormalised weights, and write the mask to the log — so a move that was never legal is distinguishable from one that was legal but never chosen.

**Worked example.** If Person 1 has cut off Person 2, then "PURSUE Person 2" is illegal. In the log you need to be able to tell: did Person 1 never pursue Person 2 because it was *illegal* (there was no tie), or because it was legal but never chosen? Without the logged mask, both look identical, and a test like "the model rarely pursues after a cutoff" becomes unreadable.

**Why it matters / what breaks if you skip it.** Makes the rare-move coverage test interpretable, and moves "what is possible right now" out of the policy into an auditable mechanism. Both theory fidelity and inference validity.

**Cost/risk.** Low.

**Trace-back:** M4.D, M16.A · Buitrago López et al. §3.2.

---

### C12 — Every decision is tagged "chosen by the policy" or "chosen by fallback"

**Plain version.** Every move-selection log record must flag whether the outcome was chosen by the scored policy or by a fallback/tie-break rule (equal scores, a non-finite score, exhausted life energy, an empty legal set). Ensemble readouts must report the fallback rate per move and per person; a criterion whose passing ensemble has a fallback rate above a declared threshold must be flagged. The tie-break rule itself must be stated, and its draw keyed (C3).

**Worked example.** Two moves score exactly equal in a week — which one does Person 1 pick? If the engine silently always takes the first-listed move, that silent rule could be what's generating the tested behaviour, not the theory. This requirement says: log that it was a tie-break, state the tie-break rule, and flag any criterion where the fallback is doing the work.

**Why it matters / what breaks if you skip it.** The spec currently has *no* tie-break or fallback rule at all (zero mentions). Softmax selection will need one for equal scores, NaN, and exhausted life energy — and a silent fallback can fake the tested behaviour.

**Cost/risk.** Low; the log stays a pure observer.

**Trace-back:** M4.D, M16.A, M11 · Ye et al. 2026 (TRAILS, 2605.18890).

---

### C13 — Relief from repeating the same move must fade (or be proven safe)

**Plain version.** Any term that credits anxiety relief for repeating the same move within a window (reassurance-seeking PURSUE, checking-type OVERFUNCTION) should decay geometrically with repetition count — with the decay constant labelled [I] — or be shown by sweep not to be bistable (ignored below a threshold, absorbing above it).

**Worked example.** If Person 1 gets anxiety relief every time she overfunctions for Person 2, and that relief never decays, the model can reach a state where the relief always outweighs the cost and the overfunctioning never stops (an absorbing state), while just below the threshold it never starts — with no middle ground. A geometric decay (each repeat gives less relief) produces the graded, in-between curve the theory expects.

**Why it matters / what breaks if you skip it.** Guards against a bistable functional form. The source paper showed a checkpoint-return bonus without habituation was bistable, and a geometrically diminishing bonus produced the graded curve.

**Cost/risk.** Low — and only if such a relief term exists (the spec search couldn't settle this).

**Trace-back:** M4, M10 · Prasad 2026 (2607.07753).

---

### C14 — Check that every move can actually happen

**Plain version.** Add a static reachability test: for a three-person sub-family over one fast tick, enumerate the move assignments the policy can produce from any state within the declared ranges, and assert that every one of the nine moves and WITHHOLD is reachable; list the unreachable ones.

**Worked example.** Take Persons 1, 2 and 3 as a triad. If, from every possible starting state in the declared ranges, the policy can never produce a TRIANGLE move (say, because the required third party is never reachable), the test lists TRIANGLE as unreachable. That's a bug — a whole move class is silently filtered out — and a normal run would never reveal it.

**Why it matters / what breaks if you skip it.** Closes the "silently filtered behaviour" risk at the source. With C11's legal mask in place, the test becomes "every move is legal somewhere in the state box, and where legal, receives non-negligible propensity."

**Cost/risk.** Moderate if the policy is piecewise linear (solvable as a linear program); otherwise dense sampling, which weakens "unreachable" to "not observed" and must be reported as such.

**Trace-back:** M11 · Kurz 2025 (2512.18016).

---

### C15 — Say where a dead person's "stuff" goes

**Plain version.** The spec must name the mechanism that removes a Person at death, and state where that person's acute and chronic anxiety, bond energy on each tie, functioning-balance debts, investment, triangle positions and share of the undifferentiation budget go — so the conservation invariants (anxiety is conserved; no one exits the field) remain checkable across a death. Add an invariant test spanning a death. If births occur within the horizon, the creating mechanism and the initial state it assigns must be named likewise (and the identifier rule C2 applies).

**Worked example.** Person 3 (a child) dies. His share of the family's anxiety budget, his bond energy on each tie, and his triangle position — do they vanish (which would destroy anxiety and fake "the family calmed down"), or are they absorbed into the multigenerational ledger? The spec tests deaths (M11.C.15) but never says where the dead person's conserved quantities go, so the conservation invariants (M6.I.6, M6.I.7) cannot be checked across a death.

**Why it matters / what breaks if you skip it.** Closes a hole in the invariants: a leak at death would fake destroyed anxiety and pass the conservation tests everywhere else. Needs an owner decision — is death an exit, or does the ledger absorb the person?

**Cost/risk.** Small if the ledger already absorbs it; *the decision* is the cost.

**Trace-back:** M1, M6, M7.C, M11 · VISA r8, r13.

---

## Part 2 — Test-design rules (P2)

### C16 — Sweep graded parameters over four or more levels, not two

**Plain version.** For each person parameter the theory treats as graded (at least `basic_level` and `chronic anxiety`), add a test that sweeps the parameter over four or more levels across the ensemble and asserts a monotone ordering of a pre-declared readout — with the ordering violated in the mutant.

**Worked example.** Run the family with Person 1's chronic anxiety at four levels (four arms) and check that a declared readout (say, how triangled she ends up) moves monotonically across the four. A two-arm test can't see a threshold or step shape — the source paper's own monotonicity claim overstates its tables (some knobs are non-monotone or step-like within the confidence intervals), which is exactly the point.

**Why it matters / what breaks if you skip it.** The corpus supplies orderings, and an ordering over four levels is a stronger use of it than a sign. Costs about four times the ensemble runs per parameter.

**Trace-back:** M11 · Prasad Tables 2, 7.

---

### C17 — Break the *contingency* of an input, not just its magnitude

**Plain version.** For each appraisal input that selection depends on, add a mutant that preserves the input's per-tick magnitude distribution but destroys its state-contingency (a seeded shuffle across persons or ticks, keyed per C3). The affected criterion must go red under it.

**Worked example.** Take the "anxiety penalty" input. Instead of deleting it, keep its size and how often it fires, but shuffle which person/tick gets which value. In the source this collapsed risky-route avoidance from 1.00 to 0.00–0.20. This rules out "any input of that size would do" — stronger than the existing delete-or-invert discipline.

**Why it matters / what breaks if you skip it.** Inference validity: it proves selection depends on the *contingency* of the input, not just its magnitude.

**Trace-back:** M11 · Prasad Fig. 3, Fig. 12.

---

### C18 — A pattern learned during a spell persists (with a learning-off arm as control)

**Plain version.** Add a test that a relational pattern established during an exogenous anxiety spell persists after the spell ends in the arm where the automatic channel learns, and remits in the arm with learning disabled; report the time-to-remission distribution per seed.

**Worked example.** A spouse's illness (an exogenous spell) drives Person 1 to distance from Person 2. After the illness resolves, in the learning-on arm the DISTANCE persists — the avoidant policy never re-encounters the evidence that would disconfirm the learned response — while in the learning-off arm it remits. This should be an emergent property with no persistence rule written in, which makes it a clean acceptance test and a clean mutation target (learning off → remits).

**Why it matters / what breaks if you skip it.** Theory fidelity; confirms the persistence is coming from the learning mechanism, not a hard-coded rule.

**Trace-back:** M11, M10 · Prasad Tables 4, 11, 12.

---

### C19 — Test the *rank order* of mechanisms, not just each sign

**Plain version.** Where the corpus supplies an ordering of effect strengths across three or more mechanisms, a criterion must assert the rank order of the per-seed paired arm differences, not only the sign of each. It passes when the observed ordering matches the corpus ordering in a pre-declared majority of seeds, and deleting the mechanism ranked first must break the ordering.

**Worked example.** If the corpus says mechanism A > B > C in how strongly each raises anxiety, run three arms and check the rank order A > B > C — not just "A raises it," "B raises it," "C raises it." The source reproduced only 4 of 8 effect sizes but ranked all 8 correctly — orderings are the testable middle ground when magnitudes aren't trusted.

**Why it matters / what breaks if you skip it.** Theory fidelity and inference validity; uses the corpus's orderings without assuming any magnitudes.

**Trace-back:** M11 · Kalluri 2026 (2603.01189).

---

### C20 — Separate "how big" from "how often" in asymmetric rules

**Plain version.** For every mechanism whose constants are asymmetric (a loss or hardening rate larger than the matching gain or recovery rate), the run log must record per-event magnitudes and event counts separately, and the asymmetry test must hold event counts equal across compared arms (or condition on them).

**Worked example.** A per-event rule where negative outcomes weigh 1.5× positive produced cumulative loss/gain ratios that varied with failure frequency — the per-event asymmetry did not produce a cumulative asymmetry. If you compare two arms with different event counts, the cumulative total confounds the rule with the frequency. Relevant to hardening, lock-in, and any "cutoff is easier than reconnection" readout.

**Why it matters / what breaks if you skip it.** A reporting and test-design rule; a cumulative total confounds rule and frequency.

**Trace-back:** M16, M11 · Kalluri Table 5, §7.1.

---

### C21 — Re-encode the numbers; the *directions* must not change

**Plain version.** Add re-encoding mutants that are numerically equivalent by construction (rescaled state ranges, changed float summation order, altered rounding or clamping thresholds within tolerance, integer vs float tick counters), and every directional criterion must be unchanged under them. A change is an encoding artefact, not a mechanism result.

**Worked example.** Rescale all anxiety values from 0–1 to 0–100. The numbers change but the *direction* of every result must not. If a result flips on rescaling, it was an artefact of the representation (e.g. a rounding threshold), not the mechanism. Because rescaling changes bytes, this test is on directions, not on byte identity (M3.D.5).

**Why it matters / what breaks if you skip it.** Catches rounding and threshold artefacts; complements C31, which locates the comparisons that rounding actually decides.

**Trace-back:** M11, M6 · TRAILS §3.1, §4.1.

---

### C22 — No rule may name its own conclusion (audit + invert both ways)

**Plain version.** For each criterion, list the minimal set of rules and constants whose joint operation produces the asserted direction, and no rule in that set may name the criterion's readout or asserted outcome as its own target. Run each such rule as both a deletion *and* a sign-inverted mutant; a criterion that flips under inversion of a single rule must be re-stated one level up or documented as a programmed premise rather than a derived result.

**Worked example.** If a criterion says "anxious families triangled more," list the rules that produce it. If one rule effectively reads "if anxiety is high, push toward more triangled," that's the criterion wearing a costume — an outcome directive, not a mechanism. Invert that rule: if the criterion flips, it was programmed, not derived. The spec already has delete-or-invert; this adds running both and a premise-vs-mechanism column.

**Why it matters / what breaks if you skip it.** Separates Bowen's premises from what follows from them; inference validity. Costs about double the mutants for the minimal set.

**Trace-back:** M11, M10 · PIMMUR §4.4.

---

### C23 — Freeze constants before the first test run

**Plain version.** Every [I] constant's value must be recorded before the acceptance suite is first run against it. Any later change that alters an acceptance outcome must be logged with the failing criterion named, and a criterion passed only after such a change must be reported as post-hoc, together with the sweep fraction over which it holds.

**Worked example.** If you tune a constant until a criterion passes, then report the criterion as "passing," that's circular. Kalluri's paper is a worked example of the failure: an asymmetry parameter was calibrated to a target, then the target was reported as a validation result. This rule records constants first, so post-hoc tuning is visible instead of laundered into a result.

**Why it matters / what breaks if you skip it.** The twin of M11.F.9 (no fitting to known histories) for fitting to the acceptance tests themselves. Inference validity; cost nil.

**Trace-back:** M10, M11, M16 · PIMMUR P5; Kalluri.

---

### C24 — A null at the ceiling is a measurement limit, not evidence

**Plain version.** Every acceptance test must record the baseline arm's position within the readout's attainable range, and a null on a readout whose baseline sits at a bound must be reported as an assay limit, not as evidence about the mechanism.

**Worked example.** If the baseline is already at maximum overt emotionality, "this mechanism doesn't increase it" is meaningless — there's no headroom. Report it as a ceiling, not as "the mechanism does nothing." Same trap as a withdrawn agent that can't express avoidance (an anxiety × depression masking).

**Why it matters / what breaks if you skip it.** Saturation is the *other* way a null can be uninformative; the spec's M11.4a already covers the underpowered-null case, but not this one.

**Trace-back:** M11, M16 · Prasad.

---

### C25 — Pick a statistic that works even when an arm never varies

**Plain version.** The direction test must be defined when one arm has zero seed-to-seed variance or discrete bounded outcomes (a rank-based paired statistic on per-seed differences), and where several readouts are tested under one criterion, a multiplicity correction must be declared.

**Worked example.** If arm A's readout is identical in every seed (zero variance), a t-test is undefined — use a rank-based test on the per-seed differences. And if you test the same criterion against ten readouts, some will "pass" by chance unless you correct for it (a Holm correction).

**Why it matters / what breaks if you skip it.** The spec has zero mentions of multiplicity/Holm/Bonferroni and one of "paired." M11.4a covers the null side only.

**Trace-back:** M11 · TRAILS App. C.1; Blando et al. 2026 (2604.04543).

---

## Part 3 — Phase E: running and reading out the experiment (P3)

### C26 — Keep adding seeds until the answer is clear, or say "can't tell"

**Plain version.** For each directional criterion, add seeds in blocks until the confidence interval of the per-seed arm difference has a half-width below a declared δ, up to a declared cap; log the seed count used. A criterion whose interval hasn't converged at the cap is reported UNDETERMINED, not pass or fail.

**Worked example.** Start with 30 seeds; if the interval is still too wide, add 30 more, and repeat up to the cap. High-variance regions get more runs. If it still won't converge, report "we can't tell," not "pass" or "fail."

**Why it matters / what breaks if you skip it.** The spec has no stopping rule and no third outcome; this fixes the underpowered-null problem from the other side.

**Trace-back:** Phase E, M11, M16 · Blando et al. §4, §7.1.

---

### C27 — Compare arms at every tick, and report power on non-rejections

**Plain version.** Test the arm difference at each reporting tick, report the first tick at which the declared direction holds, and report the power of the test wherever the null is not rejected — so "arms did not differ" is distinguishable from "ensemble too small."

**Worked example.** Track the difference week by week; report "the direction first holds at week 30," and where it doesn't hold, report the power (a non-rejection at power 1.0 = genuinely no difference; at power 0.3 = can't tell). Use a paired statistic (C25).

**Why it matters / what breaks if you skip it.** Separates "no difference" from "not enough data."

**Trace-back:** Phase E · Blando §7.4.

---

### C28 — Classify each seed as same, converted, or reversed

**Plain version.** Report, per criterion and per seed, the paired arm difference, and classify each seed as same, converted, or reversed. A directional criterion holds when converted seeds exceed reversed seeds by the pre-declared margin.

**Worked example.** For seed 1, Person 1 is more triangled in arm B than arm A → "converted." Seed 2: equal → "same." Seed 3: less → "reversed." Count them; the direction holds if converted exceeds reversed by the margin. (Requires C1 to be meaningful.)

**Why it matters / what breaks if you skip it.** A richer per-seed readout than a single pooled difference.

**Trace-back:** Phase E · Buffalo et al. §2.2.

---

### C29 — Classify each trajectory, and prove no bound is a trap

**Plain version.** Classify each bounded slow- and fast-state trajectory per seed as settled-at-a-bound, settled-at-its-attractor, oscillating, or inconclusive at the horizon, and report the fractions per arm; a directional criterion must not count inconclusive seeds as passes. Also test that no state bound (functional level, anxiety channels, conductance) is absorbing unless the theory says it is — a run starting a person at a bound must show the mechanism can leave it.

**Worked example.** A run where anxiety climbs to 1.0 and stays there forever is "settled at a bound." If that bound is absorbing and the theory doesn't say it should be, it's an artefact, not a mechanism result. Test: start a person at the bound and show the mechanism can move them off it.

**Why it matters / what breaks if you skip it.** The source paper's ±1 sinks show this is a real risk at EPModel's size.

**Trace-back:** Phase E, M11, M6 · Holland Fig. 6, Cor 4.2.

---

### C30 — If people ever act in turn, report how much order matters

**Plain version.** If Phase E runs an activation regime in which persons select or receive in sequence, the runner must, per seed, run every ordering (or a seeded sample of a declared size), and report the spread of each directional result across orderings as a variance component distinct from the across-seed spread.

**Worked example.** If you ever run "people act in turn," the order they act in can flip results (the numbers under C5). Run all orderings and report how much of the outcome's spread comes from ordering versus from the seed.

**Why it matters / what breaks if you skip it.** Keeps ordering spread from being mistaken for seed noise. Only applies if a sequential regime is ever run.

**Trace-back:** Phase E · Sachdeva §3.3.

---

### C31 — Log the distance from every threshold, and its safe range

**Plain version.** For every [I] constant used as a threshold, the engine should emit, per tick, the signed margin of each comparison against it; Phase E must report, per criterion, the interval of each threshold constant within which every seed's trajectory is unchanged (the minimum positive and negative margins over the run).

**Worked example.** If a rule is "if anxiety > 0.6, escalate," log how close each tick's anxiety is to 0.6. If the margin is 0.001 for the whole run, the result hangs on numerical noise — flag it. And report the range of the threshold (e.g. 0.58–0.63) within which nothing changes.

**Why it matters / what breaks if you skip it.** Turns the constant sweep from sampled into exact for thresholds, and flags comparisons decided within numerical noise (pairs with C21).

**Trace-back:** M16, Phase E, M10 · Kurz Lemma 3.

---

### C32 — Declare how initial people correlate, and report it separately

**Plain version.** Initial state for any run family must be a distribution over attributes that declares its correlation structure (theory-stated pairings such as spouses' matched `basic_level`; orderings among children), not independent per-attribute ranges. The runner draws from it and reports initial-condition variance separately from seed variance, spell-timing variance, and constant-sweep variance.

**Worked example.** If you set each person's `basic_level` independently, you can generate family configurations that make no sense (a matched pair that isn't matched). Declare the correlations, draw from the joint distribution, and report how much of the outcome spread comes from initial conditions versus from the seed.

**Why it matters / what breaks if you skip it.** Sampling attributes independently can break the joint distribution, and initial-condition variance must not be conflated with seed variance. (Must not be tuned to a known history — M11.F.9.)

**Trace-back:** M15, M10, Phase E · Li & Tao §3.2.3; Kurz Ex. 5.

---

### C33 — When arms differ in structure, hold the structural totals fixed

**Plain version.** When counterfactual arms differ in initial tie topology or tie strengths, the arm specification must state which structural totals are held fixed (number of ties, total conductance, total bond energy, generation structure), and the run log must verify them.

**Worked example.** If you vary "who is tied to whom" across arms, hold the degree sequence (how many ties each person has) fixed, so you're testing topology and not tie count. The source used degree-preserving edge swaps for exactly this.

**Why it matters / what breaks if you skip it.** Without it, a "structure" arm is secretly also a "more ties" arm.

**Trace-back:** M15, Phase E · TRAILS App. C.3.3.

---

### C34 — Three control arms: no-interaction (×2), endogenous-only, homogeneous

**Plain version.** Provide (a) two no-interaction controls — all conductance zero, and ties intact with event delivery suppressed — to separate drift from standing load from drift from events; (b) an endogenous-only arm with all exogenous spells removed and societal anxiety constant; and (c) a homogeneous-family arm in which all persons share identical initial `basic_level`, chronic anxiety and tie attributes, with each criterion declaring whether it is expected to pass or fail there.

**Worked example.** (a) A tie that exerts load without any traffic — one source showed a model's verdict shifting under the framing alone, before any message was exchanged. (c) A criterion that "passes" in the homogeneous arm where the theory says heterogeneity is required (e.g. projection onto the most vulnerable child) is a red flag.

**Why it matters / what breaks if you skip it.** Separates mechanism from drift, and checks null-expectations against the homogeneous case.

**Trace-back:** Phase E, M11 · Sachdeva App. D; Holland §6; PIMMUR.

---

### C35 — The engine must not know which arm it's running

**Plain version.** The engine must not receive an arm label, scenario name, test identifier, or readout definition; arms must differ only through the declared channels (initial state, constants, exogenous spells, mechanism switches declared in M10). Add a static check that no policy, appraisal or consolidation module imports from the test or readout modules.

**Worked example.** If a code path branches on "which arm am I?" (e.g. `if arm == 'anxious': do X`), that's the model cheating. The source showed that when the model could see the construct under test, the outcome was 1.77× more frequent. Arms must differ only through the declared knobs.

**Why it matters / what breaks if you skip it.** Prevents circularity; a generalisation of the existing M10.C.4b import guard. Cost trivial.

**Trace-back:** M16.B, M3, M11 · PIMMUR §2.4.

---

### C36 — Every result carries an audit of what you shook and what you didn't

**Plain version.** Every directional result must carry an audit record listing, per design dimension (seed; initial persons and ties; appraisal and belief rules; tick length and estimator windows; activation and same-tick aggregation; intervention timing, target and channel; topology; family composition), whether the dimension was perturbed and whether the direction held, was sensitive, or was **unaudited**; and state the claim grade (exploratory, mechanism, intervention) the result is used for.

**Worked example.** A result claiming "mechanism X drives triangled-ness" must show which dimensions you shook to check it, which held, which flipped, and which you never checked (unaudited) — plus whether you're claiming it as exploration, a mechanism, or an intervention. The source showed sensitivity that was wildly uneven across dimensions.

**Why it matters / what breaks if you skip it.** Makes the robustness of each claim explicit rather than implicit.

**Trace-back:** M16, Phase E · TRAILS §3.2, §5, §6.

---

### C37 — Test sensitivity at two or more configurations, not one

**Plain version.** For each dimension audited under C36, sensitivity must be measured at no fewer than two declared reference configurations of the [I] constants (for instance a low- and a high-differentiation family), because a perturbation that is null at one configuration may flip the outcome at another.

**Worked example.** A perturbation that changes nothing in a high-differentiation family may flip the outcome in a low-differentiation one (the source saw 76 percentage points in one model and about 1 in another). Test both before calling the dimension insensitive.

**Why it matters / what breaks if you skip it.** Avoids false "insensitive" conclusions. Multiplies sweep cost by the number of reference configurations.

**Trace-back:** Phase E · TRAILS §4.2.

---

### C38 — Report how far a pair's joint effect departs from "additive"

**Plain version.** For parameter pairs the theory claims interact (chronic anxiety × `basic_level`; functioning balance × conductance), run a two-dimensional grid and report the maximum residual against the additive prediction from the two one-dimensional sweeps.

**Worked example.** If `basic_level` and chronic anxiety interact, sweep both jointly and report how far the joint effect departs from "effect of A alone + effect of B alone." The source found residuals of 0.82 and 0.50, with the mechanism named (a withdrawn agent cannot express avoidance).

**Why it matters / what breaks if you skip it.** The design lessons sample the joint space but don't report interaction.

**Trace-back:** Phase E · Prasad Tables 13–14.

---

### C39 — Compare *sequences* of moves, not just their totals

**Plain version.** The trace renderer should emit, per person and per arm, the first-order move-transition count matrix; ensemble reports should compare arms on transition structure as well as move marginals; where a criterion concerns the distribution of moves, include per-seed paired Jensen–Shannon divergence (with Laplace smoothing) between arms.

**Worked example.** Does Person 1's DISTANCE tend to follow PURSUE (a pattern), or is her next move independent of the last? Two arms can have identical move *frequencies* but different *sequences*. The transition matrix and the JSD (a per-person measure of how far apart two arms' move distributions are) expose that.

**Why it matters / what breaks if you skip it.** The source's own reference policy is first-order Markov in moves, yet only marginals were scored — a gap this closes.

**Trace-back:** M16, Phase E · Buitrago López §3.

---

### C40 — Separate "habit" from "copying others"

**Plain version.** The analysis layer should fit, per person, a multinomial model of the selected move on: an indicator that the same move was selected last tick (inertia), the count of each move type received or witnessed in the current batch (conformity, within-batch), and counts in earlier ticks (prior), with person and tick effects. Acceptance tests may assert directions on these coefficients (zeroing all conductance must drive conformity toward about zero).

**Worked example.** Is Person 1 doing DISTANCE because she did it last week (inertia) or because she watched others do it (conformity)? Fit the model and read off the coefficients. The source showed the two are statistically separable and interact positively. (Rare moves will have wide intervals — report them.)

**Why it matters / what breaks if you skip it.** Separates own-past from witnessed-others, and within-batch from prior; pure post-run diagnostic.

**Trace-back:** M11, M16 · Sachdeva §2.4.

---

### C41 — Track the gap between what people believe and what's true

**Plain version.** For each belief in the belief layer that has a true-state counterpart, the run log must allow a signed discrepancy (belief − truth) per tick, and Phase E should report its distribution and trajectory per arm alongside the outcome readouts, so arms in which beliefs and outcomes move in opposite directions are identified rather than averaged away.

**Worked example.** Person 3 believes her tie to Person 1 is strong (0.8) but it is actually 0.3 — discrepancy +0.5. Track that over time. The source showed trust and task success decoupling across scenarios, with a calibration-error readout distinguishing them. With C9, this is the quantity the true-state mutant is expected to move.

**Why it matters / what breaks if you skip it.** Identifies arms where beliefs and outcomes diverge, instead of letting them cancel out.

**Trace-back:** M16, Phase E · Kalluri §6.3, Table 9.

---

### C42 — Test triangles over every configuration, not one hand-picked triad

**Plain version.** Triangle-level acceptance tests should run over an enumerated set of initial triad configurations (all sign patterns of the three ties' functioning balance and all orderings of the three conductance classes) rather than one hand-set reference triad, with per-configuration outcomes reported.

**Worked example.** For Persons 1, 2 and 3, run every combination of each tie being in positive/negative functioning balance and every ordering of the three conductance values — not just one hand-picked triad. The source enumerated all 64 initial signed states. Feasible for triads (tens of configurations × seeds); not for the twelve-person family, which stays on M15 ranges.

**Why it matters / what breaks if you skip it.** A single hand-set triad can miss configuration-dependent behaviour.

**Trace-back:** M11, Phase E · PIMMUR §4.6.2; Kurz Fig. 3.

---

## Notes on coverage (what was already there)

The source file flags each candidate as NEW, PARTLY, or COVERED against the current spec, and lists what the readers judged "already covered" (no change proposed). Those checks are the technical evidence behind "why does this candidate exist." They are deliberately omitted here to keep this file readable; consult the source file (§7 and §9) for the coverage findings and the spec-search hit counts that back each one.
