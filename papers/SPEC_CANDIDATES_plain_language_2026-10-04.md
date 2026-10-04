# Suggested enhancements from the five 2026-09-28 sweep papers, in plain language

Plain-language companion to `SWEEP_READING_REPORTS_2026-10-04.md`. That file has the requirement wording, the evidence and the spec references. None of these is in the spec. Each is a proposal for the owner, held for revision 11 like the 2026-09-27 batches.

The examples use the same invented family as the earlier batches: the grandparents, parents Tom and Ann, and children Sam (15) and Lily (10). They show how a proposal would work. They are not results.

The suggestions are grouped by what they would change, not by paper. Group 1 is a gap in the spec text. Group 2 is tests. Group 3 is how ensembles are set up. Group 4 is reporting. Group 5 is the optional narrator.

---

## 1. Gaps in the spec text (need an owner decision first)

### BV1. Give "situation", "decision domain" and "task versus relational demand" a definite form *(from Bayesian Personalized Value Alignment)*
**What:** Three requirements make behaviour depend on context:
- M1.A.3: the intellect has less licence over joint decisions.
- M1.A.3a: marriage moves many decisions into the joint domain.
- M1.A.14d: sibling-position behaviour shows on task demands, not relational ones.

A fourth, the estimator for basic level, counts how many "situations" a person functions well across (M1.A.4c, M10.B.3).

The reader could not find anywhere in the spec that says what a situation, a decision domain or a demand type *is*. No event or move carries any such field. The proposal is that each becomes a declared field with its values listed in config, so that these requirements can be built as written.
**Why:** The paper's main result is that a fixed profile does badly once context changes, and does worst when context pulls hardest against the profile. EPModel already separates stable level from current state, so that criticism does not apply. What the paper exposes is that EPModel *names* contexts it never represents.
**Owner decisions needed:**
- What counts as a "situation"?
- Are "joint decision" and "relational demand" the same axis?
- How does a move on a tie carry a decision domain?

The paper cannot answer these, and its taxonomy (Schwartz values) is not Bowen theory.
**Example:** Tom decides alone how to fix the car (a task demand, outside the joint domain). He and Ann decide together whether his mother moves in (a joint decision). At present the engine cannot tell these two apart, so M1.A.3 cannot act on the difference.

### BV3. Say what the mix between "automatic" and "self-directed" reads *(from Bayesian Personalized Value Alignment)*
**What:** M4.D.1a says the weight between the automatic channel (driven by the relationship system) and the self-directed channel (driven by the person) is "a function of differentiation". The spec should state which of these it reads: basic level, current functional level, or anxiety. It should also say whether M1.A.4h's estimator reads that weight or the share of moves actually chosen.
**Why:** If the weight reads functional level, the estimator for basic level will pick up the temporary swing unless load is matched. It would then read a bad month as a lower basic level. This is a theory question. The paper only shows that a context-dependent mix beats a fixed one in its own task.
**Example:** Ann's basic level is fixed, but in a stressful month her functional level drops. Does her automatic share rise that month because the mix changed, or only because anxiety raised the weight of reactive moves inside the automatic channel? The two readings give different estimates of her basic level.

---

## 2. Tests that check the model is doing what we think

### BV2. Same person, different context: the behaviour should differ *(from Bayesian Personalized Value Alignment; depends on BV1)*
**What:** Two tests that keep one person exactly the same and change only the context.
- (a) The share of self-directed moves should be lower on joint decisions than on other decisions.
- (b) Under raised anxiety, sibling-position behaviour should show more on a task than in a relationship.

Each test is checked against a broken version in which the person always behaves at their average across contexts. That broken version must make the test fail.
**Why:** Every existing comparison varies the person or the family. None holds the person fixed and varies only the situation, so a policy that ignored context could pass all of them.
**Example:** Same Tom, same seed. When he decides alone about the car he is more self-directed. When he and Ann decide about his mother he is less so. If Tom's behaviour is identical in both, the test fails.

### WA1. Do family members actually behave like different people? *(from Warned alike)*
**What:** For each person, measure their habitual tendency, for example their share of self-directed moves in a settled period. Then check two things:
- whether people in the family differ from one another more than identical people would by chance;
- whether a person's tendency in the first half of the period predicts it in the second half.

A test should show both are higher in the normal family than in the arm where everyone is given identical parameters.
**Why:** In the paper, 50 AI agents with a "personality" trait spread out no more than identical coin-flippers would. Their individual tendencies barely carried over from one half of a session to the other. Humans were about three times as spread out and highly consistent. A rule-based model can have the same problem: parameters can differ without producing behaviour that differs.
**Example:** Over five settled years, Lily's self-directed share is consistently higher than Sam's, in both halves. If the twelve people are no more different from each other than twelve copies of an average person would be, the parameters are not doing their job.

### WA2. Does the whole family react in lockstep to the same event? *(from Warned alike)*
**What:** When something reaches several people on the same tick (a death, a job loss, a rise in societal anxiety), record each person's first reaction and how long it took. Report how concentrated those first reactions are and how spread out their timing is, and compare the normal family with the identical-parameters family.
**Why:** In the paper even a simple rule-based learner with 20% parameter variation started moving in lockstep once everyone received one shared signal. EPModel spreads ordinary moves out through delays and witnesses, but some inputs reach everyone at once. Nothing in the spec would currently detect a whole family moving as one unit.
**Owner question:** Does the theory say more fused families react *more alike* to the same event? If so, the test should check that alignment rises as differentiation falls, not just that the normal family beats the identical-parameters one.
**Example:** Grandfather dies. If all six adults withdraw on the same tick, that is lockstep. A realistic run might have Ann pursue Tom, Tom distance, Sam act up a few weeks later, and the grandmother over-function.

### CD3. Show that each measurement actually moves when the thing it measures moves *(from Cultural Divergence Preservation)*
**What:** For every measure an acceptance test relies on, a fixed test feeds it artificial data in which the property is turned down and then turned up by set amounts. The measure must move in the right direction both times.
**Why:** The paper checked its own metric this way before using it. Some measures barely register small real changes, and others are already at their limit. Either kind can make a two-arm test look like "no difference" when there is one.
**Example:** Take a log where projection lands on Sam 60% of the time and make versions at 40% and 80%. The concentration measure must read lower for 40% and higher for 80%.

---

## 3. How ensembles of runs are set up (Phase E)

### PW1. Check which side of each turning point the starting families fall on *(from PERSONAWEAVER)*
**What:** For each ensemble, report what share of the starting families sit on each side of each named turning point (M15.D.4: marital distance, stable or unstable twosome, the outsider threshold, severity, and peace-agree versus reactive). Do this at the start and again after the settling period. If most families sit on one side, every result is reported as holding only for that side. A wide spread of parameter values does not count as coverage.
**Why:** The paper found that varied character descriptions still produced the same behaviour. Spread in the inputs did not mean spread in what happened. The spec already says errors in starting conditions can flip a result across a turning point, so the coverage that matters is coverage of those turning points.
**Example:** 500 runs with Tom and Ann's starting levels drawn widely. If 480 of them start in the peace-agree zone, the ensemble says almost nothing about reactive marriages, however wide the ranges looked.

### PW2. Vary the shape of the family, not just the numbers *(from PERSONAWEAVER)*
**What:** Draw family structure as its own factor from a declared list of shapes: how many children, which side has a cutoff, which spouse's family is distant, whether children have left home. Report results per shape and pooled. Each shape is either an anonymised imported topology or labelled invented.
**Why:** In the paper, asking a generator to "be diverse" still gave conventional characters. Drawing the parts separately gave less conventional ones. A person hand-building families has the same pull toward a default shape. The spec already notes that hand-built families come out more balanced than real ones.
**Example:** As well as the reference family, run one where Tom is cut off from his parents, and one where Ann has three siblings instead of one.

### PW3. Only rule out impossible starting families, and count what was repaired *(from PERSONAWEAVER)*
**What:** List the combinations a starting family may not have, each tied to a requirement (for example a spouse age gap outside the tolerance). Rare combinations are not excluded. A draw that breaks a rule is fixed by a declared fixed transform, not redrawn, and the log counts how many were fixed.
**Why:** The paper's repair step was allowed to fix impossible cards but not unlikely ones. Without that limit, a validity filter quietly narrows the ensemble toward the conventional family. Redrawing until valid also breaks the pairing of draws between arms.
**Example:** A drawn family with a very low-functioning grandmother and a very high-functioning grandfather is unusual, not impossible. It stays in.

### WA4. Keep development seeds and test seeds separate *(from Warned alike)*
**What:** Declare the seeds used while writing code and choosing invented constants. They must not overlap with the seeds used for acceptance tests and ensembles. Record the acceptance seed range before the suite first runs.
**Why:** The paper did its development on a separate seed and fixed its test seeds in a hashed file in advance. The spec freezes constants before testing (M10.B.4), but they could still have been tuned while watching the same seeds later used to test them.
**Example:** Seeds 1–200 are used while tuning. Acceptance uses 10,000 onward, written into the run header before the first acceptance run.

### WA5. When an intervention reaches some members, try every number of members *(from Warned alike)*
**What:** Where arms differ in who receives an event (for example, the coaching-knowledge event in the owner's answer to SV2), run several recipient counts, not only "nobody" and "everyone". Report the curve and flag a curve that goes down and then up again.
**Why:** In the paper, warning a small share of drivers helped, and warning more made things worse again. The middle cases were not the average of the extremes.
**Example:** Coaching knowledge reaches only Ann, then Ann and Tom, then Ann, Tom and his mother. The benefit might be largest when one person knows and smaller when three do.

### CB3. Don't compare invented grandparents with simulated grandchildren without checking *(from Consequential Behaviour)*
**What:** M11.C.6(b) checks that differences in basic level widen across generations. The founders' values are supplied, not simulated (M2.A.0a). So any comparison between the founders and later generations must be repeated over a range of how spread out the founders were set to be. If the result depends on that setting, it is reported as conditional.
**Why:** If the founders were given a narrow spread, the later generations will "widen" partly because of that choice, not only because of the mechanism.
**Example:** Give the four grandparents nearly equal levels and the grandchildren will look more spread out almost automatically. Give the grandparents a wide spread and the widening may disappear.

---

## 4. How results are reported

### CB1. Say whether a result is about what people believe, how they are inside, or what they did *(from Consequential Behaviour)*
**What:** Each readout and each acceptance test states which layer it reads:
- belief: what a person thinks is true;
- inner state: anxiety, bond energy, functional level;
- action: moves actually made, contact counts, held-back moves.

A result on one layer is never reported as a result on another. Where a readout uses inner state, the matching action count is shown beside it. The test fixture is the reference family's Iris and Bruno: same contact frequency, different bond energy. On the action layer the two should look the same; on the inner-state layer they should not.
**Why:** The paper's point is that what people say and what they do are different things, and validating one does not validate the other. In EPModel the cutoff readout is computed from bond energy, which no real observer can see. Nothing stops a report presenting it as visible cutoff.
**Example:** "Sam is cut off from his grandfather" might mean low bond energy (inner state) or almost no contact (action). The spec says these can differ, so the report should say which one it means.

### CB2. When a result holds for the whole family, show it for each position too *(from Consequential Behaviour)*
**What:** For every test asserted on a family total, also report the difference between arms for each position in the family:
- generation;
- spouse or child;
- projection focus or sibling;
- inside or outside of the active triangle;
- over- or under-functioning;
- married-in or born in.

Show the weakest position beside the total, flag any position where the direction reverses, and list positions with too few cases instead of dropping them.
**Why:** The paper notes that an average can be right while every subgroup is wrong, because errors cancel. In EPModel anxiety is conserved and moved around, so a family total can stay flat while one person gets better and another gets worse. This is not about "fairness": the projection focus carrying more is theory, not a defect.
**Example:** Coaching lowers family anxiety overall. Broken down, Tom and Ann improve and Lily gets worse because the triangle shifts onto her. The total hides that.

### CD1. When comparing two arms' move patterns, show how different two runs of the same arm are *(from Cultural Divergence Preservation)*
**What:** Where a report gives a divergence score between the move patterns of two arms (M17.B.5), it must also give the same score between two runs of the *same* arm. A directional test must not use the divergence score itself. It must state which move shares went up or down.
**Why:** With small numbers of moves, chance alone produces a sizeable divergence. The reader worked out that over five years for one person, chance alone averages about three times the divergence of a real 5-point shift in move shares. The score also cannot say which way things moved.
**Example:** Arm B shows a divergence of 0.02 from arm A for Tom's moves. Two runs of arm A also differ by 0.02. So there is no evidence of a real difference.

### CD2. Measure differences between family members within each run before averaging runs *(from Cultural Divergence Preservation)*
**What:** Measures of how unevenly something falls across family members (for example, how concentrated symptoms are) are calculated inside each run and then summarised across runs. They are never calculated from per-person averages taken across runs. When arms are compared, both use the same set of people: anyone dead or not yet born in either arm is left out of both, and the list of who was left out is reported.
**Why:** The paper showed that blending groups toward their average removes differences faster than you would expect. Averaging runs where projection lands on different children does the same thing.
**Example:** In half the runs projection lands fully on Sam, in the other half fully on Lily. Averaged by name, Sam and Lily each look "half affected", and the family looks even. Measured within each run, every family is highly concentrated.

### CD4. Report results at neighbouring cut-off values *(from Cultural Divergence Preservation)*
**What:** Every invented threshold used in analysis must be frozen before the run. That includes the margins, the regime boundaries, the drift threshold and the seed-stopping rule. Each result is then reported at that value and at declared values either side.
**Why:** The spec already does this for engine thresholds (M17.E.3) but not for thresholds used only in analysis. Re-running the analysis needs no new simulation runs.
**Example:** "Coaching moves the family out of the chronic-conflict regime in 70% of runs" might be 55% or 85% if the regime boundary moves a little. The report shows all three.

### WA3. Report the average and the ups and downs separately *(from Warned alike)*
**What:** For each family-level measure in a test, report both the average over the window and how much it swings, and the difference between arms on each. Flag arms where one goes up and the other goes down.
**Why:** In the paper, a warning made the average worse and the swings smaller, so the overall cost barely moved and hid two opposite effects. The theory describes change-back as an oscillation, so an intervention could lower average tension while making it swing more.
**Example:** After coaching, Tom and Ann's average tension falls, but it now rises and falls sharply around holidays. Reporting only the average misses that.

---

## 5. The optional narrator (only if it is built)

### PW-X1. Check the narrator follows each type of move, not just most of them *(from PERSONAWEAVER)*
**What:** Before using a narrator model, build a table of the move the engine made against the move a reader recovers from the narration, for each move type and each model. A model that falls short on any one move type is barred from narrating that type, or flagged.
**Why:** In the paper, one model followed most behaviour types but followed "deflection" only 31% of the time. An overall average would have hidden that.
**Example:** A narrator renders Tom's CUTOFF as "he took some space". The table would show that CUTOFF is softened.

### CD-X1. The narrator should keep people as different as the engine has them, no flatter and no more extreme *(from Cultural Divergence Preservation)*
**What:** Compare how different the people are in the narration with how different they are in the engine state for the same log. Too similar fails, and too exaggerated also fails. Getting each person roughly right is not enough on its own.
**Why:** In the paper, the richest persona prompts gave the best per-person accuracy and the worst loss of differences between groups. The thinnest prompts gave caricatures.
**Example:** The engine has Sam much more reactive than Lily. A narration where both sound mildly upset fails. So does one where Sam is a cartoon and Lily a saint.

### WA-X1. LLM personas: check they behave differently from each other before reading any result *(from Warned alike)*
**What:** Any test of LLM personas (for example "a 15-year-old brat") reports how spread out the personas' behaviour is compared with identical choosers, and whether each persona is consistent with itself. Every sentence shared by all the prompts is treated as part of the experiment.
**Why:** In the paper, a single shared sentence in the prompt froze 10 of 10 runs onto the worse road.

---

## Questions for the owner arising from this batch

1. **WA2.** Does the corpus say members of a more fused family react more alike to the same event?
2. **WA2/PD3.** Should family-wide inputs (societal anxiety, the start of an outside stress) reach each person on a staggered tick? Or is applying them to everyone at once a deliberate choice to be tested?
3. **BV1.** What is a "situation", a "decision domain" and a "demand type" in a model whose units are moves on ties?
4. **BV3.** Does anxiety shift the automatic/self-directed mix itself, or only the reactive moves inside the automatic channel?
