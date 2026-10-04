# Suggested enhancements from the 25-paper social-simulation digest, in plain language

This is the plain-language companion to `SWEEP_READING_REPORTS_2026-10-04b.md`, which has the requirement wording, the evidence and the spec references. Of the 25 papers, 21 were read in full. The other four had already been read in earlier batches. Seven of the 21 gave no proposal: AI mediators, personality-tuned LLMs, digital twins, empathetic RL (CARE), MEM, and, for the v2 spec only, illusory truth and AGIMUD's narrator item.

None of these proposals is in the spec. Each is for the owner to decide, held for revision 11.

The examples use the same invented family as the earlier batches: the grandparents, parents Tom and Ann, and children Sam (15) and Lily (10).

---

## 1. Gaps in the spec text (need an owner decision first)

### RA1. How does a person choose *whom* a move is aimed at? *(from ReAdapt)*
**What it proposes:** The spec says how a move is picked: pursue, distance, conflict, triangle and so on. It does not say how its **target** is picked: whom Ann pursues, or which third person a triangle pulls in. The proposal is to declare this. The target choice could only read the actor's own ties, old triangle circuits, beliefs and delivered events.
**Owner question:** When anxiety rises, does targeting shift from "the tie I have" toward "whoever just upset me"? Does that depend on differentiation?
**Why:** This choice decides who becomes the projection focus and who becomes the outsider. Left unstated, it becomes an unlabelled invented choice with large effects.
**Example:** Tom comes home late. Ann's anxiety rises. Does she conflict with Tom, who caused it? Pursue Tom, which is her usual pattern? Or pull in Sam, the old triangle circuit? At present the spec does not say.

### PM1. How long does a belief last? *(from PersMem)*
**What it proposes:** The spec says how a belief is *written* (by delivered events). It does not say what happens to it afterwards. It might stay until overwritten, fade back toward a default, or change only when contradicted. Declare one. Whether an anxious person holds threatening beliefs longer is an owner decision.
**Why:** Over 40 years this matters a great deal, and two implementations could differ here and both pass every test. Caution: the corpus's "the family system operates to obscure and misremember" (L09.4) may point the *opposite* way from the memory literature.
**Example:** Lily once overheard Tom say he might leave. Twenty years later, does she still believe it, has it faded, or has it been replaced by what she has seen since?

### AN1. What happens to a move that is already sent but not yet received? *(from AnthroDial)*
**What it proposes:** Moves take time to arrive (per-edge delay). If, before arrival, the sender dies, the tie is cut off, or the receiver leaves the household, the engine needs a declared rule: deliver anyway, deliver with updated witnesses, or cancel. A cancelled move is logged and counted. If it carried anxiety, that anxiety must be accounted for under conservation.
**Why:** The spec is silent on this, and it touches the conservation invariant around deaths.
**Example:** Grandfather sends a critical message to Tom, then dies before it reaches him. Does Tom receive it?

### RZ1 and RZ2. Are beliefs separate items, or linked? Does a belief resist being contradicted? *(from AI Agents are Vulnerable to Radicalization)*
**What they propose:**
- **RZ1:** State whether a write to one belief moves a related one. For example, does "Sam is the problem" shift "whose fault the marriage trouble is"? Then test whichever answer is chosen, giving each belief its own control.
- **RZ2:** Should a message that confirms what someone already believes write more strongly than one that contradicts it? The spec already says this for a coach's account (M1.E.7f). The question is whether it applies to all belief writes. If it does, test it with four arms, and check that the arms do not already differ before any message is sent. The paper's own design failed that check.

**Why:** The spec says nothing on either point. The paper is not evidence about people, only a source of test designs.
**Example:** Ann believes Sam is the problem. Tom tells her "Sam's fine, it's us". Does that change her belief as easily as Tom saying "Sam really is out of control"?

### EN (owner question only). Should randomness be memoryless? *(from Emotions as coloured noise)*
The reader found the paper's "coloured noise" means noise with a non-zero mean, not noise that is correlated over time, so the paper does not bear on this. The reader did check the spec: every random draw is independent from one tick to the next, and all persistence comes from named state. **Question:** should the spec say outright that no hidden "mood noise" may carry over between weeks? Such noise would look like chronic anxiety without being it.

---

## 2. Tests that check the model is doing what we think

### PF1. A table of which broken mechanism breaks which test *(from patch-foraging identifiability)*
**What it proposes:** The mutation tests already break each mechanism and check that its test fails. Record all the results as one table: broken mechanisms down the side, tests across the top. Flag two cases. One is two broken mechanisms that fail exactly the same tests, because the suite cannot tell them apart. The other is a test that fails when a mechanism it was not about is broken.
**Why:** The paper found that different mechanisms produce similar group outcomes. Each test proving it *can* fail does not show the suite as a whole can tell mechanisms apart. The table needs no extra runs.

### RP2. Each test names the behaviour the mechanism works through, and reports it *(from RePair)*
**What it proposes:** Each acceptance test names, in advance, the moves its mechanism should change. One example is triangling moves on the named triangle. Report that behaviour's change beside the outcome's change, and sort the result into one of three cases:
- both moved;
- the outcome moved but the behaviour did not;
- the behaviour moved but the outcome did not.

Only the first counts as support for the mechanism.
**Why:** The paper found all three cases. In one, a "contact" rule lowered contact and still changed the outcome. Choosing the behaviour for each test is a theory decision.
**Example:** Coaching lowers Sam's symptoms. If Ann's triangling moves toward Sam did not fall, coaching did not work through detriangling, whatever the outcome says.

### RP1. A person cannot change before news of the change could reach them *(from RePair)*
**What it proposes:** Paired runs share random draws, so the two arms are identical until something actually differs. For each person, find the first week their state differs between arms, and the earliest week any message from the change could have reached them. If the state differs first, it must come through a named family-wide channel, such as societal anxiety or the shared anxiety budget. Otherwise it is flagged as a leak.
**Why:** The paper found an effect present before the channel it was credited to could operate.
**Example:** Coaching starts with Ann in week 100. If Grandmother's anxiety differs between arms in week 100, before anyone could have told her, something is leaking.

### NG2. On day one, the two arms differ only in what was declared *(from neutrality gap)*
**What it proposes:** Before the first tick, compare the two arms field by field. Anything that differs must be the declared change, something the spec derives from it, or something drawn to match it under NG1. Anything else fails the run.
**Why:** In the paper, changing one persona attribute silently changed others. The spec currently checks this only by reading the code and by the null arm.
**Example:** An arm raises Ann's basic level. If Sam's starting anxiety also changed and nobody declared that, the run fails.

### NG1. When one person's level is changed, say whether others are redrawn *(from neutrality gap)*
**What it proposes:** An arm that changes one person's starting attribute declares one of two modes:
- **assign:** everyone else is left exactly as they were;
- **condition:** the others are redrawn as the theory's correlations imply. Spouses at similar levels is one such correlation.

**Owner decision:** spouses marry at nearly equal levels (M2.A.0c). An arm that raises only Ann's level describes a couple the theory says does not form. Should that be forbidden, allowed and flagged, or always run with both spouses moving together?

### NG3. Count how many runs break the "should not change" check, not just the average *(from neutrality gap)*
**What it proposes:** When a test names something that should *not* change (TM5), report the share of runs where it moved beyond the margin, not only the average.
**Why:** In the paper the average looked fine while 39–76% of cases failed.

### AG1. Reordering the list of moves must change nothing *(from AGIMUD)*
**What it proposes:** Shuffle the order in which the moves are listed in config. Every run must come out identical once the labels are mapped back. A broken version that breaks ties by "first in the list" must fail.
**Why:** In the paper, list order decided every tied choice in 50 of 50 runs.

### RA2. A test case where the most recent upset and the closest tie point to different people *(from ReAdapt; depends on RA1)*
**What it proposes:** Targeting tests include cases where the loudest recent event comes from a different person than the one the tie state favours. Results are reported for those cases separately. The harness is first checked with two scripted policies, one that follows ties and one that follows salience.
**Why:** Only cases where the two cues disagree show which one the model follows.

### PM2. Two different "off switches" for the anxiety-shapes-beliefs rule *(from PersMem; for the held MM1 answer)*
**What it proposes:** Test the owner's answer to MM1 against two broken versions:
- **(a)** beliefs ignore the receiver's state;
- **(b)** beliefs read the receiver's state with a fixed weight that does not rise with anxiety.

The first part of the answer must fail under (a). The second part, "more anxious means more weight", must fail under (b).
**Why:** Without (b), a version with a fixed weight passes while missing half the rule.

### EV4. Does anxiety colour second-hand news more than first-hand? *(from evacuation)*
**What it proposes:** Extend the held MM1 test across three ways a person can learn of an event:
- told directly;
- witnessed;
- heard second-hand, possibly garbled.

Report the anxiety effect for each.
**Owner question:** Should anxiety weigh more on indirect evidence? FE05.10 ("what might be") may bear on it.
**Example:** Ann hears from her sister that Tom was seen at a bar. Does her anxiety distort that more than seeing it herself?

### TP1. "Sudden" in time, or "sudden" across a setting? *(from tipping points; fold into held BB5)*
**What it proposes:** Any claim of a tipping point says which kind it is:
- abrupt over weeks with nothing else changed;
- abrupt as a setting is turned up.

Each needs different evidence.
**Why:** The paper defined tipping one way and measured it the other. Its biggest "jump" was the start-up period.

### EN1. An oscillation must survive a finer time step *(from Emotions as coloured noise)*
**What it proposes:** If a run is classed as oscillating, re-run it with the weekly tick split into smaller steps. If the oscillation goes away, it was caused by the step size, not by the theory.
**Why:** The paper's own model oscillates and goes chaotic purely because of how the discrete steps overshoot. The spec requires a real damped oscillation in change-back (M5.E.6), and that must not be confused with this artefact.

---

## 3. How ensembles of runs are set up and checked (Phase E)

### AR1. Deliberately search for families where a result reverses *(from AdvRole)*
**What it proposes:** As well as sampling starting families and constants at random, run a directed search, within the declared ranges, for settings where an acceptance result flips direction. A flip counts only if it holds on fresh seeds. Report flips beside the random-sample result and keep them as regression tests. The declared range must not then be narrowed to hide them.
**Why:** Random sampling can miss a small region where the result reverses.
**Owner flag:** held PD4 says "never select" samples against a target. This does select, in order to falsify. The wording of the two needs reconciling.

### AQ2. Can the model reach the corpus's stated bounds at all? *(from adequacy-aware calibration)*
**What it proposes:** For each bound the corpus states (M10.C.4) that runs miss, report whether *any* setting in the sweep reaches it, and whether any one setting reaches all the bounds together. If none does, the mechanism cannot produce what the corpus states, and changing the invented constants will not fix it. The settings that do reach a bound are never used to pick defaults.
**Owner decision:** this reads the bounds against the constants, which sits next to an earlier open question about excluding regions.

### AQ1. After a fix, check the tests the fix was *not* aimed at *(from adequacy-aware calibration)*
**What it proposes:** When a rule or constant is changed because a test failed, declare beforehand a set of other tests that did not motivate the change. Report how they did before and after. Ideally, also compare against a change made somewhere else.
**Why:** In the paper, only the held-out check caught a failure the fix had missed.

### PF2. Flag a constant that does nothing in a given setting *(from patch-foraging identifiability)*
**What it proposes:** When a constant is swept and nothing moves, mark it "inert in this setting". Do not report that as the result being robust to that constant.
**Why:** If a mechanism is dormant, a sweep of its constant looks perfectly stable, and that stability means nothing.
**Example:** The coach-rejection constant cannot matter in a family that never hears of coaching.

### PF3. Constants that only matter as a product should be swept as one *(from patch-foraging identifiability)*
**What it proposes:** List groups of constants that only ever appear multiplied or divided together. One example is event intensity × conductance. Fix one member by convention and sweep the combined value.
**Why:** Sweeping both separately samples the same thing twice, under an undeclared weighting.

### EV3. Which route does basic level actually work through? *(from evacuation)*
**What it proposes:** Basic level feeds six derived quantities. For a test that varies basic level, run one arm per route, in which only that route sees the change. Report which routes carry the result.
**Why:** The spec itself warns (M4.A.5) that one test could pass "for the wrong reason" through a side route.

### EV2. Refuse to run a test that cannot reach significance *(from evacuation)*
**What it proposes:** Before running, compute the smallest p-value the test could ever produce at the planned number of seeds, after correction. If that is above the threshold, do not run the test.
**Why:** The paper's best result, 8 out of 8, sat exactly at its design's floor of 0.078. No result could have passed.

### AQ3. Random draws in the analysis are seeded too *(from adequacy-aware calibration)*
**What it proposes:** Permutation nulls, resampling and sweep designs all use a declared analysis seed that is logged.
**Why:** The paper had to withdraw published numbers because one null distribution was not seeded.

---

## 4. How results are reported

### AT2. Every result is stated as "in this model, under these settings…" *(from anthropomorphism risks)*
**What it proposes:** Results name the runs, arms, seed count and claim grade. They are never stated as facts about families in general, and never as decisions by the model.
**Why:** M11.F.9 forbids claims about a *particular* real family. Nothing forbids "coaching lowers symptoms" as a general claim.
**Example:** Not "coaching helps the youngest child". Instead: "in this model, with these settings, the coaching arm had lower symptom load for P10 than the no-coaching arm in 412 of 500 seeds."

### AT1. Say what "anxiety" means each time it is reported *(from anthropomorphism risks)*
**What it proposes:** Any output that names a model quantity (anxiety, functional level, bond energy, belief) carries a short statement that it is a model quantity standing for a theory concept. It is not a measurement of a person or a feeling. The deterministic renderer may not turn numbers into feeling words ("Ann feels hurt"). The banned-word list lives in config.
**Owner decision:** what the statement says, especially whether model "anxiety" is felt, is a theory question.

### AT3. Show people by ID by default, not by name *(from anthropomorphism risks)*
**What it proposes:** Rendered output shows "P09 · G3" by default, with given names as an opt-in. Role words like "mother" are not used as labels, because the spec forbids reading position from role.
**Why:** Human names make readers treat the output as being about real people. The evidence for this is weak, and you may prefer names for readability.

### PP1. Measure how varied a person's moves are against what chance gives for that many moves *(from Population Physics)*
**What it proposes:** The repertoire-variety measure (M16.A.9) always reports how many moves it is based on, and a chance baseline for that count. M11.C.40 compares the gap above chance, not the raw number.
**Why:** The reader computed that with few moves the measure reads low even when behaviour has not changed: 0.56 at 5 moves against 0.90 at 260, for identical behaviour. Self-directed moves are rarer than automatic ones, so this could fake the M11.C.40 result.

### AG2. A number that never changes gives "undefined", not "zero" *(from AGIMUD)*
**What it proposes:** Flag any quantity that never varied in a set of runs. Any correlation or regression involving it is reported as undefined.
**Why:** The paper reported r = 0.00 for quantities that were constant and read it as "no effect".

### EV1. Track how news spreads through the family *(from evacuation)*
**What it proposes:** For news that starts with one person, such as a death or the held SV2 coaching-knowledge event, report who it reached, in how many steps, how many people passed it on, and who never heard.
**Why:** Your SV2 answer says knowledge spreads mostly to the closest and safest, and a partner is almost always told. Nothing currently measures whether it does.
**Example:** Ann learns about coaching. Tom hears in week 2. Her sister hears in week 6, through Ann. Tom's parents never hear.

### PP2. A curve fit stuck at its limit is "inconclusive" *(from Population Physics; low priority)*
Only relevant if regime classification is done by fitting curves.

---

## 5. The optional narrator (only if it is built)

- **NG-X1.** Check that changing one thing in the engine state changes only that thing in the narration. The first pair to test: higher anxiety must not make a person *sound* less differentiated. The paper found that telling an LLM an attribute is "neutral" does not stop it from leaking.
- **AT-X1.** The narrator writes in the third person by default. This questions DESIGN_LESSONS §7.6, which adopted first-person narration.
- **AN-X1.** A mechanical check, run before any other narrator test, throws out narration that contradicts the log. Examples: a dead person speaking, or an event described as happening now when it is in the past.
- **IT-X1.** Each narrator call starts fresh, with no earlier narration in its context.
- **PM-X1, RP-X1, AQ-X1, TP-X1.** Minor. See the reports.

---

## Questions for the owner arising from this batch

1. **RA1.** How is a move's target chosen? Under anxiety, does it shift toward whoever just caused the upset?
2. **PM1.** How long does a belief last? Does anxiety make threatening beliefs last longer, or does L09.4 point the other way?
3. **AN1.** Does a message sent before a death or cutoff still arrive?
4. **RZ1.** Are beliefs independent, or does changing one move a related one?
5. **RZ2.** Does a confirming message write more strongly than a contradicting one, for all beliefs or only for a coach's account?
6. **NG1.** May an arm separate two spouses' levels?
7. **EV4.** Does anxiety distort second-hand news more than first-hand?
8. **EN.** Should the spec forbid hidden randomness that carries over from week to week?
9. **AT1, AT2.** Should the consistency-engine framing ("if the theory holds, what follows") become a requirement? Is model "anxiety" felt?
10. **AN (from AnthroDial).** Does a held-back urge carry over to the next week, or is it gone?
11. **AR1 and PD4.** How should "search for counterexamples" and "never select samples" be worded together?
12. **FE05.17** (found while checking RA, not from a paper). The corpus says that protecting children from one's own problems is a main route of transmission, and marks this as testable. No acceptance test covers it. Should one be added?
