# Suggested enhancements from the 2026-09-27 reading, in plain language

Plain-language companion to `SWEEP_READING_REPORTS_2026-09-27.md`. The explanation lives here; the requirement wording, evidence and spec references are in that file. None of these is in the spec; each is a proposal for the owner.

The examples use an invented simulated family: grandparents, parents Tom and Ann, and children Sam (15) and Lily (10). They illustrate how a proposal would work. They are not results.

---

## From the opinion-games paper (Jiang et al.)

### J1. When a family looks calm, also show what the calm is costing
**What:** When the model reports a number that looks good when it is low, it also reports how far each person has moved from their own basic level to keep things calm. Examples of such numbers are few arguments or low tension.
**Why:** In the paper, the agents that looked least polarized were the ones that had given up their own positions. Low conflict was a sign of giving in, not of maturity. This lets the model show that readout trap as a number.
**Example:** Tom and Ann rarely argue. Over the same years, Ann's functioning has fallen well below her basic level and Tom's has risen above his, because she keeps the peace by giving way. A second simulated couple argues more, but both stay near their own levels. Counting arguments alone ranks Tom and Ann as healthier. The second number shows that their calm is bought by Ann giving up self.

### J2. Check whether a pattern comes from the family's shape rather than from the process
**What:** Before a pattern is reported as a finding, compare it with shuffled versions of the same family. The shuffled versions keep the fixed structure (who is related to whom, how many relationships each person has) but move the events around at random.
**Why:** In the paper, 44% of a headline effect came from a rule the authors had built in, not from the dynamics they were studying.
**Example:** A run shows cutoffs concentrated on Tom's side of the family. But Tom's side has more members and more relationships, so more cutoffs would land there by chance. Shuffle the cutoffs among people while keeping the counts. If the shuffled families show about the same concentration, the finding reflects the family tree, not emotional process.

### J3. Double-check when a run has "settled down"
**What:** The model uses one rule to decide when a family has reached a steady pattern. Check that point with two other measures, and also report results at fixed times chosen in advance.
**Why:** In the paper, the other measures showed the system had settled well before the rule said so. One invented threshold should not decide where the results are read.
**Example:** The rule says the family settles in year 12. Another measure shows tension between Ann and Lily still rising at year 12. Report the results at year 12, and also at years 15 and 20, which were fixed before the run. If the conclusion changes between those points, say so.

### J4. Test the model against the most obvious rival explanation
**What:** Build a deliberately different version in which people drift toward the average of the people around them, with no give-and-take between them. Confirm that the tests fail on that version.
**Why:** This "drift to the average" model is what a reviewer would propose first. If the tests pass on it too, they are not testing borrowing and lending of self.
**Example:** In EPModel, when Tom's functioning rises, Ann's falls by the same amount. In the rival version both drift toward the middle and nobody loses anything. Take the test "the overfunctioner rises while the underfunctioner falls". If it still passes on the rival version, it is not checking what we think it is checking.

---

## From the SocioVerse2 paper (Zhang et al.)

### SV1. Two versions of a family must match exactly until the point where they differ
**What:** When two runs differ only by something that starts at a given time, their records must be identical up to that time.
**Why:** This catches two hidden faults. One is the model "peeking" at scheduled future events. The other is random numbers falling out of step between the two runs.
**Example:** In run A a coach enters in year 10. Run B has no coach. Years 0 to 9 must be identical in both. If they differ in year 3, something is wrong: for instance, Ann is reacting to a coach who has not arrived yet, or the random draws have shifted.

### SV2. Enter an intervention as a scheduled event, not a hand edit
**What:** An intervention is written as "at this time, this happens". It enters the model as an ordinary event, goes through the normal rules and is logged. Each comparison is also labelled as one of two kinds:
- a **branch**: the same family with the same history up to the change;
- a **version**: a different starting family or different rules.

Only a branch can be read as the effect of an intervention on that family.
**Why:** The current spec has no way to change something partway through a run. Editing a number directly bypasses the rules the model is testing.
**Example:** "In year 10, Tom begins coaching" is entered as an event, and the model works out how Tom and the family respond. Writing "set Tom's anxiety to 0.3 in year 10" is not allowed. Comparing Tom's family with and without coaching from year 10 is a branch. Comparing Tom's family with a different family is a version.
**Question for the owner:** A coach who arrives in year 10 is a new person in the family's world. Should the coach exist from the start with no active relationship, or does adding a person make the run a version rather than a branch?

### SV3. When importing a real family diagram, keep what happened later apart from what sets up the start
**What:** Each dated item in an imported family diagram is marked as either a starting input or a later outcome. Later outcomes are held back and used only to compare against. Anything the model is supposed to produce itself must never be fed in as a scheduled event, for example a cutoff, a divorce, a symptom or the identified patient. Each rating also carries two dates: when it was made, and the time it describes.
**Why:** If later events are fed in, or hindsight shapes the starting point, the model is being handed the answer. That makes the run a fit, not a comparison. The current import contract does not say how dated events after the start should be treated.
**Example:** The diagram shows the parents divorcing in 1995 and a son cutting off in 2001, and the run starts in 1990. The divorce and the cutoff are what the model is meant to produce, so they are held out and compared with the runs afterwards. They are not scheduled. A rating made in 2020 of the mother's functioning in 1990 is made with hindsight. It is flagged and given a wider range than a rating made at the time.

### SV4. Report how many attempts came before the result
**What:** Record which version of the study a result came from, and how many earlier versions were tried. If a version was chosen after looking at the results, mark the result as chosen after the fact.
**Why:** In the paper, each result was the version the researchers accepted, and the attempts before it were not reported. This hides how much trial and error went into a result.
**Example:** "The coaching effect shown is from the fourth version of the study. Earlier versions changed the definition of tension twice and the sibling rules once."

---

## From the partnership-dynamics paper (Nair-Turkich et al.)

### PD1. Keep a record of each relationship episode, including ones still going at the end
**What:** Log each marriage, cutoff and period of distance with its start, its end, how it ended, and a flag if it was still going when the run finished. Reports on how long things last must count the unfinished ones properly.
**Why:** Averaging only the episodes that ended gives a false answer. The paper made this mistake.
**Example:** Across 100 runs, the marriages that ended lasted 8 years on average. In 60 of the runs the marriage was still intact after 40 years. "Marriages last 8 years" is wrong. The correct statement includes the 60 marriages that had not ended.

### PD2. State how the chance of something ending changes over time
**What:** For every duration in the model, declare its assumed shape. The chance of ending may stay the same, rise, or fall as time passes. Label the shape as invented, and rerun the tests with at least one other shape.
**Why:** Choosing a shape is an assumption about how relationships work. Trying different numbers within the same shape does not test that assumption.
**Example:** Is Sam's cutoff from Tom as likely to end in its first year as in its twentieth (the same chance throughout)? Or does it get harder to end the longer it lasts (hardening)? These are two different models of cutoff. Declare which one is used, and check whether the conclusions hold under the other.

### PD3. Don't make everyone's year turn over in the same week
**What:** At present, the yearly updates happen to all twelve people in the same week: ageing, life stage, mortality, chronic anxiety. Either give each person their own point in the year (for example their birthday), or record the simultaneous version as an assumption and test it.
**Why:** Updating everyone at once can create an artificial yearly jolt that looks like a pattern in the family.
**Example:** Every 52 weeks the grandmother moves into late life, Sam into adolescence, and everyone's chronic anxiety is recalculated, all in the same week. A spike in family tension every year at that week would be an artefact of the schedule, not emotional process.

### PD4. Explore the invented numbers evenly, and never pick the best one
**What:** Each invented constant has a plausible range. Draw a few hundred combinations spread evenly across all the ranges together, and run both arms with several random seeds for each combination. Report the share of combinations in which the direction holds. Never pick the combination that best matches a known family. Also state how the values were spread: evenly, or on a scale that treats 0.5 to 1 and 1 to 2 as the same size step.
**Why:** The paper picked its single best-fitting combination, and the "best" values swung by up to three times between runs. That is the fitting EPModel forbids. How the values are spread changes what "most of the range" means.
**Example:** With ten invented constants, draw 500 combinations. The report reads: "Coaching reduced cutoffs in 83% of combinations." If one constant runs from 0.5 to 6 and is spread evenly, about 90% of its draws are above 1. That should be stated, not hidden.

---

## From the persona-drift paper (Leins et al.)

These apply only if the optional narrator is built: a language model that turns the engine's results into readable descriptions. The narrator never makes decisions.

### X5. Give the narrator the full picture every time
**What:** Each time the narrator writes about a person, it receives that person's full current profile from the engine. It does not rely on remembering earlier passages.
**Why:** The paper showed that language models steadily forget a persona over a conversation. Resending the full description worked better than a short reminder.
**Example:** Each time the narrator writes a line for Sam, it is told Sam's age, his current anxiety, what the engine chose for him this week (for example, a fight with Ann), and his relationships. It is not left to "remember" that Sam is a reactive 15-year-old.

### X6. Measure how much the narrator drifts on its own
**What:** Freeze a person's state in the engine and have the narrator write many turns. Any change in how the person comes across is the narrator's drift. A narrated change smaller than that amount cannot be trusted.
**Why:** In the paper, a highly reactive persona drifted toward calm within about eight turns. No prompting technique stopped it. A narrated teenager could appear to mature when nothing in the model changed.
**Example:** Hold Sam at high reactivity and have the narrator write 30 turns. If Sam sounds noticeably calmer by turn 20, that is drift. Repeat this for shy Lily, whose drift may go the other way (becoming more talkative) or not appear at all.

### X7. Keep the judge separate from the writer
**What:** Whatever scores the narrator's output sees only the text. It does not see the engine's numbers or the narrator's instructions, and it is a different model from the narrator. If a correction step uses a checklist, the final score uses a different one. Language-model judges agreeing with each other is not proof that they are right.
**Why:** In the paper, the same checklist guided the corrections and scored the results, so the corrections learned to satisfy the checklist. The same models also served as their own judges.
**Example:** The judge rating whether Sam sounds like a reactive teenager reads only the passage. It does not know what the engine said about him, and it is not the model that wrote the passage.

### X8. Measure the narrator against the engine, not against its own first passages
**What:** The narrator is checked against what the engine says about the person at that moment. It is never checked against how it described the person at the start.
**Why:** In the paper, drift was measured against each conversation's own opening, so an inaccurate start was treated as the correct target.
**Example:** If Sam genuinely calms down in the engine by year 5, the narrator should show that. Measured against its early passages, the narrator would be "corrected" back into making Sam reactive.

### X9. If the narrator is corrected on the fly, compare it with a simple fixed schedule
**What:** Compare any "correct it when drift is detected" system with simply resending the profile on a fixed schedule, using the same number of corrections. Log how long it has been since the last correction, so that the resulting ups and downs are not read as mood swings.
**Why:** In the paper, the "smart" timing was no better than a fixed schedule. Each correction was followed by a gradual fade, which produced a sawtooth pattern.
**Example:** "Resend Sam's profile every 4 turns" is the baseline. A system that corrects only when Sam starts sounding too calm must beat that baseline with the same number of corrections. This is low priority if X5 is adopted.
