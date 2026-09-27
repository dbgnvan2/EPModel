# Suggested enhancements from the second 2026-09-27 batch, in plain language

Plain-language companion to `SWEEP_READING_REPORTS_2026-09-27b.md`. The requirement wording, evidence and spec references are in that file. None of these is in the spec; each is a proposal for the owner.

Examples use the same invented simulated family as the first batch: grandparents, parents Tom and Ann, children Sam (15) and Lily (10). They illustrate how a proposal would work; they are not results.

The suggestions are grouped by what they would change, not by paper. The first group affects how the model works. The other groups affect how it is tested and reported, and how the optional narrator would be checked.

---

## 1. Changes to how the model works

### MM1. Does anxiety change how a person reads the same event? *(from Mind or Message?)*
**What:** A test that gives one person exactly the same incoming events in two runs. The only difference between the runs is that person's own anxiety. The test checks whether the more anxious version reads the events as more threatening, both in the moment and in what they come to believe about the other person. A third run changes only a label on the sender (for example "father" versus "uncle") and should make no difference.
**Why:** The paper found that agents judged their partner mostly by projecting their own position onto them, not by reading the partner. The spec already says anxiety colours perception in the moment (M4.C.5, M4.C.6). The reader could not find where it says whether a person's lasting beliefs about another person are also coloured by their own state. If they are not, "anxiety distorts perception" happens only in the moment and never builds up into a lasting belief. That is a theory question for you, and the test would show which way the model behaves.
**Example:** Ann receives the same three messages from Tom in both runs: a short reply, a late arrival and a forgotten errand. In the low-anxiety run she reads them as "busy week". In the high-anxiety run the theory says she should read them as "he doesn't care", and that reading should shape her belief about the marriage. If both runs leave her with the same belief, anxiety isn't reaching her beliefs.
**Caution:** This is about perception coloured by one's own state. It is **not** the family projection process, which is anxiety focused on a chosen child over years. The two should be kept separate.

### AD1. Record the move a person held back *(from What People Almost Did)*
**What:** When a person catches themselves and doesn't act, the spec's WITHHOLD, the record should say which move they held back and how strong the pull was. Reports should show held-back moves next to the moves actually made.
**Why:** The paper's point is that behaviour hides what people nearly did. Someone who wanted to pursue forty times and held back looks the same in the record as someone who never felt the urge. The spec already says that a person who withholds is different from one who takes an I-position (M4.D.1c), but that difference cannot be seen unless the held-back move is recorded.
**Example:** Over a year, Ann makes no pursuing moves toward Tom. The record shows she held back a pursuit 38 times, mostly after his late arrivals. That is a very different Ann from one who felt no pull at all, and the theory treats the two differently.

---

## 2. Tests that check the model is doing what we think

### BB4. Check the engine against cases where the answer can be worked out by hand *(from Bayesian Belief Layer)*
**What:** A small set of tests that switch the engine into simple configurations where the correct answer can be calculated on paper. The engine must match that answer exactly.
**Why:** The paper validated each of its settings against a known mathematical answer. EPModel has no such checks. They catch plain coding errors before anyone interprets results, and they are tests of the code, not of the theory.
**Example:** Turn off all connections between people. Each person's anxiety should then simply fade by their own decay rule, and the engine's numbers must match the hand calculation to the last decimal. A second example: script Tom to make exactly the same move toward Ann every week. The rise in Ann's anxiety each week must equal the formula worked by hand.

### BB5. Any sudden jump the model claims must come from a named part of the rules *(from Bayesian Belief Layer)*
**What:** If a test says something flips suddenly (a tipping point, a sudden cutoff, a family locking into a pattern), the test must name which rule causes the flip. The rule is then replaced with a smooth, straight-line version, and the test must fail.
**Why:** The paper proved that a smooth, straight-line rule can never produce a tipping point. If a test of a sudden change still passes after the cause is smoothed away, the jump came from somewhere else.
**Example:** A test says Sam cuts off suddenly once tension passes a point, rather than drifting away gradually. The test names the threshold in the cutoff rule as the cause. Replace that threshold with a gradual slope. The test must now fail. If it still passes, something else is producing the "sudden" cutoff.

### AD2. Test the pull toward each move, not just the move that happened *(from What People Almost Did)*
**What:** When a test checks that rising anxiety pushes people toward simpler, more reactive moves, it should check the odds of each move, which the log already records, and not only which move was drawn.
**Why:** The odds can shift a lot without changing which move comes up. Testing the odds shows the mechanism more clearly and needs fewer runs.
**Example:** As Tom's anxiety rises, his chance of distancing goes from 20% to 45% and his chance of staying in contact falls from 50% to 30%. Over a few weeks he may still happen to stay in contact most times. A test that counts only his actual moves could miss the shift; a test on the odds will not.

### TM4. Check whether a result reverses at different levels, not just whether it holds on average *(from Diverse Minds)*
**What:** For each main test, name the things the theory says could change the result, at least the family's level of differentiation and its chronic anxiety. Run the test at three or more levels of each and report the direction of the result at each level. If the direction reverses somewhere, the result is conditional and must be reported that way.
**Why:** In the paper, one trait decided whether another trait helped or hurt: the effect of openness reversed depending on agreeableness. The traits' average effects didn't replicate across models; the reversal did. Two test settings, which the spec now uses, cannot tell a reversal from an effect that simply weakens.
**Example:** Coaching reduces cutoffs in a family at middle levels of differentiation. At a low level it might make no difference, because the family never takes it in. At a very low level it might even raise cutoffs for a while, because the change-back reaction is stronger. Running only two settings could miss the reversal completely.

### TM2. Check the setup actually took hold before reading the result *(from Diverse Minds)*
**What:** When two runs differ by a person's assigned starting quality, for example Ann starting at a higher basic level in one run, check two things before reading the result. First, that the difference is still there as the model measures it during the run. Second, that the two runs are far enough apart to matter. If either check fails, report it as "the setup didn't take hold", not as "no effect".
**Why:** In the paper, one model put its personas in the right order but so close together that its behaviour was flat. Without the check, that would have looked like "personality makes no difference". In EPModel, borrowing and lending can make a person's functioning look quite different from their assigned level.
**Example:** Ann is set to basic level 45 in one run and 55 in the other. By year 3, borrowing from Tom has lifted the 45-Ann's functioning so that the model reads both Anns at about 52. Any comparison of the two runs after that is not a comparison of a 45 and a 55 Ann, and must be reported as such.

### TM5. Name something the change should not affect, and check it doesn't *(from Diverse Minds)*
**What:** Each main test names at least one result the theory says the change should **not** move, and reports that result too. If it moves, something general may be inflating every result.
**Why:** The paper used a neutral topic (pineapple on pizza) to detect a model that exaggerated on every topic. The rule-based equivalent is a setting that raises every readout at once.
**Example:** Coaching Ann should change the marriage and the parent–child triangle. The theory gives no reason for it to change how often the grandparents fight with each other. If the grandparents' conflict also drops in the coached runs, the effect may come from something global, such as a scaling error that lowers all tension, and not from coaching.
**Caution:** The corpus rarely says outright that something is not affected. Where no such result can be found in the corpus, the test should say so rather than pick one.

---

## 3. How results are reported

### TM3. Report the parts of a combined measure separately, and count the opportunities *(from Diverse Minds)*
**What:** Measures that bundle several things, such as tension, fusion, or closeness and distance, are reported by their parts. A test names which part it checks. Every rate is reported with how many opportunities there were, so that "no chance to act" is distinguishable from "had the chance and didn't".
**Why:** In the paper, two parts of "polarization" moved in opposite directions. A low number also turned out to mean different things depending on how many opportunities there were. This extends J1 from the first batch.
**Example:** "Sam made few I-positions this year" could mean he rarely had an occasion for one (only 3 chances), or that he had 40 chances and took 2. Those are different Sams. The report should read "2 of 40", not "2".

### AD3. Show what kind of move a count is made of *(from What People Almost Did)*
**What:** When a report counts a move, it also shows what the count is made of: which channel the move went through, whether an I-position was genuine or the counterfeit assertion form, and which pressure drove it most.
**Why:** The same action can come from opposite processes. The spec already has counterfeit I-positions: the same label, the opposite process. A plain count merges them.
**Example:** Tom and a second simulated father each make 12 I-positions in a year. For Tom, 10 are genuine. For the other father, 9 are reactive assertions that look like I-positions ("I'm not going to take this anymore"). Reported as "12 and 12", they look the same.

### MM2. Check belief errors separately for similar and dissimilar pairs *(from Mind or Message?)*
**What:** When the model reports how wrong people's beliefs about each other are, report it separately for pairs who are alike and pairs who are different.
**Why:** In the paper, an agent that just assumed the other was like itself looked accurate when the two were in fact alike. Spouses in EPModel are matched on level, so this kind of error would be hidden in the marriage and show up only between parent and child.
**Example:** Ann's beliefs about Tom look accurate, because they are at similar levels and she is partly reading herself. Her beliefs about Lily, who is quite different from her, are much less accurate. Averaged together, this looks like "Ann reads her family moderately well", which hides where the error actually is.

### TM1. Record the starting mix each run actually got *(from Diverse Minds)*
**What:** When starting families are drawn at random, declare the average level and the spread as two separate settings, and record what each run actually got. Spread may be set across founding couples only. It must never be set within a couple (spouses match by rule) or across children (differences between children are something the model produces).
**Why:** In the paper, the limits on the scale made the actual spread differ from what was asked for. EPModel has the same issue: levels run from 0 to 100, and spouses must be within ±1 point.
**Example:** A setting asks for founding couples at an average level of 80 with a wide spread. Because levels cannot go above 100, the families actually drawn are closer together than asked. The log should record the spread each run actually got.

### MM4. Check the starting families are well formed before any run begins *(from Mind or Message?)*
**What:** Before a batch of runs, draw a sample of starting families and check that each one follows the declared rules. If any fails, stop before the first week runs.
**Why:** The paper checked 200 generated scenarios before starting and stopped on any failure. A malformed starting family otherwise shows up only as odd results much later.
**Example:** Draw 500 starting families. Check that every couple is within ±1 point of each other, that the dominant/adaptive pole is not tied to sex, and that every person has at least one move they are allowed to make in week 1. One failure stops the whole batch.

### TM6. Every number in a report comes from the analysis script, with a record of where it came from *(from Diverse Minds)*
**What:** No number in a report is typed by hand. A ledger records, for each number, which file, field and runs it came from. A check compares the report text against the ledger.
**Why:** The paper did this, and its text and ledger still disagreed in at least four places. So the ledger alone is not enough; the comparison check is needed too. This applies your standing rule against fabricated results.
**Example:** The report says "coaching reduced cutoffs in 83% of combinations". The ledger shows that figure comes from `sweep_results.csv`, column `direction_holds`, averaged over 500 combinations and 8 seeds. If someone edits the report to say 85%, the check fails.

---

## 4. The optional narrator (only if it is built)

### BB1. Can a reader tell who is most anxious from the narration alone? *(from Bayesian Belief Layer)*
**What:** Pick moments in a run where the engine knows the order of people on some measure, for example who is most anxious. Have the narrator describe them. Then have a separate reader, a different model or a person, who sees only the text, rank the people. The reader's ranking must match the engine's.
**Why:** The paper did this and showed the order survived the trip into language and back out. Exact amounts did not survive. It is the first test the narrator could fail on content rather than on labelling.
**Example:** In week 300, the engine ranks anxiety as Ann, then Sam, then Tom, then Lily. A reader given only the four narrated passages must produce the same order. Ann at 71 and Sam at 70 are too close to count, so pairs that close are left out, and the report says how many were left out.
**Caution:** The paper's levels were a factor of two apart, which made the test easy. EPModel's differences between people will be much finer, so the test should be run at that spacing.

### BB2. Map how the narrator bends the engine's values *(from Bayesian Belief Layer)*
**What:** For each narrator model, plot what the reader perceives against the engine's actual value across the whole range. Classify the bend as exaggerating both ends, shifting everything one way, or bending only one side.
**Why:** In the paper each model bent values in its own way. One exaggerated only the negative side. Shifts that lean one way are what distort results, because they don't cancel out.
**Example:** The narrator makes low-anxiety people sound calm, which is correct. It also makes high-anxiety people sound only moderately anxious, which flattens the top end. Every narrated crisis would then read as milder than it was in the engine.

### MM3. Change one input and see what the narration does *(from Mind or Message?)*
**What:** Give the narrator the same run record several times, each time changing exactly one thing: one person's anxiety, a role label, or the writing style. Only the anxiety change should change what the narration says about people's states. Every number in the input should also appear correctly in the output.
**Why:** The paper froze its records and changed one factor at a time to find what really drove its agents' judgements.
**Example:** Relabel Tom from "father" to "stepfather" and change nothing else. If the narration now describes him as more distant, the narrator is reacting to the label, not to the engine.

### X10. Check the narrator's persona took hold before using it *(from Diverse Minds)*
**What:** Before a narrator model is used, check two things about its personas: that they come out in the right order, and that they are far enough apart. Check behaviour as well as questionnaire answers. Include a neutral-content probe to catch a general style bias, and stop the run if too many calls fail rather than filling the gaps with blanks.
**Example:** The narrator's "reactive 15-year-old" and its "calm 15-year-old" must sound clearly different, not just slightly different in the right direction. If they are close together, stop before any narrated run is shown.

### BB3. If text ever feeds back into the model, measure the distortion it adds *(from Bayesian Belief Layer; exploratory LLM line only)*
**What:** This applies only if an experimental build ever lets narrated text change agents' state, which the v2 spec forbids. Run a matching version in which the agents receive the true values instead of the text, and report the gap between the two as distortion, never as a finding.
**Why:** In the paper, a small one-sided misreading added up each round and moved the group's final position from 0.50 to as low as 0.01. This is strong support for the current rule that the narrator never feeds back into the model.
**Example:** No example is needed for v2. This point supports the existing rule rather than adding one.
