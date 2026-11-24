# Suggested enhancements from six 2026-11-23 digest papers, in plain language

Plain-language companion to `SWEEP_READING_REPORTS_2026-11-23.md`. That file has the requirement wording, the evidence
and the spec references. None of these is in the spec. Each is a proposal for the owner, held for the next revision like
the earlier batches.

The six papers, all read in full: Chai et al. 2026 (*Alliance Rupture and Repair in Simulating Cognitive Behavioral
Therapy Dialogues*, 2611.27544), Zhou et al. 2026 (*On Evaluating User Simulators for Interactive Agents*, 2611.24892),
Keskin et al. 2026 (*Personalized Affective Perception via Continual Learning in Human-Agent Negotiation*, 2611.14247),
Qin et al. 2026 (*PsyWorld*, 2611.24999), Amar 2026 (*MANGO Arena*, 2611.23613) and Zhu et al. 2026 (*Probe What-If
Worlds Before You Ship*, 2611.14700).

Of the twenty-five candidates, only six propose a change to how the family behaves; the other nineteen are about how a
result is tested or reported. None is a theory change, and none goes to the narrator line. Three of the six model
proposals come from one paper (ALLY) and ask for something the spec currently lacks entirely: a repair that a person
can attempt and that can fail.

The examples use the same invented family as the earlier batches: the grandparents, parents Tom and Ann, and children
Sam (15) and Lily (10). They show how a proposal would work. They are not results.

---

## 1. Tests

### US1. Break the *instructions*, not the mechanism *(from the ten perturbations)*

**What it proposes:** The suite already contains mutants that delete, invert, sever, permute or re-round the
machinery. Add a second family that leaves every mechanism working and instead corrupts what one arm was told to
be: a different scripted act for one named member, a different starting belief, a removed constraint, a truncated
run, an external agent doing more of the work, and — the important one — a one-off reaction offered to the
estimator as if it were a standing trait. Each acceptance test must say in advance which of these faults it is
supposed to catch, and the suite prints a coverage count (n of the set) next to every verdict.

**Why:** The paper builds ten such faults and finds that three published metric suites catch only 5 or 6 of them,
under a rule as generous as "any metric, at any level, once." The failures they miss are the ones that matter:
a simulated user who sounds natural while approving broken work, or pursuing a different goal. EPModel is exposed
to exactly this class, because its arms differ by *declared inputs*, not by code.

**Example:** Both arms run the same 40 years on the same seeds. In both, Tom overfunctions at Sam and Ann
distances. The only difference is that Sam's declared act in year 1 is swapped from "I'll stay in contact with Dad
without managing his moods" to a plausible alternative, "I'll keep the peace by staying out of it." Every
mechanism is untouched. If no readout separates those two arms — say, Sam's share of `I-POSITION` toward Tom —
then the criterion that claims to test Sam holding a position is measuring something else. The estimator fault is
the sharpest: give the estimator a single bad week of Ann's and see whether her `basic_level` moves. If it does,
the estimator is absorbing a reaction into a trait.

### US2. Show that a test also *stays green* on healthy runs *(from the filtering results)*

**What it proposes:** A test that goes red under a mutant has shown it can fire. It has not shown it fires only
when it should. For each criterion, report the fraction of *unperturbed* seeds that still pass at the threshold
that lets 5% of the fault-injected seeds through. A criterion that also rejects 40% of its healthy seeds is
"screening but not discriminating" and cannot be cited as evidence for its mechanism. And no overall score across
criteria — the paper shows a composite that ranks broken runs *above* controls.

**Why:** Three suites in the paper scored 6 out of 10 on sensitivity while their combined scores preferred a
population instructed to accept broken work. A separate result is worse: for premature stopping, the aggregate
fidelity score *rose*, because cutting runs short brought their session lengths closer to the real-user average.
The metric rewarded the defect it was meant to catch.

**Example:** The criterion says the coach's contact raises Tom's `basic_level`. The mutant routes the coach's
contact to Ann instead, and the criterion goes red — good. But suppose that on 35 of 100 unmutated seeds the rise
never clears the declared margin, because coaching lands thinly. At the threshold that catches 95% of the routed
mutants, the criterion also throws out a third of the ordinary families. That is worth knowing before anyone
writes "the coach helps" in a report. And if the project ever prints a single headline number averaging all
criteria, the paper's evidence says it can invert.

### US3. Record *which* family member the mechanism acted on, seed by seed *(from population vs trace fidelity)*

**What it proposes:** When a criterion names a member — as the spec already requires — classify every seed as:
the mechanism acted on the named member; it acted on a different member; or it did not act. The classification
comes from the run log, not from an average over the family. The criterion passes only if the named-member count
beats the other-member count by the declared margin, and the other-member count is printed with the verdict.

**Why:** The paper's central finding is that population statistics can look right while the individual case is
wrong — the aggregate cannot see that the simulated user changed the goal. It cuts both ways, too: for a changed
goal the trace-level win rate was 0.70 and the population-level win rate 1.00, so aggregation amplified one
failure and hid another. The spec's rival-arm rule already asks that an assertion be *about* the named member,
but it checks that through the shape of the ensemble result, not by counting where the mechanism actually landed.

**Example:** The criterion says Sam's functioning falls in the year after his grandmother dies, and the mean over
100 seeds confirms it. The classification shows that in 45 of those seeds it was Lily whose functioning fell —
Sam was untouched. The mean was real and the attribution was wrong. Report both numbers, and the finding becomes
"a death in this family lowers *a* child's functioning, usually the more reactive one" rather than a claim about
Sam.

### US4. Count per week at risk, so a shorter run cannot score better *(from the premature-stop artefact)*

**What it proposes:** Any readout that counts events, or time spent in a state, must be divided by exposure —
person-weeks at risk, or weeks in the state — and the denominator must be printed. A run in which someone died,
a tie was cut, or the horizon was shortened may not be compared with a longer run without that adjustment.

**Why:** The paper's clean demonstration is that truncating runs improved a fidelity score, because the shortened
run looked more like the reference. EPModel has the same exposure problem structurally: over 40 years its arms
differ in person-weeks by construction, because members die and ties cut. A criterion counting cutoffs or symptom
onsets per run will quietly favour whichever arm lost people earlier.

**Example:** The grandmother dies in year 20 in one arm and in year 38 in the other. The first arm has roughly
1,040 fewer person-weeks. "Four cutoffs against six" then says almost nothing; "four cutoffs per 1,000
person-weeks against three" is the comparison anyone would actually want. Print the denominator, because an
adjustment can also hide a mechanism whose whole effect is to end exposure — a cutoff removes the contact-weeks
in which the next cutoff would have formed.

---

### AP3. Any mechanism that keeps learning gets tested in three states, not two *(from Personalised Affective Perception)*

**What it proposes:** Where a criterion credits something that both *starts out calibrated* and *keeps adapting*, run
three arms — off, calibrated then frozen, and adapting throughout — and report the two steps separately. If freezing it
after calibration gives the whole effect, say plainly that the continuing adaptation did nothing, in the same way a rival
arm that reproduces the result is already reported as not discriminating.

**Why:** This is the paper's own first listed limitation, and it is the cleanest thing in it. Its comparison was "six
minutes of training plus ten minutes of continued learning" against "no training and no learning", so it cannot say which
half did the work; the authors say the fixed-after-training arm was not run and would have separated them. EPModel has the
same gap in its arm vocabulary: `M10.B.5` lets learning be switched off in a test, and `M17.D.1` names five control arms,
but none of them is "learned for a while, then held still". Every result about the learner is currently all-or-nothing.

**Example:** Sam's repertoire of moves is learned from what relieved his anxiety. Three arms: no learning at all; learning
for his first five years and then frozen; learning all forty. If the frozen arm already produces the whole difference in
how he behaves at 15, then "he was still learning at 15" is decoration and must be reported as such — otherwise the
finding reads as evidence for lifelong adaptation when it is evidence for childhood.

### AP4. Anything that is read *and* updated is read from last week's numbers *(from Personalised Affective Perception)*

**What it proposes:** If a quantity is both used in appraisal and changed by it — a person's belief about what a partner
usually sends, AP1's range, the twelve-week variability, the running danger average — then every appraisal in a tick must
use its value as of the end of the previous tick, and it must not be updated until every arrival in that tick has been
appraised. Two things arriving in the same week are therefore read through the same numbers, whatever order the queue
happens to process them in.

**Why:** The spec already says an arrival is judged against the state *before* the week's batch (`M1.F.8`). It does not say
the same about a quantity that the appraisal then *writes*. Without that, an implementation can legitimately smooth a
belief after the first arrival and before the second, and then which of the two costs more depends on queue order. The
worst part is that it would not be caught: the order is deterministic, so the log is byte-identical run to run, and it is
deterministic and wrong. The paper's architecture is the same discipline in a live system — the learning thread publishes
its weights only at the end of a step, and the interaction thread picks them up at the top of a cycle.

**Example:** One week both Ann and Tom say something to Sam. If Sam's belief about what Tom usually sends is smoothed
right after Tom's remark, then Ann's remark that same week is read against a slightly different baseline, and the two
get different prices for no reason anyone in the family would recognise. The rule says both are read against last week.

**The exemption to write down:** some things are *supposed* to move inside the week — a tie's bond energy, the anxiety
returned to a sender when their contact relieved the receiver. The rule has to list what it excepts, or it would forbid
coupling the design deliberately has.

---

### PS1. Say how many people are actually doing the thing *(from PsyWorld)*

**What:** Right now a test passes if one arm produces more of something than the other. That is still right. The addition is that the report must say *how many people* produced it. For every measure, count the distinct people (and distinct ties) that contributed to the difference in each arm, and if only one or two people carry it, say so in the same sentence as the result: "the coached arm cuts off more, and in nine of ten seeds it is Sam".

**Why:** In the paper the overall accuracy was almost identical across four groups of people (0.891 to 0.913 — a spread of two points), while performance on the rare action (teleporting) ranged from 0.13 to 0.26 in those same groups. The average looked steady because it was being carried by the common actions. The authors caught it only because they broke the result down, and then warned that with 7 to 18 people per group the breakdown is descriptive, not causal. EPModel has the same exposure and more of it: there are only twelve people, so a "family" result can be one person's behaviour with eleven others unchanged.

**Example:** The test says the coached arm produces more `CUTOFF`. Over ten seeds the family mean rises. The new field says: nine of the ten seeds' rise comes from Ann cutting off her mother; in one seed Tom also does. The verdict still passes — the direction is real — but the report now reads "Ann cut off", not "the family cut off". That distinction is the whole point of the model: a cut-off *person* and a cut-off *family* are different claims, and `M11.C.28` already refuses to let symptom count stand in for differentiation.

**Owner decisions needed:**
- How many people is "enough" for a family-level reading? Two of twelve? Three? It is an invented number either way.
- Should the flag ever fail a test, or only annotate it? As written it annotates, because the theory has real mechanisms (`M7.D`'s projection onto one child) that legitimately land on one person.

---

### MG3. A result is not settled until it holds on seeds nobody has looked at yet

**What it proposes:** Every directional criterion gets two blocks of seeds. The first block gives the
verdict. Then, on a second block of seeds that has never been used for that criterion and whose size was
written down before it was drawn, the same comparison is run again. If the direction holds on both blocks,
the criterion passes. If it holds on one and not the other, the verdict is a third outcome — **not
replicated** — neither pass nor fail, and nothing may describe the mechanism as supported. Both blocks are
reported side by side, with their seed ranges and their intervals. A mutation clause is judged the same way:
if deleting the mechanism turns the test red on one block and not the other, the test is not shown to depend
on the mechanism.

**Why:** MANGO Arena's Table 4 is the cleanest demonstration I have seen of the problem. Same model, same
prompts, same seeds, no failed calls in either run: for one seed, Factory amplification went 6.9 to 113.4;
for another, Supplier went 8.0 to 134.2. Their conclusion about what the fire did to that model reversed
between the two rounds. Their own fix, in §6.3 and §8, is to report spread, set the seed count from the
variance observed, and stop making claims from small blocks.

The spec's stopping rule (`M17.A.1`) is careful but it reads one stream of seeds: it adds seeds until the
interval is narrow enough and stops. That is precisely the moment the running estimate happens to sit
furthest from zero, and with a heavy-tailed per-seed difference — which is what the paper's table is — a
small block can look decisive. Confirming on seeds that played no part in the stopping decision is the cheap
repair, and it is one-sided: only passes need confirming, so failures cost nothing extra.

**Example:** The test is that Tom's coaching of Sam raises Sam's `functional_level` relative to a matched
arm where Tom does not coach. On the primary block, 60 seeds, the difference is +3.1 with a half-width of
1.2 — passes. The confirmatory block was declared at 60 seeds before it was drawn, and gives +0.4 with a
half-width of 1.3. The verdict is **not replicated**. The record says both blocks and their intervals; the
summary sentence in the write-up cannot say "coaching raises a child's functioning in this model", because
on half the seeds nobody looked at first, it did not.

**Worth saying plainly:** this will produce a run of "not replicated" verdicts, and that is the point — it
is a smaller embarrassment than a published direction that a later block reverses silently. The owner may
want to scope it: confirm only the criteria that gate a phase, or only those above a declared effect size.
The spec already believes this for one criterion (`M5.D.3`'s note that a divergence is not read as met until
fresh, disjoint seed blocks show it); MG3 makes it a rule and gives the outcome a name.

---

### PW1. A result that only holds in one family is a fact about that family, and must be reported as one *(from Persona Simulation)*

**What it proposes:** when a test passes and we want to call it a finding, the record must say whether the direction was
also run against a differently shaped family — a second family, a different starting draw, an imported one — and it must
say **held**, **sensitive** or **reversed**. "Not checked" would no longer be allowed for anything we are calling a
finding. If the direction reverses in the other family, we report that the result belongs to the composition we built, not
to the mechanism.

**Why:** this is the paper's strongest result. When they removed the persona descriptions, the link between a person's
professional group and what they prioritised collapsed from a Cramér's V of .928 to .063 — and in another study the
ranking of two contract versions flipped from +17.25 points to −15.50. Not weakened. Reversed. The numbers are not
transferable to us, but the lesson is: with a single hand-built family we cannot tell a mechanism from a family shape
until we have run it somewhere else.

**Example:** the test says a low-level family reaches a symptom threshold sooner than a high-level one, and it passes on our
twelve-person family. Before that goes in the findings column, the same contrast runs on the Phase C family, where the
children are a different number, a different age and a different birth order. If the direction holds, say so. If it holds
only because Leo is eight at week 0 and observed for twelve years, the record says the result is about who was in the
family.

**Worth saying plainly:** this is close to something already in the spec. `M17.G.1` already asks for an audit record with
family composition as one of its dimensions — but it lets the value be "unaudited", which is exactly the hole. And
`M15.D.2`–`M15.D.4` already require an envelope and a named flip, but only for imported real families. This proposal is
those two rules joined up and made to bite on the reference family. It also must not become a way to shop for a second
family that agrees; the second family gets named before the first result is read.

### PW2. Every test names one thing it must not break, with the limit set in advance *(from Persona Simulation)*

**What it proposes:** beside the direction it asserts, each acceptance test names one guardrail — some other quantity, one
its own mechanism is not aiming at — and a bound that quantity must stay inside. If the test passes but the guardrail is
breached, the result is reported as a pass with a cost attached, and we do not describe it as support for the mechanism.

**Why:** the paper's method separates three kinds of measure — the outcome that decides, the diagnostic, and the guardrail
— and its case reports read a win against what it cost. Asking how users felt built trust; they tracked privacy confidence
in the same table because the friendly version could have bought trust with privacy. In another case the revised contract
improved acceptance overall while one group of a hundred clients accepted nothing at all in either model. That second fact
is the one a person acting on the advice needs.

**Example:** the test for an I-position at a low level says the family's anxiety rises more when the move comes from Tom at
35 than from Tom at 60. Good. The guardrail is Tom's own estimated basic level: if the rise in family anxiety is bought by
Tom's level falling, that is not the theory working, that is someone paying for the disturbance with themselves. Bound
declared before the run: his estimate may fall by no more than X. Passed with the guardrail breached reads "the family was
disturbed and Tom went backwards", not "the mechanism works".

**Worth saying plainly:** two ways this could go wrong. A guardrail can be set so tight that nothing passes, or loose
enough to be decoration — so the bound is recorded before the first run, like any other constant. And the guardrail must be
something the mechanism under test is not aiming at, or it becomes a second copy of the assertion and proves nothing.

---

## 2. Reporting rules

### AL4. Count successes against the chances there were, and never let an arm buy successes with extra attempts

**What:** Some outcomes only have a chance to happen if a situation arises — a repair after a rupture, a position held under pressure, contact after a cut-off. Every readout of that kind should report how many chances there were in each arm, and a run where the situation never arose should be marked "did not arise", not scored as a zero and not quietly dropped. A chance that arose and drew nothing counts as a failure. And if one arm of a comparison makes more attempts, each revised attempt is scored against the objective it started from, so extra attempts cannot manufacture extra successes.

**Why:** The paper does exactly this for its repair-skill rating: it is scored only where a relational difficulty is visible, a missed visible opportunity counts as a deficit rather than a blank, and the number of rated dialogues is printed beside the score — 124 of 228 for the system under study against at most 18 for any baseline, which is a very different denominator. Its plan-revision step links every rewritten goal back to its original objective for the same reason: so that adding plan items does not inflate completion. EPModel's criteria have no such rule, and a criterion like "the run of negative contacts standing at a tie's first cut-off" has no defined behaviour in a run where nobody cuts off.

**Example:** In 4 of 12 seeds Tom cuts off from his father; in the other 8 he does not. A "repair rate" averaged over all 12 divides by 12 and reports 8 runs that were never tested. With AL4 the report says 4 chances, and the 8 are marked as not arising. Separately: the arm that allows a second repair attempt in week 40 has more chances to succeed than the arm that allows one, so a bare count of successful repairs favours it. Scoring each attempt against its original objective removes that.

### AL5. Say what a confidence interval is resampled over, and report the nested version

**What:** In Phase E, every interval should name the thing it resamples — individual runs, or whole families drawn from the import distribution. When runs are nested inside family draws (12 families × 8 seeds is 96 runs but only 12 independent draws), the interval must be computed over the 12, with the naive per-run interval printed beside it. If a comparison clears the declared margin under one and not the other, it is reported as undetermined.

**Why:** The paper has this exact case and prints both answers. One contrast is +0.022 with p = 0.068 when the resampling unit is the 57 real sessions the simulated ones were matched to, and p = 0.035 when it is the profiles. The same data, two defensible units, opposite verdicts. EPModel is exposed to the same error: families are imported as ranges and each is run over many seeds, so a per-seed bootstrap will give an interval that is too narrow. The spec's existing rule about verdicts that move between two *engines* at p between 0.01 and 0.2 catches this only by luck, because a wrong unit gives a wrong interval at p = 0.001 too.

**Example:** Twelve family draws, eight seeds each. Under one arm, anxiety in Sam's last year at home is 3 points higher. Resampling the 96 runs gives one interval; resampling the 12 families gives a wider one. If the first clears the declared margin and the second does not, the honest report is "undetermined at this ensemble size", not the first number.

---

**Narrator line:** no candidate here. Nothing in this paper bears on the optional narrator, so no X- or IR-prefixed item is proposed.

### US5. "It reads like a real family" is not evidence *(from the human study)*

**What it proposes:** Any judgement that a run, a trace or a narrator line looks right — the owner's, a reader's,
or a language model's — is recorded as a **plausibility note**, names who made it, and never contributes to a
verdict or a claim grade. Where the spec already asks for human review — the three-phase curve shape, the
counterfeit-move threshold — the review is recorded as *choosing an invented value*, not as confirming a result.

**Why:** The paper had raters try to pick the deliberately broken replay out of a matched pair. They scored 0.56,
with agreement of 0.51, against a floor of 0.50 for guessing. The defects included a changed goal. If trained
raters cannot see a swapped goal in a dialogue, a reader cannot see a wrong attribution in a narrator line, and
EPModel's readable trace is exactly where that temptation lives.

**Example:** The year-12 narrator line reads well: Ann goes quiet, Tom fills the silence, Sam takes the blame.
Someone says "that's how a family like this would go." That is a plausible sentence and it may even be true, but
it is not a test result — the same reading would have been produced by a run in which the triangle had been
routed to Lily. Record the note, name the reader, and let the log predicate decide.

---

### AP5. If one member gains, report what the other members got, and check the channel the mechanism names *(from Personalised Affective Perception)*

**What it proposes:** When an arm or a criterion says a change raised one member's number, the report must show the same
number for the other people on the ties that member acted on, and at least one joint number — a named component of the
family evaluation, not the whole composite. And where the mechanism names a step in between ("she stops pursuing, so he
stops distancing"), the report must say whether that step moved, on the same seeds. If it did not, the mechanism must not
be described as supported.

**Why:** This is the paper's most instructive failure and it is not a modelling failure. The personalising robot gained
about 2 points of its own value, robustly — it survived every check the authors ran. Its partner's value did not move and
neither did the joint measure. Its *own* concession rate was identical in both conditions, so the gain came entirely from
the human conceding more. And the story the authors tell — the human concedes more, therefore the robot gains — was
tested and came back at essentially zero, with 13 of 14 exploratory models including zero, and the one that did not
pointing the wrong way. The abstract still reads as though the route was shown. EPModel's criteria are directional and
several of them read one member only (the child carrying the projection, the one who cuts off); nothing currently makes
the counterpart's number appear beside it.

**Example:** An arm says Ann's `I-POSITION` works: her acute anxiety is lower than in the control. Report Tom's anxiety
beside it, and Sam's, and the family evaluation's intensity component — if Ann is calmer because Tom is now carrying it,
that is a different finding and the report has to be able to say so. And if the arm's stated route is "she stopped
pursuing, so he stopped distancing", then the count of her pursuing acts has to be checked. If it did not fall, the two
numbers moved together for some other reason and the route is unproven, however clean the contrast looks.

**One caution on the wording:** the check must ask for the *direction* of the middle step on the same seeds, not for a
properly estimated indirect effect. Twelve agents cannot power a mediation, and a rule that demanded one would either be
ignored or produce an interval nobody can read. `UNDETERMINED` is an acceptable answer.

---

### PS2. Separate what was chosen from what arrived *(from PsyWorld; only meaningful with the social cues switched on)*

**What:** When the social cues are on, the report should say how often the event a person received was different *in kind* from the one that was sent. Two cases already in the spec produce that: a private hop under `M1.F.4` arrives with tone, face and posture empty, and a `TRIANGLE` can reach a different third party than the sender had in mind. Record the count and share per pair and per move kind, next to the move counts. Then a test may not credit a change to "who chose what" when that share is high, unless there is an arm that isolates selection.

**Why:** The paper's own model names the right high-level action 90.4% of the time and then fails to carry it out: completion of the requested action is 0.332 in its game, against 0.990 in the other simulator, and the gap between intended and realised move frequency is 0.520. It reports those as separate numbers, which is why the failure is visible at all. EPModel's version of the same gap is built in rather than learned: an event is not what its sender chose, because fidelity degrades per private hop and the nonverbal channels drop on a reported message. Without this number, a difference between the coached and uncoached arms could be a difference in what got through, not in what anyone decided.

**Example:** Ann says something sharp to Tom about his mother, and it reaches him through Lily, so he gets the words with the tone, face and posture missing. If the model says Tom distanced more in the coached arm, we want to know whether more of the sharp things actually arrived, or whether the same share arrived and Tom handled them differently. Those are different findings and only one of them is about coaching.

**Caveat, and it is a real one:** with the cues switched off, fidelity is a declared constant that scales how strongly a move lands but never changes what kind of move it is, so this number would sit at zero and mean nothing. The rule should therefore apply only when `social_cues_enabled` is on — and that costs nothing, because `M4.C.1f` already requires a cues-off run to be record-for-record identical to the engine before that revision.

**Owner decisions needed:** whether this belongs in the family evaluation readout (`M11.G`) as a standing component, or only in the trace renderer as a diagnostic. The first makes it hard to ignore; the second keeps the evaluation short.

---

### MG1. The count has to add up: every person-tick is one thing, and a forced outcome is not a choice

**What it proposes:** For each living person, each week, the log holds exactly one outcome record, and it
says which of five things happened: the policy chose a move; the tie-break chose between equal scores; the
fallback rule produced an outcome because the scores were unusable, the person's energy was spent, or no
move was legal; an I-POSITION step was deferred a week for a collision; or the person withheld, and chose
to. Every ensemble report prints that census per person and per arm, and the numbers have to add to the
living person-weeks. A week with no record, or with two, fails the run as a log defect. And — the operative
half — no move marginals, no transition matrix and no criterion readout may fold the tie-break or fallback
rows into the move they happened to resolve to; they are their own row.

**Why:** In the pilot, a failed API call was recorded as "no valid order". The company ordered zero that
period, so the zero entered the demand-variance measure as a real ordering decision, *and* the failure was
recorded as a rule violation — one fault moving two headline numbers, both upward. The authors' rule, now
in §6.1, is that an evaluation must keep "the agent decided X" apart from "the agent never decided", and
report the second as its own number.

Our engine cannot lose an API call, but it does have outcomes that are produced by something other than a
choice, and today they are flagged per record (`M16.A.3c`) with a rate that gets reported (`M11.D.18`), but
nothing keeps them out of the readouts. That is the same defect one layer down.

**Example:** Lily is 10 and both parents have effectively cut her off, so `PURSUE` has no legal target this
week. The fallback rule emits `DISTANCE`. Under the current spec, a report on "Lily's distancing from her
parents, years 8–12" can be built partly out of weeks in which she had nothing else available to do. With
MG1, those weeks sit in their own row, the census shows 41 such person-weeks for Lily against 520 total,
and the sentence the owner actually wants — "Lily stopped reaching for them" — is either true of the 479
weeks where reaching was possible, or it is not.

**Worth saying plainly:** `WITHHOLD` has to stay a *chosen* class, or this becomes a second way of writing
"no move". One sentence in `M4.D.1` fixes that, and it is the only real complication.

### MG4. Keep every ensemble record, and never quote a retired verdict bare

**What it proposes:** Ensemble records are append-only. Each one gets its own identity and is never
overwritten; it carries the seed, the config hash, the spec revision, the family, the seed range and the
verdict as rendered. If a record replaces an earlier one, it names the earlier one and states why — a spec
revision, a fixed defect, a changed readout, a changed seed set. Any document that quotes a verdict cites
the record it came from, and a verdict from a retired record is marked retired at the point it is quoted.
The engine still writes nothing; this is the runner's and the repository's job.

**Why:** The paper's §6.3 lists this among the four practices it adopted after the pilot, and §6.2 is the
case that shows why: an earlier draft explained the Luna numbers by the failed calls, and Table 4 showed
that explanation incomplete — the disruption games had had no failures at all. A retraction is only
possible if the old run still exists to compare against.

We do this by hand already. The criteria table says "Regenerated at revision 32", "superseded at revisions
2…", "measured before the baseline was matched". That is a good practice held together by care, and it
should be a requirement, because it is the only way the independence claim (no arm fitted to a known
history, `M11.F.9(c)`) can be audited after the fact.

**Example:** `M11.C.48` passes at +0.104 ± 0.070 after the baseline was matched on sides. A year later
someone reads "+0.104 ± 0.070" in a summary and treats it as current. With MG4 the summary line carries the
record identity, the record says it supersedes the unmatched version and why, and the reader can see it was
regenerated at a revision whose engine has since changed again.

**Worth saying plainly:** scope the "state why" clause to records that carry a verdict or that a report
cites, or it becomes a clerical tax on every regeneration.

---

### PW3. Say whether a difference is more people, or more effect per person *(from Persona Simulation)*

**What it proposes:** wherever a test's number is a count or a share, the report gives three things: how many people or
weeks were even in a position for the mechanism to fire, how many of those it fired in, and the rate among those it could
have. Then one sentence saying which of the three carries the difference between the arms. If the arms differ only in how
many were eligible, we do not describe that as the mechanism acting more strongly.

**Why:** in the paper's marketplace study two models disagreed by a factor of two about how many people were even in the
situation — 97.3%–100% of the cohort versus 30.9%–75.2% — while agreeing closely, once someone was in the situation, on
what stopped them: distrust of the agent, 92.1% against 95.6%, and price, 0.04% under both. Nearly the whole disagreement
sat in the reach term, not in how people behaved. The paper says that reach number is the one you most need from real
people before you scale anything.

**Example:** "cut-offs are more common in generation three" can pass because the third generation holds more ties that
could be cut, or because each tie that could be cut is cut more often. Those are different findings about the family. The
report prints all three numbers and says which one moved: Ann's generation had the same number of cuttable ties in both
arms and cut them at twice the rate — that is the mechanism; or, the rate was identical and only the number of ties
differed — that is the shape of the family, and says nothing about how anyone handles anxiety.

**Worth saying plainly:** the spec already does this for one case — asymmetric rules must be compared at matched event
counts (`M11.1e`), and the log already keeps magnitudes and counts apart (`M16.A.8`). A per-opportunity denominator for
move rates was also proposed on 2026-10-12 (PR1). This extends the same discipline to every outcome readout, which is the
part nobody has written down.

### PW4. The unit that gets resampled is the family, never the person-week *(from Persona Simulation)*

**What it proposes:** when a readout is computed across more than one person or tie in a run, the report must say so and
must resample at the level of the family. Any "n" printed beside it counts families, not people, weeks, ties or events.
A test that treats people inside one family as independent observations is not allowed.

**Why:** the paper shipped a statistical module and then wrote down its defect: the generic categorical tests assume the
two arms are independent observations and do not account for the same persona appearing in both, so their p-values need
re-checking for any repeated-person design. Where they had real group structure — 114 workers in eight teams — they did it
properly, resampling whole teams and keeping each person's answers together. EPModel has the same trap in a sharper form:
twelve people over forty years is a table of about twenty-five thousand rows that belongs to **one** family.

**Example:** Lily's weeks of high symptom load look like a big sample. She is ten, she is one child, and her anxious weeks
are not independent of each other or of Sam's. If a report says the low-level arm carried more symptom-weeks "at n = 4,160", the number is meaningless — it is one family, forty years. Said properly: "in 12 families the low-level arm carried more
symptom-weeks, 9 of 12 seeds in the same direction", with the spread over seeds beside it.

**Worth saying plainly:** this is my own extrapolation of the paper's caution, not a result in it. It does not weaken any
existing rule — the paired, per-seed statistics in `M11.4e` and `M17.A.3` stand — it just stops a person-level average
from borrowing a precision the model does not have.

---

## 3. Model changes

### AL1. Repair takes several steps, and it can fail

**What:** Repairing a damaged relationship cannot happen in one move. It has to run as an ordered sequence — notice what happened, say what it was, the person who did the damage owns their part, then re-agree how to go on — and each step only counts when the other person produces something in their own words: what stung, what they want instead, how they want to continue. The other person saying "fine" or "okay" does not count as progress, and neither does the repair move simply being delivered. If the sequence runs out of time at a step, the relationship keeps the strain.

**Why:** In the paper the repair ladder is four fixed stages, and the rule that governs it is that a counsellor's apology or a client's agreement "never completes a repair card" (App. E, Fig. 11). It is the mechanism that resolves 55 of 57 major ruptures. EPModel today has no repair a person can *choose* at all: reconciliation is a scripted event that fires and empties the whole stored strain at once. So in EPModel repair cannot fail, and "smoothed over" and "worked through" are the same event.

**Example:** Tom asks Ann about her mother's last hospital stay. Ann answers in one word. Tom presses for the exact date and Ann goes quiet. Today's model: a reconciliation event next month returns everything the tie was holding, and the relationship is as it was. Under AL1 this takes four weeks. Tom says what he noticed — Ann went quiet after the second question. Ann says what stung — "you wanted the date right, not how I was". Tom owns that he did push. Then they agree how to talk about her mother. Each of those steps only counts when Ann actually says the thing. If Tom jumps to "so shall we try again next month", the sequence stalls where it is and the tie keeps what it held.

### AL2. Repairing the wrong thing does not repair it

**What:** A relationship between two people carries three different things: how close they feel (the bond), who does what for whom (the balance), and what they do together (the shared activities). When one of those is strained, a repair aimed at a different one does not help. It should make things slightly worse, because the other person reads it as another sign of not being heard.

**Why:** In the paper the repair instruction is chosen by which of the three was damaged — if the activity was the problem, change how it is done; if the focus was the problem, let the client set the priority; if the bond was hurt, check first that they feel understood. And a repair that returns to the task too early counts as a fresh rupture (Fig. 10, "re-rupture"). EPModel has all three quantities on the tie but only one repair event, so it cannot represent an apology for the wrong thing.

**Example:** Tom keeps coaching Sam on maths. Sam has stopped bringing work home. What was actually hurt is the bond — Sam read the coaching as "you think I'm stupid" — but Tom repairs the activity: "fine, we'll do less maths." Nothing recovers, because the maths was never the injury, and Sam reads the withdrawal as Tom giving up on him. The repair that works is "I made you feel stupid about it, and I did it in front of Lily."

### AL3. Getting past a painful subject leaves the relationship in one of two states

**What:** A subject that hurts between two people should carry a sensitivity level, not just be on or off a list of forbidden topics. And there are two ways to get past it. *Bypass* — the triggering move is dropped or done more gently — lowers the sensitivity one level and leaves the subject live; do it again and it drops again, but it never goes away. *Addressed* — the specific thing is named and the feared response is disconfirmed — closes the subject for that pair permanently, so the same move no longer registers.

**Why:** In the paper a trigger carries a three-state memory, open, raised or closed, and the two exits write different states (Table 11, Algorithm 1). EPModel's forbidden-topic list is a set that grows and can be blunted by deliberate mention, which is one path with no grade and no closure. The distinction the paper adds is the one Bowen makes: avoiding a subject and working it through do not leave the same relationship behind.

**Example:** Ann's mother drinks. Twice Tom has steered the conversation away just as it got near — each time the subject got quieter but stayed live, and the third time it came up it landed as hard as the first. Once, the two of them actually named it, and Ann said what she had been afraid of: that Tom would think her family was trash. That subject has not registered since. Under today's model both routes look the same after enough quiet weeks.

---

### AP1. A cue means something different on each relationship, because each relationship has its own range *(from Personalised Affective Perception)*

**What it proposes:** When the social cues are on, each relationship should carry, for each person and each of the four
cues, how much that partner's cue normally *varies* — and how much a change in that cue costs should be measured against
that range, not against one range for the family. A partner who never varies makes a small departure mean a lot; a partner
who swings constantly makes the same departure ordinary. The range must be built only from what the receiver *perceives*,
never from what the sender actually sent, and it must start from the ordinary level-based sensitivity so a new relationship
still works on day one.

**Why:** The spec already has two halves of this and is missing the third. Each person holds a believed cue vector per
partner (`M9.9`) — that is where the partner's cue usually *sits*. How sensitive a person is, and how much they misread,
are derived from their own level (`M4.C.14`) — that is one *slope* for everyone they know. Nothing holds how much this
particular partner's cue *spreads*. The paper's whole apparatus is that third quantity: it re-centres and re-scales its
reading against the partner's own running mean and spread, and the spread is where its output range changed by a factor
of 5.7. Its own words for the problem are a generalisation gap: a generic mapping "degrades across individuals".

**Example:** Tom's face barely moves — week to week his approval toward Sam sits within a hair of the same value. Lily's
moves constantly; she is up and down all through dinner. Sam reads a 0.3 drop in approval from each. Under the current
rule those two cost Sam the same, because the drop is the same size and both are measured from Sam's belief about where
each of them usually sits. With AP1 the drop from Tom costs a great deal — Dad never does that — and the drop from Lily
costs little, because for Lily that is a Tuesday.

**The trap to write into the rule:** if the spread follows the partner too quickly, a real slow slide gets absorbed and
stops being noticed — Ann's approval of Sam can decay a hair a week for two years and never once register as a change.
So the spread has to move slowly, and there must be a test that a steady twenty-week decline still registers before the
reference eats it. The other trap: if the spread were computed from what the sender actually sent rather than what the
receiver perceived, then averaging over many deliveries would cancel the receiver's misreading, and a person would end up
with a back channel to the truth — which the whole appraisal design forbids.

### AP2. A change in a cue should count for more than the level it changes to *(from Personalised Affective Perception)*

**What it proposes:** When a cue arrives, how much it costs should be scaled up if it has just moved — by a bounded factor,
so a big jump counts for more than a slow drift to the same place. It must scale the existing two-sided deviation, not add
a new drive: the two drives (too little contact, too much impingement) stay the only drives, and the cues stay readings.

**Why:** The model's appraisal is a *level* appraisal: anxiety rises with how far a relationship sits from the person's
felt optimum. A relationship that has sat at a poor level for a year and one that has just fallen to it are, on the week
of the fall, identical. The paper's entire control rule is built on the change instead — direction from the rotation of
the valence–arousal vector, strength from its radial change, capped at one so it cannot run away. The nearest thing in
the spec, `M4.C.16`, reads how *variable* a sender's cues have been over twelve weeks and uses that to intensify the
reaction; that is dispersion, not direction, and it says nothing about a drop.

**Example:** Tom's tone toward Sam has been flat and slightly clipped for months. It costs Sam a steady, mild amount every
week. Then one week it goes sharp. Under the current rule that week costs about what the flat weeks cost, once the level
is the same. With AP2 the sharp week costs noticeably more — which is how a family actually notices a change — and because
the factor multiplies rather than adds, a family in escalating conflict cannot spiral on this term alone.

**Why multiplicative:** an added term would be a third thing pushing anxiety up with nothing to stop it. Multiplying the
existing deviation keeps the saturating shape the appraisal already has. It is still an invented premise with no passage
behind it, and should be labelled as one, like `M4.C.11`.

---

### MG2. Record what the person reached for, not only what they were allowed to do

**What it proposes:** The legal set is already logged — which moves were unavailable this week. MG2 adds the
other half: the score of the outcome the policy would have ranked first *before* the mask was applied, and
its rank. Where that outcome has no target at all (nothing to pursue, no third party to triangle in), the
score is recorded as undefined, not as zero, and those weeks are counted separately. The report then gives,
per move and per person, how often the first choice was illegal and what was enacted instead. It is a
**score**, never a "propensity": it is not normalised over the legal set and does not sum to one, and calling
it a probability would invite exactly the misreading the spec warns about for repertoire entropy.

**Why:** MANGO Arena's rule engine blocks an order that breaks a rule and logs the attempt — the violation
is the observation, not the substituted order. That is why "1, 7 and 1 of 288 decisions" is a real result
about behaviour under pressure rather than a footnote.

In a family the analogue is not compliance — a family has no referee, and its rules are its own emotional
process, which is a mechanism in our model, not a constraint. The analogue is the *pressure that shows up as
a blocked reach*: the move a person was heading for that the structure did not allow. Today the spec can say
"no one pursued"; it cannot say "three of them were about to and the tie was cut".

**Example:** Sam, 15, has been distancing for two years and the tie with Tom is at cutoff. Each week the
policy's top-ranked move is `CONFLICT` toward Tom, and each week the mask removes it because the tie will
not carry it. The fallback emits `DISTANCE`. Over 40 years the report shows Sam's `DISTANCE` share at 0.71
and says nothing about the 300 weeks the first choice was a fight that could not happen. With MG2 the same
run reports "suppressed `CONFLICT` toward Tom: 300 person-weeks, 58% of the weeks in which Sam had a live tie
to a parent", and the distance in the trace stops reading as indifference.

**Worth saying plainly:** this costs a second pass of the scoring function over moves that have no target,
and "undefined" will be common rather than rare — the fallback rule exists precisely because some scores are
non-finite. It must not consume a random draw: an extra draw would shift the stream and break the per-event
keying. The `M11.D.19` reachability test already enumerates illegal moves, so the machinery is close.

---

## 4. Narrator line (does not enter the v2 spec)

None this week. No candidate from these six papers was judged narrator/LLM-line only; the readers checked the narrator
split and left it empty.

---

## 5. Where these came from, and what was not taken

Each paper's own note: provenance, what the reader deliberately did not propose, and any owner questions.

### From Suggested additions from the ALLY paper, in plain language

Plain-language companion to `_partials_2026-11-23/PART_1_AL.md`. The requirement wording, evidence and spec references are there. None of these is in the spec; each is a proposal for the owner.

The paper simulates one counselling session between a counsellor and a client. The client carries a hidden "alliance state" — intact, strained, ruptured, or repairing — that the counsellor cannot see and must guess from what the client says. When the counsellor pushes too hard and the client pulls back, the counsellor has to notice, repair over several turns, and change the rest of the plan. That is the whole of it: two people, one hidden state on one side, and a repair that is a sequence rather than an event.

The examples use the invented family: grandparents, parents Tom and Ann, and children Sam (15) and Lily (10). They illustrate how a proposal would work. They are not results.

---

### From Suggested enhancements from *On Evaluating User Simulators for Interactive Agents* (Zhou et al. 2026), in plain language

Plain-language companion to `PART_2_US.md`. That file has the requirement wording, module guesses, evidence and
strength. This one says what each proposal would mean in practice.

The paper is about a different object than EPModel: it grades the *metrics* used to score a simulated user
talking to an agent. EPModel has no simulated user, no dialogue, no judge. Nothing here is a mechanism. All five
proposals are test-harness and reporting rules — the same kind of material as the 2026-10-12 stack — and all five
are about the same weakness: a number can look right while the thing it is supposed to measure is wrong.

None is in the spec. Each is a proposal for the owner, held for the next revision.

The examples use the same invented family as the earlier batches: the grandparents, Tom and Ann, Sam (15) and
Lily (10). They show how a proposal would work. They are not results.

---

### Note on what is deliberately not here

No candidate imports the paper's fidelity scores. Every one of them — intent preservation, constraint retention,
persona conformance, action distribution, information provenance — is computed by an LLM judge, and a judge in
the evaluation path is as corrosive to this project as one in the decision path: it is not reproducible, not
auditable by the log, and its own validity is unchecked. The five proposals above are the judge-free residue:
what the paper's *experiment design* says, re-expressed as predicates over `M16` records.

### From Suggested enhancements from one 2026-11-10 paper on personalised affect perception, in plain language

Plain-language companion to `PART_3_AP.md`. That file has the requirement wording, the evidence and the spec references.
None of these five is in the spec. Each is a proposal for the owner, held for the next revision like the earlier batches.

The paper: Keskin, Buzcu, Churamani, Aydoğan, Gunes & Jonker, *Personalized Affective Perception via Continual Learning in
Human-Agent Negotiation* (arXiv 2611.14247v1). A robot negotiates with 66 students, one at a time, for ten minutes each.
The robot reads the student's face as a point on a two-dimensional map (how pleasant, how aroused), and it keeps
re-calibrating that reading to this particular student while the negotiation runs. When the reading shifts, the robot
moves its own bargaining target: a shift it reads as favourable makes it hold firm, an unfavourable one makes it concede.

**What the paper is not.** It is not a simulation and it has no family in it. One robot, one stranger, ten minutes, no
history, no third party, no next week. Nothing in it about state representation, scheduling or seeding transfers. What
transfers is the *shape* of the coupling — a perceived change on a relationship, calibrated to that relationship, moving
the perceiver's own next move — and one honest negative result about how such a gain gets reported.

**The headline numbers, plainly.** The personalising robot got itself about 2 points more of the available value (0.73 →
0.75 on a 0–1 scale) and its partner conceded more often (28% → 37% of offers). The partner's own value did not change
(0.76 → 0.75), and how close the pair came to the best joint deal did not change either. The robot's *own* concession rate
was 38.9% in both conditions — it did not give less; the human gave more. And when the authors tested the story they tell
about why ("the human concedes more, so the robot gets more"), the test came back empty.

The examples use the same invented family as the earlier batches: the grandparents, parents Tom and Ann, and children
Sam (15) and Lily (10). They show how a proposal would work. They are not results.

---

*4. Narrator lines* — None. All five candidates enter the spec proper; none of them is a way of phrasing a run for a reader.

### From Two small changes to how we report results, from the PsyWorld paper

Plain-language companion to `PART_4_PS.md`, which has the requirement wording, the evidence and the spec references. Neither suggestion is in the spec as written. Both are about *how a result is reported*, not about how the family behaves — this paper is a learned model of one person's next move in a task with no other people in it, and nothing in its machinery can be copied into EPModel.

The examples use the same invented family as the earlier batches: the grandparents, parents Tom and Ann, and children Sam (15) and Lily (10). They show how a proposal would work. They are not results.

**What the paper did, in plain words.** It took 313 people who had filled in questionnaires and played a virtual coin-collecting game, and tried to predict each person's next kind of action (work, consume, rest, leisure, move — or, in the game, collect/change room/teleport) from three things: their questionnaire profile, the state they were in, and what they had just done. The clever part was that the training used what *actually happened next* as a teaching signal, but at prediction time only what was known *before* the event could be used. It got the next action right 90.4% of the time — which sounds good until you notice that always guessing the most common action gets 73.9%, and a plain model trained on the same data gets 75.4%. Each person was in exactly one of training, validation or test sets, so every test score comes from someone the model had never seen.

---

### What I did not propose, and why

- **Conditioning on the profile instead of a static description of the person.** Already the design: `M4.D.2` lists what the propensity score may read and `basic_level` is not on the list; `M4.D.1a` routes differentiation through the mixing weight; `M11.1b` already requires a mutant that destroys a state input's link to state and shows the criterion goes red. The paper agrees with the spec here and adds nothing to it.
- **Traits changing rates and thresholds rather than pushing actions around.** The paper does this in its simulator (questionnaire traits scale how fast each need decays and where it trips), and so does the spec, at `M4.C.1a` and `M4.C.14`.
- **Holding people out.** The paper's 240/25/48 split is the strongest thing in it, and it has no analogue here: EPModel fits nothing, so there is nothing to hold out. `M15.D.4`'s envelope over initial-condition ranges is the honest version and is already stronger than a train/test split for this purpose.
- **Comparing distributions as well as averages, breaking a result down by group, requiring a rare move to actually occur, correcting for many comparisons, refusing to claim a difference smaller than its interval.** All already in the spec or in an earlier batch of candidates (`C39`, `C37`, `M11.1f`, `M11.4e`, `M17.A.1`, `M17.G.1`). Listed in the overlap table of the main report so they are not proposed twice.

No narrator-line suggestion came out of this paper. Its language is about latent codes and vector quantisation, and the one line worth having — "the model is a predictive summary, not a mechanism" — is already what `M11.F`'s framing block says.

### From Suggested enhancements from MANGO Arena, in plain language

Plain-language companion to `PART_5_MG.md`. That file has the requirement wording, the evidence and the spec
references. None of these is in the spec as written. Each is a proposal for the owner, held for the next
revision like the earlier batches.

The paper: Sara Amar, *MANGO Arena: Same Seeds, Different Chains* (arXiv 2611.23613v1, 17 Nov 2026). Four
LLM agents each run one company in a row — Retailer, Distributor, Factory, Supplier — for twelve periods,
playing a classroom supply-chain game. The environment blocks and logs any order that breaks a rule, and
gives the four a public message board on which side payments are forbidden.

It is mostly useful to this project for what went wrong in it. The agents amplified customer demand by 7 to
25 times at the Retailer and 13 to 179 times at the Supplier; real rule-breaks were rare; and then two
things broke the evaluation. API failures got counted as "ordered nothing", so a technical fault became a
behaviour and also, wrongly, a rule violation. And when they reran the games with identical seeds and no
failed calls at all, the upstream amplification changed by up to 17 times, which reversed conclusions they
had already drawn. With three seeds, nothing held.

That second finding is worth stating plainly for us: **it does not apply to EPModel's engine.** Our draws
are keyed to the seed and the event, and two runs at one seed produce the same log byte for byte. A seed
here really is the same run twice. Their seed only fixed the customer demand cards; the agents' own choices
were unseeded, which is why "same seeds, different chains" was possible for them and cannot be for us. What
still varies across our seeds is the starting family, and we already report that as its own variance
component.

So three of the four proposals below are about bookkeeping: keeping the count honest, keeping the record,
and not letting a narrow interval stand in for a replicated direction.

---

### Where these came from

- **MG1, MG2, MG3, MG4** — Amar, MANGO Arena (2611.23613v1)
- Adjacent, already proposed: SR2 (Selective Robustness, 2026-11-09) on readout imputation; GP3 (GARP,
  2026-11-16) on amplification control. Neither is the same question; see §5.

---

The examples use the same invented family as the earlier batches: the grandparents, Tom and Ann, and their
children Sam (15) and Lily (10). They show how a proposal would work. They are not results.

---

*4. Narrator line (does not enter the v2 spec)* — None from this paper. Nothing in it is about how a family should be described, and the one line that looked
promising — "the chain amplified what nobody intended" — is about a supply chain's structure, not a family's.

---

### 5. Flagged as already covered / not taken

- **Same-seed rerun divergence.** Covered by design: keyed draws (`M3.D.4a`) and byte-identical logs
  (`M3.D.5`). Their failure mode cannot occur here; no requirement needed.
- **Seed count from observed variance, intervals, spread.** `M17.A.1`, `M17.A.4`, `M17.B.1`, `M17.C.1`.
  Already covered — MG3 is only the confirmatory block they stop short of.
- **Paired per-seed statistic, no unpaired test.** `M11.4e`, `M17.A.3`. Already covered.
- **Which moves were legal.** `M4.D.1e`, `M16.A.3b`. Covered; MG2 is the pre-mask score, which is not.
- **Fallback and tie-break provenance and rate.** `M4.D.1f`, `M16.A.3c`, `M11.D.18`. Partly covers MG1;
  what is missing is the census identity and keeping those rows out of the readouts.
- **Invariant failures recorded before the tick raises.** `M16.A.4`. Covers "log the blocked act" for
  invariants; not needed twice.
- **Refused deliveries logged as their own event class.** `M1.B.3c` in the update-order table. This is the
  delivery half of the paper's "blocked attempt"; MG2 is the selection half, and they should not be merged.
- **Agents see only their own state.** Our analogue is beliefs against truth (`M9.5`, `M16.A.5a`). Covered.
- **Readout imputation disclosure** (SR2, Selective Robustness, 2026-11-09). Adjacent to MG2 and different:
  SR2 bounds a readout built from a clamped quantity; MG2 records a score the mask removed. No overlap worth
  merging.
- **Amplification control** (GP3, GARP, 2026-11-16). Asks whether a mechanism's constant is too small for the
  readout to see. MG3 asks whether a direction survives fresh seeds. Different question.
- **Reliability table beside results, and cost reserved per call.** The first becomes worth having only once
  MG1 exists; the second is an API-budget device with no mechanism content here. Not taken.
- **Human baseline, monitored-vs-unmonitored rules, money-transfer control.** Study designs for their game.
  Nothing to import.

---

### 6. Owner questions

1. **MG3's cost.** Confirming every passing criterion on a fresh block roughly doubles the wall-clock of the
   Phase C/D ensembles. Confirm all four, or only criteria that gate a phase, or only those above a declared
   effect margin? My recommendation: all criteria that a report cites as a finding, which is narrower than
   "all criteria" and wider than "gating only", because the failures are remembered from the write-up.
2. **MG2's vocabulary.** Is "the move the person was heading for and could not make" something you want
   reported in the family-evaluation readout, or is it a trace-level observation only? It has theory content
   — a cut tie is under pressure, not simply absent — but it also invites reading a score as an intention,
   and the model has no intentions.
3. **MG1's fifth class.** Should a deferred I-POSITION step (`M5.D.9`) count as its own census class, or as
   a chosen move with a flag? Its own class is cleaner and slightly noisier.

### From Suggested enhancements from Zhu et al. 2026 (*Probe What-If Worlds Before You Ship*), in plain language

Plain-language companion to `PART_6_PW.md`. That file has the requirement wording, the evidence and the spec references.
None of these four is in the spec. Each is a proposal for the owner, held for the next revision like the earlier batches.

The paper is **Zhu, Hou, Tao, Bo, Fan et al. 2026, arXiv 2611.14700v1** — an industrial pipeline that turns a product
question into a controlled experiment run on simulated users, in twelve partner cases across five domains. Candidate
prefix: **PW**.

It is a paper about LLM personas answering surveys and using websites, and EPModel is a rule engine simulating one family
in which the people are never alone. So almost all of its machinery — cohorts of a thousand, three LLMs compared, trials
that never touch each other — is beside the point here. What survives is narrower and worth having: the paper is a long,
honest demonstration that **the description of the people you simulate is not a background detail**. Take the
description away and the result does not merely weaken; it can turn round. That, and three statistical housekeeping
lessons, are what I have taken.

---

The examples use the same invented family as the earlier batches: the grandparents, parents Tom and Ann, and children
Sam (15) and Lily (10). They show how a proposal would work. They are not results.

---

*3. Narrator line (does not enter the v2 spec)* — **None of the four candidates is narrator-line-only.** The paper writes no prose about people and offers no phrasing for
the readable trace; its transferable content is experimental design and reporting discipline, which belongs in the spec, not
in the narrator. Nothing from this paper is proposed for the LLM/narrator exploration either: the paper's personas are
survey respondents, and its findings about them are about how a model answers a single question, not about a character
holding a personality across decades.

---

### 4. Flagged as already covered / not taken

- **Matched counterfactual personas** — 50 base personas with four variants holding everything else fixed. EPModel already
  does this by hand in `M11.C.1`, `M11.C.16`, `M11.C.41` and `M11.C.49`.
- **Cohort composition strategies and quotas** (reference-data, balanced, oversampled, broad, stress-test; realized
  marginals within 0.1 and 1.4 points) — the analogue is `M17.C.1`'s `D0` and `M15.B.1`'s ranges; nothing new to add
  beyond PW1.
- **Cross-model robustness** — no counterpart; EPModel has one engine. The nearest existing rules are the constant sweep
  and two reference configurations (`M17.E.1`, `M17.E.4`).
- **Repeat variability as a floor for a between-model difference** — covered by `M11.1c`'s representation-invariance
  mutants and `M17.B.2`.
- **Declared success thresholds, honestly missed** (70% acceptable intent, 0.30 points) — `M17.A.4`, `M11.4a`.
- **Trial-level audit and provenance** (persona, rendered inputs, trajectory, outcome, failed trials reported) — `M16.A`
  goes further already.
- **Intervals describe variation within the design only** — `M11.F`'s framing block and `M17.G.1`'s claim grade.
- **Frozen study specification reused after a change** — `M16.A.1`'s config hash, `M17`'s arm specification.
- **Confounds named when arms differ in more than the intended factor** — `M17.D.3` (arms differ only through declared
  channels) and `M11.C.49`'s declared confound.
- **Per-opportunity denominators for move rates** — already proposed as PR1 on 2026-10-12; PW3 is the outcome-side
  extension.

