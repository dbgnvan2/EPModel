# Suggested enhancements from three early method papers, in plain language

Plain-language companion to `SWEEP_READING_REPORTS_2026-10-05.md`. That file has the requirement wording, the evidence
and the spec references. None of these is in the spec. Each is a proposal for the owner, held for revision 11 like the
earlier batches.

The three papers: Redish 2004 (addiction as a learning process that goes wrong), Park et al. 2023 (Generative Agents,
"Smallville"), and Argyle et al. 2022 (Out of One, Many: using a language model to stand in for survey respondents).

The examples use the same invented family as the earlier batches: the grandparents, parents Tom and Ann, and children
Sam (15) and Lily (10). They show how a proposal would work. They are not results.

---

## 1. Gaps in the spec text (need an owner decision first)

### TD2. When a learned pattern stops, is it erased or only held down? *(from Redish)*
**What it proposes:** When a reactive pattern fades, after a crisis ends or after coaching lands, the model should not
simply decay the learned weight back to zero. The weight should stay, held down, so that under new load the pattern
comes back faster than it was first learned. Test: two arms. One learns a pattern in a first crisis, the pattern fades,
then a second crisis arrives. The other has only the second crisis. The first arm must fall back into the pattern sooner.
**Owner question:** Does the theory say a family's old pattern is still there after it goes quiet? The nearest support is
Kerr's account of regression as an older system re-exposed (KS23.2). That is Kerr's extension, not Bowen.
**Why:** Without this, a coached arm's gains would count as permanent unlearning.
**Example:** In Sam's first year of high school Ann pursued him hard every evening. By the next summer it had stopped. In
the spring of his last year, when he is failing a course, does Ann's pursuing come back within a week, or does it have to
be learned again from scratch?

### GA1. How long does a belief last, and on which clock? *(from Generative Agents; adds to held question PM1)*
**What it proposes:** PM1 already asks how long a belief lasts. This adds two points. First, fading could run on the time
since the belief was **written**, or on the time since it was last **used**. Under the second, beliefs that drive
behaviour keep themselves alive and unused ones fade. Second, each belief's persistence must be computed from that belief
alone, never relative to the rest of the store. If any fading is adopted, any claim that emotional process distorted a
memory must be compared against a run where only plain fading operates, with anxiety switched off.
**Owner question:** Should a belief fade with time since it was formed, or with time since it was last acted on?
**Why:** The paper shows that plain fading alone produces gaps and fragments. The corpus's "the family system operates
to obscure and misremember" (L09.4) describes something active. The two must be told apart.
**Example:** Lily believes her father might leave. If she checks that belief every time Tom is late, under "time since
last used" it never fades. Under "time since written" it fades whatever she does.

### GA2. Are some beliefs built from other beliefs, and do they outlast their evidence? *(from Generative Agents)*
**What it proposes:** If any belief is derived from other beliefs or from accumulated events rather than written by one
event (for example, who in the family "is the problem"), the spec must say three things: what it is built from, what
triggers the rebuild (the slow clock, or enough upsetting events piling up), and whether it is revised when its sources
fade. Close to RZ1 (are beliefs linked); could be folded into it.
**Owner question:** Does a family's conclusion about who is the problem stand after the events that produced it are
forgotten?
**Why:** A clock trigger gives a crisis year and a calm year one update each. A load trigger gives the crisis year more.
That is a theory choice either way.
**Example:** The family concludes in Sam's 15th year that "Sam is the problem". Ten years on, nobody remembers the
specific fights. Does the conclusion still stand?

---

## 2. Tests

### TD1. State how learning works, and make sure a learned habit cannot grow without limit *(from Redish)*
**What it proposes:** The spec says moves that worked are reinforced (`M4.D.6`) and fixes the horizon, but not the form
of the update. Declare one of two forms: either reinforcement is the gap between what happened and what was expected (a
fully expected result stops reinforcing), or reinforcement adds up but with a declared ceiling. Test: give one move a
constant reward and check its learned weight levels off.
**Why:** In Redish's model the update form alone decides between a finite habit and runaway growth. The spec's argument
that a short-horizon learner "converges on CUTOFF by construction" assumes the additive form without saying so. The
expectation form may also make the separate habituation term (`M4.G.3`) unnecessary.
**Example:** Every time Tom withdraws to the garage his tension drops. Under the additive form his garage habit keeps
growing for 40 years. Under the expectation form it settles once the relief is fully expected.

### TD3. Check the horizon result under two discount shapes *(from Redish)*
**What it proposes:** Declare not only how far ahead the learner looks but how it weights the future: a steady decline
(exponential) or steep-then-flat (hyperbolic, which a mix of exponentials produces). `M11.C.16`'s result must hold under
both before it is relied on.
**Why:** The relief-now, cost-later timing that `M4.D.6a` relies on is exactly the case where discount shape can flip a
preference.
**Example:** Ann soothes Sam now and pays later. Under one discount shape she always picks the soothing; under the other
she might not. The test should give the same direction under both.

### OM1. Do the associations across many simulated families have the signs the corpus states? *(from Out of One, Many)*
**What it proposes:** Across a large ensemble of starting families, compute whether pairs of readouts rise or fall
together (for example, founders' level and symptom load), both pooled and within each level band. Compare each sign
with the corpus where the corpus states one. Report matches and misses together. Pairs the corpus says are not
one-directional are left unscored. The list of corpus signs lives in config, and it is a check only, never used to set
constants.
**Why:** Every existing acceptance test compares two arms. A model can pass all of them and still produce patterns
across families that the corpus contradicts.
**Example:** Across 1,000 simulated families, lower-level founders should go with more symptoms in the third
generation. If the pooled sign is right but it reverses inside each level band, that is flagged.

---

## 3. Reporting rules

### GA3. Every belief update names the events that caused it *(from Generative Agents)*
**What it proposes:** Each logged belief write carries the IDs of the events it was computed from. A test checks every
cited event was actually delivered to that person, as target or witness, before the write.
**Why:** Revision 10 lists "belief records citing event IDs" as noted but not specified, for want of a testable form.
This gives it one, and lets the trace renderer show the path from an event to a belief.
**Example:** Lily's belief that Tom might leave is logged with the ID of the argument she overheard in week 312. If no
such event reached her, the test fails.

### GA4. Say when a switched-off mechanism was switched off *(from Generative Agents)*
**What it proposes:** Every arm that disables a mechanism must say whether it was off from the start or switched off
partway through a history built with it on. A partway result must not be reported as the mechanism's total effect.
**Why:** The paper's ablations removed components only at question time, on a history the full system had built, and
called the result conservative without testing that.
**Example:** Turning off triangling in year 20, after 20 years of triangles shaped the family, measures something
different from a family that never triangled.

### OM2. Did importing a real family's diagram change the result at all? *(from Out of One, Many)*
**What it proposes:** A result on an imported family is reported beside the same arms on the generic reference family,
and re-run with each imported part (structure, tie states, dated events, ratings) replaced in turn by "no information".
If the imported result does not differ from the generic one, the report must say so and must not present the result as
being about that family's structure.
**Why:** The current import rules ask whether a direction survives the imported ranges. Nothing asks whether the import
mattered. A result identical to the generic family says nothing about this family.
**Example:** A coach arm on an imported diagram shows less cutoff. The generic family shows the same. The report says
the diagram did not change the result.

### OM3. Human reviewers judge blind *(from Out of One, Many)*
**What it proposes:** Where the spec assigns a criterion to human review (for example `M11.C.11`'s three-phase curve),
the reviewer first sees both arms' traces with labels, parameters and seeds removed, and records which arm is which
before the labels are revealed. A review that cannot tell the arms apart better than chance is reported as not
supporting the criterion.
**Why:** Today's human reviews are unblinded. A reviewer who knows which arm should show the effect can confirm it.
**Example:** The reviewer gets two unlabelled plots of the family after the grandmother's death and must say which is
the arm with a cutoff before being told.

---

## 4. Narrator line only (does not enter the v2 spec)

- **GA-X1.** Human ratings of narration must score whether the reader can recover the engine's state, not which
  rendering they prefer. In the paper, human-written answers ranked below two cut-down AI versions.
- **GA-X2.** Count narrated claims that are absent from the log separately from claims that contradict it, and count
  outside world knowledge leaking in as its own category.
- **OM-X1.** An LLM persona population must be tested for each attribute it was given, including rare categories, at two
  or more sampling temperatures. In the paper, at near-greedy sampling every respondent came out White, and the sign of
  the association error depended on temperature.

---

## Owner questions raised by this batch

1. **TD2.** Is an old pattern held down or erased when it goes quiet?
2. **GA1.** Does belief fading run on time since written, or time since last used? (Adds to PM1.)
3. **GA2.** Are some beliefs derived, and does a derived belief outlast its evidence? (Close to RZ1.)
