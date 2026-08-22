---
tags: [model-bt, alignment, resolution]
status: current
date: 2026-08-22
---

# Resolutions — the two live contradictions

Both were carried forward from pass 2 as genuinely unresolved. **Both are now resolved**, and both
resolved the same way: **against the primary source, not against the summaries.** In each case the
contradiction was an artefact of reading a summary — in one case a section *title* — rather than the text.

> **Method note, and it is the point.** Pass 1 over-read the corpus. Pass 2 corrected it by re-reading
> chapters against each other, but still worked from extractions. Both of these contradictions survived
> pass 2 because **the resolving sentence is in the source and in neither summary.** Where a contradiction
> blocks a decision, go to the text.

---

# R1 — Threesome or twosome? **Resolved: neither. Both require three positions.**

## The contradiction as recorded

`L03.4` / `L02.4`: with all three present and the structure held, the family "cannot avoid running into
intense family conflict… This results in high anxiety, action, and progress in therapy. **Any two members
of the family threesome can successfully avoid anxiety issues** and the therapy becomes more intellectual,
more sterile, and less profitable." (Ch02 and Ch03, verbatim, near-identical wording in both.)

`L14.5` item 2: Ch14 abandons the threesome for the twosome — its section is headed **"Family Systems
Therapy with Two People"** — and "this directly opposes L02.4 and L03.4, the group-size finding called
three-times corroborated."

## What the source says

**Ch14's "two people" is a triangle, and Bowen says so in the same section.**

> "The concept about modifying the entire family in **the triangle of the two most important family members
> and the therapist** was well formulated by the mid 1960s."

> "The major changes since the mid 1960s have been in the better understanding of triangles, **clearer
> definition of the therapist's functioning in the triangle**…"

And the mechanics are explicitly anti-fusion, position by position:

> "A typical session might begin with a comment from the husband to the therapist. **To respond directly to
> the husband involves risk in triangling with the husband.** Instead, the therapist asks the wife what she
> was thinking when she heard this."

**Ch03's threesome is the *family* threesome, with the therapist deliberately outside it.**

> "Our first basic rule is that **the family work on its own problem in the hour while the therapist
> observes from the sideline**."

> "There are several ways in which the family attempts to avoid the anxiety. **The most frequent is for the
> decision-making family member to engage the therapist in conversation.**"

Ch02 is explicit that the unit counted is the family: "any two members of the **family** threesome", under
the same condition — "if the family follows the plan of working on its own problem in the hour."

## The resolution

**The group-size finding survives intact. It was never about family members; it is about *live positions*.**

- **Ch03 (1959):** the therapist is structured *out* of the working group. So the working group is the
  family alone, and it needs three of them. With two, the third position is empty and they avoid.
- **Ch14 (1975):** the therapist is the third position, actively held and constantly rotated so he fuses
  with neither spouse. Two family members plus one live third = three positions.

**Both configurations satisfy one rule: three live positions, or the pair avoids.** What changed between
1959 and 1975 is not the mechanism but **who occupies the third position** — and Ch14 says the change was
"a clearer definition of the therapist's functioning in the triangle", which is exactly that.

This also dissolves item 1 of `L14.5` — the "reversal of what the same act means". In Ch03, a family member
talking to the therapist **removes** the third position from the working group, leaving a twosome that can
avoid. In Ch14 the therapist **is** the third position, so talking to him keeps three live. Same act,
opposite meaning, because the third position moved. **That is C10 — an act's identity depends on the
configuration, not the act — applied to therapeutic technique rather than to a family member.**

## What Ch14 actually changed, and why

Not group size. Two separate things, each with its own stated motive:

1. **The child was removed** — because "in the **physical presence of the child**, it was difficult to get
   the parents to focus on themselves", and the parents-plus-child format produced "high praise for family
   therapy but with no basic change in the family problem." That is an **attention/focus** argument and an
   instance of C1 (relief without structural change). It is not a claim about group size.
2. **Communication was routed through the therapist** — "Even when the emotional climate is calm, direct
   communication can increase the emotional tension." That is a **routing** finding (`L14.1`), and it is not
   new in Ch14: Ch10 states it in full in 1971 with the technique "in use more than five years", and Ch11
   dates it to "after about 1962".

**Nowhere does Ch14 claim two is better than three.** It lists three formats — multiple family members, the
two spouses, one family member — without ranking them on group size.

## Consequence for the model

**Avoidance availability is a function of live positions in the working group, not of family members.**

```
positions_live(group) = |{ p in group : p is present AND p is not fused into another member }|
avoidance_available   = positions_live < 3        # a step, not a gradient
```

- An external agent in the outside position **counts as a live position** (Ch14, Ch10 · L10.4 — "would
  probably proceed with any third person… no matter what the subject matter").
- An external agent who has taken a side **does not count** — he has fused into one of the two, and
  `positions_live` drops back to 2. This is the same predicate as R2's ally rule, and it is why test 7
  (position, not skill) and test 3 (triangling) share machinery.
- A member who is present but emotionally inactive **does not count**. Ch17 · `L17.1` names this state
  directly — the displaced member of an interlocking triangle becomes "emotionally inactive".

**Status of the affected ledger entries:** `L02.4` and `L03.4` **stand, restated in terms of positions**.
`L14.5` item 2 is **withdrawn** — it was a contradiction with a section heading, not with the chapter.

## What would falsify this

A passage in which Bowen prefers two live positions to three, or in which a two-family-member session with
an *actively triangulating* therapist is said to avoid the issue the way Ch03's twosome does. Nothing found
in Ch02, Ch03, Ch10, Ch11 or Ch14 says either.

---

# R2 — Does a witness help or displace? **Resolved: alignment is the gate, not knowledge.**

## The contradiction as recorded

`L10.15`: Ch10's differentiating step is **by construction stated aloud**, its case has the spouse present
throughout, and it **succeeds** — which sits badly beside `L21.2`, where Bowen "worked out a plan that
permitted no 'allies'" and detriangled any that appeared.

## What the source says

### Ch21 states the purpose of the ally rule, and it is not secrecy

> "In the past I had done fairly well in detriangling myself from one triangle, **only to have the tension
> slip into another triangle; this pattern had been my undoing.** In preparation for all the potential
> peripheral triangles that could align themselves with issues and prove difficult, I worked out a plan that
> permitted no 'allies' in my effort. In other words, it was an effort to keep the entire family in one big
> emotional clump, and to detriangle any ally who tried to come over **to my side** for this project."

And, on the two deliberately contradictory letters:

> "**The conflicting messages were designed to prevent any one segment of the family from getting on my
> 'side.'**"

**The gated quantity is *taking a side*.** Every statement of the rule uses the language of alignment —
"come over to my side", "getting on my side", "align themselves", "which side was right", "taking sides
with me".

### The ally case is in the chapter, and it is about a position, not about knowledge

His younger sister wrote: "**I am back of you if I can be of help.**"

> "**A red flag** had gone up from her comment about 'I am back of you,' which I handled by telling her that
> I was going to tell the family she had invited me home to help her with her Big Mother role. **She retreated
> from taking sides with me**…"

She already knew everything. Knowing was not the problem. **Offering to stand with him was**, and the
countermeasure returned her to a neutral position — it did not remove her knowledge.

### And there is a witness in Ch21 who knows nothing, takes no position, and the effort succeeds

> "My wife had no direct knowledge of what I was doing… **My wife did not ask a single question nor make
> positive or negative comments about my family at any time during the trip. This had never happened
> before.**"

Bowen singles out her *non-participation* as unprecedented and reports it approvingly, in the account of
his most successful effort. **A witness who takes no position is not an ally and carries no penalty.**

### Ch10 bars alliance in the same terms, from the helper's side

> "The therapist can help the differentiating one when family pressure is great. **He must do this without
> being perceived as against the family.**"

That is Ch21's rule stated from the other direction. And in Ch10's case history the spouse is **the target
of the move and the next mover**, never a supporter: "the process goes back and forth between the spouses in
successive small steps." The person present is the one the position is taken *toward*, and the one the
energy is withdrawn *from* — "which detracts from the former energy devoted to the system, **especially to
the important other**."

## The resolution

**An ally is defined by alignment, not by awareness.** Nothing in the corpus penalises a witness. What is
penalised is a second person taking a position on the mover's behalf, because that **opens a new peripheral
triangle** and the tension slips into it — which Bowen names as his own undoing.

**Secrecy is a risk-reduction heuristic over alignment, not an independent gate.** Inside the system,
knowledge reliably produces alignment because the system relays it: "Messages run back and forth in such a
family system as if by telepathy." That is why the rule is scoped, in the same sentence, to "another person
**who is part of the system**" — and why his wife, outside the family of origin, could be present throughout.

**Three separate objects, three separate rules.** Conflating them is what produced the contradiction:

| Object | Rule | Source |
|---|---|---|
| **The programme** — the overall differentiation effort | Tell no one **inside the system**. Not because knowledge leaks, but because knowledge inside the system produces alignment, and alignment opens a peripheral triangle. | Ch21, Ch22, Ch20 |
| **The act** — a specific move, e.g. a withdrawal | **May be announced, to the party in the dyad**, and Ch10 does so — communicating confidence in the other's competence. | Ch10 · `L10.7` |
| **A third party's position** | **Must be neutral.** A knowing, non-aligning witness is free. A supporter must be actively detriangled. A helper may help only from a position not perceived as against the family. | Ch21 · `L21.2`, Ch10 |

## A third secrecy mechanism, in the source and in neither summary

Pass 2 recorded two reasons for secrecy — sub-deliberative latency ("these decisions and actions often have
to be made instantaneously") and undivided ownership ("for better or worse, the individual has the
responsibility"). **There is a third, stated earlier in the same chapter and generalised:**

> "The second goal was **the element of surprise that is essential if a differentiating step is to be
> successful.**"

`[T]` Ch21. This is a claim about differentiating steps in general, not about the conference presentation
that occasions it. It is a genuine third mechanism and it attaches to **the target**, not to third parties —
which is exactly why it does not conflict with Ch10's announced act: announcing *what you are about to do*
in one dyad does not forfeit surprise about *the programme*.

**New ledger entry: `L21.11` — surprise as a third secrecy mechanism.**

## Consequence for the model

Replace the secrecy flag with an **alignment** predicate.

```
# a third party's effect on a differentiating move
if third.position == NEUTRAL:            effect = none          # Ch21's wife
if third.position == ALIGNED_WITH_MOVER: open_peripheral_triangle(third, mover)
                                         # tension displaces into it; the move's
                                         # gain leaks until the ally is detriangled
if third.role == EXTERNAL and perceived_against_family(third):
                                         open_peripheral_triangle(third, mover)

# knowledge is not a gate; it is a hazard rate on alignment
P(align | knows, inside_system)  = high      # "as if by telepathy"
P(align | knows, outside_system) = low       # the wife
```

Two moves follow that the nine-move repertoire does not have:

- **`DETRIANGLE`** — reactive. Push an aligned third party back to neutral. Already in the explainer §5.7;
  R2 supplies its trigger predicate and its success condition (the ally *retreats from taking sides*, and
  keeps their knowledge).
- **`PREVENT_ALIGNMENT`** — preemptive, and previously unmodelled. Ch21's two deliberately contradictory
  letters, sent in the same mail within the hour, exist for exactly this and Bowen states the purpose
  outright. The model has no move that acts on *potential* alignment before it forms.

**Status of the affected ledger entries:** `L10.15` is **resolved** and becomes a restatement of the
alignment rule. `L21.2`'s ally gate **stands, restated** as alignment rather than support. `L21.3` was
already settled at the object level and is unaffected. `L22.3` remains closed.

## What would falsify this

A case in which a third party who **took no position** nonetheless degraded a differentiating move, or one
in which an openly aligned supporter helped it. Ch21 contains the opposite of both: the non-participating
wife (success) and the "I am back of you" sister (red flag, actively neutralised).

The weaker point is the scope of "inside the system". Bowen's wife is treated as outside his family of
origin, but he still withheld the plan from her, and he states the absolute broadly before narrowing it in
the next sentence. **Implement the narrow reading — the rule as he scopes it — and treat the broad
statement as rhetorical emphasis**, which is consistent with how he restates it a third time, hedged: "it is
doubtful that any differentiation will result."

---

# What both resolutions have in common

The same mechanism sits under both, and it is the model's own:

> **A position only counts as live if its occupant is not fused into another.**

In R1 that predicate decides whether a group of three can avoid its issue. In R2 the same predicate decides
whether a third party is a witness or a peripheral triangle. **The coach's neutrality, a family member's
self-control, an ally's detriangling and the three-position floor are one rule seen from four angles** —
which is what Ch10 says directly: "the therapist's neutrality and a family member's self-control are the
same operation from different positions."

Implement it once.
