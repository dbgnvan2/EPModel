# Proposal — nodal events, cutoff read from behaviour, and C.17

> **Status: approved by the owner, 2026-10-11 (§1, §2, §3); applied at spec revision 17.** Drafted 2026-10-11. It follows the owner's answers of 2026-10-11 on B4–B6. Each item is
> `[I]` where it goes beyond the owner's words. Nothing here is built until approved. §1 changes the estimator's
> breadth. §2 changes the `cutoff` pattern readout. §3 rewrites `M11.C.17` (spec revision 17).

## 1. What a nodal event is

**The owner, 2026-10-11:** "Death, births, marriages, serious illnesses, big moves, big job changes, child moving out or
back in, divorces are 'nodal' events. Any major event on a person (a node) is a nodal event."

**Proposed:**
- **Where the list lives.** In config, `config/bowen/nodal_events.md`, as a list of event kinds. The estimator reads
  it. Adding a kind is an editorial change, not a code change.
- **Kinds the engine has now (D-I):**

  | Event kind | What it is |
  |---|---|
  | `JOB_LOSS` | a big job change |
  | `SYMPTOM_ONSET` | a serious illness: a symptom crossing its threshold (`M7.D.1`) |
  | `INSTITUTIONALIZE` | a member moving out of the household |
  | `BINDER_UNAVAILABLE` | losing what a person was bound to |

- **New kinds, so scenarios can carry several nodal kinds in D-I:**

  | Event kind | Mechanism |
  |---|---|
  | `MOVE` | exogenous stressor |
  | `SERIOUS_ILLNESS` | exogenous stressor: an illness from outside the symptom model |
  | `JOB_CHANGE` | exogenous stressor |

- **Added with the life course in D-II:** `BIRTH`, `DEATH`, `MARRIAGE`, `DIVORCE`, `LAUNCH` (a child moving out) and
  `RETURN` (a child moving back in).
- **Whose event it is.** A nodal event counts for the person it happens to, and for everyone it reaches: its targets
  and its witnesses. A death in the family is nodal for the family.
- **Breadth** stays the number of distinct kinds in the window. A job loss every two weeks is one situation.

## 2. Cutoff read from behaviour

**The owner, 2026-10-11:** "Cutoff is from the agent's behaviour, not just the engine. The designation of cutoff
emerges from behaviour. An agent saying 'I'm not going to contact you again' is not cutoff; it's just words that need
behaviour to back it up."

**Today the readout counts the words.** It counts `CUTOFF` acts sent in reply to an approach. Three such acts in four
weeks read as cutoff, whatever the person then does.

**Proposed readout, for one member A toward B, over the pattern window W (4 weeks):**
- **A refuses B.** B approached A at least `pattern_min_acts` (3) times. Each approach was refused: either the engine
  refused it (`M1.B.3c`, A's rejection was held), or A answered it with `CUTOFF` or `DISTANCE` within a week.
- **A does not approach B.** A sends B no act with contact in the window.

Reading of the result:
- **Both hold:** A's cutoff of B.
- **A `CUTOFF` act with neither behaviour:** reads as nothing; it is words.
- **No approaches either way:** distance, as now.

**Effect:** the readout reads what A does and does not do. The engine's `CUTOFF` act is one of the things A does, not
the designation. No criterion reads this readout. It changes what the fam-diag export draws.

## 3. `M11.C.17` rewritten: a coach helps, and is not required

**The owner, 2026-10-11:** "A person CAN improve functioning and level of differentiation without a coach. A coach can
help, but isn't required. The evidence is in the behaviour."

**Today:**
- **The criterion:** "Differentiation is not self-generating … the coach-free arm MUST NOT show upward drift in mean
  `basic_level`." The source is Kerr's "levels roughly conserved and differentiation is rare".
- **The conflict:** the owner's statement contradicts the "MUST NOT".

**Proposed `M11.C.17`:** *Differentiation is earned in behaviour; a coach helps and is not required.*
- **Arms.** Two arms on identical seeds, differing only in whether Dr Halim is in contact. Both carry nodal load of at
  least `estimator_breadth` kinds in every window.
- **Direction.** The coached arm's mean rise in `basic_level` MUST be larger than the coach-free arm's.
- **Coach-free rises.** The coach-free arm MAY rise. Every coach-free rise MUST be carried by the person's own
  self-directed behaviour under load. A person with no self-directed share under load (`M1.A.4h`) MUST NOT rise.
- **The relay.** Kept as a reported readout, not a gate: "leadership shifts back and forth" (FE11.6).
- **Mutation.** Remove the coach's effect (`M1.E.7`'s landed contact). The arms are then alike, and the criterion turns
  red.
- **What it replaces.** The two amendments (flat against `M7.E.4`'s deterioration; the relay) are folded in as above.
  The claim that a coach-free family stays flat is withdrawn.

**One consequence.** Kerr's "differentiation is rare" survives as a property the runs show, not a rule. Rises need
sustained self-directed behaviour under broad load, which the policy draws rarely at low levels (`M4.D.1a`). If runs
show rises that are not rare, that is a finding to report, not something to tune.

## What the owner decides

Approve, correct or reject §1, §2 and §3.
