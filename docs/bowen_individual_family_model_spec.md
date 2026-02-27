# EPModel — Bowen Family Systems Theory Spec
## Individual & Family Level Models

**Version:** 1.2
**Date:** 2026-02-26
**Source theory:** Murray Bowen, *Family Therapy in Clinical Practice*, Chapter 21 — "On the Differentiation of Self" (1970/1978)

---

## 1. Overview and Design Philosophy

This document specifies how Bowen Family Systems Theory maps onto the EPModel individual unit and family-level simulation. The goal is for every variable and every dynamic in the engine to have a theoretically grounded Bowen interpretation — not just a mechanical one.

Bowen's core insight: **the family is an emotional unit, not a collection of individuals.** Symptoms, stress, and recovery do not belong to individuals alone — they circulate through the system according to each person's level of differentiation, their current reactivity, and the structural patterns (triangles, projection, fusion) that the family has developed over time.

Emotion propagates like a contagion through the grid. The question is not *whether* it propagates, but *how much each person amplifies or dampens it* — and that is determined entirely by their differentiation level (C) and their current reactivity (R).

---

## 2. Variable Mapping — Current State Array to Bowen Concepts

The engine state array `state[N, 4]` maps to Bowen constructs as follows.

Generational-flux target: keep the four core columns (`S, TX, C, M`) in `state` and keep identity/lifecycle fields (`age`, family IDs, slot lifecycle status) in parallel arrays for performance and clarity.

| Column | Engine Name | Bowen Concept | Bowen Definition |
|--------|-------------|---------------|-----------------|
| `state[:, 0]` | **S** — Stress | **Stressors** | External and internal pressures on the unit: economic hardship, environmental threats, relationship conflict, unemployment, climate strain. Stressors drive reactivity up. They vary widely across units and over time. |
| `state[:, 1]` | **TX** — Transmission Index | **Reactivity (R)** | The unit's current level of emotional reactivity — how activated the emotional system is right now. High R means the unit is both *more susceptible* to incoming stress contagion and *more contagious* to neighbors. R accumulates when stressors exceed the differentiation-adjusted baseline; it decays naturally during calm periods. Capped at `TX_MAX = 4.0` to prevent runaway amplification. |
| `state[:, 2]` | **C** — Capacity | **Differentiation of Self** | The basic, stable level of self-differentiation on Bowen's 0–100 scale. Higher C = clearer separation between thinking and feeling, greater autonomy in relationships, less susceptibility to emotional fusion. C changes slowly across the life course. It is the primary modulator of how much stress one absorbs (as receiver) and how much one broadcasts (as sender). |
| `state[:, 3]` | **M** — Metabolic Resources | **Functional Resources** | The material and energetic capacity to sustain life functioning: income, savings, physical health reserves. M underpins the ability to maintain functioning under stress. When M → 0, the unit can no longer sustain itself; metabolic death occurs. |

### Additional Arrays (Beyond the Core State)

| Array | Bowen Concept | Description |
|-------|--------------|-------------|
| `age[N]` | **Life Stage** | Units age 1 year per cycle. Stage governs childhood (1–17), launch (18–25), partnering (22–40), childbearing (22–45), and Gompertz mortality. Chronic Anxiety (CA) is fixed at age 10. |
| `family_origin_id[N]` | **Family of Origin (FOO)** | Immutable lineage identifier assigned at birth. Never changes after initialization/birth. Used for multigenerational analysis. |
| `nuclear_family_id[N]` | **Current Nuclear Family** | Mutable current family membership. Changes through marriage/divorce/remarriage. Same-nuclear-family friction = 0 in contagion (Axiom 4). |
| `family_ids[N]` | **Compatibility Alias** | Backward-compatible alias to `nuclear_family_id` during migration from the current engine. |
| `slot_status[N]` | **Lifecycle Slot State** | `ACTIVE`, `DEPARTED`, `DEAD`, `INACTIVE_BUFFER`. `INACTIVE_BUFFER` starts as nursery capacity (`M = -1`) and can be activated by births. |
| `family_c_diff[F]` | **C-Level Mismatch at Marriage** | Absolute C-diff % between spouses at the time of partnering. Drives elevated divorce rate and chronic S-raise (see §7). |
| `family_divorce_rate_mod[F]` | **Fusion Penalty — Divorce Probability** | Extra per-cycle divorce probability from C-mismatch. Mismatched partners have a fused pseudo-self arrangement that is unstable under stress. |
| `family_s_penalty[F]` | **Fusion Penalty — Chronic Stress Raise** | Per-cycle S baseline raise applied to both members of a C-mismatched pair. Models the continuous emotional cost of living in an incompatible fusion. |
| `recovery_timers[N]` | **Post-Ejection Relief** | 20-cycle recovery window after a family member departs. Maps to the short-term relief the family system experiences after the most anxious person exits. |

---

## 3. The Differentiation of Self Scale (C)

### 3.1 Theoretical Basis

Bowen's differentiation scale runs from 0 (total emotional fusion, no autonomous self) to 100 (complete separation of thinking from feeling — theoretical maximum, never reached in practice). The engine maps this directly to `C ∈ [10, 80]`. 10 to 80 is practical limit.

Key Bowen properties of C:

- **Slow change:** Basic differentiation is acquired in the family of origin during childhood and is largely consolidated by adulthood. It changes very slowly — years, not weeks.
- **Multigenerational transmission:** Children emerge from the family projection process with approximately the same, slightly higher, or slightly lower C than their parents. The most emotionally over-involved child (projection target) receives lower C; children who remain less involved may emerge with higher C.
- **Basic Self vs. Pseudo-Self:** The engine's C represents *basic self* — the stable, non-negotiable core. Pseudo-self (borrowed beliefs and positions traded in relationships) is captured implicitly by the functional fluctuations in R (TX).
- **Protective effect:** Higher C units absorb less incoming stress (`/ clip(C, 10)` in contagion), broadcast less stress per unit of TX, and recover faster when conditions improve.

### 3.2 Engine Implementation of C (Target Specification)

**Target initialization:**
```python
self.state[:, 2] = 40.0 + np.random.uniform(-5, 5, num_units)
```
Population starts around the mid-scale (40) with ±5 scatter — representing a typical cross-section of society where most people function in the mid-range.

**Update rule (`update_c`):**
- When S < adjusted baseline: C drifts back toward its baseline value (`c_baseline`), at 50% of the gap per cycle. Represents the natural re-solidification of self during calm.
- When L (Leadership) > 1.2: all units gain +0.05 C/cycle — strong societal leadership creates conditions that support differentiation across the population.
- C is clamped to `[10, 80]` in the target implementation.

**C and coaching:** During active coaching, C grows at +0.5 per year (cycle) for adults. As C increases, R (Reactivity) drops by the same amount — a 1:1 inverse relationship during coaching. See §4.1 and §9.

### 3.3 C-Level Inheritance (Multigenerational Transmission)

When a birth occurs in a family, the newborn's C is:

```
child_C = avg(parent_C) × uniform(0.9, 1.1)   clipped to [10, 80] (target)
```

This implements Bowen's multigenerational transmission: most children emerge near the parental average, with natural ±10% variation. Some drift higher (will function better), some drift lower. Across generations with the family projection process applied, the most over-involved child consistently draws below the parental mean.

**Future extension (projection process):** In families with 3+ members, one child could be designated the *projection target* and receive `child_C = avg(parent_C) × uniform(0.7, 0.9)` — a systematically lower C from being the focal point of parental anxiety.

---

## 4. Reactivity (R) — the TX Column

### 4.1 Theoretical Basis

Bowen distinguished between:

- **Acute Anxiety:** Triggered by a real, present threat (job loss, illness, divorce shock, environmental disaster). Rises fast, falls when the threat resolves.
- **Chronic Anxiety (CA):** Persistent, background-level anxiety carried from the family of origin. Present in all units regardless of current stressors. Increases with accumulating stressors over time. The key Bowen insight: *chronic anxiety is more damaging than acute anxiety because it never resolves*.

In the engine, **TX (Reactivity / R)** captures both:
- The acute component rises when `S > S_BASELINE / L` (current stressors exceed the leadership-adjusted tolerance threshold).
- The chronic component is the floor from which R never fully drops, implicitly maintained by the `S_BASELINE` floor on S and the non-zero starting value of TX.
- **Coaching inverse:** When C increases through coaching (+0.5/cycle for adults), R decreases by the same amount (−0.5/cycle) as a direct 1:1 consequence — in addition to the coached R decay applied to all live units. See §9.2 for the combined effect.

### 4.2 Engine Formula

```python
delta_R = ((S - S_baseline) * gamma) / clip(C, 10)
R += delta_R
R = clip(R, 0.0, TX_MAX)   # TX_MAX = 4.0
```

**Interpretation:**
- When stressors (S) exceed the differentiation-adjusted baseline, R rises.
- Higher C suppresses R accumulation — differentiated people do not become as reactive under the same stressor load.
- The cap at 4.0 prevents the runaway amplification Bowen described as "acute anxiety spiraling into system-wide crisis."

### 4.3 Chronic Anxiety at Age 10

**Theoretical basis:** Bowen held that chronic anxiety level is set in childhood within the family of origin — by age 10, the child has absorbed a characteristic baseline anxiety level from the family emotional atmosphere.

**Model specification:**

At the cycle when a unit's age crosses 10, its chronic anxiety baseline is fixed as:

```
CA_unit = avg(S of all live family members) * 0.3
        + avg(R of all live family members) * 0.2
```

This value is stored in a new per-unit array `chronic_anxiety[N]` and serves as a *floor on R*: even when stressors drop, R cannot fall below `CA_unit / C_scale_factor`. Units that grew up in high-stress, high-reactivity families carry a permanently higher floor — they are chronically more anxious regardless of present circumstances.

**Implementation note:** `chronic_anxiety` is a new `np.float32` array initialized at birth to `S_BASELINE * 0.3` and updated once when `age` crosses 10. The existing `update_tx()` method would clamp R from below at `chronic_anxiety[unit] / 50.0` after the unit's CA age is passed.

---

## 5. Stressors (S) and the Stress-Reactivity Loop

### 5.1 Theoretical Basis

S represents the total stressor load on a unit at any given cycle. Stressors include:

| Stressor Source | Engine Mechanism |
|-----------------|------------------|
| Economic hardship (low M) | M drain → ejection when M < 150 |
| Unemployment | `unemployment_rate` → +10 S per cycle when unemployed |
| Relationship conflict (C-mismatch) | `family_s_penalty` → per-cycle S raise |
| Environmental pressure | `E` multiplier on metabolic drain |
| Social contagion from others | `update_contagion()` → neighbor S propagation |
| Accumulated reactivity feedback | R amplifies received contagion |
| Shock events | Keyboard events: SPACE (anxiety shock), F (global spike) |

### 5.2 The S → R → S Feedback Loop

This is the central Bowen feedback mechanism. It happens most intensely within the family unit, but some spills into society through the spatial contagion grid:

```
High S  →  R rises  →  more contagious to neighbors
                    →  more susceptible to incoming stress
                    →  neighbors' S rises
                    →  their R rises  →  broadcast back
```

C is the **damper** on this loop. High-C units:
- Accumulate R more slowly for a given S level.
- Broadcast less stress per unit of R (`/ clip(C, 10)` in contagion formula).
- Recover R faster during calm periods.

Low-C units amplify the loop — they are the system's "anxiety carriers."

### 5.3 S Baseline and Equilibrium

The stress floor is maintained by:
```python
s_floor = S_BASELINE * E
state[:, 0] = clip(state[:, 0], s_floor, 500)
```

In Bowen terms, `S_BASELINE` represents the chronic background level of societal anxiety. No one in the model functions below this level — it is the ambient social anxiety of the era. **Exception:** individuals with C > 55 subtract 1 point from their personal S floor for every C point over 55. A unit with C = 80 reduces its floor by 25 points (e.g., floor drops from 80 to 55 at E=1.0). This captures Bowen's observation that highly differentiated people genuinely operate at a lower baseline level of anxiety than the surrounding population.

Elevated E (climate/environment pressure) raises this floor, making baseline functioning harder for everyone.

**Implementation note:** This requires a per-unit floor array rather than the current scalar clip. The implementation is:
```python
s_floor_scalar = S_BASELINE * E
per_unit_floor = np.full(num_units, s_floor_scalar)
high_c_mask = state[:, 2] > 55
per_unit_floor[high_c_mask] -= (state[high_c_mask, 2] - 55)
per_unit_floor = np.maximum(per_unit_floor, s_floor_scalar * 0.5)  # cap reduction at 50%
state[:, 0] = np.maximum(state[:, 0], per_unit_floor)
```

### 5.4 Update Order (Generational-Flux Target)

To prioritize family dynamics before societal contagion, `update()` uses this order:

1. `apply_income()`  
   - Grant `base_income` to all `ACTIVE` units.  
   - Apply autonomy multiplier only to circles (`STATUS_DEPARTED`): `income *= 1.5 * (C / 100)`.
2. `apply_aging()`  
   - Increment `age += 1` for `ACTIVE` units.  
   - Keep Gompertz-Makeham mortality law (no hard age cutoff override).
3. `apply_matchmaking()`  
   - Run every `MATCH_INTERVAL` cycles (default 20; optional stress-test profile 10).  
   - Two-pass matching: primary `<= 5%` C-diff, secondary `> 5%` with compatibility penalties.
4. `apply_reproduction()`  
   - Eligible size-2 families receive per-cycle birth chance (configurable; baseline profile keeps current 8%).  
   - Birth activation uses first available slot from `INACTIVE_BUFFER`, else `DEAD`, else `DEPARTED`.
5. `apply_launching()`  
   - Launch window is age-based and data-calibrated (see §11.1): baseline `LAUNCH_BASE_P = 0.07` with stress multipliers.
6. `calculate_stress()`  
   - Run contagion / TX / C / M passes on the current living population.

This order preserves UI and class boundaries while supporting dynamic generational turnover.

---

## 6. Functional Differentiation (FD)

### 6.1 Theoretical Basis

Bowen distinguished between:
- **Basic Differentiation (C):** Stable, slow-changing. The long-term structural level of self.
- **Functional Level of Differentiation (FD):** How someone is actually functioning *right now*, given their circumstances. FD can be considerably higher or lower than basic C depending on:
  - Current stressor load (S)
  - Relationship support (healthy family = functioning up)
  - Fusion dynamics (dominant partner borrows self; adaptive partner loses self)
  - Societal conditions (L, E, X)

### 6.2 Model Specification

FD is a **derived quantity** (not stored — computed on demand):

```
FD(unit) = C × adjustment_factor

where adjustment_factor = clip(
    (S_BASELINE / L) / max(S, 1),   # high S depresses FD below C
    0.3,                             # floor: even under max stress, FD ≥ 30% of C
    1.5                              # ceiling: good conditions can lift FD 50% above C
)
```

**Interpretation:**
- When S = S_BASELINE/L (exactly at threshold): FD = C (functioning at basic level).
- When S >> baseline (under crisis): FD << C (functioning far below capacity).
- When S << baseline (thriving conditions, supportive family): FD > C (functioning above basic level — "borrowed self" from positive environment).

**Use cases in the engine:**
- FD determines the effective C used in contagion damping for the current cycle.
- FD governs the autonomy income bonus for circles: `income × 1.5 × (FD / 100)` rather than raw C.
- Coaching raises FD first (immediate effect) then C over years (structural effect).

---

## 7. Nuclear Family Emotional System

### 7.1 Theoretical Basis

Bowen identified that undifferentiation in a couple is absorbed via four mechanisms:
1. **Marital conflict** — neither partner gives in; tension stays between the pair.
2. **Spouse dysfunction** — one partner (the adaptive one) absorbs the undifferentiation; they "give up self" and become functionally lower while the dominant partner appears stronger.
3. **Child projection** — the couple's combined anxiety is focused onto one or more children, lowering that child's effective C.
4. **Distance** — one or both partners become emotionally withdrawn. The relationship remains intact but is superficially calm; partners avoid emotional contact to prevent triggering reactivity. Unlike cutoff (divorce/departure), distance keeps the pair together while reducing genuine emotional exchange.

### 7.2 Engine Implementation

**Mechanism 1 — Marital Conflict** (currently implemented):
- Families with 2 active members where avg S > 160 for 50+ cycles face divorce risk (cutoff).
- C-mismatch increases this rate: `eff_divorce_rate = 0.01 + family_divorce_rate_mod[fid]`.
- This models the conflict that arises when neither partner can adapt to the other's differentiation level, ultimately resulting in rupture.

**Mechanism 4 — Emotional Distance** (specified for future implementation):
- Two same-family embedded units whose individual R has been below 0.5 for 20+ consecutive cycles are flagged as "distanced."
- A `family_distance_flag[F]` bool array marks these families.
- Distanced families apply a cross-member contagion multiplier of 0.1 (near-zero emotional exchange despite zero friction), modeling the hollowed-out marriage that stays legally intact.
- Distanced families have a reduced divorce rate (0.005 base) but elevated child projection probability — the unresolved anxiety routes to children instead.

**Mechanism 2 — Spouse Dysfunction** (currently partially implemented):
- The `family_s_penalty` applies a continuous S raise to both members of a C-mismatched pair.
- Future enhancement: designate the lower-C partner as "adaptive" — they receive a larger S penalty while the higher-C partner receives a smaller one, modeling the asymmetric borrowing of pseudo-self.

**Mechanism 3 — Child Projection Process** (specified for future implementation):
- When a birth occurs in a family where at least one parent has elevated R (R > 2.0), the newborn is a *projection target*.
- Projection target child C = `avg(parent_C) × uniform(0.7, 0.9)` (systematically lower).
- Projection target child birth_stress = `family_avg_S × 0.8` (inherits parental anxiety directly).
- At most one child per family per generation is the primary projection target.
- The target is the child born during the family's highest-stress period.
- The target child designation can shift as more children are born and relative stress levels change.

### 7.3 Fusion in Marriage — C-Proximity Matching

Bowen observed that people choose partners at approximately their own differentiation level. The engine enforces this via two-pass matchmaking:

**Primary match (≤5% C-diff):** Partners are close in differentiation. Minimal fusion penalty. Stable marriage.

**Secondary match (>5% C-diff):** Partners differ significantly. The pseudo-selves fuse under emotional pressure. The engine records:
- `family_c_diff[fid]` — the mismatch magnitude.
- `family_divorce_rate_mod[fid]` — elevated instability (+2% per 1% of excess C-diff).
- `family_s_penalty[fid]` — chronic stress raise (+1.0 per 1% of excess C-diff, cap 50).

The S-penalty models what Bowen called "the cost of maintaining the fusion" — the chronic low-grade stress that pervades a household where two people of different differentiation levels have merged pseudo-selves.

### 7.4 Divorce, Remarriage, and Identity Continuity

Identity behavior is explicit:

- `family_origin_id` is immutable for life.
- On marriage, both partners receive a new `nuclear_family_id` (new family record).
- On divorce, each partner becomes a departed circle and exits that nuclear family (`nuclear_family_id = -1` until re-partnering).
- On remarriage, the unit gets a new `nuclear_family_id`; `family_origin_id` remains unchanged.
- Children always inherit `family_origin_id` lineage from birth logic (documented in config/implementation details), independent of later parental divorces/remarriages.

---

## 8. Triangle Dynamics

### 8.1 Theoretical Basis

Bowen's core structural concept: **the triangle is the smallest stable relationship system.** A two-person system under stress becomes unstable — the more anxious person "triangles in" a third person to diffuse the tension. The third person absorbs some of the anxiety, temporarily stabilizing the original twosome.

Key properties:
- In calm: the triangle has a comfortable twosome + an outsider.
- Under tension: all three positions shift as each person tries to gain a twosome slot (the comfortable position) or escape to the outside (when tension is very high).
- In periods of very high stress, families triangle in neighbors, schools, police, clinics — the wider community becomes part of the family's emotional system.

### 8.2 Engine Implementation (Current)

The spatial grid layout with 8-neighbor contagion is the engine's implicit triangle model: the spatial neighbors of any cell form a set of overlapping three-person (and larger) relationships through which anxiety flows.

The **same-family zero-friction rule** implements the twosome dynamic: within a family block, anxiety circulates freely (no friction), while cross-family transmission is damped (friction = 0.2). This captures:
- The close twosome (same family block) where anxiety flows without resistance.
- The outsider relationship (different family block) where anxiety must overcome friction to propagate.

### 8.3 Triangling-In Mechanism (Specification for Future Implementation)

When a size-2 family has been under high stress (avg S > 180) for 2+ consecutive cycles and has NOT yet divorced, a **triangle event** fires:

1. Locate the nearest unattached circle (departed, live, within TRIANGLE_RADIUS = 20 cells).
2. That circle is temporarily "triangled in" — given the same family_id for the next 2 cycles.
3. During those 2 cycles:
   - The circle absorbs 20% of the family's excess S (both members' S reduced by 15%).
   - The circle's own S increases by the absorbed amount × (1 / circle_C_ratio).
   - After 2 cycles, the circle detaches and resumes its original family_id.
4. This models the child (or therapist, or neighbor) who becomes the temporary anxiety container for the couple.
5. The 2-cycle process repeats each time avg S remains > 180, creating a continuous triangling pattern for chronically stressed families.

**Tracking:** A `triangle_timer[N]` array (int16) records how many cycles each unit is currently serving as a triangle vertex. Units in an active triangle cannot be matchmaking candidates.

---

## 9. Coaching Toggle

### 9.1 Theoretical Basis

Bowen's coaching method (distinct from conventional psychotherapy) works by:
1. Teaching the client how emotional systems operate — increasing *observational capacity*.
2. Keeping the therapist "detriangled" — emotionally outside the family system while staying in contact with both members.
3. Helping the client define an "I position" — a differentiated self-stance within the family of origin.
4. Over time (years, not weeks), this work enables genuine increases in basic differentiation.

The average well-motivated client, working over 4–5 years, achieves meaningful increases in C — Bowen estimated roughly +0.5 levels per year of sustained coaching effort.

Coaching is NOT about removing stressors. It is about increasing the capacity to *not be reactive* to stressors that remain present.

### 9.2 Engine Specification

**Planned global toggle:** `sim.coaching_active` (bool, default False). When implemented, it is toggled by keyboard key **[G]** in `main.py`.

**When coaching is active, each cycle:**

```python
if self.coaching_active:
    live_mask = self.state[:, 3] > 0
    adult_live = live_mask & (self.age >= 18)

    # 1. Reactivity reduction (all live units): R decays faster during coaching.
    #    Coached R decay: additional -0.5/cycle floor'd at the CA floor.
    self.state[live_mask, 1] = np.maximum(
        self.state[live_mask, 1] - 0.5,
        self.chronic_anxiety[live_mask] / 50.0   # CA floor remains
    )

    # 2. Stress reduction: coaching reduces acute reactivity to stressors.
    #    S drifts toward S_BASELINE faster: -20% of excess per cycle.
    excess_s = np.maximum(self.state[live_mask, 0] - self.S_BASELINE, 0)
    self.state[live_mask, 0] -= excess_s * 0.2

    # 3. C growth (adults only): differentiation increases at +0.5/year = +0.5/cycle.
    self.state[adult_live, 2] = np.clip(
        self.state[adult_live, 2] + 0.5,
        10.0, 80.0
    )

    # 4. C-growth inverse (adults only): per §4.1, C↑ drives R↓ by the same amount.
    #    This is additive to the coached R decay in step 1.
    self.state[adult_live, 1] = np.maximum(
        self.state[adult_live, 1] - 0.5,
        self.chronic_anxiety[adult_live] / 50.0   # CA floor still respected
    )
```

**Effects summary:**

| Unit type | Variable | Effect during coaching | Bowen rationale |
|-----------|----------|----------------------|-----------------|
| All live | R (TX) | −0.5/cycle decay | Coaching teaches observation and control of emotional responses |
| All live | S | −20% of excess/cycle toward baseline | Acute anxiety reduces as clients distinguish real from perceived threats |
| Adults only | C | +0.5/cycle growth | Sustained differentiation work yields genuine self-growth over years |
| Adults only | R (TX) | additional −0.5/cycle (C-growth inverse) | As differentiation rises, reactivity drops 1:1; total adult R reduction = −1.0/cycle |
| All | CA | Unchanged | Chronic anxiety from family of origin is not erased; its *impact* is reduced by higher C |

**Sidebar display:**
```
COACHING : ACTIVE [g]   ← green when on
  C GROWTH: +0.5/yr
```

**Planned key binding:** `K_g` in `main.py` event loop.
```python
if event.key == pygame.K_g:
    sim.coaching_active = not sim.coaching_active
    log.append(f"Coaching: {'ON' if sim.coaching_active else 'OFF'}")
```

---

## 10. Family Leader (High-C Anchor)

### 10.1 Theoretical Basis

Bowen observed that a highly differentiated person in a family or group system has a stabilizing effect on the whole system — their lower reactivity and clearer "I position" prevents the group from escalating into undifferentiated fusion. The presence of even one high-C person in a family can protect the entire family from runaway anxiety.

This maps directly onto the engine's contagion model: a high-C unit at the center of a family block broadcasts far less stress (divided by clip(C, 10)) and absorbs less incoming stress, acting as a node that dampens propagation.

### 10.2 Strong Leader Effect (Already Implemented via L)

The global `L` (Leadership) parameter captures the population-level version of this. Under the current engine formula (`S_BASELINE / L`), higher `L` lowers the stress threshold at which R starts climbing, while still adding a direct +0.05 C/cycle bonus when `L > 1.2`. This section defines the intended stabilizing interpretation; align formula and interpretation together in implementation updates.

### 10.3 Family-Level Leader (Specification)

A per-family "leader" unit can be identified as the member with the highest current C. This unit:
- Gets the full `update_c()` recovery boost first (priority ordering).
- Has its C updated with L bonus doubled: `+0.20/cycle` when L > 1.2 (vs +0.05 for others).
- In the contagion calculation, its broadcast is additionally scaled by `0.7` (high-C leaders are less emotionally contagious even relative to their C level).

Planned implementation as a per-cycle identification step in `update_c()`:
```python
# Family leader: highest-C embedded live member per family
family_leader_mask = ...  # one True per active family — the max-C member
self.state[family_leader_mask, 2] += 0.20 if self.L > 1.2 else 0.0
```

---

## 11. Multigenerational Transmission — Lifecycle Summary

The full multigenerational pathway through the engine:

```
Birth
  └─ child_C = avg(parent_C) × uniform(0.9, 1.1)       [C inheritance, currently clipped to [10, 100]; target [10, 80]]
  └─ if projection target: child_C *= uniform(0.7, 0.9) [lower C for most anxious child]
  └─ birth_stress elevated if family avg S > 160         [stress inheritance]

Age 0–10
  └─ age[unit] increments each cycle
  └─ At age 10: chronic_anxiety fixed from family avg(S, R)

Age 18–25: Launch window
  └─ Data-calibrated launch probability (baseline 7%/cycle; stress-adjusted upward)
  └─ Circles remain eligible for future matchmaking/remarriage

Age 22–40: Matchmaking window (every 20 cycles)
  └─ C-proximity matching: primary ≤5% diff, secondary wider with penalties
  └─ New family created; family_c_diff, divorce_rate_mod, s_penalty stored

Age 22–45: Childbearing window
  └─ 8% chance/cycle per eligible family
  └─ Newborn C inherited from parents

Adult: C update every cycle
  └─ Drifts toward c_baseline when S < threshold
  └─ +0.05/cycle bonus when L > 1.2
  └─ +0.5/cycle during coaching (adults only)
  └─ R drops −1.0/cycle total during coaching (−0.5 decay + −0.5 C-growth inverse)

Death: Gompertz-Makeham (P = 0.00003 × exp(0.0785 × age), age ≥ 20)
  └─ Slot transitions to DEAD and is eligible for newborn activation (new identity, no resurrection)
```

### 11.1 Launch Calibration (North America Reference)

To avoid arbitrary launch rates, use contemporary North American household data as calibration anchors:

- U.S. Census (CPS ASEC 2024): 57% of adults age 18–24 live in parental homes; 16% of adults age 25–34 do.  
- Statistics Canada (Census 2021): 35.1% of adults age 20–34 live with at least one parent.

Reference links:
- U.S. Census Bureau (Families and Living Arrangements, 2024 tip sheet): https://www.census.gov/newsroom/press-releases/2024/families-and-living-arrangements.html
- U.S. Census Bureau (Families and Living Arrangements, 2025 tip sheet): https://www.census.gov/newsroom/press-releases/2025/families-and-living-arrangements.html
- Statistics Canada, The Daily (July 13, 2022; 2021 Census living arrangements): https://www150.statcan.gc.ca/n1/daily-quotidien/220713/dq220713a-eng.htm

Calibration implication:

- A flat 20% launch probability in ages 18–22 is too aggressive for baseline runs.
- Recommended baseline profile: `LAUNCH_BASE_P = 0.07` with stress multipliers (e.g., up to ~0.12 in high-stress families), then validate with evaluator metrics in §15.1.

---

## 12. Implementation Checklist

The following items are **fully implemented** in the current engine:

- [x] C (Differentiation) as TX damper in both contagion and TX update
- [x] R (Reactivity/TX) accumulation proportional to (S − baseline) / C
- [x] TX_MAX cap (4.0) — prevents runaway contagion spiral
- [x] Multigenerational C inheritance via apply_births() — currently clipped to [10, 100] (target [10, 80] in §3.3)
- [x] C-proximity matchmaking (two-pass: primary ≤5%, secondary with penalties)
- [x] Per-family C-mismatch: divorce_rate_mod and s_penalty arrays
- [x] Same-family zero-friction contagion (Axiom 4)
- [x] Recovery curve: post-ejection R/C restoration (20-cycle timer)
- [x] avg_c telemetry in sidebar and CSV

The following items are **specified in this document but not yet implemented:**

- [ ] C range clamped to [10, 80] in engine (currently [10, 100])
- [ ] C initialization changed to 40.0 ± 5 (currently 50.0 ± 5)
- [x] Generational-flux slot model: initialize 8,000 active + 2,000 inactive buffer (`M = -1`, `slot_status = INACTIVE_BUFFER`)
- [x] `slot_status[N]` lifecycle states with dead-slot reuse for births
- [x] Split family identity arrays: immutable `family_origin_id[N]` and mutable `nuclear_family_id[N]` (with `family_ids` compatibility alias)
- [ ] `chronic_anxiety[N]` array — fixed at age 10 from family emotional atmosphere
- [ ] R floor from CA: `R ≥ chronic_anxiety[unit] / 50.0` after age 10
- [ ] Per-unit S floor for high-C units: C > 55 subtracts (C − 55) from personal S floor, requiring per-unit floor array replacing the current scalar clip
- [ ] `coaching_active` toggle with:
  - R decay −0.5/cycle for all live units
  - S reduction −20% of excess/cycle
  - C growth +0.5/cycle for adults (clipped to 80)
  - C-growth inverse: additional R −0.5/cycle for adults (total adult R reduction = −1.0/cycle)
- [ ] `[G]` key binding in main.py for coaching toggle
- [ ] Coaching sidebar display (C GROWTH: +0.5/yr)
- [ ] Functional Differentiation (FD) as derived quantity used in contagion and income bonus
- [ ] Family projection process: projection-target child receives `C × uniform(0.7, 0.9)`
- [ ] Spouse dysfunction asymmetry: lower-C partner receives larger S-penalty share
- [ ] Emotional Distance mechanic: `family_distance_flag[F]`, reduced cross-member contagion, lower divorce rate, higher projection probability
- [ ] Triangle mechanism: short-term triangling-in of nearby circle to absorb family stress (repeating every 2 cycles while avg S > 180)
- [ ] Family-level leader identification: highest-C member gets +0.20/cycle C bonus (vs +0.05 for others) and 0.7× broadcast scaling
- [ ] Config file system: all parameters sourced from markdown config file, none hard-coded (see §14)

---

## 13. Parameter Reference — Bowen Interpretation

| Engine Parameter | Bowen Meaning | Recommended Range |
|-----------------|--------------|------------------|
| `S_BASELINE` | Ambient societal chronic anxiety — the background level below which no one functions (except high-C individuals per §5.3) | 60–100 (80 = calibrated default) |
| `L` (Leadership) | Quality and effectiveness of institutional differentiation — leaders who can stay "outside" the emotional system while remaining in contact | 1.0–2.0 (1.5 = best survival) |
| `X` (Social Media) | Amplification of triangling and emotional contagion across the grid — social media as a triangling mechanism at scale | 0.5–1.5 (1.0 = optimal) |
| `E` (Climate/Environment) | Environmental stressor load — raises the S floor for all; high-C units partially offset this via §5.3 | 0.5–1.5 (1.0 = neutral; E>1.5 destabilizes) |
| `TX_MAX` (4.0) | Upper bound on emotional reactivity — prevents acute anxiety from permanently disabling the system's recovery capacity | Fixed at 4.0 |
| `C_PRIMARY_PCT` (5%) | The matching tolerance within which two people can form a stable partnership without significant fusion penalties | Fixed at 5% C-diff |
| `INITIAL_ACTIVE_UNITS` | Initial active population at simulation start | 8,000 (for N=10,000 profile) |
| `NURSERY_BUFFER_UNITS` | Preallocated inactive slots reserved for births before dead-slot reuse | 2,000 (for N=10,000 profile) |
| `MATCH_INTERVAL` | Matchmaking cadence in cycles | 20 default; 10 stress-test profile |
| `LAUNCH_BASE_P` | Baseline annual launch probability in launch window | 0.05–0.10 (0.07 recommended baseline) |
| `coaching_active` | Whether the population is engaged in systematic differentiation work (therapy, education, self-observation) | Boolean toggle |
| `C_PRACTICAL_MIN` (10) | Minimum differentiation — units cannot fall below this regardless of projection or stress | Fixed at 10 |
| `C_PRACTICAL_MAX` (80) | Practical upper limit — fully differentiated in real-world terms; theoretical 100 is never reached | Fixed at 80 |

---

## 14. Parameters as Config Settings

1. No parameters for the model are to be hard-coded in engine source files.
2. All parameters will be accessed via a config file that is a Markdown file for easy review and adjustment.
3. Each parameter will be preceded by comments explaining what it does (impact per cycle, Bowen interpretation, and recommended range).
4. The config file will be parsed at simulation startup and applied to the Simulator object before `__init__` completes.

---

## 15. Evaluator and Glossary

### 15.1 Evaluator Metrics (Acceptance Gates)

Use these checks after parameter changes:

1. Biological stability: avg age should settle in a realistic band (target 30–50 in long runs).
2. Economic differentiation: in high-stress regimes (`X > 1.5`), high-C circles should maintain higher `M` than low-C squares.
3. Pruning before collapse: circles (departures) should rise before black-pixel metabolic deaths dominate.
4. Launch realism: parental-home share implied by launch rates should remain directionally consistent with §11.1 anchors.

### 15.2 Glossary of Bowen Terms Used in This Spec

| Term | Definition |
|------|-----------|
| **Differentiation of Self** | The degree to which a person can maintain a separate, autonomous self while remaining in emotional contact with others. Scale 0–100; practical engine range 10–80. Higher = more resilient. |
| **Reactivity** | The degree to which emotional stimuli automatically trigger feeling-based responses rather than thought-based ones. Inverse of differentiation in the short term. During coaching, C↑ produces R↓ at 1:1 for adults. |
| **Chronic Anxiety** | Persistent baseline anxiety carried from the family of origin. Present in all units. Does not resolve when stressors are removed. Forms a permanent floor on R set at age 10. |
| **Acute Anxiety** | Anxiety in response to a real, present threat. Rises and falls with the threat. |
| **Undifferentiated Ego Mass** | The emotional fusion of a family system — members lose individual self in the common emotional field. The engine models this as high R + high cross-family contagion. |
| **Triangle** | The basic three-person relationship unit. A twosome under stress pulls in a third to stabilize. In the engine, fires every 2 cycles when avg S > 180 in a size-2 family. |
| **Nuclear Family Emotional System** | The emotional patterns that develop within the immediate family: marital conflict (→ divorce/cutoff), spouse dysfunction (→ adaptive partner loses self), child projection (→ one child receives lower C), and emotional distance (→ intact but hollow marriage). |
| **Multigenerational Transmission** | The process by which differentiation levels (C) pass across generations, generally declining in the most over-involved child. Implemented via `child_C = avg(parent_C) × uniform(0.9, 1.1)`. |
| **Family Projection Process** | Parental anxiety focused onto one child, lowering that child's basic differentiation level. Projection target receives `C × uniform(0.7, 0.9)`. Target designation can shift as new children are born under different stress conditions. |
| **Coaching** | Bowen's therapeutic approach: teaching clients to observe the emotional system they are in and make self-defined moves to differentiate, without therapist involvement in the system. Engine effect: +0.5 C/yr for adults, −0.5 R/cycle all, additional −0.5 R/cycle for adults (C-growth inverse). |
| **Basic Self** | The stable, non-negotiable self that does not shift under relationship pressure. Corresponds to C in the engine, bounded [10, 80]. |
| **Pseudo-Self** | The borrowed, negotiable self that is traded in relationships. Represents short-term R fluctuations — C can appear higher or lower based on current relationship conditions. |
| **Functional Level of Differentiation (FD)** | How someone is actually functioning right now, as distinct from their basic C. Computed as `C × clip((S_BASELINE/L)/S, 0.3, 1.5)`. Can be significantly above or below C depending on circumstances. |
| **I Position** | A differentiated self-stance: "This is what I think and believe and will do" — stated without attacking or accommodating to others' positions. |
| **Detriangled** | The therapist's position: emotionally outside the family system while remaining in relationship contact with all members. |
| **Emotional Distance** | A partner's withdrawal from emotional contact within an intact marriage. Not the same as cutoff (divorce). Keeps the pair together while routing unresolved anxiety toward children. Specified for future engine implementation via `family_distance_flag`. |
| **Emotional Cutoff** | The severing of emotional contact through departure or divorce. Modeled in the engine by the divorce mechanic — the extreme end of the distance–cutoff continuum. |
