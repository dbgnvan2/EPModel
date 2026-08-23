Technical Specification and State Formulas1.

# Technical Specification: State Formulas

### 1. Core State Array (NumPy)

The system maintains a 4D state space for $N$ units:

- **$S$ (Stress):** Current deviation from baseline (70.0).
- **$TX$ (Togetherness Force):** Volatile reactivity; $TX \propto S$.
- **$C$ (Individuality):** The governor; $C_{new} = C_{old} - \text{erosion} + \text{recovery}$.
- **$M$ (Material/Wealth):** The metabolic account.

### 2. Primary Dynamics

#### Echo Contagion (Refined)

A unit's stress absorption from its neighbors is calculated via vectorized Laplacian sensing. The incoming stress is the sum of neighbors' reactivity, buffered by the receiver's own differentiation ($C_i$).

$$\Delta S_i = \frac{\sum_{j \in N} \left( (S_j - 70) \times \frac{TX_j}{C_j} \right) \times \text{Friction}}{C_i}$$

**Logic Audit Keys:**

- **The Sender ($j$):** High $TX_j$ and Low $C_j$ increase the "broadcast" intensity of stress.
- **The Receiver ($i$):** High $C_i$ acts as the denominator, damping the total absorbed stress.
- **The Friction:** A constant (default 0.2) representing the attenuation of stress across the social field.

#### Togetherness Volatility

$TX$ increases by a fixed factor of the stress deviation per cycle, inversely proportional to the current $C$ level.
$$\Delta TX = \frac{(S - 70) \times \gamma}{C}$$

#### Working on Self (Recovery)

If a unit remains stable ($S < 75$) and its $C$ is below its historical $C_{baseline}$, it recovers:
$$C_{recovery} = 0.5 \times (C_{baseline} - C_{current})$$

#### Metabolic Drain

The resource account $M$ is depleted by regulation and togetherness demands:
$$\frac{dM}{dt} = -(\text{BaseMetabolism} + \text{CoolingCost}(C_{eff}) + TX)$$

### 3. Societal Architecture

- **Clustering:** Grid is partitioned into 4 countries (quadrants).
- **Border Friction:** Laplacian sensing across quadrant lines is multiplied by 0.2 by default.
- **Taxation Logic:** If active, the system taxes the top 20% of the $M$ array by 2% and redistributes the pool to units where $S > 90$ and $M < 200$.

### 4. Visualization Requirements (Pygame)

- **Heat Layer:** `R = (S-70)*2.55`, `G = 255-R`, `B = 50`.
- **Wealth Layer:** `B = (M/MaxM)*255`.
- **Dead State:** Any unit where $M \le 0$ renders as `(40, 40, 40)`.
