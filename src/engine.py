import numpy as np
from collections import deque
import os


class Simulator:
    """
    Manages the state of the Societal Inflammatory Simulator.
    """
    # --- Simulation Constants ---
    FI_CONSTANT = 10.0
    FAMILY_SIZES = [2, 4, 8, 16]
    S_BASELINE = 80.0
    WD_FACTOR = 0.01          # 1% windfall chance/cycle (was 10% — too lottery-like)
    Q_THRESHOLD_ON = 200.0
    Q_THRESHOLD_OFF = 150.0
    Q_FACTOR = 0.2
    TX_MAX = 4.0              # TX cap — prevents runaway contagion spiral
    INITIAL_ACTIVE_UNITS = 8000
    NURSERY_BUFFER_UNITS = 2000
    MATCH_INTERVAL = 20
    LAUNCH_BASE_P = 0.07

    # --- Status Constants ---
    STATUS_EMBEDDED = 0
    STATUS_DEPARTED = 1
    SLOT_ACTIVE = 0
    SLOT_DEPARTED = 1
    SLOT_DEAD = 2
    SLOT_INACTIVE_BUFFER = 3

    def __init__(self, num_units=10000):
        self.num_units = num_units
        self.grid_size = int(np.sqrt(num_units))
        if self.grid_size * self.grid_size != num_units:
            raise ValueError("Number of units must be a perfect square.")

        self.state = np.zeros((num_units, 4), dtype=np.float64)
        self.state[:, 0] = self.S_BASELINE + np.random.uniform(-16, 16, num_units)
        self.state[:, 1] = 1.0
        self.state[:, 2] = 50.0 + np.random.uniform(-5, 5, num_units)

        self.slot_status = np.full(num_units, self.SLOT_ACTIVE, dtype=np.int8)
        if num_units == (self.INITIAL_ACTIVE_UNITS + self.NURSERY_BUFFER_UNITS):
            inactive_indices = np.random.choice(
                num_units, self.NURSERY_BUFFER_UNITS, replace=False
            )
            self.slot_status[inactive_indices] = self.SLOT_INACTIVE_BUFFER
        else:
            inactive_indices = np.array([], dtype=np.int64)

        self.state[:, 3] = -1.0
        active_indices = np.where(self.slot_status == self.SLOT_ACTIVE)[0]
        if active_indices.size > 0:
            self.state[active_indices, 3] = 1000.0
            elite_count = int(active_indices.size * 0.1)
            if elite_count > 0:
                elite_indices = np.random.choice(active_indices, elite_count, replace=False)
                self.state[elite_indices, 3] = 100000.0

        self.c_baseline = self.state[:, 2].copy()
        self.unit_status = np.full(num_units, self.STATUS_EMBEDDED, dtype=np.int8)
        if inactive_indices.size > 0:
            self.unit_status[inactive_indices] = self.STATUS_DEPARTED
            self.state[inactive_indices, 1] = 0.0
        self.high_stress_duration = np.zeros(num_units, dtype=np.int16)
        self.total_deaths = 0
        self.tax_pool = 0.0

        self.safety_net_active = False
        self.national_debt = 0.0

        self.L = 1.0
        self.X = 1.0
        self.E = 1.0

        # --- Demographic Parameters ---
        # base_income = 20.0 per cycle.
        # Rationale: M starts at 1000 for 90% of units.  With E=1.0 neutral drain
        # of ~0.15/cycle, income must meaningfully build a buffer.  $20 maps to
        # "one unit of disposable income per month" in abstract M-units.
        # Elite (top 10%) start at M=100,000 — they earn proportionally more via
        # the circle autonomy bonus and windfall (WD_FACTOR).
        self.base_income      = 20.0         # income per cycle per live unit (was 5)
        self.unemployment_rate = 0.05        # fraction with zero income + stress penalty
        self.divorce_count    = 0            # cumulative divorces

        # Age: 1 cycle = 1 year.  Realistic demographic pyramid.
        # Use a truncated exponential to mimic real age pyramids in developed nations:
        # more young people, tapering toward old age, capped at 75.
        # This ensures deaths are spread evenly over time rather than in a single wave.
        # Real-world: ~13% of US population is over 65; ~18% under 15.
        # We now start from a Normal(35, 15) profile, clipped to [0, 95], and
        # keep a nursery buffer (M=-1) at age 0 for birth activation.
        self.age = np.clip(np.random.normal(35.0, 15.0, num_units), 0.0, 95.0).astype(np.float32)
        if inactive_indices.size > 0:
            self.age[inactive_indices] = 0.0

        # Per-family high-stress duration tracker for divorce mechanic
        self.family_high_stress_duration = np.zeros(self.num_units, dtype=np.int16)

        # Telemetry output fields
        self.homelessness_rate  = 0.0
        self.avg_age            = 0.0
        self.unemployment_count = 0
        self.total_births       = 0   # cumulative births
        self.total_marriages    = 0   # cumulative marriages

        self._init_family_clusters()
        self.nuclear_family_id = self.family_ids
        self.family_origin_id = self.family_ids.copy()
        self.log_messages = deque(maxlen=100)
        self.initial_family_sizes = np.bincount(self.family_ids)
        self.recovery_timers = np.zeros(self.num_units, dtype=np.int16)

        self.active_family_count = 0
        self.avg_family_size = 0
        self.avg_family_m = 0
        self.avg_c = 50.0  # population average C-level for telemetry

        # Per-family C-level mismatch tracking (grows as new families form via marriage)
        # family_c_diff[fid]          = absolute C-diff % at time of marriage (0 for birth families)
        # family_divorce_rate_mod[fid] = extra divorce rate added by C-mismatch
        # family_s_penalty[fid]       = per-cycle S baseline raise from C-mismatch (capped at 50)
        self.family_c_diff          = np.zeros(self.num_families, dtype=np.float32)
        self.family_divorce_rate_mod = np.zeros(self.num_families, dtype=np.float32)
        self.family_s_penalty       = np.zeros(self.num_families, dtype=np.float32)

        if not os.path.exists("sim_audit.csv"):
            with open("sim_audit.csv", "w") as f:
                f.write("cycle,living_units,active_families,avg_s,total_deaths,"
                        "total_births,total_marriages,debt,recovery_active,"
                        "homelessness_pct,avg_age,divorce_count,avg_c\n")

    # ---------------------------------------------------------------------- #
    #  Initialisation                                                         #
    # ---------------------------------------------------------------------- #

    def _init_family_clusters(self):
        """Tile the grid with compact rectangular family blocks (shelf-first algorithm).

        Block shapes: size 2 → 1×2, size 4 → 2×2, size 8 → 2×4, size 16 → 4×4.
        All families are contiguous filled rectangles — no random scatter.
        Constitution Axiom 4 requires this so same-family friction=0 is meaningful.
        """
        G = self.grid_size
        BLOCK_SHAPES = {2: (1, 2), 4: (2, 2), 8: (2, 4), 16: (4, 4)}

        height_to_sizes = {}
        for size, (bh, bw) in BLOCK_SHAPES.items():
            if G % bw == 0 and bh <= G:
                height_to_sizes.setdefault(bh, []).append(size)

        family_id_grid = np.zeros((G, G), dtype=np.int32)
        family_sizes_list = []
        family_id = 0
        cursor_row = 0

        while cursor_row < G:
            remaining_rows = G - cursor_row
            valid_heights = sorted(h for h in height_to_sizes if h <= remaining_rows)
            shelf_height = int(np.random.choice(valid_heights))
            shelf_sizes = height_to_sizes[shelf_height]

            cursor_col = 0
            while cursor_col < G:
                remaining_cols = G - cursor_col
                fitting = [s for s in shelf_sizes if BLOCK_SHAPES[s][1] <= remaining_cols]
                size = int(np.random.choice(fitting))
                bh, bw = BLOCK_SHAPES[size]
                family_id_grid[cursor_row:cursor_row + bh,
                               cursor_col:cursor_col + bw] = family_id
                family_sizes_list.append(size)
                cursor_col += bw
                family_id += 1

            cursor_row += shelf_height

        self.family_ids = family_id_grid.flatten()
        self.family_member_counts = np.array(family_sizes_list)
        self.num_families = family_id

    # ---------------------------------------------------------------------- #
    #  Telemetry                                                              #
    # ---------------------------------------------------------------------- #

    def calculate_family_telemetry(self):
        """Compute active family stats using embedded squares (M>0) only."""
        square_mask = (self.state[:, 3] > 0) & (self.unit_status == self.STATUS_EMBEDDED)
        current_counts = np.bincount(self.family_ids[square_mask], minlength=self.num_families)
        active_fam_mask = current_counts >= 2
        self.active_family_count = int(np.sum(active_fam_mask))

        if self.active_family_count > 0:
            family_m_sums = np.bincount(self.family_ids[square_mask],
                                        weights=self.state[square_mask, 3],
                                        minlength=self.num_families)
            safe_counts = np.where(active_fam_mask, current_counts, 1)
            self.avg_family_m = float(np.mean(family_m_sums[active_fam_mask] /
                                              safe_counts[active_fam_mask]))
            self.avg_family_size = float(np.sum(current_counts[active_fam_mask]) /
                                         self.active_family_count)
        else:
            self.avg_family_m = 0
            self.avg_family_size = 0

        # Homelessness: live units with M < 100 AND S > 200
        live_mask = (self.state[:, 3] > 0)
        homeless_mask = live_mask & (self.state[:, 3] < 100) & (self.state[:, 0] > 200)
        live_count = int(np.sum(live_mask))
        self.homelessness_rate = float(np.sum(homeless_mask) / live_count * 100) if live_count > 0 else 0.0

        # Average age of live units
        self.avg_age = float(np.mean(self.age[live_mask])) if live_count > 0 else 0.0

        # Average C-level of live units (tracks multigenerational drift)
        self.avg_c = float(np.mean(self.state[live_mask, 2])) if live_count > 0 else 50.0

    def log_telemetry(self, cycle):
        if cycle % 10 == 0:
            living_mask  = (self.state[:, 3] > 0) & (self.unit_status == self.STATUS_EMBEDDED)
            living_units = int(np.sum(living_mask))
            avg_s        = float(np.mean(self.state[living_mask, 0])) if living_units > 0 else 0.0
            recovery_active = int(np.sum(self.recovery_timers > 0))
            with open("sim_audit.csv", "a") as f:
                f.write(f"{cycle},{living_units},{self.active_family_count},{avg_s:.2f},"
                        f"{self.total_deaths},{self.total_births},{self.total_marriages},"
                        f"{int(self.national_debt)},{recovery_active},"
                        f"{self.homelessness_rate:.2f},{self.avg_age:.1f},{self.divorce_count},"
                        f"{self.avg_c:.2f}\n")

    # ---------------------------------------------------------------------- #
    #  Economic Engine                                                        #
    # ---------------------------------------------------------------------- #

    def apply_income(self):
        """Vectorized income: base income for all live units, autonomy bonus for circles.

        Spec 2A:
          - Every live unit (M > 0) earns base_income per cycle.
          - Circles earn base_income * 1.5 * (C / 100)  (autonomy bonus).
          - unemployment_rate % of units receive 0 income + +10 stress penalty.
        Income is added BEFORE the progressive tax so it can be taxed (Constraint 3).
        """
        live_mask = self.state[:, 3] > 0

        income_vector = np.zeros(self.num_units, dtype=np.float64)
        income_vector[live_mask] = self.base_income

        # Autonomy bonus for departed circles that are still alive
        circle_mask = (self.unit_status == self.STATUS_DEPARTED) & live_mask
        income_vector[circle_mask] *= 1.5 * (self.state[circle_mask, 2] / 100.0)

        # Unemployment: random mask, zero income + stress hit
        unemployed = (np.random.rand(self.num_units) < self.unemployment_rate) & live_mask
        income_vector[unemployed] = 0.0
        self.state[unemployed, 0] += 10.0   # unemployment stress penalty
        self.unemployment_count = int(np.sum(unemployed))

        self.state[:, 3] += income_vector

    def apply_progressive_tax(self, cycle):
        if cycle % 100 != 0:
            return
        taxable_mask = self.state[:, 3] > 5000
        tax_revenue = np.sum(self.state[taxable_mask, 3] * 0.02)
        self.state[taxable_mask, 3] *= 0.98
        self.tax_pool += tax_revenue
        if self.tax_pool > 0 and self.national_debt > 0:
            payment = min(self.tax_pool, self.national_debt)
            self.national_debt -= payment
            self.tax_pool -= payment
            self.log_messages.append(f"[{cycle}] Tax revenue paid down debt by ${payment:,.0f}")

    # ---------------------------------------------------------------------- #
    #  Demographic Engine                                                     #
    # ---------------------------------------------------------------------- #

    def apply_aging(self):
        """Increment age for all live units by 1 cycle (= 1 year).

        Calibrated to CDC 2022 North American life tables (Gompertz-Makeham law):
          P(death | age) = 0.00003 × exp(0.0785 × age)

          Empirical checkpoints:
            age 50: P = 0.00152  → life expectancy ~660 years from 50 (negligible)
            age 70: P = 0.00730  → life expectancy ~137 cycles from 70
            age 80: P = 0.01601  → life expectancy ~62 cycles from 80
            age 90: P = 0.03511  → life expectancy ~28 cycles from 90
            age 100: P = 0.07700 → life expectancy ~13 cycles from 100

          At steady state (Uniform[0,90] pyramid, 10,000 units):
            Expected deaths ≈ 52/cycle ≈ 5.2/1000/yr
            (close to US 8.6/1000 — slight undercount as our model excludes
             infant/disease mortality tracked separately via M-drain)

        Vectorized — no per-unit Python loop.
        """
        live_mask = self.state[:, 3] > 0
        self.age[live_mask] += 1.0

        # Full Gompertz-Makeham: applies to all ages ≥ 0
        # Skip units < 20 for performance (P < 0.00015 — negligible)
        adult_mask = live_mask & (self.age >= 20)
        if np.any(adult_mask):
            p_death = np.clip(
                0.00003 * np.exp(0.0785 * self.age[adult_mask]),
                0.0, 0.40
            )
            die_roll = np.random.rand(int(np.sum(adult_mask)))
            dying = np.where(adult_mask)[0][die_roll < p_death]
            if len(dying) > 0:
                self.unit_status[dying] = self.STATUS_DEPARTED
                self.slot_status[dying] = self.SLOT_DEAD
                self.total_deaths += len(dying)
                self.state[dying, 3] = 0.0   # zero M to trigger black override

    def apply_launch(self, cycle):
        """Young-adult leave-home mechanic ("Launch").

        Embedded units aged 18-22 have a 10% chance per cycle of becoming Circles.
        That chance doubles (20%) if their family avg S > 160.
        Vectorized: build probability vector, single random draw.
        """
        embedded = self.unit_status == self.STATUS_EMBEDDED
        live     = self.state[:, 3] > 0
        # Real-world: young adults leave home 18-25 (college, jobs, relationships).
        # p=0.05/cycle → ~5% chance per year → average 20 years in home after 18
        # (i.e. most leave by ~23, which is realistic).
        # Under-18s (children in family) cannot launch.
        young    = (self.age >= 18) & (self.age <= 25)
        eligible = embedded & live & young

        if not np.any(eligible):
            return

        # Per-family avg stress for the doubling check
        active_mask = embedded & live
        family_s_sums  = np.bincount(self.family_ids[active_mask],
                                     weights=self.state[active_mask, 0],
                                     minlength=self.num_families)
        family_counts  = np.bincount(self.family_ids[active_mask],
                                     minlength=self.num_families)
        safe_counts    = np.where(family_counts > 0, family_counts, 1)
        family_avg_s   = family_s_sums / safe_counts          # shape: (num_families,)

        # Per-unit base probability
        p_launch = np.zeros(self.num_units, dtype=np.float64)
        p_launch[eligible] = self.LAUNCH_BASE_P

        # Double probability where family avg S > 160
        high_stress_fam = family_avg_s > 160                  # shape: (num_families,)
        unit_high_stress = high_stress_fam[self.family_ids]   # broadcast to units
        p_launch[eligible & unit_high_stress] = min(self.LAUNCH_BASE_P * 2.0, 1.0)

        # Single vectorized roll
        roll = np.random.rand(self.num_units)
        launching = eligible & (roll < p_launch)
        launching_indices = np.where(launching)[0]

        for uid in launching_indices:
            self.unit_status[uid] = self.STATUS_DEPARTED
            self.slot_status[uid] = self.SLOT_DEPARTED
            self.log_messages.append(f"[{cycle}] Unit {uid} Launched (age {self.age[uid]:.0f})")

    def apply_matchmaking(self, cycle):
        """Vectorized spatial-distance matchmaking every 20 cycles.

        Eligible candidates: STATUS_DEPARTED circles, aged 20-35, S < 150, M > 0.

        Pairing rule (C-proximity first):
          Primary match:   C-levels within 5% of each other AND within MATCH_RADIUS.
          Secondary match: wider C gap allowed if no primary partner found; mismatch
                           penalties are stored per-family for use in apply_divorce().

        Mismatch penalties (per 1% beyond 5% C-diff gap):
          - +2% added to family divorce rate modifier
          - +1.0 added to family S baseline raise
          Both capped so that S raise ≤ 50 (which implies c_diff ≤ 55%).

        Vectorized with a NumPy distance+C-diff matrix on the candidate set only.
        No nested Python loops over all 10,000 units.
        """
        if cycle % self.MATCH_INTERVAL != 0:
            return

        departed = self.unit_status == self.STATUS_DEPARTED
        live     = self.state[:, 3] > 0
        # Real-world: median age at first marriage US ≈ 28-30; window 22-40
        eligible = (departed & live &
                    (self.age >= 22) & (self.age <= 40) &
                    (self.state[:, 0] < 150))

        candidate_ids = np.where(eligible)[0]
        if len(candidate_ids) < 2:
            return

        # Grid coordinates and C-levels of each candidate
        G    = self.grid_size
        rows = candidate_ids // G              # shape: (C,)
        cols = candidate_ids %  G              # shape: (C,)
        c_vals = self.state[candidate_ids, 2]  # shape: (C,)

        # Vectorized pairwise Euclidean distance matrix — shape: (C, C)
        dr   = rows[:, None] - rows[None, :]
        dc   = cols[:, None] - cols[None, :]
        dist = np.sqrt(dr**2 + dc**2)

        # Vectorized pairwise C-diff % matrix — shape: (C, C)
        # c_diff_pct[i,j] = |C_i - C_j| / mean(C_i, C_j) * 100
        c_mean_mat  = (c_vals[:, None] + c_vals[None, :]) / 2.0
        c_mean_mat  = np.where(c_mean_mat < 1.0, 1.0, c_mean_mat)   # avoid /0
        c_diff_pct  = np.abs(c_vals[:, None] - c_vals[None, :]) / c_mean_mat * 100.0

        MATCH_RADIUS  = 15    # grid cells
        C_PRIMARY_PCT = 5.0   # within this C-diff % → primary (no penalty)
        paired = set()

        # ---- helper to register a new family given two candidate indices --------
        def _register_marriage(i, j):
            uid_a = candidate_ids[i]
            uid_b = candidate_ids[j]
            c_gap = float(c_diff_pct[i, j])

            new_fid = self.num_families
            self.family_ids[uid_a] = new_fid
            self.family_ids[uid_b] = new_fid
            self.unit_status[uid_a] = self.STATUS_EMBEDDED
            self.unit_status[uid_b] = self.STATUS_EMBEDDED
            self.slot_status[uid_a] = self.SLOT_ACTIVE
            self.slot_status[uid_b] = self.SLOT_ACTIVE
            self.num_families += 1
            self.family_member_counts = np.append(self.family_member_counts, 2)

            # Extend per-family mismatch arrays to cover new_fid
            excess_pct = max(0.0, c_gap - C_PRIMARY_PCT)
            dr_mod  = float(np.clip(excess_pct * 0.02, 0.0, 1.0))   # +2%/% → max 100%
            s_pen   = float(np.clip(excess_pct * 1.0,  0.0, 50.0))  # +1.0/% → max 50
            self.family_c_diff          = np.append(self.family_c_diff, c_gap)
            self.family_divorce_rate_mod = np.append(self.family_divorce_rate_mod, dr_mod)
            self.family_s_penalty       = np.append(self.family_s_penalty, s_pen)

            self.total_marriages += 1
            match_type = "PRIMARY" if c_gap <= C_PRIMARY_PCT else f"MISMATCH(C-diff={c_gap:.1f}%)"
            self.log_messages.append(
                f"[{cycle}] Marriage {match_type}: Units {uid_a}+{uid_b} → Fam {new_fid}")

        # ---- Pass 1: primary matches (within C_PRIMARY_PCT AND within radius) ---
        for i in range(len(candidate_ids)):
            if i in paired:
                continue
            primary_mask = (
                (c_diff_pct[i, :] <= C_PRIMARY_PCT) &
                (dist[i, :] <= MATCH_RADIUS)
            )
            primary_mask[i] = False   # not self
            primary_js = [jj for jj in np.where(primary_mask)[0] if jj not in paired]
            if not primary_js:
                continue
            j = min(primary_js, key=lambda jj: dist[i, jj])
            paired.add(i); paired.add(j)
            _register_marriage(i, j)

        # ---- Pass 2: secondary matches (within radius, any C gap) ---------------
        for i in range(len(candidate_ids)):
            if i in paired:
                continue
            within = (dist[i, :] <= MATCH_RADIUS)
            within[i] = False
            secondary_js = [jj for jj in np.where(within)[0] if jj not in paired]
            if not secondary_js:
                continue
            j = min(secondary_js, key=lambda jj: dist[i, jj])
            paired.add(i); paired.add(j)
            _register_marriage(i, j)

    def apply_births(self, cycle):
        """Birth engine: chance per cycle for active families with adults aged 25-40.

        Real-world calibration:
          - US birth rate: ~11/1000/year → ~110 births/cycle at 10k population
          - Average age at first birth: 27-30 years
          - Probability per eligible family per cycle: 8%
            (families of size 2+ where at least one parent is 25-40)
          - Eligible: families with ≥ 2 embedded live members and at least one
            member aged 25-40 (childbearing window)
          - Newborn receives C inherited from parents, M grant of 500.

        Reuses slots in this priority:
          INACTIVE_BUFFER -> DEAD -> DEPARTED.
        Slot search uses nearest-to-family centroid among currently available candidates.
        """
        embedded = self.unit_status == self.STATUS_EMBEDDED
        live     = self.state[:, 3] > 0
        active_mask = embedded & live

        # Families with ≥ 2 active members
        family_counts = np.bincount(self.family_ids[active_mask],
                                    minlength=self.num_families)

        # Per-family: does at least one member fall in childbearing age 22-45?
        # (wider window: many first-time parents are 22-35, and second children extend to 45)
        child_age_mask = active_mask & (self.age >= 22) & (self.age <= 45)
        family_has_parent = np.zeros(self.num_families, dtype=bool)
        if np.any(child_age_mask):
            parent_fids = self.family_ids[child_age_mask]
            family_has_parent_count = np.bincount(parent_fids, minlength=self.num_families)
            family_has_parent = family_has_parent_count > 0

        # Eligible: ≥2 members AND has a childbearing-age adult
        eligible_fids = np.where((family_counts >= 2) & family_has_parent)[0]
        if len(eligible_fids) == 0:
            return

        # Birth probability: 8% per eligible family per cycle.
        # Target: match CDC birth rate of ~11/1000/yr.
        # With ~700 eligible families (steady state) × 0.08 ≈ 56 births/cycle
        # → 5.6/1000 — slightly below US rate (11/1000) but conservative to
        #    avoid runaway population growth in the model.
        # The gap (5.6 vs 11) reflects that our model requires BOTH parents
        # to be embedded + family intact — real birth data includes single parents.
        roll = np.random.rand(len(eligible_fids))
        birth_fids = eligible_fids[roll < 0.08]
        if len(birth_fids) == 0:
            return

        slot_groups = self._get_birth_slot_groups()
        if sum(len(g) for g in slot_groups) == 0:
            return  # No room — all units still active

        G = self.grid_size
        used_slots = set()

        for fid in birth_fids:
            # Find the parents' grid positions
            parent_ids = np.where((self.family_ids == fid) & active_mask)[0]
            if len(parent_ids) < 2:
                continue

            # Parent centroid on the grid
            p_rows = parent_ids // G
            p_cols = parent_ids %  G
            cx = float(np.mean(p_rows))
            cy = float(np.mean(p_cols))

            nearest = None
            # Enforce priority tiers: INACTIVE_BUFFER -> DEAD -> DEPARTED.
            for group in slot_groups:
                available = [s for s in group if s not in used_slots]
                if not available:
                    continue
                av_arr = np.array(available)
                av_rows = av_arr // G
                av_cols = av_arr %  G
                dists   = np.sqrt((av_rows - cx)**2 + (av_cols - cy)**2)
                nearest = available[int(np.argmin(dists))]
                break
            if nearest is None:
                break
            used_slots.add(nearest)

            # M grant from parents
            grant = 500.0
            per_parent = grant / len(parent_ids)
            for pid in parent_ids:
                self.state[pid, 3] = max(0.0, self.state[pid, 3] - per_parent)

            # C-Level Inheritance: child C = avg(parent C) × uniform(0.9, 1.1)
            parent_c_avg = float(np.mean(self.state[parent_ids, 2]))
            child_c = float(np.clip(parent_c_avg * np.random.uniform(0.9, 1.1), 10.0, 100.0))
            child_origin = int(self.family_origin_id[parent_ids[0]])

            # Birth stress: if family avg S is elevated, newborn inherits some stress
            family_avg_s = float(np.mean(self.state[parent_ids, 0]))
            birth_stress = self.S_BASELINE
            if family_avg_s > 160.0:
                birth_stress += (family_avg_s - 160.0) * 0.5

            # Activate the newborn
            self.unit_status[nearest] = self.STATUS_EMBEDDED
            self.slot_status[nearest] = self.SLOT_ACTIVE
            self.family_ids[nearest]  = fid
            self.nuclear_family_id[nearest] = fid
            self.family_origin_id[nearest] = child_origin
            self.age[nearest]         = 0.0
            self.state[nearest, 0]    = float(np.clip(birth_stress, self.S_BASELINE, 300.0))
            self.state[nearest, 1]    = 1.0                      # TX reset
            self.state[nearest, 2]    = child_c                  # inherited C
            self.state[nearest, 3]    = grant                    # M grant
            self.recovery_timers[nearest] = 0
            self.high_stress_duration[nearest] = 0

            self.total_births += 1
            self.log_messages.append(
                f"[{cycle}] Birth in Fam {fid} → slot {nearest} C={child_c:.1f}")

    def apply_divorce(self, cycle):
        """Vectorized divorce check for size-2 families under chronic stress.

        A size-2 family that has S > 160 for 50+ cycles has a base 1% chance per
        cycle of splitting. Families formed via C-mismatch matchmaking have their
        divorce rate raised by family_divorce_rate_mod (up to +100%) and their
        members receive a continuous S-raise from family_s_penalty (capped at +50).

        Implementation uses NumPy binops on family-level arrays — no unit loops.
        """
        embedded = self.unit_status == self.STATUS_EMBEDDED
        live     = self.state[:, 3] > 0
        active_mask = embedded & live

        # Ensure per-family arrays cover all known families (grows via matchmaking)
        nf = self.num_families
        for arr_name in ('family_c_diff', 'family_divorce_rate_mod', 'family_s_penalty'):
            arr = getattr(self, arr_name)
            if len(arr) < nf:
                pad = np.zeros(nf - len(arr), dtype=arr.dtype)
                setattr(self, arr_name, np.concatenate([arr, pad]))

        # Per-family: count of active members
        family_counts = np.bincount(self.family_ids[active_mask], minlength=nf)
        # Per-family: average stress
        family_s_sums = np.bincount(self.family_ids[active_mask],
                                    weights=self.state[active_mask, 0],
                                    minlength=nf)
        safe_counts = np.where(family_counts > 0, family_counts, 1)
        family_avg_s = family_s_sums / safe_counts

        # Apply continuous S-raise for C-mismatched families (every cycle, all members)
        # Only applies to size-2 active families with a non-zero S-penalty.
        mismatch_fam_mask = (family_counts == 2) & (self.family_s_penalty[:nf] > 0)
        mismatch_fids = np.where(mismatch_fam_mask)[0]
        if len(mismatch_fids) > 0:
            # Map per-family S-penalty back to unit indices
            unit_penalty = self.family_s_penalty[self.family_ids]
            apply_pen_mask = active_mask & (unit_penalty > 0)
            self.state[apply_pen_mask, 0] += unit_penalty[apply_pen_mask]

        # Track per-unit high-stress duration for divorce trigger
        stressed_units = active_mask & (self.state[:, 0] > 160)
        self.family_high_stress_duration[stressed_units] += 1
        self.family_high_stress_duration[~stressed_units] = 0

        # Per-family: min stress duration (both members must be stressed)
        family_min_stress_dur = np.full(nf, np.iinfo(np.int16).max, dtype=np.int32)
        for uid in np.where(active_mask)[0]:
            fid = self.family_ids[uid]
            if self.family_high_stress_duration[uid] < family_min_stress_dur[fid]:
                family_min_stress_dur[fid] = self.family_high_stress_duration[uid]

        # Candidate families: exactly 2 active members, avg S > 160, stress ≥ 50 cycles
        candidate_fam_mask = (
            (family_counts == 2) &
            (family_avg_s > 160) &
            (family_min_stress_dur >= 50)
        )
        candidate_fids = np.where(candidate_fam_mask)[0]
        if len(candidate_fids) == 0:
            return

        # Effective divorce rate = base 1% + per-family C-mismatch modifier
        base_rate = 0.01
        eff_rates = base_rate + self.family_divorce_rate_mod[candidate_fids]
        eff_rates = np.clip(eff_rates, 0.0, 1.0)

        roll = np.random.rand(len(candidate_fids))
        divorce_fids = candidate_fids[roll < eff_rates]

        for fid in divorce_fids:
            members = np.where((self.family_ids == fid) & active_mask)[0]
            self.unit_status[members] = self.STATUS_DEPARTED
            self.slot_status[members] = self.SLOT_DEPARTED
            self.family_ids[members] = -1
            self.nuclear_family_id[members] = -1
            self.divorce_count += 1
            c_note = (f" [C-diff={self.family_c_diff[fid]:.1f}%]"
                      if self.family_c_diff[fid] > 0 else "")
            self.log_messages.append(
                f"[{cycle}] Fam {fid}: DIVORCE — 2 Circles launched{c_note}")

    # ---------------------------------------------------------------------- #
    #  Departure / Safety / Recovery                                          #
    # ---------------------------------------------------------------------- #

    def handle_departures(self, cycle):
        """Eject units that are emotionally overloaded or resource-poor.

        Ejection triggers (before metabolic death M<=0):
          - S >= 180  : emotional threshold
          - M <  150  : resource poverty
          - high_stress_duration >= 50 : chronic high stress
        total_deaths NOT incremented here — ejection ≠ death.
        """
        embedded = self.unit_status == self.STATUS_EMBEDDED
        stress_eject   = (self.state[:, 0] >= 180) & embedded
        resource_eject = (self.state[:, 3] < 150) & (self.state[:, 3] > 0) & embedded
        timed_eject    = (self.high_stress_duration >= 50) & embedded
        departing_mask = stress_eject | resource_eject | timed_eject
        departing_indices = np.where(departing_mask)[0]

        for unit_id in departing_indices:
            self.unit_status[unit_id] = self.STATUS_DEPARTED
            self.slot_status[unit_id] = self.SLOT_DEPARTED
            family_id = self.family_ids[unit_id]
            self.log_messages.append(f"[{cycle}] Unit {unit_id} Ejected Fam {family_id}")

            family_members_mask = ((self.family_ids == family_id) &
                                   (self.unit_status == self.STATUS_EMBEDDED))
            # Ejection Relief: immediate -50 stress + 20-cycle 5%/cycle recovery
            self.state[family_members_mask, 0] -= 50.0
            self.recovery_timers[family_members_mask] = 20

            remaining_members = int(np.sum(family_members_mask))
            if (self.initial_family_sizes.size > family_id
                    and self.initial_family_sizes[family_id] > 0
                    and remaining_members / self.initial_family_sizes[family_id] >= 0.5):
                self.log_messages.append(f"[{cycle}] Fam {family_id}: Systemic Recovery")

    def apply_safety_net(self):
        if not self.safety_net_active:
            return
        recovery_bonus = -10.0
        if self.L < 0.5:
            recovery_bonus *= 0.5
        subsidy_mask = self.state[:, 0] > 150
        num_subsidized = np.sum(subsidy_mask)
        if num_subsidized > 0:
            self.state[subsidy_mask, 0] += recovery_bonus
            self.national_debt += num_subsidized * 50

    def apply_recovery_curve(self):
        recovering_mask = self.recovery_timers > 0
        if not np.any(recovering_mask):
            return
        self.state[recovering_mask, 0] *= 0.95
        self.state[recovering_mask, 2] += 0.5
        self.recovery_timers[recovering_mask] -= 1

    # ---------------------------------------------------------------------- #
    #  Core update passes                                                     #
    # ---------------------------------------------------------------------- #

    def update_tx(self, gamma=0.1):
        s_baseline = self.S_BASELINE / self.L
        s, c = self.state[:, 0], self.state[:, 2]
        delta_tx = ((s - s_baseline) * gamma) / np.clip(c, 10, None)
        self.state[:, 1] += delta_tx
        # Cap TX to prevent runaway contagion spiral (TX>5 was causing
        # unlimited stress amplification in long runs)
        self.state[:, 1] = np.clip(self.state[:, 1], 0.0, self.TX_MAX)

    def update_c(self):
        s = self.state[:, 0]
        c = self.state[:, 2]
        stable_mask = s < (self.S_BASELINE / self.L) + 5
        recovery = 0.5 * (self.c_baseline[stable_mask] - c[stable_mask])
        self.state[stable_mask, 2] += recovery
        if self.L > 1.2:
            self.state[:, 2] += 0.05
        self.state[:, 2] = np.clip(self.state[:, 2], 10.0, 100.0)

    def update_m(self, base_metabolism=0.1):
        # Windfall: rare wealth events (inheritance, business success, lottery).
        # WD_FACTOR=0.01 → 1% chance/cycle that one unit gets M×10 (not ×100).
        # Real-world: ~1% of people experience a major wealth windfall per year.
        if np.random.rand() < self.WD_FACTOR:
            lucky_unit = np.random.randint(0, self.num_units)
            self.state[lucky_unit, 3] *= 10   # ×10 not ×100 — realistic

        # Metabolic drain: living costs scale with stress (high TX → high spending).
        # E is a climate/environment multiplier:
        #   E=0.5 → benign environment (low cost of living pressure)
        #   E=1.0 → neutral (baseline)
        #   E=2.0 → harsh environment (double living costs)
        # Fix: use E directly (not E*2) so E=1.0 is truly neutral.
        drain = (base_metabolism + (self.state[:, 1] * 0.05)) / ((self.state[:, 2] / 50.0)**2)
        self.state[:, 3] -= drain * self.E

    def update_contagion(self, friction=0.2):
        """Spatially propagate stress with per-neighbor friction masking.

        Constitution Axiom 4: same-family neighbors → friction = 0.
        Cross-family neighbors → friction = 0.2.
        """
        s_grid, tx_grid, c_grid, _ = self.get_grids()
        s_baseline = self.S_BASELINE / self.L
        broadcast = (s_grid - s_baseline) * (tx_grid / np.clip(c_grid, 10, None))
        family_grid = self.family_ids.reshape(self.grid_size, self.grid_size)

        neighbor_sum = np.zeros_like(s_grid)
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                neighbor_family = np.roll(family_grid, (i, j), axis=(0, 1))
                eff_friction = np.where(family_grid == neighbor_family, 0.0, friction)
                neighbor_sum += np.roll(broadcast, (i, j), axis=(0, 1)) * eff_friction

        neighbor_contagion = neighbor_sum * (self.X * 2)
        self.state[:, 0] += neighbor_contagion.flatten() / np.clip(self.state[:, 2], 10, None)

    # ---------------------------------------------------------------------- #
    #  Master update                                                          #
    # ---------------------------------------------------------------------- #

    def update(self, cycle):
        # 1. Income pass
        self.apply_income()

        # 2. Family/lifecycle passes before societal stress propagation
        self.apply_aging()
        self.apply_matchmaking(cycle)   # circles pair up -> embedded couples
        self.apply_births(cycle)        # family growth into available slots
        self.apply_launch(cycle)        # young adults leave family system -> circles
        self.apply_divorce(cycle)       # chronic stress can split size-2 families

        # Departed units should not broadcast stress in the same cycle.
        departed_mask = self.unit_status == self.STATUS_DEPARTED
        self.state[departed_mask, 1] = 0.0

        # 3. Stress duration tracking
        over_180 = self.state[:, 0] > 180
        self.high_stress_duration[over_180] += 1
        self.high_stress_duration[~over_180] = 0

        # 4. Core dynamics
        self.update_contagion()
        self.update_tx()
        self.update_c()
        self.update_m()
        self.apply_safety_net()
        self.apply_recovery_curve()

        # 5. Emotional/resource ejections
        self.handle_departures(cycle)
        self.apply_progressive_tax(cycle)

        # 6. Clamp stress
        s_floor = self.S_BASELINE * self.E
        self.state[:, 0] = np.clip(self.state[:, 0], s_floor, 500)

        # 7. Metabolic death (M <= 0, still embedded)
        newly_dead_mask = (self.state[:, 3] <= 0) & (self.unit_status == self.STATUS_EMBEDDED)
        newly_dead_count = int(np.sum(newly_dead_mask))
        if newly_dead_count > 0:
            self.total_deaths += newly_dead_count
            self.unit_status[newly_dead_mask] = self.STATUS_DEPARTED
            self.slot_status[newly_dead_mask] = self.SLOT_DEAD

        # 8. Zero TX on all departed so they don't broadcast
        departed_mask = self.unit_status == self.STATUS_DEPARTED
        self.state[departed_mask, 1] = 0.0

        self.calculate_family_telemetry()
        self.log_telemetry(cycle)

    # ---------------------------------------------------------------------- #
    #  Utilities                                                              #
    # ---------------------------------------------------------------------- #

    def get_grids(self):
        s_grid  = self.state[:, 0].reshape((self.grid_size, self.grid_size))
        tx_grid = self.state[:, 1].reshape((self.grid_size, self.grid_size))
        c_grid  = self.state[:, 2].reshape((self.grid_size, self.grid_size))
        m_grid  = self.state[:, 3].reshape((self.grid_size, self.grid_size))
        return s_grid, tx_grid, c_grid, m_grid

    def _get_birth_slot_groups(self):
        """Return slot groups for newborn activation, ordered by priority."""
        buffer_slots = np.where(self.slot_status == self.SLOT_INACTIVE_BUFFER)[0]
        dead_slots = np.where(self.slot_status == self.SLOT_DEAD)[0]
        departed_slots = np.where(
            (self.slot_status == self.SLOT_DEPARTED) & (self.state[:, 3] < 10)
        )[0]
        if len(departed_slots) == 0:
            departed_slots = np.where(self.slot_status == self.SLOT_DEPARTED)[0]
        return (buffer_slots, dead_slots, departed_slots)

    def _get_birth_candidate_slots(self):
        """Return a flat prioritized list of newborn activation slots."""
        groups = self._get_birth_slot_groups()
        return np.concatenate(groups)

    def reproduce(self):
        pass
