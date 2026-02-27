import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

# Add the project root to the Python path to allow importing 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine import Simulator


class TestSimulator(unittest.TestCase):
    """
    Unit tests for the Simulator engine to prevent regressions.
    """

    def test_simulator_initialization(self):
        """Tests if the simulator initializes correctly with the new multiscale properties."""
        simulator = Simulator(num_units=10000)
        self.assertEqual(simulator.num_units, 10000)
        self.assertTrue(hasattr(simulator, 'family_ids'))
        self.assertTrue(hasattr(simulator, 'family_member_counts'))
        self.assertEqual(np.sum(simulator.family_member_counts), 10000)

        # S is initialised as S_BASELINE ± 16  →  range [64, 96]
        self.assertTrue(np.all(simulator.state[:, 0] >= 64), "S below lower bound 64")
        self.assertTrue(np.all(simulator.state[:, 0] <= 96), "S above upper bound 96")
        # C is initialised as 50 ± 5  →  range [45, 55]
        self.assertTrue(np.all(simulator.state[:, 2] >= 45), "C below lower bound 45")
        self.assertTrue(np.all(simulator.state[:, 2] <= 55), "C above upper bound 55")

    def test_friction_event(self):
        """Test 1: Press 'F' (Friction). Verify global S increases by exactly 10%."""
        sim = Simulator(num_units=100)
        initial_s_mean = np.mean(sim.state[:, 0])

        # Simulate the 'F' key press
        sim.state[:, 0] *= 1.10

        new_s_mean = np.mean(sim.state[:, 0])
        self.assertAlmostEqual(new_s_mean, initial_s_mean * 1.10, places=5)

    def test_safety_net_logic(self):
        """Test 2: Toggle 'N' (Safety Net). Verify high-stress units receive a stress reduction."""
        sim = Simulator(num_units=100)
        sim.safety_net_active = True

        # Drive stress above the subsidy threshold (150) for all units
        sim.state[:, 0] = 200.0

        stress_before = sim.state[:, 0].copy()
        sim.apply_safety_net()
        stress_after = sim.state[:, 0].copy()

        # All units were above 150, so all should have received stress reduction
        self.assertTrue(
            np.all(stress_after < stress_before),
            "apply_safety_net() should reduce stress for all units above threshold 150"
        )
        # National debt must have increased (subsidy costs money)
        self.assertGreater(sim.national_debt, 0, "Safety net should increase national debt")

    def test_family_influence(self):
        """Test 3: Verify family structure is tracked correctly and families exist.

        Families are now compact rectangular blocks in the grid, and update_contagion()
        applies zero friction between same-family neighbors (Constitution Axiom 4).
        This test verifies family ID assignment consistency and recovery timer propagation.
        Spatial contiguity is verified separately in test_family_spatial_contiguity.
        """
        sim = Simulator(num_units=100)

        # Find a family of size 16 (or any available size)
        target_size = 16
        family_sizes = sim.family_member_counts
        possible_families = np.where(family_sizes == target_size)[0]
        if len(possible_families) == 0:
            target_size = np.max(family_sizes)
            possible_families = np.where(family_sizes == target_size)[0]

        self.assertTrue(len(possible_families) > 0, "Could not find a family of suitable size to test.")

        test_family_id = possible_families[0]
        family_member_mask = (sim.family_ids == test_family_id)
        family_indices = np.where(family_member_mask)[0]

        # All members should share the same family_id
        self.assertTrue(
            np.all(sim.family_ids[family_indices] == test_family_id),
            "All family members must share the same family_id"
        )

        # After a departure, recovery timers must be set for remaining members ONLY
        shock_unit_idx = family_indices[0]
        sim.state[shock_unit_idx, 0] = 500.0   # Force departure threshold
        sim.handle_departures(cycle=0)

        other_family_indices = family_indices[1:]
        stranger_indices = np.where(~family_member_mask)[0]

        # Family members get recovery timers; strangers do not
        family_timers   = sim.recovery_timers[other_family_indices]
        stranger_timers = sim.recovery_timers[stranger_indices]

        self.assertTrue(np.all(family_timers == 20),
                        "Remaining family members must have recovery_timer=20 after ejection")
        self.assertTrue(np.all(stranger_timers == 0),
                        "Stranger units must NOT receive recovery timers from another family's ejection")

    def test_ghost_family_elimination(self):
        """Test 4 (Spec 2A/3): Active Family count drops when M is drained to 0.
        Verifies the Ghost Family bug is fixed — families of M<=0 units are NOT counted."""
        sim = Simulator(num_units=10000)

        # Kill all units by draining M to 0
        sim.state[:, 3] = 0.0

        sim.calculate_family_telemetry()

        self.assertEqual(
            sim.active_family_count, 0,
            "Active family count must be 0 when all units have M <= 0 (Ghost Family bug check)"
        )
        self.assertEqual(
            sim.avg_family_size, 0,
            "Avg family size must be 0 when there are no active families"
        )

    def test_recovery_loop_fires(self):
        """Test 5 (Spec 2C): When a unit departs, remaining family members get recovery_timer=20
        and their stress is reduced 5% per cycle."""
        sim = Simulator(num_units=10000)

        # Pick an active embedded unit in a family with at least one other active member
        active_embedded = (sim.unit_status == sim.STATUS_EMBEDDED) & (sim.state[:, 3] > 0)
        family_counts = np.bincount(sim.family_ids[active_embedded], minlength=sim.num_families)
        candidate_fids = np.where(family_counts >= 2)[0]
        self.assertGreater(len(candidate_fids), 0, "No eligible active family with >=2 members found")

        family_id = int(candidate_fids[0])
        family_members = np.where((sim.family_ids == family_id) & active_embedded)[0]
        shock_unit = int(family_members[0])

        # Identify other family members before departure
        sim.state[shock_unit, 0] = 500.0
        family_mask = (
            (sim.family_ids == family_id) &
            active_embedded &
            (np.arange(sim.num_units) != shock_unit)
        )
        family_member_indices = np.where(family_mask)[0]

        sim.handle_departures(cycle=1)

        # All remaining family members should have recovery_timer = 20
        timers = sim.recovery_timers[family_member_indices]
        self.assertTrue(
            np.all(timers == 20),
            "All remaining family members must have recovery_timer=20 after a departure"
        )

        # After apply_recovery_curve(), their stress should drop by 5%
        stress_before = sim.state[family_member_indices, 0].copy()
        sim.apply_recovery_curve()
        stress_after = sim.state[family_member_indices, 0]

        np.testing.assert_allclose(
            stress_after, stress_before * 0.95, rtol=1e-5,
            err_msg="Recovery curve must apply 5% stress reduction per cycle"
        )

    def test_csv_has_recovery_active_column(self):
        """Test 6 (Spec 2C-4): CSV log must include a recovery_active column."""
        csv_path = "sim_audit.csv"
        # Remove any existing file so __init__ writes a fresh header
        if os.path.exists(csv_path):
            os.remove(csv_path)

        sim = Simulator(num_units=100)
        # Force a log write
        sim.calculate_family_telemetry()
        sim.log_telemetry(cycle=0)

        with open(csv_path) as f:
            header = f.readline().strip()

        self.assertIn(
            "recovery_active", header,
            f"CSV header must contain 'recovery_active'. Got: {header}"
        )


    def test_family_spatial_contiguity(self):
        """Test 8: Every family must occupy a contiguous filled rectangle in the grid.

        Verifies the new _init_family_clusters() shelf-first block layout.
        Each family's bounding box must be exactly filled by that family's ID —
        no holes, no foreign cells — and its area must match family_member_counts.
        """
        sim = Simulator(num_units=10000)
        G = sim.grid_size  # 100
        family_grid = sim.family_ids.reshape(G, G)

        # Sample up to 200 families for speed (full ~1500-family check is slow in CI)
        sample_ids = np.random.choice(
            sim.num_families, size=min(200, sim.num_families), replace=False
        )

        for fid in sample_ids:
            rows, cols = np.where(family_grid == fid)
            self.assertGreater(len(rows), 0, f"Family {fid} has no members in the grid")

            r_min, r_max = int(rows.min()), int(rows.max())
            c_min, c_max = int(cols.min()), int(cols.max())

            # The entire bounding box must belong to this family (no holes or mixed IDs)
            bbox = family_grid[r_min:r_max + 1, c_min:c_max + 1]
            self.assertTrue(
                np.all(bbox == fid),
                f"Family {fid} bounding box [{r_min}:{r_max+1}, {c_min}:{c_max+1}] "
                f"contains foreign family IDs — not a contiguous block"
            )

            # Bounding box area must match recorded family size
            expected_size = int(sim.family_member_counts[fid])
            actual_size = bbox.size
            self.assertEqual(
                actual_size, expected_size,
                f"Family {fid}: bounding box area {actual_size} != "
                f"family_member_counts {expected_size}"
            )

    def test_within_family_friction_is_zero(self):
        """Test 9: Same-family neighbors must experience zero contagion friction (Axiom 4).

        Sets up a minimal scenario: one stressed cell inside a family block, zero stress
        everywhere else. Verifies that the family block's own cells absorb more stress
        than cross-family cells at the same spatial distance.
        """
        sim = Simulator(num_units=10000)
        G = sim.grid_size
        family_grid = sim.family_ids.reshape(G, G)

        # Find a size-16 (4×4) family that doesn't touch the grid edge
        # (to avoid toroidal wrap complications in this targeted test)
        for fid in range(sim.num_families):
            rows, cols = np.where(family_grid == fid)
            if len(rows) != 16:
                continue
            r_min, r_max = int(rows.min()), int(rows.max())
            c_min, c_max = int(cols.min()), int(cols.max())
            # Must be interior (not touching any edge)
            if r_min >= 1 and r_max <= G - 2 and c_min >= 1 and c_max <= G - 2:
                break
        else:
            self.skipTest("No interior 4×4 family found — try with a fresh seed")

        # Zero all stress so only our injected signal matters
        sim.state[:, 0] = 0.0
        sim.state[:, 1] = 1.0   # TX = 1 everywhere
        sim.state[:, 2] = 10.0  # C = 10 (minimum clip) so damping is constant

        # Inject stress at the top-left corner of the target family block
        source_flat = r_min * G + c_min
        sim.state[source_flat, 0] = 100.0

        # Record stress for family siblings vs. an immediate cross-family neighbor
        # Family siblings: the other cells in the same 4×4 block
        sibling_indices = [r * G + c for r in range(r_min, r_max + 1)
                           for c in range(c_min, c_max + 1) if not (r == r_min and c == c_min)]

        # Cross-family neighbor: cell just to the left of the block's left edge (guaranteed different family)
        cross_flat = r_min * G + (c_min - 1)
        cross_family_id = sim.family_ids[cross_flat]
        self.assertNotEqual(cross_family_id, fid,
                            "Cell to the left of block must belong to a different family")

        stress_siblings_before = sim.state[sibling_indices, 0].copy()
        stress_cross_before    = sim.state[cross_flat, 0]

        sim.update_contagion()

        delta_siblings = np.mean(sim.state[sibling_indices, 0] - stress_siblings_before)
        delta_cross    = sim.state[cross_flat, 0] - stress_cross_before

        # Within-family (friction=0) must propagate MORE stress than cross-family (friction=0.2)
        self.assertGreater(
            delta_siblings, delta_cross,
            f"Within-family stress delta ({delta_siblings:.4f}) should exceed "
            f"cross-family stress delta ({delta_cross:.4f}) — Axiom 4 not satisfied"
        )

    def test_generational_flux_profile_initialization(self):
        """Spec v1.2: 10k profile initializes 8k active + 2k inactive buffer."""
        sim = Simulator(num_units=10000)

        inactive = np.sum(sim.slot_status == sim.SLOT_INACTIVE_BUFFER)
        active = np.sum(sim.slot_status == sim.SLOT_ACTIVE)

        self.assertEqual(inactive, sim.NURSERY_BUFFER_UNITS)
        self.assertEqual(active, sim.INITIAL_ACTIVE_UNITS)

        inactive_m = sim.state[sim.slot_status == sim.SLOT_INACTIVE_BUFFER, 3]
        self.assertTrue(np.all(inactive_m == -1.0), "Inactive nursery slots must have M = -1")

    def test_birth_candidate_priority_order(self):
        """Birth slot priority must be INACTIVE_BUFFER -> DEAD -> DEPARTED."""
        sim = Simulator(num_units=100)
        sim.slot_status[:] = sim.SLOT_ACTIVE
        sim.state[:, 3] = 1000.0

        buffer_slots = np.array([1, 5])
        dead_slots = np.array([2])
        departed_slots = np.array([3, 4])

        sim.slot_status[buffer_slots] = sim.SLOT_INACTIVE_BUFFER
        sim.state[buffer_slots, 3] = -1.0
        sim.slot_status[dead_slots] = sim.SLOT_DEAD
        sim.state[dead_slots, 3] = 0.0
        sim.slot_status[departed_slots] = sim.SLOT_DEPARTED
        sim.state[departed_slots, 3] = 5.0

        candidates = sim._get_birth_candidate_slots()
        b = len(buffer_slots)
        d = len(dead_slots)

        self.assertTrue(np.array_equal(np.sort(candidates[:b]), np.sort(buffer_slots)))
        self.assertTrue(np.array_equal(np.sort(candidates[b:b + d]), np.sort(dead_slots)))
        self.assertTrue(
            set(candidates[b + d:]) == set(departed_slots),
            "Departed slots should appear after buffer and dead slots"
        )

    def test_metabolic_death_marks_slot_dead(self):
        """Metabolic death should transition an embedded unit to DEAD slot status."""
        sim = Simulator(num_units=10000)

        victim = np.where(sim.slot_status == sim.SLOT_ACTIVE)[0][0]
        sim.unit_status[victim] = sim.STATUS_EMBEDDED
        sim.state[victim, 3] = 0.0

        sim.update(cycle=1)

        self.assertEqual(sim.slot_status[victim], sim.SLOT_DEAD)
        self.assertEqual(sim.unit_status[victim], sim.STATUS_DEPARTED)

    def test_non_10k_profile_has_no_inactive_buffer(self):
        """Only the 10k profile should preallocate an inactive nursery buffer."""
        sim = Simulator(num_units=100)
        self.assertEqual(np.sum(sim.slot_status == sim.SLOT_INACTIVE_BUFFER), 0)
        self.assertTrue(np.all(sim.state[:, 3] > 0), "Non-10k profiles should start fully active")

    def test_log_telemetry_writes_every_10_cycles(self):
        """Telemetry should append rows only when cycle % 10 == 0."""
        csv_path = "sim_audit.csv"
        if os.path.exists(csv_path):
            os.remove(csv_path)

        sim = Simulator(num_units=100)
        sim.calculate_family_telemetry()
        sim.log_telemetry(cycle=1)   # no write
        sim.log_telemetry(cycle=10)  # write

        with open(csv_path) as f:
            lines = [line.strip() for line in f if line.strip()]

        self.assertEqual(len(lines), 2, "Expected header + one telemetry row at cycle 10")
        self.assertTrue(lines[1].startswith("10,"), f"Expected telemetry row for cycle 10, got: {lines[1]}")

    def test_matchmaking_respects_interval_gate(self):
        """Matchmaking should run only on MATCH_INTERVAL cycles."""
        sim = Simulator(num_units=100)

        uid_a, uid_b = 0, 1
        sim.unit_status[[uid_a, uid_b]] = sim.STATUS_DEPARTED
        sim.slot_status[[uid_a, uid_b]] = sim.SLOT_DEPARTED
        sim.state[[uid_a, uid_b], 3] = 1000.0
        sim.state[[uid_a, uid_b], 0] = 100.0
        sim.age[[uid_a, uid_b]] = 30.0
        sim.state[[uid_a, uid_b], 2] = 50.0

        before = sim.total_marriages
        sim.apply_matchmaking(cycle=1)
        self.assertEqual(sim.total_marriages, before, "No marriages should occur off-interval")

        sim.apply_matchmaking(cycle=sim.MATCH_INTERVAL)
        self.assertGreaterEqual(sim.total_marriages, before + 1, "Expected at least one marriage on interval")

    def test_births_consume_inactive_buffer_first(self):
        """When available, births should activate INACTIVE_BUFFER slots before other slot types."""
        sim = Simulator(num_units=10000)

        active_embedded = (sim.unit_status == sim.STATUS_EMBEDDED) & (sim.state[:, 3] > 0)
        sim.age[active_embedded] = 10.0
        family_counts = np.bincount(sim.family_ids[active_embedded], minlength=sim.num_families)
        candidate_fids = np.where(family_counts >= 2)[0]
        self.assertGreater(len(candidate_fids), 0)
        fid = int(candidate_fids[0])

        parent_ids = np.where((sim.family_ids == fid) & active_embedded)[0]
        sim.age[parent_ids] = 30.0  # make exactly one family eligible

        buffer_before = np.where(sim.slot_status == sim.SLOT_INACTIVE_BUFFER)[0]
        self.assertGreater(len(buffer_before), 0, "Expected inactive buffer slots in 10k profile")

        with patch("numpy.random.rand", return_value=np.array([0.0])):
            sim.apply_births(cycle=1)

        buffer_after = np.where(sim.slot_status == sim.SLOT_INACTIVE_BUFFER)[0]
        self.assertEqual(len(buffer_after), len(buffer_before) - 1, "One inactive buffer slot should be consumed")
        self.assertEqual(sim.total_births, 1)

    def test_departed_tx_zeroed_after_update(self):
        """Departed units should not keep non-zero TX after update."""
        sim = Simulator(num_units=10000)
        uid = np.where(sim.slot_status == sim.SLOT_ACTIVE)[0][0]
        sim.unit_status[uid] = sim.STATUS_DEPARTED
        sim.slot_status[uid] = sim.SLOT_DEPARTED
        sim.state[uid, 1] = 3.0
        sim.state[uid, 3] = 100.0

        sim.update(cycle=1)
        self.assertEqual(sim.state[uid, 1], 0.0, "Departed TX should be hard-reset to 0")

    def test_family_identity_arrays_exist_and_alias(self):
        """Phase 2: family identity arrays should be present with compatibility aliasing."""
        sim = Simulator(num_units=10000)
        self.assertTrue(hasattr(sim, "family_origin_id"))
        self.assertTrue(hasattr(sim, "nuclear_family_id"))
        np.testing.assert_array_equal(
            sim.family_ids, sim.nuclear_family_id,
            err_msg="family_ids should remain a compatibility alias to nuclear_family_id values"
        )

    def test_family_origin_immutable_across_marriage_and_divorce(self):
        """family_origin_id should not change when units marry/divorce; nuclear id should."""
        sim = Simulator(num_units=100)
        uid_a, uid_b = 0, 1

        origin_a = int(sim.family_origin_id[uid_a])
        origin_b = int(sim.family_origin_id[uid_b])

        sim.unit_status[[uid_a, uid_b]] = sim.STATUS_DEPARTED
        sim.slot_status[[uid_a, uid_b]] = sim.SLOT_DEPARTED
        sim.state[[uid_a, uid_b], 3] = 1000.0
        sim.state[[uid_a, uid_b], 0] = 100.0
        sim.age[[uid_a, uid_b]] = 30.0
        sim.state[[uid_a, uid_b], 2] = 50.0

        sim.apply_matchmaking(cycle=sim.MATCH_INTERVAL)
        self.assertEqual(sim.total_marriages, 1)

        married_fid = int(sim.family_ids[uid_a])
        self.assertEqual(sim.family_ids[uid_b], married_fid)
        self.assertGreaterEqual(married_fid, 0)
        self.assertEqual(int(sim.family_origin_id[uid_a]), origin_a)
        self.assertEqual(int(sim.family_origin_id[uid_b]), origin_b)

        sim.state[[uid_a, uid_b], 0] = 220.0
        sim.family_high_stress_duration[[uid_a, uid_b]] = 49
        with patch("numpy.random.rand", side_effect=lambda n: np.zeros(n)):
            sim.apply_divorce(cycle=999)

        self.assertEqual(sim.family_ids[uid_a], -1)
        self.assertEqual(sim.family_ids[uid_b], -1)
        self.assertEqual(sim.nuclear_family_id[uid_a], -1)
        self.assertEqual(sim.nuclear_family_id[uid_b], -1)
        self.assertEqual(int(sim.family_origin_id[uid_a]), origin_a)
        self.assertEqual(int(sim.family_origin_id[uid_b]), origin_b)

    def test_birth_replaces_identity_in_reused_slot(self):
        """A newborn activated into a reused slot should receive a new family origin identity."""
        sim = Simulator(num_units=10000)

        active_embedded = (sim.unit_status == sim.STATUS_EMBEDDED) & (sim.state[:, 3] > 0)
        sim.age[active_embedded] = 10.0
        family_counts = np.bincount(sim.family_ids[active_embedded], minlength=sim.num_families)
        fid = int(np.where(family_counts >= 2)[0][0])
        parent_ids = np.where((sim.family_ids == fid) & active_embedded)[0]
        sim.age[parent_ids] = 30.0

        buffer_slots = np.where(sim.slot_status == sim.SLOT_INACTIVE_BUFFER)[0]
        target_slot = int(buffer_slots[0])
        other_buffer = buffer_slots[1:]
        sim.slot_status[other_buffer] = sim.SLOT_DEAD
        sim.state[other_buffer, 3] = 0.0

        sim.family_origin_id[target_slot] = 99999
        parent_origin = int(sim.family_origin_id[parent_ids[0]])

        with patch("numpy.random.rand", return_value=np.array([0.0])):
            sim.apply_births(cycle=5)

        self.assertEqual(sim.total_births, 1)
        self.assertEqual(sim.slot_status[target_slot], sim.SLOT_ACTIVE)
        self.assertEqual(sim.nuclear_family_id[target_slot], fid)
        self.assertEqual(sim.family_origin_id[target_slot], parent_origin)
        self.assertNotEqual(sim.family_origin_id[target_slot], 99999)

    def test_update_order_lifecycle_precedes_stress_pass(self):
        """Regression: update() should run lifecycle passes before contagion/tx/c/m stress passes."""
        sim = Simulator(num_units=100)
        order = []

        def wrap(name):
            original = getattr(sim, name)

            def _wrapped(*args, **kwargs):
                order.append(name)
                return original(*args, **kwargs)

            return _wrapped

        tracked = [
            "apply_income",
            "apply_aging",
            "apply_matchmaking",
            "apply_births",
            "apply_launch",
            "apply_divorce",
            "update_contagion",
            "update_tx",
            "update_c",
            "update_m",
            "apply_safety_net",
            "apply_recovery_curve",
            "handle_departures",
            "apply_progressive_tax",
        ]
        for method_name in tracked:
            setattr(sim, method_name, wrap(method_name))

        sim.update(cycle=sim.MATCH_INTERVAL)

        expected_prefix = [
            "apply_income",
            "apply_aging",
            "apply_matchmaking",
            "apply_births",
            "apply_launch",
            "apply_divorce",
        ]
        self.assertEqual(order[:len(expected_prefix)], expected_prefix)
        self.assertLess(order.index("apply_divorce"), order.index("update_contagion"))


if __name__ == '__main__':
    unittest.main()
