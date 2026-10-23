"""The stock a cut-off tie holds: its weight on the members, its cap and decay, bridging, and repair of a distant or
conflicted tie.

Purpose: give a tie's bound anxiety (``distance_bound_anxiety``, the stock ``s``) the life the owner decided for group
         G of version 2 (`docs/PROPOSAL — CUTOFF ANXIETY, BRIDGING AND REPAIR.md`, decided 2026-10-19): while a tie is
         cut off its stock weighs on both members in proportion to bond energy; it is capped at the cut and decays to a
         floor; a negative contact across the cut tie renews it to its peak and a positive one lets most of it go;
         the tie reopens only on a positive contact once no rejection is held. A distant or conflicted tie keeps the
         stock it had when it hardened as its reference level: each positive contact removes a third of that and each
         negative adds a third back, and at zero the tie is ordinary again. Conflict binds into the tie as distance
         does, at a smaller share.
Spec:    docs/bowen_agent_model_spec_v2.md#M1.D.2b, #M1.D.2c, #M1.B.3c, #M1.B.3, #M4.G.1, #M6.4
Tests:   tests/bowen/test_cutoff_stock.py, tests/bowen/test_tie_repair.py

What each rule changes is what an act does or what a tie holds; none writes a pattern outcome (`model_explainer.md`
§19). Cutoff still comes only from a person choosing ``CUTOFF``; no count of negatives cuts a tie off, and no rule here
selects an act. Every constant is `[I]` (``config/bowen/constants.md``); where a number is the owner's, its row says so.

* **The threat** (``cutoff_threat``, step 1): each week a tie is cut off, each living member carries
  ``cutoff_threat_weight × s × bond_energy / 100`` intensity units, felt as a stressor's arrival is
  (÷ ``intensity_scale`` × their steepness). A source in `M6.4`; it does not draw down ``s``.
* **Entry, the cap and the peak** (``bridge_ties``, step 4, after the move physics): in the week a tie becomes cut off
  a stock above ``cutoff_stock_cap`` is cut to it and the excess dissipates (``cutoff_cap_dissipation``); the peak
  ``P`` is the highest ``s`` since the cut, never above the cap. ``DISTANCE`` and ``CONFLICT`` still bind across a
  cut-off tie, up to the cap (``moves.bind_into_tie``).
* **Decay** (``decay_cutoff_stock``, step 9): ``s ← max(cutoff_floor_share × P, s × f)`` with
  ``f = cutoff_floor_share ** (1 / cutoff_decay_weeks)``; the fall dissipates (``cutoff_decay``).
* **Bridging** (``bridge_ties``): one effect per tie per week, the negative winning (the owner's Q6). Negative — a
  ``negative`` contact delivered (``config/bowen/contact_classes.md``) or any move refused: ``s ← P``
  (``cutoff_threat_renewed``, a source). Positive — a ``positive`` contact delivered:
  ``s ← (1 − positive_bridge_dissipation) × s`` (``cutoff_bridge_dissipation``, a sink); and if no rejection remains
  the tie leaves ``CUT_OFF`` for its ``uncut_state``, keeping ``s``. A ``RECONCILIATION`` is the structural reopening
  and returns the stock to both members (``event_effects``, `M1.D.2a`(c)).
* **Repair** (``bridge_ties``, on a ``DISTANT`` or ``OPEN_CONFLICT`` tie): with ``R`` the reference level recorded at
  entry (``consolidate``), a positive contact removes ``R / tie_repair_steps`` from ``s``, never below 0
  (``tie_repair_dissipation``), and at 0 the tie is ``ORDINARY``; a negative adds the same back, never above ``R``
  (``tie_repair_reversed``, a source). Binding past ``R`` raises ``R`` to the stock. On an ``ORDINARY`` or
  ``RESOLVED_LOW_CONTACT`` tie a positive contact removes ``1 / tie_repair_steps`` of the stock.
"""

from __future__ import annotations

import math

from src.bowen.engine.contact import steepness
from src.bowen.engine.events import Delivery, Mechanism, Role
from src.bowen.engine.identifiers import TieId
from src.bowen.engine.log_records import EffectRecord
from src.bowen.engine.objects import SCALE_MAX, Relationship, TieState
from src.bowen.engine.params import EngineParams
from src.bowen.engine.state import RunState

REPAIRABLE = (TieState.DISTANT, TieState.OPEN_CONFLICT)


def decay_factor(params: EngineParams) -> float:
    return params.cutoff_floor_share ** (1.0 / params.cutoff_decay_weeks)


def enter_cutoff(tie: Relationship, params: EngineParams) -> float:
    """Purpose: the week a tie becomes cut off: cap its stock and set the peak; returns what dissipated.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.D.2b
    Tests:   tests/bowen/test_cutoff_stock.py::test_m1d2b_stock_capped_and_distancer_keeps_the_rest
    """
    excess = max(0.0, tie.distance_bound_anxiety - params.cutoff_stock_cap)
    tie.distance_bound_anxiety -= excess
    tie.cutoff_peak = tie.distance_bound_anxiety
    return excess


def enter_declared_cutoffs(state: RunState, params: EngineParams) -> None:
    """A tie the family declares cut off enters the cut at initialisation, as a cutoff in the run does at its week.
    A declared stock above the cap is a family-file error, refused rather than dissipated unseen (no record can
    carry the loss before the first tick)."""
    for tie_id in sorted(state.ties):
        tie = state.ties[tie_id]
        if tie.tie_state is TieState.CUT_OFF and tie.cutoff_peak is None:
            if tie.distance_bound_anxiety > params.cutoff_stock_cap:
                raise ValueError(f"{tie_id}: declared cut-off tie holds {tie.distance_bound_anxiety} above "
                                 f"cutoff_stock_cap {params.cutoff_stock_cap}")
            enter_cutoff(tie, params)
        elif tie.tie_state in REPAIRABLE:
            tie.reference_level = tie.distance_bound_anxiety


def cutoff_threat(state: RunState, params: EngineParams) -> list[EffectRecord]:
    """Purpose: step 1 — a cut-off tie's stock weighs on both living members, scaled by bond energy (the owner's Q1).
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.D.2b, #M6.4
    Tests:   tests/bowen/test_cutoff_stock.py::test_m1d2b_threat_weighs_only_while_cut_off
    """
    if params.cutoff_threat_weight <= 0:
        return []
    rows = []
    for tie_id in sorted(state.ties):
        tie = state.ties[tie_id]
        if tie.tie_state is not TieState.CUT_OFF or tie.distance_bound_anxiety <= 0:
            continue
        intensity = params.cutoff_threat_weight * tie.distance_bound_anxiety * tie.bond_energy / SCALE_MAX
        for member in tie.id.members():
            person = state.people[member]
            if not person.alive:
                continue
            load = intensity / params.intensity_scale * steepness(person, params)
            if load:
                person.acute_anxiety += load
                rows.append((member, load))
    return [EffectRecord(state.tick, "cutoff_threat", None, acute_anxiety=tuple(rows))] if rows else []


def _contacts_this_week(state: RunState, batch: tuple[Delivery, ...], refusals, kinds) -> tuple[set[TieId], set[TieId]]:
    """The ties that received a negative and a positive contact this week: delivered acts by class, and refusals."""
    negative: set[TieId] = set()
    positive: set[TieId] = set()
    for delivery in batch:
        if delivery.role is not Role.TARGET:
            continue
        event = state.store.event(delivery.event_id)
        if event.mechanism is not Mechanism.MOVE or event.sender is None:
            continue
        if not (state.people[event.sender].alive and state.people[delivery.recipient].alive):
            continue
        tie = state.tie_between(event.sender, delivery.recipient)
        if tie is None:
            continue
        if event.kind in kinds.negative_contacts:
            negative.add(tie.id)
        elif event.kind in kinds.positive_contacts:
            positive.add(tie.id)
    for record in refusals:  # a refusal is a negative contact whatever was refused (M1.B.3c: the sender bears it)
        for tie_id, field, _ in record.ties:
            if field.startswith("rejection:"):
                negative.add(tie_id)
    return negative, positive


def bridge_ties(state: RunState, batch: tuple[Delivery, ...], refusals, params: EngineParams) -> list[EffectRecord]:
    """Purpose: step 4, after the move physics — enter the week's new cutoffs, then apply this week's one bridging or
    repair effect per tie, the negative winning.
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.D.2b, #M1.D.2c, #M1.B.3c, #M6.4
    Tests:   tests/bowen/test_cutoff_stock.py::test_m1d2b_bridging_negative_and_positive,
             tests/bowen/test_tie_repair.py::test_m1d2c_three_positive_contacts_repair
    """
    negative, positive = _contacts_this_week(state, batch, refusals, state.kinds)
    records: list[EffectRecord] = []
    for tie_id in sorted(state.ties):
        tie = state.ties[tie_id]
        s = tie.distance_bound_anxiety
        if tie.tie_state is TieState.CUT_OFF:
            if tie.cutoff_peak is None:  # the week it became cut off
                lost = enter_cutoff(tie, params)
                if lost:
                    records.append(EffectRecord(state.tick, "cutoff_cap_dissipation", None,
                                                ties=((tie_id, "distance_bound_anxiety", -lost),)))
                s = tie.distance_bound_anxiety
            tie.cutoff_peak = max(tie.cutoff_peak, s)
            if tie_id in negative:
                rise = tie.cutoff_peak - s
                if rise > 0:
                    tie.distance_bound_anxiety = tie.cutoff_peak
                    records.append(EffectRecord(state.tick, "cutoff_threat_renewed", None,
                                                ties=((tie_id, "distance_bound_anxiety", rise),)))
            elif tie_id in positive:
                lost = params.positive_bridge_dissipation * s
                rows = []
                if lost > 0:
                    tie.distance_bound_anxiety = s - lost
                    rows.append((tie_id, "distance_bound_anxiety", -lost))
                if not tie.rejected():  # repair: the tie leaves the cut on a positive contact, keeping its stock
                    tie.tie_state = tie.uncut_state
                    tie.cutoff_peak = None
                    if tie.tie_state in REPAIRABLE:  # entering DISTANT or OPEN_CONFLICT records the stock as R (M1.D.2c)
                        tie.reference_level = tie.distance_bound_anxiety
                    rows.append((tie_id, f"tie_state_{tie.tie_state.value}", 1.0))
                if rows:
                    records.append(EffectRecord(state.tick, "cutoff_bridge_dissipation", None, ties=tuple(rows)))
        elif tie.tie_state in REPAIRABLE:
            reference = tie.reference_level
            step = reference / params.tie_repair_steps
            if tie_id in negative:
                added = min(step, max(0.0, reference - s))
                if added > 0:
                    tie.distance_bound_anxiety = s + added
                    records.append(EffectRecord(state.tick, "tie_repair_reversed", None,
                                                ties=((tie_id, "distance_bound_anxiety", added),)))
            elif tie_id in positive:
                removed = min(s, step)
                left = s - removed
                if left <= params.invariant_tolerance:  # the last third: the line is 0
                    removed, left = s, 0.0
                rows = []
                if removed > 0:
                    tie.distance_bound_anxiety = left
                    rows.append((tie_id, "distance_bound_anxiety", -removed))
                if left == 0.0:
                    tie.tie_state = TieState.ORDINARY
                    tie.reference_level = 0.0
                    rows.append((tie_id, "tie_state_ordinary", 1.0))
                if rows:
                    records.append(EffectRecord(state.tick, "tie_repair_dissipation", None, ties=tuple(rows)))
        elif tie_id in positive and tie_id not in negative and s > 0:  # ordinary or resolved low-contact
            removed = s / params.tie_repair_steps
            tie.distance_bound_anxiety = s - removed
            records.append(EffectRecord(state.tick, "tie_repair_dissipation", None,
                                        ties=((tie_id, "distance_bound_anxiety", -removed),)))
    return records


def decay_cutoff_stock(state: RunState, params: EngineParams) -> list[EffectRecord]:
    """Purpose: step 9 — a cut-off tie's stock falls toward its floor, a share of the peak (the owner's 10% at five years).
    Spec:    docs/bowen_agent_model_spec_v2.md#M1.D.2b, #M6.4
    Tests:   tests/bowen/test_cutoff_stock.py::test_m1d2b_decays_to_floor_and_stays
    """
    f = decay_factor(params)
    rows = []
    for tie_id in sorted(state.ties):
        tie = state.ties[tie_id]
        if tie.tie_state is not TieState.CUT_OFF or tie.cutoff_peak is None:
            continue
        s = tie.distance_bound_anxiety
        new = max(params.cutoff_floor_share * tie.cutoff_peak, s * f)
        if new < s:
            tie.distance_bound_anxiety = new
            rows.append((tie_id, "distance_bound_anxiety", new - s))
    return [EffectRecord(state.tick, "cutoff_decay", None, ties=tuple(rows))] if rows else []


def raise_reference(tie: Relationship) -> None:
    """Binding past the reference level of a distant or conflicted tie raises the level to the stock (`M1.D.2c`)."""
    if tie.tie_state in REPAIRABLE and tie.distance_bound_anxiety > tie.reference_level:
        tie.reference_level = tie.distance_bound_anxiety


def total_threat(params: EngineParams, stock: float, bond_energy: float, weeks: int) -> float:
    """The intensity a stock at ``stock`` delivers over ``weeks`` weeks from its peak, decaying with no contact, at
    bond energy ``bond_energy``: the sizing check of G1 §3.7 (a stock at the cap at bond 100 over five years ≈ a
    divorce, 504 intensity units)."""
    f = decay_factor(params)
    floor = params.cutoff_floor_share * stock
    series = math.fsum(max(floor, stock * f ** t) for t in range(weeks))
    return params.cutoff_threat_weight * series * bond_energy / SCALE_MAX
