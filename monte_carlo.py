"""
Monte Carlo Parameter Sweep — EPModel Societal Inflammatory Simulator
======================================================================
Tests ~150 random parameter combinations × 2 replicates to identify
which settings prevent population collapse (100% deaths).

Parameters swept:
  L  (Leadership)        : [0.5, 1.0, 1.5, 2.0]
  safety_net             : [True, False]
  S_BASELINE             : [60, 80, 100, 120]
  unemployment_rate      : [0.00, 0.05, 0.10, 0.20]
  X  (Social Media)      : [0.5, 1.0, 1.5, 2.0]
  E  (Climate/Environ)   : [0.5, 1.0, 1.5, 2.0]

Metric: survival_rate = % of units with M > 0 at cycle 200.
"""

import os, sys, time, itertools, random
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from src.engine import Simulator

# ──────────────────────────────────────────────────────────────────────────────
# Parameter grid
# ──────────────────────────────────────────────────────────────────────────────
L_VALUES    = [0.5, 1.0, 1.5, 2.0]
SN_VALUES   = [True, False]
SB_VALUES   = [60, 80, 100, 120]
UR_VALUES   = [0.00, 0.05, 0.10, 0.20]
X_VALUES    = [0.5, 1.0, 1.5, 2.0]
E_VALUES    = [0.5, 1.0, 1.5, 2.0]

ALL_COMBOS = list(itertools.product(
    L_VALUES, SN_VALUES, SB_VALUES, UR_VALUES, X_VALUES, E_VALUES
))
print(f"Full parameter space: {len(ALL_COMBOS):,} combinations")

# Sample 150 random combos
np.random.seed(42)
random.seed(42)
SAMPLE_SIZE   = 150
REPLICATES    = 2
CYCLES        = 200
NUM_UNITS     = 10000

sampled_idx   = np.random.choice(len(ALL_COMBOS), size=SAMPLE_SIZE, replace=False)
sampled_combos = [ALL_COMBOS[i] for i in sampled_idx]
print(f"Sampled: {SAMPLE_SIZE} combos × {REPLICATES} replicates = "
      f"{SAMPLE_SIZE * REPLICATES} simulation runs @ {CYCLES} cycles each\n")

# Silence CSV creation (redirect to /dev/null workaround: patch os.path.exists)
_real_exists = os.path.exists
def _suppress_csv(path):
    if "sim_audit.csv" in str(path):
        return True   # pretend it exists → engine won't write a new header
    return _real_exists(path)
os.path.exists = _suppress_csv
# Also patch open so CSV writes go nowhere
import builtins
_real_open = builtins.open
def _muted_open(path, mode='r', **kw):
    if "sim_audit.csv" in str(path) and ('w' in mode or 'a' in mode):
        import io
        return io.StringIO()
    return _real_open(path, mode, **kw)
builtins.open = _muted_open

# ──────────────────────────────────────────────────────────────────────────────
# Run simulations
# ──────────────────────────────────────────────────────────────────────────────
results = []
t0 = time.time()

for run_idx, (L, sn, sb, ur, X, E) in enumerate(sampled_combos):
    rep_survival = []
    rep_deaths   = []
    rep_births   = []
    rep_marriages = []
    rep_avg_s    = []
    rep_families = []

    for rep in range(REPLICATES):
        np.random.seed(run_idx * 100 + rep)

        sim = Simulator(NUM_UNITS)
        sim.L                = L
        sim.safety_net_active = sn
        sim.S_BASELINE       = sb
        sim.unemployment_rate = ur
        sim.X                = X
        sim.E                = E
        # Reinitialise S with new baseline (S was set during __init__ before we changed it)
        sim.state[:, 0]      = sb + np.random.uniform(-16, 16, NUM_UNITS)

        for cycle in range(CYCLES):
            sim.update(cycle)

        live_mask    = sim.state[:, 3] > 0
        live_count   = int(np.sum(live_mask))
        survival_pct = live_count / NUM_UNITS * 100.0
        avg_s        = float(np.mean(sim.state[live_mask, 0])) if live_count > 0 else 0.0

        rep_survival.append(survival_pct)
        rep_deaths.append(sim.total_deaths)
        rep_births.append(sim.total_births)
        rep_marriages.append(sim.total_marriages)
        rep_avg_s.append(avg_s)
        rep_families.append(sim.active_family_count)

    results.append({
        "L":            L,
        "safety_net":   sn,
        "S_BASELINE":   sb,
        "unemployment": ur,
        "X":            X,
        "E":            E,
        "survival_rate": np.mean(rep_survival),
        "survival_std":  np.std(rep_survival),
        "total_deaths":  np.mean(rep_deaths),
        "total_births":  np.mean(rep_births),
        "total_marriages": np.mean(rep_marriages),
        "avg_stress":    np.mean(rep_avg_s),
        "active_families": np.mean(rep_families),
        "collapsed":     np.mean(rep_survival) < 5.0,
    })

    if (run_idx + 1) % 10 == 0:
        elapsed = time.time() - t0
        eta     = elapsed / (run_idx + 1) * (SAMPLE_SIZE - run_idx - 1)
        print(f"  [{run_idx+1:3d}/{SAMPLE_SIZE}] done — "
              f"last survival={results[-1]['survival_rate']:5.1f}% — "
              f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s")

elapsed_total = time.time() - t0
print(f"\nAll {SAMPLE_SIZE * REPLICATES} runs complete in {elapsed_total:.1f}s\n")

# ──────────────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────────────
results.sort(key=lambda r: -r["survival_rate"])

survived_runs  = [r for r in results if r["survival_rate"] > 50]
collapsed_runs = [r for r in results if r["collapsed"]]
pct_survive    = len(survived_runs) / len(results) * 100

lines = []
def p(s=""):
    print(s)
    lines.append(s)

p("=" * 72)
p("  EPModel Monte Carlo Parameter Sweep — Results Summary")
p(f"  {SAMPLE_SIZE} combos × {REPLICATES} replicates × {CYCLES} cycles | {NUM_UNITS:,} units")
p("=" * 72)

p(f"\n{'─'*72}")
p(f"  SURVIVAL OVERVIEW")
p(f"{'─'*72}")
p(f"  Runs with >50% population alive at cycle {CYCLES}: "
  f"{len(survived_runs)}/{SAMPLE_SIZE}  ({pct_survive:.0f}%)")
p(f"  Runs with total collapse (<5% alive):              "
  f"{len(collapsed_runs)}/{SAMPLE_SIZE}  ({len(collapsed_runs)/len(results)*100:.0f}%)")
p(f"  Mean survival rate across all runs:                "
  f"{np.mean([r['survival_rate'] for r in results]):.1f}%")

# ── TOP 10 ────────────────────────────────────────────────────────────────────
p(f"\n{'─'*72}")
p(f"  TOP 10 PARAMETER COMBOS  (best survival)")
p(f"{'─'*72}")
p(f"  {'Surv%':>6}  {'L':>4}  {'SNet':>5}  {'SBase':>6}  {'Unempl':>7}  {'X':>4}  {'E':>4}  {'Births':>7}  {'Deaths':>7}")
for r in results[:10]:
    p(f"  {r['survival_rate']:6.1f}%  {r['L']:4.1f}  "
      f"{'YES' if r['safety_net'] else 'no':>5}  "
      f"{r['S_BASELINE']:6.0f}  "
      f"{r['unemployment']*100:6.0f}%  "
      f"{r['X']:4.1f}  {r['E']:4.1f}  "
      f"{r['total_births']:7.0f}  {r['total_deaths']:7.0f}")

# ── BOTTOM 10 ─────────────────────────────────────────────────────────────────
p(f"\n{'─'*72}")
p(f"  BOTTOM 10 PARAMETER COMBOS  (worst / most deadly)")
p(f"{'─'*72}")
p(f"  {'Surv%':>6}  {'L':>4}  {'SNet':>5}  {'SBase':>6}  {'Unempl':>7}  {'X':>4}  {'E':>4}  {'Births':>7}  {'Deaths':>7}")
for r in results[-10:]:
    p(f"  {r['survival_rate']:6.1f}%  {r['L']:4.1f}  "
      f"{'YES' if r['safety_net'] else 'no':>5}  "
      f"{r['S_BASELINE']:6.0f}  "
      f"{r['unemployment']*100:6.0f}%  "
      f"{r['X']:4.1f}  {r['E']:4.1f}  "
      f"{r['total_births']:7.0f}  {r['total_deaths']:7.0f}")

# ── IMPACT ANALYSIS ───────────────────────────────────────────────────────────
p(f"\n{'─'*72}")
p(f"  PARAMETER IMPACT ANALYSIS  (mean survival at each level)")
p(f"{'─'*72}")

def impact_table(param_key, values, label):
    p(f"\n  {label}:")
    for v in values:
        subset = [r for r in results if r[param_key] == v]
        if subset:
            mean_s  = np.mean([r["survival_rate"] for r in subset])
            pct_ok  = sum(1 for r in subset if r["survival_rate"] > 50) / len(subset) * 100
            p(f"    {str(v):>8}  →  mean survival {mean_s:5.1f}%   "
              f"(>{50}% alive in {pct_ok:.0f}% of runs)  n={len(subset)}")

impact_table("L",            L_VALUES,  "L  — Leadership Factor")
impact_table("safety_net",   SN_VALUES, "Safety Net (True/False)")
impact_table("S_BASELINE",   SB_VALUES, "S_BASELINE — Baseline Stress")
impact_table("unemployment", UR_VALUES, "Unemployment Rate")
impact_table("X",            X_VALUES,  "X  — Social Media Amplifier")
impact_table("E",            E_VALUES,  "E  — Climate / Environment")

# ── PAIRWISE: L × Safety Net ─────────────────────────────────────────────────
p(f"\n{'─'*72}")
p(f"  LEADERSHIP × SAFETY NET  (mean survival %)")
p(f"{'─'*72}")
header = f"  {'':13}" + "".join(f"  L={l:<4}" for l in L_VALUES)
p(header)
for sn in SN_VALUES:
    label = "SafetyNet=ON " if sn else "SafetyNet=OFF"
    row = f"  {label}"
    for L in L_VALUES:
        subset = [r for r in results if r["safety_net"] == sn and r["L"] == L]
        mean_s = np.mean([r["survival_rate"] for r in subset]) if subset else float('nan')
        row += f"  {mean_s:5.1f}%"
    p(row)

# ── KEY FINDING ───────────────────────────────────────────────────────────────
p(f"\n{'─'*72}")
p(f"  KEY FINDINGS — MINIMUM VIABLE PARAMETERS")
p(f"{'─'*72}")

# High survivors
high_surv = [r for r in results if r["survival_rate"] > 80]
if high_surv:
    L_hs    = np.mean([r["L"] for r in high_surv])
    sn_hs   = np.mean([r["safety_net"] for r in high_surv])
    sb_hs   = np.mean([r["S_BASELINE"] for r in high_surv])
    ur_hs   = np.mean([r["unemployment"] for r in high_surv])
    X_hs    = np.mean([r["X"] for r in high_surv])
    E_hs    = np.mean([r["E"] for r in high_surv])
    p(f"\n  Runs achieving >80% survival rate: {len(high_surv)}/{SAMPLE_SIZE}")
    p(f"  Their average parameters:")
    p(f"    L (Leadership)     : {L_hs:.2f}   (higher = better)")
    p(f"    Safety Net ON      : {sn_hs*100:.0f}% of those runs had it active")
    p(f"    S_BASELINE         : {sb_hs:.1f}   (lower baseline stress = better)")
    p(f"    Unemployment       : {ur_hs*100:.1f}%  (lower = better)")
    p(f"    X (Social Media)   : {X_hs:.2f}   (lower = better)")
    p(f"    E (Climate)        : {E_hs:.2f}   (lower = better)")
else:
    p("  No runs achieved >80% survival in this sample.")

# Collapse parameters
if collapsed_runs:
    L_c  = np.mean([r["L"] for r in collapsed_runs])
    sn_c = np.mean([r["safety_net"] for r in collapsed_runs])
    sb_c = np.mean([r["S_BASELINE"] for r in collapsed_runs])
    ur_c = np.mean([r["unemployment"] for r in collapsed_runs])
    X_c  = np.mean([r["X"] for r in collapsed_runs])
    E_c  = np.mean([r["E"] for r in collapsed_runs])
    p(f"\n  Collapsed runs (<5% alive): {len(collapsed_runs)}/{SAMPLE_SIZE}")
    p(f"  Their average parameters (what to AVOID):")
    p(f"    L (Leadership)     : {L_c:.2f}")
    p(f"    Safety Net ON      : {sn_c*100:.0f}% had it active")
    p(f"    S_BASELINE         : {sb_c:.1f}")
    p(f"    Unemployment       : {ur_c*100:.1f}%")
    p(f"    X (Social Media)   : {X_c:.2f}")
    p(f"    E (Climate)        : {E_c:.2f}")

# ── RECOMMENDED BASELINE ──────────────────────────────────────────────────────
p(f"\n{'─'*72}")
p(f"  RECOMMENDED DEFAULT PARAMETERS  (conservative safe zone)")
p(f"{'─'*72}")

# Best overall combo
best = results[0]
p(f"\n  Best single combo found:")
p(f"    Survival rate  : {best['survival_rate']:.1f}% (±{best['survival_std']:.1f}%)")
p(f"    L (Leadership) : {best['L']}")
p(f"    Safety Net     : {'ON' if best['safety_net'] else 'OFF'}")
p(f"    S_BASELINE     : {best['S_BASELINE']}")
p(f"    Unemployment   : {best['unemployment']*100:.0f}%")
p(f"    X (Media)      : {best['X']}")
p(f"    E (Climate)    : {best['E']}")
p(f"    Avg Stress     : {best['avg_stress']:.1f}")
p(f"    Births         : {best['total_births']:.0f}")
p(f"    Deaths         : {best['total_deaths']:.0f}")

p(f"\n  Total runtime: {elapsed_total:.1f}s")
p("=" * 72)

# ── SAVE TO FILE ──────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), "monte_carlo_results.txt")
builtins.open = _real_open   # restore real open for writing results
with open(out_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nResults saved to: {out_path}")
