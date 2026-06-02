#!/usr/bin/env python3
"""
Minimality Ladder Test

Tests a controlled sequence of model relaxations with explicit penalties
and holdout validation to avoid turning the model into an unfalsifiable
fit machine.

Levels:
0. Base: Shared Q, shared kernel params
1. Shift: +1 param (delta_H for down-type Higgs localization)
2. Width: +1 param (q_u != q_d sector-specific gear ratios)
3. Both: +2 params (delta_H + q split)
4. Full: +3 params (independent Q_u, Q_d positions)

Decision rule:
- Only accept Level N+1 if it improves HOLDOUT loss by >20%
- If it only improves training but not holdout, reject (overfitting)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.optimize import differential_evolution
from alternative_kernels import (
    compute_yukawas_clockwork,
    compute_yukawas_clockwork_shifted,
    compute_yukawas_clockwork_width_split,
    compute_yukawas_clockwork_both,
    compute_yukawas_clockwork_full_split,
    MINIMALITY_LADDER,
)
from observables import (
    compute_quark_observables,
    compute_training_loss,
    compute_holdout_loss,
    compute_penalized_loss,
    TRAINING_TARGETS,
    HOLDOUT_TARGETS,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

OPTIMIZER_SETTINGS = {
    'maxiter': 50,
    'popsize': 8,
    'tol': 1e-5,
    'mutation': (0.5, 1.0),
    'recombination': 0.7,
    'polish': False,  # Skip polishing for speed
}

N_SEEDS = 5
N_GEOMETRIES = 3
LAMBDA_PENALTY = 0.05  # Penalty per extra parameter
HOLDOUT_IMPROVEMENT_THRESHOLD = 0.20  # 20% improvement required

# =============================================================================
# GEOMETRY GENERATION
# =============================================================================

def generate_test_geometries(n_geom, seed=54321):
    """Generate reproducible test geometries."""
    rng = np.random.RandomState(seed)
    geometries = []
    for _ in range(n_geom):
        Q = tuple(sorted(rng.choice(range(15), 3, replace=False)))
        U = tuple(sorted(rng.choice(range(15), 3, replace=False)))
        D = tuple(sorted(rng.choice(range(15), 3, replace=False)))
        geometries.append((Q, U, D))
    return geometries

# =============================================================================
# OBJECTIVE FUNCTIONS FOR EACH LEVEL
# =============================================================================

def make_level0_objective(Q, U, D):
    """Level 0: Base clockwork (shared Q, shared params)."""
    bounds = MINIMALITY_LADDER[0]['bounds']
    
    def objective(theta):
        try:
            Yu, Yd = compute_yukawas_clockwork(Q, U, D, *theta)
            obs = compute_quark_observables(Yu, Yd)
            return compute_training_loss(obs)
        except:
            return 1000.0
    
    return objective, bounds, compute_yukawas_clockwork

def make_level1_objective(Q, U, D):
    """Level 1: Shift (delta_H for down-type)."""
    bounds = MINIMALITY_LADDER[1]['bounds']
    
    def objective(theta):
        try:
            Yu, Yd = compute_yukawas_clockwork_shifted(Q, U, D, *theta)
            obs = compute_quark_observables(Yu, Yd)
            return compute_training_loss(obs)
        except:
            return 1000.0
    
    return objective, bounds, compute_yukawas_clockwork_shifted

def make_level2_objective(Q, U, D):
    """Level 2: Width split (q_u != q_d)."""
    bounds = MINIMALITY_LADDER[2]['bounds']
    
    def objective(theta):
        try:
            Yu, Yd = compute_yukawas_clockwork_width_split(Q, U, D, *theta)
            obs = compute_quark_observables(Yu, Yd)
            return compute_training_loss(obs)
        except:
            return 1000.0
    
    return objective, bounds, compute_yukawas_clockwork_width_split

def make_level3_objective(Q, U, D):
    """Level 3: Both (delta_H + q split)."""
    bounds = MINIMALITY_LADDER[3]['bounds']
    
    def objective(theta):
        try:
            Yu, Yd = compute_yukawas_clockwork_both(Q, U, D, *theta)
            obs = compute_quark_observables(Yu, Yd)
            return compute_training_loss(obs)
        except:
            return 1000.0
    
    return objective, bounds, compute_yukawas_clockwork_both

def make_level4_objective(Q, U, D):
    """Level 4: Full split (independent Q_u, Q_d)."""
    # For level 4, we need to generate Q_d independently
    # We'll use a simple shift strategy: Q_d = Q + random offset
    bounds = MINIMALITY_LADDER[4]['bounds']
    
    # Generate Q_d as a perturbed version of Q
    rng = np.random.RandomState(hash(Q) % 2**31)
    Q_d = tuple(max(0, min(14, q + rng.randint(-3, 4))) for q in Q)
    Q_d = tuple(sorted(set(Q_d)))  # Ensure unique and sorted
    while len(Q_d) < 3:
        Q_d = Q_d + (max(Q_d) + 1,)
    Q_d = Q_d[:3]
    
    def objective(theta):
        try:
            Yu, Yd = compute_yukawas_clockwork_full_split(Q, Q_d, U, D, *theta)
            obs = compute_quark_observables(Yu, Yd)
            return compute_training_loss(obs)
        except:
            return 1000.0
    
    def compute_func(Q_, U_, D_, *args):
        return compute_yukawas_clockwork_full_split(Q_, Q_d, U_, D_, *args)
    
    return objective, bounds, compute_func

LEVEL_MAKERS = {
    0: make_level0_objective,
    1: make_level1_objective,
    2: make_level2_objective,
    3: make_level3_objective,
    4: make_level4_objective,
}

# =============================================================================
# MAIN LADDER TEST
# =============================================================================

def run_minimality_ladder():
    """Run the minimality ladder test."""
    
    print("=" * 80)
    print("MINIMALITY LADDER TEST")
    print("=" * 80)
    
    print("\n--- CONFIGURATION ---")
    print(f"Optimizer: maxiter={OPTIMIZER_SETTINGS['maxiter']}, popsize={OPTIMIZER_SETTINGS['popsize']}")
    print(f"Seeds per geometry: {N_SEEDS}")
    print(f"Geometries: {N_GEOMETRIES}")
    print(f"Penalty per extra param: {LAMBDA_PENALTY}")
    print(f"Holdout improvement threshold: {HOLDOUT_IMPROVEMENT_THRESHOLD*100:.0f}%")
    
    print("\n--- TRAINING TARGETS ---")
    for key, val in TRAINING_TARGETS.items():
        print(f"  {key}: {val}")
    
    print("\n--- HOLDOUT TARGETS (not used in optimization) ---")
    for key, val in HOLDOUT_TARGETS.items():
        print(f"  {key}: {val}")
    
    print("\n--- MINIMALITY LEVELS ---")
    for level, spec in MINIMALITY_LADDER.items():
        print(f"  Level {level}: {spec['name']} (+{spec['extra_params']} params)")
    
    # Generate test geometries
    geometries = generate_test_geometries(N_GEOMETRIES)
    
    # Results storage
    all_results = {level: [] for level in range(5)}
    
    print("\n" + "=" * 80)
    print("RUNNING OPTIMIZATION FOR EACH LEVEL...")
    print("=" * 80)
    
    for level in range(5):
        spec = MINIMALITY_LADDER[level]
        print(f"\n--- LEVEL {level}: {spec['name']} ---")
        
        for geom_idx, (Q, U, D) in enumerate(geometries):
            objective, bounds, compute_func = LEVEL_MAKERS[level](Q, U, D)
            
            best_loss = float('inf')
            best_params = None
            best_obs = None
            
            for seed in range(N_SEEDS):
                result = differential_evolution(
                    objective, bounds,
                    seed=seed,
                    **OPTIMIZER_SETTINGS
                )
                
                if result.fun < best_loss:
                    best_loss = result.fun
                    best_params = result.x
                    
                    # Compute observables
                    if level == 4:
                        # Level 4 needs special handling for Q_d
                        rng = np.random.RandomState(hash(Q) % 2**31)
                        Q_d = tuple(max(0, min(14, q + rng.randint(-3, 4))) for q in Q)
                        Q_d = tuple(sorted(set(Q_d)))
                        while len(Q_d) < 3:
                            Q_d = Q_d + (max(Q_d) + 1,)
                        Q_d = Q_d[:3]
                        Yu, Yd = compute_yukawas_clockwork_full_split(Q, Q_d, U, D, *result.x)
                    else:
                        Yu, Yd = compute_func(Q, U, D, *result.x)
                    best_obs = compute_quark_observables(Yu, Yd)
            
            # Compute losses
            train_loss = compute_training_loss(best_obs)
            holdout_loss = compute_holdout_loss(best_obs)
            penalized_loss = compute_penalized_loss(best_obs, spec['extra_params'], LAMBDA_PENALTY)
            
            all_results[level].append({
                'geometry': (Q, U, D),
                'train_loss': train_loss,
                'holdout_loss': holdout_loss,
                'penalized_loss': penalized_loss,
                'obs': best_obs,
                'params': best_params,
            })
            
            print(f"  Geom {geom_idx+1:2d}: train={train_loss:.4f}, holdout={holdout_loss:.2f}, penalized={penalized_loss:.4f}")
    
    # =============================================================================
    # SUMMARY STATISTICS
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS BY LEVEL")
    print("=" * 80)
    
    summary = {}
    
    for level in range(5):
        spec = MINIMALITY_LADDER[level]
        data = all_results[level]
        
        train_losses = [r['train_loss'] for r in data]
        holdout_losses = [r['holdout_loss'] for r in data]
        penalized_losses = [r['penalized_loss'] for r in data]
        
        summary[level] = {
            'train_median': np.median(train_losses),
            'train_mean': np.mean(train_losses),
            'holdout_median': np.median(holdout_losses),
            'holdout_mean': np.mean(holdout_losses),
            'penalized_median': np.median(penalized_losses),
        }
        
        print(f"\nLevel {level}: {spec['name']} (+{spec['extra_params']} params)")
        print(f"  Training:  median={np.median(train_losses):.4f}, mean={np.mean(train_losses):.4f}")
        print(f"  Holdout:   median={np.median(holdout_losses):.2f}, mean={np.mean(holdout_losses):.2f}")
        print(f"  Penalized: median={np.median(penalized_losses):.4f}")
    
    # =============================================================================
    # DECISION ANALYSIS
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("DECISION ANALYSIS")
    print("=" * 80)
    
    print("\nComparing each level to Level 0 (base):")
    base_holdout = summary[0]['holdout_median']
    
    decisions = {}
    
    for level in range(1, 5):
        spec = MINIMALITY_LADDER[level]
        level_holdout = summary[level]['holdout_median']
        
        # Calculate improvement
        if base_holdout > 0:
            improvement = (base_holdout - level_holdout) / base_holdout
        else:
            improvement = 0.0
        
        # Decision
        if improvement > HOLDOUT_IMPROVEMENT_THRESHOLD:
            decision = "ACCEPT"
            reason = f"Holdout improved by {improvement*100:.1f}% > {HOLDOUT_IMPROVEMENT_THRESHOLD*100:.0f}%"
        elif improvement > 0:
            decision = "MARGINAL"
            reason = f"Holdout improved by {improvement*100:.1f}% < {HOLDOUT_IMPROVEMENT_THRESHOLD*100:.0f}%"
        else:
            decision = "REJECT"
            reason = f"Holdout degraded by {-improvement*100:.1f}%"
        
        decisions[level] = {'decision': decision, 'improvement': improvement, 'reason': reason}
        
        print(f"\nLevel {level} ({spec['name']}):")
        print(f"  Holdout: {base_holdout:.2f} -> {level_holdout:.2f}")
        print(f"  Improvement: {improvement*100:+.1f}%")
        print(f"  Decision: {decision}")
        print(f"  Reason: {reason}")
    
    # =============================================================================
    # OVERFITTING CHECK
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("OVERFITTING CHECK")
    print("=" * 80)
    
    print("\nTrain-Holdout gap (larger gap = more overfitting):")
    
    for level in range(5):
        spec = MINIMALITY_LADDER[level]
        train_med = summary[level]['train_median']
        holdout_med = summary[level]['holdout_median']
        gap = holdout_med - train_med
        
        print(f"  Level {level}: train={train_med:.4f}, holdout={holdout_med:.2f}, gap={gap:.2f}")
    
    # Check if gap grows with complexity
    gaps = [summary[level]['holdout_median'] - summary[level]['train_median'] for level in range(5)]
    if gaps[-1] > gaps[0] * 1.5:
        print("\nWARNING: Train-holdout gap grows significantly with model complexity.")
        print("This suggests overfitting at higher levels.")
    else:
        print("\nGap does not grow significantly - no strong overfitting signal.")
    
    # =============================================================================
    # BEST OBSERVABLE VALUES
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("BEST SOLUTIONS BY LEVEL")
    print("=" * 80)
    
    for level in range(5):
        spec = MINIMALITY_LADDER[level]
        data = all_results[level]
        best = min(data, key=lambda x: x['train_loss'])
        
        print(f"\nLevel {level}: {spec['name']}")
        print(f"  Best training loss: {best['train_loss']:.6f}")
        print(f"  Holdout loss: {best['holdout_loss']:.2f}")
        print(f"  Observables:")
        
        # Training targets
        for key in ['mc', 'Vus', 'Vcb']:
            val = best['obs'].get(key, 0.0)
            target = TRAINING_TARGETS[key]
            err = abs(val - target) / target * 100 if target > 0 else 0
            print(f"    {key}: {val:.4f} (target: {target}, err: {err:.1f}%) [TRAIN]")
        
        # Holdout targets
        for key in ['ms', 'mu', 'md', 'Vub']:
            val = best['obs'].get(key, 0.0)
            target = HOLDOUT_TARGETS[key]
            if target > 0 and val > 0:
                err = abs(val - target) / target * 100
            else:
                err = float('inf')
            print(f"    {key}: {val:.6f} (target: {target}, err: {err:.1f}%) [HOLDOUT]")
    
    # =============================================================================
    # FINAL RECOMMENDATION
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)
    
    # Find the best level that passes the decision criteria
    best_level = 0
    for level in range(1, 5):
        if decisions[level]['decision'] == 'ACCEPT':
            best_level = level
    
    if best_level == 0:
        print("\nRECOMMENDATION: Stay with Level 0 (base model)")
        print("No relaxation provides sufficient holdout improvement to justify extra parameters.")
        print("\nConclusion: The shared-Q constraint is NOT the primary bottleneck.")
        print("The kernel functional form itself may be insufficient.")
    else:
        spec = MINIMALITY_LADDER[best_level]
        print(f"\nRECOMMENDATION: Use Level {best_level} ({spec['name']})")
        print(f"This level provides {decisions[best_level]['improvement']*100:.1f}% holdout improvement")
        print(f"with only +{spec['extra_params']} extra parameter(s).")
        print("\nConclusion: The shared-Q constraint IS a bottleneck that can be")
        print("partially addressed with physics-motivated relaxation.")
    
    # =============================================================================
    # SAVE RESULTS
    # =============================================================================
    
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, '09_minimality_ladder.txt'), 'w') as f:
        f.write("MINIMALITY LADDER TEST RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Seeds: {N_SEEDS}, Geometries: {N_GEOMETRIES}\n")
        f.write(f"  Penalty: {LAMBDA_PENALTY}, Threshold: {HOLDOUT_IMPROVEMENT_THRESHOLD}\n\n")
        
        f.write("Summary by level:\n")
        for level in range(5):
            spec = MINIMALITY_LADDER[level]
            s = summary[level]
            f.write(f"  Level {level} ({spec['name']}):\n")
            f.write(f"    Train median: {s['train_median']:.4f}\n")
            f.write(f"    Holdout median: {s['holdout_median']:.2f}\n")
            f.write(f"    Penalized median: {s['penalized_median']:.4f}\n")
        
        f.write("\nDecisions:\n")
        for level in range(1, 5):
            d = decisions[level]
            f.write(f"  Level {level}: {d['decision']} ({d['reason']})\n")
        
        f.write(f"\nFinal recommendation: Level {best_level}\n")
    
    print(f"\nResults saved to diagnostics/results/09_minimality_ladder.txt")
    
    return all_results, summary, decisions

if __name__ == '__main__':
    results, summary, decisions = run_minimality_ladder()
