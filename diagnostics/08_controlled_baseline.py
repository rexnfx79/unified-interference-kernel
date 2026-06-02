#!/usr/bin/env python3
"""
Controlled Baseline Diagnostic

Establishes a rigorous baseline comparison between Gaussian and Clockwork kernels
with IDENTICAL optimizer settings, pre-registered success criteria, and holdout
validation protocol.

This addresses the critique that previous comparisons may have used inconsistent
settings or cherry-picked results.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.optimize import differential_evolution
from alternative_kernels import (
    compute_yukawas_gaussian, compute_yukawas_clockwork,
    KERNELS
)
from observables import compute_quark_observables, QUARK_TARGETS

# =============================================================================
# PRE-REGISTERED SUCCESS CRITERIA (defined BEFORE running)
# =============================================================================

SUCCESS_CRITERIA = {
    'mc': {'target': 1.27, 'tolerance': 0.30},      # within 30%
    'Vus': {'target': 0.225, 'tolerance': 0.20},    # within 20%
    'Vcb': {'target': 0.042, 'tolerance': 0.30},    # within 30%
}

# Holdout observables (NOT used in optimization, only for evaluation)
HOLDOUT_TARGETS = {
    'ms': 0.093,
    'mu': 0.00216,
    'md': 0.00467,
    'Vub': 0.00382,
}

# =============================================================================
# STANDARDIZED OPTIMIZER SETTINGS (identical for all kernels)
# =============================================================================

OPTIMIZER_SETTINGS = {
    'maxiter': 200,
    'popsize': 20,
    'tol': 1e-8,
    'mutation': (0.5, 1.0),
    'recombination': 0.7,
    'polish': True,
}

N_SEEDS = 50
N_GEOMETRIES = 20

# =============================================================================
# GEOMETRY GENERATION (fixed seed for reproducibility)
# =============================================================================

def generate_test_geometries(n_geom, seed=12345):
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
# TRAINING LOSS (what we optimize)
# =============================================================================

def compute_training_loss(obs):
    """
    Training loss on pre-registered targets.
    Uses relative squared error.
    """
    loss = 0.0
    for key, spec in SUCCESS_CRITERIA.items():
        target = spec['target']
        value = obs.get(key, 0.0)
        if key == 'mc' and (value < 0.01 or value > 500):
            return 1000.0  # Invalid
        rel_err = (value - target) / target
        loss += rel_err ** 2
    return loss

# =============================================================================
# HOLDOUT LOSS (NOT used in optimization)
# =============================================================================

def compute_holdout_loss(obs):
    """
    Holdout loss on observables NOT used in training.
    Tests generalization.
    """
    loss = 0.0
    
    # Light masses (log-ratio for scale invariance)
    for key in ['ms', 'mu', 'md']:
        target = HOLDOUT_TARGETS[key]
        value = obs.get(key, 0.0)
        if value > 1e-8:
            loss += np.log(value / target) ** 2
        else:
            loss += 100.0  # Penalty for zero/negative
    
    # V_ub (relative error)
    vub = obs.get('Vub', 0.0)
    if vub > 0:
        loss += ((vub - HOLDOUT_TARGETS['Vub']) / HOLDOUT_TARGETS['Vub']) ** 2
    else:
        loss += 100.0
    
    return loss

# =============================================================================
# CHECK SUCCESS CRITERIA
# =============================================================================

def check_success(obs):
    """Check if solution meets pre-registered success criteria."""
    results = {}
    all_pass = True
    for key, spec in SUCCESS_CRITERIA.items():
        target = spec['target']
        tol = spec['tolerance']
        value = obs.get(key, 0.0)
        rel_err = abs(value - target) / target
        passed = rel_err <= tol
        results[key] = {'value': value, 'target': target, 'rel_err': rel_err, 'passed': passed}
        if not passed:
            all_pass = False
    return all_pass, results

# =============================================================================
# KERNEL-SPECIFIC OBJECTIVE FUNCTIONS
# =============================================================================

def make_gaussian_objective(Q, U, D):
    """Create objective function for Gaussian kernel."""
    bounds = KERNELS['gaussian']['bounds']
    
    def objective(theta):
        try:
            Yu, Yd = compute_yukawas_gaussian(Q, U, D, *theta)
            obs = compute_quark_observables(Yu, Yd)
            return compute_training_loss(obs)
        except:
            return 1000.0
    
    return objective, bounds

def make_clockwork_objective(Q, U, D):
    """Create objective function for Clockwork kernel."""
    bounds = KERNELS['clockwork']['bounds']
    
    def objective(theta):
        try:
            Yu, Yd = compute_yukawas_clockwork(Q, U, D, *theta)
            obs = compute_quark_observables(Yu, Yd)
            return compute_training_loss(obs)
        except:
            return 1000.0
    
    return objective, bounds

# =============================================================================
# MAIN COMPARISON
# =============================================================================

def run_comparison():
    """Run controlled comparison between Gaussian and Clockwork."""
    
    print("=" * 80)
    print("CONTROLLED BASELINE DIAGNOSTIC")
    print("=" * 80)
    
    print("\n--- PRE-REGISTERED SUCCESS CRITERIA ---")
    for key, spec in SUCCESS_CRITERIA.items():
        print(f"  {key}: within {spec['tolerance']*100:.0f}% of {spec['target']}")
    
    print("\n--- HOLDOUT OBSERVABLES (not used in optimization) ---")
    for key, target in HOLDOUT_TARGETS.items():
        print(f"  {key}: target = {target}")
    
    print("\n--- OPTIMIZER SETTINGS (identical for all) ---")
    for key, val in OPTIMIZER_SETTINGS.items():
        print(f"  {key}: {val}")
    print(f"  seeds: {N_SEEDS}")
    print(f"  geometries: {N_GEOMETRIES}")
    
    # Generate test geometries
    geometries = generate_test_geometries(N_GEOMETRIES)
    
    print("\n" + "=" * 80)
    print("RUNNING COMPARISONS...")
    print("=" * 80)
    
    results = {'gaussian': [], 'clockwork': []}
    
    for kernel_name in ['gaussian', 'clockwork']:
        print(f"\n--- {kernel_name.upper()} KERNEL ---")
        
        make_objective = make_gaussian_objective if kernel_name == 'gaussian' else make_clockwork_objective
        compute_yukawas = compute_yukawas_gaussian if kernel_name == 'gaussian' else compute_yukawas_clockwork
        
        for geom_idx, (Q, U, D) in enumerate(geometries):
            objective, bounds = make_objective(Q, U, D)
            
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
                    Yu, Yd = compute_yukawas(Q, U, D, *result.x)
                    best_obs = compute_quark_observables(Yu, Yd)
            
            # Compute holdout loss
            holdout_loss = compute_holdout_loss(best_obs)
            
            # Check success
            success, success_details = check_success(best_obs)
            
            results[kernel_name].append({
                'geometry': (Q, U, D),
                'training_loss': best_loss,
                'holdout_loss': holdout_loss,
                'success': success,
                'obs': best_obs,
                'params': best_params,
            })
            
            status = "PASS" if success else "FAIL"
            print(f"  Geom {geom_idx+1:2d}: train={best_loss:.4f}, holdout={holdout_loss:.2f}, {status}")
    
    # =============================================================================
    # SUMMARY STATISTICS
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    for kernel_name in ['gaussian', 'clockwork']:
        data = results[kernel_name]
        
        train_losses = [r['training_loss'] for r in data]
        holdout_losses = [r['holdout_loss'] for r in data]
        n_success = sum(1 for r in data if r['success'])
        
        print(f"\n--- {kernel_name.upper()} ---")
        print(f"  Training loss:  min={min(train_losses):.4f}, "
              f"median={np.median(train_losses):.4f}, "
              f"mean={np.mean(train_losses):.4f}")
        print(f"  Holdout loss:   min={min(holdout_losses):.2f}, "
              f"median={np.median(holdout_losses):.2f}, "
              f"mean={np.mean(holdout_losses):.2f}")
        print(f"  Success rate:   {n_success}/{N_GEOMETRIES} ({100*n_success/N_GEOMETRIES:.0f}%)")
    
    # =============================================================================
    # DETAILED COMPARISON
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 80)
    
    gaussian_wins_train = 0
    clockwork_wins_train = 0
    gaussian_wins_holdout = 0
    clockwork_wins_holdout = 0
    
    for i in range(N_GEOMETRIES):
        g = results['gaussian'][i]
        c = results['clockwork'][i]
        
        if g['training_loss'] < c['training_loss']:
            gaussian_wins_train += 1
        else:
            clockwork_wins_train += 1
        
        if g['holdout_loss'] < c['holdout_loss']:
            gaussian_wins_holdout += 1
        else:
            clockwork_wins_holdout += 1
    
    print(f"\nTraining loss (lower is better):")
    print(f"  Gaussian wins:  {gaussian_wins_train}/{N_GEOMETRIES}")
    print(f"  Clockwork wins: {clockwork_wins_train}/{N_GEOMETRIES}")
    
    print(f"\nHoldout loss (lower is better):")
    print(f"  Gaussian wins:  {gaussian_wins_holdout}/{N_GEOMETRIES}")
    print(f"  Clockwork wins: {clockwork_wins_holdout}/{N_GEOMETRIES}")
    
    # =============================================================================
    # BEST SOLUTIONS
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("BEST SOLUTIONS (by training loss)")
    print("=" * 80)
    
    for kernel_name in ['gaussian', 'clockwork']:
        data = results[kernel_name]
        best = min(data, key=lambda x: x['training_loss'])
        
        print(f"\n--- {kernel_name.upper()} BEST ---")
        print(f"  Geometry: Q={best['geometry'][0]}, U={best['geometry'][1]}, D={best['geometry'][2]}")
        print(f"  Training loss: {best['training_loss']:.6f}")
        print(f"  Holdout loss:  {best['holdout_loss']:.2f}")
        print(f"  Observables:")
        for key in ['mc', 'Vus', 'Vcb', 'ms', 'mu', 'md', 'Vub']:
            val = best['obs'].get(key, 0.0)
            if key in SUCCESS_CRITERIA:
                target = SUCCESS_CRITERIA[key]['target']
            else:
                target = HOLDOUT_TARGETS.get(key, 0.0)
            if target > 0:
                err = abs(val - target) / target * 100
                print(f"    {key}: {val:.6f} (target: {target}, error: {err:.1f}%)")
    
    # =============================================================================
    # SAVE RESULTS
    # =============================================================================
    
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, '08_controlled_baseline.txt'), 'w') as f:
        f.write("CONTROLLED BASELINE DIAGNOSTIC RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Pre-registered success criteria:\n")
        for key, spec in SUCCESS_CRITERIA.items():
            f.write(f"  {key}: within {spec['tolerance']*100:.0f}% of {spec['target']}\n")
        
        f.write("\nOptimizer settings:\n")
        for key, val in OPTIMIZER_SETTINGS.items():
            f.write(f"  {key}: {val}\n")
        f.write(f"  seeds: {N_SEEDS}\n")
        f.write(f"  geometries: {N_GEOMETRIES}\n")
        
        for kernel_name in ['gaussian', 'clockwork']:
            data = results[kernel_name]
            train_losses = [r['training_loss'] for r in data]
            holdout_losses = [r['holdout_loss'] for r in data]
            n_success = sum(1 for r in data if r['success'])
            
            f.write(f"\n{kernel_name.upper()}:\n")
            f.write(f"  Training: min={min(train_losses):.4f}, median={np.median(train_losses):.4f}\n")
            f.write(f"  Holdout:  min={min(holdout_losses):.2f}, median={np.median(holdout_losses):.2f}\n")
            f.write(f"  Success:  {n_success}/{N_GEOMETRIES}\n")
        
        f.write(f"\nHead-to-head:\n")
        f.write(f"  Training: Gaussian {gaussian_wins_train}, Clockwork {clockwork_wins_train}\n")
        f.write(f"  Holdout:  Gaussian {gaussian_wins_holdout}, Clockwork {clockwork_wins_holdout}\n")
    
    print(f"\nResults saved to diagnostics/results/08_controlled_baseline.txt")
    
    return results

if __name__ == '__main__':
    results = run_comparison()
