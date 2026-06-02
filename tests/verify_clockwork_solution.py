#!/usr/bin/env python3
"""
Independent Verification of Clockwork Solution

This script independently verifies the claimed Clockwork kernel solution
by re-deriving it from scratch with a fresh optimization.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.optimize import differential_evolution, minimize
from alternative_kernels import compute_yukawas_clockwork
from observables import compute_quark_observables, QUARK_TARGETS

np.random.seed(2024)  # Different seed from original analysis

print("="*70)
print("INDEPENDENT VERIFICATION OF CLOCKWORK SOLUTION")
print("="*70)

# Target values
TARGETS = {
    'mc': 1.27,
    'Vus': 0.225,
    'Vcb': 0.04182,
    'Vub': 0.00382,
}

print("\nTarget values:")
for key, val in TARGETS.items():
    print(f"  {key}: {val}")

# Claimed geometry
Q_claimed = (7, 8, 9)
U_claimed = (2, 12, 14)
D_claimed = (1, 4, 7)

# Claimed parameters
params_claimed = {
    'q': 11.636058,
    'k': 3.057852,
    'alpha': 0.627795,
    'eta': 1.486743,
    'eps_u': 0.997599,
    'eps_d': 0.999944,
}

print("\n" + "="*70)
print("STEP 1: Verify claimed solution")
print("="*70)

Yu, Yd = compute_yukawas_clockwork(
    Q_claimed, U_claimed, D_claimed,
    params_claimed['q'], params_claimed['k'], params_claimed['alpha'],
    params_claimed['eta'], params_claimed['eps_u'], params_claimed['eps_d']
)
obs_claimed = compute_quark_observables(Yu, Yd)

print("\nClaimed solution observables:")
all_within_1pct = True
for key in ['mc', 'Vus', 'Vcb', 'Vub']:
    target = TARGETS[key]
    actual = obs_claimed[key] if key != 'Vcb' else obs_claimed['Vcb']
    error = abs(actual - target) / target * 100
    status = "✓" if error < 1 else "✗"
    print(f"  {key}: {actual:.6f} (target: {target}, error: {error:.4f}%) {status}")
    if error >= 1:
        all_within_1pct = False

print(f"\nClaimed solution verification: {'PASSED' if all_within_1pct else 'FAILED'}")

print("\n" + "="*70)
print("STEP 2: Independent re-optimization")
print("="*70)

# Use the same geometry but find parameters independently
bounds = [
    (5.0, 15.0),   # q
    (0.001, 10.0), # k
    (0.0, 2*np.pi), # alpha
    (0.001, 15.0), # eta
    (0.001, 2.0),  # eps_u
    (0.001, 2.0),  # eps_d
]

def objective(theta):
    try:
        Yu, Yd = compute_yukawas_clockwork(Q_claimed, U_claimed, D_claimed, *theta)
        obs = compute_quark_observables(Yu, Yd)
        
        if obs['mc'] < 0.01 or obs['mc'] > 500:
            return 1000.0
        
        # Squared relative errors
        mc_err = ((obs['mc'] - 1.27) / 1.27)**2
        vus_err = ((obs['Vus'] - 0.225) / 0.225)**2
        vcb_err = ((obs['Vcb'] - 0.042) / 0.042)**2
        vub_err = ((obs['Vub'] - 0.00382) / 0.00382)**2
        
        return mc_err + vus_err + vcb_err + vub_err
    except:
        return 1000.0

print("\nRunning independent optimization (20 seeds)...")

best_loss = float('inf')
best_params = None

for seed in range(20):
    result = differential_evolution(
        objective, bounds, maxiter=200, seed=seed + 1000, polish=True
    )
    if result.fun < best_loss:
        best_loss = result.fun
        best_params = result.x

print(f"Best loss found: {best_loss:.10f}")

# Verify independent solution
Yu, Yd = compute_yukawas_clockwork(Q_claimed, U_claimed, D_claimed, *best_params)
obs_independent = compute_quark_observables(Yu, Yd)

print("\nIndependently optimized solution:")
all_within_1pct_independent = True
for key in ['mc', 'Vus', 'Vcb', 'Vub']:
    target = TARGETS[key]
    actual = obs_independent[key] if key != 'Vcb' else obs_independent['Vcb']
    error = abs(actual - target) / target * 100
    status = "✓" if error < 1 else "✗"
    print(f"  {key}: {actual:.6f} (target: {target}, error: {error:.4f}%) {status}")
    if error >= 1:
        all_within_1pct_independent = False

print(f"\nIndependent solution verification: {'PASSED' if all_within_1pct_independent else 'FAILED'}")

print("\nIndependently found parameters:")
param_names = ['q', 'k', 'alpha', 'eta', 'eps_u', 'eps_d']
for name, val in zip(param_names, best_params):
    claimed_val = params_claimed[name]
    diff = abs(val - claimed_val)
    print(f"  {name}: {val:.6f} (claimed: {claimed_val:.6f}, diff: {diff:.6f})")

print("\n" + "="*70)
print("STEP 3: Search for alternative geometries")
print("="*70)

print("\nSearching for other geometries that work (1000 random samples)...")

good_geometries = []

for trial in range(1000):
    # Random geometry
    Q = tuple(sorted(np.random.choice(range(15), 3, replace=False)))
    U = tuple(sorted(np.random.choice(range(15), 3, replace=False)))
    D = tuple(sorted(np.random.choice(range(15), 3, replace=False)))
    
    # Quick optimization
    def obj(theta):
        try:
            Yu, Yd = compute_yukawas_clockwork(Q, U, D, *theta)
            obs = compute_quark_observables(Yu, Yd)
            if obs['mc'] < 0.01 or obs['mc'] > 500:
                return 1000.0
            return ((obs['mc'] - 1.27) / 1.27)**2 + \
                   ((obs['Vus'] - 0.225) / 0.225)**2 + \
                   ((obs['Vcb'] - 0.042) / 0.042)**2
        except:
            return 1000.0
    
    result = differential_evolution(obj, bounds, maxiter=50, seed=trial, polish=False)
    
    if result.fun < 0.1:  # Good solution
        Yu, Yd = compute_yukawas_clockwork(Q, U, D, *result.x)
        obs = compute_quark_observables(Yu, Yd)
        good_geometries.append({
            'Q': Q, 'U': U, 'D': D,
            'loss': result.fun,
            'mc': obs['mc'],
            'Vus': obs['Vus'],
            'Vcb': obs['Vcb'],
        })

print(f"\nFound {len(good_geometries)} geometries with loss < 0.1")

if good_geometries:
    good_geometries.sort(key=lambda x: x['loss'])
    print("\nTop 5 alternative geometries:")
    for i, g in enumerate(good_geometries[:5]):
        print(f"  {i+1}. Q={g['Q']}, U={g['U']}, D={g['D']}")
        print(f"     loss={g['loss']:.6f}, mc={g['mc']:.4f}, Vus={g['Vus']:.4f}, Vcb={g['Vcb']:.4f}")

print("\n" + "="*70)
print("FINAL VERIFICATION SUMMARY")
print("="*70)

print(f"""
1. Claimed solution verification:     {'PASSED ✓' if all_within_1pct else 'FAILED ✗'}
2. Independent re-optimization:       {'PASSED ✓' if all_within_1pct_independent else 'FAILED ✗'}
3. Alternative geometries found:      {len(good_geometries)}

CONCLUSION: The Clockwork kernel solution is {'VERIFIED AND REPRODUCIBLE' if all_within_1pct and all_within_1pct_independent else 'NOT FULLY VERIFIED'}
""")

# Save verification results
results_dir = os.path.join(os.path.dirname(__file__), '..', 'diagnostics', 'results')
os.makedirs(results_dir, exist_ok=True)

with open(os.path.join(results_dir, 'clockwork_verification.txt'), 'w') as f:
    f.write("CLOCKWORK SOLUTION VERIFICATION\n")
    f.write("="*50 + "\n\n")
    f.write("Claimed solution:\n")
    for key in ['mc', 'Vus', 'Vcb', 'Vub']:
        target = TARGETS[key]
        actual = obs_claimed[key]
        error = abs(actual - target) / target * 100
        f.write(f"  {key}: {actual:.6f} (error: {error:.4f}%)\n")
    f.write(f"\nVerification: {'PASSED' if all_within_1pct else 'FAILED'}\n")
    f.write(f"\nIndependent re-optimization: {'PASSED' if all_within_1pct_independent else 'FAILED'}\n")
    f.write(f"Alternative geometries found: {len(good_geometries)}\n")

print(f"Results saved to diagnostics/results/clockwork_verification.txt")
