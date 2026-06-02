#!/usr/bin/env python3
"""
Fast Verification of Clockwork Solution

Quick verification without exhaustive geometry search.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.optimize import differential_evolution
from alternative_kernels import compute_yukawas_clockwork
from observables import compute_quark_observables

np.random.seed(2024)

print("="*70)
print("FAST VERIFICATION OF CLOCKWORK SOLUTION")
print("="*70)

# Claimed geometry and parameters
Q = (7, 8, 9)
U = (2, 12, 14)
D = (1, 4, 7)

params_claimed = [11.636058, 3.057852, 0.627795, 1.486743, 0.997599, 0.999944]

# Targets
TARGETS = {'mc': 1.27, 'Vus': 0.225, 'Vcb': 0.04182, 'Vub': 0.00382}

print("\n[1] Verify claimed solution")
print("-"*40)

Yu, Yd = compute_yukawas_clockwork(Q, U, D, *params_claimed)
obs = compute_quark_observables(Yu, Yd)

all_pass = True
for key in ['mc', 'Vus', 'Vcb', 'Vub']:
    target = TARGETS[key]
    actual = obs[key]
    error = abs(actual - target) / target * 100
    status = "PASS" if error < 1 else "FAIL"
    print(f"  {key}: {actual:.6f} (target: {target}, error: {error:.4f}%) [{status}]")
    if error >= 1:
        all_pass = False

print(f"\nClaimed solution: {'VERIFIED' if all_pass else 'FAILED'}")

print("\n[2] Independent re-optimization (5 seeds)")
print("-"*40)

bounds = [
    (5.0, 15.0), (0.001, 10.0), (0.0, 2*np.pi),
    (0.001, 15.0), (0.001, 2.0), (0.001, 2.0),
]

def objective(theta):
    try:
        Yu, Yd = compute_yukawas_clockwork(Q, U, D, *theta)
        obs = compute_quark_observables(Yu, Yd)
        if obs['mc'] < 0.01 or obs['mc'] > 500:
            return 1000.0
        return ((obs['mc'] - 1.27) / 1.27)**2 + \
               ((obs['Vus'] - 0.225) / 0.225)**2 + \
               ((obs['Vcb'] - 0.042) / 0.042)**2 + \
               ((obs['Vub'] - 0.00382) / 0.00382)**2
    except:
        return 1000.0

best_loss = float('inf')
best_params = None

for seed in range(5):
    result = differential_evolution(objective, bounds, maxiter=100, seed=seed+1000, polish=True)
    if result.fun < best_loss:
        best_loss = result.fun
        best_params = result.x
    print(f"  Seed {seed}: loss = {result.fun:.10f}")

Yu, Yd = compute_yukawas_clockwork(Q, U, D, *best_params)
obs_new = compute_quark_observables(Yu, Yd)

print(f"\nBest independent solution:")
all_pass_new = True
for key in ['mc', 'Vus', 'Vcb', 'Vub']:
    target = TARGETS[key]
    actual = obs_new[key]
    error = abs(actual - target) / target * 100
    status = "PASS" if error < 1 else "FAIL"
    print(f"  {key}: {actual:.6f} (error: {error:.4f}%) [{status}]")
    if error >= 1:
        all_pass_new = False

print(f"\nIndependent solution: {'VERIFIED' if all_pass_new else 'FAILED'}")

print("\n[3] Parameter comparison")
print("-"*40)
param_names = ['q', 'k', 'alpha', 'eta', 'eps_u', 'eps_d']
print(f"{'Param':<10} {'Claimed':>12} {'Found':>12} {'Diff':>12}")
for name, claimed, found in zip(param_names, params_claimed, best_params):
    print(f"{name:<10} {claimed:>12.6f} {found:>12.6f} {abs(claimed-found):>12.6f}")

print("\n" + "="*70)
print("VERIFICATION SUMMARY")
print("="*70)
print(f"  Claimed solution:      {'PASS' if all_pass else 'FAIL'}")
print(f"  Independent solution:  {'PASS' if all_pass_new else 'FAIL'}")
print(f"  Overall:               {'REPRODUCIBLE' if all_pass and all_pass_new else 'NOT REPRODUCIBLE'}")
print("="*70)
