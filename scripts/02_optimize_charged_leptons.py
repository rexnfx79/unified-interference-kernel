#!/usr/bin/env python3
"""
Charged Lepton Sector Optimization - Phase-Sensitive Regime

Optimizes charged lepton masses using variable phase parameters (k_e, η_e).
This allows the muon mass hierarchy to be resolved through phase interference.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from kernel import compute_charged_lepton_yukawa
from observables import compute_charged_lepton_observables, compute_charged_lepton_loss
from optimizer import optimize_parameters
from itertools import product
import argparse


def objective_lepton(theta, L, E):
    """Objective function for charged lepton optimization"""
    sigma, k_e, alpha, eta_e, eps_e = theta
    Ye = compute_charged_lepton_yukawa(L, E, sigma, k_e, alpha, eta_e, eps_e)
    obs = compute_charged_lepton_observables(Ye)
    L_mass = compute_charged_lepton_loss(obs)
    
    # Mass floor penalties
    L_me_penalty = 2.0 * (np.log(0.0003 / obs['me'])) ** 2 if obs['me'] < 0.0003 else 0.0
    L_mmu_penalty = 0.5 * (np.log(0.05 / obs['mmu'])) ** 2 if obs['mmu'] < 0.05 else 0.0
    
    return L_mass + L_me_penalty + L_mmu_penalty


def optimize_single_geometry(L, E, n_seeds=5, maxiter=100):
    """Optimize parameters for a single geometry"""
    # Parameters: sigma, k_e, alpha, eta_e, eps_e
    # Note: k_e and eta_e are variable phase parameters (phase-sensitive regime)
    bounds = [
        (0.5, 6.0),      # sigma: envelope width
        (0.1, 2.0),      # k_e: phase parameter (variable from quark baseline)
        (0.0, 2 * np.pi), # alpha: phase offset
        (1.0, 5.0),      # eta_e: phase coupling (variable from quark baseline)
        (0.01, 0.5),     # eps_e: interference strength
    ]
    
    best_loss = np.inf
    best_result = None
    
    for seed in range(n_seeds):
        result = optimize_parameters(
            lambda theta: objective_lepton(theta, L, E),
            bounds, maxiter=maxiter, seed=seed, polish=False
        )
        
        if result['fun'] < best_loss:
            best_loss = result['fun']
            best_result = result
    
    return best_result


def generate_geometries(max_coord=10):
    """Generate candidate geometries for charged leptons"""
    geometries = []
    # L: left-handed lepton doublets (2 coordinates, third fixed at 0)
    for l1, l2 in product(range(max_coord), repeat=2):
        if l1 >= l2:
            continue
        # E: right-handed charged leptons (3 coordinates, must be ordered)
        for e1, e2, e3 in product(range(max_coord), repeat=3):
            if not (e1 < e2 < e3):
                continue
            geometries.append(((l1, l2, 0), (e1, e2, e3)))
    return geometries


def main():
    parser = argparse.ArgumentParser(description='Optimize charged lepton sector')
    parser.add_argument('--max-coord', type=int, default=5, help='Maximum coordinate value')
    parser.add_argument('--n-seeds', type=int, default=5, help='Number of optimization seeds per geometry')
    parser.add_argument('--maxiter', type=int, default=200, help='Maximum iterations per optimization')
    parser.add_argument('--output', type=str, default='data/charged_lepton_results.csv', help='Output CSV file')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of geometries (for testing)')
    args = parser.parse_args()
    
    # Fix output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, '..', args.output)
    
    print("Generating geometries...")
    geometries = generate_geometries(max_coord=args.max_coord)
    if args.limit:
        geometries = geometries[:args.limit]
    
    print(f"Optimizing {len(geometries)} geometries with {args.n_seeds} seeds each...")
    print(f"Phase-sensitive regime: variable k_e and η_e parameters")
    
    results = []
    for i, (L, E) in enumerate(geometries):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(geometries)}")
        
        result = optimize_single_geometry(L, E, n_seeds=args.n_seeds, maxiter=args.maxiter)
        
        # Save ALL results, not just successful ones
        sigma, k_e, alpha, eta_e, eps_e = result['x']
        Ye = compute_charged_lepton_yukawa(L, E, sigma, k_e, alpha, eta_e, eps_e)
        obs = compute_charged_lepton_observables(Ye)
        
        results.append({
            'L1': L[0], 'L2': L[1],
            'E1': E[0], 'E2': E[1], 'E3': E[2],
            'sigma': sigma, 'k_e': k_e, 'alpha': alpha, 'eta_e': eta_e,
            'eps_e': eps_e,
            'loss_total': result['fun'],
            'loss_mass': compute_charged_lepton_loss(obs),
            **obs
        })
    
    print(f"\nCompleted! Generated {len(results)} results.")
    
    # Save results
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Best total loss: {df['loss_total'].min():.6f}")
    print(f"  Best mass loss: {df['loss_mass'].min():.6f}")
    
    # Find survivors (matching experimental masses within reasonable ranges)
    # PDG 2024: me = 0.0005109989461, mmu = 0.1056583745, mtau = 1.77686
    survivors = df[
        (df['me'] > 0.0004) & (df['me'] < 0.0006) &
        (df['mmu'] > 0.09) & (df['mmu'] < 0.12) &
        (df['mtau'] > 1.6) & (df['mtau'] < 2.0)
    ]
    print(f"  Survivors: {len(survivors)}/{len(df)} ({100*len(survivors)/len(df):.1f}%)")
    
    if len(survivors) > 0:
        print(f"  Best survivor mass loss: {survivors['loss_mass'].min():.6f}")


if __name__ == "__main__":
    main()
