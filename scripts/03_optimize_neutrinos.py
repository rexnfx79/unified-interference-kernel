#!/usr/bin/env python3
"""
Neutrino Sector Optimization - Metric-Dominated Regime

Optimizes neutrino PMNS mixing angles using envelope compression (g_env).
Information loss under compression leads to emergent anarchy in PMNS angles.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from kernel import compute_neutrino_yukawa, compute_charged_lepton_yukawa
from observables import (
    compute_neutrino_observables, compute_pmns_loss, compute_neutrino_mass_loss,
    compute_charged_lepton_observables
)
from optimizer import optimize_parameters
from itertools import product
import argparse


def objective_neutrino(theta, L, N, E, g_env):
    """Objective function for neutrino optimization"""
    sigma, k, alpha, eta, eps_nu, k_e, eta_e, eps_e = theta
    
    # Compute neutrino Yukawa (with envelope compression)
    Ynu = compute_neutrino_yukawa(L, N, sigma, k, alpha, eta, eps_nu, g_env)
    
    # Compute charged lepton Yukawa (needed for PMNS mixing)
    Ye = compute_charged_lepton_yukawa(L, E, sigma, k_e, alpha, eta_e, eps_e)
    
    # Compute observables
    obs = compute_neutrino_observables(Ynu, Ye)
    
    L_pmns = compute_pmns_loss(obs)
    L_mass = compute_neutrino_mass_loss(obs)
    
    # Penalties for mass floor
    L_m1_penalty = 2.0 * (np.log(0.001 / obs['m1'])) ** 2 if obs['m1'] < 0.001 else 0.0
    
    return L_mass + 5.0 * L_pmns + L_m1_penalty


def optimize_single_geometry(L, N, E, g_env, n_seeds=5, maxiter=100):
    """Optimize parameters for a single geometry with given g_env"""
    # Parameters: sigma, k, alpha, eta, eps_nu, k_e, eta_e, eps_e
    bounds = [
        (0.5, 6.0),      # sigma: envelope width (before compression)
        (0.1, 2.0),      # k: phase parameter
        (0.0, 2 * np.pi), # alpha: phase offset
        (1.0, 5.0),      # eta: phase coupling
        (0.01, 0.5),     # eps_nu: neutrino interference
        (0.1, 2.0),      # k_e: charged lepton phase parameter
        (1.0, 5.0),      # eta_e: charged lepton phase coupling
        (0.01, 0.5),     # eps_e: charged lepton interference
    ]
    
    best_loss = np.inf
    best_result = None
    
    for seed in range(n_seeds):
        result = optimize_parameters(
            lambda theta: objective_neutrino(theta, L, N, E, g_env),
            bounds, maxiter=maxiter, seed=seed, polish=False
        )
        
        if result['fun'] < best_loss:
            best_loss = result['fun']
            best_result = result
    
    return best_result


def generate_geometries(max_coord=10):
    """Generate candidate geometries for neutrinos"""
    geometries = []
    # L: left-handed lepton doublets (2 coordinates, third fixed at 0)
    for l1, l2 in product(range(max_coord), repeat=2):
        if l1 >= l2:
            continue
        # N: right-handed neutrinos (3 coordinates, must be ordered)
        for n1, n2, n3 in product(range(max_coord), repeat=3):
            if not (n1 < n2 < n3):
                continue
            # E: right-handed charged leptons (3 coordinates, must be ordered)
            for e1, e2, e3 in product(range(max_coord), repeat=3):
                if not (e1 < e2 < e3):
                    continue
                geometries.append(((l1, l2, 0), (n1, n2, n3), (e1, e2, e3)))
    return geometries


def main():
    parser = argparse.ArgumentParser(description='Optimize neutrino sector')
    parser.add_argument('--max-coord', type=int, default=4, help='Maximum coordinate value')
    parser.add_argument('--g-env-min', type=float, default=0.5, help='Minimum g_env value')
    parser.add_argument('--g-env-max', type=float, default=0.7, help='Maximum g_env value')
    parser.add_argument('--g-env-steps', type=int, default=5, help='Number of g_env values to scan')
    parser.add_argument('--n-seeds', type=int, default=5, help='Number of optimization seeds per geometry')
    parser.add_argument('--maxiter', type=int, default=200, help='Maximum iterations per optimization')
    parser.add_argument('--output', type=str, default='data/neutrino_results.csv', help='Output CSV file')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of geometries (for testing)')
    args = parser.parse_args()
    
    # Fix output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, '..', args.output)
    
    # Generate g_env values to scan
    g_env_values = np.linspace(args.g_env_min, args.g_env_max, args.g_env_steps)
    
    print("Generating geometries...")
    geometries = generate_geometries(max_coord=args.max_coord)
    if args.limit:
        geometries = geometries[:args.limit]
    
    total_optimizations = len(geometries) * len(g_env_values)
    print(f"Scanning {len(g_env_values)} g_env values: {g_env_values}")
    print(f"Optimizing {len(geometries)} geometries with {args.n_seeds} seeds each...")
    print(f"Total optimizations: {total_optimizations}")
    print(f"Metric-dominated regime: envelope compression (g_env)")
    
    results = []
    opt_count = 0
    
    for g_env in g_env_values:
        print(f"\n--- g_env = {g_env:.3f} ---")
        for i, (L, N, E) in enumerate(geometries):
            if opt_count % 10 == 0:
                print(f"Progress: {opt_count}/{total_optimizations} (g_env={g_env:.3f}, geom={i}/{len(geometries)})")
            
            result = optimize_single_geometry(L, N, E, g_env, n_seeds=args.n_seeds, maxiter=args.maxiter)
            opt_count += 1
            
            # Save ALL results
            sigma, k, alpha, eta, eps_nu, k_e, eta_e, eps_e = result['x']
            Ynu = compute_neutrino_yukawa(L, N, sigma, k, alpha, eta, eps_nu, g_env)
            Ye = compute_charged_lepton_yukawa(L, E, sigma, k_e, alpha, eta_e, eps_e)
            obs = compute_neutrino_observables(Ynu, Ye)
            
            results.append({
                'g_env': g_env,
                'L1': L[0], 'L2': L[1],
                'N1': N[0], 'N2': N[1], 'N3': N[2],
                'E1': E[0], 'E2': E[1], 'E3': E[2],
                'sigma': sigma, 'k': k, 'alpha': alpha, 'eta': eta,
                'eps_nu': eps_nu, 'k_e': k_e, 'eta_e': eta_e, 'eps_e': eps_e,
                'loss_total': result['fun'],
                'loss_pmns': compute_pmns_loss(obs),
                'loss_mass': compute_neutrino_mass_loss(obs),
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
    print(f"  Best PMNS loss: {df['loss_pmns'].min():.6f}")
    print(f"  Best mass loss: {df['loss_mass'].min():.6f}")
    
    # Find survivors (matching experimental PMNS angles)
    # PDG 2024: theta12 ≈ 0.583, theta23 ≈ 0.785, theta13 ≈ 0.149
    survivors = df[
        (df['theta12'] > 0.5) & (df['theta12'] < 0.7) &
        (df['theta23'] > 0.6) & (df['theta23'] < 1.0) &
        (df['theta13'] > 0.10) & (df['theta13'] < 0.20)
    ]
    print(f"  Survivors: {len(survivors)}/{len(df)} ({100*len(survivors)/len(df):.1f}%)")
    
    if len(survivors) > 0:
        print(f"  Best survivor PMNS loss: {survivors['loss_pmns'].min():.6f}")
        print(f"  Best g_env: {survivors.loc[survivors['loss_pmns'].idxmin(), 'g_env']:.3f}")


if __name__ == "__main__":
    main()
