#!/usr/bin/env python3
"""
Quark Sector Optimization - Generate fresh results
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from kernel import compute_quark_yukawas
from observables import compute_quark_observables, compute_ckm_loss, compute_mass_loss
from optimizer import optimize_parameters
from itertools import product
import argparse


def objective_quark(theta, Q, U, D):
    """Objective function for quark optimization"""
    sigma, k, alpha, eta, eps_u, eps_d = theta
    Yu, Yd = compute_quark_yukawas(Q, U, D, sigma, k, alpha, eta, eps_u, eps_d)
    obs = compute_quark_observables(Yu, Yd)
    L_ckm = compute_ckm_loss(obs)
    L_mass = compute_mass_loss(obs)
    
    # Mass floor penalties
    L_md_penalty = 2.0 * (np.log(0.002 / obs['md'])) ** 2 if obs['md'] < 0.002 else 0.0
    L_mu_penalty = 0.5 * (np.log(0.0005 / obs['mu'])) ** 2 if obs['mu'] < 0.0005 else 0.0
    
    return L_mass + 5.0 * L_ckm + L_md_penalty + L_mu_penalty


def optimize_single_geometry(Q, U, D, n_seeds=5, maxiter=100):
    """Optimize parameters for a single geometry"""
    bounds = [
        (0.5, 6.0), (0.1, 2.0), (0.0, 2 * np.pi),
        (1.0, 5.0), (0.01, 0.5), (0.01, 0.5),
    ]
    
    best_loss = np.inf
    best_result = None
    
    for seed in range(n_seeds):
        result = optimize_parameters(
            lambda theta: objective_quark(theta, Q, U, D),
            bounds, maxiter=maxiter, seed=seed, polish=False
        )
        
        if result['fun'] < best_loss:
            best_loss = result['fun']
            best_result = result
    
    return best_result


def generate_geometries(max_coord=10):
    """Generate candidate geometries"""
    geometries = []
    for q1, q2 in product(range(max_coord), repeat=2):
        if q1 >= q2:
            continue
        for u1, u2, u3 in product(range(max_coord), repeat=3):
            if not (u1 < u2 < u3):
                continue
            for d1, d2, d3 in product(range(max_coord), repeat=3):
                if not (d1 < d2 < d3):
                    continue
                geometries.append(((q1, q2, 0), (u1, u2, u3), (d1, d2, d3)))
    return geometries


def main():
    parser = argparse.ArgumentParser(description='Optimize quark sector')
    parser.add_argument('--max-coord', type=int, default=8)
    parser.add_argument('--n-seeds', type=int, default=3)
    parser.add_argument('--maxiter', type=int, default=100)
    parser.add_argument('--output', type=str, default='data/quark_results.csv')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    
    # Fix output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, '..', args.output)
    
    print("Generating geometries...")
    geometries = generate_geometries(max_coord=args.max_coord)
    if args.limit:
        geometries = geometries[:args.limit]
    
    print(f"Optimizing {len(geometries)} geometries with {args.n_seeds} seeds each...")
    
    results = []
    for i, (Q, U, D) in enumerate(geometries):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(geometries)}")
        
        result = optimize_single_geometry(Q, U, D, n_seeds=args.n_seeds, maxiter=args.maxiter)
        
        # Save ALL results, not just successful ones
        sigma, k, alpha, eta, eps_u, eps_d = result['x']
        Yu, Yd = compute_quark_yukawas(Q, U, D, sigma, k, alpha, eta, eps_u, eps_d)
        obs = compute_quark_observables(Yu, Yd)
        
        results.append({
            'Q1': Q[0], 'Q2': Q[1],
            'U1': U[0], 'U2': U[1], 'U3': U[2],
            'D1': D[0], 'D2': D[1], 'D3': D[2],
            'sigma': sigma, 'k': k, 'alpha': alpha, 'eta': eta,
            'eps_u': eps_u, 'eps_d': eps_d,
            'loss_total': result['fun'],
            'loss_ckm': compute_ckm_loss(obs),
            'loss_mass': compute_mass_loss(obs),
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
    print(f"  Best CKM loss: {df['loss_ckm'].min():.6f}")
    survivors = df[
        (df['Vus'] > 0.17) & (df['Vus'] < 0.29) &
        (df['Vcb'] > 0.025) & (df['Vcb'] < 0.060) &
        (df['Vub'] > 0.0018) & (df['Vub'] < 0.0060)
    ]
    print(f"  Survivors: {len(survivors)}/{len(df)} ({100*len(survivors)/len(df):.1f}%)")


if __name__ == "__main__":
    main()
