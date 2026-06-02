#!/usr/bin/env python3
"""
Pareto Envelope Comparison: Test Robustness to Envelope Shape

This script tests whether the CKM-mc Pareto knee is:
1. Robust to envelope choice (persists for all p) → phase/interference constraint
2. Moves with p → envelope artifact
3. Disappears for some p → model-dependent

Methodology:
- Scan p ∈ [1.0, 3.0] (exponential to super-Gaussian)
- For each p, run identical Pareto sweep
- Compare knee locations and hull shapes
- Report statistical summary

This is a parameter-matched comparison: only p changes, everything else identical.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from scipy.optimize import differential_evolution
import argparse
from datetime import datetime


# Import modules
from kernel_generalized import compute_quark_yukawas_generalized
from observables import compute_quark_observables, compute_ckm_loss, compute_mass_loss, QUARK_TARGETS


# ============================================================================
# Objective Function
# ============================================================================

def objective_quark_generalized(
    theta: np.ndarray,
    Q: Tuple[int, int, int],
    U: Tuple[int, int, int],
    D: Tuple[int, int, int],
    p: float,
    w_ckm: float = 1.0,
    w_mass: float = 1.0
) -> float:
    """
    Objective function for quark optimization with generalized envelope.
    
    Args:
        theta: Parameters [sigma, k, alpha, eta, eps_u, eps_d]
        Q, U, D: Geometry
        p: Envelope shape parameter
        w_ckm: Weight for CKM loss
        w_mass: Weight for mass loss
    
    Returns:
        Weighted total loss
    """
    sigma, k, alpha, eta, eps_u, eps_d = theta
    
    # Compute Yukawa matrices with generalized envelope
    Yu, Yd = compute_quark_yukawas_generalized(Q, U, D, sigma, k, alpha, eta, eps_u, eps_d, p)
    
    # Compute observables
    obs = compute_quark_observables(Yu, Yd)
    
    # Compute losses
    L_ckm = compute_ckm_loss(obs)
    L_mass = compute_mass_loss(obs)
    
    # Mass floor penalties (prevent collapse)
    md_floor = 0.002
    mu_floor = 0.0005
    
    L_md_penalty = 2.0 * (np.log(md_floor / obs['md'])) ** 2 if obs['md'] < md_floor else 0.0
    L_mu_penalty = 0.5 * (np.log(mu_floor / obs['mu'])) ** 2 if obs['mu'] < mu_floor else 0.0
    
    # Weighted total
    return w_mass * L_mass + w_ckm * L_ckm + L_md_penalty + L_mu_penalty


# ============================================================================
# Pareto Sweep
# ============================================================================

def pareto_sweep_single_geometry(
    Q: Tuple[int, int, int],
    U: Tuple[int, int, int],
    D: Tuple[int, int, int],
    p: float,
    weights: List[Tuple[float, float]],
    n_seeds: int = 3,
    maxiter: int = 100
) -> List[Dict]:
    """
    Run Pareto sweep for a single geometry at fixed p.
    
    Args:
        Q, U, D: Geometry
        p: Envelope shape parameter
        weights: List of (w_ckm, w_mass) weight pairs
        n_seeds: Random seeds per weight
        maxiter: Max iterations per optimization
    
    Returns:
        List of result dictionaries
    """
    bounds = [
        (0.5, 6.0),          # sigma
        (0.1, 2.0),          # k
        (0.0, 2 * np.pi),    # alpha
        (1.0, 5.0),          # eta
        (0.01, 0.5),         # eps_u
        (0.01, 0.5),         # eps_d
    ]
    
    results = []
    
    for w_ckm, w_mass in weights:
        best_loss = np.inf
        best_result = None
        
        for seed in range(n_seeds):
            try:
                result = differential_evolution(
                    lambda theta: objective_quark_generalized(theta, Q, U, D, p, w_ckm, w_mass),
                    bounds,
                    maxiter=maxiter,
                    seed=seed,
                    polish=False,
                    atol=1e-6,
                    tol=1e-6,
                )
                
                if result.fun < best_loss:
                    best_loss = result.fun
                    best_result = result
            except Exception as e:
                continue
        
        if best_result is not None:
            sigma, k, alpha, eta, eps_u, eps_d = best_result.x
            
            # Recompute observables
            Yu, Yd = compute_quark_yukawas_generalized(Q, U, D, sigma, k, alpha, eta, eps_u, eps_d, p)
            obs = compute_quark_observables(Yu, Yd)
            
            results.append({
                'p': p,
                'w_ckm': w_ckm,
                'w_mass': w_mass,
                'Q1': Q[0], 'Q2': Q[1],
                'U1': U[0], 'U2': U[1], 'U3': U[2],
                'D1': D[0], 'D2': D[1], 'D3': D[2],
                'sigma': sigma,
                'k': k,
                'alpha': alpha,
                'eta': eta,
                'eps_u': eps_u,
                'eps_d': eps_d,
                'loss_total': best_result.fun,
                'loss_ckm': compute_ckm_loss(obs),
                'loss_mass': compute_mass_loss(obs),
                **obs
            })
    
    return results


def compute_pareto_frontier(df: pd.DataFrame, x_col: str = 'loss_ckm', y_col: str = 'mc') -> pd.DataFrame:
    """
    Compute Pareto frontier from results.
    
    A point is Pareto-optimal if no other point has both lower x AND lower |y - target|.
    """
    mc_target = QUARK_TARGETS['mc']
    
    # Compute mc deviation
    df = df.copy()
    df['mc_dev'] = abs(df[y_col] - mc_target)
    
    # Sort by x
    df_sorted = df.sort_values(x_col)
    
    # Find Pareto frontier
    frontier_idx = []
    min_y = np.inf
    
    for idx, row in df_sorted.iterrows():
        if row['mc_dev'] < min_y:
            frontier_idx.append(idx)
            min_y = row['mc_dev']
    
    return df.loc[frontier_idx]


def detect_knee(frontier_df: pd.DataFrame, x_col: str = 'loss_ckm', y_col: str = 'mc') -> Dict:
    """
    Detect knee in Pareto frontier using curvature analysis.
    
    Returns dict with knee location and statistics.
    """
    if len(frontier_df) < 3:
        return {'knee_found': False, 'reason': 'Too few points'}
    
    mc_target = QUARK_TARGETS['mc']
    
    # Normalize coordinates
    x = frontier_df[x_col].values
    y = abs(frontier_df[y_col].values - mc_target)
    
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)
    
    # Compute curvature (second derivative approximation)
    if len(x_norm) < 5:
        # Simple: find point with max distance from line connecting endpoints
        line_start = np.array([x_norm[0], y_norm[0]])
        line_end = np.array([x_norm[-1], y_norm[-1]])
        line_vec = line_end - line_start
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 1e-10:
            return {'knee_found': False, 'reason': 'Degenerate frontier'}
        
        line_unit = line_vec / line_len
        
        max_dist = 0
        knee_idx = 0
        
        for i in range(len(x_norm)):
            point = np.array([x_norm[i], y_norm[i]])
            vec_to_point = point - line_start
            proj_len = np.dot(vec_to_point, line_unit)
            proj_point = line_start + proj_len * line_unit
            dist = np.linalg.norm(point - proj_point)
            
            if dist > max_dist:
                max_dist = dist
                knee_idx = i
    else:
        # Use second derivative
        dx = np.gradient(x_norm)
        dy = np.gradient(y_norm)
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        
        # Curvature = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2 + 1e-10)**1.5
        knee_idx = np.argmax(curvature)
    
    knee_row = frontier_df.iloc[knee_idx]
    
    return {
        'knee_found': True,
        'knee_ckm_loss': knee_row[x_col],
        'knee_mc': knee_row[y_col],
        'knee_mc_error_pct': 100 * (knee_row[y_col] - mc_target) / mc_target,
        'knee_idx': knee_idx,
        'frontier_size': len(frontier_df),
    }


# ============================================================================
# Main Analysis
# ============================================================================

def run_envelope_comparison(
    geometries: List[Tuple],
    p_values: List[float],
    weights: List[Tuple[float, float]],
    n_seeds: int = 3,
    maxiter: int = 100,
    output_dir: str = 'data'
) -> pd.DataFrame:
    """
    Run full envelope comparison analysis.
    
    Args:
        geometries: List of (Q, U, D) tuples
        p_values: List of envelope shape parameters to test
        weights: List of (w_ckm, w_mass) weight pairs for Pareto sweep
        n_seeds: Random seeds per optimization
        maxiter: Max iterations per optimization
        output_dir: Directory for output files
    
    Returns:
        DataFrame with all results
    """
    all_results = []
    
    total_runs = len(geometries) * len(p_values) * len(weights) * n_seeds
    print(f"Total optimization runs: {total_runs}")
    print(f"Geometries: {len(geometries)}, p values: {len(p_values)}, weights: {len(weights)}")
    
    run_count = 0
    
    for geom_idx, (Q, U, D) in enumerate(geometries):
        for p in p_values:
            print(f"  Geometry {geom_idx+1}/{len(geometries)}, p={p:.2f}...", end=" ", flush=True)
            
            results = pareto_sweep_single_geometry(Q, U, D, p, weights, n_seeds, maxiter)
            all_results.extend(results)
            
            run_count += len(weights) * n_seeds
            print(f"done ({len(results)} points)")
    
    return pd.DataFrame(all_results)


def analyze_results(df: pd.DataFrame) -> Dict:
    """
    Analyze results to determine knee robustness.
    
    Returns summary statistics.
    """
    p_values = sorted(df['p'].unique())
    
    analysis = {
        'p_values': p_values,
        'knee_locations': {},
        'frontier_sizes': {},
        'summary': {}
    }
    
    for p in p_values:
        df_p = df[df['p'] == p]
        
        # Compute Pareto frontier
        frontier = compute_pareto_frontier(df_p)
        
        # Detect knee
        knee_info = detect_knee(frontier)
        
        analysis['knee_locations'][p] = knee_info
        analysis['frontier_sizes'][p] = len(frontier)
    
    # Compute summary statistics
    knee_ckm_losses = [
        analysis['knee_locations'][p]['knee_ckm_loss']
        for p in p_values
        if analysis['knee_locations'][p].get('knee_found', False)
    ]
    
    if len(knee_ckm_losses) >= 2:
        analysis['summary'] = {
            'knee_mean': np.mean(knee_ckm_losses),
            'knee_std': np.std(knee_ckm_losses),
            'knee_cv': np.std(knee_ckm_losses) / np.mean(knee_ckm_losses) if np.mean(knee_ckm_losses) > 0 else np.inf,
            'knee_range': max(knee_ckm_losses) - min(knee_ckm_losses),
            'robust': np.std(knee_ckm_losses) / np.mean(knee_ckm_losses) < 0.3 if np.mean(knee_ckm_losses) > 0 else False,
        }
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description='Pareto envelope comparison')
    parser.add_argument('--p-min', type=float, default=1.0, help='Minimum p value')
    parser.add_argument('--p-max', type=float, default=3.0, help='Maximum p value')
    parser.add_argument('--p-steps', type=int, default=5, help='Number of p values')
    parser.add_argument('--n-geometries', type=int, default=3, help='Number of geometries')
    parser.add_argument('--n-weights', type=int, default=10, help='Number of weight pairs')
    parser.add_argument('--n-seeds', type=int, default=2, help='Seeds per optimization')
    parser.add_argument('--maxiter', type=int, default=100, help='Max iterations')
    parser.add_argument('--output', type=str, default='data/pareto_envelope_comparison.csv')
    
    args = parser.parse_args()
    
    # Setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, '..', args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate p values
    p_values = np.linspace(args.p_min, args.p_max, args.p_steps).tolist()
    print(f"Testing p values: {p_values}")
    
    # Generate weight pairs (log-spaced for Pareto sweep)
    w_ratios = np.logspace(-1, 1, args.n_weights)
    weights = [(w, 1.0) for w in w_ratios]
    print(f"Weight pairs: {len(weights)}")
    
    # Generate geometries (known-good from literature)
    geometries = [
        ((0, 1, 0), (0, 3, 6), (0, 3, 7)),   # Standard
        ((0, 2, 0), (0, 4, 8), (0, 4, 9)),   # Scaled
        ((1, 3, 0), (0, 5, 10), (0, 5, 11)), # Offset
    ][:args.n_geometries]
    print(f"Geometries: {len(geometries)}")
    
    # Run analysis
    print("\n" + "=" * 60)
    print("PARETO ENVELOPE COMPARISON")
    print("=" * 60)
    print(f"Start time: {datetime.now()}")
    
    df = run_envelope_comparison(
        geometries=geometries,
        p_values=p_values,
        weights=weights,
        n_seeds=args.n_seeds,
        maxiter=args.maxiter,
        output_dir=os.path.dirname(output_path)
    )
    
    # Save raw results
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} results to {output_path}")
    
    # Analyze
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    analysis = analyze_results(df)
    
    print("\nKnee locations by p:")
    for p in analysis['p_values']:
        knee = analysis['knee_locations'][p]
        if knee.get('knee_found', False):
            print(f"  p={p:.2f}: CKM loss = {knee['knee_ckm_loss']:.6f}, mc = {knee['knee_mc']:.3f} GeV ({knee['knee_mc_error_pct']:+.1f}%)")
        else:
            print(f"  p={p:.2f}: No knee found ({knee.get('reason', 'unknown')})")
    
    if 'summary' in analysis and analysis['summary']:
        print("\nSummary:")
        print(f"  Knee mean: {analysis['summary']['knee_mean']:.6f}")
        print(f"  Knee std: {analysis['summary']['knee_std']:.6f}")
        print(f"  Knee CV: {analysis['summary']['knee_cv']:.3f}")
        print(f"  Robust (CV < 0.3): {analysis['summary']['robust']}")
    
    print(f"\nEnd time: {datetime.now()}")
    
    # Save analysis summary
    summary_path = output_path.replace('.csv', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("PARETO ENVELOPE COMPARISON SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"p values tested: {analysis['p_values']}\n\n")
        f.write("Knee locations:\n")
        for p in analysis['p_values']:
            knee = analysis['knee_locations'][p]
            if knee.get('knee_found', False):
                f.write(f"  p={p:.2f}: CKM loss = {knee['knee_ckm_loss']:.6f}, mc = {knee['knee_mc']:.3f} GeV\n")
            else:
                f.write(f"  p={p:.2f}: No knee found\n")
        if 'summary' in analysis and analysis['summary']:
            f.write(f"\nKnee robustness: CV = {analysis['summary']['knee_cv']:.3f}\n")
            f.write(f"Conclusion: {'ROBUST' if analysis['summary']['robust'] else 'NOT ROBUST'}\n")
    
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
