#!/usr/bin/env python3
"""
True Transfer Test: Test Actual Universality

This is the rigorous test of universality:
1. Fit quarks → get (σ*, k*, α*, η*)
2. FREEZE these parameters
3. Apply to leptons with ONLY ε_e free
4. Measure: How bad is the fit?

If fit is good → evidence for universality
If fit is bad → quantifies degree of non-universality

This is what "universal kernel" claims require.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from scipy.optimize import differential_evolution, minimize
import argparse
from datetime import datetime


from kernel_generalized import compute_yukawa_matrix_generalized
from observables import (
    compute_quark_observables,
    compute_ckm_loss,
    compute_mass_loss,
    compute_lepton_observables,
    compute_lepton_loss,
    QUARK_TARGETS,
    LEPTON_TARGETS,
)


# ============================================================================
# Quark Fitting
# ============================================================================

def fit_quarks(
    Q: Tuple[int, int, int],
    U: Tuple[int, int, int],
    D: Tuple[int, int, int],
    p: float = 2.0,
    n_seeds: int = 5,
    maxiter: int = 200
) -> Dict:
    """
    Fit quark sector to get universal parameters.
    
    Returns:
        Dict with fitted parameters and observables
    """
    from kernel_generalized import compute_quark_yukawas_generalized
    
    bounds = [
        (0.5, 6.0),          # sigma
        (0.1, 2.0),          # k
        (0.0, 2 * np.pi),    # alpha
        (1.0, 5.0),          # eta
        (0.01, 0.5),         # eps_u
        (0.01, 0.5),         # eps_d
    ]
    
    def objective(theta):
        sigma, k, alpha, eta, eps_u, eps_d = theta
        Yu, Yd = compute_quark_yukawas_generalized(Q, U, D, sigma, k, alpha, eta, eps_u, eps_d, p)
        obs = compute_quark_observables(Yu, Yd)
        L_ckm = compute_ckm_loss(obs)
        L_mass = compute_mass_loss(obs)
        L_md = 2.0 * (np.log(0.002 / obs['md'])) ** 2 if obs['md'] < 0.002 else 0.0
        L_mu = 0.5 * (np.log(0.0005 / obs['mu'])) ** 2 if obs['mu'] < 0.0005 else 0.0
        return L_mass + 5.0 * L_ckm + L_md + L_mu
    
    best_loss = np.inf
    best_result = None
    
    for seed in range(n_seeds):
        result = differential_evolution(
            objective, bounds, maxiter=maxiter, seed=seed, polish=True, atol=1e-8, tol=1e-8
        )
        if result.fun < best_loss:
            best_loss = result.fun
            best_result = result
    
    sigma, k, alpha, eta, eps_u, eps_d = best_result.x
    Yu, Yd = compute_quark_yukawas_generalized(Q, U, D, sigma, k, alpha, eta, eps_u, eps_d, p)
    obs = compute_quark_observables(Yu, Yd)
    
    return {
        'sigma': sigma,
        'k': k,
        'alpha': alpha,
        'eta': eta,
        'eps_u': eps_u,
        'eps_d': eps_d,
        'loss_total': best_loss,
        'loss_ckm': compute_ckm_loss(obs),
        'loss_mass': compute_mass_loss(obs),
        **obs
    }


# ============================================================================
# Transfer Tests
# ============================================================================

def transfer_test_frozen(
    L: Tuple[int, int, int],
    E: Tuple[int, int, int],
    quark_params: Dict,
    p: float = 2.0,
    n_seeds: int = 5,
    maxiter: int = 200
) -> Dict:
    """
    Transfer test with FROZEN universal parameters.
    
    Only eps_e is allowed to vary.
    This is the TRUE universality test.
    """
    sigma = quark_params['sigma']
    k = quark_params['k']
    alpha = quark_params['alpha']
    eta = quark_params['eta']
    
    bounds = [(0.01, 0.5)]  # Only eps_e
    
    def objective(theta):
        eps_e = theta[0]
        Ye = compute_yukawa_matrix_generalized(L, E, sigma, k, alpha, eta, eps_e, p)
        obs = compute_lepton_observables(Ye)
        return compute_lepton_loss(obs)
    
    best_loss = np.inf
    best_result = None
    
    for seed in range(n_seeds):
        result = differential_evolution(
            objective, bounds, maxiter=maxiter, seed=seed, polish=True
        )
        if result.fun < best_loss:
            best_loss = result.fun
            best_result = result
    
    eps_e = best_result.x[0]
    Ye = compute_yukawa_matrix_generalized(L, E, sigma, k, alpha, eta, eps_e, p)
    obs = compute_lepton_observables(Ye)
    
    return {
        'test_type': 'frozen',
        'eps_e': eps_e,
        'loss': best_loss,
        **obs
    }


def transfer_test_free_k(
    L: Tuple[int, int, int],
    E: Tuple[int, int, int],
    quark_params: Dict,
    p: float = 2.0,
    n_seeds: int = 5,
    maxiter: int = 200
) -> Dict:
    """
    Transfer test with k_e allowed to vary.
    
    This tests: "How much does k need to change for leptons?"
    """
    sigma = quark_params['sigma']
    alpha = quark_params['alpha']
    eta = quark_params['eta']
    k_quark = quark_params['k']
    
    bounds = [
        (0.1, 2.0),   # k_e
        (0.01, 0.5),  # eps_e
    ]
    
    def objective(theta):
        k_e, eps_e = theta
        Ye = compute_yukawa_matrix_generalized(L, E, sigma, k_e, alpha, eta, eps_e, p)
        obs = compute_lepton_observables(Ye)
        return compute_lepton_loss(obs)
    
    best_loss = np.inf
    best_result = None
    
    for seed in range(n_seeds):
        result = differential_evolution(
            objective, bounds, maxiter=maxiter, seed=seed, polish=True
        )
        if result.fun < best_loss:
            best_loss = result.fun
            best_result = result
    
    k_e, eps_e = best_result.x
    Ye = compute_yukawa_matrix_generalized(L, E, sigma, k_e, alpha, eta, eps_e, p)
    obs = compute_lepton_observables(Ye)
    
    return {
        'test_type': 'free_k',
        'k_e': k_e,
        'k_quark': k_quark,
        'delta_k': k_e - k_quark,
        'eps_e': eps_e,
        'loss': best_loss,
        **obs
    }


def transfer_test_free_all(
    L: Tuple[int, int, int],
    E: Tuple[int, int, int],
    quark_params: Dict,
    p: float = 2.0,
    n_seeds: int = 5,
    maxiter: int = 200
) -> Dict:
    """
    Transfer test with all parameters free.
    
    This is the "independent fit" baseline.
    """
    bounds = [
        (0.5, 6.0),          # sigma_e
        (0.1, 2.0),          # k_e
        (0.0, 2 * np.pi),    # alpha_e
        (1.0, 5.0),          # eta_e
        (0.01, 0.5),         # eps_e
    ]
    
    def objective(theta):
        sigma_e, k_e, alpha_e, eta_e, eps_e = theta
        Ye = compute_yukawa_matrix_generalized(L, E, sigma_e, k_e, alpha_e, eta_e, eps_e, p)
        obs = compute_lepton_observables(Ye)
        return compute_lepton_loss(obs)
    
    best_loss = np.inf
    best_result = None
    
    for seed in range(n_seeds):
        result = differential_evolution(
            objective, bounds, maxiter=maxiter, seed=seed, polish=True
        )
        if result.fun < best_loss:
            best_loss = result.fun
            best_result = result
    
    sigma_e, k_e, alpha_e, eta_e, eps_e = best_result.x
    Ye = compute_yukawa_matrix_generalized(L, E, sigma_e, k_e, alpha_e, eta_e, eps_e, p)
    obs = compute_lepton_observables(Ye)
    
    return {
        'test_type': 'free_all',
        'sigma_e': sigma_e,
        'k_e': k_e,
        'alpha_e': alpha_e,
        'eta_e': eta_e,
        'eps_e': eps_e,
        'sigma_quark': quark_params['sigma'],
        'k_quark': quark_params['k'],
        'alpha_quark': quark_params['alpha'],
        'eta_quark': quark_params['eta'],
        'delta_sigma': sigma_e - quark_params['sigma'],
        'delta_k': k_e - quark_params['k'],
        'delta_alpha': alpha_e - quark_params['alpha'],
        'delta_eta': eta_e - quark_params['eta'],
        'loss': best_loss,
        **obs
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='True transfer test')
    parser.add_argument('--p', type=float, default=2.0, help='Envelope shape parameter')
    parser.add_argument('--n-seeds', type=int, default=5, help='Seeds per optimization')
    parser.add_argument('--maxiter', type=int, default=200, help='Max iterations')
    parser.add_argument('--output', type=str, default='data/transfer_test_results.csv')
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, '..', args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("=" * 60)
    print("TRUE TRANSFER TEST")
    print("=" * 60)
    print(f"Envelope shape p = {args.p}")
    print(f"Start time: {datetime.now()}")
    
    # Geometries
    quark_geom = {
        'Q': (0, 1, 0),
        'U': (0, 3, 6),
        'D': (0, 3, 7),
    }
    
    lepton_geom = {
        'L': (0, 1, 0),
        'E': (0, 3, 6),
    }
    
    # Step 1: Fit quarks
    print("\n1. Fitting quark sector...")
    quark_result = fit_quarks(
        quark_geom['Q'], quark_geom['U'], quark_geom['D'],
        p=args.p, n_seeds=args.n_seeds, maxiter=args.maxiter
    )
    
    print(f"   Quark fit: CKM loss = {quark_result['loss_ckm']:.6f}")
    print(f"   Parameters: σ={quark_result['sigma']:.3f}, k={quark_result['k']:.3f}, η={quark_result['eta']:.3f}")
    print(f"   Vus={quark_result['Vus']:.4f} (target: 0.225)")
    print(f"   mc={quark_result['mc']:.3f} GeV (target: 1.27 GeV)")
    
    # Step 2: Transfer tests
    print("\n2. Running transfer tests...")
    
    # Test A: Frozen parameters (TRUE universality test)
    print("   A. Frozen parameters (only eps_e free)...")
    frozen_result = transfer_test_frozen(
        lepton_geom['L'], lepton_geom['E'], quark_result,
        p=args.p, n_seeds=args.n_seeds, maxiter=args.maxiter
    )
    print(f"      Loss = {frozen_result['loss']:.4f}")
    print(f"      m_mu = {frozen_result['m_mu']:.6f} GeV (target: 0.1057 GeV)")
    
    # Test B: Free k
    print("   B. Free k (eps_e and k_e free)...")
    free_k_result = transfer_test_free_k(
        lepton_geom['L'], lepton_geom['E'], quark_result,
        p=args.p, n_seeds=args.n_seeds, maxiter=args.maxiter
    )
    print(f"      Loss = {free_k_result['loss']:.4f}")
    print(f"      Δk = {free_k_result['delta_k']:.4f}")
    print(f"      m_mu = {free_k_result['m_mu']:.6f} GeV")
    
    # Test C: Free all
    print("   C. Free all (independent fit)...")
    free_all_result = transfer_test_free_all(
        lepton_geom['L'], lepton_geom['E'], quark_result,
        p=args.p, n_seeds=args.n_seeds, maxiter=args.maxiter
    )
    print(f"      Loss = {free_all_result['loss']:.4f}")
    print(f"      Δσ = {free_all_result['delta_sigma']:.4f}")
    print(f"      Δk = {free_all_result['delta_k']:.4f}")
    print(f"      Δη = {free_all_result['delta_eta']:.4f}")
    print(f"      m_mu = {free_all_result['m_mu']:.6f} GeV")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\nLoss comparison:")
    print(f"  Frozen (true universality): {frozen_result['loss']:.4f}")
    print(f"  Free k only:                {free_k_result['loss']:.4f}")
    print(f"  Free all (independent):     {free_all_result['loss']:.4f}")
    
    improvement_k = (frozen_result['loss'] - free_k_result['loss']) / frozen_result['loss'] * 100
    improvement_all = (frozen_result['loss'] - free_all_result['loss']) / frozen_result['loss'] * 100
    
    print(f"\nImprovement from freeing k: {improvement_k:.1f}%")
    print(f"Improvement from freeing all: {improvement_all:.1f}%")
    
    # Interpretation
    print("\nInterpretation:")
    if frozen_result['loss'] < 0.1:
        print("  ✓ TRUE UNIVERSALITY: Frozen parameters give good lepton fit")
    elif free_k_result['loss'] < 0.1 and improvement_k > 50:
        print("  ⚠ PARTIAL UNIVERSALITY: k must change, but other params transfer")
        print(f"    Required Δk = {free_k_result['delta_k']:.4f}")
    else:
        print("  ✗ NO UNIVERSALITY: Multiple parameters must change")
        print(f"    Required Δσ = {free_all_result['delta_sigma']:.4f}")
        print(f"    Required Δk = {free_all_result['delta_k']:.4f}")
        print(f"    Required Δη = {free_all_result['delta_eta']:.4f}")
    
    # Save results
    results_df = pd.DataFrame([
        {'test': 'quark_fit', 'p': args.p, **quark_result},
        {'test': 'frozen', 'p': args.p, **frozen_result},
        {'test': 'free_k', 'p': args.p, **free_k_result},
        {'test': 'free_all', 'p': args.p, **free_all_result},
    ])
    
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")
    print(f"End time: {datetime.now()}")


if __name__ == "__main__":
    main()
