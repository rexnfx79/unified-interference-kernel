#!/usr/bin/env python3
"""
Diagnostic Test 4: Optimization Settings Verification

Test if the optimization is under-powered:
- Run a single geometry with increased settings
- Compare results to the original data
- Check if the loss landscape has many local minima
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from kernel import compute_quark_yukawas
from observables import compute_quark_observables, compute_ckm_loss, compute_mass_loss, QUARK_TARGETS
from optimizer import optimize_parameters
import time

# Test results tracking
RESULTS = []


def log_result(test_name: str, passed: bool, details: str = ""):
    """Log a test result."""
    status = "PASS" if passed else "FAIL"
    RESULTS.append((test_name, passed, details))
    print(f"[{status}] {test_name}")
    if details:
        print(f"       {details}")


def objective_quark(theta, Q, U, D):
    """Objective function for quark optimization (same as in 01_optimize_quarks.py)."""
    sigma, k, alpha, eta, eps_u, eps_d = theta
    Yu, Yd = compute_quark_yukawas(Q, U, D, sigma, k, alpha, eta, eps_u, eps_d)
    obs = compute_quark_observables(Yu, Yd)
    L_ckm = compute_ckm_loss(obs)
    L_mass = compute_mass_loss(obs)
    
    # Mass floor penalties
    L_md_penalty = 2.0 * (np.log(0.002 / obs['md'])) ** 2 if obs['md'] < 0.002 else 0.0
    L_mu_penalty = 0.5 * (np.log(0.0005 / obs['mu'])) ** 2 if obs['mu'] < 0.0005 else 0.0
    
    return L_mass + 5.0 * L_ckm + L_md_penalty + L_mu_penalty


def test_optimization_convergence():
    """Test if more iterations improve results."""
    print("\n" + "="*60)
    print("TEST: Optimization Convergence")
    print("="*60)
    
    # Use a specific geometry
    Q = (0, 2, 0)
    U = (0, 2, 4)
    D = (0, 2, 4)
    
    # Original bounds from 01_optimize_quarks.py
    bounds = [
        (0.5, 6.0), (0.1, 2.0), (0.0, 2 * np.pi),
        (1.0, 5.0), (0.01, 0.5), (0.01, 0.5),
    ]
    
    print(f"\n  Testing geometry Q={Q}, U={U}, D={D}")
    print(f"  Original settings: maxiter=100, seeds=3")
    
    # Test with different iteration counts
    iteration_tests = [50, 100, 200, 500]
    results_by_iter = {}
    
    for maxiter in iteration_tests:
        start = time.time()
        best_loss = np.inf
        
        for seed in range(5):  # Use 5 seeds for fair comparison
            result = optimize_parameters(
                lambda theta: objective_quark(theta, Q, U, D),
                bounds, maxiter=maxiter, seed=seed, polish=False
            )
            if result['fun'] < best_loss:
                best_loss = result['fun']
        
        elapsed = time.time() - start
        results_by_iter[maxiter] = best_loss
        print(f"    maxiter={maxiter}: best_loss={best_loss:.6f} ({elapsed:.1f}s)")
    
    # Check if more iterations help
    improvement = results_by_iter[50] - results_by_iter[500]
    significant_improvement = improvement > 0.1
    
    log_result("More iterations improve results", significant_improvement,
               f"Loss improved by {improvement:.4f} from 50 to 500 iterations")


def test_seed_sensitivity():
    """Test if results are sensitive to random seed."""
    print("\n" + "="*60)
    print("TEST: Seed Sensitivity (Local Minima)")
    print("="*60)
    
    Q = (0, 2, 0)
    U = (0, 2, 4)
    D = (0, 2, 4)
    
    bounds = [
        (0.5, 6.0), (0.1, 2.0), (0.0, 2 * np.pi),
        (1.0, 5.0), (0.01, 0.5), (0.01, 0.5),
    ]
    
    print(f"\n  Testing 20 different seeds with maxiter=100:")
    
    losses = []
    for seed in range(20):
        result = optimize_parameters(
            lambda theta: objective_quark(theta, Q, U, D),
            bounds, maxiter=100, seed=seed, polish=False
        )
        losses.append(result['fun'])
    
    losses = np.array(losses)
    
    print(f"\n  Loss statistics across seeds:")
    print(f"    Min:    {losses.min():.6f}")
    print(f"    Max:    {losses.max():.6f}")
    print(f"    Mean:   {losses.mean():.6f}")
    print(f"    Std:    {losses.std():.6f}")
    print(f"    Range:  {losses.max() - losses.min():.6f}")
    
    # High variance suggests many local minima
    high_variance = losses.std() > 0.5
    log_result("Loss landscape has local minima", high_variance,
               f"Std={losses.std():.4f}, suggests {'many' if high_variance else 'few'} local minima")


def test_wider_bounds():
    """Test if wider parameter bounds improve results."""
    print("\n" + "="*60)
    print("TEST: Wider Parameter Bounds")
    print("="*60)
    
    Q = (0, 2, 0)
    U = (0, 2, 4)
    D = (0, 2, 4)
    
    # Original bounds
    original_bounds = [
        (0.5, 6.0), (0.1, 2.0), (0.0, 2 * np.pi),
        (1.0, 5.0), (0.01, 0.5), (0.01, 0.5),
    ]
    
    # Wider bounds
    wider_bounds = [
        (0.1, 20.0),   # sigma: much wider
        (0.01, 5.0),   # k: wider
        (0.0, 2 * np.pi),  # alpha: same (periodic)
        (0.1, 10.0),   # eta: wider
        (0.001, 1.0),  # eps_u: wider
        (0.001, 1.0),  # eps_d: wider
    ]
    
    print(f"\n  Original bounds: sigma=[0.5,6], k=[0.1,2], eta=[1,5], eps=[0.01,0.5]")
    print(f"  Wider bounds:    sigma=[0.1,20], k=[0.01,5], eta=[0.1,10], eps=[0.001,1]")
    
    # Test original bounds
    best_original = np.inf
    for seed in range(10):
        result = optimize_parameters(
            lambda theta: objective_quark(theta, Q, U, D),
            original_bounds, maxiter=200, seed=seed, polish=False
        )
        if result['fun'] < best_original:
            best_original = result['fun']
    
    # Test wider bounds
    best_wider = np.inf
    best_wider_params = None
    for seed in range(10):
        result = optimize_parameters(
            lambda theta: objective_quark(theta, Q, U, D),
            wider_bounds, maxiter=200, seed=seed, polish=False
        )
        if result['fun'] < best_wider:
            best_wider = result['fun']
            best_wider_params = result['x']
    
    print(f"\n  Best loss with original bounds: {best_original:.6f}")
    print(f"  Best loss with wider bounds:    {best_wider:.6f}")
    
    if best_wider_params is not None:
        print(f"\n  Best parameters with wider bounds:")
        print(f"    sigma={best_wider_params[0]:.4f}, k={best_wider_params[1]:.4f}")
        print(f"    alpha={best_wider_params[2]:.4f}, eta={best_wider_params[3]:.4f}")
        print(f"    eps_u={best_wider_params[4]:.4f}, eps_d={best_wider_params[5]:.4f}")
    
    improvement = best_original - best_wider
    significant_improvement = improvement > 0.1
    
    log_result("Wider bounds improve results", significant_improvement,
               f"Improvement: {improvement:.4f}")


def test_best_possible_fit():
    """Find the best possible fit for a geometry and analyze what's limiting."""
    print("\n" + "="*60)
    print("TEST: Best Possible Fit Analysis")
    print("="*60)
    
    Q = (0, 2, 0)
    U = (0, 2, 4)
    D = (0, 2, 4)
    
    # Very wide bounds and many iterations
    wide_bounds = [
        (0.1, 50.0),   # sigma
        (0.001, 10.0), # k
        (0.0, 2 * np.pi),
        (0.01, 20.0),  # eta
        (0.0001, 2.0), # eps_u
        (0.0001, 2.0), # eps_d
    ]
    
    print(f"\n  Running intensive optimization (maxiter=500, 20 seeds)...")
    
    best_loss = np.inf
    best_params = None
    
    for seed in range(20):
        result = optimize_parameters(
            lambda theta: objective_quark(theta, Q, U, D),
            wide_bounds, maxiter=500, seed=seed, polish=True  # Enable polishing
        )
        if result['fun'] < best_loss:
            best_loss = result['fun']
            best_params = result['x']
    
    print(f"\n  Best achievable loss: {best_loss:.6f}")
    
    if best_params is not None:
        # Compute observables at best point
        sigma, k, alpha, eta, eps_u, eps_d = best_params
        Yu, Yd = compute_quark_yukawas(Q, U, D, sigma, k, alpha, eta, eps_u, eps_d)
        obs = compute_quark_observables(Yu, Yd)
        
        print(f"\n  Best parameters:")
        print(f"    sigma={sigma:.4f}, k={k:.4f}, alpha={alpha:.4f}")
        print(f"    eta={eta:.4f}, eps_u={eps_u:.4f}, eps_d={eps_d:.4f}")
        
        print(f"\n  Observables at best point:")
        print(f"    CKM: Vus={obs['Vus']:.4f} (target: {QUARK_TARGETS['Vus']:.4f})")
        print(f"         Vcb={obs['Vcb']:.4f} (target: {QUARK_TARGETS['Vcb']:.4f})")
        print(f"         Vub={obs['Vub']:.4f} (target: {QUARK_TARGETS['Vub']:.4f})")
        print(f"    Masses: mu={obs['mu']:.6f} (target: {QUARK_TARGETS['mu']:.6f})")
        print(f"            mc={obs['mc']:.4f} (target: {QUARK_TARGETS['mc']:.4f})")
        print(f"            md={obs['md']:.6f} (target: {QUARK_TARGETS['md']:.6f})")
        print(f"            ms={obs['ms']:.4f} (target: {QUARK_TARGETS['ms']:.4f})")
        
        # Compute individual losses
        L_ckm = compute_ckm_loss(obs)
        L_mass = compute_mass_loss(obs)
        
        print(f"\n  Loss breakdown:")
        print(f"    CKM loss:  {L_ckm:.6f}")
        print(f"    Mass loss: {L_mass:.6f}")
        
        # What's the limiting factor?
        mc_error = abs(obs['mc'] - QUARK_TARGETS['mc']) / QUARK_TARGETS['mc']
        ckm_ok = L_ckm < 0.1
        mc_ok = mc_error < 0.5
        
        print(f"\n  Limiting factors:")
        print(f"    CKM acceptable (loss < 0.1): {'YES' if ckm_ok else 'NO'}")
        print(f"    mc acceptable (error < 50%): {'YES' if mc_ok else 'NO'}")
        
        if not mc_ok:
            print(f"\n  *** mc is the limiting factor ***")
            print(f"  mc error: {mc_error*100:.1f}%")


def compare_with_original_data():
    """Compare intensive optimization with original data."""
    print("\n" + "="*60)
    print("TEST: Comparison with Original Data")
    print("="*60)
    
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'quark_results.csv')
    
    if not os.path.exists(data_path):
        log_result("Load original data", False, f"File not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    
    # Find the best result in original data
    best_idx = df['loss_total'].idxmin()
    best_original = df.loc[best_idx]
    
    print(f"\n  Best result in original data:")
    print(f"    Total loss: {best_original['loss_total']:.6f}")
    print(f"    CKM loss:   {best_original['loss_ckm']:.6f}")
    print(f"    Mass loss:  {best_original['loss_mass']:.6f}")
    print(f"    mc:         {best_original['mc']:.4f} (target: {QUARK_TARGETS['mc']:.4f})")
    
    # Get the geometry
    Q = (int(best_original['Q1']), int(best_original['Q2']), 0)
    U = (int(best_original['U1']), int(best_original['U2']), int(best_original['U3']))
    D = (int(best_original['D1']), int(best_original['D2']), int(best_original['D3']))
    
    print(f"\n  Re-optimizing this geometry with better settings...")
    
    # Re-optimize with better settings
    wide_bounds = [
        (0.1, 20.0), (0.01, 5.0), (0.0, 2 * np.pi),
        (0.1, 10.0), (0.001, 1.0), (0.001, 1.0),
    ]
    
    best_loss = np.inf
    for seed in range(15):
        result = optimize_parameters(
            lambda theta: objective_quark(theta, Q, U, D),
            wide_bounds, maxiter=300, seed=seed, polish=True
        )
        if result['fun'] < best_loss:
            best_loss = result['fun']
    
    improvement = best_original['loss_total'] - best_loss
    
    print(f"\n  Re-optimized loss: {best_loss:.6f}")
    print(f"  Improvement:       {improvement:.6f}")
    
    significant = improvement > 0.1
    log_result("Re-optimization improves results", significant,
               f"Improvement: {improvement:.4f} ({100*improvement/best_original['loss_total']:.1f}%)")


def main():
    """Run all optimization tests."""
    print("="*60)
    print("DIAGNOSTIC 4: OPTIMIZATION SETTINGS VERIFICATION")
    print("="*60)
    
    test_optimization_convergence()
    test_seed_sensitivity()
    test_wider_bounds()
    test_best_possible_fit()
    compare_with_original_data()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, p, _ in RESULTS if p)
    total = len(RESULTS)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    # Key insight
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print("""
  This diagnostic tests whether the optimization settings are
  limiting the model's performance. Key findings:
  
  1. If more iterations significantly improve results, the original
     optimization was under-powered.
  
  2. If seed sensitivity is high, the loss landscape has many local
     minima, requiring more seeds.
  
  3. If wider bounds help, the original bounds were too restrictive.
  
  4. If even intensive optimization cannot achieve good mc values,
     the model has a fundamental limitation.
""")
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, '04_optimization_results.txt'), 'w') as f:
        f.write("OPTIMIZATION SETTINGS VERIFICATION RESULTS\n")
        f.write("="*50 + "\n\n")
        for name, p, details in RESULTS:
            status = "PASS" if p else "FAIL"
            f.write(f"[{status}] {name}\n")
            if details:
                f.write(f"        {details}\n")
        f.write(f"\nTotal: {passed}/{total} tests passed\n")
    
    print(f"\nResults saved to diagnostics/results/04_optimization_results.txt")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
