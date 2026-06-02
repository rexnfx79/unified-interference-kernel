#!/usr/bin/env python3
"""
Diagnostic Test 2: SVD Hierarchy Verification

Check the assumption that SVD singular values map to (top, charm, up) masses.
The code assumes S[0] > S[1] > S[2] and that these correspond to the heaviest,
middle, and lightest quarks respectively.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from kernel import compute_quark_yukawas

# Test results tracking
RESULTS = []


def log_result(test_name: str, passed: bool, details: str = ""):
    """Log a test result."""
    status = "PASS" if passed else "FAIL"
    RESULTS.append((test_name, passed, details))
    print(f"[{status}] {test_name}")
    if details:
        print(f"       {details}")


def test_svd_ordering():
    """Test that SVD always returns singular values in descending order."""
    print("\n" + "="*60)
    print("TEST: SVD Ordering (S[0] >= S[1] >= S[2])")
    print("="*60)
    
    # Generate random Yukawa matrices and check ordering
    np.random.seed(42)
    n_tests = 100
    all_ordered = True
    
    for _ in range(n_tests):
        # Random 3x3 complex matrix
        Y = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
        _, S, _ = np.linalg.svd(Y)
        
        if not (S[0] >= S[1] >= S[2]):
            all_ordered = False
            print(f"  FAIL: S = {S}")
    
    log_result("SVD returns descending singular values", all_ordered,
               f"Tested {n_tests} random matrices")
    
    print("\n  Note: numpy.linalg.svd ALWAYS returns singular values in descending order.")
    print("  This is guaranteed by the SVD algorithm.")


def test_hierarchy_from_kernel():
    """Test if kernel-generated Yukawa matrices produce realistic hierarchies."""
    print("\n" + "="*60)
    print("TEST: Mass Hierarchy from Kernel")
    print("="*60)
    
    # Expected quark mass ratios (approximate)
    # mt/mc ≈ 172.5/1.27 ≈ 136
    # mc/mu ≈ 1.27/0.00216 ≈ 588
    # mt/mu ≈ 172.5/0.00216 ≈ 79861
    
    expected_mt_mc = 172.5 / 1.27  # ~136
    expected_mc_mu = 1.27 / 0.00216  # ~588
    
    print(f"\n  Expected ratios (from PDG):")
    print(f"    mt/mc ≈ {expected_mt_mc:.1f}")
    print(f"    mc/mu ≈ {expected_mc_mu:.1f}")
    
    # Test with various geometries and parameters
    test_cases = [
        # (Q, U, D, sigma, k, alpha, eta, eps_u, eps_d)
        ((0, 1, 0), (0, 1, 2), (0, 1, 2), 2.0, 0.5, 0.3, 0.1, 0.2, 0.2),
        ((0, 2, 0), (0, 2, 4), (0, 2, 4), 1.5, 0.3, 0.5, 0.2, 0.3, 0.3),
        ((1, 3, 0), (0, 3, 6), (0, 3, 6), 3.0, 0.7, 1.0, 0.3, 0.1, 0.1),
        ((0, 4, 0), (0, 4, 8), (0, 4, 8), 2.5, 0.4, 0.8, 0.15, 0.25, 0.25),
    ]
    
    print(f"\n  Testing {len(test_cases)} geometries:")
    
    hierarchies = []
    for Q, U, D, sigma, k, alpha, eta, eps_u, eps_d in test_cases:
        Yu, Yd = compute_quark_yukawas(Q, U, D, sigma, k, alpha, eta, eps_u, eps_d)
        _, Su, _ = np.linalg.svd(Yu)
        
        ratio_01 = Su[0] / Su[1] if Su[1] > 0 else np.inf  # "mt/mc" ratio
        ratio_12 = Su[1] / Su[2] if Su[2] > 0 else np.inf  # "mc/mu" ratio
        
        hierarchies.append((ratio_01, ratio_12))
        print(f"    Q={Q}, U={U}: S[0]/S[1]={ratio_01:.2f}, S[1]/S[2]={ratio_12:.2f}")
    
    # Check if any geometry produces realistic hierarchies
    realistic = any(
        r01 > 10 and r12 > 10  # At least order of magnitude hierarchies
        for r01, r12 in hierarchies
    )
    
    log_result("Kernel can produce mass hierarchies", realistic,
               "At least one geometry produces S[0]/S[1] > 10 and S[1]/S[2] > 10")
    
    # Check if hierarchies are LARGE ENOUGH
    max_ratio_01 = max(h[0] for h in hierarchies)
    max_ratio_12 = max(h[1] for h in hierarchies)
    
    sufficient_hierarchy = max_ratio_01 > 50 and max_ratio_12 > 50
    log_result("Hierarchies are sufficient magnitude", sufficient_hierarchy,
               f"Max S[0]/S[1]={max_ratio_01:.1f}, Max S[1]/S[2]={max_ratio_12:.1f}")


def test_hierarchy_from_data():
    """Check actual hierarchies in the generated data."""
    print("\n" + "="*60)
    print("TEST: Hierarchy from Generated Data")
    print("="*60)
    
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'quark_results.csv')
    
    if not os.path.exists(data_path):
        log_result("Load quark data", False, f"File not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    print(f"\n  Loaded {len(df)} rows from quark_results.csv")
    
    # Check the mc values in the data
    mc_values = df['mc'].values
    mc_target = 1.27  # GeV
    
    print(f"\n  mc statistics:")
    print(f"    Target: {mc_target} GeV")
    print(f"    Mean:   {mc_values.mean():.4f} GeV")
    print(f"    Std:    {mc_values.std():.4f} GeV")
    print(f"    Min:    {mc_values.min():.4f} GeV")
    print(f"    Max:    {mc_values.max():.4f} GeV")
    
    # How many are within 50% of target?
    within_50pct = np.sum(np.abs(mc_values - mc_target) / mc_target < 0.5)
    pct_within = 100 * within_50pct / len(mc_values)
    
    log_result("mc within 50% of target", pct_within > 10,
               f"{within_50pct}/{len(df)} ({pct_within:.1f}%) have mc within 50% of {mc_target} GeV")
    
    # Check mu values
    mu_values = df['mu'].values
    mu_target = 0.00216  # GeV
    
    print(f"\n  mu statistics:")
    print(f"    Target: {mu_target} GeV")
    print(f"    Mean:   {mu_values.mean():.6f} GeV")
    print(f"    Std:    {mu_values.std():.6f} GeV")
    print(f"    Min:    {mu_values.min():.6f} GeV")
    print(f"    Max:    {mu_values.max():.6f} GeV")
    
    # Check the actual mass ratios in the data
    # Note: mt is fixed at 172.5 by scaling
    mt = 172.5
    
    ratio_mt_mc = mt / mc_values
    ratio_mc_mu = mc_values / mu_values
    
    print(f"\n  Mass ratio statistics:")
    print(f"    mt/mc: mean={ratio_mt_mc.mean():.1f}, target={mt/mc_target:.1f}")
    print(f"    mc/mu: mean={ratio_mc_mu.mean():.1f}, target={mc_target/mu_target:.1f}")


def test_svd_phase_sensitivity():
    """Test if SVD results are sensitive to phase choices."""
    print("\n" + "="*60)
    print("TEST: SVD Phase Sensitivity")
    print("="*60)
    
    # Same geometry, different phases
    Q = (0, 2, 0)
    U = (0, 2, 4)
    D = (0, 2, 4)
    sigma = 2.0
    k = 0.5
    eta = 0.2
    eps_u = 0.3
    eps_d = 0.3
    
    print("\n  Testing how alpha affects singular values:")
    
    alphas = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]
    sv_results = []
    
    for alpha in alphas:
        Yu, _ = compute_quark_yukawas(Q, U, D, sigma, k, alpha, eta, eps_u, eps_d)
        _, Su, _ = np.linalg.svd(Yu)
        sv_results.append(Su)
        print(f"    α={alpha:.4f}: S = [{Su[0]:.4f}, {Su[1]:.4f}, {Su[2]:.4f}]")
    
    # Check variation
    sv_array = np.array(sv_results)
    variation = sv_array.std(axis=0) / sv_array.mean(axis=0)
    
    print(f"\n  Coefficient of variation: {variation}")
    
    significant_variation = np.any(variation > 0.1)
    log_result("Phase affects singular values", significant_variation,
               f"CV = {variation}, significant if > 0.1")


def main():
    """Run all SVD hierarchy tests."""
    print("="*60)
    print("DIAGNOSTIC 2: SVD HIERARCHY VERIFICATION")
    print("="*60)
    
    test_svd_ordering()
    test_hierarchy_from_kernel()
    test_hierarchy_from_data()
    test_svd_phase_sensitivity()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, p, _ in RESULTS if p)
    total = len(RESULTS)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed < total:
        print("\nFAILED TESTS:")
        for name, p, details in RESULTS:
            if not p:
                print(f"  - {name}")
                if details:
                    print(f"    {details}")
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, '02_svd_hierarchy_results.txt'), 'w') as f:
        f.write("SVD HIERARCHY VERIFICATION RESULTS\n")
        f.write("="*50 + "\n\n")
        for name, p, details in RESULTS:
            status = "PASS" if p else "FAIL"
            f.write(f"[{status}] {name}\n")
            if details:
                f.write(f"        {details}\n")
        f.write(f"\nTotal: {passed}/{total} tests passed\n")
    
    print(f"\nResults saved to diagnostics/results/02_svd_hierarchy_results.txt")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
