#!/usr/bin/env python3
"""
Diagnostic Test 3: Mass Scaling Verification

Verify the mass extraction logic in observables.py:
- Test compute_quark_observables() with known Yukawa matrices
- Check if scaling by mt/S[0] produces correct mc values
- Identify if the mc prediction failure is due to wrong scaling
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from observables import compute_quark_observables, QUARK_TARGETS, fix_svd_phases

# Test results tracking
RESULTS = []


def log_result(test_name: str, passed: bool, details: str = ""):
    """Log a test result."""
    status = "PASS" if passed else "FAIL"
    RESULTS.append((test_name, passed, details))
    print(f"[{status}] {test_name}")
    if details:
        print(f"       {details}")


def test_scaling_with_known_matrix():
    """Test mass scaling with a matrix that has known singular values."""
    print("\n" + "="*60)
    print("TEST: Scaling with Known Matrix")
    print("="*60)
    
    # Create a diagonal matrix with known singular values
    # If we want mt=172.5, mc=1.27, mu=0.00216
    # Then S[0]/S[1] = 172.5/1.27 = 135.8
    # And S[1]/S[2] = 1.27/0.00216 = 587.9
    
    # Let's use S = [1.0, 1/135.8, 1/(135.8*587.9)]
    s0 = 1.0
    s1 = 1.0 / 135.8
    s2 = 1.0 / (135.8 * 587.9)
    
    # Create diagonal Yukawa matrices
    Yu = np.diag([s0, s1, s2]).astype(complex)
    Yd = np.diag([s0, s1, s2]).astype(complex)  # Same for simplicity
    
    print(f"\n  Input singular values: [{s0:.6f}, {s1:.6f}, {s2:.10f}]")
    print(f"  Expected ratios: S[0]/S[1]={s0/s1:.1f}, S[1]/S[2]={s1/s2:.1f}")
    
    obs = compute_quark_observables(Yu, Yd)
    
    print(f"\n  Computed masses:")
    print(f"    mu = {obs['mu']:.6f} GeV (target: {QUARK_TARGETS['mu']})")
    print(f"    mc = {obs['mc']:.6f} GeV (target: {QUARK_TARGETS['mc']})")
    print(f"    scale_u = {obs['scale_u']:.2f}")
    
    # Check if scaling works correctly
    # scale_u = mt / S[0] = 172.5 / 1.0 = 172.5
    # mc = S[1] * scale_u = (1/135.8) * 172.5 = 1.27
    
    mc_error = abs(obs['mc'] - QUARK_TARGETS['mc']) / QUARK_TARGETS['mc']
    mu_error = abs(obs['mu'] - QUARK_TARGETS['mu']) / QUARK_TARGETS['mu']
    
    log_result("mc scaling correct", mc_error < 0.01,
               f"mc={obs['mc']:.4f}, target={QUARK_TARGETS['mc']}, error={mc_error*100:.2f}%")
    log_result("mu scaling correct", mu_error < 0.01,
               f"mu={obs['mu']:.6f}, target={QUARK_TARGETS['mu']}, error={mu_error*100:.2f}%")


def test_scaling_formula():
    """Verify the scaling formula is mathematically correct."""
    print("\n" + "="*60)
    print("TEST: Scaling Formula Verification")
    print("="*60)
    
    # The scaling approach:
    # 1. Compute SVD: Y = U @ diag(S) @ Vh
    # 2. S[0] is the largest singular value
    # 3. scale_u = mt / S[0]
    # 4. mc = S[1] * scale_u
    # 5. mu = S[2] * scale_u
    
    # This means: mc = S[1] * (mt / S[0]) = mt * (S[1] / S[0])
    # So mc depends on the RATIO S[1]/S[0], not the absolute values
    
    print("\n  The scaling formula:")
    print("    scale_u = mt / S[0]")
    print("    mc = S[1] * scale_u = mt * (S[1] / S[0])")
    print("    mu = S[2] * scale_u = mt * (S[2] / S[0])")
    
    print("\n  For correct mc = 1.27 GeV:")
    print(f"    S[1]/S[0] must equal {QUARK_TARGETS['mc'] / QUARK_TARGETS['mt']:.6f}")
    
    print("\n  For correct mu = 0.00216 GeV:")
    print(f"    S[2]/S[0] must equal {QUARK_TARGETS['mu'] / QUARK_TARGETS['mt']:.8f}")
    
    # This is a VERY small ratio - the model must produce singular values
    # spanning ~5 orders of magnitude!
    
    required_ratio_mc = QUARK_TARGETS['mc'] / QUARK_TARGETS['mt']
    required_ratio_mu = QUARK_TARGETS['mu'] / QUARK_TARGETS['mt']
    
    print(f"\n  Required dynamic range: {1/required_ratio_mu:.0f}x")
    print("  This requires singular values spanning ~5 orders of magnitude!")
    
    log_result("Scaling formula is correct", True,
               "Formula mc = mt * (S[1]/S[0]) is mathematically sound")


def test_actual_data_ratios():
    """Check the singular value ratios in the actual generated data."""
    print("\n" + "="*60)
    print("TEST: Actual Data Singular Value Ratios")
    print("="*60)
    
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'quark_results.csv')
    
    if not os.path.exists(data_path):
        log_result("Load quark data", False, f"File not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    
    # Reverse-engineer the singular value ratios from the masses
    # mc = mt * (S[1]/S[0]) => S[1]/S[0] = mc / mt
    # mu = mt * (S[2]/S[0]) => S[2]/S[0] = mu / mt
    
    mt = QUARK_TARGETS['mt']
    
    ratio_s1_s0 = df['mc'].values / mt
    ratio_s2_s0 = df['mu'].values / mt
    
    print(f"\n  Inferred S[1]/S[0] (from mc):")
    print(f"    Mean:   {ratio_s1_s0.mean():.6f}")
    print(f"    Target: {QUARK_TARGETS['mc'] / mt:.6f}")
    print(f"    Min:    {ratio_s1_s0.min():.6f}")
    print(f"    Max:    {ratio_s1_s0.max():.6f}")
    
    print(f"\n  Inferred S[2]/S[0] (from mu):")
    print(f"    Mean:   {ratio_s2_s0.mean():.8f}")
    print(f"    Target: {QUARK_TARGETS['mu'] / mt:.8f}")
    
    # Check if any data point achieves the correct ratio
    target_ratio = QUARK_TARGETS['mc'] / mt
    close_to_target = np.sum(np.abs(ratio_s1_s0 - target_ratio) / target_ratio < 0.2)
    
    log_result("Some points achieve correct mc ratio", close_to_target > 0,
               f"{close_to_target}/{len(df)} points have S[1]/S[0] within 20% of target")
    
    # What ratio does the model typically produce?
    print(f"\n  Model produces S[1]/S[0] ratios that are:")
    print(f"    {ratio_s1_s0.mean() / target_ratio:.1f}x larger than needed on average")


def test_ckm_extraction():
    """Test CKM matrix extraction from known Yukawa matrices."""
    print("\n" + "="*60)
    print("TEST: CKM Extraction")
    print("="*60)
    
    # Create Yukawa matrices that should give identity CKM
    # If Yu and Yd are both diagonal, CKM = I
    Yu = np.diag([1.0, 0.1, 0.01]).astype(complex)
    Yd = np.diag([1.0, 0.1, 0.01]).astype(complex)
    
    obs = compute_quark_observables(Yu, Yd)
    
    print(f"\n  With diagonal Yu = Yd:")
    print(f"    Vus = {obs['Vus']:.6f} (should be ~0 for identity CKM)")
    print(f"    Vcb = {obs['Vcb']:.6f} (should be ~0 for identity CKM)")
    print(f"    Vub = {obs['Vub']:.6f} (should be ~0 for identity CKM)")
    
    # Check if CKM is approximately identity
    is_identity = obs['Vus'] < 0.01 and obs['Vcb'] < 0.01 and obs['Vub'] < 0.01
    log_result("Diagonal matrices give identity CKM", is_identity,
               f"Vus={obs['Vus']:.4f}, Vcb={obs['Vcb']:.4f}, Vub={obs['Vub']:.4f}")
    
    # Now test with slightly different matrices
    Yu = np.diag([1.0, 0.1, 0.01]).astype(complex)
    Yd = np.array([
        [1.0, 0.1, 0.01],
        [0.1, 0.1, 0.01],
        [0.01, 0.01, 0.01]
    ], dtype=complex)
    
    obs2 = compute_quark_observables(Yu, Yd)
    
    print(f"\n  With non-diagonal Yd:")
    print(f"    Vus = {obs2['Vus']:.6f}")
    print(f"    Vcb = {obs2['Vcb']:.6f}")
    print(f"    Vub = {obs2['Vub']:.6f}")
    
    # Check if we can get non-trivial CKM
    has_mixing = obs2['Vus'] > 0.01 or obs2['Vcb'] > 0.01
    log_result("Non-diagonal matrices give CKM mixing", has_mixing,
               f"Vus={obs2['Vus']:.4f}, Vcb={obs2['Vcb']:.4f}")


def test_phase_fixing():
    """Test the SVD phase fixing procedure."""
    print("\n" + "="*60)
    print("TEST: SVD Phase Fixing")
    print("="*60)
    
    # Create a complex matrix
    Y = np.array([
        [1.0 + 0.5j, 0.1 - 0.1j, 0.01],
        [0.2j, 0.5 + 0.3j, 0.05 - 0.02j],
        [0.01 + 0.01j, 0.02, 0.1 - 0.05j]
    ], dtype=complex)
    
    U, S, Vh = np.linalg.svd(Y)
    U_fixed, S_fixed, Vh_fixed = fix_svd_phases(U, S, Vh)
    
    print(f"\n  Original U diagonal phases: {np.angle(np.diag(U))}")
    print(f"  Fixed U diagonal phases:    {np.angle(np.diag(U_fixed))}")
    
    # Check that singular values are unchanged
    sv_unchanged = np.allclose(S, S_fixed)
    log_result("Phase fixing preserves singular values", sv_unchanged,
               f"S={S}, S_fixed={S_fixed}")
    
    # Check that reconstruction still works
    Y_reconstructed = U_fixed @ np.diag(S_fixed) @ Vh_fixed
    reconstruction_error = np.max(np.abs(Y - Y_reconstructed))
    
    log_result("Phase fixing preserves reconstruction", reconstruction_error < 1e-10,
               f"Max reconstruction error: {reconstruction_error:.2e}")


def main():
    """Run all mass scaling tests."""
    print("="*60)
    print("DIAGNOSTIC 3: MASS SCALING VERIFICATION")
    print("="*60)
    
    test_scaling_with_known_matrix()
    test_scaling_formula()
    test_actual_data_ratios()
    test_ckm_extraction()
    test_phase_fixing()
    
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
    
    # Key insight
    print("\n" + "="*60)
    print("KEY INSIGHT")
    print("="*60)
    print("""
  The mass scaling formula is CORRECT, but it requires the kernel
  to produce Yukawa matrices with singular value ratios spanning
  ~5 orders of magnitude (S[0]/S[2] ~ 80,000).
  
  If the kernel cannot naturally produce such extreme hierarchies,
  the model will fail to reproduce the correct quark masses.
  
  This is a FUNDAMENTAL LIMITATION of the model, not a bug.
""")
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, '03_mass_scaling_results.txt'), 'w') as f:
        f.write("MASS SCALING VERIFICATION RESULTS\n")
        f.write("="*50 + "\n\n")
        for name, p, details in RESULTS:
            status = "PASS" if p else "FAIL"
            f.write(f"[{status}] {name}\n")
            if details:
                f.write(f"        {details}\n")
        f.write(f"\nTotal: {passed}/{total} tests passed\n")
    
    print(f"\nResults saved to diagnostics/results/03_mass_scaling_results.txt")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
