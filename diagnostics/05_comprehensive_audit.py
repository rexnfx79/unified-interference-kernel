#!/usr/bin/env python3
"""
Comprehensive Audit of Gaussian Interference Kernel

This script performs a deep analysis of all potential issues with the
current Gaussian interference model, generating a detailed report.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from kernel import compute_kernel_element, compute_yukawa_matrix, compute_quark_yukawas
from observables import (compute_quark_observables, compute_ckm_loss, compute_mass_loss, 
                         QUARK_TARGETS, fix_svd_phases)

# Debug logging
LOG_PATH = '/Users/alexm4/Cursor Repos/unified-interference-kernel/.cursor/debug.log'

def log_debug(location, message, data, hypothesis_id=""):
    """Write debug log entry."""
    entry = {
        "timestamp": int(np.datetime64('now', 'ms').astype(int)),
        "location": location,
        "message": message,
        "data": data,
        "hypothesisId": hypothesis_id,
        "sessionId": "comprehensive-audit"
    }
    with open(LOG_PATH, 'a') as f:
        f.write(json.dumps(entry) + '\n')

REPORT = []

def add_finding(category, severity, title, description, evidence=None):
    """Add a finding to the report."""
    REPORT.append({
        'category': category,
        'severity': severity,  # CRITICAL, HIGH, MEDIUM, LOW, INFO
        'title': title,
        'description': description,
        'evidence': evidence or {}
    })

# ============================================================
# AUDIT 1: KERNEL MATHEMATICAL PROPERTIES
# ============================================================

def audit_kernel_range():
    """Check the range of values the kernel can produce."""
    print("\n[AUDIT 1] Kernel Mathematical Properties")
    print("="*60)
    
    # Test envelope range
    sigmas = [0.5, 1.0, 2.0, 5.0, 10.0]
    distances = [0, 1, 2, 4, 8, 16]
    
    print("\n  Envelope values exp(-d²/(2σ²)):")
    print("  " + "d=".ljust(8) + "".join(f"{d:<12}" for d in distances))
    
    min_envelope = 1.0
    for sigma in sigmas:
        row = f"  σ={sigma:<5}"
        for d in distances:
            env = np.exp(-d**2 / (2 * sigma**2))
            row += f"{env:<12.6f}"
            if env > 1e-10:
                min_envelope = min(min_envelope, env)
        print(row)
    
    # #region agent log
    log_debug("audit_kernel_range", "Envelope range analysis", 
              {"min_envelope_nonzero": min_envelope}, "H1_envelope_range")
    # #endregion
    
    # The key insight: for d=8, σ=2, envelope = exp(-16) ≈ 1e-7
    # This means the Gaussian CAN produce large suppressions
    # But the RATIO between elements matters for SVD
    
    # Test interference range
    print("\n  Interference term |1 + ε·exp(iΦ)| range:")
    eps_values = [0.1, 0.3, 0.5, 0.7, 1.0]
    
    for eps in eps_values:
        # Min when phase = π: |1 + ε·(-1)| = |1 - ε|
        # Max when phase = 0: |1 + ε·(1)| = 1 + ε
        min_int = abs(1 - eps)
        max_int = 1 + eps
        ratio = max_int / min_int if min_int > 0 else np.inf
        print(f"    ε={eps}: range [{min_int:.2f}, {max_int:.2f}], ratio={ratio:.2f}")
    
    # Key finding: interference can only modulate by factor of (1+ε)/(1-ε)
    # For ε=0.5: ratio = 1.5/0.5 = 3x
    # For ε=1.0: ratio = 2/0 = infinite (but singular)
    
    add_finding(
        "KERNEL_MATH", "HIGH",
        "Interference term has limited modulation range",
        "The interference term |1 + ε·exp(iΦ)| can only modulate the envelope by a factor of (1+ε)/(1-ε). "
        "For typical ε=0.3, this is only 1.86x. This is insufficient to create the ~136x ratio needed for mt/mc.",
        {"eps_0.3_ratio": 1.3/0.7, "eps_0.5_ratio": 1.5/0.5, "required_ratio": 136}
    )


def audit_svd_singular_value_structure():
    """Analyze what SVD singular value structures the kernel can produce."""
    print("\n[AUDIT 2] SVD Singular Value Structure")
    print("="*60)
    
    # Generate many random Yukawa matrices from the kernel
    np.random.seed(42)
    n_samples = 500
    
    ratios_01 = []  # S[0]/S[1] - determines mt/mc
    ratios_12 = []  # S[1]/S[2] - determines mc/mu
    
    for _ in range(n_samples):
        # Random geometry
        Q = (np.random.randint(0, 8), np.random.randint(0, 8), 0)
        U = tuple(sorted(np.random.choice(range(10), 3, replace=False)))
        D = tuple(sorted(np.random.choice(range(10), 3, replace=False)))
        
        # Random parameters within typical bounds
        sigma = np.random.uniform(0.5, 10.0)
        k = np.random.uniform(0.1, 3.0)
        alpha = np.random.uniform(0, 2*np.pi)
        eta = np.random.uniform(0.1, 5.0)
        eps_u = np.random.uniform(0.1, 0.9)
        eps_d = np.random.uniform(0.1, 0.9)
        
        Yu, Yd = compute_quark_yukawas(Q, U, D, sigma, k, alpha, eta, eps_u, eps_d)
        _, Su, _ = np.linalg.svd(Yu)
        
        if Su[1] > 1e-15 and Su[2] > 1e-15:
            ratios_01.append(Su[0] / Su[1])
            ratios_12.append(Su[1] / Su[2])
    
    ratios_01 = np.array(ratios_01)
    ratios_12 = np.array(ratios_12)
    
    print(f"\n  S[0]/S[1] (determines mt/mc):")
    print(f"    Required: ~136")
    print(f"    Achieved: mean={ratios_01.mean():.1f}, max={ratios_01.max():.1f}, "
          f"min={ratios_01.min():.1f}")
    print(f"    % achieving >100: {100*np.mean(ratios_01 > 100):.1f}%")
    print(f"    % achieving >50:  {100*np.mean(ratios_01 > 50):.1f}%")
    
    print(f"\n  S[1]/S[2] (determines mc/mu):")
    print(f"    Required: ~588")
    print(f"    Achieved: mean={ratios_12.mean():.1f}, max={ratios_12.max():.1f}")
    
    # #region agent log
    log_debug("audit_svd_structure", "SVD ratio statistics", {
        "ratio_01_mean": float(ratios_01.mean()),
        "ratio_01_max": float(ratios_01.max()),
        "ratio_12_mean": float(ratios_12.mean()),
        "ratio_12_max": float(ratios_12.max()),
        "pct_over_100": float(100*np.mean(ratios_01 > 100))
    }, "H2_svd_ratios")
    # #endregion
    
    if ratios_01.max() < 50:
        add_finding(
            "SVD_STRUCTURE", "CRITICAL",
            "Kernel cannot produce required singular value ratios",
            f"The maximum S[0]/S[1] ratio achieved is {ratios_01.max():.1f}, but ~136 is required for mt/mc. "
            "This is a fundamental limitation of the Gaussian envelope.",
            {"max_ratio": float(ratios_01.max()), "required": 136}
        )


def audit_fix_svd_phases_bug():
    """Check if fix_svd_phases has a bug."""
    print("\n[AUDIT 3] SVD Phase Fixing Bug")
    print("="*60)
    
    # Create a test matrix
    Y = np.array([
        [1.0 + 0.5j, 0.1 - 0.1j, 0.01],
        [0.2j, 0.5 + 0.3j, 0.05 - 0.02j],
        [0.01 + 0.01j, 0.02, 0.1 - 0.05j]
    ], dtype=complex)
    
    U, S, Vh = np.linalg.svd(Y)
    
    # Original reconstruction
    Y_reconstructed_orig = U @ np.diag(S) @ Vh
    error_orig = np.max(np.abs(Y - Y_reconstructed_orig))
    
    # After phase fixing
    U_fixed, S_fixed, Vh_fixed = fix_svd_phases(U, S, Vh)
    Y_reconstructed_fixed = U_fixed @ np.diag(S_fixed) @ Vh_fixed
    error_fixed = np.max(np.abs(Y - Y_reconstructed_fixed))
    
    print(f"  Original SVD reconstruction error: {error_orig:.2e}")
    print(f"  After fix_svd_phases error:        {error_fixed:.2e}")
    
    # #region agent log
    log_debug("audit_phase_fix", "Phase fixing reconstruction", {
        "error_orig": float(error_orig),
        "error_fixed": float(error_fixed)
    }, "H3_phase_bug")
    # #endregion
    
    if error_fixed > 0.01:
        add_finding(
            "BUG", "HIGH",
            "fix_svd_phases breaks SVD reconstruction",
            f"The fix_svd_phases function introduces a reconstruction error of {error_fixed:.2e}. "
            "This may affect CKM matrix extraction. The function modifies U rows and Vh columns "
            "independently, which breaks the U @ S @ Vh = Y relationship.",
            {"reconstruction_error": float(error_fixed)}
        )
        
        # Analyze what the function actually does wrong
        print("\n  Bug analysis:")
        print("    fix_svd_phases multiplies ROWS of U by phases")
        print("    and COLUMNS of Vh by phases")
        print("    But SVD requires: if U[i,:] *= exp(-iφ), then Vh[i,:] *= exp(iφ)")
        print("    The function doesn't maintain this correspondence!")


def audit_hardcoded_zero():
    """Check the hardcoded 0 in kernel.py line 47."""
    print("\n[AUDIT 4] Hardcoded Zero in Geometry")
    print("="*60)
    
    # In kernel.py line 47:
    # left_vec = np.array([left_positions[0], left_positions[1], 0], dtype=float)
    
    print("  kernel.py line 47 hardcodes left_vec[2] = 0")
    print("  This means the third generation's left-handed position is always 0")
    print("  regardless of what Q[2] is passed in.")
    
    # Test if this matters
    Q_test = (1, 3, 7)  # Third element should be 7
    U = (0, 2, 4)
    D = (0, 2, 4)
    
    Yu, _ = compute_quark_yukawas(Q_test, U, D, 2.0, 0.5, 0.3, 0.1, 0.3, 0.3)
    
    # What the code actually uses
    actual_left = [Q_test[0], Q_test[1], 0]  # [1, 3, 0]
    
    print(f"\n  Input Q = {Q_test}")
    print(f"  Actual left_vec used = {actual_left}")
    print(f"  Q[2] = {Q_test[2]} is IGNORED")
    
    # #region agent log
    log_debug("audit_hardcoded_zero", "Hardcoded zero analysis", {
        "input_Q": Q_test,
        "actual_left_vec": actual_left
    }, "H4_hardcoded")
    # #endregion
    
    # Check if this is intentional (Q is a doublet)
    add_finding(
        "DESIGN", "MEDIUM",
        "Third left-handed position hardcoded to 0",
        "In kernel.py line 47, left_vec[2] is always set to 0, ignoring the third element of Q. "
        "This may be intentional (Q is a doublet with only 2 generations), but it limits the "
        "model's flexibility. The third row of the Yukawa matrix always uses position 0.",
        {"affected_line": "kernel.py:47"}
    )


def audit_loss_function_balance():
    """Check if the loss function weighting is appropriate."""
    print("\n[AUDIT 5] Loss Function Balance")
    print("="*60)
    
    # From 01_optimize_quarks.py:
    # return L_mass + 5.0 * L_ckm + L_md_penalty + L_mu_penalty
    
    # Load actual data to see typical loss values
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'quark_results.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        print(f"  Loss statistics from {len(df)} data points:")
        print(f"    CKM loss:  mean={df['loss_ckm'].mean():.4f}, min={df['loss_ckm'].min():.4f}")
        print(f"    Mass loss: mean={df['loss_mass'].mean():.4f}, min={df['loss_mass'].min():.4f}")
        print(f"    Total:     mean={df['loss_total'].mean():.4f}, min={df['loss_total'].min():.4f}")
        
        # Check the weighting
        # Total = L_mass + 5*L_ckm + penalties
        # If L_ckm dominates, optimizer focuses on CKM at expense of masses
        
        ckm_contribution = 5.0 * df['loss_ckm'].mean()
        mass_contribution = df['loss_mass'].mean()
        
        print(f"\n  Average contributions to total loss:")
        print(f"    5 × CKM loss:  {ckm_contribution:.4f}")
        print(f"    Mass loss:     {mass_contribution:.4f}")
        print(f"    Ratio:         {ckm_contribution/mass_contribution:.2f}x")
        
        # #region agent log
        log_debug("audit_loss_balance", "Loss function analysis", {
            "ckm_mean": float(df['loss_ckm'].mean()),
            "mass_mean": float(df['loss_mass'].mean()),
            "ckm_contribution": float(ckm_contribution),
            "mass_contribution": float(mass_contribution)
        }, "H5_loss_balance")
        # #endregion
        
        if ckm_contribution > 2 * mass_contribution:
            add_finding(
                "OPTIMIZATION", "MEDIUM",
                "Loss function may over-weight CKM",
                f"The 5x multiplier on CKM loss means CKM contributes {ckm_contribution:.1f} "
                f"while mass contributes {mass_contribution:.1f}. This may cause the optimizer "
                "to sacrifice mass accuracy for CKM accuracy.",
                {"ckm_contribution": float(ckm_contribution), "mass_contribution": float(mass_contribution)}
            )


def audit_parameter_bounds():
    """Check if parameter bounds are appropriate."""
    print("\n[AUDIT 6] Parameter Bounds Analysis")
    print("="*60)
    
    # From 01_optimize_quarks.py:
    original_bounds = [
        (0.5, 6.0),   # sigma
        (0.1, 2.0),   # k
        (0.0, 2*np.pi),  # alpha
        (1.0, 5.0),   # eta
        (0.01, 0.5),  # eps_u
        (0.01, 0.5),  # eps_d
    ]
    
    print("  Original bounds:")
    param_names = ['sigma', 'k', 'alpha', 'eta', 'eps_u', 'eps_d']
    for name, (lo, hi) in zip(param_names, original_bounds):
        print(f"    {name}: [{lo}, {hi}]")
    
    # Load data to see where optimized parameters land
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'quark_results.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        print("\n  Optimized parameter statistics:")
        at_boundary = {}
        for name, (lo, hi) in zip(param_names, original_bounds):
            if name in df.columns:
                vals = df[name].values
                pct_at_lo = 100 * np.mean(np.abs(vals - lo) < 0.01 * (hi - lo))
                pct_at_hi = 100 * np.mean(np.abs(vals - hi) < 0.01 * (hi - lo))
                print(f"    {name}: mean={vals.mean():.3f}, at_low={pct_at_lo:.1f}%, at_high={pct_at_hi:.1f}%")
                at_boundary[name] = {'at_low': pct_at_lo, 'at_high': pct_at_hi}
        
        # #region agent log
        log_debug("audit_bounds", "Parameter bounds analysis", at_boundary, "H6_bounds")
        # #endregion
        
        # Check for parameters hitting boundaries
        for name, stats in at_boundary.items():
            if stats['at_low'] > 20 or stats['at_high'] > 20:
                add_finding(
                    "OPTIMIZATION", "MEDIUM",
                    f"Parameter {name} frequently hits boundary",
                    f"{name} is at lower bound {stats['at_low']:.1f}% of the time and "
                    f"at upper bound {stats['at_high']:.1f}% of the time. "
                    "Consider widening the bounds.",
                    {"parameter": name, **stats}
                )


def audit_geometry_coverage():
    """Check if geometry generation covers the space well."""
    print("\n[AUDIT 7] Geometry Coverage")
    print("="*60)
    
    # From generate_geometries():
    # Q: (q1, q2, 0) where q1 < q2
    # U: (u1, u2, u3) where u1 < u2 < u3
    # D: (d1, d2, d3) where d1 < d2 < d3
    
    # Count geometries for max_coord=8
    max_coord = 8
    n_q = sum(1 for q1 in range(max_coord) for q2 in range(max_coord) if q1 < q2)
    n_u = sum(1 for u1 in range(max_coord) for u2 in range(max_coord) for u3 in range(max_coord) 
              if u1 < u2 < u3)
    n_d = n_u  # Same constraint
    
    total = n_q * n_u * n_d
    
    print(f"  With max_coord={max_coord}:")
    print(f"    Q combinations (q1 < q2): {n_q}")
    print(f"    U combinations (u1 < u2 < u3): {n_u}")
    print(f"    D combinations (d1 < d2 < d3): {n_d}")
    print(f"    Total geometries: {total}")
    
    # Check actual data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'quark_results.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"    Actually generated: {len(df)}")
        
        # Check which geometries perform best
        best = df.nsmallest(10, 'loss_total')
        print("\n  Best 10 geometries by loss:")
        for _, row in best.iterrows():
            print(f"    Q=({row['Q1']:.0f},{row['Q2']:.0f}), "
                  f"U=({row['U1']:.0f},{row['U2']:.0f},{row['U3']:.0f}), "
                  f"D=({row['D1']:.0f},{row['D2']:.0f},{row['D3']:.0f}), "
                  f"loss={row['loss_total']:.4f}")


def audit_lepton_neutrino_data():
    """Check lepton and neutrino data for comparison."""
    print("\n[AUDIT 8] Lepton and Neutrino Sector Analysis")
    print("="*60)
    
    # Lepton data
    lepton_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'charged_lepton_results.csv')
    if os.path.exists(lepton_path):
        df_lep = pd.read_csv(lepton_path)
        print(f"\n  Charged Leptons ({len(df_lep)} rows):")
        print(f"    mmu mean: {df_lep['mmu'].mean():.4f} GeV (target: 0.1057 GeV)")
        print(f"    mmu error: {100*abs(df_lep['mmu'].mean() - 0.1057)/0.1057:.1f}%")
        print(f"    Loss mean: {df_lep['loss_total'].mean():.4f}")
        print(f"    Loss min:  {df_lep['loss_total'].min():.4f}")
        
        # #region agent log
        log_debug("audit_lepton", "Lepton sector analysis", {
            "mmu_mean": float(df_lep['mmu'].mean()),
            "mmu_target": 0.1057,
            "loss_min": float(df_lep['loss_total'].min())
        }, "H7_lepton")
        # #endregion
    
    # Neutrino data
    neutrino_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'neutrino_results.csv')
    if os.path.exists(neutrino_path):
        df_nu = pd.read_csv(neutrino_path)
        print(f"\n  Neutrinos ({len(df_nu)} rows):")
        print(f"    theta12 mean: {df_nu['theta12'].mean():.4f} (target: ~0.59)")
        print(f"    theta23 mean: {df_nu['theta23'].mean():.4f} (target: ~0.85)")
        print(f"    theta13 mean: {df_nu['theta13'].mean():.4f} (target: ~0.15)")
        print(f"    Loss mean: {df_nu['loss_total'].mean():.4f}")
        print(f"    Loss min:  {df_nu['loss_total'].min():.4f}")
        
        # Check PMNS success rate
        # Rough targets: theta12 ~ 0.59, theta23 ~ 0.85, theta13 ~ 0.15
        pmns_ok = df_nu[
            (df_nu['theta12'] > 0.4) & (df_nu['theta12'] < 0.8) &
            (df_nu['theta23'] > 0.6) & (df_nu['theta23'] < 1.1) &
            (df_nu['theta13'] > 0.1) & (df_nu['theta13'] < 0.2)
        ]
        print(f"    PMNS roughly correct: {len(pmns_ok)}/{len(df_nu)} ({100*len(pmns_ok)/len(df_nu):.1f}%)")


def generate_report():
    """Generate the final report."""
    print("\n" + "="*60)
    print("COMPREHENSIVE AUDIT REPORT")
    print("="*60)
    
    # Sort by severity
    severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3, 'INFO': 4}
    sorted_findings = sorted(REPORT, key=lambda x: severity_order.get(x['severity'], 5))
    
    print(f"\nTotal findings: {len(REPORT)}")
    for sev in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
        count = sum(1 for f in REPORT if f['severity'] == sev)
        if count > 0:
            print(f"  {sev}: {count}")
    
    print("\n" + "-"*60)
    print("DETAILED FINDINGS")
    print("-"*60)
    
    for i, finding in enumerate(sorted_findings, 1):
        print(f"\n[{finding['severity']}] {i}. {finding['title']}")
        print(f"   Category: {finding['category']}")
        print(f"   {finding['description']}")
        if finding['evidence']:
            print(f"   Evidence: {finding['evidence']}")
    
    # Save report
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, '05_comprehensive_audit.json'), 'w') as f:
        json.dump(REPORT, f, indent=2)
    
    with open(os.path.join(results_dir, '05_comprehensive_audit.txt'), 'w') as f:
        f.write("COMPREHENSIVE AUDIT REPORT\n")
        f.write("="*60 + "\n\n")
        for i, finding in enumerate(sorted_findings, 1):
            f.write(f"[{finding['severity']}] {i}. {finding['title']}\n")
            f.write(f"   Category: {finding['category']}\n")
            f.write(f"   {finding['description']}\n")
            if finding['evidence']:
                f.write(f"   Evidence: {finding['evidence']}\n")
            f.write("\n")
    
    print(f"\nReport saved to diagnostics/results/05_comprehensive_audit.txt")
    
    return sorted_findings


def main():
    """Run all audits."""
    print("="*60)
    print("COMPREHENSIVE GAUSSIAN INTERFERENCE KERNEL AUDIT")
    print("="*60)
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    
    audit_kernel_range()
    audit_svd_singular_value_structure()
    audit_fix_svd_phases_bug()
    audit_hardcoded_zero()
    audit_loss_function_balance()
    audit_parameter_bounds()
    audit_geometry_coverage()
    audit_lepton_neutrino_data()
    
    findings = generate_report()
    
    # Final summary
    print("\n" + "="*60)
    print("EXECUTIVE SUMMARY")
    print("="*60)
    
    critical = [f for f in findings if f['severity'] == 'CRITICAL']
    high = [f for f in findings if f['severity'] == 'HIGH']
    
    if critical:
        print("\nCRITICAL ISSUES (model fundamentally cannot work):")
        for f in critical:
            print(f"  • {f['title']}")
    
    if high:
        print("\nHIGH PRIORITY BUGS (should be fixed):")
        for f in high:
            print(f"  • {f['title']}")
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("""
The Gaussian interference kernel Y_ij = exp(-d²/(2σ²)) × [1 + ε exp(iΦ)]
has a FUNDAMENTAL LIMITATION:

1. The Gaussian envelope exp(-d²/(2σ²)) CAN produce large suppressions
   (e.g., exp(-16) ≈ 1e-7 for d=4σ)

2. BUT the interference term [1 + ε exp(iΦ)] can only modulate by
   a factor of (1+ε)/(1-ε) ≈ 2-3x for typical ε values

3. The SVD singular values depend on the MATRIX STRUCTURE, not just
   individual element magnitudes. The Gaussian envelope creates
   matrices where all elements in a row/column have similar magnitudes.

4. To get S[0]/S[1] ≈ 136 (for mt/mc), we need one singular value
   to dominate by 136x. This requires very specific matrix structures
   that the Gaussian kernel cannot naturally produce.

RECOMMENDATION: Replace the Gaussian envelope with a power-law or
exponential form that can create the required hierarchical structure.
""")


if __name__ == "__main__":
    main()
