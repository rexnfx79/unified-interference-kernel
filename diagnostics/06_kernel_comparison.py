#!/usr/bin/env python3
"""
Kernel Comparison Test

Compare all kernel types to find which can achieve:
1. Correct charm quark mass (mc ≈ 1.27 GeV)
2. Correct CKM mixing angles (Vus ≈ 0.225, Vcb ≈ 0.042, Vub ≈ 0.004)
3. Both simultaneously
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.optimize import differential_evolution
from alternative_kernels import KERNELS
from observables import compute_quark_observables, compute_ckm_loss, compute_mass_loss, QUARK_TARGETS
import time

np.random.seed(42)


def create_objective(kernel_name, Q, U, D):
    """Create objective function for a specific kernel."""
    kernel_info = KERNELS[kernel_name]
    compute_yukawas = kernel_info['compute_yukawas']
    
    def objective(theta):
        try:
            Yu, Yd = compute_yukawas(Q, U, D, *theta)
            obs = compute_quark_observables(Yu, Yd)
            
            L_ckm = compute_ckm_loss(obs)
            L_mass = compute_mass_loss(obs)
            
            # Penalties for extreme values
            if obs['mc'] < 0.01 or obs['mc'] > 500:
                return 1000.0
            if obs['mu'] < 1e-10:
                return 1000.0
                
            return L_mass + 5.0 * L_ckm
        except:
            return 1000.0
    
    return objective


def test_kernel_random_sampling(kernel_name, n_samples=5000):
    """Test kernel with random parameter sampling."""
    kernel_info = KERNELS[kernel_name]
    bounds = kernel_info['bounds']
    compute_yukawas = kernel_info['compute_yukawas']
    
    results = {
        'mc_values': [],
        'ckm_values': [],
        'ratios': [],
        'good_mc': 0,
        'good_ckm': 0,
        'good_both': 0,
    }
    
    for _ in range(n_samples):
        # Random geometry
        Q = tuple(sorted(np.random.choice(range(10), 3, replace=False)))
        U = tuple(sorted(np.random.choice(range(10), 3, replace=False)))
        D = tuple(sorted(np.random.choice(range(10), 3, replace=False)))
        
        # Random parameters
        theta = [np.random.uniform(lo, hi) for lo, hi in bounds]
        
        try:
            Yu, Yd = compute_yukawas(Q, U, D, *theta)
            obs = compute_quark_observables(Yu, Yd)
            
            _, Su, _ = np.linalg.svd(Yu)
            if Su[1] > 1e-15:
                ratio = Su[0] / Su[1]
                results['ratios'].append(ratio)
            
            mc = obs['mc']
            results['mc_values'].append(mc)
            results['ckm_values'].append((obs['Vus'], obs['Vcb'], obs['Vub']))
            
            # Check success criteria
            mc_ok = abs(mc - 1.27) / 1.27 < 0.5  # Within 50%
            ckm_ok = (
                abs(obs['Vus'] - 0.225) / 0.225 < 0.5 and
                abs(obs['Vcb'] - 0.042) / 0.042 < 0.5 and
                abs(obs['Vub'] - 0.00382) / 0.00382 < 0.5
            )
            
            if mc_ok:
                results['good_mc'] += 1
            if ckm_ok:
                results['good_ckm'] += 1
            if mc_ok and ckm_ok:
                results['good_both'] += 1
                
        except:
            pass
    
    return results


def optimize_kernel(kernel_name, Q, U, D, n_seeds=5, maxiter=200):
    """Optimize parameters for a specific kernel and geometry."""
    kernel_info = KERNELS[kernel_name]
    bounds = kernel_info['bounds']
    objective = create_objective(kernel_name, Q, U, D)
    
    best_loss = np.inf
    best_result = None
    
    for seed in range(n_seeds):
        try:
            result = differential_evolution(
                objective, bounds, maxiter=maxiter, seed=seed,
                polish=True, atol=1e-6, tol=1e-6
            )
            if result.fun < best_loss:
                best_loss = result.fun
                best_result = result
        except:
            pass
    
    return best_result


def test_kernel_optimization(kernel_name, n_geometries=20, n_seeds=5, maxiter=200):
    """Test kernel with optimization on multiple geometries."""
    kernel_info = KERNELS[kernel_name]
    compute_yukawas = kernel_info['compute_yukawas']
    
    results = []
    
    # Generate diverse geometries
    geometries = []
    for _ in range(n_geometries * 3):  # Generate more, pick best
        Q = tuple(sorted(np.random.choice(range(10), 3, replace=False)))
        U = tuple(sorted(np.random.choice(range(10), 3, replace=False)))
        D = tuple(sorted(np.random.choice(range(10), 3, replace=False)))
        geometries.append((Q, U, D))
    
    # Remove duplicates
    geometries = list(set(geometries))[:n_geometries]
    
    for i, (Q, U, D) in enumerate(geometries):
        result = optimize_kernel(kernel_name, Q, U, D, n_seeds=n_seeds, maxiter=maxiter)
        
        if result is not None:
            try:
                Yu, Yd = compute_yukawas(Q, U, D, *result.x)
                obs = compute_quark_observables(Yu, Yd)
                
                results.append({
                    'Q': Q, 'U': U, 'D': D,
                    'loss': result.fun,
                    'mc': obs['mc'],
                    'Vus': obs['Vus'],
                    'Vcb': obs['Vcb'],
                    'Vub': obs['Vub'],
                    'params': result.x,
                })
            except:
                pass
    
    return results


def main():
    print("="*70)
    print("KERNEL COMPARISON TEST")
    print("="*70)
    
    all_results = {}
    
    for kernel_name, kernel_info in KERNELS.items():
        print(f"\n{'='*70}")
        print(f"Testing: {kernel_info['name']}")
        print(f"Formula: {kernel_info['formula']}")
        print("="*70)
        
        # Phase 1: Random sampling
        print("\n[Phase 1] Random Sampling (5000 samples)...")
        start = time.time()
        random_results = test_kernel_random_sampling(kernel_name, n_samples=5000)
        elapsed = time.time() - start
        
        mc_arr = np.array(random_results['mc_values'])
        ratio_arr = np.array(random_results['ratios']) if random_results['ratios'] else np.array([0])
        
        print(f"  Time: {elapsed:.1f}s")
        print(f"  mc statistics:")
        print(f"    Mean: {mc_arr.mean():.4f} GeV")
        print(f"    Min:  {mc_arr.min():.4f} GeV")
        print(f"    Max:  {mc_arr.max():.4f} GeV")
        print(f"    Within 50% of target: {random_results['good_mc']}/{len(mc_arr)} ({100*random_results['good_mc']/len(mc_arr):.1f}%)")
        print(f"  S[0]/S[1] ratio:")
        print(f"    Mean: {ratio_arr.mean():.1f}")
        print(f"    Max:  {ratio_arr.max():.1f}")
        print(f"  Success rates:")
        print(f"    Good mc:   {100*random_results['good_mc']/len(mc_arr):.1f}%")
        print(f"    Good CKM:  {100*random_results['good_ckm']/len(mc_arr):.1f}%")
        print(f"    BOTH:      {100*random_results['good_both']/len(mc_arr):.1f}%")
        
        # Phase 2: Optimization
        print("\n[Phase 2] Optimization (20 geometries, 5 seeds each)...")
        start = time.time()
        opt_results = test_kernel_optimization(kernel_name, n_geometries=20, n_seeds=5, maxiter=200)
        elapsed = time.time() - start
        
        if opt_results:
            opt_results.sort(key=lambda x: x['loss'])
            
            print(f"  Time: {elapsed:.1f}s")
            print(f"  Best results:")
            for i, r in enumerate(opt_results[:5]):
                mc_err = 100 * abs(r['mc'] - 1.27) / 1.27
                vus_err = 100 * abs(r['Vus'] - 0.225) / 0.225
                print(f"    {i+1}. loss={r['loss']:.4f}, mc={r['mc']:.4f} ({mc_err:.0f}% err), "
                      f"Vus={r['Vus']:.4f} ({vus_err:.0f}% err)")
            
            # Check for success
            best = opt_results[0]
            mc_ok = abs(best['mc'] - 1.27) / 1.27 < 0.5
            ckm_ok = (
                abs(best['Vus'] - 0.225) / 0.225 < 0.5 and
                abs(best['Vcb'] - 0.042) / 0.042 < 0.5 and
                abs(best['Vub'] - 0.00382) / 0.00382 < 0.5
            )
            
            print(f"\n  Best result assessment:")
            print(f"    mc within 50%:  {'YES' if mc_ok else 'NO'}")
            print(f"    CKM within 50%: {'YES' if ckm_ok else 'NO'}")
            print(f"    OVERALL:        {'SUCCESS' if mc_ok and ckm_ok else 'FAILURE'}")
        
        all_results[kernel_name] = {
            'random': random_results,
            'optimized': opt_results,
        }
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\n{:<25} {:>10} {:>10} {:>10} {:>10}".format(
        "Kernel", "Good mc%", "Good CKM%", "Both%", "Best Loss"))
    print("-"*70)
    
    for kernel_name, results in all_results.items():
        random_r = results['random']
        opt_r = results['optimized']
        
        n = len(random_r['mc_values'])
        mc_pct = 100 * random_r['good_mc'] / n if n > 0 else 0
        ckm_pct = 100 * random_r['good_ckm'] / n if n > 0 else 0
        both_pct = 100 * random_r['good_both'] / n if n > 0 else 0
        best_loss = opt_r[0]['loss'] if opt_r else float('inf')
        
        print(f"{KERNELS[kernel_name]['name']:<25} {mc_pct:>10.1f} {ckm_pct:>10.1f} {both_pct:>10.1f} {best_loss:>10.4f}")
    
    # Recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    # Find best kernel
    best_kernel = None
    best_both_pct = 0
    
    for kernel_name, results in all_results.items():
        random_r = results['random']
        n = len(random_r['mc_values'])
        both_pct = 100 * random_r['good_both'] / n if n > 0 else 0
        if both_pct > best_both_pct:
            best_both_pct = both_pct
            best_kernel = kernel_name
    
    if best_both_pct > 0:
        print(f"\nBest kernel: {KERNELS[best_kernel]['name']}")
        print(f"Success rate (both mc AND CKM): {best_both_pct:.1f}%")
    else:
        print("\nNo kernel achieved both mc AND CKM targets simultaneously.")
        print("Consider:")
        print("  1. Wider parameter bounds")
        print("  2. Different geometry constraints")
        print("  3. Modified kernel forms")
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, '06_kernel_comparison.txt'), 'w') as f:
        f.write("KERNEL COMPARISON RESULTS\n")
        f.write("="*70 + "\n\n")
        
        for kernel_name, results in all_results.items():
            f.write(f"\n{KERNELS[kernel_name]['name']}\n")
            f.write("-"*40 + "\n")
            
            random_r = results['random']
            n = len(random_r['mc_values'])
            f.write(f"Random sampling ({n} samples):\n")
            f.write(f"  Good mc:  {random_r['good_mc']} ({100*random_r['good_mc']/n:.1f}%)\n")
            f.write(f"  Good CKM: {random_r['good_ckm']} ({100*random_r['good_ckm']/n:.1f}%)\n")
            f.write(f"  Both:     {random_r['good_both']} ({100*random_r['good_both']/n:.1f}%)\n")
            
            opt_r = results['optimized']
            if opt_r:
                f.write(f"\nOptimization (best 5):\n")
                for i, r in enumerate(opt_r[:5]):
                    f.write(f"  {i+1}. loss={r['loss']:.4f}, mc={r['mc']:.4f}, Vus={r['Vus']:.4f}\n")
    
    print(f"\nResults saved to diagnostics/results/06_kernel_comparison.txt")


if __name__ == "__main__":
    main()
