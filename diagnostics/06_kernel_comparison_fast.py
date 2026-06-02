#!/usr/bin/env python3
"""
Fast Kernel Comparison Test

Quick comparison of all kernel types with reduced samples.
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


def test_kernel_random(kernel_name, n_samples=1000):
    """Quick random sampling test."""
    kernel_info = KERNELS[kernel_name]
    bounds = kernel_info['bounds']
    compute_yukawas = kernel_info['compute_yukawas']
    
    good_mc = 0
    good_ckm = 0
    good_both = 0
    mc_values = []
    best_both = None
    best_both_loss = float('inf')
    
    for _ in range(n_samples):
        # Random geometry
        Q = tuple(sorted(np.random.choice(range(8), 3, replace=False)))
        U = tuple(sorted(np.random.choice(range(8), 3, replace=False)))
        D = tuple(sorted(np.random.choice(range(8), 3, replace=False)))
        
        # Random parameters
        theta = [np.random.uniform(lo, hi) for lo, hi in bounds]
        
        try:
            Yu, Yd = compute_yukawas(Q, U, D, *theta)
            obs = compute_quark_observables(Yu, Yd)
            
            mc = obs['mc']
            if mc < 0.01 or mc > 500:
                continue
                
            mc_values.append(mc)
            
            mc_ok = abs(mc - 1.27) / 1.27 < 0.5
            ckm_ok = (abs(obs['Vus'] - 0.225) / 0.225 < 0.5 and
                      abs(obs['Vcb'] - 0.042) / 0.042 < 0.5)
            
            if mc_ok:
                good_mc += 1
            if ckm_ok:
                good_ckm += 1
            if mc_ok and ckm_ok:
                good_both += 1
                loss = compute_mass_loss(obs) + 5 * compute_ckm_loss(obs)
                if loss < best_both_loss:
                    best_both_loss = loss
                    best_both = {'mc': mc, 'Vus': obs['Vus'], 'Vcb': obs['Vcb'], 
                                 'Q': Q, 'U': U, 'D': D, 'theta': theta}
        except:
            pass
    
    return {
        'n_valid': len(mc_values),
        'good_mc': good_mc,
        'good_ckm': good_ckm,
        'good_both': good_both,
        'mc_mean': np.mean(mc_values) if mc_values else 0,
        'mc_min': np.min(mc_values) if mc_values else 0,
        'best_both': best_both,
    }


def optimize_single(kernel_name, Q, U, D, maxiter=100):
    """Quick optimization for one geometry."""
    kernel_info = KERNELS[kernel_name]
    bounds = kernel_info['bounds']
    compute_yukawas = kernel_info['compute_yukawas']
    
    def objective(theta):
        try:
            Yu, Yd = compute_yukawas(Q, U, D, *theta)
            obs = compute_quark_observables(Yu, Yd)
            if obs['mc'] < 0.01 or obs['mc'] > 500:
                return 1000.0
            return compute_mass_loss(obs) + 5 * compute_ckm_loss(obs)
        except:
            return 1000.0
    
    try:
        result = differential_evolution(objective, bounds, maxiter=maxiter, seed=42, polish=False)
        Yu, Yd = compute_yukawas(Q, U, D, *result.x)
        obs = compute_quark_observables(Yu, Yd)
        return {'loss': result.fun, 'mc': obs['mc'], 'Vus': obs['Vus'], 'Vcb': obs['Vcb']}
    except:
        return None


def main():
    print("="*70)
    print("FAST KERNEL COMPARISON")
    print("="*70)
    
    results = {}
    
    for kernel_name, kernel_info in KERNELS.items():
        print(f"\n{'='*70}")
        print(f"{kernel_info['name']}: {kernel_info['formula']}")
        print("="*70)
        
        # Random sampling
        print("\n[1] Random sampling (1000 samples)...")
        start = time.time()
        r = test_kernel_random(kernel_name, n_samples=1000)
        print(f"    Time: {time.time()-start:.1f}s")
        print(f"    Valid samples: {r['n_valid']}")
        print(f"    mc mean: {r['mc_mean']:.4f} GeV (target: 1.27)")
        print(f"    mc min:  {r['mc_min']:.4f} GeV")
        print(f"    Good mc:  {r['good_mc']} ({100*r['good_mc']/max(r['n_valid'],1):.1f}%)")
        print(f"    Good CKM: {r['good_ckm']} ({100*r['good_ckm']/max(r['n_valid'],1):.1f}%)")
        print(f"    BOTH:     {r['good_both']} ({100*r['good_both']/max(r['n_valid'],1):.1f}%)")
        
        if r['best_both']:
            b = r['best_both']
            print(f"\n    Best 'both' example:")
            print(f"      mc={b['mc']:.4f}, Vus={b['Vus']:.4f}, Vcb={b['Vcb']:.4f}")
        
        # Quick optimization on a few geometries
        print("\n[2] Quick optimization (5 geometries)...")
        start = time.time()
        opt_results = []
        
        test_geometries = [
            ((0, 1, 2), (3, 4, 5), (0, 2, 4)),
            ((1, 2, 3), (0, 4, 6), (1, 3, 5)),
            ((0, 2, 4), (1, 3, 5), (2, 4, 6)),
            ((0, 1, 3), (2, 4, 5), (0, 3, 6)),
            ((1, 3, 5), (0, 2, 4), (1, 4, 7)),
        ]
        
        for Q, U, D in test_geometries:
            opt = optimize_single(kernel_name, Q, U, D, maxiter=100)
            if opt:
                opt_results.append(opt)
        
        print(f"    Time: {time.time()-start:.1f}s")
        
        if opt_results:
            opt_results.sort(key=lambda x: x['loss'])
            best = opt_results[0]
            print(f"    Best: loss={best['loss']:.4f}, mc={best['mc']:.4f}, Vus={best['Vus']:.4f}")
            
            mc_ok = abs(best['mc'] - 1.27) / 1.27 < 0.5
            ckm_ok = abs(best['Vus'] - 0.225) / 0.225 < 0.5
            print(f"    mc OK:  {'YES' if mc_ok else 'NO'}")
            print(f"    CKM OK: {'YES' if ckm_ok else 'NO'}")
        
        results[kernel_name] = {'random': r, 'opt': opt_results}
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"\n{'Kernel':<30} {'mc OK%':>8} {'CKM OK%':>8} {'Both%':>8} {'Best mc':>10}")
    print("-"*70)
    
    for kernel_name, r in results.items():
        rand = r['random']
        n = max(rand['n_valid'], 1)
        mc_pct = 100 * rand['good_mc'] / n
        ckm_pct = 100 * rand['good_ckm'] / n
        both_pct = 100 * rand['good_both'] / n
        best_mc = r['opt'][0]['mc'] if r['opt'] else 0
        
        print(f"{KERNELS[kernel_name]['name']:<30} {mc_pct:>8.1f} {ckm_pct:>8.1f} {both_pct:>8.1f} {best_mc:>10.4f}")
    
    # Find winner
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    best_kernel = max(results.keys(), 
                      key=lambda k: results[k]['random']['good_both'])
    best_both = results[best_kernel]['random']['good_both']
    best_n = results[best_kernel]['random']['n_valid']
    
    if best_both > 0:
        print(f"\nBest kernel for BOTH mc AND CKM: {KERNELS[best_kernel]['name']}")
        print(f"Success rate: {100*best_both/best_n:.1f}%")
    else:
        print("\nNo kernel achieved BOTH targets in random sampling.")
        
        # Check which is closest
        best_mc_kernel = max(results.keys(),
                            key=lambda k: results[k]['random']['good_mc'])
        best_ckm_kernel = max(results.keys(),
                             key=lambda k: results[k]['random']['good_ckm'])
        
        print(f"\nBest for mc alone: {KERNELS[best_mc_kernel]['name']} "
              f"({100*results[best_mc_kernel]['random']['good_mc']/results[best_mc_kernel]['random']['n_valid']:.1f}%)")
        print(f"Best for CKM alone: {KERNELS[best_ckm_kernel]['name']} "
              f"({100*results[best_ckm_kernel]['random']['good_ckm']/results[best_ckm_kernel]['random']['n_valid']:.1f}%)")


if __name__ == "__main__":
    main()
