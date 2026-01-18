#!/usr/bin/env python3
"""
Boundary Shape and Commonalities Analysis

Characterizes Pareto boundary shapes across all three fermion sectors
and identifies commonalities in parameter space and boundary topology.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy import stats
from scipy.interpolate import interp1d
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Import existing analysis functions - need to import directly from module
import importlib.util
spec = importlib.util.spec_from_file_location("analyze_results", 
    os.path.join(os.path.dirname(__file__), "04_analyze_results.py"))
analyze_results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze_results)

# Extract needed functions and constants
load_results = analyze_results.load_results
find_survivors_quark = analyze_results.find_survivors_quark
find_survivors_lepton = analyze_results.find_survivors_lepton
find_survivors_neutrino = analyze_results.find_survivors_neutrino
compute_pareto_frontier = analyze_results.compute_pareto_frontier
QUARK_TARGETS = analyze_results.QUARK_TARGETS
CHARGED_LEPTON_TARGETS = analyze_results.CHARGED_LEPTON_TARGETS
NEUTRINO_TARGETS = analyze_results.NEUTRINO_TARGETS

# Create figures directory
Path('figures').mkdir(exist_ok=True)


# ============================================================================
# Boundary Shape Fitting
# ============================================================================

def fit_power_law(x: np.ndarray, y: np.ndarray) -> Tuple[Dict, float]:
    """
    Fit power-law y = a * x^b to boundary data.
    
    Returns:
        fit_params: Dict with 'a', 'b', 'r2', 'popt', 'pcov'
        r2: R-squared goodness of fit
    """
    # Filter out zeros and negatives for log
    mask = (x > 0) & (y > 0)
    if mask.sum() < 3:
        return {}, 0.0
    
    x_fit = x[mask]
    y_fit = y[mask]
    
    try:
        # Log-log linear fit: log(y) = log(a) + b*log(x)
        log_x = np.log(x_fit)
        log_y = np.log(y_fit)
        
        # Linear regression in log space
        coeffs = np.polyfit(log_x, log_y, 1)
        log_a = coeffs[1]
        b = coeffs[0]
        a = np.exp(log_a)
        
        # R-squared
        y_pred = a * (x_fit ** b)
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Parameter covariance (simplified)
        try:
            popt, pcov = curve_fit(lambda t, a_val, b_val: a_val * (t ** b_val), 
                                  x_fit, y_fit, p0=[a, b], maxfev=10000)
            a, b = popt
        except:
            popt = [a, b]
            pcov = np.eye(2)
        
        return {
            'a': a,
            'b': b,
            'r2': r2,
            'popt': popt,
            'pcov': pcov
        }, r2
    except:
        return {}, 0.0


def fit_exponential(x: np.ndarray, y: np.ndarray) -> Tuple[Dict, float]:
    """
    Fit exponential y = a * exp(-b*x) to boundary data.
    
    Returns:
        fit_params: Dict with 'a', 'b', 'r2', 'popt', 'pcov'
        r2: R-squared goodness of fit
    """
    # Filter out negatives
    mask = (x > 0) & (y > 0)
    if mask.sum() < 3:
        return {}, 0.0
    
    x_fit = x[mask]
    y_fit = y[mask]
    
    try:
        # Initial guess
        a0 = np.max(y_fit)
        b0 = 1.0 / np.mean(x_fit) if np.mean(x_fit) > 0 else 1.0
        
        popt, pcov = curve_fit(
            lambda t, a_val, b_val: a_val * np.exp(-b_val * t),
            x_fit, y_fit,
            p0=[a0, b0],
            maxfev=10000
        )
        a, b = popt
        
        # R-squared
        y_pred = a * np.exp(-b * x_fit)
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            'a': a,
            'b': b,
            'r2': r2,
            'popt': popt,
            'pcov': pcov
        }, r2
    except:
        return {}, 0.0


def fit_boundary_shape(df: pd.DataFrame, x_col: str, y_col: str) -> Dict:
    """
    Fit multiple functional forms to Pareto frontier and return best fit.
    
    Returns:
        Dict with best fit information and all fits
    """
    pareto = compute_pareto_frontier(df, x_col, y_col)
    
    if len(pareto) < 3:
        return {'best_fit': None, 'all_fits': {}}
    
    x = pareto[x_col].values
    y = pareto[y_col].values
    
    # Sort by x
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    
    # Try power-law
    power_law_fit, power_r2 = fit_power_law(x, y)
    
    # Try exponential
    exp_fit, exp_r2 = fit_exponential(x, y)
    
    # Choose best fit
    all_fits = {
        'power_law': power_law_fit,
        'exponential': exp_fit
    }
    
    if power_r2 > exp_r2:
        best_fit = {'type': 'power_law', 'params': power_law_fit, 'r2': power_r2}
    else:
        best_fit = {'type': 'exponential', 'params': exp_fit, 'r2': exp_r2}
    
    return {
        'best_fit': best_fit,
        'all_fits': all_fits,
        'pareto': pareto,
        'x': x,
        'y': y
    }


# ============================================================================
# Boundary Curvature Analysis
# ============================================================================

def compute_boundary_curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute curvature κ along boundary.
    
    κ = |y''| / (1 + y'²)^(3/2)
    
    Uses numerical differentiation with smoothing.
    """
    if len(x) < 3:
        return np.array([])
    
    # Sort by x
    idx = np.argsort(x)
    x_sorted = x[idx]
    y_sorted = y[idx]
    
    # Interpolate for smooth derivatives
    try:
        # Use cubic spline for smooth interpolation
        if len(x_sorted) > 3:
            f = interp1d(x_sorted, y_sorted, kind='cubic', bounds_error=False, fill_value='extrapolate')
            x_interp = np.linspace(x_sorted.min(), x_sorted.max(), max(50, len(x_sorted) * 2))
            y_interp = f(x_interp)
        else:
            x_interp = x_sorted
            y_interp = y_sorted
    except:
        x_interp = x_sorted
        y_interp = y_sorted
    
    # Compute first and second derivatives
    dx = np.diff(x_interp)
    dy = np.diff(y_interp)
    
    # First derivative
    dy_dx = dy / (dx + 1e-10)
    
    # Second derivative
    d2y_dx2 = np.diff(dy_dx) / (dx[1:] + 1e-10)
    
    # Curvature (at midpoints)
    x_curv = x_interp[1:-1]
    dy_dx_mid = (dy_dx[:-1] + dy_dx[1:]) / 2
    
    kappa = np.abs(d2y_dx2) / ((1 + dy_dx_mid**2)**(3/2) + 1e-10)
    
    return x_curv, kappa


def characterize_boundary_topology(x: np.ndarray, y: np.ndarray) -> Dict:
    """
    Characterize boundary topology: convexity, transition points, sharpness.
    """
    if len(x) < 3:
        return {}
    
    # Sort by x
    idx = np.argsort(x)
    x_sorted = x[idx]
    y_sorted = y[idx]
    
    # Compute curvature
    x_curv, kappa = compute_boundary_curvature(x_sorted, y_sorted)
    
    # Sharpness (mean curvature)
    sharpness = np.mean(kappa) if len(kappa) > 0 else 0.0
    
    # Transition points (where curvature changes sign or has local max)
    if len(kappa) > 2:
        # Find local maxima in curvature (knee regions)
        kappa_diff = np.diff(kappa)
        transition_idx = np.where(np.abs(kappa_diff) > np.std(kappa_diff))[0]
        transitions = x_curv[transition_idx] if len(transition_idx) > 0 else np.array([])
    else:
        transitions = np.array([])
    
    # Convexity (second derivative sign)
    # Positive second derivative = convex, negative = concave
    if len(y_sorted) > 2:
        second_deriv = np.diff(np.diff(y_sorted))
        is_convex = np.mean(second_deriv) > 0
    else:
        is_convex = None
    
    return {
        'sharpness': sharpness,
        'transitions': transitions,
        'is_convex': is_convex,
        'curvature': {'x': x_curv, 'kappa': kappa}
    }


# ============================================================================
# Parameter Commonalities Analysis
# ============================================================================

def analyze_parameter_commonalities(q_df: pd.DataFrame, l_df: pd.DataFrame, n_df: pd.DataFrame) -> Dict:
    """
    Compare parameter distributions across sectors and identify commonalities.
    """
    common_params = ['sigma', 'alpha']
    
    results = {}
    
    for param in common_params:
        if param in q_df.columns and param in l_df.columns and param in n_df.columns:
            q_vals = q_df[param].values
            l_vals = l_df[param].values
            n_vals = n_df[param].values
            
            # Basic statistics
            stats_dict = {
                'quark': {
                    'mean': np.mean(q_vals),
                    'std': np.std(q_vals),
                    'median': np.median(q_vals),
                    'range': [np.min(q_vals), np.max(q_vals)]
                },
                'lepton': {
                    'mean': np.mean(l_vals),
                    'std': np.std(l_vals),
                    'median': np.median(l_vals),
                    'range': [np.min(l_vals), np.max(l_vals)]
                },
                'neutrino': {
                    'mean': np.mean(n_vals),
                    'std': np.std(n_vals),
                    'median': np.median(n_vals),
                    'range': [np.min(n_vals), np.max(n_vals)]
                }
            }
            
            # Statistical tests for similarity
            # KS test for distribution similarity
            ks_q_l = stats.ks_2samp(q_vals, l_vals)
            ks_q_n = stats.ks_2samp(q_vals, n_vals)
            ks_l_n = stats.ks_2samp(l_vals, n_vals)
            
            # Overlap in ranges
            q_range = stats_dict['quark']['range']
            l_range = stats_dict['lepton']['range']
            n_range = stats_dict['neutrino']['range']
            
            overlap_q_l = max(0, min(q_range[1], l_range[1]) - max(q_range[0], l_range[0]))
            overlap_q_n = max(0, min(q_range[1], n_range[1]) - max(q_range[0], n_range[0]))
            overlap_l_n = max(0, min(l_range[1], n_range[1]) - max(l_range[0], n_range[0]))
            
            results[param] = {
                'statistics': stats_dict,
                'ks_tests': {
                    'quark_lepton': {'statistic': ks_q_l.statistic, 'pvalue': ks_q_l.pvalue},
                    'quark_neutrino': {'statistic': ks_q_n.statistic, 'pvalue': ks_q_n.pvalue},
                    'lepton_neutrino': {'statistic': ks_l_n.statistic, 'pvalue': ks_l_n.pvalue}
                },
                'range_overlaps': {
                    'quark_lepton': overlap_q_l,
                    'quark_neutrino': overlap_q_n,
                    'lepton_neutrino': overlap_l_n
                }
            }
    
    # Phase parameters comparison (k, eta vary by sector)
    # Compare k-like parameters
    if 'k' in q_df.columns and 'k_e' in l_df.columns and 'k' in n_df.columns:
        k_comparison = {
            'quark_k': q_df['k'].values,
            'lepton_k_e': l_df['k_e'].values,
            'neutrino_k': n_df['k'].values
        }
        results['phase_k'] = {
            'quark_mean': np.mean(k_comparison['quark_k']),
            'lepton_mean': np.mean(k_comparison['lepton_k_e']),
            'neutrino_mean': np.mean(k_comparison['neutrino_k'])
        }
    
    # Epsilon (envelope suppression) comparison
    eps_params = {
        'quark': ['eps_u', 'eps_d'],
        'lepton': ['eps_e'],
        'neutrino': ['eps_nu', 'eps_e']
    }
    
    eps_means = {}
    for sector, params in eps_params.items():
        for param in params:
            if sector == 'quark' and param in q_df.columns:
                eps_means[f'q_{param}'] = q_df[param].mean()
            elif sector == 'lepton' and param in l_df.columns:
                eps_means[f'l_{param}'] = l_df[param].mean()
            elif sector == 'neutrino' and param in n_df.columns:
                eps_means[f'n_{param}'] = n_df[param].mean()
    
    results['envelope_suppression'] = eps_means
    
    return results


# ============================================================================
# Boundary Commonalities Analysis
# ============================================================================

def analyze_boundary_commonalities(q_pareto: Dict, l_pareto: Dict, n_pareto: Dict) -> Dict:
    """
    Compare boundary shapes, exponents, and features across sectors.
    """
    results = {}
    
    # Extract fits (with None checks)
    q_fit = q_pareto.get('best_fit') or {}
    l_fit = l_pareto.get('best_fit') or {}
    n_fit = n_pareto.get('best_fit') or {}
    
    # Compare power-law exponents
    exponents = {}
    if q_fit and q_fit.get('type') == 'power_law' and 'params' in q_fit:
        exponents['quark'] = q_fit['params'].get('b', None)
    if l_fit and l_fit.get('type') == 'power_law' and 'params' in l_fit:
        exponents['lepton'] = l_fit['params'].get('b', None)
    if n_fit and n_fit.get('type') == 'power_law' and 'params' in n_fit:
        exponents['neutrino'] = n_fit['params'].get('b', None)
    
    results['power_law_exponents'] = exponents
    
    # Compare R-squared values (goodness of fit)
    results['fit_quality'] = {
        'quark': q_fit.get('r2', 0) if q_fit else 0,
        'lepton': l_fit.get('r2', 0) if l_fit else 0,
        'neutrino': n_fit.get('r2', 0) if n_fit else 0
    }
    
    # Compare boundary topologies
    topologies = {}
    if 'x' in q_pareto and 'y' in q_pareto:
        topologies['quark'] = characterize_boundary_topology(q_pareto['x'], q_pareto['y'])
    if 'x' in l_pareto and 'y' in l_pareto:
        topologies['lepton'] = characterize_boundary_topology(l_pareto['x'], l_pareto['y'])
    if 'x' in n_pareto and 'y' in n_pareto:
        topologies['neutrino'] = characterize_boundary_topology(n_pareto['x'], n_pareto['y'])
    
    results['topology'] = topologies
    
    return results


# ============================================================================
# Survivor Boundary Analysis
# ============================================================================

def analyze_survivor_boundary_location(survivors: pd.DataFrame, pareto: pd.DataFrame,
                                      x_col: str, y_col: str) -> Dict:
    """
    Analyze where survivors sit relative to the Pareto boundary.
    """
    if len(survivors) == 0 or len(pareto) == 0:
        return {'distances': [], 'on_boundary': 0, 'near_boundary': 0}
    
    # Sort Pareto by x
    pareto_sorted = pareto.sort_values(x_col)
    
    # Interpolate boundary
    try:
        f_boundary = interp1d(
            pareto_sorted[x_col].values,
            pareto_sorted[y_col].values,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
    except:
        return {'distances': [], 'on_boundary': 0, 'near_boundary': 0}
    
    # For each survivor, find distance to boundary
    distances = []
    on_boundary = 0
    near_boundary = 0
    
    for _, row in survivors.iterrows():
        x_surv = row[x_col]
        y_surv = row[y_col]
        
        # Find closest boundary point
        y_boundary = f_boundary(x_surv)
        
        # Distance (normalized)
        if y_boundary > 0:
            distance = abs(y_surv - y_boundary) / y_boundary
            distances.append(distance)
            
            # On boundary (within 1%)
            if distance < 0.01:
                on_boundary += 1
            # Near boundary (within 10%)
            elif distance < 0.10:
                near_boundary += 1
    
    return {
        'distances': distances,
        'mean_distance': np.mean(distances) if distances else 0.0,
        'on_boundary': on_boundary,
        'near_boundary': near_boundary,
        'total_survivors': len(survivors)
    }


# ============================================================================
# Figure Generation
# ============================================================================

def plot_boundary_shape_comparison(q_pareto: Dict, l_pareto: Dict, n_pareto: Dict):
    """Plot all three boundaries in normalized space for comparison."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Normalize each boundary to [0,1] in both dimensions
    def normalize_boundary(x, y):
        if len(x) == 0:
            return np.array([]), np.array([])
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)
        return x_norm, y_norm
    
    if 'x' in q_pareto and 'y' in q_pareto:
        x_q, y_q = normalize_boundary(q_pareto['x'], q_pareto['y'])
        ax.plot(x_q, y_q, 'b-', linewidth=2, label='Quark (Envelope-Dominated)', alpha=0.8)
    
    if 'x' in l_pareto and 'y' in l_pareto:
        x_l, y_l = normalize_boundary(l_pareto['x'], l_pareto['y'])
        ax.plot(x_l, y_l, 'g-', linewidth=2, label='Charged Lepton (Phase-Sensitive)', alpha=0.8)
    
    if 'x' in n_pareto and 'y' in n_pareto:
        x_n, y_n = normalize_boundary(n_pareto['x'], n_pareto['y'])
        ax.plot(x_n, y_n, 'orange', linewidth=2, label='Neutrino (Metric-Dominated)', alpha=0.8)
    
    ax.set_xlabel('Normalized Loss (x-axis)', fontsize=12)
    ax.set_ylabel('Normalized Observable (y-axis)', fontsize=12)
    ax.set_title('Pareto Boundary Shape Comparison (Normalized)', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/boundary_shape_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: figures/boundary_shape_comparison.png")


def plot_boundary_fits(q_pareto: Dict, l_pareto: Dict, n_pareto: Dict):
    """Plot fitted curves overlaid on Pareto data."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sectors = [
        ('Quark', 'b', q_pareto, axes[0]),
        ('Charged Lepton', 'g', l_pareto, axes[1]),
        ('Neutrino', 'orange', n_pareto, axes[2])
    ]
    
    for name, color, pareto_dict, ax in sectors:
        if 'x' in pareto_dict and 'y' in pareto_dict:
            x = pareto_dict['x']
            y = pareto_dict['y']
            
            # Plot data
            ax.scatter(x, y, alpha=0.5, s=20, c=color, label=f'{name} data')
            
            # Plot fits
            best_fit = pareto_dict.get('best_fit', {})
            if best_fit and 'params' in best_fit:
                fit_type = best_fit.get('type', '')
                params = best_fit['params']
                
                x_fit = np.linspace(x.min(), x.max(), 100)
                if fit_type == 'power_law' and 'a' in params and 'b' in params:
                    y_fit = params['a'] * (x_fit ** params['b'])
                    ax.plot(x_fit, y_fit, 'r--', linewidth=2, 
                           label=f'Power-law (R²={best_fit.get("r2", 0):.3f})', alpha=0.8)
                elif fit_type == 'exponential' and 'a' in params and 'b' in params:
                    y_fit = params['a'] * np.exp(-params['b'] * x_fit)
                    ax.plot(x_fit, y_fit, 'r--', linewidth=2,
                           label=f'Exponential (R²={best_fit.get("r2", 0):.3f})', alpha=0.8)
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(f'{name} Boundary Fit', fontsize=12)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/boundary_fits.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: figures/boundary_fits.png")


def plot_boundary_curvature(q_pareto: Dict, l_pareto: Dict, n_pareto: Dict):
    """Plot curvature along each boundary."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sectors = [
        ('Quark', 'b', q_pareto),
        ('Charged Lepton', 'g', l_pareto),
        ('Neutrino', 'orange', n_pareto)
    ]
    
    for name, color, pareto_dict in sectors:
        if 'x' in pareto_dict and 'y' in pareto_dict:
            x_curv, kappa = compute_boundary_curvature(pareto_dict['x'], pareto_dict['y'])
            if len(kappa) > 0:
                # Normalize x for comparison
                x_norm = (x_curv - x_curv.min()) / (x_curv.max() - x_curv.min() + 1e-10)
                ax.plot(x_norm, kappa, color=color, linewidth=2, label=name, alpha=0.8)
    
    ax.set_xlabel('Normalized Position Along Boundary', fontsize=12)
    ax.set_ylabel('Curvature κ', fontsize=12)
    ax.set_title('Boundary Curvature Profiles', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/boundary_curvature.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: figures/boundary_curvature.png")


def plot_parameter_commonalities(q_df: pd.DataFrame, l_df: pd.DataFrame, n_df: pd.DataFrame):
    """Plot parameter distributions to show commonalities."""
    common_params = ['sigma', 'alpha']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, param in enumerate(common_params):
        if idx >= 2:
            break
        
        ax = axes[idx]
        
        if param in q_df.columns:
            ax.hist(q_df[param], bins=30, alpha=0.5, label='Quark', color='b', density=True)
        if param in l_df.columns:
            ax.hist(l_df[param], bins=30, alpha=0.5, label='Lepton', color='g', density=True)
        if param in n_df.columns:
            ax.hist(n_df[param], bins=30, alpha=0.5, label='Neutrino', color='orange', density=True)
        
        ax.set_xlabel(param.capitalize(), fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{param.capitalize()} Distribution Comparison', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/parameter_commonalities.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: figures/parameter_commonalities.png")


def plot_survivor_boundary_positions(q_df: pd.DataFrame, q_surv: pd.DataFrame,
                                    l_df: pd.DataFrame, l_surv: pd.DataFrame,
                                    n_df: pd.DataFrame, n_surv: pd.DataFrame):
    """Plot survivors overlaid on Pareto boundaries."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Quark
    q_pareto = compute_pareto_frontier(q_df, 'loss_ckm', 'mc')
    if len(q_pareto) > 0:
        axes[0].scatter(q_df['loss_ckm'], q_df['mc'], alpha=0.2, s=10, c='gray')
        axes[0].plot(q_pareto['loss_ckm'], q_pareto['mc'], 'b-', linewidth=2, label='Pareto boundary')
        if len(q_surv) > 0:
            axes[0].scatter(q_surv['loss_ckm'], q_surv['mc'], alpha=0.7, s=50, c='red', 
                          label=f'Survivors ({len(q_surv)})', zorder=5)
        axes[0].set_xlabel('CKM Loss', fontsize=10)
        axes[0].set_ylabel('$m_c$ [GeV]', fontsize=10)
        axes[0].set_xscale('log')
        axes[0].set_title('Quark', fontsize=11)
        axes[0].legend(loc='best', fontsize=8)
        axes[0].grid(True, alpha=0.3)
    
    # Lepton
    l_pareto = compute_pareto_frontier(l_df, 'loss_total', 'me')
    if len(l_pareto) > 0:
        axes[1].scatter(l_df['loss_total'], l_df['me'], alpha=0.2, s=10, c='gray')
        axes[1].plot(l_pareto['loss_total'], l_pareto['me'], 'g-', linewidth=2, label='Pareto boundary')
        if len(l_surv) > 0:
            axes[1].scatter(l_surv['loss_total'], l_surv['me'], alpha=0.7, s=50, c='red',
                          label=f'Survivors ({len(l_surv)})', zorder=5)
        axes[1].set_xlabel('Total Loss', fontsize=10)
        axes[1].set_ylabel('$m_e$ [GeV]', fontsize=10)
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].set_title('Charged Lepton', fontsize=11)
        axes[1].legend(loc='best', fontsize=8)
        axes[1].grid(True, alpha=0.3)
    
    # Neutrino
    n_pareto = compute_pareto_frontier(n_df, 'loss_pmns', 'g_env')
    if len(n_pareto) > 0:
        axes[2].scatter(n_df['loss_pmns'], n_df['g_env'], alpha=0.2, s=10, c='gray')
        axes[2].plot(n_pareto['loss_pmns'], n_pareto['g_env'], color='orange', linewidth=2, 
                    label='Pareto boundary')
        if len(n_surv) > 0:
            axes[2].scatter(n_surv['loss_pmns'], n_surv['g_env'], alpha=0.7, s=50, c='red',
                          label=f'Survivors ({len(n_surv)})', zorder=5)
        axes[2].set_xlabel('PMNS Loss', fontsize=10)
        axes[2].set_ylabel('$g_{env}$', fontsize=10)
        axes[2].set_xscale('log')
        axes[2].set_title('Neutrino', fontsize=11)
        axes[2].legend(loc='best', fontsize=8)
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/survivor_boundary_positions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: figures/survivor_boundary_positions.png")


# ============================================================================
# Main Analysis Function
# ============================================================================

def main():
    """Run comprehensive boundary shape and commonalities analysis."""
    print("=" * 70)
    print("BOUNDARY SHAPE AND COMMONALITIES ANALYSIS")
    print("=" * 70)
    
    # Load data
    q_df, l_df, n_df = load_results()
    q_surv = find_survivors_quark(q_df)
    l_surv = find_survivors_lepton(l_df)
    n_surv = find_survivors_neutrino(n_df)
    
    print(f"\nLoaded data:")
    print(f"  Quark: {len(q_df)} geometries, {len(q_surv)} survivors")
    print(f"  Lepton: {len(l_df)} geometries, {len(l_surv)} survivors")
    print(f"  Neutrino: {len(n_df)} geometries, {len(n_surv)} survivors")
    
    # Fit boundary shapes
    print("\n" + "=" * 70)
    print("FITTING BOUNDARY SHAPES")
    print("=" * 70)
    
    q_pareto = fit_boundary_shape(q_df, 'loss_ckm', 'mc')
    l_pareto = fit_boundary_shape(l_df, 'loss_total', 'me')
    n_pareto = fit_boundary_shape(n_df, 'loss_pmns', 'g_env')
    
    # Print fit results
    for name, pareto_dict in [('Quark', q_pareto), ('Lepton', l_pareto), ('Neutrino', n_pareto)]:
        best_fit = pareto_dict.get('best_fit', {})
        if best_fit:
            fit_type = best_fit.get('type', 'unknown')
            r2 = best_fit.get('r2', 0)
            params = best_fit.get('params', {})
            print(f"\n{name}:")
            print(f"  Best fit: {fit_type} (R² = {r2:.4f})")
            if fit_type == 'power_law':
                print(f"    y = {params.get('a', 0):.4e} * x^{params.get('b', 0):.4f}")
            elif fit_type == 'exponential':
                print(f"    y = {params.get('a', 0):.4e} * exp(-{params.get('b', 0):.4f} * x)")
    
    # Parameter commonalities
    print("\n" + "=" * 70)
    print("PARAMETER COMMONALITIES")
    print("=" * 70)
    
    param_comm = analyze_parameter_commonalities(q_df, l_df, n_df)
    
    if 'sigma' in param_comm:
        stats_dict = param_comm['sigma']['statistics']
        print(f"\nsigma (envelope width):")
        for sector, stats in stats_dict.items():
            print(f"  {sector.capitalize()}: mean={stats['mean']:.3f} ± {stats['std']:.3f}, "
                  f"range=[{stats['range'][0]:.3f}, {stats['range'][1]:.3f}]")
    
    if 'alpha' in param_comm:
        stats_dict = param_comm['alpha']['statistics']
        print(f"\nalpha (phase parameter):")
        for sector, stats in stats_dict.items():
            print(f"  {sector.capitalize()}: mean={stats['mean']:.3f} ± {stats['std']:.3f}, "
                  f"range=[{stats['range'][0]:.3f}, {stats['range'][1]:.3f}]")
    
    # Boundary commonalities
    print("\n" + "=" * 70)
    print("BOUNDARY COMMONALITIES")
    print("=" * 70)
    
    boundary_comm = analyze_boundary_commonalities(q_pareto, l_pareto, n_pareto)
    
    if 'power_law_exponents' in boundary_comm:
        exponents = boundary_comm['power_law_exponents']
        print("\nPower-law exponents:")
        for sector, exp in exponents.items():
            if exp is not None:
                print(f"  {sector.capitalize()}: {exp:.4f}")
    
    # Generate figures
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    
    plot_boundary_shape_comparison(q_pareto, l_pareto, n_pareto)
    plot_boundary_fits(q_pareto, l_pareto, n_pareto)
    plot_boundary_curvature(q_pareto, l_pareto, n_pareto)
    plot_parameter_commonalities(q_df, l_df, n_df)
    plot_survivor_boundary_positions(q_df, q_surv, l_df, l_surv, n_df, n_surv)
    
    # Save results for report
    results = {
        'boundary_fits': {
            'quark': q_pareto,
            'lepton': l_pareto,
            'neutrino': n_pareto
        },
        'parameter_commonalities': param_comm,
        'boundary_commonalities': boundary_comm
    }
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    results = main()
