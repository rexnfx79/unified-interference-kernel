#!/usr/bin/env python3
"""
Boundary Shape and Commonalities Analysis

Characterizes Pareto boundary shapes across all three fermion sectors
with publication-quality figures using SciencePlots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from pathlib import Path
from scipy.optimize import curve_fit
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Create figures directory
Path('figures').mkdir(exist_ok=True)

# Use SciencePlots styles
plt.style.use(['science', 'ieee'])

# PDG 2024 targets
QUARK_TARGETS = {'mc': 1.27}
CHARGED_LEPTON_TARGETS = {'mmu': 0.1056583745}
NEUTRINO_TARGETS = {'theta12': 0.583}

MIN_BOUNDARY_POINTS = 5


def load_results() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load results from all three sectors."""
    quark_df = pd.read_csv('data/quark_results.csv')
    lepton_df = pd.read_csv('data/charged_lepton_results.csv')
    neutrino_df = pd.read_csv('data/neutrino_results.csv')
    return quark_df, lepton_df, neutrino_df


def compute_pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str,
                            minimize_x: bool = True, minimize_y: bool = True) -> pd.DataFrame:
    """Compute TRUE Pareto frontier (nondominated points)."""
    points = df[[x_col, y_col]].values
    n_points = len(points)
    is_dominated = np.zeros(n_points, dtype=bool)
    
    for i in range(n_points):
        if is_dominated[i]:
            continue
        for j in range(n_points):
            if i == j or is_dominated[j]:
                continue
            p, q = points[i], points[j]
            
            if minimize_x and minimize_y:
                at_least_as_good = (q[0] <= p[0]) and (q[1] <= p[1])
                strictly_better = (q[0] < p[0]) or (q[1] < p[1])
            elif minimize_x and not minimize_y:
                at_least_as_good = (q[0] <= p[0]) and (q[1] >= p[1])
                strictly_better = (q[0] < p[0]) or (q[1] > p[1])
            elif not minimize_x and minimize_y:
                at_least_as_good = (q[0] >= p[0]) and (q[1] <= p[1])
                strictly_better = (q[0] > p[0]) or (q[1] < p[1])
            else:
                at_least_as_good = (q[0] >= p[0]) and (q[1] >= p[1])
                strictly_better = (q[0] > p[0]) or (q[1] > p[1])
            
            if at_least_as_good and strictly_better:
                is_dominated[i] = True
                break
    
    pareto_df = df[~is_dominated].copy()
    pareto_df = pareto_df.sort_values(x_col)
    return pareto_df


def fit_exponential(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[Dict], float]:
    """Fit exponential y = a * exp(-b*x) to boundary data."""
    mask = (x > 0) & (y > 0)
    if mask.sum() < 3:
        return None, 0.0
    
    x_fit, y_fit = x[mask], y[mask]
    
    try:
        a0 = np.max(y_fit)
        b0 = 1.0 / np.mean(x_fit) if np.mean(x_fit) > 0 else 1.0
        
        popt, _ = curve_fit(
            lambda t, a_val, b_val: a_val * np.exp(-b_val * t),
            x_fit, y_fit, p0=[a0, b0], maxfev=10000
        )
        a, b = popt
        
        y_pred = a * np.exp(-b * x_fit)
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {'a': a, 'b': b, 'r2': r2}, r2
    except Exception:
        return None, 0.0


def compute_boundary_curvature(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute curvature along boundary using finite differences."""
    if len(x) < 5:
        return np.array([]), np.array([])
    
    idx = np.argsort(x)
    x_sorted, y_sorted = x[idx], y[idx]
    
    dx = np.gradient(x_sorted)
    dy = np.gradient(y_sorted)
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    
    numerator = np.abs(dx * d2y - dy * d2x)
    denominator = (dx**2 + dy**2)**(3/2)
    
    kappa = np.where(denominator > 1e-10, numerator / denominator, 0)
    
    return x_sorted, kappa


def get_boundary_data(df: pd.DataFrame, sector: str) -> Dict:
    """Extract boundary data for a sector with appropriate preprocessing."""
    result = {'sector': sector, 'x': None, 'y': None, 'n_points': 0, 'valid': False}
    
    if sector == 'quark':
        df_work = df.copy()
        df_work['mc_error'] = np.abs(df_work['mc'] - QUARK_TARGETS['mc'])
        pareto = compute_pareto_frontier(df_work, 'loss_ckm', 'mc_error', True, True)
        if len(pareto) >= MIN_BOUNDARY_POINTS:
            result['x'] = pareto['loss_ckm'].values
            result['y'] = pareto['mc'].values
            result['n_points'] = len(pareto)
            result['valid'] = True
            
    elif sector == 'lepton':
        df_work = df.copy()
        df_work['mmu_error'] = np.abs(df_work['mmu'] - CHARGED_LEPTON_TARGETS['mmu'])
        pareto = compute_pareto_frontier(df_work, 'loss_total', 'mmu_error', True, True)
        if len(pareto) >= MIN_BOUNDARY_POINTS:
            result['x'] = pareto['loss_total'].values
            result['y'] = pareto['mmu'].values
            result['n_points'] = len(pareto)
            result['valid'] = True
        else:
            result['x'] = df['loss_total'].values
            result['y'] = df['mmu'].values
            result['n_points'] = len(df)
            result['valid'] = True
            result['note'] = 'Using all points (Pareto too sparse)'
            
    elif sector == 'neutrino':
        best_per_genv = df.groupby('g_env').apply(
            lambda x: x.loc[x['loss_pmns'].idxmin()], include_groups=False
        ).reset_index()
        
        if len(best_per_genv) >= 3:
            result['x'] = best_per_genv['loss_pmns'].values
            result['y'] = best_per_genv['g_env'].values
            result['n_points'] = len(best_per_genv)
            result['valid'] = True
            result['discrete'] = True
    
    return result


def plot_boundary_shape_comparison(q_data: Dict, l_data: Dict, n_data: Dict):
    """
    Plot normalized boundary comparison - clean overlay.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    
    sectors = [
        ('Quark', '#0072B2', q_data),
        ('Lepton', '#009E73', l_data),
        ('Neutrino', '#D55E00', n_data)
    ]
    
    plotted_any = False
    
    for name, color, data in sectors:
        if not data.get('valid', False):
            continue
            
        x, y = data['x'], data['y']
        n_pts = data['n_points']
        
        if n_pts < MIN_BOUNDARY_POINTS:
            continue
        
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        
        if x_range < 1e-10 or y_range < 1e-10:
            continue
        
        x_norm = (x - x.min()) / x_range
        y_norm = (y - y.min()) / y_range
        
        idx = np.argsort(x_norm)
        ax.plot(x_norm[idx], y_norm[idx], color=color, linewidth=1.5, 
                label=f'{name} ($n$={n_pts})', alpha=0.9)
        ax.scatter(x_norm[idx], y_norm[idx], color=color, s=20, alpha=0.7, 
                   edgecolors='black', linewidths=0.3)
        plotted_any = True
    
    if not plotted_any:
        ax.text(0.5, 0.5, 'Insufficient data',
                ha='center', va='center', transform=ax.transAxes)
    
    ax.set_xlabel('Normalized Loss')
    ax.set_ylabel('Normalized Observable')
    ax.legend(loc='upper right', fontsize=6)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig('figures/boundary_shape_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/boundary_shape_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/boundary_shape_comparison.pdf")


def plot_boundary_fits(q_data: Dict, l_data: Dict, n_data: Dict):
    """
    Plot fitted curves overlaid on Pareto data - 3-panel figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))
    
    sectors = [
        ('Quark', '#0072B2', q_data, axes[0]),
        ('Lepton', '#009E73', l_data, axes[1]),
        ('Neutrino', '#D55E00', n_data, axes[2])
    ]
    
    for name, color, data, ax in sectors:
        if not data.get('valid', False):
            ax.text(0.5, 0.5, f'{name}\nNo Data',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(name)
            continue
        
        x, y = data['x'], data['y']
        
        # Plot data points
        ax.scatter(x, y, alpha=0.7, s=25, c=color, edgecolors='black', 
                   linewidths=0.3, label='Pareto')
        
        if data.get('discrete', False):
            idx = np.argsort(x)
            x_sorted, y_sorted = x[idx], y[idx]
            ax.step(x_sorted, y_sorted, where='post', color=color, 
                   linewidth=1.2, alpha=0.8, linestyle='--')
        else:
            fit_params, r2 = fit_exponential(x, y)
            if fit_params is not None and r2 > 0.5:
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = fit_params['a'] * np.exp(-fit_params['b'] * x_fit)
                ax.plot(x_fit, y_fit, color='#CC79A7', linestyle='--', linewidth=1.2,
                       label=f'Fit ($R^2$={r2:.2f})')
        
        ax.set_title(name)
        ax.set_xlabel('Loss')
        ax.set_ylabel('Observable')
        ax.legend(loc='best', fontsize=5)
        
        if x.max() / (x.min() + 1e-15) > 100:
            ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('figures/boundary_fits.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/boundary_fits.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/boundary_fits.pdf")


def plot_boundary_curvature(q_data: Dict, l_data: Dict, n_data: Dict):
    """Plot curvature along each boundary."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    sectors = [
        ('Quark', '#0072B2', q_data),
        ('Lepton', '#009E73', l_data),
        ('Neutrino', '#D55E00', n_data)
    ]
    
    plotted_any = False
    
    for name, color, data in sectors:
        if not data.get('valid', False):
            continue
        
        if data.get('discrete', False):
            continue
            
        x, y = data['x'], data['y']
        if len(x) < 5:
            continue
        
        x_curv, kappa = compute_boundary_curvature(x, y)
        if len(kappa) > 0:
            x_norm = (x_curv - x_curv.min()) / (x_curv.max() - x_curv.min() + 1e-10)
            ax.plot(x_norm, kappa, color=color, linewidth=1.5, label=name, alpha=0.9)
            plotted_any = True
    
    if not plotted_any:
        ax.text(0.5, 0.5, 'Insufficient continuous data',
                ha='center', va='center', transform=ax.transAxes)
    
    ax.set_xlabel('Normalized Position')
    ax.set_ylabel('Curvature $\\kappa$')
    ax.legend(loc='best', fontsize=6)
    
    plt.tight_layout()
    plt.savefig('figures/boundary_curvature.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/boundary_curvature.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/boundary_curvature.pdf")


def main():
    """Main analysis pipeline."""
    print("=" * 70)
    print("BOUNDARY SHAPE ANALYSIS")
    print("=" * 70)
    
    print("\nLoading results...")
    quark_df, lepton_df, neutrino_df = load_results()
    print(f"  Quark: {len(quark_df)} geometries")
    print(f"  Lepton: {len(lepton_df)} geometries")
    print(f"  Neutrino: {len(neutrino_df)} geometries")
    
    print("\nExtracting boundary data...")
    q_data = get_boundary_data(quark_df, 'quark')
    l_data = get_boundary_data(lepton_df, 'lepton')
    n_data = get_boundary_data(neutrino_df, 'neutrino')
    
    print(f"  Quark: {q_data['n_points']} boundary points")
    print(f"  Lepton: {l_data['n_points']} boundary points")
    print(f"  Neutrino: {n_data['n_points']} boundary points")
    
    print("\nGenerating boundary plots...")
    plot_boundary_shape_comparison(q_data, l_data, n_data)
    plot_boundary_fits(q_data, l_data, n_data)
    plot_boundary_curvature(q_data, l_data, n_data)
    
    print("\n" + "=" * 70)
    print("BOUNDARY ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nGenerated figures (PDF + PNG):")
    print("  - figures/boundary_shape_comparison.pdf")
    print("  - figures/boundary_fits.pdf")
    print("  - figures/boundary_curvature.pdf")


if __name__ == "__main__":
    main()
