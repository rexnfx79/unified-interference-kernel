#!/usr/bin/env python3
"""
Comprehensive analysis script for all three fermion sectors.

Generates publication-quality figures using:
- SciencePlots for IEEE/Nature formatting
- corner.py for parameter correlation analysis
- Proper Pareto frontier visualization

FIGURES GENERATED:
1. Pareto frontiers for each sector (scatter + frontier line)
2. Corner plots showing parameter correlations
3. Regime comparison bar chart
4. Parameter distribution comparisons
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import corner
from pathlib import Path
from scipy.optimize import curve_fit
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Create figures directory
Path('figures').mkdir(exist_ok=True)

# Use SciencePlots styles for publication-quality figures
plt.style.use(['science', 'ieee'])

# PDG 2024 targets
QUARK_TARGETS = {
    'Vus': 0.22500,
    'Vcb': 0.04182,
    'Vub': 0.00382,
    'mc': 1.27,  # GeV
    'ms': 0.093,  # GeV
}

CHARGED_LEPTON_TARGETS = {
    'me': 0.0005109989461,  # GeV
    'mmu': 0.1056583745,     # GeV
    'mtau': 1.77686,         # GeV
}

NEUTRINO_TARGETS = {
    'theta12': 0.583,  # radians
    'theta23': 0.785,  # radians
    'theta13': 0.149,  # radians
}


def load_results() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load results from all three sectors."""
    quark_df = pd.read_csv('data/quark_results.csv')
    lepton_df = pd.read_csv('data/charged_lepton_results.csv')
    neutrino_df = pd.read_csv('data/neutrino_results.csv')
    
    print(f"Loaded results:")
    print(f"  Quark: {len(quark_df)} geometries")
    print(f"  Charged Lepton: {len(lepton_df)} geometries")
    print(f"  Neutrino: {len(neutrino_df)} geometries")
    
    return quark_df, lepton_df, neutrino_df


def compute_pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str, 
                            minimize_x: bool = True, minimize_y: bool = True) -> pd.DataFrame:
    """
    Compute TRUE Pareto frontier (nondominated points).
    """
    points = df[[x_col, y_col]].values
    n_points = len(points)
    is_dominated = np.zeros(n_points, dtype=bool)
    
    for i in range(n_points):
        if is_dominated[i]:
            continue
        for j in range(n_points):
            if i == j or is_dominated[j]:
                continue
            
            p = points[i]
            q = points[j]
            
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
    
    print(f"  Pareto frontier: {len(pareto_df)} nondominated points out of {n_points}")
    return pareto_df


def find_survivors_quark(df: pd.DataFrame) -> pd.DataFrame:
    """Find quark survivors matching experimental CKM ranges."""
    survivors = df[
        (df['Vus'] > 0.17) & (df['Vus'] < 0.29) &
        (df['Vcb'] > 0.025) & (df['Vcb'] < 0.060) &
        (df['Vub'] > 0.0018) & (df['Vub'] < 0.0060)
    ]
    return survivors


def find_survivors_lepton(df: pd.DataFrame) -> pd.DataFrame:
    """Find charged lepton survivors matching experimental mass ranges."""
    survivors = df[
        (df['me'] > 0.0004) & (df['me'] < 0.0006) &
        (df['mmu'] > 0.09) & (df['mmu'] < 0.12) &
        (df['mtau'] > 1.6) & (df['mtau'] < 2.0)
    ]
    return survivors


def find_survivors_neutrino(df: pd.DataFrame) -> pd.DataFrame:
    """Find neutrino survivors matching experimental PMNS ranges."""
    survivors = df[
        (df['theta12'] > 0.5) & (df['theta12'] < 0.7) &
        (df['theta23'] > 0.6) & (df['theta23'] < 1.0) &
        (df['theta13'] > 0.10) & (df['theta13'] < 0.20)
    ]
    return survivors


def plot_pareto_quark(df: pd.DataFrame, survivors: pd.DataFrame):
    """
    Plot Pareto frontier for quark sector with publication quality.
    Clean design: background ensemble, clear frontier, target line.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    
    # Compute mc error for Pareto
    df_pareto = df.copy()
    df_pareto['mc_error'] = np.abs(df_pareto['mc'] - QUARK_TARGETS['mc'])
    
    print("\nComputing quark Pareto frontier:")
    pareto = compute_pareto_frontier(df_pareto, 'loss_ckm', 'mc_error', True, True)
    
    # Background ensemble - light gray, small markers
    ax.scatter(df['loss_ckm'], df['mc'], 
               alpha=0.15, s=8, c='gray', rasterized=True,
               label='Ensemble')
    
    # Pareto frontier points - prominent
    ax.scatter(pareto['loss_ckm'], pareto['mc'], 
               s=40, c='#0072B2', edgecolors='black', linewidths=0.5,
               zorder=10, label='Pareto front')
    
    # Connect Pareto points with step line (proper for discrete frontier)
    pareto_sorted = pareto.sort_values('loss_ckm')
    ax.step(pareto_sorted['loss_ckm'], pareto_sorted['mc'], 
            where='post', color='#0072B2', linewidth=1.2, alpha=0.8, zorder=9)
    
    # Target line
    ax.axhline(y=QUARK_TARGETS['mc'], color='#D55E00', linestyle='--', 
               linewidth=1.5, label=f'$m_c^{{\\mathrm{{exp}}}}$ = {QUARK_TARGETS["mc"]} GeV', zorder=8)
    
    # Survivors if any
    if len(survivors) > 0:
        ax.scatter(survivors['loss_ckm'], survivors['mc'], 
                   s=60, c='#009E73', marker='*', edgecolors='black', linewidths=0.3,
                   zorder=11, label=f'Survivors ({len(survivors)})')
    
    ax.set_xlabel(r'CKM Loss $\mathcal{L}_{\mathrm{CKM}}$')
    ax.set_ylabel(r'Charm Mass $m_c$ (GeV)')
    
    # Use log scale for CKM loss (better visualization of trade-off structure)
    ax.set_xscale('log')
    ax.set_ylim(0, min(df['mc'].quantile(0.95), 15))
    ax.legend(loc='upper right', framealpha=0.9, fontsize=6)
    
    plt.tight_layout()
    plt.savefig('figures/quark_pareto_ckm_mc.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/quark_pareto_ckm_mc.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/quark_pareto_ckm_mc.pdf")
    
    return pareto


def plot_pareto_lepton(df: pd.DataFrame, survivors: pd.DataFrame):
    """
    Plot for charged lepton sector - precision landscape.
    Shows muon mass error vs total loss, colored by mass loss component.
    Includes green zone for < 1% error region.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    
    # Compute muon mass error (percentage)
    df_plot = df.copy()
    df_plot['mmu_error'] = np.abs(df_plot['mmu'] - CHARGED_LEPTON_TARGETS['mmu']) / CHARGED_LEPTON_TARGETS['mmu'] * 100
    
    print("\nComputing lepton Pareto frontier:")
    pareto = compute_pareto_frontier(df_plot, 'loss_total', 'mmu_error', True, True)
    
    # Green zone (acceptable physics: < 1% error)
    ax.axhspan(0, 1.0, color='green', alpha=0.1, zorder=1, label='< 1\\% Error')
    
    # Scatter plot colored by mass loss component
    sc = ax.scatter(df_plot['loss_total'], df_plot['mmu_error'], 
                    c=df_plot['loss_mass'], cmap='viridis', 
                    s=25, alpha=0.7, edgecolors='k', linewidth=0.2,
                    rasterized=True, zorder=5)
    
    # Pareto frontier - highlight with different marker
    ax.scatter(pareto['loss_total'], pareto['mmu_error'], 
               s=60, c='#D55E00', edgecolors='black', linewidths=0.8,
               zorder=10, marker='D', label='Pareto front')
    
    # Survivors
    if len(survivors) > 0:
        surv_plot = survivors.copy()
        surv_plot['mmu_error'] = np.abs(surv_plot['mmu'] - CHARGED_LEPTON_TARGETS['mmu']) / CHARGED_LEPTON_TARGETS['mmu'] * 100
        ax.scatter(surv_plot['loss_total'], surv_plot['mmu_error'], 
                   s=70, c='#009E73', marker='*', edgecolors='black', linewidths=0.3,
                   zorder=11, label=f'Survivors ({len(survivors)})')
    
    ax.set_xlabel(r'Total Loss $\mathcal{L}_{\mathrm{total}}$')
    ax.set_ylabel(r'$m_\mu$ Relative Error (\%)')
    
    # Log scales for better visualization
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-10)  # Prevent log(0) issues
    
    ax.legend(loc='upper right', framealpha=0.9, fontsize=6)
    
    # Colorbar for mass loss component
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r'$\mathcal{L}_{\mathrm{mass}}$', fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    
    plt.tight_layout()
    plt.savefig('figures/lepton_pareto_loss_mmu.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/lepton_pareto_loss_mmu.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/lepton_pareto_loss_mmu.pdf")
    
    return pareto


def plot_pareto_neutrino(df: pd.DataFrame, survivors: pd.DataFrame):
    """
    Plot Pareto frontier for neutrino sector.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    
    # Compute theta12 error for coloring
    df_plot = df.copy()
    df_plot['theta12_error'] = np.abs(df_plot['theta12'] - NEUTRINO_TARGETS['theta12']) / NEUTRINO_TARGETS['theta12'] * 100
    
    print("\nComputing neutrino Pareto frontier:")
    
    # For neutrino with discrete g_env, find best per g_env
    best_per_genv = df.groupby('g_env').apply(
        lambda x: x.loc[x['loss_pmns'].idxmin()], include_groups=False
    ).reset_index()
    
    # Background - color by theta12 error
    scatter = ax.scatter(df_plot['loss_pmns'], df_plot['g_env'], 
                         alpha=0.4, s=12, c=df_plot['theta12_error'], 
                         cmap='viridis', vmin=0, vmax=df_plot['theta12_error'].quantile(0.9),
                         rasterized=True)
    
    # Pareto frontier (best per g_env)
    ax.scatter(best_per_genv['loss_pmns'], best_per_genv['g_env'], 
               s=80, c='#D55E00', edgecolors='black', linewidths=0.8,
               zorder=10, marker='D', label='Pareto front')
    
    # Survivors
    if len(survivors) > 0:
        ax.scatter(survivors['loss_pmns'], survivors['g_env'], 
                   s=70, c='#009E73', marker='*', edgecolors='black', linewidths=0.3,
                   zorder=11, label=f'Survivors ({len(survivors)})')
    
    ax.set_xlabel(r'PMNS Loss $\mathcal{L}_{\mathrm{PMNS}}$')
    ax.set_ylabel(r'Envelope Compression $g_{\mathrm{env}}$')
    
    # Log scale for PMNS loss
    if df['loss_pmns'].max() / (df['loss_pmns'].min() + 1e-15) > 100:
        ax.set_xscale('log')
    
    ax.legend(loc='upper right', framealpha=0.9, fontsize=6)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label(r'$\theta_{12}$ Error (\%)', fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    
    plt.tight_layout()
    plt.savefig('figures/neutrino_pareto_pmns_genv.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/neutrino_pareto_pmns_genv.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/neutrino_pareto_pmns_genv.pdf")
    
    return best_per_genv


def plot_neutrino_violin(df: pd.DataFrame):
    """
    Violin plot showing theta23 distribution vs envelope compression.
    Shows how mixing angle distributions change with g_env.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    
    # Round g_env to avoid floating-point precision issues
    df_plot = df.copy()
    df_plot['g_env_rounded'] = df_plot['g_env'].round(2)
    g_env_levels = sorted(df_plot['g_env_rounded'].unique())
    
    # Prepare data for violin plot
    data_to_plot = [df_plot[df_plot['g_env_rounded'] == g]['theta23'].values for g in g_env_levels]
    
    # Violin plot
    parts = ax.violinplot(data_to_plot, positions=g_env_levels, 
                          showmeans=True, showextrema=False, widths=0.04)
    
    # Style the violins
    for pc in parts['bodies']:
        pc.set_facecolor('#0072B2')
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.5)
    
    # Style the mean line
    if 'cmeans' in parts:
        parts['cmeans'].set_color('black')
        parts['cmeans'].set_linewidth(1.5)
    
    # Overlay scatter points with jitter for density visualization
    for g in g_env_levels:
        subset = df_plot[df_plot['g_env_rounded'] == g]
        jitter = np.random.normal(g, 0.003, size=len(subset))
        ax.scatter(jitter, subset['theta23'], s=5, color='black', alpha=0.3, marker='.', zorder=5)
    
    # Reference line (experimental value ~45 degrees)
    target_theta23 = NEUTRINO_TARGETS['theta23']
    ax.axhline(target_theta23, color='#D55E00', linestyle='--', linewidth=1.5,
               label=r'Exp $\theta_{23} \approx 45^\circ$', zorder=10)
    
    ax.set_xlabel(r'Envelope Compression $g_{\mathrm{env}}$')
    ax.set_ylabel(r'Atmospheric Mixing $\theta_{23}$ (rad)')
    ax.legend(loc='lower right', framealpha=0.9, fontsize=6)
    
    plt.tight_layout()
    plt.savefig('figures/neutrino_violin_theta23.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/neutrino_violin_theta23.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/neutrino_violin_theta23.pdf")


def plot_corner_quark(df: pd.DataFrame):
    """
    Generate corner plot for quark sector parameters.
    Shows correlations between sigma, k, alpha, eta.
    """
    # Select key parameters
    params = ['sigma', 'k', 'alpha', 'eta']
    labels = [r'$\sigma$', r'$k$', r'$\alpha$', r'$\eta$']
    
    # Filter to reasonable parameter ranges
    data = df[params].values
    
    fig = corner.corner(
        data,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 8},
        label_kwargs={"fontsize": 9},
        hist_kwargs={"density": True, "color": "#0072B2"},
        color="#0072B2",
        plot_contours=True,
        fill_contours=True,
        levels=[0.68, 0.95],
        smooth=1.0,
    )
    
    fig.suptitle('Quark Sector Parameter Correlations', fontsize=10, y=1.02)
    
    plt.savefig('figures/quark_corner.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/quark_corner.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/quark_corner.pdf")


def plot_corner_lepton(df: pd.DataFrame):
    """
    Generate corner plot for charged lepton sector parameters.
    """
    params = ['sigma', 'k_e', 'alpha']
    labels = [r'$\sigma$', r'$k_e$', r'$\alpha$']
    
    data = df[params].values
    
    fig = corner.corner(
        data,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 8},
        label_kwargs={"fontsize": 9},
        hist_kwargs={"density": True, "color": "#009E73"},
        color="#009E73",
        plot_contours=True,
        fill_contours=True,
        levels=[0.68, 0.95],
        smooth=1.0,
    )
    
    fig.suptitle('Charged Lepton Parameter Correlations', fontsize=10, y=1.02)
    
    plt.savefig('figures/lepton_corner.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/lepton_corner.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/lepton_corner.pdf")


def plot_corner_neutrino(df: pd.DataFrame):
    """
    Generate corner plot for neutrino sector parameters.
    """
    params = ['sigma', 'alpha', 'g_env']
    labels = [r'$\sigma$', r'$\alpha$', r'$g_{\mathrm{env}}$']
    
    data = df[params].values
    
    fig = corner.corner(
        data,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 8},
        label_kwargs={"fontsize": 9},
        hist_kwargs={"density": True, "color": "#D55E00"},
        color="#D55E00",
        plot_contours=True,
        fill_contours=True,
        levels=[0.68, 0.95],
        smooth=1.0,
    )
    
    fig.suptitle('Neutrino Parameter Correlations', fontsize=10, y=1.02)
    
    plt.savefig('figures/neutrino_corner.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/neutrino_corner.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/neutrino_corner.pdf")


def plot_regime_comparison(q_surv, l_surv, n_surv, q_total, l_total, n_total):
    """Plot survivor rates across regimes - clean bar chart."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    regimes = ['Quark', 'Lepton', 'Neutrino']
    survivor_rates = [
        100 * len(q_surv) / q_total,
        100 * len(l_surv) / l_total,
        100 * len(n_surv) / n_total
    ]
    
    # Colorblind-friendly palette
    colors = ['#0072B2', '#009E73', '#D55E00']
    
    bars = ax.bar(regimes, survivor_rates, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, rate, surv, total in zip(bars, survivor_rates, 
                                       [len(q_surv), len(l_surv), len(n_surv)],
                                       [q_total, l_total, n_total]):
        height = bar.get_height()
        ax.annotate(f'{rate:.1f}\\%\n({surv}/{total})',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 2), textcoords="offset points",
                   ha='center', va='bottom', fontsize=7)
    
    ax.set_ylabel('Survivor Rate (\\%)')
    ax.set_ylim(0, max(survivor_rates) * 1.3)
    
    plt.tight_layout()
    plt.savefig('figures/regime_comparison_survivors.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/regime_comparison_survivors.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/regime_comparison_survivors.pdf")


def plot_parameter_commonalities(quark_df, lepton_df, neutrino_df):
    """Plot parameter distributions across sectors - overlay histograms."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))
    
    colors = {'quark': '#0072B2', 'lepton': '#009E73', 'neutrino': '#D55E00'}
    
    # Sigma distributions
    ax1 = axes[0]
    ax1.hist(quark_df['sigma'], bins=15, alpha=0.6, label='Quark', 
             color=colors['quark'], density=True, histtype='stepfilled', edgecolor='black', linewidth=0.5)
    ax1.hist(lepton_df['sigma'], bins=15, alpha=0.6, label='Lepton', 
             color=colors['lepton'], density=True, histtype='stepfilled', edgecolor='black', linewidth=0.5)
    ax1.hist(neutrino_df['sigma'], bins=15, alpha=0.6, label='Neutrino', 
             color=colors['neutrino'], density=True, histtype='stepfilled', edgecolor='black', linewidth=0.5)
    
    # Mean lines
    ax1.axvline(x=quark_df['sigma'].mean(), color=colors['quark'], linestyle='--', linewidth=1)
    ax1.axvline(x=lepton_df['sigma'].mean(), color=colors['lepton'], linestyle='--', linewidth=1)
    ax1.axvline(x=neutrino_df['sigma'].mean(), color=colors['neutrino'], linestyle='--', linewidth=1)
    
    ax1.set_xlabel(r'Envelope Width $\sigma$')
    ax1.set_ylabel('Density')
    ax1.legend(fontsize=6, loc='upper left')
    
    # Alpha distributions
    ax2 = axes[1]
    ax2.hist(quark_df['alpha'], bins=15, alpha=0.6, label='Quark', 
             color=colors['quark'], density=True, histtype='stepfilled', edgecolor='black', linewidth=0.5)
    ax2.hist(lepton_df['alpha'], bins=15, alpha=0.6, label='Lepton', 
             color=colors['lepton'], density=True, histtype='stepfilled', edgecolor='black', linewidth=0.5)
    ax2.hist(neutrino_df['alpha'], bins=15, alpha=0.6, label='Neutrino', 
             color=colors['neutrino'], density=True, histtype='stepfilled', edgecolor='black', linewidth=0.5)
    
    # Pi reference line
    ax2.axvline(x=np.pi, color='black', linestyle=':', linewidth=1.5, label=r'$\pi$')
    
    ax2.set_xlabel(r'Phase Parameter $\alpha$')
    ax2.set_ylabel('Density')
    ax2.legend(fontsize=6, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('figures/parameter_commonalities.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/parameter_commonalities.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/parameter_commonalities.pdf")


def analyze_survivors():
    """Comprehensive survivor analysis across all sectors."""
    print("\n" + "=" * 70)
    print("SURVIVOR ANALYSIS")
    print("=" * 70)
    
    quark_df, lepton_df, neutrino_df = load_results()
    
    # Find survivors
    q_surv = find_survivors_quark(quark_df)
    l_surv = find_survivors_lepton(lepton_df)
    n_surv = find_survivors_neutrino(neutrino_df)
    
    print(f"\nSurvivors:")
    print(f"  Quark: {len(q_surv)}/{len(quark_df)} ({100*len(q_surv)/len(quark_df):.1f}%)")
    print(f"  Charged Lepton: {len(l_surv)}/{len(lepton_df)} ({100*len(l_surv)/len(lepton_df):.0f}%)")
    print(f"  Neutrino: {len(n_surv)}/{len(neutrino_df)} ({100*len(n_surv)/len(neutrino_df):.0f}%)")
    
    # Best performances
    print(f"\nBest Performances:")
    print(f"  Quark CKM loss: {quark_df['loss_ckm'].min():.6f}")
    print(f"  Lepton total loss: {lepton_df['loss_total'].min():.6e}")
    print(f"  Neutrino PMNS loss: {neutrino_df['loss_pmns'].min():.6e}")
    
    return q_surv, l_surv, n_surv, quark_df, lepton_df, neutrino_df


def main():
    """Main analysis pipeline."""
    print("=" * 70)
    print("COMPREHENSIVE ANALYSIS WITH PUBLICATION-QUALITY FIGURES")
    print("=" * 70)
    
    # Load and analyze survivors
    q_surv, l_surv, n_surv, quark_df, lepton_df, neutrino_df = analyze_survivors()
    
    # Generate Pareto plots
    print("\n" + "=" * 70)
    print("GENERATING PARETO PLOTS")
    print("=" * 70)
    
    plot_pareto_quark(quark_df, q_surv)
    plot_pareto_lepton(lepton_df, l_surv)
    plot_pareto_neutrino(neutrino_df, n_surv)
    plot_neutrino_violin(neutrino_df)
    
    # Generate corner plots for parameter correlations
    print("\n" + "=" * 70)
    print("GENERATING CORNER PLOTS (Parameter Correlations)")
    print("=" * 70)
    
    plot_corner_quark(quark_df)
    plot_corner_lepton(lepton_df)
    plot_corner_neutrino(neutrino_df)
    
    # Regime comparison
    print("\n" + "=" * 70)
    print("GENERATING REGIME COMPARISON")
    print("=" * 70)
    plot_regime_comparison(q_surv, l_surv, n_surv, 
                          len(quark_df), len(lepton_df), len(neutrino_df))
    
    # Parameter commonalities
    print("\n" + "=" * 70)
    print("GENERATING PARAMETER DISTRIBUTIONS")
    print("=" * 70)
    plot_parameter_commonalities(quark_df, lepton_df, neutrino_df)
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("PARAMETER SUMMARY")
    print("=" * 70)
    print("\nQuark sector:")
    print(f"  sigma: {quark_df['sigma'].mean():.2f} +/- {quark_df['sigma'].std():.2f}")
    print(f"  alpha: {quark_df['alpha'].mean():.2f} +/- {quark_df['alpha'].std():.2f}")
    
    print("\nCharged Lepton sector:")
    print(f"  sigma: {lepton_df['sigma'].mean():.2f} +/- {lepton_df['sigma'].std():.2f}")
    print(f"  alpha: {lepton_df['alpha'].mean():.2f} +/- {lepton_df['alpha'].std():.2f}")
    
    print("\nNeutrino sector:")
    print(f"  sigma: {neutrino_df['sigma'].mean():.2f} +/- {neutrino_df['sigma'].std():.2f}")
    print(f"  alpha: {neutrino_df['alpha'].mean():.2f} +/- {neutrino_df['alpha'].std():.2f}")
    print(f"  g_env: {neutrino_df['g_env'].mean():.2f} +/- {neutrino_df['g_env'].std():.2f}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nGenerated figures (PDF + PNG):")
    print("  Pareto plots:")
    print("    - figures/quark_pareto_ckm_mc.pdf")
    print("    - figures/lepton_pareto_loss_mmu.pdf")
    print("    - figures/neutrino_pareto_pmns_genv.pdf")
    print("    - figures/neutrino_violin_theta23.pdf")
    print("  Corner plots:")
    print("    - figures/quark_corner.pdf")
    print("    - figures/lepton_corner.pdf")
    print("    - figures/neutrino_corner.pdf")
    print("  Summary plots:")
    print("    - figures/regime_comparison_survivors.pdf")
    print("    - figures/parameter_commonalities.pdf")


if __name__ == "__main__":
    main()
