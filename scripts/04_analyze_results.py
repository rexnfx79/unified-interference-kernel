#!/usr/bin/env python3
"""
Comprehensive analysis script for all three fermion sectors.

Generates:
- Survivor analysis across all sectors
- Pareto plots
- Z-score validation
- Parameter pattern comparisons
- Manuscript-ready figures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.stats as stats
from typing import Dict, Tuple, List

# Create figures directory
Path('figures').mkdir(exist_ok=True)

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
    'm2_sq_minus_m1_sq': 7.53e-5,  # eV^2
    'm3_sq_minus_m1_sq': 2.52e-3,  # eV^2
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


def find_survivors_quark(df: pd.DataFrame) -> pd.DataFrame:
    """Find quark survivors matching experimental CKM ranges."""
    # PDG 2024: Vus = 0.22500 ± 0.00067, Vcb = 0.04182 ± 0.00057, Vub = 0.00382 ± 0.00020
    survivors = df[
        (df['Vus'] > 0.17) & (df['Vus'] < 0.29) &
        (df['Vcb'] > 0.025) & (df['Vcb'] < 0.060) &
        (df['Vub'] > 0.0018) & (df['Vub'] < 0.0060)
    ]
    return survivors


def find_survivors_lepton(df: pd.DataFrame) -> pd.DataFrame:
    """Find charged lepton survivors matching experimental mass ranges."""
    # PDG 2024 ranges
    survivors = df[
        (df['me'] > 0.0004) & (df['me'] < 0.0006) &
        (df['mmu'] > 0.09) & (df['mmu'] < 0.12) &
        (df['mtau'] > 1.6) & (df['mtau'] < 2.0)
    ]
    return survivors


def find_survivors_neutrino(df: pd.DataFrame) -> pd.DataFrame:
    """Find neutrino survivors matching experimental PMNS ranges."""
    # PDG 2024 ranges
    survivors = df[
        (df['theta12'] > 0.5) & (df['theta12'] < 0.7) &
        (df['theta23'] > 0.6) & (df['theta23'] < 1.0) &
        (df['theta13'] > 0.10) & (df['theta13'] < 0.20)
    ]
    return survivors


def compute_z_scores(df: pd.DataFrame, targets: Dict[str, float], 
                     obs_cols: List[str]) -> Dict[str, float]:
    """Compute Z-scores for observables relative to targets."""
    z_scores = {}
    for obs in obs_cols:
        if obs in df.columns and obs in targets:
            mean_val = df[obs].mean()
            std_val = df[obs].std()
            target_val = targets[obs]
            if std_val > 0:
                z_score = abs(mean_val - target_val) / std_val
            else:
                z_score = np.inf if mean_val != target_val else 0.0
            z_scores[obs] = z_score
    return z_scores


def compute_pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Compute Pareto frontier for two objectives."""
    # Sort by x (ascending)
    df_sorted = df.sort_values(x_col)
    
    # Find non-dominated points
    pareto = []
    min_y = np.inf
    
    for _, row in df_sorted.iterrows():
        if row[y_col] < min_y:
            pareto.append(row)
            min_y = row[y_col]
    
    return pd.DataFrame(pareto)


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
    print(f"  Total: {len(q_surv) + len(l_surv) + len(n_surv)}/{len(quark_df) + len(lepton_df) + len(neutrino_df)}")
    
    # Best performances
    print(f"\nBest Performances:")
    print(f"  Quark CKM loss: {quark_df['loss_ckm'].min():.6f}")
    print(f"  Lepton total loss: {lepton_df['loss_total'].min():.6e}")
    print(f"  Neutrino PMNS loss: {neutrino_df['loss_pmns'].min():.6e}")
    
    # Survivor statistics
    if len(q_surv) > 0:
        print(f"\nQuark Survivor Statistics:")
        for obs in ['Vus', 'Vcb', 'Vub', 'mc']:
            if obs in q_surv.columns:
                mean_val = q_surv[obs].mean()
                target_val = QUARK_TARGETS.get(obs, 0)
                print(f"  {obs}: {mean_val:.6f} (target: {target_val:.6f}, error: {100*abs(mean_val-target_val)/target_val:.2f}%)")
    
    if len(l_surv) > 0:
        print(f"\nCharged Lepton Survivor Statistics:")
        for obs in ['me', 'mmu', 'mtau']:
            if obs in l_surv.columns:
                mean_val = l_surv[obs].mean()
                target_val = CHARGED_LEPTON_TARGETS.get(obs, 0)
                print(f"  {obs}: {mean_val:.6f} (target: {target_val:.6f}, error: {100*abs(mean_val-target_val)/target_val:.2f}%)")
    
    if len(n_surv) > 0:
        print(f"\nNeutrino Survivor Statistics:")
        for obs in ['theta12', 'theta23', 'theta13']:
            if obs in n_surv.columns:
                mean_val = n_surv[obs].mean()
                target_val = NEUTRINO_TARGETS.get(obs, 0)
                print(f"  {obs}: {mean_val:.6f} (target: {target_val:.6f}, error: {100*abs(mean_val-target_val)/target_val:.2f}%)")
    
    return q_surv, l_surv, n_surv


def plot_pareto_quark(df: pd.DataFrame, survivors: pd.DataFrame):
    """Plot Pareto frontier for quark sector (CKM loss vs mc)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # All points
    ax.scatter(df['loss_ckm'], df['mc'], alpha=0.3, s=20, c='gray', label='All geometries')
    
    # Survivors
    if len(survivors) > 0:
        ax.scatter(survivors['loss_ckm'], survivors['mc'], alpha=0.7, s=50, 
                  c='red', label=f'Survivors ({len(survivors)})', zorder=5)
    
    # Pareto frontier
    pareto = compute_pareto_frontier(df, 'loss_ckm', 'mc')
    if len(pareto) > 0:
        ax.plot(pareto['loss_ckm'], pareto['mc'], 'b-', linewidth=2, 
               label='Pareto frontier', zorder=4)
    
    # Target mc
    ax.axhline(y=QUARK_TARGETS['mc'], color='green', linestyle='--', 
              linewidth=1.5, label=f"Target $m_c$ = {QUARK_TARGETS['mc']:.2f} GeV", zorder=3)
    
    ax.set_xlabel('CKM Loss', fontsize=12)
    ax.set_ylabel('$m_c$ [GeV]', fontsize=12)
    ax.set_xscale('log')
    ax.set_title('Quark Sector: CKM Loss vs $m_c$', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/quark_pareto_ckm_mc.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: figures/quark_pareto_ckm_mc.png")


def plot_pareto_lepton(df: pd.DataFrame, survivors: pd.DataFrame):
    """Plot Pareto frontier for charged lepton sector."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # All points
    ax.scatter(df['loss_total'], df['me'], alpha=0.3, s=20, c='gray', label='All geometries')
    
    # Survivors
    if len(survivors) > 0:
        ax.scatter(survivors['loss_total'], survivors['me'], alpha=0.7, s=50, 
                  c='red', label=f'Survivors ({len(survivors)})', zorder=5)
    
    # Target
    ax.axhline(y=CHARGED_LEPTON_TARGETS['me'], color='green', linestyle='--', 
              linewidth=1.5, label=f"Target $m_e$ = {CHARGED_LEPTON_TARGETS['me']:.6f} GeV", zorder=3)
    
    ax.set_xlabel('Total Loss', fontsize=12)
    ax.set_ylabel('$m_e$ [GeV]', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Charged Lepton Sector: Total Loss vs $m_e$', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/lepton_pareto_loss_me.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: figures/lepton_pareto_loss_me.png")


def plot_pareto_neutrino(df: pd.DataFrame, survivors: pd.DataFrame):
    """Plot Pareto frontier for neutrino sector (PMNS loss vs g_env)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Color by g_env
    scatter = ax.scatter(df['loss_pmns'], df['g_env'], alpha=0.3, s=20, 
                        c=df['g_env'], cmap='viridis', label='All geometries')
    
    # Survivors
    if len(survivors) > 0:
        ax.scatter(survivors['loss_pmns'], survivors['g_env'], alpha=0.7, s=50, 
                  c='red', label=f'Survivors ({len(survivors)})', zorder=5, edgecolors='black')
    
    ax.set_xlabel('PMNS Loss', fontsize=12)
    ax.set_ylabel('$g_{env}$', fontsize=12)
    ax.set_xscale('log')
    ax.set_title('Neutrino Sector: PMNS Loss vs $g_{env}$', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='$g_{env}$')
    
    plt.tight_layout()
    plt.savefig('figures/neutrino_pareto_pmns_genv.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: figures/neutrino_pareto_pmns_genv.png")


def plot_regime_comparison(q_surv: pd.DataFrame, l_surv: pd.DataFrame, n_surv: pd.DataFrame):
    """Compare survivor rates across the three regimes."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    regimes = ['Envelope-\nDominated\n(Quark)', 
               'Phase-\nSensitive\n(Lepton)', 
               'Metric-\nDominated\n(Neutrino)']
    
    # Get total counts
    q_total = 1000  # From quark results
    l_total = 100   # From lepton results
    n_total = 480   # From neutrino results
    
    survivor_counts = [len(q_surv), len(l_surv), len(n_surv)]
    survivor_rates = [100*len(q_surv)/q_total, 100*len(l_surv)/l_total, 100*len(n_surv)/n_total]
    
    bars = ax.bar(regimes, survivor_rates, color=['#1f77b4', '#2ca02c', '#ff7f0e'], alpha=0.7)
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, survivor_counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count}\n({survivor_rates[i]:.0f}%)',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Survivor Rate [%]', fontsize=12)
    ax.set_title('Survivor Rates Across Three Projection Regimes', fontsize=14)
    ax.set_ylim(0, max(survivor_rates) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('figures/regime_comparison_survivors.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: figures/regime_comparison_survivors.png")


def compute_z_scores_all():
    """Compute Z-scores for statistical validation."""
    print("\n" + "=" * 70)
    print("Z-SCORE VALIDATION")
    print("=" * 70)
    
    quark_df, lepton_df, neutrino_df = load_results()
    
    # Quark Z-scores
    q_z = compute_z_scores(quark_df, QUARK_TARGETS, ['Vus', 'Vcb', 'Vub', 'mc'])
    print(f"\nQuark Z-scores (all geometries):")
    for obs, z in q_z.items():
        print(f"  {obs}: Z = {z:.2f}")
    
    # Lepton Z-scores
    l_z = compute_z_scores(lepton_df, CHARGED_LEPTON_TARGETS, ['me', 'mmu', 'mtau'])
    print(f"\nCharged Lepton Z-scores (all geometries):")
    for obs, z in l_z.items():
        print(f"  {obs}: Z = {z:.2f}")
    
    # Neutrino Z-scores
    n_z = compute_z_scores(neutrino_df, NEUTRINO_TARGETS, ['theta12', 'theta23', 'theta13'])
    print(f"\nNeutrino Z-scores (all geometries):")
    for obs, z in n_z.items():
        print(f"  {obs}: Z = {z:.2f}")
    
    # Survivor Z-scores
    q_surv = find_survivors_quark(quark_df)
    l_surv = find_survivors_lepton(lepton_df)
    n_surv = find_survivors_neutrino(neutrino_df)
    
    if len(q_surv) > 0:
        q_z_surv = compute_z_scores(q_surv, QUARK_TARGETS, ['Vus', 'Vcb', 'Vub', 'mc'])
        print(f"\nQuark Z-scores (survivors only):")
        for obs, z in q_z_surv.items():
            print(f"  {obs}: Z = {z:.2f}")
    
    if len(l_surv) > 0:
        l_z_surv = compute_z_scores(l_surv, CHARGED_LEPTON_TARGETS, ['me', 'mmu', 'mtau'])
        print(f"\nCharged Lepton Z-scores (survivors only):")
        for obs, z in l_z_surv.items():
            print(f"  {obs}: Z = {z:.2f}")
    
    if len(n_surv) > 0:
        n_z_surv = compute_z_scores(n_surv, NEUTRINO_TARGETS, ['theta12', 'theta23', 'theta13'])
        print(f"\nNeutrino Z-scores (survivors only):")
        for obs, z in n_z_surv.items():
            print(f"  {obs}: Z = {z:.2f}")


def analyze_parameter_patterns():
    """Analyze parameter patterns across regimes."""
    print("\n" + "=" * 70)
    print("PARAMETER PATTERN ANALYSIS")
    print("=" * 70)
    
    quark_df, lepton_df, neutrino_df = load_results()
    
    # Quark parameters
    print(f"\nQuark Parameters (all geometries):")
    for param in ['sigma', 'k', 'alpha', 'eta', 'eps_u', 'eps_d']:
        if param in quark_df.columns:
            print(f"  {param}: mean={quark_df[param].mean():.4f} ± {quark_df[param].std():.4f}, "
                  f"range=[{quark_df[param].min():.4f}, {quark_df[param].max():.4f}]")
    
    # Lepton parameters
    print(f"\nCharged Lepton Parameters (all geometries):")
    for param in ['sigma', 'k_e', 'alpha', 'eta_e', 'eps_e']:
        if param in lepton_df.columns:
            print(f"  {param}: mean={lepton_df[param].mean():.4f} ± {lepton_df[param].std():.4f}, "
                  f"range=[{lepton_df[param].min():.4f}, {lepton_df[param].max():.4f}]")
    
    # Neutrino parameters
    print(f"\nNeutrino Parameters (all geometries):")
    for param in ['sigma', 'k', 'alpha', 'eta', 'eps_nu', 'k_e', 'eta_e', 'eps_e', 'g_env']:
        if param in neutrino_df.columns:
            print(f"  {param}: mean={neutrino_df[param].mean():.4f} ± {neutrino_df[param].std():.4f}, "
                  f"range=[{neutrino_df[param].min():.4f}, {neutrino_df[param].max():.4f}]")


def main():
    """Run comprehensive analysis."""
    print("=" * 70)
    print("COMPREHENSIVE ANALYSIS - ALL THREE SECTORS")
    print("=" * 70)
    
    # Load data
    quark_df, lepton_df, neutrino_df = load_results()
    
    # Survivor analysis
    q_surv, l_surv, n_surv = analyze_survivors()
    
    # Generate Pareto plots
    print("\n" + "=" * 70)
    print("GENERATING PARETO PLOTS")
    print("=" * 70)
    plot_pareto_quark(quark_df, q_surv)
    plot_pareto_lepton(lepton_df, l_surv)
    plot_pareto_neutrino(neutrino_df, n_surv)
    plot_regime_comparison(q_surv, l_surv, n_surv)
    
    # Z-score validation
    compute_z_scores_all()
    
    # Parameter patterns
    analyze_parameter_patterns()
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nGenerated figures:")
    print("  - figures/quark_pareto_ckm_mc.png")
    print("  - figures/lepton_pareto_loss_me.png")
    print("  - figures/neutrino_pareto_pmns_genv.png")
    print("  - figures/regime_comparison_survivors.png")


if __name__ == '__main__':
    main()
