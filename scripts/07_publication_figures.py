#!/usr/bin/env python3
"""
Publication-Quality Figures for Unified Interference Kernel Paper

Generates three key visualizations:
1. Universal Parameter Clustering - Tests if parameters cluster across sectors
2. Quark Stability Boundary - Shows the CKM loss vs charm mass trade-off
3. Neutrino Anarchy - Shows compression effect on mixing angle distributions

Uses seaborn for statistical visualizations with publication styling.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Create figures directory
Path('figures').mkdir(exist_ok=True)

# Load Data
quark_df = pd.read_csv('data/quark_results.csv')
lepton_df = pd.read_csv('data/charged_lepton_results.csv')
neutrino_df = pd.read_csv('data/neutrino_results.csv')

print(f"Loaded: Quark={len(quark_df)}, Lepton={len(lepton_df)}, Neutrino={len(neutrino_df)}")

# Set Publication Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'font.size': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.frameon': False,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ============================================================================
# FIGURE 1: Three-Panel Summary Figure
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# ==========================================
# Panel 1: Universal Parameter Map (Testing Universality)
# ==========================================
# Filter top 50 best fits from each sector to see where they live
n_best = min(50, len(quark_df), len(lepton_df), len(neutrino_df))
q_best = quark_df.nsmallest(n_best, 'loss_total').assign(Sector='Quark')
l_best = lepton_df.nsmallest(n_best, 'loss_total').assign(Sector='Charged Lepton')
n_best_df = neutrino_df.nsmallest(n_best, 'loss_total').assign(Sector='Neutrino')
param_data = pd.concat([q_best, l_best, n_best_df])

sns.scatterplot(
    data=param_data, x='sigma', y='alpha', 
    hue='Sector', style='Sector', s=80, alpha=0.8, 
    palette={'Quark': '#d62728', 'Charged Lepton': '#2ca02c', 'Neutrino': '#1f77b4'},
    ax=axes[0]
)
axes[0].set_title('Universal Parameter Clustering\n(Top 50 Geometries per Sector)')
axes[0].set_xlabel(r'Envelope Width $\sigma$')
axes[0].set_ylabel(r'Phase Parameter $\alpha$')
axes[0].legend(loc='best', fontsize=8)

# ==========================================
# Panel 2: Quark Stability Boundary (The Trade-off)
# ==========================================
# Identify Pareto Frontier (Minimize CKM Loss vs Minimize Mass Deviation)
target_mc = 1.27  # GeV
quark_df['mass_dev'] = np.abs(quark_df['mc'] - target_mc)
q_sorted = quark_df.sort_values('loss_ckm')

pareto_pts = []
min_dev = float('inf')
for i, row in q_sorted.iterrows():
    if row['mass_dev'] < min_dev:
        pareto_pts.append(row)
        min_dev = row['mass_dev']
pareto_df = pd.DataFrame(pareto_pts)

# Background scatter
axes[1].scatter(quark_df['loss_ckm'], quark_df['mc'], 
                c='gray', alpha=0.15, s=10, label='Sample Geometries', rasterized=True)

# Pareto frontier
axes[1].plot(pareto_df['loss_ckm'], pareto_df['mc'], 'r-o', 
             linewidth=2, markersize=5, label='Pareto Frontier')

# Experimental constraint band
axes[1].axhspan(1.25, 1.29, color='green', alpha=0.2, 
                label=r'Exp $m_c$ ($1.27 \pm 0.02$ GeV)')

axes[1].set_xscale('log')
axes[1].set_title('Quark Sector: Stability Boundary')
axes[1].set_xlabel(r'CKM Mixing Loss ($\mathcal{L}_{\mathrm{CKM}}$)')
axes[1].set_ylabel(r'Charm Quark Mass $m_c$ (GeV)')
axes[1].legend(loc='upper right', fontsize=8)
axes[1].set_ylim(0, min(quark_df['mc'].quantile(0.95), 20))

# ==========================================
# Panel 3: Neutrino Anarchy (Compression Trend)
# ==========================================
# Round g_env to handle floating point issues
neutrino_df['g_env_rounded'] = neutrino_df['g_env'].round(2)

# Box plot shows the distribution of mixing angles at each compression level
sns.boxplot(
    data=neutrino_df, x='g_env_rounded', y='theta23', 
    color='lightblue', width=0.5, ax=axes[2], fliersize=2
)

# Overlay Median Trend
medians = neutrino_df.groupby('g_env_rounded')['theta23'].median()
x_positions = range(len(medians))
axes[2].plot(x_positions, medians.values, 'r--', marker='o', 
             lw=1.5, markersize=6, label='Median Trend')

# Experimental Target Line
target_theta23 = 0.785  # ~45 degrees
axes[2].axhline(target_theta23, color='green', linestyle='--', 
                linewidth=2, label=r'Exp $\theta_{23} \approx 45^\circ$')

axes[2].set_title('Neutrino: Envelope Compression vs. Mixing')
axes[2].set_xlabel(r'Compression Factor $g_{\mathrm{env}}$')
axes[2].set_ylabel(r'Atmospheric Mixing Angle $\theta_{23}$ (rad)')
axes[2].legend(loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig('figures/publication_summary_3panel.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/publication_summary_3panel.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/publication_summary_3panel.pdf")


# ============================================================================
# FIGURE 2: Individual High-Quality Plots
# ============================================================================

# --- 2a. Quark Trade-off (Standalone) ---
fig, ax = plt.subplots(figsize=(6, 4.5))

ax.scatter(quark_df['loss_ckm'], quark_df['mc'], 
           c='gray', alpha=0.2, s=15, label='Sample Geometries', rasterized=True)
ax.plot(pareto_df['loss_ckm'], pareto_df['mc'], 'r-o', 
        linewidth=2.5, markersize=7, label='Pareto Frontier')
ax.axhspan(1.25, 1.29, color='green', alpha=0.2, 
           label=r'Exp $m_c$ ($1.27 \pm 0.02$ GeV)')

ax.set_xscale('log')
ax.set_xlabel(r'CKM Mixing Loss ($\mathcal{L}_{\mathrm{CKM}}$)', fontsize=12)
ax.set_ylabel(r'Charm Quark Mass $m_c$ (GeV)', fontsize=12)
ax.set_title('Quark Sector: Structural Trade-off', fontsize=14)
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0, min(quark_df['mc'].quantile(0.95), 20))

plt.tight_layout()
plt.savefig('figures/quark_tradeoff_publication.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/quark_tradeoff_publication.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/quark_tradeoff_publication.pdf")


# --- 2b. Lepton Precision Landscape ---
fig, ax = plt.subplots(figsize=(6, 4.5))

# Compute muon mass error
mmu_target = 0.1056583745  # GeV (PDG)
lepton_df['mmu_error_pct'] = 100 * np.abs(lepton_df['mmu'] - mmu_target) / mmu_target

# Scatter with color by mass loss
sc = ax.scatter(lepton_df['loss_total'], lepton_df['mmu_error_pct'], 
                c=lepton_df['loss_mass'], cmap='viridis', 
                s=50, alpha=0.8, edgecolors='k', linewidth=0.3)

# Green zone (< 1% error)
ax.axhspan(0, 1.0, color='green', alpha=0.15, label='< 1% Error Region')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Total Optimization Loss', fontsize=12)
ax.set_ylabel(r'Muon Mass Relative Error (%)', fontsize=12)
ax.set_title('Charged Lepton Sector: Precision Landscape', fontsize=14)
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(bottom=1e-10)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label(r'Mass Loss Component ($\mathcal{L}_{\mathrm{mass}}$)', fontsize=10)

plt.tight_layout()
plt.savefig('figures/lepton_precision_publication.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/lepton_precision_publication.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/lepton_precision_publication.pdf")


# --- 2c. Neutrino Box Plot (Standalone) ---
fig, ax = plt.subplots(figsize=(6, 4.5))

sns.boxplot(
    data=neutrino_df, x='g_env_rounded', y='theta23', 
    color='lightblue', width=0.5, ax=ax, fliersize=3
)

# Median trend
medians = neutrino_df.groupby('g_env_rounded')['theta23'].median()
x_positions = range(len(medians))
ax.plot(x_positions, medians.values, 'r--', marker='o', 
        lw=2, markersize=8, label='Median Trend')

# Experimental target
ax.axhline(target_theta23, color='green', linestyle='--', 
           linewidth=2.5, label=r'Exp $\theta_{23} \approx 45^\circ$')

ax.set_xlabel(r'Compression Factor $g_{\mathrm{env}}$', fontsize=12)
ax.set_ylabel(r'Atmospheric Mixing Angle $\theta_{23}$ (rad)', fontsize=12)
ax.set_title('Neutrino Sector: Compression vs. Anarchy', fontsize=14)
ax.legend(loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('figures/neutrino_anarchy_publication.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/neutrino_anarchy_publication.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/neutrino_anarchy_publication.pdf")


# --- 2d. Parameter Universality (Standalone with KDE) ---
fig, ax = plt.subplots(figsize=(7, 5))

# Scatter plot with different markers
for sector, color, marker in [('Quark', '#d62728', 'o'), 
                               ('Charged Lepton', '#2ca02c', 's'), 
                               ('Neutrino', '#1f77b4', '^')]:
    if sector == 'Quark':
        data = q_best
    elif sector == 'Charged Lepton':
        data = l_best
    else:
        data = n_best_df
    
    ax.scatter(data['sigma'], data['alpha'], 
               c=color, marker=marker, s=80, alpha=0.7, 
               label=sector, edgecolors='black', linewidths=0.3)

# Add reference lines
ax.axhline(np.pi, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label=r'$\alpha = \pi$')

ax.set_xlabel(r'Envelope Width $\sigma$', fontsize=12)
ax.set_ylabel(r'Phase Parameter $\alpha$', fontsize=12)
ax.set_title('Parameter Universality Across Fermion Sectors\n(Best 50 Geometries per Sector)', fontsize=13)
ax.legend(loc='best', fontsize=10)

plt.tight_layout()
plt.savefig('figures/parameter_universality_publication.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/parameter_universality_publication.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/parameter_universality_publication.pdf")


# ============================================================================
# FIGURE 3: Survivor Analysis Summary
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# --- 3a. Survivor Rates Bar Chart ---
# Define survivor criteria
q_survivors = quark_df[
    (quark_df['Vus'] > 0.17) & (quark_df['Vus'] < 0.29) &
    (quark_df['Vcb'] > 0.025) & (quark_df['Vcb'] < 0.060) &
    (quark_df['Vub'] > 0.0018) & (quark_df['Vub'] < 0.0060)
]
l_survivors = lepton_df[
    (lepton_df['me'] > 0.0004) & (lepton_df['me'] < 0.0006) &
    (lepton_df['mmu'] > 0.09) & (lepton_df['mmu'] < 0.12) &
    (lepton_df['mtau'] > 1.6) & (lepton_df['mtau'] < 2.0)
]
n_survivors = neutrino_df[
    (neutrino_df['theta12'] > 0.5) & (neutrino_df['theta12'] < 0.7) &
    (neutrino_df['theta23'] > 0.6) & (neutrino_df['theta23'] < 1.0) &
    (neutrino_df['theta13'] > 0.10) & (neutrino_df['theta13'] < 0.20)
]

sectors = ['Quark\n(Envelope)', 'Charged Lepton\n(Phase)', 'Neutrino\n(Metric)']
survivor_rates = [
    100 * len(q_survivors) / len(quark_df),
    100 * len(l_survivors) / len(lepton_df),
    100 * len(n_survivors) / len(neutrino_df)
]
colors = ['#d62728', '#2ca02c', '#1f77b4']

bars = axes[0].bar(sectors, survivor_rates, color=colors, edgecolor='black', linewidth=1)

for bar, rate, surv, total in zip(bars, survivor_rates, 
                                   [len(q_survivors), len(l_survivors), len(n_survivors)],
                                   [len(quark_df), len(lepton_df), len(neutrino_df)]):
    height = bar.get_height()
    axes[0].annotate(f'{rate:.1f}%\n({surv}/{total})',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

axes[0].set_ylabel('Survivor Rate (%)', fontsize=12)
axes[0].set_title('Experimental Constraint Survival by Sector', fontsize=13)
axes[0].set_ylim(0, max(survivor_rates) * 1.35)

# --- 3b. Parameter Distributions (Sigma) ---
for sector, data, color in [('Quark', quark_df, '#d62728'), 
                             ('Lepton', lepton_df, '#2ca02c'), 
                             ('Neutrino', neutrino_df, '#1f77b4')]:
    axes[1].hist(data['sigma'], bins=15, alpha=0.5, label=sector, 
                 color=color, density=True, edgecolor='black', linewidth=0.5)
    axes[1].axvline(data['sigma'].mean(), color=color, linestyle='--', linewidth=2)

axes[1].set_xlabel(r'Envelope Width $\sigma$', fontsize=12)
axes[1].set_ylabel('Probability Density', fontsize=12)
axes[1].set_title('Envelope Width Distribution by Sector', fontsize=13)
axes[1].legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('figures/survivor_analysis_publication.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/survivor_analysis_publication.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/survivor_analysis_publication.pdf")


print("\n" + "=" * 60)
print("PUBLICATION FIGURES COMPLETE")
print("=" * 60)
print("\nGenerated figures:")
print("  - figures/publication_summary_3panel.pdf  (Main summary)")
print("  - figures/quark_tradeoff_publication.pdf")
print("  - figures/lepton_precision_publication.pdf")
print("  - figures/neutrino_anarchy_publication.pdf")
print("  - figures/parameter_universality_publication.pdf")
print("  - figures/survivor_analysis_publication.pdf")
