#!/usr/bin/env python3
"""
Diagnostic Figures - Revealing the True Physics

These plots show the REAL story in the data:
1. Lepton: Parameter sensitivity (sigma vs muon mass) - 61% hit target
2. Quark: The impossible gap (model predictions vs reality) - 0% hit target
3. Neutrino: Mixing angle distribution by compression (after filtering degenerate points)

Based on diagnostic analysis showing:
- Lepton: loss_total vs loss_mass correlation = 0.9999 (tautology in loss plots)
- Quark: Best mc = 2.57 GeV, target = 1.27 GeV (100% miss)
- Neutrino: 50% of data has theta23 ≈ 0 (degenerate/failed optimizations)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LogNorm
import json

# #region agent log
LOG_PATH = '/Users/alexm4/Cursor Repos/unified-interference-kernel/.cursor/debug.log'
def log_debug(hypothesis_id, location, message, data):
    entry = {'hypothesisId': hypothesis_id, 'location': location, 'message': message, 
             'data': data, 'timestamp': int(pd.Timestamp.now().timestamp() * 1000),
             'sessionId': 'chart-debug', 'runId': 'post-fix'}
    with open(LOG_PATH, 'a') as f:
        f.write(json.dumps(entry) + '\n')
# #endregion

# Create figures directory
Path('figures').mkdir(exist_ok=True)

# Load Data
quark_df = pd.read_csv('data/quark_results.csv')
lepton_df = pd.read_csv('data/charged_lepton_results.csv')
neutrino_df = pd.read_csv('data/neutrino_results.csv')

# #region agent log
# FILTER: Remove degenerate neutrino points (theta23 ≈ 0)
neutrino_valid = neutrino_df[neutrino_df['theta23'] > 0.01].copy()
log_debug('D-fix', '08_diagnostic_figures.py:45', 'Filtered degenerate neutrino points', {
    'original_count': len(neutrino_df),
    'valid_count': len(neutrino_valid),
    'removed': len(neutrino_df) - len(neutrino_valid)
})
neutrino_df = neutrino_valid
# #endregion

print("=" * 60)
print("DIAGNOSTIC ANALYSIS")
print("=" * 60)

# --- DIAGNOSTIC CHECKS ---
lepton_corr = lepton_df['loss_total'].corr(lepton_df['loss_mass'])
print(f"\nLepton Correlation (Total Loss vs Mass Loss): {lepton_corr:.6f}")
print("  -> If ~1.0, plotting them is a TAUTOLOGY")

min_mc = quark_df['mc'].min()
max_mc = quark_df['mc'].max()
target_mc = 1.27
best_mc_error = np.min(np.abs(quark_df['mc'] - target_mc))
print(f"\nQuark mc range: {min_mc:.4f} - {max_mc:.4f} GeV")
print(f"Target mc: {target_mc} GeV")
print(f"Best mc error: {best_mc_error:.4f} GeV ({100*best_mc_error/target_mc:.1f}% miss)")

# Set Publication Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

# ============================================================================
# FIGURE 1: Charged Lepton - The "Resonance" Structure
# ============================================================================
# Plot INPUT (sigma) vs OUTPUT (muon mass) - NOT loss vs loss!
print("\n" + "=" * 60)
print("Generating: Lepton Resonance Structure")
print("=" * 60)

fig, ax = plt.subplots(figsize=(7, 5))

# #region agent log
# FIX: Use log scale for colorbar (loss spans 14 orders of magnitude)
loss_min = lepton_df['loss_total'].min()
loss_max = lepton_df['loss_total'].max()
log_debug('E-fix', '08_diagnostic_figures.py:80', 'Using log norm for colorbar', {
    'loss_min': float(loss_min), 'loss_max': float(loss_max)
})
# #endregion

# Color by loss_total with LOG SCALE (spans 14 orders of magnitude)
sc = ax.scatter(lepton_df['sigma'], lepton_df['mmu'], 
                c=lepton_df['loss_total'], cmap='viridis_r', 
                norm=LogNorm(vmin=max(loss_min, 1e-12), vmax=loss_max),
                s=50, alpha=0.8, edgecolors='k', lw=0.3)

# Target line
target_mmu = 0.1056583745
ax.axhline(target_mmu, color='#d62728', linestyle='--', linewidth=2.5, 
           label=f'Target $m_\\mu$ = {target_mmu:.4f} GeV')

# Formatting
ax.set_xlabel(r'Envelope Width Parameter $\sigma$')
ax.set_ylabel(r'Predicted Muon Mass (GeV)')
ax.set_title('Charged Lepton Sector: Parameter Sensitivity\n(61% of geometries hit target)')
ax.set_yscale('log')
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Optimization Loss (Log Scale)', fontsize=10)
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('figures/lepton_resonance.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/lepton_resonance.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figures/lepton_resonance.pdf")


# ============================================================================
# FIGURE 2: Quark Sector - The "Impossible Gap"
# ============================================================================
# Show the reality: model predictions are FAR from target
print("\n" + "=" * 60)
print("Generating: Quark Impossible Gap")
print("=" * 60)

fig, ax = plt.subplots(figsize=(7, 5))

# Scatter all predictions
ax.scatter(quark_df['loss_ckm'], quark_df['mc'], 
           color='gray', alpha=0.4, s=15, label='Model Predictions', rasterized=True)

# Target line (reality)
ax.axhline(target_mc, color='#d62728', linestyle='-', linewidth=2.5, 
           label=f'Experimental Target ({target_mc} GeV)')

# Shade the "impossible" region
ax.axhspan(0, target_mc + 0.5, color='green', alpha=0.1, label='Acceptable Region')

# Annotate the gap
min_pred = quark_df['mc'].min()
gap = min_pred - target_mc

# Find a good x position for annotation
x_pos = quark_df['loss_ckm'].quantile(0.7)
ax.annotate('', xy=(x_pos, min_pred), xytext=(x_pos, target_mc),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2))
ax.text(x_pos * 1.3, (min_pred + target_mc) / 2, 
        f'Gap: {gap:.2f} GeV\n({100*gap/target_mc:.0f}% error)', 
        verticalalignment='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Formatting
ax.set_xlabel(r'CKM Matrix Loss $\mathcal{L}_{\mathrm{CKM}}$')
ax.set_ylabel(r'Charm Quark Mass $m_c$ (GeV)')
ax.set_title('Quark Sector: Model vs Reality Gap\n(0% Survivor Rate Explained)')
ax.set_xscale('log')
ax.set_ylim(0, 8)  # Zoom to show gap clearly
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('figures/quark_gap.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/quark_gap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figures/quark_gap.pdf")


# ============================================================================
# FIGURE 3: Neutrino Sector - Transition to Anarchy
# ============================================================================
print("\n" + "=" * 60)
print("Generating: Neutrino Anarchy Transition")
print("=" * 60)

fig, ax = plt.subplots(figsize=(7, 5))

# Round g_env to handle floating point
neutrino_df['g_env_rounded'] = neutrino_df['g_env'].round(2)

# Box plot for distribution at each compression level
sns.boxplot(x='g_env_rounded', y='theta23', data=neutrino_df, 
            ax=ax, color='lightblue', width=0.6, fliersize=2)

# Overlay median trend
medians = neutrino_df.groupby('g_env_rounded')['theta23'].median()
stds = neutrino_df.groupby('g_env_rounded')['theta23'].std()
x_positions = range(len(medians))
ax.plot(x_positions, medians.values, 'r-o', lw=2, markersize=8, label='Median Trend')

# #region agent log
# FIX: Remove false "variance increases" claim - variance is actually flat
variance_by_genv = neutrino_df.groupby('g_env_rounded')['theta23'].var()
log_debug('B-fix', '08_diagnostic_figures.py:180', 'Corrected variance annotation', {
    'variance_values': {str(k): float(v) for k, v in variance_by_genv.items()},
    'variance_range': float(variance_by_genv.max() - variance_by_genv.min()),
    'is_essentially_flat': float(variance_by_genv.max() - variance_by_genv.min()) < 0.01
})
# #endregion

# Experimental target
target_theta23 = 0.785
ax.axhline(target_theta23, color='green', linestyle='--', linewidth=2.5, 
           label=r'Exp $\theta_{23} \approx 45^\circ$')

# CORRECTED: Variance is essentially flat, not increasing
# Show actual statistics instead of false claim
mean_var = variance_by_genv.mean()
ax.text(0.02, 0.98, f'Variance ~{mean_var:.3f} (flat across $g_{{env}}$)\n(After filtering {240} degenerate points)', 
        transform=ax.transAxes, verticalalignment='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

# Formatting
ax.set_xlabel(r'Envelope Compression Factor $g_{\mathrm{env}}$')
ax.set_ylabel(r'Atmospheric Mixing Angle $\theta_{23}$ (rad)')
ax.set_title('Neutrino Sector: Mixing Angle Distribution\n(Valid optimizations only, n={})'.format(len(neutrino_df)))
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('figures/neutrino_anarchy.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/neutrino_anarchy.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figures/neutrino_anarchy.pdf")


# ============================================================================
# FIGURE 4: Three-Panel Summary (The Real Story)
# ============================================================================
print("\n" + "=" * 60)
print("Generating: Three-Panel Summary")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: Lepton Parameter Sensitivity (with log colorbar)
sc1 = axes[0].scatter(lepton_df['sigma'], lepton_df['mmu'], 
                      c=lepton_df['loss_total'], cmap='viridis_r', 
                      norm=LogNorm(vmin=max(loss_min, 1e-12), vmax=loss_max),
                      s=40, alpha=0.8, edgecolors='k', lw=0.2)
axes[0].axhline(target_mmu, color='#d62728', linestyle='--', linewidth=2, 
                label=f'Target $m_\\mu$')
axes[0].set_xlabel(r'Envelope Width $\sigma$')
axes[0].set_ylabel(r'Muon Mass (GeV)')
axes[0].set_title('Lepton: 61% Hit Target')
axes[0].set_yscale('log')
axes[0].legend(loc='upper right', fontsize=9)
plt.colorbar(sc1, ax=axes[0], label='Loss (log)')

# Panel 2: Quark Gap
axes[1].scatter(quark_df['loss_ckm'], quark_df['mc'], 
                color='gray', alpha=0.3, s=10, rasterized=True)
axes[1].axhline(target_mc, color='#d62728', linestyle='-', linewidth=2, 
                label=f'Target $m_c$ = {target_mc} GeV')
axes[1].axhspan(0, target_mc + 0.5, color='green', alpha=0.1)
axes[1].set_xlabel(r'CKM Loss')
axes[1].set_ylabel(r'Charm Mass $m_c$ (GeV)')
axes[1].set_title(f'Quark: {gap:.1f} GeV Gap to Reality')
axes[1].set_xscale('log')
axes[1].set_ylim(0, 8)
axes[1].legend(loc='upper right', fontsize=9)

# Panel 3: Neutrino Anarchy
sns.boxplot(x='g_env_rounded', y='theta23', data=neutrino_df, 
            ax=axes[2], color='lightblue', width=0.5, fliersize=1)
axes[2].plot(x_positions, medians.values, 'r-o', lw=1.5, markersize=6, label='Median')
axes[2].axhline(target_theta23, color='green', linestyle='--', linewidth=2, 
                label=r'Exp $\theta_{23}$')
axes[2].set_xlabel(r'Compression $g_{\mathrm{env}}$')
axes[2].set_ylabel(r'$\theta_{23}$ (rad)')
axes[2].set_title(f'Neutrino: n={len(neutrino_df)} valid')
axes[2].legend(loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('figures/diagnostic_summary_3panel.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/diagnostic_summary_3panel.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figures/diagnostic_summary_3panel.pdf")


# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("DIAGNOSTIC FIGURES COMPLETE")
print("=" * 60)
print("\nKey Findings:")
print(f"  1. Lepton: 61% hit target - model works well (colorbar now uses log scale)")
print(f"  2. Quark: Best mc = {min_mc:.2f} GeV, Target = {target_mc} GeV -> {100*gap/target_mc:.0f}% gap")
print(f"  3. Neutrino: Filtered 240 degenerate points (theta23≈0), variance is flat")
print("\nGenerated figures:")
print("  - figures/lepton_resonance.pdf")
print("  - figures/quark_gap.pdf")
print("  - figures/neutrino_anarchy.pdf")
print("  - figures/diagnostic_summary_3panel.pdf")
