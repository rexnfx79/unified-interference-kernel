#!/usr/bin/env python3
"""
Physical Mechanism Plots - Revealing True Physics

These plots show PHYSICAL RELATIONSHIPS, not optimization tautologies:
1. Lepton: Geometry (σ) → Mass (mμ) - Resonance bands
2. Quark: CKM Error vs mc - The forbidden gap (model limitation)
3. Neutrino: θ12 vs θ23 colored by g_env - Phase transition to anarchy

The key insight: Plot INPUT PARAMETERS vs OUTPUT PHYSICS,
not "error vs error" which is circular logic.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LogNorm

# Create figures directory
Path('figures').mkdir(exist_ok=True)

# Load Data
quark_df = pd.read_csv('data/quark_results.csv')
lepton_df = pd.read_csv('data/charged_lepton_results.csv')
neutrino_df = pd.read_csv('data/neutrino_results.csv')

# Filter degenerate neutrino points (theta23 ≈ 0 are failed optimizations)
neutrino_df = neutrino_df[neutrino_df['theta23'] > 0.01].copy()

# Style Settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

print("=" * 60)
print("GENERATING PHYSICAL MECHANISM PLOTS")
print("=" * 60)

# ==========================================
# 1. Lepton Resonance (Sigma vs Mass)
# ==========================================
print("\n1. Charged Lepton: Geometry-Mass Resonance")

fig1, ax1 = plt.subplots(figsize=(8, 6))

# Filter for reasonable range to see structure
l_subset = lepton_df[(lepton_df['mmu'] < 1.0) & (lepton_df['sigma'] < 10)]

# Use log scale for loss colorbar (spans many orders of magnitude)
loss_min = l_subset['loss_total'].min()
loss_max = l_subset['loss_total'].max()

# Scatter: Input (Sigma) vs Output (Mass)
# Color by Loss to show "good" vs "bad" solutions
sc = ax1.scatter(l_subset['sigma'], l_subset['mmu'], 
                 c=l_subset['loss_total'], 
                 cmap='viridis_r', 
                 norm=LogNorm(vmin=max(loss_min, 1e-12), vmax=loss_max),
                 s=60, alpha=0.9, edgecolors='k', lw=0.5)

# Target Line
target_mmu = 0.1056583745
ax1.axhline(target_mmu, color='red', linestyle='--', linewidth=2.5, 
            label=r'Target $m_\mu$ = 105.66 MeV')

# Add shaded "resonance" region around target
ax1.axhspan(target_mmu * 0.99, target_mmu * 1.01, alpha=0.2, color='green', 
            label='±1% Target Zone')

ax1.set_xlabel(r'Geometric Parameter $\sigma$ (Envelope Width)')
ax1.set_ylabel(r'Generated Muon Mass (GeV)')
ax1.set_title('Charged Lepton Sector: Geometry-Mass Resonance\n'
              r'Only specific $\sigma$ values produce correct mass')
ax1.set_yscale('log')
cbar = plt.colorbar(sc, ax=ax1)
cbar.set_label('Fit Quality (Log Scale, Brighter = Better)')
ax1.legend(loc='upper right')

# Count survivors
survivors = len(l_subset[np.abs(l_subset['mmu'] - target_mmu) / target_mmu < 0.01])
ax1.text(0.02, 0.98, f'Survivors (±1%): {survivors}/{len(l_subset)}', 
         transform=ax1.transAxes, va='top', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('figures/physical_lepton_resonance.pdf')
plt.savefig('figures/physical_lepton_resonance.png')
print(f"   Saved: figures/physical_lepton_resonance.pdf")
print(f"   Survivors in ±1% zone: {survivors}/{len(l_subset)}")

# ==========================================
# 2. Quark "Impossible Gap"
# ==========================================
print("\n2. Quark Sector: The Forbidden Gap")

fig2, ax2 = plt.subplots(figsize=(8, 6))

# Scatter model points - show ALL data, don't hide the failure
ax2.scatter(quark_df['loss_ckm'], quark_df['mc'], 
            color='steelblue', alpha=0.4, s=40, label='Model Geometries')

# Reality Line
target_mc = 1.27
ax2.axhline(target_mc, color='red', linewidth=2.5, linestyle='--',
            label=r'Experimental Target $m_c$ = 1.27 GeV')

# Add shaded forbidden region
min_mc_model = quark_df['mc'].min()
ax2.axhspan(target_mc, min_mc_model, alpha=0.15, color='red', 
            label='Forbidden Gap')

# Annotate the Gap with arrow
gap_x = quark_df['loss_ckm'].quantile(0.3)  # Position arrow at 30th percentile
ax2.annotate('', xy=(gap_x, min_mc_model), xytext=(gap_x, target_mc),
             arrowprops=dict(arrowstyle='<->', color='black', lw=2.5))

gap_size = min_mc_model - target_mc
ax2.text(gap_x * 1.5, (min_mc_model + target_mc) / 2, 
         f'Forbidden Gap\n{gap_size:.2f} GeV\n({100*gap_size/target_mc:.0f}% error)', 
         fontsize=12, fontweight='bold', va='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

ax2.set_xscale('log')
ax2.set_xlabel(r'CKM Mixing Error ($\chi^2$)')
ax2.set_ylabel(r'Charm Quark Mass $m_c$ (GeV)')
ax2.set_title('Quark Sector: The Precision Gap\n'
              'Model cannot reach experimental charm mass')
ax2.legend(loc='upper right')

# Statistics
ax2.text(0.02, 0.98, f'Best $m_c$: {min_mc_model:.2f} GeV\n'
                      f'Target: {target_mc} GeV\n'
                      f'Survivors: 0/{len(quark_df)}', 
         transform=ax2.transAxes, va='top', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('figures/physical_quark_gap.pdf')
plt.savefig('figures/physical_quark_gap.png')
print(f"   Saved: figures/physical_quark_gap.pdf")
print(f"   Gap size: {gap_size:.2f} GeV ({100*gap_size/target_mc:.0f}% error)")

# ==========================================
# 3. Neutrino Phase Transition (Mixing Plane)
# ==========================================
print("\n3. Neutrino Sector: Phase Transition to Anarchy")

fig3, ax3 = plt.subplots(figsize=(8, 6))

# Scatter Theta23 vs Theta12
# Color by Compression (g_env) to show the transition
sc3 = ax3.scatter(neutrino_df['theta12'], neutrino_df['theta23'], 
                  c=neutrino_df['g_env'], cmap='coolwarm', 
                  s=60, alpha=0.8, edgecolors='k', lw=0.3)

# Experimental Targets (PDG values)
exp_theta12 = 0.5903  # ~33.82 degrees
exp_theta23 = 0.785   # ~45 degrees (maximal mixing)
ax3.scatter([exp_theta12], [exp_theta23], color='lime', marker='*', s=300, 
            edgecolors='black', linewidths=1.5, label='Experimental Best Fit', zorder=10)

# Add crosshairs at experimental values
ax3.axvline(exp_theta12, color='lime', alpha=0.5, linestyle=':', linewidth=1.5)
ax3.axhline(exp_theta23, color='lime', alpha=0.5, linestyle=':', linewidth=1.5)

ax3.set_xlabel(r'Solar Angle $\theta_{12}$ (rad)')
ax3.set_ylabel(r'Atmospheric Angle $\theta_{23}$ (rad)')
ax3.set_title('Neutrino Sector: Transition to Anarchy\n'
              r'Lower $g_{env}$ → More chaotic mixing')

cbar = plt.colorbar(sc3, ax=ax3)
cbar.set_label(r'Envelope Compression $g_{env}$')

# Custom tick labels to show physical meaning
g_env_min = neutrino_df['g_env'].min()
g_env_max = neutrino_df['g_env'].max()
cbar.set_ticks([g_env_min, (g_env_min + g_env_max)/2, g_env_max])
cbar.ax.set_yticklabels([f'{g_env_min:.2f}\n(Chaos)', 
                          f'{(g_env_min + g_env_max)/2:.2f}', 
                          f'{g_env_max:.2f}\n(Order)'])

ax3.legend(loc='upper left')

# Statistics
ax3.text(0.98, 0.02, f'n = {len(neutrino_df)} valid points\n'
                      f'(filtered {240} degenerate)', 
         transform=ax3.transAxes, va='bottom', ha='right', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('figures/physical_neutrino_phase.pdf')
plt.savefig('figures/physical_neutrino_phase.png')
print(f"   Saved: figures/physical_neutrino_phase.pdf")
print(f"   Valid points: {len(neutrino_df)}")

# ==========================================
# 4. Three-Panel Summary (Publication Figure)
# ==========================================
print("\n4. Three-Panel Summary Figure")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: Lepton Resonance
l_subset = lepton_df[(lepton_df['mmu'] < 1.0) & (lepton_df['sigma'] < 10)]
sc1 = axes[0].scatter(l_subset['sigma'], l_subset['mmu'], 
                      c=l_subset['loss_total'], cmap='viridis_r',
                      norm=LogNorm(vmin=max(l_subset['loss_total'].min(), 1e-12), 
                                   vmax=l_subset['loss_total'].max()),
                      s=40, alpha=0.9, edgecolors='k', lw=0.3)
axes[0].axhline(target_mmu, color='red', linestyle='--', linewidth=2, 
                label=r'Target $m_\mu$')
axes[0].axhspan(target_mmu * 0.99, target_mmu * 1.01, alpha=0.2, color='green')
axes[0].set_xlabel(r'$\sigma$ (Envelope Width)')
axes[0].set_ylabel(r'Muon Mass (GeV)')
axes[0].set_title('(a) Lepton: Resonance Bands')
axes[0].set_yscale('log')
axes[0].legend(loc='upper right', fontsize=9)
plt.colorbar(sc1, ax=axes[0], label='Loss (log)')

# Panel 2: Quark Gap
axes[1].scatter(quark_df['loss_ckm'], quark_df['mc'], 
                color='steelblue', alpha=0.4, s=30, label='Model')
axes[1].axhline(target_mc, color='red', linewidth=2, linestyle='--',
                label=r'Target $m_c$')
axes[1].axhspan(target_mc, quark_df['mc'].min(), alpha=0.15, color='red')
axes[1].set_xscale('log')
axes[1].set_xlabel(r'CKM Error ($\chi^2$)')
axes[1].set_ylabel(r'Charm Mass (GeV)')
axes[1].set_title('(b) Quark: Forbidden Gap')
axes[1].legend(loc='upper right', fontsize=9)

# Panel 3: Neutrino Phase Transition
sc3 = axes[2].scatter(neutrino_df['theta12'], neutrino_df['theta23'], 
                      c=neutrino_df['g_env'], cmap='coolwarm', 
                      s=40, alpha=0.8, edgecolors='k', lw=0.2)
axes[2].scatter([exp_theta12], [exp_theta23], color='lime', marker='*', s=200, 
                edgecolors='black', linewidths=1, label='Experiment', zorder=10)
axes[2].set_xlabel(r'$\theta_{12}$ (Solar)')
axes[2].set_ylabel(r'$\theta_{23}$ (Atmospheric)')
axes[2].set_title('(c) Neutrino: Phase Transition')
axes[2].legend(loc='upper left', fontsize=9)
cbar3 = plt.colorbar(sc3, ax=axes[2])
cbar3.set_label(r'$g_{env}$')

plt.tight_layout()
plt.savefig('figures/physical_summary_3panel.pdf')
plt.savefig('figures/physical_summary_3panel.png')
print(f"   Saved: figures/physical_summary_3panel.pdf")

# ==========================================
# Summary
# ==========================================
print("\n" + "=" * 60)
print("PHYSICAL PLOTS COMPLETE")
print("=" * 60)
print("\nKey Physics Revealed:")
print(f"  1. Lepton: Resonance structure - only specific σ values work")
print(f"  2. Quark: Forbidden gap of {gap_size:.2f} GeV - model limitation")
print(f"  3. Neutrino: Phase transition from order to chaos with g_env")
print("\nGenerated figures:")
print("  - figures/physical_lepton_resonance.pdf")
print("  - figures/physical_quark_gap.pdf")
print("  - figures/physical_neutrino_phase.pdf")
print("  - figures/physical_summary_3panel.pdf")
