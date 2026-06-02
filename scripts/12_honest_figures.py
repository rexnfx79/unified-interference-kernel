#!/usr/bin/env python3
"""
Honest Publication Figures - What Reviewers Actually Want to See

Based on rigorous data visualization critique, these figures show:
1. Prediction vs Experiment with error bars (the basic expectation)
2. Parameter sensitivity analysis (actual mechanism)
3. Success rate vs tolerance threshold (model robustness)
4. Failure mode analysis (why quark sector fails)

No more dressing up failures as insights. Honest science.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Create figures directory
Path('figures').mkdir(exist_ok=True)

# Load Data
quark_df = pd.read_csv('data/quark_results.csv')
lepton_df = pd.read_csv('data/charged_lepton_results.csv')
neutrino_df = pd.read_csv('data/neutrino_results.csv')

# Style Settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

print("=" * 70)
print("GENERATING HONEST PUBLICATION FIGURES")
print("=" * 70)

# =============================================================================
# FIGURE 1: PREDICTION VS EXPERIMENT (All Three Sectors)
# =============================================================================
print("\n1. Prediction vs Experiment Plots")

fig1, axes = plt.subplots(1, 3, figsize=(15, 5))

# --- Panel A: Quark Sector ---
ax = axes[0]

# Experimental values (PDG 2024)
quark_exp = {
    'Vus': (0.2243, 0.0005),
    'Vcb': (0.0422, 0.0008),
    'Vub': (0.00382, 0.00020),
    'mc': (1.27, 0.02),  # GeV
}

# Model predictions (best geometry)
best_quark_idx = quark_df['loss_total'].idxmin()
best_quark = quark_df.loc[best_quark_idx]

# Get distribution of predictions
quark_pred = {
    'Vus': (quark_df['Vus'].mean(), quark_df['Vus'].std()),
    'Vcb': (quark_df['Vcb'].mean(), quark_df['Vcb'].std()),
    'Vub': (quark_df['Vub'].mean(), quark_df['Vub'].std()),
    'mc': (quark_df['mc'].mean(), quark_df['mc'].std()),
}

observables = ['Vus', 'Vcb', 'Vub', 'mc']
x_pos = np.arange(len(observables))

# Normalize to experimental values for comparison
exp_vals = [quark_exp[o][0] for o in observables]
exp_errs = [quark_exp[o][1] for o in observables]
pred_vals = [quark_pred[o][0] for o in observables]
pred_errs = [quark_pred[o][1] for o in observables]

# Plot as ratio to experiment
ratios = [p/e for p, e in zip(pred_vals, exp_vals)]
ratio_errs = [pe/e for pe, e in zip(pred_errs, exp_vals)]
exp_rel_errs = [ee/e for ee, e in zip(exp_errs, exp_vals)]

ax.bar(x_pos - 0.2, [1]*len(observables), 0.35, label='Experiment', color='green', alpha=0.7)
ax.errorbar(x_pos - 0.2, [1]*len(observables), yerr=exp_rel_errs, fmt='none', color='darkgreen', capsize=5)

ax.bar(x_pos + 0.2, ratios, 0.35, label='Model', color='steelblue', alpha=0.7)
ax.errorbar(x_pos + 0.2, ratios, yerr=ratio_errs, fmt='none', color='navy', capsize=5)

ax.axhline(1, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(observables)
ax.set_ylabel('Ratio to Experiment')
ax.set_title('(a) Quark Sector: Model Fails\n(0% within 10% of experiment)')
ax.legend(loc='upper right')
ax.set_ylim(0, max(ratios) * 1.2)

# Add failure annotation
ax.text(0.5, 0.95, f'$m_c$ prediction: {quark_pred["mc"][0]:.1f} ± {quark_pred["mc"][1]:.1f} GeV\n'
                   f'$m_c$ experiment: {quark_exp["mc"][0]:.2f} ± {quark_exp["mc"][1]:.2f} GeV\n'
                   f'Error: {100*(quark_pred["mc"][0]/quark_exp["mc"][0] - 1):.0f}%',
        transform=ax.transAxes, va='top', ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# --- Panel B: Charged Lepton Sector ---
ax = axes[1]

# Experimental values
lepton_exp = {
    'me': (0.000511, 0.000000001),
    'mmu': (0.105658, 0.000001),
    'mtau': (1.77686, 0.00012),
}

# Model predictions
lepton_pred = {
    'me': (lepton_df['me'].mean(), lepton_df['me'].std()),
    'mmu': (lepton_df['mmu'].mean(), lepton_df['mmu'].std()),
    'mtau': (lepton_df['mtau'].mean(), lepton_df['mtau'].std()),
}

observables = ['me', 'mmu', 'mtau']
labels = [r'$m_e$', r'$m_\mu$', r'$m_\tau$']
x_pos = np.arange(len(observables))

exp_vals = [lepton_exp[o][0] for o in observables]
exp_errs = [lepton_exp[o][1] for o in observables]
pred_vals = [lepton_pred[o][0] for o in observables]
pred_errs = [lepton_pred[o][1] for o in observables]

ratios = [p/e for p, e in zip(pred_vals, exp_vals)]
ratio_errs = [pe/e for pe, e in zip(pred_errs, exp_vals)]

ax.bar(x_pos - 0.2, [1]*len(observables), 0.35, label='Experiment', color='green', alpha=0.7)
ax.bar(x_pos + 0.2, ratios, 0.35, label='Model', color='forestgreen', alpha=0.7)
ax.errorbar(x_pos + 0.2, ratios, yerr=ratio_errs, fmt='none', color='darkgreen', capsize=5)

ax.axhline(1, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.axhspan(0.9, 1.1, alpha=0.1, color='green', label='±10% band')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_ylabel('Ratio to Experiment')
ax.set_title('(b) Charged Lepton Sector\n(Partial success for $m_\\tau$)')
ax.legend(loc='upper right')
ax.set_ylim(0, max(ratios) * 1.2)

# Count successes
n_within_10pct = sum(1 for r in ratios if 0.9 <= r <= 1.1)
ax.text(0.5, 0.95, f'{n_within_10pct}/{len(ratios)} observables within 10%',
        transform=ax.transAxes, va='top', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

# --- Panel C: Neutrino Sector ---
ax = axes[2]

# Experimental values
neutrino_exp = {
    'theta12': (0.5903, 0.013),
    'theta23': (0.785, 0.025),
    'theta13': (0.149, 0.003),
}

# Model predictions (filter valid points)
neutrino_valid = neutrino_df[neutrino_df['theta23'] > 0.01]
neutrino_pred = {
    'theta12': (neutrino_valid['theta12'].mean(), neutrino_valid['theta12'].std()),
    'theta23': (neutrino_valid['theta23'].mean(), neutrino_valid['theta23'].std()),
    'theta13': (neutrino_valid['theta13'].mean(), neutrino_valid['theta13'].std()),
}

observables = ['theta12', 'theta23', 'theta13']
labels = [r'$\theta_{12}$', r'$\theta_{23}$', r'$\theta_{13}$']
x_pos = np.arange(len(observables))

exp_vals = [neutrino_exp[o][0] for o in observables]
exp_errs = [neutrino_exp[o][1] for o in observables]
pred_vals = [neutrino_pred[o][0] for o in observables]
pred_errs = [neutrino_pred[o][1] for o in observables]

ratios = [p/e for p, e in zip(pred_vals, exp_vals)]
ratio_errs = [pe/e for pe, e in zip(pred_errs, exp_vals)]
exp_rel_errs = [ee/e for ee, e in zip(exp_errs, exp_vals)]

ax.bar(x_pos - 0.2, [1]*len(observables), 0.35, label='Experiment', color='green', alpha=0.7)
ax.errorbar(x_pos - 0.2, [1]*len(observables), yerr=exp_rel_errs, fmt='none', color='darkgreen', capsize=5)

ax.bar(x_pos + 0.2, ratios, 0.35, label='Model', color='darkorange', alpha=0.7)
ax.errorbar(x_pos + 0.2, ratios, yerr=ratio_errs, fmt='none', color='darkorange', capsize=5)

ax.axhline(1, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.axhspan(0.9, 1.1, alpha=0.1, color='green', label='±10% band')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_ylabel('Ratio to Experiment')
ax.set_title('(c) Neutrino Sector\n(Good agreement for mixing angles)')
ax.legend(loc='upper right')
ax.set_ylim(0.5, 1.5)

# Count successes
n_within_10pct = sum(1 for r in ratios if 0.9 <= r <= 1.1)
ax.text(0.5, 0.95, f'{n_within_10pct}/{len(ratios)} observables within 10%\n'
                   f'(50% of optimizations failed)',
        transform=ax.transAxes, va='top', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='moccasin', alpha=0.9))

plt.suptitle('Model Predictions vs Experiment (with uncertainties)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('figures/fig_honest_pred_vs_exp.pdf')
plt.savefig('figures/fig_honest_pred_vs_exp.png')
print("   Saved: figures/fig_honest_pred_vs_exp.pdf")

# =============================================================================
# FIGURE 2: PARAMETER SENSITIVITY ANALYSIS
# =============================================================================
print("\n2. Parameter Sensitivity Analysis")

fig2, axes = plt.subplots(2, 3, figsize=(14, 9))

# Lepton sector: How does mmu depend on each parameter?
params = ['sigma', 'k_e', 'alpha', 'eta_e', 'eps_e']
param_labels = [r'$\sigma$', r'$k_e$', r'$\alpha$', r'$\eta_e$', r'$\varepsilon_e$']

for idx, (param, label) in enumerate(zip(params[:3], param_labels[:3])):
    ax = axes[0, idx]
    
    # Scatter plot with trend line
    x = lepton_df[param]
    y = lepton_df['mmu']
    
    ax.scatter(x, y, alpha=0.5, s=30, c='forestgreen')
    
    # Add trend line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2,
            label=f'r = {r_value:.2f}, p = {p_value:.2e}')
    
    # Target line
    ax.axhline(0.105658, color='black', linestyle='--', linewidth=1.5, label='Target')
    
    ax.set_xlabel(label)
    ax.set_ylabel(r'$m_\mu$ (GeV)')
    ax.set_title(f'Lepton: {label} vs $m_\\mu$')
    ax.legend(loc='best', fontsize=8)

# Neutrino sector: How does theta23 depend on g_env?
ax = axes[0, 2]
ax.clear()

# Use ALL data including failures
x = neutrino_df['g_env']
y = neutrino_df['theta23']

# Show all points including failures
ax.scatter(x, y, alpha=0.3, s=20, c='darkorange', label='All data (n=480)')

# Highlight valid points
valid_mask = neutrino_df['theta23'] > 0.01
ax.scatter(x[valid_mask], y[valid_mask], alpha=0.7, s=30, c='red', 
           label=f'Valid (n={valid_mask.sum()})')

# Trend line on valid data only
x_valid = x[valid_mask]
y_valid = y[valid_mask]
slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
x_line = np.linspace(x_valid.min(), x_valid.max(), 100)
ax.plot(x_line, slope * x_line + intercept, 'b-', linewidth=2,
        label=f'r = {r_value:.2f}, p = {p_value:.2e}')

ax.axhline(0.785, color='black', linestyle='--', linewidth=1.5, label='Target')
ax.set_xlabel(r'$g_{env}$')
ax.set_ylabel(r'$\theta_{23}$ (rad)')
ax.set_title(r'Neutrino: $g_{env}$ vs $\theta_{23}$')
ax.legend(loc='best', fontsize=8)

# Bottom row: Quark sector
quark_params = ['sigma', 'alpha', 'eps_u']
quark_labels = [r'$\sigma$', r'$\alpha$', r'$\varepsilon_u$']

for idx, (param, label) in enumerate(zip(quark_params, quark_labels)):
    ax = axes[1, idx]
    
    x = quark_df[param]
    y = quark_df['mc']
    
    ax.scatter(x, y, alpha=0.3, s=20, c='steelblue')
    
    # Trend line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2,
            label=f'r = {r_value:.2f}, p = {p_value:.2e}')
    
    # Target line
    ax.axhline(1.27, color='green', linestyle='--', linewidth=2, label='Target (1.27 GeV)')
    
    ax.set_xlabel(label)
    ax.set_ylabel(r'$m_c$ (GeV)')
    ax.set_title(f'Quark: {label} vs $m_c$')
    ax.legend(loc='best', fontsize=8)

plt.suptitle('Parameter Sensitivity: How Do Predictions Depend on Each Parameter?', 
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('figures/fig_honest_param_sensitivity.pdf')
plt.savefig('figures/fig_honest_param_sensitivity.png')
print("   Saved: figures/fig_honest_param_sensitivity.pdf")

# =============================================================================
# FIGURE 3: SUCCESS RATE VS TOLERANCE THRESHOLD
# =============================================================================
print("\n3. Success Rate vs Tolerance Threshold")

fig3, ax = plt.subplots(figsize=(10, 6))

tolerances = np.logspace(-3, 0, 50)  # 0.1% to 100%

# Quark sector: success = all CKM elements and mc within tolerance
def quark_success_rate(tol):
    success = (
        (np.abs(quark_df['Vus'] - 0.2243) / 0.2243 < tol) &
        (np.abs(quark_df['Vcb'] - 0.0422) / 0.0422 < tol) &
        (np.abs(quark_df['Vub'] - 0.00382) / 0.00382 < tol) &
        (np.abs(quark_df['mc'] - 1.27) / 1.27 < tol)
    )
    return success.mean() * 100

# Lepton sector: success = all masses within tolerance
def lepton_success_rate(tol):
    success = (
        (np.abs(lepton_df['me'] - 0.000511) / 0.000511 < tol) &
        (np.abs(lepton_df['mmu'] - 0.105658) / 0.105658 < tol) &
        (np.abs(lepton_df['mtau'] - 1.77686) / 1.77686 < tol)
    )
    return success.mean() * 100

# Neutrino sector: success = all angles within tolerance
def neutrino_success_rate(tol):
    success = (
        (np.abs(neutrino_df['theta12'] - 0.5903) / 0.5903 < tol) &
        (np.abs(neutrino_df['theta23'] - 0.785) / 0.785 < tol) &
        (np.abs(neutrino_df['theta13'] - 0.149) / 0.149 < tol)
    )
    return success.mean() * 100

quark_rates = [quark_success_rate(t) for t in tolerances]
lepton_rates = [lepton_success_rate(t) for t in tolerances]
neutrino_rates = [neutrino_success_rate(t) for t in tolerances]

ax.semilogx(tolerances * 100, quark_rates, 'b-', linewidth=2, label='Quark (0% at any threshold < 100%)')
ax.semilogx(tolerances * 100, lepton_rates, 'g-', linewidth=2, label='Charged Lepton')
ax.semilogx(tolerances * 100, neutrino_rates, 'orange', linewidth=2, label='Neutrino')

# Mark key thresholds
ax.axvline(1, color='gray', linestyle='--', alpha=0.5)
ax.axvline(10, color='gray', linestyle='--', alpha=0.5)
ax.text(1, 95, '1%', ha='center', fontsize=9)
ax.text(10, 95, '10%', ha='center', fontsize=9)

ax.set_xlabel('Tolerance Threshold (%)')
ax.set_ylabel('Success Rate (%)')
ax.set_title('Model Robustness: What Fraction of Geometries Succeed?')
ax.legend(loc='lower right')
ax.set_xlim(0.1, 100)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)

# Add annotations
ax.annotate('Quark sector fails\nat all thresholds', xy=(50, 5), fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('figures/fig_honest_success_threshold.pdf')
plt.savefig('figures/fig_honest_success_threshold.png')
print("   Saved: figures/fig_honest_success_threshold.pdf")

# =============================================================================
# FIGURE 4: QUARK FAILURE MODE ANALYSIS
# =============================================================================
print("\n4. Quark Failure Mode Analysis")

fig4, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Distribution of mc predictions
ax = axes[0, 0]
ax.hist(quark_df['mc'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(1.27, color='red', linewidth=3, linestyle='--', label='Experimental target')
ax.axvline(quark_df['mc'].mean(), color='blue', linewidth=2, label=f'Model mean: {quark_df["mc"].mean():.1f} GeV')

# Shade the gap
ax.axvspan(1.27, quark_df['mc'].min(), alpha=0.3, color='red', label='Forbidden gap')

ax.set_xlabel(r'Charm Mass $m_c$ (GeV)')
ax.set_ylabel('Count')
ax.set_title('(a) Charm Mass Distribution\nModel systematically overshoots target')
ax.legend(loc='upper right')

# Panel B: CKM element errors
ax = axes[0, 1]

ckm_elements = ['Vus', 'Vcb', 'Vub']
ckm_targets = [0.2243, 0.0422, 0.00382]
ckm_errors = []

for elem, target in zip(ckm_elements, ckm_targets):
    rel_error = (quark_df[elem] - target) / target * 100
    ckm_errors.append(rel_error)

bp = ax.boxplot(ckm_errors, labels=ckm_elements, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightsteelblue')

ax.axhline(0, color='green', linewidth=2, linestyle='--', label='Perfect match')
ax.axhspan(-10, 10, alpha=0.2, color='green', label='±10% band')

ax.set_ylabel('Relative Error (%)')
ax.set_title('(b) CKM Element Errors\nLarge systematic errors in all elements')
ax.legend(loc='upper right')

# Panel C: Correlation between mc and CKM loss
ax = axes[1, 0]
sc = ax.scatter(quark_df['mc'], quark_df['loss_ckm'], 
                c=quark_df['sigma'], cmap='viridis', alpha=0.5, s=30)
ax.axvline(1.27, color='red', linewidth=2, linestyle='--', label='Target $m_c$')

# Correlation
r, p = stats.pearsonr(quark_df['mc'], quark_df['loss_ckm'])
ax.text(0.05, 0.95, f'r = {r:.2f}, p = {p:.2e}', transform=ax.transAxes, 
        va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel(r'Charm Mass $m_c$ (GeV)')
ax.set_ylabel(r'CKM Loss ($\chi^2$)')
ax.set_title('(c) Trade-off: Better mass → Worse CKM\n(But neither reaches target)')
ax.legend(loc='upper right')
plt.colorbar(sc, ax=ax, label=r'$\sigma$')

# Panel D: Why does the model fail?
ax = axes[1, 1]
ax.axis('off')

failure_text = """
QUARK SECTOR FAILURE ANALYSIS

Key Finding: The model CANNOT produce m_c < 2.5 GeV

Possible Causes:
1. Envelope-dominated regime inherently produces 
   masses that are too large
2. The kernel functional form is inappropriate 
   for the quark sector
3. The optimization is stuck in local minima
   (but 1000 random geometries all fail)

Implications:
• The "three-regime framework" is not validated
• The quark sector requires different physics
• Claims of "universal kernel" are not supported

Recommendation:
• Acknowledge failure honestly in manuscript
• Do not fit curves to failed data
• Consider alternative models for quarks
"""

ax.text(0.1, 0.95, failure_text, transform=ax.transAxes, va='top', 
        fontsize=11, family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.suptitle('Quark Sector: Understanding Why the Model Fails', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('figures/fig_honest_quark_failure.pdf')
plt.savefig('figures/fig_honest_quark_failure.png')
print("   Saved: figures/fig_honest_quark_failure.pdf")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================
print("\n" + "=" * 70)
print("HONEST ASSESSMENT SUMMARY")
print("=" * 70)

print("\nQuark Sector:")
print(f"  - mc prediction: {quark_df['mc'].mean():.2f} ± {quark_df['mc'].std():.2f} GeV")
print(f"  - mc target: 1.27 GeV")
print(f"  - Systematic error: {100*(quark_df['mc'].mean()/1.27 - 1):.0f}%")
print(f"  - Success rate (10% tolerance): {quark_success_rate(0.1):.1f}%")
print(f"  - VERDICT: MODEL FAILS")

print("\nCharged Lepton Sector:")
print(f"  - mmu prediction: {lepton_df['mmu'].mean():.4f} ± {lepton_df['mmu'].std():.4f} GeV")
print(f"  - mmu target: 0.1057 GeV")
print(f"  - Success rate (10% tolerance): {lepton_success_rate(0.1):.1f}%")
print(f"  - VERDICT: PARTIAL SUCCESS (mtau only)")

print("\nNeutrino Sector:")
print(f"  - theta23 prediction: {neutrino_valid['theta23'].mean():.3f} ± {neutrino_valid['theta23'].std():.3f}")
print(f"  - theta23 target: 0.785")
print(f"  - Optimization failure rate: {100*(1 - len(neutrino_valid)/len(neutrino_df)):.0f}%")
print(f"  - Success rate (10% tolerance): {neutrino_success_rate(0.1):.1f}%")
print(f"  - VERDICT: GOOD (but 50% optimization failures)")

print("\n" + "=" * 70)
print("HONEST FIGURES COMPLETE")
print("=" * 70)
print("\nGenerated figures:")
print("  - figures/fig_honest_pred_vs_exp.pdf")
print("  - figures/fig_honest_param_sensitivity.pdf")
print("  - figures/fig_honest_success_threshold.pdf")
print("  - figures/fig_honest_quark_failure.pdf")
