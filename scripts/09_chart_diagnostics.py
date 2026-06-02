#!/usr/bin/env python3
"""
Chart Diagnostics - Verify charting consistency
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

LOG_PATH = '/Users/alexm4/Cursor Repos/unified-interference-kernel/.cursor/debug.log'

def log_debug(hypothesis_id, location, message, data):
    """Write debug log entry"""
    entry = {
        'hypothesisId': hypothesis_id,
        'location': location,
        'message': message,
        'data': data,
        'timestamp': int(pd.Timestamp.now().timestamp() * 1000),
        'sessionId': 'chart-debug',
        'runId': 'run1'
    }
    with open(LOG_PATH, 'a') as f:
        f.write(json.dumps(entry) + '\n')

# Load Data
quark_df = pd.read_csv('data/quark_results.csv')
lepton_df = pd.read_csv('data/charged_lepton_results.csv')
neutrino_df = pd.read_csv('data/neutrino_results.csv')

# #region agent log
# Hypothesis A: Lepton "resonance" claim - does model actually hit target?
target_mmu = 0.1056583745
best_mmu_error = np.min(np.abs(lepton_df['mmu'] - target_mmu))
best_mmu_pct_error = 100 * best_mmu_error / target_mmu
lepton_survivors = len(lepton_df[(lepton_df['mmu'] > 0.09) & (lepton_df['mmu'] < 0.12)])
log_debug('A', 'chart_diagnostics.py:30', 'Lepton target accuracy', {
    'best_mmu_error_GeV': float(best_mmu_error),
    'best_mmu_pct_error': float(best_mmu_pct_error),
    'survivors_in_range': lepton_survivors,
    'total': len(lepton_df),
    'claim': 'resonance structure shows physics',
    'reality': 'model hits target with <0.00001% error'
})
# #endregion

# #region agent log
# Hypothesis B: Neutrino variance - does it actually increase with g_env?
neutrino_df['g_env_rounded'] = neutrino_df['g_env'].round(2)
variance_by_genv = neutrino_df.groupby('g_env_rounded')['theta23'].var().to_dict()
variance_values = list(variance_by_genv.values())
variance_increasing = all(variance_values[i] <= variance_values[i+1] for i in range(len(variance_values)-1))
log_debug('B', 'chart_diagnostics.py:45', 'Neutrino variance trend', {
    'variance_by_genv': {str(k): float(v) for k, v in variance_by_genv.items()},
    'is_monotonically_increasing': variance_increasing,
    'claim': 'variance increases with g_env',
    'g_env_order': sorted(variance_by_genv.keys())
})
# #endregion

# #region agent log
# Hypothesis C: Quark survivor criteria mismatch
# The plot shows mc gap but survivors check CKM values
q_ckm_survivors = quark_df[
    (quark_df['Vus'] > 0.17) & (quark_df['Vus'] < 0.29) &
    (quark_df['Vcb'] > 0.025) & (quark_df['Vcb'] < 0.060) &
    (quark_df['Vub'] > 0.0018) & (quark_df['Vub'] < 0.0060)
]
q_mc_survivors = quark_df[np.abs(quark_df['mc'] - 1.27) < 0.5]
q_both_survivors = q_ckm_survivors[np.abs(q_ckm_survivors['mc'] - 1.27) < 0.5]
log_debug('C', 'chart_diagnostics.py:62', 'Quark survivor criteria mismatch', {
    'ckm_survivors': len(q_ckm_survivors),
    'mc_survivors': len(q_mc_survivors),
    'both_survivors': len(q_both_survivors),
    'total': len(quark_df),
    'issue': 'plot shows mc gap but survivor rate uses CKM criteria only'
})
# #endregion

# #region agent log
# Hypothesis D: Neutrino degenerate data points
theta23_near_zero = neutrino_df[neutrino_df['theta23'] < 0.01]
theta23_valid = neutrino_df[neutrino_df['theta23'] > 0.01]
log_debug('D', 'chart_diagnostics.py:74', 'Neutrino degenerate points', {
    'near_zero_count': len(theta23_near_zero),
    'valid_count': len(theta23_valid),
    'min_theta23': float(neutrino_df['theta23'].min()),
    'issue': 'near-zero theta23 values may be failed optimizations'
})
# #endregion

# #region agent log
# Hypothesis E: Colorbar direction check
# viridis_r means reversed - high values are dark, low values are bright
# "Lower = Better" with viridis_r means bright = good
# But loss_total varies from ~1e-11 to ~2300 - need log scale?
loss_range = lepton_df['loss_total'].max() / lepton_df['loss_total'].min()
log_debug('E', 'chart_diagnostics.py:86', 'Colorbar scale issue', {
    'loss_min': float(lepton_df['loss_total'].min()),
    'loss_max': float(lepton_df['loss_total'].max()),
    'loss_range_ratio': float(loss_range),
    'issue': 'loss spans 11 orders of magnitude - linear colormap hides variation',
    'using_log_scale': False
})
# #endregion

print("Diagnostics complete. Check debug.log for results.")
