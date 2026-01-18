"""
Observable Extraction for Flavor Physics

Compute physical observables (masses, mixing) from Yukawa matrices using SVD.
All experimental targets from PDG 2024.
"""

import numpy as np
from typing import Dict, Tuple

QUARK_TARGETS = {
    'Vus': 0.22500, 'Vcb': 0.04182, 'Vub': 0.00382,
    'mu': 0.00216, 'mc': 1.27, 'md': 0.00467, 'ms': 0.093,
    'mb': 4.18, 'mt': 172.5,
}


def fix_svd_phases(U, S, Vh):
    """Fix SVD phase ambiguities for consistent CKM extraction."""
    U_fixed = U.copy()
    Vh_fixed = Vh.copy()
    for i in range(U.shape[0]):
        if abs(U[i, 0]) > 1e-10:
            phase = np.angle(U[i, 0])
            U_fixed[i, :] *= np.exp(-1j * phase)
    for j in range(Vh.shape[1]):
        if abs(Vh[0, j]) > 1e-10:
            phase = np.angle(Vh[0, j])
            Vh_fixed[:, j] *= np.exp(-1j * phase)
    return U_fixed, S, Vh_fixed


def compute_quark_observables(Yu: np.ndarray, Yd: np.ndarray) -> Dict[str, float]:
    """Compute quark sector observables: CKM mixing and masses."""
    Uu, Su, Vuh = np.linalg.svd(Yu, full_matrices=False)
    Ud, Sd, Vdh = np.linalg.svd(Yd, full_matrices=False)
    Uu_fixed, _, _ = fix_svd_phases(Uu, Su, Vuh)
    Ud_fixed, _, _ = fix_svd_phases(Ud, Sd, Vdh)
    CKM = Uu_fixed.conj().T @ Ud_fixed
    Vus = abs(CKM[0, 1])
    Vcb = abs(CKM[1, 2])
    Vub = abs(CKM[0, 2])
    scale_u = QUARK_TARGETS['mt'] / Su[0] if Su[0] > 0 else 0.0
    scale_d = QUARK_TARGETS['mb'] / Sd[0] if Sd[0] > 0 else 0.0
    mu = Su[2] * scale_u
    mc = Su[1] * scale_u
    md = Sd[2] * scale_d
    ms = Sd[1] * scale_d
    return {
        'Vus': float(Vus), 'Vcb': float(Vcb), 'Vub': float(Vub),
        'mu': float(mu), 'mc': float(mc), 'md': float(md), 'ms': float(ms),
        'scale_u': float(scale_u), 'scale_d': float(scale_d),
    }


def compute_ckm_loss(obs: Dict[str, float]) -> float:
    """Compute CKM loss as sum of relative squared errors."""
    loss = 0.0
    for key in ['Vus', 'Vcb', 'Vub']:
        target = QUARK_TARGETS[key]
        rel_error = (obs[key] - target) / target
        loss += rel_error ** 2
    return float(loss)


def compute_mass_loss(obs: Dict[str, float]) -> float:
    """Compute mass loss as sum of squared log-ratio errors."""
    loss = 0.0
    for key in ['mu', 'mc', 'md', 'ms']:
        target = QUARK_TARGETS[key]
        value = obs[key]
        if value > 0 and target > 0:
            log_ratio = np.log(value / target)
            loss += log_ratio ** 2
        else:
            loss += 100.0
    return float(loss)
