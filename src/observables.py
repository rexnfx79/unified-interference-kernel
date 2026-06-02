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

NEUTRINO_TARGETS = {
    'theta12': 0.5903,  # radians (~33.82°), PDG 2024
    'theta23': 0.7850,  # radians (~45°)
    'theta13': 0.1490,  # radians (~8.54°)
}

# Mass-squared differences (eV²), normal ordering — PDG 2024
NEUTRINO_MASS_TARGETS = {
    'dm21': 7.53e-5,
    'dm31': 2.453e-3,
}

# Legacy neutrino optimization anchor (m2 fixed; scale from Snu[1])
NEUTRINO_M2_ANCHOR_EV = 0.0086

LEPTON_TARGETS = {
    'm_e': 0.000511,    # GeV, PDG 2024
    'm_mu': 0.1057,
    'm_tau': 1.777,
}


def fix_svd_phases(U, S, Vh):
    """Fix SVD phase ambiguities for consistent CKM extraction."""
    U_fixed = np.asarray(U, dtype=complex).copy()
    Vh_fixed = np.asarray(Vh, dtype=complex).copy()
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


def pmns_angles_from_unitary(U: np.ndarray) -> Tuple[float, float, float]:
    """Extract PMNS mixing angles (radians) from unitary matrix (PDG convention)."""
    s13 = float(np.clip(abs(U[0, 2]), 0.0, 1.0))
    theta13 = float(np.arcsin(s13))
    c13 = np.cos(theta13)
    if c13 < 1e-12:
        return 0.0, 0.0, theta13
    s12 = float(np.clip(abs(U[0, 1]) / c13, 0.0, 1.0))
    s23 = float(np.clip(abs(U[1, 2]) / c13, 0.0, 1.0))
    theta12 = float(np.arcsin(s12))
    theta23 = float(np.arcsin(s23))
    return theta12, theta23, theta13


def compute_neutrino_observables(Ynu: np.ndarray, Ye: np.ndarray) -> Dict[str, float]:
    """
    Compute neutrino sector observables: PMNS angles from SVD of Ynu and Ye.

    U_PMNS = U_e† U_nu per manuscript methodology.
    """
    Unu, Snu, Vnuh = np.linalg.svd(Ynu, full_matrices=False)
    Ue, Se, Veh = np.linalg.svd(Ye, full_matrices=False)
    Ue_fixed, _, _ = fix_svd_phases(Ue, Se, Veh)
    Unu_fixed, _, _ = fix_svd_phases(Unu, Snu, Vnuh)
    PMNS = Ue_fixed.conj().T @ Unu_fixed
    theta12, theta23, theta13 = pmns_angles_from_unitary(PMNS)

    masses = neutrino_masses_from_singular_values(Snu)
    return {
        'theta12': theta12,
        'theta23': theta23,
        'theta13': theta13,
        'Snu_0': float(Snu[0]),
        'Snu_1': float(Snu[1]),
        'Snu_2': float(Snu[2]),
        'm1': masses['m1'],
        'm2': masses['m2'],
        'm3': masses['m3'],
        'dm21': masses['dm21'],
        'dm31': masses['dm31'],
        'scale_nu': masses['scale_nu'],
        'unitarity_violation': float(np.max(np.abs(PMNS @ PMNS.conj().T - np.eye(3)))),
    }


def neutrino_masses_from_singular_values(Snu: np.ndarray) -> Dict[str, float]:
    """Neutrino masses (eV) anchored at m2 = NEUTRINO_M2_ANCHOR_EV; Δm² from ordering."""
    Snu = np.asarray(Snu, dtype=float)
    if Snu[1] <= 0:
        return {
            'm1': 0.0, 'm2': 0.0, 'm3': 0.0,
            'dm21': 0.0, 'dm31': 0.0, 'scale_nu': 0.0,
        }
    scale = NEUTRINO_M2_ANCHOR_EV / Snu[1]
    m1 = float(Snu[2] * scale)
    m2 = float(NEUTRINO_M2_ANCHOR_EV)
    m3 = float(Snu[0] * scale)
    return {
        'm1': m1,
        'm2': m2,
        'm3': m3,
        'dm21': float(m2 ** 2 - m1 ** 2),
        'dm31': float(m3 ** 2 - m1 ** 2),
        'scale_nu': float(scale),
    }


def compute_pmns_loss(obs: Dict[str, float]) -> float:
    """Compute PMNS loss as sum of relative squared errors on mixing angles."""
    loss = 0.0
    for key in ['theta12', 'theta23', 'theta13']:
        target = NEUTRINO_TARGETS[key]
        rel_error = (obs[key] - target) / target
        loss += rel_error ** 2
    return float(loss)


def compute_neutrino_mass_loss(obs: Dict[str, float]) -> float:
    """Squared log-ratio loss on Δm²21 and Δm²31 (eV²)."""
    loss = 0.0
    for key in ['dm21', 'dm31']:
        target = NEUTRINO_MASS_TARGETS[key]
        value = obs.get(key, 0.0)
        if value > 0 and target > 0:
            loss += np.log(value / target) ** 2
        else:
            loss += 100.0
    return float(loss)


def compute_neutrino_joint_loss(
    obs: Dict[str, float],
    pmns_weight: float = 5.0,
) -> float:
    """Manuscript-style joint neutrino objective: mass + weighted PMNS."""
    return compute_neutrino_mass_loss(obs) + pmns_weight * compute_pmns_loss(obs)


def compute_lepton_observables(Ye: np.ndarray) -> Dict[str, float]:
    """Compute charged lepton masses from Yukawa matrix (tau-anchored SVD scale)."""
    _, S, _ = np.linalg.svd(Ye, full_matrices=False)
    scale = LEPTON_TARGETS['m_tau'] / S[0] if S[0] > 0 else 0.0
    return {
        'm_e': float(S[2] * scale),
        'm_mu': float(S[1] * scale),
        'm_tau': float(S[0] * scale),
        'scale_e': float(scale),
    }


def compute_lepton_loss(obs: Dict[str, float]) -> float:
    """Compute lepton mass loss as sum of squared log-ratio errors."""
    loss = 0.0
    for key in ['m_e', 'm_mu', 'm_tau']:
        target = LEPTON_TARGETS[key]
        value = obs.get(key, 0.0)
        if value > 0 and target > 0:
            loss += np.log(value / target) ** 2
        else:
            loss += 100.0
    return float(loss)


# =============================================================================
# LEPTON TRAIN/HOLDOUT SPLIT
# =============================================================================
# Charged leptons have only three observables (all masses). m_tau is scale-
# anchored in compute_lepton_observables, so training on m_tau is redundant but
# kept for symmetry with quark holdout reporting. Optimize mu–tau hierarchy;
# hold out m_e (3500× gap from m_mu — hardest mass to predict).

LEPTON_TRAINING_TARGETS = {
    'm_mu': LEPTON_TARGETS['m_mu'],
    'm_tau': LEPTON_TARGETS['m_tau'],
}

LEPTON_HOLDOUT_TARGETS = {
    'm_e': LEPTON_TARGETS['m_e'],
}


def compute_lepton_training_loss(obs: Dict[str, float]) -> float:
    """Loss on training masses only (m_mu, m_tau); log-ratio for scale invariance."""
    loss = 0.0
    for key, target in LEPTON_TRAINING_TARGETS.items():
        value = obs.get(key, 0.0)
        if value > 0 and target > 0:
            loss += np.log(value / target) ** 2
        else:
            loss += 100.0
    return float(loss)


def compute_lepton_holdout_loss(obs: Dict[str, float]) -> float:
    """Holdout loss on m_e only — not used in optimization."""
    target = LEPTON_HOLDOUT_TARGETS['m_e']
    value = obs.get('m_e', 0.0)
    if value > 0 and target > 0:
        return float(np.log(value / target) ** 2)
    return 100.0


# =============================================================================
# TRAIN/HOLDOUT SPLIT FOR MINIMALITY VALIDATION (QUARKS)
# =============================================================================

# Training: CKM entries + charm mass (structural CKM–m_c trade-off axis).
# Holdout: light masses + V_ub — tests whether training fit generalizes.
TRAINING_TARGETS = {
    'mc': 1.27,
    'Vus': 0.22500,
    'Vcb': 0.04182,
}

HOLDOUT_TARGETS = {
    'ms': 0.093,
    'mu': 0.00216,
    'md': 0.00467,
    'Vub': 0.00382,
}


def compute_training_loss(obs: Dict[str, float]) -> float:
    """
    Compute loss on TRAINING targets only.
    
    This is what we optimize. Uses relative squared error.
    """
    loss = 0.0
    
    # mc: check for validity first
    mc = obs.get('mc', 0.0)
    if mc < 0.01 or mc > 500:
        return 1000.0  # Invalid solution
    
    for key, target in TRAINING_TARGETS.items():
        value = obs.get(key, 0.0)
        if target > 0 and value > 0:
            rel_err = (value - target) / target
            loss += rel_err ** 2
        else:
            loss += 100.0
    
    return float(loss)


def compute_holdout_loss(obs: Dict[str, float]) -> float:
    """
    Compute loss on HOLDOUT targets only.
    
    This is NOT used in optimization - only for evaluating generalization.
    Uses log-ratio for masses (scale-invariant) and relative error for CKM.
    """
    loss = 0.0
    
    # Light masses: use log-ratio for scale invariance
    for key in ['ms', 'mu', 'md']:
        target = HOLDOUT_TARGETS[key]
        value = obs.get(key, 0.0)
        if value > 1e-8 and target > 0:
            loss += np.log(value / target) ** 2
        else:
            loss += 100.0  # Penalty for zero/negative
    
    # V_ub: relative squared error
    vub = obs.get('Vub', 0.0)
    vub_target = HOLDOUT_TARGETS['Vub']
    if vub > 0:
        loss += ((vub - vub_target) / vub_target) ** 2
    else:
        loss += 100.0
    
    return float(loss)


def compute_penalized_loss(
    obs: Dict[str, float],
    n_extra_params: int,
    lambda_penalty: float = 0.1
) -> float:
    """
    Compute training loss with AIC-like penalty for model complexity.
    
    L_penalized = L_train + lambda * n_extra_params
    
    This discourages adding parameters unless they substantially improve fit.
    
    Parameters:
        obs: Observable dictionary
        n_extra_params: Number of extra parameters beyond base model
        lambda_penalty: Penalty weight per extra parameter
    
    Returns:
        Penalized loss value
    """
    train_loss = compute_training_loss(obs)
    penalty = lambda_penalty * n_extra_params
    return train_loss + penalty


def compute_full_ckm_observables(Yu: np.ndarray, Yd: np.ndarray) -> Dict[str, float]:
    """
    Compute full CKM matrix observables including all 9 elements and Jarlskog.
    
    Extended version of compute_quark_observables for detailed analysis.
    """
    # Get basic observables
    obs = compute_quark_observables(Yu, Yd)
    
    # Compute full CKM matrix
    Uu, Su, Vuh = np.linalg.svd(Yu, full_matrices=False)
    Ud, Sd, Vdh = np.linalg.svd(Yd, full_matrices=False)
    
    # Fix phases
    Uu_fixed, _, _ = fix_svd_phases(Uu, Su, Vuh)
    Ud_fixed, _, _ = fix_svd_phases(Ud, Sd, Vdh)
    
    CKM = Uu_fixed.conj().T @ Ud_fixed
    
    # All CKM magnitudes
    obs['Vud'] = float(abs(CKM[0, 0]))
    obs['Vcd'] = float(abs(CKM[1, 0]))
    obs['Vtd'] = float(abs(CKM[2, 0]))
    obs['Vts'] = float(abs(CKM[2, 1]))
    obs['Vtb'] = float(abs(CKM[2, 2]))
    obs['Vcs'] = float(abs(CKM[1, 1]))
    
    # Jarlskog invariant (magnitude only - sign is convention-dependent)
    J = np.imag(CKM[0, 0] * CKM[1, 1] * np.conj(CKM[0, 1]) * np.conj(CKM[1, 0]))
    obs['J_magnitude'] = float(abs(J))
    
    # Unitarity check
    VVdag = CKM @ CKM.conj().T
    obs['unitarity_violation'] = float(np.max(np.abs(VVdag - np.eye(3))))
    
    return obs
