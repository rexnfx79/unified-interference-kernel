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

CHARGED_LEPTON_TARGETS = {
    'me': 0.0005109989461,  # electron mass in GeV
    'mmu': 0.1056583745,     # muon mass in GeV
    'mtau': 1.77686,         # tau mass in GeV
}

NEUTRINO_TARGETS = {
    # PMNS mixing angles (in radians, for PDG 2024)
    'theta12': 0.583,       # solar angle ~33.4°
    'theta23': 0.785,       # atmospheric angle ~45°
    'theta13': 0.149,       # reactor angle ~8.57°
    # Neutrino masses (in eV, typical values for normal hierarchy)
    'm1': 0.0,              # Lightest (effectively zero)
    'm2': 0.0086,           # sqrt(Δm21^2) ≈ 8.6 meV
    'm3': 0.050,            # sqrt(Δm31^2) ≈ 50 meV
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


def compute_charged_lepton_observables(Ye: np.ndarray) -> Dict[str, float]:
    """Compute charged lepton sector observables: masses only."""
    Ue, Se, Veh = np.linalg.svd(Ye, full_matrices=False)
    # No phase fixing needed for leptons (no mixing to extract)
    # Scale by tau mass
    scale_e = CHARGED_LEPTON_TARGETS['mtau'] / Se[0] if Se[0] > 0 else 0.0
    me = Se[2] * scale_e  # electron (smallest singular value)
    mmu = Se[1] * scale_e  # muon
    mtau = Se[0] * scale_e  # tau (largest singular value)
    return {
        'me': float(me), 'mmu': float(mmu), 'mtau': float(mtau),
        'scale_e': float(scale_e),
    }


def compute_charged_lepton_loss(obs: Dict[str, float]) -> float:
    """Compute charged lepton loss as sum of squared log-ratio errors."""
    loss = 0.0
    for key in ['me', 'mmu', 'mtau']:
        target = CHARGED_LEPTON_TARGETS[key]
        value = obs[key]
        if value > 0 and target > 0:
            log_ratio = np.log(value / target)
            loss += log_ratio ** 2
        else:
            loss += 100.0
    return float(loss)


def compute_neutrino_observables(Ynu: np.ndarray, Ye: np.ndarray) -> Dict[str, float]:
    """Compute neutrino sector observables: PMNS mixing angles and masses.
    
    Uses envelope compression (metric-dominated regime) which causes
    information loss and emergent anarchy in PMNS angles.
    """
    Unu, Snu, Vnuh = np.linalg.svd(Ynu, full_matrices=False)
    Ue, Se, Veh = np.linalg.svd(Ye, full_matrices=False)
    
    # Fix phases similar to CKM
    Unu_fixed, _, _ = fix_svd_phases(Unu, Snu, Vnuh)
    Ue_fixed, _, _ = fix_svd_phases(Ue, Se, Veh)
    
    # PMNS = Ue^dagger * Unu (charged lepton mixing * neutrino mixing)
    PMNS = Ue_fixed.conj().T @ Unu_fixed
    
    # Extract mixing angles (using standard parameterization)
    # sin^2(theta12) = |PMNS[0,1]|^2 / (1 - |PMNS[0,2]|^2)
    # sin^2(theta23) = |PMNS[1,2]|^2 / (1 - |PMNS[0,2]|^2)
    # sin^2(theta13) = |PMNS[0,2]|^2
    abs_pmns_sq = np.abs(PMNS) ** 2
    sin2_theta13 = abs_pmns_sq[0, 2]
    sin2_theta23 = abs_pmns_sq[1, 2] / (1 - sin2_theta13 + 1e-10)
    sin2_theta12 = abs_pmns_sq[0, 1] / (1 - sin2_theta13 + 1e-10)
    
    theta12 = np.arcsin(np.sqrt(np.clip(sin2_theta12, 0, 1)))
    theta23 = np.arcsin(np.sqrt(np.clip(sin2_theta23, 0, 1)))
    theta13 = np.arcsin(np.sqrt(np.clip(sin2_theta13, 0, 1)))
    
    # Neutrino masses (scale by a reference value, e.g., m2)
    # For simplicity, use relative masses
    if Snu[0] > 0:
        # Normal hierarchy: m1 < m2 < m3
        scale_nu = NEUTRINO_TARGETS['m2'] / Snu[1] if Snu[1] > 0 else 0.0
        m1 = Snu[2] * scale_nu
        m2 = Snu[1] * scale_nu
        m3 = Snu[0] * scale_nu
    else:
        m1, m2, m3 = 0.0, 0.0, 0.0
        scale_nu = 0.0
    
    return {
        'theta12': float(theta12), 'theta23': float(theta23), 'theta13': float(theta13),
        'm1': float(m1), 'm2': float(m2), 'm3': float(m3),
        'scale_nu': float(scale_nu),
    }


def compute_pmns_loss(obs: Dict[str, float]) -> float:
    """Compute PMNS loss as sum of relative squared errors for mixing angles."""
    loss = 0.0
    for key in ['theta12', 'theta23', 'theta13']:
        target = NEUTRINO_TARGETS[key]
        rel_error = (obs[key] - target) / target
        loss += rel_error ** 2
    return float(loss)


def compute_neutrino_mass_loss(obs: Dict[str, float]) -> float:
    """Compute neutrino mass loss as sum of squared log-ratio errors."""
    loss = 0.0
    # Only fit m2 and m3 (m1 is effectively zero)
    for key in ['m2', 'm3']:
        target = NEUTRINO_TARGETS[key]
        value = obs[key]
        if value > 0 and target > 0:
            log_ratio = np.log(value / target)
            loss += log_ratio ** 2
        else:
            loss += 100.0
    return float(loss)
