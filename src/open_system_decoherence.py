"""
Open-system decoherence sketch for flavor mixing (Path A, SM only).

Minimal Lindblad-style diagonal damping:

    ρ_open = (1 - p) ρ_Y + p diag(ρ_Y)   (trace-renormalized)

Decoherence rate p is fixed by an **external** parameter (g_env for neutrinos,
mean interference ε for quarks) — not inferred post-hoc from ρ_Y off-diagonals.

Hypothesis (falsifiable): larger p ↔ stronger CKM/PMNS mixing magnitudes.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Literal, Optional

from flavor_information import yukawa_density_matrix
from qed_information import coherence_l1_norm, off_diagonal_to_diagonal_ratio

SectorKind = Literal["quark", "neutrino"]


def decoherence_rate_from_g_env(
    g_env: float,
    g_min: float = 0.40,
    g_max: float = 0.80,
) -> float:
    """
    External environment coupling → diagonal damping p ∈ [0, 1].

    Linear map: g_env at g_min → p=0, g_env at g_max → p=1.
    """
    span = g_max - g_min
    if span <= 0:
        return 0.0
    return float(np.clip((g_env - g_min) / span, 0.0, 1.0))


def decoherence_rate_from_eps(
    eps: float,
    eps_min: float = 0.05,
    eps_max: float = 0.50,
) -> float:
    """Quark sector: interference strength as external decoherence proxy."""
    span = eps_max - eps_min
    if span <= 0:
        return 0.0
    return float(np.clip((eps - eps_min) / span, 0.0, 1.0))


def off_diagonal_damping_gamma(p: float) -> float:
    """Equivalent off-diagonal damping γ with ρ_ij → (1-γ) ρ_ij, γ = p."""
    return float(np.clip(p, 0.0, 1.0))


def apply_diagonal_decoherence(rho: np.ndarray, p: float) -> np.ndarray:
    """
    ρ_open = (1-p) ρ + p diag(ρ), renormalized to unit trace.
    """
    rho = np.asarray(rho, dtype=complex)
    p = float(np.clip(p, 0.0, 1.0))
    diag_part = np.diag(np.diag(rho))
    rho_open = (1.0 - p) * rho + p * diag_part
    tr = float(np.real(np.trace(rho_open)))
    if tr > 1e-15:
        rho_open = rho_open / tr
    return rho_open


def apply_gamma_damping(rho: np.ndarray, gamma: float) -> np.ndarray:
    """Scale off-diagonal entries by (1 - γ); renormalize trace."""
    rho = np.asarray(rho, dtype=complex)
    gamma = float(np.clip(gamma, 0.0, 1.0))
    out = rho.copy()
    mask = ~np.eye(rho.shape[0], dtype=bool)
    out[mask] *= (1.0 - gamma)
    tr = float(np.real(np.trace(out)))
    if tr > 1e-15:
        out = out / tr
    return out


def open_system_coherence_proxy(rho_open: np.ndarray) -> Dict[str, float]:
    """Mixing-relevant coherence measures on open-system state."""
    return {
        "coherence_l1": coherence_l1_norm(rho_open),
        "off_diagonal_ratio": off_diagonal_to_diagonal_ratio(rho_open),
    }


def open_system_mixing_proxy(
    Y: np.ndarray,
    p: float,
    method: str = "diagonal_damping",
) -> float:
    """
    Scalar mixing proxy from externally damped Yukawa density matrix.

    Uses normalized off-diagonal l1 coherence (not post-hoc p from ρ_Y).
    """
    rho = yukawa_density_matrix(Y)
    if method == "gamma_damping":
        rho_open = apply_gamma_damping(rho, off_diagonal_damping_gamma(p))
    else:
        rho_open = apply_diagonal_decoherence(rho, p)
    coh = coherence_l1_norm(rho_open)
    diag = float(np.sum(np.abs(np.diag(rho_open))))
    if diag > 1e-15:
        return float(coh / diag)
    return float(coh)


def external_decoherence_parameter(
    sector: SectorKind,
    *,
    g_env: Optional[float] = None,
    eps_u: Optional[float] = None,
    eps_d: Optional[float] = None,
) -> float:
    """Resolve external p from sector-specific inputs."""
    if sector == "neutrino":
        if g_env is None:
            raise ValueError("g_env required for neutrino sector")
        return decoherence_rate_from_g_env(g_env)
    if sector == "quark":
        if eps_u is None or eps_d is None:
            raise ValueError("eps_u and eps_d required for quark sector")
        return 0.5 * (
            decoherence_rate_from_eps(eps_u) + decoherence_rate_from_eps(eps_d)
        )
    raise ValueError(f"Unknown sector: {sector}")


def compute_open_system_row(
    sector: SectorKind,
    Y: np.ndarray,
    p: float,
    actual_mixing: float,
) -> Dict[str, float]:
    """One diagnostic row: external p, open-system proxy, actual SM mixing."""
    proxy = open_system_mixing_proxy(Y, p)
    coh = open_system_coherence_proxy(
        apply_diagonal_decoherence(yukawa_density_matrix(Y), p)
    )
    return {
        "sector": sector,
        "p_external": p,
        "mixing_proxy_open": proxy,
        "actual_mixing": actual_mixing,
        **coh,
    }
