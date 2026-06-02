"""
QED-derived information measures for Yukawa matrices (Path A).

Quantum Fisher information (QFI), decoherence/coherence proxies, and
distinguishability from a uniform reference — SM flavor scope only.
"""

import numpy as np
from typing import Dict, Optional

from flavor_information import yukawa_density_matrix


def _hermitize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.conj().T)


def symmetric_logarithmic_derivative(
    rho: np.ndarray, drho: np.ndarray, rcond: float = 1e-10
) -> np.ndarray:
    """
    Solve L rho + rho L = 2 drho for Hermitian L (symmetric logarithmic derivative).
    """
    n = rho.shape[0]
    rho = np.asarray(rho, dtype=complex)
    drho = _hermitize(drho)
    K = np.kron(np.eye(n), rho) + np.kron(rho.conj(), np.eye(n))
    rhs = 2.0 * drho.ravel()
    vec_L, _, _, _ = np.linalg.lstsq(K, rhs, rcond=rcond)
    return _hermitize(vec_L.reshape(n, n))


def quantum_fisher_trace(rho: np.ndarray, drho: np.ndarray) -> float:
    """QFI for one parameter: F = Tr(rho L^2) with L the SLD."""
    L = symmetric_logarithmic_derivative(rho, drho)
    return float(np.real(np.trace(rho @ L @ L)))


def coherence_l1_norm(rho: np.ndarray) -> float:
    """l1-norm of off-diagonal part of rho."""
    off = rho.copy()
    np.fill_diagonal(off, 0.0)
    return float(np.sum(np.abs(off)))


def off_diagonal_to_diagonal_ratio(rho: np.ndarray, eps: float = 1e-15) -> float:
    """Sum |off-diag| / sum |diag| — decoherence proxy."""
    diag = float(np.sum(np.abs(np.diag(rho))))
    off = coherence_l1_norm(rho)
    if diag <= eps:
        return 0.0 if off <= eps else float("inf")
    return off / diag


def quantum_relative_entropy(
    rho: np.ndarray, sigma: np.ndarray, eps: float = 1e-15
) -> float:
    """S(rho || sigma) = Tr(rho (log rho - log sigma)) in nats (eigenbasis)."""
    rho = _hermitize(rho)
    sigma = _hermitize(sigma)
    eig_r, U_r = np.linalg.eigh(rho)
    eig_s, U_s = np.linalg.eigh(sigma)
    eig_r = np.clip(np.real(eig_r), eps, None)
    eig_s = np.clip(np.real(eig_s), eps, None)
    log_r = U_r @ np.diag(np.log(eig_r)) @ U_r.conj().T
    log_s = U_s @ np.diag(np.log(eig_s)) @ U_s.conj().T
    diff = log_r - log_s
    return float(np.real(np.trace(rho @ diff)))


def uniform_yukawa_density(dim: int = 3) -> np.ndarray:
    """Normalized rho for Y = ones(dim, dim) / sqrt(dim) — distinguishability reference."""
    Y = np.ones((dim, dim), dtype=complex) / np.sqrt(dim)
    return yukawa_density_matrix(Y)


def yukawa_qfi_mean_over_elements(
    Y: np.ndarray, eps: float = 1e-7, max_directions: Optional[int] = None
) -> float:
    """
    Mean QFI over finite-difference perturbations of each Y_{ij} (real step).
    """
    Y = np.asarray(Y, dtype=complex)
    n = Y.shape[0]
    rho0 = yukawa_density_matrix(Y)
    directions = [(i, j) for i in range(n) for j in range(n)]
    if max_directions is not None:
        directions = directions[:max_directions]
    qfis = []
    for i, j in directions:
        Yp = Y.copy()
        Ym = Y.copy()
        Yp[i, j] = Yp[i, j] + eps
        Ym[i, j] = Ym[i, j] - eps
        drho = (yukawa_density_matrix(Yp) - yukawa_density_matrix(Ym)) / (2.0 * eps)
        F = quantum_fisher_trace(rho0, drho)
        if np.isfinite(F) and F >= 0:
            qfis.append(F)
    return float(np.mean(qfis)) if qfis else 0.0


def yukawa_qfi_singular_values(Y: np.ndarray, eps: float = 1e-7) -> float:
    """QFI proxy from perturbing log singular values of Y (scale-sensitive texture)."""
    _, S, _ = np.linalg.svd(Y, full_matrices=False)
    S = np.clip(S, 1e-15, None)
    logS = np.log(S)
    rho0 = yukawa_density_matrix(Y)
    qfis = []
    for k in range(len(logS)):
        lp = logS.copy()
        lm = logS.copy()
        lp[k] += eps
        lm[k] -= eps
        Sp = np.exp(lp)
        Sm = np.exp(lm)
        U, _, Vh = np.linalg.svd(Y, full_matrices=False)
        Yp = U @ np.diag(Sp) @ Vh
        Ym = U @ np.diag(Sm) @ Vh
        drho = (yukawa_density_matrix(Yp) - yukawa_density_matrix(Ym)) / (2.0 * eps)
        F = quantum_fisher_trace(rho0, drho)
        if np.isfinite(F) and F >= 0:
            qfis.append(F)
    return float(np.mean(qfis)) if qfis else 0.0


def distinguishability_from_uniform(Y: np.ndarray) -> float:
    """S(rho_Y || rho_uniform) — information to distinguish from uniform Yukawa."""
    rho = yukawa_density_matrix(Y)
    rho_u = uniform_yukawa_density(Y.shape[0])
    return quantum_relative_entropy(rho, rho_u)


def cramér_rao_bound_proxy(fisher: float) -> float:
    """Minimum variance lower bound 1/F (Cramér–Rao style; inf if F<=0)."""
    if fisher <= 0 or not np.isfinite(fisher):
        return float("inf")
    return float(1.0 / fisher)


def compute_qed_yukawa_information(Y: np.ndarray) -> Dict[str, float]:
    """
    QED-derived information measures for Yukawa Y (SM flavor scope).

    Returns QFI proxies, decoherence/coherence measures, uniform distinguishability,
    and Cramér–Rao bound from mean element QFI.
    """
    rho = yukawa_density_matrix(Y)
    qfi_elem = yukawa_qfi_mean_over_elements(Y)
    qfi_sv = yukawa_qfi_singular_values(Y)
    rel_ent = distinguishability_from_uniform(Y)
    coh_l1 = coherence_l1_norm(rho)
    off_ratio = off_diagonal_to_diagonal_ratio(rho)
    return {
        "qfi_mean_elements": qfi_elem,
        "qfi_mean_singular_values": qfi_sv,
        "coherence_l1": coh_l1,
        "off_diagonal_ratio": off_ratio,
        "distinguishability_uniform": rel_ent,
        "cramer_rao_from_qfi_elements": cramér_rao_bound_proxy(qfi_elem),
    }
