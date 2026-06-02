"""
Experimental Fisher information from PDG-weighted observable Jacobians (Path A).

Fisher matrix for kernel parameters θ from SM flavor observables μ(θ):

    F_ij = Σ_k (1/σ_k²) (∂μ_k/∂θ_i)(∂μ_k/∂θ_j)

where σ_k are PDG-inspired absolute uncertainties (documented below).

This is **experimental** distinguishability — not post-hoc ρ_Y QFI.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from kernel import compute_quark_yukawas, compute_yukawa_matrix
from observables import (
    TRAINING_TARGETS,
    compute_quark_observables,
    compute_neutrino_observables,
    compute_training_loss,
)

# ---------------------------------------------------------------------------
# PDG uncertainty assumptions (relative unless noted)
# Sources: PDG 2024 Review — rounded representative values for mechanism tests.
# Masses: relative uncertainty on GeV scale; CKM: relative on |V_ij|;
# PMNS angles: absolute rad uncertainty.
# ---------------------------------------------------------------------------

PDG_RELATIVE_UNCERTAINTY = {
    "Vus": 0.003 / 0.22500,
    "Vcb": 0.00048 / 0.04182,
    "Vub": 0.00017 / 0.00382,
    "mc": 0.02 / 1.27,
    "md": 0.00048 / 0.00467,
    "ms": 0.0007 / 0.093,
    "mu": 0.00011 / 0.00216,
    "m_e": 0.000000011 / 0.000511,
    "m_mu": 0.00000012 / 0.1057,
    "m_tau": 0.00012 / 1.777,
    "theta12": 0.020 / 0.5903,
    "theta23": 0.020 / 0.7850,
    "theta13": 0.005 / 0.1490,
}

LEPTON_TARGETS = {"m_e": 0.000511, "m_mu": 0.1057, "m_tau": 1.777}

QUARK_OBS_KEYS = ["Vus", "Vcb", "Vub", "mc", "md", "ms", "mu"]
LEPTON_OBS_KEYS = ["log_m_e", "log_m_mu", "log_m_tau"]
NEUTRINO_OBS_KEYS = ["theta12", "theta23", "theta13"]

QUARK_PARAM_NAMES = ["sigma", "k", "alpha", "eta", "eps_u", "eps_d"]
LEPTON_PARAM_NAMES = ["sigma", "k", "alpha", "eta", "eps"]
NEUTRINO_PARAM_NAMES = ["sigma", "k", "alpha", "eta", "eps_nu", "eps_e", "g_env"]
UNIVERSAL_PARAM_NAMES = ["sigma", "k", "alpha", "eta"]


def _absolute_uncertainty(key: str) -> float:
    """Absolute σ_k for observable key (handles log-mass keys)."""
    if key.startswith("log_"):
        base = key[4:]
        rel = PDG_RELATIVE_UNCERTAINTY.get(base, 0.05)
        return rel
    if key in ("theta12", "theta23", "theta13"):
        from observables import NEUTRINO_TARGETS

        target_val = NEUTRINO_TARGETS[key]
        return PDG_RELATIVE_UNCERTAINTY[key] * target_val
    if key in TRAINING_TARGETS:
        rel = PDG_RELATIVE_UNCERTAINTY.get(key, 0.05)
        return rel * TRAINING_TARGETS[key]
    if key in LEPTON_TARGETS:
        rel = PDG_RELATIVE_UNCERTAINTY.get(key, 0.05)
        return rel * LEPTON_TARGETS[key]
    if key in PDG_RELATIVE_UNCERTAINTY:
        from observables import QUARK_TARGETS

        target_val = QUARK_TARGETS.get(key, 1.0)
        return PDG_RELATIVE_UNCERTAINTY[key] * target_val
    from observables import QUARK_TARGETS

    rel = PDG_RELATIVE_UNCERTAINTY.get(key, 0.05)
    return rel * QUARK_TARGETS.get(key, 1.0)


def lepton_observables_from_ye(Ye: np.ndarray) -> Dict[str, float]:
    """Charged lepton masses + log masses for Fisher (wraps observables.py)."""
    from observables import compute_lepton_observables

    out = dict(compute_lepton_observables(Ye))
    for k in ("m_e", "m_mu", "m_tau"):
        v = out[k]
        out[f"log_{k}"] = float(np.log(v)) if v > 0 else float("nan")
    return out


def observable_vector(keys: Sequence[str], obs: Dict[str, float]) -> np.ndarray:
    return np.array([obs[k] for k in keys], dtype=float)


def numerical_jacobian(
    mu_fn: Callable[[np.ndarray], np.ndarray],
    theta: np.ndarray,
    eps: float = 1e-5,
    rel_step: bool = True,
) -> np.ndarray:
    """Central-difference Jacobian ∂μ/∂θ (n_obs × n_params)."""
    theta = np.asarray(theta, dtype=float)
    n = len(theta)
    mu0 = mu_fn(theta)
    n_obs = len(mu0)
    J = np.zeros((n_obs, n), dtype=float)
    for i in range(n):
        h = eps * (abs(theta[i]) + 1e-8) if rel_step else eps
        tp = theta.copy()
        tm = theta.copy()
        tp[i] += h
        tm[i] -= h
        J[:, i] = (mu_fn(tp) - mu_fn(tm)) / (2.0 * h)
    return J


def fisher_information_matrix(
    jacobian: np.ndarray, obs_keys: Sequence[str]
) -> np.ndarray:
    """F_ij = Σ_k (1/σ_k²) J_ki J_kj."""
    sigmas = np.array([_absolute_uncertainty(k) for k in obs_keys], dtype=float)
    sigmas = np.clip(sigmas, 1e-15, None)
    weights = 1.0 / (sigmas ** 2)
    Jw = jacobian * weights[:, np.newaxis]
    return jacobian.T @ Jw


def fisher_effective_rank(F: np.ndarray, rel_tol: float = 1e-10) -> float:
    """Participation ratio of Fisher eigenvalues."""
    eigvals = np.linalg.eigvalsh(_symmetrize(F))
    eigvals = np.clip(eigvals, 0.0, None)
    s = float(np.sum(eigvals))
    if s <= 0:
        return 0.0
    lam_max = float(np.max(eigvals))
    if lam_max <= 0:
        return 0.0
    active = eigvals[eigvals > rel_tol * lam_max]
    if len(active) == 0:
        return 0.0
    return float((np.sum(active) ** 2) / np.sum(active ** 2))


def fisher_scalar_summaries(F: np.ndarray) -> Dict[str, float]:
    """Log-det, max eigenvalue, condition number, effective rank."""
    F = _symmetrize(F)
    eigvals = np.linalg.eigvalsh(F)
    eigvals = np.clip(eigvals, 0.0, None)
    lam_max = float(np.max(eigvals)) if len(eigvals) else 0.0
    lam_min = float(np.min(eigvals[eigvals > 0])) if np.any(eigvals > 0) else 0.0
    logdet = float(np.sum(np.log(eigvals[eigvals > 1e-30]))) if np.any(eigvals > 1e-30) else float("-inf")
    cond = lam_max / lam_min if lam_min > 0 else float("inf")
    return {
        "logdet_fisher": logdet,
        "max_eigenvalue": lam_max,
        "min_eigenvalue": lam_min,
        "condition_number": cond,
        "effective_rank": fisher_effective_rank(F),
    }


def parameter_correlation_from_fisher(F: np.ndarray) -> np.ndarray:
    """Approximate parameter correlations from Fisher inverse (pseudoinverse if singular)."""
    F = _symmetrize(F)
    try:
        Finv = np.linalg.pinv(F, rcond=1e-12)
        d = np.sqrt(np.clip(np.diag(Finv), 0.0, None))
        d = np.where(d > 0, d, np.nan)
        C = Finv / np.outer(d, d)
        return np.clip(C, -1.0, 1.0)
    except np.linalg.LinAlgError:
        return np.full_like(F, np.nan)


def cramér_rao_bounds(F: np.ndarray, param_names: Sequence[str]) -> Dict[str, float]:
    """Diagonal CR lower bounds 1/F_ii (inf if non-identifiable)."""
    F = _symmetrize(F)
    bounds = {}
    for i, name in enumerate(param_names):
        v = F[i, i]
        bounds[name] = float(1.0 / v) if v > 1e-30 else float("inf")
    return bounds


def principal_eigenvector(F: np.ndarray) -> np.ndarray:
    """Unit eigenvector for largest Fisher eigenvalue."""
    F = _symmetrize(F)
    eigvals, eigvecs = np.linalg.eigh(F)
    idx = int(np.argmax(eigvals))
    v = eigvecs[:, idx].astype(float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def eigenvector_alignment(v1: np.ndarray, v2: np.ndarray) -> float:
    """|cos θ| between two vectors (handles sign ambiguity)."""
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-15 or n2 < 1e-15:
        return float("nan")
    return float(abs(np.dot(v1 / n1, v2 / n2)))


def _symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def _make_quark_mu_fn(
    Q: Tuple[int, int, int],
    U: Tuple[int, int, int],
    D: Tuple[int, int, int],
) -> Callable[[np.ndarray], np.ndarray]:
    def mu_fn(theta: np.ndarray) -> np.ndarray:
        sigma, k, alpha, eta, eps_u, eps_d = theta
        Yu, Yd = compute_quark_yukawas(Q, U, D, sigma, k, alpha, eta, eps_u, eps_d)
        obs = compute_quark_observables(Yu, Yd)
        return observable_vector(QUARK_OBS_KEYS, obs)

    return mu_fn


def _make_lepton_mu_fn(
    L: Tuple[int, int, int],
    E: Tuple[int, int, int],
) -> Callable[[np.ndarray], np.ndarray]:
    def mu_fn(theta: np.ndarray) -> np.ndarray:
        sigma, k, alpha, eta, eps = theta
        Ye = compute_yukawa_matrix(L, E, sigma, k, alpha, eta, eps)
        obs = lepton_observables_from_ye(Ye)
        return observable_vector(LEPTON_OBS_KEYS, obs)

    return mu_fn


def _make_neutrino_mu_fn(
    L: Tuple[int, int, int],
    N: Tuple[int, int, int],
) -> Callable[[np.ndarray], np.ndarray]:
    def mu_fn(theta: np.ndarray) -> np.ndarray:
        sigma, k, alpha, eta, eps_nu, eps_e, g_env = theta
        Ynu = compute_yukawa_matrix(L, N, sigma * g_env, k, alpha, eta, eps_nu)
        Ye = compute_yukawa_matrix(L, N, sigma, k, alpha, eta, eps_e)
        obs = compute_neutrino_observables(Ynu, Ye)
        return observable_vector(NEUTRINO_OBS_KEYS, obs)

    return mu_fn


def compute_sector_experimental_fisher(
    sector: str,
    geometry: Dict,
    theta: np.ndarray,
    eps: float = 1e-5,
) -> Dict:
    """
    Full experimental Fisher analysis for one sector / geometry / parameter point.

    Returns Fisher matrix, summaries, CR bounds, principal eigenvector, and loss.
    """
    if sector == "quark":
        mu_fn = _make_quark_mu_fn(geometry["Q"], geometry["U"], geometry["D"])
        param_names = QUARK_PARAM_NAMES
        obs_keys = QUARK_OBS_KEYS
        sigma, k, alpha, eta, eps_u, eps_d = theta
        Yu, Yd = compute_quark_yukawas(
            geometry["Q"], geometry["U"], geometry["D"],
            sigma, k, alpha, eta, eps_u, eps_d,
        )
        obs = compute_quark_observables(Yu, Yd)
        loss = compute_training_loss(obs)
        regime = 0
    elif sector == "lepton":
        mu_fn = _make_lepton_mu_fn(geometry["L"], geometry["E"])
        param_names = LEPTON_PARAM_NAMES
        obs_keys = LEPTON_OBS_KEYS
        sigma, k, alpha, eta, eps = theta
        Ye = compute_yukawa_matrix(geometry["L"], geometry["E"], sigma, k, alpha, eta, eps)
        obs = lepton_observables_from_ye(Ye)
        loss = 0.0
        for key in ("m_e", "m_mu", "m_tau"):
            t = LEPTON_TARGETS[key]
            v = obs[key]
            if v > 0 and t > 0:
                loss += np.log(v / t) ** 2
            else:
                loss += 100.0
        loss = float(loss)
        regime = 1
    elif sector == "neutrino":
        mu_fn = _make_neutrino_mu_fn(geometry["L"], geometry["N"])
        param_names = NEUTRINO_PARAM_NAMES
        obs_keys = NEUTRINO_OBS_KEYS
        sigma, k, alpha, eta, eps_nu, eps_e, g_env = theta
        Ynu = compute_yukawa_matrix(
            geometry["L"], geometry["N"], sigma * g_env, k, alpha, eta, eps_nu
        )
        Ye = compute_yukawa_matrix(geometry["L"], geometry["N"], sigma, k, alpha, eta, eps_e)
        obs = compute_neutrino_observables(Ynu, Ye)
        loss = 0.0
        from observables import NEUTRINO_TARGETS

        for key in NEUTRINO_OBS_KEYS:
            t = NEUTRINO_TARGETS[key]
            v = obs[key]
            loss += ((v - t) / t) ** 2
        loss = float(loss)
        regime = 2
    else:
        raise ValueError(f"Unknown sector: {sector}")

    J = numerical_jacobian(mu_fn, theta, eps=eps)
    F = fisher_information_matrix(J, obs_keys)
    summaries = fisher_scalar_summaries(F)
    corr = parameter_correlation_from_fisher(F)
    cr = cramér_rao_bounds(F, param_names)
    v_princ = principal_eigenvector(F)

    return {
        "sector": sector,
        "geometry": geometry.get("name", ""),
        "regime": regime,
        "loss": loss,
        "fisher": F,
        "jacobian": J,
        "param_names": list(param_names),
        "obs_keys": list(obs_keys),
        "summaries": summaries,
        "correlation": corr,
        "cramer_rao": cr,
        "principal_eigenvector": v_princ,
        "theta": np.asarray(theta, dtype=float),
    }


def align_fisher_subspaces(
    v_a: np.ndarray,
    v_b: np.ndarray,
    names_a: Sequence[str],
    names_b: Sequence[str],
) -> float:
    """
    Align principal Fisher directions on shared parameter names (sigma, k, alpha, eta).
    """
    shared = [n for n in names_a if n in names_b]
    if not shared:
        return float("nan")
    ia = [list(names_a).index(n) for n in shared]
    ib = [list(names_b).index(n) for n in shared]
    return eigenvector_alignment(v_a[ia], v_b[ib])
