"""
Flavor information measures from Yukawa matrices.

Form normalized density matrix rho ∝ Y Y† / Tr(Y Y†) and compute
von Neumann entropy, effective rank, and off-diagonal entropy.
"""

import numpy as np
from typing import Dict


def yukawa_density_matrix(Y: np.ndarray) -> np.ndarray:
    """Normalized Hermitian density matrix rho ∝ Y Y† / Tr(Y Y†)."""
    gram = Y @ Y.conj().T
    trace = np.real(np.trace(gram))
    if trace <= 0:
        raise ValueError("Y Y† must have positive trace")
    return gram / trace


def von_neumann_entropy(rho: np.ndarray, eps: float = 1e-15) -> float:
    """Von Neumann entropy S(rho) = -Tr(rho log rho) in nats."""
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = np.clip(np.real(eigvals), eps, None)
    return float(-np.sum(eigvals * np.log(eigvals)))


def effective_rank(S: float) -> float:
    """Effective rank exp(S) for normalized density matrix."""
    return float(np.exp(S))


def off_diagonal_entropy(rho: np.ndarray, eps: float = 1e-15) -> float:
    """Entropy of normalized off-diagonal mass distribution."""
    off = rho.copy()
    np.fill_diagonal(off, 0.0)
    mass = np.sum(np.abs(off))
    if mass <= eps:
        return 0.0
    probs = np.abs(off).ravel() / mass
    probs = probs[probs > eps]
    return float(-np.sum(probs * np.log(probs)))


def compute_yukawa_information(Y: np.ndarray, include_qed: bool = False) -> Dict[str, float]:
    """Compute S(rho_Y), effective rank, and off-diagonal entropy for Yukawa Y.

    If include_qed=True, merge QFI/coherence measures from qed_information.
    """
    rho = yukawa_density_matrix(Y)
    S = von_neumann_entropy(rho)
    out = {
        'entropy': S,
        'effective_rank': effective_rank(S),
        'off_diagonal_entropy': off_diagonal_entropy(rho),
        'trace_yydag': float(np.real(np.trace(Y @ Y.conj().T))),
    }
    if include_qed:
        from qed_information import compute_qed_yukawa_information
        out.update(compute_qed_yukawa_information(Y))
    return out
