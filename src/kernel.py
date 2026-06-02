"""
Universal Interference Kernel

Core implementation of the universal interference kernel that organizes
Yukawa couplings through envelope suppression and phase interference.

Mathematical form:
    Y_ij = exp(-d²/(2σ²)) × [1 + ε exp(iΦ)]

where:
    d = |x_i - x_j| is the distance in internal flavor coordinate
    Φ = α + k(x_i + x_j)/2 + η(x_i - x_j) is the phase structure
"""

import numpy as np
from typing import Tuple


def compute_kernel_element(
    x_left: float,
    x_right: float,
    sigma: float,
    k: float,
    alpha: float,
    eta: float,
    eps: float
) -> complex:
    """Compute a single kernel element Y_ij."""
    diff = x_left - x_right
    diff_sq = diff ** 2
    envelope = np.exp(-diff_sq / (2 * sigma ** 2))
    phase = alpha + k * (x_left + x_right) / 2 + eta * diff
    interference = 1 + eps * np.exp(1j * phase)
    return envelope * interference


def compute_yukawa_matrix(
    left_positions: Tuple[int, int, int],
    right_positions: Tuple[int, int, int],
    sigma: float,
    k: float,
    alpha: float,
    eta: float,
    eps: float
) -> np.ndarray:
    """Compute full 3x3 Yukawa matrix from kernel parameters."""
    left_vec = np.array([left_positions[0], left_positions[1], 0], dtype=float)
    right_vec = np.array(right_positions, dtype=float)
    Y = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            Y[i, j] = compute_kernel_element(
                left_vec[i], right_vec[j], sigma, k, alpha, eta, eps
            )
    return Y


def compute_quark_yukawas(
    Q: Tuple[int, int, int],
    U: Tuple[int, int, int],
    D: Tuple[int, int, int],
    sigma: float,
    k: float,
    alpha: float,
    eta: float,
    eps_u: float,
    eps_d: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute up and down Yukawa matrices for quark sector (envelope-dominated regime)."""
    Yu = compute_yukawa_matrix(Q, U, sigma, k, alpha, eta, eps_u)
    Yd = compute_yukawa_matrix(Q, D, sigma, k, alpha, eta, eps_d)
    return Yu, Yd
