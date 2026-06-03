"""
1D Gaussian split-fermion overlap integrals vs discrete interference kernel.

Used by diagnostics/10 and diagnostics/33 (Tier 3 theory bridge).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple

from kernel import compute_yukawa_matrix


def gaussian_overlap_matrix(
    left_y: np.ndarray,
    right_y: np.ndarray,
    width: float,
) -> np.ndarray:
    """Analytic overlap of normalized Gaussians psi(y) ~ exp(-(y-y0)^2/(2*width^2))."""
    n_l, n_r = len(left_y), len(right_y)
    o = np.zeros((n_l, n_r), dtype=float)
    for i in range(n_l):
        for j in range(n_r):
            d = left_y[i] - right_y[j]
            o[i, j] = np.exp(-(d ** 2) / (4.0 * width ** 2))
    return o


def phase_matrix(
    left_y: np.ndarray,
    right_y: np.ndarray,
    k: float,
    alpha: float,
    eta: float,
) -> np.ndarray:
    n_l, n_r = len(left_y), len(right_y)
    phi = np.zeros((n_l, n_r), dtype=float)
    for i in range(n_l):
        for j in range(n_r):
            phi[i, j] = alpha + k * (left_y[i] + right_y[j]) / 2.0 + eta * (
                left_y[i] - right_y[j]
            )
    return phi


def build_overlap_yukawa(
    left_y: np.ndarray,
    right_y: np.ndarray,
    width: float,
    k: float,
    alpha: float,
    eta: float,
    eps: float,
) -> np.ndarray:
    mag = gaussian_overlap_matrix(left_y, right_y, width)
    phi = phase_matrix(left_y, right_y, k, alpha, eta)
    return mag * (1.0 + eps * np.exp(1j * phi))


def positions_to_legacy(
    Q: Tuple, U: Tuple, D: Tuple
) -> Tuple[Tuple, Tuple, Tuple]:
    """Phenomenology sorted triple Q → legacy (q1, q2, 0)."""
    return (int(Q[0]), int(Q[1]), 0), tuple(int(x) for x in U), tuple(int(x) for x in D)


def fit_width_to_kernel(
    left_pos: Tuple,
    right_pos: Tuple,
    sigma: float,
    k: float,
    alpha: float,
    eta: float,
    eps: float,
) -> Dict[str, float]:
    """Find Gaussian width w matching kernel envelope magnitudes."""
    left_y = np.array([left_pos[0], left_pos[1], 0.0], dtype=float)
    right_y = np.array(right_pos, dtype=float)
    y_kernel = compute_yukawa_matrix(left_pos, right_pos, sigma, k, alpha, eta, eps)
    target_mag = np.abs(y_kernel)
    tmax = float(np.abs(target_mag).max()) + 1e-15

    def loss(log_w):
        w = np.exp(log_w[0])
        mag = gaussian_overlap_matrix(left_y, right_y, w)
        mmax = float(mag.max()) + 1e-15
        rel = (mag / mmax - target_mag / tmax) ** 2
        return float(rel.sum())

    res = minimize(loss, x0=[np.log(max(sigma / np.sqrt(2.0), 1e-3))], method="Nelder-Mead")
    w_opt = float(np.exp(res.x[0]))
    y_overlap = build_overlap_yukawa(left_y, right_y, w_opt, k, alpha, eta, eps)
    mag_corr = float(
        np.corrcoef(np.abs(y_overlap).ravel(), np.abs(y_kernel).ravel())[0, 1]
    )
    phase_err = float(np.mean(np.abs(np.angle(y_overlap / (y_kernel + 1e-15)))))

    return {
        "width": w_opt,
        "sigma_relation": w_opt / sigma if sigma > 0 else float("nan"),
        "magnitude_correlation": mag_corr,
        "mean_phase_error_rad": phase_err,
        "overlap_loss": float(res.fun),
    }
