"""
Standard QED / QFT spectral sums (integer mode indices) for Tier 5.5 audit.

References: Schwinger g-2 (one-loop), Casimir ζ-regularization, vacuum polarization
coefficients (ζ(2), ζ(4)), Euler product for ζ(s). No flavor scope.
"""

import math
from typing import Optional

import numpy as np

# First 50 Riemann zeros — not used here; kept for cross-diag consistency if needed
ZETA2 = math.pi ** 2 / 6.0
ZETA4 = math.pi ** 4 / 90.0
ZETA3 = 1.2020569031595942  # Apéry
SCHWINGER_SUM = 1.0  # sum_{k>=1} 1/(k(k+1))
CASIMIR_ZETA_MINUS1 = -1.0 / 12.0  # ζ(-1) from ∑ n regularization


def primes_up_to(n_max: int) -> np.ndarray:
    """Sieve of Eratosthenes."""
    if n_max < 2:
        return np.array([], dtype=int)
    sieve = np.ones(n_max + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(n_max**0.5) + 1):
        if sieve[p]:
            sieve[p * p :: p] = False
    return np.nonzero(sieve)[0]


def integer_partial_sum_zeta(s: float, n_max: int) -> float:
    ns = np.arange(1, n_max + 1, dtype=float)
    return float(np.sum(ns ** (-s)))


def prime_partial_sum_zeta(s: float, n_max: int) -> float:
    ps = primes_up_to(n_max).astype(float)
    if len(ps) == 0:
        return 0.0
    return float(np.sum(ps ** (-s)))


def euler_product_zeta(s: float, p_max: int) -> float:
    """∏_{p≤p_max} (1 - p^{-s})^{-1} — prime factorization side of ζ(s)."""
    ps = primes_up_to(p_max).astype(float)
    if len(ps) == 0:
        return 1.0
    return float(np.prod((1.0 - ps ** (-s)) ** (-1)))


def schwinger_g2_series_integer(k_max: int) -> float:
    ks = np.arange(1, k_max + 1, dtype=float)
    return float(np.sum(1.0 / (ks * (ks + 1.0))))


def schwinger_g2_series_prime(p_max: int) -> float:
    ps = primes_up_to(p_max).astype(float)
    if len(ps) == 0:
        return 0.0
    return float(np.sum(1.0 / (ps * (ps + 1.0))))


def casimir_mode_sum_integer(n_max: int) -> float:
    """Raw ∑_{n=1}^N n (divergent; compare to ζ(-1) limit via ratio)."""
    ns = np.arange(1, n_max + 1, dtype=float)
    return float(np.sum(ns))


def casimir_mode_sum_prime(p_max: int) -> float:
    ps = primes_up_to(p_max).astype(float)
    if len(ps) == 0:
        return 0.0
    return float(np.sum(ps))


def vacuum_polarization_coefficient_integer(n_max: int) -> float:
    """
    One-loop photon vacuum pol. coefficient ∝ π² α² / (45 m_e²) uses ζ(4)=π⁴/90.
    Audit proxy: partial sum ∑ n^{-4} → ζ(4).
    """
    return integer_partial_sum_zeta(4.0, n_max)


def vacuum_polarization_coefficient_prime(p_max: int) -> float:
    return prime_partial_sum_zeta(4.0, p_max)


def relative_error(target: float, surrogate: float) -> float:
    if abs(target) < 1e-15:
        return float("inf") if abs(surrogate) > 1e-15 else 0.0
    return abs(target - surrogate) / abs(target)


def audit_observable(
    name: str,
    target: float,
    integer_value: float,
    prime_value: float,
    euler_value: Optional[float] = None,
) -> dict:
    """Compare integer-index vs prime-only surrogate vs optional Euler product."""
    out = {
        "name": name,
        "target": target,
        "integer": integer_value,
        "prime_only": prime_value,
        "rel_err_integer": relative_error(target, integer_value),
        "rel_err_prime_only": relative_error(target, prime_value),
    }
    if euler_value is not None:
        out["euler_product"] = euler_value
        out["rel_err_euler"] = relative_error(target, euler_value)
    return out
