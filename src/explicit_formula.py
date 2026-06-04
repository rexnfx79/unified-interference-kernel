"""
Riemann explicit formula utilities (Path D — arithmetic only, SM-decoupled).

psi(x) = x - sum_rho x^rho/rho - log(2*pi) - (1/2)log(1 - x^{-2})
Non-trivial zeros on critical line: rho = 1/2 + i*gamma (RH assumed for audit).
"""

from __future__ import annotations

import cmath
import math
from typing import List, Sequence

# First 50 non-trivial zero heights (Im rho), ascending — standard tables
RIEMANN_ZERO_IMAG: List[float] = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.606045, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.145896,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
    103.725538, 105.446623, 107.168611, 111.029536, 111.874659,
    114.320221, 116.226680, 118.790782, 121.370125, 122.946829,
    124.256819, 127.516684, 129.578704, 131.087688, 133.497737,
    134.756509, 136.457552, 138.116072, 139.736209, 141.123707,
]


def primes_up_to(n: int) -> List[int]:
    if n < 2:
        return []
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            start = i * i
            sieve[start : n + 1 : i] = b"\x00" * ((n - start) // i + 1)
    return [i for i in range(2, n + 1) if sieve[i]]


def chebyshev_psi(x: float) -> float:
    """psi(x) = sum_{p^k <= x} log p."""
    if x < 2:
        return 0.0
    n = int(x)
    total = 0.0
    for p in primes_up_to(n):
        pk = p
        while pk <= x:
            total += math.log(p)
            pk *= p
    return total


def archimedean_correction(x: float) -> float:
    """-log(2*pi) - (1/2)log(1 - x^{-2})."""
    if x <= 1:
        return 0.0
    return -math.log(2 * math.pi) - 0.5 * math.log(1 - x ** (-2))


def zero_pair_contribution(x: float, gamma: float) -> float:
    """Contribution from conjugate pair on critical line to -sum x^rho/rho."""
    rho = 0.5 + 1j * gamma
    try:
        lx = cmath.log(x)
        t1 = cmath.exp(rho * lx) / rho
        t2 = cmath.exp((1 - rho) * lx) / (1 - rho)
        return -(t1 + t2).real
    except (OverflowError, ValueError, ZeroDivisionError):
        return 0.0


def psi_from_explicit_formula(x: float, zero_imag: Sequence[float]) -> float:
    """psi(x) from truncated explicit formula (critical-line zeros)."""
    if x < 2:
        return 0.0
    val = x + archimedean_correction(x)
    for gamma in zero_imag:
        val += zero_pair_contribution(x, gamma)
    return val


def random_null_frequencies(
    n: int,
    gamma_min: float,
    gamma_max: float,
    seed: int,
) -> List[float]:
    import numpy as np

    rng = np.random.RandomState(seed)
    return sorted(rng.uniform(gamma_min, gamma_max, n).tolist())
