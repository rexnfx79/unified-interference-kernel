"""Shared helpers for lepton/neutrino phenomenology diagnostics."""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Tuple

import numpy as np


class JointThreeSectorGeometry(NamedTuple):
    """Shared left positions L=Q across lepton, neutrino, and quark sectors."""

    index: int
    L: Tuple[int, ...]
    E: Tuple[int, ...]
    N: Tuple[int, ...]
    U: Tuple[int, ...]
    D: Tuple[int, ...]

    @property
    def lepton(self) -> Tuple[Tuple, Tuple]:
        return (self.L, self.E)

    @property
    def neutrino(self) -> Tuple[Tuple, Tuple]:
        return (self.L, self.N)

    @property
    def quark(self) -> Tuple[Tuple, Tuple, Tuple]:
        return (self.L, self.U, self.D)

# Legacy range-based survivors (scripts/04_analyze_results.py)
LEGACY_LEPTON_RANGES = {
    "m_e": (0.0004, 0.0006),
    "m_mu": (0.09, 0.12),
    "m_tau": (1.6, 2.0),
}

LEGACY_NEUTRINO_RANGES = {
    "theta12": (0.5, 0.7),
    "theta23": (0.6, 1.0),
    "theta13": (0.10, 0.20),
}


def generate_quark_geometries(n_geom: int, seed: int) -> List[Tuple[Tuple, Tuple, Tuple]]:
    """Unique (Q, U, D) sorted triples — phenomenology convention (diag 21/32)."""
    rng = np.random.RandomState(seed)
    coords = list(range(15))
    seen = set()
    geometries: List[Tuple[Tuple, Tuple, Tuple]] = []
    attempts = 0
    max_attempts = max(n_geom * 50, 1000)
    while len(geometries) < n_geom and attempts < max_attempts:
        Q = tuple(sorted(rng.choice(coords, 3, replace=False)))
        U = tuple(sorted(rng.choice(coords, 3, replace=False)))
        D = tuple(sorted(rng.choice(coords, 3, replace=False)))
        key = (Q, U, D)
        if key not in seen:
            seen.add(key)
            geometries.append(key)
        attempts += 1
    return geometries


def generate_lepton_geometries(n_geom: int, seed: int) -> List[Tuple[Tuple, Tuple]]:
    """Unique (L, E) triple pairs sampled from a fixed coordinate grid."""
    rng = np.random.RandomState(seed)
    coords = list(range(15))
    seen = set()
    geometries: List[Tuple[Tuple, Tuple]] = []
    attempts = 0
    max_attempts = max(n_geom * 50, 1000)
    while len(geometries) < n_geom and attempts < max_attempts:
        L = tuple(sorted(rng.choice(coords, 3, replace=False)))
        E = tuple(sorted(rng.choice(coords, 3, replace=False)))
        key = (L, E)
        if key not in seen:
            seen.add(key)
            geometries.append(key)
        attempts += 1
    return geometries


def generate_joint_three_sector_geometries(
    n_geom: int, seed: int
) -> List[JointThreeSectorGeometry]:
    """Joint corpus: shared L (= quark Q) with independent right-handed triples."""
    rng = np.random.RandomState(seed)
    coords = list(range(15))
    seen = set()
    geometries: List[JointThreeSectorGeometry] = []
    attempts = 0
    max_attempts = max(n_geom * 50, 1000)
    while len(geometries) < n_geom and attempts < max_attempts:
        L = tuple(sorted(rng.choice(coords, 3, replace=False)))
        E = tuple(sorted(rng.choice(coords, 3, replace=False)))
        N = tuple(sorted(rng.choice(coords, 3, replace=False)))
        U = tuple(sorted(rng.choice(coords, 3, replace=False)))
        D = tuple(sorted(rng.choice(coords, 3, replace=False)))
        key = (L, E, N, U, D)
        if key not in seen:
            seen.add(key)
            geometries.append(
                JointThreeSectorGeometry(
                    index=len(geometries),
                    L=L,
                    E=E,
                    N=N,
                    U=U,
                    D=D,
                )
            )
        attempts += 1
    return geometries


def generate_neutrino_geometries(n_geom: int, seed: int) -> List[Tuple[Tuple, Tuple]]:
    """Unique (L, N) triple pairs sampled from a fixed coordinate grid."""
    rng = np.random.RandomState(seed)
    coords = list(range(15))
    seen = set()
    geometries: List[Tuple[Tuple, Tuple]] = []
    attempts = 0
    max_attempts = max(n_geom * 50, 1000)
    while len(geometries) < n_geom and attempts < max_attempts:
        L = tuple(sorted(rng.choice(coords, 3, replace=False)))
        N = tuple(sorted(rng.choice(coords, 3, replace=False)))
        key = (L, N)
        if key not in seen:
            seen.add(key)
            geometries.append(key)
        attempts += 1
    return geometries


def check_legacy_lepton(rec: Dict) -> bool:
    for key, (lo, hi) in LEGACY_LEPTON_RANGES.items():
        v = rec[key]
        if not (lo < v < hi):
            return False
    return True


def check_legacy_neutrino(rec: Dict) -> bool:
    for key, (lo, hi) in LEGACY_NEUTRINO_RANGES.items():
        v = rec[key]
        if not (lo < v < hi):
            return False
    return True


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def bootstrap_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 800,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, float]:
    """Bootstrap percentile CI for Pearson r."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 5:
        r = safe_corr(x, y)
        return {"r": r, "ci_lo": float("nan"), "ci_hi": float("nan"), "n_boot": 0}

    rng = np.random.RandomState(seed)
    corrs: List[float] = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        r = safe_corr(x[idx], y[idx])
        if np.isfinite(r):
            corrs.append(r)

    if not corrs:
        r = safe_corr(x, y)
        return {"r": r, "ci_lo": float("nan"), "ci_hi": float("nan"), "n_boot": 0}

    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(corrs, 100 * alpha))
    hi = float(np.percentile(corrs, 100 * (1 - alpha)))
    return {
        "r": safe_corr(x, y),
        "ci_lo": lo,
        "ci_hi": hi,
        "n_boot": len(corrs),
    }


def pareto_nondominated(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Nondominated set minimizing both coordinates."""
    nd: List[Tuple[float, float]] = []
    for i, (x, y) in enumerate(points):
        dominated = False
        for j, (x2, y2) in enumerate(points):
            if j == i:
                continue
            if x2 <= x and y2 <= y and (x2 < x or y2 < y):
                dominated = True
                break
        if not dominated:
            nd.append((x, y))
    return sorted(nd, key=lambda p: p[0])
