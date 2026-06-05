#!/usr/bin/env python3
"""
N1 — Joint loss landscape cartography (quark vs neutrino).

Maps geometry-conditioned joint objectives at DE minima:
  best loss, Hessian spectrum, gradient norm, random-direction ruggedness.

Pre-registered (N1 falsifier):
  Sectors are NOT structurally differentiated if < 2 of 5 metrics show
  Mann-Whitney p < 0.05 (two-sided) between quark and neutrino pools.

Positive N1 signal:
  >= 3 metrics differ significantly AND neutrino strict rate exceeds quark
  by >= 15 percentage points on the same N (Gaussian kernel, joint objectives).

Does not claim a fundamental mechanism — compares optimization geometry only.
"""

import argparse
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.optimize import differential_evolution
from scipy import stats

from alternative_kernels import compute_yukawas_gaussian
from kernel import compute_yukawa_matrix
from observables import (
    compute_ckm_loss,
    compute_mass_loss,
    compute_neutrino_joint_loss,
    compute_neutrino_mass_loss,
    compute_neutrino_observables,
    compute_pmns_loss,
    compute_quark_observables,
    NEUTRINO_MASS_TARGETS,
    NEUTRINO_TARGETS,
    QUARK_TARGETS,
)
from phenomenology_utils import generate_neutrino_geometries, generate_quark_geometries

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "39_joint_loss_landscape_cartography.txt"
)

N_GEOMETRIES = 50
QUARK_GEOM_SEED = 39139
NU_GEOM_SEED = 39140
N_SEEDS = 3
OPT = dict(maxiter=80, popsize=10, tol=1e-5, polish=False)

QUARK_BOUNDS = [
    (0.5, 6.0),
    (0.1, 2.0),
    (0.0, 2 * np.pi),
    (1.0, 5.0),
    (0.01, 0.5),
    (0.01, 0.5),
]

NU_BOUNDS = QUARK_BOUNDS + [(0.45, 0.75)]

QUARK_STRICT = {
    "mc": 0.30,
    "Vus": 0.20,
    "Vcb": 0.30,
    "Vub": 0.50,
    "mu": 0.50,
    "md": 0.50,
    "ms": 0.50,
}

NU_PMNS_STRICT = {"theta12": 0.15, "theta23": 0.15, "theta13": 0.20}
NU_MASS_STRICT = {"dm21": 0.30, "dm31": 0.30}

METRIC_KEYS = [
    "best_loss",
    "log10_condition",
    "frac_negative_hessian",
    "ruggedness",
    "grad_norm",
]

MIN_SIGNIFICANT_METRICS = 2
PURSUE_MIN_METRICS = 3
STRICT_RATE_GAP = 0.15
P_VALUE_MAX = 0.05
N_RUGGED_DIRS = 16
RUGGED_STEP = 0.08


def hessian_eigenvalues(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    eps: float = 2e-4,
) -> np.ndarray:
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            xpp = x0.copy()
            xpm = x0.copy()
            xmp = x0.copy()
            xmm = x0.copy()
            xpp[i] += eps
            xpp[j] += eps
            xpm[i] += eps
            xpm[j] -= eps
            xmp[i] -= eps
            xmp[j] += eps
            xmm[i] -= eps
            xmm[j] -= eps
            val = (
                objective(xpp) - objective(xpm) - objective(xmp) + objective(xmm)
            ) / (4 * eps * eps)
            H[i, j] = val
            if j != i:
                H[j, i] = val
    return np.linalg.eigvalsh(H)


def grad_norm(objective: Callable[[np.ndarray], float], x0: np.ndarray, eps: float = 2e-4) -> float:
    x0 = np.asarray(x0, dtype=float)
    g = np.zeros(len(x0))
    for i in range(len(x0)):
        xp = x0.copy()
        xm = x0.copy()
        xp[i] += eps
        xm[i] -= eps
        g[i] = (objective(xp) - objective(xm)) / (2 * eps)
    return float(np.linalg.norm(g))


def ruggedness(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: List[Tuple[float, float]],
    n_dirs: int,
    step: float,
    seed: int,
) -> float:
    rng = np.random.RandomState(seed)
    x0 = np.asarray(x0, dtype=float)
    f0 = objective(x0)
    spans = np.array([hi - lo for lo, hi in bounds])
    bumps = []
    for _ in range(n_dirs):
        v = rng.standard_normal(len(x0))
        v = v / (np.linalg.norm(v) + 1e-15)
        delta = step * spans * v
        f1 = objective(x0 + delta)
        bumps.append(abs(f1 - f0) / (step + 1e-15))
    return float(np.mean(bumps))


def compute_joint_quark_loss(obs: Dict[str, float]) -> float:
    l_mass = compute_mass_loss(obs)
    l_ckm = compute_ckm_loss(obs)
    l_md = 2.0 * (np.log(0.002 / obs["md"])) ** 2 if obs["md"] < 0.002 else 0.0
    l_mu = 0.5 * (np.log(0.0005 / obs["mu"])) ** 2 if obs["mu"] < 0.0005 else 0.0
    return float(l_mass + 5.0 * l_ckm + l_md + l_mu)


def quark_strict(obs: Dict[str, float]) -> bool:
    for key, tol in QUARK_STRICT.items():
        t, v = QUARK_TARGETS[key], obs[key]
        if v <= 0 or t <= 0 or abs(v - t) / t > tol:
            return False
    return True


def neutrino_strict(obs: Dict[str, float]) -> bool:
    for key, tol in NU_PMNS_STRICT.items():
        t, v = NEUTRINO_TARGETS[key], obs[key]
        if v <= 0 or t <= 0 or abs(v - t) / t > tol:
            return False
    for key, tol in NU_MASS_STRICT.items():
        t, v = NEUTRINO_MASS_TARGETS[key], obs[key]
        if v <= 0 or t <= 0 or abs(v - t) / t > tol:
            return False
    return True


def optimize_de(
    objective: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    geom_idx: int,
) -> Tuple[Optional[np.ndarray], float]:
    best_x, best_f = None, np.inf
    for s in range(N_SEEDS):
        try:
            res = differential_evolution(
                objective,
                bounds,
                seed=s + geom_idx * 19,
                **OPT,
            )
        except Exception:
            continue
        if res.fun < best_f and res.fun < 999:
            best_f, best_x = res.fun, res.x
    return best_x, best_f


def cartograph_quark(geometries: List[Tuple]) -> List[Dict]:
    rows = []
    for gi, (Q, U, D) in enumerate(geometries):

        def objective(theta):
            try:
                Yu, Yd = compute_yukawas_gaussian(Q, U, D, *theta)
                obs = compute_quark_observables(Yu, Yd)
                if obs["mc"] < 0.01 or obs["mc"] > 500:
                    return 1000.0
                return compute_joint_quark_loss(obs)
            except Exception:
                return 1000.0

        x0, loss = optimize_de(objective, QUARK_BOUNDS, gi)
        if x0 is None:
            continue
        Yu, Yd = compute_yukawas_gaussian(Q, U, D, *x0)
        obs = compute_quark_observables(Yu, Yd)
        w = hessian_eigenvalues(objective, x0)
        lam_pos = w[w > 1e-8]
        cond = (
            float(lam_pos.max() / lam_pos.min())
            if len(lam_pos) >= 2
            else float("nan")
        )
        rows.append(
            {
                "sector": "quark",
                "best_loss": loss,
                "mass_part": compute_mass_loss(obs),
                "mix_part": compute_ckm_loss(obs),
                "strict": quark_strict(obs),
                "log10_condition": np.log10(cond) if cond > 0 and np.isfinite(cond) else np.nan,
                "frac_negative_hessian": float(np.mean(w < -1e-6)),
                "ruggedness": ruggedness(
                    objective, x0, QUARK_BOUNDS, N_RUGGED_DIRS, RUGGED_STEP, 39000 + gi
                ),
                "grad_norm": grad_norm(objective, x0),
            }
        )
        if (gi + 1) % 10 == 0:
            print(f"  quark {gi + 1}/{len(geometries)}")
    return rows


def cartograph_neutrino(geometries: List[Tuple]) -> List[Dict]:
    rows = []
    for gi, (L, N) in enumerate(geometries):

        def objective(theta):
            try:
                sigma, k, alpha, eta, eps_nu, eps_e, g_env = theta
                Ynu = compute_yukawa_matrix(L, N, sigma * g_env, k, alpha, eta, eps_nu)
                Ye = compute_yukawa_matrix(L, N, sigma, k, alpha, eta, eps_e)
                obs = compute_neutrino_observables(Ynu, Ye)
                if obs["theta23"] < 0.01:
                    return 1000.0
                return compute_neutrino_joint_loss(obs)
            except Exception:
                return 1000.0

        x0, loss = optimize_de(objective, NU_BOUNDS, gi)
        if x0 is None:
            continue
        sigma, k, alpha, eta, eps_nu, eps_e, g_env = x0
        Ynu = compute_yukawa_matrix(L, N, sigma * g_env, k, alpha, eta, eps_nu)
        Ye = compute_yukawa_matrix(L, N, sigma, k, alpha, eta, eps_e)
        obs = compute_neutrino_observables(Ynu, Ye)
        w = hessian_eigenvalues(objective, x0)
        lam_pos = w[w > 1e-8]
        cond = (
            float(lam_pos.max() / lam_pos.min())
            if len(lam_pos) >= 2
            else float("nan")
        )
        rows.append(
            {
                "sector": "neutrino",
                "best_loss": loss,
                "mass_part": compute_neutrino_mass_loss(obs),
                "mix_part": compute_pmns_loss(obs),
                "strict": neutrino_strict(obs),
                "log10_condition": np.log10(cond) if cond > 0 and np.isfinite(cond) else np.nan,
                "frac_negative_hessian": float(np.mean(w < -1e-6)),
                "ruggedness": ruggedness(
                    objective, x0, NU_BOUNDS, N_RUGGED_DIRS, RUGGED_STEP, 49000 + gi
                ),
                "grad_norm": grad_norm(objective, x0),
            }
        )
        if (gi + 1) % 10 == 0:
            print(f"  neutrino {gi + 1}/{len(geometries)}")
    return rows


def mannwhitney(a: List[float], b: List[float]) -> Tuple[float, float]:
    a = [x for x in a if np.isfinite(x)]
    b = [x for x in b if np.isfinite(x)]
    if len(a) < 8 or len(b) < 8:
        return float("nan"), float("nan")
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(u.statistic), float(u.pvalue)


def pool(rows: List[Dict], key: str) -> List[float]:
    return [float(r[key]) for r in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    n_geom = 10 if args.smoke else N_GEOMETRIES
    print(f"N1 landscape cartography: N={n_geom} per sector (Gaussian, joint loss)...")

    q_geoms = generate_quark_geometries(n_geom, QUARK_GEOM_SEED)
    n_geoms = generate_neutrino_geometries(n_geom, NU_GEOM_SEED)

    print("Quark joint landscape...")
    q_rows = cartograph_quark(q_geoms)
    print("Neutrino joint landscape...")
    n_rows = cartograph_neutrino(n_geoms)

    q_strict = np.mean([r["strict"] for r in q_rows]) if q_rows else 0.0
    n_strict = np.mean([r["strict"] for r in n_rows]) if n_rows else 0.0

    comparisons = []
    n_sig = 0
    for key in METRIC_KEYS:
        _, p = mannwhitney(pool(q_rows, key), pool(n_rows, key))
        sig = p < P_VALUE_MAX if np.isfinite(p) else False
        if sig:
            n_sig += 1
        comparisons.append((key, p, sig))

    differentiated = n_sig >= PURSUE_MIN_METRICS and (n_strict - q_strict) >= STRICT_RATE_GAP
    falsifier_fail = n_sig < MIN_SIGNIFICANT_METRICS

    lines = [
        "=" * 72,
        "N1 JOINT LOSS LANDSCAPE CARTOGRAPHY (diagnostic 39)",
        "=" * 72,
        f"N_geom={n_geom} per sector; quark seed {QUARK_GEOM_SEED}, nu seed {NU_GEOM_SEED}",
        "Kernel: Gaussian (quark); Gaussian+g_env (neutrino, diag 28 protocol)",
        "Objectives: joint quark (diag 27) vs joint PMNS+dm^2 (diag 28)",
        "",
        f"  quark solved: {len(q_rows)}, strict rate: {100*q_strict:.1f}%",
        f"  neutrino solved: {len(n_rows)}, strict rate: {100*n_strict:.1f}%",
        "",
        "--- Medians ---",
    ]
    for key in METRIC_KEYS + ["mass_part", "mix_part"]:
        q_med = float(np.median(pool(q_rows, key))) if q_rows else float("nan")
        n_med = float(np.median(pool(n_rows, key))) if n_rows else float("nan")
        lines.append(f"  {key}: quark={q_med:.4g}  neutrino={n_med:.4g}")

    lines.append("")
    lines.append("--- Mann-Whitney (quark vs neutrino) ---")
    for key, p, sig in comparisons:
        lines.append(f"  {key}: p={p:.4e}  significant={sig}")

    lines.extend(
        [
            "",
            "--- Pre-registered N1 ---",
            f"  significant metrics (p<{P_VALUE_MAX}): {n_sig} / {len(METRIC_KEYS)}",
            f"  falsifier (indistinguishable): {falsifier_fail}",
            f"  differentiated landscape + strict gap: {differentiated}",
            "",
            "--- VERDICT ---",
        ]
    )

    if falsifier_fail:
        verdict = "indistinguishable"
        lines.append(
            "  N1 falsifier NOT rejected — landscape metrics too similar across sectors."
        )
        lines.append("  Sector success gap may be objective/target noise, not geometry of loss.")
    elif differentiated:
        verdict = "differentiated"
        lines.append(
            "  N1 POSITIVE — neutrino joint landscape differs from quark (>=3 metrics, strict gap)."
        )
        lines.append(
            "  Suggests asymmetric optimization geometry; NOT a universal-kernel proof."
        )
    else:
        verdict = "mixed"
        lines.append(
            "  MIXED — some metric separation but strict gap or metric count below pursue bar."
        )

    report = "\n".join(lines)
    print(report)

    if not args.smoke:
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            f.write(report + "\n")
            f.write(f"verdict: {verdict}\n")
            f.write(f"n_significant_metrics: {n_sig}\n")
        print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
