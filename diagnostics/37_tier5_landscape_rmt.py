#!/usr/bin/env python3
"""
Tier 5.4 — Optimization landscape spacing vs RMT (meta, not flavor law).

Tests whether local Hessian eigenvalue spacings at quark training minima resemble
GOE (GUE-class) level statistics or Poisson / uncorrelated (no repulsion).

Pre-registered falsifiers:
  - GUE-like: median spacing ratio r > 0.45 AND frac(r < 0.1) < 0.05 (level repulsion)
  - vs Poisson null: same metrics on i.i.d. exponential spacings
  - vs GOE(6) synthetic Hessian eigenvalues

Does NOT claim connection to zeta zeros or CKM (see diag 33, adversarial review).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.optimize import differential_evolution
from typing import Callable, List, Tuple

from alternative_kernels import compute_yukawas_gaussian
from observables import compute_quark_observables, compute_training_loss
from phenomenology_utils import generate_quark_geometries

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "37_tier5_landscape_rmt.txt"
)

N_GEOMETRIES = 60
GEOM_SEED = 37037
N_SEEDS = 3
OPT = dict(maxiter=70, popsize=8, tol=1e-5, polish=False)

GAUSSIAN_BOUNDS = [
    (0.5, 6.0),
    (0.1, 2.0),
    (0.0, 2 * np.pi),
    (1.0, 5.0),
    (0.01, 0.5),
    (0.01, 0.5),
]

# Pre-registered (Tier 5.4): unfolded NN spacings (mean 1 per spectrum)
SMALL_SPACING = 0.1
GUE_SMALL_FRAC_MAX = 0.03
POISSON_SMALL_FRAC_MIN = 0.06


def unfolded_spacings(evals: np.ndarray) -> np.ndarray:
    """Nearest-neighbor gaps normalized to unit mean (local unfolding)."""
    v = np.sort(np.asarray(evals, dtype=float))
    v = v[v > 1e-8]
    if len(v) < 3:
        return np.array([])
    gaps = np.diff(v)
    gaps = gaps[gaps > 1e-12]
    if len(gaps) == 0:
        return np.array([])
    return gaps / np.mean(gaps)


def hessian_eigenvalues(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    eps: float = 2e-4,
) -> np.ndarray:
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    f0 = objective(x0)
    grad = np.zeros(n)
    H = np.zeros((n, n))
    for i in range(n):
        xp = x0.copy()
        xm = x0.copy()
        xp[i] += eps
        xm[i] -= eps
        fp = objective(xp)
        fm = objective(xm)
        grad[i] = (fp - fm) / (2 * eps)
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
    w = np.linalg.eigvalsh(H)
    return w


def goe_unfolded_pool(n: int, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_samples):
        M = rng.standard_normal((n, n))
        A = (M + M.T) / 2.0
        out.extend(unfolded_spacings(np.linalg.eigvalsh(A)).tolist())
    return np.array(out)


def poisson_unfolded_pool(n_spacings: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.exponential(1.0, n_spacings)


def spacing_stats(spacings: np.ndarray) -> dict:
    if len(spacings) == 0:
        return {"n": 0, "median_s": float("nan"), "frac_small": float("nan")}
    return {
        "n": len(spacings),
        "median_s": float(np.median(spacings)),
        "mean_s": float(np.mean(spacings)),
        "frac_small": float(np.mean(spacings < SMALL_SPACING)),
    }


def optimize_geometry(Q, U, D, geom_idx: int) -> Tuple[np.ndarray, float]:
    def objective(theta):
        try:
            Yu, Yd = compute_yukawas_gaussian(Q, U, D, *theta)
            return compute_training_loss(compute_quark_observables(Yu, Yd))
        except Exception:
            return 1000.0

    best_x, best_f = None, np.inf
    for seed in range(N_SEEDS):
        try:
            res = differential_evolution(
                objective,
                GAUSSIAN_BOUNDS,
                seed=seed + geom_idx * 17,
                **OPT,
            )
        except Exception:
            continue
        if res.fun < best_f and res.fun < 999:
            best_f, best_x = res.fun, res.x
    return best_x, best_f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    n_geom = 8 if args.smoke else N_GEOMETRIES
    geometries = generate_quark_geometries(n_geom, GEOM_SEED)

    pooled_spacings = []
    train_losses = []
    n_hessian = 0

    print(f"Tier 5.4 landscape RMT: {n_geom} geometries (Hessian per solved geom)...")
    for gi, (Q, U, D) in enumerate(geometries):
        x0, train = optimize_geometry(Q, U, D, gi)
        if x0 is None:
            continue
        train_losses.append(train)

        def obj(theta):
            Yu, Yd = compute_yukawas_gaussian(Q, U, D, *theta)
            return compute_training_loss(compute_quark_observables(Yu, Yd))

        try:
            w = hessian_eigenvalues(obj, x0)
            s = unfolded_spacings(w)
            if len(s) > 0:
                pooled_spacings.extend(s.tolist())
                n_hessian += 1
        except Exception:
            continue
        if (gi + 1) % 10 == 0:
            print(f"  {gi + 1}/{n_geom} done, hessian ok: {n_hessian}")

    emp = spacing_stats(np.array(pooled_spacings))
    n_rat = max(emp["n"], 80)
    goe = spacing_stats(goe_unfolded_pool(6, n_samples=max(30, n_geom), seed=37038))
    poi = spacing_stats(poisson_unfolded_pool(n_rat, seed=37039))

    loss_emp = spacing_stats(unfolded_spacings(np.array(train_losses)))

    gue_like = (
        emp["n"] > 30
        and emp["frac_small"] <= GUE_SMALL_FRAC_MAX
        and abs(emp["frac_small"] - goe["frac_small"])
        < abs(emp["frac_small"] - poi["frac_small"])
    )
    poisson_like = (
        emp["frac_small"] >= POISSON_SMALL_FRAC_MIN
        or abs(emp["frac_small"] - poi["frac_small"])
        <= abs(emp["frac_small"] - goe["frac_small"])
    )

    lines = [
        "=" * 72,
        "TIER 5.4 LANDSCAPE RMT (diagnostic 37)",
        "=" * 72,
        f"N_geom={n_geom}, seed={GEOM_SEED}, Gaussian train-loss minima",
        f"Hessian unfolded NN spacings pooled (n_geom with Hessian={n_hessian})",
        "",
        "--- Empirical (Hessian at minima) ---",
        f"  n={emp['n']}, median_s={emp['median_s']:.4f}, "
        f"frac(s<{SMALL_SPACING})={emp['frac_small']:.4f}",
        "",
        "--- Nulls ---",
        f"  GOE(6) synthetic: median_s={goe['median_s']:.4f}, frac_small={goe['frac_small']:.4f}",
        f"  Poisson i.i.d.:  median_s={poi['median_s']:.4f}, frac_small={poi['frac_small']:.4f}",
        "",
        "--- Ensemble train-loss spacings (geometry index; not local RMT) ---",
        f"  n={loss_emp['n']}, median_s={loss_emp['median_s']:.4f}, "
        f"frac_small={loss_emp['frac_small']:.4f}",
        "",
        "--- Pre-registered ---",
        f"  GUE-like: frac_small<={GUE_SMALL_FRAC_MAX} and closer to GOE than Poisson",
        f"  Poisson falsifier: frac_small>={POISSON_SMALL_FRAC_MIN} or closer to Poisson",
        f"  gue_like={gue_like}, poisson_like={poisson_like}",
        "",
        "--- VERDICT ---",
    ]

    if gue_like and not poisson_like:
        verdict = "gue_like"
        lines.append(
            "  Local Hessian spacings show GOE-style repulsion (meta landscape only)."
        )
        lines.append("  NOT a flavor/RH/zeta claim — optimization geometry only.")
    else:
        verdict = "poisson_like"
        lines.append(
            "  FAIL GUE universality — spacings Poisson-like or not closer to GOE than Poisson."
        )
        lines.append(
            "  Do not market as quantum chaos / GUE link to Yukawa or zeros."
        )

    report = "\n".join(lines)
    print(report)

    if not args.smoke:
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            f.write(report + "\n")
            f.write(f"verdict: {verdict}\n")
            f.write("no_flavor_connection: true\n")
        print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
