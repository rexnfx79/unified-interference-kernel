#!/usr/bin/env python3
"""
Tier 3 theory bridges (diagnostic 33)

Track A — Split-fermion → interference kernel (extend diag 10):
  - N geometries, Yu + Yd width fits at fixed and optimized kernel params
  - Falsifier: stable w/σ across geometries AND geometry predicts w/σ (R² > 0.5)

Track B — Path D sanity (no flavor numerology):
  - 3×3 spacing vs GUE: structurally insufficient (2 spacings)
  - Yukawa eigenvalue spacing vs random: no special structure
  - Explicit-formula prime side: educational only (diag 14)

Path A (QIT→flavor): closed (diag 12–19) — not re-run.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.optimize import differential_evolution

from kernel import compute_quark_yukawas
from observables import compute_training_loss, compute_quark_observables
from split_fermion_overlap import fit_width_to_kernel, positions_to_legacy
from phenomenology_utils import generate_quark_geometries

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "33_tier3_theory_bridges.txt"
)

N_GEOMETRIES = 50
GEOM_SEED = 33033
DEFAULT_PARAMS = dict(sigma=4.0, k=1.4, alpha=2.5, eta=2.0, eps_u=0.15, eps_d=0.15)

# Falsifier thresholds (pre-registered)
W_SIGMA_REL_SPREAD_MAX = 0.15
GEOM_PREDICT_R2_MIN = 0.50
MAG_CORR_MIN = 0.99

OPT_SETTINGS = {
    "maxiter": 80,
    "popsize": 10,
    "tol": 1e-5,
    "polish": False,
}


def geometry_features(Q, U, D) -> np.ndarray:
    """Simple spread features for predicting w/σ."""
    coords = list(Q) + list(U) + list(D)
    arr = np.array(coords, dtype=float)
    return np.array(
        [
            arr.max() - arr.min(),
            arr.mean(),
            arr.std(),
            float(max(U) - min(Q)),
            float(max(D) - min(Q)),
        ]
    )


def fit_overlap_batch(
    geometries: list,
    sigma: float,
    k: float,
    alpha: float,
    eta: float,
    eps_u: float,
    eps_d: float,
    label: str,
) -> list:
    rows = []
    for idx, (Q, U, D) in enumerate(geometries):
        L, R_u, R_d = positions_to_legacy(Q, U, D)
        for sector, R, eps in [("Yu", R_u, eps_u), ("Yd", R_d, eps_d)]:
            fit = fit_width_to_kernel(L, R, sigma, k, alpha, eta, eps)
            fit["geom_idx"] = idx
            fit["sector"] = sector
            fit["param_set"] = label
            fit["features"] = geometry_features(Q, U, D)
            rows.append(fit)
    return rows


def predictability_r2(rows: list) -> float:
    """R² of linear fit: features → w/sigma across rows."""
    if len(rows) < 6:
        return float("nan")
    X = np.array([r["features"] for r in rows])
    y = np.array([r["sigma_relation"] for r in rows])
    X = np.column_stack([np.ones(len(X)), X])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def optimize_params_single_geometry(Q, U, D) -> dict:
    L, R_u, R_d = positions_to_legacy(Q, U, D)
    bounds = [
        (0.5, 6.0),
        (0.1, 2.5),
        (0.0, 2 * np.pi),
        (1.0, 5.0),
        (0.01, 0.5),
        (0.01, 0.5),
    ]

    def objective(theta):
        try:
            Yu, Yd = compute_quark_yukawas(L, R_u, R_d, *theta)
            return compute_training_loss(compute_quark_observables(Yu, Yd))
        except Exception:
            return 1000.0

    r = differential_evolution(objective, bounds, seed=GEOM_SEED, **OPT_SETTINGS)
    s, k, a, e, eu, ed = r.x
    return dict(sigma=s, k=k, alpha=a, eta=e, eps_u=eu, eps_d=ed, train_loss=r.fun)


def path_d_spacing_audit() -> dict:
    """Document that 3×3 spacing cannot test GUE; Yukawa vs random."""
    rng = np.random.RandomState(42)
    yukawa_spacings = []
    random_spacings = []
    for _ in range(200):
        Q = tuple(sorted(rng.choice(15, 3, replace=False)))
        U = tuple(sorted(rng.choice(15, 3, replace=False)))
        D = tuple(sorted(rng.choice(15, 3, replace=False)))
        L, Ru, Rd = positions_to_legacy(Q, U, D)
        Yu, _ = compute_quark_yukawas(L, Ru, Rd, **DEFAULT_PARAMS)
        ev = np.sort(np.abs(np.linalg.eigvalsh(Yu @ Yu.conj().T)))[::-1]
        if ev[0] > 1e-15:
            yukawa_spacings.append((ev[0] - ev[1]) / ev[0])
            yukawa_spacings.append((ev[1] - ev[2]) / ev[0])
        R = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
        evr = np.sort(np.abs(np.linalg.eigvalsh(R @ R.conj().T)))[::-1]
        if evr[0] > 1e-15:
            random_spacings.append((evr[0] - evr[1]) / evr[0])
            random_spacings.append((evr[1] - evr[2]) / evr[0])
    return {
        "n_spacing_samples": len(yukawa_spacings),
        "yukawa_spacing_mean": float(np.mean(yukawa_spacings)),
        "random_spacing_mean": float(np.mean(random_spacings)),
        "gue_testable_at_3x3": False,
        "note": "Only 2 unfolded spacings per matrix; GUE comparison not statistically meaningful.",
    }


def format_report(
    fixed_rows: list,
    opt_rows: list,
    path_d: dict,
    opt_params: dict,
) -> str:
    def summarize(rows: list, title: str) -> list:
        lines = [f"--- {title} ---"]
        ratios = [r["sigma_relation"] for r in rows]
        corrs = [r["magnitude_correlation"] for r in rows]
        phases = [r["mean_phase_error_rad"] for r in rows]
        r_mean = float(np.mean(ratios))
        r_std = float(np.std(ratios))
        rel_spread = r_std / r_mean if r_mean > 0 else float("inf")
        r2 = predictability_r2(rows)
        lines.append(f"  N fits: {len(rows)}")
        lines.append(f"  w/σ mean={r_mean:.4f} std={r_std:.4f} rel_spread={rel_spread:.4f}")
        lines.append(f"  |corr| mag: min={min(corrs):.4f} median={np.median(corrs):.4f}")
        lines.append(f"  phase err max={max(phases):.2e}")
        lines.append(f"  geometry → w/σ linear R²={r2:.4f}")
        stable = rel_spread < W_SIGMA_REL_SPREAD_MAX
        predict = r2 >= GEOM_PREDICT_R2_MIN
        mag_ok = min(corrs) >= MAG_CORR_MIN
        lines.append(f"  stable w/σ (<{W_SIGMA_REL_SPREAD_MAX} rel spread): {stable}")
        lines.append(f"  predictable w/σ (R²≥{GEOM_PREDICT_R2_MIN}): {predict}")
        lines.append(f"  all mag r≥{MAG_CORR_MIN}: {mag_ok}")
        return lines, stable, predict, mag_ok

    lines = [
        "=" * 78,
        "TIER 3 THEORY BRIDGES (diagnostic 33)",
        "=" * 78,
        "",
        f"Geometries: N={N_GEOMETRIES}, seed={GEOM_SEED}",
        f"DEFAULT kernel params: {DEFAULT_PARAMS}",
        "",
        "Track A falsifiers:",
        f"  w/σ rel spread < {W_SIGMA_REL_SPREAD_MAX}",
        f"  geometry predicts w/σ with R² ≥ {GEOM_PREDICT_R2_MIN}",
        f"  magnitude correlation ≥ {MAG_CORR_MIN} all fits",
        "",
    ]
    l1, s1, p1, m1 = summarize(fixed_rows, "TRACK A — fixed params (diag 10 style)")
    lines.extend(l1)
    lines.append("")
    lines.append(
        f"  One-geometry optimized params (train loss={opt_params.get('train_loss', float('nan')):.4f}): "
        f"σ={opt_params.get('sigma', 0):.3f} k={opt_params.get('k', 0):.3f}"
    )
    l2, s2, p2, m2 = summarize(opt_rows, "TRACK A — same geom, optimized σ,k,α,η,ε")
    lines.extend(l2)

    lines.extend(
        [
            "",
            "--- TRACK B — Path D (no flavor link) ---",
            f"  GUE testable at 3×3: {path_d['gue_testable_at_3x3']}",
            f"  {path_d['note']}",
            f"  Yukawa spacing mean={path_d['yukawa_spacing_mean']:.4f} "
            f"vs random {path_d['random_spacing_mean']:.4f} (n={path_d['n_spacing_samples']})",
            "  Explicit formula / primes: see diagnostics/14_explicit_formula_numerics.py (educational)",
            "  Flavor numerology: REFUTED — [[why-not-zeta-flavor-numerology]]",
            "",
            "--- TRACK A — Path A (QIT→flavor) ---",
            "  CLOSED — diagnostics 12–19; not re-run.",
            "",
            "--- VERDICT ---",
        ]
    )
    bridge_ok = s1 and p1 and m1
    if bridge_ok:
        lines.append(
            "  Split-fermion overlap reproduces kernel magnitudes with stable w/σ and "
            "geometry-predictable width — pursue derivation write-up only."
        )
    else:
        lines.append(
            "  Split-fermion → kernel remains POST-HOC fit (w/σ unstable or not predictable "
            "from geometry) — Tier 3 Track A does not justify mechanism claims."
        )
    lines.append(
        "  Path D: no operational hook to Yukawa/CKM in this repo; remain watch-only."
    )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="N=8 geometries")
    args = parser.parse_args()

    global N_GEOMETRIES
    if args.smoke:
        N_GEOMETRIES = 8

    geometries = generate_quark_geometries(N_GEOMETRIES, GEOM_SEED)
    print(f"Tier 3: N={len(geometries)} geometries...")

    fixed_rows = fit_overlap_batch(
        geometries, label="fixed", **DEFAULT_PARAMS
    )

    Q0, U0, D0 = geometries[0]
    opt_params = optimize_params_single_geometry(Q0, U0, D0)
    kernel_params = {k: opt_params[k] for k in ("sigma", "k", "alpha", "eta", "eps_u", "eps_d")}
    opt_rows = fit_overlap_batch(geometries, label="optimized", **kernel_params)

    path_d = path_d_spacing_audit()
    report = format_report(fixed_rows, opt_rows, path_d, opt_params)
    print(report)

    if not args.smoke:
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            f.write(report)
        print(f"Saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
