#!/usr/bin/env python3
"""
Tier 1 phase-fix audit (P1.1 + CP snapshot P1.2/P1.3).

- Re-run diag-21-style Gaussian quark holdout on N=15 geometries (seed 21021 prefix).
- Compare legacy vs repaired SVD phase conventions on same optimized Yukawas.
- Report strict survivors, CKM/mass deltas, Jarlskog J and CP phases vs PDG targets.

Falsifier (P1.1): strict rate jumps to >>0% after phase fix alone → revisit refutations.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy.optimize import differential_evolution

from alternative_kernels import compute_yukawas_gaussian, KERNELS
from observables import (
    QUARK_TARGETS,
    QUARK_CP_TARGETS,
    TRAINING_TARGETS,
    compute_quark_observables,
    compute_quark_observables_legacy_phases,
    compute_training_loss,
    compute_holdout_loss,
    compute_ckm_loss,
    svd_reconstruction_error,
)

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "36_tier1_phase_fix_audit.txt"
)

N_GEOMETRIES = 15
GEOM_SEED = 21021
N_SEEDS = 4
OPT_SETTINGS = dict(maxiter=100, popsize=10, tol=1e-6, polish=False)

STRICT_TOLERANCES = {
    "mc": 0.30,
    "Vus": 0.20,
    "Vcb": 0.30,
    "Vub": 0.50,
    "mu": 0.50,
    "md": 0.50,
    "ms": 0.50,
}


def generate_geometries(n: int, seed: int):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        Q = tuple(sorted(rng.choice(range(15), 3, replace=False)))
        U = tuple(sorted(rng.choice(range(15), 3, replace=False)))
        D = tuple(sorted(rng.choice(range(15), 3, replace=False)))
        out.append((Q, U, D))
    return out


def check_strict(obs: dict) -> bool:
    for key, tol in STRICT_TOLERANCES.items():
        t = QUARK_TARGETS[key]
        v = obs.get(key, 0.0)
        if v <= 0 or t <= 0:
            return False
        if abs(v - t) / t > tol:
            return False
    return True


def rel_diff(a: float, b: float) -> float:
    if a == 0 and b == 0:
        return 0.0
    denom = max(abs(a), abs(b), 1e-15)
    return abs(a - b) / denom


def optimize_geometry(Q, U, D, geom_idx: int):
    bounds = KERNELS["gaussian"]["bounds"]

    def objective(theta):
        try:
            Yu, Yd = compute_yukawas_gaussian(Q, U, D, *theta)
            return compute_training_loss(compute_quark_observables(Yu, Yd))
        except Exception:
            return 1000.0

    best = None
    for seed in range(N_SEEDS):
        try:
            res = differential_evolution(
                objective, bounds, seed=seed + geom_idx * 100, **OPT_SETTINGS
            )
        except Exception:
            continue
        if res.fun >= 999:
            continue
        Yu, Yd = compute_yukawas_gaussian(Q, U, D, *res.x)
        obs = compute_quark_observables(Yu, Yd)
        rec = {
            "theta": res.x,
            "Yu": Yu,
            "Yd": Yd,
            "train": compute_training_loss(obs),
            "holdout": compute_holdout_loss(obs),
            "obs": obs,
        }
        if best is None or rec["train"] < best["train"]:
            best = rec
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="N=5 geometries")
    args = parser.parse_args()

    n_geom = 5 if args.smoke else N_GEOMETRIES
    geometries = generate_geometries(n_geom, GEOM_SEED)

    lines = [
        "=" * 72,
        "TIER 1 PHASE-FIX AUDIT (diagnostic 36)",
        "=" * 72,
        f"N={n_geom}, seed={GEOM_SEED}, kernel=gaussian, train/holdout=diag 21",
        "",
    ]

    strict_new = strict_legacy = 0
    solved = 0
    phase_diffs = {k: [] for k in ("Vus", "Vcb", "Vub", "mc", "mu", "md", "ms")}
    cp_keys = ("J_abs", "delta_CKM")
    cp_new = {k: [] for k in cp_keys}
    pareto = []
    recon_errors = []

    print(f"Tier 1 audit: {n_geom} geometries...")
    for gi, (Q, U, D) in enumerate(geometries):
        best = optimize_geometry(Q, U, D, gi)
        if best is None:
            continue
        solved += 1
        Yu, Yd = best["Yu"], best["Yd"]
        obs_new = best["obs"]
        obs_old = compute_quark_observables_legacy_phases(Yu, Yd)

        Uu, Su, Vuh = np.linalg.svd(Yu, full_matrices=False)
        recon_errors.append(svd_reconstruction_error(Uu, Su, Vuh))

        if check_strict(obs_new):
            strict_new += 1
        if check_strict(obs_old):
            strict_legacy += 1

        for k in phase_diffs:
            phase_diffs[k].append(rel_diff(obs_new[k], obs_old[k]))

        for k in cp_keys:
            cp_new[k].append(obs_new[k])

        ckm_l = compute_ckm_loss(obs_new)
        mc_rel = abs(obs_new["mc"] - TRAINING_TARGETS["mc"]) / TRAINING_TARGETS["mc"]
        pareto.append((ckm_l, mc_rel))

    lines.append(f"Geometries solved: {solved}/{n_geom}")
    lines.append(f"Strict survivors (repaired phases): {strict_new}/{solved}")
    lines.append(f"Strict survivors (legacy phases):   {strict_legacy}/{solved}")
    lines.append(
        f"SVD reconstruction error (repaired, max): "
        f"{max(recon_errors) if recon_errors else float('nan'):.2e}"
    )
    lines.append("")
    lines.append("--- Legacy vs repaired relative diff (same Yukawa) ---")
    for k, vals in phase_diffs.items():
        if vals:
            lines.append(
                f"  {k}: max={max(vals):.4f} median={np.median(vals):.4f}"
            )

    if pareto:
        ckm = np.array([p[0] for p in pareto])
        mc = np.array([p[1] for p in pareto])
        corr = float(np.corrcoef(ckm, mc)[0, 1]) if len(ckm) > 2 else float("nan")
        lines.append("")
        lines.append("--- CKM–mc Pareto (repaired pipeline) ---")
        lines.append(f"  ckm_mc_corr: {corr:.4f}")
        lines.append(f"  mc_rel median: {np.median(mc):.4f}")

    lines.append("")
    lines.append("--- CP snapshot (repaired; PDG targets) ---")
    lines.append(f"  PDG |J| ~ {QUARK_CP_TARGETS['J']:.3e}, delta_CKM ~ {QUARK_CP_TARGETS['delta_CKM']:.2f} rad")
    if cp_new["J_abs"]:
        lines.append(
            f"  |J| median={np.median(cp_new['J_abs']):.3e} "
            f"max={max(cp_new['J_abs']):.3e}"
        )
        lines.append(
            f"  delta_CKM median={np.median(cp_new['delta_CKM']):.3f} "
            f"(not optimized; order-of-magnitude audit only)"
        )

    lines.append("")
    lines.append("--- VERDICT (P1.1) ---")
    if strict_new > 0:
        lines.append(
            "  WARNING: strict survivors > 0 — revisit prior 0% quark refutation narrative."
        )
    else:
        lines.append(
            "  PASS: 0% strict survivors — quark refutation stable under repaired SVD phases."
        )
    if strict_legacy != strict_new:
        lines.append(
            f"  Phase convention changed strict count {strict_legacy} → {strict_new}."
        )
    else:
        lines.append("  Strict count unchanged by phase convention on this sample.")

    report = "\n".join(lines)
    print(report)

    if not args.smoke:
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            f.write(report + "\n")
        print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
