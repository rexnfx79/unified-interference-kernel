#!/usr/bin/env python3
"""
Neutrino Masses + PMNS Joint Objective (diagnostic 28, Tier B2)

Falsifier: does adding Δm²21, Δm²31 to the objective collapse the 78.9%
PMNS-only strict survivor rate (diag 23)?

Objective: L_mass + 5*L_PMNS (manuscript-style joint neutrino loss).
Strict survivor: PMNS angles + Δm² within pre-registered tolerances.

N=100 geometries, 4 seeds, Gaussian + g_env kernel (same as diag 23).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.optimize import differential_evolution
from typing import Dict, List, Tuple

from kernel import compute_yukawa_matrix
from observables import (
    compute_neutrino_observables,
    compute_neutrino_joint_loss,
    compute_pmns_loss,
    compute_neutrino_mass_loss,
    NEUTRINO_TARGETS,
    NEUTRINO_MASS_TARGETS,
)
from phenomenology_utils import generate_neutrino_geometries

OPTIMIZER_SETTINGS = {
    "maxiter": 120,
    "popsize": 12,
    "tol": 1e-6,
    "mutation": (0.5, 1.0),
    "recombination": 0.7,
    "polish": False,
}

N_SEEDS = 4
N_GEOMETRIES = 100
GEOM_SEED = 28028

NEUTRINO_BOUNDS = [
    (0.5, 6.0),
    (0.1, 2.0),
    (0.0, 2 * np.pi),
    (1.0, 5.0),
    (0.01, 0.5),
    (0.01, 0.5),
    (0.45, 0.75),
]

PMNS_STRICT = {
    "theta12": 0.15,
    "theta23": 0.15,
    "theta13": 0.20,
}

MASS_STRICT = {
    "dm21": 0.30,
    "dm31": 0.30,
}

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "28_neutrino_masses_pmns_joint.txt"
)

DIAG23_PMNS_ONLY_STRICT_PCT = 78.9


def make_objective(L: Tuple, N: Tuple):
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

    return objective


def check_pmns_strict(rec: Dict) -> bool:
    for key, tol in PMNS_STRICT.items():
        t = NEUTRINO_TARGETS[key]
        v = rec[key]
        if v <= 0 or t <= 0:
            return False
        if abs(v - t) / t > tol:
            return False
    return True


def check_joint_strict(rec: Dict) -> bool:
    if not check_pmns_strict(rec):
        return False
    for key, tol in MASS_STRICT.items():
        t = NEUTRINO_MASS_TARGETS[key]
        v = rec[key]
        if v <= 0 or t <= 0:
            return False
        if abs(v - t) / t > tol:
            return False
    return True


def optimize_geometries(geometries: List[Tuple]) -> Dict:
    records = []
    failed_theta23 = 0

    for geom_idx, (L, N) in enumerate(geometries):
        best = None
        for seed in range(N_SEEDS):
            objective = make_objective(L, N)
            try:
                result = differential_evolution(
                    objective,
                    NEUTRINO_BOUNDS,
                    seed=seed + geom_idx * 100,
                    **OPTIMIZER_SETTINGS,
                )
            except Exception:
                continue
            if result.fun >= 999:
                failed_theta23 += 1
                continue

            sigma, k, alpha, eta, eps_nu, eps_e, g_env = result.x
            Ynu = compute_yukawa_matrix(L, N, sigma * g_env, k, alpha, eta, eps_nu)
            Ye = compute_yukawa_matrix(L, N, sigma, k, alpha, eta, eps_e)
            obs = compute_neutrino_observables(Ynu, Ye)
            joint_l = compute_neutrino_joint_loss(obs)

            rec = {
                "geom": geom_idx,
                "seed": seed,
                "joint_loss": joint_l,
                "pmns_loss": compute_pmns_loss(obs),
                "mass_loss": compute_neutrino_mass_loss(obs),
                "theta12": obs["theta12"],
                "theta23": obs["theta23"],
                "theta13": obs["theta13"],
                "dm21": obs["dm21"],
                "dm31": obs["dm31"],
                "m1": obs["m1"],
                "m2": obs["m2"],
                "m3": obs["m3"],
                "g_env": g_env,
            }
            if best is None or joint_l < best["joint_loss"]:
                best = rec

        if best is not None:
            best["pmns_strict"] = check_pmns_strict(best)
            best["joint_strict"] = check_joint_strict(best)
            records.append(best)

    return {"records": records, "failed_theta23": failed_theta23}


def format_report(geometries: List[Tuple], result: Dict) -> str:
    records = result["records"]
    lines = [
        "=" * 78,
        "NEUTRINO MASSES + PMNS JOINT (diagnostic 28, Tier B2)",
        "=" * 78,
        "",
        "Objective: L_mass(Δm²) + 5*L_PMNS",
        f"Reference diag 23 PMNS-only strict: {DIAG23_PMNS_ONLY_STRICT_PCT:.1f}%",
        "",
        "PMNS targets (rad):",
    ]
    for k, v in NEUTRINO_TARGETS.items():
        lines.append(f"  {k}: {v}")
    lines.append("Mass targets (eV²):")
    for k, v in NEUTRINO_MASS_TARGETS.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append(
        f"Geometries: {N_GEOMETRIES} requested, {len(records)} solved (seed={GEOM_SEED})"
    )
    lines.append(f"Seeds per geometry: {N_SEEDS}")
    lines.append(f"Optimizer: {OPTIMIZER_SETTINGS}")
    lines.append(f"PMNS strict tol: {PMNS_STRICT}; mass strict tol: {MASS_STRICT}")
    lines.append(f"Optimization failures (theta23 ≈ 0): {result['failed_theta23']}")
    lines.append("")

    if not records:
        lines.append("No converged solutions.")
        return "\n".join(lines)

    pmns_n = sum(1 for r in records if r.get("pmns_strict"))
    joint_n = sum(1 for r in records if r.get("joint_strict"))
    pmns_pct = 100.0 * pmns_n / len(records)
    joint_pct = 100.0 * joint_n / len(records)

    lines.extend(
        [
            "--- SUMMARY (best seed per geometry) ---",
            f"  Geometries solved: {len(records)} / {len(geometries)}",
            f"  Strict PMNS only: {pmns_n} ({pmns_pct:.1f}%)",
            f"  Strict PMNS + Δm² (joint): {joint_n} ({joint_pct:.1f}%)",
            f"  Joint loss median: {np.median([r['joint_loss'] for r in records]):.6f}",
            f"  Mass loss median: {np.median([r['mass_loss'] for r in records]):.6f}",
            "",
            "--- B2 VERDICT ---",
            f"  PMNS-only strict dropped from {DIAG23_PMNS_ONLY_STRICT_PCT:.1f}% to {pmns_pct:.1f}%",
            f"  under joint objective (same geometry count when comparable).",
            f"  Joint strict (angles + Δm²): {joint_pct:.1f}%.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="N=10 geometries")
    args = parser.parse_args()

    n_geom = 10 if args.smoke else N_GEOMETRIES
    print(f"Neutrino joint mass+PMNS diagnostic (N={n_geom})...")
    geometries = generate_neutrino_geometries(n_geom, GEOM_SEED)
    result = optimize_geometries(geometries)
    report = format_report(geometries, result)
    print(report)

    if not args.smoke:
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            f.write(report)
        print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
