#!/usr/bin/env python3
"""
N5 — PMNS CP descriptive audit (diagnostic 46).

Descriptive only: on diag 28 pool (seed 28028, joint objective), report
delta_PMNS and J_PMNS from best-per-geometry optima vs PDG targets.
No new optimization protocol; no CP-targeted sweeps.

Usage:
  python diagnostics/46_n5_pmns_cp_descriptive_audit.py
  python diagnostics/46_n5_pmns_cp_descriptive_audit.py --smoke
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.optimize import differential_evolution

from kernel import compute_yukawa_matrix
from observables import (
    NEUTRINO_CP_TARGETS,
    compute_neutrino_joint_loss,
    compute_neutrino_observables,
)
from phenomenology_utils import generate_neutrino_geometries

GEOM_SEED = 28028
N_GEOM = 100
N_SEEDS = 4
OPT = dict(maxiter=120, popsize=12, tol=1e-6, mutation=(0.5, 1.0), recombination=0.7, polish=False)
NU_BOUNDS = [
    (0.5, 6.0), (0.1, 2.0), (0.0, 2 * np.pi), (1.0, 5.0),
    (0.01, 0.5), (0.01, 0.5), (0.45, 0.75),
]

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "46_n5_pmns_cp_descriptive_audit.txt"
)


def best_obs_for_geom(L, N, geom_idx):
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

    best_obs, best_loss = None, np.inf
    for seed in range(N_SEEDS):
        try:
            res = differential_evolution(
                objective, NU_BOUNDS, seed=seed + geom_idx * 100, **OPT
            )
        except Exception:
            continue
        if res.fun >= 999:
            continue
        sigma, k, alpha, eta, eps_nu, eps_e, g_env = res.x
        Ynu = compute_yukawa_matrix(L, N, sigma * g_env, k, alpha, eta, eps_nu)
        Ye = compute_yukawa_matrix(L, N, sigma, k, alpha, eta, eps_e)
        obs = compute_neutrino_observables(Ynu, Ye)
        jl = compute_neutrino_joint_loss(obs)
        if jl < best_loss:
            best_loss, best_obs = jl, obs
    return best_obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    n_geom = 20 if args.smoke else N_GEOM

    geoms = generate_neutrino_geometries(n_geom, GEOM_SEED)
    deltas, jvals, jabs = [], [], []
    solved = 0
    for gi, (L, N) in enumerate(geoms):
        obs = best_obs_for_geom(L, N, gi)
        if obs is None:
            continue
        solved += 1
        deltas.append(obs["delta_PMNS"])
        jvals.append(obs["J_PMNS"])
        jabs.append(obs["J_PMNS_abs"])

    tgt_d = NEUTRINO_CP_TARGETS["delta_PMNS"]
    med_d = float(np.median(deltas))
    med_j = float(np.median(jabs))

    lines = [
        "N5 PMNS CP descriptive audit (diagnostic 46)",
        f"Pool: diag 28 joint objective, seed={GEOM_SEED}, N={n_geom}",
        f"Solved: {solved}/{n_geom}",
        f"PDG delta_PMNS target: {tgt_d:.3f} rad",
        f"Median delta_PMNS: {med_d:.3f} rad",
        f"Median |J_PMNS|: {med_j:.6f}",
        f"Median |delta - target|: {float(np.median([abs(d - tgt_d) for d in deltas])):.3f} rad",
    ]
    report = "\n".join(lines)
    print(report)

    if not args.smoke:
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"Saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
