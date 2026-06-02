#!/usr/bin/env python3
"""
Quark Joint 7-Observable Loss + Holdout (diagnostic 27, Tier A2)

Falsifier: is the CKM–m_c Pareto in diag 21 a train/holdout split artifact?

Optimizes L_quark = L_mass + 5*L_CKM (+ light-quark penalties) on ALL seven
quark observables simultaneously — matching legacy scripts/01_optimize_quarks.py
objective — then reports strict survivors and CKM–mc Pareto structure.

N=100 geometries, 4 seeds, same kernels as diagnostic 21.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy.optimize import differential_evolution
from typing import Callable, Dict, List, Tuple

from alternative_kernels import (
    compute_yukawas_gaussian,
    compute_yukawas_clockwork,
    KERNELS,
)
from kernel_generalized import compute_quark_yukawas_generalized
from observables import (
    compute_quark_observables,
    compute_ckm_loss,
    compute_mass_loss,
    compute_holdout_loss,
    compute_training_loss,
    QUARK_TARGETS,
    TRAINING_TARGETS,
    HOLDOUT_TARGETS,
)

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
GEOM_SEED = 27027
GENERALIZED_P_VALUES = [1.5, 2.0, 3.0]

STRICT_TOLERANCES = {
    "mc": 0.30,
    "Vus": 0.20,
    "Vcb": 0.30,
    "Vub": 0.50,
    "mu": 0.50,
    "md": 0.50,
    "ms": 0.50,
}

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "27_quark_joint_loss_holdout.txt"
)


def generate_test_geometries(n_geom: int, seed: int) -> List[Tuple]:
    rng = np.random.RandomState(seed)
    geometries = []
    for _ in range(n_geom):
        Q = tuple(sorted(rng.choice(range(15), 3, replace=False)))
        U = tuple(sorted(rng.choice(range(15), 3, replace=False)))
        D = tuple(sorted(rng.choice(range(15), 3, replace=False)))
        geometries.append((Q, U, D))
    return geometries


def compute_joint_quark_loss(obs: Dict[str, float]) -> float:
    """Full quark objective: L_mass + 5*L_CKM + light-quark floor penalties."""
    l_mass = compute_mass_loss(obs)
    l_ckm = compute_ckm_loss(obs)
    l_md = (
        2.0 * (np.log(0.002 / obs["md"])) ** 2 if obs["md"] < 0.002 else 0.0
    )
    l_mu = (
        0.5 * (np.log(0.0005 / obs["mu"])) ** 2 if obs["mu"] < 0.0005 else 0.0
    )
    return float(l_mass + 5.0 * l_ckm + l_md + l_mu)


def make_objective(compute_yukawas: Callable, Q: Tuple, U: Tuple, D: Tuple) -> Callable:
    def objective(theta):
        try:
            Yu, Yd = compute_yukawas(Q, U, D, *theta)
            obs = compute_quark_observables(Yu, Yd)
            if obs["mc"] < 0.01 or obs["mc"] > 500:
                return 1000.0
            return compute_joint_quark_loss(obs)
        except Exception:
            return 1000.0

    return objective


def check_strict_survivor(rec: Dict) -> bool:
    obs = {k: rec[k] for k in STRICT_TOLERANCES}
    targets = {k: QUARK_TARGETS[k] for k in STRICT_TOLERANCES}
    for key, tol in STRICT_TOLERANCES.items():
        t, v = targets[key], obs[key]
        if v <= 0 or t <= 0:
            return False
        if abs(v - t) / t > tol:
            return False
    return True


def pareto_summary(points: List[Tuple[float, float, float]]) -> Dict:
    if not points:
        return {}
    ckm = np.array([p[0] for p in points])
    mc = np.array([p[1] for p in points])
    nd = []
    for i, (c, m, h) in enumerate(points):
        dominated = False
        for j, (c2, m2, h2) in enumerate(points):
            if j == i:
                continue
            if c2 <= c and m2 <= m and (c2 < c or m2 < m):
                dominated = True
                break
        if not dominated:
            nd.append((c, m, h))
    corr = float(np.corrcoef(ckm, mc)[0, 1]) if len(ckm) > 2 else float("nan")
    return {
        "n_points": len(points),
        "n_pareto": len(nd),
        "ckm_mc_corr": corr,
        "ckm_median": float(np.median(ckm)),
        "mc_rel_median": float(np.median(mc)),
        "pareto_front": nd[:8],
    }


def optimize_kernel(
    kernel_label: str,
    compute_yukawas: Callable,
    bounds: list,
    geometries: List[Tuple],
) -> Dict:
    records = []
    pareto_points = []
    strict_pass = 0

    for geom_idx, (Q, U, D) in enumerate(geometries):
        best = None
        for seed in range(N_SEEDS):
            objective = make_objective(compute_yukawas, Q, U, D)
            try:
                result = differential_evolution(
                    objective,
                    bounds,
                    seed=seed + geom_idx * 100,
                    **OPTIMIZER_SETTINGS,
                )
            except Exception:
                continue
            if result.fun >= 999:
                continue
            Yu, Yd = compute_yukawas(Q, U, D, *result.x)
            obs = compute_quark_observables(Yu, Yd)
            joint_l = compute_joint_quark_loss(obs)
            train_l = compute_training_loss(obs)
            hold_l = compute_holdout_loss(obs)
            ckm_l = compute_ckm_loss(obs)
            mc_rel = abs(obs["mc"] - TRAINING_TARGETS["mc"]) / TRAINING_TARGETS["mc"]
            rec = {
                "geom": geom_idx,
                "seed": seed,
                "joint": joint_l,
                "train": train_l,
                "holdout": hold_l,
                "ckm_loss": ckm_l,
                "mc": obs["mc"],
                "mc_rel_err": mc_rel,
                "Vus": obs["Vus"],
                "Vcb": obs["Vcb"],
                "Vub": obs["Vub"],
                "mu": obs["mu"],
                "md": obs["md"],
                "ms": obs["ms"],
            }
            pareto_points.append((ckm_l, mc_rel, hold_l))
            if best is None or joint_l < best["joint"]:
                best = rec

        if best is None:
            continue
        records.append(best)
        if check_strict_survivor(best):
            strict_pass += 1

    return {
        "kernel": kernel_label,
        "records": records,
        "pareto_points": pareto_points,
        "strict_pass": strict_pass,
    }


def aggregate_kernel_result(res: Dict) -> Dict:
    recs = res["records"]
    if not recs:
        return {"kernel": res["kernel"], "n_solved": 0}
    joints = [r["joint"] for r in recs]
    holds = [r["holdout"] for r in recs]
    return {
        "kernel": res["kernel"],
        "n_solved": len(recs),
        "strict_survivors": res["strict_pass"],
        "strict_rate_pct": 100.0 * res["strict_pass"] / len(recs),
        "joint_mean": float(np.mean(joints)),
        "joint_median": float(np.median(joints)),
        "holdout_mean": float(np.mean(holds)),
        "holdout_median": float(np.median(holds)),
        "pareto": pareto_summary(res["pareto_points"]),
    }


def run_generalized(geometries: List[Tuple]) -> List[Dict]:
    base_bounds = [
        (0.5, 6.0),
        (0.1, 2.0),
        (0.0, 2 * np.pi),
        (1.0, 5.0),
        (0.01, 0.5),
        (0.01, 0.5),
    ]
    results = []
    for p in GENERALIZED_P_VALUES:

        def compute_fn(Q, U, D, sigma, k, alpha, eta, eps_u, eps_d):
            return compute_quark_yukawas_generalized(
                Q, U, D, sigma, k, alpha, eta, eps_u, eps_d, p=p
            )

        raw = optimize_kernel(f"generalized_p{p}", compute_fn, base_bounds, geometries)
        results.append(aggregate_kernel_result(raw))
    return results


def format_report(geometries: List[Tuple], kernel_results: List[Dict]) -> str:
    lines = [
        "=" * 78,
        "QUARK JOINT 7-OBSERVABLE LOSS (diagnostic 27, Tier A2)",
        "=" * 78,
        "",
        "Objective: L_mass + 5*L_CKM + md/mu floor penalties (all 7 obs in loss)",
        "Falsifier: CKM–m_c Pareto vs diag 21 train/holdout split",
        "",
        f"Geometries: {N_GEOMETRIES} requested, {len(geometries)} generated (seed={GEOM_SEED})",
        f"Seeds per geometry: {N_SEEDS}",
        f"Optimizer: {OPTIMIZER_SETTINGS}",
        f"Strict tolerances: {STRICT_TOLERANCES}",
        "",
        "Reference diag 21 (train mc+Vus+Vcb): 0% strict all kernels",
        "",
        "--- PER-KERNEL SUMMARY (best seed per geometry) ---",
    ]
    for agg in kernel_results:
        lines.append(f"\n[{agg['kernel']}]")
        if agg.get("n_solved", 0) == 0:
            lines.append("  No converged solutions.")
            continue
        lines.append(f"  Geometries solved: {agg['n_solved']}")
        lines.append(
            f"  Strict survivors: {agg['strict_survivors']} ({agg['strict_rate_pct']:.1f}%)"
        )
        lines.append(
            f"  Joint loss mean/median: {agg['joint_mean']:.4f} / {agg['joint_median']:.4f}"
        )
        lines.append(
            f"  Holdout loss mean/median: {agg['holdout_mean']:.4f} / {agg['holdout_median']:.4f}"
        )
        po = agg.get("pareto", {})
        if po:
            lines.append(
                f"  CKM–mc Pareto: {po['n_points']} pts, nondominated {po['n_pareto']}, "
                f"corr={po['ckm_mc_corr']:.4f}"
            )
            if po.get("pareto_front"):
                lines.append("  Sample Pareto front (ckm_loss, mc_rel, holdout):")
                for pt in po["pareto_front"][:5]:
                    lines.append(f"    {pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f}")

    lines.extend(
        [
            "",
            "--- A2 VERDICT ---",
            "  Compare strict rate and Pareto corr to diag 21.",
            "  If joint objective still yields 0% strict and sparse Pareto fronts,",
            "  CKM–m_c tension is structural (not a train-split artifact).",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="N=10 geometries")
    args = parser.parse_args()

    n_geom = 10 if args.smoke else N_GEOMETRIES
    print(f"Quark joint loss diagnostic (N={n_geom})...")
    geometries = generate_test_geometries(n_geom, GEOM_SEED)

    g_raw = optimize_kernel(
        "gaussian",
        compute_yukawas_gaussian,
        KERNELS["gaussian"]["bounds"],
        geometries,
    )
    c_raw = optimize_kernel(
        "clockwork",
        compute_yukawas_clockwork,
        KERNELS["clockwork"]["bounds"],
        geometries,
    )
    gen_aggs = run_generalized(geometries)
    kernel_results = [
        aggregate_kernel_result(g_raw),
        aggregate_kernel_result(c_raw),
    ] + gen_aggs

    report = format_report(geometries, kernel_results)
    print(report)

    if not args.smoke:
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            f.write(report)
        print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
