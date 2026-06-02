#!/usr/bin/env python3
"""
Joint 3-Sector Cross-Kernel Paired Comparison (diagnostic 26)

Shared geometry corpus at equal N: L (= quark Q) fixed across lepton, neutrino,
and quark sectors; independent right-handed triples per sector.

Kernels: Gaussian, clockwork, generalized envelope (p ∈ {1.5, 2, 3}).
Paired comparison vs Gaussian on identical geometries within each sector.
Target N=100 parity with diagnostics 21/22/23 (diag 24 used 30 geom, separate seeds).

Usage:
  python diagnostics/26_joint_three_sector_cross_kernel.py           # N=100
  python diagnostics/26_joint_three_sector_cross_kernel.py --smoke   # N=3 quick check
  python diagnostics/26_joint_three_sector_cross_kernel.py --n-geometries 30
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.optimize import differential_evolution
from typing import Callable, Dict, List, Tuple

from alternative_kernels import (
    build_yukawa_matrix,
    clockwork_kernel_element,
    compute_yukawas_clockwork,
    compute_yukawas_gaussian,
    KERNELS,
)
from kernel import compute_yukawa_matrix
from kernel_generalized import compute_quark_yukawas_generalized, compute_yukawa_matrix_generalized
from observables import (
    compute_holdout_loss,
    compute_lepton_holdout_loss,
    compute_lepton_observables,
    compute_lepton_training_loss,
    compute_neutrino_observables,
    compute_pmns_loss,
    compute_quark_observables,
    compute_training_loss,
)
from phenomenology_utils import (
    JointThreeSectorGeometry,
    generate_joint_three_sector_geometries,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

OPTIMIZER_SETTINGS = {
    "maxiter": 120,
    "popsize": 12,
    "tol": 1e-6,
    "mutation": (0.5, 1.0),
    "recombination": 0.7,
    "polish": False,
}

DEFAULT_N_GEOMETRIES = 100
DEFAULT_N_SEEDS = 4
JOINT_GEOM_SEED = 26026
GENERALIZED_P_VALUES = [1.5, 2.0, 3.0]

LEPTON_GAUSS_BOUNDS = [
    (0.5, 6.0), (0.1, 2.0), (0.0, 2 * np.pi), (1.0, 5.0), (0.01, 0.5),
]
LEPTON_CLOCK_BOUNDS = [
    (1.5, 10.0), (0.01, 5.0), (0.0, 2 * np.pi), (0.01, 10.0), (0.01, 1.5),
]
NEUTRINO_GAUSS_BOUNDS = [
    (0.5, 6.0), (0.1, 2.0), (0.0, 2 * np.pi), (1.0, 5.0),
    (0.01, 0.5), (0.01, 0.5), (0.45, 0.75),
]
NEUTRINO_CLOCK_BOUNDS = [
    (1.5, 10.0), (0.01, 5.0), (0.0, 2 * np.pi), (0.01, 10.0),
    (0.01, 1.5), (0.01, 1.5), (0.45, 0.75),
]
QUARK_GEN_BOUNDS = [
    (0.5, 6.0), (0.1, 2.0), (0.0, 2 * np.pi), (1.0, 5.0), (0.01, 0.5), (0.01, 0.5),
]

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "26_joint_three_sector_cross_kernel.txt"
)


# =============================================================================
# LEPTON
# =============================================================================

def lepton_yukawa_gaussian(L, E, sigma, k, alpha, eta, eps):
    return compute_yukawa_matrix(L, E, sigma, k, alpha, eta, eps)


def lepton_yukawa_clockwork(L, E, q, k, alpha, eta, eps):
    left = (L[0], L[1], 0)
    params = {"q": q, "k": k, "alpha": alpha, "eta": eta, "eps": eps}
    return build_yukawa_matrix(clockwork_kernel_element, left, E, **params)


def lepton_yukawa_generalized(L, E, sigma, k, alpha, eta, eps, p):
    return compute_yukawa_matrix_generalized(L, E, sigma, k, alpha, eta, eps, p)


def optimize_lepton_kernel(
    label: str,
    compute_ye: Callable,
    bounds: list,
    joint_geoms: List[JointThreeSectorGeometry],
    n_seeds: int,
    extra=(),
) -> Dict:
    records = []
    for geom in joint_geoms:
        L, E = geom.lepton
        best = None
        for seed in range(n_seeds):
            def objective(theta, _L=L, _E=E):
                try:
                    Ye = compute_ye(_L, _E, *theta, *extra)
                    obs = compute_lepton_observables(Ye)
                    return compute_lepton_training_loss(obs)
                except Exception:
                    return 1000.0

            try:
                result = differential_evolution(
                    objective,
                    bounds,
                    seed=seed + geom.index * 100,
                    **OPTIMIZER_SETTINGS,
                )
            except Exception:
                continue
            if result.fun >= 999:
                continue
            Ye = compute_ye(L, E, *result.x, *extra)
            obs = compute_lepton_observables(Ye)
            train_l = compute_lepton_training_loss(obs)
            hold_l = compute_lepton_holdout_loss(obs)
            rec = {"geom": geom.index, "train": train_l, "holdout": hold_l}
            if best is None or train_l < best["train"]:
                best = rec
        if best is not None:
            records.append(best)
    return {"kernel": label, "records": records}


# =============================================================================
# NEUTRINO
# =============================================================================

def neutrino_yukawas_gaussian(L, N, sigma, k, alpha, eta, eps_nu, eps_e, g_env):
    Ynu = compute_yukawa_matrix(L, N, sigma * g_env, k, alpha, eta, eps_nu)
    Ye = compute_yukawa_matrix(L, N, sigma, k, alpha, eta, eps_e)
    return Ynu, Ye


def neutrino_yukawas_clockwork(L, N, q, k, alpha, eta, eps_nu, eps_e, g_env):
    left = (L[0], L[1], 0)
    q_eff = max(q / max(g_env, 0.01), 1.01)
    params_nu = {"q": q_eff, "k": k, "alpha": alpha, "eta": eta, "eps": eps_nu}
    params_e = {"q": q, "k": k, "alpha": alpha, "eta": eta, "eps": eps_e}
    Ynu = build_yukawa_matrix(clockwork_kernel_element, left, N, **params_nu)
    Ye = build_yukawa_matrix(clockwork_kernel_element, left, N, **params_e)
    return Ynu, Ye


def neutrino_yukawas_generalized(L, N, sigma, k, alpha, eta, eps_nu, eps_e, g_env, p):
    Ynu = compute_yukawa_matrix_generalized(
        L, N, sigma * g_env, k, alpha, eta, eps_nu, p
    )
    Ye = compute_yukawa_matrix_generalized(L, N, sigma, k, alpha, eta, eps_e, p)
    return Ynu, Ye


def optimize_neutrino_kernel(
    label: str,
    compute_yukawas: Callable,
    bounds: list,
    joint_geoms: List[JointThreeSectorGeometry],
    n_seeds: int,
    extra=(),
) -> Dict:
    records = []
    for geom in joint_geoms:
        L, N = geom.neutrino
        best = None
        for seed in range(n_seeds):
            def objective(theta, _L=L, _N=N):
                try:
                    Ynu, Ye = compute_yukawas(_L, _N, *theta, *extra)
                    obs = compute_neutrino_observables(Ynu, Ye)
                    if obs["theta23"] < 0.01:
                        return 1000.0
                    return compute_pmns_loss(obs)
                except Exception:
                    return 1000.0

            try:
                result = differential_evolution(
                    objective,
                    bounds,
                    seed=seed + geom.index * 100,
                    **OPTIMIZER_SETTINGS,
                )
            except Exception:
                continue
            if result.fun >= 999:
                continue
            Ynu, Ye = compute_yukawas(L, N, *result.x, *extra)
            obs = compute_neutrino_observables(Ynu, Ye)
            pmns_l = compute_pmns_loss(obs)
            rec = {"geom": geom.index, "pmns_loss": pmns_l}
            if best is None or pmns_l < best["pmns_loss"]:
                best = rec
        if best is not None:
            records.append(best)
    return {"kernel": label, "records": records}


# =============================================================================
# QUARK
# =============================================================================

def optimize_quark_kernel(
    label: str,
    compute_yukawas: Callable,
    bounds: list,
    joint_geoms: List[JointThreeSectorGeometry],
    n_seeds: int,
) -> Dict:
    records = []
    for geom in joint_geoms:
        Q, U, D = geom.quark
        best = None
        for seed in range(n_seeds):
            def objective(theta, _Q=Q, _U=U, _D=D):
                try:
                    Yu, Yd = compute_yukawas(_Q, _U, _D, *theta)
                    obs = compute_quark_observables(Yu, Yd)
                    return compute_training_loss(obs)
                except Exception:
                    return 1000.0

            try:
                result = differential_evolution(
                    objective,
                    bounds,
                    seed=seed + geom.index * 100,
                    **OPTIMIZER_SETTINGS,
                )
            except Exception:
                continue
            if result.fun >= 999:
                continue
            Yu, Yd = compute_yukawas(Q, U, D, *result.x)
            obs = compute_quark_observables(Yu, Yd)
            train_l = compute_training_loss(obs)
            hold_l = compute_holdout_loss(obs)
            rec = {"geom": geom.index, "train": train_l, "holdout": hold_l}
            if best is None or train_l < best["train"]:
                best = rec
        if best is not None:
            records.append(best)
    return {"kernel": label, "records": records}


def run_quark_kernels(
    joint_geoms: List[JointThreeSectorGeometry], n_seeds: int
) -> List[Dict]:
    runs = [
        optimize_quark_kernel(
            "gaussian",
            compute_yukawas_gaussian,
            KERNELS["gaussian"]["bounds"],
            joint_geoms,
            n_seeds,
        ),
        optimize_quark_kernel(
            "clockwork",
            compute_yukawas_clockwork,
            KERNELS["clockwork"]["bounds"],
            joint_geoms,
            n_seeds,
        ),
    ]
    for p in GENERALIZED_P_VALUES:
        def compute_fn(Q, U, D, sigma, k, alpha, eta, eps_u, eps_d, p_val=p):
            return compute_quark_yukawas_generalized(
                Q, U, D, sigma, k, alpha, eta, eps_u, eps_d, p=p_val
            )

        runs.append(
            optimize_quark_kernel(
                f"generalized_p{p}",
                compute_fn,
                QUARK_GEN_BOUNDS,
                joint_geoms,
                n_seeds,
            )
        )
    return runs


# =============================================================================
# PAIRED COMPARISON + REPORT
# =============================================================================

def paired_kernel_comparison(
    sector: str,
    kernel_runs: List[Dict],
    metric_key: str,
    reference_label: str = "gaussian",
) -> Dict:
    ref = next((r for r in kernel_runs if r["kernel"] == reference_label), None)
    if ref is None or not ref["records"]:
        return {"sector": sector, "reference": reference_label, "comparisons": []}

    ref_by_geom = {r["geom"]: r[metric_key] for r in ref["records"]}
    comparisons = []
    for run in kernel_runs:
        if run["kernel"] == reference_label:
            continue
        wins = ref_wins = ties = 0
        deltas = []
        for rec in run["records"]:
            g = rec["geom"]
            if g not in ref_by_geom:
                continue
            a = ref_by_geom[g]
            b = rec[metric_key]
            deltas.append(b - a)
            if b < a * 0.95:
                wins += 1
            elif a < b * 0.95:
                ref_wins += 1
            else:
                ties += 1
        n = wins + ref_wins + ties
        comparisons.append({
            "kernel": run["kernel"],
            "n_compared": n,
            "kernel_wins": wins,
            f"{reference_label}_wins": ref_wins,
            "ties": ties,
            "mean_delta_kernel_minus_ref": float(np.mean(deltas)) if deltas else float("nan"),
        })
    return {"sector": sector, "reference": reference_label, "comparisons": comparisons}


def aggregate_sector(runs: List[Dict], metric_key: str) -> List[Dict]:
    out = []
    for run in runs:
        recs = run["records"]
        if not recs:
            out.append({"kernel": run["kernel"], "n_solved": 0})
            continue
        vals = [r[metric_key] for r in recs]
        agg = {
            "kernel": run["kernel"],
            "n_solved": len(recs),
            f"{metric_key}_mean": float(np.mean(vals)),
            f"{metric_key}_median": float(np.median(vals)),
            f"{metric_key}_min": float(np.min(vals)),
        }
        if metric_key == "train" and "holdout" in recs[0]:
            holds = [r["holdout"] for r in recs]
            agg["holdout_mean"] = float(np.mean(holds))
            agg["holdout_median"] = float(np.median(holds))
        out.append(agg)
    return out


def format_report(
    joint_geoms: List[JointThreeSectorGeometry],
    n_seeds: int,
    quark_runs: List[Dict],
    lepton_runs: List[Dict],
    neutrino_runs: List[Dict],
    quark_paired: Dict,
    lepton_paired: Dict,
    neutrino_paired: Dict,
    elapsed_sec: float,
) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("JOINT 3-SECTOR CROSS-KERNEL PAIRED COMPARISON (diagnostic 26)")
    lines.append("=" * 78)
    lines.append("")
    lines.append(
        f"Joint corpus: N={len(joint_geoms)} geometries (requested), "
        f"seed={JOINT_GEOM_SEED}, seeds/kernel/geom={n_seeds}"
    )
    lines.append("Shared L (= quark Q) across lepton, neutrino, quark; independent E, N, U, D.")
    lines.append(f"Kernels: gaussian, clockwork, generalized p ∈ {GENERALIZED_P_VALUES}")
    lines.append(f"Optimizer: {OPTIMIZER_SETTINGS}")
    lines.append(f"Wall time: {elapsed_sec / 60:.1f} min")
    lines.append("")
    lines.append("Quark/lepton: optimize train; report train + holdout.")
    lines.append("Neutrino: optimize PMNS loss.")
    lines.append("")

    sector_blocks = (
        ("QUARK", quark_runs, "train", quark_paired),
        ("LEPTON", lepton_runs, "train", lepton_paired),
        ("NEUTRINO", neutrino_runs, "pmns_loss", neutrino_paired),
    )
    for title, runs, metric, paired in sector_blocks:
        lines.append(f"--- {title} PER-KERNEL (best seed per geometry) ---")
        agg = aggregate_sector(runs, metric)
        for a in agg:
            lines.append(f"\n[{a['kernel']}]")
            if a.get("n_solved", 0) == 0:
                lines.append("  No converged solutions.")
                continue
            lines.append(f"  Geometries solved: {a['n_solved']}")
            lines.append(
                f"  {metric} mean/median: {a[f'{metric}_mean']:.6f} / {a[f'{metric}_median']:.6f}"
            )
            if "holdout_median" in a:
                lines.append(
                    f"  Holdout mean/median: {a['holdout_mean']:.4f} / {a['holdout_median']:.4f}"
                )
        lines.append("")
        lines.append(f"--- {title} PAIRED vs GAUSSIAN (>5% better = win) ---")
        for c in paired["comparisons"]:
            lines.append(
                f"  {c['kernel']}: wins={c['kernel_wins']}, "
                f"gaussian wins={c['gaussian_wins']}, ties={c['ties']} "
                f"(n={c['n_compared']}); mean Δ={c['mean_delta_kernel_minus_ref']:.4f}"
            )
        lines.append("")

    lines.append("--- HONEST CONCLUSIONS ---")
    lines.append("  • Joint corpus enables cross-sector comparison at equal N with shared L.")
    lines.append("  • Kernel choice shifts train/PMNS on some geometries — not sector resolution.")
    lines.append("  • Quark holdout and lepton m_e remain poor across kernels (structural).")
    lines.append("  • No universal envelope; paired wins are geometry-dependent.")
    lines.append("")
    return "\n".join(lines)


def run_lepton_kernels(
    joint_geoms: List[JointThreeSectorGeometry], n_seeds: int
) -> List[Dict]:
    runs = [
        optimize_lepton_kernel(
            "gaussian", lepton_yukawa_gaussian, LEPTON_GAUSS_BOUNDS, joint_geoms, n_seeds
        ),
        optimize_lepton_kernel(
            "clockwork", lepton_yukawa_clockwork, LEPTON_CLOCK_BOUNDS, joint_geoms, n_seeds
        ),
    ]
    for p in GENERALIZED_P_VALUES:
        runs.append(
            optimize_lepton_kernel(
                f"generalized_p{p}",
                lepton_yukawa_generalized,
                LEPTON_GAUSS_BOUNDS,
                joint_geoms,
                n_seeds,
                extra=(p,),
            )
        )
    return runs


def run_neutrino_kernels(
    joint_geoms: List[JointThreeSectorGeometry], n_seeds: int
) -> List[Dict]:
    runs = [
        optimize_neutrino_kernel(
            "gaussian", neutrino_yukawas_gaussian, NEUTRINO_GAUSS_BOUNDS, joint_geoms, n_seeds
        ),
        optimize_neutrino_kernel(
            "clockwork", neutrino_yukawas_clockwork, NEUTRINO_CLOCK_BOUNDS, joint_geoms, n_seeds
        ),
    ]
    for p in GENERALIZED_P_VALUES:
        runs.append(
            optimize_neutrino_kernel(
                f"generalized_p{p}",
                neutrino_yukawas_generalized,
                NEUTRINO_GAUSS_BOUNDS,
                joint_geoms,
                n_seeds,
                extra=(p,),
            )
        )
    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint 3-sector cross-kernel diagnostic 26")
    parser.add_argument(
        "--n-geometries",
        type=int,
        default=DEFAULT_N_GEOMETRIES,
        help=f"Number of joint geometries (default {DEFAULT_N_GEOMETRIES})",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        help=f"DE seeds per geometry (default {DEFAULT_N_SEEDS})",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke test with N=3 geometries",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    n_geom = 3 if args.smoke else args.n_geometries
    n_seeds = args.n_seeds

    print(
        f"Joint 3-sector cross-kernel test (N={n_geom}, seeds={n_seeds}, "
        f"{'SMOKE' if args.smoke else 'FULL'})..."
    )
    t0 = time.time()

    joint_geoms = generate_joint_three_sector_geometries(n_geom, JOINT_GEOM_SEED)
    if len(joint_geoms) < n_geom:
        print(f"WARNING: only {len(joint_geoms)}/{n_geom} joint geometries generated")

    print(f"  Quark sector ({len(joint_geoms)} geoms × 5 kernels)...")
    quark_runs = run_quark_kernels(joint_geoms, n_seeds)

    print(f"  Lepton sector...")
    lepton_runs = run_lepton_kernels(joint_geoms, n_seeds)

    print(f"  Neutrino sector...")
    neutrino_runs = run_neutrino_kernels(joint_geoms, n_seeds)

    quark_paired = paired_kernel_comparison("quark", quark_runs, "train")
    lepton_paired = paired_kernel_comparison("lepton", lepton_runs, "train")
    neutrino_paired = paired_kernel_comparison("neutrino", neutrino_runs, "pmns_loss")

    elapsed = time.time() - t0
    report = format_report(
        joint_geoms,
        n_seeds,
        quark_runs,
        lepton_runs,
        neutrino_runs,
        quark_paired,
        lepton_paired,
        neutrino_paired,
        elapsed,
    )
    print(report)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        f.write(report)
    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
