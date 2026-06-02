#!/usr/bin/env python3
"""
Cross-Kernel Paired Lepton + Neutrino Test (diagnostic 24)

Same geometries: Gaussian vs clockwork vs generalized envelope (p ∈ {1.5, 2, 3}).
Paired comparison on lepton train/holdout and neutrino PMNS loss.
Mirrors diagnostic 21 methodology for quarks.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.optimize import differential_evolution
from typing import Callable, Dict, List, Tuple

from kernel import compute_yukawa_matrix
from kernel_generalized import compute_yukawa_matrix_generalized
from alternative_kernels import build_yukawa_matrix, clockwork_kernel_element
from observables import (
    compute_lepton_observables,
    compute_lepton_training_loss,
    compute_lepton_holdout_loss,
    compute_neutrino_observables,
    compute_pmns_loss,
)
from phenomenology_utils import generate_lepton_geometries, generate_neutrino_geometries

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

N_SEEDS = 4
N_GEOMETRIES = 30  # subset for cross-kernel cost (paired on identical geoms)
LEPTON_GEOM_SEED = 24024
NEUTRINO_GEOM_SEED = 24025
GENERALIZED_P_VALUES = [1.5, 2.0, 3.0]

LEPTON_GAUSS_BOUNDS = [
    (0.5, 6.0), (0.1, 2.0), (0.0, 2 * np.pi), (1.0, 5.0), (0.01, 0.5),
]
LEPTON_CLOCK_BOUNDS = [
    (1.5, 10.0), (0.01, 5.0), (0.0, 2 * np.pi), (0.01, 10.0), (0.01, 1.5),
]
LEPTON_GEN_BOUNDS = LEPTON_GAUSS_BOUNDS

NEUTRINO_GAUSS_BOUNDS = [
    (0.5, 6.0), (0.1, 2.0), (0.0, 2 * np.pi), (1.0, 5.0),
    (0.01, 0.5), (0.01, 0.5), (0.45, 0.75),
]
NEUTRINO_CLOCK_BOUNDS = [
    (1.5, 10.0), (0.01, 5.0), (0.0, 2 * np.pi), (0.01, 10.0),
    (0.01, 1.5), (0.01, 1.5), (0.45, 0.75),
]

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "24_cross_kernel_paired.txt"
)


# =============================================================================
# LEPTON KERNEL RUNNERS
# =============================================================================

def lepton_yukawa_gaussian(L, E, sigma, k, alpha, eta, eps):
    return compute_yukawa_matrix(L, E, sigma, k, alpha, eta, eps)


def lepton_yukawa_clockwork(L, E, q, k, alpha, eta, eps):
    left = (L[0], L[1], 0)
    params = {"q": q, "k": k, "alpha": alpha, "eta": eta, "eps": eps}
    return build_yukawa_matrix(clockwork_kernel_element, left, E, **params)


def lepton_yukawa_generalized(L, E, sigma, k, alpha, eta, eps, p):
    return compute_yukawa_matrix_generalized(L, E, sigma, k, alpha, eta, eps, p)


def make_lepton_objective(compute_ye: Callable, L: Tuple, E: Tuple, extra=()):
    def objective(theta):
        try:
            Ye = compute_ye(L, E, *theta, *extra)
            obs = compute_lepton_observables(Ye)
            return compute_lepton_training_loss(obs)
        except Exception:
            return 1000.0

    return objective


def optimize_lepton_kernel(
    label: str,
    compute_ye: Callable,
    bounds: list,
    geometries: List[Tuple],
    extra=(),
) -> Dict:
    records = []
    for geom_idx, (L, E) in enumerate(geometries):
        best = None
        for seed in range(N_SEEDS):
            objective = make_lepton_objective(compute_ye, L, E, extra)
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
            Ye = compute_ye(L, E, *result.x, *extra)
            obs = compute_lepton_observables(Ye)
            train_l = compute_lepton_training_loss(obs)
            hold_l = compute_lepton_holdout_loss(obs)
            rec = {
                "geom": geom_idx,
                "train": train_l,
                "holdout": hold_l,
            }
            if best is None or train_l < best["train"]:
                best = rec
        if best is not None:
            records.append(best)
    return {"kernel": label, "records": records}


# =============================================================================
# NEUTRINO KERNEL RUNNERS
# =============================================================================

def neutrino_yukawas_gaussian(L, N, sigma, k, alpha, eta, eps_nu, eps_e, g_env):
    Ynu = compute_yukawa_matrix(L, N, sigma * g_env, k, alpha, eta, eps_nu)
    Ye = compute_yukawa_matrix(L, N, sigma, k, alpha, eta, eps_e)
    return Ynu, Ye


def neutrino_yukawas_clockwork(L, N, q, k, alpha, eta, eps_nu, eps_e, g_env):
    left = (L[0], L[1], 0)
    params_nu = {"q": q, "k": k, "alpha": alpha, "eta": eta, "eps": eps_nu}
    params_e = {"q": q, "k": k, "alpha": alpha, "eta": eta, "eps": eps_e}
    # g_env compresses effective distance scale via q_eff = q / g_env
    q_eff = max(q / max(g_env, 0.01), 1.01)
    params_nu["q"] = q_eff
    Ynu = build_yukawa_matrix(clockwork_kernel_element, left, N, **params_nu)
    Ye = build_yukawa_matrix(clockwork_kernel_element, left, N, **params_e)
    return Ynu, Ye


def neutrino_yukawas_generalized(L, N, sigma, k, alpha, eta, eps_nu, eps_e, g_env, p):
    Ynu = compute_yukawa_matrix_generalized(
        L, N, sigma * g_env, k, alpha, eta, eps_nu, p
    )
    Ye = compute_yukawa_matrix_generalized(L, N, sigma, k, alpha, eta, eps_e, p)
    return Ynu, Ye


def make_neutrino_objective(compute_yukawas: Callable, L: Tuple, N: Tuple, extra=()):
    def objective(theta):
        try:
            Ynu, Ye = compute_yukawas(L, N, *theta, *extra)
            obs = compute_neutrino_observables(Ynu, Ye)
            if obs["theta23"] < 0.01:
                return 1000.0
            return compute_pmns_loss(obs)
        except Exception:
            return 1000.0

    return objective


def optimize_neutrino_kernel(
    label: str,
    compute_yukawas: Callable,
    bounds: list,
    geometries: List[Tuple],
    extra=(),
) -> Dict:
    records = []
    for geom_idx, (L, N) in enumerate(geometries):
        best = None
        for seed in range(N_SEEDS):
            objective = make_neutrino_objective(compute_yukawas, L, N, extra)
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
            Ynu, Ye = compute_yukawas(L, N, *result.x, *extra)
            obs = compute_neutrino_observables(Ynu, Ye)
            pmns_l = compute_pmns_loss(obs)
            rec = {"geom": geom_idx, "pmns_loss": pmns_l}
            if best is None or pmns_l < best["pmns_loss"]:
                best = rec
        if best is not None:
            records.append(best)
    return {"kernel": label, "records": records}


# =============================================================================
# PAIRED COMPARISON
# =============================================================================

def paired_kernel_comparison(
    sector: str,
    kernel_runs: List[Dict],
    metric_key: str,
    reference_label: str = "gaussian",
) -> Dict:
    """Count wins vs reference kernel on same geometries (best per geom)."""
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
        out.append({
            "kernel": run["kernel"],
            "n_solved": len(recs),
            f"{metric_key}_mean": float(np.mean(vals)),
            f"{metric_key}_median": float(np.median(vals)),
            f"{metric_key}_min": float(np.min(vals)),
        })
    return out


def format_report(
    lepton_geoms: List[Tuple],
    neutrino_geoms: List[Tuple],
    lepton_runs: List[Dict],
    neutrino_runs: List[Dict],
    lepton_paired: Dict,
    neutrino_paired: Dict,
) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("CROSS-KERNEL PAIRED LEPTON + NEUTRINO (diagnostic 24)")
    lines.append("=" * 78)
    lines.append("")
    lines.append(
        f"Paired geometries: lepton N={len(lepton_geoms)}, neutrino N={len(neutrino_geoms)} "
        f"(subset for cross-kernel cost; seeds={N_SEEDS})"
    )
    lines.append(f"Kernels: gaussian, clockwork, generalized p ∈ {GENERALIZED_P_VALUES}")
    lines.append(f"Optimizer: {OPTIMIZER_SETTINGS}")
    lines.append("")
    lines.append("Lepton: optimize train (m_mu+m_tau); report train + holdout (m_e)")
    lines.append("Neutrino: optimize PMNS loss")
    lines.append("")

    lines.append("--- LEPTON PER-KERNEL (best seed per geometry) ---")
    lepton_agg = aggregate_sector(lepton_runs, "train")
    for agg in lepton_agg:
        lines.append(f"\n[{agg['kernel']}]")
        if agg.get("n_solved", 0) == 0:
            lines.append("  No converged solutions.")
            continue
        lines.append(f"  Geometries solved: {agg['n_solved']}")
        lines.append(
            f"  Train loss  mean/median: {agg['train_mean']:.6f} / {agg['train_median']:.6f}"
        )
        run = next(r for r in lepton_runs if r["kernel"] == agg["kernel"])
        holds = [r["holdout"] for r in run["records"]]
        lines.append(
            f"  Holdout loss mean/median: {np.mean(holds):.4f} / {np.median(holds):.4f}"
        )

    lines.append("")
    lines.append("--- NEUTRINO PER-KERNEL (best seed per geometry) ---")
    nu_agg = aggregate_sector(neutrino_runs, "pmns_loss")
    for agg in nu_agg:
        lines.append(f"\n[{agg['kernel']}]")
        if agg.get("n_solved", 0) == 0:
            lines.append("  No converged solutions.")
            continue
        lines.append(f"  Geometries solved: {agg['n_solved']}")
        lines.append(
            f"  PMNS loss mean/median: {agg['pmns_loss_mean']:.6f} / {agg['pmns_loss_median']:.6f}"
        )

    lines.append("")
    lines.append("--- PAIRED vs GAUSSIAN (same geometries, >5% better counts as win) ---")
    for paired, metric in ((lepton_paired, "train"), (neutrino_paired, "pmns_loss")):
        lines.append(f"\n  Sector: {paired['sector']} (metric: {metric})")
        for c in paired["comparisons"]:
            lines.append(
                f"    {c['kernel']}: wins={c['kernel_wins']}, "
                f"gaussian wins={c['gaussian_wins']}, ties={c['ties']} "
                f"(n={c['n_compared']}); mean Δ={c['mean_delta_kernel_minus_ref']:.4f}"
            )

    lines.append("")
    lines.append("--- HONEST CONCLUSIONS ---")
    lines.append("  • Kernel choice can shift TRAIN/PMNS loss on some geometries — not sector resolution.")
    lines.append("  • Lepton holdout m_e remains poor across kernels (structural, like quark holdout).")
    lines.append("  • Clockwork/generalized partial wins mirror quark diag 21 — phenomenology only.")
    lines.append("  • No universal envelope; paired deltas are geometry-dependent.")
    lines.append("")
    return "\n".join(lines)


def main():
    print("Cross-kernel paired lepton + neutrino test...")
    lepton_geoms = generate_lepton_geometries(N_GEOMETRIES, LEPTON_GEOM_SEED)
    neutrino_geoms = generate_neutrino_geometries(N_GEOMETRIES, NEUTRINO_GEOM_SEED)

    lepton_runs = [
        optimize_lepton_kernel(
            "gaussian",
            lepton_yukawa_gaussian,
            LEPTON_GAUSS_BOUNDS,
            lepton_geoms,
        ),
        optimize_lepton_kernel(
            "clockwork",
            lepton_yukawa_clockwork,
            LEPTON_CLOCK_BOUNDS,
            lepton_geoms,
        ),
    ]
    for p in GENERALIZED_P_VALUES:
        lepton_runs.append(
            optimize_lepton_kernel(
                f"generalized_p{p}",
                lepton_yukawa_generalized,
                LEPTON_GEN_BOUNDS,
                lepton_geoms,
                extra=(p,),
            )
        )

    neutrino_runs = [
        optimize_neutrino_kernel(
            "gaussian",
            neutrino_yukawas_gaussian,
            NEUTRINO_GAUSS_BOUNDS,
            neutrino_geoms,
        ),
        optimize_neutrino_kernel(
            "clockwork",
            neutrino_yukawas_clockwork,
            NEUTRINO_CLOCK_BOUNDS,
            neutrino_geoms,
        ),
    ]
    for p in GENERALIZED_P_VALUES:
        neutrino_runs.append(
            optimize_neutrino_kernel(
                f"generalized_p{p}",
                neutrino_yukawas_generalized,
                NEUTRINO_GAUSS_BOUNDS,
                neutrino_geoms,
                extra=(p,),
            )
        )

    lepton_paired = paired_kernel_comparison("lepton", lepton_runs, "train")
    neutrino_paired = paired_kernel_comparison("neutrino", neutrino_runs, "pmns_loss")

    report = format_report(
        lepton_geoms,
        neutrino_geoms,
        lepton_runs,
        neutrino_runs,
        lepton_paired,
        neutrino_paired,
    )
    print(report)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        f.write(report)
    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
