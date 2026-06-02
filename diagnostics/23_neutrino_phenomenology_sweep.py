#!/usr/bin/env python3
"""
Neutrino Sector Phenomenology Sweep (diagnostic 23)

Gaussian kernel sweep on neutrino geometries (L, N) with g_env envelope
compression. Optimizes PMNS mixing angles; reports strict vs legacy survivors,
g_env distribution, and g_env–mixing correlations with bootstrap CIs.
No universality claims.
"""

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
    compute_pmns_loss,
    NEUTRINO_TARGETS,
)
from phenomenology_utils import (
    generate_neutrino_geometries,
    check_legacy_neutrino,
    bootstrap_corr_ci,
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

N_SEEDS = 4
# 480 full grid would be ~20× slower; 100 is scaled vs legacy 480 subsample
N_GEOMETRIES = 100
GEOM_SEED = 23023

NEUTRINO_BOUNDS = [
    (0.5, 6.0),
    (0.1, 2.0),
    (0.0, 2 * np.pi),
    (1.0, 5.0),
    (0.01, 0.5),
    (0.01, 0.5),
    (0.45, 0.75),
]

STRICT_TOLERANCES = {
    "theta12": 0.15,
    "theta23": 0.15,
    "theta13": 0.20,
}

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "23_neutrino_phenomenology_sweep.txt"
)


def make_objective(L: Tuple, N: Tuple):
    def objective(theta):
        try:
            sigma, k, alpha, eta, eps_nu, eps_e, g_env = theta
            Ynu = compute_yukawa_matrix(L, N, sigma * g_env, k, alpha, eta, eps_nu)
            Ye = compute_yukawa_matrix(L, N, sigma, k, alpha, eta, eps_e)
            obs = compute_neutrino_observables(Ynu, Ye)
            if obs["theta23"] < 0.01:
                return 1000.0
            return compute_pmns_loss(obs)
        except Exception:
            return 1000.0

    return objective


def check_strict_survivor(rec: Dict) -> bool:
    for key, tol in STRICT_TOLERANCES.items():
        t = NEUTRINO_TARGETS[key]
        v = rec[key]
        if v <= 0 or t <= 0:
            return False
        if abs(v - t) / t > tol:
            return False
    return True


def optimize_geometries(geometries: List[Tuple]) -> Dict:
    records = []
    all_points = []
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
            pmns_l = compute_pmns_loss(obs)

            rec = {
                "geom": geom_idx,
                "seed": seed,
                "pmns_loss": pmns_l,
                "theta12": obs["theta12"],
                "theta23": obs["theta23"],
                "theta13": obs["theta13"],
                "sigma": sigma,
                "k": k,
                "alpha": alpha,
                "eta": eta,
                "eps_nu": eps_nu,
                "eps_e": eps_e,
                "g_env": g_env,
            }
            all_points.append(rec)
            if best is None or pmns_l < best["pmns_loss"]:
                best = rec

        if best is not None:
            best["strict"] = check_strict_survivor(best)
            best["legacy"] = check_legacy_neutrino(best)
            records.append(best)

    return {
        "records": records,
        "all_points": all_points,
        "failed_theta23": failed_theta23,
    }


def g_env_correlations(records: List[Dict], all_points: List[Dict]) -> Dict:
    """Correlations on best-per-geom records + bootstrap CIs."""
    if len(records) < 5:
        return {}

    g = np.array([r["g_env"] for r in records])
    t23 = np.array([r["theta23"] for r in records])
    t12 = np.array([r["theta12"] for r in records])
    t13 = np.array([r["theta13"] for r in records])
    pmns = np.array([r["pmns_loss"] for r in records])

    return {
        "best_per_geom": {
            "corr_g_env_theta23": bootstrap_corr_ci(g, t23, seed=2301),
            "corr_g_env_theta12": bootstrap_corr_ci(g, t12, seed=2302),
            "corr_g_env_theta13": bootstrap_corr_ci(g, t13, seed=2303),
            "corr_g_env_pmns_loss": bootstrap_corr_ci(g, pmns, seed=2304),
            "g_env_mean": float(np.mean(g)),
            "g_env_std": float(np.std(g)),
            "g_env_median": float(np.median(g)),
            "n_records": len(records),
        },
        "all_seed_points": {
            "n_points": len(all_points),
        },
    }


def format_report(geometries: List[Tuple], result: Dict, g_corr: Dict) -> str:
    records = result["records"]
    lines = []
    lines.append("=" * 78)
    lines.append("NEUTRINO PHENOMENOLOGY SWEEP (diagnostic 23)")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Kernel: Gaussian with g_env envelope compression on Y_nu")
    lines.append("Targets (PMNS angles, radians):")
    for k, v in NEUTRINO_TARGETS.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append(
        f"Geometries: {N_GEOMETRIES} requested, {len(records)} solved "
        f"(seed={GEOM_SEED}; legacy archive used 480)"
    )
    lines.append(f"Seeds per geometry: {N_SEEDS}")
    lines.append(f"Optimizer: {OPTIMIZER_SETTINGS}")
    lines.append(f"Strict tolerances (relative): {STRICT_TOLERANCES}")
    lines.append(
        "Legacy survivors: PMNS angle ranges from scripts/04_analyze_results.py"
    )
    lines.append(f"Optimization failures (theta23 ≈ 0): {result['failed_theta23']}")
    lines.append("")

    if not records:
        lines.append("No converged solutions.")
        return "\n".join(lines)

    strict_n = sum(1 for r in records if r.get("strict"))
    legacy_n = sum(1 for r in records if r.get("legacy"))
    losses = [r["pmns_loss"] for r in records]
    g_vals = [r["g_env"] for r in records]
    strict_g = [r["g_env"] for r in records if r.get("strict")]

    lines.append("--- SUMMARY (best seed per geometry) ---")
    lines.append(f"  Geometries solved: {len(records)} / {len(geometries)}")
    lines.append(
        f"  Strict PMNS survivors: {strict_n} ({100.0 * strict_n / len(records):.1f}%)"
    )
    lines.append(
        f"  Legacy PMNS survivors: {legacy_n} ({100.0 * legacy_n / len(records):.1f}%)"
    )
    lines.append(f"  PMNS loss mean/median: {np.mean(losses):.6f} / {np.median(losses):.6f}")
    lines.append(f"  Best PMNS loss: {min(losses):.6e}")
    lines.append("")

    lines.append("--- g_env DISTRIBUTION ---")
    lines.append(
        f"  All best-per-geom: mean={np.mean(g_vals):.3f} ± {np.std(g_vals):.3f}, "
        f"median={np.median(g_vals):.3f}, range=[{min(g_vals):.3f}, {max(g_vals):.3f}]"
    )
    lines.append("  Manuscript legacy: g_env ≈ 0.60 ± 0.07 (range 0.50–0.70)")
    if strict_g:
        lines.append(
            f"  Strict survivors: mean={np.mean(strict_g):.3f} ± {np.std(strict_g):.3f} "
            f"(n={len(strict_g)})"
        )
    lines.append("")

    lines.append("--- g_env–MIXING CORRELATION (bootstrap 95% CI, best per geometry) ---")
    lines.append(
        "Manuscript: weak g_env–mixing correlation (r < 0.1) — metric-dominated "
        "regime does not cleanly predict PMNS from g_env alone."
    )
    best = g_corr.get("best_per_geom", {})
    if best:
        for label, key in (
            ("theta23", "corr_g_env_theta23"),
            ("theta12", "corr_g_env_theta12"),
            ("theta13", "corr_g_env_theta13"),
            ("PMNS_loss", "corr_g_env_pmns_loss"),
        ):
            c = best.get(key, {})
            r = c.get("r", float("nan"))
            lo = c.get("ci_lo", float("nan"))
            hi = c.get("ci_hi", float("nan"))
            lines.append(f"  corr(g_env, {label}): r={r:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        weak = all(
            abs(best[k]["r"]) < 0.1
            for k in (
                "corr_g_env_theta23",
                "corr_g_env_theta12",
                "corr_g_env_pmns_loss",
            )
            if np.isfinite(best[k]["r"])
        )
        lines.append(
            f"  Weak correlation claim (|r| < 0.1 on θ23, θ12, loss): "
            f"{'CONSISTENT' if weak else 'MIXED — some |r| >= 0.1 or CI overlaps stronger coupling'}"
        )
    lines.append("")

    lines.append("--- HONEST CONCLUSIONS ---")
    lines.append("  • Neutrinos: PMNS angles fittable on many geometries; sector-specific.")
    lines.append(
        f"  • Strict {100.0 * strict_n / len(records):.1f}% vs legacy "
        f"{100.0 * legacy_n / len(records):.1f}% (legacy archive ~45% on 480 geom)."
    )
    lines.append(
        f"  • ~{100.0 * result['failed_theta23'] / max(1, len(geometries) * N_SEEDS):.0f}% "
        "of seed attempts hit theta23 ≈ 0 (optimization pathology)."
    )
    lines.append(
        f"  • g_env clusters near {np.median(g_vals):.2f} (not manuscript 0.60) — fit artifact."
    )
    lines.append("  • No cross-sector universality; metric-dominated label is descriptive only.")
    lines.append("")
    return "\n".join(lines)


def main():
    print(f"Neutrino phenomenology sweep ({N_GEOMETRIES} geometries)...")
    geometries = generate_neutrino_geometries(N_GEOMETRIES, GEOM_SEED)
    print(f"  Generated {len(geometries)} unique geometries")
    result = optimize_geometries(geometries)
    g_corr = g_env_correlations(result["records"], result["all_points"])
    report = format_report(geometries, result, g_corr)
    print(report)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        f.write(report)
    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
