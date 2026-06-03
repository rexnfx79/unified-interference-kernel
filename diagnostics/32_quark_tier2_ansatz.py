#!/usr/bin/env python3
"""
Quark Tier-2 ansatz diagnostic (diagnostic 32)

Pre-registered test of new matrix-level / texture kernels vs diag-21 Gaussian baseline.

Hypotheses (future-work Tier 2):
  P2.1 rank2_clockwork_sum — Y = w Y^(1) + (1-w) Y^(2)
  P2.2 fn_texture, fn_texture_split — charge-based hierarchy
  P2.3 deferred (scheme/RGE readout) — not in this script

Falsifiers:
  - Strict survivor rate > 0 at N=100 on any Tier-2 kernel
  - Median holdout loss improves >20% vs Gaussian (diag 09 rule)

Reference: diag 21 (0% strict Gaussian at N=100), geom seed 21021.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.optimize import differential_evolution
from typing import Callable, Dict, List, Tuple

from alternative_kernels import KERNELS, TIER2_QUARK_KERNELS
from observables import (
    QUARK_TARGETS,
    compute_quark_observables,
    compute_training_loss,
    compute_holdout_loss,
    compute_ckm_loss,
)
from phenomenology_utils import generate_quark_geometries

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
GEOM_SEED = 32032
HOLDOUT_IMPROVEMENT_THRESHOLD = 0.20

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
    os.path.dirname(__file__), "results", "32_quark_tier2_ansatz.txt"
)


def effective_rank(Y: np.ndarray, tol: float = 1e-3) -> float:
    s = np.linalg.svd(Y, compute_uv=False)
    s = s[s > tol * s[0]] if s[0] > tol else s
    smax = s.sum()
    if smax <= 0:
        return 0.0
    p = s / smax
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p + 1e-30))))


def check_strict_survivor(obs: Dict[str, float]) -> bool:
    for key, tol in STRICT_TOLERANCES.items():
        t, v = QUARK_TARGETS[key], obs[key]
        if v <= 0 or t <= 0:
            return False
        if abs(v - t) / t > tol:
            return False
    return True


def optimize_kernel(
    kernel_label: str,
    geometries: List[Tuple],
) -> Dict:
    spec = KERNELS[kernel_label]
    compute_yukawas = spec["compute_yukawas"]
    bounds = spec["bounds"]
    records = []
    ranks_u = []
    ranks_d = []

    for geom_idx, (Q, U, D) in enumerate(geometries):

        def objective(theta):
            try:
                Yu, Yd = compute_yukawas(Q, U, D, *theta)
                obs = compute_quark_observables(Yu, Yd)
                if obs["mc"] < 0.01 or obs["mc"] > 500:
                    return 1000.0
                return compute_training_loss(obs)
            except Exception:
                return 1000.0

        best = None
        for seed in range(N_SEEDS):
            try:
                result = differential_evolution(
                    objective,
                    bounds,
                    seed=seed + geom_idx * 100,
                    **OPTIMIZER_SETTINGS,
                )
            except Exception:
                continue
            if abs(result.fun - 1000.0) < 1e-3:
                continue
            Yu, Yd = compute_yukawas(Q, U, D, *result.x)
            obs = compute_quark_observables(Yu, Yd)
            rec = {
                "train": compute_training_loss(obs),
                "holdout": compute_holdout_loss(obs),
                "ckm": compute_ckm_loss(obs),
                "strict": check_strict_survivor(obs),
                "mc": obs["mc"],
                "rank_u": effective_rank(Yu),
                "rank_d": effective_rank(Yd),
            }
            if best is None or rec["train"] < best["train"]:
                best = rec

        if best is None:
            continue
        records.append(best)
        ranks_u.append(best["rank_u"])
        ranks_d.append(best["rank_d"])

    strict_n = sum(1 for r in records if r["strict"])
    holds = [r["holdout"] for r in records]
    trains = [r["train"] for r in records]
    return {
        "kernel": kernel_label,
        "n_solved": len(records),
        "strict_n": strict_n,
        "strict_rate_pct": 100.0 * strict_n / max(len(records), 1),
        "train_median": float(np.median(trains)) if trains else float("nan"),
        "holdout_median": float(np.median(holds)) if holds else float("nan"),
        "holdout_min": float(np.min(holds)) if holds else float("nan"),
        "rank_u_mean": float(np.mean(ranks_u)) if ranks_u else float("nan"),
        "rank_d_mean": float(np.mean(ranks_d)) if ranks_d else float("nan"),
    }


def format_report(results: List[Dict], baseline_holdout: float) -> str:
    lines = [
        "=" * 78,
        "QUARK TIER-2 ANSATZ (diagnostic 32)",
        "=" * 78,
        "",
        "Pre-registered: TIER2_QUARK_KERNELS in alternative_kernels.py",
        f"Geometries: N={N_GEOMETRIES}, seed={GEOM_SEED} (phenomenology triples)",
        f"Optimizer: {OPTIMIZER_SETTINGS}, seeds/geom={N_SEEDS}",
        f"Objective: training loss (mc, Vus, Vcb) — same split as diag 21",
        f"Holdout accept rule: median holdout improves >{100*HOLDOUT_IMPROVEMENT_THRESHOLD:.0f}% vs gaussian",
        f"Strict tolerances: {STRICT_TOLERANCES}",
        "",
        f"Reference diag 21: gaussian 0% strict at N=100",
        f"Baseline gaussian holdout median: {baseline_holdout:.4f}",
        "",
        "--- PER-KERNEL ---",
    ]
    for r in results:
        imp = ""
        if baseline_holdout > 0 and not np.isnan(r["holdout_median"]):
            pct = 100.0 * (baseline_holdout - r["holdout_median"]) / baseline_holdout
            accept = pct >= 100 * HOLDOUT_IMPROVEMENT_THRESHOLD
            imp = f" holdout_improve={pct:.1f}% accept={accept}"
        lines.append(
            f"\n[{r['kernel']}] solved={r['n_solved']} strict={r['strict_n']} "
            f"({r['strict_rate_pct']:.1f}%)"
        )
        lines.append(
            f"  train_median={r['train_median']:.4f} holdout_median={r['holdout_median']:.4f} "
            f"holdout_min={r['holdout_min']:.4f}{imp}"
        )
        lines.append(
            f"  mean effective rank Yu={r['rank_u_mean']:.3f} Yd={r['rank_d_mean']:.3f}"
        )

    any_strict = any(r["strict_n"] > 0 for r in results)
    any_holdout = any(
        baseline_holdout > 0
        and r["holdout_median"] <= baseline_holdout * (1 - HOLDOUT_IMPROVEMENT_THRESHOLD)
        for r in results
        if r["kernel"] != "gaussian"
    )
    lines.extend(["", "--- VERDICT ---"])
    if any_strict:
        lines.append("  Tier-2 produced strict survivors — ansatz class warrants scaled follow-up.")
    else:
        lines.append("  0% strict on all Tier-2 kernels at N=100 — P2.1/P2.2 falsified at strict protocol.")
    if any_holdout:
        lines.append("  At least one kernel beats Gaussian holdout by >20% — partial P2 holdout gain.")
    else:
        lines.append("  No kernel beats Gaussian holdout by >20% — no generalization gain from Tier-2 ansätze.")
    lines.append("  P2.3 (scheme/RGE readout): not tested in this diagnostic.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="N=10 geometries")
    parser.add_argument("--kernels", nargs="*", default=None)
    args = parser.parse_args()

    n_geom = 10 if args.smoke else N_GEOMETRIES
    kernels = args.kernels or TIER2_QUARK_KERNELS
    geometries = generate_quark_geometries(n_geom, GEOM_SEED)
    print(f"Tier-2 quark diagnostic: N={len(geometries)}, kernels={kernels}")

    results = []
    for label in kernels:
        if label not in KERNELS:
            print(f"  skip unknown kernel {label}")
            continue
        print(f"  optimizing {label}...")
        results.append(optimize_kernel(label, geometries))

    baseline = next((r for r in results if r["kernel"] == "gaussian"), None)
    baseline_holdout = baseline["holdout_median"] if baseline else float("nan")
    report = format_report(results, baseline_holdout)
    print(report)

    if not args.smoke:
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            f.write(report)
        print(f"Saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
