#!/usr/bin/env python3
"""
Quark geometry space extension (diagnostic 29)

Explores discrete geometries beyond the exhaustive max_coord=5 grid
(1000 geometries, coordinates 0..4) used in scripts/01_optimize_quarks.py.

Extension geometries require max_coord >= 6 (coordinates may include 5+).
Compares joint 7-observable loss and strict survivors vs the legacy grid.

Falsifier: if extended coordinates yield strict survivors or materially
lower joint loss than the max_coord=5 Pareto best, geometry coverage was
the limiting factor; otherwise the quark failure is kernel/landscape, not grid size.
"""

import argparse
import os
import sys
from itertools import product
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy.optimize import differential_evolution

from kernel import compute_quark_yukawas
from observables import (
    QUARK_TARGETS,
    compute_quark_observables,
    compute_ckm_loss,
    compute_mass_loss,
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
GEOM_SEED = 29029
STRICT_TOLERANCES = {
    "mc": 0.30,
    "Vus": 0.20,
    "Vcb": 0.30,
    "Vub": 0.50,
    "mu": 0.50,
    "md": 0.50,
    "ms": 0.50,
}

BOUNDS = [
    (0.5, 6.0),
    (0.1, 2.0),
    (0.0, 2 * np.pi),
    (1.0, 5.0),
    (0.01, 0.5),
    (0.01, 0.5),
]

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "29_quark_geometry_extension.txt"
)


def geometry_key(Q: Tuple, U: Tuple, D: Tuple) -> Tuple:
    return (tuple(Q), tuple(U), tuple(D))


def generate_geometries(max_coord: int) -> List[Tuple[Tuple, Tuple, Tuple]]:
    """Same enumeration as scripts/01_optimize_quarks.py."""
    geometries = []
    for q1, q2 in product(range(max_coord), repeat=2):
        if q1 >= q2:
            continue
        for u1, u2, u3 in product(range(max_coord), repeat=3):
            if not (u1 < u2 < u3):
                continue
            for d1, d2, d3 in product(range(max_coord), repeat=3):
                if not (d1 < d2 < d3):
                    continue
                geometries.append(((q1, q2, 0), (u1, u2, u3), (d1, d2, d3)))
    return geometries


def baseline_keys(max_coord: int = 5) -> Set[Tuple]:
    return {geometry_key(*g) for g in generate_geometries(max_coord)}


def extension_geometries(
    min_max_coord: int = 6, max_max_coord: int = 8
) -> List[Tuple[Tuple, Tuple, Tuple]]:
    """Geometries not in the exhaustive max_coord=5 grid."""
    seen = baseline_keys(5)
    out = []
    for mc in range(min_max_coord, max_max_coord + 1):
        for g in generate_geometries(mc):
            k = geometry_key(*g)
            if k not in seen:
                seen.add(k)
                out.append(g)
    return out


def sample_geometries(
    pool: List[Tuple], n_geom: int, seed: int
) -> List[Tuple]:
    if n_geom >= len(pool):
        return list(pool)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(pool), size=n_geom, replace=False)
    return [pool[i] for i in sorted(idx)]


def compute_joint_quark_loss(obs: Dict[str, float]) -> float:
    l_mass = compute_mass_loss(obs)
    l_ckm = compute_ckm_loss(obs)
    l_md = 2.0 * (np.log(0.002 / obs["md"])) ** 2 if obs["md"] < 0.002 else 0.0
    l_mu = 0.5 * (np.log(0.0005 / obs["mu"])) ** 2 if obs["mu"] < 0.0005 else 0.0
    return float(l_mass + 5.0 * l_ckm + l_md + l_mu)


def check_strict_survivor(obs: Dict[str, float]) -> bool:
    for key, tol in STRICT_TOLERANCES.items():
        t, v = QUARK_TARGETS[key], obs[key]
        if v <= 0 or t <= 0:
            return False
        if abs(v - t) / t > tol:
            return False
    return True


def optimize_geometry(Q, U, D) -> Dict:
    best = None
    for seed in range(N_SEEDS):
        def objective(theta):
            try:
                Yu, Yd = compute_quark_yukawas(Q, U, D, *theta)
                obs = compute_quark_observables(Yu, Yd)
                if obs["mc"] < 0.01 or obs["mc"] > 500:
                    return 1000.0
                return compute_joint_quark_loss(obs)
            except Exception:
                return 1000.0

        try:
            result = differential_evolution(
                objective, BOUNDS, seed=seed, **OPTIMIZER_SETTINGS
            )
        except Exception:
            continue
        if abs(result.fun - 1000.0) < 1e-3:
            continue
        Yu, Yd = compute_quark_yukawas(Q, U, D, *result.x)
        obs = compute_quark_observables(Yu, Yd)
        rec = {
            "joint": compute_joint_quark_loss(obs),
            "ckm": compute_ckm_loss(obs),
            "mass": compute_mass_loss(obs),
            "strict": check_strict_survivor(obs),
            "theta": [float(x) for x in result.x],
            "winning_seed": seed,
            **{k: obs[k] for k in STRICT_TOLERANCES},
            "Q": Q,
            "U": U,
            "D": D,
        }
        if best is None or rec["joint"] < best["joint"]:
            best = rec
    return best


def load_baseline_best(csv_path: str) -> Dict:
    import pandas as pd

    df = pd.read_csv(csv_path)
    row = df.loc[df["loss_total"].idxmin()]
    Q = (int(row.Q1), int(row.Q2), 0)
    U = (int(row.U1), int(row.U2), int(row.U3))
    D = (int(row.D1), int(row.D2), int(row.D3))
    Yu, Yd = compute_quark_yukawas(
        Q,
        U,
        D,
        row.sigma,
        row.k,
        row.alpha,
        row.eta,
        row.eps_u,
        row.eps_d,
    )
    obs = compute_quark_observables(Yu, Yd)
    return {
        "label": "legacy_csv_best_total",
        "joint": float(row.loss_total),
        "ckm": float(row.loss_ckm),
        "strict": check_strict_survivor(obs),
        "mc": obs["mc"],
        "Q": Q,
        "U": U,
        "D": D,
    }


def format_report(
    n_sampled: int,
    pool_size: int,
    baseline: Dict,
    records: List[Dict],
    counts_by_max: Dict[int, int],
) -> str:
    solved = [r for r in records if r is not None]
    strict_n = sum(1 for r in solved if r["strict"])
    lines = [
        "=" * 78,
        "QUARK GEOMETRY SPACE EXTENSION (diagnostic 29)",
        "=" * 78,
        "",
        "Baseline grid: max_coord=5 exhaustive (1000 geometries, coords 0..4)",
        f"Extension pool: {pool_size} geometries with at least one new coordinate",
        f"  (max_coord 6..8 cumulative, not in baseline)",
        f"Sampled for optimization: {n_sampled} (seed={GEOM_SEED})",
        f"Seeds per geometry: {N_SEEDS}; kernel: Gaussian (scripts/01 objective)",
        "",
        "--- BASELINE REFERENCE (data/quark_results.csv best loss_total) ---",
        f"  Q={baseline['Q']}, U={baseline['U']}, D={baseline['D']}",
        f"  joint={baseline['joint']:.4f} ckm={baseline['ckm']:.4f} mc={baseline['mc']:.3f}",
        f"  strict_survivor={baseline['strict']}",
        "",
        "--- EXTENSION SAMPLE ---",
        f"  Solved: {len(solved)}/{n_sampled}",
        f"  Strict survivors: {strict_n} ({100 * strict_n / max(len(solved), 1):.1f}%)",
    ]
    if solved:
        joints = [r["joint"] for r in solved]
        ckms = [r["ckm"] for r in solved]
        lines.append(
            f"  Joint loss min/median: {min(joints):.4f} / {np.median(joints):.4f}"
        )
        lines.append(
            f"  CKM loss min/median: {min(ckms):.4f} / {np.median(ckms):.4f}"
        )
        best = min(solved, key=lambda r: r["joint"])
        lines.append(
            f"  Best extension: Q={best['Q']}, U={best['U']}, D={best['D']}"
        )
        lines.append(
            f"    joint={best['joint']:.4f} ckm={best['ckm']:.4f} "
            f"mc={best['mc']:.3f} strict={best['strict']}"
        )
        beats = best["joint"] < baseline["joint"]
        lines.append(
            f"  Beats legacy best joint loss: {beats}"
        )
        lines.append("")
        lines.append("  Top 5 extension geometries by joint loss:")
        for r in sorted(solved, key=lambda x: x["joint"])[:5]:
            mx = max(
                max(r["Q"][:2]),
                max(r["U"]),
                max(r["D"]),
            )
            lines.append(
                f"    max_coord={mx} Q={r['Q']} U={r['U']} D={r['D']} "
                f"joint={r['joint']:.4f} ckm={r['ckm']:.4f} mc={r['mc']:.3f}"
            )
    lines.extend(
        [
            "",
            "--- POOL COMPOSITION (extension only) ---",
        ]
    )
    for mc, cnt in sorted(counts_by_max.items()):
        lines.append(f"  geometries with max coordinate {mc}: {cnt}")
    lines.extend(["", "--- VERDICT ---"])
    if strict_n > 0:
        lines.append(
            "  Extension grid produced strict survivors — geometry coverage may have been limiting."
        )
    elif solved and min(r["joint"] for r in solved) < baseline["joint"]:
        lines.append(
            "  Marginal joint-loss improvement possible beyond max_coord=5, but 0% strict survivors;"
        )
        lines.append(
            "  larger grids do not achieve simultaneous PDG match (consistent with diag 27)."
        )
    else:
        lines.append(
            "  No joint-loss improvement vs baseline; extension is not the quark bottleneck."
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-geom", type=int, default=100, help="Extension geometries to optimize"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="N=15 extension geometries"
    )
    parser.add_argument(
        "--max-max-coord",
        type=int,
        default=8,
        help="Upper max_coord for extension enumeration",
    )
    args = parser.parse_args()

    n_geom = 15 if args.smoke else args.n_geom
    pool = extension_geometries(min_max_coord=6, max_max_coord=args.max_max_coord)
    counts_by_max = {}
    for g in pool:
        mx = max(max(g[0][:2]), max(g[1]), max(g[2]))
        counts_by_max[mx] = counts_by_max.get(mx, 0) + 1

    sampled = sample_geometries(pool, n_geom, GEOM_SEED)
    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "quark_results.csv"
    )
    baseline = load_baseline_best(csv_path)

    print(
        f"Geometry extension diagnostic: pool={len(pool)}, optimizing {len(sampled)}..."
    )
    records = []
    for i, (Q, U, D) in enumerate(sampled):
        if i % 5 == 0:
            print(f"  Progress {i}/{len(sampled)}")
        records.append(optimize_geometry(Q, U, D))

    report = format_report(len(sampled), len(pool), baseline, records, counts_by_max)
    print(report)

    if not args.smoke:
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            f.write(report)
        print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
