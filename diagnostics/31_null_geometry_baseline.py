#!/usr/bin/env python3
"""
Null / scrambled geometry baseline (diagnostic 31, Tier A4)

Falsifier: do kernel fits on real discrete geometries outperform null baselines?

Conditions on identical geometry draws (phenomenology sampler, seed 21021):
  1. kernel   — Gaussian kernel, optimize training loss (diag 21 protocol)
  2. shuffled — same U,D; Q replaced by random sorted triple (structure broken)
  3. haar     — no kernel: random Yu,Yd from log-normal singular values + Haar unitaries

Pass: kernel strict rate significantly above nulls AND lower holdout/train loss.
Fail (expected): nulls match or beat kernel → geometry signal is not special.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import unitary_group
from typing import Dict, List, Tuple

from alternative_kernels import compute_yukawas_gaussian
from observables import (
    compute_quark_observables,
    compute_training_loss,
    compute_holdout_loss,
    QUARK_TARGETS,
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
GEOM_SEED = 21021
HAAR_SAMPLES = 4
STRICT_TOLERANCES = {
    "mc": 0.30,
    "Vus": 0.20,
    "Vcb": 0.30,
    "Vub": 0.50,
    "mu": 0.50,
    "md": 0.50,
    "ms": 0.50,
}

GAUSSIAN_BOUNDS = [
    (0.5, 6.0),
    (0.1, 2.0),
    (0.0, 2 * np.pi),
    (1.0, 5.0),
    (0.01, 0.5),
    (0.01, 0.5),
]

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "31_null_geometry_baseline.txt"
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


def check_strict_survivor(obs: Dict[str, float]) -> bool:
    for key, tol in STRICT_TOLERANCES.items():
        t, v = QUARK_TARGETS[key], obs[key]
        if v <= 0 or t <= 0:
            return False
        if abs(v - t) / t > tol:
            return False
    return True


def optimize_kernel(Q, U, D) -> Dict:
    best = None
    for seed in range(N_SEEDS):
        def objective(theta):
            try:
                Yu, Yd = compute_yukawas_gaussian(Q, U, D, *theta)
                obs = compute_quark_observables(Yu, Yd)
                if obs["mc"] < 0.01 or obs["mc"] > 500:
                    return 1000.0
                return compute_training_loss(obs)
            except Exception:
                return 1000.0

        try:
            result = differential_evolution(
                objective, GAUSSIAN_BOUNDS, seed=seed, **OPTIMIZER_SETTINGS
            )
        except Exception:
            continue
        if result.fun >= 999:
            continue
        Yu, Yd = compute_yukawas_gaussian(Q, U, D, *result.x)
        obs = compute_quark_observables(Yu, Yd)
        rec = {
            "train": compute_training_loss(obs),
            "holdout": compute_holdout_loss(obs),
            "strict": check_strict_survivor(obs),
            "mc": obs["mc"],
        }
        if best is None or rec["train"] < best["train"]:
            best = rec
    return best


def shuffled_q_geometry(Q, U, D, rng: np.random.RandomState) -> Tuple:
    """Replace Q with an independent sorted triple; keep U,D."""
    new_q = tuple(sorted(rng.choice(range(15), 3, replace=False)))
    return new_q, U, D


def random_yukawa_pair(rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """Haar-random unitaries + log-spaced positive singular values."""
    def one_matrix():
        u = unitary_group.rvs(3, random_state=rng)
        s = np.exp(rng.uniform(-4, 0, 3))
        s = np.sort(s)[::-1]
        return u @ np.diag(s) @ u.conj().T

    return one_matrix(), one_matrix()


def evaluate_haar_baseline(rng: np.random.RandomState) -> Dict:
    best = None
    for _ in range(HAAR_SAMPLES):
        Yu, Yd = random_yukawa_pair(rng)
        obs = compute_quark_observables(Yu, Yd)
        if obs["mc"] < 0.01 or obs["mc"] > 500:
            continue
        rec = {
            "train": compute_training_loss(obs),
            "holdout": compute_holdout_loss(obs),
            "strict": check_strict_survivor(obs),
            "mc": obs["mc"],
        }
        if best is None or rec["train"] < best["train"]:
            best = rec
    return best


def summarize(label: str, records: List[Dict]) -> Dict:
    solved = [r for r in records if r is not None]
    if not solved:
        return {"label": label, "n": 0}
    strict_n = sum(1 for r in solved if r["strict"])
    trains = [r["train"] for r in solved]
    holds = [r["holdout"] for r in solved]
    return {
        "label": label,
        "n": len(solved),
        "strict_n": strict_n,
        "strict_pct": 100.0 * strict_n / len(solved),
        "train_median": float(np.median(trains)),
        "holdout_median": float(np.median(holds)),
        "train_min": min(trains),
    }


def format_report(n_geom: int, kernel_recs, shuffled_recs, haar_recs) -> str:
    ks = summarize("kernel (Gaussian, real Q)", kernel_recs)
    ss = summarize("shuffled Q", shuffled_recs)
    hs = summarize("Haar random Yu/Yd", haar_recs)

    lines = [
        "=" * 78,
        "NULL / SCRAMBLED GEOMETRY BASELINE (diagnostic 31, Tier A4)",
        "=" * 78,
        "",
        f"Geometries: {n_geom} (phenomenology sampler, seed={GEOM_SEED})",
        f"Kernel: Gaussian, training loss, {N_SEEDS} seeds; Haar: {HAAR_SAMPLES} samples/geom",
        f"Strict tolerances: {STRICT_TOLERANCES}",
        "",
    ]

    for s in (ks, ss, hs):
        if s.get("n", 0) == 0:
            lines.append(f"--- {s['label'].upper()} --- no converged records")
            continue
        lines.extend([
            f"--- {s['label'].upper()} (n={s['n']}) ---",
            f"  Strict survivors: {s['strict_n']} ({s['strict_pct']:.1f}%)",
            f"  Train loss min/median: {s['train_min']:.4f} / {s['train_median']:.4f}",
            f"  Holdout loss median: {s['holdout_median']:.4f}",
            "",
        ])

    lines.append("--- A4 INTERPRETATION ---")
    if ks.get("n") and ss.get("n") and hs.get("n"):
        kernel_beats_null = (
            ks["strict_n"] > max(ss["strict_n"], hs["strict_n"])
            or ks["train_median"] < min(ss["train_median"], hs["train_median"]) - 0.1
        )
        if ks["strict_n"] == 0 and ss["strict_n"] == 0 and hs["strict_n"] == 0:
            lines.append(
                "  All conditions 0% strict — kernel geometry does not enable PDG match;"
            )
            lines.append(
                "  null baselines equally fail (no spurious survivor inflation from search)."
            )
        elif kernel_beats_null:
            lines.append(
                "  Kernel on real Q outperforms nulls — discrete geometry carries signal"
            )
            lines.append("  (but may still fail strict survivor protocol).")
        else:
            lines.append(
                "  Null baselines match or beat kernel — apparent fits may not require geometry."
            )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-geom", type=int, default=30)
    parser.add_argument("--smoke", action="store_true", help="N=5 geometries")
    args = parser.parse_args()

    n_geom = 5 if args.smoke else args.n_geom
    geoms = generate_test_geometries(n_geom, GEOM_SEED)
    rng = np.random.RandomState(31031)

    kernel_recs, shuffled_recs, haar_recs = [], [], []
    print(f"Null geometry baseline: {n_geom} geometries...")
    for i, (Q, U, D) in enumerate(geoms):
        if i % 5 == 0:
            print(f"  {i}/{n_geom}")
        kernel_recs.append(optimize_kernel(Q, U, D))
        q2, u2, d2 = shuffled_q_geometry(Q, U, D, rng)
        shuffled_recs.append(optimize_kernel(q2, u2, d2))
        haar_recs.append(evaluate_haar_baseline(rng))

    report = format_report(n_geom, kernel_recs, shuffled_recs, haar_recs)
    print(report)

    if not args.smoke:
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            f.write(report)
        print(f"Saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
