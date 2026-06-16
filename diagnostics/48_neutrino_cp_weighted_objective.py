#!/usr/bin/env python3
"""
Diagnostic 48 - Neutrino signed-J CP weighted objective.

Purpose:
  Test one pre-registered CP-aware objective after diag 47 established that raw
  delta_PMNS is rephasing-sensitive and signed J_PMNS is the invariant CP target.

Objective:
  L = L_mass(Delta m^2) + 5 * L_PMNS + w_CP * L_CP(J_PMNS)

Pass bar (full N=100):
  - joint strict >= 22/100 attempted (no degradation vs diag 28)
  - median signed-J relative error < 0.50
  - median PMNS and mass losses do not worsen vs diag 47 baselines
  - CP sign-match rate materially exceeds chance

Usage:
  python diagnostics/48_neutrino_cp_weighted_objective.py --smoke
  python diagnostics/48_neutrino_cp_weighted_objective.py --smoke --cp-weight 0.5
  python diagnostics/48_neutrino_cp_weighted_objective.py
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.optimize import differential_evolution

from kernel import compute_yukawa_matrix
from observables import (
    NEUTRINO_MASS_TARGETS,
    NEUTRINO_TARGETS,
    compute_neutrino_cp_joint_loss,
    compute_neutrino_joint_loss,
    compute_neutrino_mass_loss,
    compute_neutrino_observables,
    compute_pmns_cp_loss,
    compute_pmns_loss,
    target_pmns_jarlskog,
)
from phenomenology_utils import generate_neutrino_geometries


GEOM_SEED = 28028
N_GEOM = 100
N_SEEDS = 4
CP_WEIGHT = 1.0
PMNS_WEIGHT = 5.0
OPT = dict(
    maxiter=120,
    popsize=12,
    tol=1e-6,
    mutation=(0.5, 1.0),
    recombination=0.7,
    polish=False,
)
NU_BOUNDS = [
    (0.5, 6.0),
    (0.1, 2.0),
    (0.0, 2 * np.pi),
    (1.0, 5.0),
    (0.01, 0.5),
    (0.01, 0.5),
    (0.45, 0.75),
]

PMNS_STRICT = {"theta12": 0.15, "theta23": 0.15, "theta13": 0.20}
MASS_STRICT = {"dm21": 0.30, "dm31": 0.30}

# Pre-registered no-degradation bar from diag 28 (22/100 attempted).
DIAG28_JOINT_STRICT_ATTEMPTED = 22

# Full diag 47 audit baselines on the same N=100 corpus.
DIAG47_JOINT_STRICT_ATTEMPTED = 24
BASELINE_MASS_MEDIAN = 0.026755
BASELINE_PMNS_MEDIAN = 0.057684
BASELINE_SIGNED_J_REL = 1.0712
BASELINE_ABS_J_REL = 0.8435

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "48_neutrino_cp_weighted_objective.txt"
)


def results_path_for_weight(cp_weight: float) -> str:
    """Keep exploratory non-default weights from overwriting the registered w=1 result."""
    if abs(cp_weight - CP_WEIGHT) < 1e-12:
        return RESULTS_PATH
    label = f"{cp_weight:g}".replace(".", "p").replace("-", "m")
    return os.path.join(
        os.path.dirname(__file__),
        "results",
        f"48_neutrino_cp_weighted_objective_w{label}.txt",
    )


def check_pmns_strict(obs: Dict[str, float]) -> bool:
    for key, tol in PMNS_STRICT.items():
        value = obs.get(key, 0.0)
        target = NEUTRINO_TARGETS[key]
        if value <= 0 or abs(value - target) / target > tol:
            return False
    return True


def check_joint_strict(obs: Dict[str, float]) -> bool:
    if not check_pmns_strict(obs):
        return False
    for key, tol in MASS_STRICT.items():
        value = obs.get(key, 0.0)
        target = NEUTRINO_MASS_TARGETS[key]
        if value <= 0 or abs(value - target) / target > tol:
            return False
    return True


def best_record_for_geom(
    L: Tuple,
    N: Tuple,
    geom_idx: int,
    cp_weight: float,
) -> Dict | None:
    def objective(theta):
        try:
            sigma, k, alpha, eta, eps_nu, eps_e, g_env = theta
            Ynu = compute_yukawa_matrix(L, N, sigma * g_env, k, alpha, eta, eps_nu)
            Ye = compute_yukawa_matrix(L, N, sigma, k, alpha, eta, eps_e)
            obs = compute_neutrino_observables(Ynu, Ye)
            if obs["theta23"] < 0.01:
                return 1000.0
            return compute_neutrino_cp_joint_loss(
                obs, pmns_weight=PMNS_WEIGHT, cp_weight=cp_weight
            )
        except Exception:
            return 1000.0

    best = None
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
        rec = {
            **obs,
            "geom": geom_idx,
            "seed": seed,
            "sigma": float(sigma),
            "k": float(k),
            "alpha": float(alpha),
            "eta": float(eta),
            "eps_nu": float(eps_nu),
            "eps_e": float(eps_e),
            "g_env": float(g_env),
            "cp_joint_loss": compute_neutrino_cp_joint_loss(
                obs, pmns_weight=PMNS_WEIGHT, cp_weight=cp_weight
            ),
            "baseline_joint_loss": compute_neutrino_joint_loss(obs),
            "mass_loss": compute_neutrino_mass_loss(obs),
            "pmns_loss": compute_pmns_loss(obs),
            "cp_loss": compute_pmns_cp_loss(obs),
        }
        if best is None or rec["cp_joint_loss"] < best["cp_joint_loss"]:
            best = rec

    if best is not None:
        best["pmns_strict"] = check_pmns_strict(best)
        best["joint_strict"] = check_joint_strict(best)
    return best


def run_pool(n_geom: int, cp_weight: float) -> Dict:
    geoms = generate_neutrino_geometries(n_geom, GEOM_SEED)
    records: List[Dict] = []
    for gi, (L, N) in enumerate(geoms):
        print(
            f"[diag48] optimizing geometry {gi + 1}/{n_geom} (cp_weight={cp_weight:g})",
            flush=True,
        )
        rec = best_record_for_geom(L, N, gi, cp_weight)
        if rec is not None:
            records.append(rec)
    return {"records": records, "requested": n_geom, "cp_weight": cp_weight}


def summarize(result: Dict) -> Dict[str, float]:
    records = result["records"]
    target_j = target_pmns_jarlskog()
    if not records:
        return {"target_j": target_j, "requested": float(result["requested"]), "solved": 0.0}

    js = np.array([r["J_PMNS"] for r in records], dtype=float)
    jabs = np.abs(js)
    target_abs = abs(target_j)
    signed_rel = np.abs(js - target_j) / target_abs
    abs_rel = np.abs(jabs - target_abs) / target_abs
    sign_matches = np.sign(js) == np.sign(target_j)

    return {
        "target_j": float(target_j),
        "requested": float(result["requested"]),
        "cp_weight": float(result["cp_weight"]),
        "solved": float(len(records)),
        "pmns_strict": float(sum(bool(r["pmns_strict"]) for r in records)),
        "joint_strict": float(sum(bool(r["joint_strict"]) for r in records)),
        "median_cp_joint_loss": float(np.median([r["cp_joint_loss"] for r in records])),
        "median_baseline_joint_loss": float(
            np.median([r["baseline_joint_loss"] for r in records])
        ),
        "median_mass_loss": float(np.median([r["mass_loss"] for r in records])),
        "median_pmns_loss": float(np.median([r["pmns_loss"] for r in records])),
        "median_cp_loss": float(np.median([r["cp_loss"] for r in records])),
        "median_j": float(np.median(js)),
        "median_abs_j": float(np.median(jabs)),
        "median_signed_j_rel_err": float(np.median(signed_rel)),
        "median_abs_j_rel_err": float(np.median(abs_rel)),
        "sign_match_rate": float(np.mean(sign_matches)),
        "passes_signed_j_bar": float(np.median(signed_rel) < 0.50),
        "passes_joint_strict_bar": float(
            sum(bool(r["joint_strict"]) for r in records) >= DIAG28_JOINT_STRICT_ATTEMPTED
            if result["requested"] >= 100
            else 0.0
        ),
        "passes_diag47_strict_bar": float(
            sum(bool(r["joint_strict"]) for r in records) >= DIAG47_JOINT_STRICT_ATTEMPTED
            if result["requested"] >= 100
            else 0.0
        ),
        "improves_signed_j_vs_diag47": float(np.median(signed_rel) < BASELINE_SIGNED_J_REL),
        "improves_abs_j_vs_diag47": float(np.median(abs_rel) < BASELINE_ABS_J_REL),
    }


def format_report(summary: Dict[str, float], smoke: bool) -> str:
    lines = [
        "=" * 78,
        "NEUTRINO CP-WEIGHTED OBJECTIVE (diagnostic 48)",
        "=" * 78,
        "",
        f"Objective: L_mass + 5*L_PMNS + {summary.get('cp_weight', CP_WEIGHT):g}*L_CP(signed J_PMNS)",
        f"Mode: {'SMOKE' if smoke else 'FULL'}",
        f"Geometries requested: {int(summary.get('requested', 0))}",
        f"Geometries solved: {int(summary.get('solved', 0))}",
        f"Target signed J_PMNS: {summary['target_j']:.8f}",
    ]
    if summary.get("solved", 0) == 0:
        lines.append("No converged records.")
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "--- SUMMARY ---",
            f"PMNS strict: {int(summary['pmns_strict'])}/{int(summary['solved'])}",
            f"Joint strict: {int(summary['joint_strict'])}/{int(summary['solved'])}",
            f"Median CP-joint loss: {summary['median_cp_joint_loss']:.6f}",
            f"Median baseline joint loss: {summary['median_baseline_joint_loss']:.6f}",
            f"Median mass loss: {summary['median_mass_loss']:.6f}",
            f"Median PMNS loss: {summary['median_pmns_loss']:.6f}",
            f"Median CP loss: {summary['median_cp_loss']:.6f}",
            f"Median signed J_PMNS: {summary['median_j']:.8f}",
            f"Median |J_PMNS|: {summary['median_abs_j']:.8f}",
            f"Median signed-J rel err: {summary['median_signed_j_rel_err']:.4f}",
            f"Median |J|-magnitude rel err: {summary['median_abs_j_rel_err']:.4f}",
            f"CP sign-match rate: {summary['sign_match_rate']:.3f}",
            "",
            "--- DIAG 47 BASELINE COMPARISON ---",
            f"diag47 signed-J rel err: {BASELINE_SIGNED_J_REL:.4f}",
            f"diag47 |J| rel err: {BASELINE_ABS_J_REL:.4f}",
            f"diag47 mass median: {BASELINE_MASS_MEDIAN:.6f}",
            f"diag47 PMNS median: {BASELINE_PMNS_MEDIAN:.6f}",
            f"Improves signed-J vs diag47: {bool(summary['improves_signed_j_vs_diag47'])}",
            f"Improves |J| vs diag47: {bool(summary['improves_abs_j_vs_diag47'])}",
            f"Passes signed-J <50% bar: {bool(summary['passes_signed_j_bar'])}",
        ]
    )
    if not smoke:
        lines.append(
            f"Passes diag28 joint strict >=22/100 bar: {bool(summary['passes_joint_strict_bar'])}"
        )
        lines.append(
            f"Passes diag47 joint strict >=24/100 bar: {bool(summary['passes_diag47_strict_bar'])}"
        )
    else:
        lines.append("Full-N joint strict bar not evaluated in smoke mode.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run 5 geometries only")
    parser.add_argument("--n-geom", type=int, default=None, help="Override geometry count")
    parser.add_argument("--cp-weight", type=float, default=CP_WEIGHT, help="Signed-J CP loss weight")
    args = parser.parse_args()

    n_geom = args.n_geom if args.n_geom is not None else (5 if args.smoke else N_GEOM)
    result = run_pool(n_geom, args.cp_weight)
    summary = summarize(result)
    report = format_report(summary, args.smoke)
    print(report)

    if not args.smoke:
        results_path = results_path_for_weight(args.cp_weight)
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"Saved: {results_path}")


if __name__ == "__main__":
    main()
