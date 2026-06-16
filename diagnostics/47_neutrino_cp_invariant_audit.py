#!/usr/bin/env python3
"""
Diagnostic 47 - Neutrino CP invariant audit.

Purpose:
  47A: Show that raw delta_PMNS = -arg(U_e3) is rephasing-sensitive unless the
       PMNS matrix is first put in a PDG gauge.
  47B: Re-audit the diag-28 joint-objective pool using signed J_PMNS, which is
       invariant under row/column rephasings.

No CP-targeted optimization is performed here. This is the protocol gate before
any CP-aware objective or seesaw-style readout is allowed.

Usage:
  python diagnostics/47_neutrino_cp_invariant_audit.py --smoke
  python diagnostics/47_neutrino_cp_invariant_audit.py
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
    NEUTRINO_CP_TARGETS,
    NEUTRINO_MASS_TARGETS,
    NEUTRINO_TARGETS,
    compute_neutrino_joint_loss,
    compute_neutrino_mass_loss,
    compute_neutrino_observables,
    compute_pmns_loss,
    cp_phase_delta_from_unitary,
    jarlskog_invariant,
    pmns_angles_from_unitary,
    target_pmns_jarlskog,
)
from phenomenology_utils import generate_neutrino_geometries


GEOM_SEED = 28028
N_GEOM = 100
N_SEEDS = 4
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

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "47_neutrino_cp_invariant_audit.txt"
)


def pdg_pmns_matrix(theta12: float, theta23: float, theta13: float, delta: float) -> np.ndarray:
    c12, s12 = np.cos(theta12), np.sin(theta12)
    c23, s23 = np.cos(theta23), np.sin(theta23)
    c13, s13 = np.cos(theta13), np.sin(theta13)
    return np.array(
        [
            [c12 * c13, s12 * c13, s13 * np.exp(-1j * delta)],
            [
                -s12 * c23 - c12 * s23 * s13 * np.exp(1j * delta),
                c12 * c23 - s12 * s23 * s13 * np.exp(1j * delta),
                s23 * c13,
            ],
            [
                s12 * s23 - c12 * c23 * s13 * np.exp(1j * delta),
                -c12 * s23 - s12 * c23 * s13 * np.exp(1j * delta),
                c23 * c13,
            ],
        ],
        dtype=complex,
    )


def circular_distance(a: float, b: float) -> float:
    return float(abs(np.arctan2(np.sin(a - b), np.cos(a - b))))


def rephase_unitary(U: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    row = np.exp(1j * rng.uniform(-np.pi, np.pi, size=3))
    col = np.exp(1j * rng.uniform(-np.pi, np.pi, size=3))
    return np.diag(row) @ U @ np.diag(col)


def convention_audit(n_samples: int = 200) -> Dict[str, float]:
    target_delta = NEUTRINO_CP_TARGETS["delta_PMNS"]
    U0 = pdg_pmns_matrix(
        NEUTRINO_TARGETS["theta12"],
        NEUTRINO_TARGETS["theta23"],
        NEUTRINO_TARGETS["theta13"],
        target_delta,
    )
    target_j = target_pmns_jarlskog()
    base_angles = pmns_angles_from_unitary(U0)
    rng = np.random.default_rng(47047)

    j_drifts, angle_drifts, raw_delta_errors = [], [], []
    for _ in range(n_samples):
        U = rephase_unitary(U0, rng)
        j_drifts.append(abs(jarlskog_invariant(U) - target_j))
        got_angles = pmns_angles_from_unitary(U)
        angle_drifts.append(max(abs(a - b) for a, b in zip(got_angles, base_angles)))
        raw_delta_errors.append(
            circular_distance(cp_phase_delta_from_unitary(U), target_delta)
        )

    return {
        "target_j": float(target_j),
        "max_j_drift": float(max(j_drifts)),
        "max_angle_drift": float(max(angle_drifts)),
        "median_raw_delta_error": float(np.median(raw_delta_errors)),
        "max_raw_delta_error": float(max(raw_delta_errors)),
        "j_invariant_pass": float(max(j_drifts) < 1e-12),
        "angles_invariant_pass": float(max(angle_drifts) < 1e-12),
        "raw_delta_unstable": float(np.median(raw_delta_errors) > 0.1),
    }


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


def best_record_for_geom(L: Tuple, N: Tuple, geom_idx: int) -> Dict | None:
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
            "joint_loss": compute_neutrino_joint_loss(obs),
            "pmns_loss": compute_pmns_loss(obs),
            "mass_loss": compute_neutrino_mass_loss(obs),
        }
        if best is None or rec["joint_loss"] < best["joint_loss"]:
            best = rec

    if best is not None:
        best["pmns_strict"] = check_pmns_strict(best)
        best["joint_strict"] = check_joint_strict(best)
    return best


def pool_audit(n_geom: int) -> Dict:
    geoms = generate_neutrino_geometries(n_geom, GEOM_SEED)
    records: List[Dict] = []
    for gi, (L, N) in enumerate(geoms):
        print(f"[diag47] optimizing geometry {gi + 1}/{n_geom}", flush=True)
        rec = best_record_for_geom(L, N, gi)
        if rec is not None:
            records.append(rec)
    return {"records": records, "requested": n_geom}


def summarize_pool(result: Dict) -> Dict[str, float]:
    records = result["records"]
    target_j = target_pmns_jarlskog()
    target_delta = NEUTRINO_CP_TARGETS["delta_PMNS"]
    if not records:
        return {"target_j": target_j}

    js = np.array([r["J_PMNS"] for r in records], dtype=float)
    jabs = np.abs(js)
    target_abs = abs(target_j)
    signed_rel = np.abs(js - target_j) / target_abs
    abs_rel = np.abs(jabs - target_abs) / target_abs
    raw_delta_err = np.array(
        [circular_distance(r["delta_PMNS"], target_delta) for r in records],
        dtype=float,
    )
    sign_matches = np.sign(js) == np.sign(target_j)

    return {
        "target_j": float(target_j),
        "solved": float(len(records)),
        "requested": float(result["requested"]),
        "pmns_strict": float(sum(bool(r["pmns_strict"]) for r in records)),
        "joint_strict": float(sum(bool(r["joint_strict"]) for r in records)),
        "median_joint_loss": float(np.median([r["joint_loss"] for r in records])),
        "median_mass_loss": float(np.median([r["mass_loss"] for r in records])),
        "median_pmns_loss": float(np.median([r["pmns_loss"] for r in records])),
        "median_j": float(np.median(js)),
        "median_abs_j": float(np.median(jabs)),
        "median_signed_j_rel_err": float(np.median(signed_rel)),
        "median_abs_j_rel_err": float(np.median(abs_rel)),
        "sign_match_rate": float(np.mean(sign_matches)),
        "median_raw_delta": float(np.median([r["delta_PMNS"] for r in records])),
        "median_raw_delta_error": float(np.median(raw_delta_err)),
        "abs_j_close_50pct": float(np.median(abs_rel) < 0.5),
        "signed_j_close_50pct": float(np.median(signed_rel) < 0.5),
    }


def format_report(conv: Dict[str, float], pool: Dict, smoke: bool) -> str:
    lines = [
        "=" * 78,
        "NEUTRINO CP INVARIANT AUDIT (diagnostic 47)",
        "=" * 78,
        "",
        "Scope: 47A convention audit + 47B signed J_PMNS audit on diag-28 protocol.",
        "No CP-targeted optimization is performed.",
        f"Mode: {'SMOKE' if smoke else 'FULL'}",
        "",
        "--- 47A CONVENTION AUDIT ---",
        f"Target delta_PMNS: {NEUTRINO_CP_TARGETS['delta_PMNS']:.6f} rad",
        f"Target signed J_PMNS: {conv['target_j']:.8f}",
        f"Max |J rephase drift|: {conv['max_j_drift']:.3e}",
        f"Max angle rephase drift: {conv['max_angle_drift']:.3e}",
        f"Median raw-delta rephase error: {conv['median_raw_delta_error']:.6f} rad",
        f"Max raw-delta rephase error: {conv['max_raw_delta_error']:.6f} rad",
        f"J invariant pass: {bool(conv['j_invariant_pass'])}",
        f"Angles invariant pass: {bool(conv['angles_invariant_pass'])}",
        f"Raw delta unstable under rephase: {bool(conv['raw_delta_unstable'])}",
        "",
        "--- 47B DIAG-28 POOL AUDIT ---",
        f"Geometries requested: {int(pool.get('requested', 0))}",
        f"Geometries solved: {int(pool.get('solved', 0))}",
    ]

    if pool.get("solved", 0) == 0:
        lines.append("No converged records.")
        return "\n".join(lines)

    solved = pool["solved"]
    lines.extend(
        [
            f"PMNS strict: {int(pool['pmns_strict'])}/{int(solved)}",
            f"Joint strict: {int(pool['joint_strict'])}/{int(solved)}",
            f"Median joint loss: {pool['median_joint_loss']:.6f}",
            f"Median mass loss: {pool['median_mass_loss']:.6f}",
            f"Median PMNS loss: {pool['median_pmns_loss']:.6f}",
            f"Target signed J_PMNS: {pool['target_j']:.8f}",
            f"Median signed J_PMNS: {pool['median_j']:.8f}",
            f"Median |J_PMNS|: {pool['median_abs_j']:.8f}",
            f"Median signed-J rel err: {pool['median_signed_j_rel_err']:.4f}",
            f"Median |J|-magnitude rel err: {pool['median_abs_j_rel_err']:.4f}",
            f"CP sign-match rate: {pool['sign_match_rate']:.3f}",
            f"Median raw delta_PMNS: {pool['median_raw_delta']:.6f} rad",
            f"Median raw-delta circular error: {pool['median_raw_delta_error']:.6f} rad",
            f"|J| within 50% bar: {bool(pool['abs_j_close_50pct'])}",
            f"Signed J within 50% bar: {bool(pool['signed_j_close_50pct'])}",
            "",
            "--- INTERPRETATION ---",
        ]
    )
    if bool(pool["abs_j_close_50pct"]) and not bool(pool["signed_j_close_50pct"]):
        lines.append(
            "CP magnitude is not obviously dead, but the signed invariant is misaligned."
        )
    elif bool(pool["signed_j_close_50pct"]):
        lines.append("Signed CP invariant is already near target under the joint objective.")
    else:
        lines.append("Both CP magnitude and signed invariant miss the 50% audit bar.")
    lines.append(
        "Use signed J_PMNS, not raw delta_PMNS, as the next CP optimization target."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run 5 geometries only")
    parser.add_argument("--n-geom", type=int, default=None, help="Override geometry count")
    args = parser.parse_args()

    n_geom = args.n_geom if args.n_geom is not None else (5 if args.smoke else N_GEOM)
    conv = convention_audit()
    result = pool_audit(n_geom)
    pool = summarize_pool(result)
    report = format_report(conv, pool, args.smoke)
    print(report)

    if not args.smoke:
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"Saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
