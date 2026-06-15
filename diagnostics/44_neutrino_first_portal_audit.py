#!/usr/bin/env python3
"""
Neutrino-first chiral projection portal (diagnostic 44).

Portal corrections apply **only** to the neutrino Yukawa block; quark/lepton
sectors are not modified in the primary models.

Models (per geometry, diag 28 corpus seed=28028):
  baseline     — diag 28 Gaussian + g_env joint objective
  mirror_eta   — Ynu += portal(Phi_-); eps_p free after baseline theta frozen
  parity_pi    — Ynu += portal(Phi + pi)
  schur_nu     — 6×6 Schur complement on Ynu only (mirror block integrated out)
  leaky        — joint (L,Q,U) geometry: eps_nu_portal AND eps_q_portal free (P2 test)

Pre-registered falsifiers (chiral-projection-formalization-program):
  P2 structure: median eps_nu / eps_q > 5 in leaky model when both free.
  Flavor PASS: best portal median joint_loss < 0.85 * baseline median joint_loss
               AND joint_strict rate not worse than baseline - 5pp.

Usage:
  python diagnostics/44_neutrino_first_portal_audit.py
  python diagnostics/44_neutrino_first_portal_audit.py --smoke
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

from kernel import compute_yukawa_matrix
from observables import (
    NEUTRINO_MASS_TARGETS,
    NEUTRINO_TARGETS,
    compute_neutrino_joint_loss,
    compute_neutrino_mass_loss,
    compute_neutrino_observables,
    compute_pmns_loss,
    compute_training_loss,
    compute_quark_observables,
)
from phenomenology_utils import (
    generate_joint_three_sector_geometries,
    generate_neutrino_geometries,
)

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "44_neutrino_first_portal_audit.txt"
)

N_GEOMETRIES = 100
GEOM_SEED = 28028
N_SEEDS = 4
LEAKY_N_GEOM = 30
LEAKY_GEOM_SEED = 26026
N_LEAKY_SEEDS = 2

OPT = dict(maxiter=120, popsize=12, tol=1e-6, polish=False)
OPT_PORTAL = dict(maxiter=60, popsize=10, tol=1e-6, polish=False)

NEUTRINO_BOUNDS = [
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

JOINT_IMPROVE_RATIO = 0.85
P2_EPS_RATIO = 5.0
STRICT_DROP_TOL_PP = 5.0


def portal_matrix(left, right, sigma_eff, k, alpha, eta, eps_int, mode: str):
    Y = np.zeros((3, 3), dtype=complex)
    left_vec = np.array([left[0], left[1], 0], dtype=float)
    right_vec = np.array(right, dtype=float)
    for i in range(3):
        for j in range(3):
            xi, xj = left_vec[i], right_vec[j]
            diff = xi - xj
            env = np.exp(-(diff**2) / (2 * sigma_eff**2))
            if mode == "mirror_eta":
                phase = alpha + k * (xi + xj) / 2 - eta * diff
            elif mode == "parity_pi":
                phase = alpha + k * (xi + xj) / 2 + eta * diff + np.pi
            else:
                raise ValueError(mode)
            Y[i, j] = eps_int * env * np.exp(1j * phase)
    return Y


def build_yukawas(theta, L, N):
    sigma, k, alpha, eta, eps_nu, eps_e, g_env = theta
    Ynu = compute_yukawa_matrix(L, N, sigma * g_env, k, alpha, eta, eps_nu)
    Ye = compute_yukawa_matrix(L, N, sigma, k, alpha, eta, eps_e)
    return Ynu, Ye


def neutrino_obs_from_theta(theta, L, N):
    Ynu, Ye = build_yukawas(theta, L, N)
    obs = compute_neutrino_observables(Ynu, Ye)
    if obs["theta23"] < 0.01:
        return None
    return obs


def apply_nu_portal(theta, L, N, eps_p, mode: str):
    sigma, k, alpha, eta, eps_nu, eps_e, g_env = theta
    Ynu, Ye = build_yukawas(theta, L, N)
    P = portal_matrix(L, N, sigma * g_env, k, alpha, eta, eps_p, mode)
    return Ynu + P, Ye


def schur_nu_yukawas(theta, L, N, eps_x, rho):
    sigma, k, alpha, eta, eps_nu, eps_e, g_env = theta
    sig = sigma * g_env
    Y_sm = compute_yukawa_matrix(L, N, sig, k, alpha, eta, eps_nu)
    Y_m = compute_yukawa_matrix(L, N, sig, -k, alpha + np.pi, -eta, eps_nu)
    X = portal_matrix(L, N, sig, k, alpha, eta, eps_x, "mirror_eta")
    Ye = compute_yukawa_matrix(L, N, sigma, k, alpha, eta, eps_e)
    try:
        inv_m = np.linalg.inv(Y_m + rho * np.eye(3))
    except np.linalg.LinAlgError:
        return None, None
    Ynu = Y_sm - X @ inv_m @ X.conj().T
    if not np.all(np.isfinite(Ynu)):
        return None, None
    return Ynu, Ye


def check_pmns_strict(rec: dict) -> bool:
    for key, tol in PMNS_STRICT.items():
        t, v = NEUTRINO_TARGETS[key], rec[key]
        if v <= 0 or t <= 0 or abs(v - t) / t > tol:
            return False
    return True


def check_joint_strict(rec: dict) -> bool:
    if not check_pmns_strict(rec):
        return False
    for key, tol in MASS_STRICT.items():
        t, v = NEUTRINO_MASS_TARGETS[key], rec[key]
        if v <= 0 or t <= 0 or abs(v - t) / t > tol:
            return False
    return True


def record_from_obs(obs: dict, joint_loss: float, extra: dict | None = None) -> dict:
    rec = {
        "joint_loss": joint_loss,
        "pmns_loss": compute_pmns_loss(obs),
        "mass_loss": compute_neutrino_mass_loss(obs),
        "theta12": obs["theta12"],
        "theta23": obs["theta23"],
        "theta13": obs["theta13"],
        "dm21": obs["dm21"],
        "dm31": obs["dm31"],
        "g_env": obs.get("g_env", np.nan),
    }
    if extra:
        rec.update(extra)
    rec["pmns_strict"] = check_pmns_strict(rec)
    rec["joint_strict"] = check_joint_strict(rec)
    return rec


def optimize_baseline(L, N, geom_idx: int):
    best = None
    best_theta = None

    def objective(theta):
        obs = neutrino_obs_from_theta(theta, L, N)
        if obs is None:
            return 1000.0
        return compute_neutrino_joint_loss(obs)

    for seed in range(N_SEEDS):
        try:
            res = differential_evolution(
                objective,
                NEUTRINO_BOUNDS,
                seed=seed + geom_idx * 100,
                **OPT,
            )
        except Exception:
            continue
        if res.fun >= 999:
            continue
        obs = neutrino_obs_from_theta(res.x, L, N)
        if obs is None:
            continue
        jl = compute_neutrino_joint_loss(obs)
        if best is None or jl < best["joint_loss"]:
            best = record_from_obs(obs, jl)
            best_theta = res.x.copy()

    return best, best_theta


def optimize_nu_portal(theta_b, L, N, geom_idx: int, mode: str):
    if theta_b is None:
        return None

    def objective(eps_p):
        Ynu, Ye = apply_nu_portal(theta_b, L, N, float(eps_p[0]), mode)
        obs = compute_neutrino_observables(Ynu, Ye)
        if obs["theta23"] < 0.01:
            return 1000.0
        return compute_neutrino_joint_loss(obs)

    best = None
    for seed in range(N_SEEDS):
        try:
            res = differential_evolution(
                objective,
                [(0.0, 0.5)],
                seed=seed + geom_idx * 100 + 17,
                **OPT_PORTAL,
            )
        except Exception:
            continue
        if res.fun >= 999:
            continue
        Ynu, Ye = apply_nu_portal(theta_b, L, N, float(res.x[0]), mode)
        obs = compute_neutrino_observables(Ynu, Ye)
        jl = compute_neutrino_joint_loss(obs)
        if best is None or jl < best["joint_loss"]:
            best = record_from_obs(obs, jl, {"eps_p": float(res.x[0])})
    return best


def optimize_schur_nu(theta_b, L, N, geom_idx: int):
    if theta_b is None:
        return None

    def objective(x):
        eps_x, rho = float(x[0]), float(x[1])
        Ynu, Ye = schur_nu_yukawas(theta_b, L, N, eps_x, rho)
        if Ynu is None:
            return 1000.0
        obs = compute_neutrino_observables(Ynu, Ye)
        if obs["theta23"] < 0.01:
            return 1000.0
        return compute_neutrino_joint_loss(obs)

    best = None
    for seed in range(N_SEEDS):
        try:
            res = differential_evolution(
                objective,
                [(0.0, 0.3), (0.05, 2.0)],
                seed=seed + geom_idx * 100 + 31,
                **OPT_PORTAL,
            )
        except Exception:
            continue
        if res.fun >= 999:
            continue
        Ynu, Ye = schur_nu_yukawas(theta_b, L, N, float(res.x[0]), float(res.x[1]))
        if Ynu is None:
            continue
        obs = compute_neutrino_observables(Ynu, Ye)
        jl = compute_neutrino_joint_loss(obs)
        if best is None or jl < best["joint_loss"]:
            best = record_from_obs(
                obs, jl, {"eps_x": float(res.x[0]), "rho": float(res.x[1])}
            )
    return best


def optimize_leaky(geom, geom_idx: int):
    """P2: both eps_nu_portal and eps_q_portal free on joint geometry."""
    L, E, N, U, D = geom.L, geom.E, geom.N, geom.U, geom.D

    def objective(theta):
        sigma, k, alpha, eta, eps_nu, eps_e, g_env, eps_nu_p, eps_q_p = theta
        Ynu, Ye = build_yukawas(
            np.array([sigma, k, alpha, eta, eps_nu, eps_e, g_env]), L, N
        )
        Pnu = portal_matrix(L, N, sigma * g_env, k, alpha, eta, eps_nu_p, "mirror_eta")
        Ynu = Ynu + Pnu
        obs_nu = compute_neutrino_observables(Ynu, Ye)
        if obs_nu["theta23"] < 0.01:
            return 1000.0
        loss_nu = compute_neutrino_joint_loss(obs_nu)

        Yu = compute_yukawa_matrix(L, U, sigma, k, alpha, eta, 0.1)
        Yd = compute_yukawa_matrix(L, D, sigma, k, alpha, eta, 0.1)
        Pu = portal_matrix(L, U, sigma, k, alpha, eta, eps_q_p, "mirror_eta")
        Pd = portal_matrix(L, D, sigma, k, alpha, eta, eps_q_p, "mirror_eta")
        obs_q = compute_quark_observables(Yu + Pu, Yd + Pd)
        loss_q = compute_training_loss(obs_q)
        return loss_nu + 0.1 * loss_q

    bounds = NEUTRINO_BOUNDS + [(0.0, 0.5), (0.0, 0.5)]
    best = None
    for seed in range(N_LEAKY_SEEDS):
        try:
            res = differential_evolution(
                objective,
                bounds,
                seed=seed + geom_idx * 100 + 53,
                **OPT,
            )
        except Exception:
            continue
        if res.fun >= 999:
            continue
        eps_nu_p, eps_q_p = float(res.x[7]), float(res.x[8])
        rec = {
            "eps_nu_p": eps_nu_p,
            "eps_q_p": eps_q_p,
            "eps_ratio": eps_nu_p / max(eps_q_p, 1e-6),
            "joint_loss": res.fun,
        }
        if best is None or res.fun < best["joint_loss"]:
            best = rec
    return best


def run_audit(smoke: bool):
    t0 = time.time()
    n_geom = 10 if smoke else N_GEOMETRIES
    leaky_n = 5 if smoke else LEAKY_N_GEOM

    geoms = generate_neutrino_geometries(n_geom, GEOM_SEED)
    models = ["baseline", "mirror_eta", "parity_pi", "schur_nu"]
    results = {m: [] for m in models}

    for gi, (L, N) in enumerate(geoms):
        base, theta_b = optimize_baseline(L, N, gi)
        if base is not None:
            results["baseline"].append(base)
        for mode, key in [("mirror_eta", "mirror_eta"), ("parity_pi", "parity_pi")]:
            rec = optimize_nu_portal(theta_b, L, N, gi, mode)
            if rec is not None:
                results[key].append(rec)
        rec_s = optimize_schur_nu(theta_b, L, N, gi)
        if rec_s is not None:
            results["schur_nu"].append(rec_s)

    leaky_geoms = generate_joint_three_sector_geometries(leaky_n, LEAKY_GEOM_SEED)
    leaky_recs = []
    for gi, geom in enumerate(leaky_geoms):
        rec = optimize_leaky(geom, gi)
        if rec is not None:
            leaky_recs.append(rec)

    lines = [
        "Diagnostic 44: Neutrino-first chiral projection portal",
        f"N_geom={n_geom} (seed={GEOM_SEED}), leaky N={leaky_n} (seed={LEAKY_GEOM_SEED})",
        f"Wall time: {(time.time() - t0) / 60:.1f} min",
        "",
        "--- Median joint loss / joint strict % ---",
    ]

    base_losses = [r["joint_loss"] for r in results["baseline"]]
    base_median = float(np.median(base_losses)) if base_losses else float("nan")
    base_strict = (
        100.0 * sum(r["joint_strict"] for r in results["baseline"]) / len(results["baseline"])
        if results["baseline"]
        else 0.0
    )

    best_name = "baseline"
    best_median = base_median
    for m in models:
        recs = results[m]
        if not recs:
            lines.append(f"  {m:12s}  n=0")
            continue
        med = float(np.median([r["joint_loss"] for r in recs]))
        strict = 100.0 * sum(r["joint_strict"] for r in recs) / len(recs)
        lines.append(
            f"  {m:12s}  n={len(recs):3d}  joint_med={med:.4f}  strict={strict:.1f}%"
        )
        if m != "baseline" and med < best_median:
            best_median = med
            best_name = m

    lines.extend(
        [
            "",
            f"Baseline reference: joint_med={base_median:.4f}  strict={base_strict:.1f}%",
            f"Best portal model: {best_name} (joint_med={best_median:.4f})",
            "",
            "--- P2 leaky (eps_nu vs eps_q, both free) ---",
        ]
    )

    if leaky_recs:
        med_nu = float(np.median([r["eps_nu_p"] for r in leaky_recs]))
        med_q = float(np.median([r["eps_q_p"] for r in leaky_recs]))
        med_ratio = float(np.median([r["eps_ratio"] for r in leaky_recs]))
        p2_pass = med_ratio > P2_EPS_RATIO
        lines.append(f"  n={len(leaky_recs)}  median eps_nu_p={med_nu:.4f}  eps_q_p={med_q:.4f}")
        lines.append(f"  median eps_nu/eps_q={med_ratio:.2f}  (P2 threshold > {P2_EPS_RATIO})")
        lines.append(f"  P2 structure: {p2_pass}")
    else:
        p2_pass = False
        lines.append("  No leaky solutions.")

    flavor_pass = (
        best_name != "baseline"
        and best_median < JOINT_IMPROVE_RATIO * base_median
        and results[best_name]
        and (
            100.0 * sum(r["joint_strict"] for r in results[best_name]) / len(results[best_name])
            >= base_strict - STRICT_DROP_TOL_PP
        )
    )

    lines.extend(
        [
            "",
            "--- Pre-registered ---",
            f"  Flavor PASS (joint_med < {JOINT_IMPROVE_RATIO}× baseline, strict within -{STRICT_DROP_TOL_PP}pp): {flavor_pass}",
            "",
        ]
    )

    if flavor_pass:
        verdict = f"WEAK SIGNAL: {best_name} improves neutrino joint objective"
        tag = "weak_signal"
    else:
        verdict = "FALSIFIER PASS — neutrino-first portal does not beat diag 28 baseline"
        tag = "falsifier_pass"
    lines.append(f"VERDICT: {verdict}")
    lines.append(f"verdict: {tag}")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    text = "\n".join(lines)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_audit(args.smoke)
