#!/usr/bin/env python3
"""
Chiral projection portal audit (diagnostic 42).

Tests speculative formalizations where a mirror/oriented projection contributes
a portal correction to Yukawa matrices:

  F1 — Parity phase portal:
       Y_ij -> Y_ij + eps_p * envelope_ij * exp(i(Phi_ij + pi))

  F2 — Mirror coordinate flip (eta -> -eta in auxiliary term):
       Y_ij -> Y_ij + eps_p * env_ij * (1 + eps_m * exp(i Phi_mirror_ij))
       Phi_mirror = alpha + k*(xi+xj)/2 - eta*(xi-xj)

  F3 — 6x6 block Schur complement (integrate out mirror 3x3):
       Y_eff = Y_SM - Y_X @ inv(Y_mirror + rho I) @ Y_X^dag
       with Y_X = eps_X * envelope * exp(i Phi), rank-1 style portal block.

Pre-registered falsifier:
  If best portal model does not improve median |J - J_PDG|/J_PDG on holdout
  while keeping train loss within 1.2x baseline, portal ansatz is not a useful
  handhold for quark CP (does not kill mirror-projection program globally).

Usage:
  python diagnostics/42_chiral_projection_portal_audit.py
  python diagnostics/42_chiral_projection_portal_audit.py --smoke
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy.optimize import differential_evolution

from kernel import compute_yukawa_matrix
from observables import (
    QUARK_CP_TARGETS,
    compute_holdout_loss,
    compute_quark_observables,
    compute_training_loss,
)


RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "42_chiral_projection_portal_audit.txt"
)

OPT = dict(maxiter=80, popsize=10, tol=1e-6, polish=False, seed=42)
N_GEOM = 20
GEOM_SEED = 42042
N_SEEDS = 3


def generate_geometries(n: int, seed: int):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        Q = tuple(sorted(rng.choice(range(15), 3, replace=False)))
        U = tuple(sorted(rng.choice(range(15), 3, replace=False)))
        D = tuple(sorted(rng.choice(range(15), 3, replace=False)))
        out.append((Q, U, D))
    return out


def baseline_yukawas(theta, Q, U, D):
    sigma, k, alpha, eta, eps_u, eps_d = theta
    Yu = compute_yukawa_matrix(Q, U, sigma, k, alpha, eta, eps_u)
    Yd = compute_yukawa_matrix(Q, D, sigma, k, alpha, eta, eps_d)
    return Yu, Yd


def portal_matrix(left, right, sigma, k, alpha, eta, eps_int, mode: str):
    """Build portal correction matrix (same geometry as kernel)."""
    Y = np.zeros((3, 3), dtype=complex)
    left_vec = np.array([left[0], left[1], 0], dtype=float)
    right_vec = np.array(right, dtype=float)
    for i in range(3):
        for j in range(3):
            xi, xj = left_vec[i], right_vec[j]
            diff = xi - xj
            env = np.exp(-diff**2 / (2 * sigma**2))
            if mode == "parity_pi":
                phase = alpha + k * (xi + xj) / 2 + eta * diff + np.pi
            elif mode == "mirror_eta":
                phase = alpha + k * (xi + xj) / 2 - eta * diff
            else:
                raise ValueError(mode)
            Y[i, j] = eps_int * env * np.exp(1j * phase)
    return Y


def apply_portal(Yu, Yd, Q, U, D, theta_base, portal_theta, mode: str):
    sigma, k, alpha, eta, eps_u, eps_d = theta_base
    eps_p_u, eps_p_d = portal_theta
    Pu = portal_matrix(Q, U, sigma, k, alpha, eta, eps_p_u, mode)
    Pd = portal_matrix(Q, D, sigma, k, alpha, eta, eps_p_d, mode)
    return Yu + Pu, Yd + Pd


def schur_yukawas(theta, Q, U, D):
    """6x6 block with shared SM part + portal off-diagonal; integrate out mirror."""
    sigma, k, alpha, eta, eps_u, eps_d, eps_x_u, eps_x_d, rho = theta
    Yu_sm = compute_yukawa_matrix(Q, U, sigma, k, alpha, eta, eps_u)
    Yd_sm = compute_yukawa_matrix(Q, D, sigma, k, alpha, eta, eps_d)
    Yu_m = compute_yukawa_matrix(Q, U, sigma, -k, alpha + np.pi, -eta, eps_u)
    Yd_m = compute_yukawa_matrix(Q, D, sigma, -k, alpha + np.pi, -eta, eps_d)
    Xu = portal_matrix(Q, U, sigma, k, alpha, eta, eps_x_u, "mirror_eta")
    Xd = portal_matrix(Q, D, sigma, k, alpha, eta, eps_x_d, "mirror_eta")
    inv_u = np.linalg.inv(Yu_m + rho * np.eye(3))
    inv_d = np.linalg.inv(Yd_m + rho * np.eye(3))
    Yu_eff = Yu_sm - Xu @ inv_u @ Xu.conj().T
    Yd_eff = Yd_sm - Xd @ inv_d @ Xd.conj().T
    return Yu_eff, Yd_eff


def optimize_baseline(Q, U, D, seed: int):
    bounds = [
        (0.5, 6.0), (0.1, 2.0), (0.0, 2 * np.pi), (1.0, 5.0), (0.01, 0.5), (0.01, 0.5),
    ]

    def obj(theta):
        Yu, Yd = baseline_yukawas(theta, Q, U, D)
        obs = compute_quark_observables(Yu, Yd)
        return compute_training_loss(obs)

    res = differential_evolution(obj, bounds, **{**OPT, "seed": seed})
    Yu, Yd = baseline_yukawas(res.x, Q, U, D)
    obs = compute_quark_observables(Yu, Yd)
    return res.x, obs, Yu, Yd


def optimize_portal(Q, U, D, seed: int, mode: str, theta_base):
    bounds = [(0.0, 0.3), (0.0, 0.3)]

    def obj(portal_theta):
        Yu, Yd = baseline_yukawas(theta_base, Q, U, D)
        Yu, Yd = apply_portal(Yu, Yd, Q, U, D, theta_base, portal_theta, mode)
        obs = compute_quark_observables(Yu, Yd)
        return compute_training_loss(obs)

    res = differential_evolution(obj, bounds, **{**OPT, "seed": seed})
    Yu, Yd = baseline_yukawas(theta_base, Q, U, D)
    Yu, Yd = apply_portal(Yu, Yd, Q, U, D, theta_base, res.x, mode)
    obs = compute_quark_observables(Yu, Yd)
    return res.x, obs


def optimize_schur(Q, U, D, seed: int):
    bounds = [
        (0.5, 6.0), (0.1, 2.0), (0.0, 2 * np.pi), (1.0, 5.0),
        (0.01, 0.5), (0.01, 0.5), (0.0, 0.2), (0.0, 0.2), (0.01, 2.0),
    ]

    def obj(theta):
        try:
            Yu, Yd = schur_yukawas(theta, Q, U, D)
            if not (np.all(np.isfinite(Yu)) and np.all(np.isfinite(Yd))):
                return 1e6
            obs = compute_quark_observables(Yu, Yd)
            return compute_training_loss(obs)
        except np.linalg.LinAlgError:
            return 1e6

    res = differential_evolution(obj, bounds, **{**OPT, "seed": seed})
    Yu, Yd = schur_yukawas(res.x, Q, U, D)
    obs = compute_quark_observables(Yu, Yd)
    return res.x, obs


def j_rel_err(obs):
    j_tgt = QUARK_CP_TARGETS["J"]
    if obs["J_abs"] <= 0:
        return 1.0
    return abs(obs["J_abs"] - j_tgt) / j_tgt


def sign_match(obs):
    return np.sign(obs["J"]) == np.sign(QUARK_CP_TARGETS["J"])


def run_audit(smoke: bool):
    n_geom = 3 if smoke else N_GEOM
    lines = [
        "Diagnostic 42: Chiral projection portal audit",
        f"N_geom={n_geom}, seeds={N_SEEDS}",
        "",
    ]
    geoms = generate_geometries(n_geom, GEOM_SEED)

    models = ["baseline", "parity_pi", "mirror_eta", "schur_6x6"]
    stats = {m: {"train": [], "hold": [], "j_err": [], "j_sign": []} for m in models}

    for gi, (Q, U, D) in enumerate(geoms):
        for seed in range(N_SEEDS):
            theta_b, obs_b, _, _ = optimize_baseline(Q, U, D, seed)
            stats["baseline"]["train"].append(compute_training_loss(obs_b))
            stats["baseline"]["hold"].append(compute_holdout_loss(obs_b))
            stats["baseline"]["j_err"].append(j_rel_err(obs_b))
            stats["baseline"]["j_sign"].append(sign_match(obs_b))

            for mode, key in [("parity_pi", "parity_pi"), ("mirror_eta", "mirror_eta")]:
                _, obs_p = optimize_portal(Q, U, D, seed, mode, theta_b)
                stats[key]["train"].append(compute_training_loss(obs_p))
                stats[key]["hold"].append(compute_holdout_loss(obs_p))
                stats[key]["j_err"].append(j_rel_err(obs_p))
                stats[key]["j_sign"].append(sign_match(obs_p))

            _, obs_s = optimize_schur(Q, U, D, seed)
            stats["schur_6x6"]["train"].append(compute_training_loss(obs_s))
            stats["schur_6x6"]["hold"].append(compute_holdout_loss(obs_s))
            stats["schur_6x6"]["j_err"].append(j_rel_err(obs_s))
            stats["schur_6x6"]["j_sign"].append(sign_match(obs_s))

    lines.append("Median metrics (lower train/hold/j_err better; j_sign = fraction correct sign):")
    for m in models:
        s = stats[m]
        lines.append(
            f"  {m:12s}  train={np.median(s['train']):.4f}  "
            f"hold={np.median(s['hold']):.4f}  "
            f"j_err={np.median(s['j_err']):.4f}  "
            f"j_sign={np.mean(s['j_sign']):.3f}"
        )

    base_j = np.median(stats["baseline"]["j_err"])
    best_portal = min(
        ("parity_pi", np.median(stats["parity_pi"]["j_err"])),
        ("mirror_eta", np.median(stats["mirror_eta"]["j_err"])),
        ("schur_6x6", np.median(stats["schur_6x6"]["j_err"])),
        key=lambda x: x[1],
    )
    base_train = np.median(stats["baseline"]["train"])
    best_name, best_j = best_portal
    best_train = np.median(stats[best_name]["train"])

    lines.append("")
    if best_j < 0.9 * base_j and best_train <= 1.2 * base_train:
        verdict = f"WEAK SIGNAL: {best_name} improves J error with modest train cost"
    else:
        verdict = "FALSIFIER PASS (quark CP): portal ansätze do not beat baseline on this audit"
    lines.append(f"Verdict: {verdict}")
    lines.append(f"  baseline median j_err={base_j:.4f}, best portal {best_name}={best_j:.4f}")

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
