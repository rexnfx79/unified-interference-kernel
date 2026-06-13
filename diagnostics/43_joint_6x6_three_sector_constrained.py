#!/usr/bin/env python3
"""
Joint 3-sector 6×6 constrained fit (diagnostic 43, chiral projection L3 / P6).

Corpus: diag 26 joint geometries (shared L=Q, seed 26026).

Models per geometry:
  1. independent — separate DE per sector (18 kernel params total)
  2. joint_shared — shared (sigma,k,alpha,eta) + sector eps (9 params)
  3. joint_6x6 — joint_shared + quark Schur portal (eps_x_u, eps_x_d, rho) (12 params)

Pre-registered falsifiers (chiral-projection-formalization-program):
  L3 FAIL: joint_6x6 median holdout sum >= independent median holdout sum.
  P6 FAIL: joint_6x6 does not reduce param count while matching train within 10%.

Success: joint_6x6 holdout sum < 0.95 * independent AND n_params_joint < n_params_indep.
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

from alternative_kernels import compute_yukawas_gaussian
from kernel import compute_yukawa_matrix
from observables import (
    compute_holdout_loss,
    compute_lepton_holdout_loss,
    compute_lepton_observables,
    compute_lepton_training_loss,
    compute_neutrino_joint_loss,
    compute_neutrino_observables,
    compute_quark_observables,
    compute_training_loss,
)
from phenomenology_utils import (
    JointThreeSectorGeometry,
    generate_joint_three_sector_geometries,
)

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "43_joint_6x6_three_sector_constrained.txt"
)

JOINT_GEOM_SEED = 26026
DEFAULT_N_GEOM = 30
N_SEEDS = 2
OPT = dict(maxiter=80, popsize=10, tol=1e-6, polish=False)

HOLDOUT_IMPROVE_RATIO = 0.95
TRAIN_MATCH_RATIO = 1.10

N_PARAMS_INDEP = 18  # quark 6 + lepton 5 + neutrino 7
N_PARAMS_SHARED = 9
N_PARAMS_6X6 = 12


def portal_matrix(left, right, sigma, k, alpha, eta, eps_int):
    Y = np.zeros((3, 3), dtype=complex)
    left_vec = np.array([left[0], left[1], 0], dtype=float)
    right_vec = np.array(right, dtype=float)
    for i in range(3):
        for j in range(3):
            xi, xj = left_vec[i], right_vec[j]
            diff = xi - xj
            env = np.exp(-diff**2 / (2 * sigma**2))
            phase = alpha + k * (xi + xj) / 2 - eta * diff
            Y[i, j] = eps_int * env * np.exp(1j * phase)
    return Y


def schur_quark_yukawas(
    Q, U, D, sigma, k, alpha, eta, eps_u, eps_d, eps_x_u, eps_x_d, rho
):
    Yu_sm = compute_yukawa_matrix(Q, U, sigma, k, alpha, eta, eps_u)
    Yd_sm = compute_yukawa_matrix(Q, D, sigma, k, alpha, eta, eps_d)
    Yu_m = compute_yukawa_matrix(Q, U, sigma, -k, alpha + np.pi, -eta, eps_u)
    Yd_m = compute_yukawa_matrix(Q, D, sigma, -k, alpha + np.pi, -eta, eps_d)
    Xu = portal_matrix(Q, U, sigma, k, alpha, eta, eps_x_u)
    Xd = portal_matrix(Q, D, sigma, k, alpha, eta, eps_x_d)
    inv_u = np.linalg.inv(Yu_m + rho * np.eye(3))
    inv_d = np.linalg.inv(Yd_m + rho * np.eye(3))
    Yu = Yu_sm - Xu @ inv_u @ Xu.conj().T
    Yd = Yd_sm - Xd @ inv_d @ Xd.conj().T
    return Yu, Yd


def sector_losses_from_yukawas(geom: JointThreeSectorGeometry, Yu, Yd, Ye, Ynu, Ye_nu):
    obs_q = compute_quark_observables(Yu, Yd)
    obs_l = compute_lepton_observables(Ye)
    obs_n = compute_neutrino_observables(Ynu, Ye_nu)
    if obs_n["theta23"] < 0.01:
        return None
    return {
        "quark_train": compute_training_loss(obs_q),
        "quark_hold": compute_holdout_loss(obs_q),
        "lepton_train": compute_lepton_training_loss(obs_l),
        "lepton_hold": compute_lepton_holdout_loss(obs_l),
        "neutrino_joint": compute_neutrino_joint_loss(obs_n),
        "train_sum": (
            compute_training_loss(obs_q)
            + compute_lepton_training_loss(obs_l)
            + compute_neutrino_joint_loss(obs_n)
        ),
        "holdout_sum": (
            compute_holdout_loss(obs_q)
            + compute_lepton_holdout_loss(obs_l)
            + compute_neutrino_joint_loss(obs_n)
        ),
    }


def optimize_independent(geom: JointThreeSectorGeometry, seed: int) -> dict | None:
    L, E = geom.lepton
    Ln, N = geom.neutrino
    Q, U, D = geom.quark

    q_bounds = [(0.5, 6.0), (0.1, 2.0), (0.0, 2 * np.pi), (1.0, 5.0), (0.01, 0.5), (0.01, 0.5)]
    l_bounds = [(0.5, 6.0), (0.1, 2.0), (0.0, 2 * np.pi), (1.0, 5.0), (0.01, 0.5)]
    n_bounds = [
        (0.5, 6.0), (0.1, 2.0), (0.0, 2 * np.pi), (1.0, 5.0),
        (0.01, 0.5), (0.01, 0.5), (0.45, 0.75),
    ]

    def opt_q():
        def obj(theta):
            Yu, Yd = compute_yukawas_gaussian(Q, U, D, *theta)
            return compute_training_loss(compute_quark_observables(Yu, Yd))
        r = differential_evolution(obj, q_bounds, seed=seed, **OPT)
        return compute_yukawas_gaussian(Q, U, D, *r.x)

    def opt_l():
        def obj(theta):
            Ye = compute_yukawa_matrix(L, E, *theta)
            return compute_lepton_training_loss(compute_lepton_observables(Ye))
        r = differential_evolution(obj, l_bounds, seed=seed + 1, **OPT)
        return compute_yukawa_matrix(L, E, *r.x)

    def opt_n():
        def obj(theta):
            sigma, k, alpha, eta, eps_nu, eps_e, g_env = theta
            Ynu = compute_yukawa_matrix(Ln, N, sigma * g_env, k, alpha, eta, eps_nu)
            Ye = compute_yukawa_matrix(Ln, N, sigma, k, alpha, eta, eps_e)
            obs = compute_neutrino_observables(Ynu, Ye)
            if obs["theta23"] < 0.01:
                return 1000.0
            return compute_neutrino_joint_loss(obs)
        r = differential_evolution(obj, n_bounds, seed=seed + 2, **OPT)
        if r.fun >= 999:
            return None
        sigma, k, alpha, eta, eps_nu, eps_e, g_env = r.x
        Ynu = compute_yukawa_matrix(Ln, N, sigma * g_env, k, alpha, eta, eps_nu)
        Ye = compute_yukawa_matrix(Ln, N, sigma, k, alpha, eta, eps_e)
        return Ynu, Ye

    try:
        Yu, Yd = opt_q()
        Ye = opt_l()
        nu = opt_n()
        if nu is None:
            return None
        Ynu, Ye_nu = nu
        losses = sector_losses_from_yukawas(geom, Yu, Yd, Ye, Ynu, Ye_nu)
        return losses
    except Exception:
        return None


def optimize_joint_shared(geom: JointThreeSectorGeometry, seed: int) -> dict | None:
    L, E = geom.lepton
    Ln, N = geom.neutrino
    Q, U, D = geom.quark
    bounds = [
        (0.5, 6.0), (0.1, 2.0), (0.0, 2 * np.pi), (1.0, 5.0),
        (0.01, 0.5), (0.01, 0.5), (0.01, 0.5), (0.01, 0.5), (0.45, 0.75),
    ]

    def obj(theta):
        sigma, k, alpha, eta, eps_u, eps_d, eps_e, eps_nu, g_env = theta
        try:
            Yu, Yd = compute_yukawas_gaussian(Q, U, D, sigma, k, alpha, eta, eps_u, eps_d)
            Ye = compute_yukawa_matrix(L, E, sigma, k, alpha, eta, eps_e)
            Ynu = compute_yukawa_matrix(Ln, N, sigma * g_env, k, alpha, eta, eps_nu)
            Ye_nu = compute_yukawa_matrix(Ln, N, sigma, k, alpha, eta, eps_e)
            losses = sector_losses_from_yukawas(geom, Yu, Yd, Ye, Ynu, Ye_nu)
            if losses is None:
                return 1000.0
            return losses["train_sum"]
        except Exception:
            return 1000.0

    try:
        r = differential_evolution(obj, bounds, seed=seed, **OPT)
        if r.fun >= 999:
            return None
        sigma, k, alpha, eta, eps_u, eps_d, eps_e, eps_nu, g_env = r.x
        Yu, Yd = compute_yukawas_gaussian(Q, U, D, sigma, k, alpha, eta, eps_u, eps_d)
        Ye = compute_yukawa_matrix(L, E, sigma, k, alpha, eta, eps_e)
        Ynu = compute_yukawa_matrix(Ln, N, sigma * g_env, k, alpha, eta, eps_nu)
        Ye_nu = compute_yukawa_matrix(Ln, N, sigma, k, alpha, eta, eps_e)
        return sector_losses_from_yukawas(geom, Yu, Yd, Ye, Ynu, Ye_nu)
    except Exception:
        return None


def optimize_joint_6x6(geom: JointThreeSectorGeometry, seed: int) -> dict | None:
    L, E = geom.lepton
    Ln, N = geom.neutrino
    Q, U, D = geom.quark
    bounds = [
        (0.5, 6.0), (0.1, 2.0), (0.0, 2 * np.pi), (1.0, 5.0),
        (0.01, 0.5), (0.01, 0.5), (0.01, 0.5), (0.01, 0.5), (0.45, 0.75),
        (0.0, 0.2), (0.0, 0.2), (0.01, 2.0),
    ]

    def obj(theta):
        (
            sigma, k, alpha, eta, eps_u, eps_d, eps_e, eps_nu, g_env,
            eps_x_u, eps_x_d, rho,
        ) = theta
        try:
            Yu, Yd = schur_quark_yukawas(
                Q, U, D, sigma, k, alpha, eta, eps_u, eps_d, eps_x_u, eps_x_d, rho
            )
            if not (np.all(np.isfinite(Yu)) and np.all(np.isfinite(Yd))):
                return 1e6
            Ye = compute_yukawa_matrix(L, E, sigma, k, alpha, eta, eps_e)
            Ynu = compute_yukawa_matrix(Ln, N, sigma * g_env, k, alpha, eta, eps_nu)
            Ye_nu = compute_yukawa_matrix(Ln, N, sigma, k, alpha, eta, eps_e)
            losses = sector_losses_from_yukawas(geom, Yu, Yd, Ye, Ynu, Ye_nu)
            if losses is None:
                return 1000.0
            return losses["train_sum"]
        except (np.linalg.LinAlgError, FloatingPointError):
            return 1e6

    try:
        r = differential_evolution(obj, bounds, seed=seed, **OPT)
        if r.fun >= 999:
            return None
        (
            sigma, k, alpha, eta, eps_u, eps_d, eps_e, eps_nu, g_env,
            eps_x_u, eps_x_d, rho,
        ) = r.x
        Yu, Yd = schur_quark_yukawas(
            Q, U, D, sigma, k, alpha, eta, eps_u, eps_d, eps_x_u, eps_x_d, rho
        )
        Ye = compute_yukawa_matrix(L, E, sigma, k, alpha, eta, eps_e)
        Ynu = compute_yukawa_matrix(Ln, N, sigma * g_env, k, alpha, eta, eps_nu)
        Ye_nu = compute_yukawa_matrix(Ln, N, sigma, k, alpha, eta, eps_e)
        return sector_losses_from_yukawas(geom, Yu, Yd, Ye, Ynu, Ye_nu)
    except Exception:
        return None


def best_of_seeds(geom, fn, base_seed: int) -> dict | None:
    best = None
    for s in range(N_SEEDS):
        out = fn(geom, base_seed + s * 17)
        if out is None:
            continue
        if best is None or out["train_sum"] < best["train_sum"]:
            best = out
    return best


def run_audit(n_geom: int, smoke: bool) -> str:
    geoms = generate_joint_three_sector_geometries(n_geom, JOINT_GEOM_SEED)
    if len(geoms) < n_geom:
        print(f"WARNING: only {len(geoms)}/{n_geom} joint geometries")

    ind_tr, ind_ho = [], []
    shared_tr, shared_ho = [], []
    schur_tr, schur_ho = [], []

    t0 = time.time()
    for gi, geom in enumerate(geoms):
        ind = best_of_seeds(geom, optimize_independent, gi * 100)
        shared = best_of_seeds(geom, optimize_joint_shared, gi * 100 + 50)
        schur = best_of_seeds(geom, optimize_joint_6x6, gi * 100 + 75)
        if ind:
            ind_tr.append(ind["train_sum"])
            ind_ho.append(ind["holdout_sum"])
        if shared:
            shared_tr.append(shared["train_sum"])
            shared_ho.append(shared["holdout_sum"])
        if schur:
            schur_tr.append(schur["train_sum"])
            schur_ho.append(schur["holdout_sum"])
        if (gi + 1) % 10 == 0:
            print(f"  {gi + 1}/{len(geoms)} geometries")

    elapsed = time.time() - t0

    def med(xs):
        return float(np.median(xs)) if xs else float("nan")

    m_ind_tr, m_ind_ho = med(ind_tr), med(ind_ho)
    m_sh_tr, m_sh_ho = med(shared_tr), med(shared_ho)
    m_6_tr, m_6_ho = med(schur_tr), med(schur_ho)

    l3_fail = m_6_ho >= m_ind_ho if np.isfinite(m_6_ho) and np.isfinite(m_ind_ho) else True
    p6_param_ok = N_PARAMS_6X6 < N_PARAMS_INDEP
    p6_train_ok = m_6_tr <= TRAIN_MATCH_RATIO * m_ind_tr if np.isfinite(m_6_tr) else False
    l3_success = (
        np.isfinite(m_6_ho)
        and np.isfinite(m_ind_ho)
        and m_6_ho < HOLDOUT_IMPROVE_RATIO * m_ind_ho
    )
    p6_success = p6_param_ok and p6_train_ok and (
        m_6_ho <= m_ind_ho * 1.05 if np.isfinite(m_6_ho) else False
    )

    lines = [
        "Diagnostic 43: Joint 3-sector 6×6 constrained fit (diag 26 corpus)",
        f"N_geom={len(geoms)}, seed={JOINT_GEOM_SEED}, seeds/model={N_SEEDS}",
        f"Wall time: {elapsed / 60:.1f} min",
        "",
        "Param counts: independent=18, joint_shared=9, joint_6x6=12",
        "",
        "--- Median train / holdout sums (quark train + lepton train + ν joint) ---",
        f"  independent:  train={m_ind_tr:.4f}  holdout_sum={m_ind_ho:.4f}  (n={len(ind_tr)})",
        f"  joint_shared: train={m_sh_tr:.4f}  holdout_sum={m_sh_ho:.4f}  (n={len(shared_tr)})",
        f"  joint_6x6:    train={m_6_tr:.4f}  holdout_sum={m_6_ho:.4f}  (n={len(schur_tr)})",
        "",
        "--- Pre-registered ---",
        f"  L3 (6x6 holdout < {HOLDOUT_IMPROVE_RATIO}× independent): {l3_success}",
        f"  L3 falsifier (6x6 holdout >= independent): {l3_fail}",
        f"  P6 param reduction ({N_PARAMS_6X6} < {N_PARAMS_INDEP}): {p6_param_ok}",
        f"  P6 train match (6x6 train <= {TRAIN_MATCH_RATIO}× ind): {p6_train_ok}",
        "",
        "--- VERDICT ---",
    ]

    if l3_success and p6_success:
        verdict = "success"
        lines.append("  L3+P6 PASS — constrained 6×6 joint beats independent on holdout with fewer params.")
    elif l3_fail and not l3_success:
        verdict = "l3_fail"
        lines.append("  L3 FAIL — joint 6×6 does not beat independent sector fits on holdout sum.")
        lines.append("  Chiral projection F2 joint parent not supported at this constraint level.")
    elif m_6_tr < m_ind_tr and m_6_ho > m_ind_ho:
        verdict = "overfit"
        lines.append("  OVERFIT — 6×6 lowers train but worsens holdout (diag 42 pattern).")
    else:
        verdict = "mixed"
        lines.append("  MIXED — partial gains without clear L3/P6 success.")

    text = "\n".join(lines)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(text + "\n")
        f.write(f"verdict: {verdict}\n")
    print(text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-geometries", type=int, default=DEFAULT_N_GEOM)
    args = parser.parse_args()
    n_geom = 3 if args.smoke else args.n_geometries
    run_audit(n_geom, args.smoke)


if __name__ == "__main__":
    main()
