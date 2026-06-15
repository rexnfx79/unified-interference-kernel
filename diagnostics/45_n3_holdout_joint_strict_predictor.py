#!/usr/bin/env python3
"""
N3 — Holdout-block geometry predictor for joint strict (diag 45).

Extends N4 (diag 40): same geometry features and joint_strict labels, but
evaluates generalization on a **disjoint** geometry pool (different seed).

Train: seed=28028, N=70 — fit logistic on standardized geometry features.
Test:  seed=28029, N=50 — drop any (L,N) colliding with train; frozen coefficients.

Pre-registered (N3):
  FAIL:    holdout test AUC <= 0.55  (opaque luck on unseen geoms)
  PURSUE:  holdout test AUC >= 0.66  (beat N4 in-sample CV AUC 0.658)
  WEAK:    between bars

Secondary: PMNS-only strict label on same pools for comparison.

Usage:
  python diagnostics/45_n3_holdout_joint_strict_predictor.py
  python diagnostics/45_n3_holdout_joint_strict_predictor.py --smoke
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.optimize import differential_evolution, minimize

from kernel import compute_yukawa_matrix
from observables import (
    NEUTRINO_MASS_TARGETS,
    NEUTRINO_TARGETS,
    compute_neutrino_joint_loss,
    compute_neutrino_mass_loss,
    compute_neutrino_observables,
    compute_pmns_loss,
)
from phenomenology_utils import generate_neutrino_geometries

TRAIN_SEED = 28028
TEST_SEED = 28029
TRAIN_N = 70
TEST_N = 50
N_SEEDS = 4
OPT = dict(maxiter=120, popsize=12, tol=1e-6, mutation=(0.5, 1.0), recombination=0.7, polish=False)

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

FEATURE_NAMES = [
    "spread_L",
    "spread_N",
    "mean_L",
    "mean_N",
    "centroid_gap",
    "overlap_count",
    "union_count",
    "mean_pair_L",
    "mean_pair_N",
    "min_cross",
    "max_cross",
    "coord_span",
    "spread_ratio",
    "cross_span_ratio",
]

AUC_FAIL = 0.55
AUC_PURSUE = 0.66
N4_CV_REFERENCE = 0.658
L2_LAM = 1e-2
N_FOLDS = 5

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "45_n3_holdout_joint_strict_predictor.txt"
)


def geometry_features(L: Tuple[int, ...], N: Tuple[int, ...]) -> Dict[str, float]:
    Lv = np.array(L, dtype=float)
    Nv = np.array(N, dtype=float)

    def spread(v):
        return float(v.max() - v.min())

    def mean_pair(v):
        return float(np.mean([abs(v[i] - v[j]) for i in range(3) for j in range(i + 1, 3)]))

    cross = [abs(float(a) - float(b)) for a in L for b in N]
    sL, sN = spread(Lv), spread(Nv)
    allc = np.concatenate([Lv, Nv])
    return {
        "spread_L": sL,
        "spread_N": sN,
        "mean_L": float(Lv.mean()),
        "mean_N": float(Nv.mean()),
        "centroid_gap": abs(float(Lv.mean()) - float(Nv.mean())),
        "overlap_count": float(len(set(L) & set(N))),
        "union_count": float(len(set(L) | set(N))),
        "mean_pair_L": mean_pair(Lv),
        "mean_pair_N": mean_pair(Nv),
        "min_cross": float(min(cross)),
        "max_cross": float(max(cross)),
        "coord_span": float(allc.max() - allc.min()),
        "spread_ratio": sL / max(sN, 1e-6),
        "cross_span_ratio": float(max(cross) - min(cross)) / max(float(allc.max() - allc.min()), 1e-6),
    }


def check_pmns_strict(obs: Dict[str, float]) -> bool:
    for key, tol in PMNS_STRICT.items():
        t, v = NEUTRINO_TARGETS[key], obs[key]
        if v <= 0 or t <= 0 or abs(v - t) / t > tol:
            return False
    return True


def check_joint_strict(obs: Dict[str, float]) -> bool:
    if not check_pmns_strict(obs):
        return False
    for key, tol in MASS_STRICT.items():
        t, v = NEUTRINO_MASS_TARGETS[key], obs[key]
        if v <= 0 or t <= 0 or abs(v - t) / t > tol:
            return False
    return True


def optimize_one(L: Tuple, N: Tuple, geom_idx: int) -> Tuple[bool, bool, Dict]:
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

    best_obs, best_loss = None, np.inf
    for seed in range(N_SEEDS):
        try:
            res = differential_evolution(
                objective,
                NU_BOUNDS,
                seed=seed + geom_idx * 100,
                **OPT,
            )
        except Exception:
            continue
        if res.fun >= 999:
            continue
        sigma, k, alpha, eta, eps_nu, eps_e, g_env = res.x
        Ynu = compute_yukawa_matrix(L, N, sigma * g_env, k, alpha, eta, eps_nu)
        Ye = compute_yukawa_matrix(L, N, sigma, k, alpha, eta, eps_e)
        obs = compute_neutrino_observables(Ynu, Ye)
        jl = compute_neutrino_joint_loss(obs)
        if jl < best_loss:
            best_loss, best_obs = jl, obs

    if best_obs is None:
        return False, False, {}
    return (
        check_pmns_strict(best_obs),
        check_joint_strict(best_obs),
        {
            "joint_loss": best_loss,
            "pmns_loss": compute_pmns_loss(best_obs),
            "mass_loss": compute_neutrino_mass_loss(best_obs),
        },
    )


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(X: np.ndarray, y: np.ndarray, lam: float = L2_LAM) -> np.ndarray:
    n, p = X.shape
    Xd = np.column_stack([np.ones(n), X])

    def nll(beta):
        p_hat = sigmoid(Xd @ beta)
        p_hat = np.clip(p_hat, 1e-9, 1 - 1e-9)
        ll = np.sum(y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat))
        pen = lam * np.sum(beta[1:] ** 2)
        return -(ll - pen)

    beta0 = np.zeros(p + 1)
    res = minimize(nll, beta0, method="L-BFGS-B")
    return res.x


def predict_proba(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    Xd = np.column_stack([np.ones(len(X)), X])
    return sigmoid(Xd @ beta)


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for s in pos:
        wins += np.sum(s > neg) + 0.5 * np.sum(s == neg)
    return float(wins / (len(pos) * len(neg)))


def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (X - mu) / sd, mu, sd


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def cv_auc(X: np.ndarray, y: np.ndarray, n_folds: int = N_FOLDS, seed: int = 45045) -> float:
    n = len(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    scores = np.zeros(n)
    for k in range(n_folds):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != k])
        X_tr, mu, sd = standardize_fit(X[train_idx])
        X_te = standardize_apply(X[test_idx], mu, sd)
        beta = fit_logistic(X_tr, y[train_idx])
        scores[test_idx] = predict_proba(X_te, beta)
    return roc_auc(y, scores)


def univariate_aucs(X: np.ndarray, y: np.ndarray, names: List[str]) -> List[Tuple[str, float]]:
    return sorted(
        [(names[j], roc_auc(y, X[:, j])) for j in range(len(names))],
        key=lambda t: -t[1],
    )


def geom_key(L: Tuple, N: Tuple) -> Tuple:
    return (tuple(L), tuple(N))


def build_pool(geometries: List[Tuple], label: str, exclude: set | None = None):
    rows = []
    for gi, (L, N) in enumerate(geometries):
        key = geom_key(L, N)
        if exclude and key in exclude:
            continue
        feats = geometry_features(L, N)
        pmns_s, joint_s, meta = optimize_one(L, N, gi)
        rows.append(
            {
                **feats,
                "pmns_strict": int(pmns_s),
                "joint_strict": int(joint_s),
                "solved": int(meta != {}),
                "L": L,
                "N": N,
            }
        )
        if (gi + 1) % 20 == 0:
            print(f"  {label}: {gi + 1}/{len(geometries)}")
    return rows


def rows_to_xy(rows: List[dict], label_key: str):
    y = np.array([r[label_key] for r in rows], dtype=float)
    X = np.array([[r[n] for n in FEATURE_NAMES] for r in rows], dtype=float)
    return X, y


def run(smoke: bool):
    t0 = time.time()
    train_n = 15 if smoke else TRAIN_N
    test_n = 10 if smoke else TEST_N
    test_seed = 45029 if smoke else TEST_SEED

    print(f"N3 holdout predictor: train N={train_n} seed={TRAIN_SEED}, test N={test_n} seed={test_seed}")

    train_geoms = generate_neutrino_geometries(train_n, TRAIN_SEED)
    train_rows = build_pool(train_geoms, "train")
    train_keys = {geom_key(r["L"], r["N"]) for r in train_rows}

    test_geoms_raw = generate_neutrino_geometries(test_n + 20, test_seed)
    test_geoms = [g for g in test_geoms_raw if geom_key(g[0], g[1]) not in train_keys][:test_n]
    test_rows = build_pool(test_geoms, "test", exclude=train_keys)

    X_tr, y_joint_tr = rows_to_xy(train_rows, "joint_strict")
    _, y_pmns_tr = rows_to_xy(train_rows, "pmns_strict")
    X_te, y_joint_te = rows_to_xy(test_rows, "joint_strict")
    _, y_pmns_te = rows_to_xy(test_rows, "pmns_strict")

    X_tr_s, mu, sd = standardize_fit(X_tr)
    beta_joint = fit_logistic(X_tr_s, y_joint_tr)
    beta_pmns = fit_logistic(X_tr_s, y_pmns_tr)

    X_te_s = standardize_apply(X_te, mu, sd)
    scores_joint_te = predict_proba(X_te_s, beta_joint)
    scores_pmns_te = predict_proba(X_te_s, beta_pmns)

    auc_train_cv = cv_auc(X_tr, y_joint_tr)
    auc_train_in = roc_auc(y_joint_tr, predict_proba(X_tr_s, beta_joint))
    auc_test_joint = roc_auc(y_joint_te, scores_joint_te)
    auc_test_pmns = roc_auc(y_pmns_te, scores_pmns_te)
    uni_train = univariate_aucs(X_tr, y_joint_tr, FEATURE_NAMES)
    uni_test = univariate_aucs(X_te, y_joint_te, FEATURE_NAMES)

    falsifier = auc_test_joint <= AUC_FAIL if np.isfinite(auc_test_joint) else True
    pursue = auc_test_joint >= AUC_PURSUE if np.isfinite(auc_test_joint) else False
    beats_n4 = auc_test_joint > N4_CV_REFERENCE if np.isfinite(auc_test_joint) else False

    if falsifier:
        verdict, verdict_tag = "no_signal", "falsifier_pass"
    elif pursue:
        verdict, verdict_tag = "predictive_holdout", "pursue"
    else:
        verdict, verdict_tag = "weak_holdout", "weak_signal"

    lines = [
        "=" * 72,
        "N3 HOLDOUT GEOMETRY -> JOINT STRICT PREDICTOR (diagnostic 45)",
        "=" * 72,
        f"Train: seed={TRAIN_SEED}, N={len(train_rows)}",
        f"Test:  seed={test_seed}, N={len(test_rows)} (disjoint from train keys)",
        f"Wall time: {(time.time() - t0) / 60:.1f} min",
        "",
        "--- Label rates ---",
        f"  Train joint strict: {100*y_joint_tr.mean():.1f}% ({int(y_joint_tr.sum())}/{len(train_rows)})",
        f"  Test  joint strict: {100*y_joint_te.mean():.1f}% ({int(y_joint_te.sum())}/{len(test_rows)})",
        f"  Test  PMNS strict:  {100*y_pmns_te.mean():.1f}% ({int(y_pmns_te.sum())}/{len(test_rows)})",
        "",
        "--- AUC (joint strict primary) ---",
        f"  Train 5-fold CV:       {auc_train_cv:.4f}  (N4 reference CV: {N4_CV_REFERENCE})",
        f"  Train in-sample:       {auc_train_in:.4f}",
        f"  **Holdout test AUC**:  {auc_test_joint:.4f}",
        f"  Holdout test (PMNS):   {auc_test_pmns:.4f}",
        f"  Best uni train:        {uni_train[0][1]:.4f} ({uni_train[0][0]})",
        f"  Best uni test:         {uni_test[0][1]:.4f} ({uni_test[0][0]})",
        "",
        "--- Coefficients (train-fit, standardized) ---",
    ]
    for name, c in zip(FEATURE_NAMES, beta_joint[1:]):
        lines.append(f"  {name:16s} {c:+.4f}")

    lines.extend(
        [
            "",
            "--- Pre-registered N3 ---",
            f"  FAIL  holdout AUC <= {AUC_FAIL}: {falsifier}",
            f"  PURSUE holdout AUC >= {AUC_PURSUE}: {pursue}",
            f"  Beat N4 CV ({N4_CV_REFERENCE}): {beats_n4}",
            "",
            "--- VERDICT ---",
        ]
    )

    if falsifier:
        lines.append("  N3 falsifier NOT rejected — holdout geometry does not predict joint strict.")
    elif pursue:
        lines.append("  N3 PURSUE — holdout test AUC meets pursue bar (generalizes beyond N4 CV).")
    else:
        lines.append("  WEAK SIGNAL — holdout AUC above fail bar but below pursue / N4 CV.")

    report = "\n".join(lines)
    print(report)

    if not smoke:
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            f.write(report + "\n")
            f.write(f"verdict: {verdict_tag}\n")
            f.write(f"holdout_test_auc: {auc_test_joint}\n")
            f.write(f"train_cv_auc: {auc_train_cv}\n")
        print(f"\nSaved: {RESULTS_PATH}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.smoke)
