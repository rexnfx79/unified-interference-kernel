#!/usr/bin/env python3
"""
N4 — Predict neutrino joint strict success from geometry features (diag 28 pool).

Uses frozen geometry corpus: seed=28028, N=100 (diag 28).
Labels: joint_strict on best DE seed per geometry (0 if unsolved).

Pre-registered falsifier:
  max(5-fold CV AUC, best univariate AUC) <= 0.55 → no geometry signal (opaque luck).

Pursue bar (exploratory):
  CV AUC >= 0.65 with >= 1 feature |coef| > 0 in standardized logistic model.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.optimize import differential_evolution, minimize

from kernel import compute_yukawa_matrix
from observables import (
    compute_neutrino_joint_loss,
    compute_neutrino_mass_loss,
    compute_neutrino_observables,
    compute_pmns_loss,
    NEUTRINO_MASS_TARGETS,
    NEUTRINO_TARGETS,
)
from phenomenology_utils import generate_neutrino_geometries

# Import strict checks from diag 28 pattern
GEOM_SEED = 28028
N_GEOMETRIES = 100
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
]

AUC_FALSIFIER = 0.55
AUC_PURSUE = 0.65
N_FOLDS = 5
L2_LAM = 1e-2

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "40_n4_geometry_strict_predictor.txt"
)


def geometry_features(L: Tuple[int, ...], N: Tuple[int, ...]) -> Dict[str, float]:
    Lv = np.array(L, dtype=float)
    Nv = np.array(N, dtype=float)

    def spread(v):
        return float(v.max() - v.min())

    def mean_pair(v):
        return float(np.mean([abs(v[i] - v[j]) for i in range(3) for j in range(i + 1, 3)]))

    cross = [abs(float(a) - float(b)) for a in L for b in N]
    allc = np.concatenate([Lv, Nv])
    return {
        "spread_L": spread(Lv),
        "spread_N": spread(Nv),
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
    }


def check_joint_strict(obs: Dict[str, float]) -> bool:
    for key, tol in PMNS_STRICT.items():
        t, v = NEUTRINO_TARGETS[key], obs[key]
        if v <= 0 or t <= 0 or abs(v - t) / t > tol:
            return False
    for key, tol in MASS_STRICT.items():
        t, v = NEUTRINO_MASS_TARGETS[key], obs[key]
        if v <= 0 or t <= 0 or abs(v - t) / t > tol:
            return False
    return True


def optimize_one(L: Tuple, N: Tuple, geom_idx: int) -> Tuple[bool, Dict]:
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
        return False, {}
    return check_joint_strict(best_obs), {
        "joint_loss": best_loss,
        "pmns_loss": compute_pmns_loss(best_obs),
        "mass_loss": compute_neutrino_mass_loss(best_obs),
    }


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
    # Mann-Whitney U / AUC
    wins = 0.0
    for s in pos:
        wins += np.sum(s > neg) + 0.5 * np.sum(s == neg)
    return float(wins / (len(pos) * len(neg)))


def standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (X - mu) / sd, mu, sd


def cv_auc(X: np.ndarray, y: np.ndarray, n_folds: int = N_FOLDS, seed: int = 40040) -> float:
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
        X_tr, mu, sd = standardize(X[train_idx])
        X_te = (X[test_idx] - mu) / sd
        beta = fit_logistic(X_tr, y[train_idx])
        scores[test_idx] = predict_proba(X_te, beta)
    return roc_auc(y, scores)


def univariate_aucs(X: np.ndarray, y: np.ndarray, names: List[str]) -> List[Tuple[str, float]]:
    out = []
    for j, name in enumerate(names):
        out.append((name, roc_auc(y, X[:, j])))
    return sorted(out, key=lambda t: -t[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    n_geom = 20 if args.smoke else N_GEOMETRIES
    geom_seed = GEOM_SEED if not args.smoke else 40041

    print(f"N4 geometry predictor: N={n_geom}, seed={geom_seed} (diag 28 pool)...")
    geometries = generate_neutrino_geometries(n_geom, geom_seed)

    rows = []
    for gi, (L, N) in enumerate(geometries):
        feats = geometry_features(L, N)
        strict, meta = optimize_one(L, N, gi)
        rows.append({**feats, "strict": int(strict), "solved": int(meta != {})})
        if (gi + 1) % 20 == 0:
            print(f"  {gi + 1}/{n_geom} geometries")

    y = np.array([r["strict"] for r in rows], dtype=float)
    solved = np.array([r["solved"] for r in rows], dtype=bool)
    X = np.array([[r[n] for n in FEATURE_NAMES] for r in rows], dtype=float)

    # Primary: all geometries (unsolved → strict=0)
    auc_uni_best = max(univariate_aucs(X, y, FEATURE_NAMES), key=lambda t: t[1])
    auc_cv = cv_auc(X, y)
    Xs, _, _ = standardize(X)
    beta = fit_logistic(Xs, y)
    auc_insample = roc_auc(y, predict_proba(Xs, beta))

    # Secondary: solved-only pool (diag 28 comparable)
    if solved.sum() >= 10 and len(np.unique(y[solved])) == 2:
        auc_cv_solved = cv_auc(X[solved], y[solved])
        auc_uni_solved = max(univariate_aucs(X[solved], y[solved], FEATURE_NAMES), key=lambda t: t[1])
    else:
        auc_cv_solved = float("nan")
        auc_uni_solved = ("n/a", float("nan"))

    max_auc = np.nanmax([auc_cv, auc_uni_best[1]])
    falsifier = max_auc <= AUC_FALSIFIER
    pursue = auc_cv >= AUC_PURSUE if np.isfinite(auc_cv) else False

    uni_all = univariate_aucs(X, y, FEATURE_NAMES)
    coefs = beta[1:]

    lines = [
        "=" * 72,
        "N4 GEOMETRY → JOINT STRICT PREDICTOR (diagnostic 40)",
        "=" * 72,
        f"Pool: diag 28 protocol (joint ν), seed={geom_seed}, N={n_geom}",
        f"Labels: joint_strict (0 if unsolved); solved {int(solved.sum())}/{n_geom}",
        f"Strict positive rate: {100*y.mean():.1f}% ({int(y.sum())}/{n_geom})",
        "",
        "--- Model ---",
        "  Logistic regression, standardized geometry features only (no post-fit params)",
        f"  L2 λ={L2_LAM}; {N_FOLDS}-fold CV AUC",
        "",
        "--- AUC ---",
        f"  CV AUC (full model):     {auc_cv:.4f}",
        f"  In-sample AUC:           {auc_insample:.4f}",
        f"  Best univariate AUC:     {auc_uni_best[1]:.4f} ({auc_uni_best[0]})",
    ]
    if np.isfinite(auc_cv_solved):
        lines.append(f"  CV AUC (solved only):    {auc_cv_solved:.4f}")
        lines.append(
            f"  Best uni (solved only):  {auc_uni_solved[1]:.4f} ({auc_uni_solved[0]})"
        )

    lines.extend(["", "--- Univariate AUC (all geoms) ---"])
    for name, auc in uni_all:
        lines.append(f"  {name:14s} {auc:.4f}")

    lines.extend(["", "--- Standardized coefficients (in-sample) ---"])
    for name, c in zip(FEATURE_NAMES, coefs):
        lines.append(f"  {name:14s} {c:+.4f}")

    lines.extend(
        [
            "",
            "--- Pre-registered N4 ---",
            f"  falsifier AUC <= {AUC_FALSIFIER}: {falsifier}",
            f"  pursue AUC >= {AUC_PURSUE}: {pursue}",
            "",
            "--- VERDICT ---",
        ]
    )

    if falsifier:
        verdict = "no_signal"
        lines.append(
            "  N4 falsifier NOT rejected — geometry does not predict joint strict (opaque luck)."
        )
    elif pursue:
        verdict = "predictive"
        lines.append(
            "  N4 PURSUE — geometry features predict joint strict with CV AUC above bar."
        )
    else:
        verdict = "weak_signal"
        lines.append(
            "  WEAK SIGNAL — some AUC above falsifier but below pursue bar; interpret cautiously."
        )

    report = "\n".join(lines)
    print(report)

    if not args.smoke:
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            f.write(report + "\n")
            f.write(f"verdict: {verdict}\n")
            f.write(f"cv_auc: {auc_cv}\n")
            f.write(f"best_univariate: {auc_uni_best[0]} {auc_uni_best[1]}\n")
        print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
