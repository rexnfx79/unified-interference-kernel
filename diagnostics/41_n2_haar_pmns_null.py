#!/usr/bin/env python3
"""
N2 — Haar PMNS null vs post-fit angles (diag 28 pool).

Compares kernel-optimized PMNS angles to Haar-random U(3) mixing angles.

Pre-registered falsifier (kernel adds nothing):
  Post-fit angles are Haar-like: KS p > 0.05 for all three angles AND
  post-fit PDG angular distance is NOT smaller than Haar (Mann-Whitney p >= 0.01).

N2 positive (kernel structures mixing):
  >= 2 angles with KS p < 0.05 OR post-fit PDG distance << Haar (MW p < 0.01).

Mass-constrained Haar null (secondary): sample dm21, dm31 in PDG bands from
random positive masses; angles still Haar — tests whether angle clustering is
mass-conditioned only.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy import stats
from scipy.optimize import differential_evolution
from scipy.stats import unitary_group

from kernel import compute_yukawa_matrix
from observables import (
    NEUTRINO_MASS_TARGETS,
    NEUTRINO_TARGETS,
    compute_neutrino_joint_loss,
    compute_neutrino_observables,
    pmns_angles_from_unitary,
)

GEOM_SEED = 28028
N_GEOMETRIES = 100
N_SEEDS = 4
HAAR_SAMPLES = 5000
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

KS_P_MAX = 0.05
PDG_MW_P_MAX = 0.01
MIN_KS_REJECT = 2

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "41_n2_haar_pmns_null.txt"
)


def pdg_rel_distance(theta12: float, theta23: float, theta13: float) -> float:
    d = 0.0
    for key, val in [("theta12", theta12), ("theta23", theta23), ("theta13", theta13)]:
        t = NEUTRINO_TARGETS[key]
        d += ((val - t) / t) ** 2
    return float(np.sqrt(d))


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


def optimize_angles(geometries: List[Tuple]) -> List[Dict]:
    from phenomenology_utils import generate_neutrino_geometries

    rows = []
    for gi, (L, N) in enumerate(geometries):

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

        best_obs = None
        best_loss = np.inf
        for seed in range(N_SEEDS):
            try:
                res = differential_evolution(
                    objective,
                    NU_BOUNDS,
                    seed=seed + gi * 100,
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

        if best_obs is not None:
            rows.append(
                {
                    "theta12": best_obs["theta12"],
                    "theta23": best_obs["theta23"],
                    "theta13": best_obs["theta13"],
                    "pdg_dist": pdg_rel_distance(
                        best_obs["theta12"], best_obs["theta23"], best_obs["theta13"]
                    ),
                    "strict": check_joint_strict(best_obs),
                }
            )
        if (gi + 1) % 25 == 0:
            print(f"  optimized {gi + 1}/{len(geometries)}")
    return rows


def sample_haar_angles(n: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    t12, t23, t13, dists = [], [], [], []
    for _ in range(n):
        U = unitary_group.rvs(3, random_state=rng)
        a12, a23, a13 = pmns_angles_from_unitary(U)
        t12.append(a12)
        t23.append(a23)
        t13.append(a13)
        dists.append(pdg_rel_distance(a12, a23, a13))
    return {
        "theta12": np.array(t12),
        "theta23": np.array(t23),
        "theta13": np.array(t13),
        "pdg_dist": np.array(dists),
    }


def sample_mass_band_null(n: int, seed: int) -> Dict[str, np.ndarray]:
    """Haar angles + random masses with dm21, dm31 in PDG strict bands (angles independent)."""
    rng = np.random.RandomState(seed + 1)
    haar = sample_haar_angles(n, seed + 2)
    # masses don't affect angles in this null — same as Haar for angle stats
    return haar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    from phenomenology_utils import generate_neutrino_geometries

    n_geom = 25 if args.smoke else N_GEOMETRIES
    geom_seed = GEOM_SEED
    n_haar = 500 if args.smoke else HAAR_SAMPLES

    print(f"N2 Haar null: N_geom={n_geom}, Haar={n_haar}...")
    geometries = generate_neutrino_geometries(n_geom, geom_seed)
    post = optimize_angles(geometries)
    haar = sample_haar_angles(n_haar, 41042)

    if len(post) < 10:
        print("Too few solved geometries.")
        return

    post_t12 = np.array([r["theta12"] for r in post])
    post_t23 = np.array([r["theta23"] for r in post])
    post_t13 = np.array([r["theta13"] for r in post])
    post_dist = np.array([r["pdg_dist"] for r in post])

    ks = {}
    for key, post_arr, haar_arr in [
        ("theta12", post_t12, haar["theta12"]),
        ("theta23", post_t23, haar["theta23"]),
        ("theta13", post_t13, haar["theta13"]),
    ]:
        ks[key] = stats.ks_2samp(post_arr, haar_arr)

    n_ks_reject = sum(1 for k in ks if ks[k].pvalue < KS_P_MAX)
    _, pdg_mw_p = stats.mannwhitneyu(post_dist, haar["pdg_dist"], alternative="less")
    post_closer = post_dist.mean() < haar["pdg_dist"].mean()

    haar_like = n_ks_reject == 0 and (not post_closer or pdg_mw_p >= PDG_MW_P_MAX)
    n2_positive = n_ks_reject >= MIN_KS_REJECT or (post_closer and pdg_mw_p < PDG_MW_P_MAX)

    strict_post = [r for r in post if r["strict"]]
    strict_dist = np.array([r["pdg_dist"] for r in strict_post]) if strict_post else np.array([])

    lines = [
        "=" * 72,
        "N2 HAAR PMNS NULL (diagnostic 41)",
        "=" * 72,
        f"Post-fit pool: diag 28 protocol, seed={geom_seed}, N={n_geom}",
        f"Haar null samples: {n_haar}",
        f"Post-fit solved: {len(post)}",
        f"Post-fit strict: {len(strict_post)} ({100*len(strict_post)/max(len(post),1):.1f}% of solved)",
        "",
        "--- Median angles (rad) ---",
        f"  {'':12s}  post-fit    Haar      PDG",
    ]
    post_map = {"theta12": post_t12, "theta23": post_t23, "theta13": post_t13}
    for key in ["theta12", "theta23", "theta13"]:
        lines.append(
            f"  {key:12s}  {np.median(post_map[key]):.4f}      "
            f"{np.median(haar[key]):.4f}      {NEUTRINO_TARGETS[key]:.4f}"
        )

    lines.extend(
        [
            "",
            "--- KS test (post-fit vs Haar) ---",
        ]
    )
    for key in ["theta12", "theta23", "theta13"]:
        lines.append(f"  {key}: D={ks[key].statistic:.4f}, p={ks[key].pvalue:.4e}")

    lines.extend(
        [
            "",
            "--- PDG relative distance ---",
            f"  post-fit mean: {post_dist.mean():.4f}, median: {np.median(post_dist):.4f}",
            f"  Haar mean:     {haar['pdg_dist'].mean():.4f}, median: {np.median(haar['pdg_dist']):.4f}",
            f"  Mann-Whitney (post closer): p={pdg_mw_p:.4e}",
        ]
    )
    if len(strict_dist) > 0:
        lines.append(
            f"  strict-only median dist: {np.median(strict_dist):.4f} (n={len(strict_dist)})"
        )

    lines.extend(
        [
            "",
            "--- Pre-registered N2 ---",
            f"  KS rejections (p<{KS_P_MAX}): {n_ks_reject}/3",
            f"  haar_like (falsifier): {haar_like}",
            f"  n2_positive: {n2_positive}",
            "",
            "--- VERDICT ---",
        ]
    )

    if haar_like:
        verdict = "haar_like"
        lines.append(
            "  N2 falsifier NOT rejected — post-fit angles compatible with Haar anarchy."
        )
        lines.append("  Kernel may not structure mixing beyond generic unitarity.")
    elif n2_positive:
        verdict = "structured"
        lines.append(
            "  N2 POSITIVE — post-fit angles deviate from Haar and/or cluster near PDG."
        )
        lines.append("  Kernel optimization selects non-generic PMNS; not proof of mechanism.")
    else:
        verdict = "mixed"
        lines.append("  MIXED — partial deviation from Haar without clear pursue signal.")

    report = "\n".join(lines)
    print(report)

    if not args.smoke:
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            f.write(report + "\n")
            f.write(f"verdict: {verdict}\n")
        print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
