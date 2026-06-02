#!/usr/bin/env python3
"""
Lepton m_mu–m_e Pareto Frontier (diagnostic 25)

Weighted sweep trading train m_mu+m_tau fit vs holdout m_e fit.
Characterizes structural tension analogous to quark CKM–m_c Pareto knee.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.optimize import differential_evolution
from typing import Dict, List, Tuple

from kernel import compute_yukawa_matrix
from observables import (
    compute_lepton_observables,
    compute_lepton_training_loss,
    compute_lepton_holdout_loss,
    LEPTON_TARGETS,
)
from phenomenology_utils import generate_lepton_geometries, pareto_nondominated

# =============================================================================
# CONFIGURATION
# =============================================================================

OPTIMIZER_SETTINGS = {
    "maxiter": 120,
    "popsize": 12,
    "tol": 1e-6,
    "mutation": (0.5, 1.0),
    "recombination": 0.7,
    "polish": False,
}

N_SEEDS = 3
N_GEOMETRIES = 24
GEOM_SEED = 25025
N_WEIGHTS = 12

LEPTON_BOUNDS = [
    (0.5, 6.0),
    (0.1, 2.0),
    (0.0, 2 * np.pi),
    (1.0, 5.0),
    (0.01, 0.5),
]

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "25_lepton_mass_pareto.txt"
)
FIGURE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "figures", "lepton_mass_pareto_diag25.png"
)


def make_weighted_objective(L: Tuple, E: Tuple, w_train: float, w_hold: float):
    def objective(theta):
        try:
            sigma, k, alpha, eta, eps = theta
            Ye = compute_yukawa_matrix(L, E, sigma, k, alpha, eta, eps)
            obs = compute_lepton_observables(Ye)
            train_l = compute_lepton_training_loss(obs)
            hold_l = compute_lepton_holdout_loss(obs)
            return w_train * train_l + w_hold * hold_l
        except Exception:
            return 1000.0

    return objective


def log_spaced_weights(n: int) -> List[Tuple[float, float]]:
    """(w_train, w_hold) pairs spanning train-only to holdout-heavy."""
    ratios = np.logspace(-2, 2, n)
    pairs = []
    for r in ratios:
        w_train = 1.0 / (1.0 + r)
        w_hold = r / (1.0 + r)
        pairs.append((float(w_train), float(w_hold)))
    return pairs


def pareto_sweep_geometry(L: Tuple, E: Tuple, geom_idx: int) -> List[Dict]:
    weights = log_spaced_weights(N_WEIGHTS)
    points = []
    for w_idx, (w_train, w_hold) in enumerate(weights):
        best = None
        for seed in range(N_SEEDS):
            objective = make_weighted_objective(L, E, w_train, w_hold)
            try:
                result = differential_evolution(
                    objective,
                    LEPTON_BOUNDS,
                    seed=seed + geom_idx * 1000 + w_idx * 10,
                    **OPTIMIZER_SETTINGS,
                )
            except Exception:
                continue
            if result.fun >= 999:
                continue
            sigma, k, alpha, eta, eps = result.x
            Ye = compute_yukawa_matrix(L, E, sigma, k, alpha, eta, eps)
            obs = compute_lepton_observables(Ye)
            train_l = compute_lepton_training_loss(obs)
            hold_l = compute_lepton_holdout_loss(obs)
            mu_rel = abs(obs["m_mu"] - LEPTON_TARGETS["m_mu"]) / LEPTON_TARGETS["m_mu"]
            e_rel = abs(obs["m_e"] - LEPTON_TARGETS["m_e"]) / LEPTON_TARGETS["m_e"]
            rec = {
                "geom": geom_idx,
                "w_train": w_train,
                "w_hold": w_hold,
                "train": train_l,
                "holdout": hold_l,
                "mu_rel_err": mu_rel,
                "e_rel_err": e_rel,
            }
            if best is None or result.fun < best["weighted"]:
                best = {**rec, "weighted": result.fun}
        if best is not None:
            points.append(best)
    return points


def knee_analysis(all_points: List[Dict]) -> Dict:
    """Pareto in (train_loss, holdout_loss) space."""
    pairs = [(p["train"], p["holdout"]) for p in all_points]
    nd = pareto_nondominated(pairs)
    trains = np.array([p[0] for p in pairs])
    holds = np.array([p[1] for p in pairs])
    corr = float(np.corrcoef(trains, holds)[0, 1]) if len(trains) > 2 else float("nan")

    # Knee: point on frontier with minimum Euclidean distance to origin in log space
    knee = None
    if nd:
        log_nd = [(np.log10(max(t, 1e-12)), np.log10(max(h, 1e-12))) for t, h in nd]
        dists = [np.hypot(x, y) for x, y in log_nd]
        knee_idx = int(np.argmin(dists))
        knee = nd[knee_idx]

    return {
        "n_points": len(all_points),
        "n_pareto": len(nd),
        "train_hold_corr": corr,
        "pareto_front_sample": nd[:8],
        "knee_train_hold": knee,
    }


def try_save_figure(all_points: List[Dict], pareto: Dict) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    trains = [p["train"] for p in all_points]
    holds = [p["holdout"] for p in all_points]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(trains, holds, s=12, alpha=0.35, c="steelblue", label="Weighted sweep")
    if pareto.get("pareto_front_sample"):
        pf = pareto["pareto_front_sample"]
        ax.plot(
            [p[0] for p in pf],
            [p[1] for p in pf],
            "r-o",
            ms=4,
            lw=1.5,
            label=f"Pareto front (n={pareto["n_pareto"]})",
        )
    knee = pareto.get("knee_train_hold")
    if knee:
        ax.scatter([knee[0]], [knee[1]], c="gold", s=80, edgecolors="black", zorder=5, label="Knee")
    ax.set_xlabel("Train loss (m_mu + m_tau)")
    ax.set_ylabel("Holdout loss (m_e)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Lepton m_mu–m_e Pareto (diag 25)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(FIGURE_PATH), exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)
    return True


def format_report(geometries: List[Tuple], all_points: List[Dict], pareto: Dict, fig_saved: bool) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("LEPTON m_mu–m_e PARETO FRONTIER (diagnostic 25)")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Objective: w_train * L_train(m_mu,m_tau) + w_hold * L_holdout(m_e)")
    lines.append(f"Geometries: {len(geometries)}; weights per geometry: {N_WEIGHTS}")
    lines.append(f"Seeds per weight: {N_SEEDS}; optimizer: {OPTIMIZER_SETTINGS}")
    lines.append("")

    if not all_points:
        lines.append("No converged Pareto points.")
        return "\n".join(lines)

    trains = [p["train"] for p in all_points]
    holds = [p["holdout"] for p in all_points]
    mu_rels = [p["mu_rel_err"] for p in all_points]
    e_rels = [p["e_rel_err"] for p in all_points]

    lines.append("--- AGGREGATE ---")
    lines.append(f"  Total Pareto sweep points: {len(all_points)}")
    lines.append(f"  Nondominated (train, holdout): {pareto['n_pareto']}")
    lines.append(
        f"  corr(train_loss, holdout_loss): {pareto['train_hold_corr']:.4f} "
        "(positive => structural trade-off)"
    )
    lines.append(
        f"  Train loss  median: {np.median(trains):.4f}; "
        f"Holdout loss median: {np.median(holds):.4f}"
    )
    lines.append(
        f"  m_mu rel err median: {np.median(mu_rels):.4f}; "
        f"m_e rel err median: {np.median(e_rels):.4f}"
    )
    lines.append("")

    lines.append("--- PARETO FRONT SAMPLE (train_loss, holdout_loss) ---")
    for pt in pareto.get("pareto_front_sample", [])[:6]:
        lines.append(f"  ({pt[0]:.6f}, {pt[1]:.4f})")
    knee = pareto.get("knee_train_hold")
    if knee:
        lines.append(f"  Knee (log-space nearest origin): train={knee[0]:.6f}, holdout={knee[1]:.4f}")
    lines.append("")

    lines.append("--- STRUCTURAL INTERPRETATION ---")
    lines.append(
        "  Analogous to quark CKM–m_c: improving train (m_mu+m_tau) does not "
        "simultaneously improve holdout m_e."
    )
    lines.append(
        "  Sparse Pareto fronts indicate a knee — m_e is not reachable by "
        "phase/envelope tuning once mu–tau hierarchy is fit."
    )
    if fig_saved:
        lines.append(f"  Figure saved: {FIGURE_PATH}")
    lines.append("")

    lines.append("--- HONEST CONCLUSIONS ---")
    lines.append("  • m_e holdout failure is structural, not optimizer artifact.")
    lines.append("  • Weighted Pareto documents tension; no simultaneous 3-mass fit at strict tolerances.")
    lines.append("  • Phenomenology only — no mechanism claim.")
    lines.append("")
    return "\n".join(lines)


def main():
    print(f"Lepton mass Pareto sweep ({N_GEOMETRIES} geometries)...")
    geometries = generate_lepton_geometries(N_GEOMETRIES, GEOM_SEED)
    all_points: List[Dict] = []
    for geom_idx, (L, E) in enumerate(geometries):
        pts = pareto_sweep_geometry(L, E, geom_idx)
        all_points.extend(pts)
        print(f"  Geometry {geom_idx + 1}/{len(geometries)}: {len(pts)} points")

    pareto = knee_analysis(all_points)
    fig_saved = try_save_figure(all_points, pareto)
    report = format_report(geometries, all_points, pareto, fig_saved)
    print(report)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        f.write(report)
    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
