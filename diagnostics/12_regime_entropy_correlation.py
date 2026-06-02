#!/usr/bin/env python3
"""
Regime–entropy correlation diagnostic (Path A mechanism).

Sweeps geometries and parameter points per SM flavor sector, computes
S(rho_Y), effective rank, off-diagonal entropy, and correlates with
regime labels, g_env, and loss components.

Pre-registered falsifier (see knowledge/wiki/synthesis/research-strategy.md):
  If |Pearson r| < CORRELATION_THRESHOLD for all primary measures vs regime
  across >= MIN_SAMPLES pooled samples, mark regime-entropy hypothesis REFUTED.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import itertools
import numpy as np
from scipy import stats

from kernel import compute_quark_yukawas, compute_yukawa_matrix
from flavor_information import compute_yukawa_information
from observables import (
    compute_quark_observables,
    compute_ckm_loss,
    compute_mass_loss,
    compute_neutrino_observables,
    compute_pmns_loss,
)

# --- Pre-registered falsifier ---
CORRELATION_THRESHOLD = 0.30
MIN_SAMPLES = 30

# Regime encoding: envelope / phase / metric
REGIME_QUARK = 0
REGIME_LEPTON = 1
REGIME_NEUTRINO = 2

QUARK_GEOMETRIES = [
    {"name": "standard", "Q": (0, 1, 0), "U": (0, 3, 6), "D": (0, 3, 7)},
    {"name": "csv_compact", "Q": (0, 1, 0), "U": (0, 1, 2), "D": (0, 1, 3)},
    {"name": "kernel_comparison", "Q": (0, 1, 3), "U": (2, 4, 5), "D": (0, 3, 6)},
    {"name": "spread_020", "Q": (0, 2, 0), "U": (0, 2, 4), "D": (0, 2, 5)},
    {"name": "spread_246", "Q": (2, 4, 6), "U": (0, 3, 6), "D": (1, 4, 7)},
]

LEPTON_GEOMETRIES = [
    {"name": "standard", "L": (0, 1, 0), "E": (0, 3, 6)},
    {"name": "compact", "L": (0, 1, 0), "E": (0, 1, 2)},
    {"name": "spread", "L": (0, 2, 0), "E": (0, 2, 4)},
    {"name": "wide", "L": (0, 1, 3), "E": (2, 4, 5)},
]

NEUTRINO_GEOMETRIES = [
    {"name": "standard", "L": (0, 1, 0), "N": (0, 3, 6)},
    {"name": "compact", "L": (0, 1, 0), "N": (0, 1, 2)},
    {"name": "spread", "L": (0, 2, 0), "N": (0, 2, 4)},
]

# Representative parameter grid (long budget: full Cartesian subset)
SIGMA_VALS = [2.5, 4.0, 5.5]
K_VALS = [1.0, 1.4, 1.75, 2.0]
ETA_VALS = [2.0, 3.0, 3.7, 4.5]
EPS_VALS = [0.10, 0.25, 0.41]
G_ENV_VALS = [0.45, 0.55, 0.65, 0.75]
ALPHA = 2.5


def lepton_mass_loss(Ye):
    _, S, _ = np.linalg.svd(Ye, full_matrices=False)
    targets = np.array([1.777, 0.1057, 0.000511])
    scale = targets[0] / S[0] if S[0] > 0 else 0.0
    masses = S * scale
    loss = 0.0
    for m, t in zip(masses, targets):
        if m > 0 and t > 0:
            loss += np.log(m / t) ** 2
        else:
            loss += 100.0
    return float(loss)


def sample_row(sector, regime, geometry_name, params, Y, extra=None):
    info = compute_yukawa_information(Y)
    row = {
        "sector": sector,
        "regime": regime,
        "geometry": geometry_name,
        "entropy": info["entropy"],
        "effective_rank": info["effective_rank"],
        "off_diagonal_entropy": info["off_diagonal_entropy"],
        "trace_yydag": info["trace_yydag"],
    }
    row.update(params)
    if extra:
        row.update(extra)
    return row


def sweep_quarks():
    rows = []
    for geom in QUARK_GEOMETRIES:
        Q, U, D = geom["Q"], geom["U"], geom["D"]
        for sigma, k, eta, eps_u, eps_d in itertools.product(
            SIGMA_VALS, K_VALS, ETA_VALS, EPS_VALS, EPS_VALS
        ):
            Yu, Yd = compute_quark_yukawas(
                Q, U, D, sigma, k, ALPHA, eta, eps_u, eps_d
            )
            obs = compute_quark_observables(Yu, Yd)
            loss_ckm = compute_ckm_loss(obs)
            loss_mass = compute_mass_loss(obs)
            params = dict(sigma=sigma, k=k, eta=eta, eps_u=eps_u, eps_d=eps_d, g_env=np.nan)
            for label, Y in [("quark_up", Yu), ("quark_down", Yd)]:
                rows.append(
                    sample_row(
                        label,
                        REGIME_QUARK,
                        geom["name"],
                        params,
                        Y,
                        extra=dict(L_ckm=loss_ckm, L_mass=loss_mass, L_total=loss_ckm + loss_mass),
                    )
                )
    return rows


def sweep_leptons():
    rows = []
    for geom in LEPTON_GEOMETRIES:
        L, E = geom["L"], geom["E"]
        for sigma, k, eta, eps in itertools.product(SIGMA_VALS, K_VALS, ETA_VALS, EPS_VALS):
            Ye = compute_yukawa_matrix(L, E, sigma, k, ALPHA, eta, eps)
            L_mass = lepton_mass_loss(Ye)
            params = dict(sigma=sigma, k=k, eta=eta, eps=eps, g_env=np.nan)
            rows.append(
                sample_row(
                    "charged_lepton",
                    REGIME_LEPTON,
                    geom["name"],
                    params,
                    Ye,
                    extra=dict(L_ckm=np.nan, L_mass=L_mass, L_total=L_mass),
                )
            )
    return rows


def sweep_neutrinos():
    rows = []
    for geom in NEUTRINO_GEOMETRIES:
        L, N = geom["L"], geom["N"]
        for sigma, k, eta, eps_nu, eps_e, g_env in itertools.product(
            SIGMA_VALS, K_VALS, ETA_VALS, EPS_VALS, EPS_VALS, G_ENV_VALS
        ):
            Ynu = compute_yukawa_matrix(L, N, sigma * g_env, k, ALPHA, eta, eps_nu)
            Ye = compute_yukawa_matrix(L, N, sigma, k, ALPHA, eta, eps_e)
            obs = compute_neutrino_observables(Ynu, Ye)
            L_pmns = compute_pmns_loss(obs)
            params = dict(sigma=sigma, k=k, eta=eta, eps_nu=eps_nu, eps_e=eps_e, g_env=g_env)
            rows.append(
                sample_row(
                    "neutrino",
                    REGIME_NEUTRINO,
                    geom["name"],
                    params,
                    Ynu,
                    extra=dict(L_ckm=L_pmns, L_mass=np.nan, L_total=L_pmns),
                )
            )
    return rows


def pearson_safe(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    if np.std(x[mask]) < 1e-15 or np.std(y[mask]) < 1e-15:
        return np.nan, np.nan
    r, p = stats.pearsonr(x[mask], y[mask])
    return float(r), float(p)


def main():
    print("=" * 70)
    print("REGIME–ENTROPY CORRELATION (Path A, pre-registered falsifier)")
    print("=" * 70)
    print(f"Threshold |r| >= {CORRELATION_THRESHOLD}, min samples >= {MIN_SAMPLES}\n")

    rows = sweep_quarks() + sweep_leptons() + sweep_neutrinos()
    n = len(rows)
    print(f"Total samples: {n}")

    regimes = np.array([r["regime"] for r in rows])
    measures = ["entropy", "effective_rank", "off_diagonal_entropy"]

    print("\n--- Correlation vs regime label (0=quark, 1=lepton, 2=neutrino) ---")
    correlations = {}
    for m in measures:
        vals = np.array([r[m] for r in rows])
        r, p = pearson_safe(regimes, vals)
        correlations[f"{m}_vs_regime"] = r
        print(f"  {m:25s}  r = {r:+.4f}  p = {p:.2e}" if np.isfinite(r) else f"  {m:25s}  r = nan")

    # Sector-mean comparison (regime separation without param noise)
    print("\n--- Mean by sector (pooled over params/geometries) ---")
    sector_means = {}
    for sector in sorted(set(r["sector"] for r in rows)):
        subset = [r for r in rows if r["sector"] == sector]
        sector_means[sector] = {
            m: float(np.mean([r[m] for r in subset])) for m in measures
        }
        sm = sector_means[sector]
        print(
            f"  {sector:20s}  S={sm['entropy']:.4f}  rank={sm['effective_rank']:.4f}  "
            f"off_S={sm['off_diagonal_entropy']:.4f}  (n={len(subset)})"
        )

    # g_env vs entropy (neutrino only)
    nu_rows = [r for r in rows if r["sector"] == "neutrino"]
    g_vals = [r["g_env"] for r in nu_rows]
    s_vals = [r["entropy"] for r in nu_rows]
    r_g, p_g = pearson_safe(g_vals, s_vals)
    correlations["entropy_vs_g_env_neutrino"] = r_g
    print(f"\n--- Neutrino: S vs g_env ---")
    print(f"  r = {r_g:+.4f}  p = {p_g:.2e}" if np.isfinite(r_g) else "  r = nan")

    # Loss vs entropy (where defined)
    for loss_key in ["L_ckm", "L_mass", "L_total"]:
        xs = [r["entropy"] for r in rows if np.isfinite(r.get(loss_key, np.nan))]
        ys = [r[loss_key] for r in rows if np.isfinite(r.get(loss_key, np.nan))]
        r_l, p_l = pearson_safe(xs, ys)
        correlations[f"entropy_vs_{loss_key}"] = r_l
        if np.isfinite(r_l):
            print(f"\n--- S vs {loss_key} ---")
            print(f"  r = {r_l:+.4f}  p = {p_l:.2e}  (n={len(xs)})")

    # Falsifier verdict
    primary = [correlations.get(f"{m}_vs_regime") for m in measures]
    primary_finite = [abs(r) for r in primary if np.isfinite(r)]
    max_abs_r = max(primary_finite) if primary_finite else 0.0
    passed = n >= MIN_SAMPLES and max_abs_r >= CORRELATION_THRESHOLD

    print("\n" + "=" * 70)
    print("PRE-REGISTERED FALSIFIER VERDICT")
    print("=" * 70)
    print(f"  Samples: {n} (required >= {MIN_SAMPLES})")
    print(f"  Max |r| vs regime (entropy, rank, off-diag): {max_abs_r:.4f}")
    if passed:
        print("  VERDICT: NOT REFUTED — at least one measure exceeds threshold")
        verdict = "not_refuted"
    else:
        print("  VERDICT: REFUTED — regime labels do not correlate with info measures")
        verdict = "refuted"

    out = os.path.join(os.path.dirname(__file__), "results", "12_regime_entropy_correlation.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("Regime-entropy correlation diagnostic\n")
        f.write(f"threshold: {CORRELATION_THRESHOLD}\n")
        f.write(f"min_samples: {MIN_SAMPLES}\n")
        f.write(f"n_samples: {n}\n")
        f.write(f"verdict: {verdict}\n\n")
        for k, v in correlations.items():
            f.write(f"{k}: {v}\n")
        f.write("\nsector_means:\n")
        for sector, sm in sector_means.items():
            f.write(f"  {sector}: {sm}\n")
    print(f"\nWrote {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
