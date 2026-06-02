#!/usr/bin/env python3
"""
Experimental Fisher / PDG Jacobian diagnostic (Path A).

Grid over geometries + kernel parameters; for each sector compute Fisher matrix
from ∂(observables)/∂θ weighted by PDG uncertainties.

Mechanism hypothesis: cross-sector Fisher geometry predicts sector parameter splits
(k_e ≠ k_q) — compare Fisher eigenstructure between sectors at comparable loss.

Pre-registered falsifier:
  If mean Fisher eigenvector alignment across sector pairs < ALIGNMENT_THRESHOLD
  AND |r| between Fisher effective rank and regime label < RANK_REGIME_THRESHOLD,
  report experimental-Fisher mechanism REFUTED.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import itertools
import numpy as np
from scipy import stats

from experimental_fisher import (
    QUARK_PARAM_NAMES,
    LEPTON_PARAM_NAMES,
    NEUTRINO_PARAM_NAMES,
    align_fisher_subspaces,
    compute_sector_experimental_fisher,
    eigenvector_alignment,
)

ALIGNMENT_THRESHOLD = 0.50
RANK_REGIME_THRESHOLD = 0.25
MIN_SAMPLES = 30
ALPHA = 2.5

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

SIGMA_VALS = [2.5, 4.0, 5.5]
K_VALS = [1.0, 1.4, 1.75, 2.0]
ETA_VALS = [2.0, 3.0, 3.7, 4.5]
EPS_VALS = [0.10, 0.25, 0.41]
G_ENV_VALS = [0.45, 0.55, 0.65, 0.75]


def pearson_safe(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return np.nan, np.nan
    if np.std(x[mask]) < 1e-15 or np.std(y[mask]) < 1e-15:
        return np.nan, np.nan
    r, p = stats.pearsonr(x[mask], y[mask])
    return float(r), float(p)


def flatten_row(result: dict) -> dict:
    s = result["summaries"]
    row = {
        "sector": result["sector"],
        "geometry": result["geometry"],
        "regime": result["regime"],
        "loss": result["loss"],
        "logdet_fisher": s["logdet_fisher"],
        "max_eigenvalue": s["max_eigenvalue"],
        "condition_number": s["condition_number"],
        "effective_rank": s["effective_rank"],
        "k_param": float(result["theta"][1]),
    }
    return row


def sweep_quarks():
    rows = []
    full = []
    for geom in QUARK_GEOMETRIES:
        for sigma, k, eta, eps_u, eps_d in itertools.product(
            SIGMA_VALS, K_VALS, ETA_VALS, EPS_VALS, EPS_VALS
        ):
            theta = np.array([sigma, k, ALPHA, eta, eps_u, eps_d], dtype=float)
            res = compute_sector_experimental_fisher("quark", geom, theta)
            rows.append(flatten_row(res))
            full.append(res)
    return rows, full


def sweep_leptons():
    rows = []
    full = []
    for geom in LEPTON_GEOMETRIES:
        for sigma, k, eta, eps in itertools.product(SIGMA_VALS, K_VALS, ETA_VALS, EPS_VALS):
            theta = np.array([sigma, k, ALPHA, eta, eps], dtype=float)
            res = compute_sector_experimental_fisher("lepton", geom, theta)
            rows.append(flatten_row(res))
            full.append(res)
    return rows, full


def sweep_neutrinos():
    rows = []
    full = []
    for geom in NEUTRINO_GEOMETRIES:
        for sigma, k, eta, eps_nu, eps_e, g_env in itertools.product(
            SIGMA_VALS, K_VALS, ETA_VALS, EPS_VALS, EPS_VALS, G_ENV_VALS
        ):
            theta = np.array([sigma, k, ALPHA, eta, eps_nu, eps_e, g_env], dtype=float)
            res = compute_sector_experimental_fisher("neutrino", geom, theta)
            rows.append(flatten_row(res))
            full.append(res)
    return rows, full


def comparable_loss_pairs(rows_a, full_a, rows_b, full_b, n_bins=5):
    """Pair samples from two sectors at comparable loss quantiles."""
    losses_a = np.array([r["loss"] for r in rows_a])
    losses_b = np.array([r["loss"] for r in rows_b])
    finite_a = np.isfinite(losses_a)
    finite_b = np.isfinite(losses_b)
    if finite_a.sum() < 5 or finite_b.sum() < 5:
        return []
    qa = np.quantile(losses_a[finite_a], np.linspace(0, 1, n_bins + 1))
    qb = np.quantile(losses_b[finite_b], np.linspace(0, 1, n_bins + 1))
    pairs = []
    for b in range(n_bins):
        mask_a = (losses_a >= qa[b]) & (losses_a <= qa[b + 1] + 1e-12)
        mask_b = (losses_b >= qb[b]) & (losses_b <= qb[b + 1] + 1e-12)
        ia = np.where(mask_a)[0]
        ib = np.where(mask_b)[0]
        if len(ia) == 0 or len(ib) == 0:
            continue
        i_a = int(ia[len(ia) // 2])
        i_b = int(ib[len(ib) // 2])
        fa, fb = full_a[i_a], full_b[i_b]
        align = align_fisher_subspaces(
            fa["principal_eigenvector"],
            fb["principal_eigenvector"],
            fa["param_names"],
            fb["param_names"],
        )
        pairs.append(
            {
                "loss_bin": b,
                "loss_a": rows_a[i_a]["loss"],
                "loss_b": rows_b[i_b]["loss"],
                "alignment": align,
                "k_a": rows_a[i_a]["k_param"],
                "k_b": rows_b[i_b]["k_param"],
            }
        )
    return pairs


def cross_sector_k_split(full_results: dict) -> dict:
    """At median Fisher effective rank, compare k distribution across sectors."""
    out = {}
    for sector, res_list in full_results.items():
        ks = [r["theta"][1] for r in res_list]
        ranks = [r["summaries"]["effective_rank"] for r in res_list]
        out[f"{sector}_k_median"] = float(np.median(ks))
        out[f"{sector}_rank_median"] = float(np.median(ranks))
    return out


def main():
    print("=" * 70)
    print("EXPERIMENTAL FISHER / PDG JACOBIAN (Path A)")
    print("=" * 70)
    print(
        f"Falsifier: alignment < {ALIGNMENT_THRESHOLD} AND "
        f"|r| rank vs regime < {RANK_REGIME_THRESHOLD}\n"
    )

    q_rows, q_full = sweep_quarks()
    l_rows, l_full = sweep_leptons()
    n_rows, n_full = sweep_neutrinos()
    all_rows = q_rows + l_rows + n_rows
    n = len(all_rows)
    print(f"Samples: quark={len(q_rows)}, lepton={len(l_rows)}, neutrino={len(n_rows)}, total={n}")

    # Fisher rank vs regime
    regimes = np.array([r["regime"] for r in all_rows])
    eff_ranks = np.array([r["effective_rank"] for r in all_rows])
    r_rank, p_rank = pearson_safe(eff_ranks, regimes)
    print(f"\n--- Fisher effective rank vs regime ---")
    print(f"  r = {r_rank:+.4f}  p = {p_rank:.2e}")

    logdets = np.array([r["logdet_fisher"] for r in all_rows])
    r_logdet, _ = pearson_safe(logdets, regimes)
    print(f"  log det F vs regime: r = {r_logdet:+.4f}")

    # Cross-sector alignment at comparable loss
    pair_sets = [
        ("quark", "lepton", q_rows, q_full, l_rows, l_full),
        ("quark", "neutrino", q_rows, q_full, n_rows, n_full),
        ("lepton", "neutrino", l_rows, l_full, n_rows, n_full),
    ]
    alignments = []
    print("\n--- Cross-sector Fisher alignment (comparable loss bins) ---")
    for name_a, name_b, ra, fa, rb, fb in pair_sets:
        pairs = comparable_loss_pairs(ra, fa, rb, fb)
        vals = [p["alignment"] for p in pairs if np.isfinite(p["alignment"])]
        mean_align = float(np.mean(vals)) if vals else float("nan")
        alignments.extend(vals)
        print(f"  {name_a} vs {name_b}: mean |cos| = {mean_align:.4f}  (n_bins={len(vals)})")
        for p in pairs:
            if np.isfinite(p["alignment"]):
                print(
                    f"    bin {p['loss_bin']}: align={p['alignment']:.3f}  "
                    f"k_{name_a[0]}={p['k_a']:.2f} k_{name_b[0]}={p['k_b']:.2f}"
                )

    mean_alignment = float(np.mean(alignments)) if alignments else 0.0

    # k split at comparable Fisher geometry (shared sigma, eta grid points)
    print("\n--- Sector k split (Fisher geometry hypothesis) ---")
    k_by_sector = cross_sector_k_split({"quark": q_full, "lepton": l_full, "neutrino": n_full})
    for k, v in sorted(k_by_sector.items()):
        print(f"  {k}: {v:.4f}")

    # Within-sector: does Fisher max eig correlate with k?
    print("\n--- Fisher max λ vs k (by sector) ---")
    k_corrs = {}
    for label, rows in [("quark", q_rows), ("lepton", l_rows), ("neutrino", n_rows)]:
        ks = [r["k_param"] for r in rows]
        lam = [r["max_eigenvalue"] for r in rows]
        r_k, _ = pearson_safe(ks, lam)
        k_corrs[label] = r_k
        print(f"  [{label}] r(max λ, k) = {r_k:+.4f}")

    rank_regime_pass = np.isfinite(r_rank) and abs(r_rank) >= RANK_REGIME_THRESHOLD
    alignment_pass = mean_alignment >= ALIGNMENT_THRESHOLD
    passed = n >= MIN_SAMPLES and (alignment_pass or rank_regime_pass)

    print("\n" + "=" * 70)
    print("PRE-REGISTERED FALSIFIER VERDICT")
    print("=" * 70)
    print(f"  Samples: {n}")
    print(f"  Mean cross-sector Fisher alignment: {mean_alignment:.4f} (threshold {ALIGNMENT_THRESHOLD})")
    print(f"  |r| effective rank vs regime: {abs(r_rank) if np.isfinite(r_rank) else 0:.4f} (threshold {RANK_REGIME_THRESHOLD})")
    if passed:
        print("  VERDICT: NOT REFUTED — Fisher geometry shows cross-sector or rank-regime signal")
        verdict = "not_refuted"
    else:
        print("  VERDICT: REFUTED — experimental Fisher does not predict sector splits")
        verdict = "refuted"

    out = os.path.join(os.path.dirname(__file__), "results", "17_experimental_fisher_pdg.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("Experimental Fisher / PDG Jacobian diagnostic\n")
        f.write(f"alignment_threshold: {ALIGNMENT_THRESHOLD}\n")
        f.write(f"rank_regime_threshold: {RANK_REGIME_THRESHOLD}\n")
        f.write(f"n_samples: {n}\n")
        f.write(f"verdict: {verdict}\n")
        f.write(f"mean_cross_sector_alignment: {mean_alignment}\n")
        f.write(f"rank_vs_regime_r: {r_rank}\n")
        f.write(f"logdet_vs_regime_r: {r_logdet}\n")
        for label, rv in k_corrs.items():
            f.write(f"max_eig_vs_k_{label}: {rv}\n")
        for k, v in k_by_sector.items():
            f.write(f"{k}: {v}\n")
    print(f"\nWrote {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
