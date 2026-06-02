#!/usr/bin/env python3
"""
QED Fisher / coherence diagnostic (Path A pivot).

Sweeps geometries and kernel parameters (grid aligned with diagnostic 12),
computes QFI proxies, coherence measures, and correlates with SM mixing proxies.

Pre-registered falsifier:
  If max |r| between QFI/coherence measures and mixing observables < 0.25
  across pooled sectors, report NO QED-info mechanism for this measure class.

Compares to refuted S(rho_Y) baseline (diagnostic 12: max |r| vs regime ~ 0.05).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import itertools
import numpy as np
from scipy import stats

from kernel import compute_quark_yukawas, compute_yukawa_matrix
from flavor_information import compute_yukawa_information
from qed_information import compute_qed_yukawa_information
from observables import compute_quark_observables, compute_neutrino_observables

CORRELATION_THRESHOLD = 0.25
MIN_SAMPLES = 30
REFUTED_ENTROPY_MAX_R = 0.05  # diagnostic 12 regime baseline

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
ALPHA = 2.5

QED_MEASURES = [
    "qfi_mean_elements",
    "qfi_mean_singular_values",
    "coherence_l1",
    "off_diagonal_ratio",
    "distinguishability_uniform",
]


def mixing_proxy(sector, Y, obs=None):
    """SM mixing magnitude proxy per sector."""
    if sector.startswith("quark") and obs is not None:
        return obs["Vus"] + obs["Vcb"] + obs["Vub"]
    if sector == "charged_lepton":
        return float(np.sum(np.abs(Y - np.diag(np.diag(Y)))))
    if sector == "neutrino" and obs is not None:
        return obs["theta12"] + obs["theta23"] + obs["theta13"]
    return np.nan


def sample_row(sector, geometry_name, Y, mix_off, regime):
    info = compute_yukawa_information(Y)
    qed = compute_qed_yukawa_information(Y)
    row = {
        "sector": sector,
        "geometry": geometry_name,
        "regime": regime,
        "mix_off": mix_off,
        "entropy": info["entropy"],
    }
    row.update(qed)
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
            for label, Y in [("quark_up", Yu), ("quark_down", Yd)]:
                rows.append(
                    sample_row(
                        label,
                        geom["name"],
                        Y,
                        mixing_proxy(label, Y, obs),
                        0,
                    )
                )
    return rows


def sweep_leptons():
    rows = []
    for geom in LEPTON_GEOMETRIES:
        L, E = geom["L"], geom["E"]
        for sigma, k, eta, eps in itertools.product(SIGMA_VALS, K_VALS, ETA_VALS, EPS_VALS):
            Ye = compute_yukawa_matrix(L, E, sigma, k, ALPHA, eta, eps)
            rows.append(
                sample_row(
                    "charged_lepton",
                    geom["name"],
                    Ye,
                    mixing_proxy("charged_lepton", Ye),
                    1,
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
            rows.append(
                sample_row(
                    "neutrino",
                    geom["name"],
                    Ynu,
                    mixing_proxy("neutrino", Ynu, obs),
                    2,
                )
            )
    return rows


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


def main():
    print("=" * 70)
    print("QED FISHER / COHERENCE vs MIXING (Path A pivot)")
    print("=" * 70)
    print(f"Falsifier: max |r| >= {CORRELATION_THRESHOLD} vs mixing (pooled)\n")

    rows = sweep_quarks() + sweep_leptons() + sweep_neutrinos()
    n = len(rows)
    print(f"Total samples: {n}")

    mix = np.array([r["mix_off"] for r in rows])
    correlations = {}

    print("\n--- Pooled: QED measures vs mixing proxy ---")
    for m in QED_MEASURES:
        vals = np.array([r[m] for r in rows])
        r, p = pearson_safe(vals, mix)
        correlations[f"{m}_vs_mix_pooled"] = r
        if np.isfinite(r):
            print(f"  {m:32s}  r = {r:+.4f}  p = {p:.2e}")

    # Refuted baseline: S(rho_Y) vs mixing (same sweep)
    ent = np.array([r["entropy"] for r in rows])
    r_s, p_s = pearson_safe(ent, mix)
    correlations["entropy_vs_mix_pooled"] = r_s
    print(f"\n  {'entropy (refuted baseline)':32s}  r = {r_s:+.4f}")

    print("\n--- By sector type ---")
    by_type = {}
    for st in ("quark_up", "quark_down", "charged_lepton", "neutrino"):
        sub = [r for r in rows if r["sector"] == st]
        by_type[st] = sub
        if len(sub) < 5:
            continue
        print(f"\n  [{st}] n={len(sub)}")
        for m in QED_MEASURES:
            vals = [r[m] for r in sub]
            ys = [r["mix_off"] for r in sub]
            r, p = pearson_safe(vals, ys)
            correlations[f"{m}_vs_mix_{st}"] = r
            if np.isfinite(r):
                print(f"    {m:30s}  r = {r:+.4f}")

    pooled_keys = [f"{m}_vs_mix_pooled" for m in QED_MEASURES]
    pooled_finite = [
        abs(correlations[k]) for k in pooled_keys if np.isfinite(correlations.get(k, np.nan))
    ]
    max_abs_r = max(pooled_finite) if pooled_finite else 0.0

    passed = n >= MIN_SAMPLES and max_abs_r >= CORRELATION_THRESHOLD

    print("\n" + "=" * 70)
    print("PRE-REGISTERED FALSIFIER VERDICT")
    print("=" * 70)
    print(f"  Samples: {n}")
    print(f"  Max |r| (QED measures vs mixing, pooled): {max_abs_r:.4f}")
    print(f"  Refuted S(rho_Y) regime baseline (diag 12): ~{REFUTED_ENTROPY_MAX_R}")
    if passed:
        print("  VERDICT: NOT REFUTED — at least one QED measure exceeds threshold vs mixing")
        verdict = "not_refuted"
    else:
        print("  VERDICT: REFUTED — no QED-info mechanism for this measure class (pooled)")
        verdict = "refuted"

    out = os.path.join(os.path.dirname(__file__), "results", "15_qed_fisher_yukawa.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("QED Fisher/coherence vs mixing diagnostic\n")
        f.write(f"threshold: {CORRELATION_THRESHOLD}\n")
        f.write(f"n_samples: {n}\n")
        f.write(f"verdict: {verdict}\n")
        f.write(f"max_abs_r_pooled_qed: {max_abs_r}\n\n")
        for k, v in sorted(correlations.items()):
            f.write(f"{k}: {v}\n")
    print(f"\nWrote {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
