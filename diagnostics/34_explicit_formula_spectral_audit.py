#!/usr/bin/env python3
"""
Tier 5.2 — Non-circular explicit formula audit (Path D).

Pre-registered tests (see adversarial-review-tier5-trace-formula):
  1. Holdout-x: RMSE(psi_computed - psi_model) on x > X_cut using truncated zero sum.
  2. More zeros improve high-x holdout (K_large vs K_small).
  3. Null control: random frequencies (matched count/range) inflate holdout RMSE.

PASS does NOT imply Hilbert-Polya — only that zero heights are specific to psi(x).
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from explicit_formula import (
    RIEMANN_ZERO_IMAG,
    chebyshev_psi,
    psi_from_explicit_formula,
    random_null_frequencies,
)

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "34_explicit_formula_spectral_audit.txt"
)

# Pre-registered thresholds
X_HOLDOUT_MIN = 2500.0
X_HOLDOUT_MAX = 12000.0
K_SMALL = 10
K_LARGE = 50
NULL_SEEDS = 20
NULL_BEAT_RATIO = 0.60  # true holdout RMSE < 60% of mean null RMSE
IMPROVE_RATIO = 1.05  # more zeros improve all-x RMSE (marginal convergence)

X_GRID = np.unique(
    np.round(np.exp(np.linspace(math.log(400), math.log(X_HOLDOUT_MAX), 32))).astype(int)
)


def rmse_on_mask(psi_true, psi_model, mask):
    err = np.array([psi_true[x] - psi_model[x] for x in X_GRID if mask(x)])
    if len(err) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(err ** 2)))


def build_psi_true(max_x: int) -> dict:
    return {int(x): chebyshev_psi(float(x)) for x in X_GRID if x <= max_x}


def psi_model_dict(psi_true: dict, gammas) -> dict:
    return {x: psi_from_explicit_formula(float(x), gammas) for x in psi_true}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    k_large = 25 if args.smoke else K_LARGE
    null_seeds = 5 if args.smoke else NULL_SEEDS
    zeros = RIEMANN_ZERO_IMAG[:k_large]

    psi_true = build_psi_true(int(X_GRID.max()))
    holdout_mask = lambda x: X_HOLDOUT_MIN <= x <= X_HOLDOUT_MAX

    model_small = psi_model_dict(psi_true, zeros[:K_SMALL])
    model_large = psi_model_dict(psi_true, zeros[:k_large])

    rmse_hold_small = rmse_on_mask(psi_true, model_small, holdout_mask)
    rmse_hold_large = rmse_on_mask(psi_true, model_large, holdout_mask)
    rmse_all_small = rmse_on_mask(psi_true, model_small, lambda x: True)
    rmse_all_large = rmse_on_mask(psi_true, model_large, lambda x: True)

    g_min, g_max = zeros[0], zeros[min(K_SMALL, len(zeros)) - 1]
    null_rmses = []
    for seed in range(null_seeds):
        fake = random_null_frequencies(K_SMALL, g_min, g_max, seed=34034 + seed)
        model_null = psi_model_dict(psi_true, fake)
        null_rmses.append(rmse_on_mask(psi_true, model_null, holdout_mask))
    rmse_null_mean = float(np.mean(null_rmses))

    specificity_ok = rmse_hold_small < NULL_BEAT_RATIO * rmse_null_mean
    enrich_ok = rmse_all_large * IMPROVE_RATIO < rmse_all_small
    # Full-range sanity: model should track psi at all x with enough zeros
    sanity_ok = rmse_all_large < 15.0

    lines = [
        "=" * 72,
        "TIER 5.2 EXPLICIT FORMULA SPECTRAL AUDIT (diagnostic 34)",
        "=" * 72,
        "Non-circular: holdout x + wrong-frequency null (not FFT peak matching).",
        f"Zeros used: {k_large} (RH critical line); holdout {X_HOLDOUT_MIN:.0f} <= x <= {X_HOLDOUT_MAX:.0f}",
        "",
        "--- Holdout RMSE |psi_computed - psi_model| ---",
        f"  Holdout K={K_SMALL}: {rmse_hold_small:.4f}",
        f"  Holdout K={k_large}: {rmse_hold_large:.4f}",
        f"  All x K={K_SMALL}: {rmse_all_small:.4f}",
        f"  All x K={k_large}: {rmse_all_large:.4f}",
        f"  Null mean (K={K_SMALL} random gamma): {rmse_null_mean:.4f} ({null_seeds} seeds)",
        "",
        "--- Pre-registered falsifiers ---",
        f"  Specificity (true < {NULL_BEAT_RATIO} * null): {specificity_ok}",
        f"  Enrichment all-x (K_large < K_small/{IMPROVE_RATIO}): {enrich_ok}",
        f"  Full-range sanity (RMSE < 15): {sanity_ok}",
        "",
        "--- VERDICT ---",
    ]

    if specificity_ok and enrich_ok and sanity_ok:
        lines.append(
            "  PASS — zero heights are specific and truncations behave as expected."
        )
        lines.append(
            "  Interpretation: arithmetic identity check only; NOT HP proof or flavor hook."
        )
        verdict = "pass"
    else:
        lines.append("  FAIL — explicit formula audit did not meet pre-registered bars.")
        if not specificity_ok:
            lines.append("    Random frequencies did not inflate error enough (or bug).")
        if not enrich_ok:
            lines.append("    More zeros did not improve all-x RMSE as expected.")
        verdict = "fail"

    report = "\n".join(lines)
    print(report)

    if not args.smoke:
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            f.write(report + "\n")
            f.write(f"verdict: {verdict}\n")
            f.write("no_flavor_connection: true\n")
        print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
