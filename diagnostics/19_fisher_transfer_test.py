#!/usr/bin/env python3
"""
Pre-registered Fisher transfer test (quark → lepton).

Protocol (registered BEFORE run):
  1. Fit quarks → θ*_quark, experimental Fisher F_quark at that point
  2. Freeze universal subset (σ, k, α, η) from quark fit
  3. Transfer to leptons with only ε_e free; measure lepton loss
  4. Compare F_lepton at transfer point vs F_quark (alignment, CR bounds)
  5. Independent free lepton fit → actual required Δθ vs Fisher CR prediction

Prediction: If mechanism holds, Fisher at quark minimum predicts which lepton
parameters must deviate and by how much (compare to free lepton fit).

Pre-registered falsifiers:
  A) Free-fit Δθ on (σ,k,α,η) all within z·√CR from quark Fisher BUT frozen
     transfer loss ≥ FROZEN_LOSS_BAD_THRESHOLD → mechanism REFUTED
  B) Fisher principal-eigenvector alignment at transfer < ALIGNMENT_THRESHOLD
     AND frozen loss ≥ FROZEN_LOSS_BAD_THRESHOLD → mechanism REFUTED
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from fisher_transfer import (
    UNIVERSAL_PARAM_NAMES,
    evaluate_fisher_transfer_verdict,
    run_fisher_transfer_analysis,
)

ALIGNMENT_THRESHOLD = 0.50
FROZEN_LOSS_BAD_THRESHOLD = 797.0
CR_Z = 2.0
N_SEEDS = 5
MAXITER = 200


def _fmt_theta(names, theta):
    return ", ".join(f"{n}={theta[i]:.4f}" for i, n in enumerate(names))


def main():
    print("=" * 70)
    print("PRE-REGISTERED FISHER TRANSFER TEST (quark → lepton)")
    print("=" * 70)
    print(f"Alignment threshold:     {ALIGNMENT_THRESHOLD}")
    print(f"Frozen loss bad cutoff:  {FROZEN_LOSS_BAD_THRESHOLD}")
    print(f"CR bound multiplier z:   {CR_Z}")
    print(f"Seeds / maxiter:         {N_SEEDS} / {MAXITER}\n")

    analysis = run_fisher_transfer_analysis(n_seeds=N_SEEDS, maxiter=MAXITER, cr_z=CR_Z)
    verdict_info = evaluate_fisher_transfer_verdict(
        analysis,
        alignment_threshold=ALIGNMENT_THRESHOLD,
        frozen_loss_bad_threshold=FROZEN_LOSS_BAD_THRESHOLD,
    )

    q = analysis["quark_fit"]
    frozen = analysis["frozen"]
    free = analysis["free"]

    print("--- Step 1: Quark fit ---")
    print(f"  Loss (total): {q['loss_total']:.6f}  CKM: {q['loss_ckm']:.6f}  mass: {q['loss_mass']:.6f}")
    print(f"  θ_quark: {_fmt_theta(analysis['fisher_quark']['param_names'], analysis['quark_theta'])}")
    fq = analysis["fisher_quark"]["summaries"]
    print(
        f"  F_quark: logdet={fq['logdet_fisher']:.2f}  "
        f"eff_rank={fq['effective_rank']:.2f}  cond={fq['condition_number']:.1e}"
    )

    print("\n--- Step 2: Frozen transfer (σ,k,α,η fixed; ε_e free) ---")
    print(f"  Lepton loss: {frozen['loss']:.4f}")
    print(f"  ε_e = {frozen['eps_e']:.4f}")
    print(f"  m_mu = {frozen['m_mu']:.6f} GeV (target 0.1057)")
    fa = analysis["alignment_at_transfer"]
    print(f"  Fisher alignment (quark vs lepton at transfer): {fa:.4f}")

    print("\n--- Step 3: Free lepton fit (baseline) ---")
    print(f"  Lepton loss: {free['loss']:.4f}")
    print(f"  θ_lepton: {_fmt_theta(analysis['fisher_lepton_free']['param_names'], free['theta'])}")
    print(f"  Fisher alignment (quark vs lepton at free optimum): {analysis['alignment_at_free_optimum']:.4f}")

    print("\n--- Universal parameter deviations (free − quark) vs quark CR bounds ---")
    for name in UNIVERSAL_PARAM_NAMES:
        delta = analysis["universal_deltas"][name]
        cr = analysis["cr_shared"][name]
        ratio = analysis["cr_ratios"][name]
        bound = CR_Z * np.sqrt(cr) if np.isfinite(cr) and cr > 0 else float("inf")
        print(
            f"  Δ{name:5s} = {delta:+.4f}  "
            f"z√CR = {bound:.4f}  ratio = {ratio:.3f}"
        )
    print(f"  All Δ within {CR_Z}√CR? {analysis['deltas_within_cr']}")

    print("\n" + "=" * 70)
    print("PRE-REGISTERED FALSIFIER VERDICT")
    print("=" * 70)
    print(f"  Frozen loss:              {verdict_info['frozen_loss']:.4f}")
    print(f"  Loss bad (≥ {FROZEN_LOSS_BAD_THRESHOLD}): {verdict_info['loss_bad']}")
    print(f"  Alignment at transfer:    {verdict_info['alignment_at_transfer']:.4f}")
    print(f"  Falsifier A (CR + bad loss): {verdict_info['falsifier_a']}")
    print(f"  Falsifier B (align + bad loss): {verdict_info['falsifier_b']}")
    print(f"  VERDICT: {verdict_info['verdict'].upper()}")
    print(f"  Reason: {verdict_info['reason']}")

    out = os.path.join(os.path.dirname(__file__), "results", "19_fisher_transfer_test.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("Pre-registered Fisher transfer test (quark → lepton)\n")
        f.write(f"alignment_threshold: {ALIGNMENT_THRESHOLD}\n")
        f.write(f"frozen_loss_bad_threshold: {FROZEN_LOSS_BAD_THRESHOLD}\n")
        f.write(f"cr_z: {CR_Z}\n")
        f.write(f"n_seeds: {N_SEEDS}\n")
        f.write(f"verdict: {verdict_info['verdict']}\n")
        f.write(f"refuted: {verdict_info['refuted']}\n")
        f.write(f"reason: {verdict_info['reason']}\n")
        f.write(f"quark_loss_total: {q['loss_total']}\n")
        f.write(f"frozen_lepton_loss: {frozen['loss']}\n")
        f.write(f"free_lepton_loss: {free['loss']}\n")
        f.write(f"alignment_at_transfer: {analysis['alignment_at_transfer']}\n")
        f.write(f"alignment_at_free_optimum: {analysis['alignment_at_free_optimum']}\n")
        f.write(f"deltas_within_cr: {analysis['deltas_within_cr']}\n")
        f.write(f"falsifier_a: {verdict_info['falsifier_a']}\n")
        f.write(f"falsifier_b: {verdict_info['falsifier_b']}\n")
        for name in UNIVERSAL_PARAM_NAMES:
            f.write(f"delta_{name}: {analysis['universal_deltas'][name]}\n")
            f.write(f"cr_{name}: {analysis['cr_shared'][name]}\n")
            f.write(f"cr_ratio_{name}: {analysis['cr_ratios'][name]}\n")
        fq_s = analysis["fisher_quark_shared_summaries"]
        fl_s = analysis["fisher_lepton_shared_summaries"]
        f.write(f"fisher_quark_shared_logdet: {fq_s['logdet_fisher']}\n")
        f.write(f"fisher_lepton_shared_logdet: {fl_s['logdet_fisher']}\n")

    print(f"\nWrote {out}")
    return 1 if verdict_info["refuted"] else 0


if __name__ == "__main__":
    sys.exit(main())
