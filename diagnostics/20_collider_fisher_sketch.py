#!/usr/bin/env python3
"""
Collider-accessible Fisher sketch (Path D / experimental scope note).

Full collider Fisher (event-level likelihood, systematic correlations, run
combinations) is OUT OF REPO SCOPE — requires experiment likelihood codes.

This sketch compares PDG-weighted Fisher information content when restricting
observable sets to subsets tagged as more vs less "collider-direct":

  - mixing_only: Vus, Vcb, Vub (CKM from weak decays / |V_cb| from B decays)
  - collider_plus_mixing: above + mc (lattice/QCD input still dominates mc)
  - full_pdg: all seven quark observables used in diagnostic 17

Compares log det F and effective rank for the same θ point to quantify how
much kernel-parameter information lives in collider-accessible channels alone.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from experimental_fisher import (
    QUARK_OBS_KEYS,
    QUARK_PARAM_NAMES,
    fisher_information_matrix,
    fisher_scalar_summaries,
    numerical_jacobian,
    _make_quark_mu_fn,
)
from fisher_transfer import DEFAULT_QUARK_GEOM, fit_quarks

# Observable subsets (ordered)
OBS_MIXING_ONLY = ["Vus", "Vcb", "Vub"]
OBS_COLLIDER_PLUS = ["Vus", "Vcb", "Vub", "mc"]
OBS_FULL = list(QUARK_OBS_KEYS)

SUBSETS = {
    "mixing_only": OBS_MIXING_ONLY,
    "collider_plus_mixing": OBS_COLLIDER_PLUS,
    "full_pdg": OBS_FULL,
}


def fisher_for_obs_subset(geom, theta, obs_keys):
    mu_fn_full = _make_quark_mu_fn(geom["Q"], geom["U"], geom["D"])
    key_to_idx = {k: i for i, k in enumerate(QUARK_OBS_KEYS)}

    def mu_subset(th):
        return np.array([mu_fn_full(th)[key_to_idx[k]] for k in obs_keys])

    J = numerical_jacobian(mu_subset, theta, eps=1e-5)
    F = fisher_information_matrix(J, obs_keys)
    return F, fisher_scalar_summaries(F)


def main():
    print("=" * 70)
    print("COLLIDER-ACCESSIBLE FISHER SKETCH")
    print("=" * 70)
    print("Scope: PDG Jacobian Fisher only — NOT event-level collider likelihood.\n")

    quark = fit_quarks(n_seeds=3, maxiter=120)
    theta = quark["theta"]
    geom = DEFAULT_QUARK_GEOM

    print(f"Reference θ from quark fit: σ={theta[0]:.3f}, k={theta[1]:.3f}, η={theta[3]:.3f}")
    print(f"Quark fit loss: {quark['loss_total']:.6f}\n")

    rows = {}
    print("--- Fisher information by observable subset ---")
    for label, keys in SUBSETS.items():
        F, summaries = fisher_for_obs_subset(geom, theta, keys)
        rows[label] = {"keys": keys, "summaries": summaries, "fisher": F}
        print(f"  [{label}] n_obs={len(keys)}")
        print(f"    log det F = {summaries['logdet_fisher']:.2f}")
        print(f"    eff rank  = {summaries['effective_rank']:.2f}")
        print(f"    cond      = {summaries['condition_number']:.1e}")

    full_logdet = rows["full_pdg"]["summaries"]["logdet_fisher"]
    mix_logdet = rows["mixing_only"]["summaries"]["logdet_fisher"]
    if np.isfinite(full_logdet) and np.isfinite(mix_logdet) and full_logdet > 0:
        frac_msg = f"Mixing-only log det F is {mix_logdet:.2f} vs full {full_logdet:.2f} ({100 * mix_logdet / full_logdet:.1f}% of full)."
    else:
        frac_msg = (
            f"Mixing-only log det F = {mix_logdet:.2f} (non-PDG subset); "
            f"full PDG log det F = {full_logdet:.2f} — mixing alone does not carry mass information."
        )
    frac = mix_logdet / full_logdet if np.isfinite(full_logdet) and full_logdet > 0 else float("nan")

    print("\n--- Scope assessment ---")
    print("  Full collider Fisher requires:")
    print("    - Experiment likelihoods (LHCb, Belle II, CMS/BMTS τ, ...)")
    print("    - Systematic covariance across analyses")
    print("    - Joint fit with lattice/QCD for mc, md, ms — not separable in repo")
    print(f"  {frac_msg}")
    print("  Kernel parameters (σ, k, α, η) remain under-identified from mixing alone")
    print(f"    (eff rank mixing={rows['mixing_only']['summaries']['effective_rank']:.2f} "
          f"vs full={rows['full_pdg']['summaries']['effective_rank']:.2f}).")

    print("\n  VERDICT: Collider-only Fisher sketch is TOO THIN for mechanism tests here.")
    print("  Path D promotion still requires trace-formula / spectral hooks — not this Jacobian.")

    out = os.path.join(os.path.dirname(__file__), "results", "20_collider_fisher_sketch.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("Collider-accessible Fisher sketch\n")
        f.write("scope: pdg_jacobian_only_not_event_likelihood\n")
        f.write(f"quark_fit_loss: {quark['loss_total']}\n")
        for label, data in rows.items():
            s = data["summaries"]
            f.write(f"{label}_n_obs: {len(data['keys'])}\n")
            f.write(f"{label}_logdet: {s['logdet_fisher']}\n")
            f.write(f"{label}_eff_rank: {s['effective_rank']}\n")
            f.write(f"{label}_condition: {s['condition_number']}\n")
        f.write(f"mixing_logdet_fraction_of_full: {frac}\n")
        f.write("verdict: too_thin_for_mechanism_test\n")
        f.write("full_collider_fisher: out_of_repo_scope\n")

    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
