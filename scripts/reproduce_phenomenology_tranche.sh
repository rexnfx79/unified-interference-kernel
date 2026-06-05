#!/usr/bin/env bash
# Headline phenomenology artifacts + observables sanity (see knowledge/wiki/synthesis/future-work.md)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
[[ -x "$PY" ]] || PY=python3

echo "=== Frozen diagnostic reports ==="
for f in diagnostics/results/21_quark_phenomenology_holdout.txt \
         diagnostics/results/22_lepton_phenomenology_sweep.txt \
         diagnostics/results/23_neutrino_phenomenology_sweep.txt \
         diagnostics/results/27_quark_joint_loss_holdout.txt \
         diagnostics/results/28_neutrino_masses_pmns_joint.txt \
         diagnostics/results/30_quark_geometry_followup.txt \
         diagnostics/results/32_quark_tier2_ansatz.txt \
         diagnostics/results/33_tier3_theory_bridges.txt \
         diagnostics/results/36_tier1_phase_fix_audit.txt \
         diagnostics/results/34_explicit_formula_spectral_audit.txt \
         diagnostics/results/35_jacobi_inverse_kernel_phase.txt \
         diagnostics/results/37_tier5_landscape_rmt.txt \
         diagnostics/results/38_tier5_qed_prime_spectral_audit.txt \
         diagnostics/results/39_joint_loss_landscape_cartography.txt \
         diagnostics/results/40_n4_geometry_strict_predictor.txt; do
  if [[ -f "$f" ]]; then
    echo "  OK  $f"
  else
    echo "  MISSING  $f"
  fi
done

echo ""
echo "=== Observables pipeline (SVD phase fix) ==="
PYTHONPATH=src "$PY" -m pytest tests/test_observables.py::test_fix_svd_phases_preserves_reconstruction \
  tests/test_cp_observables.py -q

echo ""
echo "Protocol: knowledge/wiki/synthesis/survivor-protocol-preregistered.md"
echo "Future work: knowledge/wiki/synthesis/future-work.md"
