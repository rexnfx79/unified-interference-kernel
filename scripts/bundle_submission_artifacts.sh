#!/usr/bin/env bash
# Create submission_bundle/ with manuscript + frozen diagnostics + protocol pointers.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/submission_bundle"
rm -rf "$OUT"
mkdir -p "$OUT/diagnostics/results"
mkdir -p "$OUT/knowledge/wiki/synthesis"
mkdir -p "$OUT/scripts"

cp "$ROOT/manuscript.tex" "$OUT/"
[[ -f "$ROOT/manuscript.pdf" ]] && cp "$ROOT/manuscript.pdf" "$OUT/" || true
cp "$ROOT/BUILD_MANUSCRIPT.md" "$OUT/"

for f in \
  diagnostics/results/21_quark_phenomenology_holdout.txt \
  diagnostics/results/22_lepton_phenomenology_sweep.txt \
  diagnostics/results/23_neutrino_phenomenology_sweep.txt \
  diagnostics/results/27_quark_joint_loss_holdout.txt \
  diagnostics/results/28_neutrino_masses_pmns_joint.txt \
  diagnostics/results/30_quark_geometry_followup.txt \
  diagnostics/results/32_quark_tier2_ansatz.txt \
  diagnostics/results/36_tier1_phase_fix_audit.txt \
  diagnostics/results/39_joint_loss_landscape_cartography.txt \
  diagnostics/results/41_n2_haar_pmns_null.txt \
  diagnostics/results/45_n3_holdout_joint_strict_predictor.txt \
  diagnostics/results/46_n5_pmns_cp_descriptive_audit.txt; do
  if [[ -f "$ROOT/$f" ]]; then
    cp "$ROOT/$f" "$OUT/$f"
  fi
done

cp "$ROOT/knowledge/wiki/synthesis/survivor-protocol-preregistered.md" "$OUT/knowledge/wiki/synthesis/"
cp "$ROOT/knowledge/wiki/synthesis/manuscript-ledger-alignment.md" "$OUT/knowledge/wiki/synthesis/"
cp "$ROOT/knowledge/wiki/synthesis/phenomenology-methodology-export.md" "$OUT/knowledge/wiki/synthesis/"
cp "$ROOT/scripts/reproduce_phenomenology_tranche.sh" "$OUT/scripts/"
chmod +x "$OUT/scripts/reproduce_phenomenology_tranche.sh"

cat > "$OUT/README.txt" <<'EOF'
Unified Interference Kernel — phenomenology submission bundle.

Build PDF: see BUILD_MANUSCRIPT.md
Reproduce checks: scripts/reproduce_phenomenology_tranche.sh
Protocol: knowledge/wiki/synthesis/survivor-protocol-preregistered.md
Claim ledger: knowledge/wiki/synthesis/manuscript-ledger-alignment.md
EOF

echo "Created $OUT"
find "$OUT" -type f | sort
