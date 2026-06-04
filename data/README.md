# Legacy optimization artifacts

These CSV/JSON files come from **pre-phenomenology-tranche** optimizers (`scripts/01_optimize_quarks.py`, etc.) and early geometry follow-ups. They are **not** the canonical strict-survivor protocol.

## Protocol differences (legacy vs current)

| Aspect | Legacy (`quark_results.csv`, etc.) | Canonical (diag 21–31, 36) |
|--------|-----------------------------------|----------------------------|
| Objective | Full-sector or CSV-era totals | Train/holdout split, pre-registered tolerances |
| Survivor label | Often ±40% or 10% loose | PDG-relative strict (see wiki survivor protocol) |
| Geometry | `scripts/01` enumeration | Phenomenology sampler seed 21021 + diag 30 exhaustive |

## Files

| File | Role |
|------|------|
| `quark_results.csv` | Legacy quark DE sweep (Jan 2026); baseline for diag 29 |
| `quark_geometry_followup_*.csv/json` | Diag 30 exhaustive + shell-5 re-baseline |
| `charged_lepton_results.csv`, `neutrino_results.csv` | Legacy sector sweeps |
| `transfer_test_results.csv` | Cross-sector parameter transfer (refuted) |

## Reproduce headline results

Use frozen diagnostic reports, not these CSVs alone:

```bash
./scripts/reproduce_phenomenology_tranche.sh
```
