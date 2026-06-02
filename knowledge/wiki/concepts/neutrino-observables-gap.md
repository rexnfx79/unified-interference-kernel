---
type: concept
title: Neutrino Observables Gap
tags: [flavor]
related:
  - yukawa-observables-pipeline
  - observables-extraction
  - analysis-summary
  - projection-regimes
  - knowledge-gaps-audit
island: true
bridge_tags: [flavor]
approach: effective
plausibility: watch
status: resolved
created: 2026-06-01
updated: 2026-06-01
---

# Neutrino Observables Gap

## Status: resolved (2026-06-01)

PMNS extraction is now in `src/observables.py`:

- `compute_neutrino_observables(Ynu, Ye)` — SVD + phase fix → \(U_{\mathrm{PMNS}} = U_e^\dagger U_\nu\) → \(\theta_{12}, \theta_{23}, \theta_{13}\)
- `compute_pmns_loss`, `NEUTRINO_TARGETS` (PDG 2024 radians)
- `tests/test_neutrino_observables.py` — synthetic Yukawa roundtrip

## Historical gap

Wiki and manuscript cited **45% neutrino survivor rate** while only quark extraction lived in the library. Archived CSV results in [[analysis-summary]] predated centralized PMNS code.

## Remaining caveat

Neutrino **optimization** scripts are not in the current repo snapshot; library extraction is proven, sector fits remain phenomenological ([[projection-regimes]]).

## Wiki impact

- [[yukawa-observables-pipeline]] updated for neutrino branch
- [[information-measure-for-projection-regimes]] can use consistent Y matrices via `diagnostics/11_flavor_information_entropy.py`
