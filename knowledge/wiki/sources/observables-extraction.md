---
type: source
title: Observables Extraction (Repo)
tags: [flavor]
related:
  - yukawa-observables-pipeline
  - interference-kernel
  - diagnostics-summary
  - neutrino-observables-gap
sources:
  - raw/sources/observables-extraction.md
authors: [Alexander Seto]
year: 2026
status: established
created: 2026-06-01
updated: 2026-06-01
---

# Observables Extraction (Repo)

Distilled from `src/observables.py`.

## Proven pipeline (quarks)

SVD → phase fix → CKM from \(U_u^\dagger U_d\); masses from singular values anchored to \(m_t, m_b\) (PDG 2024 in `QUARK_TARGETS`).

## Proven pipeline (neutrinos)

SVD of \(Y_\nu, Y_e\) → phase fix → \(U_{\mathrm{PMNS}} = U_e^\dagger U_\nu\) → \(\theta_{12}, \theta_{23}, \theta_{13}\). Tests in `tests/test_neutrino_observables.py`. See [[neutrino-observables-gap]] (resolved).

## Train/holdout methodology

Training: \(m_c, |V_{us}|, |V_{cb}|\). Holdout: light masses + \(|V_{ub}|\) — used only for generalization tests ([[diagnostics-summary]] minimality ladder).

## Implementation note

Phase-fixing helper may perturb CKM at ~0.5% level; documented in Gaussian diagnostic, secondary to structural quark failure.

## Related

Entropy measures: `src/flavor_information.py` ([[information-measure-for-projection-regimes]]).
