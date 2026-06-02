---
type: concept
title: Yukawa Observables Pipeline
tags: [flavor]
related:
  - interference-kernel
  - observables-extraction
  - kernel-implementation
  - split-fermion-overlaps
  - projection-regimes
  - neutrino-observables-gap
island: true
bridge_tags: [flavor, spectral]
approach: effective
plausibility: established
status: established
created: 2026-06-01
updated: 2026-06-01
---

# Yukawa Observables Pipeline

End-to-end readout chain in the parent repo (quark, neutrino, charged lepton in `observables.py`).

## Proven (code + tests)

```text
kernel params + geometry → Y_u, Y_d → SVD → phase fix → CKM + anchored masses → losses
```

| Step | Location | Output |
|------|----------|--------|
| Kernel | `src/kernel.py` | Complex \(3\times3\) \(Y_u, Y_d\) |
| SVD + CKM | `src/observables.py` | \(V_{us}, V_{cb}, V_{ub}\), \(m_{c,u,d,s}\) |
| SVD + PMNS | `src/observables.py` | \(\theta_{12}, \theta_{23}, \theta_{13}\) from \(U_e^\dagger U_\nu\) |
| Loss | same | CKM, mass, PMNS, train/holdout, penalized |

Anchoring: \(m_t, m_b\) fix absolute scale; lighter masses from subleading singular values. PMNS angles use PDG 2024 targets in `NEUTRINO_TARGETS`.

## Phenomenological (fits, not predictions)

- Integer geometries \((Q, U, D)\) optimized per sector.
- Sector-specific \((\sigma, k, \alpha, \eta, \varepsilon)\) — **no** cross-sector transfer ([[repo-scientific-findings]]).

## Conjecture

Same pipeline with **derived** kernel elements from [[split-fermion-overlaps]] would reduce free parameters and predict sector splits.

## Sector coverage

- **Quarks:** fully in `observables.py` — proven
- **Neutrinos / PMNS:** `compute_neutrino_observables` + tests — proven ([[neutrino-observables-gap]] resolved)
- **Charged leptons:** `compute_lepton_observables`, `compute_lepton_loss`, `LEPTON_TARGETS` in `observables.py`; `tests/test_lepton_observables.py`

## Diagnostics coupling

Gaussian kernel fails holdout despite good train targets — pipeline is sound; **ansatz** is wrong for full quarks ([[diagnostics-summary]]).
