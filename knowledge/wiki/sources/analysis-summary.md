---
type: source
title: Analysis Summary (Repo)
tags: [flavor]
related:
  - proven-vs-conjecture-ledger
  - manuscript-key-results
  - neutrino-observables-gap
  - similar-fitted-scales-vs-transfer
sources:
  - raw/sources/analysis-summary.md
status: phenomenological
created: 2026-06-01
updated: 2026-06-02e
---

# Analysis Summary (Repo)

From `ANALYSIS_SUMMARY.md` and scaled diagnostics 22–25.

## Key numbers

### Legacy CSV archives (full-sector optimization, range-based survivors)

| Sector | Geometries | Survivors | Rate |
|--------|------------|-----------|------|
| Quark | 1000 | 0 | 0% |
| Charged lepton | 100 | 60 | **60%** |
| Neutrino | 480 | 216 | **45%** |

### Scaled diagnostics (train/holdout or strict PDG-relative)

| Sector | Script | N | Strict | Legacy (same solutions) |
|--------|--------|---|--------|---------------------------|
| Quark | diag 21 | 12 | 0% | — |
| Lepton | diag 22 | **100** | **1%** | **1%** (train m_μ+m_τ only; archived 60% used all 3 masses) |
| Neutrino | diag 23 | **100** (90 solved) | **78.9%** | **78.9%** |

- σ clusters ~2–5 per sector when fit independently ([[similar-fitted-scales-vs-transfer]])

## Cross-kernel + Pareto (diag 24–25)

- **Diag 24:** 30 paired geometries — clockwork wins lepton train on 8/30 vs Gaussian 22/30; holdout m_e poor for all kernels; generalized p=1.5 wins neutrino PMNS on 14/24.
- **Diag 25:** Weighted m_μ–m_e Pareto — holdout-only protocol fails (diag 22); including holdout weight can fit all masses on subset (degenerate 1-point front).

## Gap documented

Neutrino PMNS pipeline now in `observables.py` — [[neutrino-observables-gap]] partially closed.
