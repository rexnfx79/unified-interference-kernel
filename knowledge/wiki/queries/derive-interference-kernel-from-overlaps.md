---
type: query
title: Derive Interference Kernel from Overlaps
tags: [flavor, spectral, qm]
related:
  - split-fermion-overlaps
  - split-fermion-overlap-test
  - split-fermion-localization
  - interference-kernel
  - hilbert-spaces-qm
status: refuted
created: 2026-06-01
updated: 2026-06-02m
---

# Derive Interference Kernel from Overlaps

## Status: **refuted** for mechanism (diag 33)

Split-fermion → kernel does **not** predict kernel parameters from geometry. Magnitude matching remains a **width fit** per geometry. See `diagnostics/33_tier3_theory_bridges.py`.

## Question

Can the repo's Gaussian × interference ansatz be **derived** from localized 1D fermion wavefunctions?

## First test result ([[split-fermion-overlap-test]]) — **historical / partial**

Extended test (`diagnostics/10_split_fermion_overlap_derivation.py`): 4 geometries × (Yu, Yd).

| Metric | Result |
|--------|--------|
| \|Y_overlap\| vs \|Y_kernel\| correlation | **>0.99** most cases; csv_compact Yu **0.967** |
| Phase error | **0 rad** all cases |
| Width relation | **w ≈ 0.69 σ** for spread geometries; w/σ **unstable** if compact U positions included |

**Verdict:** magnitude match is post-hoc fit quality, not mechanism. w/σ instability and no sector-parameter prediction → **deprioritize**.

## Diagnostic 33 (N=50, seed 33033)

| Falsifier | Threshold | Fixed-params result |
|-----------|-----------|---------------------|
| \(w/\sigma\) stable | rel spread < 0.15 | **pass** (0.067) |
| Geometry predicts \(w/\sigma\) | linear R² ≥ 0.5 | **fail** (0.045) |
| All magnitude correlations | r ≥ 0.99 | **fail** (min 0.985) |

**Verdict:** envelope post-hoc; do not claim derivation.

## Falsifiers (historical diag 10 + diag 33)

- w/σ ratio stable at fixed \(\sigma\) — **pass** at N=50 (diag 33); was unstable in early 4-geometry compact cases
- Predict \(\sigma,k,\eta\) from geometry without optimization — **failed** (R²≈0.05)

## Related

[[split-fermion-overlaps]], [[yukawa-observables-pipeline]], [[research-strategy]]
