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
status: open
created: 2026-06-01
updated: 2026-06-01
status: deprioritized
---

# Derive Interference Kernel from Overlaps

## Status: **deprioritized** (user strategic decision)

Split-fermion → kernel is **off the pursue list** per [[research-strategy]]. Numeric results below are **historical**; primary path is QED→information (Path A).

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

## Falsifiers (met)

- w/σ ratio not stable across geometries — **failed**
- Cannot predict sector parameter splits without optimization — **open failure**

## Related

[[split-fermion-overlaps]], [[yukawa-observables-pipeline]], [[research-strategy]]
