---
type: query
title: Neutrino Geometry Predictor for Joint Strict (N4)
tags: [flavor, neutrino, optimization]
related:
  - neutrino-loss-landscape-n1
  - future-work
  - diagnostics-summary
status: established
created: 2026-06-02
updated: 2026-06-02
---

# Neutrino Geometry Predictor for Joint Strict (N4)

## Question

Do **pre-fit** discrete geometry features \((L,N)\) predict joint strict success (PMNS + \(\Delta m^2\)) on the diag 28 pool?

## Pre-registered falsifier

\(\max(\text{5-fold CV AUC}, \text{best univariate AUC}) \le 0.55\) → success is opaque luck.

Pursue bar (exploratory): CV AUC \(\ge 0.65\).

## Method

`diagnostics/40_n4_geometry_strict_predictor.py` — seed **28028**, \(N=100\), 12 geometry features, logistic regression (no \(g_{\text{env}}\) or post-fit losses).

## Results (diag 40, full pool)

| Metric | Value |
|--------|-------|
| Solved | 79/100 (matches diag 28) |
| Strict rate | 24/100 (24%) — cf. diag 28 **27.8%** on solved (22/79) |
| **CV AUC (full model)** | **0.658** |
| Best univariate AUC | **0.572** (`overlap_count`) |
| CV AUC (solved only) | **0.602** |

Falsifier **rejected** (AUC > 0.55). Pursue bar **met** marginally (CV \(\approx 0.66\)).

Top univariate hints: `overlap_count`, `mean_L` — modest; multivariate CV beats singles.

## Interpretation

Geometry carries **weak–moderate** signal for joint strict success, not a sharp classifier. Combine with **N1** (shallower ν loss basins): sector success is partly landscape + partly which \((L,N)\) triples are drawn.

**Not** a mechanism claim; do not use for quark extrapolation.

## Follow-ups

- Holdout geometry **blocks** (train on 70 geoms, test on 30 disjoint draws).
- **N2** Haar PMNS null.
