---
type: query
title: Neutrino vs Quark Joint Loss Landscape (N1)
tags: [flavor, neutrino, optimization]
related:
  - future-work
  - diagnostics-summary
  - survivor-protocol-preregistered
status: established
created: 2026-06-02
updated: 2026-06-02
---

# Neutrino vs Quark Joint Loss Landscape (N1)

## Question

On matched geometry samples, is the **joint** optimization landscape (loss level, curvature, ruggedness) statistically similar for quark and neutrino sectors?

## Pre-registered falsifier

Fewer than **2** of five metrics show Mann–Whitney \(p < 0.05\) between sectors → landscapes **indistinguishable**.

## Method

`diagnostics/39_joint_loss_landscape_cartography.py` — \(N=50\) geometries per sector, Gaussian kernel, DE minima, finite-difference Hessian + ruggedness probes.

| Metric | Quark (joint diag 27) | Neutrino (joint diag 28) |
|--------|----------------------|---------------------------|
| Median best loss | **28.4** | **0.39** |
| Median ruggedness | **1576** | **21.4** |
| Median grad norm | **766** | **4.5** |
| Strict rate | **0%** (0/50) | **34.2%** (13/38 solved) |
| log10(condition) | 4.35 | 4.15 (not significant) |

Significant separations (\(p < 0.05\)): **best_loss**, **frac_negative_hessian**, **ruggedness**, **grad_norm** (4/5).

## Verdict (diag 39)

**N1 positive** — sectors are **not** indistinguishable; neutrino joint objective sits in shallower, smoother basins with far lower loss at minima. Does **not** prove a fundamental mechanism; explains why strict joint success is sector-asymmetric under the same kernel **form**.

## Follow-ups

- **N4** — done (diag 40): [[neutrino-geometry-predictor-n4]]
- **N2** — Haar PMNS null vs post-fit angles.
- Do **not** rescale quark loss to match ν units (would confound comparison).
