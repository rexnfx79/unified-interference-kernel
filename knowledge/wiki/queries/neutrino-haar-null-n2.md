---
type: query
title: Neutrino Haar PMNS Null (N2)
tags: [flavor, neutrino]
related:
  - neutrino-loss-landscape-n1
  - neutrino-geometry-predictor-n4
  - future-work
status: established
created: 2026-06-02
updated: 2026-06-02
---

# Neutrino Haar PMNS Null (N2)

## Question

After kernel optimization, are PMNS angles **Haar-like** (generic unitary) or clustered toward PDG?

## Pre-registered falsifier

Haar-like: all three KS \(p > 0.05\) **and** post-fit not closer to PDG than Haar null.

## Results (diag 41, seed 28028, 79 solved)

| Angle | Post-fit median | Haar median | PDG | KS \(p\) |
|-------|-----------------|-------------|-----|----------|
| \(\theta_{12}\) | 0.488 | 0.795 | 0.590 | \(\ll 0.05\) |
| \(\theta_{23}\) | 0.756 | 0.781 | 0.785 | \(\ll 0.05\) |
| \(\theta_{13}\) | 0.150 | 0.568 | 0.149 | \(\ll 0.05\) |

PDG relative distance: post-fit median **0.24** vs Haar **2.93** (MW \(p \ll 0.01\)). Strict subset median **0.008**.

## Verdict

**N2 positive** — optimization selects **non-generic** PMNS, strongly peaked near PDG (especially \(\theta_{13}\)). This is **not** anarchy; kernel + objective **do** structure mixing angles. Not a fundamental mechanism proof (optimizer targets PDG).

## With N1/N4

- **N1:** shallower ν loss basins than quark.
- **N4:** weak geometry predictors for strict success.
- **N2:** post-fit angles are **target-driven**, not Haar — explains partial joint success without universal-kernel claims.
