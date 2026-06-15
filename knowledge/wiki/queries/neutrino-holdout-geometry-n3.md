---
type: query
title: Neutrino Holdout Geometry Predictor (N3)
tags: [flavor, neutrino, optimization]
related:
  - neutrino-geometry-predictor-n4
  - neutrino-loss-landscape-n1
  - future-work
  - diagnostics-summary
status: refuted
created: 2026-06-15
updated: 2026-06-15
---

# Neutrino Holdout Geometry Predictor (N3)

## Question

Does N4's geometry→joint-strict signal **generalize** to disjoint geometry draws (holdout block)?

## Pre-registered falsifier

Holdout test AUC \(\le 0.55\) on unseen \((L,N)\) triples (different seed, no key overlap with train).

Pursue bar: holdout test AUC \(\ge 0.66\) (beat N4 in-sample CV **0.658**).

## Method

`diagnostics/45_n3_holdout_joint_strict_predictor.py`

- **Train:** seed 28028, N=70 — fit logistic on 14 geometry features
- **Test:** seed 28029, N=50 — disjoint keys; frozen train coefficients
- Labels: joint strict (PMNS + \(\Delta m^2\)); secondary PMNS-only

## Results (diag 45)

| Metric | Value |
|--------|-------|
| Train joint strict rate | 21.4% (15/70) |
| Test joint strict rate | 18.0% (9/50) |
| Train 5-fold CV AUC | **0.572** |
| Train in-sample AUC | 0.852 (overfit) |
| **Holdout test AUC** | **0.529** |
| Holdout PMNS-only AUC | 0.483 |
| Beat N4 CV (0.658) | **No** |

Falsifier **not rejected** (test AUC \(\le 0.55\) — at chance).

## Verdict

**N3 refuted** — geometry features do **not** predict joint strict on unseen triples. N4's CV AUC **0.658** does not survive holdout; treat N4 as **in-sample weak signal only**, not a generalizable geometry classifier.

## Implication

- Do not fund geometry-based geometry pre-screening for joint strict survivors.
- Neutrino success remains **landscape + luck** (N1), with structured PMNS post-fit (N2).
- P-series (chiral projection) and geometry prediction both closed for flavor mechanism.

## Related

[[neutrino-geometry-predictor-n4]], `diagnostics/results/45_n3_holdout_joint_strict_predictor.txt`
