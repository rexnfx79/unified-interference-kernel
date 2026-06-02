---
type: concept
title: Generalized Envelope Kernel
tags: [flavor]
related:
  - interference-kernel
  - pareto-envelope-comparison
  - repo-scientific-findings
  - diagnostics-summary
island: true
bridge_tags: [flavor]
approach: effective
plausibility: watch
status: phenomenological
created: 2026-06-01
updated: 2026-06-01
---

# Generalized Envelope Kernel

## Definition

`src/kernel_generalized.py`:

\[
Y_{ij} = \exp\left(-\frac{(|d|/\sigma)^p}{p}\right) \cdot [1 + \varepsilon e^{i\Phi}]
\]

- \(p=2\): Gaussian (original)
- \(p=1\): exponential; \(p>2\): super-Gaussian

## Proven

- 16/16 unit tests; \(p=2\) regression matches `kernel.py`

## Pareto envelope test ([[pareto-envelope-comparison]])

| p | Knee? | mc at knee |
|---|-------|------------|
| 1 | No | — |
| 2 | No | — |
| 3 | Yes | ~5.4 GeV (target 1.27) |

**Verdict:** envelope exponent alone **does not** solve quark CKM–\(m_c\) trade-off.

## Refuted

- Super-Gaussian as complete fix for quarks
- Parameter universality ([[repo-scientific-findings]])

## Open

Larger scan (more geometries/weights) — current ingest is exploratory.
