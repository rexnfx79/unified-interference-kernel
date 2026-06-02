---
type: source
title: Kernel Implementation (Repo)
tags: [flavor]
related:
  - interference-kernel
  - yukawa-observables-pipeline
  - kernel-implementation
sources:
  - raw/sources/kernel-implementation.md
authors: [Alexander Seto]
year: 2026
status: established
created: 2026-06-01
updated: 2026-06-01
---

# Kernel Implementation (Repo)

Distilled from `src/kernel.py`.

## Proven

- Gaussian × interference formula implemented exactly as documented.
- \(3\times3\) Yukawa construction from integer coordinate triples; quark sector returns \(Y_u, Y_d\) with shared \((\sigma,k,\alpha,\eta)\) and distinct \(\varepsilon_{u,d}\).
- Generalized envelope in `kernel_generalized.py` with regression test \(p=2\) ≡ original.

## Caveats

- Third left-handed coordinate pinned to zero — reduces geometric freedom.
- No continuous overlap integral; coordinates are discrete labels.
