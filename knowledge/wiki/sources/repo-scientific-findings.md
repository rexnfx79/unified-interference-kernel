---
type: source
title: Scientific Findings (Repo Artifact)
tags: [flavor]
related:
  - interference-kernel
  - projection-regimes
  - repo-scientific-findings
sources:
  - raw/sources/scientific-findings.md
authors: [Alexander Seto]
year: 2026
status: established
created: 2026-06-01
updated: 2026-06-01
---

# Scientific Findings (Repo Artifact)

Summary of rigorous tests in `SCIENTIFIC_FINDINGS.md` (parent repo).

## Headline Result

**The "universal kernel" claim is not supported.** Quark-fitted parameters do not transfer to leptons.

## Transfer Test

| Configuration | Loss |
|---------------|------|
| Frozen quark params, only \(\varepsilon_e\) free | 797.5 |
| Free \(k\) only | 791.1 |
| All parameters free | 779.0 |

Improvement from universality to full freedom: ~2.3% — parameters are **sector-specific**.

## What Can Be Claimed

- Gaussian × interference form **parameterizes** flavor data
- Each sector needs its own parameter set
- Generalized envelope \(p\) (super-Gaussian) tested in `kernel_generalized.py`

## Implications for Wiki

- [[interference-kernel]] remains valid as **phenomenology**
- [[spectral-interpretation-of-flavor]] must explain parameter non-universality via [[projection-regimes]]
- [[information-creates-reality]] thesis is **not** supported by this data alone — needs deeper layer

## Tests

16/16 unit tests passing (`tests/test_kernel_generalized.py`).
