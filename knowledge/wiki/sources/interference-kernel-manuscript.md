---
type: source
title: Interference Kernel Manuscript
tags: [flavor]
related:
  - interference-kernel
  - projection-regimes
  - manuscript-ledger-alignment
  - phenomenology-methodology-export
  - survivor-protocol-preregistered
sources:
  - raw/sources/manuscript-summary.md
authors: [Alexander Seto]
year: 2026
status: established
created: 2026-06-01
updated: 2026-06-15
---

# Interference Kernel Manuscript

Summary of `manuscript.tex` — phenomenological flavor organization paper. **Canonical rates:** [[manuscript-ledger-alignment]], [[survivor-protocol-preregistered]].

## Model

\[
Y_{ij} = \exp(-d_{ij}^2/2\sigma^2) \cdot [1 + \varepsilon \exp(i\Phi_{ij})]
\]

Discrete fermion coordinates in internal flavor space (split-fermion style).

## Three projection labels (organizational, not validated mechanism)

1. **Quarks** — envelope-dominated; CKM–\(m_c\) Pareto trade-off; **0% strict**
2. **Charged leptons** — phase-sensitive; **1% strict** (holdout \(m_e\))
3. **Neutrinos** — metric-dominated; **22% joint strict** / **71% PMNS-only** (different objectives)

## Headline results (strict protocol, N=100)

| Sector | Strict rate | Diagnostic |
|--------|-------------|------------|
| Quark | 0% (0/5759 exhaustive) | 21, 30 |
| Lepton | 1% | 22 |
| ν joint | 22/100 attempted | 28 |
| ν PMNS-only | 71/100 attempted | 23 |

Legacy full-objective rates (~60% lepton, ~45% ν) are historical only.

## Build & bundle

- `BUILD_MANUSCRIPT.md` — local `pdflatex`
- `scripts/bundle_submission_artifacts.sh` — submission package
- [[phenomenology-methodology-export]] — exportable protocol

## Related

[[manuscript-key-results]], [[diagnostics-summary]]
