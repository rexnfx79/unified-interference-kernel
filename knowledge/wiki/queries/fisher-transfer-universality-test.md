---
type: query
title: Fisher Transfer Universality Test
tags: [information, flavor]
related:
  - similar-fitted-scales-vs-transfer
  - qm-to-information-what-is-measurable
  - repo-scientific-findings
  - research-strategy
status: refuted
created: 2026-06-02
updated: 2026-06-02
---

# Fisher Transfer Universality Test

## Question

If QED/experimental Fisher geometry is the **mechanism** behind sector parameter splits, can Fisher information at the **quark fit minimum** predict which lepton parameters must deviate — and by how much — when quark universal parameters are frozen?

## Pre-registered protocol (`diagnostics/19_fisher_transfer_test.py`)

1. Fit quarks → \(\theta^*_{\mathrm{quark}}\), compute experimental Fisher \(F_{\mathrm{quark}}\) (PDG-weighted Jacobian).
2. **Freeze** \((\sigma, k, \alpha, \eta)\) from quark fit; optimize \(\varepsilon_e\) only.
3. Measure lepton loss at frozen transfer point.
4. Compare Fisher alignment between \(F_{\mathrm{quark}}\) and \(F_{\mathrm{lepton}}\) at transfer.
5. Independent free lepton fit → actual \(\Delta\theta\) vs quark Cramér–Rao bounds on shared parameters.

## Falsifiers (registered before run)

| ID | Condition | Interpretation |
|----|-----------|----------------|
| **A** | All \(\|\Delta\theta_i\| \le z\sqrt{\mathrm{CR}_i}\) on \((\sigma,k,\alpha,\eta)\) from free lepton fit **but** frozen transfer loss ≥ 797 | Fisher says no universal-parameter change needed; transfer still fails → **refuted** |
| **B** | Fisher principal-eigenvector alignment at transfer < 0.50 **and** frozen loss ≥ 797 | Fisher geometry does not carry across sectors at failure point → **refuted** |

## Result (2026-06-02)

See `diagnostics/results/19_fisher_transfer_test.txt`.

| Quantity | Value |
|----------|-------|
| Frozen lepton loss | **805.84** (≥ 797 threshold) |
| Free lepton loss | 775.19 |
| Fisher alignment at transfer | **0.413** (< 0.50) |
| Falsifier A (Δθ within CR + bad loss) | False — Δk ratio ≈ 410× CR bound |
| Falsifier B (low alignment + bad loss) | **True** |

**Verdict: REFUTED** — Fisher geometry at quark minimum does not predict successful lepton transfer; alignment below threshold at the failure point.

Experimental Fisher does **not** rescue parameter universality. Consistent with [[repo-scientific-findings]] and diagnostic 17 pooled alignment 0.46.

## Code

- `src/fisher_transfer.py` — fit + Fisher transfer pipeline
- `src/experimental_fisher.py` — PDG Fisher, CR bounds, alignment
- `tests/test_fisher_transfer.py`

## Implication

Path A QIT→flavor **mechanism** is **deprioritized** (see [[research-strategy]]). Kernel remains a **phenomenological** fit tool. Path D stays **watch only** — no promotion hook from Fisher transfer.
