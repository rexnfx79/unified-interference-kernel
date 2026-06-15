---
type: query
title: Neutrino PMNS CP Descriptive (N5)
tags: [flavor, neutrino]
related:
  - neutrino-loss-landscape-n1
  - future-work
  - diagnostics-summary
status: established
created: 2026-06-15
updated: 2026-06-15
---

# Neutrino PMNS CP Descriptive (N5)

## Question

On the diag 28 pool (joint objective), what are post-optimization \(\delta_{\mathrm{PMNS}}\) and \(J_{\mathrm{PMNS}}\) vs PDG — without CP-targeted sweeps?

## Method

`diagnostics/46_n5_pmns_cp_descriptive_audit.py` — seed 28028, N=100, best seed per geometry.

## Results (diag 46)

| Metric | Value |
|--------|-------|
| Solved | 79/100 |
| PDG \(\delta_{\mathrm{PMNS}}\) | 3.49 rad |
| Median \(\delta_{\mathrm{PMNS}}\) | 0.004 rad |
| Median \(\|\delta - \delta_{\mathrm{PDG}}\|\) | 3.486 rad |
| Median \(\|J_{\mathrm{PMNS}}\|\) | 0.017 |

Descriptive only — not a mechanism claim.

## Related

`diagnostics/results/46_n5_pmns_cp_descriptive_audit.txt`
