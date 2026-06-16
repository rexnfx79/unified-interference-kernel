---
type: query
title: N8 — Neutrino CP Weight Pareto Scan
tags: [flavor, neutrino, cp]
related:
  - neutrino-cp-weighted-objective-n7
  - neutrino-cp-invariant-n6
  - future-work
status: open
created: 2026-06-16
updated: 2026-06-16
---

# N8 — Neutrino CP Weight Pareto Scan

**Question:** Can a small, fixed CP-weight grid find a signed-\(J_{\mathrm{PMNS}}\) objective that clears the no-degradation strict bar after N7's near miss?

**Diagnostic:** `diagnostics/49_neutrino_cp_weight_pareto_scan.py`

**Fixed grid:** \(w_{\mathrm{CP}}\in\{0,0.05,0.10,0.15,0.20,0.25,0.50,1.00\}\)

**Pass bars (full N=100):**

1. Joint strict \(\geq 22/100\) attempted.
2. Median signed-\(J\) relative error < 0.50.
3. CP sign-match \(\geq 0.60\).
4. No adaptive weights beyond the fixed grid.

## Smoke result (N=5)

Run:

```bash
PYTHONPATH=src python diagnostics/49_neutrino_cp_weight_pareto_scan.py --smoke
```

| \(w_{\mathrm{CP}}\) | Joint strict | signed-\(J\) rel err | sign-match | mass med | PMNS med |
|--------------------|--------------|----------------------|------------|----------|----------|
| 0 | 1/5 | 1.3506 | 0.250 | 0.001928 | 0.025387 |
| 0.05 | 2/5 | 0.4308 | 1.000 | 0.003458 | 0.031212 |
| 0.10 | 2/5 | 0.0759 | 1.000 | 0.001400 | 0.030997 |
| 0.15 | 2/5 | 0.0574 | 1.000 | 0.002018 | 0.031049 |
| 0.20 | 2/5 | 0.0208 | 1.000 | 0.001630 | 0.031328 |
| 0.25 | 2/5 | 0.0289 | 1.000 | 0.001525 | 0.031104 |
| 0.50 | 2/5 | 0.0344 | 1.000 | 0.004417 | 0.033971 |
| 1.00 | 2/5 | 0.0097 | 1.000 | 0.001095 | 0.031326 |

**Smoke verdict:** fixed CP weights \(\geq 0.05\) all restore CP on the probe set. \(w=0.10\) is the most promising next full run because it has strong CP, good mass/PMNS medians, and lower CP pressure than the \(w=0.25\) near miss.

## Next full run

Run focused full N=100:

```bash
PYTHONPATH=src python diagnostics/48_neutrino_cp_weighted_objective.py --cp-weight 0.1
```

Expected output: `diagnostics/results/48_neutrino_cp_weighted_objective_w0p1.txt`

## Related

[[neutrino-cp-weighted-objective-n7]], [[future-work]]
