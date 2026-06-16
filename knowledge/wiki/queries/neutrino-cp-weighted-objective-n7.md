---
type: query
title: N7 — Neutrino CP-Weighted Objective
tags: [flavor, neutrino, cp]
related:
  - neutrino-cp-invariant-n6
  - tangent-research-seeds
  - future-work
status: mixed
created: 2026-06-15
updated: 2026-06-15
---

# N7 — Neutrino CP-Weighted Objective

**Question:** Once N6 identifies signed \(J_{\mathrm{PMNS}}\) as the invariant CP target, can a single CP-aware objective recover CP without degrading mass+PMNS performance?

**Diagnostic:** `diagnostics/48_neutrino_cp_weighted_objective.py`

**Objective:**

\[
L = L_{\Delta m^2} + 5L_{\mathrm{PMNS}} + L_{\mathrm{CP}}(J_{\mathrm{PMNS}})
\]

where

\[
L_{\mathrm{CP}} =
\left(\frac{J_{\mathrm{PMNS}} - J_{\mathrm{target}}}{|J_{\mathrm{target}}|}\right)^2.
\]

No weight grid: fixed CP weight \(=1\).

## Smoke result (N=5)

Run:

```bash
PYTHONPATH=src python diagnostics/48_neutrino_cp_weighted_objective.py --smoke
```

| Metric | Value |
|--------|-------|
| Solved | 4/5 |
| PMNS strict | 2/4 |
| Joint strict | 2/4 |
| Median mass loss | 0.001095 |
| Median PMNS loss | 0.031326 |
| Median CP loss | 0.000098 |
| Median signed \(J_{\mathrm{PMNS}}\) | \(-0.011348\) |
| Median signed-\(J\) rel err | 0.0097 |
| Median \(|J|\) rel err | 0.0097 |
| CP sign-match | 1.000 |

**Smoke verdict:** strong positive signal. The fixed signed-\(J\) objective improves invariant CP vs N6 baseline (signed-\(J\) rel err **1.071 → 0.0097**) without obvious mass/PMNS degradation on the first 5 geometries.

## Full result (N=100)

Run:

```bash
PYTHONPATH=src python diagnostics/48_neutrino_cp_weighted_objective.py
```

Report: `diagnostics/results/48_neutrino_cp_weighted_objective.txt`

| Metric | Value |
|--------|-------|
| Solved | 79/100 |
| PMNS strict | 22/79 |
| Joint strict | 20/79 |
| Median mass loss | 0.053054 |
| Median PMNS loss | 0.054178 |
| Median CP loss | 0.001748 |
| Median signed \(J_{\mathrm{PMNS}}\) | \(-0.011459\) |
| Median \(|J_{\mathrm{PMNS}}|\) | 0.011459 |
| Median signed-\(J\) rel err | 0.0418 |
| Median \(|J|\) rel err | 0.0418 |
| CP sign-match | 0.975 |

**Full-N verdict:** **mixed / fails no-degradation bar.** The fixed signed-\(J\) objective decisively fixes invariant CP (signed-\(J\) rel err **1.071 → 0.0418**, sign-match **0.975**), but joint strict drops to **20/100 attempted**, below the pre-registered **≥22/100** no-degradation bar.

## Lower-weight follow-up (N=100, \(w_{\mathrm{CP}}=0.25\))

Run:

```bash
PYTHONPATH=src python diagnostics/48_neutrino_cp_weighted_objective.py --cp-weight 0.25
```

Report: `diagnostics/results/48_neutrino_cp_weighted_objective_w0p25.txt`

| Metric | Value |
|--------|-------|
| Solved | 79/100 |
| PMNS strict | 23/79 |
| Joint strict | 21/79 |
| Median mass loss | 0.054357 |
| Median PMNS loss | 0.051698 |
| Median CP loss | 0.012499 |
| Median signed \(J_{\mathrm{PMNS}}\) | \(-0.011409\) |
| Median signed-\(J\) rel err | 0.1118 |
| Median \(|J|\) rel err | 0.1118 |
| CP sign-match | 0.975 |

**Lower-weight verdict:** improves the strict count (**20 → 21**) while preserving a strong CP fit, but still misses the no-degradation bar by one geometry (**21/100 < 22/100**). This is a near miss, not a pass.

## Full-N bar

Pass bar:

1. Joint strict \(\geq 22/100\) attempted (diag 28 no-degradation bar).
2. Median signed-\(J\) relative error < 0.50.
3. Median PMNS loss and mass loss do not worsen vs N6/diag47 baselines.
4. CP sign-match rate materially exceeds chance.

## Interpretation

Seed A is not a clean pass, but N7 is scientifically useful: the existing neutrino parameter space can hit invariant CP once CP is included in the loss, yet doing so costs joint strict survivors. Lowering \(w_{\mathrm{CP}}\) reduces the cost but still misses the pre-registered no-degradation bar. This is a CP/mass+mixing trade-off, not a mechanism.

Next step should be a physically structured readout (e.g. constrained Majorana/seesaw) or an explicitly pre-registered Pareto analysis, rather than tuning CP weights.

## Related

[[neutrino-cp-invariant-n6]], [[tangent-research-seeds]], [[future-work]]
