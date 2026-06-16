---
type: query
title: N6 — Neutrino CP Invariant Audit
tags: [flavor, neutrino, cp, methodology]
related:
  - tangent-research-seeds
  - neutrino-pmns-cp-n5
  - future-work
  - survivor-protocol-preregistered
status: established
created: 2026-06-15
updated: 2026-06-15
---

# N6 — Neutrino CP Invariant Audit

**Question:** Was N5's large \(\delta_{\mathrm{PMNS}}\) error a real CP failure, or partly a phase-convention artifact?

**Diagnostic:** `diagnostics/47_neutrino_cp_invariant_audit.py`

**Scope:** Audit-only. No CP-targeted optimization.

## Protocol

### 47A — Convention audit

Construct a PDG PMNS matrix at target angles and \(\delta_{\mathrm{PMNS}}\), then randomly rephase rows and columns.

| Observable | Expected behavior |
|------------|-------------------|
| PMNS angles | invariant |
| signed \(J_{\mathrm{PMNS}}\) | invariant |
| raw `delta_PMNS = -arg(U_e3)` | **not** invariant without PDG gauge fixing |

### 47B — Diag-28 pool audit

Re-run the diag-28 joint objective and evaluate signed \(J_{\mathrm{PMNS}}\):

\[
J_{\mathrm{target}} =
c_{12}s_{12}c_{23}s_{23}c_{13}^2s_{13}\sin\delta_{\mathrm{PDG}}.
\]

Primary CP audit metrics:

- median signed-\(J\) relative error
- median \(|J|\)-magnitude relative error
- CP sign-match rate
- joint strict rate vs diag 28 baseline

## Full result (N=100)

Run:

```bash
PYTHONPATH=src python diagnostics/47_neutrino_cp_invariant_audit.py
```

Report: `diagnostics/results/47_neutrino_cp_invariant_audit.txt`

| Metric | Value |
|--------|-------|
| \(J_{\mathrm{target}}\) | \(-0.01145960\) |
| Max \(|J|\) rephase drift | \(3.1\times10^{-17}\) |
| Max angle rephase drift | \(2.2\times10^{-16}\) |
| Median raw-\(\delta\) rephase error | 1.52 rad |
| Solved | 79/100 |
| PMNS strict | 24/79 |
| Joint strict | 24/79 |
| Median signed \(J_{\mathrm{PMNS}}\) | \(-0.005736\) |
| Median \(|J_{\mathrm{PMNS}}|\) | 0.017404 |
| Median signed-\(J\) rel err | 1.071 |
| Median \(|J|\)-magnitude rel err | 0.844 |
| CP sign-match | 0.633 |
| Median raw \(\delta_{\mathrm{PMNS}}\) | 0.004 rad |
| Median raw-\(\delta\) circular error | 1.269 rad |

**Verdict:** raw \(\delta_{\mathrm{PMNS}}\) is convention-sensitive; signed \(J_{\mathrm{PMNS}}\) is the correct CP target. Under the existing diag-28 joint objective, both CP magnitude and signed invariant miss the 50% audit bar. N6 therefore confirms CP misalignment in an invariant form.

## Smoke result (N=5)

Run:

```bash
PYTHONPATH=src python diagnostics/47_neutrino_cp_invariant_audit.py --smoke
```

Result:

| Metric | Value |
|--------|-------|
| \(J_{\mathrm{target}}\) | \(-0.01145960\) |
| Max \(|J|\) rephase drift | \(3.1\times10^{-17}\) |
| Median raw-\(\delta\) rephase error | 1.52 rad |
| Solved | 4/5 |
| Joint strict | 1/4 |
| Median signed \(J_{\mathrm{PMNS}}\) | 0.004018 |
| Median signed-\(J\) rel err | 1.35 |
| Median \(|J|\) rel err | 0.84 |
| CP sign-match | 0.25 |

**Smoke verdict:** raw \(\delta_{\mathrm{PMNS}}\) is convention-sensitive; signed \(J_{\mathrm{PMNS}}\) is the correct next target. On N=5 smoke, both CP magnitude and sign miss the 50% audit bar.

## CP-extension bar

Any later CP-aware objective must satisfy:

1. Convention audit passes: angles and \(J\) invariant; raw \(\delta\) unstable.
2. Joint strict rate remains \(\geq 22/100\) attempted.
3. PMNS and mass medians do not worsen relative to diag 28 / 47.
4. Median signed-\(J\) relative error < 50%.
5. CP sign-match rate materially exceeds chance.

## Interpretation

N6 does **not** reopen flavor mechanism claims. It narrows Seed A:

> Optimize or extend against signed \(J_{\mathrm{PMNS}}\), not raw \(\delta_{\mathrm{PMNS}}\). If CP still fails at full N, the next physically meaningful extension is a neutrino-specific readout (e.g. constrained seesaw/Majorana structure), not another blind phase grid.

## Related

[[neutrino-pmns-cp-n5]], [[tangent-research-seeds]], [[future-work]]
