---
type: query
title: Information Measure for Projection Regimes
tags: [information, flavor]
related:
  - projection-regimes
  - quantum-information
  - preskill-qit-entropy
  - qed-qm-information
  - holographic-principle
  - neutrino-observables-gap
  - research-strategy
status: refuted
created: 2026-06-01
updated: 2026-06-02
---

# Information Measure for Projection Regimes

## Question

Is there a **computable** information quantity \(I(\text{sector})\) distinguishing envelope / phase / metric regimes?

## Implementation

`src/flavor_information.py` (von Neumann — **refuted** as regime mechanism):

- `yukawa_density_matrix(Y)` — \(\rho \propto Y Y^\dagger / \operatorname{Tr}(Y Y^\dagger)\)
- `von_neumann_entropy`, `effective_rank`, `off_diagonal_entropy`
- `compute_yukawa_information(Y, include_qed=False)` — entropy dict only

`src/qed_information.py` (Path A pivot — QFI / coherence / distinguishability):

- `quantum_fisher_trace`, `yukawa_qfi_mean_over_elements`, `yukawa_qfi_singular_values`
- `coherence_l1_norm`, `off_diagonal_to_diagonal_ratio`
- `distinguishability_from_uniform`, `compute_qed_yukawa_information(Y)`

Diagnostics:

| Script | Role |
|--------|------|
| `diagnostics/11_flavor_information_entropy.py` | Single-point sector comparison |
| `diagnostics/12_regime_entropy_correlation.py` | Full sweep + pre-registered falsifier (**REFUTED**) |
| `diagnostics/13_yukawa_information_inequality.py` | Mechanism inequality candidates (**REFUTED**) |
| `diagnostics/15_qed_fisher_yukawa.py` | QFI/coherence vs mixing; pooled falsifier |
| `diagnostics/16_decoherence_mixing_bound.py` | Decoherence vs CKM/PMNS magnitudes |

## Pre-registered falsifier ([[research-strategy]])

Across ≥30 samples: if **all** of \(S(\rho_Y)\), effective rank, off-diagonal entropy have \|Pearson r\| < 0.30 vs regime label (0=quark, 1=lepton, 2=neutrino), hypothesis **refuted**.

## Result (2026-06-01) — **REFUTED**

`diagnostics/12_regime_entropy_correlation.py`, **n = 10,080** (5 quark + 4 lepton + 3 neutrino geometries × parameter grid):

| Measure | r vs regime | p |
|---------|-------------|---|
| \(S(\rho_Y)\) | +0.046 | 3.9×10⁻⁶ |
| effective rank | +0.047 | 2.1×10⁻⁶ |
| off-diagonal entropy | +0.042 | 3.1×10⁻⁵ |

**Verdict:** regime labels do **not** correlate with information measures at pre-registered threshold. Small r with tiny p reflects large n, not meaningful effect size.

### Sector means (informational only, not mechanism)

| Sector | mean S | mean rank |
|--------|--------|-----------|
| charged lepton | 0.115 | 1.127 |
| quark (up/down) | ~0.146 | ~1.167 |
| neutrino | 0.158 | 1.183 |

Neutrino \(S\) vs \(g_{\text{env}}\): r = −0.18 (weak; does not rescue regime hypothesis).

### Mechanism inequalities (`diagnostics/13_*`)

No candidate holds **across sectors** with consistent sign:

- Best sector-local: off-diag entropy vs mixing (quark r = −0.38, lepton r = +0.28) — **sign flip**
- Pooled off-diag vs mixing: r ≈ 0.006

## QED pivot result (2026-06-02) — **REFUTED (pooled)**

`diagnostics/15_qed_fisher_yukawa.py`, **n = 10,080** (same geometry grid as diagnostic 12):

| Measure | r vs mixing (pooled) |
|---------|----------------------|
| QFI mean (elements) | +0.012 |
| QFI mean (singular values) | +0.017 |
| coherence l1 | +0.013 |
| off-diagonal ratio | +0.013 |
| distinguishability vs uniform | −0.010 |

**Verdict:** max \|r\| = 0.017 < 0.25 pre-registered threshold → **no QED-info mechanism** for this measure class (pooled). Sector-local \|r\| up to ~0.47 (coherence vs mixing) does **not** pass cross-sector pooled criterion — same failure mode as diagnostic 13 sign-flip.

`diagnostics/16_decoherence_mixing_bound.py`, **n = 2,592** (quark + neutrino):

- Max \|r\| (decoherence vs CKM/PMNS) = **0.241** < 0.25 → **refuted** at threshold
- `upper_bound_frac` = 1.0 (decoherence ≥ mix_sum always) — trivial bound, not a physical inequality

## Conclusion

Regime–entropy and post-hoc \(S(\rho_Y)\) labeling are **not supported** as mechanisms. QFI/coherence proxies on \(\rho_Y\) from kernel sweeps **do not** correlate with mixing at pre-registered pooled thresholds. Sector-local correlations are exploratory only until a cross-sector pre-registered test passes.

## Falsifiers (status)

- S constant across sectors despite parameter splits — **partially false** (small spread in means)
- S tracks regime labels — **refuted** (diagnostic 12)
- QFI/coherence vs mixing (pooled) — **refuted** (diagnostics 15, 16)
- S tracks optimization bounds, not regime labels — **supported** (mass loss correlation)

## Related

[[qm-to-information-what-is-measurable]], [[projection-regimes]], [[research-strategy]]
