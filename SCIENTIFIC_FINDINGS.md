# Scientific Findings: Generalized Kernel Analysis

> **Disclaimer (2026-06):** Headline survivor rates and claim status are maintained in [`knowledge/wiki/synthesis/manuscript-ledger-alignment.md`](knowledge/wiki/synthesis/manuscript-ledger-alignment.md) and [`knowledge/wiki/synthesis/survivor-protocol-preregistered.md`](knowledge/wiki/synthesis/survivor-protocol-preregistered.md). This document summarizes transfer-test refutation of parameter universality; it does **not** supersede the wiki ledger for sector survivor numbers. Reproduce frozen diagnostics: `./scripts/reproduce_phenomenology_tranche.sh`.

### Phenomenology tranche (strict protocol, 100 geometries unless noted)

| Sector | Headline rate | Diagnostic |
|--------|---------------|------------|
| Quark | **0%** strict (0/5759 exhaustive) | 21, 30 |
| Lepton | **1%** strict train/holdout | 22 |
| Neutrino | **27.8%** joint strict (PMNS + Δm²) | 28 |
| Neutrino | **78.9%** PMNS-only strict (sub-result) | 23 |

## Executive Summary

**Key Finding**: The "universal kernel" claim is NOT supported by rigorous testing.

The true transfer test shows:
- Quark-fitted parameters do NOT transfer to leptons
- Multiple parameters must change across sectors
- This is a **parameterization**, not a **universal theory**

---

## Test Methodology

### 1. Generalized Gaussian Kernel

We implemented a generalized envelope:

```
Y_ij = exp(-(|d|/σ)^p / p) × [1 + ε exp(iΦ)]
```

Where:
- p = 1: Exponential decay
- p = 2: Gaussian (original)
- p > 2: Super-Gaussian

This allows testing whether findings are robust to envelope choice.

### 2. True Transfer Test

The rigorous universality test:

1. **Fit quarks** → get (σ*, k*, α*, η*)
2. **FREEZE** these parameters
3. **Apply to leptons** with ONLY ε_e free
4. **Measure** fit quality

**Results**:
| Test | Loss | Interpretation |
|------|------|----------------|
| Frozen (true universality) | 797.5 | Very poor fit |
| Free k only | 791.1 | Marginal improvement (0.8%) |
| Free all (independent) | 779.0 | Still poor (2.3% improvement) |

**Conclusion**: Quark parameters do NOT transfer to leptons.

---

## What This Means

### What the Data Actually Shows

1. **Same functional form works** across sectors
2. **Different parameters required** for each sector
3. **No evidence of parameter universality**

### What Can Be Claimed

✅ "A Gaussian × interference kernel parameterizes flavor structure"
✅ "Different sectors require different parameter values"
✅ "We characterize which parameters must change"

### What Cannot Be Claimed

❌ "Universal kernel derives all fermion masses"
❌ "Parameters transfer across sectors"
❌ "Three regimes emerge from one kernel"

---

## Honest Reframing

### Old Claim (Overclaiming)
> "A universal interference kernel organizes Yukawa couplings across all fermion sectors through three projection regimes."

### Honest Claim
> "A Gaussian × interference functional form can parameterize Yukawa matrices across fermion sectors. Each sector requires sector-specific parameter values. Pre-registered diagnostics find 0% quark strict survivors, 1% lepton strict (holdout m_e failure), and 27.8% neutrino joint strict (78.9% PMNS-only). This is a flexible parameterization, not a predictive universal theory."

---

## Code Quality

### Tests Implemented
- 16 unit tests for generalized kernel (all passing)
- Property-based tests (monotonicity, symmetry, positivity)
- Regression tests (p=2 matches Gaussian)
- Numerical stability tests

### Scripts Created
1. `src/kernel_generalized.py` - Generalized kernel implementation
2. `scripts/02_pareto_envelope_comparison.py` - Envelope robustness test
3. `scripts/03_true_transfer_test.py` - True universality test

### QA Methods Applied
- Input validation with proper error messages
- Edge case testing
- Numerical stability verification
- Cross-validation against original kernel

---

## Recommendations

### For the Manuscript

1. **Remove "universal" from title** - Use "unified" or "common form"
2. **Add honest limitations section** - Acknowledge parameter non-universality
3. **Reframe "regimes"** - As "parameter modifications required per sector"
4. **Report transfer test results** - Show what doesn't transfer

### For Future Work

1. **Seek predictive relations** - Can k_e be derived from k_quark?
2. **Test envelope robustness** - Does Pareto knee persist for all p?
3. **Quantify complexity** - Parameters vs. observables analysis

---

## Files Created

```
unified-interference-kernel/
├── src/
│   └── kernel_generalized.py      # Generalized kernel (tested)
├── scripts/
│   ├── 02_pareto_envelope_comparison.py  # Envelope robustness
│   └── 03_true_transfer_test.py          # True universality test
├── tests/
│   └── test_kernel_generalized.py        # 16 tests (all passing)
├── data/
│   ├── pareto_envelope_comparison.csv
│   └── transfer_test_results.csv
└── SCIENTIFIC_FINDINGS.md          # This document
```

---

## Conclusion

The rigorous testing reveals that the "universal kernel" claim is not supported by evidence. The honest characterization is:

**"A flexible parameterization that can fit each sector independently, not a predictive universal theory."**

This is still valuable as a structural analysis tool, but the claims must be scaled back to match the evidence.

---

*Generated: 2026-01-20*
*Tests: 16/16 passing*
*Transfer test: NO UNIVERSALITY detected*
