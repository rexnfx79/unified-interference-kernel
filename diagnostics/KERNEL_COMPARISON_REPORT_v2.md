# Alternative Kernel Comparison Report (Revised)

## Executive Summary

The clockwork envelope `q^{-|d|}` modifies the hierarchy mechanism compared to Gaussian and admits solutions that simultaneously fit **m_c and |V_us|** within a few percent. However, the full quark sector (light masses, full CKM matrix) remains unresolved.

**Key Finding:** Clockwork improves the m_c vs CKM tradeoff but does NOT solve the complete flavor puzzle.

---

## Mass Normalization Scheme

The code uses a **top-bottom anchoring** scheme:

1. Yukawa matrices Y_u, Y_d are computed from the kernel
2. SVD decomposition: Y = U @ diag(S) @ V†
3. Singular values S = [S_0, S_1, S_2] ordered descending
4. **Anchoring:**
   - Up-type: scale_u = m_t / S_0, where m_t = 172.5 GeV (PDG pole mass)
   - Down-type: scale_d = m_b / S_0, where m_b = 4.18 GeV (PDG MS-bar)
5. **Derived masses:**
   - m_c = m_t × (S_1/S_0)
   - m_u = m_t × (S_2/S_0)
   - m_s = m_b × (S_1/S_0)_d
   - m_d = m_b × (S_2/S_0)_d

**Note:** This scheme is approximate. The "error" percentages refer to SVD ratio matching, not properly RG-evolved masses.

---

## Full Quark Sector Results

### Claimed "Best" Clockwork Solution

| Observable | Predicted | Target | Error | Status |
|------------|-----------|--------|-------|--------|
| m_t | 172.5 GeV | 172.5 GeV | 0% | ANCHOR |
| m_c | 1.270 GeV | 1.27 GeV | 0.0% | OK |
| m_u | 0.626 GeV | 0.00216 GeV | **28,872%** | FAIL |
| m_b | 4.18 GeV | 4.18 GeV | 0% | ANCHOR |
| m_s | 0.0018 GeV | 0.093 GeV | **98%** | FAIL |
| m_d | ~0 GeV | 0.00467 GeV | **100%** | FAIL |

### CKM Matrix Comparison

| Element | Predicted | PDG | Error | Status |
|---------|-----------|-----|-------|--------|
| |V_ud| | 0.974 | 0.974 | 0.0% | OK |
| |V_us| | 0.225 | 0.225 | 0.1% | OK |
| |V_ub| | 0.020 | 0.0038 | **432%** | FAIL |
| |V_cd| | 0.224 | 0.225 | 0.3% | OK |
| |V_cs| | 0.974 | 0.973 | 0.1% | OK |
| |V_cb| | 0.030 | 0.042 | **28%** | MARGINAL |
| |V_td| | 0.027 | 0.0086 | **209%** | FAIL |
| |V_ts| | 0.025 | 0.041 | **40%** | FAIL |
| |V_tb| | 0.999 | 0.999 | 0.0% | OK |

### Jarlskog Invariant

- Predicted: J = -1.29 × 10⁻⁵
- PDG: J = 3.08 × 10⁻⁵
- Error: **142%** (wrong sign!)

---

## What the Clockwork Kernel Actually Achieves

### Successes (Validated)

1. **m_c accuracy:** SVD ratio S_1/S_0 ≈ 135.8 matches m_t/m_c target
2. **|V_us| accuracy:** Within 0.1% of PDG value
3. **|V_cs|, |V_ud| accuracy:** Within 0.3% of PDG values
4. **Pareto improvement:** Best combined (mc_err + ckm_err) is 0.77 for Clockwork vs 1.84 for Gaussian

### Failures (Must Be Stated)

1. **Light quark masses completely wrong:**
   - m_u off by ~29,000%
   - m_d essentially zero
   - m_s off by ~98%

2. **Third-generation CKM elements wrong:**
   - |V_ub| off by 432%
   - |V_td| off by 209%
   - |V_ts| off by 40%

3. **CP violation (Jarlskog) wrong sign and magnitude**

4. **Solution is FRAGILE:**
   - 1% perturbation in k causes 54% mc error
   - 1% perturbation in eta causes 42% mc error

---

## Robustness Analysis

### Parameter Sensitivity (±1% perturbation)

| Parameter | mc Error After Perturbation |
|-----------|----------------------------|
| q | 1.0% (stable) |
| k | **54%** (fragile) |
| alpha | 3.5% (moderate) |
| eta | **42%** (fragile) |
| eps_u | 0.3% (stable) |
| eps_d | 0.0% (stable) |

**Conclusion:** The solution is sensitive to k and eta, suggesting fine-tuning.

### Restricted eps Test

With eps restricted to [0.3, 0.7] (avoiding phase cancellation regime):
- Still achieves mc = 1.27 GeV (0% error)
- Still achieves Vus = 0.225 (0% error)
- Uses eps_u = 0.43, eps_d = 0.63

**This suggests the mc + Vus success is NOT purely a phase-cancellation artifact.**

---

## Pareto Frontier Comparison

| Metric | Gaussian | Clockwork |
|--------|----------|-----------|
| Min mc_err | 0.65 | **0.03** |
| Min ckm_err | 0.24 | **0.21** |
| Best combined | 1.84 | **0.77** |

**Clockwork shifts the Pareto frontier favorably for the mc + Vus subset.**

---

## Kernel Naming Correction

The "Power-Law (Froggatt-Nielsen)" kernel has been renamed to **"Geometric Exponential (Cabibbo-base)"** because:
- It uses continuous distances, not discrete charges
- ε^{|d/λ|} = exp(-|d| ln(1/ε)/λ) is a geometric exponential
- True Froggatt-Nielsen requires integer charge assignments and selection rules

---

## Defensible Statement

> "The clockwork envelope q^{-|d|} modifies the hierarchy mechanism compared to Gaussian and admits solutions that simultaneously fit the charm quark mass ratio (m_c/m_t) and the Cabibbo angle (|V_us|) within a few percent, representing an improvement over the structural tradeoff exhibited by Gaussian kernels.
>
> However, this success is limited to a subset of observables. Light quark masses (m_u, m_d, m_s), third-generation CKM elements (|V_ub|, |V_td|, |V_ts|), and CP violation (Jarlskog invariant) remain poorly reproduced, indicating that additional physics beyond simple geometric interference is needed for a complete quark sector description.
>
> The solution shows sensitivity to certain parameters (k, eta), though the core mc + Vus success persists even with restricted interference strength (eps ∈ [0.3, 0.7])."

---

## Recommendations for Manuscript

1. **Do NOT claim** "reproduces the quark sector" - only a subset works
2. **DO claim** improvement in mc vs Vus tradeoff compared to Gaussian
3. **State clearly** that light masses and full CKM remain open problems
4. **Include** the mass normalization scheme explicitly
5. **Show** the full CKM matrix, not just three elements
6. **Acknowledge** parameter sensitivity in k and eta

---

## Files

- `src/alternative_kernels.py` - Kernel implementations (FN label corrected)
- `diagnostics/07_rigorous_validation.py` - Full validation script
- `diagnostics/results/07_rigorous_validation.txt` - Detailed results
