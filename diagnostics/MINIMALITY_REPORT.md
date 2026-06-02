# Minimality Ladder Analysis Report

## Executive Summary

A controlled sequence of model relaxations was tested to determine whether the shared-Q constraint is the primary bottleneck limiting quark sector fits. The results show that **adding parameters does NOT improve holdout performance** - in fact, all relaxations degrade generalization while training loss also worsens.

**Key Finding:** The shared-Q constraint is NOT the primary bottleneck. The kernel functional form itself appears insufficient for the complete quark sector.

---

## Methodology

### Train/Holdout Split

To avoid turning the model into an unfalsifiable fit machine, observables were split:

**Training targets** (used in optimization):
- m_c = 1.27 GeV
- |V_us| = 0.225
- |V_cb| = 0.042

**Holdout targets** (NOT used in optimization):
- m_s = 0.093 GeV
- m_u = 0.00216 GeV
- m_d = 0.00467 GeV
- |V_ub| = 0.00382

### Minimality Ladder

| Level | Model | Extra Params | Description |
|-------|-------|--------------|-------------|
| 0 | Base | 0 | Shared Q, shared kernel params |
| 1 | Shift | +1 | delta_H for down-type Higgs localization |
| 2 | Width | +1 | q_u != q_d (sector-specific gear ratios) |
| 3 | Both | +2 | delta_H + q split |
| 4 | Full | +3 | Independent Q_u, Q_d positions |

### Decision Rule

Accept Level N+1 only if holdout loss improves by >20% compared to Level 0.

---

## Results

### Summary Statistics

| Level | Training Loss (median) | Holdout Loss (median) | Decision |
|-------|------------------------|----------------------|----------|
| 0 (Base) | 0.0298 | 149.86 | BASELINE |
| 1 (Shift) | 0.5970 | 832.54 | REJECT (degraded by 455.6%) |
| 2 (Width) | 0.1067 | 315.63 | REJECT (degraded by 110.6%) |
| 3 (Both) | 0.1315 | 421.78 | REJECT (degraded by 181.5%) |
| 4 (Full) | 1.0695 | 477.13 | REJECT (degraded by 218.4%) |

### Key Observations

1. **Level 0 achieves the best holdout performance** despite having the fewest parameters
2. **All relaxations DEGRADE holdout loss** - the opposite of what would happen if shared-Q were the bottleneck
3. **Train-holdout gap is much larger than the base model** for all relaxations
4. **Light quark masses remain ~100% wrong** at all levels

### Best Solution (Level 0)

| Observable | Predicted | Target | Error | Type |
|------------|-----------|--------|-------|------|
| m_c | 1.27 GeV | 1.27 GeV | 0.1% | TRAIN |
| |V_us| | 0.195 | 0.225 | 13% | TRAIN |
| |V_cb| | 0.041 | 0.042 | 3% | TRAIN |
| m_s | 0.020 GeV | 0.093 GeV | 79% | HOLDOUT |
| m_u | ~0 | 0.00216 GeV | 100% | HOLDOUT |
| m_d | ~0 | 0.00467 GeV | 100% | HOLDOUT |
| |V_ub| | 0.006 | 0.00382 | 65% | HOLDOUT |

---

## Interpretation

### Why Relaxations Don't Help

The results indicate that the shared-Q constraint is **not** the primary limitation. If it were:
- Adding delta_H (Level 1) should improve CKM flexibility
- Adding q split (Level 2) should improve mass hierarchy flexibility
- Both together (Level 3) should show compounding benefits

Instead, we observe:
- Training loss worsens for all relaxations
- Holdout loss degrades significantly
- The train-holdout gap is much larger than the base model

### The Real Bottleneck

The data suggests the fundamental limitation is the **kernel functional form** itself, not the geometric constraints. Specifically:

1. **Light quark masses are structurally inaccessible**: m_u, m_d remain essentially zero regardless of model complexity. This requires SVD ratios S_2/S_0 ~ 10^{-5}, which the clockwork envelope cannot produce while maintaining non-degenerate mixing.

2. **CKM third-generation elements are correlated with masses**: |V_ub|, |V_td|, |V_ts| depend on the same matrix structure that determines mass hierarchies. You cannot fix one without affecting the other.

3. **The interference term (1 + eps*exp(i*phi)) has limited texture flexibility**: It can create zeros via phase cancellation but cannot independently control all 9 Yukawa matrix elements.

---

## Defensible Statement

> "A controlled minimality analysis was performed to test whether the shared-Q constraint between up and down Yukawa operators is the primary bottleneck limiting quark sector fits. Five model levels were tested, from the base model (shared Q, shared kernel parameters) through progressively relaxed variants including Higgs-localization shifts, sector-specific gear ratios, and fully independent Q positions.
>
> The results show that no relaxation improves holdout performance; in fact, all variants exhibit degraded generalization compared to the base model, with train-holdout gaps much larger than the baseline. This indicates degraded generalization rather than genuine improvement.
>
> We conclude that the shared-Q constraint is not the primary bottleneck. The kernel functional form itself appears insufficient to simultaneously reproduce the full quark mass spectrum and CKM matrix. The model can fit a subset of observables (m_c, |V_us|, |V_cb|) but fails to generalize to light quark masses and third-generation CKM elements, regardless of geometric freedom.
>
> This suggests that additional physics beyond simple geometric interference is required for a complete quark sector description."

---

## Recommendations

1. **Do not relax the shared-Q constraint** - it does not help and violates SU(2)_L intuition without physical justification

2. **Do not claim the model "reproduces the quark sector"** - only a subset of observables can be fit

3. **Consider alternative approaches**:
   - Different kernel functional forms (not just envelope modifications)
   - Explicit Froggatt-Nielsen charge-based textures
   - Additional flavor symmetry structure
   - Radiative corrections / RG running effects

4. **Report honestly**: The clockwork kernel improves the m_c vs |V_us| tradeoff compared to Gaussian, but this is a partial success, not a complete solution

---

## Technical Details

### Optimizer Settings

```
maxiter: 50
popsize: 8
tol: 1e-5
seeds per geometry: 5
geometries tested: 3
penalty per extra param: 0.05
holdout improvement threshold: 20%
```

### Files

- `diagnostics/09_minimality_ladder.py` - Test script
- `diagnostics/results/09_minimality_ladder.txt` - Raw results
- `src/alternative_kernels.py` - Model implementations (Levels 0-4)
- `src/observables.py` - Train/holdout loss functions

---

## Conclusion

The minimality ladder analysis provides a rigorous, falsifiable test of the shared-Q bottleneck hypothesis. The hypothesis is **rejected**: relaxing the constraint does not improve generalization. The model's limitations are structural, residing in the kernel functional form rather than the geometric constraints.

This is a scientifically meaningful negative result that constrains future model-building efforts.
