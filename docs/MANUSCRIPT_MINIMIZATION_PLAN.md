# Manuscript Minimization Plan

## Current Status
- **Current length**: 1196 lines
- **Target length**: 800-900 lines
- **Lines to cut**: ~300-400 lines

## Sections to KEEP (Core 3-Regimes Story)

### 1. Front Matter (keep as is)
- Abstract ✓
- Introduction (trim literature review significantly)

### 2. Model (Section 2) - KEEP CORE
- 2.1 Universal Interference Kernel ✓
- 2.2 Discrete Geometry ✓ (brief)
- 2.3 Projection Regimes ✓ (KEY SECTION)
- 2.4 Scope and Epistemological Framework ✓
- 2.5 Observables ✓ (brief)
- **REMOVE**: Symbol glossary (move to appendix), Physical interpretation (too speculative), Unifying principle (redundant)

### 3. Methods (Section 3) - TRIM
- 3.1 Optimization Strategy ✓ (brief)
- 3.2 Hyperparameters (condense into 3.1)
- **REMOVE**: Detailed quark/lepton analysis subsections (merge into one "Sector-Specific Methods")

### 4. Results (Sections 4-5) - KEEP ALL THREE SECTORS
- 4.1 Quark Sector: Robust Fits ✓
- 4.2 Universal Pareto Envelope ✓ (KEY FINDING)
- 4.3 Parameter Attribution (brief version only)
- 4.4 Extension Tests (TRIM heavily - one paragraph summary)
- 4.5 CP Violation (KEEP - non-targeted prediction)
  
- 5.1 Charged Leptons: Phase-Sensitive Regime ✓ (KEY)
- 5.2 Neutrinos: Metric-Dominated Regime ✓ (KEY + anarchy)
- 5.3 Joint Tests (brief)
- **REMOVE**: Statistical structure subsection (fold into discussion)

### 5. Discussion (Section 6) - TRIM
- Keep: Interpretation, Scope, Connection to anarchy
- **REMOVE**: Extensive UV speculation, detailed future directions

### 6. Conclusions (Section 7) - KEEP (already concise)

## Specific Cuts

### Introduction (save ~100 lines)
- **Lines 46-228**: Trim literature review from 180 lines to ~60 lines
  - Keep: Split-fermion foundation, Branco limitation, This work's solution
  - Cut: Excessive comparison tables, redundant historical context
  - Cut: Detailed review of alternative solutions

### Model (save ~50 lines)
- **Section 2.5**: Remove "Physical Interpretation and UV Origin" (~30 lines)
- **Section 2.6**: Remove symbol glossary (~30 lines) - move to appendix or inline

### Methods (save ~40 lines)
- Condense 3.2, 3.3, 3.4 into one "Sector-Specific Methods" (~20 lines each saved)

### Results (save ~80 lines)
- **4.3 Parameter Attribution**: Trim from ~40 lines to ~15 lines
- **4.4 Extension Tests**: Trim from ~30 lines to ~10 lines  
- **5.3 Joint Tests**: Trim from ~15 lines to ~5 lines
- **5.4 Statistical Structure**: Remove entirely (~20 lines)

### Discussion (save ~30 lines)
- Trim future directions from detailed list to brief paragraph

## Final Structure (Target: ~850 lines)

1. Abstract (30 lines)
2. Introduction (120 lines - trimmed from 200)
3. Model (200 lines - trimmed from 250)
4. Methods (80 lines - trimmed from 120)
5. Results: Quarks (150 lines - trimmed from 180)
6. Results: Leptons (120 lines - trimmed from 140)
7. Discussion (80 lines - trimmed from 110)
8. Conclusions (40 lines)
9. References (50 lines)

**Total**: ~870 lines

## Key Preservation Priorities

1. **Three Regimes Framework** - This is the core contribution
2. **Pareto Knee** - The main structural finding for quarks
3. **6σ Validation** - Statistical evidence for regime separation
4. **Anarchy as Emergence** - Key interpretive insight
5. **Scope/Epistemology** - Essential for referee-proofing

## What Gets Cut

- ❌ Extended literature comparisons
- ❌ Detailed historical context
- ❌ Symbol glossary (inline instead)
- ❌ Speculative UV discussion
- ❌ Excessive methodological detail
- ❌ Redundant statistical arguments
- ❌ Over-detailed extension testing

## Implementation

Create `manuscript_minimal.tex` with targeted cuts, preserving:
- All key equations
- All key figures
- All 6σ validation claims
- Epistemological framing
- Three-regimes story arc
