# Next Steps for Clean Repository

## Immediate Actions (Can Run Now)

### 1. Full Quark Sector Optimization
```bash
# Run overnight - will take 8-10 hours
python3 scripts/01_optimize_quarks.py \
    --max-coord 10 \
    --n-seeds 5 \
    --maxiter 200 \
    --output data/quark_full_results.csv
```

Expected output: ~1000-2000 geometries optimized, ~300-500 survivors

### 2. Manuscript Minimization

The plan is documented in `docs/MANUSCRIPT_MINIMIZATION_PLAN.md`.

**Key cuts to make:**
- Trim introduction literature review (180 lines → 60 lines)
- Remove symbol glossary and UV speculation from Model section
- Condense Methods section
- Trim Parameter Attribution and Extension Tests in Results
- Brief Discussion section

**Target**: 800-900 lines from current 1196 lines

### 3. Git Integration

Initialize the clean repo as a git repository:
```bash
cd /Users/alexm4/Cursor\ Repos/unified-interference-kernel
git init
git add .
git commit -m "Initial commit: Clean implementation of universal interference kernel

- Core modules: kernel, observables, optimizer
- Comprehensive test suite (all tests passing)
- Quark optimization infrastructure
- Complete documentation

Foundation ready for full optimization runs."
```

## Short-Term (This Week)

### 1. Analyze Quark Results
Once full optimization completes:
```python
import pandas as pd
df = pd.read_csv('data/quark_full_results.csv')

# Find survivors
survivors = df[
    (df['Vus'] > 0.17) & (df['Vus'] < 0.29) &
    (df['Vcb'] > 0.025) & (df['Vcb'] < 0.060) &
    (df['Vub'] > 0.0018) & (df['Vub'] < 0.0060)
]

print(f"Survivors: {len(survivors)} / {len(df)}")
print(f"Best CKM loss: {df['loss_ckm'].min()}")
```

### 2. Create Basic Figures
```python
import matplotlib.pyplot as plt

# CKM loss vs mc
plt.scatter(df['loss_ckm'], df['mc'], alpha=0.5)
plt.xlabel('CKM Loss')
plt.ylabel('$m_c$ [GeV]')
plt.xscale('log')
plt.savefig('figures/ckm_mc_scatter.png', dpi=300)
```

### 3. Write Charged Lepton Script
Copy structure from `01_optimize_quarks.py`, modify for:
- Different geometries (L, E instead of Q, U, D)
- Variable phase parameters (k_e, η_e)
- Charged lepton masses as observables

## Medium-Term (Next 2 Weeks)

### 1. Statistical Validation
```python
def compute_z_scores(obs, sector='quark'):
    # Implementation in src/observables.py
    pass
```

### 2. Pareto Analysis
```python
def compute_pareto_envelope(df, x='loss_ckm', y='mc'):
    # Find Pareto frontier
    # Identify knee region
    # Compute statistics
    pass
```

### 3. Publication Figures
- CKM-mc Pareto envelope with knee
- Three regimes comparison
- 6σ validation plots
- Parameter distributions

## Long-Term (Next Month)

### 1. Complete Paper
- Finalize manuscript_minimal.tex
- Generate all figures
- Write complete Methods section
- Polish Discussion

### 2. Reproducibility Package
- Archive on Zenodo
- Include all data, code, figures
- Add DOI to paper
- Write reproduction instructions

### 3. Supplementary Materials
- Extended methods
- Additional figures
- Full parameter tables
- Sensitivity analysis

## Optional Enhancements

### Code Quality
- Add type hints throughout
- Increase test coverage to 100%
- Add performance benchmarks
- Profile and optimize hot paths

### Documentation
- API reference (Sphinx)
- Tutorial notebooks
- Video walkthrough
- FAQ

### Extensions
- 2D kernel generalization
- RGE evolution integration
- Alternative kernels comparison
- ML surrogate models

## Commands Cheat Sheet

```bash
# Run tests
cd tests && python3 test_kernel.py && python3 test_observables.py

# Quick optimization test
python3 scripts/01_optimize_quarks.py --limit 50

# Full optimization
python3 scripts/01_optimize_quarks.py --max-coord 10 --n-seeds 5

# Check results
python3 -c "import pandas as pd; df = pd.read_csv('data/quark_results.csv'); print(df.describe())"

# Activate environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Questions to Answer

1. **Does the Pareto knee reproduce?**
   - Run full optimization → compute envelope → check for knee at ~9.33e-4

2. **Do we get ~500 survivors?**
   - Depends on geometry sampling and optimization budget

3. **Is mc systematically overestimated?**
   - Check mean mc for survivors vs. PDG value (1.27 GeV)

4. **Do regime separations hold?**
   - Run lepton optimizations → compute Z-scores → verify statistical significance

## Success Criteria

- [ ] Full quark optimization completes successfully
- [ ] ≥300 survivors identified
- [ ] Pareto knee reproduces at similar location
- [ ] Manuscript trimmed to <900 lines
- [ ] All tests still pass
- [ ] Clean git history
- [ ] Complete documentation

---

**Current Status**: Foundation complete, ready for production runs.
