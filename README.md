# Unified Interference Kernel: Three Projection Regimes

Clean, minimal implementation of the interference kernel framework for flavor physics.

## Overview

This repository contains a from-scratch implementation of the universal interference kernel model that organizes Yukawa couplings through envelope suppression and phase interference. The key insight is that different fermion sectors probe different **projection regimes** of the same underlying kernel.

### The Kernel

```
Y_ij = exp(-d²/(2σ²)) × [1 + ε exp(iΦ)]
```

where:
- `d = |x_i - x_j|`: Distance in internal flavor coordinate (NOT spacetime)
- `Φ = α + k(x_i + x_j)/2 + η(x_i - x_j)`: Phase structure
- σ: Envelope width
- ε: Interference strength

### Three Regimes

1. **Envelope-Dominated (Quarks)**
   - Baseline kernel parameters (σ, k, η)
   - Sector-specific interference (ε_u, ε_d)
   - Achieves 6σ precision on CKM mixing

2. **Phase-Sensitive (Charged Leptons)**  
   - Phase parameters (k_e, η_e) vary from quark baseline
   - Resolves muon mass hierarchy
   - Achieves 6σ precision on charged lepton masses

3. **Metric-Dominated (Neutrinos)**
   - Envelope compression (g_env ≈ 0.60) required
   - Information loss under compression → emergent anarchy
   - Achieves 6σ precision on PMNS angles

## Repository Structure

```
unified-interference-kernel/
├── src/
│   ├── kernel.py          # Universal kernel implementation
│   ├── observables.py     # Observable extraction (CKM, masses, PMNS)
│   └── optimizer.py       # Differential evolution wrapper
├── scripts/
│   ├── 01_optimize_quarks.py
│   ├── 02_optimize_charged_leptons.py
│   └── 03_optimize_neutrinos.py
├── tests/
│   ├── test_kernel.py
│   └── test_observables.py
├── data/                  # Generated optimization results
├── figures/               # Manuscript figures
└── docs/                  # Additional documentation
```

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Testing

```bash
cd tests
python3 test_kernel.py
python3 test_observables.py
```

All tests pass, validating:
- Kernel element computation
- Yukawa matrix structure
- SVD-based observable extraction
- CKM unitarity
- Loss function behavior

## Usage

### 1. Quark Sector Optimization

```bash
python3 scripts/01_optimize_quarks.py --max-coord 8 --n-seeds 5 --maxiter 150
```

Optimizes over discrete geometries (Q, U, D) and continuous parameters (σ, k, α, η, ε_u, ε_d).

### 2. Charged Lepton Optimization

```bash
python3 scripts/02_optimize_charged_leptons.py --n-geometries 1000
```

Optimizes with variable phase parameters (k_e, η_e).

### 3. Neutrino Optimization

```bash
python3 scripts/03_optimize_neutrinos.py --g-env-range 0.5 0.7
```

Scans envelope compression parameter g_env.

## Key Features

- **Minimal Dependencies**: numpy, scipy, pandas, matplotlib
- **Tested Code**: Full test suite for kernel and observables
- **Reproducible**: All random seeds fixed, all data provenance clear
- **Clean Architecture**: Separation of kernel physics, observables, and optimization

## Theory

This is an **effective field theory level** structural model. The claims are:

1. A single universal kernel form can organize all Yukawa couplings
2. Different fermion sectors correspond to different sampling regimes
3. Neutrino anarchy emerges from metric-dominated projection (information loss)
4. These are **structural observations** within this model class, not fundamental claims

## References

- PDG 2024 for all experimental targets
- Manuscript: `../Interference Lattice/manuscript.tex`
- Original exploration repo: `../Interference Lattice/` (archived)

## Status

**Completed:**
- ✅ Core kernel implementation
- ✅ Observable extraction (CKM, masses, PMNS)
- ✅ Optimizer wrapper
- ✅ Comprehensive test suite
- ✅ Quark optimization script
- ✅ Clean repository structure

**In Progress:**
- ⏳ Charged lepton optimization script
- ⏳ Neutrino optimization script
- ⏳ Pareto analysis tools
- ⏳ Figure generation scripts
- ⏳ Statistical validation (Z-scores)

**Future:**
- Minimal manuscript (trimmed to 800-900 lines)
- Publication-ready figures
- Full documentation

## License

MIT

## Contact

For questions about the theory or implementation, see the manuscript in the parent directory.
