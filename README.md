# Unified Interference Kernel: Phenomenological Flavor Parameterization

Clean, minimal implementation of an interference-kernel **functional form** for organizing Yukawa couplings. This is a phenomenological parameterization, not a validated universal theory.

> **Canonical status:** [`manuscript-ledger-alignment.md`](knowledge/wiki/synthesis/manuscript-ledger-alignment.md), [`survivor-protocol-preregistered.md`](knowledge/wiki/synthesis/survivor-protocol-preregistered.md). **Tier 0–1 complete** (2026-06-02o). Reproduce: `./scripts/reproduce_phenomenology_tranche.sh`. **Next:** [`future-work.md`](knowledge/wiki/synthesis/future-work.md) Tier 5 (blocked T5.2) or optional CP sweeps.

## Overview

This repository implements a kernel form reused across fermion sectors:

```
Y_ij = exp(-d²/(2σ²)) × [1 + ε exp(iΦ)]
```

where:
- `d = |x_i - x_j|`: Distance in internal flavor coordinate (NOT spacetime)
- `Φ = α + k(x_i + x_j)/2 + η(x_i - x_j)`: Phase structure
- σ: Envelope width
- ε: Interference strength

### Sector labels (organizational, not validated mechanism)

1. **Envelope-dominated setup (Quarks)** — baseline (σ, k, η) + (ε_u, ε_d). **0% strict survivors** at pre-registered PDG-relative tolerances (diag 21/27/30).
2. **Phase-sensitive setup (Charged leptons)** — variable (k_e, η_e). Legacy full-objective ~60% at 10% tolerance; **1% strict** under train/holdout (diag 22, holdout m_e fails).
3. **Metric-dominated setup (Neutrinos)** — envelope compression g_env on Y_ν. **27.8% joint strict** (PMNS + Δm², diag 28); **78.9% PMNS-only strict** (diag 23).

Universal **parameter** values across sectors are **refuted** (transfer test loss ≳ 800 frozen vs free).

## Repository Structure

```
unified-interference-kernel/
├── knowledge/             # LLM Wiki knowledge base (information → reality research)
│   ├── purpose.md         # Research goals and thesis
│   ├── wiki/              # Interlinked concept/query/synthesis pages
│   └── raw/sources/       # Source documents for ingest
├── tools/llm_wiki/        # LLM Wiki desktop app (git submodule)
├── AGENTS.md              # Cursor/agent instructions for wiki maintenance
├── src/
│   ├── kernel.py          # Universal kernel implementation
│   ├── observables.py     # Observable extraction (CKM, masses, PMNS)
│   └── optimizer.py       # Differential evolution wrapper
├── scripts/
│   ├── 01_optimize_quarks.py
│   ├── 02_pareto_envelope_comparison.py
│   └── 03_true_transfer_test.py
├── tests/
│   ├── test_kernel.py
│   └── test_observables.py
├── data/                  # Generated optimization results
├── figures/               # Manuscript figures
└── diagnostics/           # Rigorous validation scripts
```

## Knowledge Base (Conjecture Notebook)

Multi-sided map connecting math to reality — **phenomenology tranche complete**; flavor mechanism paths refuted; arithmetic bridges **watch-only** ([`conjecture-to-physics-avenues.md`](knowledge/wiki/synthesis/conjecture-to-physics-avenues.md)).

- **Framework:** [`knowledge/wiki/synthesis/multi-sided-bridge-framework.md`](knowledge/wiki/synthesis/multi-sided-bridge-framework.md)
- **Failures / verdicts:** [`knowledge/wiki/synthesis/plausibility-register.md`](knowledge/wiki/synthesis/plausibility-register.md)
- **Agent workflow:** [`knowledge/AGENTS.md`](knowledge/AGENTS.md)

Optional: run the [LLM Wiki](https://github.com/nashsu/llm_wiki) desktop app against `knowledge/`:

```bash
cd tools/llm_wiki && npm install && npm run tauri dev
# Open Project → select knowledge/
```

Verify wiki snapshots match repo: `python3 scripts/lint_wiki_manifest.py`

### Reproduce headline phenomenology

```bash
./scripts/reproduce_phenomenology_tranche.sh
```

Checks frozen reports (diagnostics 21–23, 27–28, 30, 32–33) and SVD phase unit test.

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

## Theory (honest scope)

Effective-field-theory-level **parameterization**. What is established vs refuted:

| Claim | Status |
|-------|--------|
| Single kernel **form** reused across sectors | Phenomenological |
| Universal **parameters** across sectors | Refuted |
| Three-regime **mechanism** validated by data | Refuted (quarks fail) |
| Quark CKM–m_c Pareto trade-off | Structural (diag 21/27) |
| Neutrino joint strict (PMNS + Δm²) | 27.8% (diag 28) |

See `SCIENTIFIC_FINDINGS.md` and the wiki ledger for details.

## Reproduce headline diagnostics

```bash
./scripts/reproduce_phenomenology_tranche.sh
```

Frozen reports include phenomenology tranche (21–23, 27–28, 30–33), Tier 1 phase audit (36), and Tier 5 conjecture audits (34–38). Legacy CSV protocol: `data/README.md`.

## References

- PDG 2024 for all experimental targets
- Manuscript: `manuscript.tex` (phenomenological framing; quark failure upfront)

## License

MIT
