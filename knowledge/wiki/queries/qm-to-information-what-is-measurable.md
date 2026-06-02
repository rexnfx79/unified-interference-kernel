---
type: query
title: QM to Information — What Is Measurable?
tags: [qm, qed, information]
related:
  - qed-qm-information
  - quantum-information
  - it-from-bit
  - multi-sided-bridge-framework
  - fisher-transfer-universality-test
  - research-strategy
status: refuted
created: 2026-06-01
updated: 2026-06-02
---

# QM to Information — What Is Measurable?

## Question

Working **from QED/QM toward information theory**: what quantities are **operationally measurable** and could constrain "reality as information"?

## Path A progress (diagnostics 12–20)

Pre-registered tests in [[research-strategy]]:

| Test | Result |
|------|--------|
| Regime vs \(S(\rho_Y)\) (`diagnostics/12_*`) | **Refuted** — \|r\| < 0.05 vs regime |
| Yukawa information inequalities (`diagnostics/13_*`) | **No cross-sector mechanism** — pooled \|r\| < 0.1 |
| QFI / coherence vs mixing (`diagnostics/15_*`) | **Refuted (pooled)** — max \|r\| = 0.017 < 0.25, n=10,080 |
| Decoherence vs CKM/PMNS (`diagnostics/16_*`) | **Refuted** — max \|r\| = 0.241 < 0.25; upper bound trivial |
| Experimental Fisher / PDG Jacobian (`diagnostics/17_*`) | **Dead for mechanism** — alignment 0.46 < 0.50 |
| Open-system decoherence (`diagnostics/18_*`) | **Pooled refuted** — max pooled \|r\| = 0.007 |
| Fisher transfer quark→lepton (`diagnostics/19_*`) | **Refuted** — frozen loss 805.8, alignment 0.41 — [[fisher-transfer-universality-test]] |
| Collider-accessible Fisher sketch (`diagnostics/20_*`) | **Too thin** — event likelihood out of repo scope |
| \(S(\rho_Y)\) vs optimization loss | Mass loss r = −0.45 (confound; not a new physical law) |

**Honest status (2026-06-02):** No operational QIT→flavor **mechanism** survives repo tests. Path A **deprioritized** for mechanism pursuit. The interference kernel remains a **phenomenological** fit layer ([[interference-kernel]], [[repo-scientific-findings]]).

**Implementation:**
- `src/qed_information.py` — post-hoc QFI (refuted pooled)
- `src/experimental_fisher.py` — PDG-weighted Fisher, CR bounds, alignment
- `src/fisher_transfer.py` — pre-registered transfer test (refuted)
- `src/open_system_decoherence.py` — external \(p\) sketch (pooled refuted)

**Path D:** watch only — no promotion from Fisher or ρ_Y routes. See [[research-strategy]].

## Candidate Measures (From Established Physics)

| Measure | Definition | Already in SM/QED? |
|---------|------------|-------------------|
| Cross section | \(\sigma(E)\) from S-matrix | Yes |
| Spectral function | Im propagator | Yes (hadronic) |
| Von Neumann entropy | \(-\mathrm{Tr}\,\rho\log\rho\) | Thermal / open systems |
| Fisher information | Parameter estimation bound | Metrology |
| Entanglement entropy | Subsystem of pure state | Not SM collider default |
| Kolmogorov complexity | Shortest program for data | **Not** standard lab observable |

## Bridge Task

For each measure, ask:
1. Does it **determine** Yukawa structure or only constrain it?
2. Does arithmetic (primes, zeta) appear **necessarily** or only by hand?

## Concrete Sub-questions

- Does path-integral phase interference in QED reduce to [[interference-kernel]] form in a 1D extra-dimensional limit?
- Does RG running = information loss under [[projection-regimes]]? (Metaphor vs. theorem)
- Can holographic entropy bound explain **three** generations?

## Success

Derive one **inequality** on flavor parameters from QIT + QED that is testable and **not** already known from standard EFT. **Not achieved** in diagnostics 12–20.

## Approach

**C → B → (maybe) A** per [[multi-sided-bridge-framework]]. Do not start from zeta. After 19–20, constructive work shifts to phenomenology or Path D spectral ingest — not QIT→flavor retry without new hypothesis.
