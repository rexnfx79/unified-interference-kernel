---
type: concept
title: Quantum Information
tags: [information, qm]
related:
  - preskill-qit-entropy
  - qed-qm-information
  - it-from-bit
  - projection-regimes
  - information-measure-for-projection-regimes
  - holographic-principle
island: true
bridge_tags: [information, qm]
approach: quantum
plausibility: pursue
status: established
created: 2026-06-01
updated: 2026-06-01
---

# Quantum Information

## Proven (ingested via [[preskill-qit-entropy]])

| Tool | Role |
|------|------|
| Von Neumann entropy \(S(\rho)\) | Mixed-state information |
| CP maps | Measurement, decoherence |
| Entanglement entropy | Subsystem information |
| Quantum Fisher information | Parameter distinguishability |

## Proposed flavor application (conjecture — **pursue**)

Define per sector:

\[
\rho_Y = \frac{Y Y^\dagger}{\operatorname{Tr}(Y Y^\dagger)}
\]

Compute \(S(\rho_Y)\), effective rank, off-diagonal entropy. Test whether metric-dominated neutrinos show higher entropy than quarks ([[information-measure-for-projection-regimes]]).

## Links to repo

- **Implemented:** `src/flavor_information.py`, `src/qed_information.py`, `diagnostics/11–16` (12–16 pre-registered falsifiers; 15–16 refute ρ_Y QFI/coherence mechanism)
- Consistent Yukawa sources per sector via kernel + transfer-test geometries
- [[neutrino-observables-gap]] resolved — PMNS in `observables.py`

## Refuted shortcuts

- QIT → primes → masses: **dead** without spectral intermediate
- Decoherence metaphor for anarchy without calculation: **watch only**

## Related

[[holographic-principle]], [[qm-to-information-what-is-measurable]]
