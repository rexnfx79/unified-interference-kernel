---
type: concept
title: Hilbert Spaces and Quantum Mechanics
tags: [spectral, qm]
related:
  - qed-qm-information
  - hilbert-polya-conjecture
  - multi-sided-bridge-framework
island: true
bridge_tags: [spectral, qm]
approach: spectral
status: established
created: 2026-06-01
updated: 2026-06-01
plausibility: established
---

# Hilbert Spaces and Quantum Mechanics

## Proven (physics)

- States, self-adjoint observables, unitary evolution, Born rule — standard QM formalism (textbook level).

## Conjecture (this project)

- [[interference-kernel]] is effective \(L^2\) overlap readout ([[split-fermion-overlaps]]).
- Zeta zeros share Hilbert-space structure with a physical \(H\) ([[hilbert-polya-conjecture]]).

## Refuted misuse

- Applying infinite-dimensional zero statistics directly to 3×3 flavor ([[why-not-zeta-flavor-numerology]]).
- Using non-self-adjoint \(H=xp\) as if it were standard spectral QM ([[berry-keating-hamiltonian]]).

## Established Core

Quantum mechanics is formulated on a **Hilbert space** \(\mathcal{H}\):

- States: rays in \(\mathcal{H}\) (vectors up to phase)
- Observables: self-adjoint operators \(A = A^\dagger\)
- Dynamics: unitary \(U(t) = e^{-iHt/\hbar}\) with \(H\) self-adjoint
- Born rule: probabilities from \(|\langle\psi|\phi\rangle|^2\)

This is the **strongest** bridge between math and measured reality in the notebook. Everything else should connect *through* this layer when claiming "math → physics."

## Why Hilbert Space (Not Just Banach or p-adic)

| Property | Physical role |
|----------|---------------|
| Inner product | Transition amplitudes, probability |
| Self-adjointness | Real eigenvalues = measurable quantities |
| Spectral theorem | Decomposition into modes — foundation of QFT mode sums |
| Completeness | Cauchy sequences of states converge — needed for perturbation theory |

[[p-adic-analysis]] uses different topology; **SM phenomenology** is overwhelmingly Hilbert-based. p-adics are optional enrichment, not replacement.

## Bridge to Arithmetic

[[hilbert-polya-conjecture]] asks: does **the same** Hilbert-space machinery describe arithmetic zeros?

- **If yes:** primes enter via spectral density (indirect)
- **If no:** zeta zeros are not "physical" in standard Hilbert QM — separate mystery

## Bridge to Flavor (Cautious)

Split-fermion models: fermions localized in extra dimension → overlap integrals \(\int \psi_L^* \psi_R \, dy\) on **Hilbert space** \(L^2(\mathbb{R})\).

[[interference-kernel]] may be an **effective** parameterization of such overlaps + phases — **pursue** this path before zeta numerology. See [[spectral-interpretation-of-flavor]] (deprioritized until split-fermion derivation attempted).

## Bridge to Information

- Density matrices on \(\mathcal{H}\) — mixed states, entropy
- POVMs — generalized measurements (it-from-bit)
- Channel maps — quantum information theory

See [[qed-qm-information]], [[quantum-information]].

## Failure Modes

- Using Hilbert space **language** without self-adjoint \(H\) (Berry–Keating raw \(xp\)) — **not** standard QM
- Confusing infinite-dimensional zero spectrum with 3×3 flavor — scale error
