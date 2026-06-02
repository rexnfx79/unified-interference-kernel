# Preskill Quantum Information — Entropy (Curated Summary)

> **Primary source:** Preskill, *Quantum Computation and Quantum Information* lecture notes / Ch. 2–11 (Caltech); see also Nielsen & Chuang Ch. 11.  
> **Wiki concepts:** [[quantum-information]], [[qed-qm-information]], [[information-measure-for-projection-regimes]]

## Von Neumann entropy (established)

For density matrix \(\rho\):

\[
S(\rho) = -\operatorname{Tr}(\rho \log \rho)
\]

- Pure state: \(S = 0\)
- Maximally mixed qubit: \(S = \log 2\)

## Completely positive maps (established)

Quantum operations (measurement, decoherence, channels) are **CP maps** on density matrices. Born rule and decoherence are standard QM — operational backbone for [[it-from-bit]].

## Entanglement entropy (established)

For bipartite pure state \(|\psi\rangle_{AB}\), reduced state \(\rho_A = \operatorname{Tr}_B |\psi\rangle\langle\psi|\):

\[
S(\rho_A) = -\operatorname{Tr}(\rho_A \log \rho_A)
\]

Area-law entanglement in gapped systems; volume-law in highly entangled states.

## Fisher information / distinguishability (established)

Quantum Fisher information bounds parameter estimation precision — candidate **measurable information** in [[qm-to-information-what-is-measurable]].

## Proposed application to flavor (conjecture — this repo)

For Yukawa matrix \(Y\), form \(\rho \propto Y Y^\dagger / \operatorname{Tr}(Y Y^\dagger)\) and compute \(S(\rho)\), effective rank, off-diagonal entropy.

**Hypothesis:** metric-dominated neutrino regime ([[projection-regimes]]) shows higher \(S\) or lower rank than quark sector — test in [[information-measure-for-projection-regimes]].

## What QIT does not give

- Yukawa values from entropy alone
- Prime or zeta structure
- Proof that information creates reality ([[what-proves-information-creates-reality]])
