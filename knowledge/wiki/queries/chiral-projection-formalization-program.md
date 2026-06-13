---
type: query
title: Chiral Projection Formalization Program
tags: [information, flavor, qm, spectral]
related:
  - chiral-projection-thesis
  - adversarial-review-chiral-projection-thesis
  - projection-regimes
  - interference-kernel
  - split-fermion-overlaps
  - holographic-principle
  - plausibility-register
status: open
created: 2026-06-12
updated: 2026-06-12
---

# Chiral Projection Formalization Program

**Goal:** Name \(\Pi_\Omega\) explicitly, derive flavor predictions, and pre-register falsifiers. Speculation is intentional — handhold on unknowns before proof attempts.

## Unknowns inventory

| Symbol | Role |
|--------|------|
| \(\mathcal{C}\) | Parent parity-symmetric state / theory |
| \(\Pi_\Omega\) | Oriented reconstruction map |
| \(\Omega\) | Orientation datum |
| \(\varepsilon_{\mathrm{portal}}\) | Cross-sector coupling |
| \(x_i\) | Internal flavor coordinates (kernel positions) |
| Sector \(s\) | Quark / lepton / neutrino projection channel |

## Six formalizations

### F1 — Oriented extra dimension

\[
\mathcal{Y}_{ij} = \int dy\; w_\Omega(y)\,\psi_{L,i}^*\psi_{R,j} H(y)
\]

\[
\Phi_{\pm} = \alpha + \frac{k}{2}(x_i+x_j) \pm \eta(x_i-x_j)
\]

**Flavor:** \(J_{\mathrm{mirror}} \approx -J_{\mathrm{ours}}\) under \(\eta\to-\eta\); sector \(\eta_s\) differ.

### F2 — 6×6 Schur complement

\[
\mathcal{Y} = \begin{pmatrix} Y & X \\ X^\dagger & Y' \end{pmatrix}, \quad
Y_{\mathrm{eff}} = Y - X(Y'+\rho I)^{-1}X^\dagger
\]

\(Y'\) uses \((k,\eta)\to(-k,-\eta)\), \(\alpha'=\alpha+\pi\).

**Flavor:** non-perturbative CP corrections; neutrino block largest portal.

### F3 — Holomorphic coordinates

\(x_i\in\mathbb{C}\); \(\Phi_{ij} = \alpha + k\,\mathrm{Re}(x_i+x_j) + \eta\,\mathrm{Im}(x_i-x_j)\); \(\Pi_\Omega: x\mapsto\Omega\bar{x}\).

**Flavor:** \(\alpha\approx\pi\) as sheet lock; CP from \(\mathrm{Im}\) part.

### F4 — Fibre holonomy

Shared base \(L=Q\) (diag 26); fibre holonomy \(\mathrm{Hol}_s\) per sector. Envelope / phase / metric = radial / argument / norm projections.

**Flavor:** \(g_{\mathrm{env}} = f(\|\mathrm{Hol}_\nu\|)\); test \(r(\theta_{23})\).

### F5 — Parity doublet in parameter space

\(R_P: (k,\eta,\alpha)\mapsto(-k,-\eta,\alpha+\pi)\); \(Y = Y_{\mathrm{SM}}(\theta) + \varepsilon_p \tilde{Y}(\theta_P)\).

**Flavor:** \(\varepsilon_{q} \ll \varepsilon_{\nu}\); fix wrong-sign \(J\).

### F6 — Oriented boundary (CFT-style)

\(\Psi_{\mathrm{bulk}} = \mathcal{R}_\Omega[\Psi_\Sigma]\). Must reduce to F1–F5 on finite model or stay philosophy.

## Prediction table

| ID | Statement | Kill |
|----|-----------|------|
| P1 | Portal with \(\Phi_{-}\) improves quark \(J\) | Diag 42: **killed** (simple portals) |
| P2 | \(\varepsilon_{\nu} \gg \varepsilon_{q}\) | Diag 44 (planned) |
| P3 | Shared \(L=Q\) required | Joint geometry (diag 26) |
| P4 | \(\alpha\) clusters near \(\pi\) | Complex-\(x\) fit (diag 45) |
| P5 | \(g_{\mathrm{env}} = f(\mathrm{Hol}_\nu)\) | \(r<0.3\) after holonomy fit |
| P6 | 6×6 parent reduces total joint params | Diag 43 (planned) |

## Diagnostic 42 — quark portal audit

**Script:** `diagnostics/42_chiral_projection_portal_audit.py`  
**Setup:** N=20 geom, 3 seeds; baseline vs parity-\(\pi\) portal, mirror-\(\eta\) portal, 6×6 Schur.

| Model | Median train | Median holdout | Median \(J\) err | \(J\) sign rate |
|-------|-------------|----------------|------------------|-----------------|
| baseline | 1.49 | 348 | 1.00 | 60% |
| parity_pi | 1.37 | 362 | 1.00 | 50% |
| mirror_eta | 1.58 | 362 | 1.00 | 58% |
| schur_6x6 | 1.00 | 503 | 1.00 | 60% |

**Pre-registered falsifier:** improve median \(|J-J_{\mathrm{PDG}}|/J_{\mathrm{PDG}}\) with train \(\leq 1.2\times\) baseline.

**Verdict:** **FAIL** — simple portals do not give quark CP handhold. Schur overfits (train ↓, holdout ↑).

**Implication:** projection geometry, if real, must be non-perturbative (constrained Schur), neutrino-dominated, or geometric (F3/F4), not additive tail on kernel.

## Proof ladder

| Level | Criterion | Status |
|-------|-----------|--------|
| L0 | Parent Lagrangian; anomalies cancel | Literature (mirror matter) |
| L1 | One parent \((k,\eta,\alpha)\) → all sectors | **Not met** (transfer refuted) |
| L2 | Portal improves \(J\) + holdout | **Failed** (diag 42) |
| L3 | Joint 3-sector 6×6 beats independent | **Planned** (diag 43) |
| L4 | Lab portals (n–n', \(\gamma\)–\(\gamma'\)) | External |
| L5 | Cosmology (\(N_{\mathrm{eff}}\), DM) | External |

## Planned diagnostics

| Diag | Test | Formalization |
|------|------|---------------|
| 43 | Joint 3-sector 6×6 constrained fit (diag 26 corpus) | F2, F4 |
| 44 | Neutrino-first portal only | F1, F5 |
| 45 | Complex-coordinate kernel | F3 |
| 46 | Holonomy \(g_{\mathrm{env}}\) parametrization | F4 |
| 47 | Literal split-fermion oriented overlap | F1 |

## Related

[[adversarial-review-chiral-projection-thesis]], [[information-creates-reality]], `diagnostics/results/42_chiral_projection_portal_audit.txt`
