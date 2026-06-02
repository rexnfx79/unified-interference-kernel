# Manuscript Key Results Snapshot (Quantitative + Caveats)

> **Canonical source:** `../../manuscript.tex`. Quantitative claims only; interpretive language trimmed.

## Kernel definition (honest scope)

\[
Y_{ij} = \exp\left(-\frac{d_{ij}^2}{2\sigma^2}\right)\left[1 + \varepsilon e^{i\Phi_{ij}}\right], \quad \Phi_{ij} = \alpha + k\frac{x_i+x_j}{2} + \eta(x_i-x_j).
\]

**"Universal" in manuscript** = single **functional form** reused across sectors, **not** universal parameter values or UV origin.

## Three projection regimes (phenomenological labels)

| Regime | Sector | Extra / shifted params |
|--------|--------|-------------------------|
| Envelope-dominated | Quarks | \(\varepsilon_u, \varepsilon_d\) |
| Phase-sensitive | Charged leptons | \(k_e, \eta_e\) |
| Metric-dominated | Neutrinos | \(g_{\text{env}}\), separate \(k,\eta,\varepsilon_\nu\) vs \(k_e,\eta_e,\varepsilon_e\) |

## Reported optimization statistics (with caveats)

| Sector | Survivor rate (manuscript) | Caveat |
|--------|--------------------------|--------|
| Quarks | 0% strict PDG survivors | \(m_c\) systematic failure (~15× too high in honest figure); CKM–\(m_c\) Pareto trade-off |
| Charged leptons | 60% (100 geometries) | Best loss \(\sim 10^{-11}\); muon scatter in honest revision |
| Neutrinos | 45% (480 opts) | PMNS angles often <10% error; 50% opts fail (\(\theta_{23}\approx0\)); \(g_{\text{env}}\)–mixing correlation weak (\(r<0.1\)) |

## Parameter overlap across sectors (exploratory, not predictive)

- \(\sigma \approx 4\)–5, \(\alpha \approx 2.5\)–3.0 across sectors — may reflect **bounded optimization ranges**, not physical universality ([[repo-scientific-findings]] refutes parameter transfer).

## Split fermions (established EFT context, conjectured link to kernel)

Manuscript cites Arkani-Hamed–Schmaltz split fermions: Yukawas from extra-dimensional overlaps. Present work uses **effective** interference kernel over internal coordinate — complementary to UV boundary constructions.

## What manuscript does **not** claim (post-diagnostics)

- Predictive cross-sector parameter relations.
- Complete quark sector reproduction with Gaussian kernel.
- Validated three-regime mechanism (Fig. 4 noted schematic, not data-derived).

See wiki: [[manuscript-key-results]], [[interference-kernel-manuscript]], [[proven-vs-conjecture-ledger]].
