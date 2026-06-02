# Kernel Implementation Snapshot

> **Canonical source:** `../../src/kernel.py` (parent repo). Snapshot for wiki ingest — do not edit in place of code.

## Mathematical form (Gaussian, p=2)

\[
Y_{ij} = \exp\left(-\frac{d_{ij}^2}{2\sigma^2}\right) \times \left[1 + \varepsilon \exp(i\Phi_{ij})\right]
\]

with \(d_{ij} = |x_i - x_j|\) and

\[
\Phi_{ij} = \alpha + k\frac{x_i + x_j}{2} + \eta(x_i - x_j).
\]

## Implementation facts (proven in repo)

| Function | Role |
|----------|------|
| `compute_kernel_element` | Single \(Y_{ij}\) from coordinates + \((\sigma, k, \alpha, \eta, \varepsilon)\) |
| `compute_yukawa_matrix` | Full \(3\times3\) from left/right position triples |
| `compute_quark_yukawas` | \(Y_u, Y_d\) with shared kernel params, separate \(\varepsilon_u, \varepsilon_d\) |

## Structural detail

- Left-handed positions: third component forced to `0` in `left_vec` (limits one degree of freedom).
- Coordinates are **integer labels**, not continuous wavefunction samples.
- Generalized envelope \(p \neq 2\): `src/kernel_generalized.py` (16 unit tests passing).

## Not implemented here

- Neutrino PMNS extraction (see `observables.py` partial — quark-focused in base kernel module).
- UV derivation of \(\sigma, k, \alpha, \eta\) from QFT overlaps.

See wiki: [[kernel-implementation]], [[interference-kernel]], [[yukawa-observables-pipeline]].
