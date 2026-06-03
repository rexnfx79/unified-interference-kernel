# Observables Extraction Snapshot

> **Canonical source:** `../../src/observables.py` (parent repo). Snapshot for wiki ingest.

## Pipeline (quark sector — established in code)

1. Build \(Y_u, Y_d\) from kernel.
2. SVD: \(Y = U \operatorname{diag}(S) V^\dagger\).
3. `fix_svd_phases` — removes column/row phase ambiguities for consistent CKM.
4. CKM: \(V_{\text{CKM}} = U_u^\dagger U_d\) (magnitudes \(V_{us}, V_{cb}, V_{ub}\) extracted).
5. Mass anchoring (PDG 2024 targets in `QUARK_TARGETS`):
   - `scale_u = m_t / S_0`, `scale_d = m_b / S_0`
   - \(m_c = S_1 \cdot \text{scale}_u\), \(m_u = S_2 \cdot \text{scale}_u\), etc.

## Pipeline (neutrino sector — established in code, 2026-06-01)

1. SVD: \(Y_\nu = U_\nu \Sigma_\nu V_\nu^\dagger\), \(Y_e = U_e \Sigma_e V_e^\dagger\).
2. `fix_svd_phases` on \(U_e, U_\nu\).
3. PMNS: \(U_{\text{PMNS}} = U_e^\dagger U_\nu\).
4. Mixing angles via `pmns_angles_from_unitary` (PDG convention).
5. Targets in `NEUTRINO_TARGETS`; loss via `compute_pmns_loss`.

Tests: `tests/test_neutrino_observables.py`.

## Loss functions

| Function | Use |
|----------|-----|
| `compute_ckm_loss` | Relative squared error on \(V_{us}, V_{cb}, V_{ub}\) |
| `compute_mass_loss` | Squared log-ratio on \(\mu, c, d, s\) masses |
| `compute_pmns_loss` | Relative squared error on \(\theta_{12}, \theta_{23}, \theta_{13}\) |
| `compute_training_loss` | Train split: \(m_c, V_{us}, V_{cb}\) only |
| `compute_holdout_loss` | Holdout: \(m_s, m_u, m_d, V_{ub}\) — not used in optimization |
| `compute_penalized_loss` | AIC-like penalty on extra parameters |

## Extended observables

`compute_full_ckm_observables` — all 9 \(|V_{ij}|\), Jarlskog magnitude, unitarity violation metric.

## Information measures (related)

`src/flavor_information.py` — von Neumann entropy \(S(\rho_Y)\) from \(Y Y^\dagger\); see [[information-measure-for-projection-regimes]].

## Known limitation (diagnostics)

`fix_svd_phases` (2026-06-02j): column/row paired phases preserve \(Y=U\Sigma V^\dagger\); prior row-wise bug fixed — see `tests/test_observables.py`.

See wiki: [[observables-extraction]], [[yukawa-observables-pipeline]], [[neutrino-observables-gap]].
