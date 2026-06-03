"""
Alternative Interference Kernels

Implements multiple kernel forms to compare their ability to produce
quark mass hierarchies while maintaining CKM mixing structure.

Kernel Types:
1. Gaussian (original): Y_ij = exp(-d²/(2σ²)) × [1 + ε exp(iΦ)]
2. Power-Law (Froggatt-Nielsen): Y_ij = ε^|d/λ| × [1 + ε_phase exp(iΦ)]
3. Exponential: Y_ij = exp(-|d|/λ) × [1 + ε exp(iΦ)]
4. Clockwork: Y_ij = q^(-|d|) × [1 + ε exp(iΦ)]
5. Hybrid: Y_ij = ε^(d²/λ²) × [1 + ε_phase exp(iΦ)]
6. Clockwork Dual-Phase: Y_ij = q^(-|d|) × [1 + ε1 exp(iΦ) + ε2 exp(i(Φ+φ2+β d))]
7. Clockwork Dual-Phase (u/d split): Y_ij = q^(-|d|) × [1 + ε1 exp(iΦ) + ε2 exp(i(Φ+φ2_{u/d}+β_{u/d} d))]
8. FN Texture (charge-based): Y_ij = ε^(|q_i+f_j|) × [1 + ε_phase exp(iΦ)]
9. FN Texture (u/d split): Y_ij = ε^(|q_i+f_j+offset_{u/d}|) × [1 + ε_phase exp(iΦ)]
"""

import numpy as np
from typing import Tuple, Callable


# =============================================================================
# KERNEL TYPE 1: GAUSSIAN (Original)
# =============================================================================

def gaussian_kernel_element(
    x_left: float,
    x_right: float,
    sigma: float,
    k: float,
    alpha: float,
    eta: float,
    eps: float
) -> complex:
    """
    Original Gaussian kernel: Y_ij = exp(-d²/(2σ²)) × [1 + ε exp(iΦ)]
    
    Problem: Creates smooth matrices that become rank-1 when hierarchical.
    """
    diff = x_left - x_right
    envelope = np.exp(-diff**2 / (2 * sigma**2))
    phase = alpha + k * (x_left + x_right) / 2 + eta * diff
    interference = 1 + eps * np.exp(1j * phase)
    return envelope * interference


# =============================================================================
# KERNEL TYPE 2: POWER-LAW (Froggatt-Nielsen inspired)
# =============================================================================

def power_law_kernel_element(
    x_left: float,
    x_right: float,
    epsilon: float,  # Small parameter ~0.22 (Cabibbo angle)
    lambda_scale: float,  # Distance scale
    k: float,
    alpha: float,
    eta: float,
    eps_phase: float
) -> complex:
    """
    Power-law kernel: Y_ij = ε^|d/λ| × [1 + ε_phase exp(iΦ)]
    
    Inspired by Froggatt-Nielsen mechanism where Yukawa couplings
    scale as powers of a small parameter.
    
    Key property: Can produce large hierarchies (ε^4 ~ 0.002 for ε=0.22)
    without making the matrix rank-1.
    """
    diff = abs(x_left - x_right)
    
    # Power-law envelope: ε^(d/λ)
    # For d=0: envelope = 1
    # For d=λ: envelope = ε
    # For d=2λ: envelope = ε²
    exponent = diff / lambda_scale
    envelope = epsilon ** exponent
    
    # Phase structure (same as original)
    phase = alpha + k * (x_left + x_right) / 2 + eta * (x_left - x_right)
    interference = 1 + eps_phase * np.exp(1j * phase)
    
    return envelope * interference


# =============================================================================
# KERNEL TYPE 3: EXPONENTIAL (Extra-dimension inspired)
# =============================================================================

def exponential_kernel_element(
    x_left: float,
    x_right: float,
    lambda_scale: float,  # Decay length
    k: float,
    alpha: float,
    eta: float,
    eps: float
) -> complex:
    """
    Exponential kernel: Y_ij = exp(-|d|/λ) × [1 + ε exp(iΦ)]
    
    Inspired by wavefunction overlap in extra dimensions.
    Linear decay in exponent (vs quadratic for Gaussian).
    
    Key property: Steeper decay for small distances, slower for large.
    """
    diff = abs(x_left - x_right)
    
    # Exponential envelope: exp(-|d|/λ)
    envelope = np.exp(-diff / lambda_scale)
    
    # Phase structure
    phase = alpha + k * (x_left + x_right) / 2 + eta * (x_left - x_right)
    interference = 1 + eps * np.exp(1j * phase)
    
    return envelope * interference


# =============================================================================
# KERNEL TYPE 4: CLOCKWORK
# =============================================================================

def clockwork_kernel_element(
    x_left: float,
    x_right: float,
    q: float,  # Gear ratio (typically 2-5)
    k: float,
    alpha: float,
    eta: float,
    eps: float
) -> complex:
    """
    Clockwork kernel: Y_ij = q^(-|d|) × [1 + ε exp(iΦ)]
    
    Inspired by clockwork mechanism where exponential hierarchies
    emerge from O(1) gear ratios.
    
    Key property: For q=3, each unit distance gives 3x suppression.
    """
    diff = abs(x_left - x_right)
    
    # Clockwork envelope: q^(-|d|)
    # For d=0: envelope = 1
    # For d=1: envelope = 1/q
    # For d=2: envelope = 1/q²
    envelope = q ** (-diff)
    
    # Phase structure
    phase = alpha + k * (x_left + x_right) / 2 + eta * (x_left - x_right)
    interference = 1 + eps * np.exp(1j * phase)
    
    return envelope * interference


# =============================================================================
# KERNEL TYPE 5: CLOCKWORK DUAL-PHASE
# =============================================================================

def clockwork_dual_phase_kernel_element(
    x_left: float,
    x_right: float,
    q: float,
    k: float,
    alpha: float,
    eta: float,
    eps1: float,
    eps2: float,
    beta: float,
    phi2: float
) -> complex:
    """
    Clockwork dual-phase kernel:
    Y_ij = q^(-|d|) × [1 + ε1 exp(iΦ) + ε2 exp(i(Φ + φ2 + β d))]
    """
    diff = abs(x_left - x_right)
    envelope = q ** (-diff)

    base_phase = alpha + k * (x_left + x_right) / 2 + eta * (x_left - x_right)
    phase_2 = base_phase + phi2 + beta * (x_left - x_right)
    interference = 1 + eps1 * np.exp(1j * base_phase) + eps2 * np.exp(1j * phase_2)

    return envelope * interference


# =============================================================================
# KERNEL TYPE 6: HYBRID (Power-law envelope with quadratic distance)
# =============================================================================

def hybrid_kernel_element(
    x_left: float,
    x_right: float,
    epsilon: float,  # Small parameter
    lambda_scale: float,  # Distance scale
    k: float,
    alpha: float,
    eta: float,
    eps_phase: float
) -> complex:
    """
    Hybrid kernel: Y_ij = ε^(d²/λ²) × [1 + ε_phase exp(iΦ)]
    
    Combines power-law suppression with quadratic distance dependence.
    
    Key property: Smooth like Gaussian but with power-law scaling.
    """
    diff = x_left - x_right
    
    # Hybrid envelope: ε^(d²/λ²)
    exponent = diff**2 / lambda_scale**2
    envelope = epsilon ** exponent
    
    # Phase structure
    phase = alpha + k * (x_left + x_right) / 2 + eta * diff
    interference = 1 + eps_phase * np.exp(1j * phase)
    
    return envelope * interference


# =============================================================================
# KERNEL TYPE 7: FN TEXTURE (Charge-based)
# =============================================================================

def fn_texture_kernel_element(
    x_left: float,
    x_right: float,
    epsilon: float,
    offset: float,
    k: float,
    alpha: float,
    eta: float,
    eps_phase: float
) -> complex:
    """
    FN texture kernel: Y_ij = ε^(|q_i + f_j + offset|) × [1 + ε_phase exp(iΦ)]
    """
    exponent = abs(x_left + x_right + offset)
    envelope = epsilon ** exponent
    phase = alpha + k * (x_left + x_right) / 2 + eta * (x_left - x_right)
    interference = 1 + eps_phase * np.exp(1j * phase)
    return envelope * interference


# =============================================================================
# MATRIX BUILDERS
# =============================================================================

def build_yukawa_matrix(
    kernel_func: Callable,
    left_positions: Tuple[float, float, float],
    right_positions: Tuple[float, float, float],
    **params
) -> np.ndarray:
    """Build 3x3 Yukawa matrix using specified kernel function."""
    # Note: Using all 3 positions (not hardcoding third to 0)
    left_vec = np.array(left_positions, dtype=float)
    right_vec = np.array(right_positions, dtype=float)
    
    Y = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            Y[i, j] = kernel_func(left_vec[i], right_vec[j], **params)
    
    return Y


def compute_yukawas_gaussian(Q, U, D, sigma, k, alpha, eta, eps_u, eps_d):
    """Compute Yukawa matrices using Gaussian kernel."""
    params_u = {'sigma': sigma, 'k': k, 'alpha': alpha, 'eta': eta, 'eps': eps_u}
    params_d = {'sigma': sigma, 'k': k, 'alpha': alpha, 'eta': eta, 'eps': eps_d}
    Yu = build_yukawa_matrix(gaussian_kernel_element, Q, U, **params_u)
    Yd = build_yukawa_matrix(gaussian_kernel_element, Q, D, **params_d)
    return Yu, Yd


def compute_yukawas_power_law(Q, U, D, epsilon, lambda_scale, k, alpha, eta, eps_u, eps_d):
    """Compute Yukawa matrices using power-law kernel."""
    params_u = {'epsilon': epsilon, 'lambda_scale': lambda_scale, 
                'k': k, 'alpha': alpha, 'eta': eta, 'eps_phase': eps_u}
    params_d = {'epsilon': epsilon, 'lambda_scale': lambda_scale,
                'k': k, 'alpha': alpha, 'eta': eta, 'eps_phase': eps_d}
    Yu = build_yukawa_matrix(power_law_kernel_element, Q, U, **params_u)
    Yd = build_yukawa_matrix(power_law_kernel_element, Q, D, **params_d)
    return Yu, Yd


def compute_yukawas_exponential(Q, U, D, lambda_scale, k, alpha, eta, eps_u, eps_d):
    """Compute Yukawa matrices using exponential kernel."""
    params_u = {'lambda_scale': lambda_scale, 'k': k, 'alpha': alpha, 'eta': eta, 'eps': eps_u}
    params_d = {'lambda_scale': lambda_scale, 'k': k, 'alpha': alpha, 'eta': eta, 'eps': eps_d}
    Yu = build_yukawa_matrix(exponential_kernel_element, Q, U, **params_u)
    Yd = build_yukawa_matrix(exponential_kernel_element, Q, D, **params_d)
    return Yu, Yd


def compute_yukawas_clockwork(Q, U, D, q, k, alpha, eta, eps_u, eps_d):
    """Compute Yukawa matrices using clockwork kernel."""
    params_u = {'q': q, 'k': k, 'alpha': alpha, 'eta': eta, 'eps': eps_u}
    params_d = {'q': q, 'k': k, 'alpha': alpha, 'eta': eta, 'eps': eps_d}
    Yu = build_yukawa_matrix(clockwork_kernel_element, Q, U, **params_u)
    Yd = build_yukawa_matrix(clockwork_kernel_element, Q, D, **params_d)
    return Yu, Yd


def compute_yukawas_clockwork_dual_phase(
    Q, U, D,
    q, k, alpha, eta,
    eps1_u, eps1_d,
    eps2_u, eps2_d,
    beta, phi2
):
    """Compute Yukawa matrices using clockwork dual-phase kernel."""
    params_u = {
        'q': q, 'k': k, 'alpha': alpha, 'eta': eta,
        'eps1': eps1_u, 'eps2': eps2_u, 'beta': beta, 'phi2': phi2,
    }
    params_d = {
        'q': q, 'k': k, 'alpha': alpha, 'eta': eta,
        'eps1': eps1_d, 'eps2': eps2_d, 'beta': beta, 'phi2': phi2,
    }
    Yu = build_yukawa_matrix(clockwork_dual_phase_kernel_element, Q, U, **params_u)
    Yd = build_yukawa_matrix(clockwork_dual_phase_kernel_element, Q, D, **params_d)
    return Yu, Yd


def compute_yukawas_clockwork_dual_phase_split(
    Q, U, D,
    q, k, alpha, eta,
    eps1_u, eps1_d,
    eps2_u, eps2_d,
    beta_u, beta_d,
    phi2_u, phi2_d
):
    """Compute Yukawa matrices with u/d-specific dual-phase parameters."""
    params_u = {
        'q': q, 'k': k, 'alpha': alpha, 'eta': eta,
        'eps1': eps1_u, 'eps2': eps2_u, 'beta': beta_u, 'phi2': phi2_u,
    }
    params_d = {
        'q': q, 'k': k, 'alpha': alpha, 'eta': eta,
        'eps1': eps1_d, 'eps2': eps2_d, 'beta': beta_d, 'phi2': phi2_d,
    }
    Yu = build_yukawa_matrix(clockwork_dual_phase_kernel_element, Q, U, **params_u)
    Yd = build_yukawa_matrix(clockwork_dual_phase_kernel_element, Q, D, **params_d)
    return Yu, Yd


def compute_yukawas_hybrid(Q, U, D, epsilon, lambda_scale, k, alpha, eta, eps_u, eps_d):
    """Compute Yukawa matrices using hybrid kernel."""
    params_u = {'epsilon': epsilon, 'lambda_scale': lambda_scale,
                'k': k, 'alpha': alpha, 'eta': eta, 'eps_phase': eps_u}
    params_d = {'epsilon': epsilon, 'lambda_scale': lambda_scale,
                'k': k, 'alpha': alpha, 'eta': eta, 'eps_phase': eps_d}
    Yu = build_yukawa_matrix(hybrid_kernel_element, Q, U, **params_u)
    Yd = build_yukawa_matrix(hybrid_kernel_element, Q, D, **params_d)
    return Yu, Yd


def compute_yukawas_fn_texture(Q, U, D, epsilon, offset, k, alpha, eta, eps_u, eps_d):
    """Compute Yukawa matrices using FN texture kernel."""
    params_u = {
        'epsilon': epsilon, 'offset': offset,
        'k': k, 'alpha': alpha, 'eta': eta, 'eps_phase': eps_u,
    }
    params_d = {
        'epsilon': epsilon, 'offset': offset,
        'k': k, 'alpha': alpha, 'eta': eta, 'eps_phase': eps_d,
    }
    Yu = build_yukawa_matrix(fn_texture_kernel_element, Q, U, **params_u)
    Yd = build_yukawa_matrix(fn_texture_kernel_element, Q, D, **params_d)
    return Yu, Yd


def compute_yukawas_rank2_clockwork_sum(
    Q,
    U,
    D,
    q1,
    k1,
    alpha1,
    eta1,
    eps1_u,
    eps1_d,
    q2,
    k2,
    alpha2,
    eta2,
    eps2_u,
    eps2_d,
    blend,
):
    """
    Rank-2 style ansatz: Y = w Y^(1) + (1-w) Y^(2) with independent clockwork layers.

    Each layer is rank-1 in overlap form; the sum can raise effective rank.
    """
    Yu1, Yd1 = compute_yukawas_clockwork(Q, U, D, q1, k1, alpha1, eta1, eps1_u, eps1_d)
    Yu2, Yd2 = compute_yukawas_clockwork(Q, U, D, q2, k2, alpha2, eta2, eps2_u, eps2_d)
    w = float(np.clip(blend, 0.0, 1.0))
    Yu = w * Yu1 + (1.0 - w) * Yu2
    Yd = w * Yd1 + (1.0 - w) * Yd2
    return Yu, Yd


def compute_yukawas_fn_texture_split(
    Q, U, D,
    epsilon, offset_u, offset_d,
    k, alpha, eta, eps_u, eps_d
):
    """Compute Yukawa matrices using FN texture kernel with u/d offsets."""
    params_u = {
        'epsilon': epsilon, 'offset': offset_u,
        'k': k, 'alpha': alpha, 'eta': eta, 'eps_phase': eps_u,
    }
    params_d = {
        'epsilon': epsilon, 'offset': offset_d,
        'k': k, 'alpha': alpha, 'eta': eta, 'eps_phase': eps_d,
    }
    Yu = build_yukawa_matrix(fn_texture_kernel_element, Q, U, **params_u)
    Yd = build_yukawa_matrix(fn_texture_kernel_element, Q, D, **params_d)
    return Yu, Yd


# =============================================================================
# PHYSICS-FAITHFUL RELAXATIONS (Minimality Ladder)
# =============================================================================

def compute_yukawas_with_shift(
    Q, U, D,
    delta_H: float,
    kernel_func: Callable,
    kernel_params: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Physics-faithful relaxation: same Q positions at field level, but effective
    overlap differs due to Higgs localization.
    
    Physical motivation:
    - In Type-II 2HDM, H_u and H_d have different bulk profiles
    - In warped extra dimensions, different Higgs localizations induce
      effective shifts in where the doublet "samples" the Higgs
    
    Distance maps:
        d_u(Q_i, U_j) = |Q_i - U_j|               (up-type, H_u reference)
        d_d(Q_i, D_j) = |(Q_i + delta_H) - D_j|   (down-type, H_d shifted)
    
    This preserves SU(2)_L gauge structure (one Q field) while allowing
    operator-dependent effective overlaps.
    
    Parameters:
        Q: Left-handed doublet positions (shared)
        U: Up-type right-handed positions
        D: Down-type right-handed positions
        delta_H: Higgs profile shift (single parameter)
        kernel_func: Kernel element function
        kernel_params: Parameters for kernel (excluding positions)
    
    Returns:
        Yu, Yd: Up-type and down-type Yukawa matrices
    """
    # Q positions are shared at field level
    Q_eff_u = Q
    Q_eff_d = tuple(q + delta_H for q in Q)
    
    Yu = build_yukawa_matrix(kernel_func, Q_eff_u, U, **kernel_params)
    Yd = build_yukawa_matrix(kernel_func, Q_eff_d, D, **kernel_params)
    return Yu, Yd


def compute_yukawas_clockwork_shifted(Q, U, D, q, k, alpha, eta, eps_u, eps_d, delta_H):
    """
    Clockwork kernel with Higgs-localization shift.
    
    Level 1 in minimality ladder: +1 parameter (delta_H).
    """
    params_u = {'q': q, 'k': k, 'alpha': alpha, 'eta': eta, 'eps': eps_u}
    params_d = {'q': q, 'k': k, 'alpha': alpha, 'eta': eta, 'eps': eps_d}
    
    Q_eff_d = tuple(qi + delta_H for qi in Q)
    
    Yu = build_yukawa_matrix(clockwork_kernel_element, Q, U, **params_u)
    Yd = build_yukawa_matrix(clockwork_kernel_element, Q_eff_d, D, **params_d)
    return Yu, Yd


def compute_yukawas_clockwork_width_split(Q, U, D, q_u, q_d, k, alpha, eta, eps_u, eps_d):
    """
    Clockwork kernel with sector-specific gear ratios.
    
    Level 2 in minimality ladder: +1 parameter (q_u != q_d).
    
    Physical motivation: Different bulk mass parameters for up vs down
    Yukawa operators.
    """
    params_u = {'q': q_u, 'k': k, 'alpha': alpha, 'eta': eta, 'eps': eps_u}
    params_d = {'q': q_d, 'k': k, 'alpha': alpha, 'eta': eta, 'eps': eps_d}
    
    Yu = build_yukawa_matrix(clockwork_kernel_element, Q, U, **params_u)
    Yd = build_yukawa_matrix(clockwork_kernel_element, Q, D, **params_d)
    return Yu, Yd


def compute_yukawas_clockwork_both(Q, U, D, q_u, q_d, k, alpha, eta, eps_u, eps_d, delta_H):
    """
    Clockwork kernel with both shift and width split.
    
    Level 3 in minimality ladder: +2 parameters (delta_H + q split).
    """
    params_u = {'q': q_u, 'k': k, 'alpha': alpha, 'eta': eta, 'eps': eps_u}
    params_d = {'q': q_d, 'k': k, 'alpha': alpha, 'eta': eta, 'eps': eps_d}
    
    Q_eff_d = tuple(qi + delta_H for qi in Q)
    
    Yu = build_yukawa_matrix(clockwork_kernel_element, Q, U, **params_u)
    Yd = build_yukawa_matrix(clockwork_kernel_element, Q_eff_d, D, **params_d)
    return Yu, Yd


def compute_yukawas_clockwork_full_split(Q_u, Q_d, U, D, q, k, alpha, eta, eps_u, eps_d):
    """
    Clockwork kernel with fully independent Q positions.
    
    Level 4 in minimality ladder: +3 parameters (independent Q_u, Q_d).
    
    WARNING: This violates SU(2)_L intuition unless justified by
    specific UV physics (e.g., composite Higgs with different form factors).
    """
    params_u = {'q': q, 'k': k, 'alpha': alpha, 'eta': eta, 'eps': eps_u}
    params_d = {'q': q, 'k': k, 'alpha': alpha, 'eta': eta, 'eps': eps_d}
    
    Yu = build_yukawa_matrix(clockwork_kernel_element, Q_u, U, **params_u)
    Yd = build_yukawa_matrix(clockwork_kernel_element, Q_d, D, **params_d)
    return Yu, Yd


# =============================================================================
# KERNEL REGISTRY
# =============================================================================

KERNELS = {
    'gaussian': {
        'name': 'Gaussian',
        'formula': 'Y_ij = exp(-d²/(2σ²)) × [1 + ε exp(iΦ)]',
        'compute_yukawas': compute_yukawas_gaussian,
        'params': ['sigma', 'k', 'alpha', 'eta', 'eps_u', 'eps_d'],
        'bounds': [(0.1, 20.0), (0.01, 5.0), (0.0, 2*np.pi), (0.01, 10.0), (0.01, 1.5), (0.01, 1.5)],
    },
    'power_law': {
        'name': 'Geometric Exponential (Cabibbo-base)',
        'formula': 'Y_ij = ε^|d/λ| × [1 + ε_phase exp(iΦ)]',
        'compute_yukawas': compute_yukawas_power_law,
        'params': ['epsilon', 'lambda_scale', 'k', 'alpha', 'eta', 'eps_u', 'eps_d'],
        'bounds': [(0.1, 0.5), (0.5, 5.0), (0.01, 5.0), (0.0, 2*np.pi), (0.01, 10.0), (0.01, 1.5), (0.01, 1.5)],
    },
    'exponential': {
        'name': 'Exponential',
        'formula': 'Y_ij = exp(-|d|/λ) × [1 + ε exp(iΦ)]',
        'compute_yukawas': compute_yukawas_exponential,
        'params': ['lambda_scale', 'k', 'alpha', 'eta', 'eps_u', 'eps_d'],
        'bounds': [(0.1, 10.0), (0.01, 5.0), (0.0, 2*np.pi), (0.01, 10.0), (0.01, 1.5), (0.01, 1.5)],
    },
    'clockwork': {
        'name': 'Clockwork',
        'formula': 'Y_ij = q^(-|d|) × [1 + ε exp(iΦ)]',
        'compute_yukawas': compute_yukawas_clockwork,
        'params': ['q', 'k', 'alpha', 'eta', 'eps_u', 'eps_d'],
        'bounds': [(1.5, 10.0), (0.01, 5.0), (0.0, 2*np.pi), (0.01, 10.0), (0.01, 1.5), (0.01, 1.5)],
    },
    'clockwork_dual_phase': {
        'name': 'Clockwork Dual-Phase',
        'formula': 'Y_ij = q^(-|d|) × [1 + ε1 exp(iΦ) + ε2 exp(i(Φ+φ2+β d))]',
        'compute_yukawas': compute_yukawas_clockwork_dual_phase,
        'params': ['q', 'k', 'alpha', 'eta', 'eps1_u', 'eps1_d', 'eps2_u', 'eps2_d', 'beta', 'phi2'],
        'bounds': [
            (1.5, 10.0),
            (0.01, 5.0),
            (0.0, 2*np.pi),
            (0.01, 10.0),
            (0.01, 1.5),
            (0.01, 1.5),
            (0.01, 1.5),
            (0.01, 1.5),
            (0.0, 5.0),
            (0.0, 2*np.pi),
        ],
    },
    'clockwork_dual_phase_split': {
        'name': 'Clockwork Dual-Phase (u/d split)',
        'formula': 'Y_ij = q^(-|d|) × [1 + ε1 exp(iΦ) + ε2 exp(i(Φ+φ2_{u/d}+β_{u/d} d))]',
        'compute_yukawas': compute_yukawas_clockwork_dual_phase_split,
        'params': [
            'q', 'k', 'alpha', 'eta',
            'eps1_u', 'eps1_d', 'eps2_u', 'eps2_d',
            'beta_u', 'beta_d', 'phi2_u', 'phi2_d',
        ],
        'bounds': [
            (1.5, 10.0),
            (0.01, 5.0),
            (0.0, 2*np.pi),
            (0.01, 10.0),
            (0.01, 1.5),
            (0.01, 1.5),
            (0.01, 1.5),
            (0.01, 1.5),
            (0.0, 5.0),
            (0.0, 5.0),
            (0.0, 2*np.pi),
            (0.0, 2*np.pi),
        ],
    },
    'hybrid': {
        'name': 'Hybrid (Power-Law + Quadratic)',
        'formula': 'Y_ij = ε^(d²/λ²) × [1 + ε_phase exp(iΦ)]',
        'compute_yukawas': compute_yukawas_hybrid,
        'params': ['epsilon', 'lambda_scale', 'k', 'alpha', 'eta', 'eps_u', 'eps_d'],
        'bounds': [(0.1, 0.9), (0.5, 5.0), (0.01, 5.0), (0.0, 2*np.pi), (0.01, 10.0), (0.01, 1.5), (0.01, 1.5)],
    },
    'fn_texture': {
        'name': 'FN Texture (charge-based)',
        'formula': 'Y_ij = ε^(|q_i+f_j+offset|) × [1 + ε_phase exp(iΦ)]',
        'compute_yukawas': compute_yukawas_fn_texture,
        'params': ['epsilon', 'offset', 'k', 'alpha', 'eta', 'eps_u', 'eps_d'],
        'bounds': [(0.1, 0.6), (-6.0, 6.0), (0.01, 5.0), (0.0, 2*np.pi), (0.01, 10.0), (0.01, 1.5), (0.01, 1.5)],
    },
    'fn_texture_split': {
        'name': 'FN Texture (u/d split)',
        'formula': 'Y_ij = ε^(|q_i+f_j+offset_{u/d}|) × [1 + ε_phase exp(iΦ)]',
        'compute_yukawas': compute_yukawas_fn_texture_split,
        'params': ['epsilon', 'offset_u', 'offset_d', 'k', 'alpha', 'eta', 'eps_u', 'eps_d'],
        'bounds': [
            (0.1, 0.6),
            (-6.0, 6.0),
            (-6.0, 6.0),
            (0.01, 5.0),
            (0.0, 2*np.pi),
            (0.01, 10.0),
            (0.01, 1.5),
            (0.01, 1.5),
        ],
    },
    'rank2_clockwork_sum': {
        'name': 'Rank-2 clockwork sum',
        'formula': 'Y = w Y^(q1) + (1-w) Y^(q2) (two clockwork layers)',
        'compute_yukawas': compute_yukawas_rank2_clockwork_sum,
        'params': [
            'q1', 'k1', 'alpha1', 'eta1', 'eps1_u', 'eps1_d',
            'q2', 'k2', 'alpha2', 'eta2', 'eps2_u', 'eps2_d',
            'blend',
        ],
        'bounds': [
            (1.5, 8.0), (0.01, 3.0), (0.0, 2 * np.pi), (0.5, 6.0), (0.01, 1.0), (0.01, 1.0),
            (1.5, 8.0), (0.01, 3.0), (0.0, 2 * np.pi), (0.5, 6.0), (0.01, 1.0), (0.01, 1.0),
            (0.05, 0.95),
        ],
    },
}

# Tier 2 pre-registered comparison set (diagnostic 32)
TIER2_QUARK_KERNELS = [
    'gaussian',
    'rank2_clockwork_sum',
    'clockwork_dual_phase',
    'fn_texture',
    'fn_texture_split',
    'power_law',
]

# =============================================================================
# MINIMALITY LADDER REGISTRY
# =============================================================================

MINIMALITY_LADDER = {
    0: {
        'name': 'Base (shared Q, shared params)',
        'extra_params': 0,
        'compute_yukawas': compute_yukawas_clockwork,
        'params': ['q', 'k', 'alpha', 'eta', 'eps_u', 'eps_d'],
        'bounds': [(1.5, 15.0), (0.001, 10.0), (0.0, 2*np.pi), (0.001, 15.0), (0.01, 2.0), (0.01, 2.0)],
        'description': 'Original model with shared Q positions and kernel parameters',
    },
    1: {
        'name': 'Shift (delta_H for down-type)',
        'extra_params': 1,
        'compute_yukawas': compute_yukawas_clockwork_shifted,
        'params': ['q', 'k', 'alpha', 'eta', 'eps_u', 'eps_d', 'delta_H'],
        'bounds': [(1.5, 15.0), (0.001, 10.0), (0.0, 2*np.pi), (0.001, 15.0), (0.01, 2.0), (0.01, 2.0), (-5.0, 5.0)],
        'description': 'Higgs-localization-induced shift for down-type Yukawa',
    },
    2: {
        'name': 'Width (q_u != q_d)',
        'extra_params': 1,
        'compute_yukawas': compute_yukawas_clockwork_width_split,
        'params': ['q_u', 'q_d', 'k', 'alpha', 'eta', 'eps_u', 'eps_d'],
        'bounds': [(1.5, 15.0), (1.5, 15.0), (0.001, 10.0), (0.0, 2*np.pi), (0.001, 15.0), (0.01, 2.0), (0.01, 2.0)],
        'description': 'Sector-specific gear ratios (different bulk masses)',
    },
    3: {
        'name': 'Both (delta_H + q split)',
        'extra_params': 2,
        'compute_yukawas': compute_yukawas_clockwork_both,
        'params': ['q_u', 'q_d', 'k', 'alpha', 'eta', 'eps_u', 'eps_d', 'delta_H'],
        'bounds': [(1.5, 15.0), (1.5, 15.0), (0.001, 10.0), (0.0, 2*np.pi), (0.001, 15.0), (0.01, 2.0), (0.01, 2.0), (-5.0, 5.0)],
        'description': 'Combination of shift and width split',
    },
    4: {
        'name': 'Full (independent Q_u, Q_d)',
        'extra_params': 3,
        'compute_yukawas': compute_yukawas_clockwork_full_split,
        'params': ['q', 'k', 'alpha', 'eta', 'eps_u', 'eps_d'],
        'bounds': [(1.5, 15.0), (0.001, 10.0), (0.0, 2*np.pi), (0.001, 15.0), (0.01, 2.0), (0.01, 2.0)],
        'description': 'Fully independent Q positions (WARNING: violates SU(2)_L intuition)',
        'note': 'Requires separate Q_u, Q_d geometry inputs',
    },
}
