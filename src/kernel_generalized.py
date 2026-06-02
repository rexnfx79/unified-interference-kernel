"""
Generalized Interference Kernel

Extends the base kernel with a shape parameter p that interpolates between:
- p = 1: Exponential envelope exp(-|d|/σ)
- p = 2: Gaussian envelope exp(-d²/(2σ²))
- p > 2: Super-Gaussian (sharper cutoff)

Mathematical form:
    Y_ij = exp(-(|d|/σ)^p / p) × [1 + ε exp(iΦ)]

Note: The 1/p normalization ensures consistent behavior at d=σ across all p.

This allows testing whether the Pareto knee is robust to envelope choice
or is an artifact of the Gaussian form.
"""

import numpy as np
from typing import Tuple, Optional


def compute_generalized_envelope(
    d: float,
    sigma: float,
    p: float = 2.0
) -> float:
    """
    Compute generalized envelope function.
    
    Args:
        d: Distance (can be negative, we use |d|)
        sigma: Scale parameter
        p: Shape parameter (1=exponential, 2=Gaussian)
    
    Returns:
        Envelope value in [0, 1]
    
    Raises:
        ValueError: If sigma <= 0 or p <= 0
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if p <= 0:
        raise ValueError(f"p must be positive, got {p}")
    
    abs_d = abs(d)
    # Normalized form: exp(-(|d|/σ)^p / p)
    # This ensures envelope(σ) = exp(-1/p) for all p
    exponent = -((abs_d / sigma) ** p) / p
    return np.exp(exponent)


def compute_kernel_element_generalized(
    x_left: float,
    x_right: float,
    sigma: float,
    k: float,
    alpha: float,
    eta: float,
    eps: float,
    p: float = 2.0
) -> complex:
    """
    Compute a single generalized kernel element Y_ij.
    
    Args:
        x_left: Left-handed fermion position
        x_right: Right-handed fermion position
        sigma: Envelope scale
        k: Phase velocity
        alpha: Phase offset
        eta: Differential phase
        eps: Interference strength
        p: Envelope shape (1=exponential, 2=Gaussian)
    
    Returns:
        Complex kernel value
    """
    diff = x_left - x_right
    
    # Generalized envelope
    envelope = compute_generalized_envelope(diff, sigma, p)
    
    # Phase structure (unchanged from base kernel)
    phase = alpha + k * (x_left + x_right) / 2 + eta * diff
    
    # Interference term
    interference = 1 + eps * np.exp(1j * phase)
    
    return envelope * interference


def compute_yukawa_matrix_generalized(
    left_positions: Tuple[int, int, int],
    right_positions: Tuple[int, int, int],
    sigma: float,
    k: float,
    alpha: float,
    eta: float,
    eps: float,
    p: float = 2.0
) -> np.ndarray:
    """
    Compute full 3x3 Yukawa matrix with generalized envelope.
    
    Args:
        left_positions: Left-handed fermion positions (3 values)
        right_positions: Right-handed fermion positions (3 values)
        sigma, k, alpha, eta, eps: Kernel parameters
        p: Envelope shape parameter
    
    Returns:
        3x3 complex Yukawa matrix
    """
    # Convert positions to arrays
    left_vec = np.array([left_positions[0], left_positions[1], 0], dtype=float)
    right_vec = np.array(right_positions, dtype=float)
    
    # Build matrix
    Y = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            Y[i, j] = compute_kernel_element_generalized(
                left_vec[i], right_vec[j],
                sigma, k, alpha, eta, eps, p
            )
    
    return Y


def compute_quark_yukawas_generalized(
    Q: Tuple[int, int, int],
    U: Tuple[int, int, int],
    D: Tuple[int, int, int],
    sigma: float,
    k: float,
    alpha: float,
    eta: float,
    eps_u: float,
    eps_d: float,
    p: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute up and down Yukawa matrices with generalized envelope.
    
    Args:
        Q: Left-handed quark positions (Q1, Q2, 0)
        U: Right-handed up quark positions
        D: Right-handed down quark positions
        sigma, k, alpha, eta: Universal kernel parameters
        eps_u, eps_d: Sector-specific interference strengths
        p: Envelope shape parameter
    
    Returns:
        Tuple of (Yu, Yd) Yukawa matrices
    """
    Yu = compute_yukawa_matrix_generalized(Q, U, sigma, k, alpha, eta, eps_u, p)
    Yd = compute_yukawa_matrix_generalized(Q, D, sigma, k, alpha, eta, eps_d, p)
    return Yu, Yd


# ============================================================================
# Verification Functions (for QA)
# ============================================================================

def verify_envelope_properties(sigma: float = 1.0, p_values: list = None) -> dict:
    """
    Verify envelope properties for QA.
    
    Returns dict with verification results.
    """
    if p_values is None:
        p_values = [1.0, 1.5, 2.0, 3.0]
    
    results = {
        'at_zero': {},      # envelope(0) should be 1 for all p
        'at_sigma': {},     # envelope(σ) = exp(-1/p)
        'monotonic': {},    # should decrease with |d|
        'positive': {},     # should always be > 0
    }
    
    test_distances = [0.0, sigma/2, sigma, 2*sigma, 5*sigma]
    
    for p in p_values:
        # Test at d=0
        env_zero = compute_generalized_envelope(0.0, sigma, p)
        results['at_zero'][p] = abs(env_zero - 1.0) < 1e-10
        
        # Test at d=sigma
        env_sigma = compute_generalized_envelope(sigma, sigma, p)
        expected = np.exp(-1/p)
        results['at_sigma'][p] = abs(env_sigma - expected) < 1e-10
        
        # Test monotonicity
        values = [compute_generalized_envelope(d, sigma, p) for d in test_distances]
        results['monotonic'][p] = all(values[i] >= values[i+1] for i in range(len(values)-1))
        
        # Test positivity
        results['positive'][p] = all(v > 0 for v in values)
    
    return results


def verify_gaussian_equivalence(sigma: float = 1.5, tol: float = 1e-10) -> bool:
    """
    Verify that p=2 reproduces the standard Gaussian kernel.
    
    The generalized form with p=2 should match:
    exp(-(d/σ)²/2) = exp(-d²/(2σ²))
    """
    from kernel import compute_kernel_element
    
    test_cases = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 2.0),
        (3.0, 1.0),
        (-2.0, 4.0),
    ]
    
    k, alpha, eta, eps = 0.5, 0.3, 2.0, 0.15
    
    for x_left, x_right in test_cases:
        # Standard Gaussian
        y_gaussian = compute_kernel_element(x_left, x_right, sigma, k, alpha, eta, eps)
        
        # Generalized with p=2
        y_generalized = compute_kernel_element_generalized(
            x_left, x_right, sigma, k, alpha, eta, eps, p=2.0
        )
        
        if abs(y_gaussian - y_generalized) > tol:
            return False
    
    return True
