"""
Optimization Wrapper for Differential Evolution

Standard scipy differential_evolution with project-specific defaults.
"""

from scipy.optimize import differential_evolution
from typing import Callable, List, Tuple, Dict
import numpy as np


def optimize_parameters(
    objective_function: Callable,
    bounds: List[Tuple[float, float]],
    maxiter: int = 100,
    seed: int = 0,
    polish: bool = False,
    workers: int = 1
) -> Dict:
    """
    Run differential evolution optimization.
    
    Args:
        objective_function: Function to minimize
        bounds: List of (min, max) tuples for each parameter
        maxiter: Maximum iterations
        seed: Random seed for reproducibility
        polish: Whether to polish solution with L-BFGS-B
        workers: Number of parallel workers
    
    Returns:
        Dictionary with 'x' (solution), 'fun' (final loss), 'success', 'nit'
    """
    result = differential_evolution(
        objective_function,
        bounds,
        maxiter=maxiter,
        seed=seed,
        polish=polish,
        workers=workers,
        atol=1e-6,
        tol=1e-6,
    )
    
    return {
        'x': result.x,
        'fun': result.fun,
        'success': result.success,
        'nit': result.nit,
        'message': result.message
    }
