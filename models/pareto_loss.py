"""
Mathematical formulations for Multi-Objective Reinforcement Learning:
- Weighted Sum Scalarization
- Augmented Chebyshev Scalarization (can locate non-convex Pareto optimal points)
- Pareto Dominance checking
- 5-D Hypervolume Indicator approximation
"""
import numpy as np
import torch
from typing import List, Tuple

def weighted_sum_scalarization(
    rewards_vec: torch.Tensor,
    weights: torch.Tensor
) -> torch.Tensor:
    """
    Computes linear scalarization: S = sum(w_i * r_i).
    rewards_vec: [Batch, 5]
    weights: [Batch, 5] or [5]
    """
    if weights.dim() == 1:
        weights = weights.unsqueeze(0)
    return torch.sum(rewards_vec * weights, dim=-1)

def chebyshev_scalarization(
    rewards_vec: torch.Tensor,
    weights: torch.Tensor,
    ideal_point: torch.Tensor = None,
    rho: float = 0.05
) -> torch.Tensor:
    """
    Computes Augmented Chebyshev scalarization:
    S(r, w) = - [ max_i (w_i * (z^*_i - r_i)) + rho * sum_i (w_i * (z^*_i - r_i)) ]
    Note: Formulated so that maximizing S maximizes the multi-objective reward.
    """
    if ideal_point is None:
        ideal_point = torch.ones_like(rewards_vec)
    if weights.dim() == 1:
        weights = weights.unsqueeze(0)
    
    diff = ideal_point - rewards_vec  # (z^* - r) >= 0
    weighted_diff = weights * diff
    
    # Max component
    max_term, _ = torch.max(weighted_diff, dim=-1)
    # Augmented linear term for Pareto optimality guarantees
    sum_term = torch.sum(weighted_diff, dim=-1)
    
    # Negate because we want to maximize objective rewards
    return -(max_term + rho * sum_term)

def is_pareto_dominant(candidate: np.ndarray, baseline: np.ndarray) -> bool:
    """
    Checks if candidate vector strictly Pareto-dominates baseline vector.
    True if candidate >= baseline in all 5 objectives and > in at least one.
    """
    ge = np.all(candidate >= (baseline - 1e-6))
    gt = np.any(candidate > (baseline + 1e-6))
    return bool(ge and gt)

def compute_hypervolume(points: List[np.ndarray], ref_point: np.ndarray = None, num_samples: int = 5000) -> float:
    """
    Monte Carlo approximation of the 5-Dimensional Hypervolume Indicator.
    Evaluates the total volume in [0, 1]^5 dominated by the Pareto front.
    Higher hypervolume represents superior multi-objective coverage and optimality.
    """
    if len(points) == 0:
        return 0.0
    
    points_arr = np.array(points)  # [N, 5]
    if ref_point is None:
        ref_point = np.zeros(5)
    
    # Clip points to valid range [0, 1]
    points_arr = np.clip(points_arr, 0.0, 1.0)
    
    # Generate random uniform samples in [ref_point, [1,1,1,1,1]]
    samples = np.random.uniform(0.0, 1.0, size=(num_samples, 5))
    
    # Count samples dominated by at least one point in points_arr
    # A sample S is dominated by point P if P >= S in all 5 coordinates
    dominated_count = 0
    for s in samples:
        # Check if any point >= sample
        if np.any(np.all(points_arr >= s, axis=1)):
            dominated_count += 1
            
    hv = dominated_count / float(num_samples)
    return round(float(hv), 4)
