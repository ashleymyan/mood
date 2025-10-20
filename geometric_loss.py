"""
Riemann Curvature Loss Functions

This module provides geometric loss functions for neural network training, including:
- Riemann curvature loss based on metric tensor variations
- Axis alignment loss for encouraging orthogonal feature spaces
- Repulsion loss for preventing point collapse
- Boundary loss for domain constraint enforcement

These losses are particularly useful for training neural networks to learn
well-structured embedding spaces with desirable geometric properties.
"""

from typing import Optional, Tuple

import torch
import numpy as np
from scipy.spatial import Delaunay


# ===== Utility Functions =====

def reduce_dimensionality_pca(points: torch.Tensor, target_dim: int = 2) -> torch.Tensor:
    """
    Reduce point dimensionality using Principal Component Analysis (PCA).
    
    Args:
        points (torch.Tensor): Input points of shape (N, D)
        target_dim (int): Target dimensionality for reduction
        
    Returns:
        torch.Tensor: Reduced points of shape (N, target_dim)
    """
    if points.shape[1] <= target_dim:
        return points
    
    # Perform SVD for PCA
    u, s, v = torch.svd(points)
    
    # Project onto top target_dim principal components
    return points @ v[:, :target_dim]


@torch.no_grad()
def compute_delaunay_triangulation(points: torch.Tensor) -> np.ndarray:
    """
    Compute Delaunay triangulation for a set of points.
    
    For high-dimensional points, PCA is first applied to reduce to 2D
    before computing the triangulation.
    
    Args:
        points (torch.Tensor): Input points of shape (N, D)
        
    Returns:
        np.ndarray: Simplex indices of shape (M, D+1) where M is number of simplices
        
    Raises:
        ValueError: If points have insufficient samples for triangulation
    """
    if points.shape[0] < 3:
        raise ValueError(f"Need at least 3 points for triangulation, got {points.shape[0]}")
    
    # Reduce to 2D for triangulation if needed
    if points.shape[1] > 2:
        points_2d = reduce_dimensionality_pca(points, target_dim=2)
    else:
        points_2d = points
    
    # Convert to numpy and compute triangulation
    points_numpy = points_2d.cpu().numpy()
    
    try:
        triangulation = Delaunay(points_numpy)
        return triangulation.simplices
    except Exception as e:
        raise RuntimeError(f"Delaunay triangulation failed: {str(e)}") from e


# ===== Geometric Loss Functions =====

def compute_riemann_curvature_loss(points: torch.Tensor, 
                                 simplices: Optional[np.ndarray] = None,
                                 domain_min: float = 0.0, 
                                 domain_max: float = 1.0) -> torch.Tensor:
    """
    Compute loss based on approximated Riemann curvature.
    
    This function measures deviations from uniform metric tensors across simplices,
    which approximates variations in Riemann curvature. The goal is to encourage
    a uniform geometric structure in the embedding space.
    
    The loss is computed by:
    1. Creating a Delaunay triangulation of the points
    2. Computing metric tensors (Gram matrices) for each simplex
    3. Measuring deviations of determinants from a target value
    
    Args:
        points (torch.Tensor): Input points of shape (N, D)
        simplices (Optional[np.ndarray]): Pre-computed simplex indices. 
                                        If None, computed via Delaunay triangulation
        domain_min (float): Minimum domain value (unused in current implementation)
        domain_max (float): Maximum domain value (unused in current implementation)
        
    Returns:
        torch.Tensor: Scalar curvature loss value
        
    Raises:
        ValueError: If input points are invalid
        RuntimeError: If triangulation or curvature computation fails
    """
    # Input validation
    if points.numel() == 0:
        raise ValueError("Input points cannot be empty")
    
    if points.dim() != 2:
        raise ValueError(f"Expected 2D points tensor (N, D), got shape {points.shape}")
    
    if points.shape[0] < 3:
        return torch.tensor(0.0, device=points.device, dtype=points.dtype)
    
    # Compute triangulation if not provided
    if simplices is None:
        try:
            simplices = compute_delaunay_triangulation(points)
        except (ValueError, RuntimeError):
            # Fallback: return zero loss if triangulation fails
            return torch.tensor(0.0, device=points.device, dtype=points.dtype)
    
    if len(simplices) == 0:
        return torch.tensor(0.0, device=points.device, dtype=points.dtype)
    
    # Target determinant value (uniform metric)
    ideal_determinant = torch.tensor(1.0, device=points.device, dtype=torch.float64)
    
    # Convert simplices to tensor for batch processing
    simplices_tensor = torch.tensor(simplices, device=points.device, dtype=torch.long)
    
    # Extract points that form each simplex
    simplex_points = points[simplices_tensor]  # Shape: (M, K, D)
    
    # Calculate edge vectors from the first vertex of each simplex
    edges = simplex_points[:, 1:] - simplex_points[:, 0].unsqueeze(1)  # Shape: (M, K-1, D)
    
    # Compute metric tensors (Gram matrices) for each simplex
    metric_tensors = torch.matmul(edges, edges.transpose(1, 2))  # Shape: (M, K-1, K-1)
    
    # Calculate determinants (measures volume distortion)
    try:
        determinants = torch.linalg.det(metric_tensors)
    except RuntimeError:
        # Fallback for numerical issues
        return torch.tensor(0.0, device=points.device, dtype=points.dtype)
    
    # Filter out non-positive determinants (degenerate simplices)
    valid_determinants = determinants[determinants > 0]
    
    if len(valid_determinants) == 0:
        return torch.tensor(0.0, device=points.device, dtype=points.dtype)
    
    # Penalize deviations from uniform determinant
    curvature_loss = torch.mean((valid_determinants.double() - ideal_determinant) ** 2)
    
    return curvature_loss.float()


def compute_axis_alignment_loss(data: torch.Tensor) -> torch.Tensor:
    """
    Encourage axis alignment by minimizing off-diagonal elements in the covariance matrix.
    
    This loss function promotes orthogonal feature representations by penalizing
    correlations between different dimensions. The goal is to make the covariance
    matrix as close to the identity matrix as possible.
    
    Args:
        data (torch.Tensor): Input data of shape (N, D)
        
    Returns:
        torch.Tensor: Scalar axis alignment loss
        
    Raises:
        ValueError: If input data is invalid
    """
    if data.numel() == 0:
        raise ValueError("Input data cannot be empty")
    
    if data.dim() != 2:
        raise ValueError(f"Expected 2D data tensor (N, D), got shape {data.shape}")
    
    n_samples, n_features = data.shape
    
    if n_samples < 2:
        return torch.tensor(0.0, device=data.device, dtype=data.dtype)
    
    # Center the data (remove mean)
    centered_data = data - data.mean(dim=0, keepdim=True)
    
    # Compute empirical covariance matrix
    covariance_matrix = (centered_data.T @ centered_data) / n_samples
    
    # Target: identity matrix (orthogonal features)
    identity_matrix = torch.eye(n_features, device=data.device, dtype=data.dtype)
    
    # Smooth L1 loss between covariance matrix and identity
    alignment_loss = torch.nn.functional.smooth_l1_loss(covariance_matrix, identity_matrix)
    
    return alignment_loss


def compute_repulsion_loss(points: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Compute repulsion loss to prevent point collapse.
    
    This loss encourages points to maintain some minimum distance from each other,
    preventing the optimization from collapsing all points to the same location.
    
    Args:
        points (torch.Tensor): Input points of shape (N, D)
        epsilon (float): Small value to prevent division by zero
        
    Returns:
        torch.Tensor: Scalar repulsion loss
        
    Raises:
        ValueError: If input points are invalid
    """
    if points.numel() == 0:
        raise ValueError("Input points cannot be empty")
    
    if points.dim() != 2:
        raise ValueError(f"Expected 2D points tensor (N, D), got shape {points.shape}")
    
    n_points = points.shape[0]
    
    if n_points < 2:
        return torch.tensor(0.0, device=points.device, dtype=points.dtype)
    
    # Compute pairwise distances
    distance_matrix = torch.cdist(points, points)
    
    # Create mask to exclude diagonal elements (self-distances)
    diagonal_mask = torch.eye(n_points, device=points.device, dtype=torch.bool)
    
    # Set diagonal to large value to avoid self-repulsion
    distance_matrix = distance_matrix + diagonal_mask.float() * 1000.0
    
    # Compute repulsion forces (inverse distance)
    repulsion_forces = 1.0 / (distance_matrix + epsilon)
    
    # Only consider off-diagonal elements
    off_diagonal_forces = repulsion_forces[~diagonal_mask]
    
    # Return mean repulsion loss
    repulsion_loss = torch.mean(off_diagonal_forces)
    
    return repulsion_loss


def compute_boundary_loss(points: torch.Tensor, 
                         domain_min: float = -1.0, 
                         domain_max: float = 1.0) -> torch.Tensor:
    """
    Compute boundary loss to keep points within a specified domain.
    
    This loss penalizes points that fall outside the specified domain boundaries,
    encouraging the optimization to keep all points within the valid region.
    
    Args:
        points (torch.Tensor): Input points of shape (N, D)
        domain_min (float): Minimum allowed value for any coordinate
        domain_max (float): Maximum allowed value for any coordinate
        
    Returns:
        torch.Tensor: Scalar boundary loss
        
    Raises:
        ValueError: If input points are invalid or domain bounds are invalid
    """
    if points.numel() == 0:
        raise ValueError("Input points cannot be empty")
    
    if domain_min >= domain_max:
        raise ValueError(f"domain_min ({domain_min}) must be less than domain_max ({domain_max})")
    
    # Penalty for points below minimum boundary
    min_violations = torch.relu(domain_min - points)
    min_penalty = torch.mean(min_violations)
    
    # Penalty for points above maximum boundary
    max_violations = torch.relu(points - domain_max)
    max_penalty = torch.mean(max_violations)
    
    # Total boundary loss
    boundary_loss = min_penalty + max_penalty
    
    return boundary_loss


# ===== Legacy Function Aliases =====

# Maintain backward compatibility with existing code
pca_reduce_to_2d = reduce_dimensionality_pca
compute_delaunay = compute_delaunay_triangulation
compute_axis_align_loss = compute_axis_alignment_loss