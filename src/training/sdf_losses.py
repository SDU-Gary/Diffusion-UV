"""
SDF Loss Functions

This module implements loss functions for SDF network pre-training.
The key insight is that we don't need exact SDF values (distance queries).
Instead, we use:
1. Surface loss: |SDF| = 0 on mesh surface
2. Eikonal loss: ||∇SDF|| = 1 everywhere in space

The Eikonal constraint ensures:
- Gradient magnitude = 1.0 everywhere
- Autograd normals are already unit vectors (no normalization needed)
- Prevents SDF from collapsing to zero everywhere
"""

import torch
import torch.nn.functional as F
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def compute_sdf_loss(
    sdf_pred_surface: torch.Tensor,
    sdf_pred_off_surface: torch.Tensor,
    grad_off_surface: torch.Tensor,
    lambda_eikonal: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    Compute SDF training loss

    Loss = L_surface + λ_eikonal × L_eikonal

    where:
    - L_surface = mean(|SDF_surface|)  (surface points should have SDF = 0)
    - L_eikonal = mean((||∇SDF|| - 1)^2)  (gradient magnitude should be 1)

    Args:
        sdf_pred_surface: [B_surf] SDF predictions on surface (should be ~0)
        sdf_pred_off_surface: [B_off] SDF predictions off-surface (not constrained)
        grad_off_surface: [B_off, 3] gradients at off-surface points (∇SDF)
        lambda_eikonal: Weight for Eikonal loss (default: 0.1)

    Returns:
        loss_dict: {
            "total": total loss,
            "surface": surface SDF loss,
            "eikonal": Eikonal loss,
        }
    """
    # Surface loss: |SDF - 0|
    # Points on surface should have SDF = 0
    loss_surface = torch.mean(torch.abs(sdf_pred_surface))

    # Eikonal loss: ||∇SDF|| - 1|^2
    # Gradient magnitude should be 1 everywhere in space
    grad_norm = torch.norm(grad_off_surface, dim=-1)  # [B_off]
    loss_eikonal = torch.mean((grad_norm - 1.0) ** 2)

    # Total loss
    total_loss = loss_surface + lambda_eikonal * loss_eikonal

    return {
        "total": total_loss,
        "surface": loss_surface,
        "eikonal": loss_eikonal,
    }


def compute_sdf_loss_smooth_l1(
    sdf_pred_surface: torch.Tensor,
    sdf_pred_off_surface: torch.Tensor,
    grad_off_surface: torch.Tensor,
    lambda_eikonal: float = 0.1,
    beta: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    Compute SDF training loss with smooth L1 (Huber) for surface loss

    Uses smooth L1 (Huber loss) for surface loss instead of L1:
        smooth_l1(x) = { x^2 / (2*beta)        if |x| < beta
                       { |x| - beta/2          otherwise

    This is more robust to outliers and provides smoother gradients.

    Args:
        sdf_pred_surface: [B_surf] SDF predictions on surface
        sdf_pred_off_surface: [B_off] SDF predictions off-surface
        grad_off_surface: [B_off, 3] gradients at off-surface points
        lambda_eikonal: Weight for Eikonal loss (default: 0.1)
        beta: Huber loss threshold (default: 0.1)

    Returns:
        loss_dict: {
            "total": total loss,
            "surface": surface SDF loss,
            "eikonal": Eikonal loss,
        }
    """
    # Surface loss: smooth L1 (Huber)
    abs_sdf = torch.abs(sdf_pred_surface)
    quadratic = torch.clamp(abs_sdf, max=beta)
    linear = abs_sdf - quadratic

    loss_surface = torch.mean(0.5 * quadratic ** 2 / beta + linear)

    # Eikonal loss: ||∇SDF|| - 1|^2
    grad_norm = torch.norm(grad_off_surface, dim=-1)
    loss_eikonal = torch.mean((grad_norm - 1.0) ** 2)

    # Total loss
    total_loss = loss_surface + lambda_eikonal * loss_eikonal

    return {
        "total": total_loss,
        "surface": loss_surface,
        "eikonal": loss_eikonal,
    }


def compute_eikonal_loss(
    grad: torch.Tensor,
    target: float = 1.0,
) -> torch.Tensor:
    """
    Compute Eikonal loss: (||∇SDF|| - target)^2

    The Eikonal equation states that the gradient magnitude should be 1 everywhere:
        ||∇SDF(x)|| = 1 for all x in space

    Args:
        grad: [B, 3] gradient vectors
        target: Target gradient magnitude (default: 1.0)

    Returns:
        loss: Scalar Eikonal loss
    """
    grad_norm = torch.norm(grad, dim=-1)
    loss = torch.mean((grad_norm - target) ** 2)
    return loss


def compute_surface_loss(
    sdf_pred: torch.Tensor,
    loss_type: str = "l1",
) -> torch.Tensor:
    """
    Compute surface loss: SDF should be 0 on mesh surface

    Args:
        sdf_pred: [B] SDF predictions on surface
        loss_type: Type of loss ("l1", "l2", "smooth_l1")

    Returns:
        loss: Scalar surface loss
    """
    if loss_type == "l1":
        loss = torch.mean(torch.abs(sdf_pred))
    elif loss_type == "l2":
        loss = torch.mean(sdf_pred ** 2)
    elif loss_type == "smooth_l1":
        # Huber loss with beta=0.1
        beta = 0.1
        abs_sdf = torch.abs(sdf_pred)
        quadratic = torch.clamp(abs_sdf, max=beta)
        linear = abs_sdf - quadratic
        loss = torch.mean(0.5 * quadratic ** 2 / beta + linear)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss


def validate_sdf_loss(
    sdf_pred_surface: torch.Tensor,
    grad_off_surface: torch.Tensor,
) -> Dict[str, float]:
    """
    Validate SDF predictions

    Args:
        sdf_pred_surface: [B_surf] SDF predictions on surface
        grad_off_surface: [B_off, 3] gradients at off-surface points

    Returns:
        metrics: {
            "surface_mean": mean(|SDF|) on surface,
            "surface_std": std(|SDF|) on surface,
            "surface_max": max(|SDF|) on surface,
            "grad_mean": mean(||∇SDF||),
            "grad_std": std(||∇SDF||),
            "eikonal_error": mean(||∇SDF|| - 1),
        }
    """
    with torch.no_grad():
        # Surface statistics
        abs_sdf = torch.abs(sdf_pred_surface)
        surface_mean = abs_sdf.mean().item()
        surface_std = abs_sdf.std().item()
        surface_max = abs_sdf.max().item()

        # Gradient statistics
        grad_norm = torch.norm(grad_off_surface, dim=-1)
        grad_mean = grad_norm.mean().item()
        grad_std = grad_norm.std().item()
        eikonal_error = torch.mean(torch.abs(grad_norm - 1.0)).item()

    return {
        "surface_mean": surface_mean,
        "surface_std": surface_std,
        "surface_max": surface_max,
        "grad_mean": grad_mean,
        "grad_std": grad_std,
        "eikonal_error": eikonal_error,
    }
