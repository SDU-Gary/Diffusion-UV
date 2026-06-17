"""
SDF Network for C^∞ Continuous Normal Field

This module implements a lightweight SDF network that provides smooth,
analytic normals via autograd for tangent space projection in MA-IUVF.

Key Design Principles:
- Lightweight: ~10K parameters (vs MA-IUVF's ~100K)
- Smooth: Softplus activation (C^∞ continuous, not ReLU's C^0)
- C^∞ Continuous: B-Spline Hash Grid provides analytic gradients
- Eikonal Constrained: ||∇SDF|| = 1 everywhere

Architecture:
    Input: [B, 3] positions
        ↓
    B-Spline Hash Grid (8 levels, low resolution)
        ↓
    2-layer MLP (hidden_dim=32, Softplus)
        ↓
    Output: [B] SDF values
        ↓
    Autograd → [B, 3] normals (∇SDF)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class SDFNetwork(nn.Module):
    """
    Lightweight SDF network for C^∞ continuous normal field

    This network provides smooth, analytic normals via autograd:
        normals = ∇SDF / ||∇SDF||

    The Eikonal constraint ensures ||∇SDF|| = 1 everywhere,
    making normalization almost unnecessary.

    Architecture:
    - B-Spline Hash Grid encoder (8 levels, max_res=128)
    - 2-layer MLP with Softplus activation (hidden_dim=32)
    - Scalar SDF output

    Args:
        num_levels: Number of B-Spline levels (default: 8)
        features_per_level: Features per level (default: 2)
        log2_hashmap_size: Hash table size log2 (default: 12 → 4096)
        base_res: Coarsest resolution (default: 8)
        max_res: Finest resolution (default: 128, not too high for smoothness)
        hidden_dim: MLP hidden layer dimension (default: 32)
        num_layers: MLP number of layers (default: 2)
        bbox_min: Bounding box min (default: [0, 0, 0])
        bbox_max: Bounding box max (default: [1, 1, 1])
        cuda_backend: CUDA backend (default: "torch" for gradients)
    """

    def __init__(
        self,
        num_levels: int = 8,
        features_per_level: int = 2,
        log2_hashmap_size: int = 12,
        base_res: int = 8,
        max_res: int = 128,
        hidden_dim: int = 32,
        num_layers: int = 2,
        bbox_min: Optional[Tuple[float, float, float]] = None,
        bbox_max: Optional[Tuple[float, float, float]] = None,
        cuda_backend: str = "torch",
    ):
        super().__init__()

        if bbox_min is None:
            bbox_min = (0.0, 0.0, 0.0)
        if bbox_max is None:
            bbox_max = (1.0, 1.0, 1.0)

        # Import B-Spline Hash Grid encoder
        from src.models.encoders.bspline_grid import BSplineHashGrid

        # B-Spline Hash Grid encoder (lightweight)
        self.grid_encoder = BSplineHashGrid(
            num_levels=num_levels,
            features_per_level=features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_res=base_res,
            max_res=max_res,
            init_scale=1e-4,
            cuda_backend=cuda_backend,
            normalize_positions=True,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
        )

        encoder_dim = self.grid_encoder.output_dim

        # 2-layer MLP with Softplus activation (C^∞ continuous)
        layers = []
        input_dim = encoder_dim

        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Softplus(beta=10))
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))  # Scalar SDF output

        self.mlp = nn.Sequential(*layers)

        logger.info(
            f"SDFNetwork: levels={num_levels}, max_res={max_res}, "
            f"hidden_dim={hidden_dim}, num_layers={num_layers}, "
            f"params={sum(p.numel() for p in self.parameters()):,}"
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute SDF values

        Args:
            positions: [B, 3] input positions

        Returns:
            sdf: [B] SDF values (positive = outside, negative = inside)
        """
        # Encode positions with B-Spline Hash Grid
        features = self.grid_encoder(positions)  # [B, L*F]

        # MLP to scalar SDF
        sdf = self.mlp(features).squeeze(-1)  # [B]

        return sdf

    def get_normals(
        self,
        positions: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute normals via autograd: n = ∇SDF

        This method computes the gradient of SDF with respect to positions:
            normals = ∇SDF / ||∇SDF||

        If Eikonal constraint is satisfied (||∇SDF|| = 1), normalization
        is almost unnecessary.

        Args:
            positions: [B, 3] positions (requires_grad=True)
            normalize: Whether to normalize to unit vectors (default: True)

        Returns:
            normals: [B, 3] normalized normal vectors
        """
        # Ensure positions require gradients
        if not positions.requires_grad:
            raise ValueError("positions must have requires_grad=True to compute normals")

        # Forward pass
        sdf = self.forward(positions)

        # Compute gradient via autograd
        grad = torch.autograd.grad(
            outputs=sdf.sum(),
            inputs=positions,
            create_graph=False,
            retain_graph=False,
        )[0]  # [B, 3]

        # Normalize (if Eikonal constraint is satisfied, this is ~1.0)
        if normalize:
            normals = torch.nn.functional.normalize(grad, dim=-1, eps=1e-6)
        else:
            normals = grad

        return normals

    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_sdf_network(
    num_levels: int = 8,
    features_per_level: int = 2,
    log2_hashmap_size: int = 12,
    base_res: int = 8,
    max_res: int = 128,
    hidden_dim: int = 32,
    num_layers: int = 2,
    bbox_min: Optional[Tuple[float, float, float]] = None,
    bbox_max: Optional[Tuple[float, float, float]] = None,
    cuda_backend: str = "torch",
) -> SDFNetwork:
    """
    Convenience function to create SDF network

    Args:
        num_levels: Number of B-Spline levels (default: 8)
        features_per_level: Features per level (default: 2)
        log2_hashmap_size: Hash table size log2 (default: 12)
        base_res: Coarsest resolution (default: 8)
        max_res: Finest resolution (default: 128)
        hidden_dim: MLP hidden dimension (default: 32)
        num_layers: MLP number of layers (default: 2)
        bbox_min: Bounding box min (default: [0, 0, 0])
        bbox_max: Bounding box max (default: [1, 1, 1])
        cuda_backend: CUDA backend (default: "torch")

    Returns:
        SDF network
    """
    sdf_net = SDFNetwork(
        num_levels=num_levels,
        features_per_level=features_per_level,
        log2_hashmap_size=log2_hashmap_size,
        base_res=base_res,
        max_res=max_res,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        cuda_backend=cuda_backend,
    )

    logger.info(f"Created SDF network: {sdf_net.get_num_params():,} parameters")

    return sdf_net


# =============================================================================
# SIREN-based SDF Network
# =============================================================================

class SDFNetworkSIREN(nn.Module):
    """
    SIREN-based SDF network for C∞ continuous normal field

    Based on "Implicit Neural Representations with Periodic Activation Functions"
    Sitzmann et al. 2020

    Architecture:
    - Pure MLP (no encoding layer, direct 3D coordinate input)
    - sin activation with frequency parameters (omega_0, omega_0/2, omega_0/4, ...)
    - SIREN-specific weight initialization (critical for convergence)

    This replaces the B-Spline Hash Grid encoder with a pure MLP approach,
    providing higher capacity per parameter and better gradient properties.

    Args:
        in_dim: Input dimension (default: 3 for 3D coordinates)
        hidden_dim: Hidden layer dimension (default: 128)
        num_layers: Number of hidden layers (default: 5)
        omega_0: First layer frequency parameter (default: 30.0)

    Expected parameter count:
    - 4 layers × 128 dim ≈ 66K parameters
    - 5 layers × 128 dim ≈ 86K parameters
    - 5 layers × 256 dim ≈ 391K parameters
    """

    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 5,
        omega_0: float = 30.0,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.omega_0 = omega_0

        # Build network layers with sin activation
        layers = []

        # First layer: in_dim → hidden_dim with omega_0
        layers.append(nn.Linear(in_dim, hidden_dim))
        from src.models.siren_layer import Sine
        layers.append(Sine(omega_0))

        # Hidden layers: hidden_dim → hidden_dim with omega_0/2^(i-1)
        for i in range(1, num_layers):
            omega = omega_0 / (2 ** i)
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(Sine(omega))

        # Output layer: hidden_dim → 1 (linear activation, no sin)
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

        # SIREN-specific weight initialization (CRITICAL for convergence)
        self._init_weights_siren()

        logger.info(
            f"SDFNetworkSIREN: hidden_dim={hidden_dim}, num_layers={num_layers}, "
            f"omega_0={omega_0}, params={sum(p.numel() for p in self.parameters()):,}"
        )

    def _init_weights_siren(self):
        """
        SIREN-specific weight initialization

        Based on SIREN paper:
        - First layer: std = sqrt(2/(in+out)) * omega_0
        - Hidden layers: std = sqrt(2/(in+out)) * omega
          where omega = omega_0 / 2^(layer_idx)

        This initialization is CRITICAL for SIREN convergence.
        """
        with torch.no_grad():
            for i, layer in enumerate(self.net):
                if isinstance(layer, nn.Linear):
                    in_dim = layer.in_features
                    out_dim = layer.out_features

                    if i == 0:
                        # First layer uses omega_0
                        std = np.sqrt(2 / (in_dim + out_dim)) * self.omega_0
                    else:
                        # Hidden layers use decreasing omega
                        # Every other layer is Linear (alternating Linear/Sine)
                        layer_idx = i // 2
                        omega = self.omega_0 / (2 ** layer_idx)
                        std = np.sqrt(2 / (in_dim + out_dim)) * omega

                    layer.weight.uniform_(-std, std)
                    if layer.bias is not None:
                        layer.bias.uniform_(-std, std)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute SDF values

        Args:
            positions: [B, 3] input positions (direct 3D coordinates, no encoding)

        Returns:
            sdf: [B] SDF values (positive = outside, negative = inside)
        """
        sdf = self.net(positions).squeeze(-1)  # [B]
        return sdf

    def get_normals(
        self,
        positions: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute normals via autograd: n = ∇SDF

        This method computes the gradient of SDF with respect to positions:
            normals = ∇SDF / ||∇SDF||

        Args:
            positions: [B, 3] positions (requires_grad=True)
            normalize: Whether to normalize to unit vectors (default: True)

        Returns:
            normals: [B, 3] normalized normal vectors
        """
        # Ensure positions require gradients
        if not positions.requires_grad:
            raise ValueError("positions must have requires_grad=True to compute normals")

        # Forward pass
        sdf = self.forward(positions)

        # Compute gradient via autograd
        grad = torch.autograd.grad(
            outputs=sdf.sum(),
            inputs=positions,
            create_graph=False,
            retain_graph=False,
        )[0]  # [B, 3]

        # Normalize (if Eikonal constraint is satisfied, this is ~1.0)
        if normalize:
            normals = torch.nn.functional.normalize(grad, dim=-1, eps=1e-6)
        else:
            normals = grad

        return normals

    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_sdf_network_siren(
    in_dim: int = 3,
    hidden_dim: int = 128,
    num_layers: int = 5,
    omega_0: float = 30.0,
) -> SDFNetworkSIREN:
    """
    Convenience function to create SIREN-based SDF network

    Args:
        in_dim: Input dimension (default: 3 for 3D coordinates)
        hidden_dim: Hidden layer dimension (default: 128)
        num_layers: Number of layers (default: 5)
        omega_0: Omega_0 frequency parameter (default: 30.0)

    Returns:
        SIREN-based SDF network

    Example:
        >>> sdf_net = create_sdf_network_siren(hidden_dim=128, num_layers=5)
        >>> sdf_net.get_num_params()
        86000
    """
    sdf_net = SDFNetworkSIREN(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        omega_0=omega_0,
    )

    logger.info(f"Created SDFNetworkSIREN: {sdf_net.get_num_params():,} parameters")

    return sdf_net
