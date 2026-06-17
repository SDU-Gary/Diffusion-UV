"""
Network G: Geometry Network (SDF + Low-frequency Color)

Architecture based on torch-ngp style MLP with:
- Sinusoidal positional encoding
- Skip connection at layer 4
- Two output heads: SDF and base color
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from ..interfaces import INetworkG, NetworkGOutput


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as used in NeRF/torch-ngp.

    PE(x) = [sin(2^0 * pi * x), cos(2^0 * pi * x), ..., sin(2^{L-1} * pi * x), cos(2^{L-1} * pi * x)]
    """

    def __init__(self, in_dim: int = 3, num_frequencies: int = 6, include_input: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        # Frequency bands: 2^0, 2^1, ..., 2^{L-1}
        self.register_buffer(
            "freq_bands",
            2.0 ** torch.linspace(0.0, num_frequencies - 1, num_frequencies)
        )

        # Output dimension
        self.out_dim = in_dim * num_frequencies * 2
        if include_input:
            self.out_dim += in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding.

        Args:
            x: (B, 3) or (B, N, 3) input positions

        Returns:
            Encoded tensor
        """
        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, 3)

        B, N, D = x.shape
        x = x.reshape(B * N, D)

        out = []
        if self.include_input:
            out.append(x)

        for freq in self.freq_bands:
            out.append(torch.sin(freq * math.pi * x))
            out.append(torch.cos(freq * math.pi * x))

        result = torch.cat(out, dim=-1)
        result = result.reshape(B, N, -1)

        return result.squeeze(1) if original_shape == (B, D) else result


class SDFMLP(nn.Module):
    """
    SDF MLP with skip connection.

    Architecture:
        Input: x (3D) → Positional Encoding (L=6) → [36-dim]
        MLP: Linear(36 → 256) + ReLU × 4
             Linear(256+36 → 256) + ReLU × 4
        Output: SDF (1) + color_base (3)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 8,
        positional_encoding_freqs: int = 6,
        skip_connection_layer: int = 4,
        include_raw_input: bool = True,
        sdf_output_range: float = 1.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skip_connection_layer = skip_connection_layer
        self.sdf_output_range = sdf_output_range

        # Positional encoding
        self.pe = PositionalEncoding(
            in_dim=3,
            num_frequencies=positional_encoding_freqs,
            include_input=include_raw_input,
        )
        pe_dim = self.pe.out_dim  # 36 + 3 = 39

        # First part: before skip connection
        self.fc_in = nn.Linear(pe_dim, hidden_dim)
        self.fc_skip = nn.Linear(pe_dim + hidden_dim, hidden_dim)

        # Hidden layers
        self.fc_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)
        ])

        # Output heads
        self.fc_sdf = nn.Linear(hidden_dim, 1)
        self.fc_color = nn.Linear(hidden_dim, 3)

        # Activation
        self.activation = nn.ReLU(inplace=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize MLP weights with small values for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1e-2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            positions: (B, 3) 3D positions

        Returns:
            sdf: (B,) signed distance values
            color_base: (B, 3) low-frequency base colors
        """
        # Positional encoding
        x = self.pe(positions)  # (B, 39)

        # First layer
        h = self.activation(self.fc_in(x))  # (B, 256)

        # Hidden layers before skip
        for i in range(self.skip_connection_layer - 1):
            h = self.activation(self.fc_layers[i](h))  # (B, 256)

        # Skip connection
        h = self.activation(self.fc_skip(torch.cat([x, h], dim=-1)))  # (B, 256)

        # Hidden layers after skip
        for i in range(self.skip_connection_layer, self.num_layers - 2):
            h = self.activation(self.fc_layers[i](h))  # (B, 256)

        # Output heads
        sdf = self.fc_sdf(h).squeeze(-1)  # (B,)
        sdf = torch.tanh(sdf) * self.sdf_output_range  # Scale to [-range, range]

        color_base = torch.sigmoid(self.fc_color(h))  # (B, 3), range [0, 1]

        return sdf, color_base


class NetworkG(nn.Module, INetworkG):
    """
    Geometry Network (Network G).

    Predicts SDF and low-frequency base color for 3D points.
    Includes gradient computation for Eikonal loss.

    Usage:
        model = NetworkG(hidden_dim=256, num_layers=8)
        sdf, color_base = model(positions)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 8,
        positional_encoding_freqs: int = 6,
        skip_connection_layer: int = 4,
        include_raw_input: bool = True,
        sdf_output_range: float = 1.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.positional_encoding_freqs = positional_encoding_freqs
        self.skip_connection_layer = skip_connection_layer

        self.mlp = SDFMLP(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            positional_encoding_freqs=positional_encoding_freqs,
            skip_connection_layer=skip_connection_layer,
            include_raw_input=include_raw_input,
            sdf_output_range=sdf_output_range,
        )

        # Estimated parameter count
        self._param_count = self._count_parameters()

    def _count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, positions: torch.Tensor) -> NetworkGOutput:
        """
        Forward pass.

        Args:
            positions: (B, 3) 3D positions

        Returns:
            NetworkGOutput with sdf, color_base, and normal
        """
        sdf, color_base = self.mlp(positions)

        normal = None
        if torch.is_grad_enabled():
            # Normal estimation requires an autograd graph. Phase 2 calls G
            # under torch.no_grad(), where only sdf/color_base are needed.
            if positions.requires_grad and sdf.requires_grad:
                normal = self._compute_normal(positions, sdf)
            else:
                positions_grad = positions.detach().requires_grad_(True)
                sdf_grad, _ = self.mlp(positions_grad)
                grad = torch.autograd.grad(
                    outputs=sdf_grad,
                    inputs=positions_grad,
                    grad_outputs=torch.ones_like(sdf_grad),
                    create_graph=False,
                )[0]
                normal = F.normalize(grad, dim=-1)
                if not self.training:
                    normal = normal.detach()

        return NetworkGOutput(sdf=sdf, color_base=color_base, normal=normal)

    def _compute_normal(self, positions: torch.Tensor, sdf: torch.Tensor) -> torch.Tensor:
        """
        Compute surface normal from SDF gradient.

        Args:
            positions: (B, 3) positions
            sdf: (B,) SDF values

        Returns:
            (B, 3) normalized normals
        """
        grad = torch.autograd.grad(
            outputs=sdf,
            inputs=positions,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Normalize gradient to get normal
        normal = F.normalize(grad, dim=-1, eps=1e-6)
        return normal

    def get_sdf(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Get SDF values only (for inference).

        Args:
            positions: (B, 3) 3D positions

        Returns:
            (B,) SDF values
        """
        with torch.no_grad():
            sdf, _ = self.mlp(positions)
        return sdf

    def get_color_base(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Get base colors only (for inference).

        Args:
            positions: (B, 3) 3D positions

        Returns:
            (B, 3) base colors
        """
        with torch.no_grad():
            _, color_base = self.mlp(positions)
        return color_base

    def get_sdf_and_color(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get both SDF and color without computing normal.

        Args:
            positions: (B, 3) 3D positions

        Returns:
            sdf: (B,), color_base: (B, 3)
        """
        return self.mlp(positions)

    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count network parameters.

        Args:
            trainable_only: If True, count only trainable parameters

        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


def create_network_g(
    hidden_dim: int = 256,
    num_layers: int = 8,
    positional_encoding_freqs: int = 6,
    skip_connection_layer: int = 4,
    **kwargs,
) -> NetworkG:
    """
    Create Network G with specified parameters.

    Args:
        hidden_dim: Hidden layer dimension
        num_layers: Total MLP layers
        positional_encoding_freqs: Number of frequency bands
        skip_connection_layer: Layer index for skip connection
        **kwargs: Additional arguments

    Returns:
        NetworkG instance
    """
    return NetworkG(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        positional_encoding_freqs=positional_encoding_freqs,
        skip_connection_layer=skip_connection_layer,
        **kwargs,
    )
