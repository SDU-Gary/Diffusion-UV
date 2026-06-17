"""
Positional Encoding for Neural Fields
"""

import torch
import torch.nn as nn
import math
from typing import Literal


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as used in NeRF.

    Encodes input x using:
        PE(x) = [sin(2^0 * pi * x), cos(2^0 * pi * x), ..., sin(2^{L-1} * pi * x), cos(2^{L-1} * pi * x)]

    Args:
        in_dim: Input dimension (default: 3 for xyz coordinates)
        num_frequencies: Number of frequency bands (L in the formula)
        include_input: Whether to concatenate raw input to the encoding
        log_sampling: If True, sample frequencies in log space
    """

    def __init__(
        self,
        in_dim: int = 3,
        num_frequencies: int = 6,
        include_input: bool = True,
        log_sampling: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        if log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, num_frequencies - 1, num_frequencies)
        else:
            freq_bands = torch.linspace(1.0, 2.0 ** (num_frequencies - 1), num_frequencies)

        self.register_buffer("freq_bands", freq_bands)

        # Output dimension calculation
        self.out_dim = in_dim * num_frequencies * 2  # sin and cos for each frequency
        if include_input:
            self.out_dim += in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding.

        Args:
            x: Input tensor of shape (..., in_dim)

        Returns:
            Encoded tensor of shape (..., out_dim)
        """
        out = []
        if self.include_input:
            out.append(x)

        for freq in self.freq_bands:
            out.append(torch.sin(freq * math.pi * x))
            out.append(torch.cos(freq * math.pi * x))

        return torch.cat(out, dim=-1)


class IntegratedPositionalEncoding(nn.Module):
    """
    Integrated positional encoding for anti-aliased neural fields.

    Projects Gaussians (mean + variance) onto positional encoding,
    providing anti-aliasing by attenuating high frequencies.

    Args:
        in_dim: Input dimension
        num_frequencies: Number of frequency bands
        include_input: Whether to include raw input
    """

    def __init__(
        self,
        in_dim: int = 3,
        num_frequencies: int = 6,
        include_input: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        freq_bands = 2.0 ** torch.linspace(0.0, num_frequencies - 1, num_frequencies)
        self.register_buffer("freq_bands", freq_bands)

        self.out_dim = in_dim * num_frequencies * 2
        if include_input:
            self.out_dim += in_dim

    def forward(
        self,
        mean: torch.Tensor,
        variance: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply integrated positional encoding.

        Args:
            mean: Mean of Gaussians, shape (..., in_dim)
            variance: Variance of Gaussians, shape (..., in_dim)

        Returns:
            Encoded tensor of shape (..., out_dim)
        """
        out = []
        if self.include_input:
            out.append(mean)

        for freq in self.freq_bands:
            # Attenuation factor for Gaussian
            attenuation = torch.exp(-0.5 * variance * (freq * math.pi) ** 2)
            out.append(attenuation * torch.sin(freq * math.pi * mean))
            out.append(attenuation * torch.cos(freq * math.pi * mean))

        return torch.cat(out, dim=-1)


class HashGridEncoding(nn.Module):
    """
    Hash grid encoding (Instant-NGP style) for efficient spatial encoding.

    This is a placeholder interface - actual implementation would use
    custom CUDA kernels for efficiency.

    Args:
        in_dim: Input dimension (typically 3)
        num_levels: Number of resolution levels
        base_resolution: Base resolution of the hash grid
        max_resolution: Maximum resolution
        hash_table_size: Size of hash table per level
        feature_dim: Feature dimension per level
    """

    def __init__(
        self,
        in_dim: int = 3,
        num_levels: int = 16,
        base_resolution: int = 16,
        max_resolution: int = 2048,
        hash_table_size: int = 2**19,
        feature_dim: int = 2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_levels = num_levels
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.hash_table_size = hash_table_size
        self.feature_dim = feature_dim

        # Calculate resolutions for each level
        b = math.exp((math.log(max_resolution) - math.log(base_resolution)) / (num_levels - 1))
        self.resolutions = [int(base_resolution * (b**i)) for i in range(num_levels)]

        # Hash tables - one per level
        # Note: In practice, this should be optimized with custom CUDA
        self.embeddings = nn.ModuleList([
            nn.Embedding(hash_table_size, feature_dim)
            for _ in range(num_levels)
        ])

        self.out_dim = num_levels * feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply hash grid encoding.

        Args:
            x: Normalized coordinates in [0, 1], shape (..., in_dim)

        Returns:
            Encoded features, shape (..., out_dim)
        """
        # Placeholder - actual implementation requires spatial hashing
        # and trilinear interpolation
        raise NotImplementedError("HashGridEncoding requires custom CUDA implementation")


def get_positional_encoding(
    encoding_type: Literal["sinusoidal", "integrated", "hashgrid"],
    in_dim: int = 3,
    num_frequencies: int = 6,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create positional encoding.

    Args:
        encoding_type: Type of encoding
        in_dim: Input dimension
        num_frequencies: Number of frequency bands
        **kwargs: Additional arguments for specific encodings

    Returns:
        Positional encoding module
    """
    if encoding_type == "sinusoidal":
        return PositionalEncoding(in_dim, num_frequencies, **kwargs)
    elif encoding_type == "integrated":
        return IntegratedPositionalEncoding(in_dim, num_frequencies, **kwargs)
    elif encoding_type == "hashgrid":
        return HashGridEncoding(in_dim, num_frequencies, **kwargs)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
