"""
Models Module

Contains implementations of:
- Network G: Geometry Network (SDF + low-freq color)
- Network D: Diffusion Model
- Network R: Reverse Mapping Network
"""

from abc import ABC

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple

# Re-export from concrete implementations
from .network_g import (
    NetworkG,
    SDFMLP,
    PositionalEncoding as NetworkGPositionalEncoding,
    create_network_g,
)

from .network_d import (
    NetworkD,
    DiffusionColorModel,
    FiLMLayer,
    ResBlock,
    ColorUNet,
    create_network_d,
)

from .network_r import (
    NetworkR,
    ReverseMappingMLP,
    PositionalEncoding3D,
    LabelConsistencyLoss,
    create_network_r,
)

from .texture_sampler_field import (
    TextureSamplerField,
    TextureSamplerFieldOutput,
    sample_texture,
)

# Import interfaces from core
from ..interfaces import (
    INetworkG,
    INetworkD,
    INetworkR,
    NetworkGOutput,
    NetworkDOutput,
    NetworkROutput,
    ConditionVector,
)


class BaseNetwork(nn.Module, ABC):
    """
    Base class for all networks.

    Provides common functionality:
    - Parameter counting
    - Device management
    - Checkpoint save/load
    """

    def __init__(self):
        super().__init__()

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

    def save(self, path: str):
        """Save model weights."""
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: torch.device = None):
        """Load model weights."""
        device = device or torch.device("cpu")
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)

    def freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


# =============================================================================
# Model Factory Functions
# =============================================================================


def create_network_g_from_config(config: Dict[str, Any]) -> NetworkG:
    """Create Network G from config dictionary."""
    return NetworkG(
        hidden_dim=config.get("hidden_dim", 256),
        num_layers=config.get("num_layers", 8),
        positional_encoding_freqs=config.get("positional_encoding_freqs", 6),
        skip_connection_layer=config.get("skip_connection_layer", 4),
        include_raw_input=config.get("include_raw_input", True),
        sdf_output_range=config.get("sdf_output_range", 1.0),
    )


def create_network_d_from_config(config: Dict[str, Any]) -> NetworkD:
    """Create Network D from config dictionary."""
    return NetworkD(
        hidden_channels=config.get("hidden_channels", 128),
        num_res_blocks=config.get("num_res_blocks", 4),
        cond_dim=config.get("condition_dim", 42),
        num_diffusion_steps=config.get("num_diffusion_steps", 1000),
        beta_schedule=config.get("beta_schedule", "linear"),
    )


def create_network_r_from_config(config: Dict[str, Any]) -> NetworkR:
    """Create Network R from config dictionary."""
    return NetworkR(
        input_dim=config.get("input_dim", 6),
        hidden_dims=tuple(config.get("hidden_dims", [64, 128, 256])),
        num_classes=config.get("num_classes", 100),
        dropout=config.get("dropout", 0.1),
    )


__all__ = [
    # Concrete implementations
    "NetworkG",
    "NetworkD",
    "NetworkR",
    # Texture Sampler Field (独立实验)
    "TextureSamplerField",
    "TextureSamplerFieldOutput",
    "sample_texture",
    # Sub-modules
    "SDFMLP",
    "NetworkGPositionalEncoding",
    "DiffusionColorModel",
    "FiLMLayer",
    "ResBlock",
    "ColorUNet",
    "ReverseMappingMLP",
    "PositionalEncoding3D",
    "LabelConsistencyLoss",
    # Factory functions
    "create_network_g",
    "create_network_d",
    "create_network_r",
    "create_network_g_from_config",
    "create_network_d_from_config",
    "create_network_r_from_config",
    # Base class
    "BaseNetwork",
    # Interfaces
    "INetworkG",
    "INetworkD",
    "INetworkR",
    "NetworkGOutput",
    "NetworkDOutput",
    "NetworkROutput",
    "ConditionVector",
]
