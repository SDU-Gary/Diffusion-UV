"""
SIREN Activation Layer

This module provides the sin activation function with omega parameter
for SIREN (Sinusoidal Representation Networks).

Based on "Implicit Neural Representations with Periodic Activation Functions"
Sitzmann et al. 2020
"""

import torch
import torch.nn as nn


class Sine(nn.Module):
    """
    Sine activation function with omega parameter for SIREN

    The omega parameter controls the frequency of the sin activation,
    allowing the network to represent high-frequency signals.

    Args:
        omega_0: Frequency parameter (default: 30.0 as per SIREN paper)

    Example:
        >>> sine = Sine(omega_0=30.0)
        >>> x = torch.randn(10, 3)
        >>> y = sine(x)
        >>> y.shape  # Should be [10, 3]
        torch.Size([10, 3])
    """
    def __init__(self, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sine activation with frequency scaling

        Args:
            x: Input tensor of any shape

        Returns:
            sin(omega_0 * x)
        """
        return torch.sin(self.omega_0 * x)
