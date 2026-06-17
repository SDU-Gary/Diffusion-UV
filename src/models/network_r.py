"""
Network R: Reverse Mapping Network for Closed-Loop Constraint

Architecture:
- Lightweight MLP (~50K parameters)
- Takes 3D position and predicted color
- Outputs geometry-texture joint label
- Used for hallucination detection/penalty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from ..interfaces import INetworkR, NetworkROutput


class ReverseMappingMLP(nn.Module):
    """
    Lightweight MLP for reverse mapping.

    Architecture:
        Input: position (3) + color (3) = 6 dimensions
        MLP: 6 -> 64 -> 128 -> 256 -> num_classes
        Output: class logits

    Total params: ~50K for num_classes=100
    """

    def __init__(
        self,
        input_dim: int = 6,  # position (3) + color (3)
        hidden_dims: Tuple[int, ...] = (64, 128, 256),
        num_classes: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, 6) position + color concatenated

        Returns:
            (B, num_classes) class logits
        """
        h = self.mlp(x)
        logits = self.classifier(h)
        return logits


class NetworkR(nn.Module, INetworkR):
    """
    Reverse Mapping Network (Network R).

    Predicts geometry-texture joint labels from position and color.
    Used for closed-loop constraint to prevent hallucination.

    Usage:
        model = NetworkR(num_classes=100)
        output = model(positions, colors)
        labels = model.get_label(positions, colors)
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: Tuple[int, ...] = (64, 128, 256),
        num_classes: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes

        # Positional encoding for better position awareness
        self.pos_encoding = PositionalEncoding3D(num_frequencies=4)
        self.encoded_dim = self.pos_encoding.out_dim  # 3 * 4 * 2 + 3 = 27

        # MLP with correct input dimension
        self.mlp = ReverseMappingMLP(
            input_dim=self.encoded_dim + 3,  # 27 + 3 = 30
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        positions: torch.Tensor,
        colors: torch.Tensor,
    ) -> NetworkROutput:
        """
        Predict class label from position and color.

        Args:
            positions: (B, 3) 3D positions
            colors: (B, 3) colors

        Returns:
            NetworkROutput with logits and probs
        """
        # Encode positions with positional encoding
        pos_enc = self.pos_encoding(positions)  # (B, 3 * 4 * 2 = 24) if num_frequencies=4

        # Concatenate with colors
        x = torch.cat([pos_enc, colors], dim=-1)  # (B, 24 + 3 = 27)

        # Forward through MLP
        logits = self.mlp(x)  # (B, num_classes)
        probs = F.softmax(logits, dim=-1)

        return NetworkROutput(logits=logits, probs=probs)

    def get_label(self, positions: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
        """
        Get predicted label (argmax).

        Args:
            positions: (B, 3) 3D positions
            colors: (B, 3) colors

        Returns:
            (B,) predicted labels
        """
        with torch.no_grad():
            output = self.forward(positions, colors)
            return output.logits.argmax(dim=-1)

    def entropy(self, positions: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction entropy (uncertainty).

        Args:
            positions: (B, 3) 3D positions
            colors: (B, 3) colors

        Returns:
            (B,) entropy values (high = uncertain)
        """
        output = self.forward(positions, colors)
        # H = -sum(p * log(p))
        entropy = -torch.sum(output.probs * torch.log(output.probs + 1e-8), dim=-1)
        return entropy

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count network parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class PositionalEncoding3D(nn.Module):
    """
    Sinusoidal positional encoding for 3D positions.

    Maps 3D coordinates to higher-dimensional representation
    for better position awareness in Network R.
    """

    def __init__(self, num_frequencies: int = 4, include_input: bool = True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        self.out_dim = 3 * num_frequencies * 2
        if include_input:
            self.out_dim += 3

        self.register_buffer(
            "freq_bands",
            2.0 ** torch.linspace(0.0, num_frequencies - 1, num_frequencies)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding.

        Args:
            x: (B, 3) 3D positions

        Returns:
            Encoded tensor of shape (B, out_dim)
        """
        out = []
        if self.include_input:
            out.append(x)

        for freq in self.freq_bands:
            out.append(torch.sin(freq * torch.pi * x))
            out.append(torch.cos(freq * torch.pi * x))

        return torch.cat(out, dim=-1)


class LabelConsistencyLoss(nn.Module):
    """
    Label consistency loss for Network R.

    Ensures that:
    1. Predicted labels match ground truth labels
    2. Low-entropy predictions for realistic samples
    3. High-entropy (uncertain) predictions for hallucinated samples
    """

    def __init__(self, entropy_weight: float = 0.1):
        super().__init__()
        self.entropy_weight = entropy_weight

    def forward(
        self,
        logits: torch.Tensor,
        target_labels: torch.Tensor,
        is_hallucinated: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute label consistency loss.

        Args:
            logits: (B, num_classes) predicted logits
            target_labels: (B,) ground truth labels
            is_hallucinated: (B,) binary mask (1 = hallucinated, 0 = real)

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits, target_labels)

        # Entropy regularization (encourage confident predictions)
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        entropy_loss = entropy.mean()

        # Hallucination penalty: encourage high entropy for hallucinated samples
        hallucination_penalty = torch.tensor(0.0, device=logits.device)
        if is_hallucinated is not None and is_hallucinated.any():
            hallucination_penalty = (
                entropy[is_hallucinated.bool()].mean() * 0.1
            )

        total_loss = ce_loss + self.entropy_weight * entropy_loss - hallucination_penalty * 0.1

        metrics = {
            "ce_loss": ce_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "hallucination_penalty": hallucination_penalty.item(),
        }

        return total_loss, metrics


def create_network_r(
    input_dim: int = 6,
    hidden_dims: Tuple[int, ...] = (64, 128, 256),
    num_classes: int = 100,
    dropout: float = 0.1,
    **kwargs,
) -> NetworkR:
    """
    Create Network R with specified parameters.

    Args:
        input_dim: Input dimension (position + color)
        hidden_dims: Hidden layer dimensions
        num_classes: Number of geometry-texture classes
        dropout: Dropout rate
        **kwargs: Additional arguments

    Returns:
        NetworkR instance
    """
    return NetworkR(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=dropout,
    )
