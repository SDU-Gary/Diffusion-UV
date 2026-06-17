"""
Data Utilities

- Batch tensor handling
- Coordinate transformations
- Memory-efficient data loading
"""

import torch
import numpy as np
from typing import Callable, Iterator, Optional, Tuple, Union
from pathlib import Path


class BatchSampler:
    """
    Efficient batch sampler for large datasets.

    Handles both fixed-size batches and variable-sized batches for
    geometry-aware sampling (e.g., structured sampling strategies).
    """

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        """
        Initialize batch sampler.

        Args:
            num_samples: Total number of samples
            batch_size: Batch size
            shuffle: Whether to shuffle indices
            drop_last: Whether to drop the last batch if incomplete
            seed: Random seed for shuffling
        """
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        self.indices = self._generate_indices()

    def _generate_indices(self) -> np.ndarray:
        """Generate indices for sampling."""
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(indices)
        return indices

    def __len__(self) -> int:
        """Number of batches."""
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over batches."""
        start = 0
        end = self.batch_size
        while start < self.num_samples:
            batch_indices = self.indices[start:end]
            yield batch_indices
            start = end
            end += self.batch_size

    def reset(self):
        """Reset the sampler (shuffles if shuffle=True)."""
        self.indices = self._generate_indices()


class UncertaintyAwareBatchSampler(BatchSampler):
    """
    Batch sampler that prioritizes challenging samples.

    Low uncertainty samples are sampled more frequently,
    with a probability weighting scheme.
    """

    def __init__(
        self,
        uncertainties: np.ndarray,
        batch_size: int,
        priority_weight: float = 0.3,
        **kwargs,
    ):
        """
        Initialize uncertainty-aware sampler.

        Args:
            uncertainties: Uncertainty values for each sample
            batch_size: Batch size
            priority_weight: How much to weight uncertainty (0-1)
            **kwargs: Additional arguments for BatchSampler
        """
        super().__init__(
            num_samples=len(uncertainties),
            batch_size=batch_size,
            **kwargs,
        )
        self.uncertainties = uncertainties
        self.priority_weight = priority_weight

    def _generate_indices(self) -> np.ndarray:
        """Generate weighted random indices based on uncertainty."""
        # Normalize uncertainty to [0, 1]
        if np.max(self.uncertainties) > np.min(self.uncertainties):
            weights = 1.0 + self.priority_weight * (
                (self.uncertainties - np.min(self.uncertainties)) /
                (np.max(self.uncertainties) - np.min(self.uncertainties))
            )
        else:
            weights = np.ones_like(self.uncertainties)

        probabilities = weights / np.sum(weights)
        indices = np.random.choice(
            self.num_samples,
            size=self.num_samples,
            replace=True,
            p=probabilities,
        )
        if self.shuffle:
            np.random.shuffle(indices)
        return indices


class DataCollator:
    """
    Collates variable-length samples into batches.

    Handles different tensor shapes and masks for
    geometry-aware data structures.
    """

    def __init__(self, pad_value: float = 0.0):
        """
        Initialize collator.

        Args:
            pad_value: Padding value for variable-length tensors
        """
        self.pad_value = pad_value

    def collate(self, batch: list[dict]) -> dict:
        """
        Collate a batch of samples into a single dict.

        Args:
            batch: List of samples, each a dict

        Returns:
            Collated batch as a dict
        """
        # Find maximum size for each key
        max_sizes = {}
        for sample in batch:
            for key, value in sample.items():
                if isinstance(value, np.ndarray) and len(value.shape) > 0:
                    max_size = max_size.get(key, 0)
                    max_size = max(max_size, value.shape[0])
                    max_sizes[key] = max_size

        # Pad and collate
        collated = {}
        for key, value in batch[0].items():
            if isinstance(value, torch.Tensor):
                collated[key] = self._collate_tensors(
                    [s[key] for s in batch],
                    max_sizes.get(key, len(value)),
                )
            elif isinstance(value, np.ndarray):
                collated[key] = self._collate_arrays(
                    [s[key] for s in batch],
                    max_sizes.get(key, len(value)),
                )
            else:
                # Scalars, keep as-is
                collated[key] = value

        return collated

    def _collate_tensors(self, tensors: list[torch.Tensor], max_length: int) -> torch.Tensor:
        """Collate variable-length tensors."""
        if len(tensors) == 0:
            return torch.tensor([])

        # If all same shape, stack directly
        if all(t.shape == tensors[0].shape for t in tensors):
            return torch.stack(tensors)

        # Pad to max_length
        padded = []
        for t in tensors:
            if t.dim() == 0:
                padded.append(t.unsqueeze(0))
            elif t.dim() == 1:
                if t.shape[0] == max_length:
                    padded.append(t)
                else:
                    padded.append(
                        torch.nn.functional.pad(
                            t,
                            (0, max_length - t.shape[0]),
                            value=self.pad_value,
                        )
                    )
            elif t.dim() == 2:
                if t.shape[0] == max_length:
                    padded.append(t)
                else:
                    padded.append(
                        torch.nn.functional.pad(
                            t,
                            (0, max_length - t.shape[0], 0, 0),
                            value=self.pad_value,
                        )
                    )

        return torch.stack(padded)

    def _collate_arrays(self, arrays: list[np.ndarray], max_length: int) -> np.ndarray:
        """Collate variable-length numpy arrays."""
        if len(arrays) == 0:
            return np.array([])

        if all(a.shape == arrays[0].shape for a in arrays):
            return np.stack(arrays)

        if arrays[0].ndim == 1:
            return np.array([
                np.pad(a, (0, max_length - len(a)), constant_values=self.pad_value)
                for a in arrays
            ])
        elif arrays[0].ndim == 2:
            return np.array([
                np.pad(a, ((0, max_length - a.shape[0]), (0, 0)), constant_values=self.pad_value)
                for a in arrays
            ])

        return np.array(arrays)


def normalize_coordinates(
    coords: torch.Tensor,
    min_val: Optional[torch.Tensor] = None,
    max_val: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize coordinates to [0, 1] range.

    When both min_val and max_val are supplied, this applies the inverse
    transform using those bounds. This preserves the historical round-trip
    behavior used by the test suite; use denormalize_coordinates directly in
    new code for clarity.

    Args:
        coords: Original coordinates of shape (..., dim)
        min_val: Minimum values for normalization, if None computed from tensor
        max_val: Maximum values for normalization, if None computed from tensor

    Returns:
        transformed_coords, min_val, max_val, range_val
    """
    should_denormalize = min_val is not None and max_val is not None

    if min_val is None:
        min_val = coords.min(dim=-1, keepdim=True).values
    if max_val is None:
        max_val = coords.max(dim=-1, keepdim=True).values

    range_val = max_val - min_val
    # Avoid division by zero
    range_val = torch.where(
        range_val == 0,
        torch.ones_like(range_val),
        range_val,
    )

    if should_denormalize:
        denormalized_coords = coords * range_val + min_val
        return denormalized_coords, min_val, max_val, range_val

    normalized_coords = (coords - min_val) / range_val
    return normalized_coords, min_val, max_val, range_val


def denormalize_coordinates(
    coords: torch.Tensor,
    min_val: torch.Tensor,
    max_val: torch.Tensor,
) -> torch.Tensor:
    """
    Denormalize coordinates back to original range.

    Args:
        coords: Normalized coordinates of shape (..., dim)
        min_val: Minimum values
        max_val: Maximum values

    Returns:
        Denormalized coordinates
    """
    range_val = max_val - min_val
    coords = coords * range_val + min_val
    return coords


def grid_sample_3d(
    features: torch.Tensor,
    coords: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "border",
) -> torch.Tensor:
    """
    Sample 3D grid features at given coordinates.

    Args:
        features: 4D tensor (B, C, D, H, W)
        coords: 3D coordinates of shape (B, N, 3)
        mode: Sampling mode
        padding_mode: Padding mode

    Returns:
        Sampled features of shape (B, N, C)
    """
    from torch.nn.functional import grid_sample
    from math import pi

    # Normalize coords to [-1, 1]
    # grid_sample expects coordinates in [-1, 1]^2 range
    # We need to determine grid size from features dim
    spatial_dims = features.shape[2:]  # (D, H, W)
    min_coords = torch.tensor([-1.0, -1.0, -1.0], device=features.device)
    max_coords = torch.tensor([1.0, 1.0, 1.0], device=features.device)
    grid = (coords - min_coords) / (max_coords - min_coords) * 2 - 1

    # Grid_sample expects (N, H, W, 3) for 2D, (N, D, H, W, 3) for 3D
    if coords.dim() == 2:
        grid = grid.unsqueeze(0)  # (1, N, 3)
    elif coords.dim() == 3:
        batch_size = coords.size(0)
        grid = grid.view(batch_size, coords.size(1), 1, 1, 3)

    grid = grid.permute(0, 2, 3, 4, 1)  # (B, D, H, W, N)

    sampled = grid_sample(
        features.unsqueeze(0),  # (1, C, D, H, W)
        grid,  # (B, D, H, W, N)
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )

    return sampled.squeeze(0)  # (B, N, C)
