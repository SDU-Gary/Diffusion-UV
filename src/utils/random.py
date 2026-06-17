"""
Random Number Generator with reproducibility
"""

import torch
import numpy as np
import random
from typing import Optional

from .device_manager import get_device


class GlobalRandomSetter:
    """
    Set global random seeds for reproducibility.
    Call once at the start of your script.
    """

    def __init__(self, seed: int = 42, device: Optional[torch.device] = None):
        self.seed = seed
        self.device = device if device else torch.device("cpu")

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if self.device.type == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def get_current_seed() -> int:
        """Get the current seed value."""
        return RandomNumberGenerator.seed


# Global singleton
class RandomNumberGenerator:
    """
    Centralized random number generator with state.
    Automatically handles device-specific seeding.
    """
    seed = 42

    @classmethod
    def set_seed(cls, seed: int):
        """Set global random seed."""
        cls.seed = seed
        GlobalRandomSetter(seed)

    @classmethod
    def get_seed(cls) -> int:
        """Get current seed."""
        return cls.seed

    @classmethod
    def reset(cls):
        """Reset to default seed."""
        cls.set_seed(42)


def get_rng(device=None) -> torch.Generator:
    """
    Create an RNG for device-specific random operations.

    Args:
        device: torch.device, if None uses current device

    Returns:
        torch.Generator
    """
    device = device or get_device()
    rng = torch.Generator(device=device)
    rng.manual_seed(RandomNumberGenerator.seed)
    return rng


def set_seed_for_epoch(epoch: int, seed_offset: int = 0):
    """
    Set seed with epoch offset for deterministic shuffling.

    Args:
        epoch: Current epoch number
        seed_offset: Additional offset for data shuffling
    """
    total_seed = RandomNumberGenerator.seed + epoch * 1000 + seed_offset
    RandomNumberGenerator.set_seed(total_seed)
