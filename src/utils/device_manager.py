"""
PyTorch Device Management
"""

import torch
from typing import Literal
import logging

logger = logging.getLogger(__name__)


def get_device(device_type: Literal["cuda", "cpu", "auto"] = "auto") -> torch.device:
    """
    Get appropriate torch device.

    Args:
        device_type: "cuda" for GPU, "cpu" for CPU, "auto" for auto-detect

    Returns:
        torch.device object
    """
    if device_type == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.warning("CUDA not available, falling back to CPU")
    elif device_type == "cuda":
        device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
    else:  # "cpu"
        device = torch.device("cpu")
        logger.info("Using CPU device")

    return device


def get_memory_info(device: torch.device = None) -> dict:
    """
    Get GPU/CPU memory info.

    Args:
        device: torch.device object, if None uses current device

    Returns:
        Dict with memory statistics
    """
    if device is None:
        device = get_device()

    info = {}

    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
        cached = torch.cuda.memory_reserved(device) / 1024**2  # MB
        total = torch.cuda.get_device_properties(device).total_memory / 1024**2  # MB

        info.update({
            "allocated_mb": allocated,
            "cached_mb": cached,
            "total_mb": total,
            "utilization": allocated / total * 100,
        })
    else:
        info.update({
            "type": "cpu",
        })

    return info


def print_memory_summary(device: torch.device = None):
    """Print formatted memory summary."""
    info = get_memory_info(device)
    print(f"Memory Info: {info}")
