"""
Logging Configuration
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "diffusion_uv",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Setup a logger with console and optional file output.

    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level
        format_string: Custom format string

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Default format
    if format_string is None:
        format_string = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "diffusion_uv") -> logging.Logger:
    """Get or create a logger."""
    return logging.getLogger(name)


class TensorboardLogger:
    """Wrapper for TensorBoard logging."""

    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard writer.

        Args:
            log_dir: Directory for TensorBoard logs
        """
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalars under the same main tag."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values, step: int):
        """Log a histogram."""
        self.writer.add_histogram(tag, values, step)

    def log_image(self, tag: str, image, step: int):
        """Log an image."""
        self.writer.add_image(tag, image, step)

    def log_mesh(self, tag: str, vertices, faces, step: int, colors=None):
        """Log a 3D mesh."""
        self.writer.add_mesh(tag, vertices, faces, colors, step)

    def close(self):
        """Close the writer."""
        self.writer.close()


class WAndBLogger:
    """Wrapper for Weights & Biases logging."""

    def __init__(self, project: str, name: str, config: dict, **kwargs):
        """
        Initialize W&B run.

        Args:
            project: W&B project name
            name: Run name
            config: Configuration dict to log
            **kwargs: Additional arguments for wandb.init
        """
        import wandb

        wandb.init(project=project, name=name, config=config, **kwargs)
        self.run = wandb.run

    def log(self, metrics: dict, step: int = None):
        """Log metrics."""
        wandb.log(metrics, step=step)

    def log_image(self, key: str, image, caption: str = None):
        """Log an image."""
        wandb.log({key: wandb.Image(image, caption=caption)})

    def log_mesh(self, key: str, vertices, faces, colors=None):
        """Log a 3D mesh."""
        # W&B expects vertices as (N, 3), faces as (M, 3)
        wandb.log({key: wandb.Object3D({"vertices": vertices, "faces": faces})})

    def finish(self):
        """Finish the run."""
        wandb.finish()
