"""
Weights & Biases Experiment Tracking

Comprehensive experiment tracking and training monitoring system.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from dataclasses import dataclass, field
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class LogEntry:
    """Single log entry for metrics."""
    step: int
    phase: str
    metrics: Dict[str, float]
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


class MetricsAggregator:
    """Aggregates metrics over an epoch and computes statistics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all accumulators."""
        self.values: Dict[str, List[float]] = {}
        self.counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float], batch_size: int = 1):
        """Update with new metrics."""
        for key, value in metrics.items():
            if key not in self.values:
                self.values[key] = []
                self.counts[key] = 0
            self.values[key].append(value * batch_size)
            self.counts[key] += batch_size

    def compute(self) -> Dict[str, float]:
        """Compute mean metrics."""
        result = {}
        for key in self.values:
            if self.counts[key] > 0:
                result[key] = sum(self.values[key]) / self.counts[key]
        return result


class ExperimentTracker:
    """
    Comprehensive experiment tracking with Weights & Biases.

    Features:
    - Automatic experiment versioning
    - Config logging
    - Training metrics (loss, learning rate, gradients)
    - Validation metrics
    - Model checkpointing
    - Custom visualizations (meshes, images, plots)
    - Multi-phase training support
    - Gradient and weight histograms
    """

    def __init__(
        self,
        project: str = "diffusion-uv",
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        mode: str = "online",
        log_dir: str = "./logs",
        log_interval: int = 100,
        watch_model: bool = True,
        watch_log_freq: int = 100,
    ):
        """
        Initialize experiment tracker.

        Args:
            project: W&B project name
            experiment_name: Experiment name (auto-generated if None)
            config: Configuration dict
            entity: W&B entity/team
            tags: List of tags
            notes: Experiment notes
            mode: W&B mode (online, offline, disabled)
            log_dir: Local log directory
            log_interval: Steps between logging to W&B
            watch_model: Whether to watch model gradients
            watch_log_freq: Logging frequency for gradients/weights
        """
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Run: pip install wandb")

        self.project = project
        self.entity = entity
        self.mode = mode
        self.log_dir = Path(log_dir)
        self.log_interval = log_interval
        self.watch_model = watch_model
        self.watch_log_freq = watch_log_freq

        # Auto-generate experiment name
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"exp_{timestamp}"

        self.experiment_name = experiment_name
        self.log_dir = self.log_dir / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Metrics tracking
        self.current_phase = "phase_1"
        self.global_step = 0
        self.epoch = 0
        self.metrics_aggregator = MetricsAggregator()

        # Initialize W&B
        wandb.init(
            project=project,
            name=experiment_name,
            entity=entity,
            config=config,
            tags=tags,
            notes=notes,
            mode=mode,
            dir=str(self.log_dir),
        )
        self.run = wandb.run

        # Log config summary
        if config is not None:
            self._log_config_summary(config)

    def _log_config_summary(self, config: Dict[str, Any]):
        """Log configuration summary as JSON artifact."""
        config_json = json.dumps(config, indent=2, default=str)
        (self.log_dir / "config.json").write_text(config_json)

        # Log to W&B
        wandb.define_metric("global_step")
        wandb.define_metric("epoch")

    def set_phase(self, phase: str):
        """Set current training phase."""
        self.current_phase = phase
        self.run.define_metric(f"{phase}/*", step_metric="global_step")

    def set_epoch(self, epoch: int):
        """Set current epoch."""
        self.epoch = epoch

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        phase: Optional[str] = None,
        commit: bool = True,
    ):
        """
        Log metrics to W&B.

        Args:
            metrics: Dictionary of metric names and values
            step: Global step (uses internal counter if None)
            phase: Training phase (uses current phase if None)
            commit: Whether to commit this log
        """
        if step is None:
            step = self.global_step
        if phase is None:
            phase = self.current_phase

        # Prefix metrics with phase
        prefixed_metrics = {
            f"{phase}/{key}": value for key, value in metrics.items()
        }
        prefixed_metrics["global_step"] = step
        prefixed_metrics["epoch"] = self.epoch

        wandb.log(prefixed_metrics, step=step, commit=commit)
        self.global_step = step

    def log_scalar(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
        phase: Optional[str] = None,
    ):
        """Log a single scalar."""
        if phase is not None:
            key = f"{phase}/{key}"
        wandb.log({key: value, "global_step": step or self.global_step})

    def log_scalars(
        self,
        main_tag: str,
        scalar_dict: Dict[str, float],
        step: Optional[int] = None,
        phase: Optional[str] = None,
    ):
        """Log multiple scalars."""
        if phase is not None:
            main_tag = f"{phase}/{main_tag}"
        wandb.log({main_tag: scalar_dict, "global_step": step or self.global_step})

    def log_histogram(
        self,
        key: str,
        values: Union[torch.Tensor, np.ndarray, List],
        step: Optional[int] = None,
    ):
        """Log a histogram."""
        wandb.log({key: wandb.Histogram(values), "global_step": step or self.global_step})

    def log_image(
        self,
        key: str,
        image: Union[np.ndarray, torch.Tensor],
        caption: Optional[str] = None,
        step: Optional[int] = None,
    ):
        """
        Log an image.

        Args:
            key: Image key
            image: Image as numpy array (H, W, C) or (C, H, W)
            caption: Optional caption
            step: Global step
        """
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        # Handle different tensor formats
        if image.ndim == 4:  # (B, C, H, W) or (B, H, W, C)
            image = image[0]  # Take first image

        if image.shape[0] in [1, 3]:  # (C, H, W)
            image = np.transpose(image, (1, 2, 0))

        wandb.log({key: wandb.Image(image, caption=caption), "global_step": step or self.global_step})

    def log_images(
        self,
        key: str,
        images: Union[List[np.ndarray], torch.Tensor],
        captions: Optional[List[str]] = None,
        step: Optional[int] = None,
    ):
        """Log multiple images."""
        wandb_images = []
        for i, img in enumerate(images):
            caption = captions[i] if captions and i < len(captions) else None
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            wandb_images.append(wandb.Image(img, caption=caption))
        wandb.log({key: wandb_images, "global_step": step or self.global_step})

    def log_mesh(
        self,
        key: str,
        vertices: np.ndarray,
        faces: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        step: Optional[int] = None,
    ):
        """
        Log a 3D mesh.

        Args:
            key: Mesh key
            vertices: (N, 3) vertex positions
            faces: (M, 3) face indices (optional)
            colors: (N, 3) vertex colors (optional)
            step: Global step
        """
        mesh_data = {"vertices": vertices.tolist()}
        if faces is not None:
            mesh_data["faces"] = faces.tolist()
        if colors is not None:
            mesh_data["colors"] = colors.tolist()

        wandb.log({key: wandb.Object3D(mesh_data), "global_step": step or self.global_step})

    def log_pointcloud(
        self,
        key: str,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        step: Optional[int] = None,
    ):
        """Log a point cloud."""
        point_cloud = {"type": "lidar/beta", "points": points.tolist()}
        if colors is not None:
            point_cloud["colors"] = (colors * 255).astype(np.uint8).tolist()

        wandb.log({key: wandb.Object3D(point_cloud), "global_step": step or self.global_step})

    def log_table(
        self,
        key: str,
        columns: List[str],
        data: List[List[Any]],
        step: Optional[int] = None,
    ):
        """Log a table."""
        table = wandb.Table(columns=columns, data=data)
        wandb.log({key: table, "global_step": step or self.global_step})

    def log_plot(
        self,
        key: str,
        figure: "matplotlib.figure.Figure",
        step: Optional[int] = None,
    ):
        """Log a matplotlib figure."""
        wandb.log({key: wandb.Image(figure), "global_step": step or self.global_step})

    def log_3d_scatter(
        self,
        key: str,
        points: np.ndarray,
        values: Optional[np.ndarray] = None,
        step: Optional[int] = None,
    ):
        """
        Log 3D scatter plot.

        Args:
            key: Key
            points: (N, 3) 3D points
            values: (N,) scalar values for coloring
            step: Global step
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        if values is not None:
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                               c=values, cmap='viridis')
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(key)

        wandb.log({key: wandb.Image(fig), "global_step": step or self.global_step})
        plt.close(fig)

    def watch_model_gradients(
        self,
        model: torch.nn.Module,
        log_freq: Optional[int] = None,
    ):
        """Watch model gradients."""
        if log_freq is None:
            log_freq = self.watch_log_freq
        wandb.watch(model, log_freq=log_freq, log="gradients")

    def watch_model_weights(
        self,
        model: torch.nn.Module,
        log_freq: Optional[int] = None,
    ):
        """Watch model weights."""
        if log_freq is None:
            log_freq = self.watch_log_freq
        wandb.watch(model, log_freq=log_freq, log="parameters")

    def log_model_gradients_histogram(
        self,
        model: torch.nn.Module,
        step: Optional[int] = None,
        prefix: str = "gradients",
    ):
        """Log histograms of model gradients."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                key = f"{prefix}/{name}"
                self.log_histogram(key, param.grad.detach().cpu().numpy(), step)

    def log_model_weights_histogram(
        self,
        model: torch.nn.Module,
        step: Optional[int] = None,
        prefix: str = "weights",
    ):
        """Log histograms of model weights."""
        for name, param in model.named_parameters():
            key = f"{prefix}/{name}"
            self.log_histogram(key, param.detach().cpu().numpy(), step)

    def log_learning_rate(
        self,
        lr: float,
        optimizer_name: str = "optimizer",
        step: Optional[int] = None,
    ):
        """Log learning rate."""
        key = f"hyperparameter/{optimizer_name}_lr"
        self.log_scalar(key, lr, step)

    def log_batch_metrics(
        self,
        metrics: Dict[str, float],
        batch_size: int,
        phase: Optional[str] = None,
    ):
        """
        Accumulate batch metrics for epoch aggregation.

        Args:
            metrics: Dictionary of metric names and values
            batch_size: Batch size for weighted averaging
            phase: Training phase
        """
        if phase is None:
            phase = self.current_phase

        # Add phase prefix
        prefixed = {f"{phase}/{key}": value for key, value in metrics.items()}

        # Update aggregator
        self.metrics_aggregator.update(prefixed, batch_size)

    def commit_epoch_metrics(self, step: Optional[int] = None):
        """
        Compute and log aggregated epoch metrics.

        Args:
            step: Global step
        """
        if step is None:
            step = self.global_step

        metrics = self.metrics_aggregator.compute()
        if metrics:
            self.log_metrics(metrics, step=step, commit=True)

        self.metrics_aggregator.reset()

    def log_training_step(
        self,
        step: int,
        loss: float,
        lr: float,
        phase: str,
        metrics: Optional[Dict[str, float]] = None,
        log_gradients: bool = False,
        model: Optional[torch.nn.Module] = None,
    ):
        """
        Log training step.

        Args:
            step: Global step
            loss: Training loss
            lr: Learning rate
            phase: Training phase
            metrics: Additional metrics
            log_gradients: Whether to log gradient histograms
            model: Model for gradient logging
        """
        log_dict = {
            f"{phase}/loss": loss,
            f"{phase}/learning_rate": lr,
        }

        if metrics:
            log_dict.update({f"{phase}/{k}": v for k, v in metrics.items()})

        wandb.log(log_dict, step=step)

        if log_gradients and model is not None and step % 1000 == 0:
            self.log_model_gradients_histogram(model, step)

    def log_validation(
        self,
        metrics: Dict[str, float],
        step: int,
        phase: Optional[str] = None,
    ):
        """
        Log validation metrics.

        Args:
            metrics: Validation metrics dictionary
            step: Global step
            phase: Training phase
        """
        if phase is None:
            phase = self.current_phase

        prefixed = {f"validation/{k}": v for k, v in metrics.items()}
        prefixed["global_step"] = step
        prefixed["epoch"] = self.epoch

        wandb.log(prefixed, step=step, commit=True)

    def log_checkpoint(
        self,
        path: str,
        metric: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log model checkpoint.

        Args:
            path: Path to checkpoint file
            metric: Optional metric value
            metadata: Optional metadata dictionary
        """
        artifact = wandb.Artifact(
            name=f"{self.experiment_name}_checkpoint",
            type="model",
            metadata=metadata or {},
        )
        artifact.add_file(path)

        if metric is not None:
            artifact.metadata["metric"] = metric

        self.run.log_artifact(artifact)

    def save_checkpoint(
        self,
        path: str,
        state: Dict[str, Any],
        metric: Optional[float] = None,
    ):
        """
        Save checkpoint and log to W&B.

        Args:
            path: Path to save checkpoint
            state: State dict to save
            metric: Optional metric for best model tracking
        """
        import os

        # Save locally
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)

        # Log to W&B
        self.log_checkpoint(path, metric=metric, metadata={"epoch": self.epoch})

    def finish(self):
        """Finish the experiment."""
        wandb.finish()

    @property
    def run_url(self) -> str:
        """Get W&B run URL."""
        return self.run.url

    @property
    def run_id(self) -> str:
        """Get W&B run ID."""
        return self.run.id

    @property
    def run_name(self) -> str:
        """Get W&B run name."""
        return self.run.name


class PhaseSpecificLogger:
    """
    Logger with phase-specific configuration.

    Handles different logging requirements for each training phase.
    """

    def __init__(
        self,
        tracker: ExperimentTracker,
        phase1_interval: int = 100,
        phase2_interval: int = 50,
        phase3_interval: int = 50,
    ):
        """
        Initialize phase-specific logger.

        Args:
            tracker: Base experiment tracker
            phase1_interval: Logging interval for phase 1
            phase2_interval: Logging interval for phase 2
            phase3_interval: Logging interval for phase 3
        """
        self.tracker = tracker
        self.intervals = {
            "phase_1": phase1_interval,
            "phase_2": phase2_interval,
            "phase_3": phase3_interval,
        }

    @property
    def current_interval(self) -> int:
        """Get current phase's logging interval."""
        return self.intervals.get(self.tracker.current_phase, 100)

    def should_log(self, step: int) -> bool:
        """Check if we should log at this step."""
        return step % self.current_interval == 0

    def log_phase_specific(
        self,
        step: int,
        metrics: Dict[str, float],
        model: Optional[torch.nn.Module] = None,
        log_gradients: bool = False,
    ):
        """Log metrics with phase-specific interval."""
        if self.should_log(step):
            self.tracker.log_metrics(metrics, step=step)

            if log_gradients and model is not None:
                self.tracker.log_model_gradients_histogram(model, step)
