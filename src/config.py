"""
Configuration Management

Handles:
- Experiment configurations (YAML-based)
- Hyperparameter management
- Model architecture specifications
- Training schedules
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import yaml
import json


@dataclass
class DataConfig:
    """
    Data configuration.

    Attributes:
        high_mesh_path: Path to high-poly mesh
        low_mesh_path: Path to low-poly mesh
        texture_path: Path to texture image
        cache_dir: Directory for cached features
        num_samples_per_epoch: Number of samples per epoch
        sampling_ratios: Ratios for each sampling region
    """
    high_mesh_path: str = ""
    low_mesh_path: str = ""
    texture_path: str = ""
    cache_dir: str = "./cache"

    # Sampling configuration
    num_samples_per_epoch: int = 2_000_000
    sampling_ratios: Dict[str, float] = field(default_factory=lambda: {
        "surface": 0.4,
        "near_surface": 0.4,
        "exterior": 0.1,
        "interior": 0.1,
    })

    # Near-surface epsilon (relative to bounding box)
    near_surface_epsilon: float = 0.01

    # Gaussian filter sigma for low-pass color
    lowpass_sigma: float = 5.0  # pixels


@dataclass
class NetworkGConfig:
    """
    Network G (Geometry Network) configuration.

    Attributes:
        hidden_dim: Hidden layer dimension
        num_layers: Number of MLP layers
        positional_encoding_freqs: Number of frequency bands
        skip_connection_layer: Layer index for skip connection
        sdf_output_range: Range for SDF output (tanh)
    """
    hidden_dim: int = 256
    num_layers: int = 8
    positional_encoding_freqs: int = 6  # L_pos
    skip_connection_layer: int = 4

    # Output configuration
    sdf_output_range: float = 1.0  # tanh range

    # Positional encoding
    include_raw_input: bool = True

    # Estimated parameter count: ~0.8M


@dataclass
class NetworkDConfig:
    """
    Network D (Diffusion Model) configuration.

    Attributes:
        condition_dim: Dimension of condition vector
        num_diffusion_steps: Number of diffusion steps (T)
        inference_steps: Number of inference steps (DDIM)
        hidden_channels: Base number of channels
        num_down_layers: Number of downsampling layers
        scheduler_type: Noise scheduler type
        use_ema: Whether to use EMA
        ema_decay: EMA decay rate
    """
    # Condition vector dimension (fixed by design)
    condition_dim: int = 42

    # Diffusion configuration
    num_diffusion_steps: int = 1000  # T for training
    inference_steps: int = 20  # T_inf for DDIM
    scheduler_type: str = "linear"

    # UNet configuration
    hidden_channels: int = 64
    num_down_layers: int = 3
    num_res_blocks: int = 2

    # FiLM conditioning
    use_film: bool = True
    film_hidden_dim: int = 512

    # EMA configuration
    use_ema: bool = True
    ema_decay: float = 0.9999

    # Estimated parameter count: ~4M


@dataclass
class NetworkRConfig:
    """
    Network R (Reverse Mapping Network) configuration.

    Attributes:
        hidden_dim: Hidden layer dimension
        num_layers: Number of layers
        num_classes: Number of geometry-texture joint classes (K)
        positional_encoding_freqs: Number of frequency bands
    """
    hidden_dim: int = 256
    num_layers: int = 4
    num_classes: int = 32  # K from K-means

    # Positional encoding for (position + color) = 6 dims
    positional_encoding_freqs: int = 6

    # Estimated parameter count: ~50K


@dataclass
class TrainingConfig:
    """
    Training configuration.

    Attributes:
        phase1_epochs: Epochs for phase 1 (G only)
        phase2_epochs: Epochs for phase 2 (D only)
        phase3_epochs: Epochs for phase 3 (joint)
        batch_size_phase1: Batch size for phase 1
        batch_size_phase2: Batch size for phase 2
        batch_size_phase3: Batch size for phase 3
        learning_rate_g: Learning rate for network G
        learning_rate_d: Learning rate for network D
        learning_rate_r: Learning rate for network R
        optimizer: Optimizer type
        weight_decay: Weight decay for AdamW
        gradient_clip_norm: Gradient clipping norm
        mixed_precision: Whether to use mixed precision
    """
    # Phase durations
    phase1_epochs: int = 500
    phase2_epochs: int = 200
    phase3_epochs: int = 100

    # Batch sizes (different for each phase due to memory)
    batch_size_phase1: int = 65536
    batch_size_phase2: int = 32768
    batch_size_phase3: int = 16384

    # Learning rates
    learning_rate_g: float = 5e-4
    learning_rate_d: float = 1e-4
    learning_rate_r: float = 1e-4

    # Optimizer settings
    optimizer: str = "adam"  # "adam" or "adamw"
    weight_decay: float = 0.0  # Only for AdamW

    # Gradient clipping
    gradient_clip_norm: float = 1.0

    # Mixed precision
    mixed_precision: bool = True

    # Loss weights
    lambda_sdf: float = 1.0
    lambda_eikonal: float = 0.1
    lambda_color_base: float = 1.0
    lambda_diffusion: float = 1.0
    lambda_reverse_start: float = 0.1
    lambda_reverse_end: float = 0.5
    lambda_entropy_start: float = 0.0
    lambda_entropy_end: float = 0.05


@dataclass
class LossConfig:
    """
    Loss configuration.

    Attributes:
        lambda_sdf: Weight for SDF loss
        lambda_eikonal: Weight for Eikonal regularization
        lambda_color_base: Weight for base color loss
        lambda_diffusion: Weight for diffusion loss
        lambda_reverse: Weight for reverse mapping loss
        lambda_entropy: Weight for entropy regularization
    """
    lambda_sdf: float = 1.0
    lambda_eikonal: float = 0.1
    lambda_color_base: float = 1.0
    lambda_diffusion: float = 1.0
    lambda_reverse: float = 0.1  # Will increase during phase 3
    lambda_entropy: float = 0.0  # Will increase during phase 3


@dataclass
class LoggingConfig:
    """
    Logging configuration.

    Attributes:
        log_dir: Directory for logs
        experiment_name: Experiment name
        use_wandb: Whether to use Weights & Biases
        use_tensorboard: Whether to use TensorBoard
        wandb_project: W&B project name
        log_interval: Logging interval (steps)
        render_interval: Render interval (epochs)
        save_interval: Save checkpoint interval (epochs)
    """
    log_dir: str = "./logs"
    experiment_name: str = "diffusion_uv"

    use_wandb: bool = True
    use_tensorboard: bool = True
    wandb_project: str = "diffusion-uv"
    wandb_mode: str = "online"  # "online", "offline", or "disabled"

    log_interval: int = 100  # steps
    render_interval: int = 10  # epochs
    save_interval: int = 50  # epochs


@dataclass
class EvaluationConfig:
    """
    Evaluation configuration.

    Attributes:
        eval_interval: Evaluation interval (epochs)
        num_render_views: Number of views for rendering
        render_resolution: Render resolution
        output_dir: Output directory for results
    """
    eval_interval: int = 10
    num_render_views: int = 8
    render_resolution: int = 512
    output_dir: str = "./outputs"


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.

    Combines all sub-configurations.
    """
    data: DataConfig = field(default_factory=DataConfig)
    network_g: NetworkGConfig = field(default_factory=NetworkGConfig)
    network_d: NetworkDConfig = field(default_factory=NetworkDConfig)
    network_r: NetworkRConfig = field(default_factory=NetworkRConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Random seed
    seed: int = 42

    # Device
    device: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_yaml(self, path: str):
        """Save to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        return cls(
            data=DataConfig(**data.get("data", {})),
            network_g=NetworkGConfig(**data.get("network_g", {})),
            network_d=NetworkDConfig(**data.get("network_d", {})),
            network_r=NetworkRConfig(**data.get("network_r", {})),
            training=TrainingConfig(**data.get("training", {})),
            loss=LossConfig(**data.get("loss", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {})),
            seed=data.get("seed", 42),
            device=data.get("device", "auto"),
        )


def load_config(path: Union[str, Path]) -> ExperimentConfig:
    """
    Load configuration from file.

    Supports YAML and JSON formats.

    Args:
        path: Path to configuration file

    Returns:
        ExperimentConfig instance
    """
    path = Path(path)

    if path.suffix in [".yaml", ".yml"]:
        return ExperimentConfig.from_yaml(str(path))
    elif path.suffix == ".json":
        with open(path, "r") as f:
            data = json.load(f)
        return ExperimentConfig.from_dict(data)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def save_config(config: ExperimentConfig, path: Union[str, Path]):
    """
    Save configuration to file.

    Args:
        config: ExperimentConfig instance
        path: Path to save
    """
    path = Path(path)

    if path.suffix in [".yaml", ".yml"]:
        config.to_yaml(str(path))
    elif path.suffix == ".json":
        with open(path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def get_default_config() -> ExperimentConfig:
    """
    Get default configuration.

    Returns:
        ExperimentConfig with default values
    """
    return ExperimentConfig()
