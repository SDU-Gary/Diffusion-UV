"""
MA-IUVF Configuration System

Hierarchical configuration for MA-IUVF experiments with YAML support.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import yaml
import json


@dataclass
class ExperimentMetadata:
    """Experiment metadata"""
    name: str = "maiuvf_experiment"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    version: str = "1.0"


@dataclass
class PathsConfig:
    """Input/output paths configuration"""
    input_mesh: str = ""
    texture: str = ""
    output_dir: str = ""


@dataclass
class BakingConfig:
    """Data baking configuration"""
    num_samples: int = 10000
    chart_mode: str = "uv_islands"
    use_dynamic_sampling: bool = False
    virtual_epoch_size: int = 300000
    sigma_ratio: float = 0.01


@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 100
    batch_size: int = 8192
    learning_rate: float = 1e-3
    device: str = "cuda"


@dataclass
class HashGridConfig:
    """Hash grid configuration"""
    learning_rate: Optional[float] = None
    num_levels: int = 16
    features_per_level: int = 2
    log2_size: int = 19
    base_resolution: int = 16
    max_resolution: int = 2048
    cuda_backend: str = "auto"
    weight_decay: float = 1e-6


@dataclass
class MLPConfig:
    """MLP configuration"""
    weight_decay: float = 0.0


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    encoder_type: str = "bspline_hash"
    activation: str = "silu"
    hidden_dim: int = 64
    num_layers: int = 2
    positional_encoding_freqs: int = 8
    hash_grid: HashGridConfig = field(default_factory=HashGridConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)


@dataclass
class LossWeightsConfig:
    """Loss weights configuration"""
    metric: float = 0.01
    anchor: float = 1.0
    classification: float = 1.0
    centroid: float = 0.0
    unified: float = 0.0


@dataclass
class TargetWeightsConfig:
    """Target weights for two-stage scheduling"""
    metric: float = 1.0
    anchor: float = 0.01
    classification: float = 0.1


@dataclass
class ClassificationCutoffConfig:
    """Hard classification cutoff configuration"""
    epoch: int = 20
    value: float = 0.0


@dataclass
class UnifiedLossConfig:
    """Unified local loss configuration"""
    num_neighbors: int = 4
    epsilon: float = 0.01


@dataclass
class LossScheduleConfig:
    """Loss scheduling configuration"""
    strategy: str = "two_stage"
    phase_a_epochs: int = 30
    ramp: str = "cosine"
    target_weights: TargetWeightsConfig = field(default_factory=TargetWeightsConfig)
    classification_cutoff: ClassificationCutoffConfig = field(default_factory=ClassificationCutoffConfig)
    keep_anchor_constant: bool = True
    unified: UnifiedLossConfig = field(default_factory=UnifiedLossConfig)


@dataclass
class LossConfig:
    """Complete loss configuration"""
    weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)
    schedule: LossScheduleConfig = field(default_factory=LossScheduleConfig)


@dataclass
class DynamicSamplingConfig:
    """Dynamic sampling configuration"""
    enabled: bool = False
    virtual_epoch_size: int = 300000
    sigma_ratio: float = 0.01


@dataclass
class RenderingConfig:
    """Rendering configuration"""
    target_faces: int = 500
    mode: str = "cpu"
    resolution: int = 512


@dataclass
class SystemConfig:
    """System configuration"""
    device: str = "cuda"
    quick_test: bool = False
    seed: int = 42


@dataclass
class MAIUVFConfig:
    """Complete MA-IUVF experiment configuration"""
    experiment: ExperimentMetadata = field(default_factory=ExperimentMetadata)
    paths: PathsConfig = field(default_factory=PathsConfig)
    baking: BakingConfig = field(default_factory=BakingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    dynamic_sampling: DynamicSamplingConfig = field(default_factory=DynamicSamplingConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_yaml(self, path: Union[str, Path]):
        """Save to YAML file."""
        path = Path(path)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "MAIUVFConfig":
        """Load from YAML file."""
        path = Path(path)
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MAIUVFConfig":
        """Create from dictionary."""
        # Extract nested dictionaries
        model_data = data.get("model", {})
        loss_data = data.get("loss", {})
        schedule_data = loss_data.get("schedule", {})

        return cls(
            experiment=ExperimentMetadata(**data.get("experiment", {})),
            paths=PathsConfig(**data.get("paths", {})),
            baking=BakingConfig(**data.get("baking", {})),
            training=TrainingConfig(**data.get("training", {})),
            model=ModelConfig(
                encoder_type=model_data.get("encoder_type", "bspline_hash"),
                activation=model_data.get("activation", "silu"),
                hidden_dim=model_data.get("hidden_dim", 64),
                num_layers=model_data.get("num_layers", 2),
                positional_encoding_freqs=model_data.get("positional_encoding_freqs", 8),
                hash_grid=HashGridConfig(**model_data.get("hash_grid", {})),
                mlp=MLPConfig(**model_data.get("mlp", {})),
            ),
            loss=LossConfig(
                weights=LossWeightsConfig(**loss_data.get("weights", {})),
                schedule=LossScheduleConfig(
                    strategy=schedule_data.get("strategy", "two_stage"),
                    phase_a_epochs=schedule_data.get("phase_a_epochs", 30),
                    ramp=schedule_data.get("ramp", "cosine"),
                    target_weights=TargetWeightsConfig(**schedule_data.get("target_weights", {})),
                    classification_cutoff=ClassificationCutoffConfig(**schedule_data.get("classification_cutoff", {})),
                    keep_anchor_constant=schedule_data.get("keep_anchor_constant", True),
                    unified=UnifiedLossConfig(**schedule_data.get("unified", {})),
                ),
            ),
            dynamic_sampling=DynamicSamplingConfig(**data.get("dynamic_sampling", {})),
            rendering=RenderingConfig(**data.get("rendering", {})),
            system=SystemConfig(**data.get("system", {})),
        )

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Validate paths
        if not self.paths.input_mesh:
            errors.append("paths.input_mesh is required")
        if not self.paths.texture:
            errors.append("paths.texture is required")
        if not self.paths.output_dir:
            errors.append("paths.output_dir is required")

        # Validate chart mode
        if self.baking.chart_mode not in ["uv_islands", "face_component"]:
            errors.append(f"Invalid chart_mode: {self.baking.chart_mode}")

        # Validate encoder type
        if self.model.encoder_type not in ["bspline_hash", "fourier"]:
            errors.append(f"Invalid encoder_type: {self.model.encoder_type}")

        # Validate activation
        if self.model.activation not in ["silu", "softplus", "relu"]:
            errors.append(f"Invalid activation: {self.model.activation}")

        # Validate loss schedule strategy
        if self.loss.schedule.strategy not in ["fixed", "two_stage"]:
            errors.append(f"Invalid loss.schedule.strategy: {self.loss.schedule.strategy}")

        # Validate ramp
        if self.loss.schedule.ramp not in ["cosine", "linear"]:
            errors.append(f"Invalid loss.schedule.ramp: {self.loss.schedule.ramp}")

        # Validate render mode
        if self.rendering.mode not in ["obj", "cpu"]:
            errors.append(f"Invalid rendering.mode: {self.rendering.mode}")

        # Validate device
        if self.system.device not in ["cuda", "cpu", "auto"]:
            errors.append(f"Invalid system.device: {self.system.device}")

        # Validate numerical ranges
        if self.training.epochs <= 0:
            errors.append("training.epochs must be positive")
        if self.training.batch_size <= 0:
            errors.append("training.batch_size must be positive")
        if self.training.learning_rate <= 0:
            errors.append("training.learning_rate must be positive")

        return errors


def load_maiuvf_config(path: Union[str, Path]) -> MAIUVFConfig:
    """Load MA-IUVF configuration from file."""
    path = Path(path)

    if path.suffix in [".yaml", ".yml"]:
        return MAIUVFConfig.from_yaml(path)
    elif path.suffix == ".json":
        with open(path, 'r') as f:
            data = json.load(f)
        return MAIUVFConfig.from_dict(data)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def save_maiuvf_config(config: MAIUVFConfig, path: Union[str, Path]):
    """Save MA-IUVF configuration to file."""
    path = Path(path)

    if path.suffix in [".yaml", ".yml"]:
        config.to_yaml(path)
    elif path.suffix == ".json":
        with open(path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def get_default_maiuvf_config() -> MAIUVFConfig:
    """Get default MA-IUVF configuration."""
    return MAIUVFConfig()
