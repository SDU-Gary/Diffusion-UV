"""
Core Module Initialization
"""

from .interfaces import (
    # Enums
    SamplingRegion,
    TrainingPhase,
    DiffusionSchedulerType,
    # Data Structures
    PointSample,
    PointSampleTensor,
    GeometryFeatures,
    ConditionVector,
    NetworkGOutput,
    NetworkDOutput,
    NetworkROutput,
    TrainingMetrics,
    # Interfaces (ABC)
    IMeshLoader,
    IGeometryFeatureExtractor,
    INetworkG,
    INetworkD,
    INetworkR,
    ILossFunction,
    IOptimizer,
    IScheduler,
    ITrainer,
    IEvaluator,
    # Protocol Interfaces
    HasForward,
    HasParameters,
    HasToDevice,
    # Type Aliases (not exported, use typing.Callable instead)
)

from .config import (
    DataConfig,
    NetworkGConfig,
    NetworkDConfig,
    NetworkRConfig,
    TrainingConfig,
    LossConfig,
    LoggingConfig,
    EvaluationConfig,
    ExperimentConfig,
    load_config,
    save_config,
    get_default_config,
)

__all__ = [
    # Enums
    "SamplingRegion",
    "TrainingPhase",
    "DiffusionSchedulerType",
    # Data Structures
    "PointSample",
    "PointSampleTensor",
    "GeometryFeatures",
    "ConditionVector",
    "NetworkGOutput",
    "NetworkDOutput",
    "NetworkROutput",
    "TrainingMetrics",
    # Interfaces (Abstract)
    "IMeshLoader",
    "IGeometryFeatureExtractor",
    "INetworkG",
    "INetworkD",
    "INetworkR",
    "ILossFunction",
    "IOptimizer",
    "IScheduler",
    "ITrainer",
    "IEvaluator",
    # Interfaces (Protocol)
    "HasForward",
    "HasParameters",
    "HasToDevice",
    # Config
    "DataConfig",
    "NetworkGConfig",
    "NetworkDConfig",
    "NetworkRConfig",
    "TrainingConfig",
    "LossConfig",
    "LoggingConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "load_config",
    "save_config",
    "get_default_config",
]
