"""
Core Type Definitions and Interfaces

This module defines the abstract interfaces and data structures
that connect all components of the system.

Key Design Principles:
1. All interfaces are abstract base classes (ABC) for clear contracts
2. Data structures use dataclass for immutability and type safety
3. Tensor shapes are documented in docstrings
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Optional,
    Tuple,
    List,
    Dict,
    Any,
    Union,
    Callable,
    Protocol,
    runtime_checkable,
)
from enum import Enum
import torch
import numpy as np


# =============================================================================
# Enumerations
# =============================================================================


class SamplingRegion(Enum):
    """Sampling regions for training data."""
    SURFACE = "surface"           # On high-mesh surface (s=0)
    NEAR_SURFACE = "near_surface"  # Near surface (|s| < epsilon)
    EXTERIOR = "exterior"         # Outside surface (s > epsilon)
    INTERIOR = "interior"         # Inside surface (s < -epsilon)


class TrainingPhase(Enum):
    """Training phases for progressive training."""
    PHASE_1 = "phase_1"  # Train G only (SDF + low-freq color)
    PHASE_2 = "phase_2"  # Train D only (diffusion, G frozen)
    PHASE_3 = "phase_3"  # Joint fine-tuning (G + D + R)


class DiffusionSchedulerType(Enum):
    """Diffusion noise scheduler types."""
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    COSINE = "cosine"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class PointSample:
    """
    Single point sample for training.

    Attributes:
        position: 3D position, shape (3,)
        sdf_gt: Ground truth signed distance, scalar
        color_gt: Ground truth color from texture, shape (3,)
        color_lowpass: Low-pass filtered color, shape (3,)
        curvature: Principal curvatures (kappa1, kappa2), shape (2,)
        normal: Surface normal, shape (3,)
        boundary_distance: Geodesic distance to UV chart boundary, scalar
        label_gt: Ground truth geometry-texture joint label, integer
        region: Which sampling region this point belongs to
    """
    position: np.ndarray
    sdf_gt: float
    color_gt: np.ndarray
    color_lowpass: np.ndarray
    curvature: np.ndarray
    normal: np.ndarray
    boundary_distance: float
    label_gt: int
    region: SamplingRegion

    def to_tensor(self, device: torch.device = None) -> "PointSampleTensor":
        """Convert to tensor representation."""
        device = device or torch.device("cpu")
        return PointSampleTensor(
            position=torch.from_numpy(self.position).float().to(device),
            sdf_gt=torch.tensor(self.sdf_gt).float().to(device),
            color_gt=torch.from_numpy(self.color_gt).float().to(device),
            color_lowpass=torch.from_numpy(self.color_lowpass).float().to(device),
            curvature=torch.from_numpy(self.curvature).float().to(device),
            normal=torch.from_numpy(self.normal).float().to(device),
            boundary_distance=torch.tensor(self.boundary_distance).float().to(device),
            label_gt=torch.tensor(self.label_gt).long().to(device),
            region=self.region,
        )


@dataclass
class PointSampleTensor:
    """
    Batched point samples in tensor form.

    All tensors have shape (batch_size, ...) except region which is a list.
    """
    position: torch.Tensor          # (B, 3)
    sdf_gt: torch.Tensor            # (B,)
    color_gt: torch.Tensor          # (B, 3)
    color_lowpass: torch.Tensor     # (B, 3)
    curvature: torch.Tensor         # (B, 2)
    normal: torch.Tensor            # (B, 3)
    boundary_distance: torch.Tensor # (B,)
    label_gt: torch.Tensor          # (B,)
    region: SamplingRegion

    @property
    def batch_size(self) -> int:
        return self.position.shape[0]

    def to(self, device: torch.device) -> "PointSampleTensor":
        """Move all tensors to device."""
        return PointSampleTensor(
            position=self.position.to(device),
            sdf_gt=self.sdf_gt.to(device),
            color_gt=self.color_gt.to(device),
            color_lowpass=self.color_lowpass.to(device),
            curvature=self.curvature.to(device),
            normal=self.normal.to(device),
            boundary_distance=self.boundary_distance.to(device),
            label_gt=self.label_gt.to(device),
            region=self.region,
        )


@dataclass
class GeometryFeatures:
    """
    Geometric features extracted from SDF or mesh.

    Attributes:
        sdf: Signed distance value
        normal: Surface normal (from SDF gradient)
        curvature: Principal curvatures (kappa1, kappa2)
        boundary_distance: Distance to UV chart boundary
        global_shape_code: Global shape encoding from PointNet++
    """
    sdf: torch.Tensor              # (B, 1) or (B,)
    normal: torch.Tensor           # (B, 3)
    curvature: torch.Tensor        # (B, 2)
    boundary_distance: torch.Tensor  # (B, 1) or (B,)
    global_shape_code: torch.Tensor  # (B, 32)


@dataclass
class ConditionVector:
    """
    Condition vector for diffusion model.

    Total dimension: 3 + 1 + 2 + 3 + 1 + 32 = 42

    Attributes:
        color_base: Low-frequency base color from network G
        sdf: Signed distance from network G
        curvature: Principal curvatures
        normal: Surface normal
        boundary_distance: Distance to UV chart boundary
        global_shape_code: Global shape encoding
    """
    color_base: torch.Tensor       # (B, 3)
    sdf: torch.Tensor              # (B, 1) or (B,)
    curvature: torch.Tensor        # (B, 2)
    normal: torch.Tensor           # (B, 3)
    boundary_distance: torch.Tensor  # (B, 1) or (B,)
    global_shape_code: torch.Tensor  # (B, 32)

    def to_tensor(self) -> torch.Tensor:
        """
        Concatenate all features into a single condition vector.

        Returns:
            Condition vector of shape (B, 42)
        """
        # Ensure correct shapes
        sdf = self.sdf.view(-1, 1) if self.sdf.dim() == 1 else self.sdf
        bd = self.boundary_distance.view(-1, 1) if self.boundary_distance.dim() == 1 else self.boundary_distance

        return torch.cat([
            self.color_base,      # (B, 3)
            sdf,                  # (B, 1)
            self.curvature,       # (B, 2)
            self.normal,          # (B, 3)
            bd,                   # (B, 1)
            self.global_shape_code,  # (B, 32)
        ], dim=-1)  # (B, 42)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "ConditionVector":
        """
        Parse condition vector tensor into structured form.

        Args:
            tensor: Condition vector of shape (B, 42)

        Returns:
            ConditionVector instance
        """
        return cls(
            color_base=tensor[:, :3],
            sdf=tensor[:, 3:4],
            curvature=tensor[:, 4:6],
            normal=tensor[:, 6:9],
            boundary_distance=tensor[:, 9:10],
            global_shape_code=tensor[:, 10:42],
        )


@dataclass
class NetworkGOutput:
    """
    Output from Network G (Geometry Network).

    Attributes:
        sdf: Predicted signed distance
        color_base: Predicted low-frequency base color
        normal: Computed normal from SDF gradient (optional)
    """
    sdf: torch.Tensor          # (B, 1) or (B,)
    color_base: torch.Tensor   # (B, 3)
    normal: Optional[torch.Tensor] = None  # (B, 3)


@dataclass
class NetworkDOutput:
    """
    Output from Network D (Diffusion Model).

    Attributes:
        color_pred: Predicted high-fidelity color
        noise_pred: Predicted noise (during training)
    """
    color_pred: torch.Tensor   # (B, 3)
    noise_pred: Optional[torch.Tensor] = None  # (B, 3)


@dataclass
class NetworkROutput:
    """
    Output from Network R (Reverse Mapping Network).

    Attributes:
        logits: Class logits
        probs: Class probabilities (after softmax)
    """
    logits: torch.Tensor       # (B, K) where K is number of classes
    probs: torch.Tensor        # (B, K)


@dataclass
class TrainingMetrics:
    """
    Metrics collected during training.

    All values are scalars (float).
    """
    # Geometry losses
    loss_sdf: float = 0.0
    loss_eikonal: float = 0.0
    loss_color_base: float = 0.0

    # Diffusion losses
    loss_diffusion: float = 0.0

    # Reverse mapping losses
    loss_reverse: float = 0.0
    loss_entropy: float = 0.0

    # Combined loss
    loss_total: float = 0.0

    # Metrics
    psnr: float = 0.0
    ssim: float = 0.0
    rmse: float = 0.0
    label_accuracy: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "loss_sdf": self.loss_sdf,
            "loss_eikonal": self.loss_eikonal,
            "loss_color_base": self.loss_color_base,
            "loss_diffusion": self.loss_diffusion,
            "loss_reverse": self.loss_reverse,
            "loss_entropy": self.loss_entropy,
            "loss_total": self.loss_total,
            "psnr": self.psnr,
            "ssim": self.ssim,
            "rmse": self.rmse,
            "label_accuracy": self.label_accuracy,
        }


# =============================================================================
# Abstract Interfaces
# =============================================================================


class IMeshLoader(ABC):
    """
    Interface for loading and processing 3D meshes.

    Implementations should handle:
    - OBJ, PLY, GLTF formats
    - UV coordinate extraction
    - Texture loading
    """

    @abstractmethod
    def load_high_mesh(self, path: str) -> Dict[str, Any]:
        """
        Load high-poly mesh with UV and texture.

        Args:
            path: Path to mesh file

        Returns:
            Dict containing:
                - vertices: (V, 3) array
                - faces: (F, 3) array
                - uvs: (V, 2) array
                - texture: (H, W, 3) array
                - normals: (V, 3) array (optional)
        """
        pass

    @abstractmethod
    def load_low_mesh(self, path: str) -> Dict[str, Any]:
        """
        Load low-poly mesh (geometry only, no UV).

        Args:
            path: Path to mesh file

        Returns:
            Dict containing:
                - vertices: (V, 3) array
                - faces: (F, 3) array
                - normals: (V, 3) array (optional)
        """
        pass

    @abstractmethod
    def sample_surface_points(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        num_points: int,
    ) -> np.ndarray:
        """
        Sample points uniformly on mesh surface.

        Args:
            vertices: (V, 3) vertex positions
            faces: (F, 3) face indices
            num_points: Number of points to sample

        Returns:
            (num_points, 3) sampled positions
        """
        pass


class IGeometryFeatureExtractor(ABC):
    """
    Interface for extracting geometric features.

    Features include:
    - SDF values
    - Surface normals
    - Principal curvatures
    - Geodesic distances to boundaries
    - Global shape encoding
    """

    @abstractmethod
    def compute_sdf(
        self,
        points: torch.Tensor,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute signed distance field values.

        Args:
            points: (B, 3) query points
            vertices: (V, 3) mesh vertices
            faces: (F, 3) mesh faces

        Returns:
            (B,) SDF values
        """
        pass

    @abstractmethod
    def compute_normal(
        self,
        points: torch.Tensor,
        sdf: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute surface normal from SDF gradient.

        Args:
            points: (B, 3) query points
            sdf: (B,) SDF values

        Returns:
            (B, 3) normal vectors
        """
        pass

    @abstractmethod
    def compute_curvature(
        self,
        points: torch.Tensor,
        sdf: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute principal curvatures from SDF Hessian.

        Args:
            points: (B, 3) query points
            sdf: (B,) SDF values

        Returns:
            (B, 2) principal curvatures (kappa1, kappa2)
        """
        pass

    @abstractmethod
    def compute_boundary_distance(
        self,
        points: torch.Tensor,
        uv_boundaries: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute geodesic distance to UV chart boundaries.

        Args:
            points: (B, 3) query points
            uv_boundaries: UV chart boundary points

        Returns:
            (B,) boundary distances
        """
        pass

    @abstractmethod
    def compute_global_shape_code(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute global shape encoding using PointNet++.

        Args:
            vertices: (V, 3) mesh vertices
            faces: (F, 3) mesh faces

        Returns:
            (32,) global shape code
        """
        pass

    def extract_all_features(
        self,
        points: torch.Tensor,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        uv_boundaries: Optional[torch.Tensor] = None,
    ) -> GeometryFeatures:
        """
        Extract all geometric features.

        Args:
            points: (B, 3) query points
            vertices: (V, 3) mesh vertices
            faces: (F, 3) mesh faces
            uv_boundaries: Optional UV boundary points

        Returns:
            GeometryFeatures containing all extracted features
        """
        sdf = self.compute_sdf(points, vertices, faces)
        normal = self.compute_normal(points, sdf)
        curvature = self.compute_curvature(points, sdf)

        if uv_boundaries is not None:
            boundary_distance = self.compute_boundary_distance(points, uv_boundaries)
        else:
            boundary_distance = torch.zeros(points.shape[0], device=points.device)

        global_shape_code = self.compute_global_shape_code(vertices, faces)

        return GeometryFeatures(
            sdf=sdf,
            normal=normal,
            curvature=curvature,
            boundary_distance=boundary_distance,
            global_shape_code=global_shape_code,
        )


class INetworkG(ABC):
    """
    Interface for Network G (Geometry Network).

    Predicts SDF and low-frequency base color for 3D points.
    """

    @abstractmethod
    def forward(self, positions: torch.Tensor) -> NetworkGOutput:
        """
        Forward pass.

        Args:
            positions: (B, 3) 3D positions

        Returns:
            NetworkGOutput with sdf and color_base
        """
        pass

    @abstractmethod
    def get_sdf(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Get SDF values only (for inference).

        Args:
            positions: (B, 3) 3D positions

        Returns:
            (B,) SDF values
        """
        pass

    @abstractmethod
    def get_color_base(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Get base colors only (for inference).

        Args:
            positions: (B, 3) 3D positions

        Returns:
            (B, 3) base colors
        """
        pass


class INetworkD(ABC):
    """
    Interface for Network D (Diffusion Model).

    Generates high-fidelity colors conditioned on geometry.
    """

    @abstractmethod
    def forward(
        self,
        noisy_color: torch.Tensor,
        timestep: torch.Tensor,
        condition: ConditionVector,
    ) -> torch.Tensor:
        """
        Predict noise given noisy color and condition.

        Args:
            noisy_color: (B, 3) noisy color
            timestep: (B,) diffusion timestep
            condition: ConditionVector

        Returns:
            (B, 3) predicted noise
        """
        pass

    @abstractmethod
    def sample(
        self,
        condition: ConditionVector,
        num_steps: int = 20,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """
        Sample colors using DDIM.

        Args:
            condition: ConditionVector
            num_steps: Number of denoising steps
            deterministic: If True, use deterministic sampling

        Returns:
            (B, 3) sampled colors
        """
        pass

    @abstractmethod
    def add_noise(
        self,
        color: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to clean color (forward diffusion).

        Args:
            color: (B, 3) clean color
            timestep: (B,) timestep

        Returns:
            noisy_color, noise
        """
        pass


class INetworkR(ABC):
    """
    Interface for Network R (Reverse Mapping Network).

    Predicts geometry-texture joint labels from position and color.
    """

    @abstractmethod
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
        pass

    @abstractmethod
    def get_label(self, positions: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
        """
        Get predicted label (argmax).

        Args:
            positions: (B, 3) 3D positions
            colors: (B, 3) colors

        Returns:
            (B,) predicted labels
        """
        pass


class ILossFunction(ABC):
    """Interface for loss functions."""

    @abstractmethod
    def compute(
        self,
        prediction: Any,
        target: Any,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute loss.

        Args:
            prediction: Model output
            target: Ground truth
            **kwargs: Additional arguments

        Returns:
            Scalar loss tensor
        """
        pass


class IOptimizer(ABC):
    """Interface for optimizers (wrapper around torch.optim)."""

    @abstractmethod
    def step(self, loss: torch.Tensor):
        """Perform optimization step."""
        pass

    @abstractmethod
    def zero_grad(self):
        """Zero gradients."""
        pass


class IScheduler(ABC):
    """Interface for learning rate schedulers."""

    @abstractmethod
    def step(self, metrics: Optional[float] = None):
        """Update learning rate."""
        pass

    @abstractmethod
    def get_lr(self) -> float:
        """Get current learning rate."""
        pass


class ITrainer(ABC):
    """
    Interface for training loops.

    Handles the three-phase progressive training.
    """

    @abstractmethod
    def train_phase1(self, num_epochs: int):
        """Train network G only."""
        pass

    @abstractmethod
    def train_phase2(self, num_epochs: int):
        """Train network D (G frozen)."""
        pass

    @abstractmethod
    def train_phase3(self, num_epochs: int):
        """Joint fine-tuning of G, D, R."""
        pass

    @abstractmethod
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        pass

    @abstractmethod
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        pass


class IEvaluator(ABC):
    """Interface for evaluation."""

    @abstractmethod
    def evaluate(
        self,
        network_g: INetworkG,
        network_d: INetworkD,
        network_r: INetworkR,
        low_mesh: Dict[str, Any],
        high_mesh: Dict[str, Any],
    ) -> TrainingMetrics:
        """
        Evaluate models on low-mesh.

        Args:
            network_g: Trained geometry network
            network_d: Trained diffusion network
            network_r: Trained reverse mapping network
            low_mesh: Low-poly mesh data
            high_mesh: High-poly mesh data (reference)

        Returns:
            TrainingMetrics with evaluation results
        """
        pass

    @abstractmethod
    def render_comparison(
        self,
        low_mesh_colored: Dict[str, Any],
        high_mesh: Dict[str, Any],
        output_path: str,
    ):
        """
        Render side-by-side comparison.

        Args:
            low_mesh_colored: Low-poly mesh with predicted colors
            high_mesh: High-poly mesh with ground truth
            output_path: Path to save rendered image
        """
        pass


# =============================================================================
# Protocol Interfaces (for duck typing)
# =============================================================================


@runtime_checkable
class HasForward(Protocol):
    """Protocol for anything with a forward method."""
    def forward(self, *args, **kwargs) -> Any: ...


@runtime_checkable
class HasParameters(Protocol):
    """Protocol for anything with parameters()."""
    def parameters(self) -> Any: ...


@runtime_checkable
class HasToDevice(Protocol):
    """Protocol for anything that can be moved to device."""
    def to(self, device: torch.device) -> Any: ...
