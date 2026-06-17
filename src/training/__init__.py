"""
Training Module

Contains implementations of:
- Loss functions
- Optimizers
- Learning rate schedulers
- Training loops
- Checkpoint management
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Any, Callable, List, Tuple
from pathlib import Path

from src.interfaces import (
    ILossFunction,
    IOptimizer,
    IScheduler,
    TrainingMetrics,
)

from src.config import LossConfig


# =============================================================================
# Loss Functions
# =============================================================================


class BaseLoss(nn.Module, ABC):
    """Base class for loss functions."""

    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config


class SDFLoss(BaseLoss):
    """
    SDF loss (KL divergence between SDF distributions).

    用于监督网络 G 对 SDF 的预测。
    """

    def forward(
        self,
        pred_sdf: torch.Tensor,
        target_sdf: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SDF loss.

        Args:
            pred_sdf: (B,) predicted SDF values
            target_sdf: (B,) ground truth SDF values

        Returns:
            SDF loss (MSE)
        """
        return torch.mean((pred_sdf - target_sdf) ** 2)


class EikonalLoss(BaseLoss):
    """
    Eikonal regularization loss.

    Ensures gradient of SDF is L2-norm 1 (except on surface where |∇s| = 0).

    L_eikonal = E[||∇s(x)||_2^2 - 1]^2
    """

    def __init__(
        self,
        config: LossConfig,
    ):
        super().__init__(config)

    def forward(
        self,
        sdf_values: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Eikonal loss.

        Args:
            sdf_values: (B,) SDF values
            positions: (B, 3) 3D positions

        Returns:
            Eikonal regularization loss
        """
        # Compute gradient of SDF w.r.t. positions
        grad = torch.autograd.grad(
            outputs=sdf_values.sum(),  # Scalar output for backward
            inputs=positions,
            create_graph=True,
        )[0]  # Shape: (B, 3)

        # ||∇s(x)||_2^2
        grad_norm_sq = torch.sum(grad ** 2, dim=-1)  # Shape: (B,)

        # Eikonal constraint: ||∇s|| should be 1
        eikonal_loss = (grad_norm_sq - 1.0) ** 2

        # Only penalize non-surface points
        surface_threshold = 0.01
        non_surface_mask = (torch.abs(sdf_values) > surface_threshold).float()
        loss = (eikonal_loss * non_surface_mask).mean()

        return loss


class BaseColorLoss(BaseLoss):
    """Base class for color losses."""

    def __init__(self, config: LossConfig):
        super().__init__(config)

    def reconstruction_loss(
        self,
        pred_color: torch.Tensor,
        target_color: torch.Tensor,
    ) -> torch.Tensor:
        """Compute color reconstruction loss."""
        return torch.mean((pred_color - target_color) ** 2)

    def perceptual_loss(
        self,
        pred_color: torch.Tensor,
        target_color: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute perceptual similarity loss.

        Uses simple gradient difference as basic perceptual metric.
        """
        # Compute gradients
        _, grad_x = torch.gradient(pred_color, edge_order=1)
        _, grad_y = torch.gradient(pred_color, edge_order=1)

        pred_grad = torch.mean(grad_x**2 + grad_y**2)

        _, grad_x = torch.gradient(target_color, edge_order=1)
        _, grad_y = torch.gradient(target_color, edge_order=1)

        target_grad = torch.mean(grad_x**2 + grad_y**2)

        return torch.mean((pred_grad - target_grad) ** 2)


class LowFrequencyColorLoss(BaseColorLoss):
    """
    Low-frequency color loss.

    Compares base color (smooth) with ground truth low-pass filtered color.
    """

    def forward(
        self,
        pred_color: torch.Tensor,
        target_color_lowpass: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute low-freq color loss.

        Args:
            pred_color: (B, 3) predicted base color
            target_color_lowpass: (B, 3) target low-pass color

        Returns:
            Base color loss
        """
        return self.reconstruction_loss(pred_color, target_color_lowpass)


class DiffusionLoss(BaseLoss):
    """
    Diffusion model loss.

    Standard DDPM noise prediction: L = E[||ε - ε_θ(x_t, t, c)||^2]
    """

    def forward(
        self,
        pred_noise: torch.Tensor,
        target_noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute diffusion loss.

        Args:
            pred_noise: (B, 3) predicted noise
            target_noise: (B, 3) ground truth noise

        Returns:
            MSE between predicted and target noise
        """
        return torch.mean((pred_noise - target_noise) ** 2)

    def __call__(
        self,
        noisy_color: torch.Tensor,
        clean_color: torch.Tensor,
        timestep: torch.Tensor,
        model,
        condition_vector: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute full diffusion loss pipeline.

        Args:
            noisy_color: (B, 3) noisy color
            clean_color: (B, 3) clean color
            timestep: (B,) diffusion timestep
            model: Diffusion model
            condition_vector: (B, 42) condition vector

        Returns:
            Diffusion loss
        """
        # Sample target noise
        noise = torch.randn_like(clean_color)

        # Add noise to color
        alpha = torch.sqrt(model.scheduler.alphas_cumprod[timestep])
        noisy_color_target = alpha[:, None] * clean_color + \
            torch.sqrt(1 - alpha[:, None]) * noise

        # Predict noise
        pred_noise = model(noisy_color_target, timestep, condition_vector)

        return self(pred_noise, noise), pred_noise


class ReverseMappingLoss(BaseLoss):
    """
    Reverse mapping classification loss.

    For phase 3: CrossEntropy between predicted and ground truth labels.

    For later phases: Can add adversarial version.
    """

    def __init__(self, config: LossConfig, temperature: float = 1.0):
        super().__init__(config)
        self.temperature = temperature

    def forward(
        self,
        logits: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Args:
            logits: (B, K) class logits
            target_labels: (B,) target class indices

        Returns:
            Cross-entropy loss
        """
        return nn.functional.cross_entropy(
            logits / self.temperature,
            target_labels,
            reduction="mean",
        )

    def entropy_loss(
        self,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute entropy regularization.

        Encourages productive ambiguity (avoid overconfidence collapse).

        Args:
            probs: (B, K) class probabilities

        Returns:
            Negative entropy (higher entropy)
        """
        # Entropy: -sum(p * log(p))
        eps = 1e-8
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
        return torch.mean(entropy)


class CombinedLossFunction(ILossFunction):
    """
    Combined loss function for phase 3 training.

    L_total = λ_sdf * L_sdf + λ_eikonal * L_eikonal + λ_color_base * L_color_base +
              λ_diffusion * L_diffusion + λ_reverse * L_reverse + λ_entropy * L_entropy
    """

    def __init__(self, config: LossConfig):
        super().__init__(config)

        # Initialize loss modules
        self.sdf_loss = SDFLoss(config)
        self.eikonal_loss = EikonalLoss(config)
        self.base_color_loss = LowFrequencyColorLoss(config)
        self.diffusion_loss = DiffusionLoss(config)
        self.reverse_loss = ReverseMappingLoss(config)

    def compute(
        self,
        pred_sdf: torch.Tensor,
        pred_base_color: torch.Tensor,
        pred_diffusion_output: Any,
        pred_reverse_output: Any,
        target_sdf: torch.Tensor,
        target_base_color_lowpass: torch.Tensor,
        target_satellite_outputs: Dict[str, torch.Tensor],
        training_step: int,
        total_steps: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.

        Args:
            pred_sdf: SDF predictions
            pred_base_color: Base color predictions
            pred_diffusion_output: Diffusion model output dict
            pred_reverse_output: Reverse mapping network output
            target_sdf: Target SDF
            target_base_color_lowpass: Target low-pass color
            target_satellite_outputs: Dict of other target outputs
            training_step: Current training step
            total_steps: Total training steps (for schedule)

        Returns:
            total_loss, loss_dict with individual losses
        """
        # Compute individual losses
        loss_SDF = self.sdf_loss(pred_sdf, target_sdf)

        # Eikonal loss requires positions, estimated here
        loss_Eikonal = self.eikonal_loss(
            pred_sdf,
            target_satellite_outputs.get("positions", None),
        )

        loss_color_base = self.base_color_loss(
            pred_base_color,
            target_base_color_lowpass,
        )

        # Diffusion loss
        loss_diffusion, pred_noise = pred_diffusion_output.get("loss", (None, None))

        # Reverse mapping loss
        loss_reverse = self.reverse_loss(
            pred_reverse_output["logits"],
            pred_reverse_output["target_labels"],
        )

        # Compute dynamic weights
        lambda_sdf = self.config.lambda_sdf
        lambda_eikonal = self.config.lambda_eikonal
        lambda_color_base = self.config.lambda_color_base
        lambda_diffusion = self.config.lambda_diffusion
        lambda_reverse = self.config.lambda_reverse
        lambda_entropy = self.config.lambda_entropy

        # Reverse loss increases over training (phase 3 schedule)
        if lambda_reverse > 0:
            progress = min(training_step / total_steps, 1.0)
            lambda_reverse = self.config.lambda_reverse_start + \
                (self.config.lambda_reverse_end - self.config.lambda_reverse_start) * progress

        # Entropy increases gradually
        if lambda_entropy > 0:
            progress = min(training_step / total_steps, 1.0)
            lambda_entropy = self.config.lambda_entropy_start + \
                (self.config.lambda_entropy_end - self.config.lambda_entropy_start) * progress

        # Compute total loss
        total_loss = (
            lambda_sdf * loss_SDF +
            lambda_eikonal * loss_Eikonal +
            lambda_color_base * loss_color_base +
            lambda_diffusion * loss_diffusion +
            lambda_reverse * loss_reverse
            + lambda_entropy * self.reverse_loss.entropy_loss(pred_reverse_output["probs"])
        )

        loss_dict = {
            "loss_SDF": loss_SDF.item(),
            "loss_Eikonal": loss_Eikonal.item(),
            "loss_color_base": loss_color_base.item(),
            "loss_diffusion": loss_diffusion.item(),
            "loss_reverse": loss_reverse.item(),
            "loss_entropy": 0.0,
            "lambda_sdf": lambda_sdf,
            "lambda_eikonal": lambda_eikonal,
            "lambda_color_base": lambda_color_base,
            "lambda_diffusion": lambda_diffusion,
            "lambda_reverse": lambda_reverse,
            "lambda_entropy": lambda_entropy,
        }

        return total_loss, loss_dict


# =============================================================================
# Optimizers
# =============================================================================


class AdamOptimizer(IOptimizer):
    """Adam optimizer wrapper."""

    def __init__(self, params, lr, weight_decay: float = 0.0):
        self.optimizer = optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
        )

    def step(self, loss: torch.Tensor):
        """Perform optimization step."""
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups, max_norm=1.0)
        self.optimizer.step()

    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()


class AdamWOptimizer(IOptimizer):
    """AdamW optimizer wrapper (for phase 3)."""

    def __init__(self, params, lr, weight_decay: float = 1e-4):
        self.optimizer = optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
        )

    def step(self, loss: torch.Tensor):
        """Perform optimization step."""
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups, max_norm=1.0)
        self.optimizer.step()

    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()


# =============================================================================
# Learning Rate Schedulers
# =============================================================================


class LinearScheduler(IScheduler):
    """Linear learning rate scheduler."""

    def __init__(
        self,
        optimizer: Optional[IOptimizer] = None,
        num_warmup_steps: int = 100,
        num_total_steps: int = 10000,
        start_lr: float = 1e-5,
        end_lr: float = 5e-5,
    ):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_total_steps = num_total_steps
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.current_step = 0

    def step(self, metrics: Optional[float] = None):
        """Update learning rate."""
        self.current_step += 1

        if self.current_step <= self.num_warmup_steps:
            # Warmup phase
            lr = self.start_lr + \
                (self.end_lr - self.start_lr) * \
                (self.current_step / self.num_warmup_steps)
        else:
            # Decay phase
            progress = min(
                (self.current_step - self.num_warmup_steps) /
                (self.num_total_steps - self.num_warmup_steps),
                1.0,
            )
            lr = self.end_lr * (1 - progress) + self.start_lr + \
                (self.end_lr - self.start_lr) * progress

        if self.optimizer:
            for param_group in self.optimizer.optimizer.param_groups:
                param_group["lr"] = lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        for param_group in self.optimizer.optimizer.param_groups:
            return param_group["lr"]
        return 1e-4


# =============================================================================
# Checkpoint Manager
# =============================================================================


class CheckpointManager:
    """
    Manages model checkpoints and training state saving/loading.

    Supports saving:
    - Model weights
    - Optimizer states
    - LR scheduler states
    - Training epoch/step
    - Random seed for reproducibility
    """

    def __init__(self, save_dir: str):
        """
        Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(self, checkpoint: Dict[str, Any], filename: str, epoch: int):
        """
        Save checkpoint.

        Args:
            checkpoint: Dictionary containing model, optimizer, etc.
            filename: Output filename
            epoch: Current epoch number
        """
        checkpoint.update({
            "epoch": epoch,
            "step": checkpoint.get("step", 0),
            "seed": checkpoint.get("seed", 42),
        })

        path = self.save_dir / filename
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load(
        self,
        path: str,
        device: torch.device,
    ) -> Dict[str, Any]:
        """
        Load checkpoint.

        Args:
            path: Checkpoint path
            device: Device to load model to

        Returns:
            Checkpoint dictionary
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=device)

        print(f"Checkpoint loaded from {path}")
        print(f"  Epoch: {checkpoint.get('epoch', 0)}")
        print(f"  Step: {checkpoint.get('step', 0)}")

        return checkpoint

    def list_checkpoints(self) -> List[Path]:
        """List all checkpoints in save directory."""
        return sorted(self.save_dir.glob("*.pt"))

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the latest checkpoint."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        return checkpoints[-1]
