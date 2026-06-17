"""
Main Training Script for Diffusion-UV

Three-phase progressive training:
- Phase 1: Train Network G (SDF + low-freq color) - 500-1000 epochs
- Phase 2: Train Network D (diffusion model) - 200-300 epochs
- Phase 3: Joint fine-tuning (G + D + R) - 100-200 epochs

Usage:
    python scripts/train.py --config configs/experiment.yaml [--phase 1|2|3]
    python scripts/train.py --config configs/experiment.yaml --resume checkpoint.pt
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import time
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime

# Import project modules
from src.config import load_config, ExperimentConfig
from src.utils import get_device, setup_logger, RandomNumberGenerator
from src.utils.tracking import ExperimentTracker, MetricsAggregator

# Import experiment manager
from scripts.experiment_manager import ExperimentManager, create_sampling_data_dict

# Import models
from src.models import NetworkG, NetworkD, NetworkR

# Import data pipeline
from src.data import (
    MeshData,
    TextureData,
    DataSamplingPipeline,
    create_pipeline_from_files,
    ImplicitTextureDataset,
)

# Import losses
from src.training import (
    SDFLoss,
    EikonalLoss,
    LowFrequencyColorLoss,
    DiffusionLoss,
    ReverseMappingLoss,
    CheckpointManager,
)

# Import diffusion utilities
try:
    from diffusers import DDPMScheduler, DDIMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


@dataclass
class TrainingState:
    """State of training."""
    phase: int
    epoch: int
    global_step: int
    best_loss: float


class ImplicitTextureTrainer:
    """
    Trainer for Implicit Texture Field.

    Implements three-phase progressive training with comprehensive logging
    and experiment management for sampling data preservation.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        device: torch.device,
        tracker: Optional[ExperimentTracker] = None,
        experiment_manager: Optional[ExperimentManager] = None,
    ):
        self.config = config
        self.device = device
        self.tracker = tracker
        self.exp_manager = experiment_manager
        self.state = TrainingState(phase=1, epoch=0, global_step=0, best_loss=float('inf'))

        # Initialize components
        self.network_g: Optional[NetworkG] = None
        self.network_d: Optional[NetworkD] = None
        self.network_r: Optional[NetworkR] = None

        # Optimizers
        self.optimizer_g: Optional[torch.optim.Optimizer] = None
        self.optimizer_d: Optional[torch.optim.Optimizer] = None
        self.optimizer_r: Optional[torch.optim.Optimizer] = None

        # Schedulers
        self.scheduler_g: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.scheduler_d: Optional[torch.optim.lr_scheduler._LRScheduler] = None

        # Checkpoint manager - use experiment folder if available
        if self.exp_manager:
            checkpoint_dir = self.exp_manager.get_experiment_dir() / "checkpoints"
        else:
            checkpoint_dir = Path(config.logging.log_dir) / config.logging.experiment_name / "checkpoints"

        self.checkpoint_manager = CheckpointManager(save_dir=str(checkpoint_dir))

        # Metrics aggregator
        self.metrics_agg = MetricsAggregator()

        # Track whether sampling data has been saved
        self.sampling_data_saved = False

    def setup(self, mesh_data: MeshData, texture_data: TextureData):
        """Setup networks, optimizers, and data pipeline."""
        # Create data pipeline
        self.pipeline = DataSamplingPipeline(
            mesh=mesh_data,
            texture=texture_data,
            sampling_ratios=self.config.data.sampling_ratios,
            num_classes=self.config.network_r.num_classes,
        )

        # Create datasets for each phase
        num_samples = self.config.data.num_samples_per_epoch

        self.dataset = ImplicitTextureDataset(
            mesh_data=mesh_data,
            texture_data=texture_data,
            num_samples=num_samples,
            normalize_coords=True,
            augment=True,
            augmentation_noise=0.001,
            cache_dir=self.config.data.cache_dir,  # Enable caching
            sampling_ratios=self.config.data.sampling_ratios,
            num_classes=self.config.network_r.num_classes,
        )

        # Create networks
        self._create_networks()

        # Create optimizers
        self._create_optimizers()

        print(f"Training setup complete.")
        print(f"  Network G params: {self.network_g.count_parameters():,}")
        if self.network_d:
            print(f"  Network D params: {self.network_d.count_parameters():,}")
        if self.network_r:
            print(f"  Network R params: {self.network_r.count_parameters():,}")

    def save_sampling_data(self):
        """
        保存训练前的采样数据到实验文件夹。

        这是GT（Ground Truth）数据，用于后续对比推理结果。
        """
        if not self.exp_manager or self.sampling_data_saved:
            return

        print("\n" + "=" * 60)
        print("保存训练前采样数据 (GT Data)")
        print("=" * 60)

        # 从dataset采样数据
        num_samples = self.config.data.num_samples_per_epoch
        samples = self.dataset.sample_batch(num_samples, include_labels=True)

        # 准备采样数据字典
        sampling_data = {
            'points': samples['position'].cpu().numpy().astype(np.float32),
            'colors': samples['color_gt'].cpu().numpy().astype(np.float32),
            'sdf': samples['sdf'].cpu().numpy().astype(np.float32),
            'normals': samples['normal'].cpu().numpy().astype(np.float32),
        }

        # 添加可选字段
        if 'curvature' in samples:
            sampling_data['curvatures'] = samples['curvature'].cpu().numpy().astype(np.float32)
        if 'uv' in samples:
            sampling_data['uvs'] = samples['uv'].cpu().numpy().astype(np.float32)
        else:
            # 如果没有UV，添加占位符
            sampling_data['uvs'] = np.zeros((num_samples, 2), dtype=np.float32)
        if 'label' in samples:
            sampling_data['labels'] = samples['label'].cpu().numpy().astype(np.int32)
        if 'region' in samples:
            # Region might be strings like 'exterior', 'surface', etc.
            # Store as-is (object dtype) for strings
            if isinstance(samples['region'], np.ndarray) and samples['region'].dtype.kind in ['U', 'O', 'S']:
                sampling_data['regions'] = samples['region']
            else:
                sampling_data['regions'] = samples['region'].astype(np.int32)

        # 准备元数据
        metadata = {
            'total_samples': num_samples,
            'sampling_ratios': self.config.data.sampling_ratios,
            'sampling_date': datetime.now().isoformat(),
            'mesh_path': str(self.config.data.high_mesh_path),
            'texture_path': str(self.config.data.texture_path) if self.config.data.texture_path else 'procedural',
        }

        # 保存到实验文件夹
        self.exp_manager.save_sampling_data(
            train_samples=sampling_data,
            metadata=metadata
        )

        print(f"✓ 采样数据已保存: {self.exp_manager.current_experiment_dir / 'sampling_data' / 'train_samples.npz'}")
        print(f"  包含 {num_samples} 个采样点")
        print(f"  可使用 viewer 查看: python scripts/viewer_3d.py {self.exp_manager.current_experiment_dir}/sampling_data/train_samples.npz")

        self.sampling_data_saved = True

    def _create_networks(self):
        """Create neural networks."""
        # Network G
        self.network_g = NetworkG(
            hidden_dim=self.config.network_g.hidden_dim,
            num_layers=self.config.network_g.num_layers,
            positional_encoding_freqs=self.config.network_g.positional_encoding_freqs,
            skip_connection_layer=self.config.network_g.skip_connection_layer,
            include_raw_input=self.config.network_g.include_raw_input,
            sdf_output_range=self.config.network_g.sdf_output_range,
        ).to(self.device)

        # Network D (diffusion model)
        if DIFFUSERS_AVAILABLE:
            self.network_d = NetworkD(
                cond_dim=self.config.network_d.condition_dim,
                num_diffusion_steps=self.config.network_d.num_diffusion_steps,
                hidden_channels=self.config.network_d.hidden_channels,
                num_res_blocks=self.config.network_d.num_res_blocks,
            ).to(self.device)

            # Setup diffusion scheduler
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.config.network_d.num_diffusion_steps,
                beta_schedule=self.config.network_d.scheduler_type,
            )
        else:
            self.network_d = None
            print("Warning: diffusers not available. Phase 2 training will be skipped.")

        # Network R
        self.network_r = NetworkR(
            input_dim=6,  # position (3) + color (3)
            hidden_dims=(64, 128, 256),
            num_classes=self.config.network_r.num_classes,
        ).to(self.device)

    def _create_optimizers(self):
        """Create optimizers for each phase."""
        # Phase 1: G only
        self.optimizer_g = torch.optim.Adam(
            self.network_g.parameters(),
            lr=self.config.training.learning_rate_g,
        )

        self.scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_g,
            mode="min",
            factor=0.5,
            patience=50,
        )

        # Phase 2: D only
        if self.network_d:
            self.optimizer_d = torch.optim.AdamW(
                self.network_d.parameters(),
                lr=self.config.training.learning_rate_d,
                weight_decay=self.config.training.weight_decay,
            )

            self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_d,
                T_max=self.config.training.phase2_epochs,
                eta_min=1e-6,
            )

        # Phase 3: All networks
        self.optimizer_r = torch.optim.Adam(
            self.network_r.parameters(),
            lr=self.config.training.learning_rate_r,
        )

    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch from the data pipeline."""
        # Use dataset.sample_batch() which utilizes cached data if available
        samples = self.dataset.sample_batch(batch_size, include_labels=True)

        # Move to device and set requires_grad for positions (needed for Eikonal loss)
        batch = {
            'positions': samples['position'].to(self.device).requires_grad_(True),
            'normals': samples['normal'].to(self.device),
            'colors': samples['color_gt'].to(self.device),
            'sdf': samples['sdf'].to(self.device),
            'uvs': samples.get('curvature', torch.zeros(batch_size, 2)).to(self.device),  # Use curvature as placeholder
        }

        if 'label' in samples:
            batch['labels'] = samples['label'].to(self.device)
        if 'region' in samples:
            batch['regions'] = samples['region']  # Already numpy array

        return batch

    # =========================================================================
    # Phase 1: Train Network G (SDF + Eikonal + Color)
    # =========================================================================

    def train_phase1(self, num_epochs: int):
        """
        Phase 1: Train Network G only.

        Loss: L_Geo = λ_SDF*L_SDF + λ_Eikonal*L_Eikonal + λ_Color*L_Color
        """
        print("\n" + "=" * 60)
        print("PHASE 1: Training Network G (SDF + Low-freq Color)")
        print("=" * 60)

        self.state.phase = 1
        if self.tracker:
            self.tracker.set_phase("phase_1")

        # Ensure G is trainable, others frozen
        self.network_g.train()
        if self.network_d:
            self.network_d.eval()
            for p in self.network_d.parameters():
                p.requires_grad = False
        if self.network_r:
            self.network_r.eval()
            for p in self.network_r.parameters():
                p.requires_grad = False

        # Create loss functions
        loss_config = self.config.loss
        sdf_loss_fn = SDFLoss(loss_config)
        eikonal_loss_fn = EikonalLoss(loss_config)
        color_loss_fn = LowFrequencyColorLoss(loss_config)

        batch_size = self.config.training.batch_size_phase1
        num_batches = self.config.data.num_samples_per_epoch // batch_size

        for epoch in range(num_epochs):
            self.state.epoch = epoch
            epoch_start = time.time()
            self.metrics_agg.reset()

            for batch_idx in range(num_batches):
                # Sample batch
                batch = self.sample_batch(batch_size)
                positions = batch['positions']
                target_sdf = batch['sdf']
                target_color = batch['colors']

                # Forward pass
                output = self.network_g(positions)
                pred_sdf = output.sdf
                pred_color_base = output.color_base

                # Compute losses
                loss_sdf = sdf_loss_fn(pred_sdf, target_sdf)

                # Eikonal loss (requires gradient computation)
                loss_eikonal = eikonal_loss_fn(pred_sdf, positions)

                # Color loss (low-frequency)
                loss_color = color_loss_fn(pred_color_base, target_color)

                # Combined loss with weights
                total_loss = (
                    self.config.training.lambda_sdf * loss_sdf +
                    self.config.training.lambda_eikonal * loss_eikonal +
                    self.config.training.lambda_color_base * loss_color
                )

                # Backward pass
                self.optimizer_g.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network_g.parameters(),
                    max_norm=self.config.training.gradient_clip_norm
                )
                self.optimizer_g.step()

                # Accumulate metrics
                self.metrics_agg.update({
                    'loss_total': total_loss.item(),
                    'loss_sdf': loss_sdf.item(),
                    'loss_eikonal': loss_eikonal.item(),
                    'loss_color': loss_color.item(),
                    'lr': self.optimizer_g.param_groups[0]['lr'],
                }, batch_size)

                self.state.global_step += 1

                # Log batch metrics
                if self.state.global_step % self.config.logging.log_interval == 0:
                    metrics = self.metrics_agg.compute()
                    self._log_metrics(metrics, step=self.state.global_step)

            # Epoch summary
            epoch_time = time.time() - epoch_start
            metrics = self.metrics_agg.compute()

            print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) - "
                  f"Loss: {metrics.get('loss_total', 0):.4f} | "
                  f"SDF: {metrics.get('loss_sdf', 0):.4f} | "
                  f"Eikonal: {metrics.get('loss_eikonal', 0):.4f} | "
                  f"Color: {metrics.get('loss_color', 0):.4f}")

            # Learning rate scheduling
            self.scheduler_g.step(metrics.get('loss_total', 0))

            # Save checkpoint
            if (epoch + 1) % self.config.logging.save_interval == 0:
                self.save_checkpoint(f"phase1_epoch_{epoch+1}.pt")

            # Track best
            if metrics.get('loss_total', float('inf')) < self.state.best_loss:
                self.state.best_loss = metrics.get('loss_total', 0)
                self.save_checkpoint("phase1_best.pt")

        print("Phase 1 training complete!")
        return self.network_g

    # =========================================================================
    # Phase 2: Train Network D (Diffusion Model)
    # =========================================================================

    def train_phase2(self, num_epochs: int):
        """
        Phase 2: Train Network D only (G frozen).

        Loss: L_Diff = E[||ε - ε_θ(x_t, t, c)||^2]
        """
        if not DIFFUSERS_AVAILABLE or self.network_d is None:
            print("Warning: Skipping Phase 2 - diffusers not available")
            return self.network_d

        print("\n" + "=" * 60)
        print("PHASE 2: Training Network D (Diffusion Model)")
        print("=" * 60)

        self.state.phase = 2
        if self.tracker:
            self.tracker.set_phase("phase_2")

        # Freeze G, train D
        self.network_g.eval()
        for p in self.network_g.parameters():
            p.requires_grad = False

        self.network_d.train()
        for p in self.network_d.parameters():
            p.requires_grad = True

        batch_size = self.config.training.batch_size_phase2
        num_batches = self.config.data.num_samples_per_epoch // batch_size

        diffusion_loss_fn = DiffusionLoss(self.config.loss)

        for epoch in range(num_epochs):
            self.state.epoch = epoch
            epoch_start = time.time()
            self.metrics_agg.reset()

            for batch_idx in range(num_batches):
                # Sample batch
                batch = self.sample_batch(batch_size)
                positions = batch['positions']
                target_color = batch['colors']

                # Get condition from G (frozen)
                with torch.no_grad():
                    g_output = self.network_g(positions)
                    cond_base_color = g_output.color_base
                    cond_sdf = g_output.sdf.view(-1, 1)

                    # Create condition vector (simplified: just use base color + sdf)
                    # Full condition would include curvature, normal, boundary_dist, global_shape
                    condition = torch.cat([
                        cond_base_color,
                        cond_sdf,
                        torch.zeros_like(cond_sdf).expand(-1, 38)  # Pad to 42 dims
                    ], dim=-1)

                # Sample random timesteps
                timesteps = torch.randint(
                    0,
                    self.config.network_d.num_diffusion_steps,
                    (batch_size,),
                    device=self.device,
                )

                # Add noise to target color
                noise = torch.randn_like(target_color)
                noisy_color = self.noise_scheduler.add_noise(
                    target_color,
                    noise,
                    timesteps,
                )

                # Predict noise
                noise_pred = self.network_d(noisy_color, timesteps, condition)

                # Compute loss
                loss_diffusion = torch.mean((noise_pred - noise) ** 2)

                # Backward pass
                self.optimizer_d.zero_grad()
                loss_diffusion.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network_d.parameters(),
                    max_norm=self.config.training.gradient_clip_norm
                )
                self.optimizer_d.step()

                # Accumulate metrics
                self.metrics_agg.update({
                    'loss_diffusion': loss_diffusion.item(),
                    'lr': self.optimizer_d.param_groups[0]['lr'],
                }, batch_size)

                self.state.global_step += 1

                if self.state.global_step % self.config.logging.log_interval == 0:
                    metrics = self.metrics_agg.compute()
                    self._log_metrics(metrics, step=self.state.global_step)

            # Epoch summary
            epoch_time = time.time() - epoch_start
            metrics = self.metrics_agg.compute()

            print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) - "
                  f"Loss: {metrics.get('loss_diffusion', 0):.4f}")

            # Update scheduler
            self.scheduler_d.step()

            # Save checkpoint
            if (epoch + 1) % self.config.logging.save_interval == 0:
                self.save_checkpoint(f"phase2_epoch_{epoch+1}.pt")

        print("Phase 2 training complete!")
        return self.network_d

    # =========================================================================
    # Phase 3: Joint Fine-tuning (G + D + R)
    # =========================================================================

    def train_phase3(self, num_epochs: int):
        """
        Phase 3: Joint fine-tuning of G, D, R.

        Loss: L_Total = L_Geo + λ_Diff*L_Diff + λ_Reverse*L_Reverse + λ_Entropy*L_Entropy
        """
        if not DIFFUSERS_AVAILABLE or self.network_d is None:
            print("Warning: Skipping Phase 3 - diffusers not available")
            return self.network_g, self.network_d, self.network_r

        print("\n" + "=" * 60)
        print("PHASE 3: Joint Fine-tuning (G + D + R)")
        print("=" * 60)

        self.state.phase = 3
        if self.tracker:
            self.tracker.set_phase("phase_3")

        # Unfreeze all networks
        self.network_g.train()
        for p in self.network_g.parameters():
            p.requires_grad = True

        self.network_d.train()
        for p in self.network_d.parameters():
            p.requires_grad = True

        self.network_r.train()
        for p in self.network_r.parameters():
            p.requires_grad = True

        # Create loss functions
        loss_config = self.config.loss
        sdf_loss_fn = SDFLoss(loss_config)
        eikonal_loss_fn = EikonalLoss(loss_config)
        color_loss_fn = LowFrequencyColorLoss(loss_config)
        reverse_loss_fn = ReverseMappingLoss(loss_config)

        batch_size = self.config.training.batch_size_phase3
        num_batches = self.config.data.num_samples_per_epoch // batch_size

        # Phase 3 uses lower learning rates for fine-tuning
        for param_group in self.optimizer_g.param_groups:
            param_group['lr'] = self.config.training.learning_rate_g * 0.1
        for param_group in self.optimizer_d.param_groups:
            param_group['lr'] = self.config.training.learning_rate_d * 0.1

        total_steps = num_epochs * num_batches

        for epoch in range(num_epochs):
            self.state.epoch = epoch
            epoch_start = time.time()
            self.metrics_agg.reset()

            for batch_idx in range(num_batches):
                current_step = epoch * num_batches + batch_idx

                # Sample batch
                batch = self.sample_batch(batch_size)
                positions = batch['positions']
                target_sdf = batch['sdf']
                target_color = batch['colors']
                target_labels = batch.get('labels')

                # Compute dynamic weights
                progress = current_step / total_steps
                lambda_reverse = (
                    self.config.training.lambda_reverse_start +
                    (self.config.training.lambda_reverse_end - self.config.training.lambda_reverse_start) * progress
                )
                lambda_entropy = (
                    self.config.training.lambda_entropy_start +
                    (self.config.training.lambda_entropy_end - self.config.training.lambda_entropy_start) * progress
                )

                # Forward G
                g_output = self.network_g(positions)
                pred_sdf = g_output.sdf
                pred_color_base = g_output.color_base

                # Compute geometry losses
                loss_sdf = sdf_loss_fn(pred_sdf, target_sdf)
                loss_eikonal = eikonal_loss_fn(pred_sdf, positions)
                loss_color = color_loss_fn(pred_color_base, target_color)

                # Phase 2 loss: Diffusion (D)
                with torch.no_grad():
                    cond_base_color = pred_color_base.detach()
                    cond_sdf = pred_sdf.detach().view(-1, 1)
                    condition = torch.cat([
                        cond_base_color,
                        cond_sdf,
                        torch.zeros(batch_size, 38, device=self.device),
                    ], dim=-1)

                timesteps = torch.randint(
                    0, self.config.network_d.num_diffusion_steps,
                    (batch_size,), device=self.device,
                )
                noise = torch.randn_like(target_color)
                noisy_color = self.noise_scheduler.add_noise(target_color, noise, timesteps)
                noise_pred = self.network_d(noisy_color, timesteps, condition)
                loss_diffusion = torch.mean((noise_pred - noise) ** 2)

                # Phase 3 loss: Reverse mapping (R)
                if target_labels is not None:
                    # Get "final" color (using base color for simplicity)
                    final_color = pred_color_base

                    # Forward R
                    r_output = self.network_r(positions, final_color)
                    loss_reverse = reverse_loss_fn(r_output.logits, target_labels)

                    # Entropy regularization
                    probs = r_output.probs
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                    loss_entropy = lambda_entropy * entropy.mean()
                else:
                    loss_reverse = torch.tensor(0.0, device=self.device)
                    loss_entropy = torch.tensor(0.0, device=self.device)

                # Combined loss
                total_loss = (
                    self.config.training.lambda_sdf * loss_sdf +
                    self.config.training.lambda_eikonal * loss_eikonal +
                    self.config.training.lambda_color_base * loss_color +
                    self.config.training.lambda_diffusion * loss_diffusion +
                    lambda_reverse * loss_reverse +
                    loss_entropy
                )

                # Backward pass (all networks)
                self.optimizer_g.zero_grad()
                self.optimizer_d.zero_grad()
                self.optimizer_r.zero_grad()

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(self.network_g.parameters()) +
                    list(self.network_d.parameters()) +
                    list(self.network_r.parameters()),
                    max_norm=self.config.training.gradient_clip_norm
                )

                self.optimizer_g.step()
                self.optimizer_d.step()
                self.optimizer_r.step()

                # Accumulate metrics
                self.metrics_agg.update({
                    'loss_total': total_loss.item(),
                    'loss_sdf': loss_sdf.item(),
                    'loss_eikonal': loss_eikonal.item(),
                    'loss_color': loss_color.item(),
                    'loss_diffusion': loss_diffusion.item(),
                    'loss_reverse': loss_reverse.item(),
                    'loss_entropy': loss_entropy.item(),
                    'lambda_reverse': lambda_reverse,
                    'lambda_entropy': lambda_entropy,
                    'lr_g': self.optimizer_g.param_groups[0]['lr'],
                    'lr_d': self.optimizer_d.param_groups[0]['lr'],
                    'lr_r': self.optimizer_r.param_groups[0]['lr'],
                }, batch_size)

                self.state.global_step += 1

                if self.state.global_step % self.config.logging.log_interval == 0:
                    metrics = self.metrics_agg.compute()
                    self._log_metrics(metrics, step=self.state.global_step)

            # Epoch summary
            epoch_time = time.time() - epoch_start
            metrics = self.metrics_agg.compute()

            print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) - "
                  f"Loss: {metrics.get('loss_total', 0):.4f} | "
                  f"SDF: {metrics.get('loss_sdf', 0):.4f} | "
                  f"Reverse: {metrics.get('loss_reverse', 0):.4f}")

            # Save checkpoint
            if (epoch + 1) % self.config.logging.save_interval == 0:
                self.save_checkpoint(f"phase3_epoch_{epoch+1}.pt")

        print("Phase 3 training complete!")
        return self.network_g, self.network_d, self.network_r

    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to tracker."""
        if self.tracker:
            self.tracker.log_metrics(metrics, step=step)
        else:
            # Print to console
            loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if 'loss' in k])
            print(f"  Step {step}: {loss_str}")

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.state.epoch,
            'global_step': self.state.global_step,
            'phase': self.state.phase,
            'best_loss': self.state.best_loss,
            'config': self.config.to_dict(),
            'seed': self.config.seed,
        }

        if self.network_g:
            checkpoint['network_g_state'] = self.network_g.state_dict()
            checkpoint['optimizer_g_state'] = self.optimizer_g.state_dict()

        if self.network_d:
            checkpoint['network_d_state'] = self.network_d.state_dict()
            checkpoint['optimizer_d_state'] = self.optimizer_d.state_dict()

        if self.network_r:
            checkpoint['network_r_state'] = self.network_r.state_dict()
            checkpoint['optimizer_r_state'] = self.optimizer_r.state_dict()

        self.checkpoint_manager.save(checkpoint, filename, self.state.epoch)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = self.checkpoint_manager.load(path, self.device)

        self.state.epoch = checkpoint.get('epoch', 0)
        self.state.global_step = checkpoint.get('global_step', 0)
        self.state.phase = checkpoint.get('phase', 1)
        self.state.best_loss = checkpoint.get('best_loss', float('inf'))

        if self.network_g and 'network_g_state' in checkpoint:
            self.network_g.load_state_dict(checkpoint['network_g_state'])
            if 'optimizer_g_state' in checkpoint:
                self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state'])

        if self.network_d and 'network_d_state' in checkpoint:
            self.network_d.load_state_dict(checkpoint['network_d_state'])
            if 'optimizer_d_state' in checkpoint:
                self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state'])

        if self.network_r and 'network_r_state' in checkpoint:
            self.network_r.load_state_dict(checkpoint['network_r_state'])
            if 'optimizer_r_state' in checkpoint:
                self.optimizer_r.load_state_dict(checkpoint['optimizer_r_state'])

        print(f"Checkpoint loaded: epoch {self.state.epoch}, step {self.state.global_step}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train implicit texture field")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Training phase (1, 2, or 3). If not specified, runs all phases.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    return parser.parse_args()


def load_mesh_and_texture(config: ExperimentConfig) -> Tuple[MeshData, TextureData]:
    """
    Load mesh and texture data.

    Args:
        config: Experiment configuration

    Returns:
        mesh_data, texture_data
    """
    import trimesh

    # Load high-poly mesh
    print(f"Loading mesh: {config.data.high_mesh_path}")
    mesh = trimesh.load(config.data.high_mesh_path)

    # Handle scene (multiple meshes)
    if isinstance(mesh, trimesh.Scene):
        mesh = list(mesh.geometry.values())[0]

    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)

    # Generate spherical UV coordinates for procedural texturing
    print("Generating spherical UV coordinates...")
    uvs = spherical_uv(vertices)
    print(f"  UV range: U=[{uvs[:,0].min():.3f}, {uvs[:,0].max():.3f}], V=[{uvs[:,1].min():.3f}, {uvs[:,1].max():.3f}]")

    # Extract mesh data with UV coordinates
    mesh_data = MeshData(
        vertices=vertices,
        faces=faces,
        vertex_normals=np.array(mesh.vertex_normals, dtype=np.float32) if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None else None,
        face_normals=np.array(mesh.face_normals, dtype=np.float32) if hasattr(mesh, 'face_normals') and mesh.face_normals is not None else None,
        uvs=uvs,  # ⭐ 添加UV坐标
        uv_faces=faces,  # 使用相同的face indices
    )

    # Load or create texture
    if config.data.texture_path:
        print(f"Loading texture: {config.data.texture_path}")
        texture_image = trimesh.load_texture(config.data.texture_path)
    else:
        # Create procedural texture with full detail
        print("Creating procedural texture...")
        texture_image = create_procedural_texture()

    texture_data = TextureData.from_array(texture_image)

    return mesh_data, texture_data


def spherical_uv(vertices: np.ndarray) -> np.ndarray:
    """
    生成球面UV坐标用于程序化纹理映射。

    Args:
        vertices: (N, 3) 顶点位置

    Returns:
        uvs: (N, 2) UV坐标，范围[0, 1]
    """
    # 中心化顶点
    centroid = vertices.mean(axis=0)
    v_centered = vertices - centroid

    # 归一化到单位球
    norms = np.linalg.norm(v_centered, axis=1, keepdims=True)
    v_normalized = v_centered / (norms + 1e-8)

    # 球面坐标
    x, y, z = v_normalized[:, 0], v_normalized[:, 1], v_normalized[:, 2]

    # Theta: 从XZ平面的角度（仰角）
    theta = np.arcsin(np.clip(y, -1, 1))

    # Phi: XZ平面中的角度（方位角）
    phi = np.arctan2(z, x)

    # 转换到UV [0, 1]
    u = (phi / (2 * np.pi) + 0.5).astype(np.float32)
    v = (theta / np.pi + 0.5).astype(np.float32)

    return np.stack([u, v], axis=1)


def create_procedural_texture(width=512, height=512):
    """
    创建视觉效果丰富的程序化纹理。

    包含:
    - 正弦波基础颜色
    - 棋盘格模式
    - 径向渐变

    Returns:
        texture: (H, W, 3) RGB纹理，范围[0, 255]的uint8格式
    """
    u = np.linspace(0, 1, width)
    v = np.linspace(0, 1, height)
    U, V = np.meshgrid(u, v)

    # 基础颜色 - 正弦波 (范围[0, 1])
    r = 0.7 + 0.3 * np.sin(U * 2 * np.pi * 4)
    g = 0.5 + 0.3 * np.sin(V * 2 * np.pi * 3)
    b = 0.6 + 0.3 * np.sin((U + V) * 2 * np.pi * 2)

    # 添加棋盘格模式
    checker = ((np.floor(U * 8) + np.floor(V * 8)) % 2).astype(float) * 0.15
    r = r + checker
    g = g + checker * 0.8
    b = b + checker * 0.6

    # 添加径向渐变
    center_u, center_v = 0.5, 0.5
    dist = np.sqrt((U - center_u)**2 + (V - center_v)**2)
    radial = np.exp(-dist * 3) * 0.3
    r = r + radial
    g = g + radial * 0.7
    b = b + radial * 0.4

    # 裁剪到[0, 1]然后转换到[0, 255]
    texture = np.stack([r, g, b], axis=2)
    texture = np.clip(texture, 0, 1)
    texture = (texture * 255).astype(np.uint8)

    return texture


def setup_directories(config: ExperimentConfig) -> Dict[str, Path]:
    """Create experiment directories."""
    base_dir = Path(config.logging.log_dir) / config.logging.experiment_name

    dirs = {
        'log': base_dir / "logs",
        'checkpoint': base_dir / "checkpoints",
        'cache': Path(config.data.cache_dir),
        'output': Path(config.evaluation.output_dir) / config.logging.experiment_name,
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    if args.seed is not None:
        config.seed = args.seed

    # 初始化实验管理器
    experiment_manager = ExperimentManager()

    # 创建实验文件夹
    experiment_name = config.logging.experiment_name or f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_id = experiment_manager.create_experiment(
        experiment_name=experiment_name,
        config=config.to_dict()
    )

    print(f"✓ 创建实验: {experiment_id}")
    print(f"  实验文件夹: {experiment_manager.current_experiment_dir}")

    # Setup directories (for backward compatibility)
    dirs = setup_directories(config)

    # Setup logger - 使用实验文件夹中的logs目录
    log_dir = experiment_manager.current_experiment_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    logger = setup_logger(
        name="diffusion_uv",
        log_file=str(log_dir / "train.log"),
        level=20,
    )
    logger.info(f"Configuration: {config.to_dict()}")
    logger.info(f"实验ID: {experiment_id}")

    # Set random seed
    RandomNumberGenerator.set_seed(config.seed)
    logger.info(f"Random seed: {config.seed}")

    # Get device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Initialize W&B tracker
    tracker = None
    use_wandb = config.logging.use_wandb and not args.no_wandb
    if use_wandb:
        wandb_mode = config.logging.wandb_mode
        if wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"):
            wandb_mode = "offline"
            logger.warning("WANDB_API_KEY is not set; using W&B offline mode")

        try:
            # 使用实验ID作为W&B的run名称
            tracker = ExperimentTracker(
                project=config.logging.wandb_project,
                experiment_name=experiment_id,
                config=config.to_dict(),
                mode=wandb_mode,
                log_dir=str(log_dir),
                log_interval=config.logging.log_interval,
            )
            logger.info(f"W&B tracking enabled: {tracker.run_url}")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            tracker = None
    else:
        logger.info("W&B tracking disabled")

    # Load mesh and texture data
    try:
        mesh_data, texture_data = load_mesh_and_texture(config)
        logger.info(f"Mesh: {mesh_data.num_vertices} vertices, {mesh_data.num_faces} faces")
        logger.info(f"Texture: {texture_data.width}x{texture_data.height}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1

    # Create trainer with experiment manager
    trainer = ImplicitTextureTrainer(
        config=config,
        device=device,
        tracker=tracker,
        experiment_manager=experiment_manager,
    )

    # Setup (networks, optimizers, data)
    trainer.setup(mesh_data, texture_data)

    # 保存训练前采样数据 (GT Data)
    trainer.save_sampling_data()

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Determine phases to run
    phases_to_run = [args.phase] if args.phase else [1, 2, 3]

    # Phase 1: Train G
    if 1 in phases_to_run:
        trainer.train_phase1(num_epochs=config.training.phase1_epochs)

    # Phase 2: Train D
    if 2 in phases_to_run:
        trainer.train_phase2(num_epochs=config.training.phase2_epochs)

    # Phase 3: Joint fine-tuning
    if 3 in phases_to_run:
        trainer.train_phase3(num_epochs=config.training.phase3_epochs)

    # Save final checkpoint
    trainer.save_checkpoint("final.pt")

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"实验文件夹: {experiment_manager.current_experiment_dir}")
    logger.info(f"采样数据: {experiment_manager.current_experiment_dir / 'sampling_data' / 'train_samples.npz'}")
    logger.info(f"检查点: {experiment_manager.current_experiment_dir / 'checkpoints'}")
    logger.info("=" * 60)

    if tracker:
        tracker.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
