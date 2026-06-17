"""
Network D: Diffusion Model for High-Fidelity Color Generation

Architecture based on diffusers UNet with FiLM conditioning:
- 3D color input (instead of 2D latent)
- FiLM conditioning on geometry features (condition_dim=42)
- DDPMScheduler for training, DDIMScheduler for inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ..interfaces import INetworkD, NetworkDOutput, ConditionVector


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.

    Modulates features using scale (gamma) and shift (beta) vectors:
        y = gamma * x + beta

    Conditioning: (B, cond_dim) -> (B, feat_dim) for gamma and beta
    """

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.fc_gamma = nn.Linear(cond_dim, feat_dim)
        self.fc_beta = nn.Linear(cond_dim, feat_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning.

        Args:
            x: (B, C) or (B, C, 1) feature tensor
            cond: (B, cond_dim) conditioning vector

        Returns:
            (B, C) or (B, C, 1) modulated features
        """
        gamma = self.fc_gamma(cond)  # (B, feat_dim)
        beta = self.fc_beta(cond)  # (B, feat_dim)

        # For (B, C, 1) tensors, reshape gamma/beta to (B, C, 1)
        if x.dim() == 3:
            gamma = gamma.unsqueeze(-1)  # (B, feat_dim, 1)
            beta = beta.unsqueeze(-1)  # (B, feat_dim, 1)
        else:
            gamma = gamma  # (B, feat_dim)
            beta = beta  # (B, feat_dim)

        return gamma * x + beta


class ResBlock(nn.Module):
    """
    Residual block with group normalization and SiLU activation.

    Architecture:
        x -> GroupNorm -> SiLU -> Conv1d -> FiLM -> GroupNorm -> SiLU -> Conv1d -> Add -> SiLU -> out
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        self.film = FiLMLayer(cond_dim, out_channels)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip connection projection if dimensions differ
        self.skip_proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FiLM conditioning.

        Args:
            x: (B, C, 1) input tensor (color as single "pixel")
            cond: (B, cond_dim) conditioning vector

        Returns:
            (B, out_channels, 1) output tensor
        """
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        h = self.film(h, cond)

        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip_proj(x)


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for diffusion timesteps.

    Maps scalar timestep to vector representation.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create time embeddings.

        Args:
            timesteps: (B,) timestep values

        Returns:
            (B, dim) time embeddings
        """
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ColorUNet(nn.Module):
    """
    UNet-based diffusion model for 3D color generation.

    Architecture adapted for color input (B, 3) -> (B, 3):
        - Time embedding projected to hidden dimension
        - Color input projected to hidden dimension
        - Series of ResBlocks with FiLM conditioning
        - Skip connections between layers
        - Output projection to noise prediction
    """

    def __init__(
        self,
        color_dim: int = 3,
        hidden_channels: int = 128,
        num_res_blocks: int = 4,
        cond_dim: int = 42,
        time_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.color_dim = color_dim
        self.hidden_channels = hidden_channels
        self.cond_dim = cond_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Input projection: color (3) + time encoding
        self.input_proj = nn.Linear(color_dim + hidden_channels, hidden_channels)

        # ResBlocks with fixed hidden dimension
        self.res_blocks = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.res_blocks.append(
                ResBlock(hidden_channels, hidden_channels, cond_dim + hidden_channels, dropout)
            )

        # Output projection
        self.out_norm = nn.GroupNorm(8, hidden_channels)
        self.out_act = nn.SiLU(inplace=True)
        self.out_proj = nn.Linear(hidden_channels, color_dim)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        color: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            color: (B, 3) input color (noisy during training)
            timesteps: (B,) diffusion timesteps
            condition: (B, cond_dim) geometry conditioning

        Returns:
            (B, 3) predicted noise
        """
        B = color.shape[0]

        # Time embedding
        t_emb = self.time_mlp(timesteps)  # (B, hidden_channels)

        # Concatenate color and time for input
        color_emb = torch.cat([color, t_emb], dim=-1)  # (B, 3 + hidden)
        h = self.input_proj(color_emb)  # (B, hidden)

        # Reshape for 1D conv: (B, C) -> (B, C, 1)
        h = h.unsqueeze(-1)  # (B, hidden, 1)

        # Extend condition with time embedding
        cond_full = torch.cat([condition, t_emb], dim=-1)  # (B, cond_dim + time_dim)

        # Apply ResBlocks
        for i, block in enumerate(self.res_blocks):
            h = block(h, cond_full)  # (B, hidden, 1)

        # Output
        h = self.out_norm(h)
        h = self.out_act(h)
        h = h.squeeze(-1)  # (B, hidden)
        noise_pred = self.out_proj(h)  # (B, 3)

        return noise_pred


class DiffusionColorModel(nn.Module):
    """
    Wrapper for diffusion-based color generation.

    Handles:
    - Noise scheduling
    - Training with DDPM
    - Sampling with DDIM
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_res_blocks: int = 4,
        cond_dim: int = 42,
        num_diffusion_steps: int = 1000,
        beta_schedule: str = "linear",
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_diffusion_steps = num_diffusion_steps
        self.cond_dim = cond_dim

        # UNet model
        self.model = ColorUNet(
            color_dim=3,
            hidden_channels=hidden_channels,
            num_res_blocks=num_res_blocks,
            cond_dim=cond_dim,
            dropout=0.1,
        )

        # Noise scheduler
        self._setup_scheduler(beta_schedule)

    def _setup_scheduler(self, beta_schedule: str = "linear"):
        """Setup noise schedule."""
        if beta_schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, self.num_diffusion_steps)
        elif beta_schedule == "quadratic":
            betas = torch.linspace(1e-4, 0.02, self.num_diffusion_steps) ** 2
        elif beta_schedule == "cosine":
            # Cosine schedule
            steps = self.num_diffusion_steps + 1
            s = 0.008
            x = torch.linspace(0, self.num_diffusion_steps, steps)
            alphas_cumprod = torch.cos(((x / self.num_diffusion_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # Posterior variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

    def add_noise(
        self,
        color: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to clean color (forward diffusion).

        Args:
            color: (B, 3) clean color
            timesteps: (B,) timestep values

        Returns:
            noisy_color: (B, 3) noisy color
            noise: (B, 3) added noise
        """
        noise = torch.randn_like(color)
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1)

        noisy_color = sqrt_alpha * color + sqrt_one_minus_alpha * noise
        return noisy_color, noise

    def predict_start_from_noise(
        self,
        noisy_color: torch.Tensor,
        timesteps: torch.Tensor,
        noise_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Predict clean color from noisy color and noise prediction."""
        sqrt_recip_alpha = self.sqrt_recip_alphas_cumprod[timesteps].view(-1, 1)
        sqrt_recipm1_alpha = self.sqrt_recipm1_alphas_cumprod[timesteps].view(-1, 1)

        pred_color = sqrt_recip_alpha * noisy_color - sqrt_recipm1_alpha * noise_pred
        return pred_color

    def q_posterior(
        self,
        noisy_color: torch.Tensor,
        timesteps: torch.Tensor,
        noise_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Compute posterior mean and variance for DDPM training."""
        pred_color = self.predict_start_from_noise(noisy_color, timesteps, noise_pred)

        posterior_mean = (
            self.posterior_variance[timesteps].view(-1, 1) ** 0.5
            * self.alphas[timesteps].view(-1, 1) ** 0.5
            * noisy_color
            + (1 - self.alphas[timesteps].view(-1, 1)) ** 0.5
            * self.alphas_cumprod_prev[timesteps].view(-1, 1) ** 0.5
            * pred_color
        )
        return pred_color, posterior_mean

    def forward(
        self,
        noisy_color: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise from noisy color and condition.

        Args:
            noisy_color: (B, 3) noisy color
            timesteps: (B,) diffusion timesteps
            condition: ConditionVector or (B, 42) tensor

        Returns:
            (B, 3) predicted noise
        """
        if isinstance(condition, ConditionVector):
            condition = condition.to_tensor()

        noise_pred = self.model(noisy_color, timesteps, condition)
        return noise_pred


class NetworkD(nn.Module, INetworkD):
    """
    Diffusion Network (Network D).

    Generates high-fidelity colors conditioned on geometry features.
    Uses DDPM for training and DDIM for inference.

    Usage:
        model = NetworkD(cond_dim=42)
        noise_pred = model(noisy_color, timesteps, condition)
        sampled_colors = model.sample(condition, num_steps=20)
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_res_blocks: int = 4,
        cond_dim: int = 42,
        num_diffusion_steps: int = 1000,
        beta_schedule: str = "linear",
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_diffusion_steps = num_diffusion_steps

        self.diffusion = DiffusionColorModel(
            hidden_channels=hidden_channels,
            num_res_blocks=num_res_blocks,
            cond_dim=cond_dim,
            num_diffusion_steps=num_diffusion_steps,
            beta_schedule=beta_schedule,
        )

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
        noise_pred = self.diffusion(noisy_color, timestep, condition)
        return noise_pred

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
        self.eval()
        with torch.no_grad():
            return self._ddim_sample(condition, num_steps, deterministic)

    def _ddim_sample(
        self,
        condition: ConditionVector,
        num_steps: int,
        deterministic: bool,
    ) -> torch.Tensor:
        """DDIM sampling implementation."""
        B = condition.color_base.shape[0]
        device = condition.color_base.device

        # Get timesteps for DDIM (skip steps)
        step_ratio = self.num_diffusion_steps // num_steps
        timesteps = (torch.arange(0, num_steps) * step_ratio).long().to(device)
        timesteps = timesteps.flip(0)  # Start from high noise

        # Start from random noise
        color = torch.randn(B, 3, device=device)

        # Convert condition to tensor
        cond_tensor = condition.to_tensor()

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            # Predict noise
            noise_pred = self.diffusion(color, t_tensor, cond_tensor)

            # Predict clean color
            pred_color = self.diffusion.predict_start_from_noise(color, t_tensor, noise_pred)

            # DDIM step
            if i < len(timesteps) - 1:
                next_t = timesteps[i + 1]
                alpha_cur = self.diffusion.alphas_cumprod[t]
                alpha_next = self.diffusion.alphas_cumprod[next_t]

                # Direction pointing to noisy image
                pred_noise = (color - pred_color * alpha_cur ** 0.5) / (1 - alpha_cur) ** 0.5

                # Next image
                color = alpha_next ** 0.5 * pred_color + (1 - alpha_next) ** 0.5 * pred_noise
            else:
                color = pred_color

        # Clamp to valid color range
        return torch.clamp(color, 0, 1)

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
            noisy_color: (B, 3)
            noise: (B, 3)
        """
        return self.diffusion.add_noise(color, timestep)

    def training_step(
        self,
        color: torch.Tensor,
        condition: ConditionVector,
    ) -> torch.Tensor:
        """
        Perform one training step.

        Args:
            color: (B, 3) clean color
            condition: ConditionVector

        Returns:
            MSE loss between predicted and true noise
        """
        B = color.shape[0]
        device = color.device

        # Sample random timesteps
        timesteps = torch.randint(
            0, self.num_diffusion_steps, (B,), device=device, dtype=torch.long
        )

        # Add noise
        noisy_color, noise = self.diffusion.add_noise(color, timesteps)

        # Convert condition to tensor
        cond_tensor = condition.to_tensor()

        # Predict noise
        noise_pred = self.diffusion(noisy_color, timesteps, cond_tensor)

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        return loss

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count network parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_network_d(
    hidden_channels: int = 128,
    num_res_blocks: int = 4,
    cond_dim: int = 42,
    num_diffusion_steps: int = 1000,
    beta_schedule: str = "linear",
    **kwargs,
) -> NetworkD:
    """
    Create Network D with specified parameters.

    Args:
        hidden_channels: Hidden channel dimension
        num_res_blocks: Number of residual blocks
        cond_dim: Conditioning vector dimension
        num_diffusion_steps: Number of diffusion timesteps
        beta_schedule: Noise schedule type
        **kwargs: Additional arguments

    Returns:
        NetworkD instance
    """
    return NetworkD(
        hidden_channels=hidden_channels,
        num_res_blocks=num_res_blocks,
        cond_dim=cond_dim,
        num_diffusion_steps=num_diffusion_steps,
        beta_schedule=beta_schedule,
    )
