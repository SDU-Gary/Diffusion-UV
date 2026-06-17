"""
Texture Sampler Field: 高模邻域纹理采样分布场

独立于 G/D/R 主训练流程的新实验方向。

核心思想：
- 训练一个模型 P_theta，输入高模邻域内的查询点 x 和尺度 rho
- 输出纹理空间的采样分布 {u_i, w_i, sigma_i}
- 最终颜色来自原始纹理的采样，而非网络直接生成

架构：
P_theta(x, rho) -> {u_i, w_i, sigma_i}_{i=1..K}
C(x, rho) = sum_i w_i * T(u_i)
"""

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


@dataclass
class TextureSamplerFieldOutput:
    """
    纹理采样场输出

    Args:
        uvs: UV采样坐标，范围 [0, 1]，shape (B, K, 2)
        weights: 采样权重，softmax 后和为 1，shape (B, K)
        sigmas: 采样尺度参数（用于 mipmap），正数，shape (B, K, 1)
    """
    uvs: torch.Tensor  # (B, K, 2)
    weights: torch.Tensor  # (B, K)
    sigmas: torch.Tensor  # (B, K, 1) | None


class PositionalEncoding(nn.Module):
    """位置编码，用于将坐标映射到高维空间"""

    def __init__(self, num_freqs: int = 6, max_freq: float = None):
        """
        Args:
            num_freqs: 频率数量 L
            max_freq: 最大频率（可选）
        """
        super().__init__()
        self.num_freqs = num_freqs

        # 创建频率参数: 2^0, 2^1, ..., 2^(L-1)
        if max_freq is not None:
            freqs = torch.linspace(1.0, max_freq, num_freqs)
        else:
            freqs = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)

        # 注册为 buffer，使其包含在 state_dict 中并随 model.to(device) 管理
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入坐标 (B, 3) 或 (B, 2)

        Returns:
            位置编码 (B, 3 * 2 * L) 或 (B, 2 * 2 * L)
        """
        # x: (B, D)
        B = x.shape[0]
        D = x.shape[1]

        # 对每个频率，计算 sin 和 cos
        encoded = []
        for freq in self.freqs:
            for dim in range(D):
                encoded.append(torch.sin(freq * x[:, dim]))
                encoded.append(torch.cos(freq * x[:, dim]))

        # Stack: (B, D * 2 * L)
        encoded = torch.stack(encoded, dim=-1)

        return encoded


class TextureSamplerField(nn.Module):
    """
    纹理采样场网络

    输入：
        positions: (B, 3) - 高模邻域内的查询点
        scale: (B, 1) | None - 查询尺度（可选）

    输出：
        TextureSamplerFieldOutput
        - uvs: (B, K, 2) - UV采样坐标 [0,1]
        - weights: (B, K) - 采样权重（和为1）
        - sigmas: (B, K, 1) - 采样尺度参数（正数）
    """

    def __init__(
        self,
        num_mixtures: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 4,
        positional_encoding_freqs: int = 6,
        include_scale_in_encoding: bool = True,
        use_scale_input: bool = False,
        dropout_rate: float = 0.0,
    ):
        """
        Args:
            num_mixtures: 混合成分数量 K
            hidden_dim: 隐藏层维度
            num_layers: MLP 层数
            positional_encoding_freqs: 位置编码频率数量
            include_scale_in_encoding: 是否在位置编码中包含尺度
            use_scale_input: 是否使用尺度作为额外输入
            dropout_rate: Dropout 比率
        """
        super().__init__()

        self.num_mixtures = num_mixtures
        self.include_scale_in_encoding = include_scale_in_encoding
        self.use_scale_input = use_scale_input

        # 位置编码
        self.pos_enc = PositionalEncoding(positional_encoding_freqs)

        # 计算输入维度
        pos_enc_dim = 3 * 2 * positional_encoding_freqs  # (x,y,z) * (sin,cos) * freqs
        input_dim = pos_enc_dim

        if use_scale_input:
            input_dim += 1  # 添加尺度输入

        # MLP
        layers = []
        prev_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # 输出层
        # UV: 2 * K 个值 (每个混合成分的 u, v)
        self.uv_output = nn.Linear(prev_dim, 2 * num_mixtures)

        # Weights: K 个 logit
        self.weight_output = nn.Linear(prev_dim, num_mixtures)

        # Sigmas: K 个正值
        self.sigma_output = nn.Linear(prev_dim, num_mixtures)

    def forward(
        self,
        positions: torch.Tensor,
        scale: Optional[torch.Tensor] = None
    ) -> TextureSamplerFieldOutput:
        """
        前向传播

        Args:
            positions: (B, 3) 查询点坐标
            scale: (B, 1) | None 查询尺度（可选）

        Returns:
            TextureSamplerFieldOutput
        """
        B = positions.shape[0]

        # 位置编码
        pos_enc = self.pos_enc(positions)  # (B, 3 * 2 * L)

        # 准备输入
        if self.use_scale_input and scale is not None:
            # 扩展 scale 以匹配 batch
            if scale.dim() == 1:
                scale = scale.unsqueeze(-1)  # (B,) -> (B, 1)
            mlp_input = torch.cat([pos_enc, scale], dim=-1)  # (B, input_dim)
        else:
            mlp_input = pos_enc

        # MLP 前向传播
        features = self.mlp(mlp_input)  # (B, hidden_dim)

        # 输出 UV 坐标
        uv_flat = self.uv_output(features)  # (B, 2 * K)
        uvs = uv_flat.view(B, self.num_mixtures, 2)  # (B, K, 2)
        uvs = torch.sigmoid(uvs)  # 限制到 [0, 1]

        # 输出权重（logits -> softmax）
        weight_logits = self.weight_output(features)  # (B, K)
        weights = F.softmax(weight_logits, dim=-1)  # (B, K)

        # 输出 sigma（正值）
        sigma_flat = self.sigma_output(features)  # (B, K)
        sigmas = F.softplus(sigma_flat).unsqueeze(-1)  # (B, K, 1) - 确保为正

        return TextureSamplerFieldOutput(
            uvs=uvs,
            weights=weights,
            sigmas=sigmas
        )


def sample_texture(
    texture: torch.Tensor,
    uvs: torch.Tensor,
    weights: torch.Tensor,
    flip_v: bool = True
) -> torch.Tensor:
    """
    使用 UV 坐标和权重从纹理采样颜色

    Args:
        texture: 纹理张量，shape (3, H, W) 或 (1, 3, H, W)
        uvs: UV 采样坐标，shape (B, K, 2)，范围 [0, 1]
        weights: 采样权重，shape (B, K)，和为 1
        flip_v: 是否翻转 V 坐标（默认 True，与 UVTextureSampler 一致）

    Returns:
        采样颜色，shape (B, 3)

    注意：
    - grid_sample 的坐标范围是 [-1, 1]
    - 纹理的 V 方向可能需要翻转（取决于纹理存储方式）
    - flip_v=True 时，V=0 对应图像顶部（与 UVTextureSampler 一致）
    """
    B, K, _ = uvs.shape

    # 确保 texture 是 (1, 3, H, W) 格式
    if texture.dim() == 3:
        texture = texture.unsqueeze(0)  # (3, H, W) -> (1, 3, H, W)

    _, C, H, W = texture.shape

    # 扩展 texture 到 batch size B
    texture_batch = texture.expand(B, C, H, W)  # (B, 3, H, W)

    # 复制 UV 并翻转 V（如果需要）
    grid = uvs.clone()
    if flip_v:
        grid[..., 1] = 1.0 - grid[..., 1]  # 翻转 V 坐标

    # 将 UV 从 [0, 1] 转换到 [-1, 1] (grid_sample 要求)
    grid = grid * 2.0 - 1.0  # (B, K, 2) -> (B, K, 2)

    # grid_sample 需要 (B, H_out, W_out, 2) 格式
    # 我们把 K 个 UV 点作为 (H_out, W_out) = (K, 1)
    grid = grid.unsqueeze(2)  # (B, K, 2) -> (B, K, 1, 2)
    grid = grid.permute(0, 2, 1, 3)  # (B, K, 1, 2) -> (B, 1, K, 2)

    # 使用 grid_sample 从纹理采样
    # texture_batch: (B, 3, H, W)
    # grid: (B, 1, K, 2)
    # align_corners=True 与 UVTextureSampler 的像素坐标约定一致
    sampled = F.grid_sample(
        texture_batch,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )  # (B, 3, 1, K)

    # 重新排列: (B, 3, 1, K) -> (B, K, 3)
    sampled = sampled.squeeze(2)  # (B, 3, K)
    sampled = sampled.permute(0, 2, 1)  # (B, K, 3)

    # 加权求和
    # weights: (B, K), sampled: (B, K, 3)
    weights_expanded = weights.unsqueeze(-1)  # (B, K, 1)
    weighted_colors = (sampled * weights_expanded).sum(dim=1)  # (B, 3)

    return weighted_colors


# 导出接口
__all__ = [
    'TextureSamplerFieldOutput',
    'TextureSamplerField',
    'sample_texture',
]
