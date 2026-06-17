"""
MA-IUVF 模型：多chart分支隐式UV场

Phase 1: Fourier位置编码 + Softplus MLP（可微、稳定）
Phase 2: 可替换为hash grid加速
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
import logging

from .encoders import BSplineHashGrid

logger = logging.getLogger(__name__)


@dataclass
class MetricAlignedIUVOutput:
    """MA-IUVF输出"""
    logits: torch.Tensor       # [B, C], chart分类logits
    uv_preds: torch.Tensor     # [B, C, 2], 每个chart的UV预测


class FourierPositionalEncoding(nn.Module):
    """
    Fourier位置编码

    可微、平滑，适合拟合导数
    """

    def __init__(self, num_freqs: int):
        """
        Args:
            num_freqs: 频率数量（从2^0到2^(num_freqs-1)）
        """
        super().__init__()
        self.num_freqs = num_freqs

        # 预计算频率
        freqs = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer('freqs', freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3] 输入坐标

        Returns:
            encoded: [B, 3 * num_freqs * 2] 编码特征
        """
        # [B, 3, 1]
        x = x.unsqueeze(-1)

        # [B, 3, num_freqs]
        scaled = x * self.freqs.view(1, 1, -1)

        # sin and cos
        sin_feat = torch.sin(scaled)  # [B, 3, num_freqs]
        cos_feat = torch.cos(scaled)  # [B, 3, num_freqs]

        # 交错拼接
        encoded = torch.cat([
            sin_feat, cos_feat
        ], dim=-1)  # [B, 3, num_freqs * 2]

        # 展平
        encoded = encoded.view(encoded.shape[0], -1)  # [B, 3 * num_freqs * 2]

        return encoded


class MetricAlignedIUVField(nn.Module):
    """
    多chart分支隐式UV场

    输入3D位置，输出：
    1. chart分类logits
    2. 每个chart的UV预测
    """

    def __init__(
        self,
        num_charts: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        positional_encoding_freqs: int = 8,
        encoder_type: str = "fourier",
        hash_num_levels: int = 16,
        hash_features_per_level: int = 2,
        hash_log2_size: int = 19,
        hash_base_res: int = 16,
        hash_max_res: int = 2048,
        hash_cuda_backend: str = "auto",
        bbox_min=None,
        bbox_max=None,
        activation: str = "softplus",
    ):
        """
        Args:
            num_charts: chart数量
            hidden_dim: MLP隐藏层维度
            num_layers: MLP层数
            positional_encoding_freqs: 位置编码频率数量
            encoder_type: "fourier" 或 "bspline_hash"
        """
        super().__init__()

        if encoder_type not in {"fourier", "bspline_hash"}:
            raise ValueError(f"未知 encoder_type: {encoder_type}")
        if activation not in {"softplus", "silu", "relu"}:
            raise ValueError(f"未知 activation: {activation}")

        self.num_charts = num_charts
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.positional_encoding_freqs = positional_encoding_freqs
        self.encoder_type = encoder_type
        self.hash_num_levels = hash_num_levels
        self.hash_features_per_level = hash_features_per_level
        self.hash_log2_size = hash_log2_size
        self.hash_base_res = hash_base_res
        self.hash_max_res = hash_max_res
        self.hash_cuda_backend = hash_cuda_backend
        self.activation = activation

        # 空间编码
        if encoder_type == "fourier":
            self.pos_enc = FourierPositionalEncoding(positional_encoding_freqs)
            self.grid_encoder = None
            encoder_dim = 3 * positional_encoding_freqs * 2
        else:
            self.pos_enc = None
            self.grid_encoder = BSplineHashGrid(
                num_levels=hash_num_levels,
                features_per_level=hash_features_per_level,
                log2_hashmap_size=hash_log2_size,
                base_res=hash_base_res,
                max_res=hash_max_res,
                cuda_backend=hash_cuda_backend,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
            )
            encoder_dim = self.grid_encoder.output_dim

        # 特征编码MLP
        layers = []
        input_dim = encoder_dim

        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == "softplus":
                layers.append(nn.Softplus(beta=10))
            elif activation == "silu":
                layers.append(nn.SiLU())
            else:
                layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)  # [B, hidden_dim]

        # chart分类头
        self.chart_head = nn.Linear(hidden_dim, num_charts)

        # UV预测头（每个chart一个独立的2D输出）
        # 输出 [B, num_charts * 2]
        self.uv_head = nn.Linear(hidden_dim, num_charts * 2)

        logger.info(
            f"MA-IUVF: {num_charts}charts, "
            f"encoder={encoder_type}, hidden_dim={hidden_dim}, layers={num_layers}, "
            f"pos_enc_freqs={positional_encoding_freqs}, "
            f"hash_levels={hash_num_levels}, hash_F={hash_features_per_level}"
        )

    def forward(self, pos: torch.Tensor) -> MetricAlignedIUVOutput:
        """
        前向传播

        Args:
            pos: [B, 3] 输入位置

        Returns:
            输出
        """
        # 空间编码
        if self.encoder_type == "fourier":
            pos_encoded = self.pos_enc(pos)  # [B, pos_enc_dim]
        else:
            pos_encoded = self.grid_encoder(pos)  # [B, L * F]

        # MLP编码
        features = self.encoder(pos_encoded)  # [B, hidden_dim]

        # chart分类
        logits = self.chart_head(features)  # [B, num_charts]

        # UV预测
        uv_flat = self.uv_head(features)  # [B, num_charts * 2]
        uv_preds = uv_flat.view(-1, self.num_charts, 2)  # [B, num_charts, 2]

        # 不限制UV范围（Phase 1改进：避免sigmoid影响导数幅值）
        # uv_preds保持unconstrained

        return MetricAlignedIUVOutput(
            logits=logits,
            uv_preds=uv_preds,
        )

    def get_num_params(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_optimizer_param_groups(
        self,
        lr: float,
        hash_lr: Optional[float] = None,
        hash_weight_decay: float = 1e-6,
        mlp_weight_decay: float = 0.0,
    ):
        """Return optimizer parameter groups with separate hash-grid settings."""
        if self.encoder_type != "bspline_hash":
            return [{"params": list(self.parameters()), "lr": lr, "weight_decay": mlp_weight_decay}]

        hash_lr = lr if hash_lr is None else hash_lr
        hash_params = [self.grid_encoder.hash_table]
        hash_param_ids = {id(p) for p in hash_params}
        other_params = [p for p in self.parameters() if id(p) not in hash_param_ids]
        return [
            {
                "params": hash_params,
                "lr": hash_lr,
                "weight_decay": hash_weight_decay,
            },
            {
                "params": other_params,
                "lr": lr,
                "weight_decay": mlp_weight_decay,
            },
        ]


def create_model(
    num_charts: int,
    hidden_dim: int = 64,
    num_layers: int = 3,
    positional_encoding_freqs: int = 8,
    encoder_type: str = "fourier",
    hash_num_levels: int = 16,
    hash_features_per_level: int = 2,
    hash_log2_size: int = 19,
    hash_base_res: int = 16,
    hash_max_res: int = 2048,
    hash_cuda_backend: str = "auto",
    bbox_min=None,
    bbox_max=None,
    activation: str = "softplus",
) -> MetricAlignedIUVField:
    """
    创建MA-IUVF模型的便捷函数

    Args:
        num_charts: chart数量
        hidden_dim: 隐藏层维度
        num_layers: 层数
        positional_encoding_freqs: 位置编码频率数
        encoder_type: "fourier" 或 "bspline_hash"

    Returns:
        模型
    """
    model = MetricAlignedIUVField(
        num_charts=num_charts,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        positional_encoding_freqs=positional_encoding_freqs,
        encoder_type=encoder_type,
        hash_num_levels=hash_num_levels,
        hash_features_per_level=hash_features_per_level,
        hash_log2_size=hash_log2_size,
        hash_base_res=hash_base_res,
        hash_max_res=hash_max_res,
        hash_cuda_backend=hash_cuda_backend,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        activation=activation,
    )

    logger.info(f"创建模型: {model.get_num_params()}参数")

    return model
