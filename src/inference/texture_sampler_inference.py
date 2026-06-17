"""
TextureSamplerField 推理接口

从训练好的 checkpoint 加载模型，为低模顶点预测纹理采样分布。
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import json
import logging

logger = logging.getLogger(__name__)

from ..models.texture_sampler_field import TextureSamplerField, sample_texture


class TextureSamplerFieldInference:
    """
    TextureSamplerField 推理类

    从 checkpoint 加载训练好的模型，为 3D 顶点预测纹理采样分布。
    """

    def __init__(
        self,
        checkpoint_path: str,
        texture_path: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        初始化推理器

        Args:
            checkpoint_path: 训练好的 checkpoint 路径
            texture_path: 纹理路径（可选，覆盖 checkpoint 中的路径）
            device: 推理设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")

        # 加载 checkpoint
        logger.info(f"加载 checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # 从 checkpoint metadata 恢复模型
        self.metadata = self._extract_metadata(checkpoint)

        # 创建模型
        self.model = self._create_model_from_metadata(self.metadata)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # 加载纹理
        self.texture_path = texture_path if texture_path else self.metadata.get('texture_path')
        if self.texture_path == "procedural":
            raise ValueError(
                "Checkpoint 使用程序化纹理，推理需要真实纹理文件。"
                "请使用 --texture 参数指定纹理路径。"
            )
        self.texture_tensor = self._load_texture(self.texture_path)

        logger.info(f"推理器初始化完成")
        logger.info(f"  纹理: {self.texture_path}")
        logger.info(f"  模型: {self.metadata.get('num_mixtures')} mixtures")

    def _extract_metadata(self, checkpoint: Dict) -> Dict:
        """从 checkpoint 提取 metadata"""
        required_keys = [
            'num_mixtures',
            'hidden_dim',
            'num_layers',
            'positional_encoding_freqs',
            'use_scale_input',
            'min_scale',
            'max_scale',
        ]

        metadata = {}
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(f"Checkpoint 缺少必要字段: {key}")
            metadata[key] = checkpoint[key]

        # 可选字段
        optional_keys = ['mesh_path', 'texture_path', 'bbox_min', 'bbox_max']
        for key in optional_keys:
            if key in checkpoint:
                metadata[key] = checkpoint[key]

        return metadata

    def _create_model_from_metadata(self, metadata: Dict) -> TextureSamplerField:
        """从 metadata 创建模型"""
        model = TextureSamplerField(
            num_mixtures=metadata['num_mixtures'],
            hidden_dim=metadata['hidden_dim'],
            num_layers=metadata['num_layers'],
            positional_encoding_freqs=metadata['positional_encoding_freqs'],
            use_scale_input=metadata['use_scale_input'],
            dropout_rate=0.0,
        )
        return model

    def _load_texture(self, texture_path: str) -> torch.Tensor:
        """加载纹理并转换为张量"""
        from PIL import Image

        logger.info(f"加载纹理: {texture_path}")
        texture_pil = Image.open(texture_path)

        if texture_pil.mode != 'RGB':
            texture_pil = texture_pil.convert('RGB')

        texture_np = np.array(texture_pil).astype(np.float32) / 255.0  # (H, W, 3)

        # 转换为 (3, H, W)
        texture_tensor = torch.from_numpy(texture_np).permute(2, 0, 1)  # (3, H, W)
        texture_tensor = texture_tensor.to(self.device)

        logger.info(f"  纹理形状: {texture_tensor.shape}")

        return texture_tensor

    @torch.no_grad()
    def predict_distribution(
        self,
        positions: np.ndarray,
        scales: np.ndarray,
        batch_size: int = 8192
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        预测纹理采样分布

        Args:
            positions: (N, 3) 3D 坐标（原始空间，不归一化）
            scales: (N, 1) 或 (N,) 尺度
            batch_size: 批处理大小

        Returns:
            uvs: (N, K, 2) UV 坐标，范围 [0, 1]
            weights: (N, K) 采样权重，和为 1
            sigmas: (N, K, 1) 尺度参数
        """
        N = positions.shape[0]

        # 转换为 tensor
        positions_tensor = torch.from_numpy(positions.astype(np.float32)).to(self.device)  # (N, 3)
        scales_tensor = torch.from_numpy(scales.astype(np.float32)).to(self.device)  # (N,) or (N, 1)

        # 确保 scales 是 (N, 1)
        if scales_tensor.dim() == 1:
            scales_tensor = scales_tensor.unsqueeze(-1)  # (N,) -> (N, 1)

        # 分批预测
        all_uvs = []
        all_weights = []
        all_sigmas = []

        num_batches = (N + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, N)

            batch_positions = positions_tensor[start_idx:end_idx]  # (B, 3)
            batch_scales = scales_tensor[start_idx:end_idx]  # (B, 1)

            # 模型预测
            output = self.model(batch_positions, scale=batch_scales)

            all_uvs.append(output.uvs.cpu().numpy())
            all_weights.append(output.weights.cpu().numpy())
            all_sigmas.append(output.sigmas.cpu().numpy())

        # 拼接
        uvs = np.concatenate(all_uvs, axis=0)  # (N, K, 2)
        weights = np.concatenate(all_weights, axis=0)  # (N, K)
        sigmas = np.concatenate(all_sigmas, axis=0)  # (N, K, 1)

        # 检查 NaN
        if np.isnan(uvs).any():
            raise ValueError("预测结果包含 NaN，请检查输入坐标范围")
        if np.isnan(weights).any():
            raise ValueError("预测结果包含 NaN，请检查输入坐标范围")

        return uvs, weights, sigmas

    @torch.no_grad()
    def predict_colors(
        self,
        positions: np.ndarray,
        scales: np.ndarray,
        batch_size: int = 8192
    ) -> np.ndarray:
        """
        预测颜色（用于调试）

        使用模型预测的分布从纹理采样颜色。

        Args:
            positions: (N, 3) 3D 坐标
            scales: (N, 1) 或 (N,) 尺度
            batch_size: 批处理大小

        Returns:
            colors: (N, 3) RGB 颜色，范围 [0, 1]
        """
        N = positions.shape[0]

        # 转换为 tensor
        positions_tensor = torch.from_numpy(positions.astype(np.float32)).to(self.device)  # (N, 3)
        scales_tensor = torch.from_numpy(scales.astype(np.float32)).to(self.device)  # (N,) or (N, 1)

        # 确保 scales 是 (N, 1)
        if scales_tensor.dim() == 1:
            scales_tensor = scales_tensor.unsqueeze(-1)  # (N,) -> (N, 1)

        # 分批预测和采样
        all_colors = []

        num_batches = (N + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, N)

            batch_positions = positions_tensor[start_idx:end_idx]  # (B, 3)
            batch_scales = scales_tensor[start_idx:end_idx]  # (B, 1)

            # 模型预测
            output = self.model(batch_positions, scale=batch_scales)

            # 直接采样颜色
            batch_colors = sample_texture(
                self.texture_tensor.unsqueeze(0),  # (1, 3, H, W)
                output.uvs,
                output.weights,
                flip_v=True
            )  # (B, 3)

            all_colors.append(batch_colors.cpu().numpy())

        # 拼接
        colors = np.concatenate(all_colors, axis=0)  # (N, 3)

        return colors

    def select_uvs(
        self,
        uvs: np.ndarray,
        weights: np.ndarray,
        mode: str = "argmax"
    ) -> np.ndarray:
        """
        从混合分布中选择 UV

        Args:
            uvs: (N, K, 2) UV 坐标
            weights: (N, K) 权重
            mode: 选择模式
                - "argmax": 选择最大权重的 UV（默认，避免跨 seam）
                - "weighted": 加权平均（实验性，可能跨 seam）

        Returns:
            selected_uvs: (N, 2) 选择的 UV 坐标
        """
        if mode == "argmax":
            # 选择最大权重的 UV
            max_indices = weights.argmax(axis=-1)  # (N,)
            batch_indices = np.arange(len(uvs))
            selected_uvs = uvs[batch_indices, max_indices]  # (N, 2)

        elif mode == "weighted":
            # 加权平均（实验性）
            selected_uvs = (weights[:, :, np.newaxis] * uvs).sum(axis=1)  # (N, 2)

        else:
            raise ValueError(f"未知模式: {mode}，支持 'argmax' 或 'weighted'")

        return selected_uvs
