"""
MA-IUVF 推理接口

从训练好的checkpoint加载模型，为低模预测UV
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
import json

logger = logging.getLogger(__name__)

from ..models.metric_aligned_iuv_field import MetricAlignedIUVField, MetricAlignedIUVOutput


class MetricAlignedIUVInference:
    """
    MA-IUVF 推理类

    从checkpoint加载模型，为3D位置预测UV
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
            checkpoint_path: 训练好的checkpoint路径
            texture_path: 纹理路径（可选，覆盖checkpoint）
            device: 推理设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"设备: {self.device}")

        # 加载checkpoint
        logger.info(f"加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # 提取元数据
        self.metadata = self._extract_metadata(checkpoint)

        # 创建模型
        self.model = self._create_model_from_metadata(self.metadata)

        # 尝试加载权重
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        except RuntimeError as e:
            logger.warning(f"strict=False加载权重: {e}")
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        self.model.to(self.device)
        self.model.eval()

        # 保存纹理路径（优先显式参数，然后从 baker_metadata 读取）
        if texture_path:
            self.texture_path = texture_path
        elif 'baker_metadata' in self.metadata and 'texture_path' in self.metadata['baker_metadata']:
            self.texture_path = self.metadata['baker_metadata']['texture_path']
        elif 'texture_path' in self.metadata:
            self.texture_path = self.metadata['texture_path']
        else:
            self.texture_path = None
            logger.warning("Checkpoint 中没有找到 texture_path")

        logger.info(f"推理器初始化完成")
        if self.texture_path:
            logger.info(f"  纹理: {self.texture_path}")

    def _extract_metadata(self, checkpoint: Dict) -> Dict:
        """从checkpoint提取元数据"""
        required_keys = [
            'num_charts',
            'hidden_dim',
            'num_layers',
            'positional_encoding_freqs',
        ]

        metadata = {}
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(f"Checkpoint缺少必要字段: {key}")
            metadata[key] = checkpoint[key]

        # 可选字段
        optional_keys = [
            'baker_metadata',
            'texture_path',
            'encoder_type',
            'activation',
            'hash_num_levels',
            'hash_features_per_level',
            'hash_log2_size',
            'hash_base_res',
            'hash_max_res',
            'hash_cuda_backend',
        ]
        for key in optional_keys:
            if key in checkpoint:
                metadata[key] = checkpoint[key]

        return metadata

    def _create_model_from_metadata(self, metadata: Dict) -> MetricAlignedIUVField:
        """从元数据创建模型"""
        from ..models.metric_aligned_iuv_field import create_model

        model = create_model(
            num_charts=metadata['num_charts'],
            hidden_dim=metadata['hidden_dim'],
            num_layers=metadata['num_layers'],
            positional_encoding_freqs=metadata['positional_encoding_freqs'],
            encoder_type=metadata.get('encoder_type', 'fourier'),
            hash_num_levels=metadata.get('hash_num_levels', 16),
            hash_features_per_level=metadata.get('hash_features_per_level', 2),
            hash_log2_size=metadata.get('hash_log2_size', 19),
            hash_base_res=metadata.get('hash_base_res', 16),
            hash_max_res=metadata.get('hash_max_res', 2048),
            hash_cuda_backend=metadata.get('hash_cuda_backend', 'auto'),
            bbox_min=metadata.get('baker_metadata', {}).get('bbox_min'),
            bbox_max=metadata.get('baker_metadata', {}).get('bbox_max'),
            activation=metadata.get('activation', 'softplus'),
        )

        return model

    @torch.no_grad()
    def predict(
        self,
        positions: np.ndarray,
        batch_size: int = 8192,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测UV分布

        Args:
            positions: [N, 3] 3D位置（原始空间）
            batch_size: 批处理大小

        Returns:
            logits: [N, C] chart分类logits
            uv_preds: [N, C, 2] UV预测
        """
        N = positions.shape[0]

        # 转换为tensor
        positions_tensor = torch.from_numpy(positions.astype(np.float32)).to(self.device)  # [N, 3]

        # 分批预测
        all_logits = []
        all_uv_preds = []

        num_batches = (N + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, N)

            batch_positions = positions_tensor[start_idx:end_idx]  # [B, 3]

            # 模型预测
            output = self.model(batch_positions)

            all_logits.append(output.logits.cpu().numpy())
            all_uv_preds.append(output.uv_preds.cpu().numpy())

        # 拼接
        logits = np.concatenate(all_logits, axis=0)  # [N, C]
        uv_preds = np.concatenate(all_uv_preds, axis=0)  # [N, C, 2]

        return logits, uv_preds

    @torch.no_grad()
    def select_uvs(
        self,
        logits: np.ndarray,
        uv_preds: np.ndarray,
        mode: str = "argmax",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从多chart预测中选择UV

        Args:
            logits: [N, C] chart分类logits
            uv_preds: [N, C, 2] UV预测
            mode: 选择模式
                - "argmax": 选择最大logit的chart（默认）
                - "sample": 采样chart

        Returns:
            selected_uvs: [N, 2] 选择的UV
            chart_ids: [N] chart ID
        """
        if mode == "argmax":
            # 选择最大logit的chart
            chart_ids = logits.argmax(axis=-1)  # [N]
            batch_indices = np.arange(len(logits))
            selected_uvs = uv_preds[batch_indices, chart_ids]  # [N, 2]

        elif mode == "sample":
            # 按概率采样chart
            probs = torch.nn.functional.softmax(
                torch.from_numpy(logits),
                dim=-1
            ).numpy()

            chart_ids = []
            selected_uvs = []

            for i in range(len(logits)):
                chart_id = np.random.choice(logits.shape[1], p=probs[i])
                chart_ids.append(chart_id)
                selected_uvs.append(uv_preds[i, chart_id])

            chart_ids = np.array(chart_ids)
            selected_uvs = np.array(selected_uvs)

        else:
            raise ValueError(f"未知模式: {mode}")

        return selected_uvs, chart_ids
