#!/usr/bin/env python3
"""
实验管理系统

管理实验文件夹结构，包含采样数据、训练输出、推理结果等。

实验文件夹结构:
outputs/
  └── experiments/
      └── 20250531_143000/           # 实验ID (YYYYMMDD_HHMMSS)
          ├── sampling_data/          # 训练前的采样数据
          │   ├── train_samples.npz   # 训练集采样
          │   ├── val_samples.npz     # 验证集采样
          │   └── metadata.json       # 采样元数据
          ├── checkpoints/            # 模型检查点
          │   ├── final.pt            # 最终模型
          │   ├── best.pt             # 最佳模型
          │   └── epoch_*.pt          # 中间检查点
          ├── logs/                   # 训练日志
          │   ├── train.log
          │   └── metrics.json
          ├── inference/              # 推理结果
          │   ├── colored_bunny.obj   # 彩色模型
          │   └── metrics.json
          └── config.yaml             # 实验配置
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import yaml
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ExperimentManager:
    """管理实验文件夹和文件结构"""

    def __init__(self, base_dir: str = "outputs/experiments"):
        """
        初始化实验管理器

        Args:
            base_dir: 实验基础目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.current_experiment_id = None
        self.current_experiment_dir = None

    def create_experiment(
        self,
        experiment_name: str = None,
        config: Dict = None
    ) -> str:
        """
        创建新的实验文件夹

        Args:
            experiment_name: 实验名称
            config: 实验配置字典

        Returns:
            实验ID (格式: YYYYMMDD_HHMMSS)
        """
        # 生成实验ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            experiment_id = f"{timestamp}_{experiment_name}"
        else:
            experiment_id = timestamp

        # 创建实验文件夹
        experiment_dir = self.base_dir / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # 创建子文件夹
        (experiment_dir / "sampling_data").mkdir(exist_ok=True)
        (experiment_dir / "checkpoints").mkdir(exist_ok=True)
        (experiment_dir / "logs").mkdir(exist_ok=True)
        (experiment_dir / "inference").mkdir(exist_ok=True)

        # 保存配置
        if config:
            config_file = experiment_dir / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

        # 创建元数据文件
        metadata = {
            "experiment_id": experiment_id,
            "created_at": datetime.now().isoformat(),
            "experiment_name": experiment_name,
            "status": "created"
        }

        metadata_file = experiment_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.current_experiment_id = experiment_id
        self.current_experiment_dir = experiment_dir

        print(f"✓ 创建实验文件夹: {experiment_dir}")
        return experiment_id

    def get_experiment_dir(self, experiment_id: str = None) -> Path:
        """获取实验文件夹路径"""
        if experiment_id:
            return self.base_dir / experiment_id
        elif self.current_experiment_dir:
            return self.current_experiment_dir
        else:
            raise ValueError("没有活跃的实验，请先创建或指定实验ID")

    def save_sampling_data(
        self,
        train_samples: Dict = None,
        val_samples: Dict = None,
        metadata: Dict = None
    ):
        """
        保存采样数据到实验文件夹

        Args:
            train_samples: 训练集采样数据 {'points': (N,3), 'colors': (N,3), ...}
            val_samples: 验证集采样数据
            metadata: 采样元数据
        """
        if not self.current_experiment_dir:
            raise ValueError("没有活跃的实验")

        sampling_dir = self.current_experiment_dir / "sampling_data"

        # 保存训练采样
        if train_samples:
            train_file = sampling_dir / "train_samples.npz"
            np.savez(train_file, **train_samples)
            print(f"✓ 保存训练采样: {train_file}")

        # 保存验证采样
        if val_samples:
            val_file = sampling_dir / "val_samples.npz"
            np.savez(val_file, **val_samples)
            print(f"✓ 保存验证采样: {val_file}")

        # 保存元数据
        if metadata:
            metadata_file = sampling_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"✓ 保存采样元数据: {metadata_file}")

    def save_checkpoint(
        self,
        checkpoint_data: Dict,
        checkpoint_name: str = "checkpoint.pt"
    ):
        """保存模型检查点"""
        if not self.current_experiment_dir:
            raise ValueError("没有活跃的实验")

        checkpoint_dir = self.current_experiment_dir / "checkpoints"
        checkpoint_file = checkpoint_dir / checkpoint_name

        if not TORCH_AVAILABLE:
            # 如果没有torch，保存为JSON格式
            json_file = checkpoint_file.with_suffix('.json')
            # 转换numpy数组为列表
            json_safe_data = {}
            for key, value in checkpoint_data.items():
                if isinstance(value, np.ndarray):
                    json_safe_data[key] = value.tolist()
                elif hasattr(value, 'state_dict'):
                    json_safe_data[key] = str(value)  # 简化处理
                else:
                    json_safe_data[key] = value

            with open(json_file, 'w') as f:
                json.dump(json_safe_data, f, indent=2)
            print(f"✓ 保存检查点(JSON格式): {json_file}")
        else:
            torch.save(checkpoint_data, checkpoint_file)
            print(f"✓ 保存检查点: {checkpoint_file}")

    def save_inference_results(
        self,
        results: Dict,
        result_name: str = "result"
    ):
        """保存推理结果"""
        if not self.current_experiment_dir:
            raise ValueError("没有活跃的实验")

        inference_dir = self.current_experiment_dir / "inference"

        # 保存结果字典
        if isinstance(results, dict):
            results_file = inference_dir / f"{result_name}_data.json"
            # 转换numpy数组为列表以便JSON序列化
            json_safe_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    json_safe_results[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    json_safe_results[key] = float(value)
                else:
                    json_safe_results[key] = value

            with open(results_file, 'w') as f:
                json.dump(json_safe_results, f, indent=2)
            print(f"✓ 保存推理数据: {results_file}")

    def save_config(self, config: Dict):
        """更新实验配置"""
        if not self.current_experiment_dir:
            raise ValueError("没有活跃的实验")

        config_file = self.current_experiment_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"✓ 更新配置: {config_file}")

    def list_experiments(self) -> Dict[str, Dict]:
        """列出所有实验"""
        experiments = {}

        for exp_dir in sorted(self.base_dir.iterdir()):
            if exp_dir.is_dir():
                metadata_file = exp_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    experiments[exp_dir.name] = metadata
                else:
                    experiments[exp_dir.name] = {
                        "experiment_id": exp_dir.name,
                        "status": "unknown"
                    }

        return experiments

    def load_experiment(self, experiment_id: str):
        """加载现有实验"""
        experiment_dir = self.base_dir / experiment_id
        if not experiment_dir.exists():
            raise ValueError(f"实验不存在: {experiment_id}")

        self.current_experiment_id = experiment_id
        self.current_experiment_dir = experiment_dir

        # 加载配置
        config_file = experiment_dir / "config.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            return config

        return {}


def create_sampling_data_dict(
    points: np.ndarray,
    colors: np.ndarray,
    uvs: np.ndarray = None,
    sdf: np.ndarray = None,
    normals: np.ndarray = None,
    metadata: Dict = None
) -> Dict:
    """
    创建采样数据字典，用于保存为NPZ格式

    Args:
        points: (N, 3) 3D点位置
        colors: (N, 3) RGB颜色
        uvs: (N, 2) UV坐标（可选）
        sdf: (N,) SDF值（可选）
        normals: (N, 3) 法向量（可选）
        metadata: 元数据字典

    Returns:
        可保存为NPZ的字典
    """
    data = {
        'points': points.astype(np.float32),
        'colors': colors.astype(np.float32)
    }

    if uvs is not None:
        data['uvs'] = uvs.astype(np.float32)

    if sdf is not None:
        data['sdf'] = sdf.astype(np.float32)

    if normals is not None:
        data['normals'] = normals.astype(np.float32)

    if metadata:
        data['metadata'] = metadata

    return data


def load_sampling_data_dict(npz_file: str) -> Dict:
    """从NPZ文件加载采样数据字典"""
    data = np.load(npz_file)

    result = {}
    for key in data.files:
        result[key] = data[key]

    return result


if __name__ == "__main__":
    # 测试实验管理器
    manager = ExperimentManager()

    # 创建新实验
    exp_id = manager.create_experiment(
        experiment_name="test_experiment",
        config={"batch_size": 32, "learning_rate": 0.001}
    )

    print(f"\n实验ID: {exp_id}")
    print(f"实验目录: {manager.current_experiment_dir}")

    # 测试保存采样数据
    train_samples = {
        'points': np.random.randn(100, 3).astype(np.float32),
        'colors': np.random.rand(100, 3).astype(np.float32),
        'sdf': np.random.randn(100).astype(np.float32)
    }

    manager.save_sampling_data(
        train_samples=train_samples,
        metadata={"num_samples": 100, "sampling_method": "uniform"}
    )

    print(f"\n✓ 测试完成")
