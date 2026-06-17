#!/usr/bin/env python3
"""
改进的训练脚本，集成实验管理和采样数据保存

支持:
- 自动创建实验文件夹结构
- 保存训练前的采样数据
- 组织checkpoints、logs和推理结果
- 可配置的实验ID
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import json
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from datetime import datetime

# Import experiment manager
sys.path.append('scripts')
from experiment_manager import ExperimentManager, create_sampling_data_dict

# Import project modules
from src.config import load_config, ExperimentConfig
from src.utils import get_device, setup_logger, RandomNumberGenerator
from src.utils.tracking import ExperimentTracker, MetricsAggregator

# Import models
from src.models import NetworkG, NetworkD, NetworkR

# Import data pipeline
from src.data import (
    MeshData,
    TextureData,
    DataSamplingPipeline,
    create_pipeline_from_files,
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


class EnhancedTrainer:
    """
    增强的训练器，支持实验管理和采样数据保存
    """

    def __init__(
        self,
        config: ExperimentConfig,
        experiment_manager: ExperimentManager,
        device: torch.device,
        experiment_name: str = None
    ):
        self.config = config
        self.exp_manager = experiment_manager
        self.device = device
        self.experiment_name = experiment_name

        # Create experiment
        self.experiment_id = self.exp_manager.create_experiment(
            experiment_name=experiment_name,
            config=config.__dict__ if hasattr(config, '__dict__') else config
        )

        print(f"✓ 创建实验: {self.experiment_id}")

        # Initialize logger
        self.logger = setup_logger(
            name="training",
            log_file=str(self.exp_manager.current_experiment_dir / "logs" / "train.log")
        )

        # Initialize tracker
        self.tracker = ExperimentTracker(
            project_name="Diffusion-UV",
            experiment_name=self.experiment_id
        )

        # Initialize networks
        self._initialize_networks()

        # Initialize optimizers
        self._initialize_optimizers()

        # State
        self.current_epoch = 0
        self.best_loss = float('inf')

    def _initialize_networks(self):
        """初始化网络模型"""
        cfg = self.config

        # Network G
        self.network_g = NetworkG(
            hidden_dim=cfg.network_g.hidden_dim,
            num_layers=cfg.network_g.num_layers,
            positional_encoding_freqs=cfg.network_g.positional_encoding_freqs,
            skip_connection_layer=cfg.network_g.skip_connection_layer,
            include_raw_input=cfg.network_g.include_raw_input,
            sdf_output_range=cfg.network_g.sdf_output_range,
        ).to(self.device)

        # Network D
        if cfg.training.train_network_d:
            self.network_d = NetworkD(
                condition_dim=cfg.network_d.condition_dim,
                hidden_channels=cfg.network_d.hidden_channels,
                num_res_blocks=cfg.network_d.num_res_blocks,
                num_diffusion_steps=cfg.network_d.num_diffusion_steps,
            ).to(self.device)

        # Network R
        if cfg.training.train_network_r:
            self.network_r = NetworkR(
                input_dim=cfg.network_r.input_dim,
                hidden_dim=cfg.network_r.hidden_dim,
                num_layers=cfg.network_r.num_layers,
                output_dim=cfg.network_r.output_dim,
            ).to(self.device)

        self.logger.info("网络初始化完成")

    def _initialize_optimizers(self):
        """初始化优化器"""
        cfg = self.config

        # Optimizer G
        self.optimizer_g = torch.optim.Adam(
            self.network_g.parameters(),
            lr=cfg.training.learning_rate_g,
            betas=cfg.training.adam_betas
        )

        # Optimizer D
        if hasattr(self, 'network_d') and self.network_d:
            self.optimizer_d = torch.optim.Adam(
                self.network_d.parameters(),
                lr=cfg.training.learning_rate_d,
                betas=cfg.training.adam_betas
            )

        # Optimizer R
        if hasattr(self, 'network_r') and self.network_r:
            self.optimizer_r = torch.optim.Adam(
                self.network_r.parameters(),
                lr=cfg.training.learning_rate_r,
                betas=cfg.training.adam_betas
            )

    def setup_data_pipeline(self):
        """设置数据采样管道"""
        cfg = self.config

        # Load mesh
        mesh_path = cfg.data.mesh_path
        texture_path = cfg.data.texture_path

        self.logger.info(f"加载网格: {mesh_path}")
        self.logger.info(f"加载纹理: {texture_path}")

        # Create sampling pipeline
        self.pipeline = create_pipeline_from_files(
            mesh_path=mesh_path,
            texture_path=texture_path,
            sampling_ratios=cfg.data.sampling_ratios,
            near_surface_sigma=cfg.data.near_surface_sigma,
            lowpass_sigma=cfg.data.lowpass_sigma,
            num_classes=cfg.data.num_classes
        )

        self.logger.info("数据采样管道设置完成")

    def save_sampling_data(self):
        """保存训练前的采样数据到实验文件夹"""
        self.logger.info("保存训练前采样数据...")

        cfg = self.config

        # 采样训练数据
        total_samples = cfg.data.num_samples_per_epoch
        train_samples = self.pipeline.sample(
            num_points=total_samples,
            include_labels=True,
            use_cache=True
        )

        # 准备采样数据字典
        train_data_dict = {
            'points': train_samples['positions'],
            'colors': train_samples['colors'],
            'colors_lowpass': train_samples['colors_lowpass'],
            'sdf': train_samples['sdf'],
            'normals': train_samples['normals'],
            'uvs': train_samples.get('uvs', np.zeros((len(train_samples['positions']), 2))),
            'curvatures': train_samples['curvatures'],
            'labels': train_samples.get('labels', np.zeros(len(train_samples['positions']))),
            'regions': train_samples.get('regions', np.zeros(len(train_samples['positions'])))
        }

        # 保存训练采样
        self.exp_manager.save_sampling_data(
            train_samples=train_data_dict,
            metadata={
                'total_samples': len(train_samples['positions']),
                'sampling_ratios': cfg.data.sampling_ratios,
                'sampling_date': datetime.now().isoformat(),
                'mesh_path': cfg.data.mesh_path,
                'texture_path': cfg.data.texture_path
            }
        )

        # 采样验证数据（如果需要）
        if cfg.data.validation_split > 0:
            val_samples_count = int(total_samples * cfg.data.validation_split)
            val_samples = self.pipeline.sample(
                num_points=val_samples_count,
                include_labels=True,
                use_cache=True
            )

            val_data_dict = {
                'points': val_samples['positions'],
                'colors': val_samples['colors'],
                'sdf': val_samples['sdf'],
                'normals': val_samples['normals']
            }

            self.exp_manager.save_sampling_data(
                val_samples=val_data_dict,
                metadata={'total_samples': len(val_samples['positions'])}
            )

        self.logger.info("✓ 采样数据已保存到实验文件夹")

    def train_phase1(self):
        """Phase 1: Train Network G"""
        self.logger.info("开始Phase 1训练: Network G")

        # 初始化损失函数
        sdf_loss = SDFLoss()
        color_loss = LowFrequencyColorLoss()
        eikonal_loss = EikonalLoss()

        num_epochs = self.config.training.phase1_epochs

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # TODO: 实现实际的训练循环
            # 这里只是框架，需要实现实际的批次训练

            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}/{num_epochs}")

                # 保存检查点
                if epoch % 50 == 0:
                    checkpoint_data = {
                        'epoch': epoch,
                        'network_g_state': self.network_g.state_dict(),
                        'optimizer_g_state': self.optimizer_g.state_dict(),
                        'config': self.config.__dict__ if hasattr(self.config, '__dict__') else self.config,
                        'best_loss': self.best_loss
                    }

                    self.exp_manager.save_checkpoint(
                        checkpoint_data,
                        f"phase1_epoch_{epoch}.pt"
                    )

        # 保存最终Phase 1检查点
        final_checkpoint = {
            'epoch': num_epochs,
            'network_g_state': self.network_g.state_dict(),
            'optimizer_g_state': self.optimizer_g.state_dict(),
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else self.config,
            'phase': 1
        }

        self.exp_manager.save_checkpoint(final_checkpoint, "phase1_final.pt")

    def train_phase2(self):
        """Phase 2: Train Network D"""
        self.logger.info("开始Phase 2训练: Network D")

        # TODO: 实现Network D训练

        pass

    def train_phase3(self):
        """Phase 3: Joint fine-tuning"""
        self.logger.info("开始Phase 3训练: 联合微调")

        # TODO: 实现联合训练

        pass

    def run_full_training(self):
        """运行完整的训练流程"""
        self.logger.info("开始完整训练流程")

        # 设置数据管道
        self.setup_data_pipeline()

        # 保存训练前采样数据
        self.save_sampling_data()

        # Phase 1
        if self.config.training.train_network_g:
            self.train_phase1()

        # Phase 2
        if self.config.training.train_network_d:
            self.train_phase2()

        # Phase 3
        if self.config.training.train_network_r:
            self.train_phase3()

        self.logger.info("训练完成!")


def main():
    parser = argparse.ArgumentParser(
        description="改进的训练脚本，支持实验管理",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--experiment-name", default=None, help="实验名称")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], help="训练阶段")
    parser.add_argument("--resume", help="恢复训练的检查点路径")
    parser.add_argument("--device", default="cuda", help="训练设备")

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 初始化实验管理器
    exp_manager = ExperimentManager()

    # 设置设备
    device = get_device(args.device)

    # 创建训练器
    trainer = EnhancedTrainer(
        config=config,
        experiment_manager=exp_manager,
        device=device,
        experiment_name=args.experiment_name
    )

    # 运行训练
    if args.phase:
        if args.phase == 1:
            trainer.train_phase1()
        elif args.phase == 2:
            trainer.train_phase2()
        elif args.phase == 3:
            trainer.train_phase3()
    else:
        trainer.run_full_training()


if __name__ == "__main__":
    main()
