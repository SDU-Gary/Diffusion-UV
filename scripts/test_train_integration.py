#!/usr/bin/env python3
"""
测试train.py与实验管理器的集成

验证:
1. 实验文件夹创建
2. 采样数据保存
3. 检查点保存到实验文件夹
"""

import sys
import os
from pathlib import Path
import shutil
import tempfile

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

# 导入必要的模块
from src.config import ExperimentConfig, DataConfig, NetworkGConfig, NetworkDConfig, NetworkRConfig, TrainingConfig, LossConfig, LoggingConfig, EvaluationConfig
from src.utils import get_device, RandomNumberGenerator
from src.data import MeshData, TextureData, create_pipeline_from_files
from scripts.train import ImplicitTextureTrainer
from scripts.experiment_manager import ExperimentManager

def create_minimal_config():
    """创建最小配置用于测试"""
    config_dict = {
        'seed': 42,
        'data': {
            'high_mesh_path': 'data/models/stanford-bunny.obj',
            'low_mesh_path': 'data/models/stanford-bunny.obj',
            'texture_path': '',
            'num_samples_per_epoch': 1000,
            'sampling_ratios': {'surface': 0.4, 'near_surface': 0.4, 'exterior': 0.1, 'interior': 0.1},
            'cache_dir': '.cache/test_integration',
            'near_surface_epsilon': 0.01,
            'lowpass_sigma': 1.0,
        },
        'network_g': {
            'hidden_dim': 64,
            'num_layers': 4,
            'positional_encoding_freqs': 4,
            'skip_connection_layer': 2,
            'include_raw_input': True,
            'sdf_output_range': 1.0,
        },
        'network_d': {
            'condition_dim': 42,
            'hidden_channels': 64,
            'num_res_blocks': 2,
            'num_diffusion_steps': 50,
            'scheduler_type': 'squaredcos_cap_v2',
            'num_down_layers': 2,
        },
        'network_r': {
            'hidden_dim': 128,
            'num_layers': 3,
            'num_classes': 8,
            'positional_encoding_freqs': 6,
        },
        'training': {
            'phase1_epochs': 2,
            'phase2_epochs': 1,
            'phase3_epochs': 1,
            'learning_rate_g': 0.001,
            'learning_rate_d': 0.0001,
            'learning_rate_r': 0.001,
            'batch_size_phase1': 32,
            'batch_size_phase2': 32,
            'batch_size_phase3': 32,
            'gradient_clip_norm': 1.0,
            'weight_decay': 0.0001,
            'lambda_sdf': 1.0,
            'lambda_eikonal': 0.1,
            'lambda_color_base': 1.0,
            'lambda_diffusion': 0.5,
            'lambda_reverse_start': 0.01,
            'lambda_reverse_end': 0.1,
            'lambda_entropy_start': 0.001,
            'lambda_entropy_end': 0.01,
            'optimizer': 'adam',
            'mixed_precision': False,
        },
        'loss': {
            'lambda_sdf': 1.0,
            'lambda_eikonal': 0.1,
            'lambda_color_base': 1.0,
            'lambda_diffusion': 1.0,
            'lambda_reverse': 0.1,
            'lambda_entropy': 0.0,
        },
        'logging': {
            'log_dir': 'logs',
            'experiment_name': None,
            'log_interval': 10,
            'save_interval': 50,
            'use_wandb': False,
            'use_tensorboard': False,
            'wandb_project': 'Diffusion-UV',
            'wandb_mode': 'offline',
            'render_interval': 10,
        },
        'evaluation': {
            'output_dir': 'outputs/inference_results',
            'eval_interval': 10,
            'num_render_views': 10,
            'render_resolution': 512,
        },
    }
    return ExperimentConfig.from_dict(config_dict)


def test_experiment_integration():
    """测试train.py与实验管理器的集成"""
    print("=" * 70)
    print("测试: train.py 实验管理器集成")
    print("=" * 70)

    # 加载配置
    config = create_minimal_config()

    # 设置临时输出目录
    temp_dir = Path(tempfile.mkdtemp(prefix='diffusion_uv_test_'))
    config.logging.log_dir = str(temp_dir / 'logs')
    config.data.cache_dir = str(temp_dir / 'cache')
    config.evaluation.output_dir = str(temp_dir / 'outputs')

    print(f"使用临时目录: {temp_dir}")

    try:
        # 初始化实验管理器
        exp_manager = ExperimentManager(base_dir=str(temp_dir / 'experiments'))

        # 创建实验
        experiment_id = exp_manager.create_experiment(
            experiment_name="test_train_integration",
            config=config.to_dict()
        )

        print(f"\n✓ 步骤1: 创建实验文件夹")
        print(f"  实验ID: {experiment_id}")
        print(f"  实验目录: {exp_manager.current_experiment_dir}")

        # 设置设备
        device = get_device('cpu')
        print(f"\n✓ 步骤2: 设置设备: {device}")

        # 加载数据
        print(f"\n✓ 步骤3: 加载mesh数据")
        mesh_path = config.data.high_mesh_path

        if not Path(mesh_path).exists():
            print(f"  警告: Mesh文件不存在: {mesh_path}")
            print(f"  创建模拟mesh数据...")

            # 创建简单的立方体mesh用于测试
            vertices = np.array([
                [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
            ], dtype=np.float32)

            faces = np.array([
                [0, 1, 2], [0, 2, 3],  # bottom
                [4, 5, 6], [4, 6, 7],  # top
                [0, 1, 5], [0, 5, 4],  # front
                [2, 3, 7], [2, 7, 6],  # back
                [0, 3, 7], [0, 7, 4],  # left
                [1, 2, 6], [1, 6, 5],  # right
            ], dtype=np.int32)

            mesh_data = MeshData(
                vertices=vertices,
                faces=faces,
                vertex_normals=np.array([[0, 0, -1]] * 8, dtype=np.float32),
            )
        else:
            import trimesh
            mesh = trimesh.load(mesh_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = list(mesh.geometry.values())[0]

            mesh_data = MeshData(
                vertices=np.array(mesh.vertices, dtype=np.float32),
                faces=np.array(mesh.faces, dtype=np.int32),
                vertex_normals=np.array(mesh.vertex_normals, dtype=np.float32),
            )

        # 创建程序化纹理
        print(f"✓ 创建程序化纹理")
        u = np.linspace(0, 1, 256)
        v = np.linspace(0, 1, 256)
        U, V = np.meshgrid(u, v)
        r = 0.7 + 0.3 * np.sin(U * 2 * np.pi * 4)
        g = 0.5 + 0.3 * np.sin(V * 2 * np.pi * 3)
        b = 0.6 + 0.3 * np.sin((U + V) * 2 * np.pi * 2)
        texture_array = np.stack([r, g, b], axis=2).astype(np.float32)
        texture_data = TextureData.from_array(texture_array)

        print(f"  Mesh: {mesh_data.num_vertices} vertices, {mesh_data.num_faces} faces")
        print(f"  Texture: {texture_data.width}x{texture_data.height}")

        # 创建trainer
        print(f"\n✓ 步骤4: 创建trainer (带实验管理器)")
        trainer = ImplicitTextureTrainer(
            config=config,
            device=device,
            tracker=None,
            experiment_manager=exp_manager,
        )

        # Setup
        print(f"\n✓ 步骤5: Setup (网络、优化器、数据管道)")
        trainer.setup(mesh_data, texture_data)

        # 保存采样数据
        print(f"\n✓ 步骤6: 保存采样数据到实验文件夹")
        trainer.save_sampling_data()

        # 验证采样数据文件
        sampling_file = exp_manager.current_experiment_dir / "sampling_data" / "train_samples.npz"
        if sampling_file.exists():
            print(f"  ✓ 采样数据文件已创建: {sampling_file}")

            # 验证可以加载
            data = np.load(sampling_file)
            print(f"  ✓ 采样数据可以加载: {list(data.files)}")
            print(f"    - points: {data['points'].shape}")
            print(f"    - colors: {data['colors'].shape}")
        else:
            print(f"  ✗ 采样数据文件未创建")
            return False

        # 保存一个测试检查点
        print(f"\n✓ 步骤7: 保存检查点")
        checkpoint_data = {
            'epoch': 1,
            'global_step': 100,
            'phase': 1,
            'best_loss': 0.5,
            'network_g_state': trainer.network_g.state_dict(),
            'optimizer_g_state': trainer.optimizer_g.state_dict(),
        }

        trainer.save_checkpoint("test_checkpoint.pt")

        checkpoint_file = exp_manager.current_experiment_dir / "checkpoints" / "test_checkpoint.pt"
        if checkpoint_file.exists():
            print(f"  ✓ 检查点已保存到实验文件夹: {checkpoint_file}")
        else:
            print(f"  ✗ 检查点未保存")
            return False

        # 验证实验文件夹结构
        print(f"\n✓ 步骤8: 验证实验文件夹结构")
        expected_structure = [
            "sampling_data/train_samples.npz",
            "sampling_data/metadata.json",
            "checkpoints/test_checkpoint.pt",
            "logs",
            "inference",
            "config.yaml",
            "metadata.json",
        ]

        all_ok = True
        for item in expected_structure:
            item_path = exp_manager.current_experiment_dir / item
            if item_path.exists():
                print(f"  ✓ {item}")
            else:
                print(f"  ✗ 缺失: {item}")
                all_ok = False

        print(f"\n" + "=" * 70)
        if all_ok:
            print("✓ 集成测试通过!")
            print(f"\n实验文件夹: {exp_manager.current_experiment_dir}")
            print(f"\n可以查看采样数据:")
            print(f"  python scripts/viewer_3d.py {exp_manager.current_experiment_dir}/sampling_data/train_samples.npz")
        else:
            print("✗ 集成测试失败!")
        print("=" * 70)

        return all_ok

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 清理临时文件
        if temp_dir.exists():
            print(f"\n清理临时文件: {temp_dir}")
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    success = test_experiment_integration()
    sys.exit(0 if success else 1)
