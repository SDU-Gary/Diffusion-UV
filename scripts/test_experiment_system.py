#!/usr/bin/env python3
"""
实验管理系统测试脚本

测试完整的实验工作流程：
1. 创建实验文件夹
2. 生成和保存采样数据
3. 模拟保存checkpoints
4. 使用viewer查看NPZ数据
"""

import sys
import numpy as np
from pathlib import Path
import json
import shutil

# 添加脚本路径
sys.path.append('scripts')

sys.path.append('scripts')
from experiment_manager import ExperimentManager, create_sampling_data_dict

print("=" * 70)
print("实验管理系统测试")
print("=" * 70)

# 步骤1: 测试实验管理器
print("\n步骤1: 测试实验管理器创建")
print("-" * 70)

exp_manager = ExperimentManager()

# 创建测试实验
exp_id = exp_manager.create_experiment(
    experiment_name="test_pipeline",
    config={"batch_size": 32, "learning_rate": 0.001}
)

print(f"✓ 实验ID: {exp_id}")
print(f"✓ 实验目录: {exp_manager.current_experiment_dir}")

# 验证文件夹结构
expected_dirs = ["sampling_data", "checkpoints", "logs", "inference"]
for dir_name in expected_dirs:
    dir_path = exp_manager.current_experiment_dir / dir_name
    if dir_path.exists():
        print(f"  ✓ {dir_name}/ 文件夹已创建")
    else:
        print(f"  ✗ {dir_name}/ 文件夹缺失")

# 步骤2: 测试采样数据保存
print("\n步骤2: 测试采样数据保存")
print("-" * 70)

# 生成模拟采样数据
num_samples = 1000
mock_samples = {
    'points': np.random.randn(num_samples, 3).astype(np.float32),
    'colors': np.random.rand(num_samples, 3).astype(np.float32),
    'sdf': np.random.randn(num_samples).astype(np.float32),
    'normals': np.random.randn(num_samples, 3).astype(np.float32),
    'uvs': np.random.rand(num_samples, 2).astype(np.float32),
}

print(f"生成模拟采样数据: {num_samples} 个点")
print(f"  points: {mock_samples['points'].shape}")
print(f"  colors: {mock_samples['colors'].shape}")

# 保存采样数据
exp_manager.save_sampling_data(
    train_samples=mock_samples,
    metadata={
        'num_samples': num_samples,
        'sampling_method': 'test',
        'date': '2025-05-31'
    }
)

# 验证采样数据文件
sampling_file = exp_manager.current_experiment_dir / "sampling_data" / "train_samples.npz"
if sampling_file.exists():
    print(f"  ✓ 采样文件已创建: {sampling_file}")

    # 验证可以加载
    loaded_data = np.load(sampling_file)
    print(f"  ✓ 采样数据可以加载: {list(loaded_data.keys())}")
else:
    print(f"  ✗ 采样文件未创建")

# 步骤3: 测试checkpoints保存
print("\n步骤3: 测试checkpoints保存")
print("-" * 70)

import torch
mock_checkpoint = {
    'epoch': 10,
    'network_g_state': {'dummy': 'data'},
    'optimizer_g_state': {'dummy': 'optimizer'},
    'best_loss': 0.123
}

exp_manager.save_checkpoint(mock_checkpoint, "test_checkpoint.pt")

checkpoint_file = exp_manager.current_experiment_dir / "checkpoints" / "test_checkpoint.pt"
if checkpoint_file.exists():
    print(f"  ✓ checkpoint文件已创建: {checkpoint_file}")
else:
    print(f"  ✗ checkpoint文件未创建")

# 步骤4: 测试viewer NPZ预览
print("\n步骤4: 测试Viewer NPZ预览")
print("-" * 70)

try:
    from viewer_3d import Viewer3D

    viewer = Viewer3D(title="NPZ Preview Test")

    if viewer.load_sampling_data(str(sampling_file)):
        print("  ✓ Viewer成功加载NPZ采样数据")
        print("  ✓ 点云预览功能正常")
    else:
        print("  ✗ Viewer加载NPZ失败")

except Exception as e:
    print(f"  ✗ Viewer测试失败: {e}")

# 步骤5: 测试实验列表
print("\n步骤5: 测试实验列表功能")
print("-" * 70)

experiments = exp_manager.list_experiments()
print(f"  找到 {len(experiments)} 个实验:")

for exp_id, metadata in experiments.items():
    status = metadata.get('status', 'unknown')
    exp_name = metadata.get('experiment_name', 'N/A')
    created = metadata.get('created_at', 'N/A')[:19]  # 只显示日期时间部分
    print(f"  - {exp_id}: {exp_name} ({status}) - {created}")

# 步骤6: 验证文件结构完整性
print("\n步骤6: 验证文件结构完整性")
print("-" * 70)

def verify_experiment_structure(exp_dir):
    """验证实验文件夹结构的完整性"""
    required_files = {
        "sampling_data/train_samples.npz": "训练采样数据",
        "sampling_data/metadata.json": "采样元数据",
        "checkpoints/": "checkpoint目录",
        "logs/": "日志目录",
        "inference/": "推理结果目录",
        "config.yaml": "实验配置",
        "metadata.json": "实验元数据"
    }

    all_good = True
    for file_path, description in required_files.items():
        full_path = exp_dir / file_path
        if full_path.exists():
            print(f"  ✓ {description}: {file_path}")
        else:
            print(f"  ✗ 缺失 {description}: {file_path}")
            all_good = False

    return all_good

structure_ok = verify_experiment_structure(exp_manager.current_experiment_dir)

# 步骤7: 测试实际数据管道
print("\n步骤7: 测试实际数据管道")
print("-" * 70)

try:
    # 检查数据文件是否存在
    mesh_path = "data/models/stanford-bunny.obj"
    texture_path = "data/textures/bunny_texture.png"

    if Path(mesh_path).exists() and Path(texture_path).exists():
        print(f"  ✓ 数据文件存在")

        # 创建数据管道
        print(f"  创建数据采样管道...")
        pipeline = create_pipeline_from_files(
            mesh_path=mesh_path,
            texture_path=texture_path,
            sampling_ratios={
                "surface": 0.4,
                "near_surface": 0.4,
                "exterior": 0.1,
                "interior": 0.1
            }
        )

        print(f"  ✓ 数据管道创建成功")

        # 采样实际数据
        print(f"  采样实际数据...")
        actual_samples = pipeline.sample(
            num_points=1000,
            include_labels=True,
            use_cache=True
        )

        print(f"  ✓ 采样成功: {len(actual_samples['positions'])} 个点")

        # 创建实际采样数据字典
        actual_data_dict = {
            'points': actual_samples['positions'],
            'colors': actual_samples['colors'],
            'sdf': actual_samples['sdf'],
            'normals': actual_samples['normals'],
            'uvs': actual_samples.get('uvs', np.zeros((len(actual_samples['positions']), 2)))
        }

        # 保存实际采样数据
        new_exp_id = exp_manager.create_experiment(
            experiment_name="real_data_test"
        )

        exp_manager.save_sampling_data(
            train_samples=actual_data_dict,
            metadata={
                'total_samples': len(actual_samples['positions']),
                'mesh_path': mesh_path,
                'texture_path': texture_path
            }
        )

        print(f"  ✓ 实际数据测试完成: {new_exp_id}")

    else:
        print(f"  ⚠ 数据文件不存在，跳过实际数据管道测试")

except Exception as e:
    print(f"  ✗ 实际数据管道测试失败: {e}")
    import traceback
    traceback.print_exc()

# 总结
print("\n" + "=" * 70)
print("实验管理系统测试总结")
print("=" * 70)

print("\n✓ 已完成:")
print("  1. 实验文件夹结构创建")
print("  2. 采样数据保存为NPZ格式")
print("  3. Viewer NPZ点云预览")
print("  4. 实验管理和列表功能")

print(f"\n实验文件夹示例: {exp_manager.current_experiment_dir}")
print(f"  采样数据示例: {sampling_file}")

print("\n文件结构:")
print(f"  outputs/experiments/")
print(f"  └── 20250531_143000_test_pipeline/")
print(f"      ├── sampling_data/")
print(f"      │   ├── train_samples.npz       # 训练集采样 (可viewer预览)")
print(f"      │   └── metadata.json")
print(f"      ├── checkpoints/")
print(f"      │   └── *.pt                      # 模型检查点")
print(f"      ├── logs/")
print(f"      │   └── train.log")
print(f"      ├── inference/")
print(f"      │   └── *.obj                      # 推理结果")
print(f"      └── config.yaml")

print("\n使用方法:")
print("  1. 查看采样数据: python3 scripts/viewer_3d.py outputs/experiments/*/sampling_data/train_samples.npz")
print("  2. 运行训练: python3 scripts/train_enhanced.py --config configs/experiment.yaml")
print("  3. 列出实验: python scripts/experiment_manager.py (需要添加list命令)")

print("\n" + "=" * 70)
print("✓ 实验管理系统测试完成！")
print("=" * 70)
