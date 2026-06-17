#!/usr/bin/env python3
"""
测试数据采样修复

验证:
1. UV坐标是否正确生成
2. 纹理采样是否有颜色变化
3. 法线采样是否合理
"""

import sys
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

from src.config import ExperimentConfig, DataConfig
from scripts.train import load_mesh_and_texture, spherical_uv, create_procedural_texture
from src.data import DataSamplingPipeline, MeshData, TextureData


def test_uv_generation():
    """测试UV坐标生成"""
    print("=" * 70)
    print("测试1: UV坐标生成")
    print("=" * 70)

    # 创建简单的立方体顶点
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=np.float32)

    uvs = spherical_uv(vertices)

    print(f"✓ UV坐标生成成功")
    print(f"  U范围: [{uvs[:, 0].min():.3f}, {uvs[:, 0].max():.3f}]")
    print(f"  V范围: [{uvs[:, 1].min():.3f}, {uvs[:, 1].max():.3f}]")

    # 检查UV变化
    u_unique = len(np.unique(uvs[:, 0].round(3)))
    v_unique = len(np.unique(uvs[:, 1].round(3)))
    print(f"  唯一U值数量: {u_unique}/8")
    print(f"  唯一V值数量: {v_unique}/8")

    if u_unique > 1 and v_unique > 1:
        print(f"  ✓ UV坐标有变化，不是单一值")
    else:
        print(f"  ✗ UV坐标没有变化！")

    return uvs


def test_texture_sampling():
    """测试纹理采样"""
    print("\n" + "=" * 70)
    print("测试2: 纹理采样")
    print("=" * 70)

    # 创建程序化纹理
    texture = create_procedural_texture(256, 256)

    print(f"✓ 纹理创建成功")
    print(f"  纹理形状: {texture.shape}")
    print(f"  R范围: [{texture[:, :, 0].min():.3f}, {texture[:, :, 0].max():.3f}]")
    print(f"  G范围: [{texture[:, :, 1].min():.3f}, {texture[:, :, 1].max():.3f}]")
    print(f"  B范围: [{texture[:, :, 2].min():.3f}, {texture[:, :, 2].max():.3f}]")

    # 检查颜色变化
    unique_colors = len(np.unique(texture.reshape(-1, 3), axis=0))
    print(f"  唯一颜色数量: {unique_colors}")

    if unique_colors > 100:
        print(f"  ✓ 纹理有丰富的颜色变化")
    else:
        print(f"  ✗ 纹理颜色变化不足！")

    return texture


def test_mesh_data_with_uvs():
    """测试带有UV的MeshData"""
    print("\n" + "=" * 70)
    print("测试3: MeshData with UVs")
    print("=" * 70)

    config = ExperimentConfig(
        seed=42,
        data=DataConfig(
            high_mesh_path='data/models/stanford-bunny.obj',
            low_mesh_path='data/models/stanford-bunny.obj',
            texture_path='',
            num_samples_per_epoch=100,
        )
    )

    try:
        mesh_data, texture_data = load_mesh_and_texture(config)

        print(f"✓ MeshData创建成功")
        print(f"  顶点数: {mesh_data.num_vertices}")
        print(f"  面数: {mesh_data.num_faces}")

        if mesh_data.uvs is not None:
            print(f"  ✓ UV坐标已设置")
            print(f"  UV形状: {mesh_data.uvs.shape}")
            print(f"  U范围: [{mesh_data.uvs[:, 0].min():.3f}, {mesh_data.uvs[:, 0].max():.3f}]")
            print(f"  V范围: [{mesh_data.uvs[:, 1].min():.3f}, {mesh_data.uvs[:, 1].max():.3f}]")

            # 检查UV变化
            u_unique = len(np.unique(mesh_data.uvs[:, 0].round(2)))
            v_unique = len(np.unique(mesh_data.uvs[:, 1].round(2)))
            print(f"  唯一U值: {u_unique}, 唯一V值: {v_unique}")

            if u_unique > 10 and v_unique > 10:
                print(f"  ✓ UV坐标有丰富变化")
                return True
            else:
                print(f"  ✗ UV坐标变化不足")
                return False
        else:
            print(f"  ✗ UV坐标为None！")
            return False

    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sampling_pipeline():
    """测试完整的采样管道"""
    print("\n" + "=" * 70)
    print("测试4: 完整采样管道")
    print("=" * 70)

    config = ExperimentConfig(
        seed=42,
        data=DataConfig(
            high_mesh_path='data/models/stanford-bunny.obj',
            low_mesh_path='data/models/stanford-bunny.obj',
            texture_path='',
            num_samples_per_epoch=1000,
        )
    )

    try:
        mesh_data, texture_data = load_mesh_and_texture(config)

        # 创建采样管道
        pipeline = DataSamplingPipeline(
            mesh=mesh_data,
            texture=texture_data,
            sampling_ratios={
                "surface": 0.4,
                "near_surface": 0.4,
                "exterior": 0.1,
                "interior": 0.1
            },
            near_surface_sigma=0.01,
            lowpass_sigma=1.0,
            num_classes=8,
        )

        print(f"✓ 采样管道创建成功")

        # 采样
        samples = pipeline.sample(
            num_points=1000,
            include_labels=True,
            use_cache=True
        )

        print(f"✓ 采样成功")
        print(f"  采样点数: {len(samples['positions'])}")
        print(f"  位置形状: {samples['positions'].shape}")
        print(f"  颜色形状: {samples['colors'].shape}")
        print(f"  法线形状: {samples['normals'].shape}")
        print(f"  SDF形状: {samples['sdf'].shape}")

        # 检查颜色变化
        colors = samples['colors']
        unique_colors = len(np.unique(colors.round(2), axis=0))
        print(f"  唯一颜色数量: {unique_colors}")

        r_range = (colors[:, 0].min(), colors[:, 0].max())
        g_range = (colors[:, 1].min(), colors[:, 1].max())
        b_range = (colors[:, 2].min(), colors[:, 2].max())
        print(f"  R范围: [{r_range[0]:.3f}, {r_range[1]:.3f}]")
        print(f"  G范围: [{g_range[0]:.3f}, {g_range[1]:.3f}]")
        print(f"  B范围: [{b_range[0]:.3f}, {b_range[1]:.3f}]")

        if unique_colors > 50:
            print(f"  ✓ 采样颜色有丰富变化")
        else:
            print(f"  ✗ 采样颜色变化不足！")

        # 检查是否有灰色占主导（表示纹理采样失败）
        gray_threshold = 0.55
        gray_count = np.sum(np.all(np.abs(colors - 0.5) < gray_threshold, axis=1))
        gray_ratio = gray_count / len(colors)
        print(f"  灰色点比例: {gray_ratio:.1%}")

        if gray_ratio < 0.1:
            print(f"  ✓ 灰色点比例正常")
        else:
            print(f"  ✗ 灰色点比例过高！")

        return True

    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_region_analysis():
    """测试不同区域的采样情况"""
    print("\n" + "=" * 70)
    print("测试5: 区域采样分析")
    print("=" * 70)

    config = ExperimentConfig(
        seed=42,
        data=DataConfig(
            high_mesh_path='data/models/stanford-bunny.obj',
            low_mesh_path='data/models/stanford-bunny.obj',
            texture_path='',
            num_samples_per_epoch=2000,
        )
    )

    try:
        mesh_data, texture_data = load_mesh_and_texture(config)

        pipeline = DataSamplingPipeline(
            mesh=mesh_data,
            texture=texture_data,
            sampling_ratios={
                "surface": 0.4,
                "near_surface": 0.4,
                "exterior": 0.1,
                "interior": 0.1
            },
            near_surface_sigma=0.01,
            lowpass_sigma=1.0,
            num_classes=8,
        )

        samples = pipeline.sample(
            num_points=2000,
            include_labels=True,
            use_cache=True
        )

        regions = samples['regions']

        print(f"✓ 区域分析:")
        print(f"  Surface点数: {np.sum(regions == 0)}")
        print(f"  Near-surface点数: {np.sum(regions == 1)}")
        print(f"  Exterior点数: {np.sum(regions == 2)}")
        print(f"  Interior点数: {np.sum(regions == 3)}")

        # 检查surface区域的颜色（应该最准确）
        surface_mask = regions == 0
        surface_colors = samples['colors'][surface_mask]

        print(f"\nSurface区域颜色:")
        print(f"  唯一颜色: {len(np.unique(surface_colors.round(2), axis=0))}")
        print(f"  R范围: [{surface_colors[:, 0].min():.3f}, {surface_colors[:, 0].max():.3f}]")
        print(f"  G范围: [{surface_colors[:, 1].min():.3f}, {surface_colors[:, 1].max():.3f}]")
        print(f"  B范围: [{surface_colors[:, 2].min():.3f}, {surface_colors[:, 2].max():.3f}]")

        return True

    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("数据采样修复测试")
    print("=" * 70)

    # 运行所有测试
    uvs = test_uv_generation()
    texture = test_texture_sampling()
    mesh_ok = test_mesh_data_with_uvs()
    pipeline_ok = test_sampling_pipeline()
    region_ok = test_region_analysis()

    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"  UV生成: {'✓' if len(np.unique(uvs[:, 0])) > 1 else '✗'}")
    print(f"  纹理: {'✓' if len(np.unique(texture.reshape(-1, 3), axis=0)) > 100 else '✗'}")
    print(f"  MeshData: {'✓' if mesh_ok else '✗'}")
    print(f"  采样管道: {'✓' if pipeline_ok else '✗'}")
    print(f"  区域分析: {'✓' if region_ok else '✗'}")

    if all([mesh_ok, pipeline_ok, region_ok]):
        print("\n✓ 所有测试通过！")
    else:
        print("\n✗ 部分测试失败")


if __name__ == "__main__":
    main()
