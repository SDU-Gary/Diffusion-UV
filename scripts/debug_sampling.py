#!/usr/bin/env python3
"""
调试数据采样流程
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.config import ExperimentConfig, DataConfig
from scripts.train import load_mesh_and_texture
from src.data import DataSamplingPipeline, MeshData, TextureData


def debug_pipeline():
    """调试采样管道"""
    print("=" * 70)
    print("调试采样管道")
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

    # 加载mesh和texture
    mesh_data, texture_data = load_mesh_and_texture(config)

    print(f"\n1. MeshData检查:")
    print(f"   mesh_data类型: {type(mesh_data)}")
    print(f"   顶点数: {mesh_data.num_vertices}")
    print(f"   面数: {mesh_data.num_faces}")
    print(f"   mesh_data.uvs: {mesh_data.uvs is not None}")
    if mesh_data.uvs is not None:
        print(f"   mesh_data.uvs形状: {mesh_data.uvs.shape}")
        print(f"   mesh_data.uvs范围: U=[{mesh_data.uvs[:, 0].min():.3f}, {mesh_data.uvs[:, 0].max():.3f}]")
    else:
        print(f"   ✗ mesh_data.uvs是None!")

    print(f"\n2. TextureData检查:")
    print(f"   texture_data类型: {type(texture_data)}")
    print(f"   纹理形状: {texture_data.image.shape}")
    print(f"   纹理范围: R=[{texture_data.image[:, :, 0].min():.3f}, {texture_data.image[:, :, 0].max():.3f}]")

    # 创建采样管道
    print(f"\n3. 创建采样管道:")
    pipeline = DataSamplingPipeline(
        mesh=mesh_data,
        texture=texture_data,
        sampling_ratios={"surface": 0.4, "near_surface": 0.4, "exterior": 0.1, "interior": 0.1},
        near_surface_sigma=0.01,
        lowpass_sigma=1.0,
        num_classes=8,
    )

    print(f"   pipeline.mesh类型: {type(pipeline.mesh)}")
    print(f"   pipeline.mesh.uvs: {pipeline.mesh.uvs is not None}")
    if pipeline.mesh.uvs is not None:
        print(f"   pipeline.mesh.uvs形状: {pipeline.mesh.uvs.shape}")
    else:
        print(f"   ✗ pipeline.mesh.uvs是None!")

    print(f"   pipeline.texture_sampler: {pipeline.texture_sampler is not None}")

    # 测试表面采样
    print(f"\n4. 测试表面采样:")
    surface_pos, surface_norm, _ = pipeline.mesh_sampler.sample_surface_batch(10)
    print(f"   采样位置形状: {surface_pos.shape}")
    print(f"   采样法线形状: {surface_norm.shape}")

    # 测试颜色获取
    print(f"\n5. 测试颜色获取:")
    colors = pipeline._get_surface_colors(surface_pos)
    print(f"   颜色形状: {colors.shape}")
    print(f"   颜色范围: R=[{colors[:, 0].min():.3f}, {colors[:, 0].max():.3f}]")
    print(f"   是否全是灰色: {np.allclose(colors, 0.5, atol=0.01)}")

    # 测试UV插值
    print(f"\n6. 测试UV插值:")
    uvs = pipeline._interpolate_uvs(surface_pos)
    print(f"   UV形状: {uvs.shape}")
    print(f"   UV范围: U=[{uvs[:, 0].min():.3f}, {uvs[:, 0].max():.3f}]")
    print(f"   是否全是0.5: {np.allclose(uvs, 0.5, atol=0.01)}")

    # 测试纹理采样
    if pipeline.texture_sampler is not None:
        print(f"\n7. 测试纹理采样:")
        test_uvs = np.array([[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]])
        test_colors = pipeline.texture_sampler.sample(test_uvs)
        print(f"   测试UV: {test_uvs}")
        print(f"   采样颜色: {test_colors}")

    # 完整采样
    print(f"\n8. 完整采样测试:")
    samples = pipeline.sample(num_points=100, include_labels=True, use_cache=True)

    print(f"   采样点数: {len(samples['positions'])}")
    print(f"   颜色形状: {samples['colors'].shape}")
    print(f"   颜色范围: R=[{samples['colors'][:, 0].min():.3f}, {samples['colors'][:, 0].max():.3f}]")
    print(f"   唯一颜色数: {len(np.unique(samples['colors'].round(2), axis=0))}")
    print(f"   是否全是灰色: {np.allclose(samples['colors'], 0.5, atol=0.01)}")


if __name__ == "__main__":
    debug_pipeline()
