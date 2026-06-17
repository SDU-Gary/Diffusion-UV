"""
测试 UV chart 分割
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.metric_aligned_iuv_baker import MetricAlignedIUVBaker

def test_uv_chart_segmentation():
    """测试 UV chart 分割"""
    print("=== 测试 UV Chart 分割 ===\n")

    # 测试 OBJ 解析器 + UV chart 分割
    obj_path = "test_data/uv_seam_cube.obj"

    print(f"1. 测试 OBJ 解析器加载")
    baker = MetricAlignedIUVBaker(obj_path, seed=42, use_obj_parser=True)

    print(f"   - 顶点数: {len(baker.vertices)}")
    print(f"   - UV数: {len(baker.uvs)}")
    print(f"   - 面数: {len(baker.face_vertices)}")
    print(f"   - Face vertex indices shape: {baker.face_vertex_indices.shape}")
    print(f"   - Face UV indices shape: {baker.face_uv_indices.shape}")

    print(f"\n2. 测试 face_component 模式")
    data_face_comp = baker.bake(
        num_samples=100,
        extrusion_sigma_ratio=0.01,
        chart_mode="face_component",
    )
    print(f"   - 样本数: {len(data_face_comp.pos)}")
    print(f"   - Charts: {data_face_comp.chart_id.max().item() + 1}")
    print(f"   - 所有chart_id为0: {torch.all(data_face_comp.chart_id == 0)}")

    print(f"\n3. 测试 uv_islands 模式")
    data_uv_islands = baker.bake(
        num_samples=100,
        extrusion_sigma_ratio=0.01,
        chart_mode="uv_islands",
    )
    print(f"   - 样本数: {len(data_uv_islands.pos)}")
    print(f"   - Charts: {data_uv_islands.chart_id.max().item() + 1}")
    print(f"   - Chart ID分布: {torch.bincount(data_uv_islands.chart_id)}")

    print(f"\n4. 测试保存（带完整元数据）")
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())

    output_path = temp_dir / "baked.pt"
    baker.save(
        data_uv_islands,
        str(output_path),
        chart_mode="uv_islands",
        extrusion_sigma_ratio=0.01,
        texture_path="test.png",
    )

    print(f"   - 保存路径: {output_path}")
    print(f"   - 文件存在: {output_path.exists()}")

    # 加载并检查元数据
    from src.data.metric_aligned_iuv_baker import load_baked_data
    loaded_data, metadata = load_baked_data(str(output_path))

    print(f"\n5. 元数据检查")
    print(f"   - mesh_path: {metadata.get('mesh_path')}")
    print(f"   - num_samples: {metadata.get('num_samples')}")
    print(f"   - num_charts: {metadata.get('num_charts')}")
    print(f"   - chart_mode: {metadata.get('chart_mode')}")
    print(f"   - extrusion_sigma: {metadata.get('extrusion_sigma'):.6f}")
    print(f"   - bbox_min: {metadata.get('bbox_min')}")
    print(f"   - bbox_max: {metadata.get('bbox_max')}")
    print(f"   - uv_convention: {metadata.get('uv_convention')}")
    print(f"   - texture_path: {metadata.get('texture_path')}")

    if 'chart_stats' in metadata:
        stats = metadata['chart_stats']
        print(f"   - chart_stats:")
        print(f"     - num_charts: {stats.get('num_charts')}")
        print(f"     - chart_sizes: {stats.get('chart_sizes')}")
        print(f"     - num_uv_seams: {stats.get('num_uv_seams')}")

    print(f"\n6. 验证加载一致性")
    assert torch.allclose(loaded_data.pos, data_uv_islands.pos), "pos不一致"
    assert torch.allclose(loaded_data.j_3d_gt, data_uv_islands.j_3d_gt), "j_3d_gt不一致"
    assert torch.allclose(loaded_data.uv_anchor, data_uv_islands.uv_anchor), "uv_anchor不一致"
    assert torch.all(loaded_data.chart_id == data_uv_islands.chart_id), "chart_id不一致"
    print(f"   - 数据一致性验证通过")

    # 清理
    import shutil
    shutil.rmtree(temp_dir)

    print(f"\n✓ 所有测试通过")


if __name__ == "__main__":
    import torch
    test_uv_chart_segmentation()
